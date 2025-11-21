import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import math
from improved_model import GalaxyModel
import time
import random
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import _LRScheduler

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def count_parameters(model):
    """Count model parameters (in millions)"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6
    }


def measure_inference_speed(model, device, image_size=256, num_runs=100, warmup_runs=10):
    """Measure model inference speed (FPS)"""
    model.eval()
    batch_size = 1  # Single sample inference speed
    
    # Create dummy input
    dummy_image = torch.randn(batch_size, 3, image_size, image_size).to(device)
    dummy_coords = torch.randn(batch_size, 2).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_image, dummy_coords)
    
    # Synchronize GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual test
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_image, dummy_coords)
    
    # Synchronize GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_sample = total_time / num_runs
    fps = 1.0 / avg_time_per_sample
    
    model.train()  # Restore training mode
    return {
        'fps': fps,
        'avg_time_ms': avg_time_per_sample * 1000,
        'total_time_s': total_time
    }


class UBAScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=0,
        max_lr=None,
        min_lr=1e-6,
        initial_lr=None,
        phi=0.8,
        last_epoch=-1,
        verbose=False,
        enable_adaptive_phi=True,
        base_phi=0.90,
        phi_min=0.80,
        phi_max=1.15,
        enable_val_feedback=True,
        val_loss_history_size=5,
        val_improvement_threshold=1e-4,
        final_decay_phase_ratio=0.15,  # Final 15% of epochs enter final decay phase
        disable_adaptive_phi_in_final_phase=True  # Disable adaptive phi in final phase
    ):
        self.total_steps = total_steps
        self._initial_total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.phi = phi
        self.base_phi = base_phi if enable_adaptive_phi else phi
        
        # Adaptive parameter settings
        self.enable_adaptive_phi = enable_adaptive_phi
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.disable_adaptive_phi_in_final_phase = disable_adaptive_phi_in_final_phase
        
        # Final decay phase settings
        self.final_decay_phase_ratio = final_decay_phase_ratio
        self.final_decay_start_step = None  # Will be calculated in get_lr
        
        # Validation feedback settings
        self.enable_val_feedback = enable_val_feedback
        self.val_loss_history = []  # Store recent validation losses
        self.val_loss_history_size = val_loss_history_size
        self.val_improvement_threshold = val_improvement_threshold
        self._val_feedback_factor = 1.0
        
        # Gradient history for adaptive phi
        self.grad_norm_history = []
        self.grad_norm_history_size = 10
        
        # Get initial learning rate
        if initial_lr is None:
            initial_lr = optimizer.param_groups[0]['lr']
        
        # Get maximum learning rate
        if max_lr is None:
            max_lr = optimizer.param_groups[0]['lr']
        
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        
        # Calculate total steps after warmup
        self.decay_steps = total_steps - warmup_steps
        
        # Validate parameters
        assert 0 <= phi <= 2, f"phi must be in [0, 2], got {phi}"
        assert warmup_steps < total_steps, f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})"
        assert min_lr <= max_lr, f"min_lr ({min_lr}) must be <= max_lr ({max_lr})"
        
        # PyTorch version compatibility: some versions don't support verbose parameter
        try:
            super(UBAScheduler, self).__init__(optimizer, last_epoch, verbose)
        except TypeError:
            # If verbose parameter not supported, pass only required parameters
            super(UBAScheduler, self).__init__(optimizer, last_epoch)
    
    def update_grad_norm(self, grad_norm):
        """Update gradient norm for adaptive phi"""
        if not self.enable_adaptive_phi:
            return
        
        self.grad_norm_history.append(float(grad_norm))
        if len(self.grad_norm_history) > self.grad_norm_history_size:
            self.grad_norm_history.pop(0)
    
    def update_validation_loss(self, val_loss):
        """Update validation loss for feedback mechanism"""
        if not self.enable_val_feedback:
            return
        
        self.val_loss_history.append(float(val_loss))
        if len(self.val_loss_history) > self.val_loss_history_size:
            self.val_loss_history.pop(0)
        
        # Adjust decay speed based on validation loss
        if len(self.val_loss_history) >= 3:
            recent_improvement = self.val_loss_history[-3] - self.val_loss_history[-1]
            if recent_improvement < -self.val_improvement_threshold:
                # Validation loss increasing, accelerate decay (faster LR reduction)
                self._val_feedback_factor = min(1.2, self._val_feedback_factor * 1.05)
            elif recent_improvement > self.val_improvement_threshold:
                # Validation loss decreasing well, maintain current LR longer (slow decay)
                self._val_feedback_factor = max(0.8, self._val_feedback_factor * 0.98)
    
    def _adaptive_phi(self):
        """Adaptively adjust phi based on gradient stability"""
        if not self.enable_adaptive_phi or len(self.grad_norm_history) < 5:
            return self.base_phi
        
        # Calculate coefficient of variation (stability metric)
        grad_mean = np.mean(self.grad_norm_history)
        if grad_mean < 1e-8:
            return self.base_phi
        
        grad_std = np.std(self.grad_norm_history)
        coeff_var = grad_std / grad_mean  # Coefficient of variation
        
        # If gradient unstable (high CV), increase phi (smoother decay)
        if coeff_var > 0.5:  # Unstable gradient
            new_phi = min(self.phi_max, self.base_phi * 1.2)
        elif coeff_var < 0.2:  # Stable gradient
            new_phi = max(self.phi_min, self.base_phi * 0.9)
        else:
            new_phi = self.base_phi
        
        return new_phi
    
    def get_lr(self):
        """Calculate learning rate for current step"""
        # Calculate final decay phase start step (only once)
        if self.final_decay_start_step is None:
            self.final_decay_start_step = int(self.total_steps * (1 - self.final_decay_phase_ratio))
        
        # Check if we're in final decay phase
        in_final_phase = self.last_epoch >= self.final_decay_start_step
        
        # Adaptively adjust phi (lightweight operation, computed only when needed)
        # Disable adaptive phi in final phase if configured
        if self.enable_adaptive_phi and self.last_epoch >= self.warmup_steps:
            if in_final_phase and self.disable_adaptive_phi_in_final_phase:
                # In final phase: fix phi to base_phi to prevent learning rate increase
                self.phi = self.base_phi
            else:
                # Normal adaptive phi adjustment
                self.phi = self._adaptive_phi()
        
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear growth from initial_lr to max_lr
            progress = self.last_epoch / max(1, self.warmup_steps)
            current_lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # UBA scheduling phase: use UBA formula
            # Apply validation feedback factor (adjust decay speed) - but disable in final phase
            if in_final_phase:
                # In final phase: disable validation feedback, use fixed decay
                effective_decay_steps = self.decay_steps
            else:
                effective_decay_steps = int(self.decay_steps * self._val_feedback_factor)
            effective_decay_steps = max(1, effective_decay_steps)
            
            # Calculate relative position in decay phase [0, 1]
            step_in_decay = self.last_epoch - self.warmup_steps
            progress = step_in_decay / max(1, effective_decay_steps)
            progress = min(1.0, progress)  # Ensure not exceeding 1
            
            # UBA core formula: use cosine function
            theta = progress * math.pi
            
            # Calculate cosine term
            cos_theta = math.cos(theta)
            one_plus_cos = 1 + cos_theta
            
            # UBA formula: η_t = (η_max - η_min) * [2(1+cos)] / [2φ + (2-φ)(1+cos)] + η_min
            numerator = 2 * one_plus_cos
            denominator = 2 * self.phi + (2 - self.phi) * one_plus_cos
            
            # Avoid division by zero (theoretically won't happen, but safe)
            if abs(denominator) < 1e-10:
                denominator = 1e-10
            
            lr_factor = numerator / denominator
            current_lr = (self.max_lr - self.min_lr) * lr_factor + self.min_lr
            
            # In final phase: apply additional exponential decay to ensure LR → 0
            if in_final_phase:
                # Calculate progress within final phase [0, 1]
                final_phase_steps = self.total_steps - self.final_decay_start_step
                final_phase_progress = (self.last_epoch - self.final_decay_start_step) / max(1, final_phase_steps)
                final_phase_progress = min(1.0, final_phase_progress)
                
                # Apply exponential decay: decay_factor = exp(-5 * progress)
                # This ensures LR → min_lr (essentially 0) at the end
                decay_factor = math.exp(-5.0 * final_phase_progress)
                current_lr = self.min_lr + (current_lr - self.min_lr) * decay_factor
            
        # Ensure learning rate is within valid range
        current_lr = max(self.min_lr, min(self.max_lr, current_lr))
        
        return [current_lr for _ in self.optimizer.param_groups]
    
    def get_current_phi(self):
        """Get current phi value for monitoring"""
        return self.phi


def compute_log_mse_unified(redshift_pred, redshift_true, min_val=1e-6, max_val=None):

    if isinstance(redshift_pred, torch.Tensor):
        pred_safe = torch.clamp(redshift_pred.squeeze() if redshift_pred.dim() > 1 else redshift_pred, min=min_val)
        true_safe = torch.clamp(redshift_true.squeeze() if redshift_true.dim() > 1 else redshift_true, min=min_val)
        if max_val is not None:
            pred_safe = torch.clamp(pred_safe, max=max_val)
            true_safe = torch.clamp(true_safe, max=max_val)
        log_pred = torch.log1p(pred_safe)
        log_true = torch.log1p(true_safe)
        return F.mse_loss(log_pred, log_true)
    else:
        pred_safe = np.clip(np.atleast_1d(redshift_pred).flatten(), min_val, max_val if max_val is not None else np.inf)
        true_safe = np.clip(np.atleast_1d(redshift_true).flatten(), min_val, max_val if max_val is not None else np.inf)
        log_pred = np.log1p(pred_safe)
        log_true = np.log1p(true_safe)
        return np.mean((log_pred - log_true) ** 2)


def custom_collate_fn(batch):
    """Custom collate function"""
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)




def get_redshift_output(outputs, default_redshift=None):
    """Get redshift prediction output (prefer redshift_final, backward compatible with redshift_pred)"""
    pred_redshifts = outputs.get('redshift_final', outputs.get('redshift_pred', None))
    if pred_redshifts is None:
        if default_redshift is not None:
            return default_redshift
        else:
            raise ValueError(f"No redshift prediction found in outputs. Keys: {list(outputs.keys())}")
    return pred_redshifts


def state_dicts_equal(dict1, dict2):
    """Check if two state_dicts are equal"""
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1.keys():
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True


class Galaxy10Dataset(Dataset):
    """Galaxy10 dataset loader (compatible with modified model interface)"""
    
    _split_cache = {}
    
    def __init__(self, h5_path, num_samples=120, image_size=128, split='train', 
                 use_augmentation=True):
        self.h5_path = h5_path
        self.num_samples = num_samples
        self.image_size = image_size
        self.split = split  # 'train' or 'val'
        self.use_augmentation = use_augmentation and split == 'train'
        
        val_split = getattr(self, '_val_split', 0.3)
        cache_key = (h5_path, num_samples, val_split)
        
        if cache_key not in Galaxy10Dataset._split_cache:
            all_images, all_labels, all_redshifts, all_ra, all_dec = self._load_data()
            
            train_indices, val_indices = train_test_split(
                np.arange(len(all_images)), 
                test_size=val_split, random_state=42, stratify=all_labels
            )
            
            train_images = all_images[train_indices]
            val_images = all_images[val_indices]
            train_labels = all_labels[train_indices]
            val_labels = all_labels[val_indices]
            train_redshifts = all_redshifts[train_indices]
            val_redshifts = all_redshifts[val_indices]
            train_ra = all_ra[train_indices]
            val_ra = all_ra[val_indices]
            train_dec = all_dec[train_indices]
            val_dec = all_dec[val_indices]
            
            Galaxy10Dataset._split_cache[cache_key] = {
                'train': (train_images, train_labels, train_redshifts, train_ra, train_dec),
                'val': (val_images, val_labels, val_redshifts, val_ra, val_dec)
            }
        
        split_data = Galaxy10Dataset._split_cache[cache_key][split]
        self.images, self.labels, self.redshifts, self.ra, self.dec = split_data
    
    def _load_data(self):
        """Load data from H5 file"""
        with h5py.File(self.h5_path, 'r') as f:
            images = f['images'][:]
            labels = f['labels'][:] if 'labels' in f else f.get('ans', None)
            if labels is None:
                raise ValueError("Cannot find labels in h5 file")
            labels = labels.astype(np.int64)
            redshifts = f['redshift'][:] if 'redshift' in f else None
            ra = f['ra'][:] if 'ra' in f else None
            dec = f['dec'][:] if 'dec' in f else None
            
            if redshifts is None:
                raise ValueError(
                    "ERROR: No redshift data found in H5 file!\n"
                    "Real dataset must contain 'redshift' field.\n"
                    "Please check data file or use complete dataset with redshift data."
                )
            
            if ra is None or dec is None:
                raise ValueError(
                    "ERROR: No coordinate data found in H5 file!\n"
                    "Real dataset must contain 'ra' and 'dec' fields.\n"
                    "Please check data file or use complete dataset with coordinate data."
                )
            
            valid_mask = np.ones(len(images), dtype=bool)
            if redshifts is not None:
                nan_mask = np.isnan(redshifts)
                zero_mask = redshifts == 0
                negative_mask = redshifts < 0
                valid_mask = ~nan_mask & ~zero_mask & ~negative_mask
            
            images = images[valid_mask]
            labels = labels[valid_mask]
            redshifts = redshifts[valid_mask]
            ra = ra[valid_mask]
            dec = dec[valid_mask]
            
            if self.num_samples is not None and self.num_samples > 0:
                images = images[:self.num_samples]
                labels = labels[:self.num_samples]
                redshifts = redshifts[:self.num_samples]
                ra = ra[:self.num_samples]
                dec = dec[:self.num_samples]
            
            return images, labels, redshifts, ra, dec
    
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        redshift = self.redshifts[idx]
        ra = self.ra[idx]
        dec = self.dec[idx]
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
        else:
            label = torch.tensor(label, dtype=torch.long)
        
        if isinstance(redshift, np.ndarray):
            redshift = torch.from_numpy(redshift).float()
        else:
            redshift = torch.tensor(redshift, dtype=torch.float)
        
        if isinstance(ra, np.ndarray):
            ra = torch.from_numpy(ra).float()
        else:
            ra = torch.tensor(ra, dtype=torch.float)
        
        if isinstance(dec, np.ndarray):
            dec = torch.from_numpy(dec).float()
        else:
            dec = torch.tensor(dec, dtype=torch.float)
        
        if len(image.shape) == 3:  # (H, W, C)
            image = image.permute(2, 0, 1)  # (C, H, W)
        elif len(image.shape) == 4:  # (N, H, W, C)
            image = image[0].permute(2, 0, 1)
        image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        image = image / 255.0 if image.max() > 1.0 else image
        
        if self.use_augmentation:
            image = self._apply_augmentation(image)
        
        coordinates = torch.stack([ra, dec])
        
        return {
            'image': image,
            'coordinates': coordinates,
            'redshift': redshift,
            'label': label
        }
    
    def _apply_augmentation(self, image):
        if random.random() > 0.6:  
            noise_std = random.uniform(0.008, 0.025)
            noise = torch.randn_like(image) * noise_std
            image = image + noise
        
        if random.random() > 0.8:
            angle = random.uniform(-4.0, 4.0)
            scale = random.uniform(0.99, 1.01)
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad) * scale
            sin_a = math.sin(angle_rad) * scale
            theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32).unsqueeze(0)
            grid = torch.nn.functional.affine_grid(theta, image.unsqueeze(0).size(), align_corners=False)
            image = torch.nn.functional.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)
        
        if random.random() > 0.8:
            mean = image.mean()
            contrast_factor = random.uniform(0.985, 1.015)
            image = (image - mean) * contrast_factor + mean
        
        return torch.clamp(image, 0, 1)

def train_model(checkpoint_path=None, continue_epochs=None, seed=7, **config_overrides):

    set_seed(seed)
    print('=== Galaxy10 Real Data Training ===')
    print(f'Random seed: {seed}')
    print()
    
    config = {
        # Dataset configuration
        'h5_path': 'data/Galaxy10_DECals.h5',
        'num_samples': None,
        'image_size': 32,
        'val_split': 0.3,

        'batch_size': 512,
        'num_epochs': 120,
        'initial_lr': 1e-4,
        'max_lr': 0.001,
        'min_lr': 5e-6,
        'weight_decay': 5e-4,
        'optimizer': 'adamw',  # Keep AdamW (more stable, suitable for complex loss functions)
        
        # Training collapse detection and auto-rollback mechanism
        'enable_collapse_detection': True,  # Enable training collapse detection
        'train_acc_collapse_threshold': 0.10,  # Training accuracy drop threshold (10%): trigger rollback if drop exceeds 10%
        'collapse_check_window': 3,  # Check window: average training accuracy of recent 3 epochs
        'collapse_auto_rollback': True,  # Auto-rollback to best checkpoint
        'collapse_lr_reduction': 0.5,  # LR reduction factor after collapse (reduce by 50%)
        

        'warmup_epochs': 10,  # Warmup epochs
        'phi': 0.70,  # Keep UBA parameter unchanged
        'eta_min': 1e-6,
        
        # Data augmentation
        'use_augmentation': True,
        'num_workers': 12,  # Increased for 25 vCPU system (optimal: 8-16 for data loading)
        'prefetch_factor': 4,  # Increased for better data pipeline throughput
        'save_path': None,  # Will be set dynamically based on image_size later
        'save_every': 10,
        'seed': seed,
        
        # Ablation study configuration (default: full model)
        'use_task_relationship': True,   # Enable task relationship model
        'use_coord_encoding': True,     # Enable learnable coordinate encoding
        'use_hk_from_start': False,      # Enable HK from epoch 0 (False = enable after epoch 2)
        'disable_hk_completely': False,  # Completely disable HK distance (for ablation: WaveletMamba only)
    }
    
    if continue_epochs is not None:
        config['num_epochs'] = continue_epochs
    for key, value in config_overrides.items():
        if key in config:
            config[key] = value
    
    # Set save path based on image_size and ablation configuration
    image_size = config.get('image_size', 256)
    if config['save_path'] is None:
        # Check if this is an ablation experiment
        is_ablation = not config.get('use_task_relationship', True) or not config.get('use_coord_encoding', True) or config.get('disable_hk_completely', False)
        
        if is_ablation:
            # Ablation experiment naming: identify enabled components
            ablation_parts = []
            if config.get('use_task_relationship', True):
                ablation_parts.append('TR')
            if config.get('use_coord_encoding', True):
                ablation_parts.append('CE')
            if not config.get('disable_hk_completely', False):
                ablation_parts.append('HK')
            
            if ablation_parts:
                ablation_suffix = '_' + '+'.join(ablation_parts)
            else:
                ablation_suffix = '_WM'  # WaveletMamba only
            
            if image_size == 256:
                config['save_path'] = f'seed{seed}_ablation{ablation_suffix}_galaxy_model.pth'
            else:
                config['save_path'] = f'{image_size}x{image_size}_seed{seed}_ablation{ablation_suffix}_galaxy_model.pth'
        else:
            # Full model naming
            if image_size == 256:
                config['save_path'] = f'seed{seed}_galaxy_model.pth'
            else:
                config['save_path'] = f'{image_size}x{image_size}_seed{seed}_galaxy_model.pth'
    
    if not torch.cuda.is_available():
        raise RuntimeError("ERROR: CUDA is not available! This training script requires GPU. Please use a GPU-enabled environment.")
    
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Model will be saved to: {config['save_path']}")
    print(f"Checkpoint will be saved to: {config['save_path'].replace('.pth', '_checkpoint.pth')}")
    print()
    
    print(f"Device: {device}")
    print()
    print('=' * 70)
    print('Training Configuration')
    print('=' * 70)
    print(f'  Dataset path: {config["h5_path"]}')
    print(f'  Total samples: {config["num_samples"] if config["num_samples"] is not None else "All"}')
    print(f'  Image size: {config["image_size"]}×{config["image_size"]}')
    print(f'  Batch size: {config["batch_size"]}')
    print(f'  Number of epochs: {config["num_epochs"]}')
    print(f'  Initial learning rate: {config["initial_lr"]:.6f}')
    print(f'  Peak learning rate: {config["max_lr"]:.6f}')
    print(f'  Minimum learning rate: {config["min_lr"]:.6f}')
    print(f'  Weight decay: {config["weight_decay"]}')
    print(f'  Learning rate scheduler: UBA (Unified Budget-Aware)')
    print(f'  UBA parameter φ: {config.get("phi", 0.8):.2f}')
    print(f'  Warmup epochs: {config.get("warmup_epochs", 5)}')
    print('=' * 70)
    print()
    
    h5_path = config['h5_path']
    num_samples = config['num_samples']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    image_size = config['image_size']
    initial_lr = config['initial_lr']
    max_lr = config['max_lr']
    weight_decay = config['weight_decay']
    
    
    try:
        val_split = config.get('val_split', 0.3)
        train_dataset = Galaxy10Dataset(
            h5_path, num_samples, image_size, 
            split='train', 
            use_augmentation=True
        )
        train_dataset._val_split = val_split
        
        val_dataset = Galaxy10Dataset(
            h5_path, num_samples, image_size, 
            split='val', 
            use_augmentation=False
        )
        val_dataset._val_split = val_split
        
   
        train_labels_array = np.array(train_dataset.labels)
        class_counts = np.bincount(train_labels_array)
        total = len(train_labels_array)
        
     
        sample_weights = np.array([
            total / (len(class_counts) * class_counts[label]) 
            for label in train_labels_array
        ])
        
        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=weighted_sampler,
            num_workers=config['num_workers'], pin_memory=True, 
            prefetch_factor=config.get('prefetch_factor', 4),
            generator=generator, 
            worker_init_fn=lambda worker_id: set_seed(seed + worker_id),
            persistent_workers=True,
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=config['num_workers'], pin_memory=True, 
            prefetch_factor=config.get('prefetch_factor', 4),
            generator=generator, 
            worker_init_fn=lambda worker_id: set_seed(seed + worker_id),
            persistent_workers=True,
            collate_fn=custom_collate_fn
        )
        
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        total_size = train_size + val_size
        
        print(f'Dataset split info:')
        print(f'  Total samples: {total_size}')
        print(f'  Train set: {train_size} ({100*train_size/total_size:.1f}%)')
        print(f'  Val set: {val_size} ({100*val_size/total_size:.1f}%)')
        
        print(f'  Train set class distribution:')
        train_labels_array = np.array(train_dataset.labels)
        unique, counts = np.unique(train_labels_array, return_counts=True)
        for cls, count in zip(unique, counts):
            weight = sample_weights[train_labels_array == cls][0]
            print(f'    Class {cls}: {count} samples (weight: {weight:.2f}x)')
        print()
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Ablation study configuration
    use_task_relationship = config.get('use_task_relationship', True)
    use_coord_encoding = config.get('use_coord_encoding', True)
    use_hk_from_start = config.get('use_hk_from_start', False)  # If True, enable HK from epoch 0
    disable_hk_completely = config.get('disable_hk_completely', False)  # Completely disable HK distance
    
    model = GalaxyModel(
        num_classes=10,
        device='cuda',
        use_morph=True,
        use_wavelet_mamba=True,  # Always True (core feature extractor)
        use_task_relationship=use_task_relationship,
        use_coord_encoding=use_coord_encoding
    )
    model = model.to(device)
    model.train()
    
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    
    # Print ablation study configuration
    print("=" * 70)
    print("Ablation Study Configuration:")
    print("=" * 70)
    print(f"  WaveletMamba: {'Enabled' if True else 'Disabled'} (Core feature extractor, always enabled)")
    print(f"  Task Relationship: {'Enabled' if use_task_relationship else 'Disabled'}")
    print(f"  Coord Encoding: {'Enabled' if use_coord_encoding else 'Disabled'}")
    if disable_hk_completely:
        print(f"  HK Distance: Completely Disabled (for WaveletMamba-only ablation)")
    elif use_hk_from_start:
        print(f"  HK Distance: Enabled from Epoch 0")
    else:
        print(f"  HK Distance: Enabled from Epoch 3 (default)")
    print("=" * 70)
    print()
    
    param_info = count_parameters(model)
    print(f"Model parameters: {param_info['total_params_M']:.2f}M (trainable: {param_info['trainable_params_M']:.2f}M)")
    print()
    
    total_steps = num_epochs * len(train_loader)
    warmup_epochs_config = config.get('warmup_epochs', 10)
    warmup_steps = warmup_epochs_config * len(train_loader)
    
    optimizer_type = config.get('optimizer', 'adamw').lower()
    if optimizer_type == 'sgd':
        optimizer_lr = max_lr
        momentum = config.get('momentum', 0.9)
        nesterov = config.get('nesterov', True)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    else:
        optimizer_lr = max_lr
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=1e-8
        )
    
    phi = config.get('phi', 0.8)
    min_lr = config.get('min_lr', 1e-6)
    
    scheduler = UBAScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        max_lr=max_lr,
        min_lr=min_lr,
        initial_lr=initial_lr,
        phi=phi,
        verbose=False,
        enable_adaptive_phi=True,
        base_phi=phi,
        enable_val_feedback=True
    )
    current_cls_weight = 0.90
    current_redshift_weight = 0.10
    model.loss_function.set_weights(current_cls_weight, current_redshift_weight)
    
    fixed_reg_weights = {'lambda_vib': 0.25, 'lambda_hk': 0.035, 'lsi_weight': 0.12}
    model.loss_function.set_regularization_weights(
        lambda_vib=fixed_reg_weights['lambda_vib'],
        lambda_hk=fixed_reg_weights['lambda_hk'],
        lsi_weight=fixed_reg_weights['lsi_weight']
    )
    
    start_epoch = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    detailed_losses = {
        'cls_losses': [],
        'redshift_losses': [],
        'hk_losses': [],
        'vib_kl_losses': [],
        'lsi_losses': []
    }
    
    training_history = {
        'learning_rates': [],
        'epoch_times': [],
        'grad_norms': []
    }
    
    validation_details = {
        'log_mses': [],
        'train_log_mses': [],
        'per_class_accuracies': [],
        'redshift_stats': []
    }
    
    best_epoch = 0
    best_model_state = None
    
    if checkpoint_path is not None:
        print(f'\nLoading checkpoint: {checkpoint_path}')
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            has_training_history = 'train_losses' in checkpoint or 'train_accuracies' in checkpoint
            is_weight_only = 'model_state_dict' in checkpoint and not has_training_history
            
            if is_weight_only:
                checkpoint_file_path = checkpoint_path.replace('.pth', '_checkpoint.pth')
                
                possible_paths = [
                    checkpoint_file_path,
                    os.path.join(os.path.dirname(checkpoint_path), 'galaxy_model_checkpoint.pth'),
                    checkpoint_path.replace('.pth', '_checkpoint.pth'),
                ]
                
                if not os.path.isabs(checkpoint_path):
                    possible_paths.append('galaxy_model_checkpoint.pth')
                
                found_checkpoint = None
                for path in possible_paths:
                    if os.path.exists(path):
                        found_checkpoint = path
                        break
                
                if found_checkpoint:
                    try:
                        full_checkpoint = torch.load(found_checkpoint, map_location=device, weights_only=False)
                        checkpoint = full_checkpoint
                    except Exception as e:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        start_epoch = 0
                        checkpoint = None
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    start_epoch = 0
                    checkpoint = None
            
            if checkpoint is not None:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'best_model_state' in checkpoint:
                    model.load_state_dict(checkpoint['best_model_state'])
                
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint and hasattr(scheduler, 'load_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                if start_epoch > 0 and 'train_loader' in locals() and train_loader is not None:
                    start_steps = start_epoch * len(train_loader)
                    scheduler.last_epoch = start_steps - 1
                    current_lr_before = optimizer.param_groups[0]['lr']
                    
                    lr_reduction_factor = 0.85
                    reduced_lr = current_lr_before * lr_reduction_factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = reduced_lr
                    
                    scheduler.step()
                    current_lr_after = optimizer.param_groups[0]['lr']
                    
                    if 'training_history' in checkpoint and 'learning_rates' in checkpoint['training_history']:
                        prev_lrs = checkpoint['training_history']['learning_rates']
                        if len(prev_lrs) > 0:
                            expected_lr = prev_lrs[-1]
                            final_lr = expected_lr * lr_reduction_factor
                            if abs(current_lr_after - final_lr) > 1e-5:
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = final_lr
                
                if 'train_losses' in checkpoint:
                    train_losses = checkpoint['train_losses'].copy()
                    
                if 'train_accuracies' in checkpoint:
                    train_accuracies = checkpoint['train_accuracies'].copy()
                    
                if 'val_losses' in checkpoint:
                    val_losses = checkpoint['val_losses'].copy()
                    
                if 'val_accuracies' in checkpoint:
                    val_accuracies = checkpoint['val_accuracies'].copy()
                
                if 'detailed_losses' in checkpoint:
                    detailed_losses = checkpoint['detailed_losses'].copy()
                if 'training_history' in checkpoint:
                    training_history = checkpoint['training_history'].copy()
                if 'validation_details' in checkpoint:
                    validation_details = checkpoint['validation_details'].copy()
                
                if 'best_epoch' in checkpoint:
                    best_epoch = checkpoint['best_epoch']
                if 'best_model_state' in checkpoint:
                    best_model_state = checkpoint['best_model_state'].copy()
                
                if 'train_accuracies' in checkpoint and len(checkpoint['train_accuracies']) > 0:
                    start_epoch = len(checkpoint['train_accuracies'])
                elif 'val_accuracies' in checkpoint and len(checkpoint['val_accuracies']) > 0:
                    start_epoch = len(checkpoint['val_accuracies'])
                elif 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                else:
                    start_epoch = 0
                
                if continue_epochs is not None:
                    num_epochs = continue_epochs
                else:
                    remaining_epochs = config['num_epochs'] - start_epoch
                    if remaining_epochs > 0:
                        num_epochs = remaining_epochs
                    else:
                        num_epochs = config['num_epochs']
                
                print(f'Resuming from Epoch {start_epoch + 1}')
            
        except Exception as e:
            print(f'Failed to load checkpoint: {e}')
            start_epoch = 0
    
    best_epoch = 0
    best_model_state = None
    
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    

    for epoch in range(num_epochs):
        current_epoch = start_epoch + epoch + 1
        print(f'Epoch {current_epoch} (Training {epoch + 1}/{num_epochs}):')
        
        model.loss_function.set_training_stage(current_epoch, num_epochs)
        
        # HK distance control for ablation study
        if disable_hk_completely:
            # Completely disable HK distance (for ablation: WaveletMamba only)
            model.loss_function.set_hk_enabled(False)
        elif use_hk_from_start:
            # Enable HK from the beginning (for ablation: +HK configuration)
            model.loss_function.set_hk_enabled(True)
        else:
            # Default: enable HK after epoch 2 (for stability)
            if epoch == 2:
                model.loss_function.set_hk_enabled(True)
            elif epoch < 2:
                model.loss_function.set_hk_enabled(False)
        
        
        epoch_start_time = time.time()
        
        model.train()
        epoch_train_loss_tensor = torch.tensor(0.0, device=device)
        epoch_train_correct_tensor = torch.tensor(0, dtype=torch.long, device=device)
        epoch_train_total = 0
        epoch_loss_dict = {
            'cls_loss': torch.tensor(0.0, device=device),
            'redshift_loss': torch.tensor(0.0, device=device),
            'hk_loss': torch.tensor(0.0, device=device),
            'vib_kl': torch.tensor(0.0, device=device),
            'lsi_enhancement': torch.tensor(0.0, device=device),
            'grad_norm': torch.tensor(0.0, device=device)
        }
        epoch_batch_count = 0
        train_redshifts_pred_list = []
        train_redshifts_true_list = []
        
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device, non_blocking=True)
            coordinates = batch['coordinates'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            redshifts = batch['redshift'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(images, coordinates)
                loss, loss_dict = model.compute_loss(outputs, {
                    'labels': labels,
                    'redshift': redshifts
                })
            
            scaler.scale(loss).backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            epoch_loss_dict['grad_norm'] += grad_norm.detach()
            scheduler.update_grad_norm(grad_norm.item())
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Clear cache periodically to save memory
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
            
            # Extract all loss components from loss_dict
            epoch_loss_dict['cls_loss'] += loss_dict.get('cls_loss', torch.tensor(0.0, device=device)).detach() if isinstance(loss_dict.get('cls_loss'), torch.Tensor) else torch.tensor(0.0, device=device)
            epoch_loss_dict['redshift_loss'] += loss_dict.get('redshift_loss', torch.tensor(0.0, device=device)).detach() if isinstance(loss_dict.get('redshift_loss'), torch.Tensor) else torch.tensor(0.0, device=device)
            epoch_loss_dict['hk_loss'] += loss_dict.get('hk_loss', torch.tensor(0.0, device=device)).detach() if isinstance(loss_dict.get('hk_loss'), torch.Tensor) else torch.tensor(0.0, device=device)
            epoch_loss_dict['vib_kl'] += loss_dict.get('vib_kl', torch.tensor(0.0, device=device)).detach() if isinstance(loss_dict.get('vib_kl'), torch.Tensor) else torch.tensor(0.0, device=device)
            epoch_loss_dict['lsi_enhancement'] += loss_dict.get('lsi_enhancement', torch.tensor(0.0, device=device)).detach() if isinstance(loss_dict.get('lsi_enhancement'), torch.Tensor) else torch.tensor(0.0, device=device)
            
            # Accumulate loss for logging
            epoch_train_loss_tensor += loss.detach()
            epoch_batch_count += 1
            predictions = torch.argmax(outputs['cls_logits'], dim=1)
            epoch_train_correct_tensor += (predictions == labels).sum().detach()
            epoch_train_total += labels.size(0)
            
            train_redshift_pred = get_redshift_output(outputs, redshifts)
            if train_redshift_pred.dim() > 1:
                train_redshift_pred = train_redshift_pred.squeeze()
                if train_redshift_pred.dim() > 1:
                    train_redshift_pred = train_redshift_pred.view(-1)
            if train_redshift_pred.dim() == 0:
                train_redshift_pred = train_redshift_pred.unsqueeze(0)
            
            redshifts_flat = redshifts.squeeze()
            if redshifts_flat.dim() > 1:
                redshifts_flat = redshifts_flat.view(-1)
            if redshifts_flat.dim() == 0:
                redshifts_flat = redshifts_flat.unsqueeze(0)
            
            train_redshifts_pred_list.append(train_redshift_pred.detach().cpu())
            train_redshifts_true_list.append(redshifts_flat.detach().cpu())
        
        train_loss = (epoch_train_loss_tensor / len(train_loader)).item()
        train_acc = (epoch_train_correct_tensor.float() / epoch_train_total).item()
        
        if len(train_redshifts_pred_list) > 0:
            train_redshifts_pred = torch.cat(train_redshifts_pred_list, dim=0).numpy()
            train_redshifts_true = torch.cat(train_redshifts_true_list, dim=0).numpy()
            train_log_mse = compute_log_mse_unified(
                train_redshifts_pred,
                train_redshifts_true,
                min_val=1e-6,
                max_val=None
            )
            if isinstance(train_log_mse, torch.Tensor):
                train_log_mse = train_log_mse.item()
        else:
            train_log_mse = 0.0
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        validation_details['train_log_mses'].append(train_log_mse)
        
        collapse_detected = False
        if config.get('enable_collapse_detection', False) and len(train_accuracies) >= config.get('collapse_check_window', 3):
            recent_window = config.get('collapse_check_window', 3)
            recent_train_acc = np.mean(train_accuracies[-recent_window:])
            
            if len(train_accuracies) > recent_window:
                best_prev_train_acc = max(train_accuracies[:-recent_window])
                train_acc_drop = best_prev_train_acc - recent_train_acc
                collapse_threshold = config.get('train_acc_collapse_threshold', 0.10)
                
                if train_acc_drop > collapse_threshold:
                    collapse_detected = True
                    print(f'\nTraining collapse detected')
                    print(f'  Historical best train accuracy: {best_prev_train_acc:.2%}')
                    print(f'  Recent {recent_window} epochs avg train accuracy: {recent_train_acc:.2%}')
                    print(f'  Train accuracy drop: {train_acc_drop:.2%} (threshold: {collapse_threshold:.2%})')
                    
                    if config.get('collapse_auto_rollback', True) and best_model_state is not None and best_epoch > 0:
                        print(f'  Auto-rolling back to best epoch {best_epoch} model state...')
                        model.load_state_dict(best_model_state)
                        
                        current_lr = optimizer.param_groups[0]['lr']
                        lr_reduction = config.get('collapse_lr_reduction', 0.5)
                        new_lr = current_lr * lr_reduction
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f'  Learning rate reduced: {current_lr:.6f} → {new_lr:.6f} (reduce {(1-lr_reduction)*100:.0f}%)')
                        
                        if len(train_accuracies) > best_epoch:
                            train_losses = train_losses[:best_epoch]
                            train_accuracies = train_accuracies[:best_epoch]
                            val_losses = val_losses[:best_epoch]
                            val_accuracies = val_accuracies[:best_epoch]
                            if 'learning_rates' in training_history:
                                training_history['learning_rates'] = training_history['learning_rates'][:best_epoch]
                            if 'grad_norms' in training_history:
                                training_history['grad_norms'] = training_history['grad_norms'][:best_epoch]
                            if 'log_mses' in validation_details:
                                validation_details['log_mses'] = validation_details['log_mses'][:best_epoch]
                            if 'per_class_accuracies' in validation_details:
                                validation_details['per_class_accuracies'] = validation_details['per_class_accuracies'][:best_epoch]
                        
                        print(f'  Rolled back to Epoch {best_epoch}, continuing training...\n')
                    else:
                        print(f'  Auto-rollback not enabled or best model state not found, continuing training...\n')
        
        if epoch_batch_count > 0:
            detailed_losses['cls_losses'].append((epoch_loss_dict['cls_loss'] / epoch_batch_count).item())
            detailed_losses['redshift_losses'].append((epoch_loss_dict['redshift_loss'] / epoch_batch_count).item())
            detailed_losses['hk_losses'].append((epoch_loss_dict['hk_loss'] / epoch_batch_count).item())
            detailed_losses['vib_kl_losses'].append((epoch_loss_dict['vib_kl'] / epoch_batch_count).item())
            detailed_losses['lsi_losses'].append((epoch_loss_dict['lsi_enhancement'] / epoch_batch_count).item())
            training_history['grad_norms'].append((epoch_loss_dict['grad_norm'] / epoch_batch_count).item())
        
        model.eval()
        epoch_val_loss_tensor = torch.tensor(0.0, device=device)
        epoch_val_correct_tensor = torch.tensor(0, dtype=torch.long, device=device)
        epoch_val_total = 0
        val_redshifts_pred_tensor = []
        val_redshifts_true_tensor = []
        all_val_predictions = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                coordinates = batch['coordinates'].to(device)
                labels = batch['label'].to(device)
                redshifts = batch['redshift'].to(device)
                
                outputs = model(images, coordinates)
                loss, loss_dict = model.compute_loss(outputs, {
                    'labels': labels,
                    'redshift': redshifts
                })
                
                epoch_val_loss_tensor += loss.detach()
                predictions = torch.argmax(outputs['cls_logits'], dim=1)
                epoch_val_correct_tensor += (predictions == labels).sum().detach()
                epoch_val_total += labels.size(0)
                
                redshift_output = get_redshift_output(outputs, redshifts)
                if redshift_output.dim() > 1:
                    redshift_output = redshift_output.squeeze()
                    if redshift_output.dim() > 1:
                        redshift_output = redshift_output.view(-1)
                if redshift_output.dim() == 0:
                    redshift_output = redshift_output.unsqueeze(0)
                
                redshifts_flattened = redshifts.squeeze()
                if redshifts_flattened.dim() > 1:
                    redshifts_flattened = redshifts_flattened.view(-1)
                if redshifts_flattened.dim() == 0:
                    redshifts_flattened = redshifts_flattened.unsqueeze(0)
                
                val_redshifts_pred_tensor.append(redshift_output.detach())
                val_redshifts_true_tensor.append(redshifts_flattened.detach())
                all_val_predictions.append(predictions.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())
        
        val_loss = (epoch_val_loss_tensor / len(val_loader)).item()
        val_acc = (epoch_val_correct_tensor.float() / epoch_val_total).item()
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        val_redshifts_pred = torch.cat(val_redshifts_pred_tensor, dim=0).cpu().numpy()
        val_redshifts_true = torch.cat(val_redshifts_true_tensor, dim=0).cpu().numpy()
        
        log_mse = compute_log_mse_unified(
            val_redshifts_pred,
            val_redshifts_true,
            min_val=1e-6,
            max_val=None
        )
        if isinstance(log_mse, torch.Tensor):
            log_mse = log_mse.item()
        composite_score = 0.4 * val_acc + 0.6 * (1 - log_mse)
        if isinstance(composite_score, torch.Tensor):
            composite_score = composite_score.item()
        
        scheduler.update_validation_loss(val_loss)
        
        new_lr = optimizer.param_groups[0]['lr']
        current_phi = scheduler.get_current_phi() if hasattr(scheduler, 'get_current_phi') else config.get('phi', 0.8)
        
        epoch_time = time.time() - epoch_start_time
        
        training_history['learning_rates'].append(new_lr)
        training_history['epoch_times'].append(epoch_time)
        validation_details['log_mses'].append(log_mse)
        
        if len(all_val_predictions) > 0:
            all_val_predictions = np.concatenate(all_val_predictions)
            all_val_labels = np.concatenate(all_val_labels)
            per_class_acc = []
            for cls in range(model.num_classes):
                cls_mask = all_val_labels == cls
                if cls_mask.sum() > 0:
                    cls_acc = (all_val_predictions[cls_mask] == cls).sum() / cls_mask.sum()
                    per_class_acc.append(float(cls_acc))
                else:
                    per_class_acc.append(0.0)
            validation_details['per_class_accuracies'].append(per_class_acc)
        else:
            validation_details['per_class_accuracies'].append([0.0] * model.num_classes)
        
        redshift_stats = {
            'mean': float(np.mean(val_redshifts_pred)),
            'std': float(np.std(val_redshifts_pred)),
            'min': float(np.min(val_redshifts_pred)),
            'max': float(np.max(val_redshifts_pred))
        }
        validation_details['redshift_stats'].append(redshift_stats)
        
        if len(val_accuracies) == 1 or val_acc > max(val_accuracies[:-1]):
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
        
        current_reg_vib = model.loss_function.lambda_vib
        current_reg_hk = model.loss_function.lambda_hk
        current_reg_lsi = model.loss_function.lsi_weight
        
        print(f'  Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}')
        print(f'  Train Log-MSE: {train_log_mse:.6f}, Val Log-MSE: {log_mse:.6f}, Composite Score: {composite_score:.4f}')
        print(f'  LR: {new_lr:.6f}, UBA φ: {current_phi:.3f}')
        print(f'  Regularization weights: VIB={current_reg_vib:.3f}, HK={current_reg_hk:.3f}, LSI={current_reg_lsi:.3f}')
        print(f'  Time: {epoch_time:.2f}s')
        print()
    
    print('=' * 70)
    print('Training Summary:')
    print('=' * 70)
    print(f'Best Val Accuracy: {max(val_accuracies):.2%}')
    print()
    
    model_save_path = config['save_path']
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    final_model_state = best_model_state if best_model_state is not None else model.state_dict()
    
    total_epochs = start_epoch + epoch + 1
    
    torch.save(final_model_state, model_save_path)
    print(f'Model weights saved to: {model_save_path}')
    
    checkpoint_path = model_save_path.replace('.pth', '_checkpoint.pth')
    checkpoint_dict = {
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses.copy(),
        'train_accuracies': train_accuracies.copy(),
        'val_losses': val_losses.copy(),
        'val_accuracies': val_accuracies.copy(),
        'final_train_acc': train_accuracies[-1] if train_accuracies else 0.0,
        'final_val_acc': val_accuracies[-1] if val_accuracies else 0.0,
        'epoch': total_epochs,
        'start_epoch': start_epoch,
        'current_training_epochs': epoch + 1,
        'best_epoch': best_epoch if best_epoch is not None else total_epochs,
        'best_val_acc': max(val_accuracies) if val_accuracies else 0.0,
        'detailed_losses': detailed_losses.copy() if isinstance(detailed_losses, dict) else detailed_losses,
        'training_history': {
            'learning_rates': training_history['learning_rates'].copy() if 'learning_rates' in training_history else [],
            'epoch_times': training_history['epoch_times'].copy() if 'epoch_times' in training_history else [],
            'grad_norms': training_history['grad_norms'].copy() if 'grad_norms' in training_history else [],
        },
        'validation_details': {
            'log_mses': validation_details['log_mses'].copy() if 'log_mses' in validation_details else [],
            'per_class_accuracies': validation_details['per_class_accuracies'].copy() if 'per_class_accuracies' in validation_details else [],
            'redshift_stats': validation_details['redshift_stats'].copy() if 'redshift_stats' in validation_details else [],
        },
        'config': config,
    }
    
    if best_model_state is not None:
        checkpoint_dict['best_model_state'] = best_model_state.copy()
    
    if scheduler is not None:
        try:
            checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()
        except:
            pass
    
    total_training_time_s = sum(training_history.get('epoch_times', []))
    total_training_time_h = total_training_time_s / 3600.0
    
    print('Measuring inference speed...')
    inference_stats = measure_inference_speed(model, device, image_size=config['image_size'])
    
    checkpoint_dict['model_stats'] = {
        'num_params_M': param_info['total_params_M'],
        'num_trainable_params_M': param_info['trainable_params_M'],
        'total_params': param_info['total_params'],
        'trainable_params': param_info['trainable_params'],
        'total_training_time_h': total_training_time_h,
        'total_training_time_s': total_training_time_s,
        'inference_fps': inference_stats['fps'],
        'inference_time_ms': inference_stats['avg_time_ms'],
        'image_size': config['image_size']
    }
    
    torch.save(checkpoint_dict, checkpoint_path)
    print(f'Full checkpoint saved to: {checkpoint_path}')
    print(f'Training history: {len(train_accuracies)} epochs (from epoch {start_epoch + 1} to epoch {total_epochs})')
    if val_accuracies:
        print(f'Best epoch: {best_epoch}, Best validation accuracy: {max(val_accuracies):.2%}')
    print(f'Total training time: {total_training_time_h:.2f} hours ({total_training_time_s/60:.1f} minutes)')
    print(f'Inference speed: {inference_stats["fps"]:.2f} FPS ({inference_stats["avg_time_ms"]:.2f} ms/sample)')
    print()
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Galaxy Model with ablation study support')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_task_relationship', type=str, default='True',
                        choices=['True', 'False', 'true', 'false'],
                        help='Enable task relationship model (default: True)')
    parser.add_argument('--use_coord_encoding', type=str, default='True',
                        choices=['True', 'False', 'true', 'false'],
                        help='Enable learnable coordinate encoding (default: True)')
    parser.add_argument('--use_hk_from_start', type=str, default='False',
                        choices=['True', 'False', 'true', 'false'],
                        help='Enable HK distance from epoch 0 (default: False, enables after epoch 2)')
    parser.add_argument('--disable_hk_completely', type=str, default='True',
                        choices=['True', 'False', 'true', 'false'],
                        help='Completely disable HK distance (for ablation: WaveletMamba only)')
    parser.add_argument('--image_size', type=int, default=32, help='Image size (default: from config)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size (default: from config)')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--continue_epochs', type=int, default=None, help='Number of epochs to continue training')
    
    args = parser.parse_args()
    
    # Convert string to boolean
    use_task_relationship = args.use_task_relationship.lower() == 'true'
    use_coord_encoding = args.use_coord_encoding.lower() == 'true'
    use_hk_from_start = args.use_hk_from_start.lower() == 'true'
    disable_hk_completely = args.disable_hk_completely.lower() == 'true'
    
    # Prepare config overrides
    config_overrides = {
        'use_task_relationship': use_task_relationship,
        'use_coord_encoding': use_coord_encoding,
        'use_hk_from_start': use_hk_from_start,
        'disable_hk_completely': disable_hk_completely
    }
    
    # Add image_size and batch_size if provided
    if args.image_size is not None:
        config_overrides['image_size'] = args.image_size
    if args.batch_size is not None:
        config_overrides['batch_size'] = args.batch_size
    
    train_model(
        checkpoint_path=args.checkpoint_path,
        continue_epochs=args.continue_epochs,
        seed=args.seed,
        **config_overrides
    )
