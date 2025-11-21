"""
HK Distance Training Script for DR17 Dataset
- Uses magnitude channels (mag_g, mag_i, mag_r, mag_u, mag_z) instead of RGB
- Only redshift prediction task (no classification)
- Uses HK distance loss
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from galaxy10_training import (
    set_seed, 
    compute_log_mse_unified, get_redshift_output,
    count_parameters, measure_inference_speed
)

try:
    from hk_distance_experiments.resnet34_hk_magnitude import ResNet34HKMagnitude
    from improved_model import UnifiedLossFunction, HellingerKantorovichDistance
except ImportError:
    from resnet34_hk_magnitude import ResNet34HKMagnitude
    from improved_model import UnifiedLossFunction, HellingerKantorovichDistance

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
import time
import argparse
from sklearn.model_selection import train_test_split
import h5py
from torchvision.transforms import Resize
import torchvision.transforms.functional as F


class DR17MagnitudeDataset(Dataset):
    """Dataset for DR17 with magnitude channels"""
    
    def __init__(self, images, magnitudes, redshifts, ra, dec, 
                 image_size=64, use_augmentation=False):
        """
        Args:
            images: (N, H, W) single channel grayscale images (not RGB)
                     DR17 dataset uses grayscale images, not color images
            magnitudes: (N, 5) magnitude values [mag_g, mag_i, mag_r, mag_u, mag_z]
                        All values should be valid (already filtered during data loading)
            redshifts: (N,) redshift values (all > 0, already filtered)
            ra: (N,) RA coordinates
            dec: (N,) DEC coordinates
            image_size: target image size
            use_augmentation: whether to use data augmentation
        """
        self.images = torch.from_numpy(images).float()
        self.magnitudes = torch.from_numpy(magnitudes).float()  # [mag_g, mag_i, mag_r, mag_u, mag_z]
        self.redshifts = torch.from_numpy(redshifts).float()
        self.coords = torch.stack([
            torch.from_numpy(ra).float(),
            torch.from_numpy(dec).float()
        ], dim=1)
        
        # Compute color indices from magnitudes for color-aware weighting
        # color_features: [g-r, g-i] (main color indices)
        mag_g = self.magnitudes[:, 0]
        mag_i = self.magnitudes[:, 1]
        mag_r = self.magnitudes[:, 2]
        self.color_features = torch.stack([
            mag_g - mag_r,  # g-r
            mag_g - mag_i,  # g-i
        ], dim=1)
        
        # Resize images if needed and ensure 4D format (N, 1, H, W)
        if len(self.images.shape) == 3:  # (N, H, W)
            self.images = self.images.unsqueeze(1)  # (N, 1, H, W)
        
        if self.images.shape[2] != image_size or self.images.shape[3] != image_size:
            # Resize to target size
            self.images = F.resize(self.images, (image_size, image_size))  # (N, 1, H, W)
        
        # Expand magnitude channels to match image spatial dimensions
        # Create 5-channel image: [image, mag_g, mag_i, mag_r, mag_u, mag_z]
        # Each magnitude is expanded to (H, W) spatial dimensions
        H, W = self.images.shape[2], self.images.shape[3]  # Get spatial dimensions
        mag_expanded = self.magnitudes.unsqueeze(-1).unsqueeze(-1)  # (N, 5, 1, 1)
        mag_expanded = mag_expanded.expand(-1, -1, H, W)  # (N, 5, H, W)
        
        # Normalize magnitudes (typical range: 0-40 for DR17, normalize to [0, 1])
        # Using wider range to accommodate all valid magnitudes
        mag_min, mag_max = 0.0, 40.0
        mag_expanded = (mag_expanded - mag_min) / (mag_max - mag_min)
        mag_expanded = torch.clamp(mag_expanded, 0, 1)
        
        # Concatenate: [original_image, mag_g, mag_i, mag_r, mag_u, mag_z]
        self.images = torch.cat([self.images, mag_expanded], dim=1)  # (N, 6, H, W)
        
        self.use_augmentation = use_augmentation
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.use_augmentation:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                image = torch.flip(image, [2])
            # Random vertical flip
            if np.random.rand() > 0.5:
                image = torch.flip(image, [1])
        
        return {
            'image': image,
            'coordinates': self.coords[idx],
            'redshift': self.redshifts[idx],
            'color_features': self.color_features[idx]  # For color-aware weighting
        }


def train_hk_experiment(experiment_type: str, seed: int, **config_overrides):
    """
    Train HK distance experiment on DR17 dataset
    
    Args:
        experiment_type: 'baseline', 'color_aware', 'hk', or 'color_aware_hk'
            - 'baseline': Only MSE loss
            - 'color_aware': MSE + color-aware weighting
            - 'hk': MSE + HK distance
            - 'color_aware_hk': Full complementary mechanism (color-aware + HK)
        seed: Random seed
        **config_overrides: Configuration overrides
    """
    if experiment_type not in ['baseline', 'color_aware', 'hk', 'color_aware_hk']:
        raise ValueError(f"Unknown experiment type: {experiment_type}. "
                        f"Must be one of: ['baseline', 'color_aware', 'hk', 'color_aware_hk']")
    
    set_seed(seed)
    print('=' * 70)
    print(f'Color-Aware & HK Distance Experiment: {experiment_type}')
    print(f'Dataset: DR17 (magnitude channels)')
    print(f'Random seed: {seed}')
    print('=' * 70)
    print()
    
    # Base configuration
    config = {
        'h5_path': 'data/dr17_dataset.h5',
        'num_samples': None,  # Use all samples
        'image_size': 64,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'batch_size': 512,  # Increased from 256
        'num_epochs': 100,
        'initial_lr': 1e-4,
        'weight_decay': 5e-4,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',  # 'cosine' or 'step' or None
        'lr_gamma': 0.1,  # For step scheduler
        'lr_step_size': 20,  # For step scheduler (epochs)
        'num_workers': 4,  # Reduced from 12 to avoid OOM (each worker copies dataset)
        'prefetch_factor': 2,  # Reduced from 4 to lower memory usage
        'seed': seed,
        'experiment_type': experiment_type,
        'early_stopping': True,
        'early_stopping_patience': 20,  # For 100 epochs
        'early_stopping_min_delta': 0.001,  # For log-MSE
        'early_stopping_monitor': 'val_log_mse',
        'early_stopping_mode': 'min',  # Minimize log-MSE
    }
    
    # Set save path based on experiment type
    exp_prefix_map = {
        'baseline': 'baseline_mse',
        'color_aware': 'color_aware',
        'hk': 'hk_distance',
        'color_aware_hk': 'color_aware_hk'
    }
    config['save_path'] = f'64x64_seed{seed}_{exp_prefix_map[experiment_type]}_dr17_resnet34_model.pth'
    
    config.update(config_overrides)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    device = torch.device('cuda:0')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load data from DR17 dataset
    print("Loading DR17 data...")
    with h5py.File(config['h5_path'], 'r') as f:
        all_images = f['images'][:]  # (N, 256, 256)
        all_redshifts = f['meta/redshift'][:]  # (N,)
        all_ra = f['meta/ra'][:]  # (N,)
        all_dec = f['meta/dec'][:]  # (N,)
        
        # Load magnitude channels with validation
        print("Loading magnitude channels...")
        mag_keys = ['meta/mag_g', 'meta/mag_i', 'meta/mag_r', 'meta/mag_u', 'meta/mag_z']
        magnitudes_list = []
        
        for key in mag_keys:
            if key in f:
                mag_data = f[key][:]
                print(f"  {key}: shape={mag_data.shape}, range=[{np.nanmin(mag_data):.2f}, {np.nanmax(mag_data):.2f}]")
                magnitudes_list.append(mag_data)
            else:
                raise ValueError(f"Missing magnitude channel: {key}")
        
        # Stack magnitudes: (N, 5)
        all_magnitudes = np.stack(magnitudes_list, axis=1)
    
    print(f"Loaded {len(all_images)} samples")
    print(f"Image shape: {all_images.shape} (single channel grayscale)")
    print(f"  Image range: [{np.nanmin(all_images):.2f}, {np.nanmax(all_images):.2f}]")
    print(f"Magnitude shape: {all_magnitudes.shape}")
    print(f"Redshift range: [{np.nanmin(all_redshifts):.6f}, {np.nanmax(all_redshifts):.6f}]")
    print()
    
    # Filter invalid data (redshifts and magnitudes)
    print("Filtering invalid data...")
    
    # Check for NaN/inf in magnitudes
    mag_valid_mask = ~(np.isnan(all_magnitudes).any(axis=1) | np.isinf(all_magnitudes).any(axis=1))
    print(f"  Magnitudes (NaN/Inf): {mag_valid_mask.sum()}/{len(all_magnitudes)} valid ({(1-mag_valid_mask.sum()/len(all_magnitudes))*100:.2f}% invalid)")
    
    # Check for invalid magnitude values (e.g., -9999.00 is a common missing value marker in astronomy)
    # Valid magnitudes are typically in range [0, 40] for SDSS/DR17
    # Exclude extreme negative values (missing data markers)
    # Note: We require ALL 5 magnitude channels to be valid (no partial missing data)
    mag_range_mask = (
        (all_magnitudes >= 0.0).all(axis=1) & 
        (all_magnitudes <= 40.0).all(axis=1)
    )
    invalid_mag_count = (~mag_range_mask).sum()
    if invalid_mag_count > 0:
        # Show examples of invalid magnitudes
        invalid_indices = np.where(~mag_range_mask)[0][:5]  # First 5 examples
        print(f"  Magnitude ranges: {mag_range_mask.sum()}/{len(all_magnitudes)} in valid range [0, 40]")
        print(f"  Invalid examples (first 5):")
        for idx in invalid_indices:
            invalid_mags = all_magnitudes[idx]
            invalid_channels = np.where((invalid_mags < 0.0) | (invalid_mags > 40.0))[0]
            mag_names = ['mag_g', 'mag_i', 'mag_r', 'mag_u', 'mag_z']
            invalid_names = [mag_names[i] for i in invalid_channels]
            print(f"    Sample {idx}: {invalid_names} = {invalid_mags[invalid_channels]}")
    else:
        print(f"  Magnitude ranges: {mag_range_mask.sum()}/{len(all_magnitudes)} in valid range [0, 40]")
    
    # Check for valid redshifts (only filter negative/invalid, no upper bound)
    redshift_valid_mask = (
        ~np.isnan(all_redshifts) & 
        ~np.isinf(all_redshifts) &
        (all_redshifts > 0)  # Only filter negative/invalid redshifts, no upper bound
    )
    print(f"  Redshifts: {redshift_valid_mask.sum()}/{len(all_redshifts)} valid (range: [{np.nanmin(all_redshifts[redshift_valid_mask]):.6f}, {np.nanmax(all_redshifts[redshift_valid_mask]):.6f}])")
    
    # Combined valid mask
    valid_mask = mag_valid_mask & mag_range_mask & redshift_valid_mask
    print(f"  Combined: {valid_mask.sum()}/{len(all_images)} valid samples ({(1-valid_mask.sum()/len(all_images))*100:.2f}% filtered out)")
    print()
    
    if valid_mask.sum() == 0:
        raise ValueError("No valid samples after filtering! Please check the data.")
    
    all_images = all_images[valid_mask]
    all_redshifts = all_redshifts[valid_mask]
    all_ra = all_ra[valid_mask]
    all_dec = all_dec[valid_mask]
    all_magnitudes = all_magnitudes[valid_mask]
    
    print(f"Final dataset: {len(all_images)} valid samples")
    print(f"  Magnitude ranges: g=[{all_magnitudes[:, 0].min():.2f}, {all_magnitudes[:, 0].max():.2f}], "
          f"r=[{all_magnitudes[:, 2].min():.2f}, {all_magnitudes[:, 2].max():.2f}]")
    print(f"  Redshift range: [{all_redshifts.min():.6f}, {all_redshifts.max():.6f}]")
    print()
    
    if config['num_samples'] is not None:
        indices = np.random.RandomState(seed=seed).choice(
            len(all_images), config['num_samples'], replace=False
        )
        all_images = all_images[indices]
        all_redshifts = all_redshifts[indices]
        all_ra = all_ra[indices]
        all_dec = all_dec[indices]
        all_magnitudes = all_magnitudes[indices]
    
    # Split data (no stratification needed since no classification)
    # Create indices array
    indices = np.arange(len(all_images))
    
    # First split: train (70%) vs temp (30%)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.3,
        random_state=seed
    )
    
    # Second split: temp -> val (20%) and test (10%)
    # Create new indices for temp_indices array
    temp_array_indices = np.arange(len(temp_indices))
    val_array_indices, test_array_indices = train_test_split(
        temp_array_indices,
        test_size=1/3,  # 10% / 30% = 1/3
        random_state=seed
    )
    
    # Map back to original indices
    val_indices = temp_indices[val_array_indices]
    test_indices = temp_indices[test_array_indices]
    
    # Split data
    train_images = all_images[train_indices]
    train_redshifts = all_redshifts[train_indices]
    train_ra = all_ra[train_indices]
    train_dec = all_dec[train_indices]
    train_magnitudes = all_magnitudes[train_indices]
    
    val_images = all_images[val_indices]
    val_redshifts = all_redshifts[val_indices]
    val_ra = all_ra[val_indices]
    val_dec = all_dec[val_indices]
    val_magnitudes = all_magnitudes[val_indices]
    
    test_images = all_images[test_indices]
    test_redshifts = all_redshifts[test_indices]
    test_ra = all_ra[test_indices]
    test_dec = all_dec[test_indices]
    test_magnitudes = all_magnitudes[test_indices]
    
    # Verify data split
    total_samples = len(all_images)
    train_count = len(train_indices)
    val_count = len(val_indices)
    test_count = len(test_indices)
    
    print(f'Data split verification:')
    print(f'  Total samples: {total_samples}')
    print(f'  Train: {train_count} ({train_count/total_samples*100:.2f}%)')
    print(f'  Val: {val_count} ({val_count/total_samples*100:.2f}%)')
    print(f'  Test: {test_count} ({test_count/total_samples*100:.2f}%)')
    print(f'  Sum: {train_count + val_count + test_count} (should be {total_samples})')
    
    # Verify no overlap
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    if train_set & val_set:
        raise ValueError(f"Overlap between train and val: {len(train_set & val_set)} samples")
    if train_set & test_set:
        raise ValueError(f"Overlap between train and test: {len(train_set & test_set)} samples")
    if val_set & test_set:
        raise ValueError(f"Overlap between val and test: {len(val_set & test_set)} samples")
    
    print(f'  No overlap verified: ✓')
    
    # Verify redshift distribution consistency across splits
    train_redshifts = all_redshifts[train_indices]
    val_redshifts = all_redshifts[val_indices]
    test_redshifts = all_redshifts[test_indices]
    
    print(f'\nRedshift distribution across splits:')
    print(f'  Train: mean={train_redshifts.mean():.4f}, std={train_redshifts.std():.4f}, range=[{train_redshifts.min():.4f}, {train_redshifts.max():.4f}]')
    print(f'  Val:   mean={val_redshifts.mean():.4f}, std={val_redshifts.std():.4f}, range=[{val_redshifts.min():.4f}, {val_redshifts.max():.4f}]')
    print(f'  Test:  mean={test_redshifts.mean():.4f}, std={test_redshifts.std():.4f}, range=[{test_redshifts.min():.4f}, {test_redshifts.max():.4f}]')
    
    # Check redshift bin distribution
    redshift_bins = [0.0, 0.5, 1.0, 1.5, 2.0]
    bin_labels = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0']
    print(f'\nRedshift bin distribution:')
    for i in range(len(redshift_bins) - 1):
        bin_mask_all = (all_redshifts >= redshift_bins[i]) & (all_redshifts < redshift_bins[i+1])
        bin_mask_train = (train_redshifts >= redshift_bins[i]) & (train_redshifts < redshift_bins[i+1])
        bin_mask_val = (val_redshifts >= redshift_bins[i]) & (val_redshifts < redshift_bins[i+1])
        bin_mask_test = (test_redshifts >= redshift_bins[i]) & (test_redshifts < redshift_bins[i+1])
        
        total_bin = bin_mask_all.sum()
        train_bin = bin_mask_train.sum()
        val_bin = bin_mask_val.sum()
        test_bin = bin_mask_test.sum()
        
        print(f'  {bin_labels[i]}: Total={total_bin}, Train={train_bin} ({train_bin/total_bin*100:.1f}%), '
              f'Val={val_bin} ({val_bin/total_bin*100:.1f}%), Test={test_bin} ({test_bin/total_bin*100:.1f}%)')
    
    print()
    
    # Create datasets
    train_dataset = DR17MagnitudeDataset(
        train_images, train_magnitudes, train_redshifts,
        train_ra, train_dec,
        image_size=config['image_size'],
        use_augmentation=True
    )
    val_dataset = DR17MagnitudeDataset(
        val_images, val_magnitudes, val_redshifts,
        val_ra, val_dec,
        image_size=config['image_size'],
        use_augmentation=False
    )
    test_dataset = DR17MagnitudeDataset(
        test_images, test_magnitudes, test_redshifts,
        test_ra, test_dec,
        image_size=config['image_size'],
        use_augmentation=False
    )
    
    print(f'Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}')
    print()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], prefetch_factor=config['prefetch_factor'],
        pin_memory=True, persistent_workers=False  # Don't keep workers alive to save memory
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], prefetch_factor=config['prefetch_factor'],
        pin_memory=True, persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], prefetch_factor=config['prefetch_factor'],
        pin_memory=True, persistent_workers=False
    )
    
    # Create model (6 channels: 1 image + 5 magnitudes)
    model = ResNet34HKMagnitude(in_channels=6, device='cuda').to(device)
    
    param_info = count_parameters(model)
    if isinstance(param_info, dict):
        num_params = param_info.get('total_params', param_info.get('num_params', 0))
        num_params_M = param_info.get('total_params_M', num_params / 1e6)
    else:
        num_params = param_info
        num_params_M = num_params / 1e6
    
    print(f"Model parameters: {num_params:,} ({num_params_M:.2f}M)")
    print()
    
    # Create loss function based on experiment type
    # Import color-aware weighting from improved_model
    try:
        from improved_model import ConfidenceWeighter, compute_color_quality_masks
    except ImportError:
        ConfidenceWeighter = None
        compute_color_quality_masks = None
        print("Warning: Could not import ConfidenceWeighter, color-aware weighting will be disabled")
    
    if experiment_type == 'baseline':
        # Simple MSE loss for redshift
        loss_fn = nn.MSELoss()
        use_hk = False
        use_color_aware = False
    elif experiment_type == 'color_aware':
        # MSE + color-aware weighting
        class ColorAwareLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse_loss = nn.MSELoss(reduction='none')
                if ConfidenceWeighter is not None:
                    self.confidence_weighting = ConfidenceWeighter()
                else:
                    self.confidence_weighting = None
            
            def forward(self, outputs, targets):
                pred_redshift = outputs['redshift_final']
                target_redshift = targets['redshift']
                
                # Base MSE loss (per sample)
                mse_per_sample = self.mse_loss(pred_redshift, target_redshift)
                
                # Color-aware weighting (if available)
                if self.confidence_weighting is not None and 'color_features' in targets:
                    color_features = targets['color_features']  # [batch, 4] or [batch, 2] (g-r, g-i)
                    # Ensure color_features has shape [batch, 4] for ConfidenceWeighter
                    if color_features.shape[1] == 2:
                        # Expand to [batch, 4]: [u-g, g-r, r-i, i-z]
                        # Use g-r and g-i, set u-g=0, r-i=0, i-z=0 as placeholder
                        color_features_expanded = torch.zeros(
                            color_features.shape[0], 4, 
                            device=color_features.device, dtype=color_features.dtype
                        )
                        color_features_expanded[:, 1] = color_features[:, 0]  # g-r
                        color_features_expanded[:, 2] = color_features[:, 1]  # g-i (as r-i placeholder)
                        color_features = color_features_expanded
                    
                    sample_weights, quality_masks, _ = self.confidence_weighting(color_features)
                    mse_weighted = (sample_weights * mse_per_sample).mean()
                else:
                    # Fallback to simple MSE if color-aware not available
                    mse_weighted = mse_per_sample.mean()
                
                return mse_weighted, {'mse': mse_weighted.item()}
        
        loss_fn = ColorAwareLoss()
        use_hk = False
        use_color_aware = True
    elif experiment_type == 'hk' or experiment_type == 'color_aware_hk':
        # MSE + HK distance (with optional color-aware weighting)
        use_color_aware_in_loss = (experiment_type == 'color_aware_hk')
        
        class HKLoss(nn.Module):
            def __init__(self, use_color_aware=False):
                super().__init__()
                self.hk_distance = HellingerKantorovichDistance(
                    delta=1.0,
                    lambda_reg=0.1,
                    max_iter=50,
                    eps=1e-4,
                    n_bins=40
                )
                self.lambda_hk_target = 0.30  # Target HK weight
                self.mse_loss = nn.MSELoss(reduction='none' if use_color_aware else 'mean')
                self.lambda_mse = 1.0  # Increased from 0.1 for better balance
                self.hk_start_epoch = 2  # Start HK from epoch 2 (0-indexed: epoch 2, second epoch)
                self.hk_warmup_epochs = 8  # Gradually increase HK weight over 8 epochs
                self.use_color_aware = use_color_aware
                if use_color_aware and ConfidenceWeighter is not None:
                    self.confidence_weighting = ConfidenceWeighter()
                else:
                    self.confidence_weighting = None
            
            def forward(self, outputs, targets, current_epoch=0):
                pred_redshift = outputs['redshift_final']
                target_redshift = targets['redshift']
                
                # MSE component (with optional color-aware weighting)
                if self.use_color_aware:
                    # Per-sample MSE
                    mse_per_sample = self.mse_loss(pred_redshift, target_redshift)
                    
                    # Color-aware weighting
                    if self.confidence_weighting is not None and 'color_features' in targets:
                        color_features = targets['color_features']
                        if color_features.shape[1] == 2:
                            color_features_expanded = torch.zeros(
                                color_features.shape[0], 4,
                                device=color_features.device, dtype=color_features.dtype
                            )
                            color_features_expanded[:, 1] = color_features[:, 0]  # g-r
                            color_features_expanded[:, 2] = color_features[:, 1]  # g-i
                            color_features = color_features_expanded
                        
                        sample_weights, quality_masks, _ = self.confidence_weighting(color_features)
                        mse = (sample_weights * mse_per_sample).mean()
                    else:
                        mse = mse_per_sample.mean()
                else:
                    # Standard MSE
                    mse = self.mse_loss(pred_redshift, target_redshift)
                
                # HK distance component with gradual introduction
                if current_epoch < self.hk_start_epoch:
                    # Before epoch 2, use MSE only
                    hk_loss = torch.tensor(0.0, device=pred_redshift.device)
                    lambda_hk_current = 0.0
                    total_loss = self.lambda_mse * mse
                else:
                    # Gradually increase HK weight from epoch 2 onwards
                    # Linear warmup: from 0 at epoch 2 to lambda_hk_target at epoch (2 + warmup_epochs)
                    if current_epoch < self.hk_start_epoch + self.hk_warmup_epochs:
                        # Linear interpolation during warmup
                        progress = (current_epoch - self.hk_start_epoch) / self.hk_warmup_epochs
                        lambda_hk_current = self.lambda_hk_target * progress
                    else:
                        # After warmup, use full target weight
                        lambda_hk_current = self.lambda_hk_target
                    
                    # From epoch 2 onwards, add HK distance (with gradual weight)
                    # Clamp values to reasonable range [1e-6, 2.0] for HK distance
                    hk_pred = torch.clamp(pred_redshift, min=1e-6, max=2.0)
                    hk_target = torch.clamp(target_redshift, min=1e-6, max=2.0)
                    
                    # Use fixed z_max for stability (DR17 redshift range is [0, 2.0])
                    z_max = 2.0
                    
                    try:
                        # Batch-level HK distance computation
                        # Aggregate all samples in batch into a single distribution
                        # This captures overall distribution matching rather than per-sample matching
                        with autocast(device_type='cuda', enabled=False):  # Disable mixed precision for HK
                            hk_pred_f32 = hk_pred.float()
                            hk_target_f32 = hk_target.float()
                            
                            # Convert each sample to distribution [batch, n_bins]
                            pred_dist = self.hk_distance.values_to_distribution(
                                hk_pred_f32, z_min=1e-6, z_max=z_max
                            )
                            target_dist = self.hk_distance.values_to_distribution(
                                hk_target_f32, z_min=1e-6, z_max=z_max
                            )
                            
                            # Normalize distributions
                            pred_dist_norm = self.hk_distance._normalize_distribution(pred_dist)
                            target_dist_norm = self.hk_distance._normalize_distribution(target_dist)
                            
                            # Aggregate batch: average all samples to get batch-level distribution
                            # Shape: [batch, n_bins] -> [n_bins]
                            pred_batch_dist = pred_dist_norm.mean(dim=0)  # [n_bins]
                            target_batch_dist = target_dist_norm.mean(dim=0)  # [n_bins]
                            
                            # Ensure distributions are valid (re-normalize)
                            pred_batch_dist = pred_batch_dist / (pred_batch_dist.sum() + 1e-8)
                            target_batch_dist = target_batch_dist / (target_batch_dist.sum() + 1e-8)
                            
                            # Compute cost matrix
                            cost_matrix = self.hk_distance._compute_cost_matrix(
                                1e-6, z_max, hk_pred_f32.device
                            )
                            
                            # Compute batch-level HK distance using Sinkhorn
                            # Add batch dimension [1, n_bins] for sinkhorn_hk_distance
                            pred_batch_dist = pred_batch_dist.unsqueeze(0)  # [1, n_bins]
                            target_batch_dist = target_batch_dist.unsqueeze(0)  # [1, n_bins]
                            
                            hk_loss = self.hk_distance.sinkhorn_hk_distance(
                                pred_batch_dist, target_batch_dist, cost_matrix
                            )
                            
                            # Extract scalar from [1] tensor
                            hk_loss = hk_loss.squeeze(0) if hk_loss.dim() > 0 else hk_loss
                            
                            # Ensure hk_loss is float32 and on correct device
                            if hk_loss.dtype != torch.float32:
                                hk_loss = hk_loss.float()
                            hk_loss = hk_loss.to(pred_redshift.device)
                        
                        # Clamp HK loss to prevent explosion
                        hk_loss = torch.clamp(hk_loss, max=10.0)
                            
                    except Exception as e:
                        import warnings
                        warnings.warn(f"HK distance computation failed: {str(e)}. Using MSE only.", UserWarning)
                        hk_loss = torch.tensor(0.0, device=pred_redshift.device, dtype=pred_redshift.dtype)
                    
                    total_loss = self.lambda_mse * mse + lambda_hk_current * hk_loss
                
                return total_loss, {'mse': mse.item(), 'hk': hk_loss.item()}
        
        loss_fn = HKLoss(use_color_aware=use_color_aware_in_loss)
        use_hk = True
        use_color_aware = use_color_aware_in_loss
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['initial_lr'],
        weight_decay=config['weight_decay']
    )
    
    # Standard learning rate scheduler
    if config.get('lr_scheduler', 'cosine') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=1e-6
        )
    elif config.get('lr_scheduler') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('lr_step_size', 20),
            gamma=config.get('lr_gamma', 0.1)
        )
    else:
        # Default: no scheduler
        scheduler = None
    
    scaler = torch.cuda.amp.GradScaler()
    
    train_losses = []
    val_losses = []
    val_log_mses = []
    learning_rates = []
    epoch_times = []
    
    best_val_log_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    lr_accelerated = False  # Flag to track if LR has been accelerated
    
    print("Starting training...")
    print()
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        print(f'Epoch {epoch + 1}/{config["num_epochs"]}')
        
        model.train()
        train_loss_sum = 0.0
        train_batch_count = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            coordinates = batch['coordinates'].to(device)
            redshifts = batch['redshift'].to(device)
            color_features = batch.get('color_features', None)
            if color_features is not None:
                color_features = color_features.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images, coordinates)
                
                # Prepare targets dict
                targets = {'redshift': redshifts}
                if use_color_aware and color_features is not None:
                    targets['color_features'] = color_features
                
                if use_hk or use_color_aware:
                    loss, _ = loss_fn(outputs, targets, current_epoch=epoch)
                else:
                    pred_redshift = outputs['redshift_final']
                    loss = loss_fn(pred_redshift, redshifts)
            
            train_loss_sum += loss.item()
            train_batch_count += 1
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Clear batch data to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        avg_train_loss = train_loss_sum / train_batch_count if train_batch_count > 0 else 0.0
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss_sum = 0.0
        val_batch_count = 0
        all_val_redshifts_pred = []
        all_val_redshifts_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                coordinates = batch['coordinates'].to(device)
                redshifts = batch['redshift'].to(device)
                color_features = batch.get('color_features', None)
                if color_features is not None:
                    color_features = color_features.to(device)
                
                outputs = model(images, coordinates)
                
                # Prepare targets dict
                targets = {'redshift': redshifts}
                if use_color_aware and color_features is not None:
                    targets['color_features'] = color_features
                
                if use_hk or use_color_aware:
                    loss, _ = loss_fn(outputs, targets, current_epoch=epoch)
                else:
                    pred_redshift = outputs['redshift_final']
                    loss = loss_fn(pred_redshift, redshifts)
                
                val_loss_sum += loss.item()
                val_batch_count += 1
                
                pred_redshifts = outputs['redshift_final']
                all_val_redshifts_pred.extend(pred_redshifts.cpu().numpy())
                all_val_redshifts_true.extend(redshifts.cpu().numpy())
        
        avg_val_loss = val_loss_sum / val_batch_count if val_batch_count > 0 else 0.0
        val_losses.append(avg_val_loss)
        
        log_mse = compute_log_mse_unified(
            np.array(all_val_redshifts_pred),
            np.array(all_val_redshifts_true),
            min_val=1e-6
        )
        val_log_mses.append(log_mse)
        
        # Clear validation lists to free memory
        del all_val_redshifts_pred, all_val_redshifts_true
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update learning rate scheduler (at end of epoch)
        if scheduler is not None:
            scheduler.step()
        
        is_better = log_mse < best_val_log_mse - config['early_stopping_min_delta']
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        if is_better:
            best_val_log_mse = log_mse
            patience_counter = 0
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            lr_accelerated = False  # Reset flag when performance improves
        else:
            patience_counter += 1
            
            # Accelerate learning rate decay when patience_counter > 5
            # Only trigger once when first crossing the threshold
            if patience_counter > 5 and not lr_accelerated:
                # Additional learning rate reduction for faster convergence
                # Reduce LR by 0.5x when patience > 5
                # This helps escape local minima when stuck
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    # Only reduce if LR hasn't been reduced too much (avoid going too low)
                    if old_lr > 1e-7:  # Don't reduce if already very small
                        new_lr = old_lr * 0.5
                        param_group['lr'] = new_lr
                        print(f'  [LR Decay] Patience > 5, accelerating LR decay: {old_lr:.6f} -> {new_lr:.6f}')
                
                # For StepLR scheduler, also reduce step_size to decay more frequently
                if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                    # Reduce step_size by half to decay more frequently
                    if scheduler.step_size > 1:
                        scheduler.step_size = max(1, scheduler.step_size // 2)
                        print(f'  [LR Decay] StepLR step_size reduced to {scheduler.step_size}')
                
                # For CosineAnnealingLR, we can't easily modify T_max, but the manual LR reduction above helps
                # The scheduler will continue with its cosine schedule from the new lower LR
                
                lr_accelerated = True  # Mark as accelerated to avoid repeated reductions
        
        print(f'  Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        print(f'  Val log-MSE: {log_mse:.6f} (best: {best_val_log_mse:.6f})')
        print(f'  LR: {current_lr:.6f}, Early Stop: {patience_counter}/{config["early_stopping_patience"]}')
        print(f'  Time: {epoch_time:.2f}s')
        print()
        
        if config['early_stopping'] and patience_counter >= config['early_stopping_patience']:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            print(f'Best validation log-MSE: {best_val_log_mse:.6f} at epoch {best_epoch}')
            break
    
    print('=' * 70)
    print('Training completed!')
    print(f'Best validation log-MSE: {best_val_log_mse:.6f} at epoch {best_epoch}')
    print('=' * 70)
    print()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model from epoch {best_epoch}')
    
    # Evaluate on test set
    print('Evaluating on test set...')
    model.eval()
    all_test_redshifts_pred = []
    all_test_redshifts_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            coordinates = batch['coordinates'].to(device)
            redshifts = batch['redshift'].to(device)
            
            outputs = model(images, coordinates)
            pred_redshifts = outputs['redshift_final']
            all_test_redshifts_pred.extend(pred_redshifts.cpu().numpy())
            all_test_redshifts_true.extend(redshifts.cpu().numpy())
    
    test_log_mse = compute_log_mse_unified(
        np.array(all_test_redshifts_pred),
        np.array(all_test_redshifts_true),
        min_val=1e-6
    )
    
    # Calculate comprehensive metrics
    test_pred = np.array(all_test_redshifts_pred)
    test_true = np.array(all_test_redshifts_true)
    
    # Overall metrics
    test_mae = np.mean(np.abs(test_pred - test_true))
    test_bias = np.mean(test_pred - test_true)
    test_std = np.std(test_pred - test_true)
    
    # Outlier rate (|delta_z| > 0.15 * (1 + z_true))
    outlier_mask = np.abs(test_pred - test_true) > 0.15 * (1 + test_true)
    test_outlier_rate = np.mean(outlier_mask)
    
    # NMAD (Normalized Median Absolute Deviation)
    delta_z = (test_pred - test_true) / (1 + test_true)
    test_nmad = 1.4826 * np.median(np.abs(delta_z - np.median(delta_z)))
    
    print(f'Test Metrics:')
    print(f'  Log-MSE: {test_log_mse:.6f}')
    print(f'  MAE: {test_mae:.6f}')
    print(f'  Bias: {test_bias:.6f}')
    print(f'  Std: {test_std:.6f}')
    print(f'  Outlier Rate: {test_outlier_rate:.4f}')
    print(f'  NMAD: {test_nmad:.6f}')
    print()
    
    # Calculate metrics for different redshift bins
    redshift_bins = [0.0, 0.5, 1.0, 1.5, 2.0]
    bin_labels = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0']
    redshift_bin_metrics = {}
    
    for i in range(len(redshift_bins) - 1):
        bin_mask = (test_true >= redshift_bins[i]) & (test_true < redshift_bins[i+1])
        if bin_mask.sum() > 0:
            bin_pred = test_pred[bin_mask]
            bin_true = test_true[bin_mask]
            
            bin_log_mse = compute_log_mse_unified(bin_pred, bin_true, min_val=1e-6)
            bin_mae = np.mean(np.abs(bin_pred - bin_true))
            bin_bias = np.mean(bin_pred - bin_true)
            bin_std = np.std(bin_pred - bin_true)
            
            bin_outlier_mask = np.abs(bin_pred - bin_true) > 0.15 * (1 + bin_true)
            bin_outlier_rate = np.mean(bin_outlier_mask)
            
            bin_delta_z = (bin_pred - bin_true) / (1 + bin_true)
            bin_nmad = 1.4826 * np.median(np.abs(bin_delta_z - np.median(bin_delta_z))) if len(bin_delta_z) > 0 else 0.0
            
            redshift_bin_metrics[bin_labels[i]] = {
                'log_mse': float(bin_log_mse),
                'mae': float(bin_mae),
                'bias': float(bin_bias),
                'std': float(bin_std),
                'outlier_rate': float(bin_outlier_rate),
                'nmad': float(bin_nmad),
                'count': int(bin_mask.sum())
            }
            
            print(f'  Redshift bin {bin_labels[i]} (n={bin_mask.sum()}):')
            print(f'    Log-MSE: {bin_log_mse:.6f}, Bias: {bin_bias:.6f}, MAE: {bin_mae:.6f}')
            print(f'    Std: {bin_std:.6f}, Outlier Rate: {bin_outlier_rate:.4f}, NMAD: {bin_nmad:.6f}')
        else:
            redshift_bin_metrics[bin_labels[i]] = {
                'log_mse': None,
                'mae': None,
                'bias': None,
                'std': None,
                'outlier_rate': None,
                'nmad': None,
                'count': 0
            }
    print()
    
    # Measure inference speed with correct input shape
    def measure_inference_speed_custom(model, device, image_size, in_channels, num_runs=100, warmup_runs=10):
        model.eval()
        dummy_input = torch.randn(1, in_channels, image_size, image_size).to(device)
        dummy_coords = torch.randn(1, 2).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input, dummy_coords)
        
        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input, dummy_coords)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        elapsed = end_time - start_time
        fps = num_runs / elapsed
        return fps
    
    inference_fps = measure_inference_speed_custom(model, device, config['image_size'], 6)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_log_mses': val_log_mses,
        'learning_rates': learning_rates,
        'epoch_times': epoch_times,
        'best_val_log_mse': best_val_log_mse,
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses),
        'test_log_mse': test_log_mse,
        'test_metrics': {
            'log_mse': float(test_log_mse),
            'mae': float(test_mae),
            'bias': float(test_bias),
            'std': float(test_std),
            'outlier_rate': float(test_outlier_rate),
            'nmad': float(test_nmad)
        },
        'redshift_bin_metrics': redshift_bin_metrics,
        'config': config,
        'model_stats': {
            'num_params': num_params,
            'num_params_M': num_params_M,
            'inference_fps': inference_fps
        }
    }
    
    checkpoint_path = config['save_path'].replace('.pth', '_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to: {checkpoint_path}')
    
    torch.save(model.state_dict(), config['save_path'])
    print(f'Model saved to: {config["save_path"]}')
    print()
    
    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HK distance experiment on DR17 dataset')
    parser.add_argument('--experiment_type', type=str, required=True,
                       choices=['baseline', 'color_aware', 'hk', 'color_aware_hk'],
                       help='Experiment type: baseline (MSE only), color_aware (MSE + color weighting), '
                            'hk (MSE + HK distance), or color_aware_hk (full complementary mechanism)')
    parser.add_argument('--seed', type=int, default=None,
                       choices=[36, 42, 199],
                       help='Random seed (default: None, runs all seeds [36, 42, 199])')
    
    args = parser.parse_args()
    
    seeds = [36, 42, 199] if args.seed is None else [args.seed]
    
    for seed in seeds:
        print(f'\n{"="*70}')
        print(f'Starting experiment: {args.experiment_type}, seed: {seed}')
        print(f'{"="*70}\n')
        train_hk_experiment(args.experiment_type, seed)

