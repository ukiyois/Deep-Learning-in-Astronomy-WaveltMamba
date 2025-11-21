"""
Swin-T Multi-Task Baseline Training Script
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from galaxy10_training import (
    set_seed, Galaxy10Dataset, UBAScheduler, 
    compute_log_mse_unified, get_redshift_output,
    count_parameters, measure_inference_speed
)

try:
    from baseline_models.swin_multitask import SwinMultiTask
    from baseline_models.simple_loss import SimpleMultiTaskLoss
except ImportError:
    from swin_multitask import SwinMultiTask
    from simple_loss import SimpleMultiTaskLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast
import time

def train_swin(seed=42, **config_overrides):
    """Train Swin-T multi-task baseline model"""
    set_seed(seed)
    print('=' * 70)
    print('Swin-T Multi-Task Baseline Training')
    print(f'Random seed: {seed}')
    print('=' * 70)
    print()
    
    config = {
        'h5_path': 'data/Galaxy10_DECals.h5',
        'num_samples': None,
        'image_size': 244,
        'val_split': 0.3,
        'batch_size': 36,
        'num_epochs': 120,
        'initial_lr': 1e-4,
        'max_lr': 0.001,
        'min_lr': 5e-6,
        'weight_decay': 5e-4,
        'optimizer': 'adamw',
        'warmup_epochs': 10,
        'phi': 0.70,
        'num_workers': 16,
        'prefetch_factor': 4,
        'save_path': f'244x244_seed{seed}_baseline_swin_model.pth',
        'seed': seed,
        'early_stopping': True,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.005,  
        'early_stopping_monitor': 'val_acc',
        'early_stopping_mode': 'max',
        
        'restore_lr_on_early_stop': False,  
    }
    
    config.update(config_overrides)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    device = torch.device('cuda:0')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    train_dataset = Galaxy10Dataset(
        config['h5_path'], config['num_samples'], config['image_size'],
        split='train', use_augmentation=True
    )
    train_dataset._val_split = config['val_split']
    
    val_dataset = Galaxy10Dataset(
        config['h5_path'], config['num_samples'], config['image_size'],
        split='val', use_augmentation=False
    )
    val_dataset._val_split = config['val_split']
    
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
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=weighted_sampler,
        num_workers=config['num_workers'], pin_memory=True,
        prefetch_factor=config['prefetch_factor']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        prefetch_factor=config['prefetch_factor']
    )
    
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')
    print()
    
    model = SwinMultiTask(num_classes=10, device='cuda', use_spectral=False)
    model = model.to(device)
    
    param_info = count_parameters(model)
    print(f"Model parameters: {param_info['total_params_M']:.2f}M (trainable: {param_info['trainable_params_M']:.2f}M)")
    print()
    
    loss_fn = SimpleMultiTaskLoss(num_classes=10, cls_weight=0.90, redshift_weight=0.10)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['max_lr'],
        weight_decay=config['weight_decay']
    )
    
    total_steps = config['num_epochs'] * len(train_loader)
    warmup_steps = config['warmup_epochs'] * len(train_loader)
    scheduler = UBAScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        max_lr=config['max_lr'],
        min_lr=config['min_lr'],
        initial_lr=config['initial_lr'],
        phi=config['phi'],
        base_phi=config['phi'],
        enable_adaptive_phi=True,
        enable_val_feedback=True
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    best_lr_at_best_epoch = None  
    train_accuracies = []
    val_accuracies = []
    val_log_mses = []
    composite_scores = []
    train_losses = []  
    val_losses = []    
    learning_rates = []  
    epoch_times = []    
    
    print("Starting training...")
    print()
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        print(f'Epoch {epoch + 1}/{config["num_epochs"]}:')
        
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0
        train_batch_count = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            coordinates = batch['coordinates'].to(device)
            labels = batch['label'].to(device)
            redshifts = batch['redshift'].to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=True):
                outputs = model(images, coordinates)
                targets = {'labels': labels, 'redshift': redshifts}
                loss, _ = loss_fn(outputs, targets)
            
            train_loss_sum += loss.item()
            train_batch_count += 1
            
            scaler.scale(loss).backward()
            # Gradient clipping removed - allow gradients to update freely
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            preds = torch.argmax(outputs['cls_logits'], dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss_sum / train_batch_count if train_batch_count > 0 else 0.0
        train_accuracies.append(train_acc)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        val_batch_count = 0
        all_val_redshifts_pred = []
        all_val_redshifts_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                coordinates = batch['coordinates'].to(device)
                labels = batch['label'].to(device)
                redshifts = batch['redshift'].to(device)
                
                outputs = model(images, coordinates)
                targets = {'labels': labels, 'redshift': redshifts}
                loss, _ = loss_fn(outputs, targets)
                val_loss_sum += loss.item()
                val_batch_count += 1
                
                preds = torch.argmax(outputs['cls_logits'], dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                pred_redshifts = get_redshift_output(outputs, redshifts)
                all_val_redshifts_pred.extend(pred_redshifts.cpu().numpy())
                all_val_redshifts_true.extend(redshifts.cpu().numpy())
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss_sum / val_batch_count if val_batch_count > 0 else 0.0
        val_accuracies.append(val_acc)
        val_losses.append(avg_val_loss)
        
        log_mse = compute_log_mse_unified(
            np.array(all_val_redshifts_pred),
            np.array(all_val_redshifts_true),
            min_val=1e-6
        )
        composite_score = 0.4 * val_acc + 0.6 * (1 - log_mse)
        
        val_log_mses.append(log_mse)
        composite_scores.append(composite_score)
        
        scheduler.update_validation_loss(0.0)
        
        is_better = val_acc > best_val_acc + config['early_stopping_min_delta']
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)  
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)  
        
        if is_better:
            best_val_acc = val_acc
            patience_counter = 0
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            best_lr_at_best_epoch = current_lr  
            # Note: Only checkpoint is saved, not separate .pth file (for Table 3 data only)
        else:
            patience_counter += 1
        
        print(f'  Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        print(f'  Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}')
        print(f'  Val log-MSE: {log_mse:.6f}, Composite Score: {composite_score:.4f}')
        print(f'  LR: {current_lr:.6f}, Early Stop: {patience_counter}/{config["early_stopping_patience"]}')
        print(f'  Time: {epoch_time:.2f}s')
        print()
        
        # Check if early stopping is triggered
        if patience_counter >= config['early_stopping_patience']:
            # Standard early stopping: revert to best model
            print(f'\nEarly stopping triggered, reverting to best model (Epoch {best_epoch})')
            lr_str = f'{best_lr_at_best_epoch:.6f}' if best_lr_at_best_epoch else 'N/A'
            print(f'Best epoch LR: {lr_str}')
            print(f'Current LR: {current_lr:.6f}')
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                
                # Optionally restore LR to best epoch's value
                if best_lr_at_best_epoch is not None and config.get('restore_lr_on_early_stop', False):
                    best_epoch_steps = best_epoch * len(train_loader)
                    scheduler.last_epoch = best_epoch_steps - 1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = best_lr_at_best_epoch
                    print(f'Restored LR to best epoch ({best_epoch}): {best_lr_at_best_epoch:.6f}')
            
            break
    
    print(f'Best Val Acc: {best_val_acc:.2%}')
    if best_epoch > 0:
        print(f'Best model from Epoch {best_epoch}')
    
    # Ensure best model state is saved in checkpoint
    # This handles both early stopping and normal completion
    if best_model_state is not None:
        final_model_state = best_model_state
        model.load_state_dict(best_model_state)
        print(f'Best model state loaded (from Epoch {best_epoch})')
    else:
        final_model_state = model.state_dict()
        print('No best model state found, using current model state')
    
    total_training_time_s = sum(epoch_times)
    total_training_time_h = total_training_time_s / 3600.0
    
    print('Measuring inference speed...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference_stats = measure_inference_speed(model, device, image_size=config['image_size'])
    
    checkpoint_path = config['save_path'].replace('.pth', '_checkpoint.pth')
    full_checkpoint = {
        'model_state_dict': final_model_state,
        'train_losses': train_losses,          
        'val_losses': val_losses,              
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'val_log_mses': val_log_mses,
        'composite_scores': composite_scores,
        'learning_rates': learning_rates,      
        'epoch_times': epoch_times,            
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'best_lr_at_best_epoch': best_lr_at_best_epoch,
        'total_epochs': len(train_accuracies),
        'config': config,
        'model_stats': {
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
    }
    torch.save(full_checkpoint, checkpoint_path)
    print(f'Total training time: {total_training_time_h:.2f} hours ({total_training_time_s/60:.1f} minutes)')
    print(f'Inference speed: {inference_stats["fps"]:.2f} FPS ({inference_stats["avg_time_ms"]:.2f} ms/sample)')
    print(f'Full training history saved to: {checkpoint_path}')

if __name__ == '__main__':
    seeds = [3, 7, 42]
    completed_seeds = []
    failed_seeds = []
    
    for seed in seeds:
        print(f'\n{"="*70}')
        print(f'Starting training with seed {seed}')
        print(f'{"="*70}\n')
        
        try:
            train_swin(seed=seed)
            
            # Verify checkpoint was created
            checkpoint_path = f'244x244_seed{seed}_baseline_swin_model_checkpoint.pth'
            if os.path.exists(checkpoint_path):
                completed_seeds.append(seed)
                print(f'\n{"="*70}')
                print(f'Completed training with seed {seed}')
                print(f'Checkpoint saved: {checkpoint_path}')
                print(f'{"="*70}\n')
            else:
                failed_seeds.append(seed)
                print(f'\n{"="*70}')
                print(f'Training completed but checkpoint not found: {checkpoint_path}')
                print(f'{"="*70}\n')
        except Exception as e:
            failed_seeds.append(seed)
            print(f'\n{"="*70}')
            print(f'Training failed with seed {seed}: {e}')
            print(f'{"="*70}\n')
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f'\n{"="*70}')
    print('Training Summary:')
    print(f'{"="*70}')
    print(f'Completed seeds: {completed_seeds} ({len(completed_seeds)}/3)')
    if failed_seeds:
        print(f'Failed seeds: {failed_seeds} ({len(failed_seeds)}/3)')
    print(f'{"="*70}\n')
