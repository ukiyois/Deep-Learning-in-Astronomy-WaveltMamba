"""
ResNet-34 Multi-Task Baseline Training Script
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
    from baseline_models.resnet34_multitask import ResNet34MultiTask
    from baseline_models.simple_loss import SimpleMultiTaskLoss
except ImportError:
    from resnet34_multitask import ResNet34MultiTask
    from simple_loss import SimpleMultiTaskLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast
import time

def train_resnet34(seed=42, **config_overrides):
    """Train ResNet-34 multi-task baseline model"""
    set_seed(seed)
    print('=' * 70)
    print('ResNet-34 Multi-Task Baseline Training')
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
        
        'enable_collapse_detection': True,
        'train_acc_collapse_threshold': 0.10,
        'collapse_check_window': 3,
        'collapse_auto_rollback': True,
        'collapse_lr_reduction': 0.5,
        
        'warmup_epochs': 10,
        'phi': 0.70,
        'num_workers': 12,
        'prefetch_factor': 4,
        'save_path': f'244x244_seed{seed}_baseline_resnet34_model.pth',
        'seed': seed,
        'early_stopping': True,
        'early_stopping_patience': 25,
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
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=weighted_sampler,
        num_workers=config['num_workers'], pin_memory=True,
        prefetch_factor=config['prefetch_factor'],
        generator=generator,
        worker_init_fn=lambda worker_id: set_seed(seed + worker_id),
        persistent_workers=True  # Keep worker processes alive, reduce restart overhead
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        prefetch_factor=config['prefetch_factor'],
        generator=generator,
        worker_init_fn=lambda worker_id: set_seed(seed + worker_id),
        persistent_workers=True  # Keep worker processes alive, reduce restart overhead
    )
    
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')
    print()
    
    model = ResNet34MultiTask(num_classes=10, device='cuda', use_spectral=False)
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
            
            # Compute gradient norm (for monitoring and scheduler adaptive phi, no clipping)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            
            # Update UBA scheduler gradient norm (for adaptive phi)
            scheduler.update_grad_norm(grad_norm.item())
            
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
        
        # Training collapse detection: monitor if training accuracy drops significantly
        collapse_detected = False
        if config.get('enable_collapse_detection', False) and len(train_accuracies) >= config.get('collapse_check_window', 3):
            # Calculate average training accuracy of recent N epochs
            recent_window = config.get('collapse_check_window', 3)
            recent_train_acc = np.mean(train_accuracies[-recent_window:])
            
            # Calculate historical best training accuracy (for comparison)
            if len(train_accuracies) > recent_window:
                # Compare with previous best training accuracy
                best_prev_train_acc = max(train_accuracies[:-recent_window])
                train_acc_drop = best_prev_train_acc - recent_train_acc
                collapse_threshold = config.get('train_acc_collapse_threshold', 0.10)
                
                if train_acc_drop > collapse_threshold:
                    collapse_detected = True
                    print(f'\nTraining Collapse Detected')
                    print(f'  Historical best training accuracy: {best_prev_train_acc:.2%}')
                    print(f'  Recent {recent_window} epochs average training accuracy: {recent_train_acc:.2%}')
                    print(f'  Training accuracy drop: {train_acc_drop:.2%} (threshold: {collapse_threshold:.2%})')
                    
                    if config.get('collapse_auto_rollback', True) and best_model_state is not None and best_epoch > 0:
                        print(f'  Auto-rolling back to best epoch {best_epoch} model state...')
                        # Rollback to best model state
                        model.load_state_dict(best_model_state)
                        
                        # Reduce learning rate
                        current_lr = optimizer.param_groups[0]['lr']
                        lr_reduction = config.get('collapse_lr_reduction', 0.5)
                        new_lr = current_lr * lr_reduction
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f'  Learning rate reduced: {current_lr:.6f} → {new_lr:.6f} (reduced by {(1-lr_reduction)*100:.0f}%)')
                        
                        # Remove recent collapsed epoch records (rollback to best epoch state)
                        if len(train_accuracies) > best_epoch:
                            print(f'  Removing training records from Epoch {best_epoch + 1} to Epoch {len(train_accuracies)}')
                            train_losses = train_losses[:best_epoch]
                            train_accuracies = train_accuracies[:best_epoch]
                            val_losses = val_losses[:best_epoch]
                            val_accuracies = val_accuracies[:best_epoch]
                            val_log_mses = val_log_mses[:best_epoch]
                            learning_rates = learning_rates[:best_epoch]
                            epoch_times = epoch_times[:best_epoch]
                        
                        print(f'  Rolled back to Epoch {best_epoch}, continuing training...\n')
                    else:
                        print(f'   Auto-rollback not enabled or best model state not found, continuing training...\n')
        
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
                
                # Standard prediction
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
        
        # Update UBA scheduler validation loss (for feedback mechanism)
        scheduler.update_validation_loss(avg_val_loss)
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResNet34 baseline model')
    parser.add_argument('--image_size', type=int, default=244, help='Image size (default: 244)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[3, 7, 42], help='Random seeds (default: 3 7 42)')
    args = parser.parse_args()
    
    image_size = args.image_size
    seeds = args.seeds
    
    # 根据图像尺寸调整batch_size
    batch_size_map = {
        32: 512,
        64: 256,
        128: 128,
        244: 36
    }
    batch_size = batch_size_map.get(image_size, 36)
    
    completed_seeds = []
    failed_seeds = []
    
    print(f'\n{"="*70}')
    print(f'ResNet34 Baseline Training Configuration')
    print(f'{"="*70}')
    print(f'Image Size: {image_size}×{image_size}')
    print(f'Batch Size: {batch_size}')
    print(f'Seeds: {seeds}')
    print(f'{"="*70}\n')
    
    for seed in seeds:
        print(f'\n{"="*70}')
        print(f'Starting training with seed {seed} (Image Size: {image_size}×{image_size})')
        print(f'{"="*70}\n')
        
        try:
            train_resnet34(
                seed=seed,
                image_size=image_size,
                batch_size=batch_size,
                save_path=f'{image_size}x{image_size}_seed{seed}_baseline_resnet34_model.pth'
            )
            
            # Verify checkpoint was created
            checkpoint_path = f'{image_size}x{image_size}_seed{seed}_baseline_resnet34_model_checkpoint.pth'
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
    print(f'Image Size: {image_size}×{image_size}')
    print(f'Completed seeds: {completed_seeds} ({len(completed_seeds)}/{len(seeds)})')
    if failed_seeds:
        print(f'Failed seeds: {failed_seeds} ({len(failed_seeds)}/{len(seeds)})')
    print(f'{"="*70}\n')
