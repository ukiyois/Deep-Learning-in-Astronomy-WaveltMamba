"""
ConvNeXt Multi-Task Baseline Training Script
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
    from baseline_models.convnext_multitask import ConvNeXtMultiTask
    from baseline_models.simple_loss import SimpleMultiTaskLoss
except ImportError:
    from convnext_multitask import ConvNeXtMultiTask
    from simple_loss import SimpleMultiTaskLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast
import time
import argparse

def train_convnext(seed=42, **config_overrides):
    """Train ConvNeXt multi-task baseline model"""
    set_seed(seed)
    print('=' * 70)
    print('ConvNeXt Multi-Task Baseline Training')
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
        'save_path': None,  # Will be set based on image_size
        'seed': seed,
        'early_stopping': True,
        'early_stopping_patience': 25,
        'early_stopping_min_delta': 0.005,
        'early_stopping_monitor': 'val_acc',
        'early_stopping_mode': 'max',
        
        'restore_lr_on_early_stop': False,
        'model_name': 'convnext_tiny',
    }
    
    config.update(config_overrides)
    
    # 根据图像尺寸调整batch_size
    batch_size_map = {
        32: 512,
        64: 256,
        128: 128,
        244: 36
    }
    config['batch_size'] = batch_size_map.get(config['image_size'], 36)
    
    # 更新保存路径以包含图像尺寸
    config['save_path'] = f'{config["image_size"]}x{config["image_size"]}_seed{seed}_baseline_convnext_model.pth'
    
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
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=sampler,
        num_workers=config['num_workers'], prefetch_factor=config['prefetch_factor'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], prefetch_factor=config['prefetch_factor'],
        pin_memory=True
    )
    
    model = ConvNeXtMultiTask(
        num_classes=10, device='cuda', use_spectral=False,
        model_name=config['model_name']
    ).to(device)
    
    param_info = count_parameters(model)
    if isinstance(param_info, dict):
        num_params = param_info.get('total_params', param_info.get('num_params', 0))
        num_params_M = param_info.get('total_params_M', num_params / 1e6)
        trainable_params = param_info.get('trainable_params', 0)
        trainable_params_M = param_info.get('trainable_params_M', trainable_params / 1e6)
    else:
        num_params = param_info
        num_params_M = num_params / 1e6
        trainable_params = num_params
        trainable_params_M = num_params_M
    
    print(f"Model parameters: {num_params:,} ({num_params_M:.2f}M)")
    if isinstance(param_info, dict) and 'trainable_params' in param_info:
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params_M:.2f}M)")
    print()
    
    loss_fn = SimpleMultiTaskLoss(
        num_classes=10,
        cls_weight=0.90,
        redshift_weight=0.10
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['initial_lr'],
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
        phi=config['phi']
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_log_mses = []
    composite_scores = []
    learning_rates = []
    epoch_times = []
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    best_lr_at_best_epoch = config['initial_lr']
    
    print("Starting training...")
    print()
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        print(f'Epoch {epoch + 1}/{config["num_epochs"]}')
        
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
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images, coordinates)
                targets = {'labels': labels, 'redshift': redshifts}
                loss, _ = loss_fn(outputs, targets)
            
            train_loss_sum += loss.item()
            train_batch_count += 1
            
            scaler.scale(loss).backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
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
                    print(f'\nTraining Collapse Detected')
                    print(f'  Historical best training accuracy: {best_prev_train_acc:.2%}')
                    print(f'  Recent {recent_window} epochs average training accuracy: {recent_train_acc:.2%}')
                    print(f'  Training accuracy drop: {train_acc_drop:.2%} (threshold: {collapse_threshold:.2%})')
                    
                    if config.get('collapse_auto_rollback', True) and best_model_state is not None and best_epoch > 0:
                        print(f'  Auto-rolling back to best epoch {best_epoch} model state...')
                        model.load_state_dict(best_model_state)
                        
                        current_lr = optimizer.param_groups[0]['lr']
                        lr_reduction = config.get('collapse_lr_reduction', 0.5)
                        new_lr = current_lr * lr_reduction
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f'  Learning rate reduced: {current_lr:.6f} → {new_lr:.6f} (reduced by {(1-lr_reduction)*100:.0f}%)')
                        
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
        else:
            patience_counter += 1
        
        print(f'  Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        print(f'  Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}')
        print(f'  Val log-MSE: {log_mse:.6f}, Composite Score: {composite_score:.4f}')
        print(f'  LR: {current_lr:.6f}, Early Stop: {patience_counter}/{config["early_stopping_patience"]}')
        print(f'  Time: {epoch_time:.2f}s')
        print()
        
        if config['early_stopping'] and patience_counter >= config['early_stopping_patience']:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            print(f'Best validation accuracy: {best_val_acc:.2%} at epoch {best_epoch}')
            break
    
    print('=' * 70)
    print('Training completed!')
    print(f'Best validation accuracy: {best_val_acc:.2%} at epoch {best_epoch}')
    print('=' * 70)
    print()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model from epoch {best_epoch}')
    
    inference_fps = measure_inference_speed(model, device, config['image_size'])
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
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
            'num_params': num_params,
            'num_params_M': num_params_M,
            'trainable_params': trainable_params if 'trainable_params' in locals() else num_params,
            'trainable_params_M': trainable_params_M if 'trainable_params_M' in locals() else num_params_M,
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
    parser = argparse.ArgumentParser(description='Train ConvNeXt baseline model')
    parser.add_argument('--image_size', type=int, default=244, help='Image size')
    parser.add_argument('--seeds', type=int, nargs='+', default=[3, 7, 42], help='Random seeds')
    parser.add_argument('--model_name', type=str, default='convnext_tiny', 
                       help='ConvNeXt model name (convnext_tiny, convnext_small, convnext_base)')
    
    args = parser.parse_args()
    
    for seed in args.seeds:
        train_convnext(
            seed=seed,
            image_size=args.image_size,
            model_name=args.model_name
        )

