"""
Load and evaluate model from saved .pth checkpoint file.
If only model weights are available, training history cannot be fully restored, but final performance can be evaluated.

Usage:
    python baseline_models/load_training_history.py baseline_resnet34_model.pth
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_checkpoint_and_plot(checkpoint_path):
    """
    Load checkpoint and plot training curves.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth or _checkpoint.pth)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'train_accuracies' in checkpoint:
            print("Found training history in checkpoint!")
            
            train_accs = checkpoint['train_accuracies']
            val_accs = checkpoint['val_accuracies']
            epochs = list(range(1, len(train_accs) + 1))
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(epochs, [a * 100 for a in train_accs], 'b-', label='Train Acc', linewidth=2)
            plt.plot(epochs, [a * 100 for a in val_accs], 'r-', label='Val Acc', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if 'val_log_mses' in checkpoint:
                plt.subplot(1, 2, 2)
                plt.plot(epochs, checkpoint['val_log_mses'], 'g-', label='Val Log-MSE', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Log-MSE')
                plt.title('Validation Log-MSE')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_path = checkpoint_path.replace('.pth', '_training_curves.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {output_path}")
            
            print("\nTraining Statistics:")
            print(f"  Total epochs: {len(train_accs)}")
            print(f"  Best val acc: {max(val_accs)*100:.2f}%")
            print(f"  Final train acc: {train_accs[-1]*100:.2f}%")
            print(f"  Final val acc: {val_accs[-1]*100:.2f}%")
            
            if 'best_val_acc' in checkpoint:
                print(f"  Best val acc (recorded): {checkpoint['best_val_acc']*100:.2f}%")
            
            return checkpoint
        else:
            print("Warning: Checkpoint does not contain training history")
            print("  Only model weights are available (model_state_dict)")
            print("  You can only evaluate the model, but cannot plot training curves")
            return checkpoint
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python load_training_history.py <checkpoint_path>")
        print("Example: python load_training_history.py baseline_resnet34_model_checkpoint.pth")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    load_checkpoint_and_plot(checkpoint_path)

