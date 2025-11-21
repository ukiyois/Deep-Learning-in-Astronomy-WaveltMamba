# Baseline Models

This directory contains baseline model implementations for comparison with the improved model. These models are used to demonstrate the performance improvements of our proposed architecture.

## Overview

The baseline models include:
- **ResNet-34**: Multi-task ResNet-34 baseline
- **DenseNet-121**: Multi-task DenseNet-121 baseline
- **ViT (Vision Transformer)**: Multi-task ViT baseline
- **Swin Transformer**: Multi-task Swin Transformer baseline

All baseline models are trained for the same multi-task learning problem: galaxy classification (10 classes) and redshift prediction.

## File Structure

```
baseline_models/
├── README.md                    # This file
├── train_resnet34.py           # ResNet-34 training script
├── train_densenet.py            # DenseNet-121 training script
├── train_vit.py                 # ViT training script
├── train_swin.py                # Swin Transformer training script
├── resnet34_multitask.py        # ResNet-34 model architecture
├── densenet_multitask.py         # DenseNet-121 model architecture
├── vit_multitask.py              # ViT model architecture
├── swin_multitask.py             # Swin Transformer model architecture
├── simple_loss.py                # Simple multi-task loss function
└── load_training_history.py      # Utility to load and visualize training history
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)
- Required packages: `torch`, `torchvision`, `timm`, `numpy`, `h5py`, `scikit-learn`

### Training a Baseline Model

#### ResNet-34

```bash
python baseline_models/train_resnet34.py
```

Default configuration:
- Image size: 64×64
- Batch size: 256
- Random seed: 3
- Output: `64x64_seed3_baseline_resnet34_model.pth`

#### DenseNet-121

```bash
python baseline_models/train_densenet.py
```

Default configuration:
- Image size: 128×128
- Batch size: 128
- Random seed: 42
- Output: `128x128_seed42_baseline_densenet_model.pth`

#### ViT (Vision Transformer)

```bash
python baseline_models/train_vit.py
```

Default configuration:
- Image size: 256×256
- Batch size: 64
- Random seeds: 3, 7, 42
- Output: `seed{seed}_baseline_vit_model.pth`

#### Swin Transformer

```bash
python baseline_models/train_swin.py
```

Default configuration:
- Image size: 256×256
- Batch size: 64
- Random seeds: 3, 7, 42
- Output: `seed{seed}_baseline_swin_model.pth`

## Model Architectures

### ResNet-34 Multi-Task

- **Backbone**: ResNet-34 (from torchvision)
- **Coordinate Encoder**: CoordPyramid (multi-scale coordinate encoding)
- **Classification Head**: 10-class classification
- **Redshift Head**: Regression with coordinate and color features
- **Parameters**: ~21.69M

### DenseNet-121 Multi-Task

- **Backbone**: DenseNet-121 (from timm)
- **Coordinate Encoder**: CoordPyramid
- **Classification Head**: 10-class classification
- **Redshift Head**: Regression with coordinate and color features
- **Parameters**: ~8.38M

### ViT Multi-Task

- **Backbone**: ViT-Base (from timm, patch size 16)
- **Coordinate Encoder**: CoordPyramid
- **Classification Head**: 10-class classification
- **Redshift Head**: Regression with coordinate and color features
- **Parameters**: ~86M

### Swin Transformer Multi-Task

- **Backbone**: Swin-Tiny (from timm)
- **Coordinate Encoder**: CoordPyramid
- **Classification Head**: 10-class classification
- **Redshift Head**: Regression with coordinate and color features
- **Parameters**: ~28M

## Training Configuration

All baseline models use the same training configuration:

- **Dataset**: Galaxy10_DECals.h5 (all available samples)
- **Validation Split**: 30%
- **Sample Limit**: None (uses all available data)
- **Epochs**: 160
- **Optimizer**: AdamW
- **Learning Rate**: 
  - Initial: 5e-5
  - Max: 0.0006
  - Min: 1e-6
- **Scheduler**: UBAScheduler (Unified Budget-Aware)
- **Weight Decay**: 3e-4 to 5e-4
- **Loss Function**: SimpleMultiTaskLoss (weighted combination of classification and redshift loss)

## Output Files

Each training script generates:

1. **Model weights**: `{image_size}_seed{seed}_baseline_{model}_model.pth`
2. **Checkpoint**: `{image_size}_seed{seed}_baseline_{model}_model_checkpoint.pth`
   - Contains: model weights, training history, validation metrics, model statistics

## Loading Training History

To visualize training curves from a checkpoint:

```bash
python baseline_models/load_training_history.py <checkpoint_path>
```

Example:
```bash
python baseline_models/load_training_history.py 64x64_seed42_baseline_resnet34_model_checkpoint.pth
```

This will:
- Load the checkpoint
- Plot training/validation accuracy curves
- Plot validation Log-MSE curve (if available)
- Print training statistics
- Save plots to `{checkpoint_path}_training_curves.png`

## Customizing Training

You can customize training by modifying the `config` dictionary in each training script, or by passing parameters:

```python
from baseline_models.train_resnet34 import train_resnet34

train_resnet34(
    seed=42,
    image_size=128,
    batch_size=128,
    num_epochs=200
)
```

## Comparison with Improved Model

The baseline models are used for comparison in the paper. Key differences:

| Feature | Baseline Models | Improved Model |
|---------|----------------|----------------|
| Feature Extractor | Standard CNN/Transformer | WaveletMamba |
| Coordinate Encoding | Fixed CoordPyramid | LearnableSinusoidalCoordEncoder |
| Task Modeling | Independent heads | TaskRelationshipModel |
| Loss Function | Simple weighted loss | UnifiedLossFunction (Focal + VIB + HK) |
| Parameters | 8.38M - 86M | 3.54M |

## Notes

- All baseline models are trained from scratch (no pretrained weights)
- Training uses the same dataset and data augmentation as the improved model
- Results are reported in the paper's comparison tables
- Baseline models serve as reference points for ablation studies

## Citation

If you use these baseline models, please cite the paper and acknowledge the baseline architectures (ResNet, DenseNet, ViT, Swin).

