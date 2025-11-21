# DR17 HK Distance and Color-Aware Complementary Mechanism Experiments

## Overview

This experiment validates the effectiveness of the HK distance and color-aware complementary mechanism on the DR17 dataset. Magnitude channels (mag_g, mag_i, mag_r, mag_u, mag_z) are used instead of RGB color channels, and only redshift prediction is performed (no classification).

## Experiment Groups

According to Table 3 design, the experiments include 4 groups:

1. **baseline**: MSE loss only
2. **color_aware**: MSE + color-aware weighting (based on magnitude color indices g-r, g-i)
3. **hk**: MSE + HK distance (distribution-level matching)
4. **color_aware_hk**: Complete complementary mechanism (color-aware + HK distance)

## Data Validation

The code automatically checks:
- Whether magnitude data exists (mag_g, mag_i, mag_r, mag_u, mag_z)
- Whether magnitude data is valid (no NaN/Inf, range in [10, 30])
- Whether redshift data is valid (no NaN/Inf, range in [0, 3.0])
- Filters invalid samples and reports filtering ratio

## Usage

### Running a Single Experiment

```bash
# Baseline experiment (MSE only)
python hk_distance_experiments/train_hk_dr17.py --experiment_type baseline --seed 42

# Color-aware experiment
python hk_distance_experiments/train_hk_dr17.py --experiment_type color_aware --seed 42

# HK distance experiment
python hk_distance_experiments/train_hk_dr17.py --experiment_type hk --seed 42

# Complete complementary mechanism experiment
python hk_distance_experiments/train_hk_dr17.py --experiment_type color_aware_hk --seed 42
```

### Running All Seeds

```bash
# Run all 3 seeds (36, 42, 199)
python hk_distance_experiments/train_hk_dr17.py --experiment_type baseline
python hk_distance_experiments/train_hk_dr17.py --experiment_type color_aware
python hk_distance_experiments/train_hk_dr17.py --experiment_type hk
python hk_distance_experiments/train_hk_dr17.py --experiment_type color_aware_hk
```

## Output Files

Each experiment generates:
- `64x64_seed{seed}_{experiment_type}_dr17_resnet34_model.pth`: Model weights
- `64x64_seed{seed}_{experiment_type}_dr17_resnet34_model_checkpoint.pth`: Complete checkpoint (including training history, metrics, etc.)

## Data Requirements

- H5 file path: `data/dr17_dataset.h5`
- Required fields:
  - `images`: (N, 256, 256) image data
  - `meta/redshift`: (N,) redshift data
  - `meta/ra`, `meta/dec`: (N,) coordinate data
  - `meta/mag_g`, `meta/mag_i`, `meta/mag_r`, `meta/mag_u`, `meta/mag_z`: (N,) magnitude data

## Data Filtering Rules

- **Magnitude data**:
  - Filter NaN/Inf values
  - Filter range: exclude values <0 or >40 (exclude invalid value markers like -9999.00)
  - **Requirement**: All 5 magnitude channels (mag_g, mag_i, mag_r, mag_u, mag_z) must be valid; partial channel missing is not allowed
  - Normalization range: [0, 40] → [0, 1]
- **Redshift data**:
  - Filter NaN/Inf values
  - Only filter values ≤0 (no upper limit, keep all valid redshift values)
- **Image data**:
  - DR17 dataset uses single-channel grayscale images (not RGB color images)
  - Image shape: (N, 256, 256) single channel
  - Final input: 6 channels = 1 original image channel + 5 magnitude channels (expanded to spatial dimensions)

## Invalid Data Handling

- **Data loading stage**: Filter all invalid samples immediately after loading data
- **Filtering strategy**:
  - If any magnitude channel of a sample is invalid (NaN/Inf or out of range), the entire sample is excluded
  - If redshift data of a sample is invalid (NaN/Inf or ≤0), the entire sample is excluded
  - Filtered data ensures all samples have complete valid data
- **Advantages**:
  - Avoid encountering invalid data during training
  - Ensure color-aware module can correctly compute color indices (g-r, g-i)
  - Ensure HK distance computation uses valid distributions

## Color-Aware Implementation

- Compute color indices from magnitudes: `g-r = mag_g - mag_r`, `g-i = mag_g - mag_i`
- Use `ConfidenceWeighter` for quality stratification:
  - High quality: mean ± 0.5σ, weight 1.0
  - Medium quality: mean ± 1.5σ (excluding high quality), weight 0.7
  - Low quality: out of range, weight 0.3

## HK Distance Implementation

- Use `HellingerKantorovichDistance` for distribution-level matching
- Parameters: delta=1.0, lambda_reg=0.1, n_bins=40
- Weights: lambda_hk=0.035, lambda_mse=0.1

## Notes

1. **IPTW removed**: Table 3 no longer includes IPTW method, only 4 experiment groups remain
2. **Data validation**: Code automatically checks and filters invalid data, ensuring all samples have complete magnitude data
3. **Color features**: Color indices computed from magnitudes are used for color-aware weighting
4. **No classification task**: Only redshift prediction is performed, no classification head included
