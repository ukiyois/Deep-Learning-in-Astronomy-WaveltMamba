# Deep-Learning-in-Astronomy-WaveltMamba
PyTorch implementation of WaveletMamba for galaxy morphology classification and redshift prediction. Features multi-scale feature extraction, resolution invariance, HK distance regularization, and LSI-enhanced VIB. Achieves state-of-the-art performance with compact 3.54M-parameter design. Includes training scripts and baseline model comparisons.
Prerequisites
- Python 3.8+
- PyTorch 1.12+ (with CUDA support recommended)
- CUDA-capable GPU (recommended for training)

Installation

pip install -r requirements.txt

Data Preparation
1. Download the Galaxy10 DECals dataset from [here](https://astronn.readthedocs.io/en/latest/galaxy10.html)
2. Download the DR17 dataset form[here](https://zenodo.org/records/17649622)
Training

Train the model with default settings:
python galaxy10_training.py --seed 42


Customize training parameters:
python galaxy10_training.py \
    --seed 42 \
    --image_size 64 \
    --batch_size 256 \
    --use_task_relationship True \
    --use_coord_encoding True


WaveletMamba
The core innovation is the WaveletMamba module, which:
- Uses learnable Gabor filters for directional feature extraction (spiral, radial, tangential)
- Applies state space models (Mamba) for efficient sequence modeling with O(n) complexity
- Combines multi-resolution features through adaptive fusion
- Captures both local and global structures in galaxy images



Project Structure
galaxy-classification/
├── README.md                    # This file
├── requirements.txt            
├── LICENSE                      
├── improved_model.py            # Core model architecture
├── galaxy10_training.py         # Main training script
│
│
├── baseline_models/  .....            
│
└── hk_distance_experiments/     
    └── train_hk_dr17.py


Collaborator's link：[here]https://github.com/zhehanruoyan1234-hash    [here]https://github.com/ryxikn
