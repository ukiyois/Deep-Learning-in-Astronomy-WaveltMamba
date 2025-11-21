"""
ResNet-34 Model for HK Distance Experiments with Magnitude Channels
- 6 input channels: 1 image + 5 magnitude channels (g, i, r, u, z)
- Only redshift prediction (no classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CoordPyramid(nn.Module):
    """Coordinate pyramid encoder"""
    
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.local_scale = nn.Sequential(
            nn.Linear(input_dim, target_dim // 4),
            nn.ReLU(),
            nn.Linear(target_dim // 4, target_dim // 4),
            nn.ReLU()
        )
        
        self.medium_scale = nn.Sequential(
            nn.Linear(input_dim, target_dim // 2),
            nn.ReLU(),
            nn.Linear(target_dim // 2, target_dim // 2),
            nn.ReLU()
        )
        
        self.global_scale = nn.Sequential(
            nn.Linear(input_dim, target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim),
            nn.ReLU()
        )
        
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(target_dim * 3, target_dim * 2),
            nn.ReLU(),
            nn.Linear(target_dim * 2, target_dim),
            nn.ReLU()
        )
        
        self.scale_alignment = nn.ModuleDict({
            'local_to_target': nn.Linear(target_dim // 4, target_dim),
            'medium_to_target': nn.Linear(target_dim // 2, target_dim),
            'global_to_target': nn.Linear(target_dim, target_dim)
        })
    
    def forward(self, x: torch.Tensor):
        """Forward pass"""
        local_features = self.local_scale(x)
        medium_features = self.medium_scale(x)
        global_features = self.global_scale(x)
        
        local_aligned = self.scale_alignment['local_to_target'](local_features)
        medium_aligned = self.scale_alignment['medium_to_target'](medium_features)
        global_aligned = self.scale_alignment['global_to_target'](global_features)
        
        fused_features = torch.cat([local_aligned, medium_aligned, global_aligned], dim=1)
        final_output = self.cross_scale_fusion(fused_features)
        
        return final_output, {
            'local_features': local_features,
            'medium_features': medium_features,
            'global_features': global_features
        }


class ResNet34HKMagnitude(nn.Module):
    """
    ResNet-34 Model for HK Distance Experiments with Magnitude Channels
    
    Architecture:
    - ResNet-34 as image feature extractor (modified for 6 input channels)
    - CoordPyramid as coordinate encoder
    - Redshift prediction head only (no classification)
    """
    
    def __init__(self, in_channels: int = 6, device: str = 'cuda'):
        """
        Args:
            in_channels: Number of input channels (default: 6 = 1 image + 5 magnitudes)
            device: Device to use
        """
        super().__init__()
        self.in_channels = in_channels
        
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Load ResNet-34 backbone
        resnet = models.resnet34(pretrained=False)
        
        # Modify first conv layer to accept 6 channels instead of 3
        original_conv1 = resnet.conv1
        self.image_encoder_conv1 = nn.Conv2d(
            in_channels, 
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        
        # Initialize weights: use pretrained weights for first 3 channels, 
        # initialize new channels with small random values
        if in_channels == 6:
            with torch.no_grad():
                self.image_encoder_conv1.weight[:, :3] = original_conv1.weight
                # Initialize magnitude channels with small values
                nn.init.kaiming_normal_(
                    self.image_encoder_conv1.weight[:, 3:], 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
                self.image_encoder_conv1.weight[:, 3:] *= 0.1  # Smaller initial weights
        
        # Build image encoder
        self.image_encoder = nn.Sequential(
            self.image_encoder_conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten()
        )
        image_feature_dim = 512
        
        # Coordinate encoder
        self.coordinate_encoder = CoordPyramid(input_dim=2, target_dim=64)
        
        # Redshift predictor only (no classification head)
        self.redshift_predictor = nn.Sequential(
            nn.Linear(image_feature_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        self.redshift_scale = nn.Parameter(torch.tensor(2.0))
        
        # Initialize redshift predictor bias
        with torch.no_grad():
            self.redshift_predictor[-1].bias.fill_(0.5)
    
    def forward(self, images: torch.Tensor, coordinates: torch.Tensor) -> dict:
        """
        Forward pass
        
        Args:
            images: [batch, 6, H, W] tensor (1 image channel + 5 magnitude channels)
            coordinates: [batch, 2] (RA, DEC)
        
        Returns:
            dict with keys:
                - 'redshift_final': [batch] final redshift prediction
                - 'redshift_pred': [batch] alias for redshift_final
                - 'features': dict (for loss function compatibility)
        """
        batch_size = images.size(0)
        
        # Check for NaN/Inf in inputs
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f"Warning: NaN/Inf detected in input images!")
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(coordinates).any() or torch.isinf(coordinates).any():
            print(f"Warning: NaN/Inf detected in input coordinates!")
            coordinates = torch.nan_to_num(coordinates, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Extract image features
        image_features = self.image_encoder(images)
        
        # Check for NaN/Inf in image features
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            print(f"ERROR: NaN/Inf in image_features! Range: [{image_features.min():.2f}, {image_features.max():.2f}]")
            raise RuntimeError("NaN/Inf detected in image_features. Training terminated.")
        
        # Encode coordinates
        coord_output = self.coordinate_encoder(coordinates)
        coord_features = coord_output[0] if isinstance(coord_output, tuple) else coord_output
        
        # Check for NaN/Inf in coordinate features
        if torch.isnan(coord_features).any() or torch.isinf(coord_features).any():
            print(f"ERROR: NaN/Inf in coord_features! Range: [{coord_features.min():.2f}, {coord_features.max():.2f}]")
            raise RuntimeError("NaN/Inf detected in coord_features. Training terminated.")
        
        # Fuse image and coordinate features
        fused_features = torch.cat([image_features, coord_features], dim=1)
        
        # Check for NaN/Inf in fused features
        if torch.isnan(fused_features).any() or torch.isinf(fused_features).any():
            print(f"ERROR: NaN/Inf in fused_features! Range: [{fused_features.min():.2f}, {fused_features.max():.2f}]")
            raise RuntimeError("NaN/Inf detected in fused_features. Training terminated.")
        
        # Predict redshift
        redshift_raw = self.redshift_predictor(fused_features).squeeze(-1)
        
        # Check for NaN/Inf in raw predictions before softplus
        if torch.isnan(redshift_raw).any() or torch.isinf(redshift_raw).any():
            print(f"ERROR: NaN/Inf in redshift_raw! Range: [{redshift_raw.min():.2f}, {redshift_raw.max():.2f}]")
            raise RuntimeError("NaN/Inf detected in redshift_raw. Training terminated.")
        
        # Use more numerically stable softplus with clamping
        # Clamp input to softplus to prevent overflow
        redshift_raw_clamped = torch.clamp(redshift_raw, min=-10.0, max=10.0)
        redshift_final = F.softplus(redshift_raw_clamped) * torch.clamp(self.redshift_scale, min=0.5, max=2.0)
        redshift_final = torch.clamp(redshift_final, min=1e-6, max=2.0)
        
        # Final check for NaN/Inf in final predictions
        if torch.isnan(redshift_final).any() or torch.isinf(redshift_final).any():
            print(f"ERROR: NaN/Inf in redshift_final after processing!")
            raise RuntimeError("NaN/Inf detected in redshift_final. Training terminated.")
        
        # Features dict for loss function compatibility
        features = {
            'classification_features': image_features,  # Not used but required
            'raw_features': {
                'color': torch.zeros(batch_size, 4, device=images.device)  # Placeholder
            }
        }
        
        return {
            'redshift_final': redshift_final,
            'redshift_pred': redshift_final,
            'features': features
        }
    
    def compute_loss(self, outputs: dict, targets: dict) -> tuple:
        """Compute loss (interface compatibility, handled by training script)"""
        raise NotImplementedError("Loss computation handled by training script")

