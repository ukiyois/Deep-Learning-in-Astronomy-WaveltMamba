"""
ResNet-34 Multi-Task Baseline Model
Used for performance comparison with improved_model.py

Architecture:
- ResNet-34 as image feature extractor
- CoordPyramid as coordinate encoder
- Classification head: 10-class classification
- Redshift prediction head: regression task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CoordPyramid(nn.Module):
    """Coordinate pyramid encoder (consistent with improved_model)"""
    
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


class ResNet34MultiTask(nn.Module):
    """ResNet-34 Multi-Task Model"""
    
    def __init__(self, num_classes: int = 10, device: str = 'cuda', 
                 use_spectral: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.use_spectral = use_spectral
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        resnet = models.resnet34(pretrained=False)
        self.image_encoder = nn.Sequential(
            resnet.conv1,
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
        
        self.coordinate_encoder = CoordPyramid(input_dim=2, target_dim=64)
        
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + 64, 256),  # 512 + 64 = 576
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.redshift_predictor = nn.Sequential(
            nn.Linear(image_feature_dim + 64, 256),  # 512 + 64 = 576
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.redshift_scale = nn.Parameter(torch.tensor(2.0))
        
        with torch.no_grad():
            self.redshift_predictor[-1].bias.fill_(0.5)
    
    def forward(self, images: torch.Tensor, coordinates: torch.Tensor, 
                spectral_features: torch.Tensor = None) -> dict:
        """
        Forward pass
        
        Args:
            images: [batch, 3, 128, 128] tensor
            coordinates: [batch, 2] (RA, DEC)
            spectral_features: unused (interface consistency)
        
        Returns:
            dict with keys:
                - 'cls_logits': [batch, num_classes]
                - 'redshift_final': [batch]
                - 'redshift_pred': [batch] (alias)
                - 'features': dict (for loss function compatibility)
        """
        batch_size = images.size(0)
        
        image_features = self.image_encoder(images)
        
        coord_output = self.coordinate_encoder(coordinates)
        coord_features = coord_output[0] if isinstance(coord_output, tuple) else coord_output
        
        fused_features = torch.cat([image_features, coord_features], dim=1)
        
        cls_logits = self.classifier(fused_features)
        
        redshift_raw = self.redshift_predictor(fused_features).squeeze(-1)
        redshift_final = F.softplus(redshift_raw) * torch.clamp(self.redshift_scale, min=0.5)
        redshift_final = torch.clamp(redshift_final, min=1e-6)
        
        features = {
            'classification_features': image_features,
            'raw_features': {
                'color': torch.zeros(batch_size, 4, device=images.device)
            }
        }
        
        return {
            'cls_logits': cls_logits,
            'redshift_final': redshift_final,
            'redshift_pred': redshift_final,
            'features': features
        }
    
    def compute_loss(self, outputs: dict, targets: dict) -> tuple:
        """Compute loss (interface compatibility, handled by training script)"""
        raise NotImplementedError("Loss computation handled by training script")

