"""
Simplified Multi-Task Loss Function for Baseline Models
Maintains interface consistency with UnifiedLossFunction in improved_model.py
but removes advanced features for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMultiTaskLoss(nn.Module):
    """Simple multi-task loss function (classification + redshift regression)"""
    
    def __init__(self, num_classes=10, cls_weight=0.85, redshift_weight=0.15):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.redshift_weight = redshift_weight
        
        self.gamma = 2.5
        self.alpha = 0.6
    
    def focal_loss(self, logits, targets):
        """Focal Loss"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p = F.softmax(logits, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, outputs, targets):
        """
        Compute loss
        
        Args:
            outputs: dict with keys:
                - 'cls_logits': [batch, num_classes]
                - 'redshift_final': [batch]
            targets: dict with keys:
                - 'labels': [batch]
                - 'redshift': [batch]
        
        Returns:
            total_loss: scalar
            loss_dict: loss dictionary
        """
        cls_logits = outputs['cls_logits']
        labels = targets['labels']
        cls_loss = self.focal_loss(cls_logits, labels)
        
        redshift_pred = outputs['redshift_final']
        redshift_true = targets['redshift']
        
        redshift_pred = redshift_pred.squeeze() if redshift_pred.dim() > 1 else redshift_pred
        redshift_true = redshift_true.squeeze() if redshift_true.dim() > 1 else redshift_true
        
        redshift_pred_safe = torch.clamp(redshift_pred, min=1e-6)
        redshift_true_safe = torch.clamp(redshift_true, min=1e-6)
        
        log_pred = torch.log1p(redshift_pred_safe)
        log_true = torch.log1p(redshift_true_safe)
        
        redshift_loss = F.mse_loss(log_pred, log_true)
        
        total_loss = self.cls_weight * cls_loss + self.redshift_weight * redshift_loss
        
        loss_dict = {
            'cls_loss': cls_loss.detach(),
            'redshift_loss': redshift_loss.detach(),
            'total_loss': total_loss.detach(),
            'focal_gamma': torch.tensor(self.gamma)
        }
        
        return total_loss, loss_dict
    
    def set_weights(self, cls_weight, redshift_weight):
        """Dynamically set weights (interface compatibility)"""
        self.cls_weight = cls_weight
        self.redshift_weight = redshift_weight
