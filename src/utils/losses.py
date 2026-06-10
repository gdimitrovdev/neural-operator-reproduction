import torch
import torch.nn as nn

class RelativeL2Loss(nn.Module):
    """
    Computes the relative L2 loss between prediction and ground truth.
    Supports batched inputs across arbitrary spatial dimensions.
    """
    def __init__(self, reduction='mean', eps=1e-8):
        super(RelativeL2Loss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pr, gt):
        # Flatten all spatial and channel dimensions
        batch_size = pr.size(0)
        pr_flat = pr.view(batch_size, -1)
        gt_flat = gt.view(batch_size, -1)
        
        # Calculate L2 norms
        diff_norms = torch.norm(pr_flat - gt_flat, p=2, dim=1)
        gt_norms = torch.norm(gt_flat, p=2, dim=1)
        
        # Compute relative error per instance
        relative_errors = diff_norms / (gt_norms + self.eps)
        
        if self.reduction == 'mean':
            return relative_errors.mean()
        elif self.reduction == 'sum':
            return relative_errors.sum()
        else:
            return relative_errors
