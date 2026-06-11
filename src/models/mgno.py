import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gno import GNOLayer

class MGNO2d(nn.Module):
    """
    Multipole Graph Neural Operator.
    Implements a 2-level V-cycle (Fine Grid -> Coarse Grid -> Fine Grid)
    using coordinate-based downsampling and GNO layers.
    """
    def __init__(self, in_channels, out_channels, width, radius_fine=0.15, radius_coarse=0.45):
        super(MGNO2d, self).__init__()
        self.width = width

        # Lifting
        self.lift = nn.Linear(in_channels, self.width)

        # Multi-scale layers
        self.fine_layer1 = GNOLayer(self.width, self.width, radius=radius_fine)
        self.coarse_layer = GNOLayer(self.width, self.width, radius=radius_coarse)
        self.fine_layer2 = GNOLayer(self.width, self.width, radius=radius_fine)

        # Projection
        self.proj1 = nn.Linear(self.width, 128)
        self.proj2 = nn.Linear(128, out_channels)

    def forward(self, x, coords):
        """
        Args:
            x: Fine grid features (batch, num_nodes, in_channels)
            coords: Fine grid coordinates (batch, num_nodes, 2)
        """
        batch_size, num_nodes, _ = x.shape
        h_fine = self.lift(x)

        # 1. Fine-scale convolution
        h_fine = F.gelu(self.fine_layer1(h_fine, coords))

        # 2. Downward Pass: Coarsen the coordinates (Subsample every 4th node for simplicity)
        # In practice, this represents spatial clustering or grid decimation
        coarse_indices = torch.arange(0, num_nodes, step=4, device=x.device)
        coords_coarse = coords[:, coarse_indices, :]
        h_coarse_pooled = h_fine[:, coarse_indices, :]

        # 3. Coarse-scale convolution (large-range interactions)
        h_coarse = F.gelu(self.coarse_layer(h_coarse_pooled, coords_coarse))

        # 4. Upward Pass: Interpolate coarse features back to fine coordinates
        # Compute inverse distances between fine coordinates and coarse coordinates
        # Shape: (batch, num_nodes, num_coarse_nodes)
        dists = torch.cdist(coords, coords_coarse, p=2)
        weights = 1.0 / (dists + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize weights

        # Perform coordinate-based linear interpolation
        h_coarse_upsampled = torch.bmm(weights, h_coarse)

        # 5. Fuse fine-scale and upsampled coarse-scale features
        h_fused = F.gelu(h_fine + h_coarse_upsampled)

        # 6. Final fine-scale refinement convolution
        h_fused = F.gelu(self.fine_layer2(h_fused, coords))

        # Output projection
        out = F.gelu(self.proj1(h_fused))
        return self.proj2(out)
