import torch
import torch.nn as nn
import torch.nn.functional as F

class GNOLayer(nn.Module):
    """
    Pure-PyTorch Graph Neural Operator Layer.
    Computes kernel integration on coordinate-based graphs using a threshold radius.
    """
    def __init__(self, in_channels, out_channels, coord_dim=2, radius=0.25):
        super(GNOLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius

        # Kernel MLP: maps Concatenated coordinates (x_i, y_j) -> weight matrix
        # For simplicity, we project to (in_channels * out_channels) to construct the kernel matrix
        self.kernel_mlp = nn.Sequential(
            nn.Linear(coord_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, in_channels * out_channels)
        )
        self.local_linear = nn.Linear(in_channels, out_channels)

    def forward(self, h, coords):
        """
        Args:
            h: Node hidden features of shape (batch, num_nodes, in_channels)
            coords: Node coordinates of shape (batch, num_nodes, coord_dim)
        """
        batch_size, num_nodes, _ = h.shape
        device = h.device

        # 1. Compute pairwise distance matrix
        # coords_i shape: (batch, num_nodes, 1, coord_dim)
        # coords_j shape: (batch, 1, num_nodes, coord_dim)
        coords_i = coords.unsqueeze(2)
        coords_j = coords.unsqueeze(1)
        diff = coords_i - coords_j
        dist = torch.norm(diff, p=2, dim=-1)  # (batch, num_nodes, num_nodes)

        # 2. Build adjacency mask based on threshold radius
        adj_mask = (dist <= self.radius).float()  # (batch, num_nodes, num_nodes)

        # 3. Compute kernel weights for all node pairs
        # Concat coords_i and coords_j -> shape: (batch, num_nodes, num_nodes, coord_dim * 2)
        concat_coords = torch.cat([coords_i.expand(-1, -1, num_nodes, -1),
                                   coords_j.expand(-1, num_nodes, -1, -1)], dim=-1)
        
        # Flatten batch and spatial dimensions for MLP forward pass
        kernel_weights = self.kernel_mlp(concat_coords.view(-1, concat_coords.size(-1)))
        kernel_weights = kernel_weights.view(batch_size, num_nodes, num_nodes, self.in_channels, self.out_channels)

        # 4. Perform weighted message aggregation over neighbors
        # h expanded shape: (batch, 1, num_nodes, in_channels)
        h_expanded = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        # Message calculation: elementwise multiply features with the generated kernel weights
        # (batch, num_nodes, num_nodes, out_channels)
        messages = torch.einsum("bijn,bijnm->bijm", h_expanded, kernel_weights)

        # Mask out non-neighbors
        messages = messages * adj_mask.unsqueeze(-1)

        # Average aggregation
        num_neighbors = adj_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        aggregated = messages.sum(dim=2) / num_neighbors  # (batch, num_nodes, out_channels)

        # 5. Combine non-local kernel integral with pointwise local linear transform
        out = aggregated + self.local_linear(h)
        return out


class GNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, radius=0.25, num_layers=4):
        super(GNO2d, self).__init__()
        self.width = width
        self.num_layers = num_layers

        self.lift = nn.Linear(in_channels, self.width)
        self.gno_layers = nn.ModuleList([GNOLayer(self.width, self.width, coord_dim=2, radius=radius) for _ in range(num_layers)])
        
        self.proj1 = nn.Linear(self.width, 128)
        self.proj2 = nn.Linear(128, out_channels)

    def forward(self, x, coords):
        """
        Args:
            x: Input grid of shape (batch, num_nodes, in_channels)
            coords: Node coordinates of shape (batch, num_nodes, 2)
        """
        h = self.lift(x)
        for i in range(self.num_layers):
            h = F.gelu(self.gno_layers[i](h, coords))
            
        h = F.gelu(self.proj1(h))
        out = self.proj2(h)
        return out
