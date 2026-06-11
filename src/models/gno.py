import torch
import torch.nn as nn
import torch.nn.functional as F

class GNOLayer(nn.Module):
    """
    Pure-PyTorch Graph Neural Operator Layer.
    Computes kernel integration on coordinate-based graphs using a threshold radius.
    """
    def __init__(self, in_channels, out_channels, coord_dim=2, radius=0.25, chunk_size=64):
        super(GNOLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.chunk_size = chunk_size

        # Kernel MLP: maps concatenated coordinates (x_i, y_j) to output-channel kernels.
        self.kernel_mlp = nn.Sequential(
            nn.Linear(coord_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, out_channels)
        )
        self.source_linear = nn.Linear(in_channels, out_channels)
        self.local_linear = nn.Linear(in_channels, out_channels)

    def forward(self, h, coords):
        """
        Args:
            h: Node hidden features of shape (batch, num_nodes, in_channels)
            coords: Node coordinates of shape (batch, num_nodes, coord_dim)
        """
        batch_size, num_nodes, _ = h.shape
        outputs = []
        quadrature_weight = 1.0 / num_nodes

        for start in range(0, num_nodes, self.chunk_size):
            stop = min(start + self.chunk_size, num_nodes)
            target_coords = coords[:, start:stop, :]
            chunk_nodes = stop - start

            coords_i = target_coords.unsqueeze(2)
            coords_j = coords.unsqueeze(1)
            diff = coords_i - coords_j
            dist = torch.norm(diff, p=2, dim=-1)
            adj_mask = (dist <= self.radius).to(dtype=h.dtype)

            concat_coords = torch.cat([
                coords_i.expand(-1, -1, num_nodes, -1),
                coords_j.expand(-1, chunk_nodes, -1, -1),
            ], dim=-1)

            kernel_weights = self.kernel_mlp(concat_coords.reshape(-1, concat_coords.size(-1)))
            kernel_weights = kernel_weights.view(batch_size, chunk_nodes, num_nodes, self.out_channels)
            source_values = self.source_linear(h)
            messages = source_values.unsqueeze(1) * kernel_weights
            messages = messages * adj_mask.unsqueeze(-1)
            aggregated_chunk = messages.sum(dim=2) * quadrature_weight
            outputs.append(aggregated_chunk)

        aggregated = torch.cat(outputs, dim=1)

        # 5. Combine non-local kernel integral with pointwise local linear transform
        out = aggregated + self.local_linear(h)
        return out


class GNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, radius=0.25, num_layers=4, chunk_size=64):
        super(GNO2d, self).__init__()
        self.width = width
        self.num_layers = num_layers

        self.lift = nn.Linear(in_channels, self.width)
        self.gno_layers = nn.ModuleList([
            GNOLayer(self.width, self.width, coord_dim=2, radius=radius, chunk_size=chunk_size)
            for _ in range(num_layers)
        ])
        
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
