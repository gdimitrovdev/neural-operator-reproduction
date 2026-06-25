import torch
import torch.nn as nn
import torch.nn.functional as F

class GNOLayer(nn.Module):
    """
    Pure-PyTorch Graph Neural Operator Layer.

    Discretizes the truncated kernel integral u(x) = (1/J) sum_{y in B(x,r)} k(x,y) v(y)
    (paper section 4.1). The neighborhood B(x,r) is the paper's domain truncation to a
    ball of radius r.

    `chunk_size` is retained for config compatibility but unused (the sparse path needs
    no chunking).
    """
    def __init__(self, in_channels, out_channels, coord_dim=2, radius=0.25, chunk_size=64, kernel_hidden_dim=64):
        super(GNOLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.chunk_size = chunk_size

        # Kernel MLP: maps concatenated coordinates (x_i, y_j) to output-channel kernels.
        self.kernel_mlp = nn.Sequential(
            nn.Linear(coord_dim * 2, kernel_hidden_dim),
            nn.GELU(),
            nn.Linear(kernel_hidden_dim, out_channels)
        )
        self.source_linear = nn.Linear(in_channels, out_channels)
        self.local_linear = nn.Linear(in_channels, out_channels)
        self._edge_cache = {}

    def _radius_edges(self, coords_2d):
        """Edge list (target, source) of pairs within self.radius, for a single grid.

        The grid is identical across the batch and fixed across epochs, so the radius
        graph is built once (a one-off dense distance computation, no autograd) and
        cached by node count. Self-pairs are included (dist 0 <= r), matching the
        previous dense mask.
        """
        n = coords_2d.size(0)
        cached = self._edge_cache.get(n)
        if cached is None:
            with torch.no_grad():
                dist = torch.cdist(coords_2d, coords_2d)
                tgt, src = torch.where(dist <= self.radius)
            cached = (tgt, src)
            self._edge_cache[n] = cached
        tgt, src = cached
        return tgt.to(coords_2d.device), src.to(coords_2d.device)

    def forward(self, h, coords):
        """
        Args:
            h: Node hidden features of shape (batch, num_nodes, in_channels)
            coords: Node coordinates of shape (batch, num_nodes, coord_dim)
        """
        batch_size, num_nodes, _ = h.shape
        tgt, src = self._radius_edges(coords[0])

        edge_coords = torch.cat([coords[0][tgt], coords[0][src]], dim=-1)
        kernel_weights = self.kernel_mlp(edge_coords)

        source_values = self.source_linear(h)
        messages = kernel_weights.unsqueeze(0) * source_values[:, src]

        aggregated = h.new_zeros(batch_size, num_nodes, self.out_channels)
        aggregated.index_add_(1, tgt, messages)
        aggregated = aggregated / num_nodes

        return aggregated + self.local_linear(h)


class GNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, radius=0.25, num_layers=4, chunk_size=64, kernel_hidden_dim=64, proj_width=128):
        super(GNO2d, self).__init__()
        self.width = width
        self.num_layers = num_layers

        self.lift = nn.Linear(in_channels, self.width)
        self.gno_layers = nn.ModuleList([
            GNOLayer(self.width, self.width, coord_dim=2, radius=radius, chunk_size=chunk_size, kernel_hidden_dim=kernel_hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.proj1 = nn.Linear(self.width, proj_width)
        self.proj2 = nn.Linear(proj_width, out_channels)

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
