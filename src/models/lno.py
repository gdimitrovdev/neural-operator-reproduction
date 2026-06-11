import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.data_utils import get_grid2d


class LowRankKernel2d(nn.Module):
    """
    Low-rank neural-operator layer.

    The kernel integral is parameterized as a separable coordinate-dependent
    kernel K(x, y) ~= sum_r phi_r(x) psi_r(y), applied to hidden features.
    """
    def __init__(self, channels, rank=8, coord_dim=2, hidden_dim=64):
        super(LowRankKernel2d, self).__init__()
        self.channels = channels
        self.rank = rank

        self.phi = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * rank),
        )
        self.psi = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * rank),
        )
        self.local_linear = nn.Linear(channels, channels)

    def forward(self, h, coords):
        batch_size, num_nodes, channels = h.shape
        phi = self.phi(coords).view(batch_size, num_nodes, channels, self.rank)
        psi = self.psi(coords).view(batch_size, num_nodes, channels, self.rank)

        projected = torch.einsum("bncr,bnc->bcr", psi, h) / num_nodes
        integral = torch.einsum("bncr,bcr->bnc", phi, projected)
        return integral + self.local_linear(h)


class LNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, rank=8, num_layers=4, append_grid=True):
        super(LNO2d, self).__init__()
        self.width = width
        self.num_layers = num_layers
        self.append_grid = append_grid

        lift_channels = in_channels + 2 if append_grid else in_channels
        self.lift = nn.Linear(lift_channels, self.width)
        self.low_rank_layers = nn.ModuleList([
            LowRankKernel2d(self.width, rank=rank) for _ in range(num_layers)
        ])
        self.proj1 = nn.Linear(self.width, 128)
        self.proj2 = nn.Linear(128, out_channels)

    def forward(self, x, coords=None):
        batch_size, height, width, _ = x.shape
        if coords is None:
            coords = get_grid2d(height, batch_size=batch_size, device=x.device, flatten=True).to(dtype=x.dtype)
        elif coords.dim() == 4:
            coords = coords.view(batch_size, -1, coords.size(-1))

        x_flat = x.view(batch_size, height * width, -1)
        if self.append_grid:
            x_flat = torch.cat((x_flat, coords), dim=-1)

        h = self.lift(x_flat)
        for layer_idx, layer in enumerate(self.low_rank_layers):
            h = layer(h, coords)
            if layer_idx < self.num_layers - 1:
                h = F.gelu(h)

        h = F.gelu(self.proj1(h))
        out = self.proj2(h)
        return out.view(batch_size, height, width, -1)
