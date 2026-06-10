import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankConv2d(nn.Module):
    """
    LNO layer that approximates the integral kernel in the spatial domain
    via rank-r matrix-vector factorizations.
    """
    def __init__(self, in_channels, out_channels, rank=4):
        super(LowRankConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank

        # Factor networks represented as pointwise convolutions (MLPs)
        self.psi = nn.Conv2d(in_channels, rank, 1)
        self.phi = nn.Conv2d(rank, out_channels, 1)

    def forward(self, x):
        # x shape: (batch, in_channels, H, W)
        batch_size, _, h, w = x.shape
        
        # 1. Compute projection coordinates (psi weight mapping)
        psi_x = self.psi(x)  # Shape: (batch, rank, H, W)
        
        # 2. Integrate globally (spatial average pooling simulates the integral)
        # Effectively computes the inner product over the spatial domain
        inner_prod = torch.mean(psi_x.view(batch_size, self.rank, -1), dim=-1)  # Shape: (batch, rank)
        
        # 3. Project back to output space via phi mapping
        # Expand spatial dimensions back
        inner_prod_expanded = inner_prod.view(batch_size, self.rank, 1, 1).expand(-1, -1, h, w)
        out = self.phi(inner_prod_expanded)  # Shape: (batch, out_channels, H, W)
        
        return out


class LNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, rank=4, num_layers=4):
        super(LNO2d, self).__init__()
        self.width = width
        self.num_layers = num_layers

        # 1. Lifting
        self.lift = nn.Conv2d(in_channels, self.width, 1)

        # 2. LNO Layers
        self.low_rank_convs = nn.ModuleList([LowRankConv2d(self.width, self.width, rank) for _ in range(num_layers)])
        self.local_convs = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(num_layers)])

        # 3. Projection
        self.proj1 = nn.Conv2d(self.width, 128, 1)
        self.proj2 = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        # Convert format to (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        x = self.lift(x)
        
        for i in range(self.num_layers):
            x1 = self.low_rank_convs[i](x)
            x2 = self.local_convs[i](x)
            x = F.gelu(x1 + x2)
            
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        
        return x.permute(0, 2, 3, 1)
