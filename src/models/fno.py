import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_grid_1d(batch_size, size, device, dtype):
    grid = torch.linspace(0, 1, size, device=device, dtype=dtype)
    return grid.view(1, size, 1).repeat(batch_size, 1, 1)


def _get_grid_2d(batch_size, height, width, device, dtype):
    grid_x = torch.linspace(0, 1, height, device=device, dtype=dtype)
    grid_y = torch.linspace(0, 1, width, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep (Height)
        self.modes2 = modes2  # Number of Fourier modes to keep (Width)

        # Scale factor for parameter initialization
        scale = 1.0 / (in_channels * out_channels)
        
        # Initialize complex weights for low-frequency modes
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # Einstein summation for batched complex matrix multiplication
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Compute 2D Fourier transform (real to complex)
        # Output shape: (batch, channels, H, W//2 + 1)
        x_ft = torch.fft.rfft2(x)

        # Initialize output complex spectrum with zeros
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # Slice and multiply low-frequency corners
        # Top-left and bottom-left corners (handles positive and negative wrap-around frequencies)
        modes1 = min(self.modes1, x_ft.size(-2))
        modes2 = min(self.modes2, x_ft.size(-1))

        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2]
        )
        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
            x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2]
        )

        # Compute inverse Fourier transform back to spatial domain
        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x_out


class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width, num_layers=4, padding=0, append_grid=True):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.padding = padding
        self.append_grid = append_grid

        # 1. Lifting Step (maps input channels to model width)
        lift_channels = in_channels + 2 if append_grid else in_channels
        self.lift = nn.Conv2d(lift_channels, self.width, 1)

        # 2. Fourier Layers
        self.spectral_convs = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(num_layers)])
        self.local_convs = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(num_layers)])

        # 3. Projection Step (maps model width back to output channels)
        self.proj1 = nn.Conv2d(self.width, 128, 1)
        self.proj2 = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        if self.append_grid:
            grid = _get_grid_2d(x.size(0), x.size(1), x.size(2), x.device, x.dtype)
            x = torch.cat((x, grid), dim=-1)

        # Input shape: (batch, height, width, channels) -> Permute to (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        x = self.lift(x)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        
        for i in range(self.num_layers):
            # Parallel path: Spectral Conv (non-local) + Local Conv (residual linear mapping)
            x1 = self.spectral_convs[i](x)
            x2 = self.local_convs[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
            
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        
        # Permute back to spatial format (batch, height, width, channels)
        return x.permute(0, 2, 3, 1)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        modes1 = min(self.modes1, x_ft.size(-1))
        out_ft[:, :, :modes1] = self.compl_mul1d(x_ft[:, :, :modes1], self.weights1[:, :, :modes1])
        return torch.fft.irfft(out_ft, n=x.size(-1))

class FNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, width, num_layers=4, append_grid=True):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.num_layers = num_layers
        self.append_grid = append_grid

        lift_channels = in_channels + 1 if append_grid else in_channels
        self.lift = nn.Conv1d(lift_channels, self.width, 1)
        self.spectral_convs = nn.ModuleList([SpectralConv1d(self.width, self.width, self.modes1) for _ in range(num_layers)])
        self.local_convs = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(num_layers)])
        self.proj1 = nn.Conv1d(self.width, 128, 1)
        self.proj2 = nn.Conv1d(128, out_channels, 1)

    def forward(self, x):
        if self.append_grid:
            grid = _get_grid_1d(x.size(0), x.size(1), x.device, x.dtype)
            x = torch.cat((x, grid), dim=-1)

        x = x.permute(0, 2, 1) # (batch, channels, x)
        x = self.lift(x)
        for i in range(self.num_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.local_convs[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x.permute(0, 2, 1)
