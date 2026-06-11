import unittest

import torch

from src.models.fno import FNO2d
from src.models.lno import LowRankKernel2d
from src.utils.data_utils import UnitGaussianNormalizer, get_grid2d


def sample_smooth_function(resolution):
    """Sample a fixed smooth function on a [0, 1]^2 grid: same underlying
    function at every resolution, so outputs are comparable across grids."""
    ticks = torch.linspace(0, 1, resolution)
    grid_x, grid_y = torch.meshgrid(ticks, ticks, indexing="ij")
    f = torch.sin(2 * torch.pi * grid_x) * torch.cos(2 * torch.pi * grid_y)
    return f.unsqueeze(0).unsqueeze(-1)  # (1, res, res, 1)


class TestDiscretizationInvariance(unittest.TestCase):
    """The defining property of a neural operator (paper Section 2): the same
    network evaluated on finer discretizations of the same input function
    should produce (approximately) the same output function."""

    def test_fno2d_output_is_resolution_invariant(self):
        torch.manual_seed(0)
        # padding=0: fixed-pixel padding spans a different physical length at
        # each resolution, which would confound the invariance check.
        model = FNO2d(in_channels=1, out_channels=1, modes1=4, modes2=4,
                      width=16, num_layers=4, padding=0, append_grid=True)
        model.eval()

        # 33 -> 65 keeps the grids nested: linspace(0,1,65)[::2] == linspace(0,1,33)
        with torch.no_grad():
            out_coarse = model(sample_smooth_function(33))
            out_fine = model(sample_smooth_function(65))
        out_fine_on_coarse = out_fine[:, ::2, ::2, :]

        diff = torch.norm(out_fine_on_coarse - out_coarse)
        rel = (diff / torch.norm(out_coarse)).item()
        self.assertLess(rel, 0.1, f"FNO output changed by {rel:.3f} relative L2 across resolutions")
        # Sanity: the output is not trivially zero/constant.
        self.assertGreater(out_coarse.std().item(), 1e-4)


class TestLNOSpatialVariation(unittest.TestCase):
    """Regression test for the constant-field bug: the low-rank kernel
    integral sum_r phi_r(x) <psi_r, v> must depend on the output location x,
    i.e. the non-local term cannot be spatially constant."""

    def test_low_rank_integral_varies_spatially(self):
        torch.manual_seed(0)
        layer = LowRankKernel2d(channels=8, rank=4)
        with torch.no_grad():
            # Zero the pointwise local path so only the kernel integral remains.
            layer.local_linear.weight.zero_()
            layer.local_linear.bias.zero_()

        coords = get_grid2d(8, batch_size=2, flatten=True)
        h = torch.randn(2, 64, 8)
        with torch.no_grad():
            out = layer(h, coords)

        spatial_std = out.std(dim=1)  # (batch, channels)
        self.assertGreater(spatial_std.max().item(), 1e-4,
                           "low-rank kernel integral is spatially constant")


class TestUnitGaussianNormalizer(unittest.TestCase):

    def test_encode_decode_roundtrip_and_statistics(self):
        torch.manual_seed(0)
        x = torch.randn(50, 9, 9, 1) * 3.0 + 5.0
        normalizer = UnitGaussianNormalizer(x)
        encoded = normalizer.encode(x)
        self.assertLess(encoded.mean().abs().item(), 1e-5)
        self.assertLess((encoded.std(dim=0) - 1).abs().max().item(), 1e-2)
        decoded = normalizer.decode(encoded)
        self.assertLess((decoded - x).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
