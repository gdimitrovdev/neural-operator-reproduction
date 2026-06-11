import unittest
import torch
from src.models.fno import FNO2d
from src.models.lno import LNO2d
from src.models.gno import GNO2d
from src.models.mgno import MGNO2d
from src.utils.data_utils import get_grid2d

class TestNeuralOperators(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 2
        # LOWERED RESOLUTION to prevent Out-Of-Memory (OOM) on laptops/WSL
        self.resolution = 8  
        self.in_channels = 1
        self.out_channels = 1
        self.num_nodes = self.resolution ** 2
        
        # Grid input for FNO/LNO: (batch, H, W, in_channels)
        self.grid_input = torch.randn(self.batch_size, self.resolution, self.resolution, self.in_channels)
        
        # Graph input for GNO/MGNO: (batch, num_nodes, in_channels)
        self.graph_input = torch.randn(self.batch_size, self.num_nodes, self.in_channels)
        self.coords = get_grid2d(self.resolution, batch_size=self.batch_size, flatten=True)

    @torch.no_grad() # Disables gradient tracking to save memory during testing
    def test_fno2d_forward(self):
        model = FNO2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                      modes1=4, modes2=4, width=16, num_layers=4) # lowered modes for small resolution
        output = model(self.grid_input)
        self.assertEqual(output.shape, (self.batch_size, self.resolution, self.resolution, self.out_channels))

    @torch.no_grad()
    def test_lno2d_forward(self):
        model = LNO2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                      width=16, rank=4, num_layers=4)
        output = model(self.grid_input)
        self.assertEqual(output.shape, (self.batch_size, self.resolution, self.resolution, self.out_channels))

    @torch.no_grad()
    def test_gno2d_forward(self):
        model = GNO2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                      width=16, radius=0.25, num_layers=4)
        output = model(self.graph_input, self.coords)
        self.assertEqual(output.shape, (self.batch_size, self.num_nodes, self.out_channels))

    @torch.no_grad()
    def test_mgno2d_forward(self):
        model = MGNO2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                       width=16, radius_fine=0.15, radius_coarse=0.45)
        output = model(self.graph_input, self.coords)
        self.assertEqual(output.shape, (self.batch_size, self.num_nodes, self.out_channels))

if __name__ == '__main__':
    unittest.main()
