import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Ensure the src directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.fno import FNO2d
from src.utils.losses import RelativeL2Loss
from src.utils.data_utils import MatDataset

def main():
    # 1. Load Config
    with open("configs/darcy_fno2d.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # 2. Load Data
    # Darcy data is (N, 421, 421). The paper subsamples it to 85x85 by taking every 5th point.
    full_dataset = MatDataset(config['data']['file_path'], x_key=config['data']['x_key'], y_key=config['data']['y_key'])
    
    # Manually split and subsample
    x_train = full_dataset.x[:1000, ::5, ::5, :]  # 85x85
    y_train = full_dataset.y[:1000, ::5, ::5, :]
    x_test = full_dataset.x[-200:, ::5, ::5, :]
    y_test = full_dataset.y[-200:, ::5, ::5, :]

    train_loader = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=config['training']['batch_size'], shuffle=False)

    # 3. Initialize Model
    model = FNO2d(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        modes1=config['model']['modes1'],
        modes2=config['model']['modes2'],
        width=config['model']['width'],
        num_layers=config['model']['num_layers']
    ).to(device)

    # 4. Optimizer, Scheduler, and Loss
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['step_size'], gamma=config['training']['gamma'])
    criterion = RelativeL2Loss()

    # 5. Training Loop
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()

        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_loss += criterion(out, y).item()
                
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Rel L2: {train_loss/len(train_loader):.4f} | Test Rel L2: {test_loss/len(test_loader):.4f}")

if __name__ == "__main__":
    main()
