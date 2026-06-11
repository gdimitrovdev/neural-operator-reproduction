import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.fno import FNO1d
from src.utils.losses import RelativeL2Loss
from src.utils.data_utils import MatDataset

def main():
    with open("configs/burgers_fno1d.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    full_dataset = MatDataset(config['data']['file_path'], x_key=config['data']['x_key'], y_key=config['data']['y_key'])
    
    sub = config['data']['subsample']
    # Burgers data is (N, 8192). Subsample to (N, 1024)
    num_train = config['data']['num_train']
    num_test = config['data']['num_test']
    x_train = full_dataset.x[:num_train, ::sub, :]
    y_train = full_dataset.y[:num_train, ::sub, :]
    x_test = full_dataset.x[-num_test:, ::sub, :]
    y_test = full_dataset.y[-num_test:, ::sub, :]

    train_loader = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=config['training']['batch_size'], shuffle=False)

    # 2. Init Model
    model = FNO1d(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        modes=config['model']['modes'],
        width=config['model']['width'],
        num_layers=config['model']['num_layers'],
        append_grid=config['model'].get('append_grid', True)
    ).to(device)

    # 3. Optimizers
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['step_size'], gamma=config['training']['gamma'])
    criterion = RelativeL2Loss()

    # 4. Training Loop
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

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_loss += criterion(out, y).item()
                
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train L2: {train_loss/len(train_loader):.4f} | Test L2: {test_loss/len(test_loader):.4f}")

if __name__ == "__main__":
    main()
