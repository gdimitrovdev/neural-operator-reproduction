import argparse
import os
import sys

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.fno import FNO1d
from src.utils.losses import RelativeL2Loss
from src.utils.data_utils import MatDataset
from src.utils.training import ExperimentLogger, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/burgers_fno1d.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed_config = config.get("seed", {})
    set_seed(seed_config.get("value", 0), deterministic=seed_config.get("deterministic", False))
    
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
    model_config = config['model']
    model_kwargs = {k: v for k, v in model_config.items() if k != 'name'}
    model = FNO1d(**model_kwargs).to(device)

    # 3. Optimizers
    lr = config['training']['learning_rate']
    weight_decay = float(config['training'].get('weight_decay', 1e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['step_size'], gamma=config['training']['gamma'])
    criterion = RelativeL2Loss()
    logger = ExperimentLogger(config, config_path=args.config)

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

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        learning_rate = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch, train_loss, test_loss, learning_rate)
        logger.save_checkpoint(model, optimizer, scheduler, epoch, train_loss, test_loss)
                
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train L2: {train_loss:.4f} | Test L2: {test_loss:.4f}")

    logger.save_summary(config['training']['epochs'] - 1, train_loss, test_loss)
    if logger.run_dir is not None:
        print(f"Saved run artifacts to: {logger.run_dir}")

if __name__ == "__main__":
    main()
