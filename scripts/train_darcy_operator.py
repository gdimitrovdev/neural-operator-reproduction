import argparse
import os
import sys

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.fno import FNO2d
from src.models.gno import GNO2d
from src.models.lno import LNO2d
from src.models.mgno import MGNO2d
from src.utils.data_utils import MatDataset, get_grid2d
from src.utils.losses import RelativeL2Loss


def build_model(config):
    model_config = config["model"]
    name = model_config["name"]
    if name == "FNO2d":
        return FNO2d(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            modes1=model_config["modes1"],
            modes2=model_config["modes2"],
            width=model_config["width"],
            num_layers=model_config["num_layers"],
            padding=model_config.get("padding", 0),
            append_grid=model_config.get("append_grid", True),
        )
    if name == "GNO2d":
        return GNO2d(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            width=model_config["width"],
            radius=model_config["radius"],
            num_layers=model_config["num_layers"],
            chunk_size=model_config.get("chunk_size", 64),
        )
    if name == "LNO2d":
        return LNO2d(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            width=model_config["width"],
            rank=model_config["rank"],
            num_layers=model_config["num_layers"],
            append_grid=model_config.get("append_grid", True),
        )
    if name == "MGNO2d":
        return MGNO2d(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            width=model_config["width"],
            radius_fine=model_config["radius_fine"],
            radius_coarse=model_config["radius_coarse"],
            chunk_size=model_config.get("chunk_size", 64),
        )
    raise ValueError(f"Unsupported model name: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/darcy_fno2d.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = config["data"]
    dataset = MatDataset(data_config["file_path"], x_key=data_config["x_key"], y_key=data_config["y_key"])

    num_train = data_config["num_train"]
    num_test = data_config["num_test"]
    sub = data_config["subsample"]
    x_train = dataset.x[:num_train, ::sub, ::sub, :]
    y_train = dataset.y[:num_train, ::sub, ::sub, :]
    x_test = dataset.x[-num_test:, ::sub, ::sub, :]
    y_test = dataset.y[-num_test:, ::sub, ::sub, :]

    graph_model = config["model"]["name"] in {"GNO2d", "MGNO2d"}
    if graph_model:
        x_train = x_train.view(x_train.size(0), -1, x_train.size(-1))
        y_train = y_train.view(y_train.size(0), -1, y_train.size(-1))
        x_test = x_test.view(x_test.size(0), -1, x_test.size(-1))
        y_test = y_test.view(y_test.size(0), -1, y_test.size(-1))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=config["training"]["batch_size"], shuffle=False)

    model = build_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["step_size"],
        gamma=config["training"]["gamma"],
    )
    criterion = RelativeL2Loss()

    resolution = data_config["resolution"]
    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if graph_model:
                coords = get_grid2d(resolution, batch_size=x.size(0), device=device, flatten=True).to(dtype=x.dtype)
                out = model(x, coords)
            else:
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
                if graph_model:
                    coords = get_grid2d(resolution, batch_size=x.size(0), device=device, flatten=True).to(dtype=x.dtype)
                    out = model(x, coords)
                else:
                    out = model(x)
                test_loss += criterion(out, y).item()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | Train Rel L2: {train_loss / len(train_loader):.4f} "
                f"| Test Rel L2: {test_loss / len(test_loader):.4f}"
            )


if __name__ == "__main__":
    main()
