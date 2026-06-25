import argparse
import os
import sys
import time

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.fno import FNO2d
from src.models.gno import GNO2d
from src.models.lno import LNO2d
from src.models.mgno import MGNO2d
from src.utils.data_utils import MatDataset, UnitGaussianNormalizer, get_grid2d
from src.utils.losses import RelativeL2Loss
from src.utils.training import ExperimentLogger, set_seed


def build_model(config):
    model_config = config["model"]
    name = model_config["name"]
    model_kwargs = {k: v for k, v in model_config.items() if k != "name"}
    if name == "FNO2d":
        return FNO2d(**model_kwargs)
    if name == "GNO2d":
        return GNO2d(**model_kwargs)
    if name == "LNO2d":
        return LNO2d(**model_kwargs)
    if name == "MGNO2d":
        return MGNO2d(**model_kwargs)
    raise ValueError(f"Unsupported model name: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/darcy_fno2d.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed_config = config.get("seed", {})
    set_seed(seed_config.get("value", 0), deterministic=seed_config.get("deterministic", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = config["data"]
    dataset = MatDataset(data_config["file_path"], x_key=data_config["x_key"], y_key=data_config["y_key"])

    num_train = data_config["num_train"]
    num_test = data_config["num_test"]
    sub = data_config["subsample"]
    x_train = dataset.x[:num_train, ::sub, ::sub, :]
    y_train = dataset.y[:num_train, ::sub, ::sub, :]

    test_file = data_config.get("test_file_path")
    if test_file:
        # Paper protocol: train on smooth1, test on the independent smooth2 file.
        test_dataset = MatDataset(test_file, x_key=data_config["x_key"], y_key=data_config["y_key"])
        x_test = test_dataset.x[:num_test, ::sub, ::sub, :]
        y_test = test_dataset.y[:num_test, ::sub, ::sub, :]
    else:
        total = dataset.x.size(0)
        if num_train + num_test > total:
            raise ValueError(
                f"num_train + num_test = {num_train + num_test} exceeds the {total} samples in "
                f"{data_config['file_path']}, so the test split would overlap the training split. "
                "Set data.test_file_path to a separate file or reduce the split sizes."
            )
        x_test = dataset.x[-num_test:, ::sub, ::sub, :]
        y_test = dataset.y[-num_test:, ::sub, ::sub, :]

    graph_model = config["model"]["name"] in {"GNO2d", "MGNO2d"}
    if graph_model:
        x_train = x_train.reshape(x_train.size(0), -1, x_train.size(-1))
        y_train = y_train.reshape(y_train.size(0), -1, y_train.size(-1))
        x_test = x_test.reshape(x_test.size(0), -1, x_test.size(-1))
        y_test = y_test.reshape(y_test.size(0), -1, y_test.size(-1))

    x_normalizer = y_normalizer = None
    if data_config.get("normalize", False):
        # Pointwise Gaussian normalization as in the official FNO Darcy code:
        # inputs are encoded for train and test; targets are encoded for
        # training only and predictions decoded before the loss, so reported
        # errors stay in physical units.
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=config["training"]["batch_size"], shuffle=False)

    model = build_model(config).to(device)
    lr = config["training"]["learning_rate"]
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["step_size"],
        gamma=config["training"]["gamma"],
    )
    criterion = RelativeL2Loss()
    logger = ExperimentLogger(config, config_path=args.config)

    if y_normalizer is not None:
        y_normalizer.to(device)

    resolution = data_config["resolution"]
    total_epochs = config["training"]["epochs"]
    start_time = time.time()
    for epoch in range(total_epochs):
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
            if y_normalizer is not None:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
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
                if y_normalizer is not None:
                    out = y_normalizer.decode(out)
                test_loss += criterion(out, y).item()

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        learning_rate = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch, train_loss, test_loss, learning_rate)
        logger.save_checkpoint(model, optimizer, scheduler, epoch, train_loss, test_loss)

        done = epoch + 1
        elapsed = time.time() - start_time
        per_ep = elapsed / done
        eta = per_ep * (total_epochs - done)
        print(
            f"epoch {done:>4}/{total_epochs} ({100 * done // total_epochs:3d}%) "
            f"| train {train_loss:.4f} | test {test_loss:.4f} "
            f"| {per_ep:.1f}s/ep | ETA {eta / 60:.1f}m",
            flush=True,
        )

    logger.save_summary(config["training"]["epochs"] - 1, train_loss, test_loss)
    if logger.run_dir is not None:
        print(f"Saved run artifacts to: {logger.run_dir}")


if __name__ == "__main__":
    main()
