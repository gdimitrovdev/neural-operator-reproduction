"""Resolution-sweep evaluation of a trained neural operator.

Reproduces the core experiment of Tables 2 and 3 in Kovachki et al. (2022):
a model trained at one resolution is evaluated at several test resolutions,
and the relative L2 error should stay (approximately) constant.

Usage:
    python scripts/evaluate.py --run runs/darcy_fno2d_20260611_171715
    python scripts/evaluate.py --run runs/burgers_fno1d_... --resolutions 256 512 1024 2048 4096 8192
"""
import argparse
import csv
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.fno import FNO1d, FNO2d
from src.models.gno import GNO2d
from src.models.lno import LNO2d
from src.models.mgno import MGNO2d
from src.utils.data_utils import MatDataset, UnitGaussianNormalizer, get_grid2d
from src.utils.losses import RelativeL2Loss

MODEL_REGISTRY = {
    "FNO1d": FNO1d,
    "FNO2d": FNO2d,
    "LNO2d": LNO2d,
    "GNO2d": GNO2d,
    "MGNO2d": MGNO2d,
}
GRAPH_MODELS = {"GNO2d", "MGNO2d"}

DEFAULT_RESOLUTIONS_2D = [85, 141, 211, 421]
DEFAULT_RESOLUTIONS_1D = [256, 512, 1024, 2048, 4096, 8192]


def build_model(config):
    model_config = config["model"]
    name = model_config["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model name: {name}")
    kwargs = {k: v for k, v in model_config.items() if k != "name"}
    return MODEL_REGISTRY[name](**kwargs)


def reachable(full, res, two_d):
    if res > full:
        return False
    if two_d:
        return res > 1 and (full - 1) % (res - 1) == 0
    return full % res == 0


def subsample(t, res, two_d):
    full = t.size(1)
    if two_d:
        stride = (full - 1) // (res - 1)
        return t[:, ::stride, ::stride, :][:, :res, :res, :]
    return t[:, ::full // res, :]


def evaluate_at_resolution(model, x, y, batch_size, device, graph_model, y_normalizer):
    criterion = RelativeL2Loss(reduction='none')
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    errors = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if graph_model:
                res = int(round(xb.size(1) ** 0.5))
                coords = get_grid2d(res, batch_size=xb.size(0), device=device, flatten=True).to(dtype=xb.dtype)
                out = model(xb, coords)
            else:
                out = model(xb)
            if y_normalizer is not None:
                out = y_normalizer.decode(out)
            errors.append(criterion(out, yb))
    return torch.cat(errors).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run directory written by training (contains config.yaml and best_model.pt)")
    parser.add_argument("--checkpoint", default=None, help="checkpoint path (default: <run>/best_model.pt)")
    parser.add_argument("--mode", choices=["same", "sweep"], default="same",
                        help="'same' (default): evaluate only at the model's training resolution — this is the "
                             "Table 2/3 protocol ('train and test on the same resolution'). 'sweep': evaluate "
                             "across resolutions the model was NOT trained on (zero-shot super-resolution, paper "
                             "section 7.2.3) — only comparable to the paper's super-resolution experiments, not Tables 2/3.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=None,
                        help="override the resolutions to evaluate (defaults: training resolution for 'same', "
                             "the paper's full list for 'sweep')")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(os.path.join(args.run, "config.yaml")) as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    device = torch.device(args.device)
    batch_size = args.batch_size or config["training"]["batch_size"]
    graph_model = config["model"]["name"] in GRAPH_MODELS

    checkpoint_path = args.checkpoint or os.path.join(args.run, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test data at full file resolution; the sweep subsamples from it.
    num_test = data_config["num_test"]
    test_file = data_config.get("test_file_path")
    if test_file:
        test_dataset = MatDataset(test_file, x_key=data_config["x_key"], y_key=data_config["y_key"])
        x_test_full = test_dataset.x[:num_test]
        y_test_full = test_dataset.y[:num_test]
    else:
        dataset = MatDataset(data_config["file_path"], x_key=data_config["x_key"], y_key=data_config["y_key"])
        num_train = data_config["num_train"]
        if num_train + num_test > dataset.x.size(0):
            raise ValueError("test split would overlap the training split; set data.test_file_path")
        x_test_full = dataset.x[-num_test:]
        y_test_full = dataset.y[-num_test:]

    two_d = x_test_full.dim() == 4
    full = x_test_full.size(1)

    # Normalizer statistics are pointwise over the training set, so subsampling
    # the training data to each evaluation resolution reproduces exactly the
    # statistics training would have used at that resolution.
    x_train_full = y_train_full = None
    if data_config.get("normalize", False):
        train_dataset = MatDataset(data_config["file_path"], x_key=data_config["x_key"], y_key=data_config["y_key"])
        x_train_full = train_dataset.x[:data_config["num_train"]]
        y_train_full = train_dataset.y[:data_config["num_train"]]

    train_res = data_config.get("resolution", full // data_config.get("subsample", 1))

    if args.resolutions:
        requested = args.resolutions
    elif args.mode == "same":
        requested = [train_res]
    else:
        requested = DEFAULT_RESOLUTIONS_2D if two_d else DEFAULT_RESOLUTIONS_1D
    resolutions = []
    for res in requested:
        if reachable(full, res, two_d):
            resolutions.append(res)
        else:
            print(f"skipping resolution {res}: not reachable from {full} by uniform striding")
    if not resolutions:
        raise ValueError(f"no requested resolution is reachable from full resolution {full}")

    print(f"{config['model']['name']} | trained at resolution {train_res} | checkpoint {checkpoint_path}")
    if args.mode == "same":
        print("mode: same-resolution (Table 2/3 protocol — train and test on the same resolution)")
    else:
        print("mode: zero-shot super-resolution (trained at one resolution, tested at others; paper section 7.2.3)")
        if config["model"].get("padding", 0):
            print(f"  WARNING: model uses padding={config['model']['padding']}, which spans a different physical "
                  f"fraction of the domain at each resolution and inflates zero-shot error away from train res {train_res}.")
    if graph_model and any(r > train_res for r in resolutions):
        print("note: graph-model cost grows as O(num_nodes^2); high resolutions may be very slow")

    results = []
    for res in resolutions:
        x_res = subsample(x_test_full, res, two_d)
        y_res = subsample(y_test_full, res, two_d)
        x_train_res = subsample(x_train_full, res, two_d) if x_train_full is not None else None
        y_train_res = subsample(y_train_full, res, two_d) if y_train_full is not None else None
        if graph_model:
            # Flatten before building normalizers so their pointwise statistics
            # match the (batch, num_nodes, channels) layout, as in training.
            x_res = x_res.reshape(x_res.size(0), -1, x_res.size(-1))
            y_res = y_res.reshape(y_res.size(0), -1, y_res.size(-1))
            if x_train_res is not None:
                x_train_res = x_train_res.reshape(x_train_res.size(0), -1, x_train_res.size(-1))
                y_train_res = y_train_res.reshape(y_train_res.size(0), -1, y_train_res.size(-1))
        y_normalizer = None
        if x_train_res is not None:
            x_normalizer = UnitGaussianNormalizer(x_train_res)
            y_normalizer = UnitGaussianNormalizer(y_train_res).to(device)
            x_res = x_normalizer.encode(x_res)
        error = evaluate_at_resolution(model, x_res, y_res, batch_size, device, graph_model, y_normalizer)
        results.append((res, error))
        print(f"s = {res:5d} | relative L2 = {error:.4f}")

    out_name = "same_resolution_eval.csv" if args.mode == "same" else "zero_shot_sweep.csv"
    out_path = os.path.join(args.run, out_name)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["resolution", "test_rel_l2", "num_samples", "train_resolution", "mode"])
        for res, error in results:
            writer.writerow([res, f"{error:.6f}", x_test_full.size(0), train_res, args.mode])
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
