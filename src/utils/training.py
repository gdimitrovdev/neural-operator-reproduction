import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExperimentLogger:
    def __init__(self, config, config_path=None):
        logging_config = config.get("logging", {})
        enabled = logging_config.get("enabled", True)
        self.enabled = enabled
        self.best_metric = float("inf")
        self.run_dir = None

        if not enabled:
            return

        output_dir = logging_config.get("output_dir", "runs")
        experiment_name = logging_config.get("experiment_name", config["model"]["name"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        config_copy = dict(config)
        if config_path is not None:
            config_copy["_config_path"] = config_path
        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config_copy, f, sort_keys=False)

        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_rel_l2", "test_rel_l2", "learning_rate"])
            writer.writeheader()

    def log_epoch(self, epoch, train_loss, test_loss, learning_rate):
        if not self.enabled:
            return

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_rel_l2", "test_rel_l2", "learning_rate"])
            writer.writerow({
                "epoch": epoch,
                "train_rel_l2": train_loss,
                "test_rel_l2": test_loss,
                "learning_rate": learning_rate,
            })

    def save_checkpoint(self, model, optimizer, scheduler, epoch, train_loss, test_loss):
        if not self.enabled:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_rel_l2": train_loss,
            "test_rel_l2": test_loss,
        }
        torch.save(checkpoint, os.path.join(self.run_dir, "last_model.pt"))

        if test_loss < self.best_metric:
            self.best_metric = test_loss
            torch.save(checkpoint, os.path.join(self.run_dir, "best_model.pt"))

    def save_summary(self, final_epoch, train_loss, test_loss):
        if not self.enabled:
            return

        summary = {
            "final_epoch": final_epoch,
            "final_train_rel_l2": train_loss,
            "final_test_rel_l2": test_loss,
            "best_test_rel_l2": self.best_metric,
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
