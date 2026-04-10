"""Google Drive persistence for Colab training.

Saves checkpoints, training logs, and final models to Drive so
intermediate results survive across Colab sessions.
"""

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

DRIVE_BASE = Path("/content/drive/MyDrive/kanue")


class DriveCheckpointer:
    """Manages model checkpoints and training logs on Google Drive.

    Directory structure on Drive:
        kanue/
        ├── checkpoints/
        │   ├── baseline_epoch_10.pt
        │   ├── kan_epoch_10.pt
        │   └── ...
        ├── logs/
        │   ├── baseline_train.json
        │   ├── kan_train.json
        │   └── ...
        ├── data/
        │   └── (preprocessed .npz files)
        └── results/
            └── (analysis outputs)
    """

    def __init__(self, experiment_name: str, drive_base: Path = DRIVE_BASE):
        self.experiment_name = experiment_name
        self.base = drive_base
        self.ckpt_dir = self.base / "checkpoints"
        self.log_dir = self.base / "logs"
        self.data_dir = self.base / "data"
        self.results_dir = self.base / "results"

        for d in [self.ckpt_dir, self.log_dir, self.data_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        extra: dict | None = None,
    ) -> Path:
        """Save model checkpoint to Drive."""
        path = self.ckpt_dir / f"{self.experiment_name}_epoch_{epoch}.pt"
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def load_checkpoint(
        self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
    ) -> dict:
        """Load a checkpoint from Drive."""
        path = self.ckpt_dir / f"{self.experiment_name}_epoch_{epoch}.pt"
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def latest_checkpoint(self) -> int | None:
        """Find the latest epoch checkpoint available."""
        pattern = f"{self.experiment_name}_epoch_*.pt"
        files = list(self.ckpt_dir.glob(pattern))
        if not files:
            return None
        epochs = []
        for f in files:
            try:
                epoch = int(f.stem.split("_epoch_")[1])
                epochs.append(epoch)
            except (ValueError, IndexError):
                continue
        return max(epochs) if epochs else None

    def save_log(self, log: dict) -> Path:
        """Save training log (loss curves, metrics) to Drive as JSON."""
        path = self.log_dir / f"{self.experiment_name}_train.json"
        with path.open("w") as f:
            json.dump(log, f, indent=2)
        return path

    def load_log(self) -> dict | None:
        """Load training log from Drive."""
        path = self.log_dir / f"{self.experiment_name}_train.json"
        if not path.exists():
            return None
        with path.open() as f:
            return json.load(f)
