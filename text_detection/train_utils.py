import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


class TrainLoop:
    epoch: int
    """Count of completed training epochs."""

    def __init__(self, max_epochs: Optional[int] = None, convergence_patience=5):
        """
        :param max_epochs: Max epochs to train for, regardless of whether convergence is reached.
        :param convergence_patience:
            Number of epochs that must complete without improvement before
            training is deemed complete.
        """
        self.convergence_patience = 5
        self.epoch = 0
        self.max_epochs = max_epochs

        self._epochs_without_improvement = 0
        self._min_loss = math.inf

    def done(self) -> bool:
        """
        Return True if training has completed.
        """
        if self.max_epochs is not None and self.epoch >= self.max_epochs:
            return True
        return self._epochs_without_improvement >= self.convergence_patience

    def step(self, val_loss: float):
        """
        Report completion of the current epoch.
        """
        if val_loss < self._min_loss:
            self._min_loss = val_loss
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        self.epoch += 1


def format_metrics(metrics: dict[str, float]) -> dict[str, str]:
    return {k: f"{v:.3f}" for k, v in metrics.items()}


def load_checkpoint(
    filename: str, model: nn.Module, optimizer: Optimizer, device: torch.device
):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def save_checkpoint(filename: str, model: nn.Module, optimizer: Optimizer, epoch: int):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        filename,
    )
