from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .datasets import WebLayout
from .model import LayoutModel
from .train import load_checkpoint, save_checkpoint, trainable_params


class LayoutAccuracyStats:
    def __init__(self):
        self.total_line_start_acc = 0.0
        self.total_line_end_acc = 0.0
        self.updates = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.updates += 1

        line_starts = target[:, :, 0].bool()
        line_ends = target[:, :, 1].bool()
        threshold = 0.5
        pred_line_starts = pred[:, :, 0] >= threshold
        pred_line_ends = pred[:, :, 1] >= threshold

        self.total_line_start_acc += (
            (pred_line_starts == line_starts).float().mean().item()
        )
        self.total_line_end_acc += (pred_line_ends == line_ends).float().mean().item()

    def line_start_acc(self) -> float:
        return self.total_line_start_acc / self.updates

    def line_end_acc(self) -> float:
        return self.total_line_end_acc / self.updates

    def summary(self) -> str:
        return (
            f"line start acc {self.line_start_acc()} line end acc {self.line_end_acc()}"
        )

    def stats_dict(self) -> dict:
        return {
            "line_start_acc": self.line_start_acc(),
            "line_end_acc": self.line_end_acc(),
        }


def train(
    epoch: int,
    dataloader: DataLoader,
    model: LayoutModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, LayoutAccuracyStats]:
    """
    Run one epoch of training.

    Return the mean loss and accuracy statistics.
    """

    model.train()
    train_iterable = tqdm(dataloader)
    train_iterable.set_description(f"Training (epoch {epoch})")
    total_loss = 0.0
    total_line_start_acc = 0.0
    total_line_end_acc = 0.0
    loss = nn.BCELoss()

    stats = LayoutAccuracyStats()

    for batch_idx, batch in enumerate(train_iterable):
        optimizer.zero_grad()
        input, target = batch

        pred = model(input)

        batch_loss = loss(pred, target)
        batch_loss.backward()

        stats.update(pred, target)

        optimizer.step()

        total_loss += batch_loss.item()

    mean_loss = total_loss / len(dataloader)
    return (mean_loss, stats)


def test(
    dataloader: DataLoader,
    model: LayoutModel,
) -> tuple[float, LayoutAccuracyStats]:
    """
    Run evaluation of a layout model.

    Return the mean loss and accuracy statistics.
    """
    model.eval()

    test_iterable = tqdm(dataloader)
    test_iterable.set_description("Testing")
    total_loss = 0.0
    stats = LayoutAccuracyStats()
    loss = nn.BCELoss()

    with torch.no_grad():
        for batch in test_iterable:
            input, target = batch

            pred = model(input)

            total_loss += loss(pred, target)
            stats.update(pred, target)

    test_iterable.clear()

    return (total_loss / len(dataloader)), stats


def main():
    parser = ArgumentParser(description="Train text layout model.")
    parser.add_argument("data_dir")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint to load")
    parser.add_argument("--export", type=str, help="Export model to ONNX format")
    parser.add_argument(
        "--max-epochs", type=int, help="Maximum number of epochs to train for"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Run validation only"
    )
    args = parser.parse_args()

    # Set to aid debugging
    pytorch_seed = 1234
    torch.manual_seed(pytorch_seed)

    # Maximum number of words to use from each item in the dataset. The first
    # dimension of inputs and targets are padded to this length.
    n_words = 500

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LayoutModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_dataset = WebLayout(
        args.data_dir, randomize=True, padded_size=n_words, train=True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=10, shuffle=True, pin_memory=True
    )

    val_dataset = WebLayout(
        args.data_dir, randomize=False, padded_size=n_words, train=False
    )
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=True)

    total_params = trainable_params(model)
    print(f"Model param count {total_params}")

    epoch = 0

    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer, device)
        epoch = checkpoint["epoch"]

    if args.export:
        dummy_input, dummy_target = next(iter(train_dataloader))
        torch.onnx.export(
            model,
            dummy_input[0:1],
            args.export,
            input_names=["word_boxes"],
            output_names=["preds"],
            dynamic_axes={
                "word_boxes": {0: "batch", 1: "box"},
                "preds": {0: "box"},
            },
            opset_version=16,
        )
        return

    if args.validate_only:
        val_loss, val_stats = test(val_dataloader, model)
        line_start_acc, line_end_acc = (
            val_stats.line_start_acc(),
            val_stats.line_end_acc(),
        )
        print(f"Epoch {epoch} val stats: {val_stats.summary()}")
        return

    # Enable experiment tracking via Weights and Biases if API key set.
    enable_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if enable_wandb:
        wandb.init(
            project="text-layout",
            config={
                "dataset_size": len(train_dataset),
                "model_params": total_params,
                "pytorch_seed": pytorch_seed,
            },
        )
        wandb.watch(model)

    best_val_loss = float("inf")

    while args.max_epochs is None or epoch < args.max_epochs:
        train_loss, train_stats = train(epoch, train_dataloader, model, optimizer)
        val_loss, val_stats = test(val_dataloader, model)
        line_start_acc, line_end_acc = (
            val_stats.line_start_acc(),
            val_stats.line_end_acc(),
        )

        print(f"Epoch {epoch} train loss {train_loss} val loss {val_loss}")
        print(f"Epoch {epoch} train stats: {train_stats.summary()}")
        print(f"Epoch {epoch} val stats: {val_stats.summary()}")

        if enable_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_stats.stats_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_stats.stats_dict(),
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint("text-layout-checkpoint.pt", model, optimizer, epoch=epoch)

        epoch += 1


if __name__ == "__main__":
    main()
