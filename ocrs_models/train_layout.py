from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .datasets.web_layout import WebLayout
from .models import LayoutModel
from .train_detection import load_checkpoint, save_checkpoint, trainable_params


def f1_score(precision: float, recall: float) -> float:
    """
    Return the F1 mean of precision and recall.

    See https://en.wikipedia.org/wiki/F-score.
    """
    return 2 * (precision * recall) / (precision + recall)


def precision_recall(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """
    Compute the precision and recall for a set of binary classifications.

    :param preds: Boolean tensor of predicted classifications
    :param targets: Boolean tensor of target classifications
    :return: (precision, recall) tuple
    """
    true_results = torch.logical_and(preds, targets).sum()
    precision = true_results / preds.sum()
    recall = true_results / targets.sum()
    return (precision.item(), recall.item())


class LayoutAccuracyStats:
    def __init__(self):
        self.total_line_start_precision = 0.0
        self.total_line_start_recall = 0.0
        self.total_line_end_precision = 0.0
        self.total_line_end_recall = 0.0
        self.updates = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.updates += 1

        line_starts = target[:, :, 0].bool()
        line_ends = target[:, :, 1].bool()
        threshold = 0.5
        pred_line_starts = pred[:, :, 0] >= threshold
        pred_line_ends = pred[:, :, 1] >= threshold

        line_start_prec, line_start_recall = precision_recall(
            pred_line_starts, line_starts
        )
        line_end_prec, line_end_recall = precision_recall(pred_line_ends, line_ends)

        self.total_line_start_precision += line_start_prec
        self.total_line_start_recall += line_start_recall
        self.total_line_end_precision += line_end_prec
        self.total_line_end_recall += line_end_recall

    def line_start_precision_recall(self) -> tuple[float, float]:
        return (
            self.total_line_start_precision / self.updates,
            self.total_line_start_recall / self.updates,
        )

    def line_end_precision_recall(self) -> tuple[float, float]:
        return (
            self.total_line_end_precision / self.updates,
            self.total_line_end_recall / self.updates,
        )

    def summary(self) -> str:
        line_start_prec, line_start_rec = self.line_start_precision_recall()
        line_end_prec, line_end_rec = self.line_end_precision_recall()

        return f"line start prec/recall {line_start_prec:.3f}/{line_start_rec:.3f} line end prec/recall {line_end_prec:.3f}/{line_end_rec:.3f}"

    def stats_dict(self) -> dict:
        line_start_prec, line_start_rec = self.line_start_precision_recall()
        line_end_prec, line_end_rec = self.line_end_precision_recall()
        return {
            "line_start_precision": line_start_prec,
            "line_start_recall": line_start_rec,
            "line_end_precision": line_end_prec,
            "line_end_recall": line_end_rec,
        }


def weighted_loss():
    # Weights for +ve classes are based on an estimate of 7-9% of words being
    # a +ve for each class.
    return nn.BCEWithLogitsLoss(pos_weight=torch.Tensor((10.0, 10.0)))


def train(
    epoch: int,
    device: torch.device,
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

    loss = weighted_loss().to(device)

    stats = LayoutAccuracyStats()

    for batch_idx, batch in enumerate(train_iterable):
        optimizer.zero_grad()
        input, target = [x.to(device) for x in batch]

        pred = model(input)

        batch_loss = loss(pred, target)
        batch_loss.backward()

        pred = torch.clamp(pred.sigmoid(), 0.0, 1.0)
        stats.update(pred, target)

        optimizer.step()

        total_loss += batch_loss.item()

    mean_loss = total_loss / len(dataloader)
    return (mean_loss, stats)


def test(
    device: torch.device,
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
    loss = weighted_loss().to(device)

    with torch.no_grad():
        for batch in test_iterable:
            input, target = [x.to(device) for x in batch]

            pred = model(input).sigmoid()

            total_loss += loss(pred, target)
            stats.update(pred, target)

    test_iterable.clear()

    return (total_loss / len(dataloader)), stats


def lr_scale_for_epoch(epoch: int) -> float:
    """
    Return scale factor applied to initial learning rate for a given epoch.
    """
    warmup_epochs = 50

    if warmup_epochs > 0:
        return min((epoch + 1) / (warmup_epochs + 1), 1)
    else:
        return 1


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

    # Whether to normalize bounding box coordinates in the dataset loader.
    normalize_coords = False

    # Whether to apply small augmentations (eg. small random translations)
    # to bounding box coordinates.
    use_data_augmentation = True
    max_jitter = 10  # Max random translation

    # Bounding box coordinate encoding to use
    pos_embedding = "sin"

    batch_size = 64

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LayoutModel(return_probs=False, pos_embedding=pos_embedding).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale_for_epoch)
    train_dataset = WebLayout(
        args.data_dir,
        max_jitter=max_jitter,
        normalize_coords=normalize_coords,
        randomize=use_data_augmentation,
        padded_size=n_words,
        train=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    val_dataset = WebLayout(
        args.data_dir,
        normalize_coords=normalize_coords,
        randomize=False,
        padded_size=n_words,
        train=False,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

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
        val_loss, val_stats = test(device, val_dataloader, model)
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
        train_loss, train_stats = train(
            epoch, device, train_dataloader, model, optimizer
        )
        val_loss, val_stats = test(device, val_dataloader, model)
        lr = optimizer.state_dict()["param_groups"][0]["lr"]

        print(f"Epoch {epoch} train loss {train_loss} val loss {val_loss}")
        print(f"Epoch {epoch} train stats: {train_stats.summary()}")
        print(f"Epoch {epoch} val stats: {val_stats.summary()}")
        print(f"Epoch {epoch} lr {lr}")

        if enable_wandb:
            wandb.log(
                {
                    "lr": lr,
                    "train_loss": train_loss,
                    "train_accuracy": train_stats.stats_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_stats.stats_dict(),
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint("text-layout-checkpoint.pt", model, optimizer, epoch=epoch)

        scheduler.step()
        epoch += 1


if __name__ == "__main__":
    main()
