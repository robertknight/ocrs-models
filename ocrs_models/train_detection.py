from argparse import ArgumentParser, BooleanOptionalAction
import os
import shutil
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb

from .datasets.ddi100 import DDI100
from .datasets.hiertext import HierText
from .models import DetectionModel
from .postprocess import box_match_metrics, extract_cc_quads

mask_height = 800
mask_width = int(mask_height * 0.75)
mask_size = (mask_height, mask_width)
"""
Size of the input image and output targets that the model is trained with.

This was originally chosen as ~1/10th of the size of images in the DDI-100
dataset, which consists of scanned A4 pages.
"""


def binarize_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return torch.where(mask > threshold, 1.0, 0.0)


def save_img_and_predicted_mask(
    basename: str,
    img_filename: str,
    img: torch.Tensor,
    pred_masks: list[torch.Tensor],
    target_masks: Optional[list[torch.Tensor]] = None,
):
    # Datasets yield images with values in [-0.5, 0.5]. Convert these to [0, 1]
    # as required by `to_pil_image`.
    img = img + 0.5

    shutil.copyfile(img_filename, f"{basename}_input.png")

    pil_img = to_pil_image(img)
    pil_img.save(f"{basename}_input_scaled.png")

    for i, pred_mask in enumerate(pred_masks):
        pil_pred_mask = to_pil_image(pred_mask)
        pil_pred_mask.save(f"{basename}_pred_mask_{i}.png")

    if target_masks is not None:
        for i, target_mask in enumerate(target_masks):
            pil_target_mask = to_pil_image(target_mask)
            pil_target_mask.save(f"{basename}_mask_{i}.png")


LossFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def train(
    epoch: int,
    device: torch.device,
    dataloader: DataLoader,
    model: DetectionModel,
    loss_fn: LossFunc,
    optimizer: torch.optim.Optimizer,
    save_debug_images=False,
):
    model.train()

    train_iterable = tqdm(dataloader)
    train_iterable.set_description(f"Training (epoch {epoch})")

    train_loss = 0.0

    for batch_idx, batch in enumerate(train_iterable):
        img_fname = batch["path"]
        img = batch["image"]
        masks = batch["text_mask"]

        img = img.to(device)
        masks = masks.to(device)

        start = time.time()

        pred_masks = model(img)

        loss = loss_fn(pred_masks, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        time_per_img = (time.time() - start) / img.shape[0]

        if save_debug_images:
            save_img_and_predicted_mask(
                "train-sample",
                img_fname[0],
                img[0],
                pred_masks[0],
                [masks[0]],
            )

        train_iterable.set_postfix({"loss": loss.item(), "sec/img": time_per_img})

    train_iterable.clear()

    train_loss /= len(dataloader)
    return train_loss


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def get_metric_means(metrics_dicts: list[dict[str, float]]) -> dict[str, float]:
    """
    Compute means of all metrics in a list of metrics dicts.

    All metrics dicts should have the same keys. Any keys that are missing
    from a dict are treated as being present but with a value of zero.

    :param metrics_dicts: Dicts mapping metric names to values
    :return: Dict mapping metric names to values
    """
    if not len(metrics_dicts):
        return {}

    keys = set(k for md in metrics_dicts for k in md.keys())
    return {k: mean([md.get(k, 0.0) for md in metrics_dicts]) for k in keys}


def format_metrics(metrics: dict[str, float]) -> dict[str, str]:
    return {k: f"{v:.3f}" for k, v in metrics.items()}


def test(
    device: torch.device,
    dataloader: DataLoader,
    model: DetectionModel,
    loss_fn: LossFunc,
    save_debug_images=False,
) -> tuple[float, dict[str, float]]:
    """
    Run evaluation on a model.

    Returns a tuple of (mean pixel-level loss, word-level metrics).
    """
    model.eval()

    test_loss = 0.0
    n_batches = len(dataloader)

    test_iterable = tqdm(dataloader)
    test_iterable.set_description("Testing")

    metrics = []
    with torch.inference_mode():
        for batch in test_iterable:
            img_fname = batch["path"]
            img = batch["image"]
            masks = batch["text_mask"]

            img = img.to(device)
            masks = masks.to(device)

            pred_masks = model(img)

            test_loss += loss_fn(pred_masks, masks).item()

            bin_pred_masks = binarize_mask(pred_masks).cpu()
            bin_masks = binarize_mask(masks).cpu()

            for item_index, bin_pred_mask in enumerate(bin_pred_masks):
                pred_quads = extract_cc_quads(bin_pred_mask)
                target_quads = extract_cc_quads(bin_masks[item_index])
                metrics.append(box_match_metrics(pred_quads, target_quads))

            if save_debug_images:
                save_img_and_predicted_mask(
                    "test-sample", img_fname[0], img[0], pred_masks[0], masks[0]
                )
    test_iterable.clear()

    mean_metrics = get_metric_means(metrics)

    test_loss /= n_batches
    return test_loss, mean_metrics


def save_checkpoint(filename: str, model: nn.Module, optimizer: Optimizer, epoch: int):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        filename,
    )


def load_checkpoint(
    filename: str, model: nn.Module, optimizer: Optimizer, device: torch.device
):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def trainable_params(model: nn.Module) -> int:
    """
    Return the number of trainable parameters (weights, biases etc.) in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def balanced_cross_entropy_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Compute balanced binary cross-entropy loss between two images.

    This loss accounts for the targets being unbalanced between text and
    non-text pixels by computing per-pixel losses and then taking the mean
    of losses of an equal number of text and non-text pixels.

    :param pred: NCHW tensor of probabilities
    :param target: NCHW tensor of targets
    """

    pos_mask = target > 0.5
    neg_mask = target < 0.5

    # Clamp target values to ensure they are valid for use with BCE loss.
    #
    # The PyTorch transforms used for data augmentation can sometimes result in
    # values slightly outside the [0, 1] range.
    #
    # We assume the predictions have been generated via a sigmoid or similar
    # that guarantees values in [0, 1].
    target = target.clamp(0.0, 1.0)

    pixel_loss = F.binary_cross_entropy(pred, target, reduction="none")

    pos_loss = pos_mask * pixel_loss
    neg_loss = neg_mask * pixel_loss

    pos_elements = torch.count_nonzero(pos_mask).item()
    neg_elements = torch.count_nonzero(neg_mask).item()
    n_els = int(min(pos_elements, neg_elements))

    pos_topk_vals, _ = pos_loss.flatten().topk(k=n_els, sorted=False)
    neg_topk_vals, _ = neg_loss.flatten().topk(k=n_els, sorted=False)

    return torch.cat([pos_topk_vals, neg_topk_vals]).mean()


def prepare_transform(mask_size: tuple[int, int], augment) -> nn.Module:
    """
    Prepare image transforms to be applied to input images and text masks.

    :param mask_size: HxW output image size
    :param augment: Whether to apply randomized data augmentation
    """
    resize_transform = transforms.Resize(mask_size, antialias=False)
    if not augment:
        return resize_transform

    augmentations = transforms.RandomApply(
        [
            transforms.RandomChoice(
                [
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.RandomAffine(degrees=5, scale=(0.8, 1.2), shear=5),
                    transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
                    transforms.RandomCrop(size=600, pad_if_needed=True),
                ]
            )
        ],
        p=0.5,
    )
    return transforms.Compose([augmentations, resize_transform])


def main():
    parser = ArgumentParser(description="Train text detection model.")
    parser.add_argument(
        "dataset_type", type=str, choices=["ddi", "hiertext"], help="Format of dataset"
    )
    parser.add_argument("data_dir")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint to load")
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Save debugging images during training",
    )
    parser.add_argument("--export", type=str, help="Export model to ONNX format")
    parser.add_argument(
        "--max-epochs", type=int, help="Maximum number of epochs to train for"
    )
    parser.add_argument(
        "--max-images", type=int, help="Maximum number of images to load"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation on an existing model",
    )
    parser.add_argument(
        "--augment",
        default=True,
        action=BooleanOptionalAction,
        help="Enable data augmentation",
    )
    args = parser.parse_args()

    if args.dataset_type == "ddi":
        load_dataset = DDI100
    elif args.dataset_type == "hiertext":
        load_dataset = HierText
    else:
        raise Exception(f"Unknown dataset type {args.dataset_type}")

    batch_size = args.batch_size
    max_images = args.max_images

    # Set to aid debugging of initial text detection model
    pytorch_seed = 1234
    torch.manual_seed(pytorch_seed)

    if max_images:
        validation_max_images = max(10, int(max_images * 0.1))
    else:
        validation_max_images = None

    transform = prepare_transform(mask_size, augment=args.augment)

    train_dataset = load_dataset(
        args.data_dir, transform=transform, train=True, max_images=max_images
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    val_dataset = load_dataset(
        args.data_dir,
        transform=transform,
        train=False,
        max_images=validation_max_images,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, pin_memory=True, num_workers=2
    )

    print(
        f"Training dataset: images {len(train_dataset)} in {len(train_dataloader)} batches"
    )
    print(
        f"Validation dataset: images {len(val_dataset)} in {len(val_dataloader)} batches"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DetectionModel().to(device)

    optimizer = torch.optim.Adam(model.parameters())

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model param count: {total_params}")

    epochs_without_improvement = 0
    min_train_loss = 1.0
    epoch = 0

    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer, device)
        epoch = checkpoint["epoch"]

    if args.export:
        if not args.checkpoint:
            raise Exception("ONNX export requires a checkpoint to load")

        test_batch = next(iter(val_dataloader))
        test_image = test_batch["image"][0:1].to(device)

        torch.onnx.export(
            model,
            test_image,
            args.export,
            input_names=["image"],
            output_names=["mask"],
            dynamic_axes={"image": {0: "batch"}, "mask": {0: "batch"}},
        )
        return

    if args.validate_only:
        if not args.checkpoint:
            parser.exit(
                1,
                "Existing model should be specified with --checkpoint when using --validate-only",
            )

        val_loss, val_metrics = test(
            device,
            val_dataloader,
            model,
            balanced_cross_entropy_loss,
            save_debug_images=args.debug_images,
        )
        print(f"Validation loss {val_loss:.4f}")
        print("Validation metrics:", format_metrics(val_metrics))
        return

    # Enable experiment tracking via Weights and Biases if API key set.
    enable_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if enable_wandb:
        wandb.init(
            project="text-detection",
            config={
                "batch_size": args.batch_size,
                "dataset_size": len(train_dataset),
                "model_params": total_params,
                "pytorch_seed": pytorch_seed,
            },
        )
        wandb.watch(model)

    while args.max_epochs is None or epoch < args.max_epochs:
        train_loss = train(
            epoch,
            device,
            train_dataloader,
            model,
            balanced_cross_entropy_loss,
            optimizer,
            save_debug_images=args.debug_images,
        )
        val_loss, val_metrics = test(
            device,
            val_dataloader,
            model,
            balanced_cross_entropy_loss,
            save_debug_images=args.debug_images,
        )
        print(
            f"Epoch {epoch} train loss {train_loss:.4f} validation loss {val_loss:.4f}"
        )
        print(f"Epoch {epoch} validation metrics:", format_metrics(val_metrics))

        if enable_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                }
            )

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            epochs_without_improvement = 0
            save_checkpoint(
                "text-detection-checkpoint.pt", model, optimizer, epoch=epoch
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > 3:
            print(
                f"Stopping after {epochs_without_improvement} epochs without train loss improvement"
            )

        epoch += 1


if __name__ == "__main__":
    main()
