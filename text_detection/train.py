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
from torchvision.transforms.functional import resize, to_pil_image
import torchvision.transforms as transforms
from tqdm import tqdm

from .datasets import DDI100, HierText
from .model import DetectionModel
from .postprocess import box_match_metrics, extract_cc_quads

mask_height = 800
# mask_height = 200
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
    pred_masks: torch.Tensor,
    target_masks: Optional[torch.Tensor] = None,
):
    # Datasets yield images with values in [-0.5, 0.5]. Convert these to [0, 1]
    # as required by `to_pil_image`.
    img = img + 0.5

    shutil.copyfile(img_filename, f"{basename}_input.png")

    pil_img = to_pil_image(img)
    pil_img.save(f"{basename}_input_scaled.png")

    n_masks = pred_masks.shape[0]
    for i in range(n_masks):
        pil_pred_mask = to_pil_image(pred_masks[i])
        pil_pred_mask.save(f"{basename}_pred_{i}.png")

    if target_masks is not None:
        n_target_masks = target_masks.shape[0]
        for i in range(n_target_masks):
            pil_target_mask = to_pil_image(target_masks[i])
            pil_target_mask.save(f"{basename}_target_{i}.png")


def text_mask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    border_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute loss for text detection masks.

    `pred` and `target` may have one or two channels. The first channel is
    the text probability mask. The second, if present, is the approximate
    binary mask.

    :param pred: NCHW tensor of predicted masks
    :param target: NCHW tensor of target masks
    :param border_mask: N1HW tensor of border regions of text instances, used to adjust loss weights
    """

    # Create weight map, assigning higher weight to losses around borders of
    # text instances.
    batch_size, _, height, width = pred.shape
    weights = (
        torch.full((batch_size, height, width), fill_value=0.5, device=pred.device)
        + border_mask[:, 0]
    )

    # Compute loss for text probability mask.
    pred_prob = pred[:, 0]
    target_prob = target[:, 0]
    prob_loss = F.binary_cross_entropy(pred_prob, target_prob, weights)

    pred_channels = pred.shape[1]
    if pred_channels == 1:
        return prob_loss

    # Compute loss for approximate binary mask.
    pred_bin = pred[:, 1]
    target_bin = target[:, 1]
    bin_loss = F.binary_cross_entropy(pred_bin, target_bin, weights)

    # Compute mean losses per pixel in [0, 1].
    return torch.stack([prob_loss, bin_loss]).mean()


def train(
    epoch: int,
    device: torch.device,
    dataloader: DataLoader,
    model: DetectionModel,
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
        border_masks = batch["border_mask"]

        img = img.to(device)
        masks = masks.to(device)
        border_masks = border_masks.to(device)
        start = time.time()

        # Duplicate binary mask along channel axis so we can use it as a target
        # for both the probability and binary masks.
        target_masks = torch.cat([masks, masks], dim=1)

        pred_masks = model(img)

        # Extract the first two masks (probs, binary class) from the
        # three masks returned (probs, bin class, threshold map).
        prob_bin_masks = pred_masks[:, 0:2]

        loss = text_mask_loss(prob_bin_masks, target_masks, border_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        time_per_img = (time.time() - start) / img.shape[0]

        if save_debug_images:
            # TODO - Visualize all of the masks
            save_img_and_predicted_mask(
                "train-sample", img_fname[0], img[0], pred_masks[0], masks[0]
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
    test_iterable.set_description(f"Testing")

    metrics = []
    with torch.inference_mode():
        for batch in test_iterable:
            img_fname = batch["path"]
            img = batch["image"]
            masks = batch["text_mask"]
            border_masks = batch["border_mask"]

            img = img.to(device)
            masks = masks.to(device)
            border_masks = border_masks.to(device)

            pred_masks = model(img)

            test_loss += text_mask_loss(pred_masks, masks, border_masks).item()

            for item_index, pred_mask in enumerate(pred_masks):
                pred_quads = extract_cc_quads(binarize_mask(pred_mask).cpu())
                target_quads = extract_cc_quads(binarize_mask(masks[item_index]).cpu())
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


def prepare_transform(mask_size: tuple[int, int], augment) -> nn.Module:
    """
    Prepare image transforms to be applied to input images and text masks.

    :param mask_size: HxW output image size
    :param augment: Whether to apply randomized data augmentation
    """
    resize_transform = transforms.Resize(mask_size)
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
    parser = ArgumentParser()
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

    if max_images:
        validation_max_images = max(10, int(max_images * 0.1))
    else:
        validation_max_images = None

    transform = prepare_transform(mask_size, augment=args.augment)

    train_dataset = load_dataset(
        args.data_dir, transform=transform, train=True, max_images=max_images
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_dataset = load_dataset(
        args.data_dir,
        transform=transform,
        train=False,
        max_images=validation_max_images,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Train images {len(train_dataset)} in {len(train_dataloader)} batches")
    print(f"Validation images {len(val_dataset)} in {len(val_dataloader)} batches")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DetectionModel().to(device)

    optimizer = torch.optim.Adam(model.parameters())

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model param count {total_params}")

    epochs_without_improvement = 0
    min_train_loss = 1.0
    min_test_loss = 1.0
    epoch = 0

    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer, device)
        epoch = checkpoint["epoch"]

    if args.validate_only:
        if not args.checkpoint:
            parser.exit(
                1,
                f"Existing model should be specified with --checkpoint when using --validate-only",
            )

        val_loss, val_metrics = test(
            device, val_dataloader, model, save_debug_images=args.debug_images
        )
        print(f"Validation loss {val_loss:.4f}")
        print("Validation metrics:", format_metrics(val_metrics))
        return

    while True:
        train_loss = train(
            epoch,
            device,
            train_dataloader,
            model,
            optimizer,
            save_debug_images=args.debug_images,
        )
        val_loss, val_metrics = test(
            device,
            val_dataloader,
            model,
            save_debug_images=args.debug_images,
        )
        print(
            f"Epoch {epoch} train loss {train_loss:.4f} validation loss {val_loss:.4f}"
        )
        print(f"Epoch {epoch} validation metrics:", format_metrics(val_metrics))

        if train_loss < min_train_loss:
            min_loss = train_loss
            epochs_without_improvement = 0
            save_checkpoint("checkpoint.pt", model, optimizer, epoch=epoch)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > 3:
            print(
                f"Stopping after {epochs_without_improvement} epochs without train loss improvement"
            )

        epoch += 1


if __name__ == "__main__":
    main()
