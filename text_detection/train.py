from argparse import ArgumentParser
import os
import shutil
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize, to_pil_image
import torchvision.transforms as transforms
from tqdm import tqdm

from .datasets import DDI100, HierText
from .model import DetectionModel

mask_height = 800
mask_width = int(mask_height * 0.75)
mask_size = (mask_height, mask_width)
"""
Size of the input image and output targets that the model is trained with.

This was originally chosen as ~1/10th of the size of images in the DDI-100
dataset, which consists of scanned A4 pages.
"""


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


LossFunc = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


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

    for batch_idx, (img_fname, img, masks, border_masks) in enumerate(train_iterable):
        img = img.to(device)
        masks = masks.to(device)
        border_masks = border_masks.to(device)
        start = time.time()

        pred_masks = model(img)

        loss = loss_fn(pred_masks, masks, border_masks)
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
                [masks[0], border_masks[0]],
            )

        train_iterable.set_postfix({"loss": loss.item(), "sec/img": time_per_img})

    train_iterable.clear()

    train_loss /= len(dataloader)
    return train_loss


def test(
    device: torch.device,
    dataloader: DataLoader,
    model: DetectionModel,
    loss_fn: LossFunc,
    save_debug_images=False,
):
    model.eval()

    test_loss = 0.0
    n_batches = len(dataloader)

    test_iterable = tqdm(dataloader)
    test_iterable.set_description(f"Testing")

    with torch.inference_mode():
        for img_fname, img, masks, border_masks in test_iterable:
            img = img.to(device)
            masks = masks.to(device)
            border_masks = border_masks.to(device)

            pred_masks = model(img)

            test_loss += loss_fn(pred_masks, masks, border_masks).item()

            if save_debug_images:
                save_img_and_predicted_mask(
                    "test-sample", img_fname[0], img[0], pred_masks[0], masks[0]
                )

    test_iterable.clear()

    test_loss /= n_batches
    return test_loss


def save_checkpoint(filename: str, model, optimizer, epoch: int):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        filename,
    )


def load_checkpoint(filename: str, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


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
    transform = transforms.Compose([augmentations, transforms.Resize(mask_size)])

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

    def loss_fn(pred, target, border_mask):
        weights = torch.full(pred.shape, fill_value=0.5, device=device) + border_mask
        return F.binary_cross_entropy(pred, target, weights)

    optimizer = torch.optim.Adam(model.parameters())

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model param count {total_params}")

    epochs_without_improvement = 0
    min_train_loss = 1.0
    min_test_loss = 1.0
    epoch = 0

    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer)
        epoch = checkpoint["epoch"]

    while True:
        train_loss = train(
            epoch,
            device,
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            save_debug_images=args.debug_images,
        )
        val_loss = test(
            device, val_dataloader, model, loss_fn, save_debug_images=args.debug_images
        )
        print(
            f"Epoch {epoch} train loss {train_loss:.4f} validation loss {val_loss:.4f}"
        )

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
