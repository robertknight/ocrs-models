from argparse import ArgumentParser
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize, to_pil_image
from torchvision.transforms import Resize
from tqdm import tqdm

from .datasets import DDI100, HierText
from .model import DetectionModel

# This is approx 1/5th of the DDI-100 input image size.
# mask_size = (771, 545)

# This is approx 1/10th of the DDI-100 input image size.
mask_size = (385, 272)


def save_img_and_predicted_mask(basename: str, img, pred_mask, target_mask=None):
    # DDI100 dataset returns images with values in [-0.5, 0.5]. Adjust these
    # to be in the range [0, 255].
    img = img + 0.5

    pil_img = to_pil_image(img)
    pil_img.save(f"{basename}_input.png")

    pil_pred_mask = to_pil_image(pred_mask)
    pil_pred_mask.save(f"{basename}_pred_mask.png")

    if target_mask is not None:
        pil_target_mask = to_pil_image(target_mask)
        pil_target_mask.save(f"{basename}_mask.png")


def train(
    epoch: int,
    device: torch.device,
    dataloader: DataLoader,
    model: DetectionModel,
    loss_fn,
    optimizer,
):
    model.train()

    train_iterable = tqdm(dataloader)
    train_iterable.set_description(f"Epoch {epoch}")

    train_loss = 0.0

    for batch_idx, (img_fname, img, mask) in enumerate(train_iterable):
        img = img.to(device)
        mask = mask.to(device)
        start = time.time()

        pred_mask = model(img)

        loss = loss_fn(pred_mask, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_per_img = (time.time() - start) / img.shape[0]
        train_loss += loss
        save_img_and_predicted_mask("train-sample", img[0], pred_mask[0], mask[0])
        train_iterable.set_postfix({"loss": loss.item(), "sec/img": time_per_img})

    train_iterable.clear()

    train_loss /= len(dataloader)
    return train_loss


def test(device: torch.device, dataloader: DataLoader, model: DetectionModel, loss_fn):
    model.eval()

    test_loss = 0.0
    n_batches = len(dataloader)

    with torch.inference_mode():
        for img_fname, img, mask in dataloader:
            img = img.to(device)
            mask = mask.to(device)

            pred_mask = model(img)

            test_loss += loss_fn(pred_mask, mask).item()
            save_img_and_predicted_mask("test-sample", img[0], pred_mask[0], mask[0])

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

    transform = nn.Sequential(Resize(mask_size))

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
    loss_fn = torch.nn.BCELoss()
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
        train_loss = train(epoch, device, train_dataloader, model, loss_fn, optimizer)
        val_loss = test(device, val_dataloader, model, loss_fn)
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
