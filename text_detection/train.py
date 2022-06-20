from argparse import ArgumentParser
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize, to_pil_image

from .datasets import DDI100
from .model import DetectionModel

# This is approx 1/5th of the DDI-100 input image size.
# mask_size = (771, 545)

# This is approx 1/10th of the DDI-100 input image size.
mask_size = (385, 272)


def save_img_and_predicted_mask(basename, img, pred_mask, target_mask=None):
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


def train(epoch, dataloader, model, loss_fn, optimizer):
    train_loss = 0
    for batch_idx, (img_fname, img, mask) in enumerate(dataloader):
        start = time.time()
        img = resize(img, mask_size)
        mask = resize(mask, mask_size)
        pred_mask = model(img)

        save_img_and_predicted_mask("train-sample", img[0], pred_mask[0], mask[0])

        loss = loss_fn(pred_mask, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        time_per_img = (end - start) / img.shape[0]

        train_loss += loss

        print(
            f"Epoch {epoch} train batch {batch_idx} loss {loss} ({time_per_img:.2f} sec/img)"
        )

    train_loss /= len(dataloader)
    return train_loss


def test(dataloader, model, loss_fn):
    test_loss = 0
    n_batches = len(dataloader)

    with torch.no_grad():
        for img_fname, img, mask in dataloader:
            img = resize(img, mask_size)
            mask = resize(mask, mask_size)
            pred_mask = model(img)
            test_loss += loss_fn(pred_mask, mask).item()
            save_img_and_predicted_mask("test-sample", img[0], pred_mask[0], mask[0])

    test_loss /= n_batches
    return test_loss


def save_checkpoint(filename, model, optimizer, epoch):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        filename,
    )


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def main():
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint to load")
    args = parser.parse_args()

    print(f"Torch threads {torch.get_num_threads()}")

    max_images = 100
    batch_size = 4

    train_dataset = DDI100(args.data_dir, train=True, max_images=max_images)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_dataset = DDI100(args.data_dir, train=False, max_images=max_images)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Train images {len(train_dataset)} in {len(train_dataloader)} batches")
    print(f"Validation images {len(val_dataset)} in {len(val_dataloader)} batches")

    model = DetectionModel()
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
        train_loss = train(epoch, train_dataloader, model, loss_fn, optimizer)
        val_loss = test(val_dataloader, model, loss_fn)
        print(f"Epoch {epoch} train loss {train_loss} validation loss {val_loss}")

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
