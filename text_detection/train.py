from argparse import ArgumentParser
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from .datasets import DDI100
from .model import DetectionModel

# This is approx 1/5th of the DDI-100 input image size.
# mask_size = (771, 545)

# This is approx 1/10th of the DDI-100 input image size.
mask_size = (385, 272)


def save_img_and_predicted_mask(basename, img, mask):
    pass


def train(epoch, dataloader, model, loss_fn, optimizer):
    for batch_idx, (img, mask) in enumerate(dataloader):
        start = time.time()
        img = resize(img, mask_size)
        mask = resize(mask, mask_size)

        pred_mask = model(img)

        loss = loss_fn(pred_mask, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        time_per_img = (end - start) / img.shape[0]

        print(
            f"Epoch {epoch} train batch {batch_idx} loss {loss} ({time_per_img:.2f} sec/img)"
        )


def test(epoch, dataloader, model, loss_fn):
    test_loss = 0
    n_batches = len(dataloader)

    with torch.no_grad():
        for img, mask in dataloader:
            img = resize(img, mask_size)
            mask = resize(mask, mask_size)
            pred_mask = model(img)
            test_loss += loss_fn(pred_mask, mask).item()

    test_loss /= n_batches
    print(f"Epoch {epoch} test loss {test_loss}")


def main():
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()

    print(f"Torch threads {torch.get_num_threads()}")

    max_images = 100
    train_dataset = DDI100(args.data_dir, train=True, max_images=max_images)
    train_dataloader = DataLoader(
        train_dataset, batch_size=10, shuffle=True, num_workers=2
    )

    val_dataset = DDI100(args.data_dir, train=False, max_images=max_images)
    val_dataloader = DataLoader(val_dataset, batch_size=10)

    print(f"Train images {len(train_dataset)} in {len(train_dataloader)} batches")
    print(f"Validation images {len(val_dataset)} in {len(val_dataloader)} batches")

    learning_rate = 0.005
    model = DetectionModel()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model param count {total_params}")

    epochs = 30
    for epoch in range(epochs):
        train(epoch, train_dataloader, model, loss_fn, optimizer)
        test(epoch, val_dataloader, model, loss_fn)


if __name__ == "__main__":
    main()
