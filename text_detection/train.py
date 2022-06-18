from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from .datasets import DDI100
from .model import DetectionModel


def train(dataloader, model, loss_fn, optimizer):
    for batch_idx, (img, mask) in enumerate(dataloader):
        img = resize(img, (771, 545))
        mask = resize(mask, (771, 545))
        pred_mask = model(img)
        loss = loss_fn(pred_mask, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx} loss {loss}")


def main():
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()

    learning_rate = 1e-3
    dataset = DDI100(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=10)
    model = DetectionModel()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for epoch in range(epochs):
        train(dataloader, model, loss_fn, optimizer)


if __name__ == "__main__":
    main()
