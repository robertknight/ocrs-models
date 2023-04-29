from argparse import ArgumentParser
import math
from typing import Callable

import torch
from torch.nn import CTCLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from .datasets import (
    DEFAULT_ALPHABET,
    HierTextRecognition,
    ctc_greedy_decode_text,
    decode_text,
)
from .model import RecognitionModel
from .train import save_checkpoint


def train(
    epoch: int,
    device: torch.device,
    dataloader: DataLoader,
    model: RecognitionModel,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Run one epoch of training.

    Returns the mean loss for the epoch.
    """
    model.train()

    train_iterable = tqdm(dataloader)
    train_iterable.set_description(f"Training (epoch {epoch})")
    mean_loss = 0.0

    loss = CTCLoss()

    for batch_idx, batch in enumerate(train_iterable):
        # nb. Divide input_lengths by 4 to match the downsampling that the
        # model's CNN does.
        input_lengths = batch["image_width"].div(4, rounding_mode="floor")
        img = batch["image"].to(device)

        text_seq = batch["text_seq"].to(device)
        target_lengths = batch["text_len"]

        # Predict [seq, batch, class] from [batch, 1, height, width].
        pred_seq = model(img)

        # Preview decoded text for first batch in the dataset.
        if batch_idx == 0:
            for i in range(len(text_seq)):
                y = text_seq[i]
                x = pred_seq[:, i, :].argmax(-1)
                target_text = decode_text(y, list(DEFAULT_ALPHABET))
                pred_text = ctc_greedy_decode_text(x, list(DEFAULT_ALPHABET))
                print(f'Train pred "{pred_text}" target "{target_text}"')

        batch_loss = loss(pred_seq, text_seq, input_lengths, target_lengths)

        if math.isnan(batch_loss.item()):
            raise Exception(
                "Training produced invalid loss. Check input and target lengths are compatible with CTC loss"
            )

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        mean_loss += batch_loss.item()

    train_iterable.clear()
    mean_loss /= len(dataloader)
    return mean_loss


def test(
    device: torch.device,
    dataloader: DataLoader,
    model: RecognitionModel,
) -> float:
    """
    Run evaluation on a set of images.

    Returns the mean loss for the dataset.
    """
    model.eval()

    test_iterable = tqdm(dataloader)
    test_iterable.set_description(f"Testing")
    mean_loss = 0.0

    loss = CTCLoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iterable):
            # nb. Divide input_lengths by 4 to match the downsampling that the
            # model's CNN does.
            input_lengths = batch["image_width"].div(4, rounding_mode="floor")
            img = batch["image"].to(device)

            text_seq = batch["text_seq"].to(device)
            target_lengths = batch["text_len"]

            # Predict [seq, batch, class] from [batch, 1, height, width].
            pred_seq = model(img)

            # Preview decoded text for first batch in the dataset.
            for i in range(len(text_seq)):
                y = text_seq[i]
                x = pred_seq[:, i, :].argmax(-1)
                target_text = decode_text(y, list(DEFAULT_ALPHABET))
                pred_text = ctc_greedy_decode_text(x, list(DEFAULT_ALPHABET))
                print(f'Test pred "{pred_text}" target "{target_text}"')

            batch_loss = loss(pred_seq, text_seq, input_lengths, target_lengths)
            mean_loss += batch_loss.item()

    test_iterable.clear()
    mean_loss /= len(dataloader)
    return mean_loss


def collate_samples(samples: list[dict]) -> dict:
    """
    Collate samples from a text recognition dataset.
    """

    def text_len(sample: dict) -> int:
        return sample["text_seq"].shape[0]

    def image_width(sample: dict) -> int:
        return sample["image"].shape[-1]

    max_img_len = max([image_width(s) for s in samples])
    max_text_len = max([text_len(s) for s in samples])

    # Remove samples where the target text is longer than the image width
    # after downsampling by the model's CNN, which reduces the width by 4x.
    # CTC loss requires that the target sequence length is <= the input length.
    samples = [s for s in samples if text_len(s) <= image_width(s) // 4]

    for s in samples:
        s["text_len"] = text_len(s)
        s["text_seq"] = F.pad(s["text_seq"], [0, max_text_len - s["text_len"]])

        s["image_width"] = image_width(s)
        s["image"] = F.pad(s["image"], [0, max_img_len - s["image_width"]])

    return default_collate(samples)


def main():
    parser = ArgumentParser(description="Train text recognition model.")
    parser.add_argument("dataset_type", type=str, choices=["hiertext"])
    parser.add_argument("data_dir")
    args = parser.parse_args()

    # Set to aid debugging of initial text recognition model
    torch.manual_seed(1234)

    if args.dataset_type == "hiertext":
        load_dataset = HierTextRecognition
    else:
        raise Exception(f"Unknown dataset type {args.dataset_type}")

    train_dataset = load_dataset(args.data_dir, train=True, max_images=2000)
    val_dataset = load_dataset(args.data_dir, train=False, max_images=20)

    # TODO - Check how shuffling affects HierTextRecognition caching of
    # individual images.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        collate_fn=collate_samples,
        num_workers=2,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=20,
        shuffle=True,
        collate_fn=collate_samples,
        num_workers=2,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = RecognitionModel(alphabet=DEFAULT_ALPHABET).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modal param count {total_params}")

    epoch = 0
    while True:
        train_loss = train(epoch, device, train_dataloader, model, optimizer)

        print(f"Epoch {epoch} train loss {train_loss}")

        val_loss = test(device, val_dataloader, model)
        print(f"Epoch {epoch} validation loss {val_loss}")

        save_checkpoint("text-rec-checkpoint.pt", model, optimizer, epoch=epoch)

        epoch += 1


if __name__ == "__main__":
    main()
