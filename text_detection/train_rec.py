from argparse import ArgumentParser
import math
import os
from typing import Callable

import torch
from torch.nn import CTCLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
import wandb

from .datasets import (
    DEFAULT_ALPHABET,
    HierTextRecognition,
    ctc_greedy_decode_text,
    decode_text,
)
from .model import RecognitionModel
from .train import load_checkpoint, save_checkpoint


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

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # Predict [seq, batch, class] from [batch, 1, height, width].
            pred_seq = model(img)
            batch_loss = loss(pred_seq, text_seq, input_lengths, target_lengths)

        # Preview decoded text for first batch in the dataset.
        if batch_idx == 0:
            for i in range(min(10, len(text_seq))):
                y = text_seq[i]
                x = pred_seq[:, i, :].argmax(-1)
                target_text = decode_text(y, list(DEFAULT_ALPHABET))
                pred_text = ctc_greedy_decode_text(x, list(DEFAULT_ALPHABET))
                print(f'Train pred "{pred_text}" target "{target_text}"')

        if math.isnan(batch_loss.item()):
            raise Exception(
                "Training produced invalid loss. Check input and target lengths are compatible with CTC loss"
            )

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
            if batch_idx == 0:
                for i in range(min(10, len(text_seq))):
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


def ctc_input_and_target_compatible(input_len: int, target: torch.Tensor) -> bool:
    """
    Return true if a given input and target are compatible with CTC loss.

    The CTC loss function requires `input_length >= max(1, target_length)`.

    Additionally for every position in the target that has the same label as
    the previous position, the input will need an extra blank symbol to separate
    the repeated labels. This is because CTC decoding discards adjacent
    repeated symbols.

    :param input_len: Length of input sequence / width of image
    :param target: 1D tensor of class indices
    """
    target_len = target.shape[0]
    min_input_len = max(1, target_len)
    for i in range(1, target_len):
        if target[i - 1] == target[i]:
            min_input_len += 1
    return input_len >= min_input_len


def round_up(val: int, unit: int) -> int:
    """Round up `val` to the nearest multiple of `unit`."""
    rem = unit - val % unit
    return val + rem


def collate_samples(samples: list[dict]) -> dict:
    """
    Collate samples from a text recognition dataset.
    """

    def text_len(sample: dict) -> int:
        return sample["text_seq"].shape[0]

    def image_width(sample: dict) -> int:
        return sample["image"].shape[-1]

    # Determine width of batched tensors. We round up the value to reduce the
    # variation in tensor sizes across batches. Having too many distinct tensor
    # sizes has been observed to lead to memory fragmentation and ultimately
    # memory exhaustion when training on GPUs.
    max_img_len = round_up(max([image_width(s) for s in samples]), 250)
    max_text_len = round_up(max([text_len(s) for s in samples]), 250)

    # Remove samples where the target text is incompatible with the width of
    # the image after downsampling by the model's CNN, which reduces the
    # width by 4x.
    samples = [
        s
        for s in samples
        if ctc_input_and_target_compatible(image_width(s) // 4, s["text_seq"])
    ]

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
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint to load")
    parser.add_argument("--export", type=str, help="Export model to ONNX format")
    parser.add_argument(
        "--max-epochs", type=int, help="Maximum number of epochs to train for"
    )
    parser.add_argument(
        "--max-images", type=int, help="Maximum number of items to train on"
    )
    args = parser.parse_args()

    # Set to aid debugging of initial text recognition model
    pytorch_seed = 1234
    torch.manual_seed(pytorch_seed)

    if args.dataset_type == "hiertext":
        load_dataset = HierTextRecognition
    else:
        raise Exception(f"Unknown dataset type {args.dataset_type}")

    max_images = args.max_images
    if max_images:
        validation_max_images = max(10, int(max_images * 0.1))
    else:
        validation_max_images = None

    train_dataset = load_dataset(args.data_dir, train=True, max_images=max_images)
    val_dataset = load_dataset(
        args.data_dir, train=False, max_images=validation_max_images
    )

    # TODO - Check how shuffling affects HierTextRecognition caching of
    # individual images.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_samples,
        num_workers=2,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_samples,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = RecognitionModel(alphabet=DEFAULT_ALPHABET).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modal param count {total_params}")

    epoch = 0

    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer, device)
        epoch = checkpoint["epoch"]

    if args.export:
        test_batch = next(iter(val_dataloader))
        torch.onnx.export(
            model,
            test_batch["image"],
            args.export,
            input_names=["line_image"],
            output_names=["chars"],
            dynamic_axes={
                "line_image": {0: "batch", 3: "seq"},
                "chars": {0: "out_seq"},
            },
        )
        return

    # Enable experiment tracking via Weights and Biases if API key set.
    enable_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if enable_wandb:
        wandb.init(
            project="text-recognition",
            config={
                "batch_size": args.batch_size,
                "dataset_size": len(train_dataset),
                "model_params": total_params,
                "pytorch_seed": pytorch_seed,
            },
        )
        wandb.watch(model)

    while args.max_epochs is None or epoch < args.max_epochs:
        train_loss = train(epoch, device, train_dataloader, model, optimizer)

        print(f"Epoch {epoch} train loss {train_loss}")

        val_loss = test(device, val_dataloader, model)
        print(f"Epoch {epoch} validation loss {val_loss}")

        if enable_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        save_checkpoint("text-rec-checkpoint.pt", model, optimizer, epoch=epoch)

        epoch += 1


if __name__ == "__main__":
    main()
