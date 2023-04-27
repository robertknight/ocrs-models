from argparse import ArgumentParser
from typing import Callable

import torch
from torch.nn import CTCLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from .datasets import DEFAULT_ALPHABET, HierTextRecognition, decode_text
from .model import RecognitionModel


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
        input_lengths = batch["image_width"] // 4
        img = batch["image"]

        text_seq = batch["text_seq"]
        target_lengths = batch["text_len"]

        with torch.autograd.set_detect_anomaly(True):
            # Predict [seq, batch, class] from [batch, 1, height, width].
            pred_seq = model(img)

            # for i in range(len(text_seq)):
            #     target_text = decode_text(text_seq[i], list(DEFAULT_ALPHABET))
            #     pred_text = decode_text(
            #         pred_seq[:, i, :].argmax(-1), list(DEFAULT_ALPHABET)
            #     )
            #     print(f"Pred {pred_text} target {target_text} pred len {len(pred_text)} target len {len(target_text)}")

            batch_loss = loss(pred_seq, text_seq, input_lengths, target_lengths)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        mean_loss += batch_loss.item()

    train_iterable.clear()
    mean_loss /= len(dataloader)
    return mean_loss


def collate_samples(samples: list[dict]) -> dict:
    """
    Collate samples from a text recognition dataset.
    """

    # TODO - Handle the case where a sample's text sequence is longer than
    # the maximum allowed given the width of the image. This is needed as the
    # CTC loss restricts the length of the target sequence to be at most the
    # length of the input.

    max_img_len = max([s["image"].shape[-1] for s in samples])
    max_text_len = max([s["text_seq"].shape[0] for s in samples])

    for s in samples:
        s["text_len"] = s["text_seq"].shape[0]
        s["text_seq"] = F.pad(s["text_seq"], [0, max_text_len - s["text_len"]])

        s["image_width"] = s["image"].shape[-1]
        s["image"] = F.pad(s["image"], [0, max_img_len - s["image_width"]])

    return default_collate(samples)


def main():
    parser = ArgumentParser(description="Train text recognition model.")
    parser.add_argument("dataset_type", type=str, choices=["hiertext"])
    parser.add_argument("data_dir")
    args = parser.parse_args()

    if args.dataset_type == "hiertext":
        load_dataset = HierTextRecognition
    else:
        raise Exception(f"Unknown dataset type {args.dataset_type}")

    train_dataset = load_dataset(args.data_dir, train=True, max_images=177)

    # TODO - Investigate issue with image index 176 causing `nan` loss values.
    # item = train_dataset[-1]
    # print("last item", item["image"].shape, "text seq", item["text_seq"].shape)
    # return

    # TODO - Check how shuffling affects HierTextRecognition caching of
    # individual images.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=10,
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

        # TODO - Compute validation loss

        epoch += 1
        # TODO - Save checkpoints


if __name__ == "__main__":
    main()
