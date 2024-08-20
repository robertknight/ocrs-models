from argparse import ArgumentParser, BooleanOptionalAction
import math
import os

from pylev import levenshtein
import torch
from torch.nn import CTCLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
import wandb

from .datasets.hiertext import DEFAULT_ALPHABET, HierTextRecognition
from .datasets.util import ctc_greedy_decode_text, decode_text
from .datasets import text_recognition_data_augmentations
from .models import RecognitionModel
from .train_detection import load_checkpoint, save_checkpoint


class RecognitionAccuracyStats:
    """
    Computes text recognition accuracy statistics.
    """

    def __init__(self):
        self.total_chars = 0
        self.char_errors = 0

    def update(
        self,
        targets: torch.Tensor,
        target_lengths: list[int],
        preds: torch.Tensor,
        pred_lengths: list[int],
    ):
        """
        Update running statistics given targets and predictions for a batch of images.

        :param targets: [batch, seq] tensor of target character indices
        :param target_lengths: Lengths of target sequences
        :param preds: [seq, batch, class] tensor of character predictions
        :param pred_lengths: Lengths of predicted sequences
        """

        assert len(target_lengths) == targets.size(0)
        assert len(pred_lengths) == preds.size(1)

        total_chars = sum(target_lengths)
        char_errors = 0

        # Convert [seq, batch, class] to [batch, seq] of char indices.
        preds = preds.argmax(-1).transpose(0, 1)

        # Convert targets and preds to `list[list[int]]`, as this is much faster
        # for text decoding to operate on, especially if the tensors are on the
        # GPU.
        preds_list = preds.tolist()
        targets_list = targets.tolist()

        alphabet_chars = list(DEFAULT_ALPHABET)

        for y, x, x_len in zip(targets_list, preds_list, pred_lengths):
            target_text = decode_text(y, alphabet_chars)
            pred_text = ctc_greedy_decode_text(x[:x_len], alphabet_chars)
            char_errors += levenshtein(target_text, pred_text)

        self.total_chars += total_chars
        self.char_errors += char_errors

    def char_error_rate(self) -> float:
        """
        Return the overall fraction of character-level errors.
        """
        return self.char_errors / self.total_chars

    def stats_dict(self) -> dict:
        """
        Return a dict of stats that is convenient for logging etc.
        """
        return {
            "char_error_rate": self.char_error_rate(),
        }


def train(
    epoch: int,
    device: torch.device,
    dataloader: DataLoader,
    model: RecognitionModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, RecognitionAccuracyStats]:
    """
    Run one epoch of training.

    Returns the mean loss and accuracy statistics.
    """
    model.train()

    train_iterable = tqdm(dataloader)
    train_iterable.set_description(f"Training (epoch {epoch})")
    mean_loss = 0.0
    stats = RecognitionAccuracyStats()

    loss = CTCLoss()
    total_grad_norm = 0.0

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

        stats.update(text_seq, target_lengths, pred_seq, input_lengths)

        # Preview decoded text for first batch in the dataset.
        if batch_idx == 0:
            for i in range(min(10, len(text_seq))):
                y = text_seq[i]
                x = pred_seq[:, i, :].argmax(-1)
                x_len = input_lengths[i]
                target_text = decode_text(y, list(DEFAULT_ALPHABET))
                pred_text = ctc_greedy_decode_text(x[:x_len], list(DEFAULT_ALPHABET))
                print(f'Sample train prediction "{pred_text}" target "{target_text}"')

        if math.isnan(batch_loss.item()):
            raise Exception(
                "Training produced invalid loss. Check input and target lengths are compatible with CTC loss"
            )

        batch_loss.backward()

        # Clip to prevent exploding gradients.
        #
        # See https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191.
        #
        # `max_norm` value was taken from observing typical mean norms of
        # "healthy" minibatches.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        total_grad_norm += grad_norm.item()

        optimizer.step()

        mean_loss += batch_loss.item()

    mean_grad_norm = total_grad_norm / len(train_iterable)
    print(f"Mean grad norm {mean_grad_norm}")

    train_iterable.clear()
    mean_loss /= len(dataloader)
    return mean_loss, stats


def test(
    device: torch.device,
    dataloader: DataLoader,
    model: RecognitionModel,
) -> tuple[float, RecognitionAccuracyStats]:
    """
    Run evaluation on a set of images.

    Returns the mean loss and accuracy statistics.
    """
    model.eval()

    test_iterable = tqdm(dataloader)
    test_iterable.set_description("Testing")
    mean_loss = 0.0
    stats = RecognitionAccuracyStats()

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

            stats.update(text_seq, target_lengths, pred_seq, input_lengths)

            # Preview decoded text for first batch in the dataset.
            if batch_idx == 0:
                for i in range(min(10, len(text_seq))):
                    y = text_seq[i]
                    x = pred_seq[:, i, :].argmax(-1)
                    x_len = input_lengths[i]

                    target_text = decode_text(y, list(DEFAULT_ALPHABET))
                    pred_text = ctc_greedy_decode_text(
                        x[:x_len], list(DEFAULT_ALPHABET)
                    )
                    print(
                        f'Sample test prediction "{pred_text}" target "{target_text}"'
                    )

            batch_loss = loss(pred_seq, text_seq, input_lengths, target_lengths)
            mean_loss += batch_loss.item()

    test_iterable.clear()
    mean_loss /= len(dataloader)
    return mean_loss, stats


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

    # Factor by which the model's output sequence length is reduced compared to
    # the width of the input image.
    downsample_factor = 4

    # Determine width of batched tensors. We round up the value to reduce the
    # variation in tensor sizes across batches. Having too many distinct tensor
    # sizes has been observed to lead to memory fragmentation and ultimately
    # memory exhaustion when training on GPUs.
    img_width_step = 256
    max_img_width = max(image_width(s) for s in samples)
    max_img_width = round_up(max_img_width, img_width_step)

    max_text_len = max(text_len(s) for s in samples)
    max_text_len = round_up(max_text_len, img_width_step // downsample_factor)

    # Remove samples where the target text is incompatible with the width of
    # the image after downsampling by the model's CNN, which reduces the
    # width by `downsample_factor`.
    samples = [
        s
        for s in samples
        if ctc_input_and_target_compatible(
            image_width(s) // downsample_factor, s["text_seq"]
        )
    ]

    for sample in samples:
        text_pad_value = 0  # CTC blank label
        sample["text_len"] = text_len(sample)
        sample["text_seq"] = F.pad(
            sample["text_seq"],
            [0, max_text_len - sample["text_len"]],
            mode="constant",
            value=text_pad_value,
        )

        image_pad_value = 0.0  # Grey, since image values are in [-0.5, 0.5]
        sample["image_width"] = image_width(sample)
        sample["image"] = F.pad(
            sample["image"],
            [0, max_img_width - sample["image_width"]],
            mode="constant",
            value=image_pad_value,
        )

    return default_collate(samples)


def main():
    parser = ArgumentParser(description="Train text recognition model.")
    parser.add_argument("dataset_type", type=str, choices=["hiertext"])
    parser.add_argument("data_dir")
    parser.add_argument(
        "--augment",
        default=True,
        action=BooleanOptionalAction,
        help="Enable data augmentations",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint to load")
    parser.add_argument("--export", type=str, help="Export model to ONNX format")
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument(
        "--max-epochs", type=int, help="Maximum number of epochs to train for"
    )
    parser.add_argument(
        "--max-images", type=int, help="Maximum number of items to train on"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation on an exiting model",
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

    if args.augment:
        augmentations = text_recognition_data_augmentations()
    else:
        augmentations = None

    train_dataset = load_dataset(
        args.data_dir, train=True, max_images=max_images, transform=augmentations
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_samples,
        num_workers=2,
        pin_memory=True,
    )

    val_dataset = load_dataset(
        args.data_dir, train=False, max_images=validation_max_images
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

    initial_lr = args.lr or 1e-3  # 1e-3 is the Adam default
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=3
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model param count {total_params}")

    epoch = 0

    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer, device)
        epoch = checkpoint["epoch"]

    if args.export:
        test_batch = next(iter(val_dataloader))
        torch.onnx.export(
            model,
            test_batch["image"].to(device),
            args.export,
            input_names=["line_image"],
            output_names=["chars"],
            dynamic_axes={
                "line_image": {0: "batch", 3: "seq"},
                "chars": {0: "out_seq"},
            },
        )
        return

    if args.validate_only:
        val_loss, val_stats = test(device, val_dataloader, model)
        print(
            f"Validation loss {val_loss} char error rate {val_stats.char_error_rate()}"
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
        train_loss, train_stats = train(
            epoch, device, train_dataloader, model, optimizer
        )

        print(
            f"Epoch {epoch} train loss {train_loss} char error rate {train_stats.char_error_rate()}"
        )

        val_loss, val_stats = test(device, val_dataloader, model)
        print(
            f"Epoch {epoch} validation loss {val_loss} char error rate {val_stats.char_error_rate()}"
        )

        scheduler.step(val_loss)

        print(f"Current learning rate {scheduler.get_last_lr()}")

        if enable_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_stats.stats_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_stats.stats_dict(),
                }
            )

        save_checkpoint("text-rec-checkpoint.pt", model, optimizer, epoch=epoch)

        epoch += 1


if __name__ == "__main__":
    main()
