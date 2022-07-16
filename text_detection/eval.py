from argparse import ArgumentParser
import time
import sys

import torch
from torchvision.io import ImageReadMode, read_image
import torch
from torchvision.transforms.functional import InterpolationMode, resize, to_pil_image

from .model import DetectionModel
from .train import mask_size


def binarize_mask(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.where(mask > threshold, 1.0, 0.0)


def main():
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("out_basename")
    args = parser.parse_args()

    model = DetectionModel()
    model.eval()

    checkpoint = torch.load(args.model, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])

    input_img = read_image(args.image, ImageReadMode.GRAY)
    _, input_height, input_width = input_img.shape

    # Input and prediction target size. This matches the training process.
    target_size = mask_size

    img = input_img.float() / 255.0 - 0.5
    img = resize(img, target_size)

    # Save eval input for inspection
    to_pil_image(img + 0.5).save(f"{args.out_basename}-input.png")

    img = img.unsqueeze(0)  # Add dummy batch dimension
    start = time.time()
    with torch.inference_mode():
        pred_masks = model(img)
    end = time.time()

    print(f"Predicted text in {end - start:.2f}s", file=sys.stderr)

    pred_masks = pred_masks[0]  # Remove dummy batch dimension
    threshold = 0.5
    binary_mask = binarize_mask(pred_masks, threshold=threshold)
    binary_mask = resize(
        binary_mask, (input_height, input_width), InterpolationMode.NEAREST
    )
    text_mask = binary_mask[0]
    text_regions = (input_img.float() / 255.0) * text_mask

    to_pil_image(text_regions).save(f"{args.out_basename}-text-regions.png")
    to_pil_image(pred_masks[0]).save(f"{args.out_basename}-text-probs.png")
    to_pil_image(text_mask).save(f"{args.out_basename}-text-mask.png")


if __name__ == "__main__":
    main()
