from argparse import ArgumentParser
import time
import sys

import torch
from torchvision.io import ImageReadMode, read_image
import torch
from torchvision.transforms.functional import InterpolationMode, resize, to_pil_image

from .model import DetectionModel


def binarize_mask(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.where(mask > threshold, 1.0, 0.0)


def main():
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("out_file")
    args = parser.parse_args()

    model = DetectionModel()
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint["model_state"])

    input_img = read_image(args.image, ImageReadMode.GRAY)
    _, input_height, input_width = input_img.shape

    # Target size and image processing here matches the model.
    target_size = (385, 272)
    # target_size = (input_height // 2, input_width // 2)

    img = input_img.float() / 255.0 - 0.5
    img = resize(img, target_size)

    # Save eval input for inspection
    to_pil_image(img + 0.5).save("eval-input.png")

    img = img.unsqueeze(0)  # Add dummy batch dimension
    start = time.time()
    pred_mask = model(img)
    end = time.time()

    print(f"Predicted text in {end - start:.2f}s", file=sys.stderr)

    pred_mask = pred_mask[0]  # Remove dummy batch dimension
    binary_mask = binarize_mask(pred_mask, threshold=0.3)
    binary_mask = resize(
        binary_mask, (input_height, input_width), InterpolationMode.NEAREST
    )

    pred_mask_pil = to_pil_image(binary_mask)
    pred_mask_pil.save(args.out_file)


if __name__ == "__main__":
    main()
