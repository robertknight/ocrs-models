from argparse import ArgumentParser
import time
import sys

import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import resize, to_pil_image

from .model import DetectionModel


def main():
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("out_file")
    args = parser.parse_args()

    model = DetectionModel()
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint["model_state"])

    # Target size and image processing here matches the model.
    target_size = (385, 272)
    img = read_image(args.image, ImageReadMode.GRAY).float() / 255.0 - 0.5

    img = resize(img, target_size)

    # Save eval input for inspection
    to_pil_image(img + 0.5).save("eval-input.png")

    img = img.unsqueeze(0)  # Add dummy batch dimension
    start = time.time()
    pred_mask = model(img)
    end = time.time()

    print(f"Predicted text in {end - start:.2f}s", file=sys.stderr)

    pred_mask = pred_mask[0]  # Remove dummy batch dimension

    pred_mask_pil = to_pil_image(pred_mask)
    pred_mask_pil.save(args.out_file)


if __name__ == "__main__":
    main()
