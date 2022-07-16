from argparse import ArgumentParser
import time
import sys

import torch
from torchvision.io import ImageReadMode, read_image
import torch
from torchvision.transforms.functional import InterpolationMode, resize, to_pil_image

from .model import DetectionModel
from .datasets import SHRINK_DISTANCE
from .postprocess import draw_quads, expand_quads, extract_cc_quads
from .train import mask_size


def binarize_mask(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.where(mask > threshold, 1.0, 0.0)


def main():
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("out_basename")
    args = parser.parse_args()

    model = DetectionModel(eval_only=True)
    model.eval()

    checkpoint = torch.load(args.model, map_location=torch.device("cpu"))
    model_state = DetectionModel.get_eval_only_state(checkpoint["model_state"])
    model.load_state_dict(model_state)

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

    text_prob_mask = binary_mask[0]
    text_prob_regions = (input_img.float() / 255.0) * text_prob_mask

    to_pil_image(text_prob_regions).save(f"{args.out_basename}-prob-regions.png")
    to_pil_image(text_prob_mask).save(f"{args.out_basename}-prob-mask.png")

    prob_pred = pred_masks[0]
    to_pil_image(prob_pred).save(f"{args.out_basename}-prob-pred.png")

    # Extract word quads and expand them to compensate for the shrinking that is
    # applied to text polygons when text masks are generated during training.
    word_quads = extract_cc_quads(text_prob_mask)
    word_quads = expand_quads(word_quads, dist=SHRINK_DISTANCE)
    word_quad_image = draw_quads(input_img, word_quads)
    word_quad_image.save(f"{args.out_basename}-text-words.png")

    # By default the model only outputs the probability mask during inference.
    # If binary and threshold mask output has also been enabled, then save those.
    #
    # The binary mask is trained with the same supervision as the probability
    # mask, so it should be almost identical.
    if len(pred_masks) > 1:
        text_bin_mask = binary_mask[1]
        text_bin_regions = (input_img.float() / 255.0) * text_bin_mask
        bin_prob_diff = text_prob_mask.logical_xor(text_bin_mask).float()

        to_pil_image(text_bin_regions).save(f"{args.out_basename}-bin-regions.png")
        to_pil_image(text_bin_mask).save(f"{args.out_basename}-bin-mask.png")
        to_pil_image(bin_prob_diff).save(f"{args.out_basename}-prob-bin-diff.png")

        bin_pred = pred_masks[1]
        to_pil_image(bin_pred).save(f"{args.out_basename}-bin-pred.png")

        thresh_pred = pred_masks[2]
        to_pil_image(thresh_pred).save(f"{args.out_basename}-thresh-pred.png")


if __name__ == "__main__":
    main()
