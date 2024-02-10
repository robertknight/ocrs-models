from argparse import ArgumentParser
import json

import torch

from .datasets.util import draw_word_boxes
from .models import LayoutModel


def word_box_tensor(
    word_boxes: list[list[float]],
    img_width: int,
    img_height: int,
    normalize_coords=False,
) -> torch.Tensor:
    """
    Convert a list of word box coordinates into input for the layout model.

    :param word_boxes: A list of word coordinates. Each item is a list of
        [left, top, right, bottom] coordinates for a single word.
    :param normalize_coords: Whether to normalize coordinates such that (0, 0)
        is the center of the image and coordinates are in the range [-0.5, 0.5].
    :return: A `(len(word_boxes), D)` item for the layout model.
    """
    n_features = 4
    x = torch.zeros((len(word_boxes), n_features))

    def norm_x(coord: float):
        if not normalize_coords:
            return coord
        return (coord / img_width) - 0.5

    def norm_y(coord: float):
        if not normalize_coords:
            return coord
        return (coord / img_height) - 0.5

    for i, word_coords in enumerate(word_boxes):
        left, top, right, bottom = word_coords
        x[i, 0] = norm_x(left)
        x[i, 1] = norm_y(top)
        x[i, 2] = norm_x(right)
        x[i, 3] = norm_y(bottom)

    return x


def main():
    parser = ArgumentParser("Evaluate text layout model and preview results.")
    parser.add_argument("word_box_file")
    parser.add_argument("output_file")
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="Model checkpoint to load"
    )
    parser.add_argument(
        "--colors",
        choices=["labels", "line-start-probs", "line-end-probs"],
        help="Meaning of box colors",
    )
    args = parser.parse_args()

    model = LayoutModel(return_probs=True)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])

    # True if model requires coordinates to be normalized into the range [-0.5,
    # 0.5], with (0, 0) being the center of the image.
    normalize_coords = False

    with open(args.word_box_file) as f:
        wb_json = json.load(f)

        img_width = wb_json["resolution"]["width"]
        img_height = wb_json["resolution"]["height"]
        word_list = []
        for para in wb_json["paragraphs"]:
            for word in para["words"]:
                coords = [float(c) for c in word["coords"]]
                word_list.append(coords)

        word_boxes = word_box_tensor(
            word_list, img_width, img_height, normalize_coords=normalize_coords
        )
        word_boxes = word_boxes.unsqueeze(0)  # Add batch dim
        label_probs = model(word_boxes)

        labels = None
        probs = None

        match args.colors:
            case "labels":
                threshold = 0.5
                labels = label_probs > threshold
                n_line_starts = labels[:, :, 0].sum()
                n_line_ends = labels[:, :, 1].sum()
                labels = labels[0]
                print(
                    f"Words {len(word_list)} predicted line starts {n_line_starts} line ends {n_line_ends}"
                )
            case "line-start-probs":
                probs = label_probs[0, :, 0]
            case "line-end-probs":
                probs = label_probs[0, :, 1]

        label_img = args.output_file
        draw_word_boxes(
            label_img,
            img_width,
            img_height,
            word_boxes[0],
            labels,
            probs,
            normalized_coords=normalize_coords,
        )


if __name__ == "__main__":
    main()
