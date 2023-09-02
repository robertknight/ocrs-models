from argparse import ArgumentParser
import json

import torch

from .datasets import draw_word_boxes
from .model import LayoutModel


def word_box_tensor(
    word_boxes: list[list[float]], img_width: int, img_height: int
) -> torch.Tensor:
    """
    Convert a list of word box coordinates into input for the layout model.

    :param word_boxes: A list of word coordinates. Each item is a list of
        [left, top, right, bottom] coordinates for a single word.
    :return: A `(len(word_boxes), D)` item for the layout model.
    """
    n_features = 4
    x = torch.zeros((len(word_boxes), n_features))

    def norm_x(coord: float):
        return (coord / img_width) - 0.5

    def norm_y(coord: float):
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
    args = parser.parse_args()

    model = LayoutModel()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state"])

    with open(args.word_box_file) as f:
        wb_json = json.load(f)

        img_width = wb_json["resolution"]["width"]
        img_height = wb_json["resolution"]["height"]
        word_list = []
        for para in wb_json["paragraphs"]:
            for word in para["words"]:
                word_list.append(word["coords"])

        word_boxes = word_box_tensor(word_list, img_width, img_height)
        word_boxes = word_boxes.unsqueeze(0)  # Add batch dim
        label_probs = model(word_boxes)

        threshold = 0.5
        labels = label_probs > threshold
        n_line_starts = labels[:, :, 0].sum()
        n_line_ends = labels[:, :, 1].sum()
        print(
            f"Words {len(word_list)} predicted line starts {n_line_starts} line ends {n_line_ends}"
        )

        label_img = args.output_file
        draw_word_boxes(label_img, img_width, img_height, word_boxes[0], labels[0])


if __name__ == "__main__":
    main()
