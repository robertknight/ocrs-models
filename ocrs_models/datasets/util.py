from collections.abc import Sized
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiLineString, JOIN_STYLE
from shapely.geometry.polygon import LinearRing
import torch
from torch.utils.data import Dataset


class SizedDataset(Dataset, Sized):
    """Dataset with a known size."""

    pass


SHRINK_DISTANCE = 3.0
"""
Number of pixels by which text regions are shrunk in the text mask.

Each edge in a text polygon is offset by this number of pixels before being
used to draw and fill an area in the text mask.
"""


def transform_image(img: torch.Tensor) -> torch.Tensor:
    """
    Transform an image into the format expected by models in this package.

    :param img: 8-bit grayscale image in CHW format.
    :return: Float CHW tensor with pixel values in [-0.5, 0.5]
    """

    return img.float() / 255.0 - 0.5


def untransform_image(img: torch.Tensor) -> torch.Tensor:
    """
    Invert the transforms done by `transform_image`.

    :param img: CHW tensor with pixel values in [-0.5, 0.5]
    :return: 8-bit CHW tensor with values in [0, 255]
    """
    return ((img + 0.5) * 255.0).type(torch.uint8)


Polygon = list[tuple[int, int]]
"""
Polygon specified as a list of (x, y) coordinates of corners in clockwise order.
"""


def shrink_polygon(poly: Polygon, dist: float) -> Polygon:
    """
    Construct a shrunk version of `poly`.

    In the shrunk polygon, each edge will be at an offset of `dist` from the
    corresponding edge in the input. The returned polygon may be empty if
    it is thin relative to `dist`.
    """
    ring = LinearRing(poly)

    # The offset side needed to shrink the input depends on the orientation of
    # the points.
    side = "left" if ring.is_ccw else "right"
    shrunk_line = ring.parallel_offset(dist, side, join_style=JOIN_STYLE.mitre)

    if isinstance(shrunk_line, MultiLineString):
        # The input polygon may be split if it is thin in places. To simplify
        # consuming code, we return an empty result as if the whole polygon
        # was thin and became empty after shrinking.
        return []

    return list(shrunk_line.coords)


def generate_mask(
    width: int, height: int, polys: list[Polygon], shrink_dist: float = SHRINK_DISTANCE
) -> torch.Tensor:
    """
    Generate a mask in CHW format from polygons of words or lines.

    Returns a binary mask indicating text regions in the image. The text regions
    are shrunk by `shrink_dist` along each edge to create more space between
    adjacent regions.

    :param width: Width of output image
    :param height: Height of output image
    :param polys: List of polygons to draw on the mask.
    """

    mask_img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for poly in polys:
        if shrink_dist != 0.0:
            shrunk_poly = shrink_polygon(poly, dist=shrink_dist)
        else:
            shrunk_poly = poly
        if not shrunk_poly:
            continue
        draw.polygon(shrunk_poly, fill="white", outline=None)

    # Use numpy to convert the mask from bool -> float rather than PyTorch to
    # work around https://github.com/pytorch/pytorch/issues/54789. This caused
    # True values to be mapped to 255.0 instead of 1.0 on Linux (but not macOS).
    mask = np.array(mask_img, dtype=np.float32)

    return torch.Tensor(mask)


def encode_text(text: str, alphabet: list[str], unknown_char: str) -> torch.Tensor:
    """
    Convert `text` into a `[len(text)]` tensor of class indices.

    Each class index is the index of the character in `alphabet` + 1. The class
    0 is reserved for the blank character. If a character is encountered in
    `text` which does not appear in `alphabet`, the character `unknown_char` is
    substituted.
    """
    x = torch.zeros(len(text), dtype=torch.int32)
    for i, ch in enumerate(text):
        try:
            char_idx = alphabet.index(ch)
        except ValueError:
            char_idx = alphabet.index(unknown_char)
        x[i] = char_idx + 1
    return x


def decode_text(x: torch.Tensor | list[int], alphabet: list[str]) -> str:
    """
    Convert a vector of character class labels into a string.

    The range of class labels is `len(alphabet) + 1`. The label 0 is reserved
    for the blank character.
    """

    # Indexing into a list is much faster than indexing into a 1D tensor.
    if isinstance(x, torch.Tensor):
        x = x.tolist()

    return "".join([alphabet[char_idx - 1] for char_idx in x if char_idx > 0])


def ctc_greedy_decode_text(x: torch.Tensor | list[int], alphabet: list[str]) -> str:
    """
    Perform greedy CTC decoding of a sequence to text.

    The difference between this function and `decode_text` is that this function
    skips repeated characters.

    `x` is a vector of character class labels, where the range of labels is
    `len(alphabet) + 1`. The label 0 is reserved for the blank character.
    """
    chars = ""

    # Indexing into a list is much faster than indexing into a 1D tensor.
    if isinstance(x, torch.Tensor):
        x = x.tolist()

    last_cls = None
    for cls in x:
        # Skip repeated labels.
        if cls == last_cls:
            continue

        last_cls = cls

        # Skip blanks.
        if cls == 0:
            continue

        chars += alphabet[cls - 1]

    return chars


def clamp(val: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(val, max_val))


def bounding_box_size(vertices: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Return the width and height of the bounding box of a list of vertices.

    Vertices are specified as [x, y] coordinate tuples or lists.
    """
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    return (max_x - min_x, max_y - min_y)


def intervals_overlap(a: float, b: float, c: float, d: float) -> bool:
    """
    Return true if the interval `[a, b]` overlaps `[c, d]`.
    """
    if a <= c:
        return b > c
    else:
        return d > a


def draw_word_boxes(
    img_path: str,
    width: int,
    height: int,
    word_boxes: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
    threshold=0.5,
    normalized_coords=False,
):
    """
    Draw word bounding boxes on an image and color them according to labels
    or probabilities.

    :param word_boxes: A (W, D) tensor of word bounding boxes, where D is a
        [left, top, right, bottom] feature vector.

        If `normalized_coords` is True, coordinates are assumed to be scaled and
        offset such that (0, 0) is the center of the image and coordinates are
        in the range [-0.5, 0.5].
    :param labels: A (W, L) tensor of labels, where L is a (line_start, line_end)
        category vector.
    :param probs: A (W,) tensor of probabilities for each word.
    :param threshold: If `probs` is specified, probabilities above this threshold
        are colored differently to indicate positive instances
    """
    n_words, n_feats = word_boxes.shape
    assert n_feats == 4

    if labels is not None:
        n_labels, n_cats = labels.shape
        assert n_labels == n_words
        assert n_cats == 2

    if probs is not None:
        assert probs.shape == (n_words,)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    color: str | tuple[int, int, int]

    def scale_x(coord: float) -> float:
        if not normalized_coords:
            return coord
        return (coord + 0.5) * width

    def scale_y(coord: float) -> float:
        if not normalized_coords:
            return coord
        return (coord + 0.5) * height

    for i in range(n_words):
        left, top, right, bottom = word_boxes[i].tolist()
        left, top, right, bottom = (
            scale_x(left),
            scale_y(top),
            scale_x(right),
            scale_y(bottom),
        )

        if labels is not None:
            line_start, line_end = labels[i].tolist()
            match (line_start, line_end):
                case (1, 1):
                    color = "green"
                case (1, 0):
                    color = "blue"
                case (0, 1):
                    color = "red"
                case _:
                    color = "black"
        elif probs is not None:
            word_prob = probs[i].item()
            if word_prob > threshold:
                color = (255, 0, 0)
            else:
                min_val = 20
                max_val = 255
                prob_color = max_val - round(word_prob * (max_val - min_val))
                color = (prob_color, prob_color, prob_color)
        else:
            color = "black"

        draw.rectangle((left, top, right, bottom), fill=None, outline=color, width=2)

    img.save(img_path)
