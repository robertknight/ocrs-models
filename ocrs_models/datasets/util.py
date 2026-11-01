from collections.abc import Sized

import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.geometry import JOIN_STYLE, MultiLineString
from shapely.geometry.polygon import LinearRing
from torch.utils.data import Dataset


class SizedDataset(Dataset, Sized):
    """Dataset with a known size."""


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


def generate_mask_u8(
    width: int, height: int, polys: list[Polygon], shrink_dist: float = SHRINK_DISTANCE
) -> torch.Tensor:
    """
    Generate a mask from polygons of words or lines.

    Returns a `[height, width]` uint8 mask where text regions have value 255
    and other pixels have value 0. The text regions are shrunk by `shrink_dist`
    along each edge to create more space between adjacent regions.

    :param width: Width of output image
    :param height: Height of output image
    :param polys: List of polygons to draw on the mask.
    """

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for poly in polys:
        if shrink_dist != 0.0:
            shrunk_poly = shrink_polygon(poly, dist=shrink_dist)
        else:
            shrunk_poly = poly
        if not shrunk_poly:
            continue
        draw.polygon(shrunk_poly, fill=255, outline=None)

    return torch.from_numpy(np.array(mask_img))


def generate_mask(
    width: int, height: int, polys: list[Polygon], shrink_dist: float = SHRINK_DISTANCE
) -> torch.Tensor:
    """
    Generate a binary float mask from polygons of words or lines.

    Like `generate_mask_u8`, but returns a float mask with values 0.0 and 1.0.
    """
    return generate_mask_u8(width, height, polys, shrink_dist).float() / 255.0


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
