from argparse import ArgumentParser, BooleanOptionalAction
import gzip
import json
import os
from os.path import basename
import pickle
import pickletools
from typing import Callable, Optional, TextIO, cast

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image, write_png
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import resize
from torchvision import transforms
from shapely.geometry import JOIN_STYLE, MultiLineString
from shapely.geometry.polygon import LinearRing


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


class DDI100Unpickler(pickle.Unpickler):
    """
    Restrictive unpickler for the DDI-100 dataset.

    Pickles are in general not a safe format for untrusted data as they can
    by default execute arbitrary code when deserialized.

    This unpickler attempts to reduce the risk by limiting which globals are
    allowed, per https://docs.python.org/3/library/pickle.html#restricting-globals.
    """

    def find_class(self, module, name):
        path = f"{module}.{name}"
        if path == "numpy.dtype":
            return np.dtype
        elif path == "numpy.ndarray":
            return np.ndarray
        elif path == "numpy.core.multiarray._reconstruct":
            return np.core.multiarray._reconstruct
        else:
            raise pickle.UnpicklingError(f"Disallowed class {module}.{name}")


class DDI100(Dataset):
    """
    Distorted Document Images (DDI-100) dataset.

    See https://github.com/machine-intelligence-laboratory/DDI-100
    and https://arxiv.org/abs/1912.11658.

    License: MIT
    """

    def __init__(self, root_dir: str, train=True, transform=None, max_images=None):
        self._img_dir = f"{root_dir}/gen_imgs"
        self._boxes_dir = f"{root_dir}/gen_boxes"
        self._img_filenames = sorted(os.listdir(self._img_dir))

        if max_images is not None:
            self._img_filenames = self._img_filenames[:max_images]

        train_split_idx = int(len(self._img_filenames) * 0.9)

        if train:
            self._img_filenames = self._img_filenames[:train_split_idx]
        else:
            self._img_filenames = self._img_filenames[train_split_idx:]

        if not os.path.exists(self._img_dir):
            raise Exception(f"Dataset images not found in {self._img_dir}")
        if not os.path.exists(self._boxes_dir):
            raise Exception(f"Dataset masks not found in {self._boxes_dir}")

        self.transform = transform

    def __len__(self):
        return len(self._img_filenames)

    def __getitem__(self, idx: int):
        """
        Return tuple of (document image, binary_mask) tensors.

        The document image is an CHW tensor with one greyscale color channel.
        The binary mask is an CHW tensor.
        """

        img_fname = self._img_filenames[idx]
        img_basename, _ = os.path.splitext(img_fname)
        img_path = f"{self._img_dir}/{self._img_filenames[idx]}"

        img = transform_image(read_image(img_path))

        # See https://github.com/machine-intelligence-laboratory/DDI-100/tree/master/dataset
        # for details of dataset structure.
        pickle_path = f"{self._boxes_dir}/{img_basename}.pickle"

        with open(pickle_path, "rb") as pickle:
            words = DDI100Unpickler(pickle).load()
            word_quads = [w["box"] for w in words]

        _, height, width = img.shape

        mask = generate_mask(width, height, word_quads)
        mask = torch.unsqueeze(mask, 0)  # Add channel dimension

        if self.transform:
            # Input and target are transformed in one call to ensure same
            # parameters are used for both, if transform is randomized.
            transformed = self.transform(torch.stack([img, mask]))
            img = transformed[0]
            mask = transformed[1]

        return {
            "path": img_path,
            "image": img,
            "text_mask": mask,
        }

    @staticmethod
    def _generate_mask(width: int, height: int, word_quads):
        def reorder_quad(quad):
            # Swap (x, y) coordinates
            coords = [(coord[1], coord[0]) for coord in quad.tolist()]

            # Sort corners into clockwise order starting from top-left
            bottom_left, top_left, bottom_right, top_right = coords
            return [top_left, top_right, bottom_right, bottom_left]

        reordered_quads = [reorder_quad(q) for q in word_quads]
        return generate_mask(width, height, reordered_quads)


class HierText(Dataset):
    """
    HierText dataset.

    See https://github.com/google-research-datasets/hiertext and
    https://arxiv.org/abs/2203.15143.

    Photos from the Open Images dataset [1], containing text in natural scenes
    and documents, annotated with paragraph, line and word-level polygons and
    bounding boxes.

    [1] https://storage.googleapis.com/openimages/web/index.html

    License: CC BY-SA 4.0
    """

    def __init__(self, root_dir: str, train=True, transform=None, max_images=None):
        if train:
            self._img_dir = f"{root_dir}/train"
            annotations_file = f"{root_dir}/gt/train.jsonl.gz"
        else:
            self._img_dir = f"{root_dir}/validation"
            annotations_file = f"{root_dir}/gt/validation.jsonl.gz"

        if not os.path.exists(self._img_dir):
            raise Exception(f'Image directory "{self._img_dir}" not found')

        if not os.path.exists(annotations_file):
            raise Exception(f'Label data file "{annotations_file}" not found')

        lines_file = annotations_file.replace(".jsonl.gz", ".jsonl")
        self._generate_json_lines_annotations(annotations_file, lines_file)

        with open(lines_file) as fp:
            self._annotations = [line for line in fp]

        if max_images:
            self._annotations = self._annotations[:max_images]

        self.transform = transform

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, idx: int):
        """
        Return tuple of (document image, binary_mask) tensors.

        The document image is a CHW tensor with one greyscale color channel.
        The binary mask is a CHW tensor.
        """
        annotations = json.loads(self._annotations[idx])
        img_id = annotations["image_id"]
        img_path = f"{self._img_dir}/{img_id}.jpg"

        word_polys: list[Polygon] = []
        for para in annotations["paragraphs"]:
            for line in para["lines"]:
                for word in line["words"]:
                    # This currently adds all words, regardless of whether
                    # they are legible, handwritten vs printed and vertical vs
                    # horizontal. We might want to filter based on those
                    # criteria.
                    poly = [tuple(coord) for coord in word["vertices"]]
                    word_polys.append(cast(Polygon, poly))

        img = transform_image(read_image(img_path, ImageReadMode.GRAY))
        _, height, width = img.shape

        mask = generate_mask(width, height, word_polys)
        mask = torch.unsqueeze(mask, 0)  # Add channel dimension

        if self.transform:
            # Input and target are transformed in one call to ensure same
            # parameters are used for both, if transform is randomized.
            transformed = self.transform(torch.stack([img, mask]))
            img = transformed[0]
            mask = transformed[1]

        return {
            "path": img_path,
            "image": img,
            "text_mask": mask,
        }

    @staticmethod
    def _generate_json_lines_annotations(annotations_file: str, lines_file: str):
        """
        Generate a JSON Lines version of annotation data in `annotations_file`.

        The training data is a large gzipped JSON file which is slow to parse
        (despite the ".jsonl.gz" suffix, it is just JSON).

        Convert this to JSONL, with one line per image, which can loaded much
        more quickly, as individual entries can be parsed only when needed.
        """
        if os.path.exists(lines_file) and (
            os.path.getmtime(lines_file) >= os.path.getmtime(annotations_file)
        ):
            return

        print("Converting annotations from JSON to JSONL format...")
        with gzip.open(annotations_file) as in_fp:
            annotations = json.load(in_fp)["annotations"]

            with open(lines_file, "w") as out_fp:
                for ann in tqdm(annotations):
                    ann_json = json.dumps(ann)
                    out_fp.write(f"{ann_json}\n")


DEFAULT_ALPHABET = (
    " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    + chr(8364)  # Euro symbol. Escaped to work around issue in Vim + tmux.
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
"""
Default alphabet used by text recognition models.

This closely matches the English "gen2" model from EasyOCR.
"""


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


class HierTextRecognition(Dataset):
    """
    HierText dataset for text recognition.

    See the `HierText` class for general notes on the HierText dataset. This
    class yields greyscale images of text lines from the dataset, along with
    the associated text content as a one-hot encoded sequence of class labels.
    """

    def __init__(
        self,
        root_dir: str,
        train=True,
        transform=None,
        max_images=None,
        alphabet: Optional[list[str]] = None,
        output_height: int = 64,
    ):
        super().__init__()

        if alphabet is None:
            alphabet = [c for c in DEFAULT_ALPHABET]
        self.alphabet = cast(list[str], alphabet)

        if train:
            self._img_dir = f"{root_dir}/train"
            self._cache_dir = f"{root_dir}/train-lines-cache"
            annotations_file = f"{root_dir}/gt/train.jsonl.gz"
        else:
            self._img_dir = f"{root_dir}/validation"
            self._cache_dir = f"{root_dir}/validation-lines-cache"
            annotations_file = f"{root_dir}/gt/validation.jsonl.gz"

        if not os.path.exists(self._img_dir):
            raise Exception(f'Image directory "{self._img_dir}" not found')

        if not os.path.exists(annotations_file):
            raise Exception(f'Label data file "{annotations_file}" not found')

        lines_file = annotations_file.replace(".jsonl.gz", "-lines.jsonl")
        self._generate_text_line_annotations(annotations_file, lines_file)

        with open(lines_file) as fp:
            self._text_lines = [line for line in fp]

        if max_images:
            self._text_lines = self._text_lines[:max_images]

        self.transform = transform
        self.output_height = output_height

    def _get_line_image(
        self, image_id: str, min_x: int, max_x: int, min_y: int, max_y: int
    ) -> torch.Tensor:
        """
        Load a cached greyscale text line image.

        `image_id` is the full image from which the text line is being extracted,
        the other params are the bounding box coordinates of the text line
        within the image.
        """

        assert min_x >= 0 and min_y >= 0 and max_x >= min_x and max_y >= min_y

        cache_path = f"{self._cache_dir}/{image_id}/{min_x}_{min_y}_{max_x}_{max_y}.png"
        if not os.path.exists(cache_path):
            img_path = f"{self._img_dir}/{image_id}.jpg"
            img = read_image(img_path, ImageReadMode.GRAY)
            _, img_height, img_width = img.shape

            min_x = clamp(min_x, 0, img_width - 1)
            max_x = clamp(max_x, 0, img_width - 1)
            min_y = clamp(min_y, 0, img_height - 1)
            max_y = clamp(max_y, 0, img_height - 1)

            line_img = img[:, min_y:max_y, min_x:max_x]

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # Write image to a temporary path. If the dataset is being read
            # concurrently by multiple processes, this avoids one of them
            # potentially seeing an incomplete cached image written by another.
            tmp_path = cache_path + ".tmp"
            write_png(line_img, tmp_path)
            os.rename(tmp_path, cache_path)

        return transform_image(read_image(cache_path, ImageReadMode.GRAY))

    def __len__(self):
        return len(self._text_lines)

    def __getitem__(self, idx: int):
        """
        Return a dict containing a line image and sequence label vector.

        The line image is a CHW tensor with one greyscale color channel.
        The text sequence is a [seq, class] tensor.
        """
        text_line = json.loads(self._text_lines[idx])
        img_id = text_line["image_id"]

        # Get line bounding box in image
        line_poly = [(coord[0], coord[1]) for coord in text_line["vertices"]]
        min_x = max(0, min([x for x, y in line_poly]))
        max_x = max(min_x, max([x for x, y in line_poly]))
        min_y = max(0, min([y for x, y in line_poly]))
        max_y = max(min_y, max([y for x, y in line_poly]))

        # Load or create cached line image
        line_img = self._get_line_image(img_id, min_x, max_x, min_y, max_y)
        _, line_height, line_width = line_img.shape

        for i, (x, y) in enumerate(line_poly):
            line_poly[i] = (x - min_x, y - min_y)

        mask = generate_mask(line_width, line_height, [line_poly], shrink_dist=0.0)
        mask = torch.unsqueeze(mask, 0)  # Add channel dimension

        if line_img.shape != mask.shape:
            print(
                f"Shape mismatch {line_img.shape} vs {mask.shape} line height {line_height} min y {min_y} max y {max_y} image size {img.shape}"
            )

        # Line images have a single channel with values in [-0.5, 0.5]. Mask off
        # the part of the image outside the text line and set their value to
        # -0.5 (representing black).
        background = torch.full(line_img.shape, -0.5) * (1.0 - mask)
        line_img = background + line_img * mask

        # Apply data augmentations. This may change the size of the image.
        if self.transform:
            line_img = self.transform(line_img)

            # Brightness / contrast transforms may have moved pixel values
            # outside of the legal range of [-0.5, 0.5] for normalized images.
            #
            # Clamp to bring these values back into that range.
            line_img = line_img.clamp(-0.5, 0.5)

            _, line_height, line_width = line_img.shape

        # Scale the width along with the height, within limits. The lower limit
        # avoids errors caused by images being zero-width after downsampling.
        # The upper limit bounds memory use by a single batch.
        aspect_ratio = line_width / line_height
        output_width = min(800, max(10, int(self.output_height * aspect_ratio)))

        line_img = resize(line_img, [self.output_height, output_width], antialias=True)

        # Encode the corresponding character sequence as a one-hot vector.
        text = text_line["text"]
        text_seq = encode_text(text, self.alphabet, unknown_char="?")

        return {
            "image_id": img_id,
            "image": line_img,
            "text_seq": text_seq,
        }

    @staticmethod
    def _generate_text_line_annotations(annotations_file: str, lines_file: str):
        """
        Generate text line annotations from the raw HierText dataset.

        The training data is a large gzipped JSON file which is slow to parse
        (despite the ".jsonl.gz" suffix, it is just JSON).

        Convert this to JSONL, with one line per text line in the input dataset.
        This can be loaded efficiently as individual entries can be parsed only
        when needed.
        """
        if os.path.exists(lines_file) and (
            os.path.getmtime(lines_file) >= os.path.getmtime(annotations_file)
        ):
            return

        # Minimum size of lines that will be used for training.
        MIN_WIDTH = 10
        MIN_HEIGHT = 10

        # Min ratio of area(union of word bounding boxes) and area(line
        # bounding box). A ratio below this threshold likely indicates that the
        # line is partially illegible.
        MIN_WORD_TO_LINE_AREA_RATIO = 0.8

        # Min width/height ratio of line bounding box. This filters out text
        # lines which are severely rotated (eg. by 90 degrees).
        MIN_ASPECT_RATIO = 1.0

        total = 0
        total_usable = 0
        total_legible = 0
        total_horizontal = 0
        total_size_ok = 0
        total_handwritten = 0
        total_word_to_line_area_ratio_ok = 0
        total_aspect_ok = 0

        print(f"Extracting text line annotations from {annotations_file}")
        with gzip.open(annotations_file) as in_fp:
            annotations = json.load(in_fp)["annotations"]

            with open(lines_file, "w") as out_fp:
                for ann in tqdm(annotations):
                    lines: list[dict] = []
                    for para in ann["paragraphs"]:
                        for line in para["lines"]:
                            vertices = line["vertices"]
                            width, height = bounding_box_size(vertices)
                            aspect_ratio = width / height
                            aspect_ratio_ok = aspect_ratio >= MIN_ASPECT_RATIO

                            words_width, words_height = bounding_box_size(
                                [
                                    vertex
                                    for word in line["words"]
                                    for vertex in word["vertices"]
                                ]
                            )
                            word_line_area_ratio = (words_width * words_height) / (
                                width * height
                            )
                            area_ratio_ok = (
                                word_line_area_ratio >= MIN_WORD_TO_LINE_AREA_RATIO
                            )
                            legible = line["legible"]

                            # Exclude vertical lines, as the text recognition
                            # model is not set up to handle these.
                            horizontal = not line["vertical"]

                            # Exclude very small lines, as these are likely to
                            # be illegible.
                            size_ok = width >= MIN_WIDTH and height >= MIN_HEIGHT

                            total += 1
                            if legible:
                                total_legible += 1
                            if horizontal:
                                total_horizontal += 1
                            if size_ok:
                                total_size_ok += 1
                            if area_ratio_ok:
                                total_word_to_line_area_ratio_ok += 1
                            if aspect_ratio_ok:
                                total_aspect_ok += 1
                            if line["handwritten"]:
                                total_handwritten += 1

                            usable = (
                                legible
                                and size_ok
                                and horizontal
                                and area_ratio_ok
                                and aspect_ratio_ok
                            )
                            if not usable:
                                continue
                            total_usable += 1

                            line_data = {
                                "image_id": ann["image_id"],
                                "vertices": vertices,
                                "text": line["text"],
                            }
                            ann_json = json.dumps(line_data)
                            out_fp.write(f"{ann_json}\n")

        # Display statistics about the percentages of lines which were kept /
        # excluded for different reasons.
        stats = {
            "Total lines": total,
            "Total usable for training": total_usable,
            "Legible": total_legible,
            "Horizontal": total_horizontal,
            f"Aspect ratio (width/height) >= {MIN_ASPECT_RATIO}": total_aspect_ok,
            f"Width >= {MIN_WIDTH} and Height >= {MIN_HEIGHT}": total_size_ok,
            f"Words/line area ratio >= {MIN_WORD_TO_LINE_AREA_RATIO}": total_word_to_line_area_ratio_ok,
        }
        for description, value in stats.items():
            percent = round((value / total) * 100, 1)
            print(f"{description}: {value} ({percent}%)")


def intervals_overlap(a: float, b: float, c: float, d: float) -> bool:
    """
    Return true if the interval `[a, b]` overlaps `[c, d]`.
    """
    if a <= c:
        return b > c
    else:
        return d > a


class WebLayout(Dataset):
    """
    Layout analysis dataset produced from rendering web pages.
    """

    def __init__(
        self,
        root_dir: str,
        randomize=False,
        padded_size: Optional[int] = None,
        train=True,
        max_images: Optional[int] = None,
        filter: Optional[Callable[[str], bool]] = None,
    ):
        """
        Construct dataset from JSON files in `root_dir`.

        :param root_dir: Directory to load dataset from
        :param filter: Filter images by file path
        :param max_images: Maximum number of images to load from dataset
        :param randomize:
            If true, coordinates of OCR boxes will be transformed randomly
            before being returned by `__getitem__`.
        :param padded_size:
            If set, pad the first dimension of the returned inputs and targets
            to the given length.
        :param train:
            If true, loading the training set. Otherwise load the validation
            split.
        """
        super().__init__()

        self.randomize = randomize
        self.root_dir = root_dir
        self.padded_size = padded_size

        files = [
            f
            for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f)) and f.endswith(".json")
        ]

        train_split = round(len(files) * 4 / 5)
        if train:
            self._files = files[:train_split]
        else:
            self._files = files[train_split:]

        if max_images is not None:
            self._files = self._files[:max_images]

        if filter:
            self._files = [f for f in self._files if filter(f)]

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx: int):
        """
        Returns tuple of (word_features, labels) tensors.

        `word_features` is an `[N, D]` tensor of feature vectors for each
        word box. The feature vector for each word has the format
        `[left, top, right, bottom, width, height]`.

        `labels` is a tensor of `[word_index, C]` binary classifications
        for the word. The classifications for each word are [line_start, line_end].
        """

        words = []
        labels = []
        in_path = os.path.join(self.root_dir, self._files[idx])

        if self.randomize:
            a, b, c = torch.rand(3).tolist()
            max_offset = 25
            max_scale = 0.1
            jitter_x = -max_offset + a * (max_offset * 2)
            jitter_y = -max_offset + b * (max_offset * 2)
            scale = (1.0 - max_scale) + c * (max_scale * 2)
        else:
            jitter_x = 0.0
            jitter_y = 0.0
            scale = 1.0

        with open(in_path) as file:
            content = json.load(file)
            viewport_width = int(content["resolution"]["width"])
            viewport_height = int(content["resolution"]["height"])

            def norm_x_coord(coord):
                return (coord / viewport_width) - 0.5

            def norm_y_coord(coord):
                return (coord / viewport_height) - 0.5

            def transform(coords):
                left, top, right, bottom = coords

                # Apply augmentation to the un-normalized coordinates.
                left = left * scale + jitter_x
                right = right * scale + jitter_x
                top = top * scale + jitter_y
                bottom = bottom * scale + jitter_y

                # Normalize coordinates such that the center of the image is
                # at (0, 0).
                left = norm_x_coord(left)
                top = norm_y_coord(top)
                right = norm_x_coord(right)
                bottom = norm_y_coord(bottom)

                return [left, top, right, bottom]

            for para in content["paragraphs"]:
                para_words = para["words"]

                for idx, word in enumerate(para_words):
                    left, top, right, bottom = transform(word["coords"])
                    words.append([left, top, right, bottom])

                    line_start = False
                    line_end = False
                    prev_word = para_words[idx - 1] if idx > 0 else None

                    if prev_word is None:
                        line_start = True
                    else:
                        # If there is no vertical overlap between the prev and
                        # current words, assume a new line. There is a case
                        # where this will give the wrong result: if a paragraph
                        # is split over multiple columns and the last line of
                        # a column overlaps the first line of the next.
                        prev_word_coords = transform(prev_word["coords"])
                        prev_word_top = prev_word_coords[1]
                        prev_word_bottom = prev_word_coords[3]
                        if not intervals_overlap(
                            prev_word_top, prev_word_bottom, top, bottom
                        ):
                            line_start = True

                    if idx == len(para_words) - 1:
                        line_end = True
                    else:
                        next_word_coords = transform(para_words[idx + 1]["coords"])
                        next_word_top = next_word_coords[1]
                        next_word_bottom = next_word_coords[3]
                        if not intervals_overlap(
                            top, bottom, next_word_top, next_word_bottom
                        ):
                            line_end = True

                    labels.append([int(line_start), int(line_end)])

        input_ = torch.Tensor(words)
        labels = torch.Tensor(labels)

        if self.padded_size:
            pad_len = self.padded_size - input_.shape[0]
            if pad_len > 0:
                input_ = F.pad(input_, (0, 0, 0, pad_len))
                labels = F.pad(labels, (0, 0, 0, pad_len))
            else:
                input_ = input_[0 : self.padded_size]
                labels = labels[0 : self.padded_size]

        return (input_, labels)


def text_recognition_data_augmentations():
    """
    Create a set of data augmentations for use with text recognition.
    """

    # Fill color for empty space created by transforms.
    # This is the "black" value for normalized images.
    transform_fill = -0.5

    augmentations = transforms.RandomApply(
        [
            transforms.RandomChoice(
                [
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.RandomRotation(
                        degrees=5,
                        fill=transform_fill,
                        expand=True,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.Pad(padding=(5, 5), fill=transform_fill),
                ]
            )
        ],
        p=0.5,
    )
    return augmentations


def draw_word_boxes(
    img_path: str,
    width: int,
    height: int,
    word_boxes: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Draw word bounding boxes on an image and color them according to their labels.

    :param word_boxes: A (W, D) tensor of word bounding boxes, where D is a
        [left, top, right, bottom] feature vector. Coordinates are assumed to
        be scaled and offset such that (0, 0) is the center of the image and
        the edges have coordinates of 1.
    :param labels: A (W, L) tensor of labels, where L is a (line_start, line_end)
        category vector.
    """
    n_words, n_feats = word_boxes.shape
    assert n_feats == 4

    n_labels, n_cats = labels.shape
    assert n_labels == n_words
    assert n_cats == 2

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    def scale_x(coord: float) -> float:
        return (coord + 0.5) * width

    def scale_y(coord: float) -> float:
        return (coord + 0.5) * height

    for i in range(n_words):
        left, top, right, bottom = word_boxes[i].tolist()
        left, top, right, bottom = (
            scale_x(left),
            scale_y(top),
            scale_x(right),
            scale_y(bottom),
        )
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

        draw.rectangle((left, top, right, bottom), fill=None, outline=color, width=1)

    img.save(img_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""
Preview items from a dataset.

The dataset must have been downloaded and extracted to a directory before
running this command.
"""
    )
    parser.add_argument(
        "dataset_type",
        choices=["ddi", "hiertext", "hiertext-rec", "web-layout"],
        help="Dataset to load",
    )
    parser.add_argument("root_dir", help="Root directory of dataset")
    parser.add_argument("out_dir", help="Directory to write output images to")
    parser.add_argument(
        "--augment",
        default=True,
        action=BooleanOptionalAction,
        help="Enable data augmentations",
    )
    parser.add_argument(
        "--filter", type=str, help="Filter input data items by file path"
    )
    parser.add_argument(
        "--max-images", type=int, help="Maximum number of items to process"
    )
    parser.add_argument(
        "--subset",
        choices=["train", "validation"],
        help="Subset of dataset to load",
        default="train",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def filter_item(path: str) -> bool:
        if args.filter is None:
            return True
        return args.filter in path

    match args.dataset_type:
        case "ddi" | "hiertext":
            # Explicitly cast dataset constructors to a common type to avoid mypy error.
            DatasetConstructor = Callable[..., DDI100 | HierText | HierTextRecognition]
            if args.dataset_type == "ddi":
                load_dataset = cast(DatasetConstructor, DDI100)
            else:
                load_dataset = cast(DatasetConstructor, HierText)

            dataset = load_dataset(
                args.root_dir,
                train=args.subset == "train",
                max_images=args.max_images,
            )

            for i in range(len(dataset)):
                # This loop doesn't use `enumerate` due to mypy error.
                item = dataset[i]

                print(f"Processing image {i+1}...")

                img = item["image"]
                mask = item["text_mask"]

                grey_img = untransform_image(img)
                rgb_img = grey_img.expand((3, grey_img.shape[1], grey_img.shape[2]))
                mask_hw = mask[0] > 0.5

                seg_img = draw_segmentation_masks(
                    rgb_img, mask_hw, alpha=0.3, colors="red"
                )
                write_png(seg_img, f"{args.out_dir}/seg-{i}.png")
        case "hiertext-rec":
            if args.augment:
                augmentations = text_recognition_data_augmentations()
            else:
                augmentations = None

            dataset = HierTextRecognition(
                args.root_dir,
                train=args.subset == "train",
                max_images=args.max_images,
                transform=augmentations,
            )

            for i in range(len(dataset)):
                item = dataset[i]
                img = item["image"]
                image_id = item["image_id"]
                text_seq = item["text_seq"]
                text = decode_text(text_seq, dataset.alphabet)

                print(
                    f'Text line {i} image {image_id} size {list(img.shape[1:])} text "{text}"'
                )
                text_path_safe = text.replace("/", "_").replace(":", "_")
                line_img_path = (
                    f"{args.out_dir}/line-{i}-{image_id}-{text_path_safe}.png"
                )

                write_png(untransform_image(img), line_img_path)
        case "web-layout":
            dataset = WebLayout(
                args.root_dir,
                train=args.subset == "train",
                max_images=args.max_images,
                filter=filter_item,
            )

            for i in range(len(dataset)):
                word_boxes, labels = dataset[i]
                out_img = f"{args.out_dir}/img-{i}.png"
                width = 1024
                height = 768
                draw_word_boxes(out_img, width, height, word_boxes, labels)
