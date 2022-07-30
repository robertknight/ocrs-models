from argparse import ArgumentParser
import gzip
import json
import os
import pickle
import pickletools
from typing import Callable, TextIO, cast

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image, write_png
from torchvision.utils import draw_segmentation_masks
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
    width: int, height: int, polys: list[Polygon]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a mask in CHW format from polygons of words or lines.

    Returns a tuple of (text_mask, border_mask) where `text_mask` is a mask
    indicating text regions in the image and `border_mask` indicates the border
    of text regions. The text regions are shrunk by `SHRINK_DISTANCE` along each
    edge to create more space between adjacent regions.

    :param width: Width of output image
    :param height: Height of output image
    :param polys: List of polygons to draw on the mask.
    """

    mask_img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for poly in polys:
        shrunk_poly = shrink_polygon(poly, dist=SHRINK_DISTANCE)
        if not shrunk_poly:
            continue
        draw.polygon(shrunk_poly, fill="white", outline=None)

    # Use numpy to convert the mask from bool -> float rather than PyTorch to
    # work around https://github.com/pytorch/pytorch/issues/54789. This caused
    # True values to be mapped to 255.0 instead of 1.0 on Linux (but not macOS).
    mask = np.array(mask_img, dtype=np.float32)

    kernel = np.ones((3, 3), dtype=np.float32)

    border_mask = cv2.dilate(mask, kernel, iterations=10)
    border_mask = border_mask - mask

    return (torch.Tensor(mask), torch.Tensor(border_mask))


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

        mask, border_mask = generate_mask(width, height, word_quads)
        mask = torch.unsqueeze(mask, 0)  # Add channel dimension
        border_mask = torch.unsqueeze(border_mask, 0)  # Add channel dimension

        if self.transform:
            # Input and target are transformed in one call to ensure same
            # parameters are used for both, if transform is randomized.
            transformed = self.transform(torch.stack([img, mask, border_mask]))
            img = transformed[0]
            mask = transformed[1]
            border_mask = transformed[2]

        return {
            "path": img_path,
            "image": img,
            "text_mask": mask,
            "border_mask": border_mask,
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

        mask, border_mask = generate_mask(width, height, word_polys)
        mask = torch.unsqueeze(mask, 0)  # Add channel dimension
        border_mask = torch.unsqueeze(border_mask, 0)  # Add channel dimension

        if self.transform:
            # Input and target are transformed in one call to ensure same
            # parameters are used for both, if transform is randomized.
            transformed = self.transform(torch.stack([img, mask, border_mask]))
            img = transformed[0]
            mask = transformed[1]
            border_mask = transformed[2]

        return {
            "path": img_path,
            "image": img,
            "text_mask": mask,
            "border_mask": border_mask,
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "dataset_type", choices=["ddi", "hiertext"], help="Type of dataset to load"
    )
    parser.add_argument("root_dir", help="Root directory of dataset")
    parser.add_argument(
        "--max-images", type=int, help="Maximum number of images to process"
    )
    args = parser.parse_args()

    load_dataset: Callable[..., DDI100 | HierText]
    if args.dataset_type == "ddi":
        load_dataset = DDI100
    elif args.dataset_type == "hiertext":
        load_dataset = HierText
    else:
        raise Exception(f"Unknown dataset type {args.dataset_type}")

    dataset = load_dataset(args.root_dir, train=False, max_images=args.max_images)

    print(f"Dataset length {len(dataset)}")

    for i in range(len(dataset)):
        # This loop doesn't use `enumerate` due to mypy error.
        item = dataset[i]

        print(f"Processing image {i+1}...")

        img = item["image"]
        mask = item["text_mask"]

        grey_img = untransform_image(img)
        rgb_img = grey_img.expand((3, grey_img.shape[1], grey_img.shape[2]))
        mask_hw = mask[0] > 0.5

        seg_img = draw_segmentation_masks(rgb_img, mask_hw, alpha=0.3, colors="red")
        write_png(seg_img, f"seg-{i}.png")
