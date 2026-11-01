import gzip
import json
import os
import random
from typing import cast

import torch
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image, write_png
from torchvision.transforms.functional import resize
from tqdm import tqdm

from .util import (
    Polygon,
    SizedDataset,
    bounding_box_size,
    clamp,
    encode_text,
    generate_mask,
    generate_mask_u8,
    transform_image,
)


class HierText(SizedDataset):
    """
    HierText dataset for text detection.

    See https://github.com/google-research-datasets/hiertext and
    https://arxiv.org/abs/2203.15143.

    Photos from the Open Images dataset [1], containing text in natural scenes
    and documents, annotated with paragraph, line and word-level polygons and
    bounding boxes.

    License: CC BY-SA 4.0

    [1] https://storage.googleapis.com/openimages/web/index.html
    """

    def __init__(
        self,
        root_dir: str,
        train=True,
        max_images=None,
        size: tuple[int, int] | None = None,
        augment=False,
    ):
        """
        :param size: (height, width) to resize images and masks to. If `None`,
            items are returned at their original size.
        :param augment: Whether to apply randomized data augmentation. Requires
            `size` to be set.
        """
        if augment and size is None:
            raise ValueError("Data augmentation requires `size` to be set")

        if train:
            self._img_dir = f"{root_dir}/train"
            annotations_file = f"{root_dir}/gt/train.jsonl.gz"
        else:
            self._img_dir = f"{root_dir}/validation"
            annotations_file = f"{root_dir}/gt/validation.jsonl.gz"

        if not os.path.exists(self._img_dir):
            raise FileNotFoundError(f'Image directory "{self._img_dir}" not found')

        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f'Label data file "{annotations_file}" not found')

        lines_file = annotations_file.replace(".jsonl.gz", ".jsonl")
        self._generate_json_lines_annotations(annotations_file, lines_file)

        with open(lines_file) as fp:
            self._annotations = [line for line in fp]

        if max_images:
            self._annotations = self._annotations[:max_images]

        self.size = size
        self.augment = augment
        self._color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
        self._geometric_augmentations = [
            transforms.RandomAffine(degrees=5, scale=(0.8, 1.2), shear=5),
            transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
        ]

    def __len__(self):
        return len(self._annotations)

    def _augment(
        self, img: torch.Tensor, mask: torch.Tensor, orig_height: int, orig_width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a randomized augmentation to an image and its mask.

        Half of images get one augmentation. Geometric augmentations are applied
        to both the image and mask. Photometric augmentations are applied to
        the image only.

        :param img: CHW image, resized to `self.size`
        :param mask: CHW mask, resized to `self.size`
        :param orig_height: Height of the image before resizing
        :param orig_width: Width of the image before resizing
        """
        assert self.size is not None

        if random.random() < 0.5:
            choice = random.randrange(4)
            if choice == 0:
                # Shift the image into the [0, 1] range that ColorJitter
                # expects.
                img = self._color_jitter(img + 0.5) - 0.5
            elif choice < 3:
                # Transform image and mask in one call so that both use the
                # same randomized parameters.
                transformed = self._geometric_augmentations[choice - 1](
                    torch.stack([img, mask])
                )
                img, mask = transformed[0], transformed[1]
            else:
                # Crop a 600x600 region of the original image, scaled to the
                # resized image's coordinates, then resize back to `self.size`.
                height, width = self.size
                crop_size = 600
                crop = transforms.RandomCrop(
                    size=(
                        min(round(crop_size * height / orig_height), 2 * height),
                        min(round(crop_size * width / orig_width), 2 * width),
                    ),
                    pad_if_needed=True,
                )
                cropped = resize(
                    crop(torch.stack([img, mask])), list(self.size), antialias=False
                )
                img, mask = cropped[0], cropped[1]

        return img, mask

    def __getitem__(self, idx: int):
        """
        Return dict containing document image and binary text mask tensors.

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

        img_u8 = read_image(img_path, ImageReadMode.GRAY)
        _, height, width = img_u8.shape

        mask_u8 = generate_mask_u8(width, height, word_polys)
        mask_u8 = torch.unsqueeze(mask_u8, 0)  # Add channel dimension

        # Resize before converting to float and augmenting, as doing those at
        # the original resolution is much slower.
        img_and_mask = torch.cat([img_u8, mask_u8])
        if self.size is not None:
            img_and_mask = resize(img_and_mask, list(self.size), antialias=False)
        img = transform_image(img_and_mask[0:1])
        mask = img_and_mask[1:2].float() / 255.0

        if self.augment:
            img, mask = self._augment(img, mask, height, width)

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


class HierTextRecognition(SizedDataset):
    """
    HierText dataset for text recognition.

    See the `HierText` class for general notes on the HierText dataset. This
    class yields greyscale images of text lines from the dataset, along with
    the associated text content as a one-hot encoded sequence of class labels.

    License: CC BY-SA 4.0
    """

    def __init__(
        self,
        root_dir: str,
        train=True,
        transform=None,
        max_images=None,
        alphabet: list[str] | None = None,
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
            raise FileNotFoundError(f'Image directory "{self._img_dir}" not found')

        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f'Label data file "{annotations_file}" not found')

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
                f"Shape mismatch {line_img.shape} vs {mask.shape} line height {line_height} min y {min_y} max y {max_y}"
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
