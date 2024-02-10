import json
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .util import SizedDataset, intervals_overlap


class WebLayout(SizedDataset):
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
        normalize_coords=True,
        max_jitter: int = 25,
    ):
        """
        Construct dataset from JSON files in `root_dir`.

        :param root_dir: Directory to load dataset from
        :param filter: Filter images by file path
        :param normalize_coords: Normalize coordinates to be in the range (-0.5, 0.5)
        :param max_images: Maximum number of images to load from dataset
        :param max_jitter:
            Maximum amount of random translation to apply, only used if `randomize`
            is true.
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

        self.max_jitter = max_jitter
        self.normalize_coords = normalize_coords
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
        labels_list = []
        in_path = os.path.join(self.root_dir, self._files[idx])

        if self.randomize:
            a, b, c = torch.rand(3).tolist()
            jitter_x = a * self.max_jitter
            jitter_y = b * self.max_jitter
            scale = 1.0
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

                assert left >= 0 and right >= 0 and top >= 0 and bottom >= 0

                if self.normalize_coords:
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

                    labels_list.append([int(line_start), int(line_end)])

        input_ = torch.Tensor(words)
        labels = torch.Tensor(labels_list)

        if self.padded_size:
            pad_len = self.padded_size - input_.shape[0]
            if pad_len > 0:
                input_ = F.pad(input_, (0, 0, 0, pad_len))
                labels = F.pad(labels, (0, 0, 0, pad_len))
            else:
                input_ = input_[0 : self.padded_size]
                labels = labels[0 : self.padded_size]

        return (input_, labels)
