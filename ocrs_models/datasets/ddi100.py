import os
import pickle

import numpy as np
import torch
from torchvision.io import read_image

from .util import SizedDataset, generate_mask, transform_image


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


class DDI100(SizedDataset):
    """
    Distorted Document Images (DDI-100) dataset for text detection.

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
