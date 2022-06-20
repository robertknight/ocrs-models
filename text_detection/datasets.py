from argparse import ArgumentParser
import os
import pickle
import pickletools

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


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

    def __init__(self, root_dir: str, train=True, max_images=None):
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

    def __len__(self):
        return len(self._img_filenames)

    def __getitem__(self, idx: int):
        """
        Return tuple of (document image, binary_mask) tensors.

        The document image is an NCHW tensor with one greyscale color channel.
        The binary mask is an NCHW tensor.
        """

        img_fname = self._img_filenames[idx]
        img_basename, _ = os.path.splitext(img_fname)
        img_path = f"{self._img_dir}/{self._img_filenames[idx]}"

        # Read image as floats in [-0.5, 0.5]
        img = (read_image(img_path).float() / 255.0) - 0.5

        # See https://github.com/machine-intelligence-laboratory/DDI-100/tree/master/dataset
        # for details of dataset structure.
        pickle_path = f"{self._boxes_dir}/{img_basename}.pickle"

        with open(pickle_path, "rb") as pickle:
            words = DDI100Unpickler(pickle).load()
            word_quads = [w["box"] for w in words]

        _, height, width = img.shape
        return img_path, img, self._generate_mask(width, height, word_quads)

    @staticmethod
    def _generate_mask(width: int, height: int, word_quads):
        mask_img = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        for quad in word_quads:
            # Re-order the corners in the polygon and order of (x, y)
            # coordinates to get the format that PIL expects.
            coords = [(coord[1], coord[0]) for coord in quad.tolist()]
            bottom_left, top_left, bottom_right, top_right = coords
            draw.polygon(
                [top_left, top_right, bottom_right, bottom_left],
                fill="white",
                outline="white",
                width=1,
            )

        mask = torch.Tensor(np.array(mask_img))
        mask = torch.unsqueeze(mask, 0)  # Add channel dimension
        return mask


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("root_dir", help="Root directory of dataset")
    args = parser.parse_args()

    dataset = DDI100(args.root_dir)
    print(f"Dataset length {len(dataset)}")

    for i in range(len(dataset)):
        print(f"Processing image {i+1}...")
        img, mask = dataset[i]
        print("Img shape", img.shape, "mask shape", mask.shape)
