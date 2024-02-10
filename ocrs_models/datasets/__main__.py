from argparse import ArgumentParser, BooleanOptionalAction
from typing import Callable, cast
import os

from torchvision.io import write_png
from torchvision.utils import draw_segmentation_masks

from . import text_recognition_data_augmentations
from .ddi100 import DDI100
from .hiertext import HierText, HierTextRecognition
from .util import SizedDataset, untransform_image, decode_text, draw_word_boxes
from .web_layout import WebLayout

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
parser.add_argument("--filter", type=str, help="Filter input data items by file path")
parser.add_argument("--max-images", type=int, help="Maximum number of items to process")
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


dataset: SizedDataset
match args.dataset_type:
    case "ddi" | "hiertext":
        # Explicitly cast dataset constructors to a common type to avoid mypy error.
        DatasetConstructor = Callable[..., SizedDataset]
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

            seg_img = draw_segmentation_masks(rgb_img, mask_hw, alpha=0.3, colors="red")
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
            line_img_path = f"{args.out_dir}/line-{i}-{image_id}-{text_path_safe}.png"

            write_png(untransform_image(img), line_img_path)
    case "web-layout":
        dataset = WebLayout(
            args.root_dir,
            train=args.subset == "train",
            max_images=args.max_images,
            filter=filter_item,
            randomize=args.augment,
        )

        for i in range(len(dataset)):
            word_boxes, labels = dataset[i]
            out_img = f"{args.out_dir}/img-{i}.png"
            width = 1024
            height = 768
            draw_word_boxes(out_img, width, height, word_boxes, labels)
