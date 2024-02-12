import importlib
from typing import Generator, Optional

from torch.utils.data import IterableDataset
from torchvision.transforms.functional import pil_to_tensor, to_grayscale
import trdg
from trdg.data_generator import FakeTextDataGenerator
from trdg.string_generator import create_strings_from_dict
from trdg.utils import load_dict, load_fonts

from .util import (
    DEFAULT_ALPHABET,
    REC_INPUT_HEIGHT,
    TextRecSample,
    encode_text,
    transform_image,
)


class TRDGRecognition(IterableDataset):
    """
    Synthetic image generator for text recognition using TextRecognitionDataGenerator.

    License: MIT

    [1] https://github.com/Belval/TextRecognitionDataGenerator
    """

    max_images: Optional[int]
    """
    Maximum number of images to yield.

    If `None` this dataset will yield an infinite number of images.
    """

    fonts: list[str]
    """Paths of font files to use when generating text."""

    def __init__(self, max_images=None, max_length=None):
        self.alphabet = [c for c in DEFAULT_ALPHABET]
        self.max_images = max_images

        # See `fonts` directory in trdg package for names of supported font
        # collections.
        lang = "latin"
        self.fonts = load_fonts(lang)

        # Generate `string_count` strings with between 1 and `max_words` words,
        # each randomly chosen from a dictionary.
        max_words = 10
        string_count = max_images or 1024
        trdg_files = importlib.resources.files(trdg)
        dict_path = trdg_files.joinpath("dicts").joinpath("en.txt")
        lang_dict = load_dict(dict_path)
        self.strings = create_strings_from_dict(
            max_words, allow_variable=True, count=string_count, lang_dict=lang_dict
        )

    def __iter__(self) -> Generator[TextRecSample, None, None]:
        image_index = 0
        while self.max_images is None or image_index < self.max_images:
            text = self.strings[image_index % len(self.strings)]

            # Most of the arguments for FakeTextDataGenerator.generate in
            # the `trdg` tool come directly from CLI arguments. See
            # https://github.com/robertknight/TextRecognitionDataGenerator/blob/c1103c99d01b3181ddba50aa17bd880e3bc6f0bd/trdg/run.py#L440.
            #
            # For values that we don't want to customize, we use values that
            # correspond to the CLI defaults. We can't just omit arguments
            # because most don't have defaults.
            gen_args = {
                "index": image_index,
                "text": text,
                "font": self.fonts[image_index % len(self.fonts)],
                # If `out_dir` is None, `generate` returns a PIL Image, otherwise
                # it writes the image to a file.
                "out_dir": None,
                "size": REC_INPUT_HEIGHT,
                # Arguments with default values
                "extension": "jpg",
                "skewing_angle": 0,
                "random_skew": False,  # Randomize skew in +/- `skewing_angle`
                "blur": 0,  # Gaussian blur radius
                "random_blur": False,  # Randomize blur between 0 and `blur`
                "background_type": 0,
                # nb. Misspelling of "distortion" is intentional here.
                "distorsion_type": 0,
                "distorsion_orientation": 0,
                "is_handwritten": False,
                "name_format": 0,
                "width": -1,  # -1 means "adjust to width of text"
                "alignment": 0,  # 0 means left-align
                "text_color": "black",
                "orientation": 0,  # 0 means horizontal, 1 means vertical
                "space_width": 1.0,  # Scaling factor for normal width of words
                "character_spacing": 0,  # Width of spaces between chars in pixels
                "margins": (5, 5, 5, 5),  # Typed as `int`, but actually `tuple(int)`
                "fit": False,  # Whether to apply tight crop around text
                "output_mask": False,  # Whether to return masks for text.
                "word_split": False,
                "image_dir": None,
            }
            pil_image = FakeTextDataGenerator.generate(**gen_args)
            pil_image = to_grayscale(pil_image)
            image = pil_to_tensor(pil_image)
            image = transform_image(image)

            text_seq = encode_text(text, self.alphabet, unknown_char="?")
            sample: TextRecSample = {
                "image_id": image_index,
                "image": image,
                "text_seq": text_seq,
            }
            yield sample
            image_index += 1
