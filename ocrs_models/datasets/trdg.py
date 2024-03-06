import importlib
import os.path
import random
import sys
from typing import Callable, Optional

import torch
from torch import Tensor
from torchvision.transforms.functional import (
    invert,
    pil_to_tensor,
    resize,
    to_grayscale,
)
import trdg
from trdg.data_generator import FakeTextDataGenerator
from trdg.string_generator import create_strings_from_dict
from trdg.utils import load_dict, load_fonts

from .util import (
    DEFAULT_ALPHABET,
    REC_INPUT_HEIGHT,
    SizedDataset,
    TextRecSample,
    encode_text,
    transform_image,
)


def load_trdg_dict(filename: str) -> list[str]:
    """
    Load a list of words from one of the dictionaries in the `trdg` package.

    :param filename: Name of the dict file inside trdg's `dicts/` dir
    """
    trdg_files = importlib.resources.files(trdg)
    dict_path = trdg_files.joinpath("dicts").joinpath(filename)
    return load_dict(dict_path)


ALL_CAPS_LATIN_FONTS = [
    "Amatic",
    "BEBAS",
    "Capture",
    "SEASRN",
]
"""Latin fonts supplied with trdg which use all-uppercase letters."""


def _is_uppercase_font(path: str) -> bool:
    """Return true if the font with a given path is known to be all-caps."""

    filename, _ext = os.path.splitext(os.path.basename(path))

    for prefix in ALL_CAPS_LATIN_FONTS:
        if filename.startswith(prefix):
            return True

    return False


def gen_random_strings(alphabet: str, count: int, max_words: int) -> list[str]:
    """
    Generate strings of words with random characters.

    :param alphabet: Alphabet to draw from
    :param count: Number of strings to generate
    :param max_words: Max words per string
    """
    max_word_len = 10
    strings: list[str] = []
    alphabet_chars = [ch for ch in alphabet if not ch.isspace()]

    for _ in range(count):
        n_words = random.randint(1, max_words)
        words: list[str] = []
        for _ in range(n_words):
            word_len = random.randint(1, max_word_len)
            word = ""
            for _ in range(word_len):
                char_idx = random.randrange(len(alphabet_chars))
                word += alphabet_chars[char_idx]
            words.append(word)
        strings.append(" ".join(words))
    return strings


TensorTransform = Callable[[Tensor], Tensor]


class TRDGRecognition(SizedDataset):
    """
    Synthetic image generator for text recognition using TextRecognitionDataGenerator.

    License: MIT

    [1] https://github.com/Belval/TextRecognitionDataGenerator
    """

    image_count: int
    """
    Number of images to generate.
    """

    fonts: list[str]
    """Paths of font files to use when generating text."""

    transform: Optional[TensorTransform]
    """
    Additional data augmentations to apply.

    These augmentations are applied after the built-in randomization of the
    synthetic text generator.
    """

    def __init__(
        self,
        image_count: int,
        max_words=10,
        transform: Optional[TensorTransform] = None,
    ):
        super().__init__()

        self.alphabet = [c for c in DEFAULT_ALPHABET]
        self.image_count = image_count
        self.transform = transform

        # See `fonts` directory in trdg package for names of supported font
        # collections.
        lang = "latin"
        self.fonts = load_fonts(lang)

        lang_dict = load_trdg_dict("en.txt")

        # Generate `image_count` strings from a mixture of sources.
        n_random_strings = 0
        n_dict_strings = 0
        for _ in range(image_count):
            x = random.random()
            if x > 0.8:
                n_random_strings += 1
            else:
                n_dict_strings += 1

        self.strings = []
        self.strings += create_strings_from_dict(
            max_words, allow_variable=True, count=n_dict_strings, lang_dict=lang_dict
        )
        self.strings += gen_random_strings(
            "".join(self.alphabet), count=n_random_strings, max_words=max_words
        )
        random.shuffle(self.strings)

    def __len__(self):
        return self.image_count

    def __getitem__(self, image_index: int) -> TextRecSample:
        text = self.strings[image_index]
        font = self.fonts[image_index % len(self.fonts)]

        # If the font uses all-caps, it is unknown which case the original
        # letters used. Resolve the ambiguity by making all target letters
        # uppercase too.
        if _is_uppercase_font(font):
            text = text.upper()

        # Randomized scaling factor for standard word spacing.
        word_space_scale = (torch.rand(()).item() * 2) + 0.5

        # Char spacing in pixels
        char_space = int(torch.rand(()).item() * 10)

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
            "font": font,
            # If `out_dir` is None, `generate` returns a PIL Image, otherwise
            # it writes the image to a file.
            "out_dir": None,
            "size": REC_INPUT_HEIGHT,
            "extension": "jpg",  # Unused
            # Max skew angle in degrees. Even though this is small, it ends
            # up being quite signficant for long lines.
            "skewing_angle": 2,
            "random_skew": True,  # Randomize skew in +/- `skewing_angle`
            # Max random blur. This is about the maximum value that can be
            # used for (almost) all samples to remain human readable.
            "blur": 2,
            "random_blur": True,  # Randomize blur between 0 and `blur`
            "background_type": 0,  # Gaussian noise
            # nb. Misspelling of "distortion" is intentional here.
            "distorsion_type": 3,  # Random distortion (sine wave, cosine wave, none)
            "distorsion_orientation": 0,
            "is_handwritten": False,
            "name_format": 0,  # Filename format. Unused.
            "width": -1,  # -1 means "adjust to width of text"
            "alignment": 0,  # 0 means left-align
            "text_color": "black",
            "orientation": 0,  # 0 means horizontal, 1 means vertical
            "space_width": word_space_scale,  # Scaling factor for normal width of words
            "character_spacing": char_space,  # Width of spaces between chars in pixels
            "margins": (0, 0, 0, 0),  # Typed as `int`, but actually `tuple(int)`
            "fit": True,  # Whether to apply tight crop around text
            "output_mask": False,  # Whether to return masks for text.
            "word_split": False,
            # Directory for background images. Only used if
            # `background_type` specifies images.
            "image_dir": None,
        }
        pil_image = FakeTextDataGenerator.generate(**gen_args)

        if not pil_image:
            print("Failed to generate image with args", gen_args, file=sys.stderr)

        pil_image = to_grayscale(pil_image)
        image = pil_to_tensor(pil_image)

        # Make a fraction of generations light text on dark backgrounds.
        # This must be applied before pixel values are shifted from [0, 255] to
        # [-0.5, 0.5].
        invert_prob = 0.2
        if torch.rand(()) > (1 - invert_prob):
            image = invert(image)

        # Shift pixel values from [0, 255] to [-0.5, 0.5]
        image = transform_image(image)

        if self.transform:
            image = self.transform(image)

            # Brightness / contrast transforms may have moved pixel values
            # outside of the legal range of [-0.5, 0.5] for normalized images.
            #
            # Clamp to bring these values back into that range.
            image = image.clamp(-0.5, 0.5)

        # Data augmentations may alter the size of the image. Resize back to
        # `REC_INPUT_HEIGHT`, as all images in a batch need to have the same
        # height. Widths can vary, the caller will pad images.
        _chans, height, width = image.shape
        if height != REC_INPUT_HEIGHT:
            image = resize(image, [REC_INPUT_HEIGHT, width], antialias=True)

        text_seq = encode_text(text, self.alphabet, unknown_char="?")
        sample: TextRecSample = {
            "image_id": image_index,
            "image": image,
            "text_seq": text_seq,
        }
        return sample
