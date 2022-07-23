from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.functional import to_pil_image


def extract_cc_quads(mask: torch.Tensor) -> torch.Tensor:
    """
    Extract bounding quads of connected components in a segmentation mask.

    Returns an Nx4x2 tensor, where N is the number of connected components
    in the mask, the second dimension is the vertex index, and the last dimension
    contains the X and Y coordinates for the vertex.

    :param mask: Greyscale mask indicating text regions in the image
    """

    mask = mask.to(torch.uint8)
    contours, _ = cv2.findContours(
        mask.numpy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    quads = np.array(
        [cv2.boxPoints(cv2.minAreaRect(contour[:, 0])) for contour in contours]
    )
    return torch.Tensor(quads)


def draw_quads(img: torch.Tensor, quads: torch.Tensor) -> Image.Image:
    """
    Draw outlines of words on a mask or image.

    Returns a copy of `img` with outlines specified by `quads` drawn on top.

    :param img: Greyscale image
    :param quads: Nx4x2 tensor of quads. See `extract_cc_quads`
    """
    out_img = to_pil_image(img).convert("RGB")
    draw = ImageDraw.Draw(out_img)

    for quad in quads:
        verts = [(v[0].item(), v[1].item()) for v in quad]

        # Draw quad edge-by-edge because `draw.polygon` is really slow when
        # there is no fill and outline width > 1.
        for i, start in enumerate(verts):
            end = verts[i + 1] if i < len(verts) - 1 else verts[0]
            draw.line((start, end), fill="red", width=2)

    return out_img
