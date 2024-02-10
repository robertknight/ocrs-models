import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.functional import to_pil_image

from shapely.geometry import JOIN_STYLE
from shapely.geometry.polygon import LinearRing, Polygon


def extract_cc_quads(mask: torch.Tensor) -> torch.Tensor:
    """
    Extract bounding quads of connected components in a segmentation mask.

    Returns an Nx4x2 tensor, where N is the number of connected components
    in the mask, the second dimension is the vertex index, and the last dimension
    contains the X and Y coordinates for the vertex.

    :param mask: Greyscale mask (HxW or 1xHxW) indicating text regions in the image
    """

    # Strip channel dimension.
    if len(mask.shape) > 2:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise ValueError("Expected mask to be an HxW or 1xHxW tensor")

    mask = mask.to(torch.uint8)
    contours, _ = cv2.findContours(
        mask.numpy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    quads = np.array(
        [cv2.boxPoints(cv2.minAreaRect(contour[:, 0])) for contour in contours]
    )
    return torch.Tensor(quads)


def expand_quad(quad: torch.Tensor, dist: float) -> torch.Tensor:
    """
    Construct an enlarged version of `quad`.

    In the returned quad, each edge will be at an (approximate) offset of `dist`
    from the corresponding edge in the input.

    :param quad: 4x2 tensor
    :param dist: Units to offset quad by
    """
    ring = LinearRing(quad.tolist())

    # If the input quad is a point, it can't be offset.
    if ring.length == 0.0:
        return quad

    # The offset side needed to enlarge the input depends on the orientation of
    # the points.
    side = "right" if ring.is_ccw else "left"
    expanded_rect = ring.parallel_offset(
        dist, side, join_style=JOIN_STYLE.mitre
    ).minimum_rotated_rectangle

    # expanded_rect has 5 vertices, where the first and last are the same.
    quad_verts = list(expanded_rect.exterior.coords)[:-1]

    return torch.tensor(quad_verts)


def expand_quads(quads: torch.Tensor, dist: float) -> torch.Tensor:
    """
    Expand/dilate each quad in a list.

    :param quads: Nx4x2 tensor of quads. See `extract_cc_quads`
    :param dist: Number of pixels by which to offset each edge in each quad
    :return: Tensor of same shape as `quads`
    """
    return torch.stack([expand_quad(quad, dist) for quad in quads])


def lines_intersect(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """
    Return true if the lines (a_start, a_end) and (b_start, b_end) intersect.
    """
    if a_start <= b_start:
        return a_end > b_start
    else:
        return b_end > a_start


def bounds_intersect(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    """
    Return true if the rects defined by two (min_x, min_y, max_x, max_y) tuples intersect.
    """
    a_min_x, a_min_y, a_max_x, a_max_y = a
    b_min_x, b_min_y, b_max_x, b_max_y = b
    return lines_intersect(a_min_x, a_max_x, b_min_x, b_max_x) and lines_intersect(
        a_min_y, a_max_y, b_min_y, b_max_y
    )


def box_match_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """
    Compute metrics for quality of matches between two sets of rotated rects.

    :param pred: Nx4x2 tensor of (box index, vertex index, coord index)
    :param target: Same as `pred`
    """

    # Map of (pred index, target index) for targets with a "good" match
    matches: dict[int, int] = {}

    pred_polys = [Polygon(p.tolist()) for p in pred]
    target_polys = [Polygon(t.tolist()) for t in target]

    # Areas of intersections of predictions and targets
    intersection = torch.zeros((len(pred), len(target)))

    # Areas of unions of predictions and targets
    union = torch.zeros((len(pred), len(target)))

    # Get bounding boxes of polys for a cheap intersection test.
    pred_polys_bounds = [poly.bounds for poly in pred_polys]
    target_polys_bounds = [poly.bounds for poly in target_polys]

    pred_areas = torch.zeros((len(pred),))
    for pred_index, pred_poly in enumerate(pred_polys):
        pred_areas[pred_index] = pred_poly.area
        pred_bounds = pred_polys_bounds[pred_index]

        for target_index, target_poly in enumerate(target_polys):
            # Do a cheap intersection test and skip computing the actual
            # union/intersection if that fails.
            target_bounds = target_polys_bounds[target_index]
            if not bounds_intersect(pred_bounds, target_bounds):
                continue

            pt_intersection = pred_poly.intersection(target_poly)
            intersection[pred_index, target_index] = pt_intersection.area

            pt_union = pred_poly.union(target_poly)
            union[pred_index, target_index] = pt_union.area

    target_areas = torch.zeros((len(target),))
    for target_index, target_poly in enumerate(target_polys):
        target_areas[target_index] = target_poly.area

    # Find (pred_index, target_index) pairs of "good" matches
    iou = intersection / union
    good_match_threshold = 0.5
    good_match_indexes = torch.nonzero(iou > good_match_threshold)
    for match_ in good_match_indexes:
        pred_index, target_index = match_.tolist()
        matches[pred_index] = target_index

    # Find target boxes that got merged together in the predictions.
    merged_boxes = 0
    for pred_index in range(len(pred_polys)):
        covered_targets = len(
            torch.nonzero((intersection[pred_index] / target_areas) > 0.5)
        )
        if covered_targets > 1:
            merged_boxes += covered_targets

    # Find target boxes that got split up in the predictions.
    split_boxes = 0
    for target_index in range(len(target_polys)):
        covered_preds = len(
            torch.nonzero((intersection[:, target_index] / pred_areas) > 0.5)
        )
        if covered_preds > 1:
            split_boxes += 1

    metrics = {
        # Proportion of predictions that are good matches
        "precision": len(matches) / len(pred) if len(pred) > 0 else 1.0,
        # Proportion of expected matches that have a good matching prediction
        "recall": len(matches) / len(target) if len(target) > 0 else 1.0,
        # Fraction of target boxes that were merged with one or more other targets
        # in the predictions.
        "merged_frac": merged_boxes / len(target) if len(target) > 0 else 0.0,
        # Fraction of target boxes that were split into two or more boxes in the
        # predictions
        "split_frac": split_boxes / len(target) if len(target) > 0 else 0.0,
    }

    return metrics


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
