"""Detection validation utilities."""

from .base import Detection


def single_person_check(detections: list[list[Detection]]) -> bool:
    """Check that at most one person was detected in every frame.

    Returns True if all frames have 0 or 1 detections, meaning we can
    confidently assume a single signer.
    """
    return all(len(frame_dets) <= 1 for frame_dets in detections)


def union_bboxes(
    detections: list[list[Detection]],
) -> tuple[float, float, float, float] | None:
    """Compute the enclosing bounding box across all frames.

    Returns (x1, y1, x2, y2) covering all detections in pixel
    coordinates, or ``None`` if no frames contained any detections.
    Picks the largest-area detection per frame (most likely the signer).
    """
    all_bboxes = []
    for frame_dets in detections:
        if frame_dets:
            # Pick largest-area bbox per frame
            best = max(
                frame_dets,
                key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            )
            all_bboxes.append(best.bbox)

    if not all_bboxes:
        return None

    x1 = min(b[0] for b in all_bboxes)
    y1 = min(b[1] for b in all_bboxes)
    x2 = max(b[2] for b in all_bboxes)
    y2 = max(b[3] for b in all_bboxes)
    return (x1, y1, x2, y2)


def union_bbox_tuples(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Compute the enclosing bounding box from raw bbox tuples."""
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return (x1, y1, x2, y2)


def apply_bbox_padding(
    bbox: tuple[float, float, float, float],
    padding: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Apply padding to a bounding box and clamp to frame dimensions.

    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        padding: Padding ratio (0.25 = 25% of bbox dimensions).
        frame_width: Frame width for clamping.
        frame_height: Frame height for clamping.

    Returns:
        (x1, y1, x2, y2) as integers, clamped to frame bounds.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_x = w * padding
    pad_y = h * padding

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(frame_width, int(x2 + pad_x))
    y2 = min(frame_height, int(y2 + pad_y))

    return (x1, y1, x2, y2)
