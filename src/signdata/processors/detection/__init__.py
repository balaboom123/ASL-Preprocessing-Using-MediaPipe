"""Detection package exports."""

from .base import Detection, PersonDetector, create_detector
from .validation import apply_bbox_padding, single_person_check, union_bboxes, union_bbox_tuples

__all__ = [
    "Detection",
    "PersonDetector",
    "create_detector",
    "single_person_check",
    "union_bboxes",
    "apply_bbox_padding",
    "union_bbox_tuples",
]
