"""Dataset-ingestion text normalization helpers.

These helpers are used only by dataset adapters during ``dataset.download``
and ``dataset.manifest``. Pipeline-wide manifest and processing utilities
remain in ``signdata.utils``.
"""

import re

import ftfy
from pydantic import BaseModel


class TextProcessingConfig(BaseModel):
    """Text normalization options passed to ``normalize_text()``."""

    fix_encoding: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False
    strip_punctuation: bool = False


def normalize_text(
    text: str,
    *,
    fix_encoding: bool = True,
    normalize_whitespace: bool = True,
    lowercase: bool = False,
    strip_punctuation: bool = False,
) -> str:
    """Normalize text with configurable processing steps."""
    if fix_encoding:
        text = ftfy.fix_text(text)

    if normalize_whitespace:
        text = " ".join(text.split())

    if lowercase:
        text = text.lower()

    if strip_punctuation:
        text = re.sub(r"[^\w\s]", "", text)

    return text
