"""Map model name → extractor callable."""

from collections.abc import Callable
from pathlib import Path

from packages.extraction.baseline.claude_sonnet import extract as claude_sonnet_extract
from packages.extraction.baseline.local_vlms import (
    donut,
    nanonets_ocr_s,
    qwen25_vl_3b,
    qwen25_vl_7b,
)
from packages.extraction.baseline.pymupdf_regex import extract as pymupdf_regex_extract
from packages.extraction.schema import Paper

Extractor = Callable[[Path], Paper]

REGISTRY: dict[str, Extractor] = {
    "pymupdf_regex": pymupdf_regex_extract,
    "claude_sonnet": claude_sonnet_extract,
    "qwen2.5-vl-3b": qwen25_vl_3b,
    "qwen2.5-vl-7b": qwen25_vl_7b,
    "nanonets-ocr-s": nanonets_ocr_s,
    "donut": donut,
}


def get(name: str) -> Extractor:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
