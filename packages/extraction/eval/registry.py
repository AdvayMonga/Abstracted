"""Map model name → extractor callable."""

from collections.abc import Callable
from pathlib import Path

from packages.extraction.baseline.pymupdf_regex import extract as pymupdf_regex_extract
from packages.extraction.schema import Paper

Extractor = Callable[[Path], Paper]

REGISTRY: dict[str, Extractor] = {
    "pymupdf_regex": pymupdf_regex_extract,
}


def get(name: str) -> Extractor:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
