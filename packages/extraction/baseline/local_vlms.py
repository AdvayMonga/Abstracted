"""Stubs for local VLM baselines.

These are intentional placeholders. Wrappers are registered so the eval CLI
recognizes the names; calling them raises with installation instructions
instead of silently downloading multi-GB weights.

Fill in when we move to a GPU box (Phase 4 setup).
"""

from __future__ import annotations

from pathlib import Path

from packages.extraction.schema import Paper

_NOT_READY = (
    "{name} extractor is a stub. Implement render-and-prompt loop and install "
    "weights before running. Skipped here to avoid a multi-GB download on a "
    "machine without a GPU."
)


def _stub(name: str) -> Paper:
    raise NotImplementedError(_NOT_READY.format(name=name))


def qwen25_vl_3b(_pdf_path: Path) -> Paper:
    return _stub("qwen2.5-vl-3b")


def qwen25_vl_7b(_pdf_path: Path) -> Paper:
    return _stub("qwen2.5-vl-7b")


def nanonets_ocr_s(_pdf_path: Path) -> Paper:
    return _stub("nanonets-ocr-s")


def donut(_pdf_path: Path) -> Paper:
    return _stub("donut")
