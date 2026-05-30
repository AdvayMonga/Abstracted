"""Smoke tests for the model registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.extraction.eval.registry import REGISTRY, get


def test_registry_contains_phase3_models() -> None:
    for name in [
        "pymupdf_regex",
        "claude_sonnet",
        "qwen2.5-vl-3b",
        "qwen2.5-vl-7b",
        "nanonets-ocr-s",
        "donut",
    ]:
        assert name in REGISTRY


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        get("not-a-model")


def test_local_stubs_raise_not_implemented() -> None:
    for name in ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "nanonets-ocr-s", "donut"]:
        with pytest.raises(NotImplementedError):
            get(name)(Path("/tmp/anything.pdf"))
