"""Tests for the key_results regex heuristic."""

from __future__ import annotations

from packages.shared.groundtruth.key_results import extract


def test_empty_input_returns_empty() -> None:
    assert extract(None) == []
    assert extract("") == []


def test_picks_percentage_improvements() -> None:
    text = (
        "We start with a brief overview. "
        "Our method achieves 92.3% accuracy on the held-out set, "
        "outperforming the prior best by 4.1 points. "
        "No numbers in this sentence at all."
    )
    out = extract(text)
    assert any("92.3%" in s for s in out)


def test_picks_sota_mentions() -> None:
    text = "Our model achieves state-of-the-art performance on three benchmarks."
    out = extract(text)
    assert len(out) == 1


def test_respects_max_n() -> None:
    text = " ".join(
        f"On benchmark number {i}, our proposed model achieves {80 + i}% top-1 accuracy."
        for i in range(10)
    )
    assert len(extract(text, max_n=3)) == 3
