"""Unit tests for the deterministic graders."""

from __future__ import annotations

from pathlib import Path

from packages.agent.eval.graders import grade
from packages.agent.state import AgentState
from packages.extraction.schema import ExtractedField, Paper


def _good_state(note_path: Path) -> AgentState:
    return AgentState(
        arxiv_id="2506.11419",
        pdf_path=Path("/tmp/x.pdf"),
        paper=Paper(title=ExtractedField(value="X", confidence=0.9)),
        note_path=str(note_path),
        related_papers=[{"title": "Related"}],
    )


def test_all_pass_on_good_note(tmp_path) -> None:
    note = tmp_path / "n.md"
    note.write_text(
        "---\narxiv_id: 2506.11419\ntitle: X\n---\n"
        "# X\n## Abstract\n\nbody\n## Related work\n- foo\n"
    )
    scores = grade(_good_state(note), tmp_path)
    assert all(scores.values()), scores


def test_missing_note_fails_most(tmp_path) -> None:
    state = AgentState(arxiv_id="x", pdf_path=Path("/tmp/x.pdf"))
    scores = grade(state, tmp_path)
    assert scores["note_written"] is False
    assert scores["has_frontmatter"] is False
    assert scores["has_expected_sections"] is False
    # pipeline_completed and no_errors should still be True (no escalation, no errors)
    assert scores["pipeline_completed"] is True


def test_escalation_fails_pipeline_completed(tmp_path) -> None:
    state = AgentState(
        arxiv_id="x",
        pdf_path=Path("/tmp/x.pdf"),
        review_decision="reject",
        review_item_id=1,
    )
    scores = grade(state, tmp_path)
    assert scores["pipeline_completed"] is False
