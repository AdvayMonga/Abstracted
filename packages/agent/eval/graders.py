"""Deterministic graders for an agent run.

Each grader takes (final_state, vault_dir) and returns a bool. The harness
aggregates pass-rates across documents.
"""

from __future__ import annotations

from pathlib import Path

from packages.agent.state import AgentState

# Sections we expect in any agent-generated note.
EXPECTED_SECTIONS = ("## Abstract", "## Related work")
REQUIRED_FRONTMATTER_KEYS = ("arxiv_id", "title")


def note_written(state: AgentState, _vault: Path) -> bool:
    return state.note_path is not None and Path(state.note_path).exists()


def pipeline_completed(state: AgentState, _vault: Path) -> bool:
    """The agent went all the way through, didn't escalate to the review queue."""
    return state.review_decision is None and state.review_item_id is None


def no_errors(state: AgentState, _vault: Path) -> bool:
    return state.errors == []


def has_frontmatter(state: AgentState, _vault: Path) -> bool:
    if state.note_path is None:
        return False
    raw = Path(state.note_path).read_text()
    if not raw.startswith("---\n"):
        return False
    end = raw.find("\n---\n", 4)
    if end == -1:
        return False
    block = raw[4:end]
    return all(f"{k}:" in block for k in REQUIRED_FRONTMATTER_KEYS)


def has_expected_sections(state: AgentState, _vault: Path) -> bool:
    if state.note_path is None:
        return False
    body = Path(state.note_path).read_text()
    return all(section in body for section in EXPECTED_SECTIONS)


def found_related_work(state: AgentState, _vault: Path) -> bool:
    return len(state.related_papers) >= 1


GRADERS = {
    "note_written": note_written,
    "pipeline_completed": pipeline_completed,
    "no_errors": no_errors,
    "has_frontmatter": has_frontmatter,
    "has_expected_sections": has_expected_sections,
    "found_related_work": found_related_work,
}


def grade(state: AgentState, vault: Path) -> dict[str, bool]:
    return {name: fn(state, vault) for name, fn in GRADERS.items()}
