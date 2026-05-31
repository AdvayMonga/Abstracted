"""obsidian_vault MCP server.

The agent drops Markdown notes into an Obsidian vault directory. Files are
plain `<slug>.md` with optional YAML frontmatter. Vault location:
`ABSTRACTED_VAULT_DIR` env var, defaulting to `data/vault` under the repo.

Tools:
  write_note(slug, body, frontmatter) -> path
  read_note(slug) -> {frontmatter, body}
  append_to_note(slug, text) -> path
  list_notes() -> [slug, ...]
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("obsidian_vault")

_SLUG_OK = re.compile(r"[^A-Za-z0-9._\- ]")
_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _vault() -> Path:
    d = Path(os.environ.get("ABSTRACTED_VAULT_DIR", "data/vault"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sanitize(slug: str) -> str:
    s = _SLUG_OK.sub("", slug).strip()
    if not s or s.startswith("."):
        raise ValueError(f"Invalid slug: {slug!r}")
    return s


def _path(slug: str) -> Path:
    return _vault() / f"{_sanitize(slug)}.md"


def _render(frontmatter: dict | None, body: str) -> str:
    if not frontmatter:
        return body
    fm = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).rstrip()
    return f"---\n{fm}\n---\n{body}"


@mcp.tool()
def write_note(slug: str, body: str, frontmatter: dict | None = None) -> str:
    """Create or overwrite a note in the vault. Returns the path written.

    Args:
        slug: filename without extension. Alphanumerics, dots, underscores,
            hyphens, and spaces only.
        body: Markdown body, written after the frontmatter.
        frontmatter: optional dict serialized as YAML frontmatter.
    """
    path = _path(slug)
    path.write_text(_render(frontmatter, body))
    return str(path)


@mcp.tool()
def read_note(slug: str) -> dict[str, Any]:
    """Read a note. Returns {"frontmatter": dict|None, "body": str}.

    Args:
        slug: filename without extension.
    """
    path = _path(slug)
    if not path.exists():
        raise FileNotFoundError(f"No note named {slug!r} in vault")
    raw = path.read_text()
    m = _FRONTMATTER.match(raw)
    if not m:
        return {"frontmatter": None, "body": raw}
    fm = yaml.safe_load(m.group(1)) or {}
    body = raw[m.end():]
    return {"frontmatter": fm, "body": body}


@mcp.tool()
def append_to_note(slug: str, text: str) -> str:
    """Append text to an existing note (or create one if absent). Returns the path.

    Args:
        slug: filename without extension.
        text: text to append. A leading newline is inserted if needed.
    """
    path = _path(slug)
    if path.exists() and not path.read_text().endswith("\n"):
        text = "\n" + text
    with path.open("a") as f:
        f.write(text)
    return str(path)


@mcp.tool()
def list_notes() -> list[str]:
    """Return all note slugs (filenames without `.md`) currently in the vault."""
    return sorted(p.stem for p in _vault().glob("*.md"))
