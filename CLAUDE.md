# CLAUDE.md

This file gives Claude Code the context it needs to work on this project. See `ROADMAP.md` for the full master plan.

---

## Project: Abstracted

A two-layer agentic document processing system that mirrors Nanonets' production stack, applied to research papers instead of invoices. Built so the author learns the technical machinery (VLM fine-tuning, agent orchestration, observability) on a domain they actually use.

### Architecture (4 layers)

1. **Extraction** — Fine-tuned Qwen2.5-VL-3B emitting JSON + bbox + per-field confidence. Served via vLLM behind FastAPI. Local GPU.
2. **Tools** — One MCP server per integration: `obsidian_vault`, `semantic_scholar`, `github_lookup`, `arxiv_fetch`, `review_queue`. Standalone processes.
3. **Agent** — LangGraph stateful graph with Postgres checkpointer. `interrupt()` for human-in-the-loop. Anthropic API (Claude Sonnet) for LLM calls inside nodes.
4. **Observability** — LangSmith traces, evals, prompt versioning, replay.

Storage: Postgres + pgvector. Queue: Redis. Invocation-based (not a daemon).

### End-to-end goal

Drop a PDF in a watched folder → within ~60s an Obsidian note appears with structured frontmatter, summary, related-paper backlinks, code-repo link, citation context. Low-confidence cases escalate to a review queue. Every action is replayable from a LangSmith trace.

### Phase plan (each phase ends with something working end-to-end)

- **Phase 0** — Foundations: read LayoutLMv3, Donut, Nougat, Qwen2.5-VL, Nanonets OCR-3, LangGraph/MCP/LangSmith docs. Write `NOTES.md`.
- **Phase 1** — Repo scaffold + eval harness. Define Pydantic extraction schema. Build trivial PyMuPDF+regex baseline. Measure first, build second.
- **Phase 2** — Dataset: collect ~150 papers, label in Label Studio, synthetic augmentation, DVC versioning, train/val/test/held-out splits.
- **Phase 3** — Off-the-shelf VLM baselines: Qwen2.5-VL-3B/7B, Nanonets-OCR-s, Donut, one frontier API model. Compute confidence from logprobs (not self-report).
- **Phase 4** — Fine-tune Qwen2.5-VL-3B via Unsloth (QLoRA). Hyperparameter sweep, ablations, temperature scaling for calibrated confidence.
- **Phase 5** — Serving: vLLM + FastAPI gateway (`/extract`, async via Redis), PDF preprocessing, OpenTelemetry traces, load test.
- **Phase 6** — MCP servers (one per tool), Python SDK, tested independently with MCP Inspector.
- **Phase 7** — LangGraph agent: state schema, one file per node, conditional edges (code, not LLM), subgraphs, streaming, checkpointing, `interrupt()` for HITL. Use Anthropic SDK directly inside nodes.
- **Phase 8** — Observability + evals: 30-example agent eval set, deterministic + LLM-as-judge graders, CI gating, replay CLI, per-paper audit log.
- **Phase 9** — Active learning: corrections (extraction edits, agent overrides) become training data. Retrain triggers, A/B harness, drift detection.
- **Phase 10** — One ambitious extension. Recommended: distill agent layer to a local model for offline operation.

### First three concrete tasks

1. Initialize repo per Phase 1 layout with `uv`. `docker-compose.yml` with Postgres (pgvector), Redis, MinIO. Verify all start cleanly.
2. Define extraction schema in `packages/extraction/schema.py` (generic `ExtractedField` with `value`/`bbox`/`confidence`, concrete `Paper` model).
3. PyMuPDF+regex baseline + eval harness skeleton. Run on 3-5 sample papers before labeled data exists.

### Operating principles

- Build evals before the thing being evaluated.
- Don't add a library without justifying it. Prefer 30 lines of Python.
- No LangChain core / LCEL / agent toolkits. Just LangGraph + LangSmith + LLM SDKs directly.
- Don't simulate behavior we should actually build (past Phase 6, no mocking tools).
- Every node and tool gets a unit test. Integration tests at the graph level.
- Trace everything from day one.
- When stuck, re-read the primary source before guessing.
- Pydantic everywhere for data interchange. No raw dicts across module boundaries.
- Agent structure is code, not prompts. Routing in conditional edges, not "please decide X or Y."
- Don't build for offline prematurely. Use the Anthropic API in agent nodes; no provider abstraction.

### Style

Plain prose in commits and docs. No em dashes. No marketing language. Short paragraphs. Honest about tradeoffs.

---

## Coding Guidelines

**1. Think before coding.** State assumptions. Surface tradeoffs. If something is unclear, ask before implementing. If multiple interpretations exist, present them.

**2. Simplicity first.** Minimum code that solves the problem. No speculative abstractions, no unrequested config knobs, no error handling for impossible cases. If 200 lines could be 50, rewrite.

**3. Surgical changes.** Touch only what you must. No drive-by refactors, no reformatting untouched code. Remove orphans your changes created; leave pre-existing dead code alone (mention it, don't delete it). Every changed line should trace to the stated goal.

**4. Goal-driven execution.** Turn tasks into verifiable goals ("write the test, then make it pass"). State a brief plan for multi-step work and check off as you go.

**5. Explain before coding.** Before writing code, say what you're building, why you're building it, and how it works conceptually. Then code.

**6. Concise code docs.** Comments and docstrings are short one-liners. Detailed explanations go in chat, not source.
