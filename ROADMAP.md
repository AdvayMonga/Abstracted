# Abstracted — Project Roadmap

> A two-layer agentic document processing system that mirrors Nanonets' production stack, built on top of a real workflow I use (reading research papers). The name is a play on paper abstracts and on the agent abstracting away the work of triaging papers.
>
> **Audience:** Claude Code. This document is the master plan. Treat it as a contract for what we're building and why. When in doubt, refer back.

---

## 0. Context for Claude Code

I'm a CS student at UMD interning at Nanonets soon. Nanonets sells two products that plug into each other:

1. **Agentic Data Extraction** — a VLM (Nanonets OCR-3) that turns documents into markdown + structured JSON with **bounding boxes and per-field confidence scores**.
2. **Nanonets Agents** — agents that read the structured output, apply rules, call tools, and complete work in external systems with full auditability.

I'm rebuilding both layers from scratch on a different document type (research papers instead of invoices) so I learn the technical machinery without faking a use case I don't have. The goal is to walk into the internship having essentially reimplemented their core product on a substrate I actually use.

**Hard constraints I want enforced:**
- Maximum technical depth on both ML and systems sides.
- Use the *real* production frameworks: LangGraph for orchestration, MCP for tools, LangSmith for observability, Unsloth/LLaMA-Factory for fine-tuning, vLLM for serving.
- No shortcuts that skip the learning. If a framework is hiding something important, we go look at what it's hiding before we use it.
- Build evals *before* building the thing being evaluated. Always.
- Every phase ends with something that works end-to-end, even if narrow.

**Stack I have:**
- Local GPU for training/inference
- Obsidian as my note system (the agent's main "system of record")
- Python is primary; Go/TypeScript fine for specific services if justified

---

## 1. What we're building

A system that ingests research papers (PDFs) and:

1. **Extracts** structured content: title, authors, abstract, methods, datasets used, tools/code, key results, citations — with bounding boxes and confidence scores per field.
2. **Reasons** about each paper using an agent: how does it relate to my existing notes? Is there a GitHub repo? Should I deep-read, skim, or skip? What are the open questions?
3. **Acts** in real systems: creates an Obsidian note with structured frontmatter and a templated body, cross-links to related notes I already have, fetches the code repo if mentioned, queries Semantic Scholar for citation context, and pushes a summary card.
4. **Escalates** when uncertain: low-confidence extractions or ambiguous classification decisions go to a review queue I can act on, and corrections feed back into training data.

This is functionally identical to what Nanonets does for invoices/POs in AP automation. The architecture is the same; only the domain differs.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 4 — Observability                                     │
│  LangSmith traces, evals, prompt versioning, replay          │
├──────────────────────────────────────────────────────────────┤
│  Layer 3 — Agent (LangGraph)                                 │
│  Stateful graph, Postgres checkpointer, interrupt() for HITL │
│  Calls tools via MCP                                         │
├──────────────────────────────────────────────────────────────┤
│  Layer 2 — Tools (MCP servers, one process each)             │
│   - obsidian_vault       (read/write notes, find related)    │
│   - semantic_scholar     (search, citations, recommendations)│
│   - github_lookup        (find/inspect referenced repos)     │
│   - arxiv_fetch          (pull paper metadata + PDF)         │
│   - review_queue         (human-in-the-loop escalation)      │
├──────────────────────────────────────────────────────────────┤
│  Layer 1 — Extraction                                        │
│  Fine-tuned Qwen2.5-VL emitting JSON + bbox + confidence     │
│  Served via vLLM behind FastAPI                              │
├──────────────────────────────────────────────────────────────┤
│  Storage: Postgres + pgvector  |  Queue: Redis               │
└──────────────────────────────────────────────────────────────┘
```

### What "done" looks like

- I can drop a PDF in a watched folder (or hit an API endpoint) and within ~60s have a fully-formed Obsidian note with structured frontmatter, summary, related-paper backlinks, code-repo link, and citation context.
- For uncertain cases, I get a notification with a review UI showing the original PDF region (via bbox), the extracted value, and the agent's reasoning trace.
- Every action the agent took is replayable from a LangSmith trace.
- The fine-tuned extraction model demonstrably beats the off-the-shelf baseline on a held-out test set of research papers.

### Deployment topology

The two layers have different runtime characteristics, so they're deployed differently. This is intentional and mirrors how production IDP systems are built.

**Extraction layer — local, small, fine-tuned**

- Runs on my local GPU (sized for ~8GB VRAM-class hardware)
- Base model: Qwen2.5-VL-3B
- Training: QLoRA (4-bit base + LoRA adapters) via Unsloth. Falls back to LLaMA-Factory if Unsloth lacks support for Qwen2.5-VL at the time we build it.
- Inference: vLLM if VRAM permits; otherwise GGUF + llama.cpp. Both options get implemented and benchmarked.
- Why local: high call volume per paper (every page), strict schema, narrow task, latency-tolerant. Fine-tuned small model wins on cost and is sufficient on quality after the work in Phase 4.

**Agent layer — API, large, general-purpose**

- Uses the Anthropic API (Claude Sonnet for most nodes, Opus for the LLM-as-judge graders in Phase 8)
- Why API: low call volume per paper (handful of LLM calls per run), tasks vary (classify, synthesize, decide), benefits from a strong general model. Cost per paper is bounded because volume is bounded — I process maybe 5-20 papers a day, not thousands.
- The split is the same pattern Nanonets uses in production: fine-tuned in-house model for extraction, customer's choice of frontier model for agent work on top.

**Execution mode — invocation-based**

- Not a long-running daemon. I trigger it: `abstracted process paper.pdf` or via a watched folder.
- vLLM/llama.cpp comes up on demand if not running, or stays warm if I leave it running while actively using the system.
- Postgres and Redis run in `docker-compose` and can stay up cheaply.
- Tradeoff accepted: cold-start latency on the first paper of a session. Worth it to keep the system simple and laptop-friendly.

**Offline mode — deferred but in scope**

- A long-term goal is to make the whole pipeline (including the agent layer) runnable offline.
- Path to offline: replace the Anthropic API in agent nodes with a locally-served reasoning model (Qwen3-32B or Llama-3.3-70B quantized, if VRAM permits, or a smaller model if not).
- We'll revisit this in Phase 10 as one of the candidate ambitious extensions. Don't build for it now — it would force premature abstractions in Phase 7.

---

## 2. Mapping to Nanonets' product

| Nanonets feature | Our equivalent |
|---|---|
| Nanonets OCR-3 (VLM with bbox + confidence) | Fine-tuned Qwen2.5-VL emitting same schema |
| Agentic Data Extraction API | FastAPI `/extract` endpoint, markdown + JSON |
| Nanonets Agents (rules + tool calls) | LangGraph agent with MCP tools |
| "Every extraction is traceable" | LangSmith traces + custom audit log |
| Confidence-based HITL routing | LangGraph `interrupt()` driven by confidence thresholds |
| ERP integrations (SAP, QBO, Xero) | Obsidian vault, Semantic Scholar, GitHub |
| "When you correct it, it learns" | Active learning loop, corrections become fine-tuning data |
| Templates / no-template extraction | Schema-driven prompt + structured-output enforcement |
| Auditability | LangSmith + bbox citations on every field |

---

## 3. Phase plan

Each phase has: **goal**, **deliverables**, **learning objectives**, **definition of done**. Order matters — later phases assume earlier ones.

### Phase 0 — Foundations (2-3 days, no code)

**Goal:** Build the conceptual map before touching tools.

**Reading list (in order):**
1. **LayoutLMv3** paper (Microsoft, 2022) — why joint text + position + image embeddings matter.
2. **Donut** paper (Clova/NAVER) — end-to-end OCR-free document understanding.
3. **Nougat** paper (Meta) — academic doc parsing specifically, very close to our domain.
4. **Qwen2.5-VL technical report** — the base model we'll fine-tune.
5. **Nanonets OCR-3 announcement** (nanonets.com/research/nanonets-ocr-3) — what their flagship model emits and why bbox+confidence is the unlock.
6. **LangGraph docs**: Conceptual Guide section (state, nodes, edges, checkpointers, interrupts, subgraphs). Skip the tutorials for now.
7. **MCP specification** (modelcontextprotocol.io) — tools, resources, prompts; stdio vs SSE transport.
8. **LangSmith docs**: Tracing and Evals sections.
9. **Anthropic's "Building effective agents"** post — patterns for tool use, reflection, routing.

**Deliverable:** A 1-2 page `NOTES.md` in the repo summarizing each in my own words. If I can't explain why each thing exists, I haven't read it carefully enough.

**Done when:** I can answer these without looking anything up:
- Why is end-to-end VLM-OCR replacing detection+recognition pipelines? When would I still pick the old way?
- What's the difference between Donut and Nougat? Why does Nougat matter for our use case?
- What does a LangGraph state graph give me that a Python `while` loop with structured outputs doesn't?
- What's MCP's value proposition vs. just defining tool functions in code?
- Why does Nanonets emit bounding boxes alongside extracted values?

---

### Phase 1 — Repo scaffolding + eval harness (2-3 days)

**Goal:** Set up the project skeleton and build the eval framework *before* any models exist. We measure first, build second.

**Repo structure:**
```
paper-agent/
├── pyproject.toml              # uv or poetry, Python 3.11+
├── docker-compose.yml          # Postgres, Redis, MinIO for PDFs
├── .env.example
├── NOTES.md                    # my Phase 0 readings summary
├── README.md
├── packages/
│   ├── extraction/             # Layer 1: VLM + serving
│   │   ├── schema.py           # Pydantic models for extraction output
│   │   ├── baseline/           # off-the-shelf model wrappers
│   │   ├── train/              # fine-tuning scripts
│   │   ├── serve/              # FastAPI + vLLM
│   │   └── eval/               # extraction eval harness
│   ├── agent/                  # Layer 3: LangGraph
│   │   ├── graph.py            # the agent graph definition
│   │   ├── nodes/              # one file per node
│   │   ├── state.py            # TypedDict state schema
│   │   └── eval/               # agent eval harness
│   ├── tools/                  # Layer 2: MCP servers
│   │   ├── obsidian/
│   │   ├── semantic_scholar/
│   │   ├── github_lookup/
│   │   ├── arxiv_fetch/
│   │   └── review_queue/
│   └── shared/                 # pydantic models, db client, etc.
├── data/
│   ├── raw/                    # raw PDFs (gitignored)
│   ├── labeled/                # ground truth JSON (gitignored, DVC-tracked)
│   └── eval/                   # held-out test set
├── infra/
│   ├── postgres/init.sql       # schema for traces, queue, labels
│   └── obsidian_vault_test/    # a test vault for dev
└── notebooks/                  # exploration only, not source of truth
```

**Tasks:**
1. Initialize repo with `uv` for Python deps. Set up `ruff`, `pyright`, `pytest`.
2. Spin up `docker-compose` with Postgres (with `pgvector`), Redis, MinIO.
3. Define the **extraction schema** in `packages/extraction/schema.py` as Pydantic models. Every field is an object with `value`, `bbox` (or `null`), `confidence` (0-1). Mirror Nanonets OCR-3's output shape exactly.
4. Build the **eval harness** for extraction:
   - Loader for `data/eval/` (PDF + ground truth JSON pairs)
   - Field-level metrics: exact match, fuzzy match (rapidfuzz), normalized field accuracy
   - Document-level: % fully correct, average per-field F1
   - List-field metrics: Hungarian matching on similarity for authors/citations
   - CLI: `python -m extraction.eval --model <name> --dataset eval/v1` → markdown report
5. Build a **trivial baseline**: PyMuPDF text dump + regex for title/abstract. Run the eval. Get a number. That number is our floor.

**Learning objectives:**
- How do you measure document understanding? (Spoiler: it's hard and metric choice matters as much as model choice.)
- Why does ground truth need versioning? (Because label noise + you'll re-label, and you need reproducibility.)
- How does uv differ from pip/poetry, and why do production Python projects use it now?

**Done when:** `python -m extraction.eval --model pymupdf_regex` produces a markdown report with metrics, and I have a baseline number for every field type in the schema.

---

### Phase 2 — Dataset construction (4-6 days)

**Goal:** Build a labeled dataset of research papers with bbox-grounded ground truth. This phase is unglamorous and the hardest to do well.

**Tasks:**

1. **Collect raw papers** (~150-200):
   - Pull 100 papers from arXiv across CS (ML, systems, NLP) and bio (genomics, scRNA-seq — my domain).
   - Pull 30 from biorxiv (different formatting conventions).
   - Pull 20 older PDFs that are scans (OCR'd or not), to test the messy case.

2. **Define the labeling schema**: same Pydantic schema from Phase 1, but for ground truth we'll fill it manually.

3. **Set up Label Studio** in Docker. Configure a labeling template that:
   - Shows the PDF page
   - Lets me draw bboxes for each field
   - Pre-populates with off-the-shelf VLM output to speed up labeling
   - Exports to our JSON schema

4. **Label 80 papers manually**. Yes, this is tedious. The lesson here is *exactly* why Nanonets sells a labeling UI.

5. **Generate synthetic augmentation**: take labeled papers, render with random fonts/spacing/noise, regenerate the bboxes. This gives us ~5x more training data for the cost of one synthesis script.

6. **Split**: 60% train, 15% val, 15% test, 10% held-out (only touched at the very end).

7. **Version with DVC** pointing at S3-compatible storage (MinIO locally is fine, or a real bucket).

8. **Compute dataset statistics**: token length distributions, field presence rates, layout complexity heuristics. Add to a `data/STATS.md`. Knowing your data is half the battle.

**Learning objectives:**
- How label noise creeps in (especially on subjective fields like "methods summary").
- Inter-annotator agreement isn't a thing when there's one of you, but you'll notice your *own* drift over time. Re-label 20 random papers after a week and measure disagreement.
- The synthetic-vs-real tradeoff: synthetic gets you volume, real gets you the long tail.

**Done when:** I have ~150 labeled papers in version control, train/val/test splits, and `data/STATS.md` populated.

---

### Phase 3 — Off-the-shelf VLM baselines (2-3 days)

**Goal:** Run several off-the-shelf models through the eval harness to establish meaningful baselines.

**Models to test:**
1. **Qwen2.5-VL-3B** — small, fast, will be our fine-tuning candidate
2. **Qwen2.5-VL-7B** — bigger sibling
3. **Nanonets-OCR-s** (HuggingFace) — Nanonets' open release, important comparison point
4. **Donut-base** — older end-to-end model, useful reference
5. **GPT-4o or Claude Sonnet** (one API model) — what's the ceiling if we don't care about cost?

For each:
- Write a wrapper in `extraction/baseline/<model>.py` that takes a PDF, returns our schema.
- Use structured output enforcement where possible (Outlines, Instructor, or native JSON mode).
- Run through eval harness.
- Add results to a `BENCHMARKS.md` table.

**Critical step:** For at least one open model, extract per-token logprobs and use them to *compute* a confidence score (mean logprob of the field tokens, normalized). Don't ask the model to self-report confidence — that's notoriously unreliable. This is the bedrock of trustworthy uncertainty estimation.

**Learning objectives:**
- The cost/quality/latency Pareto frontier on real document tasks.
- Why "ask the LLM for confidence" is bad and what to do instead.
- How different models fail differently (Donut hallucinates, big VLMs are slow, small VLMs miss layout cues).

**Done when:** `BENCHMARKS.md` has a table comparing all 5 models on all metrics, and one of the open models exposes calibrated per-field confidence.

---

### Phase 4 — Fine-tuning the extraction model (1-2 weeks)

**Goal:** Take Qwen2.5-VL-3B and fine-tune it on our paper dataset to emit our exact schema with bbox + confidence. Beat the baselines.

**Tasks:**

1. **Choose framework**: Unsloth (fastest, best for single GPU) for LoRA. Fall back to LLaMA-Factory if Unsloth doesn't support Qwen2.5-VL well at the time. *Verify support before committing.*

2. **Format training data**: instruction format with image + system prompt + JSON output. Include the bboxes in the output (as `[x0,y0,x1,y1]` arrays) — the model learns to emit them alongside values.

3. **First run**: LoRA rank 16, lr 2e-4, 3 epochs, save every 200 steps. Use W&B for tracking.

4. **Sanity checks before declaring victory**:
   - Loss curve doesn't go to zero (that's memorization, not learning)
   - Eval on val set actually improves epoch over epoch
   - Spot-check 5 random val outputs — do bboxes actually align?

5. **Hyperparameter sweep**: rank ∈ {8, 16, 32}, lr ∈ {5e-5, 1e-4, 2e-4}, epochs ∈ {2, 3, 5}. Use W&B sweeps. Pick best by val field-F1.

6. **Ablation studies** (these are what actually teach you):
   - Data scaling: train on 25%, 50%, 100% of data. Plot accuracy vs. data size. This curve tells you whether to label more.
   - Synthetic-only vs. real-only vs. mixed: which wins?
   - LoRA vs. QLoRA (4-bit base) vs. full fine-tune on the smallest model
   - Different prompt formats (zero-shot vs. few-shot in the system prompt)

7. **Confidence calibration**: take the best model, compute confidence from logprobs as in Phase 3, then:
   - Plot reliability diagram on val set (predicted confidence vs. actual accuracy in buckets)
   - Apply **temperature scaling** to calibrate
   - Verify on held-out test set
   - This is the step most projects skip; don't.

8. **Final eval on held-out test set** (which we haven't touched). Numbers are honest only here.

**Learning objectives:**
- What LoRA actually does (frozen base + low-rank adapters on attention projections). Read the LoRA paper if you haven't.
- How to read loss curves (training loss decreasing + val loss increasing = overfitting; both flat = lr too low; both jagged = lr too high).
- Why calibrated confidence is the precondition for everything in Layer 2.
- The data-scaling curve is more informative than a single accuracy number.

**Done when:** Fine-tuned model beats the best off-the-shelf open model on `BENCHMARKS.md` *and* has calibrated confidence scores within 5% on the reliability diagram.

---

### Phase 5 — Extraction serving (3-4 days)

**Goal:** Production-grade serving of the extraction model. Not "it runs in a notebook."

**Tasks:**

1. **Serve with vLLM**: convert the LoRA adapter or merge it into the base for inference; spin up vLLM as a separate service. Verify throughput with concurrent requests.

2. **FastAPI gateway** in `packages/extraction/serve/`:
   - `POST /extract` — multipart PDF upload, returns extraction JSON
   - `POST /extract/async` — enqueues to Redis, returns job ID
   - `GET /extract/{job_id}` — poll for status
   - Streaming endpoint with Server-Sent Events for live updates

3. **PDF preprocessing**: PyMuPDF to convert pages to images at appropriate DPI. Handle multi-page docs (chunked or stitched, your choice — try both).

4. **Async worker** (RQ or arq, not Celery — too heavy): pulls jobs from Redis, calls vLLM, writes results to Postgres.

5. **Observability**: OpenTelemetry traces from request → preprocessing → model → response. Export to LangSmith *and* a local Jaeger if you want to compare.

6. **Load test**: Use `oha` or `k6` to hit 50 concurrent requests. Profile bottlenecks.

**Learning objectives:**
- ML inference is its own engineering discipline. vLLM exists because batching dynamics for autoregressive models are non-trivial.
- Throughput vs. latency tradeoff: batching boosts throughput, hurts p99 latency.
- Why async queues exist for ML workloads (slow, variable, GPU-bound).

**Done when:** I can run `curl -X POST http://localhost:8000/extract -F file=@paper.pdf` and get JSON back in <10s for typical papers, with traces visible in LangSmith.

---

### Phase 6 — Tools as MCP servers (4-5 days)

**Goal:** Each external integration is a standalone MCP server. The agent doesn't import these directly; it talks to them over MCP.

**Why this architecture:** It's the production pattern Anthropic ships with Claude Code, and it's where the industry is converging. Tools become independently deployable, testable, and swappable. Also, your future employer (Nanonets) almost certainly will adopt or has adopted MCP.

**Servers to build:**

1. **`obsidian_vault`** (Python, MCP Python SDK):
   - `create_note(title, frontmatter, body, folder)` → note path
   - `find_related_notes(query, k=5)` → list of paths + relevance scores (uses pgvector over note embeddings)
   - `read_note(path)` → contents
   - `update_note(path, append=None, replace=None)` → bool
   - `list_tags()` → tags in the vault
   - Indexes the vault into Postgres+pgvector on startup; watches for changes

2. **`semantic_scholar`**:
   - `search(query, limit)` → papers
   - `get_paper(id)` → details
   - `get_citations(id)` → who cites this
   - `get_references(id)` → what this cites
   - `get_recommendations(id)` → related papers
   - Rate-limited; uses the official S2 API

3. **`github_lookup`**:
   - `find_repo(paper_title, authors)` → likely repo URL
   - `inspect_repo(url)` → README, language, last commit, stars
   - Smart heuristics: check abstract/paper for github.com URLs first, then search

4. **`arxiv_fetch`**:
   - `get_metadata(arxiv_id_or_url)` → title, authors, abstract, categories
   - `download_pdf(arxiv_id)` → local path

5. **`review_queue`**:
   - `escalate(invoice_id, reason, fields_in_question, snapshot)` → review_id
   - `wait_for_resolution(review_id, timeout)` → resolution (or timeout)
   - Backed by Postgres; has a small web UI for me to act on

**Tasks:**

1. For each server, write the MCP server using the Python SDK, run it as a standalone process.
2. Test each independently with MCP Inspector before wiring to the agent.
3. Add a `docker-compose` service for each so the whole stack starts with one command.
4. Write integration tests for each tool (mocked external APIs).

**Learning objectives:**
- The Tools vs. Resources vs. Prompts distinction in MCP and why it matters.
- stdio vs. SSE transport tradeoffs.
- Why separating tools from the agent process is good architecture (independent scaling, language flexibility, security boundaries).

**Done when:** All 5 MCP servers run independently, are testable with MCP Inspector, and the `obsidian_vault` server can actually create a note in my test vault.

---

### Phase 7 — The agent graph (1-2 weeks)

**Goal:** Build the LangGraph agent that reads extraction output and acts.

**The graph design (sketch on paper first):**

```
                ┌──────────────────┐
                │ ingest_extraction │
                └──────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ check_confidence  │──low──┐
                └──────────────────┘       │
                         │ high             │
                         ▼                  ▼
                ┌──────────────────┐  ┌──────────────────┐
                │ classify_paper    │  │ escalate_review   │ ──► interrupt()
                │ (deep/skim/skip)  │  └──────────────────┘
                └──────────────────┘
                         │
                ┌────────┼────────┐
                ▼        ▼        ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │ find_repo │ │find_     │ │get_citation  │
        │ (github)  │ │ related  │ │ context      │
        │           │ │ (vault)  │ │ (s2)         │
        └──────────┘ └──────────┘ └──────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ synthesize_note   │
                └──────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ write_to_obsidian │
                └──────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ done              │
                └──────────────────┘
```

**Tasks:**

1. **Define state schema** (`packages/agent/state.py`): TypedDict with extraction result, confidence summary, classification, fetched repo info, related notes, S2 context, draft note, errors. Use reducers (e.g., `Annotated[list, operator.add]`) where appropriate.

2. **One file per node** (`packages/agent/nodes/*.py`). Each node:
   - Takes state, returns state delta
   - Has its own unit test
   - Logs structured events
   - Catches its own errors and emits an error state delta rather than throwing

3. **Conditional edges** for routing — these are code, not LLM calls. Route by confidence thresholds and classification result.

4. **`interrupt()` for human review**: when the `escalate_review` node fires, the graph persists state to the Postgres checkpointer and pauses. My review UI resolves it via `Command(resume=...)`.

5. **Use the Anthropic SDK directly inside nodes** for LLM calls (Claude Sonnet for classification and synthesis). Don't wrap in LangChain — call the SDK directly. This keeps the surface area small and the behavior obvious.

6. **MCP client setup**: LangGraph nodes that need tools connect to the MCP servers via `langchain-mcp-adapters` (or equivalent current package). Verify tool names and schemas at startup.

7. **Subgraphs**: `find_repo`, `find_related`, `get_citation_context` are independent enough to be subgraphs — each with its own retry/error logic. Compose into the main graph.

8. **Streaming**: yield state updates from each node so the frontend (eventually) can show progress.

**Learning objectives:**
- State graphs as a mental model — once it clicks, you'll never want to write an agent loop by hand again.
- Why "the LLM decides everything" is a bad agent architecture. The *structure* should be deterministic; only the *content* is LLM-generated.
- Checkpointing: how durable execution actually works, and why it matters for slow workflows.
- Interrupts as a first-class primitive vs. ad-hoc human-in-the-loop.

**Done when:** I can run the agent on a paper end-to-end, watch its trace in LangSmith, and see a note appear in my Obsidian vault. When confidence is low, I can resolve the escalation via my review UI and it resumes cleanly.

---

### Phase 8 — Observability, evals, replay (1 week)

**Goal:** Turn the agent into something you can debug, test, and trust.

**Tasks:**

1. **LangSmith tracing**: ensure every node, every tool call, every LLM call is traced. Add custom metadata: paper ID, confidence levels, classification result.

2. **Agent eval dataset**: ~30 hand-curated papers with expected agent decisions: should classify as X, should find repo Y, should link to existing note Z, should escalate Y/N. This is *much* harder to build than an extraction eval; that's why most teams skip it. Don't skip it.

3. **Custom graders** in LangSmith:
   - Deterministic: "did it create the expected note path?", "did it correctly identify the repo URL?"
   - LLM-as-judge: "is the synthesized summary faithful to the paper?" (use a stronger model as judge, e.g., Claude Opus)
   - Trace-based: "did the agent call the right tools in the right order?" (this catches regressions in graph structure)

4. **CI integration**: every PR runs the eval suite and posts results. Block merge if accuracy drops >5%.

5. **Replay tooling**: build a CLI that takes a LangSmith trace ID, fetches the original state at any node, and re-runs from that point with optionally-modified inputs. This is your debugger.

6. **Audit log per paper**: in addition to LangSmith, write a human-readable trace file alongside the Obsidian note showing every decision, every confidence, every tool call. The "every extraction is traceable" promise from Nanonets, made literal.

**Learning objectives:**
- Why LLM evals are different from classical software tests (non-determinism, subjective correctness, expensive to grade).
- The cost of evals: LLM-as-judge isn't free; budget for it.
- Replay is the single most underrated piece of agent infrastructure.

**Done when:** I have a 30-example eval set, automated graders, and CI that runs them. I can replay any historical run from a LangSmith trace.

---

### Phase 9 — Active learning loop (3-5 days)

**Goal:** Corrections turn into training data. The system improves with use.

**Tasks:**

1. **Capture corrections at both layers**:
   - Extraction layer: when I edit the auto-generated Obsidian note's frontmatter, diff it against the model output and store as a correction.
   - Agent layer: when I override an agent decision (e.g., mark a paper as "deep read" that it classified as "skim"), store the override with the full state context.

2. **Periodic retraining triggers**: when N corrections accumulate, kick off a retrain job that LoRA-fine-tunes on the new data.

3. **For agent corrections**: store as few-shot examples that get injected into the relevant node's system prompt. Later, when there are enough, fine-tune a smaller model on these decisions specifically.

4. **A/B harness**: when a new model version is ready, route 10% of papers through it and compare metrics to the previous version. Auto-promote if it wins.

5. **Drift detection**: track per-field accuracy over time; alert if it drops.

**Learning objectives:**
- Why "the system gets better with use" isn't marketing — it's an architecture choice you have to bake in.
- The feedback loop is a real engineering surface: capture, store, version, retrain, deploy, evaluate.

**Done when:** Corrections logged automatically, a retrain script exists, and I've done at least one retrain that demonstrably moves a metric.

---

### Phase 10 — Pick one ambitious extension

After everything above works, choose one to go deep on. **Distillation toward fully-local operation is the recommended pick** given my stated goal of eventually running offline.

1. **Distillation toward offline operation** (recommended): replace the API-based agent layer with a locally-served reasoning model. Capture traces from Claude doing the agent task during Phases 7-9, then fine-tune a smaller open model (Qwen3-7B or 14B depending on VRAM) on those traces. End state: the whole pipeline runs without network access. This is the deepest, most employable extension and directly serves the offline goal.
2. **Multi-agent**: split into specialized agents (extractor reviewer, citation analyst, code inspector) coordinating via shared state. Coordination problems will be real.
3. **RLHF-lite**: train a reward model on my corrections, use DPO to fine-tune the classification model on preferences.
4. **Few-shot template adaptation**: a new venue with a new layout shows up; can the system adapt with 3 examples and no retraining?
5. **Self-correcting extraction**: agent reads its own extraction, calls back to the VLM with targeted questions ("re-read the methods section, focus on the dataset name"). A loop between agent and extraction model.

---

## 4. Operating principles for Claude Code

When working on this project:

1. **Always view skill files before producing artifacts.** Read SKILL.md when relevant.
2. **Build evals before the thing being evaluated.** Always.
3. **Don't add a library without justifying it.** If we can do it in 30 lines of Python, prefer that. The stack is already large.
4. **Don't use LangChain core, LCEL, or agent toolkits.** Just LangGraph + LangSmith + the LLM SDKs directly.
5. **Don't simulate behavior we should actually build.** If the agent should call a tool, build the tool — don't mock it past Phase 6.
6. **Every node and tool gets a unit test.** Integration tests at the graph level.
7. **Trace everything from day one.** It's easier than adding tracing later.
8. **When stuck, re-read the relevant primary source** (a paper, an MCP spec section, the LangGraph conceptual docs) before guessing.
9. **Use Pydantic everywhere for data interchange.** No raw dicts crossing module boundaries.
10. **The agent's structure is code, not prompts.** Routing logic lives in `if`/conditional edges, not in "please decide between X and Y."
11. **Don't build for offline operation prematurely.** Offline is a Phase 10 goal, not a Phase 7 constraint. Use the Anthropic API directly in agent nodes. Don't add an LLM abstraction layer "in case we swap providers later" — the swap, when we do it, will be informed by traces we haven't collected yet.

---

## 5. Style preferences

- Plain prose in commits and docs. No em dashes. No marketing language. Short paragraphs.
- Question-format section titles where natural.
- Don't bold-emphasize things that don't need it.
- Speak honestly about tradeoffs — every choice has a cost.

---

## 6. Stretch / nice-to-haves

- Browser extension that sends papers from arXiv/biorxiv to the agent with one click.
- iOS Shortcut that does the same from Safari.
- A small web dashboard showing the queue, recent processings, and metrics.
- Public-facing demo (sanitized) for the internship portfolio.

---

## 7. First three concrete tasks for Claude Code

1. Initialize the repo per Phase 1 structure with `uv`. Set up `docker-compose.yml` with Postgres (pgvector), Redis, MinIO. Verify all three start cleanly.
2. Define the extraction schema in `packages/extraction/schema.py` as Pydantic models (one ExtractedField generic with `value`, `bbox`, `confidence`, then concrete Paper model using it).
3. Build the trivial PyMuPDF+regex baseline and the eval harness skeleton. Run them on 3-5 sample papers to verify the loop works end-to-end before we have any labeled data.

After that, we start Phase 2 (data collection + labeling).

---

*End of roadmap.*
