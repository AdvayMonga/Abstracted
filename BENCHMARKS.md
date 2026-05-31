# Extraction benchmarks

Eval set: `val` split, 26 CS papers. Labels built programmatically in Phase 2
(arXiv API + GROBID + GitHub-URL scan + regex). See `data/STATS.md`.

Metric: per-field mean similarity (scalars) or mean F1 (lists), threshold 0.7
for list match. Macro F1 = mean across all seven fields.

## Headline

| model | macro F1 | title | abstract | authors | citations | tools_code |
|---|---|---|---|---|---|---|
| `pymupdf_regex` (floor) | 0.415 | 0.295 | 0.720 | 0.000 | 0.000 | 0.538 |
| `claude_sonnet` | **0.581** | **0.999** | **0.977** | **0.962** | **0.413** | **0.699** |

Claude Sonnet wins on every field where the ground truth is non-degenerate.

## Per-field details (claude_sonnet)

| field | score | notes |
|---|---|---|
| title | 0.999 | Effectively solved. |
| abstract | 0.977 | Two misses are PDF-vs-arXiv-metadata version drift, same caveat as Phase 2. |
| authors | 0.962 | One paper at 0.00 — anonymous-review version with no listed authors. |
| methods | 0.288 | Metric artifact: GROBID labels are verbatim full sections (often 300+ tokens with inline citation markers and equation residue). Claude returns clean verbatim too but the strings rarely fuzz-match end-to-end. |
| datasets | 0.231 | Ground truth is mostly `[]` (Phase 2 left this field null). Claude sometimes returns real dataset names → counted as false positives. Not a Claude problem. |
| tools_code | 0.699 | Limited by label coverage (39% of papers now have labeled GitHub URLs after the Phase 2.5 backfill). |
| key_results | 0.077 | Metric artifact: ground truth was produced by regex on results/conclusion text. The baseline shares that regex, so it matches itself. Claude returns paraphrased highlights instead. |
| citations | 0.413 | High variance: 11/26 papers score 0.0, 7/26 score > 0.8. The 0.0s correlate with references appearing on pages past our render window (first 6 + last 3). |

## Known caveats

1. **methods** scoring is too strict for "extract the methods section." Two
   honest verbatim copies of the same section can still fuzz-mismatch on
   whitespace, inline citation tokens, and equation residue.
2. **datasets** ground truth is null across the board. Cannot evaluate this
   field until labels exist.
3. **key_results** ground truth was built by the same regex that scores the
   baseline. Self-comparison, not informative.
4. **citations** scoring is honest but page-window-limited. Papers with
   references on pages 7-(N-3) get zero coverage. Fix is to render the
   references section explicitly (detect "References" page in the TOC text)
   or rendering more back pages.

## Cost

- `claude_sonnet`: ~$3 total for the 26-paper val split.
  - 9 page images at 144 DPI per paper (first 6 + last 3, deduped).
  - Median wall time per paper: ~75 s (one outlier at 2014 s, suspected
    Anthropic-side backoff).

## Reproduce

```bash
uv run python -m packages.extraction.eval --model pymupdf_regex --split val \
  --out reports/val_pymupdf_regex.md
uv run python -m packages.extraction.eval --model claude_sonnet  --split val \
  --out reports/val_claude_sonnet.md
```

Per-doc predictions are cached at `data/cache/predictions/<model>/<id>.json`
so re-runs are free. Add `--no-cache` to force.

## Status of Phase 3 models

| model | status |
|---|---|
| `pymupdf_regex` | floor, run |
| `claude_sonnet` | run |
| `qwen2.5-vl-3b` | stub registered, weights not pulled |
| `qwen2.5-vl-7b` | stub registered, weights not pulled |
| `nanonets-ocr-s` | stub registered, weights not pulled |
| `donut` | stub registered, weights not pulled |

Local VLMs deferred to Phase 4 setup (GPU box).
