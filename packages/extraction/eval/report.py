"""Render a DatasetReport as markdown."""

from packages.extraction.eval.scorer import DatasetReport


def to_markdown(report: DatasetReport, model_name: str, dataset_name: str) -> str:
    lines: list[str] = []
    lines.append(f"# Extraction eval: `{model_name}` on `{dataset_name}`")
    lines.append("")
    lines.append(f"- documents: {len(report.per_doc)}")
    lines.append(f"- macro F1: {report.macro_f1:.3f}")
    lines.append(f"- fully-correct docs: {report.pct_fully_correct:.1%}")
    lines.append("")
    lines.append("## Scalar fields (mean similarity)")
    lines.append("")
    lines.append("| field | score |")
    lines.append("|---|---|")
    for name, val in report.scalar_mean.items():
        lines.append(f"| {name} | {val:.3f} |")
    lines.append("")
    lines.append("## List fields (mean F1)")
    lines.append("")
    lines.append("| field | F1 |")
    lines.append("|---|---|")
    for name, val in report.list_f1_mean.items():
        lines.append(f"| {name} | {val:.3f} |")
    lines.append("")
    lines.append("## Per-document")
    lines.append("")
    lines.append("| doc | fully correct | title | abstract | authors F1 | citations F1 |")
    lines.append("|---|---|---|---|---|---|")
    for s in report.per_doc:
        lines.append(
            f"| {s.doc_id} | {'yes' if s.all_fields_correct else 'no'} "
            f"| {s.scalar.get('title', 0):.2f} "
            f"| {s.scalar.get('abstract', 0):.2f} "
            f"| {s.lists['authors'].f1:.2f} "
            f"| {s.lists['citations'].f1:.2f} |"
        )
    return "\n".join(lines) + "\n"
