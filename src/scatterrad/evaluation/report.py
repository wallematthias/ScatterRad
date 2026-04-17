from __future__ import annotations

from pathlib import Path

from scatterrad.evaluation.aggregate import aggregate_folds
from scatterrad.paths import ScatterRadPaths


def _find_result_dirs(paths: ScatterRadPaths, task_name: str, model_kind: str) -> list[Path]:
    return paths.result_fold_dirs(task_name, model_kind)


def render_report(
    paths: ScatterRadPaths,
    task_name: str,
    model_kind: str,
    output_path: Path | None = None,
) -> str:
    """Render a markdown report for all folds of one task/model."""

    result_dirs = _find_result_dirs(paths, task_name, model_kind)
    if not result_dirs:
        raise FileNotFoundError(f"No fold results found for task={task_name}, model={model_kind}")

    agg = aggregate_folds(result_dirs)
    lines = [
        f"# ScatterRad Report: {task_name}",
        "",
        f"- Model: `{model_kind}`",
        f"- Folds: {agg['n_folds']}",
        f"- Runtime (s): {agg['total_runtime_seconds']:.2f}",
        "",
        "## Metrics Summary",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]

    for metric, stats in agg["metrics_summary"].items():
        lines.append(
            f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )

    if agg["attention_weights_mean"]:
        lines.extend(["", "## Mean Attention Weights", ""])
        for label, weight in sorted(agg["attention_weights_mean"].items()):
            bar = "#" * int(round(weight * 20))
            lines.append(f"- Label {label}: {weight:.3f} `{bar}`")

    md = "\n".join(lines) + "\n"
    out = output_path or (paths.results_dataset_dir / task_name / model_kind / "report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    return md
