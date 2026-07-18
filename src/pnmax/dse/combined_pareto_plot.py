from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import ticker

from pnmax.dse.pareto_export import PAIRWISE_FRONTIER_METRICS, pairwise_frontiers

BYTES_PER_MB = 1024.0 * 1024.0
PICOJOULES_PER_MILLIJOULE = 1_000_000_000.0
AXIS_LABELS: dict[str, str] = {
    "latency_cycles": "Latency (cycles)",
    "mem_footprint_bytes": "Memory Footprint (MB)",
    "energy": "Energy (mJ)",
}
PDF_SUFFIX = ".pdf"
PLOT_FIGSIZE = (10.0, 5.0)
TITLE_FONT_SIZE = 16.0
AXIS_LABEL_FONT_SIZE = 14.0
TICK_LABEL_FONT_SIZE = 12.0
ENERGY_TICK_DECIMALS = 12


def render_combined_frontier_plots(
    combined_dir: Path | str,
    *,
    dpi: int = 200,
    show: bool = False,
    output_base: Path | str | None = None,
) -> dict[str, str]:
    combined_dir_path = Path(combined_dir).resolve()
    if not (combined_dir_path / "summary.json").is_file():
        _raise_missing_summary_error(combined_dir_path)
    summary = _load_json(combined_dir_path / "summary.json")
    base_output = (
        _canonical_pdf_output_path(Path(output_base))
        if output_base is not None
        else (combined_dir_path / "combined_pareto.pdf").resolve()
    )
    title = str(summary.get("plot_title") or summary.get("arch_name") or "combined")
    frontier_csv_path = _resolve_combined_frontier_csv_path(
        combined_dir_path=combined_dir_path,
        summary=summary,
    )
    return render_frontier_csv_plots(
        frontier_csv_path,
        title=title,
        dpi=dpi,
        show=show,
        output_base=base_output,
    )


def render_frontier_csv_plots(
    frontier_csv: Path | str,
    *,
    title: str,
    output_base: Path | str,
    dpi: int = 200,
    show: bool = False,
) -> dict[str, str]:
    frontier_csv_path = Path(frontier_csv).resolve()
    rows = _load_frontier_rows(frontier_csv_path)
    if not rows:
        raise ValueError(f"No frontier rows found in {frontier_csv_path}.")

    frontiers = pairwise_frontiers(rows)
    base_output = _canonical_pdf_output_path(Path(output_base))
    outputs: dict[str, str] = {}
    for frontier_name, x_metric, y_metric in PAIRWISE_FRONTIER_METRICS:
        frontier_rows = list(frontiers.get(frontier_name, ()))
        if not frontier_rows:
            continue
        output_path = _output_path_for_frontier(base_output, frontier_name)
        _render_frontier_plot(
            frontier_rows=frontier_rows,
            frontier_name=frontier_name,
            x_metric=x_metric,
            y_metric=y_metric,
            title=title,
            output_path=output_path,
            dpi=dpi,
            show=show,
        )
        outputs[frontier_name] = str(output_path)
    return outputs


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping JSON payload in {path}.")
    return payload


def _resolve_combined_frontier_csv_path(
    *, combined_dir_path: Path, summary: dict[str, Any]
) -> Path:
    attempted_paths: list[Path] = []
    combined_frontier_csv = summary.get("combined_frontier_csv")
    if combined_frontier_csv is not None:
        raw_path = Path(str(combined_frontier_csv))
        if raw_path.is_absolute():
            candidate = raw_path.resolve()
            attempted_paths.append(candidate)
            if candidate.is_file():
                return candidate
        else:
            combined_relative = (combined_dir_path / raw_path).resolve()
            attempted_paths.append(combined_relative)
            if combined_relative.is_file():
                return combined_relative
            cwd_relative = raw_path.resolve()
            attempted_paths.append(cwd_relative)
            if cwd_relative.is_file():
                return cwd_relative

    fallback_path = (combined_dir_path / "combined_frontier.csv").resolve()
    attempted_paths.append(fallback_path)
    if fallback_path.is_file():
        return fallback_path

    unique_attempts: list[str] = []
    seen_attempts: set[str] = set()
    for attempted_path in attempted_paths:
        attempted = str(attempted_path)
        if attempted in seen_attempts:
            continue
        seen_attempts.add(attempted)
        unique_attempts.append(attempted)
    raise ValueError(
        "Missing frontier CSV. Checked: " + ", ".join(unique_attempts)
    )


def _raise_missing_summary_error(combined_dir_path: Path) -> None:
    validation_jsons = ("upmem.json", "hbm_pim.json", "attacc.json")
    if any((combined_dir_path / name).is_file() for name in validation_jsons):
        raise ValueError(
            "Missing JSON file: "
            f"{combined_dir_path / 'summary.json'}\n"
            "This looks like a validation report directory. "
            "Use `plot/combined_validation.py <validation_dir>` instead."
        )
    raise ValueError(f"Missing JSON file: {combined_dir_path / 'summary.json'}")


def _load_frontier_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing frontier CSV: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            state_hash = str(row.get("state_hash", "")).strip()
            if not state_hash:
                continue
            try:
                rows.append(
                    {
                        "state_hash": state_hash,
                        "latency_cycles": float(row["latency_cycles"]),
                        "mem_footprint_bytes": float(row["mem_footprint_bytes"]),
                        "energy": float(row["energy"]),
                        "weighted_cost": float(row.get("weighted_cost") or "inf"),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _metric_plot_value(row: dict[str, Any], metric_name: str) -> float:
    value = float(row[metric_name])
    if metric_name == "mem_footprint_bytes":
        return value / BYTES_PER_MB
    if metric_name == "energy":
        return value / PICOJOULES_PER_MILLIJOULE
    return value


def _output_path_for_frontier(base_output: Path, frontier_name: str) -> Path:
    return base_output.with_name(
        f"{base_output.stem}_{frontier_name}{base_output.suffix}"
    )


def _canonical_pdf_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.suffix.lower() == PDF_SUFFIX:
        return resolved
    return resolved.with_suffix(PDF_SUFFIX)


def _render_frontier_plot(
    *,
    frontier_rows: list[dict[str, Any]],
    frontier_name: str,
    x_metric: str,
    y_metric: str,
    title: str,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    sorted_rows = sorted(
        frontier_rows,
        key=lambda row: (
            float(row[x_metric]),
            float(row[y_metric]),
            str(row.get("state_hash", "")),
        ),
    )
    x_values = [_metric_plot_value(row, x_metric) for row in sorted_rows]
    y_values = [_metric_plot_value(row, y_metric) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(
        x_values,
        y_values,
        color="#264653",
        marker="o",
        markersize=4.5,
        linewidth=2.0,
    )
    ax.set_xlabel(AXIS_LABELS.get(x_metric, x_metric), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(AXIS_LABELS.get(y_metric, y_metric), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(f"{title} [{frontier_name}]", fontsize=TITLE_FONT_SIZE)
    _apply_metric_axis_formatting(ax=ax, x_metric=x_metric, y_metric=y_metric)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def _apply_metric_axis_formatting(
    *, ax: plt.Axes, x_metric: str, y_metric: str
) -> None:
    energy_formatter = ticker.FuncFormatter(_format_energy_tick)
    if x_metric == "energy":
        ax.xaxis.set_major_formatter(energy_formatter)
    if y_metric == "energy":
        ax.yaxis.set_major_formatter(energy_formatter)


def _format_energy_tick(value: float, _position: float) -> str:
    if abs(float(value)) < (0.5 / PICOJOULES_PER_MILLIJOULE):
        return "0"
    return (
        f"{float(value):.{ENERGY_TICK_DECIMALS}f}"
        .rstrip("0")
        .rstrip(".")
    )
