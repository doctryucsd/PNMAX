#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(os.environ.get("PNMAX_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_ROOT = Path(os.environ.get("PNMAX_RESULTS_ROOT", str(REPO_ROOT / "results")))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (
    FuncFormatter,
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    NullLocator,
)

STEP_ORDER: tuple[str, ...] = ("a", "b", "c", "d")
ARCH_ORDER: tuple[str, ...] = ("upmem", "hbm_pim")
CURRENT_MODE_ID = "current"
NON_STREAMING_MODE_ID = "non_streaming"
MODE_ORDER: tuple[str, ...] = (CURRENT_MODE_ID, NON_STREAMING_MODE_ID)
PARETO_FRONTIERS: tuple[tuple[str, str, str], ...] = (
    ("latency_mem", "latency_cycles", "mem_footprint_bytes"),
    ("latency_energy", "latency_cycles", "energy"),
    ("mem_energy", "mem_footprint_bytes", "energy"),
)
PARETO_FRONT_COLORS: tuple[str, ...] = (
    "#e76f51",
    "#f4a261",
    "#e9c46a",
    "#2a9d8f",
    "#264653",
)
STEP_COLORS: dict[str, str] = {
    step_id: PARETO_FRONT_COLORS[index]
    for index, step_id in enumerate(STEP_ORDER)
}
DEFAULT_STEP_COLOR = PARETO_FRONT_COLORS[-1]
STEP_MARKERS: dict[str, str] = {
    "a": "o",
    "b": "s",
    "c": "^",
    "d": "D",
}
DEFAULT_STEP_MARKER = "o"
MODE_LINESTYLES: dict[str, str] = {
    CURRENT_MODE_ID: "-",
    NON_STREAMING_MODE_ID: "--",
}
DEFAULT_MODE_LINESTYLE = "-"
MAX_WORKLOADS_PER_BAND = 5
SUBPLOT_WIDTH_IN = 3.6
MIN_FIGURE_HEIGHT_IN = 5.0
SUBPLOT_BOX_ASPECT = 9.0 / 16.0
FIGURE_LEFT = 0.09
FIGURE_RIGHT = 0.95
FIGURE_TOP = 0.95
FIGURE_BOTTOM = 0.1
INNER_BAND_WSPACE = 0.2
INNER_BAND_HSPACE = 0.2
INTER_BAND_HSPACE = 0.2
SUPXLABEL_Y = 0.03
SUPYLABEL_X = 0.02
LEGEND_BBOX_ANCHOR = (0.988, 0.2)
DEFAULT_MAJOR_TICKS = 4
REDUCED_MAJOR_TICKS = 3
MIN_MAJOR_TICKS = 2
DEFAULT_LOG_MAJOR_TICKS = 4
REDUCED_LOG_MAJOR_TICKS = 3
MIN_LOG_MAJOR_TICKS = 2
XTICK_ROTATION_DEGREES = 25.0
AXIS_LABELS: dict[str, str] = {
    "latency_cycles": "Latency (normalized to baseline)",
    "mem_footprint_bytes": "Memory footprint (normalized to baseline)",
    "energy": "Energy (normalized to baseline)",
}
BASELINE_LABEL = "UniNDP baseline"
DEFAULT_OPTIPIM_PROXY_CSV = RESULTS_ROOT / "fig09_pareto" / "optipim_proxy.csv"
OPTIPIM_LABEL = "OptiPIM"
OPTIPIM_MARKER_STYLE = dict(color="#2196F3", marker="*", s=120, linewidths=1.5, zorder=6)
# CINM (Cinnamon) compiler-tiling marker on the UPMEM subfigs (mirror of OptiPIM).
DEFAULT_CINM_PROXY_CSV = RESULTS_ROOT / "fig09_pareto" / "cinm_proxy.csv"
CINM_LABEL = "CINM"
CINM_MARKER_STYLE = dict(color="#C2185B", marker="D", s=80, linewidths=1.5, zorder=6)
MISSING_CURRENT_MODE_LABEL = "No current-mode data"


@dataclass(frozen=True)
class RunArtifact:
    arch: str
    workload: str
    workload_label: str
    run_dir: Path
    baseline_metrics: dict[str, Any]
    rows_by_frontier: dict[str, list[dict[str, Any]]]


def _display_step_label(raw_label: str) -> str:
    return raw_label.replace("+ cache", "+ buffer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render report-root Pareto grid figures with one subplot per "
            "(arch, workload) pair."
        )
    )
    parser.add_argument(
        "report_root",
        type=Path,
        help="Path to a workload-space Pareto evaluation output root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory (default: <report_root>/plots).",
    )
    parser.add_argument(
        "--optipim-proxy-csv",
        type=Path,
        default=DEFAULT_OPTIPIM_PROXY_CSV,
        help="OptiPIM proxy CSV (markers skipped when the file is absent).",
    )
    parser.add_argument(
        "--cinm-proxy-csv",
        type=Path,
        default=DEFAULT_CINM_PROXY_CSV,
        help="CINM proxy CSV (markers skipped when the file is absent).",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help=(
            "Comma-separated frontier pair names to render "
            "(latency_mem, latency_energy, mem_energy); default: all."
        ),
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving images.",
    )
    return parser.parse_args()


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


def _resolve_path(path_like: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _select_latest_run(workload_dir: Path) -> Path:
    run_dirs = sorted(
        path
        for path in workload_dir.iterdir()
        if path.is_dir() and _is_valid_run_dir(path)
    )
    if not run_dirs:
        raise ValueError(f"No run directories found under {workload_dir}.")
    return run_dirs[-1]


def _is_valid_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "summary.json").is_file() and (
        (path / "pareto_frontiers.csv").is_file() or (path / "manifest.json").is_file()
    )


def _candidate_workload_dirs(arch_dir: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for workload_dir in sorted(path for path in arch_dir.iterdir() if path.is_dir()):
        try:
            _select_latest_run(workload_dir)
        except ValueError:
            continue
        candidates.append(workload_dir)
    return tuple(candidates)


def _ordered_arch_names(arch_names: set[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            arch_names,
            key=lambda arch_name: (
                ARCH_ORDER.index(arch_name)
                if arch_name in ARCH_ORDER
                else len(ARCH_ORDER),
                arch_name,
            ),
        )
    )


def _discover_latest_runs(
    report_root: Path,
) -> tuple[dict[tuple[str, str], Path], tuple[str, ...], tuple[str, ...]]:
    if not report_root.is_dir():
        raise ValueError(f"Report root does not exist: {report_root}")

    arch_dirs = sorted(
        path
        for path in report_root.iterdir()
        if path.is_dir() and _candidate_workload_dirs(path)
    )
    if not arch_dirs:
        raise ValueError(f"No workload-space Pareto arch directories found under {report_root}.")

    workload_names_by_arch: dict[str, set[str]] = {}
    run_dirs: dict[tuple[str, str], Path] = {}
    discovered_workloads: set[str] = set()

    for arch_dir in arch_dirs:
        arch_name = arch_dir.name
        workload_dirs = list(_candidate_workload_dirs(arch_dir))
        if not workload_dirs:
            raise ValueError(f"No valid workload directories found under {arch_dir}.")

        workload_names_by_arch[arch_name] = {path.name for path in workload_dirs}
        discovered_workloads.update(workload_names_by_arch[arch_name])

    missing_pairs: list[str] = []
    for arch_name, workload_names in workload_names_by_arch.items():
        for workload_name in discovered_workloads:
            if workload_name not in workload_names:
                missing_pairs.append(f"{arch_name}/{workload_name}")
                continue
            workload_dir = report_root / arch_name / workload_name
            try:
                run_dirs[(arch_name, workload_name)] = _select_latest_run(workload_dir)
            except ValueError:
                missing_pairs.append(f"{arch_name}/{workload_name}")

    if missing_pairs:
        formatted = ", ".join(sorted(missing_pairs))
        raise ValueError(
            "Incomplete arch/workload matrix under "
            f"{report_root}; missing runs for: {formatted}"
        )

    ordered_archs = _ordered_arch_names({path.name for path in arch_dirs})
    ordered_workloads = tuple(sorted(discovered_workloads))
    return run_dirs, ordered_archs, ordered_workloads


def _load_frontier_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing Pareto CSV: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            step_id = str(row.get("step_id", "")).strip()
            frontier_name = str(row.get("frontier_name", "")).strip()
            if not frontier_name or not step_id:
                continue
            try:
                rows.append(
                    {
                        "frontier_name": frontier_name,
                        "x_metric": str(row.get("x_metric", "")).strip(),
                        "y_metric": str(row.get("y_metric", "")).strip(),
                        "step_id": step_id,
                        "base_step_id": str(row.get("base_step_id", "")).strip(),
                        "mode_id": str(row.get("mode_id", "")).strip(),
                        "mode_label": str(row.get("mode_label", "")).strip(),
                        "step_label": str(row.get("step_label", step_id)),
                        "order": int(row.get("order", 0) or 0),
                        "latency_cycles": float(row["latency_cycles"]),
                        "mem_footprint_bytes": float(row["mem_footprint_bytes"]),
                        "energy": float(row["energy"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _load_optipim_proxy(proxy_path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Load criterion=latency rows from the OptiPIM proxy CSV.

    Returns a dict keyed by (workload_short_name, streaming_flag) -> {norm_latency, norm_mem, norm_energy}.
    streaming_flag is "false" (streaming enabled) or "true" (streaming disabled).
    """
    result: dict[tuple[str, str], dict[str, float]] = {}
    if not proxy_path.is_file():
        return result
    with proxy_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("criterion", "")).strip() != "latency":
                continue
            workload = str(row.get("workload", "")).strip()
            streaming = str(row.get("streaming", "")).strip()
            try:
                result[(workload, streaming)] = {
                    "norm_latency": float(row["norm_latency"]),
                    "norm_mem": float(row["norm_mem"]),
                    "norm_energy": float(row["norm_energy"]),
                }
            except (KeyError, ValueError):
                continue
    return result


def _load_cinm_proxy(proxy_path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Load CINM compiler-tiling proxy rows (Approach A).

    Returns a dict keyed by (workload_short_name, streaming_flag) ->
    {norm_latency, norm_mem, norm_energy}. One row per workload (streaming "false",
    matching the grid's current-mode frontier, as the OptiPIM proxy does).
    """
    result: dict[tuple[str, str], dict[str, float]] = {}
    if not proxy_path.is_file():
        return result
    with proxy_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            workload = str(row.get("workload", "")).strip()
            streaming = str(row.get("streaming", "")).strip()
            try:
                result[(workload, streaming)] = {
                    "norm_latency": float(row["norm_latency"]),
                    "norm_mem": float(row["norm_mem"]),
                    "norm_energy": float(row["norm_energy"]),
                }
            except (KeyError, ValueError):
                continue
    return result


def _workload_short_name(workload_dir_name: str) -> str:
    """Strip op-type prefix (e.g. bmm_, conv2d_, fc_) to get the short name."""
    m = re.match(r"^[a-z0-9]+_(.+)$", workload_dir_name)
    return m.group(1) if m else workload_dir_name


def _mode_id(row: dict[str, Any]) -> str:
    mode_id = str(row.get("mode_id", "")).strip()
    if mode_id:
        return mode_id
    step_id = str(row.get("step_id", "")).strip()
    if step_id.startswith("ns_"):
        return NON_STREAMING_MODE_ID
    return CURRENT_MODE_ID


def _base_step_id(row: dict[str, Any]) -> str:
    base_step_id = str(row.get("base_step_id", "")).strip()
    if base_step_id:
        return base_step_id
    step_id = str(row.get("step_id", "")).strip()
    if step_id.startswith("ns_"):
        return step_id[3:]
    return step_id


def _series_sort_key(step_id: str, row: dict[str, Any]) -> tuple[int, int, str]:
    base_step_id = _base_step_id(row)
    mode_id = _mode_id(row)
    try:
        mode_index = MODE_ORDER.index(mode_id)
    except ValueError:
        mode_index = len(MODE_ORDER)
    try:
        base_index = STEP_ORDER.index(base_step_id)
    except ValueError:
        base_index = len(STEP_ORDER)
    return (mode_index, base_index, step_id)


def _plot_style(row: dict[str, Any]) -> dict[str, Any]:
    base_step_id = _base_step_id(row)
    mode_id = _mode_id(row)
    return {
        "color": STEP_COLORS.get(base_step_id, DEFAULT_STEP_COLOR),
        "marker": STEP_MARKERS.get(base_step_id, DEFAULT_STEP_MARKER),
        "markersize": 7,
        "linewidth": 2.0,
        "linestyle": MODE_LINESTYLES.get(mode_id, DEFAULT_MODE_LINESTYLE),
    }


def _normalized_metric_plot_value(
    *,
    row: dict[str, Any],
    metric_name: str,
    baseline_metrics: dict[str, Any],
    context: str,
) -> float:
    try:
        baseline_value = float(baseline_metrics[metric_name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing or invalid baseline metric '{metric_name}' for {context}."
        ) from exc
    if baseline_value == 0.0:
        raise ValueError(f"Baseline metric '{metric_name}' must be non-zero for {context}.")

    try:
        value = float(row[metric_name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid metric '{metric_name}' for {context}.") from exc
    return value / baseline_value


def _compact_tick_label(value: float, _position: float | None = None) -> str:
    numeric = float(value)
    if abs(numeric) < 1e-12:
        return "0"
    magnitude = abs(numeric)
    if magnitude >= 1e4 or magnitude < 1e-3:
        text = f"{numeric:.2e}"
        return (
            text.replace("e+0", "e")
            .replace("e-0", "e-")
            .replace("e+", "e")
        )
    return f"{numeric:.3g}"


def _configure_linear_axis_ticks(
    ax,
    *,
    axis_name: str,
    major_tick_count: int = DEFAULT_MAJOR_TICKS,
) -> None:
    axis = ax.xaxis if axis_name == "x" else ax.yaxis
    axis.set_major_locator(MaxNLocator(nbins=major_tick_count, min_n_ticks=2))
    axis.set_major_formatter(FuncFormatter(_compact_tick_label))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())
    if axis_name == "x":
        ax.tick_params(axis="x", which="minor", labelbottom=False, labeltop=False)
    else:
        ax.tick_params(axis="y", which="minor", labelleft=False, labelright=False)


def _configure_log_x_ticks(ax, *, major_tick_count: int = DEFAULT_LOG_MAJOR_TICKS) -> None:
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0,), numticks=major_tick_count)
    )
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=True))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", labelbottom=False, labeltop=False)


def _visible_tick_labels(labels: list[Any]) -> list[Any]:
    return [
        label
        for label in labels
        if label.get_visible() and str(label.get_text()).strip()
    ]


def _tick_labels_overlap(labels: list[Any], renderer: Any, *, axis: str) -> bool:
    visible_labels = _visible_tick_labels(labels)
    if len(visible_labels) < 2:
        return False

    boxes = [
        label.get_window_extent(renderer).expanded(1.02, 1.08)
        for label in visible_labels
    ]
    if axis == "x":
        boxes.sort(key=lambda box: box.x0)
        return any(curr.x0 < prev.x1 for prev, curr in zip(boxes, boxes[1:]))

    boxes.sort(key=lambda box: box.y0)
    return any(curr.y0 < prev.y1 for prev, curr in zip(boxes, boxes[1:]))


def _draw_renderer(fig) -> Any:
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _rotate_x_tick_labels(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_rotation(XTICK_ROTATION_DEGREES)
        label.set_ha("right")
        label.set_rotation_mode("anchor")


def _resolve_axis_tick_overlap(
    fig,
    ax,
    *,
    x_metric: str,
    y_metric: str,
) -> None:
    renderer = _draw_renderer(fig)

    if _tick_labels_overlap(ax.get_xticklabels(), renderer, axis="x"):
        if x_metric == "latency_cycles":
            _configure_log_x_ticks(ax, major_tick_count=REDUCED_LOG_MAJOR_TICKS)
        else:
            _configure_linear_axis_ticks(
                ax,
                axis_name="x",
                major_tick_count=REDUCED_MAJOR_TICKS,
            )
        renderer = _draw_renderer(fig)

    if _tick_labels_overlap(ax.get_yticklabels(), renderer, axis="y"):
        _configure_linear_axis_ticks(
            ax,
            axis_name="y",
            major_tick_count=REDUCED_MAJOR_TICKS,
        )
        renderer = _draw_renderer(fig)

    if _tick_labels_overlap(ax.get_xticklabels(), renderer, axis="x"):
        if x_metric == "latency_cycles":
            _configure_log_x_ticks(ax, major_tick_count=MIN_LOG_MAJOR_TICKS)
        else:
            _configure_linear_axis_ticks(
                ax,
                axis_name="x",
                major_tick_count=MIN_MAJOR_TICKS,
            )
        renderer = _draw_renderer(fig)

    if _tick_labels_overlap(ax.get_yticklabels(), renderer, axis="y"):
        _configure_linear_axis_ticks(
            ax,
            axis_name="y",
            major_tick_count=MIN_MAJOR_TICKS,
        )
        renderer = _draw_renderer(fig)

    if _tick_labels_overlap(ax.get_xticklabels(), renderer, axis="x"):
        _rotate_x_tick_labels(ax)
        _draw_renderer(fig)


def _load_artifact(run_dir: Path, *, arch: str, workload: str) -> RunArtifact:
    summary = _load_json(run_dir / "summary.json")
    pareto_csv = _resolve_path(
        str(
            summary.get("pairwise_frontiers_csv")
            or summary.get("pareto_front_csv")
            or (run_dir / "pareto_frontiers.csv")
        ),
        base_dir=run_dir,
    )
    rows = _load_frontier_rows(pareto_csv)
    rows_by_frontier: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_frontier.setdefault(str(row["frontier_name"]), []).append(row)

    baseline_metrics: dict[str, Any] = {}
    raw_baseline_path = summary.get("baseline_json")
    baseline_path = (
        _resolve_path(str(raw_baseline_path), base_dir=run_dir)
        if raw_baseline_path
        else (run_dir / "baseline.json").resolve()
    )
    if baseline_path.is_file():
        baseline_payload = _load_json(baseline_path)
        raw_metrics = baseline_payload.get("metrics")
        if raw_metrics is not None:
            if not isinstance(raw_metrics, dict):
                raise ValueError(
                    f"Expected baseline metrics mapping in {baseline_path}."
                )
            baseline_metrics = dict(raw_metrics)

    workload_label = str(summary.get("workload_display_name") or workload).strip()
    if not workload_label:
        workload_label = workload

    return RunArtifact(
        arch=arch,
        workload=workload,
        workload_label=workload_label,
        run_dir=run_dir,
        baseline_metrics=baseline_metrics,
        rows_by_frontier=rows_by_frontier,
    )


def _load_artifacts(
    run_dirs: dict[tuple[str, str], Path],
) -> dict[tuple[str, str], RunArtifact]:
    artifacts: dict[tuple[str, str], RunArtifact] = {}
    for (arch, workload), run_dir in sorted(run_dirs.items()):
        artifacts[(arch, workload)] = _load_artifact(
            run_dir,
            arch=arch,
            workload=workload,
        )
    return artifacts


def _resolve_workload_labels(
    *,
    workload_names: tuple[str, ...],
    arch_names: tuple[str, ...],
    artifacts: dict[tuple[str, str], RunArtifact],
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for workload_name in workload_names:
        seen_labels = {
            artifacts[(arch_name, workload_name)].workload_label
            for arch_name in arch_names
            if artifacts[(arch_name, workload_name)].workload_label.strip()
        }
        if not seen_labels:
            labels[workload_name] = workload_name
            continue
        if len(seen_labels) > 1:
            formatted = ", ".join(sorted(seen_labels))
            raise ValueError(
                f"Conflicting workload labels for {workload_name}: {formatted}"
            )
        labels[workload_name] = next(iter(seen_labels))
    return labels


def _workload_bands(
    workload_names: tuple[str, ...], *, max_band_width: int = MAX_WORKLOADS_PER_BAND
) -> tuple[tuple[str, ...], ...]:
    if max_band_width <= 0:
        raise ValueError("max_band_width must be positive.")
    if not workload_names:
        return tuple()
    return tuple(
        tuple(workload_names[index : index + max_band_width])
        for index in range(0, len(workload_names), max_band_width)
    )


def _frontier_current_rows(
    artifact: RunArtifact, frontier_name: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in artifact.rows_by_frontier.get(frontier_name, [])
        if _mode_id(row) == CURRENT_MODE_ID
    ]


def _build_legend_handles(
    *,
    artifacts: dict[tuple[str, str], RunArtifact],
    arch_names: tuple[str, ...],
    workload_names: tuple[str, ...],
    frontier_name: str,
    optipim_proxy: dict[tuple[str, str], dict[str, float]] | None = None,
    cinm_proxy: dict[tuple[str, str], dict[str, float]] | None = None,
) -> list[Line2D]:
    handles: list[Line2D] = []
    if any(
        artifacts[(arch_name, workload_name)].baseline_metrics
        for arch_name in arch_names
        for workload_name in workload_names
    ):
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker="x",
                linestyle="None",
                markersize=8.0,
                markeredgewidth=2.0,
                label=BASELINE_LABEL,
            )
        )

    representatives: dict[str, tuple[tuple[int, int, str], dict[str, Any]]] = {}
    for arch_name in arch_names:
        for workload_name in workload_names:
            for row in _frontier_current_rows(
                artifacts[(arch_name, workload_name)],
                frontier_name,
            ):
                label = str(row.get("step_label", row.get("step_id", ""))).strip()
                step_id = str(row.get("step_id", "")).strip()
                if not label or not step_id:
                    continue
                sort_key = _series_sort_key(step_id, row)
                existing = representatives.get(label)
                if existing is None or sort_key < existing[0]:
                    representatives[label] = (sort_key, row)

    for _, row in sorted(representatives.values(), key=lambda item: item[0]):
        style = _plot_style(row)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                label=_display_step_label(
                    str(row.get("step_label", row.get("step_id", "")))
                ),
            )
        )
    if optipim_proxy:
        handles.append(
            Line2D(
                [0],
                [0],
                color=OPTIPIM_MARKER_STYLE["color"],
                marker=OPTIPIM_MARKER_STYLE["marker"],
                linestyle="None",
                markersize=10.0,
                label=OPTIPIM_LABEL,
            )
        )
    if cinm_proxy:
        handles.append(
            Line2D(
                [0],
                [0],
                color=CINM_MARKER_STYLE["color"],
                marker=CINM_MARKER_STYLE["marker"],
                linestyle="None",
                markersize=9.0,
                label=CINM_LABEL,
            )
        )
    return handles


def _render_subplot(
    *,
    ax,
    artifact: RunArtifact,
    frontier_name: str,
    x_metric: str,
    y_metric: str,
    optipim_point: dict[str, float] | None = None,
    cinm_point: dict[str, float] | None = None,
) -> None:
    context = f"{artifact.arch}/{artifact.workload}"
    if not artifact.baseline_metrics:
        raise ValueError(f"Missing baseline metrics for {context}.")

    if artifact.baseline_metrics:
        baseline_x = _normalized_metric_plot_value(
            row=artifact.baseline_metrics,
            metric_name=x_metric,
            baseline_metrics=artifact.baseline_metrics,
            context=context,
        )
        baseline_y = _normalized_metric_plot_value(
            row=artifact.baseline_metrics,
            metric_name=y_metric,
            baseline_metrics=artifact.baseline_metrics,
            context=context,
        )
        ax.scatter(
            [baseline_x],
            [baseline_y],
            color="black",
            marker="x",
            s=80,
            linewidths=2.0,
            label=BASELINE_LABEL,
            zorder=5,
        )

        if optipim_point is not None:
            metric_map = {
                "latency_cycles": optipim_point["norm_latency"],
                "mem_footprint_bytes": optipim_point["norm_mem"],
                "energy": optipim_point["norm_energy"],
            }
            optipim_x = metric_map.get(x_metric)
            optipim_y = metric_map.get(y_metric)
            if optipim_x is not None and optipim_y is not None:
                ax.scatter(
                    [optipim_x],
                    [optipim_y],
                    label=OPTIPIM_LABEL,
                    **OPTIPIM_MARKER_STYLE,
                )

        if cinm_point is not None:
            cinm_map = {
                "latency_cycles": cinm_point["norm_latency"],
                "mem_footprint_bytes": cinm_point["norm_mem"],
                "energy": cinm_point["norm_energy"],
            }
            cinm_x = cinm_map.get(x_metric)
            cinm_y = cinm_map.get(y_metric)
            if cinm_x is not None and cinm_y is not None:
                ax.scatter(
                    [cinm_x],
                    [cinm_y],
                    label=CINM_LABEL,
                    **CINM_MARKER_STYLE,
                )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _frontier_current_rows(artifact, frontier_name):
        grouped.setdefault(str(row["step_id"]), []).append(row)

    ordered_step_ids = sorted(
        grouped,
        key=lambda step_id: _series_sort_key(step_id, grouped[step_id][0]),
    )
    for step_id in ordered_step_ids:
        points = grouped[step_id]
        ordered = sorted(
            points,
            key=lambda row: (
                int(row["order"]),
                _normalized_metric_plot_value(
                    row=row,
                    metric_name=x_metric,
                    baseline_metrics=artifact.baseline_metrics,
                    context=context,
                ),
                _normalized_metric_plot_value(
                    row=row,
                    metric_name=y_metric,
                    baseline_metrics=artifact.baseline_metrics,
                    context=context,
                ),
            ),
        )
        label = _display_step_label(str(ordered[0].get("step_label", step_id)))
        x_values = [
            _normalized_metric_plot_value(
                row=row,
                metric_name=x_metric,
                baseline_metrics=artifact.baseline_metrics,
                context=context,
            )
            for row in ordered
        ]
        y_values = [
            _normalized_metric_plot_value(
                row=row,
                metric_name=y_metric,
                baseline_metrics=artifact.baseline_metrics,
                context=context,
            )
            for row in ordered
        ]
        ax.plot(
            x_values,
            y_values,
            label=label,
            **_plot_style(ordered[0]),
        )

    if not grouped:
        ax.text(
            0.5,
            0.5,
            MISSING_CURRENT_MODE_LABEL,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.0,
            color="#555555",
        )

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
def _validate_positive_layout_factor(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive; got {value}.")


def _figure_size(num_cols: int, arch_count: int, band_count: int) -> tuple[float, float]:
    if num_cols <= 0:
        raise ValueError(f"num_cols must be positive; got {num_cols}.")
    if arch_count <= 0:
        raise ValueError(f"arch_count must be positive; got {arch_count}.")
    if band_count <= 0:
        raise ValueError(f"band_count must be positive; got {band_count}.")
    if FIGURE_RIGHT <= FIGURE_LEFT:
        raise ValueError(
            f"FIGURE_RIGHT must be greater than FIGURE_LEFT; got {FIGURE_RIGHT} <= {FIGURE_LEFT}."
        )
    if FIGURE_TOP <= FIGURE_BOTTOM:
        raise ValueError(
            f"FIGURE_TOP must be greater than FIGURE_BOTTOM; got {FIGURE_TOP} <= {FIGURE_BOTTOM}."
        )
    if SUBPLOT_BOX_ASPECT <= 0.0:
        raise ValueError(
            f"SUBPLOT_BOX_ASPECT must be positive; got {SUBPLOT_BOX_ASPECT}."
        )

    width = max((SUBPLOT_WIDTH_IN * float(num_cols)) + 2.8, 10.5)
    inner_width = width * (FIGURE_RIGHT - FIGURE_LEFT)

    width_layout_factor = float(num_cols) + (
        INNER_BAND_WSPACE * float(max(num_cols - 1, 0))
    )
    _validate_positive_layout_factor("inner band width layout factor", width_layout_factor)

    inner_band_height_factor = float(arch_count) + (
        INNER_BAND_HSPACE * float(max(arch_count - 1, 0))
    )
    _validate_positive_layout_factor(
        "inner band height layout factor",
        inner_band_height_factor,
    )

    inter_band_height_factor = float(band_count) + (
        INTER_BAND_HSPACE * float(max(band_count - 1, 0))
    )
    _validate_positive_layout_factor(
        "inter band height layout factor",
        inter_band_height_factor,
    )

    axis_width = inner_width / width_layout_factor
    axis_height = axis_width * SUBPLOT_BOX_ASPECT
    band_height = axis_height * inner_band_height_factor
    stacked_height = band_height * inter_band_height_factor
    height = max(
        stacked_height / (FIGURE_TOP - FIGURE_BOTTOM),
        MIN_FIGURE_HEIGHT_IN,
    )
    return (width, height)


def _save_figure(
    *,
    fig,
    output_path: Path,
    dpi: int,
    show: bool,
    message: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"{message} {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def _render_frontier_grid(
    *,
    artifacts: dict[tuple[str, str], RunArtifact],
    arch_names: tuple[str, ...],
    workload_names: tuple[str, ...],
    workload_labels: dict[str, str],
    frontier_name: str,
    x_metric: str,
    y_metric: str,
    output_path: Path,
    dpi: int,
    show: bool,
    optipim_proxy: dict[tuple[str, str], dict[str, float]] | None = None,
    cinm_proxy: dict[tuple[str, str], dict[str, float]] | None = None,
) -> None:
    workload_bands = _workload_bands(workload_names)
    if not workload_bands:
        raise ValueError("At least one workload is required to render the grid.")

    arch_count = len(arch_names)
    band_count = len(workload_bands)
    num_cols = max(len(band) for band in workload_bands)
    fig = plt.figure(figsize=_figure_size(num_cols, arch_count, band_count))
    outer_grid: GridSpec = fig.add_gridspec(
        band_count,
        1,
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        top=FIGURE_TOP,
        bottom=FIGURE_BOTTOM,
        hspace=INTER_BAND_HSPACE,
    )
    axes: list[list[Any]] = []

    for band_index, workload_band in enumerate(workload_bands):
        band_grid = outer_grid[band_index].subgridspec(
            arch_count,
            num_cols,
            wspace=INNER_BAND_WSPACE,
            hspace=INNER_BAND_HSPACE,
        )
        for arch_index, arch_name in enumerate(arch_names):
            row_index = band_index * arch_count + arch_index
            row_axes: list[Any] = []
            for col_index in range(num_cols):
                ax = fig.add_subplot(band_grid[arch_index, col_index])
                row_axes.append(ax)
                if col_index >= len(workload_band):
                    ax.set_visible(False)
                    continue

                workload_name = workload_band[col_index]
                artifact = artifacts[(arch_name, workload_name)]
                optipim_point = None
                cinm_point = None
                short_name = _workload_short_name(workload_name)
                if arch_name == "hbm_pim":
                    optipim_point = (
                        optipim_proxy.get((short_name, "false"))
                        if optipim_proxy
                        else None
                    )
                elif arch_name == "upmem":
                    cinm_point = (
                        cinm_proxy.get((short_name, "false"))
                        if cinm_proxy
                        else None
                    )
                _render_subplot(
                    ax=ax,
                    artifact=artifact,
                    frontier_name=frontier_name,
                    x_metric=x_metric,
                    y_metric=y_metric,
                    optipim_point=optipim_point,
                    cinm_point=cinm_point,
                )
                if col_index == 0:
                    ax.text(
                        -0.25,
                        0.5,
                        arch_name,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=18.0,
                    )
                if arch_index == 0:
                    ax.set_title(
                        workload_labels[workload_name],
                        fontsize=18.0,
                        pad=10.0,
                    )
                if x_metric == "latency_cycles":
                    ax.set_xscale("log")
                    _configure_log_x_ticks(ax)
                else:
                    _configure_linear_axis_ticks(
                        ax,
                        axis_name="x",
                        major_tick_count=DEFAULT_MAJOR_TICKS,
                    )
                _configure_linear_axis_ticks(
                    ax,
                    axis_name="y",
                    major_tick_count=DEFAULT_MAJOR_TICKS,
                )
                ax.set_box_aspect(SUBPLOT_BOX_ASPECT)
                ax.tick_params(axis="both", labelsize=13.0)
            axes.append(row_axes)

    fig.supxlabel(AXIS_LABELS[x_metric], y=SUPXLABEL_Y, fontsize=22.0)
    fig.supylabel(AXIS_LABELS[y_metric], x=SUPYLABEL_X, fontsize=22.0)

    legend_handles = _build_legend_handles(
        artifacts=artifacts,
        arch_names=arch_names,
        workload_names=workload_names,
        frontier_name=frontier_name,
        optipim_proxy=optipim_proxy,
        cinm_proxy=cinm_proxy,
    )
    if legend_handles:
        baseline_handles = [
            handle
            for handle in legend_handles
            if handle.get_label() in (BASELINE_LABEL, OPTIPIM_LABEL, CINM_LABEL)
        ]
        step_handles = [
            handle
            for handle in legend_handles
            if handle.get_label() not in (BASELINE_LABEL, OPTIPIM_LABEL, CINM_LABEL)
        ]
        if baseline_handles and step_handles:
            baseline_legend = fig.legend(
                handles=baseline_handles,
                loc="lower right",
                bbox_to_anchor=(
                    LEGEND_BBOX_ANCHOR[0],
                    LEGEND_BBOX_ANCHOR[1] + 0.18,
                ),
                ncol=1,
                frameon=False,
                fontsize=18.0,
            )
            fig.add_artist(baseline_legend)
            fig.legend(
                handles=step_handles,
                loc="lower right",
                bbox_to_anchor=LEGEND_BBOX_ANCHOR,
                ncol=1,
                frameon=False,
                fontsize=18.0,
            )
        elif step_handles:
            fig.legend(
                handles=step_handles,
                loc="lower right",
                bbox_to_anchor=LEGEND_BBOX_ANCHOR,
                ncol=1,
                frameon=False,
                fontsize=18.0,
            )
        else:
            fig.legend(
                handles=baseline_handles,
                loc="lower right",
                bbox_to_anchor=(
                    LEGEND_BBOX_ANCHOR[0],
                    LEGEND_BBOX_ANCHOR[1] + 0.18,
                ),
                ncol=1,
                frameon=False,
                fontsize=18.0,
            )

    for row_axes in axes:
        for ax in row_axes:
            if not ax.get_visible():
                continue
            _resolve_axis_tick_overlap(
                fig,
                ax,
                x_metric=x_metric,
                y_metric=y_metric,
            )

    _save_figure(
        fig=fig,
        output_path=output_path,
        dpi=dpi,
        show=show,
        message="Saved workload-space Pareto grid to",
    )


def main() -> int:
    args = parse_args()
    report_root = args.report_root.resolve()
    run_dirs, arch_names, workload_names = _discover_latest_runs(report_root)
    artifacts = _load_artifacts(run_dirs)
    optipim_proxy = _load_optipim_proxy(args.optipim_proxy_csv)
    cinm_proxy = _load_cinm_proxy(args.cinm_proxy_csv)
    workload_labels = _resolve_workload_labels(
        workload_names=workload_names,
        arch_names=arch_names,
        artifacts=artifacts,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (report_root / "plots").resolve()
    )

    if args.pairs is not None:
        selected = {name.strip() for name in args.pairs.split(",") if name.strip()}
        known = {name for name, _, _ in PARETO_FRONTIERS}
        unknown = selected - known
        if unknown:
            raise ValueError(
                f"unknown --pairs entries {sorted(unknown)}; known: {sorted(known)}"
            )
        frontiers = tuple(f for f in PARETO_FRONTIERS if f[0] in selected)
    else:
        frontiers = PARETO_FRONTIERS

    for frontier_name, x_metric, y_metric in frontiers:
        _render_frontier_grid(
            artifacts=artifacts,
            arch_names=arch_names,
            workload_names=workload_names,
            workload_labels=workload_labels,
            frontier_name=frontier_name,
            x_metric=x_metric,
            y_metric=y_metric,
            output_path=output_dir / f"workload_space_pareto_grid_{frontier_name}.pdf",
            dpi=args.dpi,
            show=args.show,
            optipim_proxy=optipim_proxy,
            cinm_proxy=cinm_proxy,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
