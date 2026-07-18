#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

ARCH_NAME = "hbm_pim"
CURRENT_MODE_ID = "current"
FRONTIER_ORDER: tuple[str, ...] = ("latency_mem", "latency_energy", "mem_energy")
STEP_ORDER: tuple[str, ...] = ("a", "b", "c", "d")
STEP_TICK_LABELS: dict[str, str] = {
    "a": "(a)",
    "b": "(b)",
    "c": "(c)",
    "d": "(d)",
}
STEP_HATCHES: dict[str, str] = {
    "a": "",
    "b": "/",
    "c": "x",
    "d": ".",
}
COMPONENT_ORDER: tuple[str, ...] = (
    "streaming",
    "load from bank to pu",
    "pu_compute",
    "store to bank from pu",
    "row change",
    "bank to bank transfer",
    "output save",
)
COMPONENT_COLORS: dict[str, str] = {
    "streaming": "#4C78A8",
    "load from bank to pu": "#F58518",
    "pu_compute": "#E45756",
    "store to bank from pu": "#72B7B2",
    "row change": "#54A24B",
    "bank to bank transfer": "#EECA3B",
    "output save": "#B279A2",
}
# Display overrides for the legend (data keys above stay unchanged).
COMPONENT_LEGEND_LABELS: dict[str, str] = {
    "load from bank to pu": "load: bank → pu",
    "store to bank from pu": "store: pu → bank",
}
DEFAULT_OUTPUT_FILENAME = "workload_space_pareto_latency_breakdown_hbm_pim_streaming_only.pdf"
STEP_A_NORMALIZED_OUTPUT_FILENAME = (
    "workload_space_pareto_latency_breakdown_hbm_pim_streaming_only_"
    "normalized_to_a.pdf"
)
NORMALIZE_PER_BAR = "per_bar"
NORMALIZE_TO_STEP_A = "step_a"
MIN_FIGURE_WIDTH_IN = 12.0
FIGURE_WIDTH_PER_WORKLOAD_IN = 0.8
FIGURE_HEIGHT_IN = 9.5
BAR_WIDTH = 0.008
STEP_SPACING = 0.01
WORKLOAD_GAP = 0.005
OUTER_X_PADDING = 0.005
WORKLOAD_LABEL_Y = -0.05
WORKLOAD_LABEL_ROTATION_DEGREES = 12.0
FIGURE_LEFT = 0.08
FIGURE_RIGHT = 0.985
FIGURE_TOP = 0.84
FIGURE_BOTTOM = 0.14
GRID_ALPHA = 0.35
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 0.6
WORKLOAD_SEPARATOR_COLOR = "#C8C8C8"
WORKLOAD_SEPARATOR_LINEWIDTH = 0.8
WORKLOAD_SEPARATOR_LINESTYLE = ":"
BAR_EDGE_COLOR = "#303030"
BAR_EDGE_LINEWIDTH = 0.45
TICK_LABEL_FONT_SIZE = 27.0
WORKLOAD_LABEL_FONT_SIZE = 27.0
LEGEND_FONT_SIZE = 27.0
LEGEND_TITLE_FONT_SIZE = 27.0
STEP_LABEL_FONT_SIZE = 25.0  # (a)/(b)/(c)/(d) drawn on top of each bar
MAX_COMPONENT_RESIDUAL_SHARE = 1e-4
SAVEFIG_PAD_INCHES = 0.01


@dataclass(frozen=True)
class PlotBar:
    workload_name: str
    workload_label: str
    step_id: str
    component_shares: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot streaming-only hbm_pim workload-space Pareto latency "
            "breakdowns as normalized stacked bars."
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
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument(
        "--normalized-only",
        action="store_true",
        help=(
            "Render only the (a)-normalized figure (Fig. 11); "
            "skip the per-bar-normalized variant."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the PDF.",
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


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _load_json(path)


def _resolve_path(path_like: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _is_valid_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "summary.json").is_file() or (
        (path / "pareto_frontiers.csv").is_file()
        and (path / "evaluations.csv").is_file()
    )


def _select_latest_run(workload_dir: Path) -> Path:
    run_dirs = sorted(
        path
        for path in workload_dir.iterdir()
        if path.is_dir() and _is_valid_run_dir(path)
    )
    if not run_dirs:
        raise ValueError(f"No run directories found under {workload_dir}.")
    return run_dirs[-1]


def _discover_latest_runs(report_root: Path) -> dict[str, Path]:
    arch_dir = report_root / ARCH_NAME
    if not arch_dir.is_dir():
        raise ValueError(f"Report root missing {ARCH_NAME} directory: {arch_dir}")

    selected: dict[str, Path] = {}
    for workload_dir in sorted(path for path in arch_dir.iterdir() if path.is_dir()):
        if workload_dir.name == "plots":
            continue
        selected[workload_dir.name] = _select_latest_run(workload_dir)

    if not selected:
        raise ValueError(f"No {ARCH_NAME} workload runs found under {report_root}.")
    return selected


def _resolve_existing_frontier_csv(run_dir: Path, summary: dict[str, Any]) -> Path:
    checked_candidates: list[Path] = []
    for key in ("pairwise_frontiers_csv", "pareto_front_csv"):
        raw_path = summary.get(key)
        if not raw_path:
            continue
        candidate = _resolve_path(str(raw_path), base_dir=run_dir)
        checked_candidates.append(candidate)
        if candidate.is_file():
            return candidate

    default_candidate = (run_dir / "pareto_frontiers.csv").resolve()
    checked_candidates.append(default_candidate)
    if default_candidate.is_file():
        return default_candidate

    formatted = ", ".join(str(path) for path in checked_candidates)
    raise ValueError(f"Unable to resolve Pareto CSV for {run_dir}; checked: {formatted}")


def _resolve_existing_evaluations_csv(run_dir: Path, summary: dict[str, Any]) -> Path:
    checked_candidates: list[Path] = []
    raw_path = summary.get("evaluations_csv")
    if raw_path:
        candidate = _resolve_path(str(raw_path), base_dir=run_dir)
        checked_candidates.append(candidate)
        if candidate.is_file():
            return candidate

    default_candidate = (run_dir / "evaluations.csv").resolve()
    checked_candidates.append(default_candidate)
    if default_candidate.is_file():
        return default_candidate

    formatted = ", ".join(str(path) for path in checked_candidates)
    raise ValueError(
        f"Unable to resolve evaluations CSV for {run_dir}; checked: {formatted}"
    )


def _mode_id(row: dict[str, Any]) -> str:
    mode_id = str(row.get("mode_id", "")).strip()
    if mode_id:
        return mode_id
    step_id = str(row.get("step_id", "")).strip()
    if step_id.startswith("ns_"):
        return "non_streaming"
    return CURRENT_MODE_ID


def _base_step_id(row: dict[str, Any]) -> str:
    base_step_id = str(row.get("base_step_id", "")).strip()
    if base_step_id:
        return base_step_id
    step_id = str(row.get("step_id", "")).strip()
    if step_id.startswith("ns_"):
        return step_id[3:]
    return step_id


def _coerce_float(value: Any, *, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric value for {context}.") from exc


def _coerce_int(value: Any, *, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer value for {context}.") from exc


def _load_frontier_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing Pareto CSV: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            frontier_name = str(row.get("frontier_name", "")).strip()
            step_id = str(row.get("step_id", "")).strip()
            if not frontier_name or not step_id:
                continue
            try:
                rows.append(
                    {
                        "frontier_name": frontier_name,
                        "step_id": step_id,
                        "base_step_id": str(row.get("base_step_id", "")).strip(),
                        "mode_id": str(row.get("mode_id", "")).strip(),
                        "order": _coerce_int(row.get("order", 0) or 0, context="order"),
                        "latency_cycles": _coerce_float(
                            row["latency_cycles"], context="latency_cycles"
                        ),
                        "mem_footprint_bytes": _coerce_float(
                            row["mem_footprint_bytes"],
                            context="mem_footprint_bytes",
                        ),
                        "energy": _coerce_float(row["energy"], context="energy"),
                        "state_hash": str(row.get("state_hash", "")).strip(),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def _load_evaluation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing evaluations CSV: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("status", "")).strip() != "ok":
                continue
            state_hash = str(row.get("state_hash", "")).strip()
            if not state_hash:
                continue
            try:
                rows.append(
                    {
                        "step_id": str(row.get("step_id", "")).strip(),
                        "base_step_id": str(row.get("base_step_id", "")).strip(),
                        "mode_id": str(row.get("mode_id", "")).strip(),
                        "attempt_index": _coerce_int(
                            row.get("attempt_index", 0) or 0,
                            context="attempt_index",
                        ),
                        "latency_cycles": _coerce_float(
                            row["latency_cycles"], context="latency_cycles"
                        ),
                        "mem_footprint_bytes": _coerce_float(
                            row["mem_footprint_bytes"],
                            context="mem_footprint_bytes",
                        ),
                        "energy": _coerce_float(row["energy"], context="energy"),
                        "state_hash": state_hash,
                        "breakdown_json": str(row.get("breakdown_json", "")),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def _frontier_selection_key(
    row: dict[str, Any],
) -> tuple[float, float, float, int, str, str]:
    return (
        float(row["latency_cycles"]),
        float(row["mem_footprint_bytes"]),
        float(row["energy"]),
        int(row.get("order", 0)),
        str(row.get("state_hash", "")).strip(),
        str(row.get("frontier_name", "")).strip(),
    )


def _frontier_row_dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    state_hash = str(row.get("state_hash", "")).strip()
    if state_hash:
        return ("state_hash", state_hash)
    return (
        "metrics",
        float(row["latency_cycles"]),
        float(row["mem_footprint_bytes"]),
        float(row["energy"]),
    )


def _dedupe_frontier_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        dedupe_key = _frontier_row_dedupe_key(row)
        existing = deduped.get(dedupe_key)
        if existing is None or _frontier_selection_key(row) < _frontier_selection_key(
            existing
        ):
            deduped[dedupe_key] = dict(row)
    return list(deduped.values())


def _select_best_rows_across_frontiers(
    rows: list[dict[str, Any]], *, context: str
) -> dict[str, dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    observed_frontiers: set[str] = set()
    for row in rows:
        frontier_name = str(row.get("frontier_name", "")).strip()
        step_id = _base_step_id(row)
        if frontier_name not in FRONTIER_ORDER or step_id not in STEP_ORDER:
            continue
        if _mode_id(row) != CURRENT_MODE_ID:
            continue
        observed_frontiers.add(frontier_name)
        grouped_rows.setdefault(step_id, []).append(row)

    missing_frontiers = [
        frontier_name
        for frontier_name in FRONTIER_ORDER
        if frontier_name not in observed_frontiers
    ]
    if missing_frontiers:
        formatted = ", ".join(missing_frontiers)
        raise ValueError(f"Missing streaming Pareto frontiers for {context}: {formatted}.")

    selected: dict[str, dict[str, Any]] = {}
    for step_id in STEP_ORDER:
        candidates = grouped_rows.get(step_id)
        if not candidates:
            raise ValueError(f"Missing streaming Pareto rows for {context}/{step_id}.")
        deduped_candidates = _dedupe_frontier_rows(candidates)
        selected[step_id] = min(deduped_candidates, key=_frontier_selection_key)
    return selected


def _evaluation_match_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("state_hash", "")).strip(),
        _base_step_id(row),
        _mode_id(row),
    )


def _evaluation_selection_key(
    row: dict[str, Any],
) -> tuple[float, float, float, int, str]:
    return (
        float(row["latency_cycles"]),
        float(row["mem_footprint_bytes"]),
        float(row["energy"]),
        int(row.get("attempt_index", 0)),
        str(row.get("state_hash", "")).strip(),
    )


def _build_evaluation_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        index.setdefault(_evaluation_match_key(row), []).append(row)
    return index


def _metrics_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        abs(float(left["latency_cycles"]) - float(right["latency_cycles"])) <= 1e-9
        and abs(float(left["mem_footprint_bytes"]) - float(right["mem_footprint_bytes"]))
        <= 1e-9
        and abs(float(left["energy"]) - float(right["energy"])) <= 1e-9
    )


def _match_evaluation_row(
    frontier_row: dict[str, Any],
    evaluation_index: dict[tuple[str, str, str], list[dict[str, Any]]],
    *,
    context: str,
) -> dict[str, Any]:
    match_key = _evaluation_match_key(frontier_row)
    candidates = list(evaluation_index.get(match_key, ()))
    if not candidates:
        raise ValueError(
            "Missing evaluation row for "
            f"{context} with key {match_key!r}."
        )

    metric_matches = [row for row in candidates if _metrics_match(row, frontier_row)]
    if metric_matches:
        candidates = metric_matches
    return min(candidates, key=_evaluation_selection_key)


def _load_breakdown_dict(row: dict[str, Any], *, context: str) -> dict[str, Any]:
    raw_payload = str(row.get("breakdown_json", "")).strip()
    if not raw_payload:
        raise ValueError(f"Missing breakdown_json for {context}.")
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid breakdown_json for {context}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping breakdown_json for {context}.")
    return payload


def _breakdown_value(
    breakdown: dict[str, Any], key: str, *, context: str
) -> float:
    if key not in breakdown:
        raise ValueError(f"Missing breakdown key '{key}' for {context}.")
    value = _coerce_float(breakdown[key], context=f"{context}/{key}")
    if value < 0.0:
        raise ValueError(f"Negative breakdown value '{key}' for {context}.")
    return value


def _component_cycles(row: dict[str, Any], *, context: str) -> dict[str, float]:
    breakdown = _load_breakdown_dict(row, context=context)
    return {
        "streaming": _breakdown_value(
            breakdown, "streaming (cycles)", context=context
        ),
        "load from bank to pu": _breakdown_value(
            breakdown, "compute local bank load (cycles)", context=context
        ),
        "pu_compute": _breakdown_value(
            breakdown, "compute arithmetic (cycles)", context=context
        ),
        "store to bank from pu": _breakdown_value(
            breakdown, "compute bank save (cycles)", context=context
        ),
        "row change": _breakdown_value(
            breakdown, "compute row-change load (cycles)", context=context
        )
        + _breakdown_value(breakdown, "save row-change (cycles)", context=context),
        "bank to bank transfer": _breakdown_value(
            breakdown, "compute inter-bank load (cycles)", context=context
        ),
        "output save": _breakdown_value(
            breakdown, "save transfer (cycles)", context=context
        ),
    }


def _normalized_component_shares(
    row: dict[str, Any], *, context: str, baseline_latency_cycles: float
) -> dict[str, float]:
    latency_cycles = float(row["latency_cycles"])
    if latency_cycles <= 0.0:
        raise ValueError(f"Latency must be positive for {context}.")
    if baseline_latency_cycles <= 0.0:
        raise ValueError(f"Baseline latency must be positive for {context}.")

    cycles = _component_cycles(row, context=context)
    shares = {
        component_name: cycles[component_name] / baseline_latency_cycles
        for component_name in COMPONENT_ORDER
    }
    expected_total = latency_cycles / baseline_latency_cycles
    residual = expected_total - sum(shares.values())
    if abs(residual) > MAX_COMPONENT_RESIDUAL_SHARE:
        raise ValueError(
            "Latency breakdown residual too large for "
            f"{context}: {residual:.6g}."
        )

    final_component = COMPONENT_ORDER[-1]
    shares[final_component] += residual
    if shares[final_component] < -1e-12:
        raise ValueError(f"Negative normalized share for {context}/{final_component}.")
    if abs(shares[final_component]) < 1e-12:
        shares[final_component] = 0.0
    return shares


def _baseline_latency_cycles_for_mode(
    *,
    evaluation_row: dict[str, Any],
    selected_evaluation_rows: dict[str, dict[str, Any]],
    normalization_mode: str,
    context: str,
) -> float:
    if normalization_mode == NORMALIZE_PER_BAR:
        baseline_latency_cycles = float(evaluation_row["latency_cycles"])
    elif normalization_mode == NORMALIZE_TO_STEP_A:
        baseline_latency_cycles = float(selected_evaluation_rows["a"]["latency_cycles"])
    else:
        raise ValueError(f"Unsupported normalization mode for {context}: {normalization_mode}")

    if baseline_latency_cycles <= 0.0:
        raise ValueError(f"Baseline latency must be positive for {context}.")
    return baseline_latency_cycles


def _build_plot_bars_for_run(
    *,
    run_dir: Path,
    workload_name: str,
    normalization_mode: str = NORMALIZE_PER_BAR,
) -> list[PlotBar]:
    summary = _load_optional_json(run_dir / "summary.json")
    frontier_csv = _resolve_existing_frontier_csv(run_dir, summary)
    evaluations_csv = _resolve_existing_evaluations_csv(run_dir, summary)

    frontier_rows = _load_frontier_rows(frontier_csv)
    if not frontier_rows:
        raise ValueError(f"No valid Pareto rows found in {frontier_csv}.")
    selected_rows = _select_best_rows_across_frontiers(
        frontier_rows, context=f"{ARCH_NAME}/{workload_name}"
    )

    evaluation_rows = _load_evaluation_rows(evaluations_csv)
    if not evaluation_rows:
        raise ValueError(f"No successful evaluation rows found in {evaluations_csv}.")
    evaluation_index = _build_evaluation_index(evaluation_rows)

    workload_label = str(summary.get("workload_display_name", workload_name)).strip()
    if not workload_label:
        workload_label = workload_name

    selected_evaluation_rows: dict[str, dict[str, Any]] = {}
    for step_id in STEP_ORDER:
        selected_row = selected_rows[step_id]
        selected_frontier = str(selected_row.get("frontier_name", "")).strip()
        context = f"{ARCH_NAME}/{workload_name}/{selected_frontier}/{step_id}"
        selected_evaluation_rows[step_id] = _match_evaluation_row(
            selected_row, evaluation_index, context=context
        )

    plot_bars: list[PlotBar] = []
    for step_id in STEP_ORDER:
        evaluation_row = selected_evaluation_rows[step_id]
        selected_frontier = str(selected_rows[step_id].get("frontier_name", "")).strip()
        context = f"{ARCH_NAME}/{workload_name}/{selected_frontier}/{step_id}"
        baseline_latency_cycles = _baseline_latency_cycles_for_mode(
            evaluation_row=evaluation_row,
            selected_evaluation_rows=selected_evaluation_rows,
            normalization_mode=normalization_mode,
            context=context,
        )
        plot_bars.append(
            PlotBar(
                workload_name=workload_name,
                workload_label=workload_label,
                step_id=step_id,
                component_shares=_normalized_component_shares(
                    evaluation_row,
                    context=context,
                    baseline_latency_cycles=baseline_latency_cycles,
                ),
            )
        )
    return plot_bars


def _build_plot_bars(
    report_root: Path, *, normalization_mode: str = NORMALIZE_PER_BAR
) -> list[PlotBar]:
    plot_bars: list[PlotBar] = []
    for workload_name, run_dir in sorted(_discover_latest_runs(report_root).items()):
        plot_bars.extend(
            _build_plot_bars_for_run(
                run_dir=run_dir,
                workload_name=workload_name,
                normalization_mode=normalization_mode,
            )
        )
    if not plot_bars:
        raise ValueError(f"No plot bars generated under {report_root}.")
    return plot_bars


def _workload_label_map(plot_bars: list[PlotBar]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for plot_bar in plot_bars:
        labels.setdefault(plot_bar.workload_name, plot_bar.workload_label)
    return labels


def _ordered_workloads(plot_bars: list[PlotBar]) -> tuple[str, ...]:
    return tuple(sorted({plot_bar.workload_name for plot_bar in plot_bars}))


def _bar_positions(
    workload_names: tuple[str, ...],
) -> tuple[dict[tuple[str, str], float], dict[str, float], list[float]]:
    positions: dict[tuple[str, str], float] = {}
    workload_centers: dict[str, float] = {}
    separators: list[float] = []
    current_x = 0.0

    for workload_index, workload_name in enumerate(workload_names):
        group_positions: list[float] = []
        for step_index, step_id in enumerate(STEP_ORDER):
            x_position = current_x + (step_index * STEP_SPACING)
            positions[(workload_name, step_id)] = x_position
            group_positions.append(x_position)
        workload_centers[workload_name] = sum(group_positions) / len(group_positions)
        current_x = group_positions[-1] + STEP_SPACING + WORKLOAD_GAP
        if workload_index < len(workload_names) - 1:
            separators.append(
                group_positions[-1] + (STEP_SPACING / 2.0) + (WORKLOAD_GAP / 2.0)
            )

    return positions, workload_centers, separators


def _percentage_tick_label(value: float, _position: float | None = None) -> str:
    percentage = float(value) * 100.0
    if abs(percentage - round(percentage)) < 1e-9:
        return f"{int(round(percentage))}%"
    return f"{percentage:.1f}%"


def _plot_latency_breakdown(plot_bars: list[PlotBar]):
    workload_names = _ordered_workloads(plot_bars)
    workload_labels = _workload_label_map(plot_bars)
    positions, workload_centers, separators = _bar_positions(workload_names)
    plot_index = {
        (plot_bar.workload_name, plot_bar.step_id): plot_bar
        for plot_bar in plot_bars
    }

    figure_width = max(
        MIN_FIGURE_WIDTH_IN,
        FIGURE_WIDTH_PER_WORKLOAD_IN * float(len(workload_names) * len(STEP_ORDER)),
    )
    fig, axis = plt.subplots(figsize=(figure_width, FIGURE_HEIGHT_IN))

    bar_x_positions = [
        positions[(workload_name, step_id)]
        for workload_name in workload_names
        for step_id in STEP_ORDER
    ]
    bar_step_ids = [
        step_id for workload_name in workload_names for step_id in STEP_ORDER
    ]

    bottoms = [0.0] * len(bar_x_positions)
    for component_name in COMPONENT_ORDER:
        heights = []
        for workload_name in workload_names:
            for step_id in STEP_ORDER:
                plot_bar = plot_index.get((workload_name, step_id))
                if plot_bar is None:
                    raise ValueError(f"Missing plot bar for {workload_name}/{step_id}.")
                heights.append(plot_bar.component_shares[component_name])

        bars = axis.bar(
            bar_x_positions,
            heights,
            width=BAR_WIDTH,
            bottom=bottoms,
            color=COMPONENT_COLORS[component_name],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LINEWIDTH,
            label=component_name,
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    top_height = max(1.0, max(bottoms, default=1.0))
    axis.set_ylim(0.0, top_height * 1.10)
    # DSE step labelled on top of each bar (replaces the hatch-by-step encoding,
    # so the legend now carries a single meaning: latency component = color).
    step_label_y = top_height * 1.025  # all (a)-(d) labels share one height
    for x_pos, step_id in zip(bar_x_positions, bar_step_ids):
        axis.text(
            x_pos,
            step_label_y,
            STEP_TICK_LABELS[step_id],
            ha="center",
            va="bottom",
            fontsize=STEP_LABEL_FONT_SIZE,
        )
    axis.yaxis.set_major_formatter(FuncFormatter(_percentage_tick_label))
    axis.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    axis.grid(
        axis="y",
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
    )
    axis.set_axisbelow(True)
    for separator_x in separators:
        axis.axvline(
            separator_x,
            color=WORKLOAD_SEPARATOR_COLOR,
            linewidth=WORKLOAD_SEPARATOR_LINEWIDTH,
            linestyle=WORKLOAD_SEPARATOR_LINESTYLE,
            zorder=0,
        )

    axis.set_xticks(bar_x_positions, ["" for _ in bar_x_positions])
    axis.tick_params(axis="x", pad=3, length=0, labelsize=TICK_LABEL_FONT_SIZE)
    for workload_name in workload_names:
        axis.text(
            workload_centers[workload_name],
            WORKLOAD_LABEL_Y,
            workload_labels[workload_name],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            clip_on=False,
            fontsize=WORKLOAD_LABEL_FONT_SIZE,
            rotation=WORKLOAD_LABEL_ROTATION_DEGREES,
            rotation_mode="anchor",
        )

    left_edge = min(bar_x_positions) - (BAR_WIDTH / 2.0) - OUTER_X_PADDING
    right_edge = max(bar_x_positions) + (BAR_WIDTH / 2.0) + OUTER_X_PADDING
    axis.set_xlim(left_edge, right_edge)
    # Only legend the components that are non-zero in at least one bar (drop
    # components that are zero across every bar, e.g. bank-to-bank transfer).
    present_components = [
        name
        for name in COMPONENT_ORDER
        if max(
            (bar.component_shares.get(name, 0.0) for bar in plot_index.values()),
            default=0.0,
        )
        > 1e-9
    ]
    component_handles = [
        Patch(
            facecolor=COMPONENT_COLORS[name],
            label=COMPONENT_LEGEND_LABELS.get(name, name),
        )
        for name in present_components
    ]
    leg1 = fig.legend(
        handles=component_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(component_handles),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.add_artist(leg1)
    fig.subplots_adjust(
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        top=FIGURE_TOP,
        bottom=FIGURE_BOTTOM,
    )
    return fig


def main() -> int:
    args = parse_args()
    report_root = args.report_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else report_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = (
        ((NORMALIZE_TO_STEP_A, STEP_A_NORMALIZED_OUTPUT_FILENAME),)
        if args.normalized_only
        else (
            (NORMALIZE_PER_BAR, DEFAULT_OUTPUT_FILENAME),
            (NORMALIZE_TO_STEP_A, STEP_A_NORMALIZED_OUTPUT_FILENAME),
        )
    )
    figures = []
    for normalization_mode, output_filename in variants:
        plot_bars = _build_plot_bars(
            report_root, normalization_mode=normalization_mode
        )
        fig = _plot_latency_breakdown(plot_bars)
        figures.append(fig)
        output_path = output_dir / output_filename
        fig.savefig(
            output_path,
            dpi=args.dpi,
            bbox_inches="tight",
            pad_inches=SAVEFIG_PAD_INCHES,
        )

    if args.show:
        plt.show()
    for fig in figures:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
