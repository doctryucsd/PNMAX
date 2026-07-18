#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

CURRENT_MODE_ID = "current"
NON_STREAMING_MODE_ID = "non_streaming"
ARCH_ORDER: tuple[str, ...] = ("upmem", "hbm_pim")
SPACE_ORDER: tuple[str, ...] = ("a", "b", "c", "d")
ARCH_MARKERS: dict[str, str] = {
    "upmem": "o",
    "hbm_pim": "s",
}
SPACE_COLORS: dict[str, str] = {
    "a": "#e76f51",
    "b": "#f4a261",
    "c": "#e9c46a",
    "d": "#2a9d8f",
}
# DSE-space (a)-(d) names, matching Figure 8 (workload_space_pareto_grid.py)
# step_label after its "+ cache" -> "+ buffer" display rewrite.
SPACE_LABELS: dict[str, str] = {
    "a": "(a) block-sharing",
    "b": "(b) + PU-level sharing",
    "c": "(c) + system-level sharing",
    "d": "(d) + buffer",
}
SCATTER_FIGSIZE: tuple[float, float] = (6.0, 4.5)
AXES_BOX_ASPECT = 1.0 / 3.0
SCATTER_ALPHA = 0.8
SCATTER_SIZE = 40
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 0.6
GRID_ALPHA = 0.35
REFERENCE_LINE_COLOR = "#7A7A7A"
REFERENCE_LINESTYLE = ":"
REFERENCE_LINEWIDTH = 0.8
MAX_NORMALIZED_MEM_FOOTPRINT = 10.0
TIGHT_LAYOUT_PAD = 0.02
SAVEFIG_PAD_INCHES = 0.01
FILTERED_OUTPUT_FILENAME = "workload_space_streaming_delta_scatter.pdf"
UNFILTERED_OUTPUT_FILENAME = "workload_space_streaming_delta_scatter_unfiltered.pdf"


@dataclass(frozen=True)
class ScatterPoint:
    arch: str
    workload: str
    space: str
    centered_latency: float
    centered_mem_footprint: float


def _marker_for_arch(arch: str) -> str:
    return ARCH_MARKERS.get(arch, "o")


def _color_for_space(space: str) -> str:
    return SPACE_COLORS.get(space, "#4C78A8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot centered non-streaming-vs-streaming latency and memory "
            "deltas from a workload-space Pareto evaluation output root."
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


def _is_valid_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "summary.json").is_file()


def _select_latest_run(workload_dir: Path) -> Path:
    run_dirs = sorted(
        path for path in workload_dir.iterdir() if path.is_dir() and _is_valid_run_dir(path)
    )
    if not run_dirs:
        raise ValueError(f"No run directories found under {workload_dir}.")
    return run_dirs[-1]


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


def _ordered_space_names(space_names: set[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            space_names,
            key=lambda space_name: (
                SPACE_ORDER.index(space_name)
                if space_name in SPACE_ORDER
                else len(SPACE_ORDER),
                space_name,
            ),
        )
    )


def _discover_latest_runs(report_root: Path) -> dict[tuple[str, str], Path]:
    if not report_root.is_dir():
        raise ValueError(f"Report root does not exist: {report_root}")

    arch_dirs = [
        path
        for path in report_root.iterdir()
        if path.is_dir() and path.name != "plots"
    ]
    ordered_archs = _ordered_arch_names({path.name for path in arch_dirs})

    selected: dict[tuple[str, str], Path] = {}
    for arch_name in ordered_archs:
        arch_dir = report_root / arch_name
        for workload_dir in sorted(path for path in arch_dir.iterdir() if path.is_dir()):
            try:
                latest_run = _select_latest_run(workload_dir)
            except ValueError:
                continue
            selected[(arch_name, workload_dir.name)] = latest_run

    if not selected:
        raise ValueError(f"No workload-space Pareto runs found under {report_root}.")
    return selected


def _resolve_existing_pareto_csv(run_dir: Path, summary: dict[str, Any]) -> Path:
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


def _load_frontier_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing Pareto CSV: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            step_id = str(row.get("step_id", "")).strip()
            if not step_id:
                continue
            try:
                rows.append(
                    {
                        "frontier_name": str(row.get("frontier_name", "")).strip(),
                        "step_id": step_id,
                        "base_step_id": str(row.get("base_step_id", "")).strip(),
                        "mode_id": str(row.get("mode_id", "")).strip(),
                        "order": int(row.get("order", 0) or 0),
                        "latency_cycles": float(row["latency_cycles"]),
                        "mem_footprint_bytes": float(row["mem_footprint_bytes"]),
                        "energy": float(row["energy"]),
                        "state_hash": str(row.get("state_hash", "")).strip(),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


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


def _row_dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    state_hash = str(row.get("state_hash", "")).strip()
    if state_hash:
        return ("state_hash", state_hash)
    return (
        "metrics",
        float(row["latency_cycles"]),
        float(row["mem_footprint_bytes"]),
        float(row["energy"]),
    )


def _row_selection_key(row: dict[str, Any]) -> tuple[float, float, float, str, str, int]:
    return (
        float(row["latency_cycles"]),
        float(row["mem_footprint_bytes"]),
        float(row["energy"]),
        str(row.get("state_hash", "")).strip(),
        str(row.get("frontier_name", "")).strip(),
        int(row.get("order", 0) or 0),
    )


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        dedupe_key = _row_dedupe_key(row)
        existing = deduped.get(dedupe_key)
        if existing is None or _row_selection_key(row) < _row_selection_key(existing):
            deduped[dedupe_key] = dict(row)
    return list(deduped.values())


def _select_best_rows_by_space_and_mode(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        group_key = (_base_step_id(row), _mode_id(row))
        grouped_rows.setdefault(group_key, []).append(row)

    best_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for group_key, group_rows in grouped_rows.items():
        deduped_rows = _dedupe_rows(group_rows)
        if not deduped_rows:
            continue
        best_rows[group_key] = min(deduped_rows, key=_row_selection_key)
    return best_rows


def _build_scatter_points_for_run(
    *, run_dir: Path, arch: str, workload: str
) -> list[ScatterPoint]:
    summary = _load_json(run_dir / "summary.json")
    pareto_csv = _resolve_existing_pareto_csv(run_dir, summary)
    rows = _load_frontier_rows(pareto_csv)
    if not rows:
        raise ValueError(f"No valid Pareto rows found in {pareto_csv}.")

    best_rows = _select_best_rows_by_space_and_mode(rows)
    spaces = sorted({space for space, _ in best_rows})
    if not spaces:
        raise ValueError(f"No workload spaces found in {pareto_csv}.")

    scatter_points: list[ScatterPoint] = []
    for space in spaces:
        current_row = best_rows.get((space, CURRENT_MODE_ID))
        non_streaming_row = best_rows.get((space, NON_STREAMING_MODE_ID))
        if current_row is None or non_streaming_row is None:
            missing_modes: list[str] = []
            if current_row is None:
                missing_modes.append(CURRENT_MODE_ID)
            if non_streaming_row is None:
                missing_modes.append(NON_STREAMING_MODE_ID)
            formatted_modes = ", ".join(missing_modes)
            raise ValueError(
                f"Missing paired Pareto rows for {arch}/{workload}/{space}: {formatted_modes}"
            )

        streaming_latency = float(current_row["latency_cycles"])
        streaming_mem = float(current_row["mem_footprint_bytes"])
        if streaming_latency <= 0.0:
            raise ValueError(
                f"Streaming latency must be positive for {arch}/{workload}/{space}."
            )
        if streaming_mem <= 0.0:
            raise ValueError(
                f"Streaming memory footprint must be positive for {arch}/{workload}/{space}."
            )

        centered_latency = (
            float(non_streaming_row["latency_cycles"]) / streaming_latency
        ) - 1.0
        normalized_mem_footprint = (
            float(non_streaming_row["mem_footprint_bytes"]) / streaming_mem
        )
        centered_mem_footprint = (
            normalized_mem_footprint - 1.0
        )
        scatter_points.append(
            ScatterPoint(
                arch=arch,
                workload=workload,
                space=space,
                centered_latency=centered_latency,
                centered_mem_footprint=centered_mem_footprint,
            )
        )
    return scatter_points


def _filter_scatter_points(points: list[ScatterPoint]) -> list[ScatterPoint]:
    return [
        point
        for point in points
        if point.centered_mem_footprint <= (MAX_NORMALIZED_MEM_FOOTPRINT - 1.0)
    ]


def _plot_streaming_delta_scatter(*, ax, points: list[ScatterPoint]) -> None:
    if not points:
        raise ValueError("No scatter points available to plot.")

    ax.set_box_aspect(AXES_BOX_ASPECT)
    present_archs = _ordered_arch_names({point.arch for point in points})
    present_spaces = _ordered_space_names({point.space for point in points})
    for arch in present_archs:
        for space in present_spaces:
            subset_points = [
                point
                for point in points
                if point.arch == arch and point.space == space
            ]
            if not subset_points:
                continue
            ax.scatter(
                [point.centered_latency for point in subset_points],
                [point.centered_mem_footprint for point in subset_points],
                s=SCATTER_SIZE,
                color=_color_for_space(space),
                alpha=SCATTER_ALPHA,
                edgecolors="none",
                marker=_marker_for_arch(arch),
            )
    ax.axvline(
        0.0,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=0,
    )
    ax.axhline(
        0.0,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=0,
    )
    ax.set_xlabel("Latency delta (norm.)")
    ax.set_ylabel("Mem. fp. delta (norm.)")
    ax.xaxis.set_major_formatter(FuncFormatter(_percentage_tick_label))
    ax.yaxis.set_major_formatter(FuncFormatter(_percentage_tick_label))
    ax.grid(True, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
    arch_handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker=_marker_for_arch(arch),
            markersize=7,
            markerfacecolor="#404040",
            markeredgecolor="#404040",
            label=arch,
        )
        for arch in present_archs
    ]
    space_handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=7,
            markerfacecolor=_color_for_space(space),
            markeredgecolor=_color_for_space(space),
            label=SPACE_LABELS.get(space, f"({space})"),
        )
        for space in present_spaces
    ]
    # One legend, two columns: architecture on the left, DSE space on the right.
    # Pad the shorter group with blank rows so both columns top-align (matplotlib
    # fills the legend grid column-major).
    def _blank_handle() -> Line2D:
        return Line2D([], [], linestyle="None", marker="None", label=" ")

    n_rows = max(len(arch_handles), len(space_handles))
    left_col = arch_handles + [_blank_handle() for _ in range(n_rows - len(arch_handles))]
    right_col = space_handles + [_blank_handle() for _ in range(n_rows - len(space_handles))]
    ax.legend(
        handles=left_col + right_col,
        loc="upper left",
        frameon=False,
        ncol=2,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    ax.set_title("")


def _percentage_tick_label(value: float, _position: float | None = None) -> str:
    percentage = float(value) * 100.0
    if abs(percentage) < 1e-12:
        return "0%"
    if abs(percentage - round(percentage)) < 1e-12:
        return f"{int(round(percentage))}%"
    return f"{percentage:.3g}%"


def _save_figure(
    *,
    fig,
    output_path: Path,
    dpi: int,
    show: bool,
    message: str,
) -> None:
    fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=SAVEFIG_PAD_INCHES,
    )
    print(f"{message} {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    report_root = args.report_root.resolve()
    selected_runs = _discover_latest_runs(report_root)

    points: list[ScatterPoint] = []
    for (arch, workload), run_dir in sorted(selected_runs.items()):
        points.extend(
            _build_scatter_points_for_run(
                run_dir=run_dir,
                arch=arch,
                workload=workload,
            )
        )

    filtered_points = _filter_scatter_points(points)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (report_root / "plots").resolve()
    )

    filtered_fig, filtered_ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    _plot_streaming_delta_scatter(ax=filtered_ax, points=filtered_points)
    _save_figure(
        fig=filtered_fig,
        output_path=output_dir / FILTERED_OUTPUT_FILENAME,
        dpi=args.dpi,
        show=args.show,
        message="Saved filtered workload-space streaming delta scatter to",
    )

    unfiltered_fig, unfiltered_ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    _plot_streaming_delta_scatter(ax=unfiltered_ax, points=points)
    _save_figure(
        fig=unfiltered_fig,
        output_path=output_dir / UNFILTERED_OUTPUT_FILENAME,
        dpi=args.dpi,
        show=args.show,
        message="Saved unfiltered workload-space streaming delta scatter to",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
