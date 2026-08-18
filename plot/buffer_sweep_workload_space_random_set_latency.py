#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUTPUT_DIRNAME = "plots"
OUTPUT_FILENAME = "selected_mappings.pdf"
# Fig. 15 shows the published line set: the UniNDP baseline row is eval-stage
# metadata, not a plotted line, and labels drop the internal "(+ cache)"
# qualifier.
EXCLUDED_LINE_KINDS = frozenset({"baseline"})
LINE_LABEL_DROPPED_SUFFIX = " (+ cache)"
# Fig. 15 layout: latency-only single panel. The optional --with-energy
# layout stacks an energy panel below.
FIGSIZE: tuple[float, float] = (7.2, 2.6)
FIGSIZE_WITH_ENERGY: tuple[float, float] = (7.2, 4.6)
YLABEL = "Latency (norm. to 512B)"
LINEWIDTH = 2.0
MARKERSIZE = 6.0
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 0.6
GRID_ALPHA = 0.35
TIGHT_LAYOUT_PAD = 0.12
SAVEFIG_PAD_INCHES = 0.01
LINE_COLORS: tuple[str, ...] = (
    "#264653",
    "#2a9d8f",
    "#8ab17d",
    "#e76f51",
    "#6d597a",
)
LINE_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "x")


@dataclass(frozen=True)
class PlotRow:
    set_id: int
    line_order: int
    line_kind: str
    line_id: str
    line_label: str
    variant_order: int
    variant_id: str
    variant_label: str
    buffer_bytes: int
    area_mm2: float
    energy: float
    latency_cycles: int
    normalized_latency: float
    midpoint_variant_id: str
    midpoint_buffer_bytes: int
    midpoint_latency_cycles: int
    is_midpoint: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one unified sampled-pool normalized-latency figure across "
            "HBM-PIM buffer-sweep variants for one workload run or workload "
            "artifact directory."
        )
    )
    parser.add_argument(
        "pipeline_dir",
        type=Path,
        help=(
            "Path to a completed unified-sampled-pool buffer-sweep run "
            "directory, its summary.json, or a workload directory containing "
            "timestamped runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            f"Optional output directory for the workload PDF (default: summary "
            f"plots_dir or <pipeline_dir>/{OUTPUT_DIRNAME})."
        ),
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument(
        "--with-energy",
        action="store_true",
        help=(
            "Also render the normalized-energy panel below the latency panel "
            "(2-panel layout). Default is the latency-only Fig. 15 layout."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure in addition to saving it.",
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
        path.resolve() for path in workload_dir.iterdir() if _is_valid_run_dir(path)
    )
    if not run_dirs:
        raise ValueError(f"No run directories found under {workload_dir}.")
    return run_dirs[-1]


def _resolve_pipeline_dir(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        if candidate.name == "summary.json":
            return candidate.parent.resolve()
        raise ValueError(
            "Expected a run directory, workload directory, or summary.json file, "
            f"got file: {candidate}"
        )
    if _is_valid_run_dir(candidate):
        return candidate
    if candidate.is_dir():
        try:
            return _select_latest_run(candidate)
        except ValueError as exc:
            raise ValueError(
                "Expected a completed unified-sampled-pool buffer-sweep run "
                "directory containing summary.json, or a workload directory "
                "containing timestamped run directories with summary.json."
            ) from exc
    raise ValueError(f"Plot input path does not exist: {candidate}")


def _resolve_plot_rows_csv(pipeline_dir: Path, summary: Mapping[str, Any]) -> Path:
    raw_path = summary.get("plot_rows_csv")
    if raw_path:
        candidate = _resolve_path(str(raw_path), base_dir=pipeline_dir)
        if candidate.is_file():
            return candidate
    default_candidate = (pipeline_dir / "plot_rows.csv").resolve()
    if default_candidate.is_file():
        return default_candidate
    raise ValueError(f"Missing plot_rows.csv under {pipeline_dir}.")


def _resolve_evaluations_csv(
    pipeline_dir: Path, summary: Mapping[str, Any]
) -> Path | None:
    raw_path = summary.get("evaluations_csv")
    if raw_path:
        candidate = _resolve_path(str(raw_path), base_dir=pipeline_dir)
        if candidate.is_file():
            return candidate
    default_candidate = (pipeline_dir / "evaluations.csv").resolve()
    if default_candidate.is_file():
        return default_candidate
    return None


def _resolve_output_dir(
    *,
    pipeline_dir: Path,
    summary: Mapping[str, Any],
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    raw_path = summary.get("plots_dir")
    if raw_path:
        return _resolve_path(str(raw_path), base_dir=pipeline_dir)
    return (pipeline_dir / OUTPUT_DIRNAME).resolve()


def _load_energy_by_key(path: Path | None) -> dict[tuple[str, str, int], float]:
    if path is None or not path.is_file():
        return {}

    energy_by_key: dict[tuple[str, str, int], float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                key = (
                    str(row["line_id"]).strip(),
                    str(row["variant_id"]).strip(),
                    int(row["buffer_bytes"]),
                )
                energy_by_key[key] = float(row["energy"])
            except (KeyError, TypeError, ValueError):
                continue
    return energy_by_key


def _load_plot_rows(
    path: Path,
    *,
    energy_by_key: Mapping[tuple[str, str, int], float],
) -> list[PlotRow]:
    if not path.is_file():
        raise ValueError(f"Missing plot-row CSV: {path}")

    rows: list[PlotRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                line_kind = str(row["line_kind"]).strip()
                if line_kind in EXCLUDED_LINE_KINDS:
                    continue
                line_id = str(row["line_id"]).strip()
                variant_id = str(row["variant_id"]).strip()
                buffer_bytes = int(row["buffer_bytes"])
                rows.append(
                    PlotRow(
                        set_id=int(row["set_id"]),
                        line_order=int(row["line_order"]),
                        line_kind=line_kind,
                        line_id=line_id,
                        line_label=str(row["line_label"])
                        .strip()
                        .removesuffix(LINE_LABEL_DROPPED_SUFFIX),
                        variant_order=int(row["variant_order"]),
                        variant_id=variant_id,
                        variant_label=str(row["variant_label"]).strip(),
                        buffer_bytes=buffer_bytes,
                        area_mm2=float(row["area_mm2"]),
                        energy=energy_by_key.get(
                            (line_id, variant_id, buffer_bytes),
                            0.0,
                        ),
                        latency_cycles=int(row["latency_cycles"]),
                        normalized_latency=float(row["normalized_latency"]),
                        midpoint_variant_id=str(row["midpoint_variant_id"]).strip(),
                        midpoint_buffer_bytes=int(row["midpoint_buffer_bytes"]),
                        midpoint_latency_cycles=int(row["midpoint_latency_cycles"]),
                        is_midpoint=_parse_bool(row["is_midpoint"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise ValueError(f"No valid plot rows found in {path}.")
    return sorted(
        rows,
        key=lambda row: (
            row.set_id,
            row.line_order,
            row.variant_order,
            row.buffer_bytes,
            row.variant_id,
        ),
    )


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    return token in {"1", "true", "yes", "on"}


def _group_rows_by_line(rows: list[PlotRow]) -> list[list[PlotRow]]:
    grouped: dict[str, list[PlotRow]] = {}
    for row in rows:
        grouped.setdefault(row.line_id, []).append(row)
    return sorted(
        (
            sorted(
                line_rows,
                key=lambda row: (
                    row.variant_order,
                    row.buffer_bytes,
                    row.variant_id,
                ),
            )
            for line_rows in grouped.values()
        ),
        key=lambda line_rows: (
            line_rows[0].line_order,
            line_rows[0].line_id,
        ),
    )


def _validate_plot_rows(
    *,
    rows: list[PlotRow],
    expected_line_count: int | None,
) -> list[list[PlotRow]]:
    set_ids = sorted({row.set_id for row in rows})
    if len(set_ids) != 1:
        raise ValueError(
            f"Expected exactly one unified set in plot_rows.csv, got {set_ids}."
        )

    line_groups = _group_rows_by_line(rows)
    if not line_groups:
        raise ValueError("The unified plot did not contain any plot lines.")
    if expected_line_count is not None and len(line_groups) != expected_line_count:
        print(
            "Warning: expected "
            f"{expected_line_count} lines for the unified plot, got "
            f"{len(line_groups)}. Plotting available lines from plot_rows.csv.",
            file=sys.stderr,
        )

    expected_variant_count = len(line_groups[0])
    if expected_variant_count == 0:
        raise ValueError("The unified plot did not contain any plot points.")
    for line_rows in line_groups:
        if len(line_rows) != expected_variant_count:
            raise ValueError(
                "The unified plot contains inconsistent variant counts across lines."
            )
        midpoint_rows = [row for row in line_rows if row.is_midpoint]
        if len(midpoint_rows) != 1:
            raise ValueError(
                "Expected exactly one midpoint row for unified-plot line "
                f"{line_rows[0].line_id}."
            )
    return line_groups


def _format_buffer_size(buffer_bytes: int) -> str:
    units = (
        ("MB", 1024**2),
        ("KB", 1024),
        ("B", 1),
    )
    for suffix, factor in units:
        if buffer_bytes >= factor or factor == 1:
            value = float(buffer_bytes) / float(factor)
            if abs(value - round(value)) < 1e-12:
                return f"{int(round(value))}{suffix}"
            return f"{value:.3g}{suffix}"
    return f"{buffer_bytes}B"


def _compact_tick_label(value: float, _position: float | None = None) -> str:
    numeric = float(value)
    if abs(numeric) < 1e-12:
        return "0"
    if abs(numeric - round(numeric)) < 1e-12:
        return str(int(round(numeric)))
    return f"{numeric:.3g}"


def _normalized_energy_values(line_rows: list[PlotRow]) -> list[float]:
    midpoint_energy = next(row.energy for row in line_rows if row.is_midpoint)
    if abs(midpoint_energy) < 1e-12:
        return [0.0 for _row in line_rows]
    return [row.energy / midpoint_energy for row in line_rows]


def _plot_selected_mappings(
    *,
    ax_lat,
    ax_energy,
    summary: Mapping[str, Any],
    line_groups: list[list[PlotRow]],
) -> None:
    """Draw the latency panel and, when ax_energy is not None, the energy panel."""
    x_values = list(range(len(line_groups[0])))
    xticklabels = [
        _format_buffer_size(row.buffer_bytes) for row in line_groups[0]
    ]
    axes = [ax_lat] if ax_energy is None else [ax_lat, ax_energy]

    for index, line_rows in enumerate(line_groups):
        ax_lat.plot(
            x_values,
            [row.normalized_latency for row in line_rows],
            color=LINE_COLORS[index % len(LINE_COLORS)],
            marker=LINE_MARKERS[index % len(LINE_MARKERS)],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            label=line_rows[0].line_label,
        )
        if ax_energy is not None:
            ax_energy.plot(
                x_values,
                _normalized_energy_values(line_rows),
                color=LINE_COLORS[index % len(LINE_COLORS)],
                marker=LINE_MARKERS[index % len(LINE_MARKERS)],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                label=line_rows[0].line_label,
            )

    for ax in axes:
        ax.grid(
            True,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
        )
        ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick_label))

    bottom_ax = axes[-1]
    bottom_ax.set_xticks(x_values)
    bottom_ax.set_xticklabels(xticklabels)
    bottom_ax.set_xlabel("Buffer size")
    ax_lat.set_ylabel(YLABEL)
    if ax_energy is not None:
        ax_energy.set_ylabel("Energy (norm. to 512B)")
    ax_lat.legend(frameon=False, ncol=2)

    workload_display_name = str(summary.get("workload_display_name", "")).strip()
    arch = str(summary.get("arch", "")).strip()
    if workload_display_name and arch:
        # ax_lat.set_title(f"{workload_display_name} ({arch})")
        pass
    elif workload_display_name:
        ax_lat.set_title(workload_display_name)
    else:
        ax_lat.set_title("Selected mappings")


def _save_figure(
    *,
    fig,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=SAVEFIG_PAD_INCHES,
    )
    print(f"Saved unified sampled-pool latency plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    pipeline_dir = _resolve_pipeline_dir(args.pipeline_dir)
    summary = _load_json(pipeline_dir / "summary.json")
    plot_rows_csv = _resolve_plot_rows_csv(pipeline_dir, summary)
    evaluations_csv = _resolve_evaluations_csv(pipeline_dir, summary)
    plot_rows = _load_plot_rows(
        plot_rows_csv,
        energy_by_key=(
            _load_energy_by_key(evaluations_csv) if args.with_energy else {}
        ),
    )
    output_dir = _resolve_output_dir(
        pipeline_dir=pipeline_dir,
        summary=summary,
        output_dir=args.output_dir,
    )

    raw_selected_topk = summary.get("selected_topk", 3)
    expected_line_count = None
    try:
        # top-k change lines + the best-latency line (the baseline row is
        # excluded at load).
        expected_line_count = int(raw_selected_topk) + 1
    except (TypeError, ValueError):
        expected_line_count = None

    line_groups = _validate_plot_rows(
        rows=plot_rows,
        expected_line_count=expected_line_count,
    )
    if args.with_energy:
        fig, (ax_lat, ax_energy) = plt.subplots(
            2, 1, sharex=True, figsize=FIGSIZE_WITH_ENERGY
        )
    else:
        # Fig. 15 layout: latency-only single panel.
        fig, ax_lat = plt.subplots(figsize=FIGSIZE)
        ax_energy = None
    _plot_selected_mappings(
        ax_lat=ax_lat,
        ax_energy=ax_energy,
        summary=summary,
        line_groups=line_groups,
    )
    _save_figure(
        fig=fig,
        output_path=output_dir / OUTPUT_FILENAME,
        dpi=args.dpi,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
