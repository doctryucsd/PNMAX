#!/usr/bin/env python3
"""Plot reduction-place Pareto fronts from per-workload evaluations.csv files."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter, NullLocator


# INCLUSIVE-ONLY roots (hardcoded; non-inclusive mode is no longer supported).
# Inclusivity is an eval-side property: each reduction-placement level pools the
# more-local levels' mappings and RE-COSTS them under its own evaluation -- bank =
# config1; channel = bank PU mappings re-cost under channel_level ∪ channel pool (the
# _PUunion root); base-die = config1 ∪ channel ∪ config2 under base-die. The plot just
# draws each config's own front -- it must NOT union points across configs here.
REPO_ROOT = Path(os.environ.get("PNMAX_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_ROOT = Path(os.environ.get("PNMAX_RESULTS_ROOT", str(REPO_ROOT / "results")))
DEFAULT_INCLUSIVE_ROOT = (
    RESULTS_ROOT / "fig17_reduction" / "pareto_eval_channel_inclusive" / "hbm_pim"
)
DEFAULT_INCLUSIVE_CHANNEL_ROOT = (
    REPO_ROOT
    / "results"
    / "fig17_reduction"
    / "pareto_eval_channel_inclusive_PUunion"
    / "hbm_pim"
)
DEFAULT_OUTDIR = RESULTS_ROOT / "fig17_reduction"

METRICS = ("latency_cycles", "mem_footprint_bytes", "energy")
REQUIRED_EVAL_COLUMNS = ("config_id", "status", "state_hash", *METRICS)
CONFIG_LABELS = {
    "config1": "Bank-level",
    "config2": "Base-die-level",
    "channel": "Channel-level",
}
CONFIG_ORDER = ("config1", "channel", "config2")
CONFIG_COLORS = {
    "config1": "#e76f51",
    "config2": "#f4a261",
    "channel": "#e9c46a",
}
CONFIG_MARKERS = {"config1": "o", "config2": "s", "channel": "^"}
# Channel is dashed so it stays visible where it rides on top of the base-die front.
CONFIG_LINESTYLES = {"config1": "-", "config2": "-", "channel": "--"}
ATTACC_MARKER_STYLE = dict(
    color="#2196F3",
    marker="*",
    s=120,
    linewidths=1.5,
    zorder=6,
)
SUBPLOT_WIDTH_IN = 4.9
SUBPLOT_BOX_ASPECT = 9.0 / 16.0
SUBPLOT_HEIGHT_IN = SUBPLOT_WIDTH_IN * SUBPLOT_BOX_ASPECT
FIGURE_WIDTH_IN = 8.0
FIGURE_HEIGHT_IN = 5.0
AXES_LEFT_IN = 0.95
AXES_BOTTOM_IN = 1.45
LEGEND_BBOX_ANCHOR = (0.988, 0.2)
DEFAULT_LOG_MAJOR_TICKS = 4
REDUCED_LOG_MAJOR_TICKS = 3
MIN_LOG_MAJOR_TICKS = 2
XTICK_ROTATION_DEGREES = 25.0
METRIC_LABELS = {
    "latency_cycles": "Latency (cycles)",
    "mem_footprint_bytes": "Memory footprint (bytes)",
    "energy": "Energy (pJ)",
}
METRIC_SHORT = {
    "latency_cycles": "Latency",
    "mem_footprint_bytes": "Mem. footprint",
    "energy": "Energy",
}
METRIC_PAIRS = (
    ("latency_cycles", "mem_footprint_bytes", "lat-mem"),
    ("latency_cycles", "energy", "lat-energy"),
    ("mem_footprint_bytes", "energy", "mem-energy"),
)


@dataclass(frozen=True)
class MetricValues:
    latency_cycles: float
    mem_footprint_bytes: float
    energy: float

    def value(self, metric: str) -> float:
        return float(getattr(self, metric))


@dataclass(frozen=True)
class EvalPoint:
    config_id: str
    state_hash: str
    metrics: MetricValues

    def value(self, metric: str) -> float:
        return self.metrics.value(metric)


@dataclass
class WorkloadSummary:
    workload_token: str
    eval_path: Path
    ok_counts: dict[str, int]
    front_sizes: dict[str, dict[str, int]]
    channel_path: Path | None
    channel_present: bool
    attacc_drawn: bool
    attacc_note: str
    outputs: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reduction-place Pareto fronts for each workload "
        "(INCLUSIVE-only: inputs are the inclusive + _PUunion eval runs).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INCLUSIVE_ROOT,
        help=f"inclusive pareto-eval root (default: {DEFAULT_INCLUSIVE_ROOT})",
    )
    parser.add_argument(
        "--channel-input-root",
        type=Path,
        default=DEFAULT_INCLUSIVE_CHANNEL_ROOT,
        help=(
            "channel-level (PU-union) pareto-eval root "
            f"(default: {DEFAULT_INCLUSIVE_CHANNEL_ROOT})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--outdir",
        dest="outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"output directory (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument("--dpi", type=int, default=200, help="output figure DPI")
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help=(
            "Comma-separated metric pair tags to render "
            "(lat-mem, lat-energy, mem-energy); default: all."
        ),
    )
    return parser.parse_args()


def latest_evaluations_csv(workload_dir: Path) -> Path | None:
    candidates = list(workload_dir.glob("*/evaluations.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.parent.name)


def require_columns(path: Path, fieldnames: Sequence[str] | None, columns: Sequence[str]) -> None:
    present = set(fieldnames or [])
    missing = [column for column in columns if column not in present]
    if missing:
        raise ValueError(f"{path} missing required column(s): {', '.join(missing)}")


def parse_metric(row: dict[str, str], metric: str, path: Path) -> float:
    try:
        return float(row[metric])
    except ValueError as exc:
        raise ValueError(f"{path}: invalid {metric} value {row[metric]!r}") from exc


def metric_values_from_row(row: dict[str, str], path: Path) -> MetricValues:
    return MetricValues(
        latency_cycles=parse_metric(row, "latency_cycles", path),
        mem_footprint_bytes=parse_metric(row, "mem_footprint_bytes", path),
        energy=parse_metric(row, "energy", path),
    )


def read_evaluations(path: Path) -> list[EvalPoint]:
    points: list[EvalPoint] = []
    seen: set[tuple[str, str]] = set()

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        require_columns(path, reader.fieldnames, REQUIRED_EVAL_COLUMNS)
        for row in reader:
            if row["status"].strip() != "ok":
                continue

            config_id = row["config_id"].strip()
            state_hash = row["state_hash"].strip()
            key = (config_id, state_hash)
            if key in seen:
                continue
            seen.add(key)

            points.append(
                EvalPoint(
                    config_id=config_id,
                    state_hash=state_hash,
                    metrics=metric_values_from_row(row, path),
                )
            )

    return points


def read_channel_points(
    channel_root: Path, workload_token: str
) -> tuple[list[EvalPoint], Path | None]:
    """Read the channel-level reduction front from a separate eval run.

    The channel run is produced by the activation driver with
    ``--config3-arch-file data/archs/lowered/activation/channel_level.yaml``, which emits
    ``config3`` rows. As a fallback we also accept ``config2`` rows (for a run that
    swapped channel_level.yaml in as ``--config2-arch-file`` instead). Either way
    the points are relabeled to the ``channel`` front.
    """
    workload_dir = channel_root / workload_token
    if not workload_dir.is_dir():
        return [], None

    evaluations_path = latest_evaluations_csv(workload_dir)
    if evaluations_path is None:
        return [], None

    all_points = read_evaluations(evaluations_path)
    source_config = "config3" if any(p.config_id == "config3" for p in all_points) else "config2"
    points = [
        EvalPoint(config_id="channel", state_hash=point.state_hash, metrics=point.metrics)
        for point in all_points
        if point.config_id == source_config
    ]
    return points, evaluations_path


def group_points(points: Sequence[EvalPoint]) -> dict[str, list[EvalPoint]]:
    grouped: dict[str, list[EvalPoint]] = {config_id: [] for config_id in CONFIG_ORDER}
    for point in points:
        if point.config_id in grouped:
            grouped[point.config_id].append(point)
    return grouped


def pareto_front(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the 2-D minimization Pareto front sorted by x, then y."""
    front: list[tuple[float, float]] = []
    best_y = math.inf

    for x_value, y_value in sorted(points, key=lambda point: (point[0], point[1])):
        if y_value < best_y:
            front.append((x_value, y_value))
            best_y = y_value

    return front


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(
        f"{config_id}={counts.get(config_id, 0)}" for config_id in CONFIG_ORDER
    )


def configure_log_axis_ticks(axis, *, major_tick_count: int = DEFAULT_LOG_MAJOR_TICKS) -> None:
    axis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0,), numticks=major_tick_count)
    )
    axis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=True))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())
    axis_name = getattr(axis, "axis_name", "")
    if axis_name == "x":
        axis.axes.tick_params(
            axis="x",
            which="minor",
            labelbottom=False,
            labeltop=False,
        )
    elif axis_name == "y":
        axis.axes.tick_params(
            axis="y",
            which="minor",
            labelleft=False,
            labelright=False,
        )


def _visible_tick_labels(labels: list) -> list:
    return [
        label
        for label in labels
        if label.get_visible() and str(label.get_text()).strip()
    ]


def _tick_labels_overlap(labels: list, renderer, *, axis: str) -> bool:
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


def _draw_renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _rotate_x_tick_labels(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_rotation(XTICK_ROTATION_DEGREES)
        label.set_ha("right")
        label.set_rotation_mode("anchor")


def _resolve_axis_tick_overlap(fig, ax) -> None:
    renderer = _draw_renderer(fig)

    if _tick_labels_overlap(ax.get_xticklabels(), renderer, axis="x"):
        configure_log_axis_ticks(ax.xaxis, major_tick_count=REDUCED_LOG_MAJOR_TICKS)
        renderer = _draw_renderer(fig)

    if _tick_labels_overlap(ax.get_yticklabels(), renderer, axis="y"):
        configure_log_axis_ticks(ax.yaxis, major_tick_count=REDUCED_LOG_MAJOR_TICKS)
        renderer = _draw_renderer(fig)

    if _tick_labels_overlap(ax.get_xticklabels(), renderer, axis="x"):
        configure_log_axis_ticks(ax.xaxis, major_tick_count=MIN_LOG_MAJOR_TICKS)
        renderer = _draw_renderer(fig)

    if _tick_labels_overlap(ax.get_yticklabels(), renderer, axis="y"):
        configure_log_axis_ticks(ax.yaxis, major_tick_count=MIN_LOG_MAJOR_TICKS)
        renderer = _draw_renderer(fig)

    if _tick_labels_overlap(ax.get_xticklabels(), renderer, axis="x"):
        _rotate_x_tick_labels(ax)
        _draw_renderer(fig)


def _figure_size() -> tuple[float, float]:
    return (FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN)


def _axes_rect() -> tuple[float, float, float, float]:
    return (
        AXES_LEFT_IN / FIGURE_WIDTH_IN,
        AXES_BOTTOM_IN / FIGURE_HEIGHT_IN,
        SUBPLOT_WIDTH_IN / FIGURE_WIDTH_IN,
        SUBPLOT_HEIGHT_IN / FIGURE_HEIGHT_IN,
    )


def _front_legend_handles(present_config_ids: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=CONFIG_COLORS[config_id],
            marker=CONFIG_MARKERS[config_id],
            linestyle=CONFIG_LINESTYLES[config_id],
            linewidth=2.0,
            markersize=7,
            label=CONFIG_LABELS[config_id],
        )
        for config_id in CONFIG_ORDER
        if config_id in present_config_ids
    ]


def _attacc_legend_handles(attacc_point: MetricValues | None) -> list[Line2D]:
    if attacc_point is None:
        return []
    return [
        Line2D(
            [0],
            [0],
            color=ATTACC_MARKER_STYLE["color"],
            marker=ATTACC_MARKER_STYLE["marker"],
            linestyle="None",
            markersize=10.0,
            label="AttAcc",
        )
    ]


def _add_figure_legends(
    fig,
    *,
    front_handles: list[Line2D],
    attacc_handles: list[Line2D],
) -> None:
    if attacc_handles:
        attacc_legend = fig.legend(
            handles=attacc_handles,
            loc="lower right",
            bbox_to_anchor=(
                LEGEND_BBOX_ANCHOR[0],
                LEGEND_BBOX_ANCHOR[1] + 0.18,
            ),
            ncol=1,
            frameon=False,
            fontsize=12.0,
        )
        fig.add_artist(attacc_legend)
    if front_handles:
        fig.legend(
            handles=front_handles,
            loc="lower right",
            bbox_to_anchor=LEGEND_BBOX_ANCHOR,
            ncol=1,
            frameon=False,
            fontsize=12.0,
        )


def plot_pair(
    workload_token: str,
    grouped_points: dict[str, list[EvalPoint]],
    attacc_point: MetricValues | None,
    channel_root: Path,
    x_metric: str,
    y_metric: str,
    pairtag: str,
    out: Path,
    dpi: int,
) -> dict[str, int]:
    fig, ax = plt.subplots(figsize=(13.5, 3.6))
    front_sizes: dict[str, int] = {}
    handles: list = []

    # AttAcc evaluated with the base-die model: AttAcc's GEMV problem is identical to
    # this reduction-place workload, and AttAcc's own mapping is infeasible on our
    # base-die arch, so we take the best-latency mapping from our base-die search
    # (config2) as the AttAcc reference point.
    base_die_pts = grouped_points.get("config2", [])
    if base_die_pts:
        attacc_point = min(
            base_die_pts, key=lambda p: p.value("latency_cycles")
        ).metrics

    # Normalize every metric to the AttAcc reference point (AttAcc -> (1, 1)).
    norm_x = attacc_point.value(x_metric) if attacc_point is not None else 1.0
    norm_y = attacc_point.value(y_metric) if attacc_point is not None else 1.0

    # Inclusivity is built on the EVAL side, not here: each config's pool already holds
    # the right inclusive mapping set re-costed under that level's evaluation -- bank =
    # config1 (bank eval); channel = config1 PU mappings re-cost under channel_level
    # ∪ channel pool (the _PUunion channel-root); base-die = config1 ∪ channel ∪ config2
    # re-cost under base-die. So we plot each config's own front directly (no pooling
    # of points evaluated under different archs).
    for config_id in CONFIG_ORDER:
        points = grouped_points.get(config_id, [])
        if not points:
            continue
        coords = [(p.value(x_metric) / norm_x, p.value(y_metric) / norm_y) for p in points]
        front = pareto_front(coords)
        front_sizes[config_id] = len(front)
        front_x = [c[0] for c in front]
        front_y = [c[1] for c in front]
        (line,) = ax.plot(
            front_x,
            front_y,
            color=CONFIG_COLORS[config_id],
            marker=CONFIG_MARKERS[config_id],
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=0.9,
            linewidth=4.8,
            linestyle=CONFIG_LINESTYLES[config_id],
            label=CONFIG_LABELS[config_id],
            zorder=3 if config_id == "channel" else 2,
        )
        handles.append(line)

    if attacc_point is not None:
        handles.append(
            ax.scatter(
                [1.0],
                [1.0],
                marker="*",
                s=420,
                color=ATTACC_MARKER_STYLE["color"],
                edgecolor="black",
                linewidths=1.0,
                zorder=5,
                label="AttAcc",
            )
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    configure_log_axis_ticks(ax.xaxis)
    configure_log_axis_ticks(ax.yaxis)
    ax.set_xlabel(METRIC_SHORT[x_metric], fontsize=26)
    ax.set_ylabel(METRIC_SHORT[y_metric], fontsize=26)
    ax.tick_params(axis="both", labelsize=22)
    ax.margins(0.07)
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.55)

    # Legend boxed in the top-right corner of the axes (the empty region: the data
    # trends lower-left, PU sits mid-right at lower y).
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=22,
        frameon=True,
        framealpha=0.92,
        borderpad=0.5,
        labelspacing=0.4,
    )

    if not grouped_points.get("channel"):
        ax.text(0.02, 0.02, "Channel-level front pending", transform=ax.transAxes,
                fontsize=7, ha="left", va="bottom")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return front_sizes


def discover_workload_csvs(root: Path) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise SystemExit(f"root directory not found: {root}")

    workload_csvs: list[tuple[str, Path]] = []
    for workload_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        evaluations_path = latest_evaluations_csv(workload_dir)
        if evaluations_path is None:
            print(
                f"warning: skipping {workload_dir}: no evaluations.csv found",
                file=sys.stderr,
            )
            continue
        workload_csvs.append((workload_dir.name, evaluations_path))

    return workload_csvs


def run_workload(
    workload_token: str,
    evaluations_path: Path,
    channel_root: Path,
    outdir: Path,
    dpi: int,
    metric_pairs: Sequence[tuple[str, str, str]] = METRIC_PAIRS,
) -> WorkloadSummary:
    main_points = read_evaluations(evaluations_path)
    channel_points, channel_path = read_channel_points(channel_root, workload_token)
    grouped_points = group_points([*main_points, *channel_points])
    ok_counts = {
        config_id: len(grouped_points.get(config_id, [])) for config_id in CONFIG_ORDER
    }

    # The AttAcc marker is computed inside plot_pair from the best-latency base-die
    # (config2) mapping, so it is drawn whenever config2 has points.
    base_die_present = bool(grouped_points.get("config2"))
    attacc_note = (
        "drawn (best-latency base-die)" if base_die_present else "skipped (no base-die)"
    )

    outputs: list[Path] = []
    front_sizes_by_pair: dict[str, dict[str, int]] = {}
    for x_metric, y_metric, pairtag in metric_pairs:
        out = (
            outdir
            / f"reduction_place_pareto_{workload_token}_{pairtag}.pdf"
        ).resolve()
        front_sizes_by_pair[pairtag] = plot_pair(
            workload_token=workload_token,
            grouped_points=grouped_points,
            attacc_point=None,
            channel_root=channel_root,
            x_metric=x_metric,
            y_metric=y_metric,
            pairtag=pairtag,
            out=out,
            dpi=dpi,
        )
        outputs.append(out)

    return WorkloadSummary(
        workload_token=workload_token,
        eval_path=evaluations_path.resolve(),
        ok_counts=ok_counts,
        front_sizes=front_sizes_by_pair,
        channel_path=channel_path.resolve() if channel_path is not None else None,
        channel_present=bool(grouped_points.get("channel")),
        attacc_drawn=base_die_present,
        attacc_note=attacc_note,
        outputs=outputs,
    )


def print_summary(
    summaries: Sequence[WorkloadSummary],
    metric_pairs: Sequence[tuple[str, str, str]] = METRIC_PAIRS,
) -> None:
    print(f"workloads plotted: {len(summaries)}")
    for summary in summaries:
        print(f"workload {summary.workload_token}:")
        print(f"  evaluations: {summary.eval_path}")
        print(f"  ok points: {format_counts(summary.ok_counts)}")
        if summary.channel_present:
            print(f"  channel: present ({summary.channel_path})")
        else:
            print("  channel: pending")
        print(f"  attacc: {summary.attacc_note}")

        front_parts = []
        for _x_metric, _y_metric, pairtag in metric_pairs:
            front_parts.append(
                f"{pairtag}: {format_counts(summary.front_sizes.get(pairtag, {}))}"
            )
        print(f"  fronts: {'; '.join(front_parts)}")
        print("  outputs:")
        for out in summary.outputs:
            print(f"    {out}")


def main() -> None:
    args = parse_args()

    root = Path(args.input_root)
    channel_root = Path(args.channel_input_root)
    workload_csvs = discover_workload_csvs(root)
    if not workload_csvs:
        raise SystemExit(f"no workloads with evaluations.csv found under {root}")

    if args.pairs is not None:
        selected = {tag.strip() for tag in args.pairs.split(",") if tag.strip()}
        known = {pairtag for _, _, pairtag in METRIC_PAIRS}
        unknown = selected - known
        if unknown:
            raise SystemExit(
                f"unknown --pairs entries {sorted(unknown)}; known: {sorted(known)}"
            )
        metric_pairs = tuple(p for p in METRIC_PAIRS if p[2] in selected)
    else:
        metric_pairs = METRIC_PAIRS

    summaries = [
        run_workload(
            workload_token=workload_token,
            evaluations_path=evaluations_path,
            channel_root=channel_root,
            outdir=args.outdir,
            dpi=args.dpi,
            metric_pairs=metric_pairs,
        )
        for workload_token, evaluations_path in workload_csvs
    ]

    print_summary(summaries, metric_pairs=metric_pairs)


if __name__ == "__main__":
    main()
