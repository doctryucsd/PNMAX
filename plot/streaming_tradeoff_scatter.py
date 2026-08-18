#!/usr/bin/env python3
"""Streaming trade-off scatter, facets by architecture (Fig 11).

Renders the "no-streaming" latency-memory trade-off figure.
Each point is the best-latency mapping of a Fig-8 Pareto front, plotted as the
PERCENT CHANGE in latency (x) and memory footprint (y) when streaming is DISABLED,
relative to the streaming-enabled mapping (the origin, 0,0).

Message conveyed:
  - Disabling streaming greatly INCREASES memory footprint (points sit high above 0%).
  - Latency change is ARCHITECTURE-DEPENDENT: UPMEM has high streaming overhead, so
    disabling streaming makes it much FASTER (large negative latency -> far left);
    HBM-PIM hugs ~0%.
  - UPMEM's buffer step (DSE space d) gives a LESS-evident latency improvement: its (d)
    points sit near 0%, unlike (a)(b)(c) ~ -48%.

Color encodes DSE space (a-d), matching Fig 8 (SPACE_COLORS). Each panel uses its OWN
independent x/y axis limits so every architecture fits to its own latency/memory range
(UPMEM spreads far left across its full range; HBM-PIM zooms into its much smaller range).
Both panels keep tick labels visible so each scale is clear.

Reuses the shared data pipeline by importing
``workload_space_streaming_delta_scatter.py`` (same directory) by path.

LEGEND: compact (3,2) column-major legend (5 entries = DSE spaces a-d + the baseline
origin marker) tucked into the UPMEM (left) panel's top-left corner. No averaged a->d
arrow and no in-plot "streaming" text annotation (cleaner panels).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ---- import the production figure module by path (data pipeline + constants) ----
PLOT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_fig11", PLOT_DIR / "workload_space_streaming_delta_scatter.py"
)
m = importlib.util.module_from_spec(_spec)
sys.modules["_fig11"] = m
_spec.loader.exec_module(m)

REPO_ROOT = Path(os.environ.get("PNMAX_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_ROOT = Path(os.environ.get("PNMAX_RESULTS_ROOT", str(REPO_ROOT / "results")))
DEFAULT_INPUT_DIR = RESULTS_ROOT / "workload_space_pareto_eval"
ARCH_TITLE = {"upmem": "UPMEM", "hbm_pim": "HBM-PIM"}
ARCH_ORDER = ("upmem", "hbm_pim")
DEFAULT_OUTPUT_PATH = (
    RESULTS_ROOT / "fig11_streaming" / "streaming_tradeoff_scatter.png"
)
DPI = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming trade-off scatter (percent change vs streaming run)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Workload-space Pareto evaluation output root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=DPI, help="Figure DPI.")
    return parser.parse_args()


def load_points(report_root: Path):
    runs = m._discover_latest_runs(report_root)
    pts = []
    for (arch, workload), run_dir in sorted(runs.items()):
        pts.extend(
            m._build_scatter_points_for_run(run_dir=run_dir, arch=arch, workload=workload)
        )
    return m._filter_scatter_points(pts)


def main():
    args = parse_args()
    points = load_points(args.input_dir)
    spaces = m._ordered_space_names({p.space for p in points})
    archs = [a for a in ARCH_ORDER if any(p.arch == a for p in points)]

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
        }
    )

    # Per-arch INDEPENDENT axes: do NOT share x/y so each architecture fits to its
    # own latency/memory range (UPMEM full range; HBM-PIM zooms into its small range).
    fig, axes = plt.subplots(
        1, len(archs), figsize=(13.0, 4.8), sharex=False, sharey=False
    )
    if len(archs) == 1:
        axes = [axes]

    pct = FuncFormatter(m._percentage_tick_label)

    for ax, arch in zip(axes, archs):
        ap = [p for p in points if p.arch == arch]

        # vertical 0% reference line (streaming-enabled latency baseline)
        ax.axvline(0.0, ls="--", color="0.45", lw=1.0, zorder=1)
        # horizontal 0% reference line (streaming-enabled memory baseline)
        ax.axhline(0.0, ls="--", color="0.45", lw=1.0, zorder=1)

        # scatter points, colored by DSE space (Fig 8 colors)
        for space in spaces:
            sub = [p for p in ap if p.space == space]
            if not sub:
                continue
            ax.scatter(
                [p.centered_latency for p in sub],
                [p.centered_mem_footprint for p in sub],
                marker="o",
                s=210,
                alpha=0.82,
                facecolor=m._color_for_space(space),
                edgecolor="white",
                linewidths=0.7,
                zorder=4,
            )

        # explicit ORIGIN marker = streaming-enabled baseline (unlabeled in-plot;
        # the in-plot "streaming" text annotation has been removed for cleanliness).
        ax.scatter(
            [0.0],
            [0.0],
            marker="*",
            s=430,
            facecolor="none",
            edgecolor="black",
            linewidths=1.3,
            zorder=5,
        )

        ax.set_title(ARCH_TITLE.get(arch, arch), fontweight="bold")
        ax.set_xlabel("Latency delta (%)")
        ax.xaxis.set_major_formatter(pct)
        ax.yaxis.set_major_formatter(pct)
        # keep tick labels visible on BOTH panels so each independent scale is clear
        ax.tick_params(axis="both", labelleft=True, labelbottom=True)
        # each panel sets its own y-axis label since the y-scales now differ
        ax.set_ylabel("Mem. footprint delta (%)")
        ax.grid(True, ls="--", lw=0.6, alpha=0.3, zorder=0)

    # Compact (3,2) DSE-space legend tucked into the UPMEM (left) panel's TOP-LEFT.
    # 5 entries = DSE spaces (a)-(d) + the baseline origin marker. matplotlib fills
    # legend columns COLUMN-MAJOR, so ncol=2 over 5 handles yields 3 in col 1, 2 in
    # col 2 (the requested "(3,2)" shape).
    space_handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=9,
            markerfacecolor=m._color_for_space(s),
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=m.SPACE_LABELS.get(s, f"({s})"),
        )
        for s in spaces
    ]
    origin_handle = Line2D(
        [],
        [],
        linestyle="None",
        marker="*",
        markersize=13,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.2,
        label="baseline (streaming)",
    )
    axes[0].legend(
        handles=space_handles + [origin_handle],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.97),
        ncol=1,
        frameon=True,
        framealpha=0.85,
        edgecolor="0.7",
        fontsize=16,
        columnspacing=1.0,
        handletextpad=0.4,
        labelspacing=0.35,
        borderpad=0.5,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output} ({len(points)} points, archs={archs}, spaces={list(spaces)})")


if __name__ == "__main__":
    main()
