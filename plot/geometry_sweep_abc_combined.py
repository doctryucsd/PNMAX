#!/usr/bin/env python3
"""Combined Figure-11 redesign in ONE figure: rows = panels (a)/(b)/(c),
columns = architectures (UPMEM, HBM-PIM):

    UPMEM (a)   HBM-PIM (a)      # SIMD lanes (specs.simd_lanes per arch YAML)
    UPMEM (b)   HBM-PIM (b)      # BG-size vs row size (hmat), capacity fixed
    UPMEM (c)   HBM-PIM (c)      # BG-size vs #rows (vmat),  capacity fixed

All cells are on the 16x corrected (iso_capacity_archs) basis. Every metric
(Latency=blue, Area=orange, Energy=green) is normalized to the baseline
(x1) rung, geomean over the workloads, with a dashed y=1.0 reference.

A SHARED x-position grid is used per row so the spacing between points is
identical across UPMEM and HBM-PIM. HBM-PIM (b)/(c) only have the /2,1x,x2
rungs (the /4 and x4 extremes are uncharacterized), so they appear at the
same positions as UPMEM with gaps at /4 and x4. Rows (b)/(c) reuse
geometry_sweep_compensated_bc.py.

Inputs (each a workload-space pareto-eval root <root>/<arch>/<workload>/<ts>/pareto_frontiers.csv):
  --bc-root     compensated (b)/(c) sweep   (default results/fig12_geometry/compensated_bc_host_congested/pareto_eval)
  --burst-root  isolated burst (a) sweep    (default results/fig12_geometry/isolated_burst_sweep_host_congested/pareto_eval)
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Reuse the compensated (b)/(c) machinery so the rows are byte-for-byte consistent.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_sweep_compensated_bc as comp  # noqa: E402

REPO_ROOT = Path(os.environ.get("PNMAX_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_ROOT = Path(os.environ.get("PNMAX_RESULTS_ROOT", str(REPO_ROOT / "results")))
DEFAULT_RESULTS_DIR = RESULTS_ROOT / "fig12_geometry"
DEFAULT_BC_ROOT = DEFAULT_RESULTS_DIR / "compensated_bc_host_congested" / "pareto_eval"
DEFAULT_BURST_ROOT = DEFAULT_RESULTS_DIR / "isolated_burst_sweep_host_congested" / "pareto_eval"

ARCHES = ("upmem", "hbm_pim")
METRICS = comp.METRICS
METRIC_PLOT_ORDER = comp.METRIC_PLOT_ORDER
METRIC_COLORS = comp.METRIC_COLORS
METRIC_MARKERS = comp.METRIC_MARKERS
ARCH_TITLES = comp.ARCH_TITLES

# The (a)-row rungs are SIMD-lane counts read from each geometry's arch YAML
# (specs.simd_lanes), not inferred from geometry file names. The baseline (1x)
# rung is the family's native lane count, read from its lowered baseline arch.
BASELINE_ARCH_DIR = REPO_ROOT / "data" / "archs" / "lowered" / "baseline"


def _specs_simd_lanes(payload) -> int | None:
    specs = payload.get("specs") if isinstance(payload, dict) else None
    lanes = specs.get("simd_lanes") if isinstance(specs, dict) else None
    return lanes if isinstance(lanes, int) and lanes > 0 else None


def family_baseline_lanes(arch: str) -> int:
    path = BASELINE_ARCH_DIR / f"{arch}.yaml"
    lanes = _specs_simd_lanes(yaml.safe_load(path.read_text()))
    if lanes is None:
        raise ValueError(f"missing specs.simd_lanes in {path}")
    return lanes


BASELINE_LANES = {arch: family_baseline_lanes(arch) for arch in ARCHES}
_BASELINE_LANE_SET = set(BASELINE_LANES.values())
if len(_BASELINE_LANE_SET) != 1:
    raise ValueError(
        f"shared (a)-row labels expect one baseline lane count, got {BASELINE_LANES}"
    )
_SHARED_BASELINE_LANES = next(iter(_BASELINE_LANE_SET))

# Shared x-position grids (so UPMEM and HBM-PIM share identical point spacing).
LANES = (4, 8, 16, 32, 64)
LANE_LABELS = tuple("1x" if l == _SHARED_BASELINE_LANES else str(l) for l in LANES)
BC_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
BC_LABELS = ("/4", "/2", "1x", "x2", "x4")

# >>> ADJUST HERE: one knob scaling every font in the figure (1.0 = original).
FONT_SCALE = 2.3
FS_TITLE = 9 * FONT_SCALE      # subplot titles
FS_LABEL = 8 * FONT_SCALE      # axis labels
FS_TICK = 8 * FONT_SCALE       # tick labels
FS_ANNOT = 6.5 * FONT_SCALE    # per-point value annotations
FS_LEGEND = 7.5 * FONT_SCALE   # legend entries

# Line/marker weight, and how much blank padding sits above/below the data
# (smaller Y_MARGIN / LAYOUT_PAD = less white background).
LINE_WIDTH = 4.2
MARKER_SIZE = 11
Y_MARGIN = 0.05
LAYOUT_PAD = 0.3


_SIMD_LANES_CACHE: dict[str, int] = {}


def geometry_arch_files(csv_path: Path) -> dict[str, str]:
    """geometry_id -> geometry_arch_file as recorded in a pareto_frontiers.csv."""
    arch_files: dict[str, str] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            geometry_id = row.get("geometry_id", "")
            arch_file = (row.get("geometry_arch_file") or "").strip()
            if geometry_id and arch_file:
                arch_files.setdefault(geometry_id, arch_file)
    return arch_files


def simd_lanes_of(geometry_id: str, arch_file: str) -> int:
    """SIMD lanes for one geometry, read from its arch YAML (specs.simd_lanes)."""
    if arch_file in _SIMD_LANES_CACHE:
        return _SIMD_LANES_CACHE[arch_file]
    path = comp.find_arch_file(arch_file)
    lanes = _specs_simd_lanes(yaml.safe_load(path.read_text())) if path is not None else None
    if lanes is None:
        raise ValueError(
            f"cannot read specs.simd_lanes for geometry '{geometry_id}' "
            f"from arch file {arch_file!r}"
        )
    _SIMD_LANES_CACHE[arch_file] = lanes
    return lanes


def discover_burst_data(burst_root: Path, arch: str, latest_before=None):
    """{simd_lanes: {metric: geomean-over-workloads ratio-to-baseline}} for the (a) row."""
    arch_dir = burst_root / arch
    if not arch_dir.is_dir():
        return {}, 0

    acc: dict[int, dict[str, list[float]]] = {}
    workload_count = 0
    for workload_dir in sorted(p for p in arch_dir.iterdir() if p.is_dir()):
        run_dir = comp.latest_run_dir(workload_dir, latest_before)
        if run_dir is None:
            continue
        csv_path = run_dir / "pareto_frontiers.csv"
        if not csv_path.is_file():
            continue
        bests = comp.read_geometry_best_latency(csv_path)
        arch_files = geometry_arch_files(csv_path)

        baseline_record = None
        parsed = {}
        for geometry_id, values in bests.items():
            lanes = simd_lanes_of(geometry_id, arch_files.get(geometry_id, ""))
            parsed[lanes] = values
            if lanes == BASELINE_LANES[arch]:
                baseline_record = values
        if baseline_record is None:
            continue
        workload_count += 1
        for lanes, values in parsed.items():
            rung = acc.setdefault(lanes, {})
            for metric in METRICS:
                base_value = baseline_record.get(metric)
                metric_value = values.get(metric)
                if (
                    base_value is None
                    or metric_value is None
                    or base_value <= 0
                    or metric_value <= 0
                ):
                    continue
                rung.setdefault(metric, []).append(metric_value / base_value)

    normalized = {}
    for lanes, metrics in acc.items():
        for metric, ratios in metrics.items():
            if ratios:
                normalized.setdefault(lanes, {})[metric] = comp.geomean(ratios)
    return normalized, workload_count


def plot_cell(ax, keys, labels, source, chart, xlabel, spread=True):
    full = len(keys)
    if spread:
        # Plot only the rungs that have data, spread EVENLY across the full canonical
        # width. So a subplot missing the /4 and x4 extremes (HBM-PIM (b)/(c)) stretches
        # its present points to fill the same box -> bigger intervals; the baseline (1x)
        # stays centered, so the subfigures align column-to-column.
        present = [(k, lab) for k, lab in zip(keys, labels) if k in source]
        n = len(present)
        if n == 0:
            ax.set_xlim(-0.5, full - 0.5)
            return
        positions = ([(full - 1) / 2.0] if n == 1
                     else [i * (full - 1) / (n - 1) for i in range(n)])
        pkeys = [k for k, _ in present]
        plabels = [lab for _, lab in present]
    else:
        # Fixed canonical slots: each rung sits at its own x position, missing rungs
        # leave a gap. Keeps the baseline column-aligned even with asymmetric gaps.
        positions = list(range(full))
        pkeys = list(keys)
        plabels = list(labels)
    interval = (positions[1] - positions[0]) if len(positions) > 1 else 1.0

    ordered = list(METRIC_PLOT_ORDER)
    if chart == "bar":
        width = (0.8 * interval) / max(len(ordered), 1)
        for idx, metric in enumerate(ordered):
            offset = (idx - (len(ordered) - 1) / 2) * width
            values = [source.get(k, {}).get(metric, float("nan")) for k in pkeys]
            ax.bar([p + offset for p in positions], values, width=width,
                   color=METRIC_COLORS[metric], alpha=0.9, label=METRICS[metric][1])
    else:
        for metric in ordered:
            values = [source.get(k, {}).get(metric, float("nan")) for k in pkeys]
            ax.plot(positions, values, color=METRIC_COLORS[metric],
                    marker=METRIC_MARKERS[metric], markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH, label=METRICS[metric][1])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(plabels)
    ax.set_xlim(-0.5, full - 0.5)  # same box across columns; present points fill it
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.margins(y=Y_MARGIN)
    ax.tick_params(labelsize=FS_TICK)


def build_figure(bc_all, burst_by_arch, chart, output_dir, dpi,
                 name_stem="geometry_sweep_abc_combined"):
    # Subplot x:y aspect = (width/2) : (height/3). Bump WIDTH up / HEIGHT down
    # here to make each subplot wider (bigger x:y); reverse to make them taller.
    fig, axes = plt.subplots(3, 2, figsize=(15, 11.0))

    # The x-axis label carries the panel identity + what differs between rows
    # (since per-subplot titles are removed; the arch name is the column header).
    row_specs = [
        ("a", LANES, LANE_LABELS, "(a) PU tile size",
         lambda arch: burst_by_arch[arch][0]),
        ("b", BC_RATIOS, BC_LABELS, "(b) BG-size ratio, row size (hmat) comp.",
         lambda arch: bc_all.get(arch, {}).get("normalized", {}).get("b", {})),
        ("c", BC_RATIOS, BC_LABELS, "(c) BG-size ratio, #rows (vmat) comp.",
         lambda arch: bc_all.get(arch, {}).get("normalized", {}).get("c", {})),
    ]

    any_data = False
    for r, (panel, keys, labels, xlabel, getter) in enumerate(row_specs):
        for c, arch in enumerate(ARCHES):
            ax = axes[r][c]
            source = getter(arch)
            if source:
                any_data = True
            plot_cell(ax, keys, labels, source, chart, xlabel)
            if r == 0:  # arch name as a single column header above all 3 rows
                ax.set_title(ARCH_TITLES[arch], fontsize=FS_TITLE * 1.3, fontweight="bold")
            if c == 0:
                ax.set_ylabel("norm. to 1x", fontsize=FS_LABEL)

    if not any_data:
        plt.close(fig)
        return None

    handles = [
        Line2D([0], [0], color=METRIC_COLORS[m], marker=METRIC_MARKERS[m], lw=LINE_WIDTH,
               label=METRICS[m][1])
        for m in METRIC_PLOT_ORDER
    ]
    axes[0][0].legend(handles=handles, fontsize=FS_LEGEND, loc="best",
                      title="Metric", title_fontsize=FS_LEGEND)

    fig.tight_layout(pad=LAYOUT_PAD)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{name_stem}_{chart}.pdf"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bc-root", type=Path, default=DEFAULT_BC_ROOT)
    p.add_argument("--burst-root", type=Path, default=DEFAULT_BURST_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--chart", choices=("bar", "line", "both"), default="both")
    p.add_argument("--latest-before", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    charts = ("bar", "line") if args.chart == "both" else (args.chart,)
    bc_all = comp.discover_arch_data(args.bc_root, args.latest_before)
    burst_by_arch = {
        arch: discover_burst_data(args.burst_root, arch, args.latest_before)
        for arch in ARCHES
    }

    saved = 0
    for chart in charts:
        out = build_figure(bc_all, burst_by_arch, chart, args.output_dir, args.dpi)
        if out is None:
            print(f"{chart}: no data")
            continue
        saved += 1
        print(out.resolve())
    if saved == 0:
        print("no plots generated")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
