#!/usr/bin/env python3
"""Plot the TRUE compensated fixed-capacity geometry sweep for PNMAX Figure 11 (b)/(c).

This is the *compensated* ("compromise") construction: each subplot is ONE sweep over
bank-group size (pu), with the OTHER knob compensating to hold die capacity FIXED.

  (b) pu<->hmat  (vmat fixed at baseline):  pu 2/4/8*/16/32  =>  hmat 64/32/16*/8/4
  (c) pu<->vmat  (hmat fixed at baseline 16): pu 2/4/8*/16/32 =>  vmat 256/128/64*/32/16 (upmem)
                                                                       128/64/32*/16/8  (hbm)

Because capacity is held fixed, the x-axis is the bank-group-size (pu) ratio relative to the
baseline (pu-8): /4, /2, 1x, x2, x4. Each subplot is a single sweep with one line per metric
(Latency=blue, Area=orange, Energy=green), normalized to the baseline (x1) rung, with a dashed
y=1.0 reference and the geomean taken over the workloads. Styling matches the burst (a) plots.

Input: a geometry-sweep workload-space pareto-eval root laid out as
  <pareto_root>/<arch>/<workload>/<timestamp>/pareto_frontiers.csv

HBM-PIM carries only the /2 / x1 / x2 rungs (pu 4/8/16); the /4 and x4 extremes (pu 2/32) are
not characterized, so the HBM panels show 3 rungs (gaps at /4 and x4 are expected).
"""

import argparse
import csv
import math
import os
import re
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(os.environ.get("PNMAX_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_ROOT = Path(os.environ.get("PNMAX_RESULTS_ROOT", str(REPO_ROOT / "results")))
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "fig12_geometry"

ARCHES = ("upmem", "hbm_pim")
METRICS = {
    "latency": ("latency_cycles", "Latency"),
    "energy": ("energy", "Energy"),
    "area": ("total_area_mm2", "Area"),
}
# Combined view: all three metric trends on one axes, colour-coded to match the
# existing isolated-knob (burst) plots (Latency=blue, Area=orange, Energy=green).
METRIC_PLOT_ORDER = ("latency", "area", "energy")
METRIC_COLORS = {
    "latency": "#1f77b4",
    "area": "#ff7f0e",
    "energy": "#2ca02c",
}
METRIC_MARKERS = {
    "latency": "o",
    "area": "s",
    "energy": "^",
}
BASELINES = {
    "upmem": {"pu": 8, "hmat": 16, "vmat": 64},
    "hbm_pim": {"pu": 8, "hmat": 16, "vmat": 32},
}
# x-axis = pu (bank-group-size) ratio vs the pu-8 baseline.
PU_RATIO_LABELS = {
    0.25: "/4",
    0.5: "/2",
    1.0: "1x",
    2.0: "x2",
    4.0: "x4",
}
ARCH_TITLES = {
    "upmem": "UPMEM",
    "hbm_pim": "HBM-PIM",
}
PANEL_INFO = {
    # panel letter, compensating-knob label, which knob is held fixed at baseline
    "b": {"comp": "hmat", "fixed": "vmat", "comp_label": "row size (hmat)"},
    "c": {"comp": "vmat", "fixed": "hmat", "comp_label": "#rows (vmat)"},
}

_ARCH_CACHE: dict[str, dict[str, float] | None] = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the TRUE compensated fixed-capacity PU sweep (Fig 11 b/c)."
    )
    parser.add_argument(
        "pareto_root",
        type=Path,
        help=(
            "geometry-sweep workload-space pareto-eval root laid out as "
            "<pareto_root>/<arch>/<workload>/<timestamp>/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory for PDF figures (default: results/fig12_geometry)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="output figure DPI")
    parser.add_argument(
        "--latest-before",
        default=None,
        help="only consider run timestamps strictly before this value; default: latest",
    )
    parser.add_argument(
        "--metrics",
        default="latency,area,energy",
        help="comma-separated subset of latency,area,energy (default: all)",
    )
    parser.add_argument(
        "--chart",
        choices=("bar", "line", "both"),
        default="both",
        help="chart style to emit (default: both)",
    )
    return parser.parse_args()


def latest_run_dir(workload_dir, latest_before=None):
    candidates = [
        path
        for path in workload_dir.iterdir()
        if path.is_dir() and (path / "summary.json").is_file()
    ]
    if latest_before is not None:
        candidates = [path for path in candidates if path.name < latest_before]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_unit_bytes(value):
    if value is None:
        return None
    text = str(value).strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?B)", text, re.IGNORECASE)
    if match is None:
        return None
    number = float(match.group(1))
    unit = match.group(2).upper()
    scale = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit]
    return number * scale


def find_arch_file(arch_file):
    if not arch_file:
        return None
    path = Path(arch_file)
    if path.is_file():
        return path
    # Fallback: search by basename under data/archs. This path is relative to
    # the current working directory, so it only resolves when the script is run
    # from the repo root (as the experiment run.sh drivers do). An absolute or
    # repo-root-relative arch_file above always resolves regardless of cwd.
    matches = list(Path("data/archs").rglob(path.name))
    return matches[0] if matches else None


def load_arch_area(arch_file: str):
    """Load total area (mm^2) from a geometry arch YAML, cached by input path."""
    if arch_file in _ARCH_CACHE:
        return _ARCH_CACHE[arch_file]

    result = None
    path = find_arch_file(arch_file)
    if path is not None:
        try:
            payload = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            payload = None
        if isinstance(payload, dict):
            result = parse_float(payload.get("total_area_mm2"))

    _ARCH_CACHE[arch_file] = result
    return result


def geometry_knobs(geometry_id):
    values = {}
    for knob in ("pu", "hmat", "vmat"):
        match = re.search(rf"{knob}-(\d+)", geometry_id)
        if match is None:
            return None
        values[knob] = int(match.group(1))
    return values


def geomean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def read_geometry_best_latency(path):
    """Take all metrics from each geometry's best-latency frontier mapping."""
    best_rows = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geometry_id = row.get("geometry_id", "")
            if not geometry_id:
                continue
            latency = parse_float(row.get("latency_cycles"))
            energy = parse_float(row.get("energy"))
            mem = parse_float(row.get("mem_footprint_bytes"))
            if latency is None or energy is None:
                continue
            key = (latency, energy, mem if mem is not None else float("inf"))
            if geometry_id not in best_rows or key < best_rows[geometry_id][0]:
                best_rows[geometry_id] = (key, row)

    bests = {}
    for geometry_id, ((latency, energy, _mem), row) in best_rows.items():
        area = load_arch_area(row.get("geometry_arch_file", ""))
        record = {"latency": latency, "energy": energy}
        if area is not None:
            record["area"] = area
        bests[geometry_id] = record
    return bests


def is_baseline_geometry(knobs, arch):
    baseline = BASELINES[arch]
    return all(knobs[knob] == baseline[knob] for knob in baseline)


def classify_panel(knobs, arch):
    """Which compensated panel a geometry belongs to.

    (b) pu<->hmat keeps vmat fixed at baseline (and varies pu & hmat together).
    (c) pu<->vmat keeps hmat fixed at baseline (and varies pu & vmat together).
    The shared baseline (pu-8/hmat-16/vmat-base) belongs to BOTH panels (x1 rung).
    """
    baseline = BASELINES[arch]
    panels = []
    if knobs["vmat"] == baseline["vmat"]:  # (b): vmat held fixed
        panels.append("b")
    if knobs["hmat"] == baseline["hmat"]:  # (c): hmat held fixed
        panels.append("c")
    return panels


def workload_panel_ratios(run_dir, arch):
    """Return {panel: {pu_ratio: {metric: normalized_to_baseline}}} for one workload."""
    csv_path = run_dir / "pareto_frontiers.csv"
    if not csv_path.is_file():
        return None

    bests = read_geometry_best_latency(csv_path)
    parsed = {}
    baseline_record = None
    for geometry_id, values in bests.items():
        knobs = geometry_knobs(geometry_id)
        if knobs is None:
            continue
        parsed[geometry_id] = (knobs, values)
        if is_baseline_geometry(knobs, arch):
            baseline_record = values
    if baseline_record is None:
        return None

    out = {"b": {}, "c": {}}
    base_pu = BASELINES[arch]["pu"]
    for geometry_id, (knobs, values) in parsed.items():
        pu_ratio = knobs["pu"] / base_pu
        for panel in classify_panel(knobs, arch):
            rung = out[panel].setdefault(pu_ratio, {})
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
                ratio = metric_value / base_value
                # best (min) across replicas if a rung recurs
                if metric not in rung or ratio < rung[metric]:
                    rung[metric] = ratio
    return out


def discover_arch_data(pareto_root, latest_before=None):
    arch_data = {}
    for arch in ARCHES:
        arch_dir = pareto_root / arch
        if not arch_dir.is_dir():
            continue

        # {panel: {pu_ratio: {metric: [per-workload ratios]}}}
        acc = {"b": {}, "c": {}}
        workload_count = 0
        for workload_dir in sorted(p for p in arch_dir.iterdir() if p.is_dir()):
            run_dir = latest_run_dir(workload_dir, latest_before)
            if run_dir is None:
                continue
            ratios = workload_panel_ratios(run_dir, arch)
            if ratios is None:
                continue
            workload_count += 1
            for panel, rung_ratios in ratios.items():
                for pu_ratio, metrics in rung_ratios.items():
                    for metric, value in metrics.items():
                        acc[panel].setdefault(pu_ratio, {}).setdefault(
                            metric, []
                        ).append(value)

        normalized = {"b": {}, "c": {}}
        for panel, rung_values in acc.items():
            for pu_ratio, metrics in rung_values.items():
                for metric, values in metrics.items():
                    if values:
                        normalized[panel].setdefault(pu_ratio, {})[metric] = geomean(
                            values
                        )

        arch_data[arch] = {
            "workload_count": workload_count,
            "normalized": normalized,
        }
    return arch_data


def pu_tick_label(ratio):
    for expected, label in PU_RATIO_LABELS.items():
        if math.isclose(ratio, expected, rel_tol=1e-6, abs_tol=1e-9):
            return label
    return f"{ratio:g}x"


def parse_metrics(metrics_text):
    metrics = [m.strip() for m in metrics_text.split(",") if m.strip()]
    invalid = [m for m in metrics if m not in METRICS]
    if invalid:
        raise SystemExit(f"unknown metric(s): {', '.join(invalid)}")
    return metrics


def panel_pu_ratios(normalized, panel, metrics):
    ratios = set()
    for pu_ratio, values in normalized.get(panel, {}).items():
        if any(metric in values for metric in metrics):
            ratios.add(pu_ratio)
    return sorted(ratios)


def values_for(normalized, panel, pu_ratios, metric):
    return [
        normalized.get(panel, {}).get(pu_ratio, {}).get(metric, float("nan"))
        for pu_ratio in pu_ratios
    ]


def annotate_points(ax, x_positions, values):
    for x, value in zip(x_positions, values):
        if math.isnan(value):
            continue
        ax.annotate(
            f"{value:.2g}",
            (x, value),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
            color="0.25",
        )


def plot_panel(ax, arch, panel, normalized, metrics, chart):
    info = PANEL_INFO[panel]
    ordered = [m for m in METRIC_PLOT_ORDER if m in metrics]
    pu_ratios = panel_pu_ratios(normalized, panel, metrics)
    x_positions = list(range(len(pu_ratios)))

    if chart == "bar":
        n_series = max(len(ordered), 1)
        width = 0.8 / n_series
        for idx, metric in enumerate(ordered):
            offset = (idx - (n_series - 1) / 2) * width
            values = values_for(normalized, panel, pu_ratios, metric)
            ax.bar(
                [x + offset for x in x_positions],
                values,
                width=width,
                color=METRIC_COLORS[metric],
                alpha=0.9,
                label=METRICS[metric][1],
            )
    else:
        for metric in ordered:
            values = values_for(normalized, panel, pu_ratios, metric)
            ax.plot(
                x_positions,
                values,
                color=METRIC_COLORS[metric],
                marker=METRIC_MARKERS[metric],
                markersize=5,
                linewidth=1.9,
                label=METRICS[metric][1],
            )
            annotate_points(ax, x_positions, values)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([pu_tick_label(r) for r in pu_ratios])
    ax.set_xlabel("BG-size (pu) ratio vs baseline  [capacity fixed]")
    ax.set_title(
        f"{ARCH_TITLES[arch]}  ({panel}) BG-size sweep, {info['comp_label']} compensating",
        fontsize=10,
    )
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.margins(y=0.25)
    return x_positions


def plot_arch(arch, arch_record, chart, output_dir, dpi, metrics):
    normalized = arch_record["normalized"]
    panels = ("b", "c")
    if not any(panel_pu_ratios(normalized, p, metrics) for p in panels):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, panel in zip(axes, panels):
        plot_panel(ax, arch, panel, normalized, metrics, chart)

    axes[0].set_ylabel("Normalized to baseline (x1)")
    handles = [
        Line2D(
            [0],
            [0],
            color=METRIC_COLORS[m],
            marker=METRIC_MARKERS[m],
            lw=2.2,
            label=METRICS[m][1],
        )
        for m in METRIC_PLOT_ORDER
        if m in metrics
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="best", title="Metric")
    fig.suptitle(
        f"{ARCH_TITLES[arch]} compensated fixed-capacity BG-size sweep "
        f"(geomean over {arch_record['workload_count']} workloads)"
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"geometry_sweep_compensated_bc_{arch}_{chart}.pdf"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main() -> int:
    args = parse_args()
    metrics = parse_metrics(args.metrics)
    charts = ("bar", "line") if args.chart == "both" else (args.chart,)
    arch_data = discover_arch_data(args.pareto_root, args.latest_before)

    saved = 0
    for arch, arch_record in sorted(arch_data.items()):
        for chart in charts:
            out = plot_arch(arch, arch_record, chart, args.output_dir, args.dpi, metrics)
            if out is None:
                print(f"{arch} {chart}: no data to plot")
                continue
            saved += 1
            print(out.resolve())

    if saved == 0:
        print(f"no plots generated from {args.pareto_root}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
