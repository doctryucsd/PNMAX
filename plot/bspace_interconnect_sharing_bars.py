#!/usr/bin/env python3
"""Best (b)-space mapping under interconnect/sharing configs.

Per (arch, workload):
1. Evaluate every spaces/b mapping (block + PU sharing, no system sharing) on the
   interconnect-sweep base arch and keep the best-latency one.  b mappings have
   sharing.system = 0, so the bank-to-bank cost never fires and the selection is
   interconnect-variant independent.
2. Re-evaluate that one mapping under the requested configs (cache and
   sharing.pu forced True like the interconnect sweep; the model rejects
   system sharing without PU sharing):
     host-x16  — sharing.system = 5 (host scope), bank_to_bank = degraded x16 host
     host-x1   — sharing.system = 5 (host scope), bank_to_bank = original x1 host
     intra-bg  — sharing.system = 2 (bank-group scope), bank_to_bank = intra-bg
     inter-bg  — sharing.system = 5 (host scope), bank_to_bank = inter-bg
3. Grouped bar chart, one group per (arch, workload), one bar per config.
   All bars are normalized to the same mapping evaluated with ALL sharing off
   (block=False, pu=False, system=0) and cache on — the no-sharing reference,
   for which the bank-to-bank cost never fires.

Default run: host-x16 + intra-bg + inter-bg.
Comparison run: --host-cost x1 --no-intra-bg --drop-no-duplication
(host-x1 + inter-bg, groups with filter and output inter-bank ratios 0 dropped).

Outputs: <output>.pdf and <output>.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(os.environ.get("PNMAX_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_ROOT = Path(os.environ.get("PNMAX_RESULTS_ROOT", str(REPO_ROOT / "results")))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pnmax.cli._workload_eval_overrides import apply_interconnect_workload_overrides
from pnmax.cli.run_workload_space_pareto_eval import _load_arch_for_worker
from pnmax.database.workload import Workload
from pnmax.dse.workload_eval import evaluate_workload_analytical_report

DEFAULT_SEARCH_ROOT = (
    RESULTS_ROOT / "workload_space_random_search" / "no_streaming_false"
)
SEARCH_ROOT = DEFAULT_SEARCH_ROOT  # overridden by --search-root in main()
ARCH_SEARCH_DIR = {"upmem": "upmem", "hbm_pim": "hbm-pim"}
WORKLOAD_TOKENS = (
    "alexnet-4",
    "bert-3",
    "bert-72",
    "llama128-6",
    "llama256-3",
    "llama512-3",
    "resnet50-2",
    "resnet50-53",
    "vgg16-0",
)
ARCH_FILES = {
    "upmem": {
        "base": "data/archs/lowered/interconnect_sweep/upmem/upmem__pu-8__hmat-16__vmat-64.yaml",
        "host-x1": "data/archs/lowered/interconnect_sweep/upmem/upmem__pu-8__hmat-16__vmat-64_interconnect-host-x1.yaml",
        "intra-bg": "data/archs/lowered/interconnect_sweep/upmem/upmem__pu-8__hmat-16__vmat-64_interconnect-intra-bg.yaml",
        "inter-bg": "data/archs/lowered/interconnect_sweep/upmem/upmem__pu-8__hmat-16__vmat-64_interconnect-inter-bg.yaml",
    },
    "hbm_pim": {
        "base": "data/archs/lowered/interconnect_sweep/hbm_pim/hbm_pim__pu-8__hmat-16__vmat-32.yaml",
        "host-x1": "data/archs/lowered/interconnect_sweep/hbm_pim/hbm_pim__pu-8__hmat-16__vmat-32_interconnect-host-x1.yaml",
        "intra-bg": "data/archs/lowered/interconnect_sweep/hbm_pim/hbm_pim__pu-8__hmat-16__vmat-32_interconnect-intra-bg.yaml",
        "inter-bg": "data/archs/lowered/interconnect_sweep/hbm_pim/hbm_pim__pu-8__hmat-16__vmat-32_interconnect-inter-bg.yaml",
    },
}
# config_id -> (arch-file key, forced sharing.system level)
ALL_CONFIGS = {
    "host-x16": ("base", 5),
    "host-x1": ("host-x1", 5),
    "intra-bg": ("intra-bg", 2),
    "inter-bg": ("inter-bg", 5),
}
CONFIG_LABELS = {
    "host-x16": "host x16 (sys-share L5)",
    "host-x1": "host x1 (sys-share L5)",
    "intra-bg": "intra-bg (sys-share L2)",
    "inter-bg": "inter-bg (sys-share L5)",
}
# Short legend names used in the combined ("all") figure.
CONFIG_SHORT = {
    "host-x16": "via host",
    "host-x1": "via host",
    "intra-bg": "intra-bank",
    "inter-bg": "inter-bank",
}
CONFIG_COLORS = {
    "host-x16": "#d62728",
    "host-x1": "#d62728",
    "intra-bg": "#2ca02c",
    "inter-bg": "#1f77b4",
}
ARCH_LABEL = {"upmem": "UPMEM", "hbm_pim": "HBM-PIM"}

DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "fig14_interconnect"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "bspace_interconnect_sharing_bars.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--search-root",
        type=Path,
        default=DEFAULT_SEARCH_ROOT,
        help="workload-space random-search root holding spaces/b mappings",
    )
    parser.add_argument(
        "--host-cost",
        choices=("congested", "x1"),
        default="congested",
        help=(
            "bank_to_bank cost for the host config: 'congested' (the 16x "
            "degraded host interconnect baked into the sweep center files, "
            "the Fig. 14 default) or 'x1' (original host cost)"
        ),
    )
    parser.add_argument(
        "--no-intra-bg",
        action="store_true",
        help="drop the intra-bg config (baseline then becomes the host config)",
    )
    parser.add_argument(
        "--drop-no-duplication",
        action="store_true",
        help="drop groups whose mapping has inter-bank ratio 0 under forced sharing",
    )
    parser.add_argument(
        "--metric",
        choices=("latency", "energy", "all"),
        default="latency",
        help=(
            "which metric the bars show; 'all' merges latency + energy (per "
            "config) and the config-independent memory footprint into one chart "
            "(all metrics are recorded in the CSV either way)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "64")),
        help="process pool size for the spaces/b selection pass",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--replot-csv",
        type=Path,
        default=None,
        help="skip evaluation; re-render the figure from an existing results CSV",
    )
    return parser.parse_args()


def _load_records_from_csv(path: Path) -> list[dict[str, object]]:
    """Load a previously written results CSV back into plot records (norm_* -> float)."""
    records: list[dict[str, object]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rec: dict[str, object] = dict(row)
            for key in rec:
                if key.startswith("norm_"):
                    rec[key] = float(rec[key])
            records.append(rec)
    return records


def _eval_native(task: tuple[str, str, str]) -> tuple[str, tuple[int, int, int] | None]:
    """Evaluate one b-space trace as-is on the base arch. Returns (path, key|None)."""
    trace_path, arch_file, target = task
    try:
        arch = _load_arch_for_worker(arch_file)
        workload = Workload.from_file(trace_path)
        report = evaluate_workload_analytical_report(
            workload,
            arch,
            target=target,
            latency_coeff=1.0,
            mem_footprint_coeff=1.0,
            energy_coeff=1.0,
            require_constraints=True,
            refresh=True,
        )
    except Exception:
        return trace_path, None
    if report is None:
        return trace_path, None
    return trace_path, (
        int(report["latency_cycles"]),
        int(report["mem_footprint_bytes"]),
        int(report["energy"]),
    )


def _select_best_b_mapping(
    arch_name: str, workload_token: str, workers: int
) -> tuple[Path, tuple[int, int, int]]:
    b_dir = SEARCH_ROOT / ARCH_SEARCH_DIR[arch_name] / workload_token / "spaces" / "b"
    traces = sorted(b_dir.glob("trace_*.yaml"))
    if not traces:
        raise ValueError(f"No spaces/b traces under {b_dir}")
    arch_file = str(REPO_ROOT / ARCH_FILES[arch_name]["base"])
    tasks = [(str(path), arch_file, arch_name) for path in traces]

    results: list[tuple[str, tuple[int, int, int] | None]] = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_eval_native, tasks, chunksize=32))
    except BrokenExecutor:
        print(f"  [warn] worker pool died for {arch_name}/{workload_token}; retrying serially", flush=True)
        results = [_eval_native(task) for task in tasks]

    scored = [(key, path) for path, key in results if key is not None]
    if not scored:
        raise ValueError(f"No successful spaces/b evaluations for {arch_name}/{workload_token}")
    (latency, mem, energy), best_path = min(scored)
    print(
        f"  best b mapping: {Path(best_path).name} latency={latency} "
        f"({len(scored)}/{len(tasks)} ok)",
        flush=True,
    )
    return Path(best_path), (latency, mem, energy)


def _eval_config(trace_path: Path, arch_name: str, config_id: str) -> dict[str, object]:
    arch_key, sharing_level = ALL_CONFIGS[config_id]
    arch = _load_arch_for_worker(str(REPO_ROOT / ARCH_FILES[arch_name][arch_key]))
    workload = Workload.from_file(trace_path)
    apply_interconnect_workload_overrides(
        workload,
        sharing_system_level=int(sharing_level),
        context="bspace interconnect bars",
    )
    report = evaluate_workload_analytical_report(
        workload,
        arch,
        target=arch_name,
        latency_coeff=1.0,
        mem_footprint_coeff=1.0,
        energy_coeff=1.0,
        require_constraints=True,
        refresh=True,
    )
    if report is None:
        raise ValueError(
            f"Constraint violation: {arch_name}/{trace_path.name} under {config_id}"
        )
    breakdown = report["breakdown"]
    return {
        "workload_name": workload.name,
        "latency_cycles": int(report["latency_cycles"]),
        "energy": int(report["energy"]),
        "mem_footprint_bytes": int(report["mem_footprint_bytes"]),
        "inter_bank_ratio": float(breakdown.get("inter-bank ratio", 0.0)),
        "output_inter_bank_ratio": float(
            breakdown.get("output inter-bank ratio", 0.0)
        ),
        "inter_bank_load_cycles": float(
            breakdown.get("compute inter-bank load (cycles)", 0.0)
        ),
        "output_inter_bank_reduce_cycles": float(
            breakdown.get("compute output inter-bank reduce (cycles)", 0.0)
        ),
    }


def _eval_no_sharing_baseline(trace_path: Path, arch_name: str) -> tuple[int, int, int]:
    """The chosen mapping with all sharing off and cache on (base arch)."""
    arch = _load_arch_for_worker(str(REPO_ROOT / ARCH_FILES[arch_name]["base"]))
    workload = Workload.from_file(trace_path)
    layout = workload.layout
    if layout is None or layout.sharing is None:
        raise ValueError(f"{trace_path} is missing Layout-Pragma.sharing")
    layout.cache = True
    layout.sharing.block = False
    layout.sharing.pu = False
    layout.sharing.system = 0
    layout.refresh()
    workload.refresh()
    report = evaluate_workload_analytical_report(
        workload,
        arch,
        target=arch_name,
        latency_coeff=1.0,
        mem_footprint_coeff=1.0,
        energy_coeff=1.0,
        require_constraints=True,
        refresh=True,
    )
    if report is None:
        raise ValueError(
            f"Constraint violation: {arch_name}/{trace_path.name} under the "
            f"no-sharing baseline"
        )
    return (
        int(report["latency_cycles"]),
        int(report["energy"]),
        int(report["mem_footprint_bytes"]),
    )


def _plot(
    records: list[dict[str, object]],
    configs: tuple[str, ...],
    metric: str,
    output: Path,
    dpi: int,
) -> None:
    # Stacked vertically: UPMEM on top, HBM-PIM below (each arch keeps its own
    # x-axis since the two archs may retain different workload sets).
    archs = ["upmem", "hbm_pim"]
    rows_by_arch = {a: [r for r in records if r["arch"] == a] for a in archs}
    fig, axes = plt.subplots(
        2, 1, figsize=(9 if metric == "all" else 8, 6.6), sharey=True,
    )
    # series: (record key, legend label, color, hatch)
    if metric == "all":
        series = [
            (
                f"norm_latency_{config_id}",
                f"Latency ({CONFIG_SHORT[config_id]})",
                CONFIG_COLORS[config_id],
                None,
            )
            for config_id in configs
        ] + [
            (
                f"norm_energy_{config_id}",
                f"Energy ({CONFIG_SHORT[config_id]})",
                CONFIG_COLORS[config_id],
                "//",
            )
            for config_id in configs
        ] + [
            (f"norm_mem_{configs[0]}", "Mem footprint", "#7f7f7f", None),
        ]
    else:
        series = [
            (
                f"norm_{metric}_{config_id}",
                CONFIG_LABELS[config_id],
                CONFIG_COLORS[config_id],
                None,
            )
            for config_id in configs
        ]
    num_series = len(series)
    bar_width = 0.8 / num_series
    for ax, arch_name in zip(axes, archs):
        rows = rows_by_arch[arch_name]
        x_positions = range(len(rows))
        for series_index, (key, label, color, hatch) in enumerate(series):
            offsets = [
                x + (series_index - (num_series - 1) / 2) * bar_width
                for x in x_positions
            ]
            values = [r[key] for r in rows]
            ax.bar(
                offsets,
                values,
                width=bar_width,
                color=color,
                hatch=hatch,
                edgecolor="white" if hatch else None,
                label=label,
            )
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(
            [str(r["workload_name"]) for r in rows], rotation=30, ha="right", fontsize=12
        )
        ax.set_title(ARCH_LABEL[arch_name], fontsize=17)
        ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ylabel = (
        "Norm. to best latency"
        if metric == "all"
        else f"{metric.capitalize()} (norm. to best latency)"
    )
    for ax in axes:
        ax.set_ylabel(ylabel, fontsize=15)
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    if metric == "all":
        # Compact the HBM-PIM (bottom) subplot and anchor it left; the freed space
        # on its right holds the legend, laid out 1 column x 5 rows (1:5). Width is
        # scaled by the workload-count ratio so HBM-PIM bars are the same width
        # (same "compactness") as UPMEM's.
        pos = axes[1].get_position()
        n_top = len(rows_by_arch[archs[0]])
        n_bot = len(rows_by_arch[archs[1]])
        compact = (n_bot / n_top) if n_top else 1.0
        new_w = pos.width * compact
        axes[1].set_position([pos.x0, pos.y0, new_w, pos.height])
        legend_x = pos.x0 + new_w + (pos.width - new_w) / 2.0
        legend_y = pos.y0 + pos.height / 2.0
        fig.legend(
            handles, labels, fontsize=12, loc="center",
            bbox_to_anchor=(legend_x, legend_y), ncol=1, framealpha=0.85,
        )
    else:
        fig.legend(
            handles, labels, fontsize=12, loc="center",
            bbox_to_anchor=(0.5, 0.40), ncol=1, framealpha=0.85,
        )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global SEARCH_ROOT
    args = parse_args()
    SEARCH_ROOT = Path(args.search_root)
    # Behavior-identical value mapping: the user-facing 'congested' value
    # selects the internal host-x16 config (16x-degraded host b2b baked
    # into the sweep center files).
    host_config = "host-x16" if args.host_cost == "congested" else "host-x1"
    configs: tuple[str, ...] = (
        (host_config, "inter-bg")
        if args.no_intra_bg
        else (host_config, "intra-bg", "inter-bg")
    )
    if args.output is None:
        suffix = "" if (args.host_cost == "congested" and not args.no_intra_bg) else f"_{args.host_cost}"
        if args.metric != "latency":
            suffix += f"_{args.metric}"
        args.output = DEFAULT_OUTPUT_DIR / f"bspace_interconnect_sharing_bars{suffix}.pdf"

    if args.replot_csv is not None:
        records = _load_records_from_csv(args.replot_csv)
        _plot(records, configs, args.metric, args.output, args.dpi)
        print(f"re-rendered {args.output} from {args.replot_csv}")
        return

    records: list[dict[str, object]] = []
    for arch_name in ("upmem", "hbm_pim"):
        for workload_token in WORKLOAD_TOKENS:
            print(f"== {arch_name}/{workload_token}", flush=True)
            best_path, (b_latency, _, _) = _select_best_b_mapping(
                arch_name, workload_token, args.workers
            )
            record: dict[str, object] = {
                "arch": arch_name,
                "workload_token": workload_token,
                "trace": str(best_path),
                "b_native_latency_cycles": b_latency,
            }
            for config_id in configs:
                result = _eval_config(best_path, arch_name, config_id)
                record["workload_name"] = result["workload_name"]
                record[f"latency_{config_id}"] = result["latency_cycles"]
                record[f"energy_{config_id}"] = result["energy"]
                record[f"mem_{config_id}"] = result["mem_footprint_bytes"]
                record[f"inter_bank_ratio_{config_id}"] = result["inter_bank_ratio"]
                record[f"output_inter_bank_ratio_{config_id}"] = result[
                    "output_inter_bank_ratio"
                ]
                record[f"inter_bank_load_cycles_{config_id}"] = result[
                    "inter_bank_load_cycles"
                ]
                record[f"output_inter_bank_reduce_cycles_{config_id}"] = result[
                    "output_inter_bank_reduce_cycles"
                ]
            baseline_latency, baseline_energy, baseline_mem = (
                _eval_no_sharing_baseline(best_path, arch_name)
            )
            record["latency_no_sharing_baseline"] = baseline_latency
            record["energy_no_sharing_baseline"] = baseline_energy
            record["mem_no_sharing_baseline"] = baseline_mem
            baselines = {
                "latency": baseline_latency,
                "energy": baseline_energy,
                "mem": baseline_mem,
            }
            for metric in ("latency", "energy", "mem"):
                for config_id in configs:
                    record[f"norm_{metric}_{config_id}"] = (
                        float(record[f"{metric}_{config_id}"])
                        / float(baselines[metric])
                    )
            mem_values = {record[f"mem_{config_id}"] for config_id in configs}
            if len(mem_values) > 1:
                print(
                    f"  [warn] mem footprint differs across configs: {mem_values}",
                    flush=True,
                )
            report_metric = "latency" if args.metric == "all" else args.metric
            print(
                f"  {report_metric}: no-sharing-baseline={baselines[report_metric]}  "
                + "  ".join(
                    f"{config_id}={record[f'{report_metric}_{config_id}']}"
                    f" ({record[f'norm_{report_metric}_{config_id}']:.3f}x)"
                    for config_id in configs
                )
                + f"  | mem={record[f'norm_mem_{configs[0]}']:.3f}x",
                flush=True,
            )
            if args.drop_no_duplication and all(
                float(record[f"inter_bank_ratio_{config_id}"]) == 0.0
                and float(record[f"output_inter_bank_ratio_{config_id}"]) == 0.0
                for config_id in configs
            ):
                print(
                    "  [drop] filter and output inter-bank ratios 0 (no duplication)",
                    flush=True,
                )
                continue
            records.append(record)

    if not records:
        raise ValueError("All groups were dropped; nothing to plot.")
    csv_path = args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    _plot(records, configs, args.metric, args.output, args.dpi)
    print(f"wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
