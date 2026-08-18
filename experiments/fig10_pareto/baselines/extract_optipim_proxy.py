#!/usr/bin/env python3
"""Extract the OptiPIM *searched-proxy* baseline from exhaustive random-search traces.

The OptiPIM comparison point in the Fig. 10 Pareto grid is a **searched proxy**, not
a run of the OptiPIM ILP tool: it is the best-of-N PNMAX random mappings for each
HBM-PIM workload (N = 4096 at full scale, i.e. two independent 2048-trace random
searches). See README.md for the rationale.

Artifact adaptation (PNMAX AE): imports use the renamed ``pnmax`` package; the arch
file and the trace/baseline result roots resolve repo-relatively and are
env-overridable. Wired into ``run.sh`` and exercised by the full Fig-9 campaign —
see README.md.
"""

import argparse
import csv
import glob
import json
import math
import multiprocessing
import os
from pathlib import Path

from pnmax.database import Arch, Workload
from pnmax.dse.workload_eval import evaluate_workload_analytical_report


# experiments/fig10_pareto/baselines/ -> repo root is 3 levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = Path(os.environ.get("PNMAX_RESULTS_ROOT", str(REPO_ROOT / "results")))

WORKLOADS = [
    "alexnet-4",
    "bert-3",
    "bert-72",
    "llama128-6",
    "llama256-3",
    "llama512-3",
    "resnet50-2",
    "resnet50-53",
    "vgg16-0",
]
STREAMING = ["false", "true"]
DEFAULT_ARCH_FILE = str(REPO_ROOT / "data" / "archs" / "lowered" / "baseline" / "hbm_pim.yaml")
TARGET = "hbm_pim"
# Best-of-N proxy sources (produced by the Fig-9 HBM-PIM random-search runs;
# run.sh wires the exact result subpaths). Anchored to the results root, env-overridable.
BASELINE_GLOB = str(RESULTS_ROOT / "workload_space_pareto_eval" / "hbm_pim" / "*")

_ARCH = None


def init_worker(arch_file):
    global _ARCH
    _ARCH = Arch.from_yaml_file(arch_file)


def eval_trace(path):
    wl = Workload.from_file(path)
    result = evaluate_workload_analytical_report(
        wl,
        _ARCH,
        target=TARGET,
        latency_coeff=1.0,
        mem_footprint_coeff=1.0,
        energy_coeff=1.0,
    )
    return path, result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the OptiPIM searched-proxy results from HBM-PIM traces."
    )
    parser.add_argument("--arch-file", default=DEFAULT_ARCH_FILE)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--output",
        default=str(RESULTS_ROOT / "fig10_pareto" / "optipim" / "optipim_proxy.csv"),
    )
    return parser.parse_args()


def build_baseline_metrics():
    baseline_metrics = {}
    for directory in sorted(glob.glob(BASELINE_GLOB)):
        root = Path(directory)
        if not root.is_dir():
            continue

        candidates = sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and (child / "baseline.json").is_file()
        )
        if not candidates:
            continue

        baseline_path = candidates[-1] / "baseline.json"
        with baseline_path.open() as f:
            payload = json.load(f)

        token = payload["baseline_yaml_name"].split("_", 1)[1]
        baseline_metrics[token] = payload["metrics"]
    return baseline_metrics


def trace_paths_for(workload, streaming):
    first_dir = (
        RESULTS_ROOT
        / f"workload_space_random_search/no_streaming_{streaming}/hbm-pim/{workload}/spaces/a"
    )
    second_dir = (
        RESULTS_ROOT
        / f"optipim_proxy/random_search_seed2/no_streaming_{streaming}/hbm_pim/{workload}/spaces/a"
    )
    first = sorted(glob.glob(os.fspath(first_dir / "trace_*.yaml")))
    second = sorted(glob.glob(os.fspath(second_dir / "trace_*.yaml")))
    return first, second


def norm_value(result, baseline, key):
    if not baseline:
        return ""
    value = result[key] / baseline[key]
    if not math.isfinite(value):
        return ""
    return value


def norm_weighted_value(result, baseline):
    if not baseline:
        return ""
    values = [
        norm_value(result, baseline, "latency_cycles"),
        norm_value(result, baseline, "mem_footprint_bytes"),
        norm_value(result, baseline, "energy"),
    ]
    if any(value == "" for value in values):
        return ""
    return sum(values)


def append_result_row(rows, workload, streaming, counts, criterion, chosen, baseline):
    path, result = chosen
    rows.append(
        {
            "workload": workload,
            "streaming": streaming,
            "n_first": counts["n_first"],
            "n_second": counts["n_second"],
            "n_total": counts["n_total"],
            "n_legal": counts["n_legal"],
            "n_rejected": counts["n_rejected"],
            "criterion": criterion,
            "latency_cycles": result["latency_cycles"],
            "mem_footprint_bytes": result["mem_footprint_bytes"],
            "energy": result["energy"],
            "weighted_cost": result["weighted_cost"],
            "norm_latency": norm_value(result, baseline, "latency_cycles"),
            "norm_mem": norm_value(result, baseline, "mem_footprint_bytes"),
            "norm_energy": norm_value(result, baseline, "energy"),
            "trace_path": path,
        }
    )


def print_summary(workload, streaming, counts, best_latency, best_weighted, baseline):
    if best_latency and baseline:
        best_latency_norm = norm_value(best_latency[1], baseline, "latency_cycles")
    else:
        best_latency_norm = "n/a"

    if best_weighted and baseline:
        best_weighted_norm = norm_weighted_value(best_weighted[1], baseline)
    else:
        best_weighted_norm = "n/a"

    print(
        f"{workload} streaming={streaming} n_total={counts['n_total']} "
        f"n_legal={counts['n_legal']} n_rejected={counts['n_rejected']} "
        f"best_latency_norm={best_latency_norm} "
        f"best_weighted_norm={best_weighted_norm}"
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    baseline_metrics = build_baseline_metrics()
    rows = []
    warnings = []

    for wl in WORKLOADS:
        for ns in STREAMING:
            first, second = trace_paths_for(wl, ns)
            paths = first + second
            n_first = len(first)
            n_second = len(second)
            n_total = n_first + n_second

            if paths:
                with multiprocessing.Pool(
                    processes=args.workers,
                    initializer=init_worker,
                    initargs=(args.arch_file,),
                ) as pool:
                    evaluated = pool.map(eval_trace, paths)
            else:
                evaluated = []

            legal = [(path, result) for path, result in evaluated if result is not None]
            n_legal = len(legal)
            n_rejected = n_total - n_legal
            counts = {
                "n_first": n_first,
                "n_second": n_second,
                "n_total": n_total,
                "n_legal": n_legal,
                "n_rejected": n_rejected,
            }

            best_weighted = (
                min(legal, key=lambda item: item[1]["weighted_cost"]) if legal else None
            )
            best_latency = (
                min(legal, key=lambda item: item[1]["latency_cycles"]) if legal else None
            )
            baseline = baseline_metrics.get(wl)

            if best_weighted is not None:
                append_result_row(
                    rows,
                    wl,
                    ns,
                    counts,
                    "weighted_cost",
                    best_weighted,
                    baseline,
                )
            if best_latency is not None:
                append_result_row(
                    rows,
                    wl,
                    ns,
                    counts,
                    "latency",
                    best_latency,
                    baseline,
                )

            print_summary(wl, ns, counts, best_latency, best_weighted, baseline)
            if n_total != 4096 or n_rejected > 0:
                warnings.append((wl, ns, n_total, n_rejected))

    fieldnames = [
        "workload",
        "streaming",
        "n_first",
        "n_second",
        "n_total",
        "n_legal",
        "n_rejected",
        "criterion",
        "latency_cycles",
        "mem_footprint_bytes",
        "energy",
        "weighted_cost",
        "norm_latency",
        "norm_mem",
        "norm_energy",
        "trace_path",
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for wl, ns, n_total, n_rejected in warnings:
        print(
            f"WARNING {wl} streaming={ns} n_total={n_total} "
            f"n_rejected={n_rejected}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
