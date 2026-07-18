from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from pnmax.analytical_model import AnalyticalModel
from pnmax.database import Arch, Workload
from pnmax.simulators.attacc.attacc_simulation import attacc_sim


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return asdict(value)

    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        try:
            return scalar_item()
        except Exception:  # pragma: no cover - defensive fallback for foreign scalars
            pass

    if isinstance(value, set):
        return list(value)

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate AttAcc analytical model quality against simulator outputs over "
            "a deterministic set of workloads."
        )
    )
    parser.add_argument(
        "--workload_dir",
        type=str,
        required=True,
        help="Folder containing workload YAMLs (recursively scanned).",
    )
    parser.add_argument(
        "--num_workloads",
        type=int,
        required=True,
        help="Number of workload YAML files to evaluate.",
    )
    parser.add_argument(
        "--arch_file",
        type=str,
        required=True,
        help="Architecture YAML file used by the analytical model.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="attacc",
        help='Target architecture for analytical model (default: "attacc").',
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional root directory for per-workload AttAcc artefacts.",
    )
    parser.add_argument(
        "--ramulator_dir",
        type=str,
        default=None,
        help="Optional override for AttAcc ramulator2 directory.",
    )
    parser.add_argument(
        "--generator_script",
        type=str,
        default=None,
        help="Optional override for AttAcc trace generator script path.",
    )
    parser.add_argument(
        "--ramulator_bin",
        type=str,
        default=None,
        help="Optional override for ramulator2 executable path.",
    )
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--dhead", type=int, default=None)
    parser.add_argument("--seqlen", type=int, default=None)
    parser.add_argument("--maxlen", type=int, default=None)
    parser.add_argument("--dtype_bytes", type=int, default=None)
    parser.add_argument(
        "--no_power_constraint",
        dest="power_constraint",
        action="store_false",
        help="Disable AttAcc DRAM power constraint (NPC preset).",
    )
    parser.set_defaults(power_constraint=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads per method stream.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to write validation JSON report.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=quiet, 1=summary, 2=verbose).",
    )
    return parser.parse_args()


def _collect_workloads(workload_dir: Path, num_workloads: int) -> list[Path]:
    if num_workloads <= 0:
        raise ValueError(
            f"num_workloads must be a positive integer, got {num_workloads}."
        )
    if not workload_dir.is_dir():
        raise ValueError(f"workload_dir must be an existing directory: {workload_dir}")

    workload_paths = sorted(
        path for path in workload_dir.rglob("*.yaml") if path.is_file()
    )
    if len(workload_paths) < num_workloads:
        raise ValueError(
            f"Requested {num_workloads} workloads, but only found "
            f"{len(workload_paths)} YAML files under {workload_dir}."
        )
    return workload_paths[:num_workloads]


def _run_simulator_job(
    workload_path: Path,
    workload_index: int,
    outdir_root: Path | None,
    ramulator_dir: str | None,
    generator_script: str | None,
    ramulator_bin: str | None,
    nhead: int | None,
    dhead: int | None,
    seqlen: int | None,
    maxlen: int | None,
    dtype_bytes: int | None,
    power_constraint: bool,
    verbose: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        workload_outdir: Path | None = None
        if outdir_root is not None:
            workload_outdir = outdir_root / f"{workload_index:04d}_{workload_path.stem}"
            workload_outdir.mkdir(parents=True, exist_ok=True)

        result = attacc_sim(
            workload_file=workload_path,
            ramulator_dir=ramulator_dir,
            generator_script=generator_script,
            ramulator_bin=ramulator_bin,
            artifact_dir=workload_outdir,
            nhead=nhead,
            dhead=dhead,
            seqlen=seqlen,
            maxlen=maxlen,
            dtype_bytes=dtype_bytes,
            power_constraint=power_constraint,
            verbose=verbose >= 1,
        )
    except Exception as exc:  # pragma: no cover - delegated backend failures
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "runtime_s": time.perf_counter() - started,
        }

    return {
        "status": "ok",
        "cycles": result.cycles,
        "ticks": result.cycles,
        "raw_cycles": result.cycles,
        "raw_ticks": result.cycles,
        "latency_ns": result.latency_ns(),
        "mac": result.mac,
        "softmax": result.softmax,
        "mvgb": result.mvgb,
        "mvsb": result.mvsb,
        "wrgb": result.wrgb,
        "trace_params": asdict(result.trace),
        "outdir": str(workload_outdir) if workload_outdir is not None else None,
        "runtime_s": time.perf_counter() - started,
    }


def _run_analytical_job(
    workload_path: Path,
    arch_file: Path,
    target: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        workload = Workload.from_file(workload_path)
        arch = Arch.from_yaml_file(arch_file)
        analytical_model = AnalyticalModel(workload, arch)

        cycles = analytical_model.latency(target=target)
        energy = analytical_model.energy(target=target)
        mem_footprint = analytical_model.mem_footprint()
    except Exception as exc:  # pragma: no cover - delegated backend failures
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "runtime_s": time.perf_counter() - started,
        }

    return {
        "status": "ok",
        "cycles": cycles,
        "energy": energy,
        "mem_footprint": mem_footprint,
        "runtime_s": time.perf_counter() - started,
    }


def _compute_pearson_r2(
    sim_cycles: list[float], ana_cycles: list[float]
) -> float | None:
    if len(sim_cycles) != len(ana_cycles):
        raise ValueError(
            f"sim_cycles and ana_cycles must have equal length, got "
            f"{len(sim_cycles)} and {len(ana_cycles)}."
        )
    if len(sim_cycles) < 2:
        return None

    sim_mean = sum(sim_cycles) / len(sim_cycles)
    ana_mean = sum(ana_cycles) / len(ana_cycles)
    sim_ss = sum((value - sim_mean) ** 2 for value in sim_cycles)
    ana_ss = sum((value - ana_mean) ** 2 for value in ana_cycles)

    if math.isclose(sim_ss, 0.0, abs_tol=1e-12) or math.isclose(
        ana_ss, 0.0, abs_tol=1e-12
    ):
        return None

    covariance = sum(
        (sim_value - sim_mean) * (ana_value - ana_mean)
        for sim_value, ana_value in zip(sim_cycles, ana_cycles)
    )
    correlation = covariance / math.sqrt(sim_ss * ana_ss)
    correlation = max(-1.0, min(1.0, correlation))
    return correlation**2


def _effective_sim_workers(workers: int) -> int:
    return max(1, min(workers, os.cpu_count() or 1))


def _make_sim_executor(sim_workers: int):
    if sim_workers == 1:
        return ThreadPoolExecutor(max_workers=1)
    return ProcessPoolExecutor(
        max_workers=sim_workers,
        mp_context=mp.get_context("spawn"),
    )


def run(
    workload_dir: str,
    num_workloads: int,
    arch_file: str,
    target: str,
    outdir_path: str | None,
    ramulator_dir: str | None,
    generator_script: str | None,
    ramulator_bin: str | None,
    nhead: int | None,
    dhead: int | None,
    seqlen: int | None,
    maxlen: int | None,
    dtype_bytes: int | None,
    power_constraint: bool,
    workers: int,
    output_json: str,
    verbose: int = 1,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError(f"workers must be a positive integer, got {workers}.")

    effective_sim_workers = _effective_sim_workers(workers)
    if verbose >= 1 and workers > effective_sim_workers:
        print(
            "Capping simulator workers from "
            f"{workers} to {effective_sim_workers} (CPU count limit)."
        )

    workload_dir_path = Path(workload_dir)
    workload_paths = _collect_workloads(workload_dir_path, num_workloads)
    arch_path = Path(arch_file)
    output_path = Path(output_json)
    outdir_root = Path(outdir_path) if outdir_path is not None else None

    num_selected = len(workload_paths)
    sim_results: list[dict[str, Any] | None] = [None for _ in range(num_selected)]
    ana_results: list[dict[str, Any] | None] = [None for _ in range(num_selected)]

    simulator_verbose = max(0, verbose - 1)
    progress = (
        tqdm(total=num_selected, desc="Validating workloads", unit="workload")
        if verbose >= 0
        else None
    )
    sim_done = [False for _ in range(num_selected)]
    ana_done = [False for _ in range(num_selected)]
    pair_progress_counted = [False for _ in range(num_selected)]

    try:
        with (
            _make_sim_executor(effective_sim_workers) as sim_executor,
            ThreadPoolExecutor(max_workers=workers) as ana_executor,
        ):
            all_futures: dict[Future[dict[str, Any]], tuple[int, str]] = {}

            for idx, workload_path in enumerate(workload_paths):
                sim_future = sim_executor.submit(
                    _run_simulator_job,
                    workload_path,
                    idx,
                    outdir_root,
                    ramulator_dir,
                    generator_script,
                    ramulator_bin,
                    nhead,
                    dhead,
                    seqlen,
                    maxlen,
                    dtype_bytes,
                    power_constraint,
                    simulator_verbose,
                )
                ana_future = ana_executor.submit(
                    _run_analytical_job,
                    workload_path,
                    arch_path,
                    target,
                )
                all_futures[sim_future] = (idx, "sim")
                all_futures[ana_future] = (idx, "ana")

            for future in as_completed(all_futures):
                idx, method = all_futures[future]
                if method == "sim":
                    sim_results[idx] = future.result()
                    sim_done[idx] = True
                else:
                    ana_results[idx] = future.result()
                    ana_done[idx] = True

                if sim_done[idx] and ana_done[idx] and not pair_progress_counted[idx]:
                    pair_progress_counted[idx] = True
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    results: list[dict[str, Any]] = []
    sim_cycles: list[float] = []
    ana_cycles: list[float] = []
    total_simulator_runtime_s = 0.0
    total_analytical_runtime_s = 0.0
    paired_simulator_runtime_s = 0.0
    paired_analytical_runtime_s = 0.0

    for idx, workload_path in enumerate(workload_paths):
        simulator = sim_results[idx]
        analytical = ana_results[idx]
        if simulator is None or analytical is None:
            raise RuntimeError(
                f"Internal error: missing result for workload index {idx}."
            )

        pair_status = (
            "ok"
            if simulator.get("status") == "ok" and analytical.get("status") == "ok"
            else "error"
        )

        if pair_status == "ok":
            sim_cycles.append(float(simulator["cycles"]))
            ana_cycles.append(float(analytical["cycles"]))
            paired_simulator_runtime_s += float(simulator.get("runtime_s", 0.0) or 0.0)
            paired_analytical_runtime_s += float(
                analytical.get("runtime_s", 0.0) or 0.0
            )

        if simulator.get("status") == "ok":
            total_simulator_runtime_s += float(simulator.get("runtime_s", 0.0) or 0.0)

        if analytical.get("status") == "ok":
            total_analytical_runtime_s += float(analytical.get("runtime_s", 0.0) or 0.0)

        results.append(
            {
                "workload": str(workload_path),
                "simulator": simulator,
                "analytical": analytical,
                "pair_status": pair_status,
            }
        )

    r2 = _compute_pearson_r2(sim_cycles, ana_cycles)
    success_pairs = len(sim_cycles)
    failed_pairs = len(workload_paths) - success_pairs
    paired_runtime_ratio_analytical_to_simulator: float | None = None
    if not math.isclose(paired_simulator_runtime_s, 0.0, abs_tol=1e-12):
        paired_runtime_ratio_analytical_to_simulator = (
            paired_analytical_runtime_s / paired_simulator_runtime_s
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "workload_dir": str(workload_dir_path),
            "num_requested_workloads": num_workloads,
            "num_selected_workloads": num_selected,
            "arch_file": str(arch_path),
            "target": target,
            "outdir": str(outdir_root) if outdir_root is not None else None,
            "ramulator_dir": ramulator_dir,
            "generator_script": generator_script,
            "ramulator_bin": ramulator_bin,
            "nhead": nhead,
            "dhead": dhead,
            "seqlen": seqlen,
            "maxlen": maxlen,
            "dtype_bytes": dtype_bytes,
            "power_constraint": power_constraint,
            "workers_per_method": workers,
            "successful_pairs": success_pairs,
            "failed_pairs": failed_pairs,
            "r2": r2,
            "total_simulator_runtime_s": total_simulator_runtime_s,
            "total_analytical_runtime_s": total_analytical_runtime_s,
            "paired_simulator_runtime_s": paired_simulator_runtime_s,
            "paired_analytical_runtime_s": paired_analytical_runtime_s,
            "paired_runtime_ratio_analytical_to_simulator": (
                paired_runtime_ratio_analytical_to_simulator
            ),
        },
        "selected_workloads": [str(path) for path in workload_paths],
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{json.dumps(report, indent=2, default=_json_default)}\n", encoding="utf-8"
    )

    if verbose >= 0:
        print(
            f"Validated {num_selected} workloads "
            f"(successful pairs: {success_pairs}, failed pairs: {failed_pairs})."
        )
        if r2 is None:
            print(
                "Pearson r^2 (simulator ground truth): N/A "
                "(need >= 2 successful pairs with non-constant values)."
            )
        else:
            print(f"Pearson r^2 (simulator ground truth): {r2:.6f}")
        print(
            "Simulator runtime total (successful workloads): "
            f"{total_simulator_runtime_s:.6f} s"
        )
        print(
            "Analytical runtime total (successful workloads): "
            f"{total_analytical_runtime_s:.6f} s"
        )
        if paired_runtime_ratio_analytical_to_simulator is None:
            print(
                "Analytical/Simulator runtime ratio (successful pairs): N/A "
                "(simulator runtime sum is zero)."
            )
        else:
            print(
                "Analytical/Simulator runtime ratio (successful pairs): "
                f"{paired_runtime_ratio_analytical_to_simulator:.6f}"
            )
        print(f"Wrote validation report to {output_path}")

    return report


def main():
    args = parse_args()
    run(
        workload_dir=args.workload_dir,
        num_workloads=args.num_workloads,
        arch_file=args.arch_file,
        target=args.target,
        outdir_path=args.outdir,
        ramulator_dir=args.ramulator_dir,
        generator_script=args.generator_script,
        ramulator_bin=args.ramulator_bin,
        nhead=args.nhead,
        dhead=args.dhead,
        seqlen=args.seqlen,
        maxlen=args.maxlen,
        dtype_bytes=args.dtype_bytes,
        power_constraint=args.power_constraint,
        workers=args.workers,
        output_json=args.output_json,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
