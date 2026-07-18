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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from pnmax.analytical_model import AnalyticalModel
from pnmax.database import Arch, Workload
from pnmax.simulators.unindp import run_unindp_sim


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

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
            "Validate UniNDP analytical model quality against simulator outputs over "
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
        default="upmem",
        help='Target architecture for both methods ("upmem" or "hbm_pim").',
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Sampling rate for simulator instruction generation.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional root directory for per-workload simulator debug artefacts.",
    )
    parser.add_argument(
        "--debug_op_type",
        nargs="+",
        type=str,
        default=[],
        help=(
            'Debug helper: emit only operations of this type ("stream", "pu", "load") '
            "during codegen."
        ),
    )
    parser.add_argument(
        "--single_pu",
        action="store_true",
        help="Generate instructions for a single bank/PU (bg_idx=0, pu_idx=0).",
    )
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
    target: str,
    sample_rate: float,
    outdir_root: Path | None,
    debug_op_type: list[str],
    single_pu: bool,
    verbose: int,
    pu_frequency_hz: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        workload_outdir: Path | None = None
        if outdir_root is not None:
            workload_outdir = outdir_root / f"{workload_index:04d}_{workload_path.stem}"
            workload_outdir.mkdir(parents=True, exist_ok=True)

        result = run_unindp_sim(
            workload_file=workload_path,
            arch=target,
            sample_rate=sample_rate,
            outdir=workload_outdir,
            dbg_op_types=debug_op_type,
            single_pu=single_pu,
            verbose=verbose,
            show_codegen_progress=False,
            show_sim_progress=False,
            pu_frequency_hz=pu_frequency_hz,
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
        "ticks": result.ticks,
        "raw_cycles": result.raw_cycles,
        "raw_ticks": result.raw_ticks,
        "operation_summary": dict(sorted(result.operation_summary.items())),
        "outdir": result.outdir,
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
    # Clamp to handle slight numerical drift outside [-1, 1] before squaring.
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
    sample_rate: float,
    outdir_path: str | None,
    debug_op_type: list[str] | None,
    single_pu: bool,
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
    # PU clock for the simulator's tick -> PU-cycle conversion: read from the
    # arch file in use (single source of truth, never duplicated in code).
    arch_pu_frequency_hz = float(Arch.from_yaml_file(arch_path).specs.pu_frequency_hz)
    output_path = Path(output_json)
    outdir_root = Path(outdir_path) if outdir_path is not None else None

    if debug_op_type is None:
        debug_op_type = []

    num_selected = len(workload_paths)
    sim_results: list[dict[str, Any] | None] = [None for _ in range(num_selected)]
    ana_results: list[dict[str, Any] | None] = [None for _ in range(num_selected)]

    # Keep method-specific progress disabled and expose only workload-level progress.
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
                    target,
                    sample_rate,
                    outdir_root,
                    list(debug_op_type),
                    single_pu,
                    simulator_verbose,
                    arch_pu_frequency_hz,
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
            "sample_rate": sample_rate,
            "outdir": str(outdir_root) if outdir_root is not None else None,
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
        sample_rate=args.sample_rate,
        outdir_path=args.outdir,
        debug_op_type=args.debug_op_type,
        single_pu=args.single_pu,
        workers=args.workers,
        output_json=args.output_json,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
