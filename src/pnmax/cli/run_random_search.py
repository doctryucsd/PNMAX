from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pnmax.database import Arch, Workload
from pnmax.seeding import default_seed
from pnmax.dse.random_search import (
    RandomSearchConfig,
    run_random_search,
    run_random_search_multi,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample unique valid workload tilings."
    )
    parser.add_argument(
        "--workload",
        type=str,
        nargs="+",
        required=True,
        help="One or more input workload YAML paths.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help="Input architecture YAML path.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for traces.",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=100,
        help="Number of unique valid traces to persist.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10_000,
        help="Maximum number of attempts before stopping.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of processes (1 means single-process mode).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
        help="Base seed for reproducibility (default: --seed > PNMAX_SEED > 42).",
    )
    parser.add_argument(
        "--template-id",
        type=str,
        default=None,
        help="Optional template id used to derive deterministic seeds.",
    )
    parser.add_argument(
        "--machine-id",
        type=str,
        default=None,
        help="Optional machine id used to derive deterministic seeds.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="*",
        default=None,
        help="Optional dimension order override (remaining dims are appended).",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit level ids (must be contiguous from 0).",
    )
    parser.add_argument(
        "--batch-attempts",
        type=int,
        default=128,
        help="Attempt batch size per worker in multiprocessing mode.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display.",
    )
    parser.add_argument(
        "--disable-early-checks",
        action="store_true",
        help="Disable cheap early architecture checks before full verification.",
    )
    parser.add_argument(
        "--disable-cache-warmup",
        action="store_true",
        help="Disable divisor/factorization cache warmup.",
    )
    parser.add_argument(
        "--search-pu-sharing",
        action="store_true",
        help="Search both disabled/enabled states for Layout-Pragma.sharing.PU.",
    )
    parser.add_argument(
        "--search-system-sharing",
        action="store_true",
        help="Search both disabled/enabled states for Layout-Pragma.sharing.system.",
    )
    parser.add_argument(
        "--search-cache",
        action="store_true",
        help="Search both disabled/enabled states for Layout-Pragma.cache.",
    )
    return parser.parse_args()


def _normalize_workload_args(raw_workload_args: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for token in raw_workload_args:
        pieces = str(token).split()
        normalized.extend(piece for piece in pieces if piece)
    if not normalized:
        raise ValueError("At least one workload path must be provided to --workload.")
    return normalized


def run(
    workload_files: Sequence[str],
    arch_file: str,
    out_dir: str,
    num_traces: int,
    max_attempts: int,
    workers: int = 1,
    seed: int | None = None,
    template_id: str | None = None,
    machine_id: str | None = None,
    dimensions: Sequence[str] | None = None,
    levels: Sequence[int] | None = None,
    batch_attempts: int = 128,
    show_progress: bool = True,
    use_early_checks: bool = True,
    warmup_cache: bool = True,
    search_pu_sharing: bool = False,
    search_system_sharing: bool = False,
    search_cache: bool = False,
) -> None:
    raw_workloads: Sequence[str]
    if isinstance(workload_files, str):
        raw_workloads = (workload_files,)
    else:
        raw_workloads = workload_files

    workload_paths = _normalize_workload_args(raw_workloads)
    workloads = [
        Workload.from_file(Path(workload_path)) for workload_path in workload_paths
    ]
    arch = Arch.from_yaml_file(Path(arch_file))

    config = RandomSearchConfig(
        out_dir=out_dir,
        num_traces=num_traces,
        max_attempts=max_attempts,
        num_workers=workers,
        dimensions=dimensions,
        level_ids=levels,
        seed=seed,
        template_id=template_id,
        machine_id=machine_id,
        batch_attempts=batch_attempts,
        show_progress=show_progress,
        use_early_checks=use_early_checks,
        warmup_cache=warmup_cache,
        search_pu_sharing=search_pu_sharing,
        search_system_sharing=search_system_sharing,
        search_cache=search_cache,
    )
    if len(workloads) == 1:
        result = run_random_search(workloads[0], arch, config)
    else:
        result = run_random_search_multi(workloads, arch, config)

    print(f"output_dir: {result.output_dir}")
    print(f"seed: {result.seed}")
    print(f"attempts: {result.attempts}")
    print(f"accepted: {result.accepted}")
    print(f"duplicates: {result.duplicates}")
    print(f"rejected: {result.rejected}")


def main():
    args = parse_args()

    workload_files: Sequence[str] = args.workload
    arch_file: str = args.arch
    out_dir: str = args.outdir
    num_traces: int = args.num_traces
    max_attempts: int = args.max_attempts
    workers: int = args.workers
    seed: int | None = args.seed
    template_id: str | None = args.template_id
    machine_id: str | None = args.machine_id
    dimensions: Sequence[str] | None = args.dimensions
    levels: Sequence[int] | None = args.levels
    batch_attempts: int = args.batch_attempts
    show_progress: bool = not args.no_progress
    use_early_checks: bool = not args.disable_early_checks
    warmup_cache: bool = not args.disable_cache_warmup
    search_pu_sharing: bool = args.search_pu_sharing
    search_system_sharing: bool = args.search_system_sharing
    search_cache: bool = args.search_cache

    run(
        workload_files=workload_files,
        arch_file=arch_file,
        out_dir=out_dir,
        num_traces=num_traces,
        max_attempts=max_attempts,
        workers=workers,
        seed=seed,
        template_id=template_id,
        machine_id=machine_id,
        dimensions=dimensions,
        levels=levels,
        batch_attempts=batch_attempts,
        show_progress=show_progress,
        use_early_checks=use_early_checks,
        warmup_cache=warmup_cache,
        search_pu_sharing=search_pu_sharing,
        search_system_sharing=search_system_sharing,
        search_cache=search_cache,
    )


if __name__ == "__main__":
    main()
