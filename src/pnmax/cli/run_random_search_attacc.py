from __future__ import annotations

import argparse
from pathlib import Path

from pnmax.dse.random_search_attacc import (
    AttaccRandomSearchConfig,
    AttaccRandomSearchResult,
    run_attacc_random_search,
)
from pnmax.paths import repo_root, results_root
from pnmax.seeding import default_seed

FULL_MODE_DEFAULT = 12_000
SANITY_MODE_DEFAULT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate randomized AttAcc GEMV workloads without duplicates."
    )
    parser.add_argument(
        "--workload",
        type=str,
        default=str(repo_root() / "data" / "workloads" / "samples" / "attacc_gemv.yaml"),
        help="Base AttAcc workload YAML path.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(results_root() / "validation_traces" / "attacc"),
        help="Output directory for generated workloads.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "sanity"),
        default="full",
        help='Generation mode, either "full" or "sanity".',
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=None,
        help="Requested number of workloads (highest priority).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Legacy alias for requested number of workloads.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of processes for YAML materialization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
        help="RNG seed for deterministic pair ordering "
        "(default: --seed > PNMAX_SEED > 42).",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Simulate generation without writing files.",
    )
    return parser.parse_args()


def _determine_num_traces(
    mode: str,
    num_traces: int | None,
    num_samples: int | None,
) -> tuple[int, str]:
    if num_traces is not None:
        return int(num_traces), "--num-traces"
    if num_samples is not None:
        return int(num_samples), "--num_samples"
    if mode == "full":
        return FULL_MODE_DEFAULT, "mode=full default"
    if mode == "sanity":
        return SANITY_MODE_DEFAULT, "mode=sanity default"
    raise ValueError(f"Unsupported mode: {mode}")


def run(
    workload_file: str,
    out_dir: str,
    mode: str,
    num_traces: int | None,
    num_samples: int | None,
    workers: int,
    seed: int | None,
    dry_run: bool,
) -> AttaccRandomSearchResult:
    requested, source = _determine_num_traces(mode, num_traces, num_samples)
    if requested < 0:
        raise ValueError(f"Requested workload count must be >= 0, got {requested}.")

    config = AttaccRandomSearchConfig(
        workload_file=Path(workload_file),
        out_dir=Path(out_dir),
        num_traces=requested,
        workers=workers,
        seed=seed,
        dry_run=dry_run,
    )
    result = run_attacc_random_search(config)

    print(f"Mode: {mode}")
    print(f"Requested workloads: {requested} ({source})")
    print(f"Unique feasible limit: {result.unique_limit}")
    if result.capped:
        print(
            "Requested count exceeds unique feasible limit; "
            f"capping to {result.effective}."
        )
    print(f"Base workload: {result.workload_file}")
    print(f"Output dir: {result.output_dir}")
    print(f"Workers: {workers}")
    print(f"Seed: {result.seed}")
    print(f"Dry run: {dry_run}")
    if result.removed_existing > 0:
        print(f"Removed existing traces: {result.removed_existing}")

    if result.dry_run:
        generated_line = (
            f"Dry run complete: would generate {result.effective} / {result.requested} "
            f"AttAcc workloads in {result.output_dir}"
        )
    else:
        generated_line = (
            f"Generated {result.generated} / {result.requested} "
            f"AttAcc workloads in {result.output_dir}"
        )
    if result.capped:
        generated_line += " (capped)"
    print(generated_line)

    return result


def main() -> None:
    args = parse_args()
    run(
        workload_file=args.workload,
        out_dir=args.outdir,
        mode=args.mode,
        num_traces=args.num_traces,
        num_samples=args.num_samples,
        workers=args.workers,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
