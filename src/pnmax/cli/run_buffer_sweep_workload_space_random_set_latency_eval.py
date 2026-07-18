from __future__ import annotations

import random
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import argparse

from pnmax.cli import run_buffer_sweep_workload_space_latency_eval as _base_buffer
from pnmax.cli import run_end_to_end_workload_space as _base_e2e
from pnmax.cli import run_workload_space_pareto_eval as _base_eval
from pnmax.database import Workload

from pnmax.paths import repo_root, results_root
from pnmax.seeding import default_seed
DEFAULT_ARCH_ROOT = repo_root() / "data" / "archs" / "lowered" / "buffer_sweep"
DEFAULT_BASELINE_ROOT = repo_root() / "data" / "workloads" / "unindp_baseline"
DEFAULT_OUTPUT_ROOT = results_root() / "buffer_sweep_workload_space_random_set_latency_eval"
DEFAULT_SAMPLE_COUNT = 30
DEFAULT_SET_SIZE = 3
DEFAULT_SEED = default_seed()  # --seed > PNMAX_SEED > 42
SELECTED_TOPK = 3
SELECTION_METHOD = "total_adjacent_pct_change"
UNIFIED_SET_ID = 1
SAMPLED_POINTS_FIELDNAMES: tuple[str, ...] = (
    "seed",
    "sample_order",
    "state_hash",
    "task_key",
    "attempt_index",
    "workload_name",
    "workload_path",
)
EVALUATIONS_FIELDNAMES: tuple[str, ...] = (
    "set_id",
    "line_order",
    "line_kind",
    "line_id",
    "line_label",
    "variant_order",
    "variant_id",
    "variant_label",
    "variant_arch_file",
    "buffer_bytes",
    "area_mm2",
    "attempt_index",
    "workload_name",
    "workload_path",
    "status",
    "latency_cycles",
    "mem_footprint_bytes",
    "energy",
    "weighted_cost",
    "runtime_s",
    "state_hash",
    "error_type",
    "error_message",
    "breakdown_json",
)
PLOT_ROWS_FIELDNAMES: tuple[str, ...] = (
    "set_id",
    "line_order",
    "line_kind",
    "line_id",
    "line_label",
    "variant_order",
    "variant_id",
    "variant_label",
    "buffer_bytes",
    "area_mm2",
    "latency_cycles",
    "normalized_latency",
    "midpoint_variant_id",
    "midpoint_buffer_bytes",
    "midpoint_latency_cycles",
    "is_midpoint",
)
TOP_CHANGE_MAPPING_LINE_KIND = "top_change_mapping"
BASELINE_LINE_KIND = "baseline"
BEST_LATENCY_LINE_KIND = "best_latency"
BASELINE_LINE_ID = "baseline"
BEST_LATENCY_LINE_ID = "best_latency"


@dataclass(frozen=True)
class ResolvedInput:
    input_dir: Path
    root_dir: Path
    mapping_dir: Path
    manifest_path: Path | None
    manifest_payload: Mapping[str, Any] | None
    workload_token: str


@dataclass(frozen=True)
class CandidateMapping:
    unique_mapping: _base_buffer.UniqueMapping
    state_hash: str


@dataclass(frozen=True)
class BaselineSpec:
    path: Path
    file_token: str
    baseline_name: str
    validation_name: str


@dataclass(frozen=True)
class ScoredMapping:
    candidate_mapping: CandidateMapping
    total_adjacent_pct_change: float
    max_adjacent_pct_change: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a seeded unified mapping pool from streaming space-d workload-space "
            "outputs, evaluate only that pool across HBM-PIM buffer-sweep variants, "
            "reevaluate the sampled midpoint-best mapping across the full sweep, and "
            "export the baseline, best-latency, and top-change mapping lines."
        )
    )
    parser.add_argument(
        "--mapping-dir",
        type=str,
        required=True,
        help=(
            "One workload-space root containing spaces/d, an explicit spaces/d "
            "directory, or a Pareto report directory/workload directory whose "
            "manifest references streaming step d."
        ),
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help='Architecture family name. This flow expects "hbm_pim".',
    )
    parser.add_argument(
        "--arch-root",
        type=str,
        default=str(DEFAULT_ARCH_ROOT),
        help="Architecture-root directory containing HBM-PIM buffer-sweep YAMLs.",
    )
    parser.add_argument(
        "--baseline-root",
        type=str,
        default=str(DEFAULT_BASELINE_ROOT),
        help="Root directory containing UniNDP baseline workload YAMLs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory root for generated artifacts.",
    )
    parser.add_argument(
        "--output-dir-file",
        type=str,
        default=None,
        help="Optional path to write the resolved artifact output directory.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional analytical-model target. Defaults to --arch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of analytical evaluation worker processes.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=quiet, 1=summary).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=(
            "Unified sample-budget multiplier. The sampled pool size is "
            "--sample-count * --set-size."
        ),
    )
    parser.add_argument(
        "--set-size",
        type=int,
        default=DEFAULT_SET_SIZE,
        help=(
            "Unified sample-budget multiplier. The sampled pool size is "
            "--sample-count * --set-size."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used for deterministic unified-pool sampling.",
    )
    parser.add_argument(
        "--midpoint-buffer-bytes",
        type=int,
        default=None,
        help=(
            "Optional buffer size in bytes to use as the midpoint/selection variant. "
            "When set, the odd-variant-count requirement is skipped and the variant "
            "whose buffer_bytes equals this value is used everywhere the midpoint "
            "variant is used. When unset, the middle variant (variants[len(variants) "
            "// 2]) is used and an odd variant count is required."
        ),
    )
    return parser.parse_args()


def run(
    *,
    mapping_dir: str,
    arch_name: str,
    arch_root: str = str(DEFAULT_ARCH_ROOT),
    baseline_root: str = str(DEFAULT_BASELINE_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    output_dir_file: str | None = None,
    target: str | None = None,
    workers: int = 4,
    verbose: int = 1,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    set_size: int = DEFAULT_SET_SIZE,
    seed: int = DEFAULT_SEED,
    midpoint_buffer_bytes: int | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    canonical_arch_name = _base_e2e._canonical_arch_name(arch_name)
    if canonical_arch_name != "hbm_pim":
        raise ValueError(
            "Buffer-sweep random-set latency evaluation currently supports only "
            f"hbm_pim, got {canonical_arch_name}."
        )
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}.")
    if set_size <= 0:
        raise ValueError(f"set_size must be positive, got {set_size}.")
    sample_budget = int(sample_count) * int(set_size)
    if sample_budget < SELECTED_TOPK:
        raise ValueError(
            f"Unified sample budget {sample_budget} is too small to select the top "
            f"{SELECTED_TOPK} mappings. Increase --sample-count and/or --set-size."
        )

    resolved_input = _resolve_mapping_input(
        input_dir=_base_eval._resolve_path(mapping_dir),
        arch_name=canonical_arch_name,
    )
    mapping_source = _build_mapping_source(resolved_input)
    unique_mappings, mapping_summary = _discover_unique_mappings((mapping_source,))
    unique_mappings = tuple(
        sorted(
            unique_mappings,
            key=lambda item: (str(item.task_key), str(item.workload_path)),
        )
    )
    if not unique_mappings:
        raise ValueError("No streaming space-d mappings were discovered.")

    display_name = _resolve_workload_display_name(
        resolved_input=resolved_input,
        unique_mappings=unique_mappings,
    )
    variants = _discover_buffer_variants(
        arch_root=_base_eval._resolve_path(arch_root),
        arch_name=canonical_arch_name,
    )
    if midpoint_buffer_bytes is None and len(variants) % 2 == 0:
        raise ValueError(
            "Expected an odd number of buffer-sweep variants so the midpoint buffer "
            "size is well defined."
        )
    midpoint_variant = _resolve_midpoint_variant(
        variants=variants,
        midpoint_buffer_bytes=midpoint_buffer_bytes,
    )

    output_dir = _resolve_pipeline_output_dir(
        output_root=_base_eval._resolve_path(output_root),
        arch_name=canonical_arch_name,
        workload_token=resolved_input.workload_token,
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir_file is not None:
        _base_eval._write_output_dir_file(
            output_dir_file=output_dir_file,
            output_dir=output_dir,
        )

    sampled_points = _sample_mapping_points(
        unique_mappings=unique_mappings,
        sample_budget=sample_budget,
        seed=seed,
    )
    sampled_point_rows = _serialize_sampled_point_rows(sampled_points, seed=seed)

    effective_target = target if target is not None else canonical_arch_name
    sampled_pool_evaluation_rows = _run_evaluations(
        tasks=_build_evaluation_tasks(
            unique_mappings=tuple(
                sampled_point.unique_mapping for sampled_point in sampled_points
            ),
            variants=variants,
        ),
        target=effective_target,
        workers=workers,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    successful_sampled_mappings = _count_complete_success_mappings(
        sampled_points=sampled_points,
        evaluation_rows=sampled_pool_evaluation_rows,
        variants=variants,
    )
    selected_mappings = _select_top_change_mappings(
        sampled_points=sampled_points,
        evaluation_rows=sampled_pool_evaluation_rows,
        variants=variants,
        topk=SELECTED_TOPK,
    )
    best_latency_mapping = _select_midpoint_best_latency_mapping(
        sampled_points=sampled_points,
        evaluation_rows=sampled_pool_evaluation_rows,
        variants=variants,
        midpoint_variant=midpoint_variant,
    )
    best_latency_evaluation_rows = _run_evaluations(
        tasks=_build_evaluation_tasks(
            unique_mappings=(best_latency_mapping.unique_mapping,),
            variants=variants,
        ),
        target=effective_target,
        workers=workers,
        verbose=0 if verbose >= 1 else verbose,
    )
    try:
        best_latency_rows_by_variant = _require_complete_success_line(
            line_label="Best latency (+ cache)",
            line_rows=best_latency_evaluation_rows,
            variants=variants,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Best latency (+ cache) midpoint winner "
            f"'{best_latency_mapping.unique_mapping.workload_path.name}' "
            f"(state_hash={best_latency_mapping.state_hash}) failed during reevaluation "
            f"across all variants after being selected at {midpoint_variant.buffer_bytes}B."
        ) from exc
    best_latency_rows = _serialize_best_latency_rows(
        rows_by_variant=best_latency_rows_by_variant,
        variants=variants,
    )

    baseline_spec = _resolve_baseline_spec(
        arch_name=canonical_arch_name,
        baseline_root=_base_eval._resolve_path(baseline_root),
        workload_token=resolved_input.workload_token,
        display_name=display_name,
        manifest_payload=resolved_input.manifest_payload,
        unique_mappings=unique_mappings,
    )
    baseline_mapping = _build_unique_mapping(
        workload_path=baseline_spec.path,
        attempt_index=0,
    )
    baseline_evaluation_rows = _run_evaluations(
        tasks=_build_evaluation_tasks(
            unique_mappings=(baseline_mapping,),
            variants=variants,
        ),
        target=effective_target,
        workers=workers,
        verbose=0 if verbose >= 1 else verbose,
    )
    baseline_rows_by_variant = _require_complete_success_line(
        line_label="UniNDP baseline (+ cache)",
        line_rows=baseline_evaluation_rows,
        variants=variants,
    )

    selected_evaluation_rows = _build_selected_evaluation_rows(
        selected_mappings=selected_mappings,
        sampled_evaluation_rows=sampled_pool_evaluation_rows,
        baseline_rows_by_variant=baseline_rows_by_variant,
        best_latency_rows=best_latency_rows,
        variants=variants,
    )
    plot_rows = _build_plot_rows(
        evaluation_rows=selected_evaluation_rows,
        variants=variants,
        midpoint_variant=midpoint_variant,
    )

    sampled_points_csv = output_dir / "sampled_points.csv"
    _base_eval._write_csv(
        sampled_points_csv,
        fieldnames=SAMPLED_POINTS_FIELDNAMES,
        rows=sampled_point_rows,
    )
    sampled_points_json = output_dir / "sampled_points.json"
    _base_eval._write_json(
        sampled_points_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "arch": canonical_arch_name,
            "workload_token": resolved_input.workload_token,
            "workload_display_name": display_name,
            "seed": int(seed),
            "sample_count": int(sample_count),
            "set_size": int(set_size),
            "sample_budget": int(sample_budget),
            "successful_sampled_mappings": int(successful_sampled_mappings),
            "selection_method": SELECTION_METHOD,
            "selected_topk": int(SELECTED_TOPK),
            "selected_mappings": [
                {
                    "selection_rank": int(rank),
                    "state_hash": scored_mapping.candidate_mapping.state_hash,
                    "task_key": scored_mapping.candidate_mapping.unique_mapping.task_key,
                    "attempt_index": int(
                        scored_mapping.candidate_mapping.unique_mapping.attempt_index
                    ),
                    "workload_name": (
                        scored_mapping.candidate_mapping.unique_mapping.workload_name
                    ),
                    "workload_path": str(
                        scored_mapping.candidate_mapping.unique_mapping.workload_path
                    ),
                    "total_adjacent_pct_change": float(
                        scored_mapping.total_adjacent_pct_change
                    ),
                    "max_adjacent_pct_change": float(
                        scored_mapping.max_adjacent_pct_change
                    ),
                }
                for rank, scored_mapping in enumerate(selected_mappings, start=1)
            ],
            "points": list(sampled_point_rows),
        },
    )
    evaluations_csv = output_dir / "evaluations.csv"
    _base_eval._write_csv(
        evaluations_csv,
        fieldnames=EVALUATIONS_FIELDNAMES,
        rows=selected_evaluation_rows,
    )
    best_latency_csv = output_dir / "best_latency.csv"
    _base_eval._write_csv(
        best_latency_csv,
        fieldnames=_base_buffer.BEST_LATENCY_FIELDNAMES,
        rows=best_latency_rows,
    )
    plot_rows_csv = output_dir / "plot_rows.csv"
    _base_eval._write_csv(
        plot_rows_csv,
        fieldnames=PLOT_ROWS_FIELDNAMES,
        rows=plot_rows,
    )

    ordered_variants = [
        {
            "variant_id": variant.variant_id,
            "variant_label": variant.variant_label,
            "variant_arch_file": str(variant.arch_file),
            "buffer_bytes": int(variant.buffer_bytes),
            "area_mm2": float(variant.area_mm2),
        }
        for variant in variants
    ]
    plots_dir = (output_dir / "plots").resolve()
    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": canonical_arch_name,
        "target": effective_target,
        "output_dir": str(output_dir),
        "requested_mapping_dir": str(resolved_input.input_dir),
        "resolved_mapping_dir": str(resolved_input.mapping_dir),
        "source_manifest": (
            str(resolved_input.manifest_path) if resolved_input.manifest_path else None
        ),
        "workload_token": resolved_input.workload_token,
        "workload_display_name": display_name,
        "baseline_workload": str(baseline_spec.path),
        "best_latency_mapping_state_hash": best_latency_mapping.state_hash,
        "best_latency_mapping_workload_path": str(
            best_latency_mapping.unique_mapping.workload_path
        ),
        "selected_variants": [variant.variant_id for variant in variants],
        "ordered_variants": ordered_variants,
        "midpoint_variant_id": midpoint_variant.variant_id,
        "midpoint_buffer_bytes": int(midpoint_variant.buffer_bytes),
        "sample_count": int(sample_count),
        "set_size": int(set_size),
        "seed": int(seed),
        "sample_budget": int(sample_budget),
        "successful_sampled_mappings": int(successful_sampled_mappings),
        "selection_method": SELECTION_METHOD,
        "selected_topk": int(SELECTED_TOPK),
        "selected_mappings_total": int(mapping_summary["selected_mappings_total"]),
        "unique_mappings_total": int(mapping_summary["unique_mappings_total"]),
        "duplicate_mappings_total": int(mapping_summary["duplicate_mappings_total"]),
        "sampled_points_csv": str(sampled_points_csv.resolve()),
        "sampled_points_json": str(sampled_points_json.resolve()),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "best_latency_csv": str(best_latency_csv.resolve()),
        "plot_rows_csv": str(plot_rows_csv.resolve()),
        "plots_dir": str(plots_dir),
        "selected_mapping_state_hashes": [
            scored_mapping.candidate_mapping.state_hash
            for scored_mapping in selected_mappings
        ],
    }
    manifest_json = output_dir / "manifest.json"
    _base_eval._write_json(manifest_json, manifest_payload)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": canonical_arch_name,
        "target": effective_target,
        "workload_token": resolved_input.workload_token,
        "workload_display_name": display_name,
        "baseline_workload": str(baseline_spec.path),
        "best_latency_mapping_state_hash": best_latency_mapping.state_hash,
        "best_latency_mapping_workload_path": str(
            best_latency_mapping.unique_mapping.workload_path
        ),
        "selected_variants": [variant.variant_id for variant in variants],
        "ordered_variants": ordered_variants,
        "midpoint_variant_id": midpoint_variant.variant_id,
        "midpoint_buffer_bytes": int(midpoint_variant.buffer_bytes),
        "sample_count": int(sample_count),
        "set_size": int(set_size),
        "seed": int(seed),
        "sample_budget": int(sample_budget),
        "successful_sampled_mappings": int(successful_sampled_mappings),
        "selection_method": SELECTION_METHOD,
        "selected_topk": int(SELECTED_TOPK),
        "selected_mappings_total": int(mapping_summary["selected_mappings_total"]),
        "unique_mappings_total": int(mapping_summary["unique_mappings_total"]),
        "duplicate_mappings_total": int(mapping_summary["duplicate_mappings_total"]),
        "sampled_points_csv": str(sampled_points_csv.resolve()),
        "sampled_points_json": str(sampled_points_json.resolve()),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "best_latency_csv": str(best_latency_csv.resolve()),
        "plot_rows_csv": str(plot_rows_csv.resolve()),
        "plots_dir": str(plots_dir),
        "selected_mapping_state_hashes": [
            scored_mapping.candidate_mapping.state_hash
            for scored_mapping in selected_mappings
        ],
    }
    summary_json = output_dir / "summary.json"
    _base_eval._write_json(summary_json, summary_payload)

    report = {
        "status": "ok",
        "output_dir": str(output_dir),
        "manifest_json": str(manifest_json.resolve()),
        "summary_json": str(summary_json.resolve()),
        "sampled_points_csv": str(sampled_points_csv.resolve()),
        "sampled_points_json": str(sampled_points_json.resolve()),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "best_latency_csv": str(best_latency_csv.resolve()),
        "plot_rows_csv": str(plot_rows_csv.resolve()),
    }
    if verbose >= 1:
        print(
            f"Completed unified sampled-pool buffer-sweep latency evaluation for "
            f"{resolved_input.workload_token} ({canonical_arch_name})."
        )
        print(f"Output directory: {output_dir}")
    return report


def _resolve_mapping_input(*, input_dir: Path, arch_name: str) -> ResolvedInput:
    input_dir = input_dir.resolve()
    derived_mapping_input = _resolve_mapping_input_from_pareto_reports(
        input_dir=input_dir,
        arch_name=arch_name,
    )
    if derived_mapping_input is not None:
        resolved = _resolve_mapping_input(
            input_dir=derived_mapping_input,
            arch_name=arch_name,
        )
        return ResolvedInput(
            input_dir=input_dir,
            root_dir=resolved.root_dir,
            mapping_dir=resolved.mapping_dir,
            manifest_path=resolved.manifest_path,
            manifest_payload=resolved.manifest_payload,
            workload_token=resolved.workload_token,
        )

    if not input_dir.is_dir():
        raise ValueError(f"Mapping directory does not exist: {input_dir}")

    if input_dir.name == "d" and input_dir.parent.name == "spaces":
        root_dir = input_dir.parent.parent.resolve()
        mapping_dir = input_dir.resolve()
    else:
        candidate_mapping_dir = (input_dir / "spaces" / "d").resolve()
        if not candidate_mapping_dir.is_dir():
            raise ValueError(
                f"Unsupported mapping directory '{input_dir}'. Expected a workload "
                "root containing spaces/d, an explicit spaces/d directory, or a "
                "Pareto report directory/workload directory."
            )
        root_dir = input_dir.resolve()
        mapping_dir = candidate_mapping_dir

    if any(part == "no_streaming_true" for part in mapping_dir.parts):
        raise ValueError(
            "Buffer-sweep random-set latency evaluation requires streaming inputs; "
            f"received non-streaming path '{mapping_dir}'."
        )

    manifest_path, manifest_payload = _base_eval._load_optional_manifest(
        root_dir / "manifest.json"
    )
    normalized_manifest_arch_name = _base_eval._normalize_manifest_value(
        "arch_name",
        None if manifest_payload is None else manifest_payload.get("arch_name"),
    )
    if (
        normalized_manifest_arch_name is not None
        and normalized_manifest_arch_name != arch_name
    ):
        raise ValueError(
            f"Manifest arch_name '{normalized_manifest_arch_name}' does not match "
            f"--arch '{arch_name}'."
        )
    all_streaming_false = bool(
        False
        if manifest_payload is None
        else manifest_payload.get("all_streaming_false", False)
    )
    if all_streaming_false:
        raise ValueError(
            "Buffer-sweep random-set latency evaluation requires streaming inputs; "
            "received all_streaming_false=true."
        )

    return ResolvedInput(
        input_dir=input_dir.resolve(),
        root_dir=root_dir,
        mapping_dir=mapping_dir,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        workload_token=_base_eval._input_root_token(root_dir),
    )


def _resolve_midpoint_variant(
    *,
    variants: Sequence[_base_buffer.BufferVariant],
    midpoint_buffer_bytes: int | None,
) -> _base_buffer.BufferVariant:
    if midpoint_buffer_bytes is None:
        return variants[len(variants) // 2]
    matches = [v for v in variants if int(v.buffer_bytes) == int(midpoint_buffer_bytes)]
    if not matches:
        available = ", ".join(str(int(v.buffer_bytes)) for v in variants)
        raise ValueError(
            f"No buffer-sweep variant has buffer_bytes={int(midpoint_buffer_bytes)}. "
            f"Available buffer sizes: {available}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple buffer-sweep variants have buffer_bytes={int(midpoint_buffer_bytes)}."
        )
    return matches[0]


def _resolve_mapping_input_from_pareto_reports(
    *,
    input_dir: Path,
    arch_name: str,
) -> Path | None:
    report_mapping_dir = _resolve_mapping_dir_from_pareto_report_dir(
        report_dir=input_dir,
        arch_name=arch_name,
    )
    if report_mapping_dir is not None:
        return report_mapping_dir

    report_dirs = _discover_pareto_report_dirs(input_dir)
    if report_dirs:
        return _require_pareto_report_mapping_dir(
            report_dir=report_dirs[-1],
            arch_name=arch_name,
        )

    arch_root = _resolve_pareto_arch_root(input_dir=input_dir, arch_name=arch_name)
    if arch_root is None:
        return None

    workload_report_dirs: list[Path] = []
    for workload_dir in sorted(path for path in arch_root.iterdir() if path.is_dir()):
        report_dirs = _discover_pareto_report_dirs(workload_dir)
        if report_dirs:
            workload_report_dirs.append(report_dirs[-1])

    if not workload_report_dirs:
        return None

    preview = ", ".join(str(path) for path in workload_report_dirs[:3])
    if len(workload_report_dirs) != 1:
        raise ValueError(
            f"Ambiguous Pareto report directory '{input_dir}'. Expected a single "
            f"workload report or timestamp directory, found "
            f"{len(workload_report_dirs)} candidates under {arch_root} "
            f"(for example: {preview})."
        )

    return _require_pareto_report_mapping_dir(
        report_dir=workload_report_dirs[0],
        arch_name=arch_name,
    )


def _discover_pareto_report_dirs(root_dir: Path) -> tuple[Path, ...]:
    if not root_dir.is_dir():
        return ()
    return tuple(
        sorted(
            path.resolve()
            for path in root_dir.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        )
    )


def _resolve_pareto_arch_root(*, input_dir: Path, arch_name: str) -> Path | None:
    normalized_arch_names = (arch_name, arch_name.replace("_", "-"))
    if input_dir.name in normalized_arch_names and input_dir.is_dir():
        return input_dir.resolve()
    for arch_dir_name in normalized_arch_names:
        candidate = (input_dir / arch_dir_name).resolve()
        if candidate.is_dir():
            return candidate
    return None


def _require_pareto_report_mapping_dir(*, report_dir: Path, arch_name: str) -> Path:
    mapping_dir = _resolve_mapping_dir_from_pareto_report_dir(
        report_dir=report_dir,
        arch_name=arch_name,
    )
    if mapping_dir is None:
        raise ValueError(
            f"Pareto report directory '{report_dir}' does not expose a streaming "
            "step-d input_dir in its manifest."
        )
    return mapping_dir


def _resolve_mapping_dir_from_pareto_report_dir(
    *,
    report_dir: Path,
    arch_name: str,
) -> Path | None:
    manifest_path, manifest_payload = _base_eval._load_optional_manifest(
        report_dir / "manifest.json"
    )
    if manifest_payload is None:
        return None

    manifest_arch = _base_eval._normalize_manifest_value(
        "arch_name",
        manifest_payload.get("arch"),
    )
    if manifest_arch is not None and manifest_arch != arch_name:
        raise ValueError(
            f"Pareto manifest arch '{manifest_arch}' does not match --arch "
            f"'{arch_name}'."
        )

    steps_payload = manifest_payload.get("steps")
    if isinstance(steps_payload, MappingABC):
        step_d_payload = steps_payload.get("d")
        if isinstance(step_d_payload, MappingABC):
            if bool(step_d_payload.get("all_streaming_false", False)):
                raise ValueError(
                    "Buffer-sweep random-set latency evaluation requires streaming "
                    f"inputs; Pareto report '{manifest_path}' marks step d as "
                    "all_streaming_false=true."
                )
            step_input_dir = step_d_payload.get("input_dir")
            if isinstance(step_input_dir, str) and step_input_dir.strip():
                return _base_eval._resolve_path(step_input_dir.strip())

    discovered_step_dirs = manifest_payload.get("discovered_step_dirs")
    if isinstance(discovered_step_dirs, MappingABC):
        step_input_dir = discovered_step_dirs.get("d")
        if isinstance(step_input_dir, str) and step_input_dir.strip():
            return _base_eval._resolve_path(step_input_dir.strip())

    return None


def _build_mapping_source(resolved_input: ResolvedInput) -> _base_buffer.MappingSource:
    workload_paths = _base_eval._collect_step_workload_paths(resolved_input.mapping_dir)
    if not workload_paths:
        raise ValueError(f"No workload YAML files found under {resolved_input.mapping_dir}.")
    return _base_buffer.MappingSource(
        input_dir=resolved_input.input_dir,
        mapping_dir=resolved_input.mapping_dir,
        workload_paths=workload_paths,
        manifest_path=resolved_input.manifest_path,
        manifest_payload=resolved_input.manifest_payload,
    )


def _resolve_workload_display_name(
    *,
    resolved_input: ResolvedInput,
    unique_mappings: Sequence[_base_buffer.UniqueMapping],
) -> str:
    manifest_name = str(
        None
        if resolved_input.manifest_payload is None
        else resolved_input.manifest_payload.get("workload_name", "")
    ).strip()
    if manifest_name:
        return manifest_name

    discovered_names = sorted(
        {
            str(item.workload_name).strip()
            for item in unique_mappings
            if item.workload_name is not None and str(item.workload_name).strip()
        }
    )
    if len(discovered_names) == 1:
        return discovered_names[0]
    return resolved_input.workload_token


def _resolve_baseline_spec(
    *,
    arch_name: str,
    baseline_root: Path,
    workload_token: str,
    display_name: str,
    manifest_payload: Mapping[str, Any] | None,
    unique_mappings: Sequence[_base_buffer.UniqueMapping],
) -> BaselineSpec:
    if not baseline_root.is_dir():
        raise ValueError(f"Baseline directory does not exist: {baseline_root}")

    queries = [workload_token, display_name]
    if manifest_payload is not None:
        workload_name = str(manifest_payload.get("workload_name", "")).strip()
        if workload_name:
            queries.append(workload_name)
    queries.extend(
        str(item.workload_name).strip()
        for item in unique_mappings
        if item.workload_name is not None and str(item.workload_name).strip()
    )

    normalized_queries = {query.strip().lower() for query in queries if query.strip()}
    if not normalized_queries:
        raise ValueError("Unable to resolve a baseline workload query.")

    suffix = f"_{arch_name}"
    matches: dict[Path, BaselineSpec] = {}
    for baseline_path in sorted(baseline_root.glob(f"*_{arch_name}.yaml")):
        workload = Workload.from_file(baseline_path)
        baseline_name = workload.name
        file_stem = baseline_path.stem
        if not file_stem.endswith(suffix):
            continue
        file_token = file_stem[: -len(suffix)]
        validation_name = baseline_name.replace("-", "_")
        aliases = {
            file_token.lower(),
            baseline_name.lower(),
            validation_name.lower(),
        }
        if normalized_queries.isdisjoint(aliases):
            continue
        matches[baseline_path.resolve()] = BaselineSpec(
            path=baseline_path.resolve(),
            file_token=file_token,
            baseline_name=baseline_name,
            validation_name=validation_name,
        )

    if not matches:
        raise ValueError(
            "No UniNDP baseline workload matched the workload token or YAML name for "
            f"'{workload_token}'."
        )
    if len(matches) > 1:
        raise ValueError(
            "Ambiguous UniNDP baseline workload resolution: "
            f"{[path.name for path in sorted(matches)]}"
        )
    return next(iter(matches.values()))


def _sample_mapping_points(
    *,
    unique_mappings: Sequence[_base_buffer.UniqueMapping],
    sample_budget: int,
    seed: int,
) -> tuple[CandidateMapping, ...]:
    if sample_budget > len(unique_mappings):
        raise ValueError(
            f"Unified sample budget {sample_budget} exceeds the "
            f"{len(unique_mappings)} unique mappings available."
        )

    rng = random.Random(seed)
    sampled_indices = tuple(rng.sample(range(len(unique_mappings)), sample_budget))
    return tuple(
        CandidateMapping(
            unique_mapping=unique_mappings[index],
            state_hash=_state_hash_for_unique_mapping(unique_mappings[index]),
        )
        for index in sampled_indices
    )


def _serialize_sampled_point_rows(
    sampled_points: Sequence[CandidateMapping],
    *,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for sample_order, sampled_point in enumerate(sampled_points, start=1):
        rows.append(
            {
                "seed": int(seed),
                "sample_order": int(sample_order),
                "state_hash": sampled_point.state_hash,
                "task_key": sampled_point.unique_mapping.task_key,
                "attempt_index": int(sampled_point.unique_mapping.attempt_index),
                "workload_name": sampled_point.unique_mapping.workload_name,
                "workload_path": str(sampled_point.unique_mapping.workload_path),
            }
        )
    return tuple(rows)


def _count_complete_success_mappings(
    *,
    sampled_points: Sequence[CandidateMapping],
    evaluation_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[_base_buffer.BufferVariant],
) -> int:
    rows_by_state = _index_success_rows_by_state(
        evaluation_rows=evaluation_rows,
        variants=variants,
    )
    return sum(
        1
        for sampled_point in sampled_points
        if all(
            variant.variant_id in rows_by_state.get(sampled_point.state_hash, {})
            for variant in variants
        )
    )


def _select_top_change_mappings(
    *,
    sampled_points: Sequence[CandidateMapping],
    evaluation_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[_base_buffer.BufferVariant],
    topk: int,
) -> tuple[ScoredMapping, ...]:
    rows_by_state = _index_success_rows_by_state(
        evaluation_rows=evaluation_rows,
        variants=variants,
    )

    scored_mappings: list[ScoredMapping] = []
    for sampled_point in sampled_points:
        rows_by_variant = rows_by_state.get(sampled_point.state_hash, {})
        if any(variant.variant_id not in rows_by_variant for variant in variants):
            continue
        (
            total_adjacent_pct_change,
            max_adjacent_pct_change,
        ) = _score_latency_curve(
            rows_by_variant=rows_by_variant,
            variants=variants,
        )
        scored_mappings.append(
            ScoredMapping(
                candidate_mapping=sampled_point,
                total_adjacent_pct_change=total_adjacent_pct_change,
                max_adjacent_pct_change=max_adjacent_pct_change,
            )
        )

    scored_mappings = sorted(
        scored_mappings,
        key=lambda scored_mapping: (
            -float(scored_mapping.total_adjacent_pct_change),
            -float(scored_mapping.max_adjacent_pct_change),
            str(scored_mapping.candidate_mapping.state_hash),
        ),
    )
    if len(scored_mappings) < topk:
        raise RuntimeError(
            "Unified sampled pool yielded only "
            f"{len(scored_mappings)} mappings that completed every buffer-sweep "
            f"variant; need at least {topk}. Increase --sample-count and/or "
            "--set-size."
        )
    return tuple(scored_mappings[:topk])


def _score_latency_curve(
    *,
    rows_by_variant: Mapping[str, Mapping[str, Any]],
    variants: Sequence[_base_buffer.BufferVariant],
) -> tuple[float, float]:
    previous_latency: float | None = None
    total_adjacent_pct_change = 0.0
    max_adjacent_pct_change = 0.0
    for variant in variants:
        latency = float(rows_by_variant[variant.variant_id]["latency_cycles"])
        if latency <= 0.0:
            raise ValueError(
                f"Latency must be positive when scoring variant '{variant.variant_id}'."
            )
        if previous_latency is not None:
            adjacent_pct_change = abs(latency - previous_latency) / previous_latency
            total_adjacent_pct_change += adjacent_pct_change
            max_adjacent_pct_change = max(
                max_adjacent_pct_change,
                adjacent_pct_change,
            )
        previous_latency = latency
    return total_adjacent_pct_change, max_adjacent_pct_change


def _select_midpoint_best_latency_mapping(
    *,
    sampled_points: Sequence[CandidateMapping],
    evaluation_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[_base_buffer.BufferVariant],
    midpoint_variant: _base_buffer.BufferVariant,
) -> CandidateMapping:
    rows_by_state = _index_success_rows_by_state(
        evaluation_rows=evaluation_rows,
        variants=variants,
    )

    candidates: list[tuple[CandidateMapping, Mapping[str, Any]]] = []
    for sampled_point in sampled_points:
        midpoint_row = rows_by_state.get(sampled_point.state_hash, {}).get(
            midpoint_variant.variant_id
        )
        if midpoint_row is None:
            continue
        candidates.append((sampled_point, midpoint_row))

    if not candidates:
        raise RuntimeError(
            "No successful sampled mappings were produced for midpoint buffer variant "
            f"'{midpoint_variant.variant_id}'."
        )
    best_mapping, _ = min(
        candidates,
        key=lambda item: _base_buffer._best_latency_sort_key(item[1]),
    )
    return best_mapping


def _serialize_best_latency_rows(
    *,
    rows_by_variant: Mapping[str, Mapping[str, Any]],
    variants: Sequence[_base_buffer.BufferVariant],
) -> tuple[dict[str, Any], ...]:
    serialized_rows: list[dict[str, Any]] = []
    for variant_order, variant in enumerate(variants, start=1):
        serialized_rows.append(
            {
                "variant_order": int(variant_order),
                **dict(rows_by_variant[variant.variant_id]),
            }
        )
    return tuple(serialized_rows)


def _build_selected_evaluation_rows(
    *,
    selected_mappings: Sequence[ScoredMapping],
    sampled_evaluation_rows: Sequence[Mapping[str, Any]],
    baseline_rows_by_variant: Mapping[str, Mapping[str, Any]],
    best_latency_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[_base_buffer.BufferVariant],
) -> tuple[dict[str, Any], ...]:
    rows_by_state = _index_success_rows_by_state(
        evaluation_rows=sampled_evaluation_rows,
        variants=variants,
    )
    best_rows_by_variant = _require_complete_success_line(
        line_label="Best latency (+ cache)",
        line_rows=best_latency_rows,
        variants=variants,
    )

    serialized_rows: list[dict[str, Any]] = []
    serialized_rows.extend(
        _serialize_line_rows(
            set_id=UNIFIED_SET_ID,
            line_order=1,
            line_kind=BASELINE_LINE_KIND,
            line_id=BASELINE_LINE_ID,
            line_label="UniNDP baseline (+ cache)",
            rows_by_variant=baseline_rows_by_variant,
            variants=variants,
        )
    )
    serialized_rows.extend(
        _serialize_line_rows(
            set_id=UNIFIED_SET_ID,
            line_order=2,
            line_kind=BEST_LATENCY_LINE_KIND,
            line_id=BEST_LATENCY_LINE_ID,
            line_label="Best latency (+ cache)",
            rows_by_variant=best_rows_by_variant,
            variants=variants,
        )
    )
    for selection_rank, scored_mapping in enumerate(selected_mappings, start=1):
        serialized_rows.extend(
            _serialize_line_rows(
                set_id=UNIFIED_SET_ID,
                line_order=selection_rank + 2,
                line_kind=TOP_CHANGE_MAPPING_LINE_KIND,
                line_id=f"top_change_{selection_rank}",
                line_label=f"Top-change mapping {selection_rank}",
                rows_by_variant=rows_by_state[
                    scored_mapping.candidate_mapping.state_hash
                ],
                variants=variants,
            )
        )
    return tuple(serialized_rows)


def _serialize_line_rows(
    *,
    set_id: int,
    line_order: int,
    line_kind: str,
    line_id: str,
    line_label: str,
    rows_by_variant: Mapping[str, Mapping[str, Any]],
    variants: Sequence[_base_buffer.BufferVariant],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for variant_order, variant in enumerate(variants, start=1):
        row = dict(rows_by_variant[variant.variant_id])
        rows.append(
            {
                "set_id": int(set_id),
                "line_order": int(line_order),
                "line_kind": line_kind,
                "line_id": line_id,
                "line_label": line_label,
                "variant_order": int(variant_order),
                "variant_id": str(row["variant_id"]),
                "variant_label": str(row["variant_label"]),
                "variant_arch_file": str(row["variant_arch_file"]),
                "buffer_bytes": int(row["buffer_bytes"]),
                "area_mm2": float(row["area_mm2"]),
                "attempt_index": int(row["attempt_index"]),
                "workload_name": str(row["workload_name"]),
                "workload_path": str(row["workload_path"]),
                "status": str(row["status"]),
                "latency_cycles": int(row["latency_cycles"]),
                "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
                "energy": int(row["energy"]),
                "weighted_cost": float(row["weighted_cost"]),
                "runtime_s": float(row["runtime_s"]),
                "state_hash": str(row["state_hash"]),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
                "breakdown_json": row.get("breakdown_json"),
            }
        )
    return tuple(rows)


def _build_plot_rows(
    *,
    evaluation_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[_base_buffer.BufferVariant],
    midpoint_variant: _base_buffer.BufferVariant,
) -> tuple[dict[str, Any], ...]:
    rows_by_line: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in evaluation_rows:
        key = (int(row["set_id"]), str(row["line_id"]))
        rows_by_line.setdefault(key, []).append(row)

    plot_rows: list[dict[str, Any]] = []
    expected_variants = len(variants)
    for (set_id, line_id), line_rows in sorted(rows_by_line.items()):
        ordered_rows = sorted(
            line_rows,
            key=lambda row: (
                int(row["variant_order"]),
                int(row["buffer_bytes"]),
                str(row["variant_id"]),
            ),
        )
        if len(ordered_rows) != expected_variants:
            raise RuntimeError(
                f"Expected {expected_variants} rows for set {set_id} line {line_id}, "
                f"got {len(ordered_rows)}."
            )
        midpoint_row = next(
            (
                row
                for row in ordered_rows
                if str(row["variant_id"]) == midpoint_variant.variant_id
            ),
            None,
        )
        if midpoint_row is None:
            raise RuntimeError(
                f"Midpoint variant '{midpoint_variant.variant_id}' was not found for "
                f"set {set_id} line {line_id}."
            )
        midpoint_latency = float(midpoint_row["latency_cycles"])
        if midpoint_latency <= 0.0:
            raise ValueError(
                f"Midpoint latency must be positive for set {set_id} line {line_id}."
            )
        for row in ordered_rows:
            plot_rows.append(
                {
                    "set_id": int(set_id),
                    "line_order": int(row["line_order"]),
                    "line_kind": str(row["line_kind"]),
                    "line_id": str(row["line_id"]),
                    "line_label": str(row["line_label"]),
                    "variant_order": int(row["variant_order"]),
                    "variant_id": str(row["variant_id"]),
                    "variant_label": str(row["variant_label"]),
                    "buffer_bytes": int(row["buffer_bytes"]),
                    "area_mm2": float(row["area_mm2"]),
                    "latency_cycles": int(row["latency_cycles"]),
                    "normalized_latency": float(row["latency_cycles"]) / midpoint_latency,
                    "midpoint_variant_id": str(midpoint_row["variant_id"]),
                    "midpoint_buffer_bytes": int(midpoint_row["buffer_bytes"]),
                    "midpoint_latency_cycles": int(midpoint_row["latency_cycles"]),
                    "is_midpoint": bool(
                        str(row["variant_id"]) == str(midpoint_row["variant_id"])
                    ),
                }
            )
    return tuple(plot_rows)


def _index_success_rows_by_state(
    *,
    evaluation_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[_base_buffer.BufferVariant],
) -> dict[str, dict[str, dict[str, Any]]]:
    valid_variant_ids = {variant.variant_id for variant in variants}
    rows_by_state: dict[str, dict[str, dict[str, Any]]] = {}
    for row in evaluation_rows:
        if str(row.get("status", "")).strip() != "ok":
            continue
        variant_id = str(row["variant_id"])
        if variant_id not in valid_variant_ids:
            continue
        state_hash = str(row.get("state_hash", "")).strip()
        if not state_hash:
            continue
        state_rows = rows_by_state.setdefault(state_hash, {})
        existing = state_rows.get(variant_id)
        candidate = dict(row)
        if existing is None or _base_buffer._best_latency_sort_key(candidate) < _base_buffer._best_latency_sort_key(
            existing
        ):
            state_rows[variant_id] = candidate
    return rows_by_state


def _require_complete_success_line(
    *,
    line_label: str,
    line_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[_base_buffer.BufferVariant],
) -> dict[str, dict[str, Any]]:
    rows_by_variant: dict[str, dict[str, Any]] = {}
    for row in line_rows:
        if str(row.get("status", "")).strip() != "ok":
            continue
        rows_by_variant[str(row["variant_id"])] = dict(row)
    missing_variant_ids = [
        variant.variant_id
        for variant in variants
        if variant.variant_id not in rows_by_variant
    ]
    if missing_variant_ids:
        raise RuntimeError(
            f"{line_label} is missing successful evaluations for variants: "
            f"{', '.join(missing_variant_ids)}."
        )
    return rows_by_variant


def _state_hash_for_unique_mapping(unique_mapping: _base_buffer.UniqueMapping) -> str:
    task_key = str(unique_mapping.task_key)
    if task_key.startswith("state:"):
        return task_key.split(":", 1)[1]
    workload = Workload.from_file(unique_mapping.workload_path)
    return _base_eval._state_hash(workload.to_dict())


def _resolve_pipeline_output_dir(
    *,
    output_root: Path,
    arch_name: str,
    workload_token: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        output_root
        / _base_eval._sanitize_token(arch_name)
        / _base_eval._sanitize_token(workload_token)
        / timestamp
    )


def _discover_unique_mappings(
    mapping_sources: Sequence[_base_buffer.MappingSource],
) -> tuple[tuple[_base_buffer.UniqueMapping, ...], dict[str, int]]:
    return _base_buffer._discover_unique_mappings(mapping_sources)


def _discover_buffer_variants(
    *,
    arch_root: Path,
    arch_name: str,
) -> tuple[_base_buffer.BufferVariant, ...]:
    return _base_buffer._discover_buffer_variants(
        arch_root=arch_root,
        arch_name=arch_name,
    )


def _build_evaluation_tasks(
    *,
    unique_mappings: Sequence[_base_buffer.UniqueMapping],
    variants: Sequence[_base_buffer.BufferVariant],
) -> tuple[_base_buffer.EvaluationTask, ...]:
    return _base_buffer._build_evaluation_tasks(
        unique_mappings=unique_mappings,
        variants=variants,
    )


def _build_unique_mapping(
    *,
    workload_path: Path,
    attempt_index: int,
) -> _base_buffer.UniqueMapping:
    return _base_buffer._build_unique_mapping(
        workload_path=workload_path,
        attempt_index=attempt_index,
    )


def _run_evaluations(
    *,
    tasks: Sequence[_base_buffer.EvaluationTask],
    target: str | None,
    workers: int,
    verbose: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    return _base_buffer._run_evaluations(
        tasks=tasks,
        target=target,
        workers=workers,
        verbose=verbose,
        progress_callback=progress_callback,
    )


def _select_best_latency_rows(
    *,
    variants: Sequence[_base_buffer.BufferVariant],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return _base_buffer._select_best_latency_rows(
        variants=variants,
        evaluation_rows=evaluation_rows,
    )


def main() -> int:
    args = parse_args()
    run(
        mapping_dir=args.mapping_dir,
        arch_name=args.arch,
        arch_root=args.arch_root,
        baseline_root=args.baseline_root,
        output_root=args.output_root,
        output_dir_file=args.output_dir_file,
        target=args.target,
        workers=args.workers,
        verbose=args.verbose,
        sample_count=args.sample_count,
        set_size=args.set_size,
        seed=args.seed,
        midpoint_buffer_bytes=args.midpoint_buffer_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
