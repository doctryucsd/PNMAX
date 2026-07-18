from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tqdm import tqdm

from pnmax.database import Arch, Workload
from pnmax.dse.pareto_export import (
    PAIRWISE_FRONTIER_METRICS,
    augment_frontier_row,
    pairwise_frontiers,
)
from pnmax.dse.workload_eval import evaluate_workload_analytical_report
from pnmax.dse.workload_search_spaces import SPACE_LABELS, SPACE_ORDER
from pnmax.paths import repo_root, results_root

DEFAULT_ARCH_FILES: dict[str, Path] = {
    "upmem": repo_root() / "data" / "archs" / "lowered" / "baseline/upmem.yaml",
    "hbm_pim": repo_root() / "data" / "archs" / "lowered" / "baseline/hbm_pim.yaml",
}
LEGACY_ARCH_FILE_BASENAME_ALIASES: dict[str, str] = {
    "upmem_arch.yaml": "upmem.yaml",
    "hbmpim_arch.yaml": "hbm_pim.yaml",
}
DEFAULT_OUTPUT_ROOT = results_root() / "workload_space_pareto_eval"
EVALUATIONS_FIELDNAMES: tuple[str, ...] = (
    "step_id",
    "base_step_id",
    "mode_id",
    "mode_label",
    "all_streaming_false",
    "step_label",
    "input_dir",
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
PARETO_FIELDNAMES: tuple[str, ...] = (
    "frontier_name",
    "x_metric",
    "y_metric",
    "step_id",
    "base_step_id",
    "mode_id",
    "mode_label",
    "all_streaming_false",
    "step_label",
    "order",
    "source",
    "attempt_index",
    "replica_id",
    "state_hash",
    "latency_cycles",
    "mem_footprint_bytes",
    "energy",
    "weighted_cost",
    "total_runtime_s",
    "workload_yaml",
)
_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TRACE_NAME_RE = re.compile(r"^trace_(\d+)\.yaml$")
_STEP_IDS = set(SPACE_ORDER)
_MANIFEST_COMPARE_FIELDS: tuple[str, ...] = (
    "workload_path",
    "workload_name",
    "arch_name",
    "arch_file",
)
CURRENT_MODE_ID = "current"
NON_STREAMING_MODE_ID = "non_streaming"
MODE_LABELS: dict[str, str] = {
    CURRENT_MODE_ID: "current",
    NON_STREAMING_MODE_ID: "non-streaming",
}
MODE_ORDER: tuple[str, ...] = (
    CURRENT_MODE_ID,
    NON_STREAMING_MODE_ID,
)


@dataclass(frozen=True)
class StepInput:
    step_id: str
    base_step_id: str
    mode_id: str
    mode_label: str
    all_streaming_false: bool
    step_label: str
    dir_path: Path
    workload_paths: tuple[Path, ...]
    manifest_path: Path | None
    manifest_payload: Mapping[str, Any] | None


@dataclass(frozen=True)
class EvaluationTask:
    task_key: str
    workload_path: Path
    first_step_id: str


@dataclass(frozen=True)
class EvaluationOccurrence:
    step_id: str
    base_step_id: str
    mode_id: str
    mode_label: str
    all_streaming_false: bool
    step_label: str
    input_dir: Path
    workload_path: Path
    attempt_index: int


@dataclass(frozen=True)
class EvaluationPlan:
    tasks: tuple[EvaluationTask, ...]
    occurrences_by_task_key: Mapping[str, tuple[EvaluationOccurrence, ...]]
    selected_workloads_by_step: Mapping[str, int]
    evaluated_workloads_by_step: Mapping[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate workload-space YAML traces with the analytical model and "
            "export Pareto-front artifacts consumable by plot/pareto.py."
        )
    )
    parser.add_argument(
        "--workload-dirs",
        nargs="+",
        required=True,
        help=(
            "One or more workload-space roots containing spaces/a,b,c,d, or "
            "explicit per-step directories whose basename is a, b, c, or d."
        ),
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=sorted(DEFAULT_ARCH_FILES),
        help='Architecture name ("upmem" or "hbm_pim").',
    )
    parser.add_argument(
        "--allow-manifest-arch-mismatch",
        action="store_true",
        help=(
            "Evaluate with --arch-file even when the search manifests record a "
            "different arch_file (e.g. re-scoring searched mappings under a "
            "cost-perturbed arch variant). Warns instead of failing."
        ),
    )
    parser.add_argument(
        "--arch-file",
        type=str,
        default=None,
        help="Optional architecture YAML path override.",
    )
    parser.add_argument(
        "--baseline-workload",
        type=str,
        default=None,
        help="Optional baseline workload YAML to evaluate and include in the plot.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory root for generated Pareto artifacts.",
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
    return parser.parse_args()


def run(
    *,
    workload_dirs: Sequence[str],
    arch_name: str,
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    arch_file: str | None = None,
    baseline_workload: str | None = None,
    output_dir_file: str | None = None,
    target: str | None = None,
    workers: int = 4,
    verbose: int = 1,
    allow_arch_mismatch: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    resolved_input_dirs = tuple(_resolve_path(path) for path in workload_dirs)
    arch_path = _resolve_arch_path(arch_name, arch_file)
    arch = Arch.from_yaml_file(arch_path)
    step_inputs = _discover_step_inputs(resolved_input_dirs)
    manifest_metadata = _validate_manifest_consistency(
        step_inputs=step_inputs,
        arch_name=arch_name,
        arch_path=arch_path,
        allow_arch_mismatch=allow_arch_mismatch,
    )

    baseline_path = (
        _resolve_path(baseline_workload) if baseline_workload is not None else None
    )
    baseline_payload: dict[str, Any] | None = None
    baseline_workload_name: str | None = None
    effective_target = target if target is not None else arch_name
    if baseline_path is not None:
        baseline_payload, baseline_workload_name = _evaluate_baseline(
            baseline_path=baseline_path,
            arch=arch,
            arch_name=arch_name,
            arch_path=arch_path,
            target=effective_target,
        )

    display_name = _resolve_display_name(
        input_dirs=resolved_input_dirs,
        manifest_metadata=manifest_metadata,
        baseline_workload_name=baseline_workload_name,
    )
    output_dir = _resolve_pipeline_output_dir(
        output_root=_resolve_path(output_root),
        arch_name=arch_name,
        workload_token=display_name,
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir_file is not None:
        _write_output_dir_file(output_dir_file=output_dir_file, output_dir=output_dir)

    evaluation_plan = _build_evaluation_plan(step_inputs)
    unique_evaluation_rows = _run_evaluations(
        tasks=evaluation_plan.tasks,
        arch_path=arch_path,
        target=effective_target,
        workers=workers,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    evaluation_rows = _expand_evaluation_rows(
        evaluation_plan=evaluation_plan,
        unique_evaluation_rows=unique_evaluation_rows,
    )
    successful_rows = [row for row in evaluation_rows if row["status"] == "ok"]
    if not successful_rows:
        raise RuntimeError(
            "No legal workloads were produced by the analytical evaluation."
        )

    evaluations_csv = output_dir / "evaluations.csv"
    _write_csv(
        evaluations_csv,
        fieldnames=EVALUATIONS_FIELDNAMES,
        rows=evaluation_rows,
    )

    step_summaries, pareto_rows = _build_step_artifacts(
        step_inputs=step_inputs,
        evaluation_plan=evaluation_plan,
        evaluation_rows=evaluation_rows,
        unique_evaluation_rows=unique_evaluation_rows,
    )
    pareto_csv = output_dir / "pareto_frontiers.csv"
    _write_csv(
        pareto_csv,
        fieldnames=PARETO_FIELDNAMES,
        rows=pareto_rows,
    )

    baseline_json_path: Path | None = None
    if baseline_payload is not None:
        baseline_json_path = output_dir / "baseline.json"
        _write_json(baseline_json_path, baseline_payload)

    source_manifests = tuple(
        dict.fromkeys(
            step_input.manifest_path.resolve()
            for step_input in step_inputs
            if step_input.manifest_path is not None
        )
    )
    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": arch_name,
        "arch_file": str(arch_path),
        "target": effective_target,
        "output_dir": str(output_dir),
        "requested_workload_dirs": [str(path) for path in resolved_input_dirs],
        "discovered_step_dirs": {
            step_input.step_id: str(step_input.dir_path) for step_input in step_inputs
        },
        "selected_steps": [step_input.step_id for step_input in step_inputs],
        "selected_workloads_total": int(
            sum(len(step.workload_paths) for step in step_inputs)
        ),
        "unique_evaluation_tasks": int(len(evaluation_plan.tasks)),
        "baseline_workload": (
            str(baseline_path) if baseline_path is not None else None
        ),
        "source_manifests": [str(path) for path in source_manifests],
        "manifest_metadata": dict(manifest_metadata),
        "steps": step_summaries,
    }
    manifest_json = output_dir / "manifest.json"
    _write_json(manifest_json, manifest_payload)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": arch_name,
        "arch_file": str(arch_path),
        "target": effective_target,
        "workload_display_name": display_name,
        "baseline_json": (
            str(baseline_json_path.resolve())
            if baseline_json_path is not None
            else None
        ),
        "selected_steps": [step_input.step_id for step_input in step_inputs],
        "selected_workloads_total": int(
            sum(len(step.workload_paths) for step in step_inputs)
        ),
        "unique_evaluation_tasks": int(len(evaluation_plan.tasks)),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "pairwise_frontiers_csv": str(pareto_csv.resolve()),
        "pareto_front_csv": str(pareto_csv.resolve()),
        "steps": step_summaries,
    }
    summary_json = output_dir / "summary.json"
    _write_json(summary_json, summary_payload)

    report = {
        "status": "ok",
        "output_dir": str(output_dir),
        "manifest_json": str(manifest_json.resolve()),
        "summary_json": str(summary_json.resolve()),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "pairwise_frontiers_csv": str(pareto_csv.resolve()),
        "baseline_json": (
            str(baseline_json_path.resolve())
            if baseline_json_path is not None
            else None
        ),
        "selected_steps": [step_input.step_id for step_input in step_inputs],
        "unique_evaluation_tasks": int(len(evaluation_plan.tasks)),
    }

    if verbose >= 1:
        print(
            f"Completed workload-space Pareto evaluation for {display_name} ({arch_name})."
        )
        print(f"Output directory: {output_dir}")

    return report


def merge_into_existing_output(
    *,
    workload_dirs: Sequence[str | Path],
    arch_name: str,
    output_dir: str | Path,
    arch_file: str | None = None,
    baseline_workload: str | None = None,
    target: str | None = None,
    workers: int = 4,
    verbose: int = 1,
) -> dict[str, Any]:
    resolved_input_dirs = tuple(_resolve_path(path) for path in workload_dirs)
    output_dir_path = _resolve_path(output_dir)
    if not output_dir_path.is_dir():
        raise ValueError(f"Pareto output directory does not exist: {output_dir_path}")

    arch_path = _resolve_arch_path(arch_name, arch_file)
    step_inputs = _discover_step_inputs(resolved_input_dirs)
    manifest_metadata = _validate_manifest_consistency(
        step_inputs=step_inputs,
        arch_name=arch_name,
        arch_path=arch_path,
    )
    existing_manifest = _load_optional_manifest(output_dir_path / "manifest.json")[1]
    existing_summary = _load_optional_manifest(output_dir_path / "summary.json")[1]

    baseline_path = (
        _resolve_path(baseline_workload) if baseline_workload is not None else None
    )
    baseline_payload: dict[str, Any] | None = None
    baseline_workload_name: str | None = None
    baseline_json_path = output_dir_path / "baseline.json"
    effective_target = target if target is not None else arch_name
    if baseline_json_path.is_file():
        baseline_payload = dict(_load_json_mapping(baseline_json_path))
        raw_name = baseline_payload.get("baseline_yaml_name") or baseline_payload.get(
            "validation_yaml_name"
        )
        if isinstance(raw_name, str) and raw_name.strip():
            baseline_workload_name = raw_name.strip()
    elif baseline_path is not None:
        arch = Arch.from_yaml_file(arch_path)
        baseline_payload, baseline_workload_name = _evaluate_baseline(
            baseline_path=baseline_path,
            arch=arch,
            arch_name=arch_name,
            arch_path=arch_path,
            target=effective_target,
        )
        _write_json(baseline_json_path, baseline_payload)

    display_name = _resolve_incremental_display_name(
        output_dir=output_dir_path,
        existing_summary=existing_summary,
        existing_manifest=existing_manifest,
        input_dirs=resolved_input_dirs,
        manifest_metadata=manifest_metadata,
        baseline_workload_name=baseline_workload_name,
    )

    evaluation_plan = _build_evaluation_plan(step_inputs)
    existing_evaluation_rows = _read_existing_evaluation_rows(
        output_dir_path / "evaluations.csv"
    )
    unique_rows_by_task_key = _existing_unique_rows_by_task_key(
        existing_evaluation_rows
    )
    missing_tasks = tuple(
        task
        for task in evaluation_plan.tasks
        if task.task_key not in unique_rows_by_task_key
    )
    if missing_tasks:
        new_unique_rows = _run_evaluations(
            tasks=missing_tasks,
            arch_path=arch_path,
            target=effective_target,
            workers=workers,
            verbose=verbose,
        )
        unique_rows_by_task_key.update(
            {str(row["task_key"]): dict(row) for row in new_unique_rows}
        )

    ordered_unique_rows = tuple(
        dict(unique_rows_by_task_key[task.task_key]) for task in evaluation_plan.tasks
    )
    evaluation_rows = _expand_evaluation_rows(
        evaluation_plan=evaluation_plan,
        unique_evaluation_rows=ordered_unique_rows,
    )
    successful_rows = [row for row in evaluation_rows if row["status"] == "ok"]
    if not successful_rows:
        raise RuntimeError(
            "No legal workloads were produced by the analytical evaluation."
        )

    evaluations_csv = output_dir_path / "evaluations.csv"
    _write_csv(
        evaluations_csv,
        fieldnames=EVALUATIONS_FIELDNAMES,
        rows=evaluation_rows,
    )

    step_summaries, pareto_rows = _build_step_artifacts(
        step_inputs=step_inputs,
        evaluation_plan=evaluation_plan,
        evaluation_rows=evaluation_rows,
        unique_evaluation_rows=ordered_unique_rows,
    )
    pareto_csv = output_dir_path / "pareto_frontiers.csv"
    _write_csv(
        pareto_csv,
        fieldnames=PARETO_FIELDNAMES,
        rows=pareto_rows,
    )

    source_manifests = tuple(
        dict.fromkeys(
            step_input.manifest_path.resolve()
            for step_input in step_inputs
            if step_input.manifest_path is not None
        )
    )
    previous_unique_tasks = len(
        _existing_unique_rows_by_task_key(existing_evaluation_rows)
    )
    incremental_update = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "previous_unique_evaluation_tasks": int(previous_unique_tasks),
        "new_unique_evaluation_tasks": int(len(missing_tasks)),
        "unique_evaluation_tasks_after_merge": int(len(evaluation_plan.tasks)),
    }
    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": arch_name,
        "arch_file": str(arch_path),
        "target": effective_target,
        "output_dir": str(output_dir_path),
        "requested_workload_dirs": [str(path) for path in resolved_input_dirs],
        "discovered_step_dirs": {
            step_input.step_id: str(step_input.dir_path) for step_input in step_inputs
        },
        "selected_steps": [step_input.step_id for step_input in step_inputs],
        "selected_workloads_total": int(
            sum(len(step.workload_paths) for step in step_inputs)
        ),
        "unique_evaluation_tasks": int(len(evaluation_plan.tasks)),
        "baseline_workload": (
            str(baseline_path)
            if baseline_path is not None
            else (
                existing_manifest.get("baseline_workload")
                if isinstance(existing_manifest, Mapping)
                else None
            )
        ),
        "source_manifests": [str(path) for path in source_manifests],
        "manifest_metadata": dict(manifest_metadata),
        "steps": step_summaries,
        "incremental_update": incremental_update,
    }
    manifest_json = output_dir_path / "manifest.json"
    _write_json(manifest_json, manifest_payload)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": arch_name,
        "arch_file": str(arch_path),
        "target": effective_target,
        "workload_display_name": display_name,
        "baseline_json": (
            str(baseline_json_path.resolve()) if baseline_json_path.is_file() else None
        ),
        "selected_steps": [step_input.step_id for step_input in step_inputs],
        "selected_workloads_total": int(
            sum(len(step.workload_paths) for step in step_inputs)
        ),
        "unique_evaluation_tasks": int(len(evaluation_plan.tasks)),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "pairwise_frontiers_csv": str(pareto_csv.resolve()),
        "pareto_front_csv": str(pareto_csv.resolve()),
        "steps": step_summaries,
        "incremental_update": incremental_update,
    }
    summary_json = output_dir_path / "summary.json"
    _write_json(summary_json, summary_payload)

    return {
        "status": "ok",
        "output_dir": str(output_dir_path),
        "manifest_json": str(manifest_json.resolve()),
        "summary_json": str(summary_json.resolve()),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "pairwise_frontiers_csv": str(pareto_csv.resolve()),
        "baseline_json": (
            str(baseline_json_path.resolve()) if baseline_json_path.is_file() else None
        ),
        "selected_steps": [step_input.step_id for step_input in step_inputs],
        "unique_evaluation_tasks": int(len(evaluation_plan.tasks)),
        "new_unique_evaluation_tasks": int(len(missing_tasks)),
    }


def _discover_step_inputs(input_dirs: Sequence[Path]) -> tuple[StepInput, ...]:
    discovered: dict[str, StepInput] = {}
    for input_dir in input_dirs:
        for step_input in _expand_workload_dir_input(input_dir):
            existing = discovered.get(step_input.step_id)
            if existing is not None and existing.dir_path != step_input.dir_path:
                raise ValueError(
                    f"Duplicate step '{step_input.step_id}' discovered in both "
                    f"{existing.dir_path} and {step_input.dir_path}."
                )
            discovered[step_input.step_id] = step_input

    if not discovered:
        raise ValueError("No workload step directories were discovered.")

    return tuple(sorted(discovered.values(), key=_step_input_sort_key))


def _expand_workload_dir_input(input_dir: Path) -> tuple[StepInput, ...]:
    if not input_dir.is_dir():
        raise ValueError(f"Workload directory does not exist: {input_dir}")

    spaces_dir = input_dir / "spaces"
    if spaces_dir.is_dir():
        manifest_path, manifest_payload = _load_optional_manifest(
            input_dir / "manifest.json"
        )
        mode_id, mode_label, all_streaming_false = _mode_metadata_from_manifest(
            manifest_payload
        )
        step_inputs: list[StepInput] = []
        for base_step_id in SPACE_ORDER:
            step_dir = spaces_dir / base_step_id
            if not step_dir.is_dir():
                continue
            workload_paths = _collect_step_workload_paths(step_dir)
            if not workload_paths:
                continue
            step_inputs.append(
                StepInput(
                    step_id=_series_step_id(base_step_id, mode_id),
                    base_step_id=base_step_id,
                    mode_id=mode_id,
                    mode_label=mode_label,
                    all_streaming_false=all_streaming_false,
                    step_label=_series_step_label(base_step_id, mode_id),
                    dir_path=step_dir.resolve(),
                    workload_paths=workload_paths,
                    manifest_path=manifest_path,
                    manifest_payload=manifest_payload,
                )
            )

        if not step_inputs:
            raise ValueError(f"No workload YAML files found under {spaces_dir}.")
        return tuple(step_inputs)

    base_step_id = input_dir.name.strip().lower()
    if base_step_id not in _STEP_IDS:
        raise ValueError(
            f"Unsupported workload directory '{input_dir}'. Expected a directory "
            "containing spaces/a,b,c,d or a directory named a, b, c, or d."
        )

    workload_paths = _collect_step_workload_paths(input_dir)
    if not workload_paths:
        raise ValueError(f"No workload YAML files found under {input_dir}.")

    manifest_path: Path | None = None
    manifest_payload: Mapping[str, Any] | None = None
    if input_dir.parent.name == "spaces":
        manifest_path, manifest_payload = _load_optional_manifest(
            input_dir.parent.parent / "manifest.json"
        )
    mode_id, mode_label, all_streaming_false = _mode_metadata_from_manifest(
        manifest_payload
    )

    return (
        StepInput(
            step_id=_series_step_id(base_step_id, mode_id),
            base_step_id=base_step_id,
            mode_id=mode_id,
            mode_label=mode_label,
            all_streaming_false=all_streaming_false,
            step_label=_series_step_label(base_step_id, mode_id),
            dir_path=input_dir.resolve(),
            workload_paths=workload_paths,
            manifest_path=manifest_path,
            manifest_payload=manifest_payload,
        ),
    )


def _collect_step_workload_paths(step_dir: Path) -> tuple[Path, ...]:
    trace_paths = tuple(
        sorted(
            path.resolve()
            for path in step_dir.glob("trace_*.yaml")
            if path.is_file() and _TRACE_NAME_RE.match(path.name)
        )
    )
    if trace_paths:
        return trace_paths

    return tuple(
        sorted(path.resolve() for path in step_dir.glob("*.yaml") if path.is_file())
    )


def _load_optional_manifest(
    path: Path,
) -> tuple[Path | None, Mapping[str, Any] | None]:
    if not path.is_file():
        return None, None
    payload = _load_json_mapping(path)
    return path.resolve(), payload


def _validate_manifest_consistency(
    *,
    step_inputs: Sequence[StepInput],
    arch_name: str,
    arch_path: Path,
    allow_arch_mismatch: bool = False,
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    expected_arch_file = str(arch_path.resolve())
    for step_input in step_inputs:
        payload = step_input.manifest_payload
        if payload is None:
            continue
        for field_name in _MANIFEST_COMPARE_FIELDS:
            raw_value = payload.get(field_name)
            value = _normalize_manifest_value(field_name, raw_value)
            if value is None:
                continue
            existing = metadata.get(field_name)
            if existing is None:
                metadata[field_name] = value
                continue
            if existing != value:
                raise ValueError(
                    f"Conflicting manifest metadata for '{field_name}': "
                    f"{existing!r} vs {value!r}."
                )

    manifest_arch_name = metadata.get("arch_name")
    if manifest_arch_name is not None and manifest_arch_name != arch_name:
        raise ValueError(
            f"Manifest arch_name '{manifest_arch_name}' does not match --arch "
            f"'{arch_name}'."
        )

    manifest_arch_file = metadata.get("arch_file")
    if manifest_arch_file is not None and manifest_arch_file != expected_arch_file:
        if allow_arch_mismatch:
            print(
                f"WARNING: evaluating with {expected_arch_file} although the "
                f"search manifests record arch_file '{manifest_arch_file}' "
                f"(--allow-manifest-arch-mismatch).",
                file=sys.stderr,
            )
        else:
            raise ValueError(
                f"Manifest arch_file '{manifest_arch_file}' does not match {expected_arch_file}."
            )

    return metadata


def _normalize_manifest_value(field_name: str, raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if field_name == "workload_path":
        return str(_resolve_path(raw_value))
    if field_name == "arch_file":
        path = _resolve_path(raw_value)
        canonical_name = LEGACY_ARCH_FILE_BASENAME_ALIASES.get(path.name)
        if canonical_name is not None:
            path = path.with_name(canonical_name)
        return str(path.resolve())
    value = str(raw_value).strip()
    if field_name == "arch_name":
        candidate = value.replace("-", "_")
        if candidate in DEFAULT_ARCH_FILES:
            value = candidate
    return value or None


def _mode_metadata_from_manifest(
    manifest_payload: Mapping[str, Any] | None,
) -> tuple[str, str, bool]:
    all_streaming_false = False
    if manifest_payload is not None:
        all_streaming_false = bool(manifest_payload.get("all_streaming_false", False))
    mode_id = NON_STREAMING_MODE_ID if all_streaming_false else CURRENT_MODE_ID
    return mode_id, MODE_LABELS[mode_id], all_streaming_false


def _series_step_id(base_step_id: str, mode_id: str) -> str:
    if mode_id == CURRENT_MODE_ID:
        return base_step_id
    if mode_id == NON_STREAMING_MODE_ID:
        return f"ns_{base_step_id}"
    return f"{mode_id}_{base_step_id}"


def _series_step_label(base_step_id: str, mode_id: str) -> str:
    base_label = SPACE_LABELS[base_step_id]
    if mode_id == CURRENT_MODE_ID:
        return base_label
    return f"{base_label} [{MODE_LABELS.get(mode_id, mode_id)}]"


def _step_input_sort_key(step_input: StepInput) -> tuple[int, int, str]:
    try:
        mode_index = MODE_ORDER.index(step_input.mode_id)
    except ValueError:
        mode_index = len(MODE_ORDER)
    try:
        base_index = SPACE_ORDER.index(step_input.base_step_id)
    except ValueError:
        base_index = len(SPACE_ORDER)
    return (mode_index, base_index, step_input.step_id)


def _evaluate_baseline(
    *,
    baseline_path: Path,
    arch: Arch,
    arch_name: str,
    arch_path: Path,
    target: str | None,
) -> tuple[dict[str, Any], str]:
    workload = Workload.from_file(baseline_path)
    report = evaluate_workload_analytical_report(
        workload,
        arch,
        target=target,
        latency_coeff=1.0,
        mem_footprint_coeff=1.0,
        energy_coeff=1.0,
        require_constraints=False,
        refresh=True,
    )
    if report is None:
        raise RuntimeError(f"Unable to evaluate baseline workload: {baseline_path}")

    return (
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "arch": arch_name,
            "arch_file": str(arch_path),
            "target": target,
            "validation_yaml_name": workload.name,
            "baseline_yaml_name": workload.name,
            "source_workload": str(baseline_path),
            "metrics": {
                "latency_cycles": report["latency_cycles"],
                "mem_footprint_bytes": report["mem_footprint_bytes"],
                "energy": report["energy"],
                "weighted_cost": report["weighted_cost"],
            },
            "breakdown": report["breakdown"],
        },
        workload.name,
    )


def _resolve_display_name(
    *,
    input_dirs: Sequence[Path],
    manifest_metadata: Mapping[str, Any],
    baseline_workload_name: str | None,
) -> str:
    manifest_name = str(manifest_metadata.get("workload_name", "")).strip()
    if manifest_name:
        return manifest_name

    if baseline_workload_name is not None:
        return baseline_workload_name

    input_tokens = [_input_root_token(path) for path in input_dirs]
    if input_tokens and len(set(input_tokens)) == 1:
        return input_tokens[0]
    if input_tokens:
        return input_tokens[0]
    return "workload-space"


def _resolve_incremental_display_name(
    *,
    output_dir: Path,
    existing_summary: Mapping[str, Any] | None,
    existing_manifest: Mapping[str, Any] | None,
    input_dirs: Sequence[Path],
    manifest_metadata: Mapping[str, Any],
    baseline_workload_name: str | None,
) -> str:
    if existing_summary is not None:
        raw_name = existing_summary.get("workload_display_name")
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()
    if existing_manifest is not None:
        metadata = existing_manifest.get("manifest_metadata")
        if isinstance(metadata, Mapping):
            raw_name = metadata.get("workload_name")
            if isinstance(raw_name, str) and raw_name.strip():
                return raw_name.strip()
    if output_dir.parent.name:
        return output_dir.parent.name
    return _resolve_display_name(
        input_dirs=input_dirs,
        manifest_metadata=manifest_metadata,
        baseline_workload_name=baseline_workload_name,
    )


def _input_root_token(path: Path) -> str:
    if path.name in _STEP_IDS and path.parent.name == "spaces":
        return path.parent.parent.name or path.parent.parent.stem
    return path.name or path.stem


def _build_evaluation_plan(step_inputs: Sequence[StepInput]) -> EvaluationPlan:
    tasks: list[EvaluationTask] = []
    occurrences_by_task_key: dict[str, list[EvaluationOccurrence]] = {}
    selected_workloads_by_step: dict[str, int] = {}
    evaluated_workloads_by_step: dict[str, int] = {}

    for step_input in step_inputs:
        selected_workloads_by_step[step_input.step_id] = len(step_input.workload_paths)
        evaluated_workloads_by_step.setdefault(step_input.step_id, 0)
        for attempt_index, workload_path in enumerate(step_input.workload_paths):
            task_key = _evaluation_task_key(workload_path)
            occurrences_by_task_key.setdefault(task_key, []).append(
                EvaluationOccurrence(
                    step_id=step_input.step_id,
                    base_step_id=step_input.base_step_id,
                    mode_id=step_input.mode_id,
                    mode_label=step_input.mode_label,
                    all_streaming_false=step_input.all_streaming_false,
                    step_label=step_input.step_label,
                    input_dir=step_input.dir_path,
                    workload_path=workload_path,
                    attempt_index=attempt_index,
                )
            )
            if len(occurrences_by_task_key[task_key]) != 1:
                continue
            tasks.append(
                EvaluationTask(
                    task_key=task_key,
                    workload_path=workload_path,
                    first_step_id=step_input.step_id,
                )
            )
            evaluated_workloads_by_step[step_input.step_id] += 1

    return EvaluationPlan(
        tasks=tuple(tasks),
        occurrences_by_task_key={
            task_key: tuple(occurrences)
            for task_key, occurrences in occurrences_by_task_key.items()
        },
        selected_workloads_by_step=dict(selected_workloads_by_step),
        evaluated_workloads_by_step=dict(evaluated_workloads_by_step),
    )


def _evaluation_task_key(workload_path: Path) -> str:
    try:
        workload = Workload.from_file(workload_path)
        return f"state:{_state_hash(workload.to_dict())}"
    except Exception:
        return f"path:{workload_path.resolve()}"


def _read_existing_evaluation_rows(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return tuple()
    with path.open("r", encoding="utf-8", newline="") as fh:
        return tuple(dict(row) for row in csv.DictReader(fh))


def _existing_unique_rows_by_task_key(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    unique_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_key = _task_key_for_existing_evaluation_row(row)
        if task_key is None:
            continue
        unique_rows.setdefault(task_key, dict(row, task_key=task_key))
    return unique_rows


def _task_key_for_existing_evaluation_row(row: Mapping[str, Any]) -> str | None:
    raw_state_hash = str(row.get("state_hash", "")).strip()
    if raw_state_hash:
        return f"state:{raw_state_hash}"
    raw_workload_path = str(row.get("workload_path", "")).strip()
    if raw_workload_path:
        return f"path:{_resolve_path(raw_workload_path)}"
    return None


def _run_evaluations(
    *,
    tasks: Sequence[EvaluationTask],
    arch_path: Path,
    target: str | None,
    workers: int,
    verbose: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    if not tasks:
        return tuple()

    eval_workers = _effective_eval_workers(workers)
    _emit_progress_callback(
        progress_callback,
        completed=0,
        total=len(tasks),
    )
    progress = (
        tqdm(
            total=len(tasks),
            desc=_evaluation_progress_description(0, len(tasks)),
            leave=False,
        )
        if verbose >= 1
        else None
    )
    try:

        def _on_progress(completed_count: int) -> None:
            if progress is not None:
                progress.update(completed_count - progress.n)
                progress.set_description(
                    _evaluation_progress_description(completed_count, len(tasks))
                )
            _emit_progress_callback(
                progress_callback,
                completed=completed_count,
                total=len(tasks),
            )

        rows_by_task_index = _run_tasks_on_recovering_pool(
            total_tasks=len(tasks),
            eval_workers=eval_workers,
            submit_task=lambda executor, task_index: executor.submit(
                _evaluate_task,
                tasks[task_index],
                str(arch_path),
                target,
            ),
            on_progress=_on_progress,
        )
    finally:
        if progress is not None:
            progress.close()

    return tuple(rows_by_task_index[index] for index in range(len(tasks)))


def _expand_evaluation_rows(
    *,
    evaluation_plan: EvaluationPlan,
    unique_evaluation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows_by_task_key = {
        str(row["task_key"]): dict(row) for row in unique_evaluation_rows
    }
    expanded_rows: list[dict[str, Any]] = []
    for task in evaluation_plan.tasks:
        base_row = rows_by_task_key[task.task_key]
        for occurrence in evaluation_plan.occurrences_by_task_key[task.task_key]:
            payload = dict(base_row)
            payload.update(
                {
                    "step_id": occurrence.step_id,
                    "base_step_id": occurrence.base_step_id,
                    "mode_id": occurrence.mode_id,
                    "mode_label": occurrence.mode_label,
                    "all_streaming_false": occurrence.all_streaming_false,
                    "step_label": occurrence.step_label,
                    "input_dir": str(occurrence.input_dir),
                    "attempt_index": int(occurrence.attempt_index),
                    "workload_path": str(occurrence.workload_path),
                }
            )
            expanded_rows.append(payload)
    return tuple(expanded_rows)


def _build_step_artifacts(
    *,
    step_inputs: Sequence[StepInput],
    evaluation_plan: EvaluationPlan,
    evaluation_rows: Sequence[Mapping[str, Any]],
    unique_evaluation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
    rows_by_step: dict[str, list[Mapping[str, Any]]] = {
        step_input.step_id: [] for step_input in step_inputs
    }
    for row in evaluation_rows:
        rows_by_step.setdefault(str(row["step_id"]), []).append(row)

    unique_runtime_by_step: dict[str, float] = {
        step_id: 0.0 for step_id in evaluation_plan.selected_workloads_by_step
    }
    for task, row in zip(evaluation_plan.tasks, unique_evaluation_rows, strict=True):
        runtime_s = _coerce_optional_float(row.get("runtime_s"))
        if runtime_s is None:
            continue
        unique_runtime_by_step[task.first_step_id] = (
            unique_runtime_by_step.get(task.first_step_id, 0.0) + runtime_s
        )

    step_summaries: dict[str, dict[str, Any]] = {}
    pareto_rows: list[dict[str, Any]] = []
    for step_input in step_inputs:
        step_rows = tuple(rows_by_step.get(step_input.step_id, ()))
        successful_rows = tuple(row for row in step_rows if row["status"] == "ok")
        deduped_rows = _dedupe_success_rows(successful_rows)
        frontiers = pairwise_frontiers(deduped_rows)
        step_runtime_s = float(unique_runtime_by_step.get(step_input.step_id, 0.0))
        step_pareto_rows = tuple(
            {
                **augment_frontier_row(
                    frontier_name=frontier_name,
                    x_metric=x_metric,
                    y_metric=y_metric,
                    step_id=step_input.step_id,
                    step_label=step_input.step_label,
                    step_runtime_s=step_runtime_s,
                    order=index,
                    row=row,
                ),
                "base_step_id": step_input.base_step_id,
                "mode_id": step_input.mode_id,
                "mode_label": step_input.mode_label,
                "all_streaming_false": step_input.all_streaming_false,
            }
            for frontier_name, x_metric, y_metric in PAIRWISE_FRONTIER_METRICS
            for index, row in enumerate(frontiers[frontier_name], start=1)
        )
        pareto_rows.extend(step_pareto_rows)
        step_summaries[step_input.step_id] = {
            "step_id": step_input.step_id,
            "base_step_id": step_input.base_step_id,
            "mode_id": step_input.mode_id,
            "mode_label": step_input.mode_label,
            "all_streaming_false": step_input.all_streaming_false,
            "step_label": step_input.step_label,
            "input_dir": str(step_input.dir_path),
            "selected_workloads": int(
                evaluation_plan.selected_workloads_by_step.get(step_input.step_id, 0)
            ),
            "evaluated_workloads": int(
                evaluation_plan.evaluated_workloads_by_step.get(step_input.step_id, 0)
            ),
            "successful_points": len(successful_rows),
            "unique_successful_points": len(deduped_rows),
            "invalid_points": sum(1 for row in step_rows if row["status"] == "invalid"),
            "error_points": sum(1 for row in step_rows if row["status"] == "error"),
            "pareto_points": len(step_pareto_rows),
            "total_runtime_s": step_runtime_s,
        }

    return step_summaries, tuple(pareto_rows)


def _dedupe_success_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_hash = str(row.get("state_hash", "")).strip()
        if not state_hash:
            continue
        candidate = dict(row)
        existing = deduped.get(state_hash)
        if existing is None or _evaluation_sort_key(candidate) < _evaluation_sort_key(
            existing
        ):
            deduped[state_hash] = candidate

    return tuple(sorted(deduped.values(), key=_evaluation_sort_key))


def _evaluation_sort_key(row: Mapping[str, Any]) -> tuple[float, int, int, int, str]:
    return (
        float(row["weighted_cost"]),
        int(row["latency_cycles"]),
        int(row["mem_footprint_bytes"]),
        int(row["energy"]),
        str(row["state_hash"]),
    )


def _effective_eval_workers(workers: int) -> int:
    return max(1, min(int(workers), os.cpu_count() or 1))


def _max_in_flight_evaluations(*, total_tasks: int, eval_workers: int) -> int:
    # Keep the pool fed without overwhelming the inter-process work queue.
    return max(1, min(int(total_tasks), int(max(eval_workers, 1) * 4)))


def _evaluation_progress_description(completed: int, total: int) -> str:
    return f"Evaluating unique workloads [{int(completed)}/{int(total)}]"


def _emit_progress_callback(
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    completed: int,
    total: int,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "completed": int(completed),
            "total": int(total),
        }
    )


def _make_eval_executor(eval_workers: int):
    if eval_workers == 1:
        return ThreadPoolExecutor(max_workers=1)
    return ProcessPoolExecutor(
        max_workers=eval_workers,
        mp_context=mp.get_context("spawn"),
    )


# Eval-pool stall recovery. A Python 3.10 spawn
# ProcessPoolExecutor can stop delivering completions without marking itself
# broken when a worker dies in the wrong window; the untimed wait() used here
# previously turned that into a permanent silent hang of the whole pipeline.
# The stall limit must comfortably exceed the slowest legitimate single task
# (the worst observed fig12 eval task ran ~405 s).
_EVAL_POOL_STALL_ENV = "PNMAX_EVAL_POOL_STALL_S"
_EVAL_POOL_RESTARTS_ENV = "PNMAX_EVAL_POOL_RESTARTS"
_DEFAULT_EVAL_POOL_STALL_S = 900.0
_DEFAULT_EVAL_POOL_RESTARTS = 3


def _eval_pool_stall_limit_s() -> float:
    raw = os.environ.get(_EVAL_POOL_STALL_ENV)
    if raw is None:
        return _DEFAULT_EVAL_POOL_STALL_S
    try:
        value = float(raw)
    except ValueError:
        print(
            f"WARNING: {_EVAL_POOL_STALL_ENV}={raw!r} is not a number; "
            f"using the default {_DEFAULT_EVAL_POOL_STALL_S} s.",
            file=sys.stderr,
        )
        return _DEFAULT_EVAL_POOL_STALL_S
    return max(1.0, value)


def _eval_pool_max_restarts() -> int:
    raw = os.environ.get(_EVAL_POOL_RESTARTS_ENV)
    if raw is None:
        return _DEFAULT_EVAL_POOL_RESTARTS
    try:
        value = int(raw)
    except ValueError:
        print(
            f"WARNING: {_EVAL_POOL_RESTARTS_ENV}={raw!r} is not an integer; "
            f"using the default {_DEFAULT_EVAL_POOL_RESTARTS}.",
            file=sys.stderr,
        )
        return _DEFAULT_EVAL_POOL_RESTARTS
    return max(0, value)


def _describe_alive_eval_workers(executor: Any) -> str:
    try:
        processes = list(executor._processes.values())
    except AttributeError:
        return "n/a"
    return str(sum(1 for process in processes if process.is_alive()))


def _dispose_wedged_eval_executor(executor: Any) -> None:
    """Best-effort teardown of a stalled or broken eval pool. Never raises.

    Workers are SIGKILLed first: a still-healthy manager thread then observes
    the deaths and completes its own broken-pool teardown. If the manager
    thread is itself wedged (the silent-wedge mode diagnosed on 3.10, e.g.
    blocked in a partial result read), it can never be joined — unhook it from
    BOTH interpreter-exit paths that would otherwise wait on it forever: the
    concurrent.futures atexit join list (``_threads_wakeups``) and the
    ``threading._shutdown()`` wait set (``_shutdown_locks``), which holds the
    tstate lock of every live non-daemon thread and is drained during
    interpreter finalization.
    """
    try:
        processes = list(executor._processes.values())
    except AttributeError:
        processes = []
    for process in processes:
        try:
            process.kill()
        except Exception:
            pass
    # ProcessPoolExecutor.shutdown() unconditionally sets
    # _executor_manager_thread = None (even with wait=False), so the manager
    # thread reference MUST be captured before calling it — otherwise the
    # wedged-manager unhooking below can never run on a real pool.
    manager = getattr(executor, "_executor_manager_thread", None)
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    if manager is None or not manager.is_alive():
        return
    manager.join(timeout=10.0)
    if not manager.is_alive():
        return
    try:
        from concurrent.futures import process as _cf_process

        _cf_process._threads_wakeups.pop(manager, None)
    except Exception:
        pass
    try:
        import threading

        lock = getattr(manager, "_tstate_lock", None)
        shutdown_locks = getattr(threading, "_shutdown_locks", None)
        if lock is not None and shutdown_locks is not None:
            shutdown_locks_lock = getattr(threading, "_shutdown_locks_lock", None)
            if shutdown_locks_lock is not None:
                with shutdown_locks_lock:
                    shutdown_locks.discard(lock)
            else:
                shutdown_locks.discard(lock)
    except Exception:
        pass


def _run_tasks_on_recovering_pool(
    *,
    total_tasks: int,
    eval_workers: int,
    submit_task: Callable[[Any, int], Any],
    on_progress: Callable[[int], None] | None = None,
) -> dict[int, Any]:
    """Collect results of pure, index-keyed eval tasks with stall recovery.

    ``submit_task(executor, task_index)`` must submit the task with the given
    index and return its future. Tasks must be pure functions of their inputs
    (analytical evaluations of on-disk YAMLs are), so re-executing a
    not-yet-completed task on a fresh pool cannot change any result value:
    completed results are never re-executed nor overwritten, and submission
    order is preserved. If the pool stops delivering completions for
    ``PNMAX_EVAL_POOL_STALL_S`` seconds (default 900) or turns broken, it is
    killed and replaced; after ``PNMAX_EVAL_POOL_RESTARTS`` (default 3)
    replacements the run fails loudly instead of hanging.
    """
    rows_by_task_index: dict[int, Any] = {}
    if total_tasks <= 0:
        return rows_by_task_index

    stall_limit_s = _eval_pool_stall_limit_s()
    max_restarts = _eval_pool_max_restarts()
    wait_poll_s = max(0.25, min(15.0, stall_limit_s / 4.0))
    restarts = 0

    while len(rows_by_task_index) < total_tasks:
        executor = _make_eval_executor(eval_workers)
        # Thread-based executors (eval_workers == 1) cannot be killed and have
        # never exhibited the wedge; keep their historical blocking behavior.
        recovery_enabled = isinstance(executor, ProcessPoolExecutor)
        pending_indices = [
            index for index in range(total_tasks) if index not in rows_by_task_index
        ]
        max_in_flight = _max_in_flight_evaluations(
            total_tasks=len(pending_indices),
            eval_workers=eval_workers,
        )
        in_flight_futures: dict[Any, int] = {}
        next_pending_pos = 0
        stalled = False
        broken: BrokenProcessPool | None = None
        last_completion = time.monotonic()

        try:
            while (
                next_pending_pos < len(pending_indices)
                and len(in_flight_futures) < max_in_flight
            ):
                task_index = pending_indices[next_pending_pos]
                in_flight_futures[submit_task(executor, task_index)] = task_index
                next_pending_pos += 1

            while in_flight_futures:
                completed, _ = wait(
                    tuple(in_flight_futures),
                    timeout=wait_poll_s if recovery_enabled else None,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    if (time.monotonic() - last_completion) >= stall_limit_s:
                        stalled = True
                        break
                    continue
                last_completion = time.monotonic()
                for future in completed:
                    task_index = in_flight_futures.pop(future)
                    rows_by_task_index[task_index] = future.result()
                    while (
                        next_pending_pos < len(pending_indices)
                        and len(in_flight_futures) < max_in_flight
                    ):
                        pending_index = pending_indices[next_pending_pos]
                        in_flight_futures[
                            submit_task(executor, pending_index)
                        ] = pending_index
                        next_pending_pos += 1
                if on_progress is not None:
                    on_progress(len(rows_by_task_index))
        except BrokenProcessPool as exc:
            broken = exc
        except BaseException:
            # Any other escape (a task exception re-raised by future.result(),
            # MemoryError, KeyboardInterrupt, ...) must not leak the pool: its
            # spawn workers would be orphaned and, if the manager thread later
            # wedges, interpreter exit would hang on it. Dispose of the pool,
            # then propagate the exception unchanged.
            _dispose_wedged_eval_executor(executor)
            raise

        if not stalled and broken is None:
            executor.shutdown(wait=True)
            continue

        restarts += 1
        reason = (
            "stopped delivering completions "
            f"(no result for >= {stall_limit_s:.0f}s)"
            if stalled
            else f"turned broken ({broken})"
        )
        print(
            f"[eval-pool-recovery] evaluation worker pool {reason}: "
            f"{len(rows_by_task_index)}/{total_tasks} tasks completed, "
            f"{len(in_flight_futures)} in flight, "
            f"{_describe_alive_eval_workers(executor)} worker process(es) "
            f"alive; killing and replacing the pool "
            f"(restart {restarts}/{max_restarts}).",
            file=sys.stderr,
            flush=True,
        )
        _dispose_wedged_eval_executor(executor)
        if restarts > max_restarts:
            raise RuntimeError(
                f"Evaluation worker pool {reason} after "
                f"{len(rows_by_task_index)}/{total_tasks} completed tasks; "
                f"exhausted {max_restarts} pool replacement(s); aborting "
                f"instead of hanging (tune {_EVAL_POOL_STALL_ENV} / "
                f"{_EVAL_POOL_RESTARTS_ENV} if this was a false stall)."
            ) from broken

    return rows_by_task_index


@lru_cache(maxsize=None)
def _load_arch_for_worker(arch_file: str) -> Arch:
    return Arch.from_yaml_file(Path(arch_file))


def _evaluate_task(
    task: EvaluationTask,
    arch_file: str,
    target: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        arch = _load_arch_for_worker(arch_file)
        workload = Workload.from_file(task.workload_path)
        workload_payload = workload.to_dict()
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
        if report is None:
            return {
                "task_key": task.task_key,
                "workload_name": workload.name,
                "workload_path": str(task.workload_path),
                "status": "invalid",
                "latency_cycles": None,
                "mem_footprint_bytes": None,
                "energy": None,
                "weighted_cost": None,
                "runtime_s": float(time.perf_counter() - started),
                "state_hash": _state_hash(workload_payload),
                "error_type": "ConstraintViolation",
                "error_message": "Analytical model constraints were not satisfied.",
                "breakdown_json": None,
            }
        return {
            "task_key": task.task_key,
            "workload_name": workload.name,
            "workload_path": str(task.workload_path),
            "status": "ok",
            "latency_cycles": int(report["latency_cycles"]),
            "mem_footprint_bytes": int(report["mem_footprint_bytes"]),
            "energy": int(report["energy"]),
            "weighted_cost": float(report["weighted_cost"]),
            "runtime_s": float(report["runtime_s"]),
            "state_hash": _state_hash(workload_payload),
            "error_type": None,
            "error_message": None,
            "breakdown_json": _serialize_compact_json(report.get("breakdown")),
        }
    except Exception as exc:
        return {
            "task_key": task.task_key,
            "workload_name": None,
            "workload_path": str(task.workload_path),
            "status": "error",
            "latency_cycles": None,
            "mem_footprint_bytes": None,
            "energy": None,
            "weighted_cost": None,
            "runtime_s": float(time.perf_counter() - started),
            "state_hash": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "breakdown_json": None,
        }


def _state_hash(workload_payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(workload_payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


def _resolve_arch_path(arch_name: str, arch_file: str | None) -> Path:
    if arch_file is not None:
        path = _resolve_path(arch_file)
        if not path.is_file():
            raise ValueError(f"Architecture file does not exist: {path}")
        return path

    try:
        path = _resolve_path(DEFAULT_ARCH_FILES[arch_name])
    except KeyError as exc:
        raise ValueError(f"Unsupported arch: {arch_name}") from exc
    if not path.is_file():
        raise ValueError(f"Architecture file does not exist: {path}")
    return path


def _resolve_pipeline_output_dir(
    *,
    output_root: Path,
    arch_name: str,
    workload_token: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        output_root
        / _sanitize_token(arch_name)
        / _sanitize_token(workload_token)
        / timestamp
    )


def _sanitize_token(token: str) -> str:
    sanitized = _ID_RE.sub("-", str(token).strip())
    sanitized = sanitized.strip("-")
    return sanitized or "item"


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        return (Path.cwd() / path).resolve()
    return path.resolve()


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping JSON payload in {path}.")
    return payload


def _write_output_dir_file(*, output_dir_file: str, output_dir: Path) -> None:
    path = _resolve_path(output_dir_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{output_dir}\n", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(dict(payload), indent=2)}\n", encoding="utf-8")


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _serialize_compact_json(payload: Any) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    run(
        workload_dirs=args.workload_dirs,
        arch_name=args.arch,
        output_root=args.output_root,
        arch_file=args.arch_file,
        baseline_workload=args.baseline_workload,
        output_dir_file=args.output_dir_file,
        target=args.target,
        workers=args.workers,
        verbose=args.verbose,
        allow_arch_mismatch=args.allow_manifest_arch_mismatch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
