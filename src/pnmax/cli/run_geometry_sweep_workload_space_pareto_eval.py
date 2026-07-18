from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tqdm import tqdm

from pnmax.cli import run_workload_space_pareto_eval as _base_eval
from pnmax.database import Workload
from pnmax.dse.pareto_export import PAIRWISE_FRONTIER_METRICS, pairwise_frontiers
from pnmax.dse.workload_eval import evaluate_workload_analytical_report

from pnmax.paths import repo_root, results_root
DEFAULT_OUTPUT_ROOT = results_root() / "geometry_sweep_workload_space_pareto_eval"
GEOMETRY_FIELDNAMES: tuple[str, ...] = (
    "geometry_id",
    "geometry_label",
    "geometry_arch_file",
)
EVALUATIONS_FIELDNAMES: tuple[str, ...] = (
    *GEOMETRY_FIELDNAMES,
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
    *GEOMETRY_FIELDNAMES,
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
_SHARED_MANIFEST_COMPARE_FIELDS: tuple[str, ...] = (
    "workload_path",
    "workload_name",
    "arch_name",
)
_GEOMETRY_MANIFEST_COMPARE_FIELDS: tuple[str, ...] = (
    *_SHARED_MANIFEST_COMPARE_FIELDS,
    "arch_file",
)


@dataclass(frozen=True)
class GeometryInput:
    geometry_id: str
    geometry_label: str
    arch_file: Path
    input_dirs: tuple[Path, ...]
    step_inputs: tuple[_base_eval.StepInput, ...]
    source_manifests: tuple[Path, ...]


@dataclass(frozen=True)
class GeometryEvaluationTask:
    task_key: str
    geometry_id: str
    geometry_label: str
    geometry_arch_file: Path
    workload_path: Path


@dataclass(frozen=True)
class GeometryEvaluationOccurrence:
    geometry_id: str
    geometry_label: str
    geometry_arch_file: Path
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
class GeometryEvaluationPlan:
    tasks: tuple[GeometryEvaluationTask, ...]
    occurrences_by_task_key: Mapping[str, tuple[GeometryEvaluationOccurrence, ...]]
    selected_workloads_by_geometry: Mapping[str, int]
    evaluated_workloads_by_geometry: Mapping[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate geometry-sweep workload-space YAML traces with the "
            "analytical model and export geometry-comparison Pareto artifacts."
        )
    )
    parser.add_argument(
        "--geometry-dirs",
        nargs="+",
        required=True,
        help=(
            "One or more geometry-specific workload-space roots, each containing "
            "spaces/a,b,c,d and a manifest.json with the geometry arch_file."
        ),
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=sorted(_base_eval.DEFAULT_ARCH_FILES),
        help='Architecture family name ("upmem" or "hbm_pim").',
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory root for generated geometry-sweep Pareto artifacts.",
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
    geometry_dirs: Sequence[str],
    arch_name: str,
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    output_dir_file: str | None = None,
    target: str | None = None,
    workers: int = 4,
    verbose: int = 1,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    resolved_input_dirs = tuple(_base_eval._resolve_path(path) for path in geometry_dirs)
    geometry_inputs, manifest_metadata = _discover_geometry_inputs(
        input_dirs=resolved_input_dirs,
        arch_name=arch_name,
    )

    display_name = _base_eval._resolve_display_name(
        input_dirs=resolved_input_dirs,
        manifest_metadata=manifest_metadata,
        baseline_workload_name=None,
    )
    output_dir = _resolve_pipeline_output_dir(
        output_root=_base_eval._resolve_path(output_root),
        arch_name=arch_name,
        workload_token=display_name,
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir_file is not None:
        _base_eval._write_output_dir_file(
            output_dir_file=output_dir_file,
            output_dir=output_dir,
        )

    evaluation_plan = _build_geometry_evaluation_plan(geometry_inputs)
    effective_target = target if target is not None else arch_name
    unique_evaluation_rows = _run_geometry_evaluations(
        tasks=evaluation_plan.tasks,
        target=effective_target,
        workers=workers,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    evaluation_rows = _expand_geometry_evaluation_rows(
        evaluation_plan=evaluation_plan,
        unique_evaluation_rows=unique_evaluation_rows,
    )

    successful_rows = [row for row in evaluation_rows if row["status"] == "ok"]
    if not successful_rows:
        raise RuntimeError(
            "No legal workloads were produced by the analytical evaluation."
        )

    evaluations_csv = output_dir / "evaluations.csv"
    _base_eval._write_csv(
        evaluations_csv,
        fieldnames=EVALUATIONS_FIELDNAMES,
        rows=evaluation_rows,
    )

    geometry_summaries, pareto_rows = _build_geometry_artifacts(
        geometry_inputs=geometry_inputs,
        evaluation_plan=evaluation_plan,
        evaluation_rows=evaluation_rows,
        unique_evaluation_rows=unique_evaluation_rows,
    )
    pareto_csv = output_dir / "pareto_frontiers.csv"
    _base_eval._write_csv(
        pareto_csv,
        fieldnames=PARETO_FIELDNAMES,
        rows=pareto_rows,
    )

    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": arch_name,
        "target": effective_target,
        "output_dir": str(output_dir),
        "requested_geometry_dirs": [str(path) for path in resolved_input_dirs],
        "selected_geometries": [geometry.geometry_id for geometry in geometry_inputs],
        "selected_workloads_total": int(
            sum(
                geometry_summary["selected_workloads"]
                for geometry_summary in geometry_summaries.values()
            )
        ),
        "unique_evaluation_tasks": int(len(evaluation_plan.tasks)),
        "source_manifests": [
            str(path)
            for path in dict.fromkeys(
                manifest
                for geometry in geometry_inputs
                for manifest in geometry.source_manifests
            )
        ],
        "manifest_metadata": dict(manifest_metadata),
        "geometries": geometry_summaries,
    }
    manifest_json = output_dir / "manifest.json"
    _base_eval._write_json(manifest_json, manifest_payload)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": arch_name,
        "target": effective_target,
        "workload_display_name": display_name,
        "selected_geometries": [geometry.geometry_id for geometry in geometry_inputs],
        "selected_workloads_total": int(
            sum(
                geometry_summary["selected_workloads"]
                for geometry_summary in geometry_summaries.values()
            )
        ),
        "unique_evaluation_tasks": int(len(evaluation_plan.tasks)),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "pairwise_frontiers_csv": str(pareto_csv.resolve()),
        "pareto_front_csv": str(pareto_csv.resolve()),
        "geometries": geometry_summaries,
    }
    summary_json = output_dir / "summary.json"
    _base_eval._write_json(summary_json, summary_payload)

    report = {
        "status": "ok",
        "output_dir": str(output_dir),
        "manifest_json": str(manifest_json.resolve()),
        "summary_json": str(summary_json.resolve()),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "pairwise_frontiers_csv": str(pareto_csv.resolve()),
        "selected_geometries": [geometry.geometry_id for geometry in geometry_inputs],
        "unique_evaluation_tasks": int(len(evaluation_plan.tasks)),
    }

    if verbose >= 1:
        print(
            f"Completed geometry-sweep workload-space Pareto evaluation for "
            f"{display_name} ({arch_name})."
        )
        print(f"Output directory: {output_dir}")

    return report


def _discover_geometry_inputs(
    *,
    input_dirs: Sequence[Path],
    arch_name: str,
) -> tuple[tuple[GeometryInput, ...], dict[str, str]]:
    groups: dict[str, dict[str, Any]] = {}
    shared_metadata: dict[str, str] = {}

    for input_dir in input_dirs:
        step_inputs = _base_eval._expand_workload_dir_input(input_dir)
        geometry_metadata = _collect_geometry_manifest_metadata(
            step_inputs=step_inputs,
            arch_name=arch_name,
        )
        geometry_arch_file = Path(geometry_metadata["arch_file"]).resolve()
        geometry_id = _geometry_id_for_input_dir(
            input_dir=input_dir,
            arch_file=geometry_arch_file,
        )
        geometry_label = geometry_id

        for field_name in _SHARED_MANIFEST_COMPARE_FIELDS:
            value = geometry_metadata.get(field_name)
            if value is None:
                continue
            existing = shared_metadata.get(field_name)
            if existing is None:
                shared_metadata[field_name] = value
                continue
            if existing != value:
                raise ValueError(
                    f"Conflicting manifest metadata for '{field_name}': "
                    f"{existing!r} vs {value!r}."
                )

        group = groups.setdefault(
            geometry_id,
            {
                "geometry_label": geometry_label,
                "arch_file": geometry_arch_file,
                "input_dirs": [],
                "step_inputs": {},
                "source_manifests": [],
            },
        )
        existing_arch_file = Path(group["arch_file"]).resolve()
        if existing_arch_file != geometry_arch_file:
            raise ValueError(
                f"Geometry '{geometry_id}' resolved multiple arch_file values: "
                f"{existing_arch_file} vs {geometry_arch_file}."
            )

        group["input_dirs"].append(input_dir.resolve())
        for step_input in step_inputs:
            existing_step = group["step_inputs"].get(step_input.step_id)
            if existing_step is not None and existing_step.dir_path != step_input.dir_path:
                raise ValueError(
                    f"Duplicate step '{step_input.step_id}' discovered for geometry "
                    f"'{geometry_id}' in both {existing_step.dir_path} and "
                    f"{step_input.dir_path}."
                )
            group["step_inputs"][step_input.step_id] = step_input
            if step_input.manifest_path is not None:
                group["source_manifests"].append(step_input.manifest_path.resolve())

    if not groups:
        raise ValueError("No geometry workload directories were discovered.")

    geometry_inputs = tuple(
        GeometryInput(
            geometry_id=geometry_id,
            geometry_label=str(group["geometry_label"]),
            arch_file=Path(group["arch_file"]).resolve(),
            input_dirs=tuple(
                sorted(
                    dict.fromkeys(Path(path).resolve() for path in group["input_dirs"])
                )
            ),
            step_inputs=tuple(
                sorted(
                    group["step_inputs"].values(),
                    key=_base_eval._step_input_sort_key,
                )
            ),
            source_manifests=tuple(
                sorted(
                    dict.fromkeys(
                        Path(path).resolve() for path in group["source_manifests"]
                    )
                )
            ),
        )
        for geometry_id, group in sorted(groups.items())
    )
    return geometry_inputs, shared_metadata


def _collect_geometry_manifest_metadata(
    *,
    step_inputs: Sequence[_base_eval.StepInput],
    arch_name: str,
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for step_input in step_inputs:
        payload = step_input.manifest_payload
        if payload is None:
            continue
        for field_name in _GEOMETRY_MANIFEST_COMPARE_FIELDS:
            raw_value = payload.get(field_name)
            value = _base_eval._normalize_manifest_value(field_name, raw_value)
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

    arch_file = metadata.get("arch_file")
    if arch_file is None:
        raise ValueError(
            "Geometry workload directories must provide manifest.json with an "
            "'arch_file' field."
        )
    resolved_arch_file = _base_eval._resolve_path(arch_file)
    if not resolved_arch_file.is_file():
        raise ValueError(f"Architecture file does not exist: {resolved_arch_file}")
    metadata["arch_file"] = str(resolved_arch_file.resolve())
    return metadata


def _geometry_id_for_input_dir(*, input_dir: Path, arch_file: Path) -> str:
    stem = arch_file.stem.strip()
    if stem:
        return stem
    return input_dir.parent.name or input_dir.name or "geometry"


def _build_geometry_evaluation_plan(
    geometry_inputs: Sequence[GeometryInput],
) -> GeometryEvaluationPlan:
    tasks: list[GeometryEvaluationTask] = []
    occurrences_by_task_key: dict[str, list[GeometryEvaluationOccurrence]] = {}
    selected_workloads_by_geometry: dict[str, int] = {}
    evaluated_workloads_by_geometry: dict[str, int] = {}

    for geometry_input in geometry_inputs:
        geometry_id = geometry_input.geometry_id
        selected_workloads_by_geometry[geometry_id] = int(
            sum(len(step_input.workload_paths) for step_input in geometry_input.step_inputs)
        )
        evaluated_workloads_by_geometry.setdefault(geometry_id, 0)

        for step_input in geometry_input.step_inputs:
            for attempt_index, workload_path in enumerate(step_input.workload_paths):
                task_key = _geometry_evaluation_task_key(
                    geometry_id=geometry_id,
                    workload_path=workload_path,
                )
                occurrences_by_task_key.setdefault(task_key, []).append(
                    GeometryEvaluationOccurrence(
                        geometry_id=geometry_id,
                        geometry_label=geometry_input.geometry_label,
                        geometry_arch_file=geometry_input.arch_file,
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
                    GeometryEvaluationTask(
                        task_key=task_key,
                        geometry_id=geometry_id,
                        geometry_label=geometry_input.geometry_label,
                        geometry_arch_file=geometry_input.arch_file,
                        workload_path=workload_path,
                    )
                )
                evaluated_workloads_by_geometry[geometry_id] += 1

    return GeometryEvaluationPlan(
        tasks=tuple(tasks),
        occurrences_by_task_key={
            task_key: tuple(occurrences)
            for task_key, occurrences in occurrences_by_task_key.items()
        },
        selected_workloads_by_geometry=dict(selected_workloads_by_geometry),
        evaluated_workloads_by_geometry=dict(evaluated_workloads_by_geometry),
    )


def _geometry_evaluation_task_key(*, geometry_id: str, workload_path: Path) -> str:
    try:
        workload = Workload.from_file(workload_path)
        state_hash = _base_eval._state_hash(workload.to_dict())
        return f"{geometry_id}:state:{state_hash}"
    except Exception:
        return f"{geometry_id}:path:{workload_path.resolve()}"


def _run_geometry_evaluations(
    *,
    tasks: Sequence[GeometryEvaluationTask],
    target: str | None,
    workers: int,
    verbose: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    if not tasks:
        _base_eval._emit_progress_callback(
            progress_callback,
            completed=0,
            total=0,
        )
        return tuple()

    eval_workers = _base_eval._effective_eval_workers(workers)
    _base_eval._emit_progress_callback(
        progress_callback,
        completed=0,
        total=len(tasks),
    )
    progress = (
        tqdm(
            total=len(tasks),
            desc=_base_eval._evaluation_progress_description(0, len(tasks)),
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
                    _base_eval._evaluation_progress_description(
                        completed_count, len(tasks)
                    )
                )
            _base_eval._emit_progress_callback(
                progress_callback,
                completed=completed_count,
                total=len(tasks),
            )

        rows_by_task_index = _base_eval._run_tasks_on_recovering_pool(
            total_tasks=len(tasks),
            eval_workers=eval_workers,
            submit_task=lambda executor, task_index: executor.submit(
                _evaluate_geometry_task,
                tasks[task_index],
                target,
            ),
            on_progress=_on_progress,
        )
    finally:
        if progress is not None:
            progress.close()

    return tuple(rows_by_task_index[index] for index in range(len(tasks)))


def _evaluate_geometry_task(
    task: GeometryEvaluationTask,
    target: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        arch = _base_eval._load_arch_for_worker(str(task.geometry_arch_file))
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
                "geometry_id": task.geometry_id,
                "geometry_label": task.geometry_label,
                "geometry_arch_file": str(task.geometry_arch_file),
                "workload_name": workload.name,
                "workload_path": str(task.workload_path),
                "status": "invalid",
                "latency_cycles": None,
                "mem_footprint_bytes": None,
                "energy": None,
                "weighted_cost": None,
                "runtime_s": float(time.perf_counter() - started),
                "state_hash": _base_eval._state_hash(workload_payload),
                "error_type": "ConstraintViolation",
                "error_message": "Analytical model constraints were not satisfied.",
                "breakdown_json": None,
            }
        return {
            "task_key": task.task_key,
            "geometry_id": task.geometry_id,
            "geometry_label": task.geometry_label,
            "geometry_arch_file": str(task.geometry_arch_file),
            "workload_name": workload.name,
            "workload_path": str(task.workload_path),
            "status": "ok",
            "latency_cycles": int(report["latency_cycles"]),
            "mem_footprint_bytes": int(report["mem_footprint_bytes"]),
            "energy": int(report["energy"]),
            "weighted_cost": float(report["weighted_cost"]),
            "runtime_s": float(report["runtime_s"]),
            "state_hash": _base_eval._state_hash(workload_payload),
            "error_type": None,
            "error_message": None,
            "breakdown_json": _base_eval._serialize_compact_json(
                report.get("breakdown")
            ),
        }
    except Exception as exc:
        return {
            "task_key": task.task_key,
            "geometry_id": task.geometry_id,
            "geometry_label": task.geometry_label,
            "geometry_arch_file": str(task.geometry_arch_file),
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


def _expand_geometry_evaluation_rows(
    *,
    evaluation_plan: GeometryEvaluationPlan,
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
                    "geometry_id": occurrence.geometry_id,
                    "geometry_label": occurrence.geometry_label,
                    "geometry_arch_file": str(occurrence.geometry_arch_file),
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


def _build_geometry_artifacts(
    *,
    geometry_inputs: Sequence[GeometryInput],
    evaluation_plan: GeometryEvaluationPlan,
    evaluation_rows: Sequence[Mapping[str, Any]],
    unique_evaluation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
    rows_by_geometry: dict[str, list[Mapping[str, Any]]] = {
        geometry_input.geometry_id: [] for geometry_input in geometry_inputs
    }
    for row in evaluation_rows:
        rows_by_geometry.setdefault(str(row["geometry_id"]), []).append(row)

    unique_runtime_by_geometry: dict[str, float] = {
        geometry_id: 0.0
        for geometry_id in evaluation_plan.selected_workloads_by_geometry
    }
    for task, row in zip(evaluation_plan.tasks, unique_evaluation_rows, strict=True):
        runtime_s = _base_eval._coerce_optional_float(row.get("runtime_s"))
        if runtime_s is None:
            continue
        unique_runtime_by_geometry[task.geometry_id] = (
            unique_runtime_by_geometry.get(task.geometry_id, 0.0) + runtime_s
        )

    geometry_summaries: dict[str, dict[str, Any]] = {}
    pareto_rows: list[dict[str, Any]] = []
    for geometry_input in geometry_inputs:
        geometry_id = geometry_input.geometry_id
        geometry_rows = tuple(rows_by_geometry.get(geometry_id, ()))
        successful_rows = tuple(row for row in geometry_rows if row["status"] == "ok")
        deduped_rows = _base_eval._dedupe_success_rows(successful_rows)
        frontiers = pairwise_frontiers(deduped_rows)
        geometry_runtime_s = float(unique_runtime_by_geometry.get(geometry_id, 0.0))

        geometry_pareto_rows = tuple(
            _augment_geometry_frontier_row(
                frontier_name=frontier_name,
                x_metric=x_metric,
                y_metric=y_metric,
                geometry_input=geometry_input,
                total_runtime_s=geometry_runtime_s,
                order=index,
                row=row,
            )
            for frontier_name, x_metric, y_metric in PAIRWISE_FRONTIER_METRICS
            for index, row in enumerate(frontiers[frontier_name], start=1)
        )
        pareto_rows.extend(geometry_pareto_rows)

        geometry_summaries[geometry_id] = {
            "geometry_id": geometry_id,
            "geometry_label": geometry_input.geometry_label,
            "geometry_arch_file": str(geometry_input.arch_file),
            "input_dirs": [str(path) for path in geometry_input.input_dirs],
            "selected_steps": [
                step_input.step_id for step_input in geometry_input.step_inputs
            ],
            "selected_workloads": int(
                evaluation_plan.selected_workloads_by_geometry.get(geometry_id, 0)
            ),
            "evaluated_workloads": int(
                evaluation_plan.evaluated_workloads_by_geometry.get(geometry_id, 0)
            ),
            "successful_points": len(successful_rows),
            "unique_successful_points": len(deduped_rows),
            "invalid_points": sum(
                1 for row in geometry_rows if row["status"] == "invalid"
            ),
            "error_points": sum(
                1 for row in geometry_rows if row["status"] == "error"
            ),
            "pareto_points": len(geometry_pareto_rows),
            "total_runtime_s": geometry_runtime_s,
        }

    return geometry_summaries, tuple(pareto_rows)


def _augment_geometry_frontier_row(
    *,
    frontier_name: str,
    x_metric: str,
    y_metric: str,
    geometry_input: GeometryInput,
    total_runtime_s: float,
    order: int,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(row)
    payload.update(
        {
            "frontier_name": frontier_name,
            "x_metric": x_metric,
            "y_metric": y_metric,
            "geometry_id": geometry_input.geometry_id,
            "geometry_label": geometry_input.geometry_label,
            "geometry_arch_file": str(geometry_input.arch_file),
            "order": int(order),
            "total_runtime_s": float(total_runtime_s),
        }
    )
    return payload


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


def main() -> int:
    args = parse_args()
    run(
        geometry_dirs=args.geometry_dirs,
        arch_name=args.arch,
        output_root=args.output_root,
        output_dir_file=args.output_dir_file,
        target=args.target,
        workers=args.workers,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
