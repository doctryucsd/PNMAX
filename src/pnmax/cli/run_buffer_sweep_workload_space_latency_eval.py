from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from tqdm import tqdm

from pnmax.cli import _workload_eval_overrides as _eval_overrides
from pnmax.cli import run_end_to_end_workload_space as _base_e2e
from pnmax.cli import run_workload_space_pareto_eval as _base_eval
from pnmax.database import Arch, Workload
from pnmax.dse.workload_eval import evaluate_workload_analytical_report

from pnmax.paths import repo_root, results_root
DEFAULT_ARCH_ROOT = repo_root() / "data" / "archs" / "lowered" / "buffer_sweep"
DEFAULT_OUTPUT_ROOT = results_root() / "buffer_sweep_workload_space_latency_eval"
MANIFEST_COMPARE_FIELDS: tuple[str, ...] = (
    "workload_path",
    "workload_name",
    "arch_name",
    "arch_file",
)
EVALUATIONS_FIELDNAMES: tuple[str, ...] = (
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
BEST_LATENCY_FIELDNAMES: tuple[str, ...] = (
    "variant_order",
    *EVALUATIONS_FIELDNAMES,
)


@dataclass(frozen=True)
class MappingSource:
    input_dir: Path
    mapping_dir: Path
    workload_paths: tuple[Path, ...]
    manifest_path: Path | None
    manifest_payload: Mapping[str, Any] | None


@dataclass(frozen=True)
class UniqueMapping:
    task_key: str
    workload_name: str | None
    workload_path: Path
    attempt_index: int


@dataclass(frozen=True)
class BufferVariant:
    variant_id: str
    variant_label: str
    arch_file: Path
    buffer_bytes: int
    area_mm2: float


@dataclass(frozen=True)
class EvaluationTask:
    task_key: str
    variant_id: str
    variant_label: str
    variant_arch_file: Path
    buffer_bytes: int
    area_mm2: float
    attempt_index: int
    workload_name: str | None
    workload_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate direct-(d) workload-space mappings against buffer-sweep "
            "architecture variants and export best-latency artifacts."
        )
    )
    parser.add_argument(
        "--mapping-dirs",
        nargs="+",
        required=True,
        help=(
            "One or more workload roots containing spaces/d or explicit spaces/d "
            "directories for a single (arch, workload) pair."
        ),
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help='Architecture family name ("upmem" or "hbm_pim").',
    )
    parser.add_argument(
        "--arch-root",
        type=str,
        default=str(DEFAULT_ARCH_ROOT),
        help="Architecture-root directory containing per-family buffer-sweep YAMLs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory root for generated buffer-sweep artifacts.",
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
    mapping_dirs: Sequence[str],
    arch_name: str,
    arch_root: str = str(DEFAULT_ARCH_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    output_dir_file: str | None = None,
    target: str | None = None,
    workers: int = 4,
    verbose: int = 1,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    canonical_arch_name = _base_e2e._canonical_arch_name(arch_name)
    resolved_input_dirs = tuple(_base_eval._resolve_path(path) for path in mapping_dirs)
    _validate_input_pair_consistency(
        input_dirs=resolved_input_dirs,
        arch_name=canonical_arch_name,
    )
    mapping_sources, manifest_metadata = _discover_mapping_sources(resolved_input_dirs)
    _validate_manifest_consistency(
        mapping_sources=mapping_sources,
        arch_name=canonical_arch_name,
    )
    variants = _discover_buffer_variants(
        arch_root=_base_eval._resolve_path(arch_root),
        arch_name=canonical_arch_name,
    )
    unique_mappings, mapping_summary = _discover_unique_mappings(mapping_sources)
    display_name = _resolve_display_name(
        input_dirs=resolved_input_dirs,
        manifest_metadata=manifest_metadata,
        unique_mappings=unique_mappings,
    )
    output_dir = _resolve_pipeline_output_dir(
        output_root=_base_eval._resolve_path(output_root),
        arch_name=canonical_arch_name,
        workload_token=display_name,
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir_file is not None:
        _base_eval._write_output_dir_file(
            output_dir_file=output_dir_file,
            output_dir=output_dir,
        )

    evaluation_tasks = _build_evaluation_tasks(
        unique_mappings=unique_mappings,
        variants=variants,
    )
    effective_target = target if target is not None else canonical_arch_name
    evaluation_rows = _run_evaluations(
        tasks=evaluation_tasks,
        target=effective_target,
        workers=workers,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    best_latency_rows = _select_best_latency_rows(
        variants=variants,
        evaluation_rows=evaluation_rows,
    )

    evaluations_csv = output_dir / "evaluations.csv"
    _base_eval._write_csv(
        evaluations_csv,
        fieldnames=EVALUATIONS_FIELDNAMES,
        rows=evaluation_rows,
    )
    best_latency_csv = output_dir / "best_latency.csv"
    _base_eval._write_csv(
        best_latency_csv,
        fieldnames=BEST_LATENCY_FIELDNAMES,
        rows=best_latency_rows,
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
    baseline_variant = variants[len(variants) // 2]
    variant_winners = {
        str(row["variant_id"]): {
            "variant_order": int(row["variant_order"]),
            "variant_id": str(row["variant_id"]),
            "variant_label": str(row["variant_label"]),
            "variant_arch_file": str(row["variant_arch_file"]),
            "buffer_bytes": int(row["buffer_bytes"]),
            "area_mm2": float(row["area_mm2"]),
            "attempt_index": int(row["attempt_index"]),
            "workload_name": str(row["workload_name"]),
            "workload_path": str(row["workload_path"]),
            "latency_cycles": int(row["latency_cycles"]),
            "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
            "energy": int(row["energy"]),
            "weighted_cost": float(row["weighted_cost"]),
            "state_hash": str(row["state_hash"]),
        }
        for row in best_latency_rows
    }

    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": canonical_arch_name,
        "target": effective_target,
        "output_dir": str(output_dir),
        "requested_mapping_dirs": [str(path) for path in resolved_input_dirs],
        "discovered_mapping_dirs": [str(source.mapping_dir) for source in mapping_sources],
        "source_manifests": [
            str(path)
            for path in dict.fromkeys(
                source.manifest_path.resolve()
                for source in mapping_sources
                if source.manifest_path is not None
            )
        ],
        "manifest_metadata": dict(manifest_metadata),
        "selected_variants": [variant.variant_id for variant in variants],
        "ordered_variants": ordered_variants,
        "selected_mappings_total": int(mapping_summary["selected_mappings_total"]),
        "unique_mappings_total": int(mapping_summary["unique_mappings_total"]),
        "duplicate_mappings_total": int(mapping_summary["duplicate_mappings_total"]),
        "unique_evaluation_tasks": int(len(evaluation_tasks)),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "best_latency_csv": str(best_latency_csv.resolve()),
    }
    manifest_json = output_dir / "manifest.json"
    _base_eval._write_json(manifest_json, manifest_payload)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": canonical_arch_name,
        "target": effective_target,
        "workload_display_name": display_name,
        "selected_variants": [variant.variant_id for variant in variants],
        "ordered_variants": ordered_variants,
        "baseline_variant_id": baseline_variant.variant_id,
        "selected_mappings_total": int(mapping_summary["selected_mappings_total"]),
        "unique_mappings_total": int(mapping_summary["unique_mappings_total"]),
        "duplicate_mappings_total": int(mapping_summary["duplicate_mappings_total"]),
        "unique_evaluation_tasks": int(len(evaluation_tasks)),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "best_latency_csv": str(best_latency_csv.resolve()),
        "variant_winners": variant_winners,
    }
    summary_json = output_dir / "summary.json"
    _base_eval._write_json(summary_json, summary_payload)

    report = {
        "status": "ok",
        "output_dir": str(output_dir),
        "manifest_json": str(manifest_json.resolve()),
        "summary_json": str(summary_json.resolve()),
        "evaluations_csv": str(evaluations_csv.resolve()),
        "best_latency_csv": str(best_latency_csv.resolve()),
        "selected_variants": [variant.variant_id for variant in variants],
        "unique_evaluation_tasks": int(len(evaluation_tasks)),
    }
    if verbose >= 1:
        print(
            f"Completed buffer-sweep workload-space latency evaluation for "
            f"{display_name} ({canonical_arch_name})."
        )
        print(f"Output directory: {output_dir}")
    return report


def _discover_mapping_sources(
    input_dirs: Sequence[Path],
) -> tuple[tuple[MappingSource, ...], dict[str, str]]:
    discovered_sources: dict[Path, MappingSource] = {}
    metadata: dict[str, str] = {}
    for input_dir in input_dirs:
        mapping_dir, manifest_path, manifest_payload = _resolve_mapping_dir(input_dir)
        workload_paths = _base_eval._collect_step_workload_paths(mapping_dir)
        if not workload_paths:
            raise ValueError(f"No workload YAML files found under {mapping_dir}.")
        source = MappingSource(
            input_dir=input_dir.resolve(),
            mapping_dir=mapping_dir.resolve(),
            workload_paths=workload_paths,
            manifest_path=manifest_path,
            manifest_payload=manifest_payload,
        )
        discovered_sources.setdefault(source.mapping_dir, source)
        if manifest_payload is None:
            continue
        for field_name in MANIFEST_COMPARE_FIELDS:
            value = _base_eval._normalize_manifest_value(
                field_name, manifest_payload.get(field_name)
            )
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
    if not discovered_sources:
        raise ValueError("No direct-(d) mapping directories were discovered.")
    return (
        tuple(sorted(discovered_sources.values(), key=lambda source: str(source.mapping_dir))),
        metadata,
    )


def _resolve_mapping_dir(
    input_dir: Path,
) -> tuple[Path, Path | None, Mapping[str, Any] | None]:
    if not input_dir.is_dir():
        raise ValueError(f"Mapping directory does not exist: {input_dir}")
    if input_dir.name == "d" and input_dir.parent.name == "spaces":
        manifest_path, manifest_payload = _base_eval._load_optional_manifest(
            input_dir.parent.parent / "manifest.json"
        )
        return input_dir.resolve(), manifest_path, manifest_payload

    mapping_dir = input_dir / "spaces" / "d"
    if not mapping_dir.is_dir():
        raise ValueError(
            f"Unsupported mapping directory '{input_dir}'. Expected a workload "
            "root containing spaces/d or an explicit spaces/d directory."
        )
    manifest_path, manifest_payload = _base_eval._load_optional_manifest(
        input_dir / "manifest.json"
    )
    return mapping_dir.resolve(), manifest_path, manifest_payload


def _validate_manifest_consistency(
    *,
    mapping_sources: Sequence[MappingSource],
    arch_name: str,
) -> None:
    manifest_arch_names: set[str] = set()
    for source in mapping_sources:
        payload = source.manifest_payload
        if payload is None:
            continue
        value = _base_eval._normalize_manifest_value("arch_name", payload.get("arch_name"))
        if value is None:
            continue
        manifest_arch_names.add(value)
    if manifest_arch_names and manifest_arch_names != {arch_name}:
        formatted = ", ".join(sorted(manifest_arch_names))
        raise ValueError(
            f"Discovered mapping inputs expose conflicting arch names: {formatted}"
        )


def _validate_input_pair_consistency(
    *,
    input_dirs: Sequence[Path],
    arch_name: str,
) -> None:
    inferred_arch_names: set[str] = set()
    inferred_workload_tokens: set[str] = set()
    for input_dir in input_dirs:
        inferred_arch_name, inferred_workload_token = _infer_input_pair_metadata(
            input_dir
        )
        if inferred_arch_name is not None:
            inferred_arch_names.add(inferred_arch_name)
        if inferred_workload_token is not None:
            inferred_workload_tokens.add(inferred_workload_token)

    if len(inferred_arch_names) > 1:
        formatted = ", ".join(sorted(inferred_arch_names))
        raise ValueError(
            f"Discovered mapping inputs span multiple architecture tokens: {formatted}"
        )
    if inferred_arch_names and inferred_arch_names != {arch_name}:
        formatted = ", ".join(sorted(inferred_arch_names))
        raise ValueError(
            f"Discovered mapping inputs expose arch token(s) {formatted}, "
            f"which do not match --arch {arch_name!r}."
        )
    if len(inferred_workload_tokens) > 1:
        formatted = ", ".join(sorted(inferred_workload_tokens))
        raise ValueError(
            f"Discovered mapping inputs span multiple workload tokens: {formatted}"
        )


def _infer_input_pair_metadata(input_dir: Path) -> tuple[str | None, str | None]:
    resolved_input_dir = input_dir.resolve()
    if resolved_input_dir.name == "d" and resolved_input_dir.parent.name == "spaces":
        workload_token = _nonempty_token(resolved_input_dir.parent.parent.name)
        arch_name = _try_canonical_arch_name(
            resolved_input_dir.parent.parent.parent.name
        )
        return arch_name, workload_token

    workload_token = _nonempty_token(resolved_input_dir.name)
    arch_name = _try_canonical_arch_name(resolved_input_dir.parent.name)
    return arch_name, workload_token


def _try_canonical_arch_name(raw_value: str) -> str | None:
    token = str(raw_value).strip()
    if not token:
        return None
    try:
        return _base_e2e._canonical_arch_name(token)
    except ValueError:
        return None


def _nonempty_token(raw_value: str) -> str | None:
    token = str(raw_value).strip()
    return token or None


def _discover_buffer_variants(
    *,
    arch_root: Path,
    arch_name: str,
) -> tuple[BufferVariant, ...]:
    arch_dir = arch_root / arch_name
    if not arch_dir.is_dir():
        raise ValueError(f"Buffer-sweep architecture directory does not exist: {arch_dir}")

    variants: list[BufferVariant] = []
    for arch_file in sorted(path.resolve() for path in arch_dir.glob("*.yaml") if path.is_file()):
        arch = Arch.from_yaml_file(arch_file)
        area_mm2 = _load_area_mm2(arch_file)
        variant_id = arch_file.stem
        variants.append(
            BufferVariant(
                variant_id=variant_id,
                variant_label=variant_id,
                arch_file=arch_file,
                buffer_bytes=int(arch.specs.cache_bytes),
                area_mm2=float(area_mm2),
            )
        )
    if not variants:
        raise ValueError(f"No buffer-sweep architecture YAMLs were found under {arch_dir}.")
    return tuple(
        sorted(
            variants,
            key=lambda variant: (
                int(variant.buffer_bytes),
                variant.variant_id,
            ),
        )
    )


def _load_area_mm2(path: Path) -> float:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML payload in {path}.")
    raw_area = payload.get("total_area_mm2")
    if raw_area is None:
        raise ValueError(f"Missing total_area_mm2 in {path}.")
    try:
        area_mm2 = float(raw_area)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid total_area_mm2 in {path}.") from exc
    if area_mm2 <= 0.0:
        raise ValueError(f"Nonpositive total_area_mm2 in {path}.")
    return area_mm2


def _discover_unique_mappings(
    mapping_sources: Sequence[MappingSource],
) -> tuple[tuple[UniqueMapping, ...], dict[str, int]]:
    unique_mappings_by_key: dict[str, UniqueMapping] = {}
    workload_names: set[str] = set()
    selected_mappings_total = 0
    for source in mapping_sources:
        for attempt_index, workload_path in enumerate(source.workload_paths):
            selected_mappings_total += 1
            unique_mapping = _build_unique_mapping(
                workload_path=workload_path,
                attempt_index=attempt_index,
            )
            if unique_mapping.workload_name is not None:
                workload_names.add(unique_mapping.workload_name)
            unique_mappings_by_key.setdefault(unique_mapping.task_key, unique_mapping)

    if not unique_mappings_by_key:
        raise ValueError("No direct-(d) mappings were discovered.")
    if len(workload_names) > 1:
        formatted = ", ".join(sorted(workload_names))
        raise ValueError(
            f"Discovered multiple workload names in one run: {formatted}"
        )

    unique_mappings = tuple(
        sorted(
            unique_mappings_by_key.values(),
            key=lambda item: (
                "" if item.workload_name is None else item.workload_name,
                str(item.workload_path),
            ),
        )
    )
    unique_total = len(unique_mappings)
    return unique_mappings, {
        "selected_mappings_total": int(selected_mappings_total),
        "unique_mappings_total": int(unique_total),
        "duplicate_mappings_total": int(selected_mappings_total - unique_total),
    }


def _build_unique_mapping(
    *,
    workload_path: Path,
    attempt_index: int,
) -> UniqueMapping:
    try:
        workload = Workload.from_file(workload_path)
        return UniqueMapping(
            task_key=f"state:{_base_eval._state_hash(workload.to_dict())}",
            workload_name=workload.name,
            workload_path=workload_path.resolve(),
            attempt_index=int(attempt_index),
        )
    except Exception:
        return UniqueMapping(
            task_key=f"path:{workload_path.resolve()}",
            workload_name=None,
            workload_path=workload_path.resolve(),
            attempt_index=int(attempt_index),
        )


def _resolve_display_name(
    *,
    input_dirs: Sequence[Path],
    manifest_metadata: Mapping[str, Any],
    unique_mappings: Sequence[UniqueMapping],
) -> str:
    manifest_name = str(manifest_metadata.get("workload_name", "")).strip()
    if manifest_name:
        return manifest_name
    discovered_names = sorted(
        {
            str(mapping.workload_name).strip()
            for mapping in unique_mappings
            if mapping.workload_name is not None and str(mapping.workload_name).strip()
        }
    )
    if len(discovered_names) == 1:
        return discovered_names[0]
    input_tokens = [_base_eval._input_root_token(path) for path in input_dirs]
    if input_tokens and len(set(input_tokens)) == 1:
        return input_tokens[0]
    if input_tokens:
        return input_tokens[0]
    return "workload-space-d"


def _build_evaluation_tasks(
    *,
    unique_mappings: Sequence[UniqueMapping],
    variants: Sequence[BufferVariant],
) -> tuple[EvaluationTask, ...]:
    tasks: list[EvaluationTask] = []
    for variant in variants:
        for mapping in unique_mappings:
            tasks.append(
                EvaluationTask(
                    task_key=f"{variant.variant_id}:{mapping.task_key}",
                    variant_id=variant.variant_id,
                    variant_label=variant.variant_label,
                    variant_arch_file=variant.arch_file,
                    buffer_bytes=variant.buffer_bytes,
                    area_mm2=variant.area_mm2,
                    attempt_index=mapping.attempt_index,
                    workload_name=mapping.workload_name,
                    workload_path=mapping.workload_path,
                )
            )
    return tuple(tasks)


def _run_evaluations(
    *,
    tasks: Sequence[EvaluationTask],
    target: str | None,
    workers: int,
    verbose: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    if not tasks:
        return tuple()

    eval_workers = _base_eval._effective_eval_workers(workers)
    rows_by_task_index: dict[int, dict[str, Any]] = {}
    _base_eval._emit_progress_callback(
        progress_callback,
        completed=0,
        total=len(tasks),
    )
    with _base_eval._make_eval_executor(eval_workers) as executor:
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
            in_flight_futures: dict[Any, int] = {}
            next_task_index = 0
            max_in_flight = _base_eval._max_in_flight_evaluations(
                total_tasks=len(tasks),
                eval_workers=eval_workers,
            )
            while next_task_index < len(tasks) and len(in_flight_futures) < max_in_flight:
                in_flight_futures[
                    executor.submit(
                        _evaluate_task,
                        tasks[next_task_index],
                        target,
                    )
                ] = next_task_index
                next_task_index += 1

            while in_flight_futures:
                completed, _ = _base_eval.wait(
                    tuple(in_flight_futures),
                    return_when=_base_eval.FIRST_COMPLETED,
                )
                for future in completed:
                    task_index = in_flight_futures.pop(future)
                    rows_by_task_index[task_index] = future.result()
                    while (
                        next_task_index < len(tasks)
                        and len(in_flight_futures) < max_in_flight
                    ):
                        in_flight_futures[
                            executor.submit(
                                _evaluate_task,
                                tasks[next_task_index],
                                target,
                            )
                        ] = next_task_index
                        next_task_index += 1
                if progress is not None:
                    progress.update(len(completed))
                    progress.set_description(
                        _evaluation_progress_description(progress.n, len(tasks))
                    )
                _base_eval._emit_progress_callback(
                    progress_callback,
                    completed=len(rows_by_task_index),
                    total=len(tasks),
                )
        finally:
            if progress is not None:
                progress.close()
    return tuple(rows_by_task_index[index] for index in range(len(tasks)))


def _evaluation_progress_description(completed: int, total: int) -> str:
    return f"Evaluating buffer variants [{int(completed)}/{int(total)}]"


def _evaluate_task(
    task: EvaluationTask,
    target: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        arch = _base_eval._load_arch_for_worker(str(task.variant_arch_file))
        workload = Workload.from_file(task.workload_path)
        _eval_overrides.enable_workload_cache(
            workload,
            context="buffer-sweep workload-space",
        )
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
                "variant_id": task.variant_id,
                "variant_label": task.variant_label,
                "variant_arch_file": str(task.variant_arch_file),
                "buffer_bytes": int(task.buffer_bytes),
                "area_mm2": float(task.area_mm2),
                "attempt_index": int(task.attempt_index),
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
            "variant_id": task.variant_id,
            "variant_label": task.variant_label,
            "variant_arch_file": str(task.variant_arch_file),
            "buffer_bytes": int(task.buffer_bytes),
            "area_mm2": float(task.area_mm2),
            "attempt_index": int(task.attempt_index),
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
            "breakdown_json": _base_eval._serialize_compact_json(report.get("breakdown")),
        }
    except Exception as exc:
        return {
            "variant_id": task.variant_id,
            "variant_label": task.variant_label,
            "variant_arch_file": str(task.variant_arch_file),
            "buffer_bytes": int(task.buffer_bytes),
            "area_mm2": float(task.area_mm2),
            "attempt_index": int(task.attempt_index),
            "workload_name": task.workload_name,
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


def _select_best_latency_rows(
    *,
    variants: Sequence[BufferVariant],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows_by_variant: dict[str, list[dict[str, Any]]] = {
        variant.variant_id: [] for variant in variants
    }
    for row in evaluation_rows:
        if str(row.get("status", "")).strip() != "ok":
            continue
        rows_by_variant.setdefault(str(row["variant_id"]), []).append(dict(row))

    best_rows: list[dict[str, Any]] = []
    for variant_order, variant in enumerate(variants, start=1):
        variant_rows = rows_by_variant.get(variant.variant_id, [])
        deduped_rows = _dedupe_success_rows(variant_rows)
        if not deduped_rows:
            raise RuntimeError(
                f"No successful mappings were produced for buffer variant '{variant.variant_id}'."
            )
        best_row = min(deduped_rows, key=_best_latency_sort_key)
        best_rows.append({"variant_order": int(variant_order), **best_row})
    return tuple(best_rows)


def _dedupe_success_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    deduped: dict[str, dict[str, Any]] = {}
    fallback_rows: list[dict[str, Any]] = []
    for row in rows:
        state_hash = str(row.get("state_hash", "")).strip()
        candidate = dict(row)
        if not state_hash:
            fallback_rows.append(candidate)
            continue
        existing = deduped.get(state_hash)
        if existing is None or _best_latency_sort_key(candidate) < _best_latency_sort_key(
            existing
        ):
            deduped[state_hash] = candidate
    combined = list(deduped.values()) + fallback_rows
    return tuple(sorted(combined, key=_best_latency_sort_key))


def _best_latency_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(row["latency_cycles"]),
        int(row["mem_footprint_bytes"]),
        int(row["energy"]),
        str(row.get("state_hash", "")),
    )


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
        mapping_dirs=args.mapping_dirs,
        arch_name=args.arch,
        arch_root=args.arch_root,
        output_root=args.output_root,
        output_dir_file=args.output_dir_file,
        target=args.target,
        workers=args.workers,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
