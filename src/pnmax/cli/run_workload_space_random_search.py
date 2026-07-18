from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from pnmax.database import Arch, Workload
from pnmax.dse.random_search import (
    OutputDimensionFactorizationConstraint,
    RandomSearchConfig,
    _build_dedup_key,
    _dedup_key_for_workload,
    _has_existing_search_artifacts,
    _next_trace_index,
    _prepare_workload_contexts,
    _serialize_output_dim_constraint,
    run_random_search,
)
from pnmax.paths import repo_root
from pnmax.seeding import default_seed
from pnmax.dse.workload_search_spaces import (
    SPACE_ORDER,
    SPACE_LABELS,
    plan_search_space_execution,
    prepare_workload_for_space,
)

PROJECT_ROOT = repo_root()
DEFAULT_ARCH_FILES: dict[str, Path] = {
    "upmem": repo_root() / "data" / "archs" / "lowered" / "baseline/upmem.yaml",
    "hbm_pim": repo_root() / "data" / "archs" / "lowered" / "baseline/hbm_pim.yaml",
}
_TRACE_NAME_RE = re.compile(r"^trace_(\d+)\.yaml$")
SEARCH_SEMANTICS = "exclusive-union-v2"
SEARCH_SEMANTICS_VERSION = 3
_ROOT_COMPARE_FIELDS: tuple[str, ...] = (
    "workload_path",
    "workload_name",
    "arch_name",
    "arch_file",
    "dimensions",
    "levels",
    "all_streaming_false",
    "direct_c_search",
    "direct_d_search",
    "direct_l4_search",
    "output_dim_constraint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample legal workloads across visible search-space unions "
            "where a is the base space, b adds only PU-sharing points, c adds "
            "only PU-sharing plus max-system-level points, and d is derived "
            "from the full visible c set with cache enabled unless '--direct-d-search' "
            "is used."
        )
    )
    parser.add_argument(
        "--workload",
        type=str,
        required=True,
        help="Input workload YAML path.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=sorted(DEFAULT_ARCH_FILES),
        help='Architecture name ("upmem" or "hbm_pim").',
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory root for the search-space run.",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        required=True,
        help=(
            "Number of new unique legal workloads to add per executed search "
            "space. Visible b/c totals include imported predecessor traces."
        ),
    )
    parser.add_argument(
        "--arch-file",
        type=str,
        default=None,
        help="Optional architecture YAML path override.",
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=list(SPACE_ORDER),
        help=(
            'Requested visible spaces, e.g. "--spaces a d". Prerequisite '
            "visible spaces are materialized automatically, and d is derived "
            "from the full visible c set."
        ),
    )
    parser.add_argument(
        "--direct-c-search",
        action="store_true",
        help=(
            "Search the full (c) superset directly by sampling both "
            "Layout-Pragma.sharing.PU and Layout-Pragma.sharing.system across "
            "their legal off/on states. Only valid with '--spaces c'."
        ),
    )
    parser.add_argument(
        "--direct-d-search",
        action="store_true",
        help=(
            "Search the full (d) superset directly by sampling both "
            "Layout-Pragma.sharing.PU and Layout-Pragma.sharing.system across "
            "their legal off/on states while forcing cache enabled. Only valid "
            "with '--spaces d'."
        ),
    )
    parser.add_argument(
        "--direct-l4-search",
        action="store_true",
        help=(
            "Search the (c) superset with PU-sharing sampled but "
            "Layout-Pragma.sharing.system FORCED to the within-channel cap "
            "(channel hierarchy level - 1), i.e. an L4 within-channel mapping "
            "pool. Only valid with '--spaces c'."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
        help="Base seed for reproducibility (default: --seed > PNMAX_SEED > 42).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10_000,
        help="Maximum number of attempts per selected space.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of processes per selected space.",
    )
    parser.add_argument(
        "--batch-attempts",
        type=int,
        default=1,
        help="Attempt batch size per worker; defaults to 1 for better kill safety.",
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
        "--no-resume",
        action="store_true",
        help="Disable resume and require empty per-space output directories.",
    )
    parser.add_argument(
        "--all-streaming-false",
        action="store_true",
        help="Force all Layout-Pragma.streaming flags (In/F/Out) to false before space mutations.",
    )
    return parser.parse_args()


def run(
    *,
    workload_file: str,
    arch_name: str,
    out_dir: str,
    num_traces: int,
    spaces: Sequence[str],
    arch_file: str | None = None,
    seed: int | None = None,
    max_attempts: int = 10_000,
    workers: int = 1,
    batch_attempts: int = 1,
    dimensions: Sequence[str] | None = None,
    levels: Sequence[int] | None = None,
    show_progress: bool = True,
    use_early_checks: bool = True,
    warmup_cache: bool = True,
    resume: bool = True,
    all_streaming_false: bool = False,
    direct_c_search: bool = False,
    direct_d_search: bool = False,
    direct_l4_search: bool = False,
    output_dim_constraint: OutputDimensionFactorizationConstraint | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    execution_plan = plan_search_space_execution(
        spaces,
        direct_c_search=direct_c_search,
        direct_d_search=direct_d_search,
        direct_l4_search=direct_l4_search,
    )
    workload_path = _resolve_path(workload_file)
    arch_path = _resolve_arch_path(arch_name, arch_file)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workload = Workload.from_file(workload_path)
    arch = Arch.from_yaml_file(arch_path)
    migration_payload = _prepare_output_dir_for_run(
        output_dir=output_dir,
        workload_path=workload_path,
        workload_name=workload.name,
        arch_name=arch_name,
        arch_path=arch_path,
        dimensions=dimensions,
        levels=levels,
        all_streaming_false=all_streaming_false,
        requested_spaces=execution_plan.requested_spaces,
        resume=resume,
        direct_c_search=execution_plan.direct_c_search,
        direct_d_search=execution_plan.direct_d_search,
        direct_l4_search=execution_plan.direct_l4_search,
        output_dim_constraint=output_dim_constraint,
    )

    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "search_semantics": SEARCH_SEMANTICS,
        "search_semantics_version": SEARCH_SEMANTICS_VERSION,
        "search_plan_mode": execution_plan.search_plan_mode,
        "workload_path": str(workload_path),
        "workload_name": workload.name,
        "arch_name": arch_name,
        "arch_file": str(arch_path),
        "output_dir": str(output_dir),
        "direct_c_search": bool(execution_plan.direct_c_search),
        "direct_d_search": bool(execution_plan.direct_d_search),
        "direct_l4_search": bool(execution_plan.direct_l4_search),
        "spaces": list(execution_plan.requested_spaces),
        "requested_spaces": list(execution_plan.requested_spaces),
        "executed_search_spaces": list(execution_plan.executed_search_spaces),
        "derived_spaces": list(execution_plan.derived_spaces),
        "visible_spaces": list(execution_plan.visible_spaces),
        "num_traces_per_space_delta": int(num_traces),
        "executed_stage_delta_targets": {
            space_id: int(num_traces)
            for space_id in execution_plan.executed_search_spaces
        },
        "executed_stage_target_totals": {
            space_id: _stage_target_total(
                execution_plan.executed_search_spaces,
                num_traces,
                space_id,
            )
            for space_id in execution_plan.executed_search_spaces
        },
        "seed": seed,
        "max_attempts": int(max_attempts),
        "workers": int(workers),
        "batch_attempts": int(batch_attempts),
        "dimensions": list(dimensions) if dimensions is not None else None,
        "levels": list(levels) if levels is not None else None,
        "all_streaming_false": bool(all_streaming_false),
        "resume": bool(resume),
        "output_dim_constraint": _serialize_output_dim_constraint(
            output_dim_constraint
        ),
        "legacy_output_migration": migration_payload,
    }
    _write_json(output_dir / "manifest.json", manifest_payload)

    prepared_by_space = {
        space_id: prepare_workload_for_space(
            workload,
            arch,
            space_id,
            all_streaming_false=all_streaming_false,
            direct_c_search=execution_plan.direct_c_search,
            direct_d_search=execution_plan.direct_d_search,
            direct_l4_search=execution_plan.direct_l4_search,
        )
        for space_id in execution_plan.visible_spaces
    }
    space_summaries: dict[str, dict[str, Any]] = {}
    previous_search_space_id: str | None = None
    for space_id in execution_plan.executed_search_spaces:
        prepared = prepared_by_space[space_id]
        space_out_dir = output_dir / "spaces" / space_id
        stage_target_total = _stage_target_total(
            execution_plan.executed_search_spaces,
            num_traces,
            space_id,
        )
        config = RandomSearchConfig(
            out_dir=space_out_dir,
            num_traces=stage_target_total,
            max_attempts=max_attempts,
            num_workers=workers,
            dimensions=dimensions,
            level_ids=levels,
            seed=seed,
            batch_attempts=batch_attempts,
            show_progress=show_progress,
            use_early_checks=use_early_checks,
            warmup_cache=warmup_cache,
            search_pu_sharing=prepared.search_pu_sharing,
            search_system_sharing=prepared.search_system_sharing,
            search_cache=prepared.search_cache,
            output_dim_constraint=output_dim_constraint,
            resume=resume,
            resume_metadata={
                "space_id": space_id,
                "progress_label": f"{arch_name}/{workload.name} space {space_id}",
                "workload_path": str(workload_path),
                "workload_name": workload.name,
                "arch_name": arch_name,
                "arch_file": str(arch_path),
            },
            progress_callback=progress_callback,
        )
        if previous_search_space_id is not None:
            _import_missing_predecessor_traces(
                source_dir=output_dir / "spaces" / previous_search_space_id,
                dest_dir=space_out_dir,
                prepared=prepared,
                arch=arch,
                config=config,
                resume=resume,
            )
        run_random_search(prepared.workload, arch, config)
        annotated_summary = _annotate_space_summary(
            space_id=space_id,
            summary=_read_json_object(space_out_dir / "summary.json"),
            requested=(space_id in execution_plan.requested_spaces),
            derived=False,
            derived_from=None,
            delta_target=num_traces,
            target_total=stage_target_total,
            direct_c_search=execution_plan.direct_c_search,
            direct_d_search=execution_plan.direct_d_search,
            direct_l4_search=execution_plan.direct_l4_search,
            search_plan_mode=execution_plan.search_plan_mode,
        )
        _write_json(space_out_dir / "summary.json", annotated_summary)
        space_summaries[space_id] = annotated_summary
        previous_search_space_id = space_id

    for space_id in execution_plan.derived_spaces:
        if space_id != "d":
            raise ValueError(f"Unsupported derived search space '{space_id}'.")
        derived_summary = _derive_space_d_from_c(
            source_dir=output_dir / "spaces" / "c",
            dest_dir=output_dir / "spaces" / space_id,
            arch=arch,
            resume=resume,
        )
        annotated_summary = _annotate_space_summary(
            space_id=space_id,
            summary=derived_summary,
            requested=(space_id in execution_plan.requested_spaces),
            derived=True,
            derived_from="c",
            delta_target=None,
            target_total=space_summaries["c"]["target_total"],
            direct_c_search=execution_plan.direct_c_search,
            direct_d_search=execution_plan.direct_d_search,
            direct_l4_search=execution_plan.direct_l4_search,
            search_plan_mode=execution_plan.search_plan_mode,
        )
        _write_json(
            output_dir / "spaces" / space_id / "summary.json", annotated_summary
        )
        space_summaries[space_id] = annotated_summary

    summary_payload = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "search_semantics": SEARCH_SEMANTICS,
        "search_semantics_version": SEARCH_SEMANTICS_VERSION,
        "search_plan_mode": execution_plan.search_plan_mode,
        "workload_path": str(workload_path),
        "workload_name": workload.name,
        "arch_name": arch_name,
        "arch_file": str(arch_path),
        "output_dir": str(output_dir),
        "direct_c_search": bool(execution_plan.direct_c_search),
        "direct_d_search": bool(execution_plan.direct_d_search),
        "direct_l4_search": bool(execution_plan.direct_l4_search),
        "requested_spaces": list(execution_plan.requested_spaces),
        "executed_search_spaces": list(execution_plan.executed_search_spaces),
        "derived_spaces": list(execution_plan.derived_spaces),
        "visible_spaces": list(execution_plan.visible_spaces),
        "num_traces_per_space_delta": int(num_traces),
        "all_streaming_false": bool(all_streaming_false),
        "output_dim_constraint": _serialize_output_dim_constraint(
            output_dim_constraint
        ),
        "spaces": {
            space_id: space_summaries[space_id]
            for space_id in execution_plan.visible_spaces
        },
        "manifest_json": str(output_dir / "manifest.json"),
        "summary_json": str(output_dir / "summary.json"),
        "legacy_output_migration": migration_payload,
    }
    _write_json(output_dir / "summary.json", summary_payload)

    print(f"output_dir: {output_dir}")
    for space_id in execution_plan.visible_spaces:
        space_summary = space_summaries[space_id]
        print(
            f"space {space_id}: status={space_summary['status']} "
            f"saved_total={space_summary['saved_total']} attempts={space_summary['attempts']} "
            f"accepted={space_summary['accepted']}"
        )

    return summary_payload


def _annotate_space_summary(
    *,
    space_id: str,
    summary: Mapping[str, Any],
    requested: bool,
    derived: bool,
    derived_from: str | None,
    delta_target: int | None,
    target_total: int,
    direct_c_search: bool,
    direct_d_search: bool,
    direct_l4_search: bool,
    search_plan_mode: str,
) -> dict[str, Any]:
    payload = dict(summary)
    payload["search_semantics"] = SEARCH_SEMANTICS
    payload["search_semantics_version"] = SEARCH_SEMANTICS_VERSION
    payload["search_plan_mode"] = str(search_plan_mode)
    payload["space_id"] = space_id
    payload["label"] = SPACE_LABELS[space_id]
    payload["requested"] = bool(requested)
    payload["derived"] = bool(derived)
    payload["implicit"] = not requested
    payload["derived_from"] = derived_from
    payload["direct_c_search"] = bool(direct_c_search)
    payload["direct_d_search"] = bool(direct_d_search)
    payload["direct_l4_search"] = bool(direct_l4_search)
    if delta_target is not None:
        payload["delta_target"] = int(delta_target)
    payload["target_total"] = int(target_total)
    return payload


def _prepare_output_dir_for_run(
    *,
    output_dir: Path,
    workload_path: Path,
    workload_name: str,
    arch_name: str,
    arch_path: Path,
    dimensions: Sequence[str] | None,
    levels: Sequence[int] | None,
    all_streaming_false: bool,
    requested_spaces: Sequence[str],
    resume: bool,
    direct_c_search: bool,
    direct_d_search: bool,
    direct_l4_search: bool,
    output_dim_constraint: OutputDimensionFactorizationConstraint | None,
) -> dict[str, Any]:
    if not resume:
        return {
            "detected_legacy_output": False,
            "migrated_legacy_output": False,
            "preserved_a_traces": False,
        }

    manifest_path = output_dir / "manifest.json"
    existing_manifest = _read_optional_json_object(manifest_path)
    legacy_output = _is_legacy_output_root(output_dir, existing_manifest)
    if not legacy_output:
        _validate_existing_output_root(
            existing_manifest=existing_manifest,
            workload_path=workload_path,
            workload_name=workload_name,
            arch_name=arch_name,
            arch_path=arch_path,
            dimensions=dimensions,
            levels=levels,
            all_streaming_false=all_streaming_false,
            direct_c_search=direct_c_search,
            direct_d_search=direct_d_search,
            direct_l4_search=direct_l4_search,
            output_dim_constraint=output_dim_constraint,
        )
        return {
            "detected_legacy_output": False,
            "migrated_legacy_output": False,
            "preserved_a_traces": False,
        }

    preserve_a = _legacy_output_matches_current_run(
        existing_manifest=existing_manifest,
        workload_path=workload_path,
        workload_name=workload_name,
        arch_name=arch_name,
        arch_path=arch_path,
        dimensions=dimensions,
        levels=levels,
        all_streaming_false=all_streaming_false,
        direct_c_search=direct_c_search,
        direct_d_search=direct_d_search,
        direct_l4_search=direct_l4_search,
        output_dim_constraint=output_dim_constraint,
    )
    if preserve_a:
        _clear_space_dir(output_dir / "spaces" / "a", preserve_trace_yaml=True)
    else:
        _clear_space_dir(output_dir / "spaces" / "a")
    for space_id in ("b", "c", "d"):
        _clear_space_dir(output_dir / "spaces" / space_id)
    for root_artifact in ("summary.json",):
        artifact_path = output_dir / root_artifact
        if artifact_path.exists():
            artifact_path.unlink()

    return {
        "detected_legacy_output": True,
        "migrated_legacy_output": True,
        "preserved_a_traces": bool(preserve_a),
    }


def _is_legacy_output_root(
    output_dir: Path, existing_manifest: Mapping[str, Any] | None
) -> bool:
    if existing_manifest is None:
        return any(
            _has_existing_search_artifacts(output_dir / "spaces" / space_id)
            for space_id in SPACE_ORDER
        )
    if (
        existing_manifest.get("search_semantics") == SEARCH_SEMANTICS
        and existing_manifest.get("search_semantics_version")
        == SEARCH_SEMANTICS_VERSION
    ):
        return False
    return True


def _legacy_output_matches_current_run(
    *,
    existing_manifest: Mapping[str, Any] | None,
    workload_path: Path,
    workload_name: str,
    arch_name: str,
    arch_path: Path,
    dimensions: Sequence[str] | None,
    levels: Sequence[int] | None,
    all_streaming_false: bool,
    direct_c_search: bool,
    direct_d_search: bool,
    direct_l4_search: bool,
    output_dim_constraint: OutputDimensionFactorizationConstraint | None,
) -> bool:
    if existing_manifest is None:
        return False
    current_payload = {
        "workload_path": str(workload_path),
        "workload_name": str(workload_name),
        "arch_name": str(arch_name),
        "arch_file": str(arch_path),
        "dimensions": list(dimensions) if dimensions is not None else None,
        "levels": [int(level) for level in levels] if levels is not None else None,
        "all_streaming_false": bool(all_streaming_false),
        "direct_c_search": bool(direct_c_search),
        "direct_d_search": bool(direct_d_search),
        "direct_l4_search": bool(direct_l4_search),
        "output_dim_constraint": _serialize_output_dim_constraint(
            output_dim_constraint
        ),
    }
    for field in _ROOT_COMPARE_FIELDS:
        existing_value = _normalize_root_compare_value(
            field, existing_manifest.get(field)
        )
        if existing_value != current_payload[field]:
            return False
    return True


def _validate_existing_output_root(
    *,
    existing_manifest: Mapping[str, Any] | None,
    workload_path: Path,
    workload_name: str,
    arch_name: str,
    arch_path: Path,
    dimensions: Sequence[str] | None,
    levels: Sequence[int] | None,
    all_streaming_false: bool,
    direct_c_search: bool,
    direct_d_search: bool,
    direct_l4_search: bool,
    output_dim_constraint: OutputDimensionFactorizationConstraint | None,
) -> None:
    if existing_manifest is None:
        return

    current_payload = {
        "workload_path": str(workload_path),
        "workload_name": str(workload_name),
        "arch_name": str(arch_name),
        "arch_file": str(arch_path),
        "dimensions": list(dimensions) if dimensions is not None else None,
        "levels": [int(level) for level in levels] if levels is not None else None,
        "all_streaming_false": bool(all_streaming_false),
        "direct_c_search": bool(direct_c_search),
        "direct_d_search": bool(direct_d_search),
        "direct_l4_search": bool(direct_l4_search),
        "output_dim_constraint": _serialize_output_dim_constraint(
            output_dim_constraint
        ),
    }
    for field in _ROOT_COMPARE_FIELDS:
        existing_value = _normalize_root_compare_value(
            field, existing_manifest.get(field)
        )
        if existing_value == current_payload[field]:
            continue
        raise ValueError(
            "Existing search output is incompatible with the current workload, "
            "architecture, or search mode."
        )


def _normalize_root_compare_value(field: str, value: Any) -> Any:
    if field in {"all_streaming_false", "direct_c_search", "direct_d_search", "direct_l4_search"} and value is None:
        return False
    return value


def _clear_space_dir(space_dir: Path, *, preserve_trace_yaml: bool = False) -> None:
    if not space_dir.exists():
        return
    if not preserve_trace_yaml:
        shutil.rmtree(space_dir)
        return

    for child in tuple(space_dir.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
            continue
        if child.is_file() and _TRACE_NAME_RE.match(child.name):
            continue
        child.unlink()


def _stage_target_total(
    executed_search_spaces: Sequence[str],
    num_traces_increment: int,
    space_id: str,
) -> int:
    try:
        stage_index = tuple(executed_search_spaces).index(space_id)
    except ValueError as exc:
        raise ValueError(
            f"Search space '{space_id}' is not present in the executed search plan."
        ) from exc
    return int(num_traces_increment) * (stage_index + 1)


def _import_missing_predecessor_traces(
    *,
    source_dir: Path,
    dest_dir: Path,
    prepared,
    arch: Arch,
    config: RandomSearchConfig,
    resume: bool,
) -> int:
    if not source_dir.is_dir():
        raise ValueError(f"Previous search-space output is missing: '{source_dir}'.")
    if not resume and _has_existing_search_artifacts(dest_dir):
        raise ValueError(
            f"Output directory '{dest_dir}' already contains search artifacts; "
            "remove them or rerun without --no-resume."
        )

    source_trace_paths = _trace_paths(source_dir)
    if not source_trace_paths:
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    context = _prepare_workload_contexts((prepared.workload,), arch, config)[0]
    existing_keys = {
        _stage_dedup_key_for_workload(Workload.from_file(trace_path), context)
        for trace_path in _trace_paths(dest_dir)
    }
    next_trace_index = _next_trace_index(dest_dir)
    imported = 0
    for trace_path in source_trace_paths:
        source_workload = Workload.from_file(trace_path)
        dedup_key = _stage_dedup_key_for_workload(source_workload, context)
        if dedup_key in existing_keys:
            continue
        _write_yaml(
            dest_dir / f"trace_{next_trace_index:06d}.yaml", source_workload.to_dict()
        )
        existing_keys.add(dedup_key)
        next_trace_index += 1
        imported += 1
    return imported


def _derive_space_d_from_c(
    *,
    source_dir: Path,
    dest_dir: Path,
    arch: Arch,
    resume: bool,
) -> dict[str, Any]:
    if not source_dir.is_dir():
        raise ValueError(f"Search-space 'c' output is missing: '{source_dir}'.")
    if not resume and _has_existing_search_artifacts(dest_dir):
        raise ValueError(
            f"Output directory '{dest_dir}' already contains search artifacts; "
            "remove them or rerun without --no-resume."
        )

    source_summary_path = source_dir / "summary.json"
    source_summary = _read_json_object(source_summary_path)
    source_trace_paths = _trace_paths(source_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[str] = []
    source_names = {path.name for path in source_trace_paths}
    for trace_path in source_trace_paths:
        workload = Workload.from_file(trace_path)
        derived = prepare_workload_for_space(workload, arch, "d").workload
        out_path = dest_dir / trace_path.name
        _write_yaml(out_path, derived.to_dict())
        generated_files.append(str(out_path))

    for existing_trace in _trace_paths(dest_dir):
        if existing_trace.name not in source_names:
            existing_trace.unlink()
    for artifact_name in ("accepted.jsonl", "state.json"):
        artifact_path = dest_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()

    summary = {
        "version": 1,
        "status": "derived",
        "output_dir": str(dest_dir),
        "seed": source_summary.get("seed"),
        "resume_enabled": bool(resume),
        "existing_trace_count": 0,
        "attempts": 0,
        "accepted": 0,
        "duplicates": 0,
        "rejected": 0,
        "saved_total": len(source_trace_paths),
        "generated_files": generated_files,
        "source_space": "c",
        "source_output_dir": str(source_dir),
    }
    _write_json(dest_dir / "summary.json", summary)
    return summary


def _resolve_arch_path(arch_name: str, arch_file: str | None) -> Path:
    if arch_file is not None:
        return _resolve_path(arch_file)
    try:
        relative = DEFAULT_ARCH_FILES[arch_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported architecture '{arch_name}'.") from exc
    return (PROJECT_ROOT / relative).resolve()


def _resolve_path(candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read JSON file '{path}'.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse JSON file '{path}'.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    return payload


def _read_optional_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json_object(path)


def _trace_paths(out_dir: Path) -> tuple[Path, ...]:
    indexed_paths: list[tuple[int, Path]] = []
    for file_path in out_dir.glob("trace_*.yaml"):
        match = _TRACE_NAME_RE.match(file_path.name)
        if match is None:
            continue
        indexed_paths.append((int(match.group(1)), file_path))
    indexed_paths.sort(key=lambda item: item[0])
    return tuple(path for _, path in indexed_paths)


def _stage_dedup_key_for_workload(workload: Workload, context) -> str:
    if all(workload.bounds.has_level(level) for level in context.levels):
        return _dedup_key_for_workload(workload, context)

    factors_by_dim = {
        dim: tuple(
            (
                workload.bounds.extent(dim, level=level)
                if workload.bounds.has_level(level)
                else 1
            )
            for level in context.levels
        )
        for dim in context.dimensions
    }
    sharing = workload.layout.sharing if workload.layout is not None else None
    return _build_dedup_key(
        workload_name=workload.name,
        compute=workload.compute,
        dimensions=context.dimensions,
        levels=context.levels,
        factors_by_dim=factors_by_dim,
        sharing_pu=sharing.pu if sharing is not None else None,
        sharing_system=sharing.system if sharing is not None else None,
        cache_enabled=workload.layout.cache if workload.layout is not None else None,
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    run(
        workload_file=args.workload,
        arch_name=args.arch,
        out_dir=args.outdir,
        num_traces=args.num_traces,
        spaces=args.spaces,
        arch_file=args.arch_file,
        seed=args.seed,
        max_attempts=args.max_attempts,
        workers=args.workers,
        batch_attempts=args.batch_attempts,
        dimensions=args.dimensions,
        levels=args.levels,
        show_progress=not args.no_progress,
        use_early_checks=not args.disable_early_checks,
        warmup_cache=not args.disable_cache_warmup,
        resume=not args.no_resume,
        all_streaming_false=args.all_streaming_false,
        direct_c_search=args.direct_c_search,
        direct_d_search=args.direct_d_search,
        direct_l4_search=args.direct_l4_search,
    )


if __name__ == "__main__":
    main()
