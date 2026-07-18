from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pnmax.cli import run_end_to_end_workload_space as _base_e2e
from pnmax.cli import (
    run_geometry_sweep_end_to_end_workload_space as _geometry_end_to_end,
)
from pnmax.cli import run_workload_space_pareto_eval as _base_eval
from pnmax.dse.pareto_export import pareto_front_nd
from pnmax.paths import repo_root, results_root

DEFAULT_INPUT_ROOT = results_root() / "geometry_sweep_end_to_end_workload_space"
DEFAULT_APPROX_MAX_POINTS = _geometry_end_to_end.DEFAULT_APPROX_MAX_POINTS
DEFAULT_APPROX_EPSILON = _geometry_end_to_end.DEFAULT_APPROX_EPSILON
_TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}$")
_WORKSPACE_ROOT = repo_root()
_REROOT_MARKERS: tuple[str, ...] = ("dbg", "data", "workloads", "results")


@dataclass(frozen=True)
class TokenTraceSource:
    token: str
    yaml_path: Path
    layer_indices: tuple[int, ...]
    mapping_dir: Path
    workload_paths: tuple[Path, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.layer_indices)


@dataclass(frozen=True)
class EndToEndModelInput:
    arch_name: str
    model_name: str
    input_root: Path
    run_dir: Path
    summary_path: Path
    baseline_geometry_id: str
    generated_workloads: tuple[_base_e2e.GeneratedWorkload, ...]
    token_sources: tuple[TokenTraceSource, ...]
    approx_max_points: int
    approx_epsilon: float
    top_summary: Mapping[str, Any]
    model_summary: Mapping[str, Any]


def resolve_end_to_end_model_input(
    *,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    input_run: str | Path | None,
    arch_name: str,
    model_name: str,
) -> EndToEndModelInput:
    canonical_arch_name = _base_e2e._canonical_arch_name(arch_name)
    resolved_input_root = _base_eval._resolve_path(input_root)
    run_dir = resolve_end_to_end_run_dir(
        input_root=resolved_input_root,
        input_run=input_run,
        arch_name=canonical_arch_name,
    )
    summary_path = (run_dir / "summary.json").resolve()
    top_summary = load_json_mapping(summary_path)
    models = top_summary.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"End-to-end summary is missing models metadata: {summary_path}")
    model_summary = models.get(model_name)
    if not isinstance(model_summary, dict):
        available = ", ".join(sorted(str(name) for name in models))
        raise ValueError(
            f"Model '{model_name}' was not found under {summary_path}. "
            f"Available models: {available}"
        )
    baseline_geometry_id = str(top_summary.get("baseline_geometry_id", "")).strip()
    if not baseline_geometry_id:
        raise ValueError(
            f"End-to-end summary is missing baseline_geometry_id: {summary_path}"
        )
    generated_workloads = load_generated_workloads(model_summary)
    token_sources = discover_token_trace_sources(
        model_summary=model_summary,
        baseline_geometry_id=baseline_geometry_id,
        generated_workloads=generated_workloads,
    )
    approx_max_points = int(
        top_summary.get("approx_max_points", DEFAULT_APPROX_MAX_POINTS)
    )
    approx_epsilon = float(top_summary.get("approx_epsilon", DEFAULT_APPROX_EPSILON))
    return EndToEndModelInput(
        arch_name=canonical_arch_name,
        model_name=str(model_name),
        input_root=resolved_input_root.resolve(),
        run_dir=run_dir.resolve(),
        summary_path=summary_path,
        baseline_geometry_id=baseline_geometry_id,
        generated_workloads=generated_workloads,
        token_sources=token_sources,
        approx_max_points=approx_max_points,
        approx_epsilon=approx_epsilon,
        top_summary=top_summary,
        model_summary=model_summary,
    )


def resolve_end_to_end_run_dir(
    *,
    input_root: Path,
    input_run: str | Path | None,
    arch_name: str,
) -> Path:
    arch_dir = (input_root / arch_name).resolve()
    if input_run is not None:
        direct = _base_eval._resolve_path(input_run)
        if _is_valid_end_to_end_run_dir(direct):
            return direct.resolve()
        relative = (arch_dir / str(input_run)).resolve()
        if _is_valid_end_to_end_run_dir(relative):
            return relative
        raise ValueError(
            f"End-to-end run directory was not found for {arch_name}: {input_run}"
        )

    if not arch_dir.is_dir():
        raise ValueError(f"Missing end-to-end input directory for {arch_name}: {arch_dir}")
    candidates = sorted(
        path.resolve()
        for path in arch_dir.iterdir()
        if _is_valid_end_to_end_run_dir(path)
    )
    if not candidates:
        raise ValueError(f"No end-to-end runs were found under {arch_dir}.")
    return candidates[-1]


def _is_valid_end_to_end_run_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and _TIMESTAMP_DIR_RE.match(path.name) is not None
        and (path / "summary.json").is_file()
    )


def load_generated_workloads(
    model_summary: Mapping[str, Any],
) -> tuple[_base_e2e.GeneratedWorkload, ...]:
    manifest_payload = model_summary.get("workloads_manifest")
    if isinstance(manifest_payload, dict):
        return _generated_workloads_from_entries(manifest_payload.get("unique_workloads"))

    manifest_json = model_summary.get("workloads_manifest_json")
    if manifest_json is not None:
        manifest_payload = load_json_mapping(resolve_workspace_path(manifest_json))
        return _generated_workloads_from_entries(manifest_payload.get("unique_workloads"))

    unique_workloads = model_summary.get("unique_workloads")
    return _generated_workloads_from_entries(unique_workloads)


def _generated_workloads_from_entries(
    raw_entries: Any,
) -> tuple[_base_e2e.GeneratedWorkload, ...]:
    if not isinstance(raw_entries, list):
        raise ValueError("End-to-end model metadata is missing unique_workloads.")
    generated: list[_base_e2e.GeneratedWorkload] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            raise ValueError("Expected mapping entries in unique_workloads.")
        token = str(item.get("token", "")).strip()
        if not token:
            raise ValueError("Generated workload entry is missing token.")
        layer_indices = tuple(int(value) for value in item.get("layer_indices", ()))
        generated.append(
            _base_e2e.GeneratedWorkload(
                token=token,
                op_type=str(item.get("op_type", "")),
                n=int(item.get("n", 0)),
                p=int(item.get("p", 0)),
                c=int(item.get("c", 0)),
                k=int(item.get("k", 0)),
                yaml_path=resolve_workspace_path(item.get("yaml_path", "")),
                layer_indices=layer_indices,
            )
        )
    if not generated:
        raise ValueError("No generated workloads were recovered from end-to-end metadata.")
    return tuple(generated)


def discover_token_trace_sources(
    *,
    model_summary: Mapping[str, Any],
    baseline_geometry_id: str,
    generated_workloads: Sequence[_base_e2e.GeneratedWorkload],
) -> tuple[TokenTraceSource, ...]:
    token_artifacts = model_summary.get("token_artifacts")
    if not isinstance(token_artifacts, dict):
        raise ValueError("End-to-end model metadata is missing token_artifacts.")
    sources: list[TokenTraceSource] = []
    for workload in generated_workloads:
        artifact = token_artifacts.get(workload.token)
        if not isinstance(artifact, dict):
            raise ValueError(
                f"Missing token_artifacts metadata for workload token '{workload.token}'."
            )
        search_dirs_by_geometry = artifact.get("search_dirs_by_geometry")
        if not isinstance(search_dirs_by_geometry, dict):
            raise ValueError(
                f"Missing search_dirs_by_geometry for workload token '{workload.token}'."
            )
        baseline_dir = search_dirs_by_geometry.get(baseline_geometry_id)
        if baseline_dir is None:
            raise ValueError(
                f"Missing baseline geometry '{baseline_geometry_id}' search output for "
                f"workload token '{workload.token}'."
            )
        mapping_dir = (resolve_workspace_path(baseline_dir) / "spaces" / "c").resolve()
        if not mapping_dir.is_dir():
            raise ValueError(
                f"Baseline token mapping directory does not exist for "
                f"'{workload.token}': {mapping_dir}"
            )
        workload_paths = _base_eval._collect_step_workload_paths(mapping_dir)
        if not workload_paths:
            raise ValueError(f"No mapping YAML files were found under {mapping_dir}.")
        sources.append(
            TokenTraceSource(
                token=workload.token,
                yaml_path=workload.yaml_path.resolve(),
                layer_indices=tuple(int(value) for value in workload.layer_indices),
                mapping_dir=mapping_dir,
                workload_paths=workload_paths,
            )
        )
    return tuple(sources)


def resolve_workspace_path(path_like: str | Path) -> Path:
    raw_path = str(path_like)
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path.resolve()

    candidates: list[Path] = []
    path_parts: tuple[str, ...]
    if path.is_absolute():
        path_parts = path.parts
    elif raw_path.startswith("/"):
        path_parts = tuple(part for part in raw_path.split("/") if part)
    else:
        path_parts = tuple()

    if path_parts:
        for marker in _REROOT_MARKERS:
            if marker not in path_parts:
                continue
            index = path_parts.index(marker)
            candidates.append((_WORKSPACE_ROOT / Path(*path_parts[index:])).resolve())
            if marker == "workloads":
                # Pre-artifact trees kept workloads/ at the repo root; the
                # artifact ships them under data/workloads/.
                candidates.append(
                    (_WORKSPACE_ROOT / "data" / Path(*path_parts[index:])).resolve()
                )
    else:
        candidates.append((_WORKSPACE_ROOT / path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    return path.resolve()


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_workspace_path(path)
    if not resolved_path.is_file():
        raise ValueError(f"JSON file not found: {resolved_path}")
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {resolved_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping JSON payload in {resolved_path}.")
    return payload


def evaluation_sort_key(row: Mapping[str, Any]) -> tuple[float, int, int, int, str]:
    return (
        float(row.get("weighted_cost", 0.0)),
        int(row["latency_cycles"]),
        int(row["mem_footprint_bytes"]),
        int(row["energy"]),
        str(row.get("state_hash", "")),
    )


def best_latency_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(row["latency_cycles"]),
        int(row["mem_footprint_bytes"]),
        int(row["energy"]),
        str(row.get("state_hash", "")),
    )


def build_token_frontier(
    *,
    rows: Sequence[Mapping[str, Any]],
    workload_token: str,
) -> tuple[dict[str, Any], ...]:
    deduped_rows = dedupe_success_rows(rows)
    frontier_rows = pareto_front_nd(
        deduped_rows,
        metrics=_base_e2e.COMBINED_METRICS,
    )
    return tuple(
        _base_e2e._decorate_frontier_component(row=row, workload_token=workload_token)
        for row in frontier_rows
    )


def dedupe_success_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    deduped: dict[str, dict[str, Any]] = {}
    fallback_rows: list[dict[str, Any]] = []
    for row in rows:
        state_hash = str(row.get("state_hash", "")).strip()
        candidate = dict(row)
        if not state_hash:
            fallback_rows.append(candidate)
            continue
        existing = deduped.get(state_hash)
        if existing is None or evaluation_sort_key(candidate) < evaluation_sort_key(
            existing
        ):
            deduped[state_hash] = candidate
    combined_rows = list(deduped.values()) + fallback_rows
    return tuple(sorted(combined_rows, key=evaluation_sort_key))


def compose_variant_frontier(
    *,
    generated_workloads: Sequence[_base_e2e.GeneratedWorkload],
    per_workload_frontiers: Mapping[str, Sequence[Mapping[str, Any]]],
    approx_max_points: int,
    approx_epsilon: float,
    variant_label: str,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    missing_tokens = [
        workload.token
        for workload in generated_workloads
        if not per_workload_frontiers.get(workload.token)
    ]
    if missing_tokens:
        formatted = ", ".join(sorted(missing_tokens))
        raise RuntimeError(
            f"Cannot compose end-to-end frontier for {variant_label}; missing valid "
            f"per-workload frontiers for: {formatted}"
        )
    return _base_e2e._compose_full_frontier(
        generated_workloads=generated_workloads,
        per_workload_frontiers=per_workload_frontiers,
        approx_max_points=approx_max_points,
        approx_epsilon=approx_epsilon,
    )


def serialize_per_workload_frontier_rows(
    *,
    variant_fields: Mapping[str, Any],
    workload_token: str,
    frontier_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    sorted_rows = sorted(
        frontier_rows,
        key=lambda row: _base_e2e._row_sort_key(row, _base_e2e.COMBINED_METRICS),
    )
    return tuple(
        {
            "order": int(index),
            **dict(variant_fields),
            "workload_token": workload_token,
            "state_hash": str(row["state_hash"]),
            "latency_cycles": int(row["latency_cycles"]),
            "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
            "energy": int(row["energy"]),
            "weighted_cost": float(row.get("weighted_cost", 0.0)),
            "workload_path": str(row.get("workload_path", "")),
        }
        for index, row in enumerate(sorted_rows, start=1)
    )


def serialize_combined_frontier_rows(
    *,
    variant_fields: Mapping[str, Any],
    model_name: str,
    frontier_rows: Sequence[Mapping[str, Any]],
    generated_workloads: Sequence[_base_e2e.GeneratedWorkload],
) -> tuple[dict[str, Any], ...]:
    sorted_rows = sorted(frontier_rows, key=best_latency_sort_key)
    return tuple(
        {
            "order": int(index),
            **dict(variant_fields),
            "model_name": str(model_name),
            "state_hash": str(row["state_hash"]),
            "latency_cycles": int(row["latency_cycles"]),
            "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
            "energy": int(row["energy"]),
            "weighted_cost": float(row.get("weighted_cost", 0.0)),
            "component_count": len(tuple(row.get("components", ()))),
            "layer_assignments_json": json.dumps(
                _base_e2e._build_layer_assignments(
                    components=tuple(row.get("components", ())),
                    generated_workloads=generated_workloads,
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for index, row in enumerate(sorted_rows, start=1)
    )
