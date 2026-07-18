from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from pnmax.cli import run_workload_space_pareto_eval, run_workload_space_random_search
from pnmax.database import Arch
from pnmax.dse.combined_pareto_plot import (
    render_combined_frontier_plots,
    render_frontier_csv_plots,
)
from pnmax.dse.pareto_export import pareto_front_nd
from pnmax.paths import repo_root, results_root
from pnmax.seeding import default_seed

DEFAULT_MODEL_DIR = repo_root() / "data" / "nn_models" / "llama3-8B-1024"
DEFAULT_WORKLOAD_ROOT = repo_root() / "data" / "workloads" / "end_to_end"
DEFAULT_OUTPUT_ROOT = results_root() / "end_to_end_workload_space"
DEFAULT_ARCHES: tuple[str, ...] = ("upmem", "hbm_pim")
DEFAULT_COST_ARCH_FILES: dict[str, Path] = {
    "upmem": repo_root() / "data" / "archs" / "lowered" / "geometry_sweep/upmem/upmem__pu-8__hmat-16__vmat-64.yaml",
    "hbm_pim": repo_root() / "data" / "archs" / "lowered" / "geometry_sweep/hbm_pim/hbm_pim__pu-8__hmat-16__vmat-32.yaml",
}
DEFAULT_NUM_LAYERS = 9
DEFAULT_NUM_TRACES = 4096
DEFAULT_MAX_ATTEMPTS = 1_000_000_000
DEFAULT_SEARCH_WORKERS = 100
DEFAULT_EVAL_WORKERS = 100
DEFAULT_APPROX_MAX_POINTS = 100_000
DEFAULT_APPROX_EPSILON = 0.01
DEFAULT_PLOT_DPI = 200
SOURCE_SEARCH_SPACE_ID = "c"
SELECTED_FRONTIER_STEP_ID = "d"
GENERATED_PROBLEM_P = 1
INCOMPATIBLE_SEARCH_OUTPUT_ERROR = (
    "Existing search output is incompatible with the current workload, "
    "architecture, or search mode."
)
COMBINED_METRICS: tuple[str, ...] = (
    "latency_cycles",
    "mem_footprint_bytes",
    "energy",
)
PER_WORKLOAD_FRONTIER_FIELDNAMES: tuple[str, ...] = (
    "order",
    "arch_name",
    "workload_token",
    "state_hash",
    "latency_cycles",
    "mem_footprint_bytes",
    "energy",
    "weighted_cost",
    "workload_path",
)
COMBINED_FRONTIER_FIELDNAMES: tuple[str, ...] = (
    "order",
    "arch_name",
    "state_hash",
    "latency_cycles",
    "mem_footprint_bytes",
    "energy",
    "weighted_cost",
    "component_count",
    "layer_assignments_json",
)


@dataclass(frozen=True)
class LayerShape:
    layer_index: int
    op_type: str
    type_layer_index: int
    n: int
    p: int
    c: int
    k: int
    raw_row: tuple[str, ...]

    @property
    def token(self) -> str:
        prefix = "fc" if self.op_type == "FC" else "bmm"
        return f"{prefix}_n{self.n}_c{self.c}_k{self.k}"


@dataclass(frozen=True)
class GeneratedWorkload:
    token: str
    op_type: str
    n: int
    p: int
    c: int
    k: int
    yaml_path: Path
    layer_indices: tuple[int, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.layer_indices)


class _ProgressCoordinator:
    def __init__(self, *, enabled: bool, overall_total: int) -> None:
        self.enabled = bool(enabled)
        self._overall = None
        self._combo = None
        self._combo_key: tuple[str, str, str] | None = None
        if self.enabled:
            self._overall = tqdm(
                total=int(overall_total),
                desc="Overall progress",
                unit="step",
                position=0,
                leave=True,
            )

    def close(self) -> None:
        if self._combo is not None:
            self._combo.close()
            self._combo = None
        if self._overall is not None:
            self._overall.close()
            self._overall = None

    def advance_overall(self, delta: int = 1) -> None:
        if self._overall is None:
            return
        self._overall.update(int(delta))

    def update_search(
        self,
        *,
        arch_name: str,
        workload_token: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._update_combo(
            key=(arch_name, workload_token, "search"),
            desc=f"{arch_name}/{workload_token} search",
            total=int(payload["target_traces"]),
            completed=int(payload["legally_found"]),
            postfix=f"attempts={int(payload['attempts'])}",
        )

    def update_eval(
        self,
        *,
        arch_name: str,
        workload_token: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._update_combo(
            key=(arch_name, workload_token, "eval"),
            desc=f"{arch_name}/{workload_token} eval",
            total=int(payload["total"]),
            completed=int(payload["completed"]),
            postfix=f"completed={int(payload['completed'])}/{int(payload['total'])}",
        )

    def start_compose(self, *, arch_name: str) -> None:
        self._update_combo(
            key=(arch_name, "combined", "compose"),
            desc=f"{arch_name}/combined compose",
            total=1,
            completed=0,
            postfix="aggregating",
        )

    def finish_compose(self, *, arch_name: str) -> None:
        self._update_combo(
            key=(arch_name, "combined", "compose"),
            desc=f"{arch_name}/combined compose",
            total=1,
            completed=1,
            postfix="done",
        )

    def _update_combo(
        self,
        *,
        key: tuple[str, str, str],
        desc: str,
        total: int,
        completed: int,
        postfix: str,
    ) -> None:
        if not self.enabled:
            return

        normalized_total = max(1, int(total))
        normalized_completed = max(0, min(int(completed), normalized_total))
        if self._combo is None:
            self._combo = tqdm(
                total=normalized_total,
                desc=desc,
                unit="item",
                position=1,
                leave=False,
            )
            self._combo_key = None

        current_total = int(self._combo.total or 0)
        current_n = int(getattr(self._combo, "n", 0))
        if (
            self._combo_key != key
            or current_total != normalized_total
            or normalized_completed < current_n
        ):
            self._combo.reset(total=normalized_total)
            self._combo_key = key

        self._combo.set_description(desc)
        current_n = int(getattr(self._combo, "n", 0))
        if normalized_completed > current_n:
            self._combo.update(normalized_completed - current_n)
        self._combo.set_postfix_str(postfix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate end-to-end Llama workloads, run cache-enabled (d) "
            "workload-space search/eval, compose a combined 3-metric frontier, "
            "and plot it."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Model directory containing layer_params.csv.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="Number of leading layers to include from layer_params.csv.",
    )
    parser.add_argument(
        "--workload-root",
        type=str,
        default=str(DEFAULT_WORKLOAD_ROOT),
        help="Root directory for generated base workloads.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for end-to-end search/eval/frontier artifacts.",
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        default=list(DEFAULT_ARCHES),
        help='Architectures to run, e.g. "--arch upmem hbm_pim".',
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=DEFAULT_NUM_TRACES,
        help=(
            "Number of valid cache-enabled (d) traces to save per unique "
            "workload/arch pair."
        ),
    )
    parser.add_argument(
        "--search-workers",
        type=int,
        default=DEFAULT_SEARCH_WORKERS,
        help="Number of workload-space random-search workers.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=DEFAULT_EVAL_WORKERS,
        help="Number of workload-space Pareto-eval workers.",
    )
    parser.add_argument(
        "--approx-max-points",
        type=int,
        default=DEFAULT_APPROX_MAX_POINTS,
        help="Maximum exact intermediate frontier size before epsilon compression.",
    )
    parser.add_argument(
        "--approx-epsilon",
        type=float,
        default=DEFAULT_APPROX_EPSILON,
        help="Multiplicative epsilon used once approximation becomes necessary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
        help="Base seed for workload-space random search "
        "(default: --seed > PNMAX_SEED > 42).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum random-search attempts per workload/arch pair.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable overall and per-combination progress bars.",
    )
    mode_group.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Rerender saved combined Pareto plots from existing combined/<arch> "
            "artifacts without rerunning search, eval, or composition."
        ),
    )
    mode_group.add_argument(
        "--recompose-only",
        action="store_true",
        help=(
            "Rebuild per-workload and combined Pareto frontiers from saved "
            "evaluations.csv artifacts without rerunning search or evaluation."
        ),
    )
    return parser.parse_args()


def run(
    *,
    model_dir: str = str(DEFAULT_MODEL_DIR),
    num_layers: int = DEFAULT_NUM_LAYERS,
    workload_root: str = str(DEFAULT_WORKLOAD_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    arch_names: Sequence[str] = DEFAULT_ARCHES,
    num_traces: int = DEFAULT_NUM_TRACES,
    search_workers: int = DEFAULT_SEARCH_WORKERS,
    eval_workers: int = DEFAULT_EVAL_WORKERS,
    approx_max_points: int = DEFAULT_APPROX_MAX_POINTS,
    approx_epsilon: float = DEFAULT_APPROX_EPSILON,
    seed: int | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    plot_dpi: int = DEFAULT_PLOT_DPI,
    plot_show: bool = False,
    show_progress: bool = True,
    plot_only: bool = False,
    recompose_only: bool = False,
) -> dict[str, Any]:
    resolved_model_dir = Path(model_dir).resolve()
    canonical_arches = _normalize_arches(arch_names)
    model_name = resolved_model_dir.name
    pipeline_dir = Path(output_root).resolve() / model_name

    if plot_only and recompose_only:
        raise ValueError("plot_only and recompose_only cannot both be enabled.")
    if plot_only:
        return _run_plot_only(
            model_dir=resolved_model_dir,
            pipeline_dir=pipeline_dir,
            arch_names=canonical_arches,
            plot_dpi=plot_dpi,
            plot_show=plot_show,
        )
    if recompose_only:
        if approx_max_points <= 0:
            raise ValueError(
                f"approx_max_points must be positive, got {approx_max_points}."
            )
        if approx_epsilon <= 0:
            raise ValueError(f"approx_epsilon must be positive, got {approx_epsilon}.")
        return _run_recompose_only(
            model_dir=resolved_model_dir,
            pipeline_dir=pipeline_dir,
            arch_names=canonical_arches,
            approx_max_points=approx_max_points,
            approx_epsilon=approx_epsilon,
            plot_dpi=plot_dpi,
            plot_show=plot_show,
        )

    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")
    if num_traces <= 0:
        raise ValueError(f"num_traces must be positive, got {num_traces}.")
    if search_workers <= 0:
        raise ValueError(f"search_workers must be positive, got {search_workers}.")
    if eval_workers <= 0:
        raise ValueError(f"eval_workers must be positive, got {eval_workers}.")
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}.")
    if approx_max_points <= 0:
        raise ValueError(
            f"approx_max_points must be positive, got {approx_max_points}."
        )
    if approx_epsilon <= 0:
        raise ValueError(f"approx_epsilon must be positive, got {approx_epsilon}.")

    csv_path = resolved_model_dir / "layer_params.csv"
    if not csv_path.is_file():
        raise ValueError(f"Missing layer_params.csv under {resolved_model_dir}.")

    workloads_dir = Path(workload_root).resolve() / model_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    layer_shapes = _load_layer_shapes(csv_path=csv_path, num_layers=num_layers)
    generated_workloads = _write_generated_workloads(
        layer_shapes=layer_shapes,
        workloads_dir=workloads_dir,
    )
    workloads_manifest_path = workloads_dir / "manifest.json"
    workloads_manifest = _write_workload_manifest(
        path=workloads_manifest_path,
        csv_path=csv_path,
        model_dir=resolved_model_dir,
        layer_shapes=layer_shapes,
        generated_workloads=generated_workloads,
    )

    arch_summaries: dict[str, Any] = {}
    progress = _ProgressCoordinator(
        enabled=show_progress,
        overall_total=(len(canonical_arches) * len(generated_workloads))
        + len(canonical_arches),
    )
    try:
        for arch_name in canonical_arches:
            arch_summaries[arch_name] = _run_arch_pipeline(
                arch_name=arch_name,
                generated_workloads=generated_workloads,
                pipeline_dir=pipeline_dir,
                num_traces=num_traces,
                search_workers=search_workers,
                eval_workers=eval_workers,
                approx_max_points=approx_max_points,
                approx_epsilon=approx_epsilon,
                seed=seed,
                max_attempts=max_attempts,
                plot_dpi=plot_dpi,
                plot_show=plot_show,
                progress=progress,
            )
    finally:
        progress.close()

    summary = {
        "generated_at": _utcnow(),
        "model_dir": str(resolved_model_dir),
        "model_name": model_name,
        "source_csv": str(csv_path),
        "selected_step_id": SELECTED_FRONTIER_STEP_ID,
        "num_layers": int(num_layers),
        "arch_names": list(canonical_arches),
        "num_traces": int(num_traces),
        "search_workers": int(search_workers),
        "eval_workers": int(eval_workers),
        "approx_max_points": int(approx_max_points),
        "approx_epsilon": float(approx_epsilon),
        "seed": seed,
        "max_attempts": int(max_attempts),
        "workloads_dir": str(workloads_dir),
        "workloads_manifest_json": str(workloads_manifest_path),
        "output_dir": str(pipeline_dir),
        "unique_workloads": [
            {
                "token": workload.token,
                "yaml_path": str(workload.yaml_path),
                "layer_indices": list(workload.layer_indices),
                "multiplicity": workload.multiplicity,
            }
            for workload in generated_workloads
        ],
        "arches": arch_summaries,
        "workload_manifest": workloads_manifest,
    }
    summary_path = pipeline_dir / "summary.json"
    _write_json(summary_path, summary)
    return {**summary, "summary_json": str(summary_path)}


def main() -> int:
    args = parse_args()
    run(
        model_dir=args.model_dir,
        num_layers=args.num_layers,
        workload_root=args.workload_root,
        output_root=args.output_root,
        arch_names=args.arch,
        num_traces=args.num_traces,
        search_workers=args.search_workers,
        eval_workers=args.eval_workers,
        approx_max_points=args.approx_max_points,
        approx_epsilon=args.approx_epsilon,
        seed=args.seed,
        max_attempts=args.max_attempts,
        show_progress=not args.no_progress,
        plot_only=args.plot_only,
        recompose_only=args.recompose_only,
    )
    return 0


def _run_plot_only(
    *,
    model_dir: Path,
    pipeline_dir: Path,
    arch_names: Sequence[str],
    plot_dpi: int,
    plot_show: bool,
) -> dict[str, Any]:
    arch_summaries: dict[str, Any] = {}
    combined_dirs: dict[str, Path] = {}
    top_summary_path = pipeline_dir / "summary.json"

    for arch_name in arch_names:
        combined_dir = pipeline_dir / "combined" / arch_name
        combined_summary_path = combined_dir / "summary.json"
        combined_csv_path = combined_dir / "combined_frontier.csv"
        if not combined_summary_path.is_file():
            raise ValueError(
                f"Plot-only mode requires saved combined summary for '{arch_name}': "
                f"{combined_summary_path}"
            )
        if not combined_csv_path.is_file():
            raise ValueError(
                f"Plot-only mode requires saved combined frontier CSV for '{arch_name}': "
                f"{combined_csv_path}"
            )
        combined_dirs[arch_name] = combined_dir

    top_summary = _try_load_json(top_summary_path)
    saved_generated_workloads: tuple[GeneratedWorkload, ...] | None = None
    if top_summary is not None:
        try:
            saved_generated_workloads = _load_saved_generated_workloads(
                top_summary=top_summary
            )
        except ValueError:
            saved_generated_workloads = None
    for arch_name in arch_names:
        combined_dir = combined_dirs[arch_name]
        combined_summary_path = combined_dir / "summary.json"
        arch_summary = _load_json_mapping(combined_summary_path)
        plot_payload = _render_arch_plot_outputs(
            arch_name=arch_name,
            combined_dir=combined_dir,
            combined_summary=arch_summary,
            generated_workloads=saved_generated_workloads,
            plot_dpi=plot_dpi,
            plot_show=plot_show,
        )
        arch_summary.update(plot_payload)
        _write_json(combined_summary_path, arch_summary)
        if top_summary is not None:
            arches_payload = top_summary.setdefault("arches", {})
            if not isinstance(arches_payload, dict):
                raise ValueError(
                    f"Expected 'arches' to be a mapping in {top_summary_path}."
                )
            arch_payload = arches_payload.setdefault(arch_name, {})
            if not isinstance(arch_payload, dict):
                raise ValueError(
                    f"Expected arches['{arch_name}'] to be a mapping in "
                    f"{top_summary_path}."
                )
            arch_payload["plot_outputs"] = dict(plot_payload["plot_outputs"])
            arch_payload["per_workload_plot_dir"] = str(
                plot_payload["per_workload_plot_dir"]
            )
            arch_payload["per_workload_plot_outputs"] = dict(
                plot_payload["per_workload_plot_outputs"]
            )
        arch_summaries[arch_name] = {
            **arch_summary,
            "summary_json": str(combined_summary_path.resolve()),
            "output_dir": str(combined_dir.resolve()),
        }

    if top_summary is not None:
        top_summary["selected_step_id"] = str(
            top_summary.get("selected_step_id") or SELECTED_FRONTIER_STEP_ID
        )
        _write_json(top_summary_path, top_summary)

    summary: dict[str, Any] = {
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "output_dir": str(pipeline_dir.resolve()),
        "selected_step_id": str(
            top_summary.get("selected_step_id") if top_summary is not None else None
            or SELECTED_FRONTIER_STEP_ID
        ),
        "arch_names": list(arch_names),
        "plot_only": True,
        "arches": arch_summaries,
    }
    if top_summary_path.is_file():
        summary["summary_json"] = str(top_summary_path.resolve())
    return summary


def _run_recompose_only(
    *,
    model_dir: Path,
    pipeline_dir: Path,
    arch_names: Sequence[str],
    approx_max_points: int,
    approx_epsilon: float,
    plot_dpi: int,
    plot_show: bool,
) -> dict[str, Any]:
    top_summary_path = pipeline_dir / "summary.json"
    top_summary = _try_load_json(top_summary_path)
    if top_summary is None:
        raise ValueError(
            "Recompose-only mode requires saved pipeline summary: "
            f"{top_summary_path}"
        )

    generated_workloads = _load_saved_generated_workloads(top_summary=top_summary)
    arches_payload = top_summary.get("arches")
    if not isinstance(arches_payload, dict):
        raise ValueError(f"Expected 'arches' to be a mapping in {top_summary_path}.")

    arch_summaries: dict[str, Any] = {}
    for arch_name in arch_names:
        existing_arch_payload = arches_payload.get(arch_name)
        if not isinstance(existing_arch_payload, dict):
            raise ValueError(
                "Recompose-only mode requires saved arch summary metadata for "
                f"'{arch_name}' in {top_summary_path}."
            )
        arch_summary = _run_recompose_arch(
            arch_name=arch_name,
            generated_workloads=generated_workloads,
            pipeline_dir=pipeline_dir,
            existing_arch_payload=existing_arch_payload,
            approx_max_points=approx_max_points,
            approx_epsilon=approx_epsilon,
            plot_dpi=plot_dpi,
            plot_show=plot_show,
        )
        arches_payload[arch_name] = {
            **existing_arch_payload,
            **{
                key: value
                for key, value in arch_summary.items()
                if key not in {"summary_json", "output_dir"}
            },
            "summary_json": arch_summary["summary_json"],
            "output_dir": arch_summary["output_dir"],
        }
        arch_summaries[arch_name] = arch_summary

    top_summary["approx_max_points"] = int(approx_max_points)
    top_summary["approx_epsilon"] = float(approx_epsilon)
    top_summary["recomposed_at"] = _utcnow()
    _write_json(top_summary_path, top_summary)

    return {
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "output_dir": str(pipeline_dir.resolve()),
        "arch_names": list(arch_names),
        "recompose_only": True,
        "arches": arch_summaries,
        "summary_json": str(top_summary_path.resolve()),
    }


def _run_recompose_arch(
    *,
    arch_name: str,
    generated_workloads: Sequence[GeneratedWorkload],
    pipeline_dir: Path,
    existing_arch_payload: Mapping[str, Any],
    approx_max_points: int,
    approx_epsilon: float,
    plot_dpi: int,
    plot_show: bool,
) -> dict[str, Any]:
    combined_dir = pipeline_dir / "combined" / arch_name
    per_workload_frontier_dir = combined_dir / "per_workload_frontier_3d"
    per_workload_frontier_dir.mkdir(parents=True, exist_ok=True)
    combined_summary_path = combined_dir / "summary.json"

    existing_combined_summary = _load_saved_arch_summary(
        combined_summary_path=combined_summary_path,
        existing_arch_payload=existing_arch_payload,
    )
    saved_eval_artifacts = _load_saved_eval_artifacts(
        arch_name=arch_name,
        existing_arch_payload=existing_arch_payload,
        existing_combined_summary=existing_combined_summary,
    )

    per_workload_frontiers: dict[str, tuple[dict[str, Any], ...]] = {}
    for workload in generated_workloads:
        evaluations_csv = _resolve_saved_evaluations_csv(
            arch_name=arch_name,
            workload_token=workload.token,
            saved_eval_artifacts=saved_eval_artifacts,
        )
        frontier_rows = _load_per_workload_frontier_rows(evaluations_csv)
        if not frontier_rows:
            raise ValueError(
                "No valid cache-enabled evaluation rows were found for "
                f"{arch_name}/{workload.token} under {evaluations_csv.parent}."
            )
        per_workload_frontier = tuple(
            _decorate_frontier_component(row=row, workload_token=workload.token)
            for row in pareto_front_nd(frontier_rows, metrics=COMBINED_METRICS)
        )
        if not per_workload_frontier:
            raise ValueError(
                f"Per-workload frontier is empty for {arch_name}/{workload.token}."
            )
        per_workload_frontiers[workload.token] = per_workload_frontier
        per_workload_frontier_csv = per_workload_frontier_dir / f"{workload.token}.csv"
        _write_csv(
            per_workload_frontier_csv,
            fieldnames=PER_WORKLOAD_FRONTIER_FIELDNAMES,
            rows=_serialize_per_workload_frontier_rows(
                arch_name=arch_name,
                workload_token=workload.token,
                frontier_rows=per_workload_frontier,
            ),
        )

    combined_rows, approximate = _compose_full_frontier(
        generated_workloads=generated_workloads,
        per_workload_frontiers=per_workload_frontiers,
        approx_max_points=approx_max_points,
        approx_epsilon=approx_epsilon,
    )
    if not combined_rows:
        raise ValueError(f"Combined frontier is empty for architecture '{arch_name}'.")

    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = combined_dir / "combined_frontier.csv"
    combined_rows_serialized = _serialize_combined_rows(
        arch_name=arch_name,
        frontier_rows=combined_rows,
        generated_workloads=generated_workloads,
    )
    _write_csv(
        combined_csv,
        fieldnames=COMBINED_FRONTIER_FIELDNAMES,
        rows=combined_rows_serialized,
    )

    summary = {
        "generated_at": _utcnow(),
        "arch_name": arch_name,
        "plot_title": str(
            existing_combined_summary.get("plot_title")
            or existing_arch_payload.get("plot_title")
            or f"{arch_name} combined frontier"
        ),
        "selected_step_id": str(
            existing_combined_summary.get("selected_step_id")
            or existing_arch_payload.get("selected_step_id")
            or SELECTED_FRONTIER_STEP_ID
        ),
        "combined_frontier_csv": str(combined_csv.resolve()),
        "per_workload_frontier_dir": str(per_workload_frontier_dir.resolve()),
        "per_workload_frontier_csvs": {
            workload.token: str(
                (per_workload_frontier_dir / f"{workload.token}.csv").resolve()
            )
            for workload in generated_workloads
        },
        "approximate": bool(approximate),
        "approx_epsilon": float(approx_epsilon),
        "approx_max_points": int(approx_max_points),
        "combined_frontier_points": len(combined_rows_serialized),
        "search_artifacts": dict(existing_combined_summary.get("search_artifacts", {})),
        "eval_artifacts": dict(saved_eval_artifacts),
    }
    _write_json(combined_summary_path, summary)
    plot_payload = _render_arch_plot_outputs(
        arch_name=arch_name,
        combined_dir=combined_dir,
        combined_summary=summary,
        generated_workloads=generated_workloads,
        plot_dpi=plot_dpi,
        plot_show=plot_show,
    )
    summary.update(plot_payload)
    _write_json(combined_summary_path, summary)
    return {
        **summary,
        "summary_json": str(combined_summary_path.resolve()),
        "output_dir": str(combined_dir.resolve()),
    }


def _load_saved_generated_workloads(
    *, top_summary: Mapping[str, Any]
) -> tuple[GeneratedWorkload, ...]:
    workload_manifest = top_summary.get("workload_manifest")
    if isinstance(workload_manifest, dict):
        return _generated_workloads_from_manifest_payload(workload_manifest)

    workloads_manifest_json = top_summary.get("workloads_manifest_json")
    if workloads_manifest_json is not None:
        manifest_payload = _load_json_mapping(Path(str(workloads_manifest_json)).resolve())
        return _generated_workloads_from_manifest_payload(manifest_payload)

    unique_workloads = top_summary.get("unique_workloads")
    if not isinstance(unique_workloads, list):
        raise ValueError(
            "Recompose-only mode requires saved workload manifest metadata in the "
            "pipeline summary."
        )
    generated: list[GeneratedWorkload] = []
    for item in unique_workloads:
        if not isinstance(item, dict):
            raise ValueError("Expected mapping entries in saved unique_workloads.")
        layer_indices = tuple(int(value) for value in item.get("layer_indices", ()))
        token = str(item.get("token", "")).strip()
        yaml_path = Path(str(item.get("yaml_path", ""))).resolve()
        if not token:
            raise ValueError("Saved unique_workloads entry is missing token.")
        generated.append(
            GeneratedWorkload(
                token=token,
                op_type=str(item.get("op_type", "")),
                n=int(item.get("n", 0)),
                p=int(item.get("p", 0)),
                c=int(item.get("c", 0)),
                k=int(item.get("k", 0)),
                yaml_path=yaml_path,
                layer_indices=layer_indices,
            )
        )
    return tuple(generated)


def _generated_workloads_from_manifest_payload(
    manifest_payload: Mapping[str, Any],
) -> tuple[GeneratedWorkload, ...]:
    unique_workloads = manifest_payload.get("unique_workloads")
    if not isinstance(unique_workloads, list):
        raise ValueError("Saved workload manifest is missing unique_workloads.")
    generated: list[GeneratedWorkload] = []
    for item in unique_workloads:
        if not isinstance(item, dict):
            raise ValueError("Expected mapping entries in workload manifest.")
        token = str(item.get("token", "")).strip()
        yaml_path = Path(str(item.get("yaml_path", ""))).resolve()
        layer_indices = tuple(int(value) for value in item.get("layer_indices", ()))
        if not token:
            raise ValueError("Saved workload manifest entry is missing token.")
        generated.append(
            GeneratedWorkload(
                token=token,
                op_type=str(item.get("op_type", "")),
                n=int(item.get("n", 0)),
                p=int(item.get("p", 0)),
                c=int(item.get("c", 0)),
                k=int(item.get("k", 0)),
                yaml_path=yaml_path,
                layer_indices=layer_indices,
            )
        )
    return tuple(generated)


def _render_arch_plot_outputs(
    *,
    arch_name: str,
    combined_dir: Path,
    combined_summary: Mapping[str, Any],
    generated_workloads: Sequence[GeneratedWorkload] | None,
    plot_dpi: int,
    plot_show: bool,
) -> dict[str, Any]:
    plot_outputs = render_combined_frontier_plots(
        combined_dir,
        dpi=plot_dpi,
        show=plot_show,
    )
    workload_tokens, layer_indices_by_token = _resolve_plot_workload_metadata(
        combined_dir=combined_dir,
        combined_summary=combined_summary,
        generated_workloads=generated_workloads,
    )
    per_workload_frontier_csvs = _resolve_current_per_workload_frontier_csvs(
        combined_dir=combined_dir,
        combined_summary=combined_summary,
        workload_tokens=workload_tokens,
    )
    per_workload_plot_dir = (combined_dir / "per_workload_plots").resolve()
    per_workload_plot_outputs: dict[str, dict[str, str]] = {}
    for workload_token in workload_tokens:
        per_workload_plot_outputs[workload_token] = render_frontier_csv_plots(
            per_workload_frontier_csvs[workload_token],
            title=_build_per_workload_plot_title(
                arch_name=arch_name,
                workload_token=workload_token,
                layer_indices=layer_indices_by_token[workload_token],
            ),
            output_base=per_workload_plot_dir / workload_token / "pareto.pdf",
            dpi=plot_dpi,
            show=plot_show,
        )
    return {
        "plot_outputs": dict(plot_outputs),
        "per_workload_plot_dir": str(per_workload_plot_dir),
        "per_workload_plot_outputs": per_workload_plot_outputs,
    }


def _resolve_plot_workload_metadata(
    *,
    combined_dir: Path,
    combined_summary: Mapping[str, Any],
    generated_workloads: Sequence[GeneratedWorkload] | None,
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    if generated_workloads:
        workload_tokens = tuple(workload.token for workload in generated_workloads)
        layer_indices_by_token = {
            workload.token: tuple(int(index) for index in workload.layer_indices)
            for workload in generated_workloads
        }
        return workload_tokens, layer_indices_by_token

    per_workload_frontier_csvs = combined_summary.get("per_workload_frontier_csvs")
    if not isinstance(per_workload_frontier_csvs, dict) or not per_workload_frontier_csvs:
        raise ValueError(
            "Saved combined summary is missing per_workload_frontier_csvs needed "
            "for per-workload plot refresh."
        )

    combined_csv_path = _resolve_combined_frontier_csv_path(
        combined_dir=combined_dir,
        combined_summary=combined_summary,
    )
    layer_indices_by_token = _load_layer_indices_by_token_from_combined_frontier_csv(
        combined_csv_path
    )
    workload_tokens: list[str] = []
    resolved_layers: dict[str, tuple[int, ...]] = {}
    for workload_token in per_workload_frontier_csvs:
        token = str(workload_token).strip()
        if not token:
            continue
        layer_indices = layer_indices_by_token.get(token)
        if layer_indices is None:
            raise ValueError(
                "Saved combined frontier is missing layer assignments for "
                f"workload token '{token}' in {combined_csv_path}."
            )
        workload_tokens.append(token)
        resolved_layers[token] = layer_indices
    if not workload_tokens:
        raise ValueError(
            "Saved combined summary does not contain any current workload tokens "
            "for per-workload plot refresh."
        )
    return tuple(workload_tokens), resolved_layers


def _resolve_current_per_workload_frontier_csvs(
    *,
    combined_dir: Path,
    combined_summary: Mapping[str, Any],
    workload_tokens: Sequence[str],
) -> dict[str, Path]:
    per_workload_frontier_csvs = combined_summary.get("per_workload_frontier_csvs")
    per_workload_frontier_dir = combined_summary.get("per_workload_frontier_dir")
    frontier_dir = (
        Path(str(per_workload_frontier_dir)).resolve()
        if per_workload_frontier_dir is not None
        else (combined_dir / "per_workload_frontier_3d").resolve()
    )

    resolved: dict[str, Path] = {}
    for workload_token in workload_tokens:
        csv_path = None
        if isinstance(per_workload_frontier_csvs, dict):
            raw_path = per_workload_frontier_csvs.get(workload_token)
            if raw_path is not None:
                csv_path = Path(str(raw_path)).resolve()
        if csv_path is None:
            csv_path = frontier_dir / f"{workload_token}.csv"
        if not csv_path.is_file():
            raise ValueError(
                "Missing saved per-workload frontier CSV for "
                f"{workload_token}: {csv_path}"
            )
        resolved[workload_token] = csv_path
    return resolved


def _resolve_combined_frontier_csv_path(
    *,
    combined_dir: Path,
    combined_summary: Mapping[str, Any],
) -> Path:
    combined_frontier_csv = combined_summary.get("combined_frontier_csv")
    if combined_frontier_csv is not None:
        return Path(str(combined_frontier_csv)).resolve()
    return (combined_dir / "combined_frontier.csv").resolve()


def _load_layer_indices_by_token_from_combined_frontier_csv(
    combined_frontier_csv: Path,
) -> dict[str, tuple[int, ...]]:
    if not combined_frontier_csv.is_file():
        raise ValueError(f"Missing combined frontier CSV: {combined_frontier_csv}")

    layer_indices_by_token: dict[str, set[int]] = {}
    with combined_frontier_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            assignments_json = str(row.get("layer_assignments_json", "")).strip()
            if not assignments_json:
                continue
            try:
                assignments = json.loads(assignments_json)
            except json.JSONDecodeError:
                continue
            if not isinstance(assignments, list):
                continue
            for assignment in assignments:
                if not isinstance(assignment, dict):
                    continue
                workload_token = str(assignment.get("workload_token", "")).strip()
                if not workload_token:
                    continue
                try:
                    layer_index = int(assignment["layer_index"])
                except (KeyError, TypeError, ValueError):
                    continue
                layer_indices_by_token.setdefault(workload_token, set()).add(
                    layer_index
                )
    return {
        workload_token: tuple(sorted(layer_indices))
        for workload_token, layer_indices in layer_indices_by_token.items()
    }


def _build_per_workload_plot_title(
    *,
    arch_name: str,
    workload_token: str,
    layer_indices: Sequence[int],
) -> str:
    layer_label = ",".join(str(int(index)) for index in layer_indices)
    return f"{arch_name} {workload_token} [layers {layer_label}]"


def _load_saved_arch_summary(
    *,
    combined_summary_path: Path,
    existing_arch_payload: Mapping[str, Any],
) -> dict[str, Any]:
    summary_json_path = existing_arch_payload.get("summary_json")
    if summary_json_path is not None:
        candidate_path = Path(str(summary_json_path)).resolve()
        if candidate_path.is_file():
            return _load_json_mapping(candidate_path)
    if combined_summary_path.is_file():
        return _load_json_mapping(combined_summary_path)
    return {}


def _load_saved_eval_artifacts(
    *,
    arch_name: str,
    existing_arch_payload: Mapping[str, Any],
    existing_combined_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    eval_artifacts = existing_arch_payload.get("eval_artifacts")
    if not isinstance(eval_artifacts, dict):
        eval_artifacts = existing_combined_summary.get("eval_artifacts")
    if not isinstance(eval_artifacts, dict):
        raise ValueError(
            "Recompose-only mode requires saved eval_artifacts for architecture "
            f"'{arch_name}'."
        )
    return eval_artifacts


def _resolve_saved_evaluations_csv(
    *,
    arch_name: str,
    workload_token: str,
    saved_eval_artifacts: Mapping[str, Any],
) -> Path:
    artifact = saved_eval_artifacts.get(workload_token)
    if not isinstance(artifact, Mapping):
        raise ValueError(
            "Recompose-only mode requires saved evaluation metadata for "
            f"{arch_name}/{workload_token}."
        )
    evaluations_csv = artifact.get("evaluations_csv")
    if evaluations_csv is not None:
        path = Path(str(evaluations_csv)).resolve()
    else:
        output_dir = artifact.get("output_dir")
        if output_dir is None:
            raise ValueError(
                "Saved evaluation metadata is missing evaluations_csv/output_dir for "
                f"{arch_name}/{workload_token}."
            )
        path = Path(str(output_dir)).resolve() / "evaluations.csv"
    if not path.is_file():
        raise ValueError(
            "Recompose-only mode requires saved evaluations CSV for "
            f"{arch_name}/{workload_token}: {path}"
        )
    return path


def _load_layer_shapes(*, csv_path: Path, num_layers: int) -> tuple[LayerShape, ...]:
    rows: list[LayerShape] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(rows) >= num_layers:
                break
            if not row:
                continue
            rows.append(_parse_layer_shape_row(row))
    if len(rows) < num_layers:
        raise ValueError(
            f"Requested {num_layers} layers from {csv_path}, but only found {len(rows)}."
        )
    return tuple(rows)


def _parse_layer_shape_row(row: Sequence[str]) -> LayerShape:
    if len(row) < 9:
        raise ValueError(f"Expected at least 9 CSV fields, got {len(row)}: {row}")
    layer_index = int(row[0])
    raw_op_type = str(row[1]).strip()
    op_type = raw_op_type.upper()
    if op_type not in {"FC", "BATCH_MATMUL"}:
        raise ValueError(f"Unsupported operator type '{raw_op_type}' in row {row}.")
    return LayerShape(
        layer_index=layer_index,
        op_type=op_type,
        type_layer_index=int(row[2]),
        n=int(row[3]),
        p=int(row[5]),
        c=int(row[6]),
        k=int(row[8]),
        raw_row=tuple(str(item) for item in row),
    )


def _write_generated_workloads(
    *,
    layer_shapes: Sequence[LayerShape],
    workloads_dir: Path,
) -> tuple[GeneratedWorkload, ...]:
    workloads_dir.mkdir(parents=True, exist_ok=True)
    grouped_layers: dict[tuple[str, int, int, int], list[LayerShape]] = {}
    ordered_keys: list[tuple[str, int, int, int]] = []
    for layer_shape in layer_shapes:
        key = (
            layer_shape.op_type,
            layer_shape.n,
            layer_shape.c,
            layer_shape.k,
        )
        if key not in grouped_layers:
            grouped_layers[key] = []
            ordered_keys.append(key)
        grouped_layers[key].append(layer_shape)

    generated: list[GeneratedWorkload] = []
    for key in ordered_keys:
        exemplar = grouped_layers[key][0]
        token = exemplar.token
        yaml_path = workloads_dir / f"{token}.yaml"
        payload = _build_generated_workload_payload(exemplar)
        _write_text(yaml_path, _dump_yaml(payload))
        generated.append(
            GeneratedWorkload(
                token=token,
                op_type=exemplar.op_type,
                n=exemplar.n,
                p=GENERATED_PROBLEM_P,
                c=exemplar.c,
                k=exemplar.k,
                yaml_path=yaml_path.resolve(),
                layer_indices=tuple(
                    int(layer_shape.layer_index) for layer_shape in grouped_layers[key]
                ),
            )
        )
    return tuple(generated)


def _build_generated_workload_payload(layer_shape: LayerShape) -> dict[str, Any]:
    problem = {
        "N": int(layer_shape.n),
        "K": int(layer_shape.k),
        "P": GENERATED_PROBLEM_P,
        "Q": 1,
        "C": int(layer_shape.c),
        "R": 1,
        "S": 1,
    }
    full_level = {
        "N": int(layer_shape.n),
        "K": int(layer_shape.k),
        "P": GENERATED_PROBLEM_P,
        "Q": 1,
        "C": int(layer_shape.c),
        "R": 1,
        "S": 1,
    }
    bounds = {
        5: [{"N": 1}, {"K": 1}, {"P": 1}, {"Q": 1}, {"C": 1}, {"R": 1}, {"S": 1}],
        4: [{"N": 1}, {"K": 1}, {"P": 1}, {"Q": 1}, {"C": 1}, {"R": 1}, {"S": 1}],
        3: [{"N": 1}, {"K": 1}, {"P": 1}, {"Q": 1}, {"C": 1}, {"R": 1}, {"S": 1}],
        2: [{"N": 1}, {"K": 1}, {"P": 1}, {"Q": 1}, {"C": 1}, {"R": 1}, {"S": 1}],
        1: [{name: int(value)} for name, value in full_level.items()],
        0: [{"N": 1}, {"K": 1}, {"P": 1}, {"Q": 1}, {"C": 1}, {"R": 1}, {"S": 1}],
    }
    return {
        "Name": layer_shape.token,
        "Problem": problem,
        "Bound": bounds,
        "TensorAttr": {
            "In": "16b",
            "F": "16b",
            "Out": "16b",
        },
        "Compute": "Out[n,p,q,k] += In[n,p+r,q+s,c] * F[k,r,s,c]",
        "Layout-Pragma": {
            "cache": False,
            "bit": "bit-parallel",
            "streaming": {
                "In": True,
                "F": False,
                "Out": False,
            },
            "sharing": {
                "block": True,
                "PU": False,
                "system": 0,
            },
            "interleaving": True,
        },
    }


def _write_workload_manifest(
    *,
    path: Path,
    csv_path: Path,
    model_dir: Path,
    layer_shapes: Sequence[LayerShape],
    generated_workloads: Sequence[GeneratedWorkload],
) -> dict[str, Any]:
    payload = {
        "generated_at": _utcnow(),
        "model_dir": str(model_dir),
        "source_csv": str(csv_path),
        "num_layers": len(layer_shapes),
        "layers": [
            {
                "layer_index": int(layer_shape.layer_index),
                "op_type": layer_shape.op_type,
                "type_layer_index": int(layer_shape.type_layer_index),
                "n": int(layer_shape.n),
                "p": int(layer_shape.p),
                "c": int(layer_shape.c),
                "k": int(layer_shape.k),
                "workload_token": layer_shape.token,
                "raw_row": list(layer_shape.raw_row),
            }
            for layer_shape in layer_shapes
        ],
        "unique_workloads": [
            {
                "token": workload.token,
                "op_type": workload.op_type,
                "n": int(workload.n),
                "p": int(workload.p),
                "c": int(workload.c),
                "k": int(workload.k),
                "yaml_path": str(workload.yaml_path),
                "layer_indices": list(workload.layer_indices),
                "multiplicity": workload.multiplicity,
            }
            for workload in generated_workloads
        ],
        "multiplicities": {
            workload.token: int(workload.multiplicity)
            for workload in generated_workloads
        },
    }
    _write_json(path, payload)
    return payload


def _run_arch_pipeline(
    *,
    arch_name: str,
    generated_workloads: Sequence[GeneratedWorkload],
    pipeline_dir: Path,
    num_traces: int,
    search_workers: int,
    eval_workers: int,
    approx_max_points: int,
    approx_epsilon: float,
    seed: int | None,
    max_attempts: int,
    plot_dpi: int,
    plot_show: bool,
    progress: _ProgressCoordinator | None,
) -> dict[str, Any]:
    search_artifacts: dict[str, Any] = {}
    eval_artifacts: dict[str, Any] = {}
    per_workload_frontiers: dict[str, tuple[dict[str, Any], ...]] = {}
    cost_arch_file = _resolve_cost_arch_file(arch_name)

    combined_dir = pipeline_dir / "combined" / arch_name
    per_workload_frontier_dir = combined_dir / "per_workload_frontier_3d"
    per_workload_frontier_dir.mkdir(parents=True, exist_ok=True)

    for workload in generated_workloads:
        search_output_dir = (
            pipeline_dir / "random_search" / arch_name / workload.token
        ).resolve()
        search_progress_callback = None
        eval_progress_callback = None
        if progress is not None and progress.enabled:
            search_progress_callback = (
                lambda payload, arch_name=arch_name, workload_token=workload.token: (
                    progress.update_search(
                        arch_name=arch_name,
                        workload_token=workload_token,
                        payload=payload,
                    )
                )
            )
            eval_progress_callback = (
                lambda payload, arch_name=arch_name, workload_token=workload.token: (
                    progress.update_eval(
                        arch_name=arch_name,
                        workload_token=workload_token,
                        payload=payload,
                    )
                )
            )
        search_summary = _run_random_search_with_cleanup_on_incompatible_output(
            workload=workload,
            arch_name=arch_name,
            arch_file=cost_arch_file,
            search_output_dir=search_output_dir,
            num_traces=num_traces,
            seed=seed,
            max_attempts=max_attempts,
            search_workers=search_workers,
            progress_callback=search_progress_callback,
        )
        search_output_dir = Path(str(search_summary["output_dir"])).resolve()
        selected_step_dir = Path(str(search_summary["selected_step_dir"])).resolve()
        search_artifacts[workload.token] = {
            "output_dir": str(search_output_dir),
            "summary_json": str((search_output_dir / "summary.json").resolve()),
            "manifest_json": str((search_output_dir / "manifest.json").resolve()),
            "selected_step_id": str(search_summary["selected_step_id"]),
            "selected_step_dir": str(selected_step_dir),
            "space_c_dir": str((search_output_dir / "spaces" / "c").resolve()),
            "space_d_dir": str((search_output_dir / "spaces" / "d").resolve()),
        }

        eval_summary = run_workload_space_pareto_eval.run(
            workload_dirs=[str(selected_step_dir)],
            arch_name=arch_name,
            arch_file=str(cost_arch_file),
            output_root=str((pipeline_dir / "pareto_eval").resolve()),
            target=arch_name,
            workers=eval_workers,
            verbose=0,
            progress_callback=eval_progress_callback,
        )
        eval_output_dir = Path(str(eval_summary["output_dir"])).resolve()
        frontier_rows = _load_per_workload_frontier_rows(
            eval_output_dir / "evaluations.csv"
        )
        if not frontier_rows:
            raise ValueError(
                "No valid cache-enabled evaluation rows were found for "
                f"{arch_name}/{workload.token} under {eval_output_dir}."
            )
        per_workload_frontier = tuple(
            _decorate_frontier_component(row=row, workload_token=workload.token)
            for row in pareto_front_nd(frontier_rows, metrics=COMBINED_METRICS)
        )
        if not per_workload_frontier:
            raise ValueError(
                f"Per-workload frontier is empty for {arch_name}/{workload.token}."
            )
        per_workload_frontiers[workload.token] = per_workload_frontier
        per_workload_frontier_csv = per_workload_frontier_dir / f"{workload.token}.csv"
        _write_csv(
            per_workload_frontier_csv,
            fieldnames=PER_WORKLOAD_FRONTIER_FIELDNAMES,
            rows=_serialize_per_workload_frontier_rows(
                arch_name=arch_name,
                workload_token=workload.token,
                frontier_rows=per_workload_frontier,
            ),
        )
        eval_artifacts[workload.token] = {
            "output_dir": str(eval_output_dir),
            "summary_json": str((eval_output_dir / "summary.json").resolve()),
            "manifest_json": str((eval_output_dir / "manifest.json").resolve()),
            "evaluations_csv": str((eval_output_dir / "evaluations.csv").resolve()),
            "pareto_frontiers_csv": str(
                (eval_output_dir / "pareto_frontiers.csv").resolve()
            ),
            "per_workload_frontier_csv": str(per_workload_frontier_csv.resolve()),
            "per_workload_frontier_points": len(per_workload_frontier),
        }
        if progress is not None:
            progress.advance_overall()

    if progress is not None:
        progress.start_compose(arch_name=arch_name)
    combined_rows, approximate = _compose_full_frontier(
        generated_workloads=generated_workloads,
        per_workload_frontiers=per_workload_frontiers,
        approx_max_points=approx_max_points,
        approx_epsilon=approx_epsilon,
    )
    if not combined_rows:
        raise ValueError(f"Combined frontier is empty for architecture '{arch_name}'.")
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = combined_dir / "combined_frontier.csv"
    combined_summary_path = combined_dir / "summary.json"
    combined_rows_serialized = _serialize_combined_rows(
        arch_name=arch_name,
        frontier_rows=combined_rows,
        generated_workloads=generated_workloads,
    )
    _write_csv(
        combined_csv,
        fieldnames=COMBINED_FRONTIER_FIELDNAMES,
        rows=combined_rows_serialized,
    )
    summary = {
        "generated_at": _utcnow(),
        "arch_name": arch_name,
        "plot_title": f"{arch_name} combined frontier",
        "selected_step_id": SELECTED_FRONTIER_STEP_ID,
        "combined_frontier_csv": str(combined_csv.resolve()),
        "per_workload_frontier_dir": str(per_workload_frontier_dir.resolve()),
        "per_workload_frontier_csvs": {
            workload.token: str(
                (per_workload_frontier_dir / f"{workload.token}.csv").resolve()
            )
            for workload in generated_workloads
        },
        "approximate": bool(approximate),
        "approx_epsilon": float(approx_epsilon),
        "approx_max_points": int(approx_max_points),
        "combined_frontier_points": len(combined_rows_serialized),
        "search_artifacts": search_artifacts,
        "eval_artifacts": eval_artifacts,
    }
    _write_json(combined_summary_path, summary)
    plot_payload = _render_arch_plot_outputs(
        arch_name=arch_name,
        combined_dir=combined_dir,
        combined_summary=summary,
        generated_workloads=generated_workloads,
        plot_dpi=plot_dpi,
        plot_show=plot_show,
    )
    summary.update(plot_payload)
    _write_json(combined_summary_path, summary)
    if progress is not None:
        progress.finish_compose(arch_name=arch_name)
        progress.advance_overall()
    return {
        **summary,
        "summary_json": str(combined_summary_path.resolve()),
        "output_dir": str(combined_dir.resolve()),
    }


def _run_random_search_with_cleanup_on_incompatible_output(
    *,
    workload: GeneratedWorkload,
    arch_name: str,
    arch_file: Path,
    search_output_dir: Path,
    num_traces: int,
    seed: int | None,
    max_attempts: int,
    search_workers: int,
    progress_callback: Any,
) -> dict[str, Any]:
    kwargs = {
        "workload_file": str(workload.yaml_path),
        "arch_name": arch_name,
        "arch_file": str(arch_file),
        "out_dir": str(search_output_dir),
        "num_traces": num_traces,
        "spaces": (SOURCE_SEARCH_SPACE_ID,),
        "seed": seed,
        "max_attempts": max_attempts,
        "workers": search_workers,
        "batch_attempts": 1,
        "show_progress": False,
        "direct_c_search": True,
        "progress_callback": progress_callback,
    }
    try:
        search_summary = run_workload_space_random_search.run(**kwargs)
    except ValueError as exc:
        if str(exc) != INCOMPATIBLE_SEARCH_OUTPUT_ERROR:
            raise
        if search_output_dir.exists():
            shutil.rmtree(search_output_dir)
        search_summary = run_workload_space_random_search.run(**kwargs)
    selected_step_dir = _derive_selected_frontier_step(
        search_output_dir=Path(str(search_summary["output_dir"])).resolve(),
        arch_file=arch_file,
    )
    return {
        **search_summary,
        "selected_step_id": SELECTED_FRONTIER_STEP_ID,
        "selected_step_dir": str(selected_step_dir.resolve()),
    }


def _derive_selected_frontier_step(*, search_output_dir: Path, arch_file: Path) -> Path:
    selected_step_dir = (search_output_dir / "spaces" / SELECTED_FRONTIER_STEP_ID).resolve()
    if SELECTED_FRONTIER_STEP_ID == SOURCE_SEARCH_SPACE_ID:
        if not selected_step_dir.is_dir():
            raise ValueError(
                f"Selected step directory is missing after search: {selected_step_dir}"
            )
        return selected_step_dir

    source_step_dir = (search_output_dir / "spaces" / SOURCE_SEARCH_SPACE_ID).resolve()
    if not source_step_dir.is_dir():
        raise ValueError(
            f"Source search-space '{SOURCE_SEARCH_SPACE_ID}' output is missing: "
            f"{source_step_dir}"
        )

    arch = Arch.from_yaml_file(arch_file)
    if SELECTED_FRONTIER_STEP_ID == "d":
        run_workload_space_random_search._derive_space_d_from_c(
            source_dir=source_step_dir,
            dest_dir=selected_step_dir,
            arch=arch,
            resume=True,
        )
        return selected_step_dir

    raise ValueError(
        f"Unsupported selected frontier step '{SELECTED_FRONTIER_STEP_ID}'."
    )


def _load_per_workload_frontier_rows(
    evaluations_csv: Path,
) -> tuple[dict[str, Any], ...]:
    if not evaluations_csv.is_file():
        raise ValueError(f"Missing evaluations CSV: {evaluations_csv}")

    deduped_by_state_hash: dict[str, dict[str, Any]] = {}
    with evaluations_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("status", "")).strip() != "ok":
                continue
            if str(row.get("step_id", "")).strip() != SELECTED_FRONTIER_STEP_ID:
                continue
            state_hash = str(row.get("state_hash", "")).strip()
            if not state_hash or state_hash in deduped_by_state_hash:
                continue
            deduped_by_state_hash[state_hash] = {
                "state_hash": state_hash,
                "latency_cycles": int(row["latency_cycles"]),
                "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
                "energy": int(row["energy"]),
                "weighted_cost": float(row["weighted_cost"]),
                "workload_path": str(row["workload_path"]).strip(),
            }
    return tuple(deduped_by_state_hash.values())


def _decorate_frontier_component(
    *, row: Mapping[str, Any], workload_token: str
) -> dict[str, Any]:
    component = {
        "workload_token": workload_token,
        "state_hash": str(row["state_hash"]),
        "workload_path": str(row.get("workload_path", "")),
        "latency_cycles": int(row["latency_cycles"]),
        "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
        "energy": int(row["energy"]),
    }
    return {
        "state_hash": str(row["state_hash"]),
        "latency_cycles": int(row["latency_cycles"]),
        "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
        "energy": int(row["energy"]),
        "weighted_cost": float(row.get("weighted_cost", 0.0)),
        "workload_path": str(row.get("workload_path", "")),
        "components": (component,),
    }


def _compose_full_frontier(
    *,
    generated_workloads: Sequence[GeneratedWorkload],
    per_workload_frontiers: Mapping[str, Sequence[Mapping[str, Any]]],
    approx_max_points: int,
    approx_epsilon: float,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    approximate = False
    repeated_frontiers: dict[str, tuple[dict[str, Any], ...]] = {}
    for workload in generated_workloads:
        repeated_frontier, became_approximate = _repeat_frontier(
            frontier=per_workload_frontiers[workload.token],
            multiplicity=workload.multiplicity,
            approx_max_points=approx_max_points,
            approx_epsilon=approx_epsilon,
        )
        repeated_frontiers[workload.token] = repeated_frontier
        approximate = approximate or became_approximate

    combined_frontier: tuple[dict[str, Any], ...] = (_identity_row(),)
    for workload in generated_workloads:
        combined_frontier, became_approximate = _combine_frontiers(
            left_frontier=combined_frontier,
            right_frontier=repeated_frontiers[workload.token],
            approx_max_points=approx_max_points,
            approx_epsilon=approx_epsilon,
        )
        approximate = approximate or became_approximate
    return combined_frontier, approximate


def _repeat_frontier(
    *,
    frontier: Sequence[Mapping[str, Any]],
    multiplicity: int,
    approx_max_points: int,
    approx_epsilon: float,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    if multiplicity <= 0:
        raise ValueError(f"multiplicity must be positive, got {multiplicity}.")

    approximate = False
    result: tuple[dict[str, Any], ...] = (_identity_row(),)
    power: tuple[dict[str, Any], ...] = tuple(dict(row) for row in frontier)
    remaining = int(multiplicity)
    while remaining > 0:
        if remaining & 1:
            result, became_approximate = _combine_frontiers(
                left_frontier=result,
                right_frontier=power,
                approx_max_points=approx_max_points,
                approx_epsilon=approx_epsilon,
            )
            approximate = approximate or became_approximate
        remaining >>= 1
        if remaining == 0:
            break
        power, became_approximate = _combine_frontiers(
            left_frontier=power,
            right_frontier=power,
            approx_max_points=approx_max_points,
            approx_epsilon=approx_epsilon,
        )
        approximate = approximate or became_approximate
    return result, approximate


def _combine_frontiers(
    *,
    left_frontier: Sequence[Mapping[str, Any]],
    right_frontier: Sequence[Mapping[str, Any]],
    approx_max_points: int,
    approx_epsilon: float,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    if not left_frontier or not right_frontier:
        return tuple(), False

    approximate = False
    current_frontier: tuple[dict[str, Any], ...] = tuple()
    batch_size = max(1024, min(8192, max(1, approx_max_points // 4)))
    pending: list[dict[str, Any]] = []
    for left_row in left_frontier:
        for right_row in right_frontier:
            pending.append(_sum_frontier_rows(left_row, right_row))
            if len(pending) >= batch_size:
                current_frontier, became_approximate = _merge_frontier_batch(
                    current_frontier=current_frontier,
                    pending_rows=pending,
                    approx_max_points=approx_max_points,
                    approx_epsilon=approx_epsilon,
                    already_approximate=approximate,
                )
                approximate = approximate or became_approximate
                pending = []
    if pending:
        current_frontier, became_approximate = _merge_frontier_batch(
            current_frontier=current_frontier,
            pending_rows=pending,
            approx_max_points=approx_max_points,
            approx_epsilon=approx_epsilon,
            already_approximate=approximate,
        )
        approximate = approximate or became_approximate
    return current_frontier, approximate


def _merge_frontier_batch(
    *,
    current_frontier: Sequence[Mapping[str, Any]],
    pending_rows: Sequence[Mapping[str, Any]],
    approx_max_points: int,
    approx_epsilon: float,
    already_approximate: bool,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    exact_frontier = pareto_front_nd(
        tuple(current_frontier) + tuple(pending_rows),
        metrics=COMBINED_METRICS,
    )
    if len(exact_frontier) <= approx_max_points:
        return exact_frontier, already_approximate

    compressed_rows = _compress_frontier_epsilon(
        exact_frontier,
        metrics=COMBINED_METRICS,
        epsilon=approx_epsilon,
    )
    approx_frontier = pareto_front_nd(
        compressed_rows,
        metrics=COMBINED_METRICS,
    )
    return approx_frontier, True


def _compress_frontier_epsilon(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    epsilon: float,
) -> tuple[dict[str, Any], ...]:
    log_base = math.log1p(float(epsilon))
    bucketed: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in rows:
        bucket = tuple(
            _epsilon_bucket(float(row[metric]), log_base) for metric in metrics
        )
        existing = bucketed.get(bucket)
        if existing is None or _row_sort_key(row, metrics) < _row_sort_key(
            existing, metrics
        ):
            bucketed[bucket] = dict(row)
    return tuple(bucketed.values())


def _epsilon_bucket(value: float, log_base: float) -> int:
    if value <= 0:
        return 0
    return int(math.floor(math.log(value) / log_base))


def _sum_frontier_rows(
    left_row: Mapping[str, Any], right_row: Mapping[str, Any]
) -> dict[str, Any]:
    components = tuple(left_row.get("components", ())) + tuple(
        right_row.get("components", ())
    )
    workload_path = str(left_row.get("workload_path", "")).strip()
    if not workload_path:
        workload_path = str(right_row.get("workload_path", "")).strip()
    return {
        "state_hash": _aggregate_state_hash(components),
        "latency_cycles": int(left_row["latency_cycles"])
        + int(right_row["latency_cycles"]),
        "mem_footprint_bytes": int(left_row["mem_footprint_bytes"])
        + int(right_row["mem_footprint_bytes"]),
        "energy": int(left_row["energy"]) + int(right_row["energy"]),
        "weighted_cost": float(left_row.get("weighted_cost", 0.0))
        + float(right_row.get("weighted_cost", 0.0)),
        "workload_path": workload_path,
        "components": components,
    }


def _aggregate_state_hash(components: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for component in components:
        digest.update(str(component.get("workload_token", "")).encode("utf-8"))
        digest.update(b":")
        digest.update(str(component.get("state_hash", "")).encode("utf-8"))
        digest.update(b";")
    return digest.hexdigest()[:16]


def _identity_row() -> dict[str, Any]:
    return {
        "state_hash": "identity",
        "latency_cycles": 0,
        "mem_footprint_bytes": 0,
        "energy": 0,
        "weighted_cost": 0.0,
        "workload_path": "",
        "components": tuple(),
    }


def _serialize_per_workload_frontier_rows(
    *,
    arch_name: str,
    workload_token: str,
    frontier_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    sorted_rows = sorted(
        frontier_rows, key=lambda row: _row_sort_key(row, COMBINED_METRICS)
    )
    return tuple(
        {
            "order": index,
            "arch_name": arch_name,
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


def _serialize_combined_rows(
    *,
    arch_name: str,
    frontier_rows: Sequence[Mapping[str, Any]],
    generated_workloads: Sequence[GeneratedWorkload],
) -> tuple[dict[str, Any], ...]:
    sorted_rows = sorted(
        frontier_rows, key=lambda row: _row_sort_key(row, COMBINED_METRICS)
    )
    return tuple(
        {
            "order": index,
            "arch_name": arch_name,
            "state_hash": str(row["state_hash"]),
            "latency_cycles": int(row["latency_cycles"]),
            "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
            "energy": int(row["energy"]),
            "weighted_cost": float(row.get("weighted_cost", 0.0)),
            "component_count": len(tuple(row.get("components", ()))),
            "layer_assignments_json": json.dumps(
                _build_layer_assignments(
                    components=tuple(row.get("components", ())),
                    generated_workloads=generated_workloads,
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for index, row in enumerate(sorted_rows, start=1)
    )


def _build_layer_assignments(
    *,
    components: Sequence[Mapping[str, Any]],
    generated_workloads: Sequence[GeneratedWorkload],
) -> list[dict[str, Any]]:
    layer_indices_by_token = {
        workload.token: list(workload.layer_indices) for workload in generated_workloads
    }
    offsets_by_token = {workload.token: 0 for workload in generated_workloads}
    assignments: list[dict[str, Any]] = []
    for component in components:
        token = str(component.get("workload_token", ""))
        if token not in layer_indices_by_token:
            raise ValueError(
                f"Unknown workload token '{token}' in combined frontier component."
            )
        layer_indices = layer_indices_by_token[token]
        offset = offsets_by_token[token]
        if offset >= len(layer_indices):
            raise ValueError(
                f"Too many components were assigned to workload token '{token}'."
            )
        layer_index = layer_indices[offset]
        offsets_by_token[token] = offset + 1
        assignments.append(
            {
                "layer_index": int(layer_index),
                "workload_token": token,
                "state_hash": str(component.get("state_hash", "")),
                "workload_path": str(component.get("workload_path", "")),
                "latency_cycles": int(component.get("latency_cycles", 0)),
                "mem_footprint_bytes": int(component.get("mem_footprint_bytes", 0)),
                "energy": int(component.get("energy", 0)),
            }
        )
    return sorted(assignments, key=lambda item: int(item["layer_index"]))


def _normalize_arches(arch_names: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for arch_name in arch_names:
        canonical = _canonical_arch_name(arch_name)
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    if not normalized:
        raise ValueError("At least one architecture must be selected.")
    return tuple(normalized)


def _resolve_cost_arch_file(arch_name: str) -> Path:
    try:
        path = DEFAULT_COST_ARCH_FILES[arch_name].resolve()
    except KeyError as exc:
        raise ValueError(f"Unsupported architecture '{arch_name}'.") from exc
    if not path.is_file():
        raise ValueError(f"Missing cost-model architecture YAML for '{arch_name}': {path}")
    return path


def _canonical_arch_name(arch_name: str) -> str:
    token = str(arch_name).strip().lower().replace("-", "_")
    if token not in {"upmem", "hbm_pim"}:
        raise ValueError(f"Unsupported architecture '{arch_name}'.")
    return token


def _row_sort_key(
    row: Mapping[str, Any], metrics: Sequence[str]
) -> tuple[float, ...] | tuple[float, ...]:
    return (
        *tuple(float(row[metric]) for metric in metrics),
        float(row.get("weighted_cost", 0.0)),
        float(len(tuple(row.get("components", ())))),
    )


def _dump_yaml(payload: Mapping[str, Any]) -> str:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError("PyYAML is required to write workloads.") from exc
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping JSON payload in {path}.")
    return payload


def _try_load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_json_mapping(path)


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, f"{json.dumps(payload, indent=2, sort_keys=True)}\n")


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
