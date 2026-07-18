from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from tqdm import tqdm

from pnmax.cli import run_end_to_end_workload_space as _base_e2e
from pnmax.cli import (
    run_geometry_sweep_workload_space_pareto_eval as _geometry_pareto_eval,
)
from pnmax.dse.pareto_export import pareto_front_nd
from pnmax.paths import repo_root, results_root
from pnmax.seeding import default_seed

DEFAULT_ARCH_FAMILY = "upmem"
DEFAULT_GEOMETRY_ROOT = repo_root() / "data" / "archs" / "lowered" / "geometry_sweep"
DEFAULT_WORKLOAD_ROOT = repo_root() / "data" / "workloads" / "end_to_end"
DEFAULT_OUTPUT_ROOT = results_root() / "geometry_sweep_end_to_end_workload_space"
DEFAULT_NUM_TRACES = 2048
DEFAULT_SEARCH_WORKERS = 100
DEFAULT_EVAL_WORKERS = 100
DEFAULT_MAX_ATTEMPTS = 1_000_000_000
DEFAULT_APPROX_MAX_POINTS = 100_000
DEFAULT_APPROX_EPSILON = 0.01
# Layer counts are derived from the shipped end-to-end workload manifests
# (DEFAULT_WORKLOAD_ROOT/<model>/manifest.json), not hardcoded here: the layer
# CSV loader truncates at the requested count, so a stale count would silently
# drop model layers.
DEFAULT_MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("llama3-8B-1024", "data/nn_models/llama3-8B-1024"),
    ("gpt2-medium-1024", "data/nn_models/gpt2-medium-1024"),
    ("opt-2.7B-2048", "data/nn_models/opt-2.7B-2048"),
)
BASELINE_GEOMETRY_BY_ARCH: dict[str, str] = {
    "upmem": "upmem__pu-8__hmat-16__vmat-64",
    "hbm_pim": "hbm_pim__pu-8__hmat-16__vmat-32",
}
COMBINED_FRONTIER_FIELDNAMES: tuple[str, ...] = (
    "order",
    "arch_family",
    "geometry_id",
    "geometry_label",
    "geometry_arch_file",
    "model_name",
    "state_hash",
    "latency_cycles",
    "mem_footprint_bytes",
    "energy",
    "area_mm2",
    "edap",
    "weighted_cost",
    "component_count",
    "layer_assignments_json",
)


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_dir: Path
    num_layers: int


@dataclass(frozen=True)
class GeometryVariant:
    geometry_id: str
    geometry_label: str
    arch_file: Path
    area_mm2: float


class _ProgressCoordinator:
    def __init__(self, *, enabled: bool, total_models: int) -> None:
        self.enabled = bool(enabled)
        self._models = None
        self._model_stages = None
        self._tokens = None
        self._stage_detail = None
        self._search_detail = None
        self._current_model_name: str | None = None
        self._current_search_geometry: str | None = None
        if self.enabled:
            self._models = tqdm(
                total=max(1, int(total_models)),
                desc="Models",
                unit="model",
                position=0,
                leave=True,
            )

    def close(self) -> None:
        self._close_search_detail()
        self._close_stage_detail()
        self._close_tokens()
        self._close_model_stages()
        if self._models is not None:
            self._models.close()
            self._models = None

    def start_model(self, *, model_name: str, num_layers: int) -> None:
        if not self.enabled:
            return
        self._close_search_detail()
        self._close_stage_detail()
        self._close_tokens()
        self._close_model_stages()
        self._current_model_name = model_name
        self._current_search_geometry = None
        self._model_stages = tqdm(
            total=3,
            desc=f"{model_name} stages",
            unit="stage",
            position=1,
            leave=False,
        )
        self._model_stages.set_postfix_str(f"layers={int(num_layers)}")
        if self._models is not None:
            self._models.set_postfix_str(f"current={model_name}")

    def finish_prepare(self, *, token_total: int) -> None:
        if not self.enabled:
            return
        self._advance_model_stage(postfix=f"prepare ({int(token_total)} tokens)")
        self._close_tokens()
        self._tokens = tqdm(
            total=max(1, int(token_total)),
            desc=f"{self._current_model_name} workloads",
            unit="token",
            position=2,
            leave=False,
        )
        self._tokens.set_postfix_str(f"completed=0/{int(token_total)}")

    def start_token_search(self, *, workload_token: str, geometry_total: int) -> None:
        if not self.enabled:
            return
        self._close_search_detail()
        self._close_stage_detail()
        if self._tokens is not None:
            self._tokens.set_postfix_str(f"current={workload_token}")
        self._stage_detail = tqdm(
            total=max(1, int(geometry_total)),
            desc=f"{self._current_model_name}/{workload_token} search",
            unit="geometry",
            position=3,
            leave=False,
        )
        self._current_search_geometry = None

    def update_search_trace(
        self,
        *,
        workload_token: str,
        geometry_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return
        total = max(1, int(payload["target_traces"]))
        completed = max(0, min(int(payload["legally_found"]), total))
        attempts = int(payload["attempts"])
        if self._current_search_geometry != geometry_id or self._search_detail is None:
            self._close_search_detail()
            self._search_detail = tqdm(
                total=total,
                desc=f"{self._current_model_name}/{workload_token}/{geometry_id} traces",
                unit="trace",
                position=4,
                leave=False,
            )
            self._current_search_geometry = geometry_id
        current_total = int(self._search_detail.total or 0)
        current_completed = int(getattr(self._search_detail, "n", 0))
        if current_total != total or completed < current_completed:
            self._search_detail.reset(total=total)
            current_completed = int(getattr(self._search_detail, "n", 0))
        if completed > current_completed:
            self._search_detail.update(completed - current_completed)
        self._search_detail.set_postfix_str(
            f"attempts={attempts} valid={completed}/{total}"
        )

    def finish_search_geometry(self, *, geometry_id: str) -> None:
        if not self.enabled or self._stage_detail is None:
            return
        current_completed = int(getattr(self._stage_detail, "n", 0))
        target_total = int(self._stage_detail.total or 0)
        if current_completed < target_total:
            self._stage_detail.update(1)
        self._stage_detail.set_postfix_str(f"done={geometry_id}")

    def start_token_eval(self, *, workload_token: str) -> None:
        if not self.enabled:
            return
        self._close_search_detail()
        self._close_stage_detail()
        if self._tokens is not None:
            self._tokens.set_postfix_str(f"eval={workload_token}")

    def update_eval(
        self,
        *,
        workload_token: str,
        payload: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return
        total = max(1, int(payload["total"]))
        completed = max(0, min(int(payload["completed"]), total))
        if self._stage_detail is None:
            self._stage_detail = tqdm(
                total=total,
                desc=f"{self._current_model_name}/{workload_token} eval",
                unit="task",
                position=3,
                leave=False,
            )
        current_total = int(self._stage_detail.total or 0)
        current_completed = int(getattr(self._stage_detail, "n", 0))
        if current_total != total or completed < current_completed:
            self._stage_detail.reset(total=total)
            current_completed = int(getattr(self._stage_detail, "n", 0))
        if completed > current_completed:
            self._stage_detail.update(completed - current_completed)
        self._stage_detail.set_postfix_str(f"completed={completed}/{int(payload['total'])}")

    def finish_token(self) -> None:
        if not self.enabled:
            return
        self._close_search_detail()
        self._close_stage_detail()
        if self._tokens is None:
            return
        current_completed = int(getattr(self._tokens, "n", 0))
        target_total = int(self._tokens.total or 0)
        if current_completed < target_total:
            self._tokens.update(1)
        self._tokens.set_postfix_str(
            f"completed={self._tokens.n}/{int(self._tokens.total or 0)}"
        )

    def finish_token_stage(self) -> None:
        if not self.enabled:
            return
        self._advance_model_stage(postfix="search/eval complete")

    def start_compose(self, *, geometry_total: int) -> None:
        if not self.enabled:
            return
        self._close_search_detail()
        self._close_stage_detail()
        self._stage_detail = tqdm(
            total=max(1, int(geometry_total)),
            desc=f"{self._current_model_name} compose",
            unit="geometry",
            position=3,
            leave=False,
        )

    def advance_compose(self, *, geometry_id: str) -> None:
        if not self.enabled or self._stage_detail is None:
            return
        current_completed = int(getattr(self._stage_detail, "n", 0))
        target_total = int(self._stage_detail.total or 0)
        if current_completed < target_total:
            self._stage_detail.update(1)
        self._stage_detail.set_postfix_str(f"done={geometry_id}")

    def finish_model(self) -> None:
        if not self.enabled:
            return
        self._close_search_detail()
        self._close_stage_detail()
        self._advance_model_stage(postfix="compose complete")
        if self._models is not None:
            self._models.update(1)
        self._close_tokens()
        self._close_model_stages()

    def _advance_model_stage(self, *, postfix: str) -> None:
        if self._model_stages is None:
            return
        current_completed = int(getattr(self._model_stages, "n", 0))
        target_total = int(self._model_stages.total or 0)
        if current_completed < target_total:
            self._model_stages.update(1)
        self._model_stages.set_postfix_str(postfix)

    def _close_model_stages(self) -> None:
        if self._model_stages is not None:
            self._model_stages.close()
            self._model_stages = None

    def _close_tokens(self) -> None:
        if self._tokens is not None:
            self._tokens.close()
            self._tokens = None

    def _close_stage_detail(self) -> None:
        if self._stage_detail is not None:
            self._stage_detail.close()
            self._stage_detail = None

    def _close_search_detail(self) -> None:
        if self._search_detail is not None:
            self._search_detail.close()
            self._search_detail = None
        self._current_search_geometry = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end workload-space search/eval across every geometry "
            "variant in one architecture family using cache-enabled mappings "
            "and select EDAP winners."
        )
    )
    parser.add_argument(
        "--arch-family",
        type=str,
        default=DEFAULT_ARCH_FAMILY,
        choices=sorted(_base_e2e.DEFAULT_COST_ARCH_FILES),
        help='Geometry family to evaluate ("upmem" or "hbm_pim").',
    )
    parser.add_argument(
        "--geometry-root",
        type=str,
        default=str(DEFAULT_GEOMETRY_ROOT),
        help="Root directory containing geometry-sweep YAMLs grouped by architecture.",
    )
    parser.add_argument(
        "--workload-root",
        type=str,
        default=str(DEFAULT_WORKLOAD_ROOT),
        help="Root directory for generated end-to-end workloads.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for geometry-sweep end-to-end artifacts.",
    )
    parser.add_argument(
        "--output-dir-file",
        type=str,
        default=None,
        help="Optional path to write the resolved pipeline output directory.",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=DEFAULT_NUM_TRACES,
        help=(
            "Number of valid cache-enabled (d) mappings to generate per "
            "geometry/workload."
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
        help="Number of analytical evaluation workers.",
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
        help="Base seed for reproducibility (default: --seed > PNMAX_SEED > 42).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum random-search attempts per geometry/workload pair.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable lightweight progress logging.",
    )
    return parser.parse_args()


def run(
    *,
    arch_family: str = DEFAULT_ARCH_FAMILY,
    geometry_root: str = str(DEFAULT_GEOMETRY_ROOT),
    workload_root: str = str(DEFAULT_WORKLOAD_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    output_dir_file: str | None = None,
    num_traces: int = DEFAULT_NUM_TRACES,
    search_workers: int = DEFAULT_SEARCH_WORKERS,
    eval_workers: int = DEFAULT_EVAL_WORKERS,
    approx_max_points: int = DEFAULT_APPROX_MAX_POINTS,
    approx_epsilon: float = DEFAULT_APPROX_EPSILON,
    seed: int | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    show_progress: bool = True,
) -> dict[str, Any]:
    canonical_arch_family = _base_e2e._canonical_arch_name(arch_family)
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

    resolved_geometry_root = Path(geometry_root).resolve()
    resolved_workload_root = Path(workload_root).resolve()
    resolved_output_root = Path(output_root).resolve()
    geometry_dir = resolved_geometry_root / canonical_arch_family
    model_specs = _resolve_default_model_specs()
    geometries = _discover_geometries(geometry_dir=geometry_dir)
    if not geometries:
        raise ValueError(f"No geometry variants found under {geometry_dir}.")

    pipeline_dir = _resolve_pipeline_output_dir(
        output_root=resolved_output_root,
        arch_family=canonical_arch_family,
    )
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    if output_dir_file is not None:
        _write_output_dir_file(output_dir_file=output_dir_file, output_dir=pipeline_dir)

    baseline_geometry_id = BASELINE_GEOMETRY_BY_ARCH[canonical_arch_family]
    if show_progress:
        progress = _ProgressCoordinator(
            enabled=True,
            total_models=len(model_specs),
        )
    else:
        progress = _ProgressCoordinator(enabled=False, total_models=len(model_specs))

    model_summaries: dict[str, dict[str, Any]] = {}
    try:
        for model_spec in model_specs:
            model_summaries[model_spec.model_name] = _run_model_pipeline(
                model_spec=model_spec,
                arch_family=canonical_arch_family,
                geometries=geometries,
                workload_root=resolved_workload_root,
                pipeline_dir=pipeline_dir,
                num_traces=num_traces,
                search_workers=search_workers,
                eval_workers=eval_workers,
                approx_max_points=approx_max_points,
                approx_epsilon=approx_epsilon,
                seed=seed,
                max_attempts=max_attempts,
                progress=progress,
            )
    finally:
        progress.close()

    mean_geometry_results = _build_mean_geometry_results(
        geometries=geometries,
        model_summaries=model_summaries,
    )
    if baseline_geometry_id not in mean_geometry_results:
        raise ValueError(
            f"Baseline geometry '{baseline_geometry_id}' is missing from the mean "
            f"geometry summary for {canonical_arch_family}."
        )

    radar_plot_outputs = {
        "winner_radar_pdf": str(
            (pipeline_dir / "radar_plots" / f"{canonical_arch_family}_winners.pdf").resolve()
        )
    }
    summary = {
        "generated_at": _utcnow(),
        "arch_family": canonical_arch_family,
        "selected_step_id": _base_e2e.SELECTED_FRONTIER_STEP_ID,
        "geometry_root": str(resolved_geometry_root),
        "geometry_dir": str(geometry_dir),
        "workload_root": str(resolved_workload_root),
        "output_dir": str(pipeline_dir.resolve()),
        "baseline_geometry_id": baseline_geometry_id,
        "num_traces": int(num_traces),
        "search_workers": int(search_workers),
        "eval_workers": int(eval_workers),
        "approx_max_points": int(approx_max_points),
        "approx_epsilon": float(approx_epsilon),
        "seed": seed,
        "max_attempts": int(max_attempts),
        "model_specs": [
            {
                "model_name": model_spec.model_name,
                "model_dir": str(model_spec.model_dir),
                "num_layers": int(model_spec.num_layers),
            }
            for model_spec in model_specs
        ],
        "geometries": {
            geometry.geometry_id: {
                "geometry_id": geometry.geometry_id,
                "geometry_label": geometry.geometry_label,
                "geometry_arch_file": str(geometry.arch_file),
                "area_mm2": float(geometry.area_mm2),
            }
            for geometry in geometries
        },
        "models": model_summaries,
        "mean_geometry_results": mean_geometry_results,
        "mean_winner": _select_mean_winner(mean_geometry_results),
        "radar_plot_outputs": radar_plot_outputs,
    }
    summary_path = pipeline_dir / "summary.json"
    _base_e2e._write_json(summary_path, summary)
    return {**summary, "summary_json": str(summary_path.resolve())}


def main() -> int:
    args = parse_args()
    run(
        arch_family=args.arch_family,
        geometry_root=args.geometry_root,
        workload_root=args.workload_root,
        output_root=args.output_root,
        output_dir_file=args.output_dir_file,
        num_traces=args.num_traces,
        search_workers=args.search_workers,
        eval_workers=args.eval_workers,
        approx_max_points=args.approx_max_points,
        approx_epsilon=args.approx_epsilon,
        seed=args.seed,
        max_attempts=args.max_attempts,
        show_progress=not args.no_progress,
    )
    return 0


def _resolve_default_model_specs() -> tuple[ModelSpec, ...]:
    specs: list[ModelSpec] = []
    for model_name, model_dir in DEFAULT_MODEL_SPECS:
        manifest_path = DEFAULT_WORKLOAD_ROOT / str(model_name) / "manifest.json"
        specs.append(
            ModelSpec(
                model_name=str(model_name),
                model_dir=(repo_root() / model_dir).resolve(),
                num_layers=_manifest_layer_count(manifest_path=manifest_path),
            )
        )
    return tuple(specs)


def _manifest_layer_count(*, manifest_path: Path) -> int:
    """Authoritative layer count: the model's end-to-end workload manifest.

    The manifest embeds one ``layers`` row per model layer and declares the
    same count as ``num_layers``; both views must agree. Deriving the count
    here keeps ``_load_layer_shapes`` (which stops reading the layer CSV at
    the requested count) from ever silently dropping rows.
    """
    if not manifest_path.is_file():
        raise ValueError(f"Missing end-to-end workload manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping JSON payload in {manifest_path}.")
    layers = payload.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"Missing or empty layers rows in {manifest_path}.")
    declared = payload.get("num_layers")
    if not isinstance(declared, int) or declared != len(layers):
        raise ValueError(
            f"num_layers={declared!r} disagrees with {len(layers)} layers rows "
            f"in {manifest_path}."
        )
    return len(layers)


def _require_csv_layer_rows(
    *,
    csv_path: Path,
    model_name: str,
    num_layers: int,
) -> None:
    """Refuse to load a layer CSV whose row count differs from the manifest.

    ``_load_layer_shapes`` stops reading after ``num_layers`` rows, so extra
    CSV rows would otherwise be truncated silently.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        row_count = sum(1 for row in csv.reader(fh) if row)
    if row_count != num_layers:
        raise ValueError(
            f"{csv_path} has {row_count} layer rows, but the {model_name} "
            f"end-to-end manifest declares {num_layers}."
        )


def _discover_geometries(*, geometry_dir: Path) -> tuple[GeometryVariant, ...]:
    if not geometry_dir.is_dir():
        raise ValueError(f"Geometry directory does not exist: {geometry_dir}")

    geometries: list[GeometryVariant] = []
    for arch_file in sorted(geometry_dir.glob("*.yaml")):
        geometry_id = arch_file.stem
        geometries.append(
            GeometryVariant(
                geometry_id=geometry_id,
                geometry_label=geometry_id,
                arch_file=arch_file.resolve(),
                area_mm2=_load_area_mm2(arch_file),
            )
        )
    return tuple(geometries)


def _run_model_pipeline(
    *,
    model_spec: ModelSpec,
    arch_family: str,
    geometries: Sequence[GeometryVariant],
    workload_root: Path,
    pipeline_dir: Path,
    num_traces: int,
    search_workers: int,
    eval_workers: int,
    approx_max_points: int,
    approx_epsilon: float,
    seed: int | None,
    max_attempts: int,
    progress: _ProgressCoordinator,
) -> dict[str, Any]:
    csv_path = model_spec.model_dir / "layer_params.csv"
    if not csv_path.is_file():
        raise ValueError(f"Missing layer_params.csv under {model_spec.model_dir}.")
    _require_csv_layer_rows(
        csv_path=csv_path,
        model_name=model_spec.model_name,
        num_layers=model_spec.num_layers,
    )

    progress.start_model(
        model_name=model_spec.model_name,
        num_layers=model_spec.num_layers,
    )
    model_workloads_dir = (workload_root / model_spec.model_name).resolve()
    model_workloads_dir.mkdir(parents=True, exist_ok=True)
    layer_shapes = _base_e2e._load_layer_shapes(
        csv_path=csv_path,
        num_layers=model_spec.num_layers,
    )
    generated_workloads = _base_e2e._write_generated_workloads(
        layer_shapes=layer_shapes,
        workloads_dir=model_workloads_dir,
    )
    workloads_manifest_path = model_workloads_dir / "manifest.json"
    workloads_manifest = _base_e2e._write_workload_manifest(
        path=workloads_manifest_path,
        csv_path=csv_path,
        model_dir=model_spec.model_dir,
        layer_shapes=layer_shapes,
        generated_workloads=generated_workloads,
    )
    progress.finish_prepare(token_total=len(generated_workloads))

    per_geometry_workload_frontiers: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {
        geometry.geometry_id: {} for geometry in geometries
    }
    token_artifacts: dict[str, dict[str, Any]] = {}

    for workload in generated_workloads:
        search_dirs_by_geometry = _run_geometry_searches_for_workload(
            model_spec=model_spec,
            arch_family=arch_family,
            geometries=geometries,
            pipeline_dir=pipeline_dir,
            workload=workload,
            num_traces=num_traces,
            search_workers=search_workers,
            seed=seed,
            max_attempts=max_attempts,
            progress=progress,
        )
        progress.start_token_eval(workload_token=workload.token)
        eval_progress_callback: Callable[[Mapping[str, Any]], None] | None = (
            lambda payload, workload_token=workload.token: progress.update_eval(
                workload_token=workload_token,
                payload=payload,
            )
        )
        eval_summary = _geometry_pareto_eval.run(
            geometry_dirs=[
                str(search_dirs_by_geometry[geometry.geometry_id])
                for geometry in geometries
            ],
            arch_name=arch_family,
            output_root=str((pipeline_dir / "pareto_eval" / model_spec.model_name).resolve()),
            target=arch_family,
            workers=eval_workers,
            verbose=0,
            progress_callback=eval_progress_callback,
        )
        eval_output_dir = Path(str(eval_summary["output_dir"])).resolve()
        geometry_frontiers = _load_geometry_frontiers_for_workload(
            evaluations_csv=eval_output_dir / "evaluations.csv",
            workload_token=workload.token,
        )
        for geometry in geometries:
            geometry_frontier = geometry_frontiers.get(geometry.geometry_id)
            if not geometry_frontier:
                raise ValueError(
                    f"No valid cache-enabled frontier rows were found for "
                    f"{model_spec.model_name}/{geometry.geometry_id}/{workload.token}."
                )
            per_geometry_workload_frontiers[geometry.geometry_id][
                workload.token
            ] = geometry_frontier

        token_artifacts[workload.token] = {
            "search_dirs_by_geometry": {
                geometry_id: str(path.resolve())
                for geometry_id, path in search_dirs_by_geometry.items()
            },
            "pareto_eval": {
                "output_dir": str(eval_output_dir),
                "summary_json": str((eval_output_dir / "summary.json").resolve()),
                "manifest_json": str((eval_output_dir / "manifest.json").resolve()),
                "evaluations_csv": str((eval_output_dir / "evaluations.csv").resolve()),
                "pareto_frontiers_csv": str(
                    (eval_output_dir / "pareto_frontiers.csv").resolve()
                ),
            },
        }
        progress.finish_token()

    progress.finish_token_stage()
    geometry_results: dict[str, dict[str, Any]] = {}
    progress.start_compose(geometry_total=len(geometries))
    for geometry in geometries:
        geometry_results[geometry.geometry_id] = _compose_geometry_model_frontier(
            model_spec=model_spec,
            arch_family=arch_family,
            geometry=geometry,
            pipeline_dir=pipeline_dir,
            generated_workloads=generated_workloads,
            per_workload_frontiers=per_geometry_workload_frontiers[geometry.geometry_id],
            approx_max_points=approx_max_points,
            approx_epsilon=approx_epsilon,
        )
        progress.advance_compose(geometry_id=geometry.geometry_id)

    progress.finish_model()

    return {
        "model_name": model_spec.model_name,
        "model_dir": str(model_spec.model_dir),
        "selected_step_id": _base_e2e.SELECTED_FRONTIER_STEP_ID,
        "num_layers": int(model_spec.num_layers),
        "source_csv": str(csv_path.resolve()),
        "workloads_dir": str(model_workloads_dir.resolve()),
        "workloads_manifest_json": str(workloads_manifest_path.resolve()),
        "workloads_manifest": workloads_manifest,
        "unique_workloads": [
            {
                "token": workload.token,
                "yaml_path": str(workload.yaml_path),
                "layer_indices": list(workload.layer_indices),
                "multiplicity": workload.multiplicity,
            }
            for workload in generated_workloads
        ],
        "token_artifacts": token_artifacts,
        "geometry_results": geometry_results,
        "winner": _select_model_winner(geometry_results),
    }


def _run_geometry_searches_for_workload(
    *,
    model_spec: ModelSpec,
    arch_family: str,
    geometries: Sequence[GeometryVariant],
    pipeline_dir: Path,
    workload: _base_e2e.GeneratedWorkload,
    num_traces: int,
    search_workers: int,
    seed: int | None,
    max_attempts: int,
    progress: _ProgressCoordinator,
) -> dict[str, Path]:
    search_dirs_by_geometry: dict[str, Path] = {}
    progress.start_token_search(
        workload_token=workload.token,
        geometry_total=len(geometries),
    )
    for geometry in geometries:
        progress_callback: Callable[[Mapping[str, Any]], None] | None = (
            lambda payload, workload_token=workload.token, geometry_id=geometry.geometry_id: (
                progress.update_search_trace(
                    workload_token=workload_token,
                    geometry_id=geometry_id,
                    payload=payload,
                )
            )
        )
        search_output_dir = (
            pipeline_dir
            / "random_search"
            / model_spec.model_name
            / geometry.geometry_id
            / workload.token
        ).resolve()
        search_summary = _base_e2e._run_random_search_with_cleanup_on_incompatible_output(
            workload=workload,
            arch_name=arch_family,
            arch_file=geometry.arch_file,
            search_output_dir=search_output_dir,
            num_traces=num_traces,
            seed=_derived_seed(
                seed=seed,
                model_name=model_spec.model_name,
                geometry_id=geometry.geometry_id,
                workload_token=workload.token,
            ),
            max_attempts=max_attempts,
            search_workers=search_workers,
            progress_callback=progress_callback,
        )
        search_dirs_by_geometry[geometry.geometry_id] = Path(
            str(search_summary["output_dir"])
        ).resolve()
        progress.finish_search_geometry(geometry_id=geometry.geometry_id)
    return search_dirs_by_geometry


def _load_geometry_frontiers_for_workload(
    *,
    evaluations_csv: Path,
    workload_token: str,
) -> dict[str, tuple[dict[str, Any], ...]]:
    if not evaluations_csv.is_file():
        raise ValueError(f"Missing evaluations CSV: {evaluations_csv}")

    rows_by_geometry: dict[str, dict[str, dict[str, Any]]] = {}
    with evaluations_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("status", "")).strip() != "ok":
                continue
            if (
                str(row.get("step_id", "")).strip()
                != _base_e2e.SELECTED_FRONTIER_STEP_ID
            ):
                continue
            geometry_id = str(row.get("geometry_id", "")).strip()
            state_hash = str(row.get("state_hash", "")).strip()
            if not geometry_id or not state_hash:
                continue
            candidate = {
                "geometry_id": geometry_id,
                "state_hash": state_hash,
                "latency_cycles": int(row["latency_cycles"]),
                "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
                "energy": int(row["energy"]),
                "weighted_cost": float(row["weighted_cost"]),
                "workload_path": str(row.get("workload_path", "")).strip(),
            }
            deduped_rows = rows_by_geometry.setdefault(geometry_id, {})
            existing = deduped_rows.get(state_hash)
            if existing is None or _evaluation_sort_key(candidate) < _evaluation_sort_key(
                existing
            ):
                deduped_rows[state_hash] = candidate

    frontiers: dict[str, tuple[dict[str, Any], ...]] = {}
    for geometry_id, deduped_rows in rows_by_geometry.items():
        frontier_rows = pareto_front_nd(
            tuple(deduped_rows.values()),
            metrics=_base_e2e.COMBINED_METRICS,
        )
        frontiers[geometry_id] = tuple(
            _base_e2e._decorate_frontier_component(
                row=row,
                workload_token=workload_token,
            )
            for row in frontier_rows
        )
    return frontiers


def _compose_geometry_model_frontier(
    *,
    model_spec: ModelSpec,
    arch_family: str,
    geometry: GeometryVariant,
    pipeline_dir: Path,
    generated_workloads: Sequence[_base_e2e.GeneratedWorkload],
    per_workload_frontiers: Mapping[str, Sequence[Mapping[str, Any]]],
    approx_max_points: int,
    approx_epsilon: float,
) -> dict[str, Any]:
    combined_rows, approximate = _base_e2e._compose_full_frontier(
        generated_workloads=generated_workloads,
        per_workload_frontiers=per_workload_frontiers,
        approx_max_points=approx_max_points,
        approx_epsilon=approx_epsilon,
    )
    if not combined_rows:
        raise ValueError(
            f"Combined frontier is empty for {model_spec.model_name}/{geometry.geometry_id}."
        )

    combined_dir = (
        pipeline_dir / "combined" / model_spec.model_name / geometry.geometry_id
    ).resolve()
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = combined_dir / "combined_frontier.csv"
    combined_summary_path = combined_dir / "summary.json"
    combined_rows_serialized = _serialize_combined_frontier_rows(
        arch_family=arch_family,
        geometry=geometry,
        model_name=model_spec.model_name,
        generated_workloads=generated_workloads,
        frontier_rows=combined_rows,
    )
    _base_e2e._write_csv(
        combined_csv,
        fieldnames=COMBINED_FRONTIER_FIELDNAMES,
        rows=combined_rows_serialized,
    )
    best_edap_point = _select_best_edap_point(
        geometry=geometry,
        generated_workloads=generated_workloads,
        frontier_rows=combined_rows,
    )
    summary = {
        "generated_at": _utcnow(),
        "arch_family": arch_family,
        "model_name": model_spec.model_name,
        "selected_step_id": _base_e2e.SELECTED_FRONTIER_STEP_ID,
        "geometry_id": geometry.geometry_id,
        "geometry_label": geometry.geometry_label,
        "geometry_arch_file": str(geometry.arch_file),
        "area_mm2": float(geometry.area_mm2),
        "combined_frontier_csv": str(combined_csv.resolve()),
        "combined_frontier_points": len(combined_rows_serialized),
        "approximate": bool(approximate),
        "approx_epsilon": float(approx_epsilon),
        "approx_max_points": int(approx_max_points),
        "best_edap_point": best_edap_point,
    }
    _base_e2e._write_json(combined_summary_path, summary)
    return {
        **summary,
        "summary_json": str(combined_summary_path.resolve()),
        "output_dir": str(combined_dir.resolve()),
    }


def _serialize_combined_frontier_rows(
    *,
    arch_family: str,
    geometry: GeometryVariant,
    model_name: str,
    generated_workloads: Sequence[_base_e2e.GeneratedWorkload],
    frontier_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    sorted_rows = sorted(
        frontier_rows,
        key=lambda row: _base_e2e._row_sort_key(row, _base_e2e.COMBINED_METRICS),
    )
    return tuple(
        {
            "order": index,
            "arch_family": arch_family,
            "geometry_id": geometry.geometry_id,
            "geometry_label": geometry.geometry_label,
            "geometry_arch_file": str(geometry.arch_file),
            "model_name": model_name,
            "state_hash": str(row["state_hash"]),
            "latency_cycles": int(row["latency_cycles"]),
            "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
            "energy": int(row["energy"]),
            "area_mm2": float(geometry.area_mm2),
            "edap": float(_edap(row=row, area_mm2=geometry.area_mm2)),
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


def _select_best_edap_point(
    *,
    geometry: GeometryVariant,
    generated_workloads: Sequence[_base_e2e.GeneratedWorkload],
    frontier_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not frontier_rows:
        raise ValueError(
            f"Cannot select an EDAP point for empty frontier {geometry.geometry_id}."
        )

    best_row = min(
        frontier_rows,
        key=lambda row: (
            _edap(row=row, area_mm2=geometry.area_mm2),
            int(row["latency_cycles"]),
            int(row["energy"]),
            int(row["mem_footprint_bytes"]),
            str(row.get("state_hash", "")),
        ),
    )
    layer_assignments = _base_e2e._build_layer_assignments(
        components=tuple(best_row.get("components", ())),
        generated_workloads=generated_workloads,
    )
    return {
        "geometry_id": geometry.geometry_id,
        "geometry_label": geometry.geometry_label,
        "geometry_arch_file": str(geometry.arch_file),
        "state_hash": str(best_row["state_hash"]),
        "latency_cycles": int(best_row["latency_cycles"]),
        "mem_footprint_bytes": int(best_row["mem_footprint_bytes"]),
        "energy": int(best_row["energy"]),
        "area_mm2": float(geometry.area_mm2),
        "edap": float(_edap(row=best_row, area_mm2=geometry.area_mm2)),
        "weighted_cost": float(best_row.get("weighted_cost", 0.0)),
        "layer_assignments_json": json.dumps(
            layer_assignments,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "layer_assignments": layer_assignments,
    }


def _select_model_winner(
    geometry_results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if not geometry_results:
        raise ValueError("geometry_results must not be empty.")
    ordered_results = sorted(
        geometry_results.values(),
        key=lambda result: (
            float(result["best_edap_point"]["edap"]),
            int(result["best_edap_point"]["latency_cycles"]),
            int(result["best_edap_point"]["energy"]),
            float(result["area_mm2"]),
            str(result["geometry_id"]),
        ),
    )
    winner = dict(ordered_results[0])
    winner["winner_kind"] = "per_model_edap"
    return winner


def _build_mean_geometry_results(
    *,
    geometries: Sequence[GeometryVariant],
    model_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    mean_geometry_results: dict[str, dict[str, Any]] = {}
    for geometry in geometries:
        per_model_points: dict[str, Mapping[str, Any]] = {}
        edaps: list[float] = []
        energies: list[float] = []
        latencies: list[float] = []
        for model_name, model_summary in model_summaries.items():
            geometry_results = model_summary.get("geometry_results")
            if not isinstance(geometry_results, Mapping):
                raise ValueError(
                    f"Expected geometry_results mapping for model '{model_name}'."
                )
            geometry_result = geometry_results.get(geometry.geometry_id)
            if not isinstance(geometry_result, Mapping):
                raise ValueError(
                    f"Missing geometry result for {model_name}/{geometry.geometry_id}."
                )
            best_point = geometry_result.get("best_edap_point")
            if not isinstance(best_point, Mapping):
                raise ValueError(
                    f"Missing best_edap_point for {model_name}/{geometry.geometry_id}."
                )
            per_model_points[model_name] = best_point
            edaps.append(float(best_point["edap"]))
            energies.append(float(best_point["energy"]))
            latencies.append(float(best_point["latency_cycles"]))

        mean_geometry_results[geometry.geometry_id] = {
            "geometry_id": geometry.geometry_id,
            "geometry_label": geometry.geometry_label,
            "geometry_arch_file": str(geometry.arch_file),
            "area_mm2": float(geometry.area_mm2),
            "model_points": {
                model_name: dict(point) for model_name, point in per_model_points.items()
            },
            "gmean_edap": float(_geometric_mean(edaps)),
            "gmean_energy": float(_geometric_mean(energies)),
            "gmean_latency": float(_geometric_mean(latencies)),
        }
    return mean_geometry_results


def _select_mean_winner(
    mean_geometry_results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if not mean_geometry_results:
        raise ValueError("mean_geometry_results must not be empty.")
    ordered_results = sorted(
        mean_geometry_results.values(),
        key=lambda result: (
            float(result["gmean_edap"]),
            float(result["gmean_latency"]),
            float(result["gmean_energy"]),
            float(result["area_mm2"]),
            str(result["geometry_id"]),
        ),
    )
    winner = dict(ordered_results[0])
    winner["winner_kind"] = "mean_gmean_edap"
    return winner


def _edap(*, row: Mapping[str, Any], area_mm2: float) -> float:
    return float(row["energy"]) * float(row["latency_cycles"]) * float(area_mm2)


def _derived_seed(
    *,
    seed: int | None,
    model_name: str,
    geometry_id: str,
    workload_token: str,
) -> int | None:
    if seed is None:
        return None
    digest = hashlib.blake2b(
        f"{seed}:{model_name}:{geometry_id}:{workload_token}".encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return int(digest, 16) % (2**31)


def _load_area_mm2(path: Path) -> float:
    if not path.is_file():
        raise ValueError(f"Geometry architecture YAML not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML file: {path}") from exc
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


def _evaluation_sort_key(row: Mapping[str, Any]) -> tuple[float, int, int, int, str]:
    return (
        float(row["weighted_cost"]),
        int(row["latency_cycles"]),
        int(row["mem_footprint_bytes"]),
        int(row["energy"]),
        str(row["state_hash"]),
    )


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean requires at least one value.")
    if any(float(value) <= 0.0 for value in values):
        raise ValueError(f"geometric mean requires positive values, got {values}.")
    return math.exp(sum(math.log(float(value)) for value in values) / len(values))


def _resolve_pipeline_output_dir(*, output_root: Path, arch_family: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return output_root / arch_family / timestamp


def _write_output_dir_file(*, output_dir_file: str, output_dir: Path) -> None:
    path = Path(output_dir_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{output_dir.resolve()}\n", encoding="utf-8")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
