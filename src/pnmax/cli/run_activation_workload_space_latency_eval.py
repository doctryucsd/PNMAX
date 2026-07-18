from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from tqdm import tqdm

from pnmax.analytical_model import AnalyticalModel
from pnmax.cli import _end_to_end_sweep_latency_eval_common as _common
from pnmax.cli import run_end_to_end_workload_space as _base_e2e
from pnmax.cli import run_workload_space_pareto_eval as _base_eval
from pnmax.cli import run_workload_space_random_search
from pnmax.database import Arch, Workload
from pnmax.dse.random_search import OutputDimensionFactorizationConstraint
from pnmax.dse.workload_eval import evaluate_workload_analytical_report

from pnmax.paths import repo_root, results_root
PROJECT_ROOT = repo_root()
DEFAULT_OUTPUT_ROOT = results_root() / "activation_workload_space_latency_eval"
DEFAULT_CONFIG1_ARCH_FILE = repo_root() / "data" / "archs" / "lowered" / "activation/hbm_pim__pu-8__hmat-16__vmat-32.yaml"
DEFAULT_CONFIG2_ARCH_FILE = repo_root() / "data" / "archs" / "lowered" / "activation/base_die.yaml"
# Config 3 (channel-level reduction) is OPT-IN: default None keeps the legacy
# two-config (PU-side vs base-die) runs bit-identical. When only the arch is
# provided, config3 evaluates the SAME mapping pool as config2 under the
# channel_level toggle; --config3-mapping-dirs can opt it into its own pool.
DEFAULT_CONFIG3_ARCH_FILE = repo_root() / "data" / "archs" / "lowered" / "activation/channel_level.yaml"
DEFAULT_MAX_ATTEMPTS = 3000
DEFAULT_FALLBACK_NUM_TRACES = 2048
BASELINE_GEOMETRY_ID = "hbm_pim__pu-8__hmat-16__vmat-32"
CONFIG1_SOURCE_WORKLOAD_ROOTS: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "workloads" / "b_fc",
    PROJECT_ROOT / "data" / "workloads" / "conv2d",
    PROJECT_ROOT / "data" / "workloads" / "fc",
)
MANIFEST_COMPARE_FIELDS: tuple[str, ...] = (
    "workload_path",
    "workload_name",
    "arch_name",
    "arch_file",
)
CONFIG_ORDER: tuple[str, ...] = ("config1", "config2")
CONFIG1_SEARCH_MODE = "searched"
CONFIG2_SEARCH_MODE = "baseline-reference"
CONFIG3_SEARCH_MODE = "baseline-reference"
EVALUATIONS_FIELDNAMES: tuple[str, ...] = (
    "config_id",
    "config_label",
    "config_arch_file",
    "search_mode",
    "activation",
    "base_die",
    "channel_level",
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
BEST_LATENCY_FIELDNAMES: tuple[str, ...] = ("config_order", *EVALUATIONS_FIELDNAMES)
RELATIVE_METRICS_FIELDNAMES: tuple[str, ...] = (
    "workload_token",
    "workload_display_name",
    "baseline_geometry_id",
    "config1_state_hash",
    "config2_state_hash",
    "config1_latency_cycles",
    "config2_latency_cycles",
    "latency_ratio",
    "config1_energy",
    "config2_energy",
    "energy_ratio",
    "config1_area_mm2",
    "config2_area_mm2",
    "area_ratio",
    "config1_mem_footprint_bytes",
    "config2_mem_footprint_bytes",
    "mem_footprint_ratio",
)


@dataclass(frozen=True)
class MappingSource:
    input_dir: Path
    mapping_dir: Path
    workload_paths: tuple[Path, ...]
    manifest_path: Path | None
    manifest_payload: Mapping[str, Any] | None
    summary_path: Path | None
    summary_payload: Mapping[str, Any] | None


@dataclass(frozen=True)
class UniqueMapping:
    task_key: str
    workload_name: str | None
    workload_path: Path
    attempt_index: int


@dataclass(frozen=True)
class ActivationConfig:
    config_id: str
    config_label: str
    arch_file: Path
    search_mode: str
    activation: bool
    base_die: bool
    area_mm2: float
    channel_level: bool = False
    mapping_pool: str = "own"


@dataclass(frozen=True)
class EvaluationTask:
    task_key: str
    config_id: str
    config_label: str
    config_arch_file: Path
    search_mode: str
    activation: bool
    base_die: bool
    area_mm2: float
    attempt_index: int
    workload_name: str | None
    workload_path: Path
    channel_level: bool = False


class _ProgressCoordinator:
    def __init__(self, *, enabled: bool, workload_display_name: str) -> None:
        self.enabled = bool(enabled)
        self._workload_display_name = str(workload_display_name)
        self._stages = None
        self._stage_detail = None
        self._nested_detail = None
        self._nested_desc: str | None = None
        if self.enabled:
            self._stages = tqdm(
                total=4,
                desc=f"{self._workload_display_name} stages",
                unit="stage",
                position=0,
                leave=True,
            )

    def close(self) -> None:
        self._close_nested()
        self._close_stage_detail()
        if self._stages is not None:
            self._stages.close()
            self._stages = None

    def start_stage(self, *, stage_name: str, total: int, unit: str) -> None:
        if not self.enabled:
            return
        self._close_nested()
        self._close_stage_detail()
        if self._stages is not None:
            self._stages.set_postfix_str(f"current={stage_name}")
        self._stage_detail = tqdm(
            total=max(1, int(total)),
            desc=stage_name,
            unit=unit,
            position=1,
            leave=False,
        )

    def advance_stage(self, *, delta: int = 1, postfix: str | None = None) -> None:
        if not self.enabled or self._stage_detail is None:
            return
        target_total = int(self._stage_detail.total or 0)
        current_completed = int(getattr(self._stage_detail, "n", 0))
        remaining = max(0, target_total - current_completed)
        applied = max(0, min(int(delta), remaining))
        if applied > 0:
            self._stage_detail.update(applied)
        if postfix:
            self._stage_detail.set_postfix_str(postfix)

    def sync_stage(
        self,
        *,
        completed: int,
        total: int | None = None,
        postfix: str | None = None,
    ) -> None:
        if not self.enabled or self._stage_detail is None:
            return
        normalized_total = max(
            1,
            int(total if total is not None else int(self._stage_detail.total or 0)),
        )
        normalized_completed = max(0, min(int(completed), normalized_total))
        current_total = int(self._stage_detail.total or 0)
        current_completed = int(getattr(self._stage_detail, "n", 0))
        if current_total != normalized_total or normalized_completed < current_completed:
            self._stage_detail.reset(total=normalized_total)
            current_completed = int(getattr(self._stage_detail, "n", 0))
        if normalized_completed > current_completed:
            self._stage_detail.update(normalized_completed - current_completed)
        if postfix:
            self._stage_detail.set_postfix_str(postfix)

    def finish_stage(self, *, postfix: str | None = None) -> None:
        if not self.enabled:
            return
        self._close_nested()
        if self._stage_detail is not None:
            target_total = int(self._stage_detail.total or 0)
            current_completed = int(getattr(self._stage_detail, "n", 0))
            if current_completed < target_total:
                self._stage_detail.update(target_total - current_completed)
            if postfix:
                self._stage_detail.set_postfix_str(postfix)
        self._close_stage_detail()
        if self._stages is not None:
            self._stages.update(1)

    def update_search(
        self,
        *,
        workload_display_name: str,
        payload: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return
        total = max(1, int(payload["target_traces"]))
        completed = max(0, min(int(payload["legally_found"]), total))
        attempts = int(payload["attempts"])
        desc = f"Config 1 search/{workload_display_name} traces"
        if self._nested_detail is None or self._nested_desc != desc:
            self._close_nested()
            self._nested_detail = tqdm(
                total=total,
                desc=desc,
                unit="trace",
                position=2,
                leave=False,
            )
            self._nested_desc = desc
        current_total = int(self._nested_detail.total or 0)
        current_completed = int(getattr(self._nested_detail, "n", 0))
        if current_total != total or completed < current_completed:
            self._nested_detail.reset(total=total)
            current_completed = int(getattr(self._nested_detail, "n", 0))
        if completed > current_completed:
            self._nested_detail.update(completed - current_completed)
        self._nested_detail.set_postfix_str(
            f"attempts={attempts} valid={completed}/{total}"
        )

    def evaluation_progress_callback(self) -> Callable[[Mapping[str, Any]], None]:
        def _callback(payload: Mapping[str, Any]) -> None:
            total = max(1, int(payload["total"]))
            completed = max(0, min(int(payload["completed"]), total))
            self.sync_stage(
                completed=completed,
                total=total,
                postfix=f"completed={completed}/{total}",
            )

        return _callback

    def _close_stage_detail(self) -> None:
        if self._stage_detail is not None:
            self._stage_detail.close()
            self._stage_detail = None

    def _close_nested(self) -> None:
        if self._nested_detail is not None:
            self._nested_detail.close()
            self._nested_detail = None
            self._nested_desc = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HBM-PIM single-layer baseline mappings against activation "
            "configurations and export best-latency comparison artifacts."
        )
    )
    parser.add_argument(
        "--mapping-dirs",
        nargs="+",
        required=True,
        help=(
            "One or more workload roots containing spaces/c or explicit spaces/c "
            "directories for a single (arch, geometry, workload) pair."
        ),
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help='Architecture family name. Activation workload evaluation expects "hbm_pim".',
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for activation workload evaluation outputs.",
    )
    parser.add_argument(
        "--output-dir-file",
        type=str,
        default=None,
        help="Optional file path that receives the resolved output directory.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional analytical-model target override. Defaults to --arch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker count shared by Config 1 search and config evaluations.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0 disables progress output).",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=None,
        help=(
            "Config 1 random-search target trace count. Defaults to the selected "
            "source workload-space run's num_traces_per_space_delta."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Config 1 random-search max_attempts cap.",
    )
    parser.add_argument(
        "--max-system-sharing",
        type=int,
        default=None,
        help=(
            "Eval-only mapping-pool filter: keep only mappings whose Layout-Pragma "
            "sharing.system is <= this hierarchy level (drop the rest). Default: "
            "no filtering. Intended for the BG-level reduction variant, where "
            "the cross-PU sharing level must not exceed the reduction level."
        ),
    )
    parser.add_argument(
        "--inclusive-reduction-place",
        action="store_true",
        help=(
            "Feasibility-nested reduction-place mode: compose config1, "
            "config2, and config3 mapping pools inclusively, with the "
            "max-system-sharing cap applied only to the config3 front."
        ),
    )
    parser.add_argument(
        "--config1-arch-file",
        type=str,
        default=str(DEFAULT_CONFIG1_ARCH_FILE),
        help="Config 1 evaluation architecture YAML path.",
    )
    parser.add_argument(
        "--config2-arch-file",
        type=str,
        default=str(DEFAULT_CONFIG2_ARCH_FILE),
        help="Config 2 evaluation architecture YAML path.",
    )
    parser.add_argument(
        "--config3-arch-file",
        type=str,
        default=None,
        help=(
            "Optional Config 3 (channel-level reduction) architecture YAML path. "
            "Default: disabled (legacy two-config run). When provided (e.g. "
            f"{DEFAULT_CONFIG3_ARCH_FILE}), config3 re-evaluates the SAME mapping "
            "pool as config2 under the channel_level toggle, adding a third "
            "reduction-place front to the Pareto plots."
        ),
    )
    parser.add_argument(
        "--config3-mapping-dirs",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional: one or more mapping-source dirs for Config 3 "
            "(channel-level). When provided, Config 3 evaluates THIS pool (e.g. "
            "an L4 within-channel pool) instead of reusing Config 2's pool. "
            "Discovered/filtered exactly like the positional config2 mapping dirs "
            "(max-system-sharing applies). Requires --config3-arch-file."
        ),
    )
    parser.add_argument(
        "--config1-reuse-root",
        type=str,
        default=None,
        help=(
            "Eval-only: re-score Config 1 (PU-side) using the EXISTING searched "
            "mappings under this root instead of re-running the Config 1 search. "
            "Accepts a prior activation run dir, its random_search/config1/<token> "
            "dir, or a dir containing spaces/d. The Config 1 mapping pool is held "
            "fixed; only the analytical-model scores change."
        ),
    )
    return parser.parse_args()


def run(
    *,
    mapping_dirs: Sequence[str],
    arch_name: str,
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    output_dir_file: str | None = None,
    target: str | None = None,
    workers: int = 4,
    verbose: int = 1,
    num_traces: int | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_system_sharing: int | None = None,
    inclusive_reduction_place: bool = False,
    config1_arch_file: str = str(DEFAULT_CONFIG1_ARCH_FILE),
    config2_arch_file: str = str(DEFAULT_CONFIG2_ARCH_FILE),
    config3_arch_file: str | None = None,
    config3_mapping_dirs: Sequence[str] | None = None,
    config1_reuse_root: str | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    canonical_arch_name = _base_e2e._canonical_arch_name(arch_name)
    if canonical_arch_name != "hbm_pim":
        raise ValueError(
            "Activation workload-space latency evaluation currently supports only "
            f"hbm_pim, got {canonical_arch_name}."
        )

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
    source_num_traces = _resolve_source_num_traces(
        mapping_sources=mapping_sources,
        requested_num_traces=num_traces,
    )
    baseline_geometry_id, baseline_geometry_arch_file = _resolve_baseline_geometry(
        input_dirs=resolved_input_dirs,
        mapping_sources=mapping_sources,
        manifest_metadata=manifest_metadata,
    )
    workload_token = _resolve_workload_token(resolved_input_dirs)
    source_workload_path = _resolve_source_workload_path(workload_token=workload_token)
    workload_display_name = _resolve_display_name(
        input_dirs=resolved_input_dirs,
        manifest_metadata=manifest_metadata,
        workload_token=workload_token,
    )
    config1_arch_path = _base_eval._resolve_path(config1_arch_file).resolve()
    config2_arch_path = _base_eval._resolve_path(config2_arch_file).resolve()
    config3_arch_path = (
        _base_eval._resolve_path(config3_arch_file).resolve()
        if config3_arch_file is not None
        else None
    )
    if config3_arch_path is None and config3_mapping_dirs is not None:
        raise ValueError("--config3-mapping-dirs requires --config3-arch-file.")
    if inclusive_reduction_place:
        if config3_arch_path is None:
            raise ValueError(
                "--inclusive-reduction-place requires --config3-arch-file."
            )
        if config3_mapping_dirs is None:
            raise ValueError(
                "--inclusive-reduction-place requires --config3-mapping-dirs."
            )
        if config1_reuse_root is None:
            raise ValueError(
                "--inclusive-reduction-place requires --config1-reuse-root."
            )
    configs = _build_configs(
        config1_arch_file=config1_arch_path,
        config2_arch_file=config2_arch_path,
        config3_arch_file=config3_arch_path,
        config3_uses_own_pool=(
            config3_arch_path is not None and config3_mapping_dirs is not None
        ),
    )
    effective_target = target if target is not None else canonical_arch_name
    output_dir = _resolve_pipeline_output_dir(
        output_root=_base_eval._resolve_path(output_root),
        arch_name=canonical_arch_name,
        workload_token=workload_token,
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir_file is not None:
        _base_eval._write_output_dir_file(
            output_dir_file=output_dir_file,
            output_dir=output_dir,
        )

    progress = _ProgressCoordinator(
        enabled=verbose >= 1,
        workload_display_name=workload_display_name,
    )
    try:
        progress.start_stage(
            stage_name="Config 2 baseline",
            total=len(mapping_sources),
            unit="source",
        )
        config2_mapping_sources = _reuse_config2_mapping_sources(
            mapping_sources=mapping_sources,
            progress=progress,
        )
        progress.finish_stage(postfix=f"reused={len(config2_mapping_sources)}")

        if not inclusive_reduction_place:
            config2_unique_mappings, config2_mapping_summary = (
                _discover_unique_mappings(
                    config2_mapping_sources,
                    max_system_sharing=max_system_sharing,
                )
            )

        config3_input_dirs = None
        if config3_arch_path is not None and config3_mapping_dirs is not None:
            config3_input_dirs = tuple(
                _base_eval._resolve_path(p) for p in config3_mapping_dirs
            )
            config3_mapping_sources, _config3_manifest_metadata = (
                _discover_mapping_sources(config3_input_dirs)
            )
            if not inclusive_reduction_place:
                config3_unique_mappings, config3_mapping_summary = (
                    _discover_unique_mappings(
                        config3_mapping_sources,
                        max_system_sharing=max_system_sharing,
                    )
                )
        else:
            config3_mapping_sources = None
            config3_unique_mappings = None
            config3_mapping_summary = None

        progress.start_stage(stage_name="Config 1 search", total=1, unit="run")
        if config1_reuse_root is not None:
            config1_mapping_sources, config1_search_artifacts = (
                _reuse_config1_mapping_sources(
                    config1_reuse_root=_base_eval._resolve_path(config1_reuse_root),
                    workload_token=workload_token,
                )
            )
            progress.finish_stage(postfix="reused=1")
        else:
            config1_mapping_sources, config1_search_artifacts = _run_config1_search(
                arch_name=canonical_arch_name,
                baseline_geometry_arch_file=baseline_geometry_arch_file,
                workload_path=source_workload_path,
                workload_token=workload_token,
                output_dir=output_dir,
                num_traces=source_num_traces,
                max_attempts=max_attempts,
                workers=workers,
                progress=progress,
                workload_display_name=workload_display_name,
            )
            progress.finish_stage(postfix="searched=1")

        if inclusive_reduction_place:
            assert config3_mapping_sources is not None
            config1_unique_mappings, config1_mapping_summary = (
                _discover_unique_mappings(
                    config1_mapping_sources,
                    max_system_sharing=None,
                    allow_multiple_workload_names=True,
                )
            )
            config2_unique_mappings, config2_mapping_summary = (
                _discover_unique_mappings(
                    (
                        config1_mapping_sources
                        + config2_mapping_sources
                        + config3_mapping_sources
                    ),
                    max_system_sharing=None,
                    allow_multiple_workload_names=True,
                )
            )
            config3_unique_mappings, config3_mapping_summary = (
                _discover_unique_mappings(
                    config1_mapping_sources + config3_mapping_sources,
                    max_system_sharing=max_system_sharing,
                    allow_multiple_workload_names=True,
                )
            )
        else:
            config1_unique_mappings, config1_mapping_summary = (
                _discover_unique_mappings(
                    config1_mapping_sources,
                    max_system_sharing=max_system_sharing,
                )
            )

        unique_mappings_by_pool = {
            "config1": config1_unique_mappings,
            "config2": config2_unique_mappings,
        }
        if config3_unique_mappings is not None:
            unique_mappings_by_pool["config3"] = config3_unique_mappings

        evaluation_tasks = _build_evaluation_tasks(
            configs=configs,
            unique_mappings_by_pool=unique_mappings_by_pool,
        )
        progress.start_stage(
            stage_name="Evaluate mappings",
            total=len(evaluation_tasks),
            unit="task",
        )
        evaluation_progress_callback = _compose_progress_callbacks(
            progress.evaluation_progress_callback(),
            progress_callback,
        )
        evaluation_rows = _run_evaluations(
            tasks=evaluation_tasks,
            target=effective_target,
            workers=workers,
            verbose=0 if progress.enabled else verbose,
            progress_callback=evaluation_progress_callback,
        )
        progress.finish_stage(postfix=f"evaluated={len(evaluation_tasks)}")

        best_latency_rows = _select_best_latency_rows(
            configs=configs,
            evaluation_rows=evaluation_rows,
        )
        relative_metrics_row = _build_relative_metrics_row(
            workload_token=workload_token,
            workload_display_name=workload_display_name,
            baseline_geometry_id=baseline_geometry_id,
            best_latency_rows=best_latency_rows,
        )

        progress.start_stage(stage_name="Write reports", total=5, unit="artifact")
        evaluations_csv = output_dir / "evaluations.csv"
        _base_eval._write_csv(
            evaluations_csv,
            fieldnames=EVALUATIONS_FIELDNAMES,
            rows=evaluation_rows,
        )
        progress.advance_stage(postfix="evaluations.csv")
        best_latency_csv = output_dir / "best_latency.csv"
        _base_eval._write_csv(
            best_latency_csv,
            fieldnames=BEST_LATENCY_FIELDNAMES,
            rows=best_latency_rows,
        )
        progress.advance_stage(postfix="best_latency.csv")
        relative_metrics_csv = output_dir / "relative_metrics.csv"
        _base_eval._write_csv(
            relative_metrics_csv,
            fieldnames=RELATIVE_METRICS_FIELDNAMES,
            rows=(relative_metrics_row,),
        )
        progress.advance_stage(postfix="relative_metrics.csv")

        ordered_configs = [
            {
                "config_id": config.config_id,
                "config_label": config.config_label,
                "config_arch_file": str(config.arch_file),
                "search_mode": config.search_mode,
                "activation": bool(config.activation),
                "base_die": bool(config.base_die),
                "channel_level": bool(config.channel_level),
                "area_mm2": float(config.area_mm2),
            }
            for config in configs
        ]
        config_winners = {
            str(row["config_id"]): {
                "config_order": int(row["config_order"]),
                "config_id": str(row["config_id"]),
                "config_label": str(row["config_label"]),
                "config_arch_file": str(row["config_arch_file"]),
                "search_mode": str(row["search_mode"]),
                "activation": bool(row["activation"]),
                "base_die": bool(row["base_die"]),
                "channel_level": bool(row["channel_level"]),
                "area_mm2": float(row["area_mm2"]),
                "state_hash": str(row["state_hash"]),
                "latency_cycles": int(row["latency_cycles"]),
                "mem_footprint_bytes": int(row["mem_footprint_bytes"]),
                "energy": int(row["energy"]),
                "weighted_cost": float(row["weighted_cost"]),
                "attempt_index": int(row["attempt_index"]),
                "workload_name": str(row["workload_name"]),
                "workload_path": str(row["workload_path"]),
            }
            for row in best_latency_rows
        }
        invalid_points = sum(
            1 for row in evaluation_rows if str(row["status"]).strip() == "invalid"
        )
        error_points = sum(
            1 for row in evaluation_rows if str(row["status"]).strip() == "error"
        )
        successful_points = sum(
            1 for row in evaluation_rows if str(row["status"]).strip() == "ok"
        )
        config3_has_own_pool = config3_unique_mappings is not None
        if config3_has_own_pool:
            assert config3_mapping_summary is not None
        config3_mapping_dirs_payload = (
            [str(p) for p in config3_input_dirs]
            if config3_has_own_pool and config3_input_dirs is not None
            else None
        )
        config3_pool = (
            "own"
            if config3_has_own_pool
            else ("config2" if config3_arch_path is not None else None)
        )

        manifest_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "arch": canonical_arch_name,
            "target": effective_target,
            "workload_token": workload_token,
            "workload_display_name": workload_display_name,
            "output_dir": str(output_dir),
            "requested_mapping_dirs": [str(path) for path in resolved_input_dirs],
            "discovered_mapping_dirs": [
                str(source.mapping_dir) for source in config2_mapping_sources
            ],
            "source_manifest_jsons": [
                str(path)
                for path in dict.fromkeys(
                    source.manifest_path.resolve()
                    for source in mapping_sources
                    if source.manifest_path is not None
                )
            ],
            "source_summary_jsons": [
                str(path)
                for path in dict.fromkeys(
                    source.summary_path.resolve()
                    for source in mapping_sources
                    if source.summary_path is not None
                )
            ],
            "manifest_metadata": dict(manifest_metadata),
            "source_workload_path": str(source_workload_path),
            "baseline_geometry_id": baseline_geometry_id,
            "baseline_geometry_arch_file": str(baseline_geometry_arch_file),
            "config1_num_traces": int(source_num_traces),
            "config1_max_attempts": int(max_attempts),
            "max_system_sharing": max_system_sharing,
            "inclusive_reduction_place": bool(inclusive_reduction_place),
            "config3_mapping_dirs": config3_mapping_dirs_payload,
            "config3_pool": config3_pool,
            "ordered_configs": ordered_configs,
            "selected_mappings_total_by_config": {
                "config1": int(config1_mapping_summary["selected_mappings_total"]),
                "config2": int(config2_mapping_summary["selected_mappings_total"]),
            },
            "filtered_out_mappings_total_by_config": {
                "config1": int(
                    config1_mapping_summary["filtered_out_mappings_total"]
                ),
                "config2": int(
                    config2_mapping_summary["filtered_out_mappings_total"]
                ),
            },
            "unique_mappings_total_by_config": {
                "config1": int(config1_mapping_summary["unique_mappings_total"]),
                "config2": int(config2_mapping_summary["unique_mappings_total"]),
            },
            "duplicate_mappings_total_by_config": {
                "config1": int(config1_mapping_summary["duplicate_mappings_total"]),
                "config2": int(config2_mapping_summary["duplicate_mappings_total"]),
            },
            "unique_evaluation_tasks": int(len(evaluation_tasks)),
            "successful_points_total": int(successful_points),
            "discarded_invalid_points_total": int(invalid_points),
            "error_points_total": int(error_points),
            "config1_search_artifacts": config1_search_artifacts,
            "evaluations_csv": str(evaluations_csv.resolve()),
            "best_latency_csv": str(best_latency_csv.resolve()),
            "relative_metrics_csv": str(relative_metrics_csv.resolve()),
        }
        manifest_json = output_dir / "manifest.json"
        _base_eval._write_json(manifest_json, manifest_payload)
        progress.advance_stage(postfix="manifest.json")

        summary_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "arch": canonical_arch_name,
            "target": effective_target,
            "workload_token": workload_token,
            "workload_display_name": workload_display_name,
            "source_workload_path": str(source_workload_path),
            "baseline_geometry_id": baseline_geometry_id,
            "baseline_geometry_arch_file": str(baseline_geometry_arch_file),
            "config1_num_traces": int(source_num_traces),
            "config1_max_attempts": int(max_attempts),
            "max_system_sharing": max_system_sharing,
            "inclusive_reduction_place": bool(inclusive_reduction_place),
            "config3_mapping_dirs": config3_mapping_dirs_payload,
            "config3_pool": config3_pool,
            "ordered_configs": ordered_configs,
            "selected_mappings_total_by_config": {
                "config1": int(config1_mapping_summary["selected_mappings_total"]),
                "config2": int(config2_mapping_summary["selected_mappings_total"]),
            },
            "filtered_out_mappings_total_by_config": {
                "config1": int(
                    config1_mapping_summary["filtered_out_mappings_total"]
                ),
                "config2": int(
                    config2_mapping_summary["filtered_out_mappings_total"]
                ),
            },
            "unique_mappings_total_by_config": {
                "config1": int(config1_mapping_summary["unique_mappings_total"]),
                "config2": int(config2_mapping_summary["unique_mappings_total"]),
            },
            "duplicate_mappings_total_by_config": {
                "config1": int(config1_mapping_summary["duplicate_mappings_total"]),
                "config2": int(config2_mapping_summary["duplicate_mappings_total"]),
            },
            "unique_evaluation_tasks": int(len(evaluation_tasks)),
            "successful_points_total": int(successful_points),
            "discarded_invalid_points_total": int(invalid_points),
            "error_points_total": int(error_points),
            "evaluations_csv": str(evaluations_csv.resolve()),
            "best_latency_csv": str(best_latency_csv.resolve()),
            "relative_metrics_csv": str(relative_metrics_csv.resolve()),
            "config_winners": config_winners,
            "relative_metrics": dict(relative_metrics_row),
        }
        summary_json = output_dir / "summary.json"
        _base_eval._write_json(summary_json, summary_payload)
        progress.advance_stage(postfix="summary.json")
        progress.finish_stage(postfix="reports ready")

        report = {
            "status": "ok",
            "output_dir": str(output_dir),
            "manifest_json": str(manifest_json.resolve()),
            "summary_json": str(summary_json.resolve()),
            "evaluations_csv": str(evaluations_csv.resolve()),
            "best_latency_csv": str(best_latency_csv.resolve()),
            "relative_metrics_csv": str(relative_metrics_csv.resolve()),
            "selected_configs": [config.config_id for config in configs],
            "unique_evaluation_tasks": int(len(evaluation_tasks)),
        }
    finally:
        progress.close()
    if verbose >= 1:
        print(
            f"Completed activation workload-space latency evaluation for "
            f"{workload_display_name} ({canonical_arch_name})."
        )
        print(f"Output directory: {output_dir}")
    return report


def _build_configs(
    *,
    config1_arch_file: Path,
    config2_arch_file: Path,
    config3_arch_file: Path | None = None,
    config3_uses_own_pool: bool = False,
) -> tuple[ActivationConfig, ...]:
    Arch.from_yaml_file(config1_arch_file)
    Arch.from_yaml_file(config2_arch_file)
    configs = [
        ActivationConfig(
            config_id="config1",
            config_label="config1",
            arch_file=config1_arch_file,
            search_mode=CONFIG1_SEARCH_MODE,
            activation=True,
            base_die=False,
            channel_level=False,
            area_mm2=float(_load_area_mm2(config1_arch_file)),
            mapping_pool="own",
        ),
        ActivationConfig(
            config_id="config2",
            config_label="config2",
            arch_file=config2_arch_file,
            search_mode=CONFIG2_SEARCH_MODE,
            activation=True,
            base_die=True,
            channel_level=False,
            area_mm2=float(_load_area_mm2(config2_arch_file)),
            mapping_pool="own",
        ),
    ]
    if config3_arch_file is not None:
        Arch.from_yaml_file(config3_arch_file)
        configs.append(
            ActivationConfig(
                config_id="config3",
                config_label="config3",
                arch_file=config3_arch_file,
                search_mode=CONFIG3_SEARCH_MODE,
                activation=True,
                base_die=False,
                channel_level=True,
                # config3 (channel-level) re-evaluates config2's pool by default,
                # or its own distinct pool when --config3-mapping-dirs is set.
                mapping_pool="config3" if config3_uses_own_pool else "config2",
                area_mm2=float(_load_area_mm2(config3_arch_file)),
            )
        )
    return tuple(configs)


def _compose_progress_callbacks(
    *callbacks: Callable[[Mapping[str, Any]], None] | None,
) -> Callable[[Mapping[str, Any]], None] | None:
    active_callbacks = tuple(callback for callback in callbacks if callback is not None)
    if not active_callbacks:
        return None

    def _callback(payload: Mapping[str, Any]) -> None:
        for callback in active_callbacks:
            callback(payload)

    return _callback


def _discover_mapping_sources(
    input_dirs: Sequence[Path],
) -> tuple[tuple[MappingSource, ...], dict[str, str]]:
    discovered_sources: dict[Path, MappingSource] = {}
    metadata: dict[str, str] = {}
    for input_dir in input_dirs:
        (
            mapping_dir,
            manifest_path,
            manifest_payload,
            summary_path,
            summary_payload,
        ) = _resolve_mapping_dir(input_dir)
        workload_paths = _base_eval._collect_step_workload_paths(mapping_dir)
        if not workload_paths:
            raise ValueError(f"No workload YAML files found under {mapping_dir}.")
        source = MappingSource(
            input_dir=input_dir.resolve(),
            mapping_dir=mapping_dir.resolve(),
            workload_paths=workload_paths,
            manifest_path=manifest_path,
            manifest_payload=manifest_payload,
            summary_path=summary_path,
            summary_payload=summary_payload,
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
        raise ValueError("No direct-(c) mapping directories were discovered.")
    return (
        tuple(
            sorted(
                discovered_sources.values(),
                key=lambda source: str(source.mapping_dir),
            )
        ),
        metadata,
    )


def _resolve_mapping_dir(
    input_dir: Path,
) -> tuple[
    Path,
    Path | None,
    Mapping[str, Any] | None,
    Path | None,
    Mapping[str, Any] | None,
]:
    if not input_dir.is_dir():
        raise ValueError(f"Mapping directory does not exist: {input_dir}")
    if input_dir.name == "c" and input_dir.parent.name == "spaces":
        root_dir = input_dir.parent.parent
        manifest_path, manifest_payload = _base_eval._load_optional_manifest(
            root_dir / "manifest.json"
        )
        summary_path, summary_payload = _base_eval._load_optional_manifest(
            root_dir / "summary.json"
        )
        return (
            input_dir.resolve(),
            manifest_path,
            manifest_payload,
            summary_path,
            summary_payload,
        )

    mapping_dir = input_dir / "spaces" / "c"
    if not mapping_dir.is_dir():
        raise ValueError(
            f"Unsupported mapping directory '{input_dir}'. Expected a workload "
            "root containing spaces/c or an explicit spaces/c directory."
        )
    manifest_path, manifest_payload = _base_eval._load_optional_manifest(
        input_dir / "manifest.json"
    )
    summary_path, summary_payload = _base_eval._load_optional_manifest(
        input_dir / "summary.json"
    )
    return (
        mapping_dir.resolve(),
        manifest_path,
        manifest_payload,
        summary_path,
        summary_payload,
    )


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
        value = _base_eval._normalize_manifest_value(
            "arch_name", payload.get("arch_name")
        )
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
    inferred_geometry_ids: set[str] = set()
    inferred_workload_tokens: set[str] = set()
    for input_dir in input_dirs:
        inferred_arch_name, inferred_geometry_id, inferred_workload_token = (
            _infer_input_pair_metadata(input_dir)
        )
        if inferred_arch_name is not None:
            inferred_arch_names.add(inferred_arch_name)
        if inferred_geometry_id is not None:
            inferred_geometry_ids.add(inferred_geometry_id)
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
    if len(inferred_geometry_ids) > 1:
        formatted = ", ".join(sorted(inferred_geometry_ids))
        raise ValueError(
            f"Discovered mapping inputs span multiple geometry tokens: {formatted}"
        )
    if inferred_geometry_ids and inferred_geometry_ids != {BASELINE_GEOMETRY_ID}:
        formatted = ", ".join(sorted(inferred_geometry_ids))
        raise ValueError(
            f"Activation workload-space latency evaluation expects baseline geometry "
            f"{BASELINE_GEOMETRY_ID!r}, got {formatted}."
        )
    if len(inferred_workload_tokens) > 1:
        formatted = ", ".join(sorted(inferred_workload_tokens))
        raise ValueError(
            f"Discovered mapping inputs span multiple workload tokens: {formatted}"
        )


def _infer_input_pair_metadata(
    input_dir: Path,
) -> tuple[str | None, str | None, str | None]:
    resolved_input_dir = input_dir.resolve()
    if resolved_input_dir.name == "c" and resolved_input_dir.parent.name == "spaces":
        workload_token = _nonempty_token(resolved_input_dir.parent.parent.name)
        geometry_id = _nonempty_token(resolved_input_dir.parent.parent.parent.name)
        arch_name = _try_canonical_arch_name(
            resolved_input_dir.parent.parent.parent.parent.name
        )
        return arch_name, geometry_id, workload_token

    workload_token = _nonempty_token(resolved_input_dir.name)
    geometry_id = _nonempty_token(resolved_input_dir.parent.name)
    arch_name = _try_canonical_arch_name(resolved_input_dir.parent.parent.name)
    return arch_name, geometry_id, workload_token


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


def _resolve_source_workload_path(
    *,
    workload_token: str,
) -> Path:
    normalized_token = str(workload_token).strip()
    if not normalized_token:
        raise ValueError("workload_token cannot be empty when resolving Config 1 source.")

    filename = f"{normalized_token}-unindp.yaml"
    candidates = tuple(
        root.resolve() / filename for root in CONFIG1_SOURCE_WORKLOAD_ROOTS
    )
    matches = tuple(path for path in candidates if path.is_file())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        formatted = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Ambiguous Config 1 source workload for token {normalized_token!r}: {formatted}"
        )
    searched = ", ".join(str(path) for path in candidates)
    raise ValueError(
        f"Unable to resolve Config 1 source workload for token {normalized_token!r}. "
        f"Searched: {searched}"
    )


def _resolve_source_num_traces(
    *,
    mapping_sources: Sequence[MappingSource],
    requested_num_traces: int | None,
) -> int:
    if requested_num_traces is not None:
        if int(requested_num_traces) <= 0:
            raise ValueError(
                f"--num-traces must be positive, got {requested_num_traces}."
            )
        return int(requested_num_traces)

    discovered_values: list[int] = []
    for source in mapping_sources:
        payload = source.summary_payload
        if not isinstance(payload, Mapping):
            continue
        raw_value = payload.get("num_traces_per_space_delta")
        if raw_value is None:
            continue
        discovered_values.append(int(raw_value))
    unique_values = tuple(dict.fromkeys(discovered_values))
    if len(unique_values) > 1:
        formatted = ", ".join(str(value) for value in unique_values)
        raise ValueError(
            f"Conflicting num_traces_per_space_delta values were discovered: {formatted}"
        )
    if len(unique_values) == 1:
        resolved = unique_values[0]
        if resolved <= 0:
            raise ValueError(
                f"Derived num_traces_per_space_delta must be positive, got {resolved}."
            )
        return resolved
    return DEFAULT_FALLBACK_NUM_TRACES


def _resolve_baseline_geometry(
    *,
    input_dirs: Sequence[Path],
    mapping_sources: Sequence[MappingSource],
    manifest_metadata: Mapping[str, Any],
) -> tuple[str, Path]:
    geometry_ids: set[str] = set()
    for input_dir in input_dirs:
        _, inferred_geometry_id, _ = _infer_input_pair_metadata(input_dir)
        if inferred_geometry_id is not None:
            geometry_ids.add(inferred_geometry_id)

    arch_files: list[Path] = []
    raw_arch_file = manifest_metadata.get("arch_file")
    if raw_arch_file is not None and str(raw_arch_file).strip():
        arch_files.append(_common.resolve_workspace_path(str(raw_arch_file)).resolve())
    for source in mapping_sources:
        payload = source.summary_payload
        if not isinstance(payload, Mapping):
            continue
        candidate = payload.get("arch_file")
        if candidate is None or not str(candidate).strip():
            continue
        arch_files.append(_common.resolve_workspace_path(str(candidate)).resolve())
    unique_arch_files = tuple(dict.fromkeys(arch_files))
    if len(unique_arch_files) > 1:
        formatted = ", ".join(str(path) for path in unique_arch_files)
        raise ValueError(f"Conflicting baseline arch_file values were discovered: {formatted}")
    if not unique_arch_files:
        raise ValueError("Unable to resolve the baseline geometry arch_file.")
    baseline_arch_file = unique_arch_files[0]
    geometry_ids.add(baseline_arch_file.stem)
    if len(geometry_ids) != 1:
        formatted = ", ".join(sorted(geometry_ids))
        raise ValueError(f"Conflicting baseline geometry ids were discovered: {formatted}")
    geometry_id = next(iter(geometry_ids))
    if geometry_id != BASELINE_GEOMETRY_ID:
        raise ValueError(
            f"Activation workload-space latency evaluation expects baseline geometry "
            f"{BASELINE_GEOMETRY_ID!r}, got {geometry_id!r}."
        )
    return geometry_id, baseline_arch_file


def _resolve_workload_token(input_dirs: Sequence[Path]) -> str:
    tokens = [
        workload_token
        for _, _, workload_token in (
            _infer_input_pair_metadata(path) for path in input_dirs
        )
        if workload_token is not None
    ]
    unique_tokens = tuple(dict.fromkeys(tokens))
    if len(unique_tokens) != 1:
        formatted = ", ".join(unique_tokens)
        raise ValueError(f"Expected exactly one workload token, got: {formatted}")
    return unique_tokens[0]


def _resolve_display_name(
    *,
    input_dirs: Sequence[Path],
    manifest_metadata: Mapping[str, Any],
    workload_token: str,
) -> str:
    manifest_name = str(manifest_metadata.get("workload_name", "")).strip()
    if manifest_name:
        return manifest_name
    input_tokens = [_base_eval._input_root_token(path) for path in input_dirs]
    if input_tokens and len(set(input_tokens)) == 1:
        return input_tokens[0]
    return workload_token


def _reuse_config2_mapping_sources(
    *,
    mapping_sources: Sequence[MappingSource],
    progress: _ProgressCoordinator | None = None,
) -> tuple[MappingSource, ...]:
    sources: list[MappingSource] = []
    for source in mapping_sources:
        sources.append(source)
        if progress is not None:
            progress.advance_stage(postfix=f"ready={source.mapping_dir.parent.parent.name}")
    return tuple(sources)


def _reuse_config1_mapping_sources(
    *,
    config1_reuse_root: Path,
    workload_token: str,
) -> tuple[tuple[MappingSource, ...], dict[str, Any]]:
    """Eval-only Config 1: build a MappingSource from EXISTING searched PU-side
    mappings (prior run's ``random_search/config1/<token>/spaces/d``) rather than
    re-running the Config 1 search. Mirrors the source built by ``_run_config1_search``.
    """
    sanitized = _base_eval._sanitize_token(workload_token)
    candidate_search_dirs = (
        config1_reuse_root,
        config1_reuse_root / "random_search" / "config1" / sanitized,
        config1_reuse_root / "config1" / sanitized,
    )
    # The Config 1 search step changed over time (April runs landed mappings under
    # spaces/c; current runs use spaces/d), so accept whichever holds the mappings.
    search_dir: Path | None = None
    mapping_dir: Path | None = None
    workload_paths: tuple[Path, ...] = ()
    for candidate in candidate_search_dirs:
        spaces_dir = candidate / "spaces"
        if not spaces_dir.is_dir():
            continue
        for step_id in ("d", "c"):
            step_dir = spaces_dir / step_id
            if not step_dir.is_dir():
                continue
            paths = _base_eval._collect_step_workload_paths(step_dir)
            if paths:
                search_dir = candidate.resolve()
                mapping_dir = step_dir.resolve()
                workload_paths = paths
                break
        if mapping_dir is not None:
            break
    if mapping_dir is None or search_dir is None:
        raise ValueError(
            "--config1-reuse-root: no Config 1 'spaces/{d,c}' mappings found for "
            f"workload '{workload_token}' under {config1_reuse_root}."
        )
    manifest_path = (search_dir / "manifest.json").resolve()
    summary_path = (search_dir / "summary.json").resolve()
    source = MappingSource(
        input_dir=search_dir,
        mapping_dir=mapping_dir,
        workload_paths=workload_paths,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        manifest_payload=_base_eval._load_optional_manifest(manifest_path)[1],
        summary_path=summary_path if summary_path.is_file() else None,
        summary_payload=_base_eval._load_optional_manifest(summary_path)[1],
    )
    return (source,), {
        "search_mode": "reused",
        "reused_from": str(search_dir),
        "selected_step_id": mapping_dir.name,
        "selected_step_dir": str(mapping_dir),
        f"space_{mapping_dir.name}_dir": str(mapping_dir),
    }


def _run_config1_search(
    *,
    arch_name: str,
    baseline_geometry_arch_file: Path,
    workload_path: Path,
    workload_token: str,
    output_dir: Path,
    num_traces: int,
    max_attempts: int,
    workers: int,
    progress: _ProgressCoordinator | None = None,
    workload_display_name: str,
) -> tuple[tuple[MappingSource, ...], dict[str, Any]]:
    search_output_dir = (
        output_dir / "random_search" / "config1" / _base_eval._sanitize_token(workload_token)
    ).resolve()
    constraint = OutputDimensionFactorizationConstraint(
        tensor_name="Out",
        allowed_level_ids=(0, 1),
        force_other_levels_to_one=True,
    )
    summary = _run_config1_random_search(
        workload_path=workload_path,
        arch_name=arch_name,
        baseline_geometry_arch_file=baseline_geometry_arch_file,
        search_output_dir=search_output_dir,
        num_traces=num_traces,
        max_attempts=max_attempts,
        workers=workers,
        output_dim_constraint=constraint,
        progress_callback=(
            None
            if progress is None
            else (
                lambda payload: progress.update_search(
                    workload_display_name=workload_display_name,
                    payload=payload,
                )
            )
        ),
    )
    if progress is not None:
        progress.advance_stage(postfix=f"done={workload_token}")
    summary_path = (search_output_dir / "summary.json").resolve()
    manifest_path = (search_output_dir / "manifest.json").resolve()
    mapping_dir = (search_output_dir / "spaces" / "d").resolve()
    workload_paths = _base_eval._collect_step_workload_paths(mapping_dir)
    if not workload_paths:
        raise ValueError(f"No Config 1 mapping YAML files were found under {mapping_dir}.")
    source = MappingSource(
        input_dir=search_output_dir,
        mapping_dir=mapping_dir,
        workload_paths=workload_paths,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        manifest_payload=_base_eval._load_optional_manifest(manifest_path)[1],
        summary_path=summary_path if summary_path.is_file() else None,
        summary_payload=_base_eval._load_optional_manifest(summary_path)[1],
    )
    return (source,), {
        "output_dir": str(Path(str(summary["output_dir"])).resolve()),
        "summary_json": str(summary_path),
        "manifest_json": str(manifest_path),
        "selected_step_id": "d",
        "selected_step_dir": str(mapping_dir),
        "space_d_dir": str(mapping_dir),
        "num_traces": int(num_traces),
        "max_attempts": int(max_attempts),
    }


def _run_config1_random_search(
    *,
    workload_path: Path,
    arch_name: str,
    baseline_geometry_arch_file: Path,
    search_output_dir: Path,
    num_traces: int,
    max_attempts: int,
    workers: int,
    output_dim_constraint: OutputDimensionFactorizationConstraint,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    kwargs = {
        "workload_file": str(workload_path),
        "arch_name": arch_name,
        "arch_file": str(baseline_geometry_arch_file),
        "out_dir": str(search_output_dir),
        "num_traces": int(num_traces),
        "spaces": ("d",),
        "max_attempts": int(max_attempts),
        "workers": int(workers),
        "batch_attempts": 1,
        "show_progress": False,
        "direct_d_search": True,
        "output_dim_constraint": output_dim_constraint,
        "progress_callback": progress_callback,
    }
    try:
        return run_workload_space_random_search.run(**kwargs)
    except ValueError as exc:
        if str(exc) != _base_e2e.INCOMPATIBLE_SEARCH_OUTPUT_ERROR:
            raise
        if search_output_dir.exists():
            shutil.rmtree(search_output_dir)
        return run_workload_space_random_search.run(**kwargs)


def _discover_unique_mappings(
    mapping_sources: Sequence[MappingSource],
    *,
    max_system_sharing: int | None = None,
    allow_multiple_workload_names: bool = False,
) -> tuple[tuple[UniqueMapping, ...], dict[str, int]]:
    unique_mappings_by_key: dict[str, UniqueMapping] = {}
    workload_names: set[str] = set()
    selected_mappings_total = 0
    filtered_out_mappings_total = 0
    for source in mapping_sources:
        for attempt_index, workload_path in enumerate(source.workload_paths):
            unique_mapping, sharing_system = _build_unique_mapping(
                workload_path=workload_path,
                attempt_index=attempt_index,
            )
            if (
                max_system_sharing is not None
                and sharing_system is not None
                and sharing_system > max_system_sharing
            ):
                filtered_out_mappings_total += 1
                continue
            selected_mappings_total += 1
            if unique_mapping.workload_name is not None:
                workload_names.add(unique_mapping.workload_name)
            unique_mappings_by_key.setdefault(unique_mapping.task_key, unique_mapping)

    if not unique_mappings_by_key:
        raise ValueError("No direct-(c) mappings were discovered.")
    # The inclusive (feasibility-nested) composition unions the config1 reuse pool
    # with the config2/L4 pools. Those pools differ only in a cosmetic workload-name
    # convention for the SAME workload (e.g. the April run names it 'bmm_llama256_3'
    # while the newer runs use 'bmm_llama256-3'), so the single-name guard must be
    # relaxed there. The legacy per-pool discovery keeps the strict guard.
    if not allow_multiple_workload_names and len(workload_names) > 1:
        formatted = ", ".join(sorted(workload_names))
        raise ValueError(f"Discovered multiple workload names in one run: {formatted}")

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
        "filtered_out_mappings_total": int(filtered_out_mappings_total),
        "unique_mappings_total": int(unique_total),
        "duplicate_mappings_total": int(selected_mappings_total - unique_total),
    }


def _build_unique_mapping(
    *,
    workload_path: Path,
    attempt_index: int,
) -> tuple[UniqueMapping, int | None]:
    try:
        workload = Workload.from_file(workload_path)
        sharing_system = None
        if workload.layout is not None and workload.layout.sharing is not None:
            sharing_system = int(workload.layout.sharing.system)
        return (
            UniqueMapping(
                task_key=f"state:{_base_eval._state_hash(workload.to_dict())}",
                workload_name=workload.name,
                workload_path=workload_path.resolve(),
                attempt_index=int(attempt_index),
            ),
            sharing_system,
        )
    except Exception:
        return (
            UniqueMapping(
                task_key=f"path:{workload_path.resolve()}",
                workload_name=None,
                workload_path=workload_path.resolve(),
                attempt_index=int(attempt_index),
            ),
            None,
        )


def _build_evaluation_tasks(
    *,
    configs: Sequence[ActivationConfig],
    unique_mappings_by_pool: Mapping[str, Sequence[UniqueMapping]],
) -> tuple[EvaluationTask, ...]:
    tasks: list[EvaluationTask] = []
    for config in configs:
        # config3 can reuse config2's pool or point at its own distinct pool;
        # config1/config2 use their own pool.
        pool_key = config.config_id if config.mapping_pool == "own" else config.mapping_pool
        for mapping in unique_mappings_by_pool[pool_key]:
            tasks.append(
                EvaluationTask(
                    task_key=f"{config.config_id}:{mapping.task_key}",
                    config_id=config.config_id,
                    config_label=config.config_label,
                    config_arch_file=config.arch_file,
                    search_mode=config.search_mode,
                    activation=config.activation,
                    base_die=config.base_die,
                    channel_level=config.channel_level,
                    area_mm2=config.area_mm2,
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
    _base_eval._emit_progress_callback(progress_callback, completed=0, total=len(tasks))
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
                    executor.submit(_evaluate_task, tasks[next_task_index], target)
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
                            executor.submit(_evaluate_task, tasks[next_task_index], target)
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
    return f"Evaluating activation workload mappings [{int(completed)}/{int(total)}]"


def _cache_config_override_for_evaluation(task: EvaluationTask) -> bool | None:
    # config2 (base-die) and config3 (channel-level) both re-cost the
    # baseline-reference pool with cache effects enabled.
    if task.config_id in ("config2", "config3"):
        return True
    return None


def _evaluate_task(
    task: EvaluationTask,
    target: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        arch = _base_eval._load_arch_for_worker(str(task.config_arch_file))
        workload = Workload.from_file(task.workload_path)
        workload_payload = workload.to_dict()
        cache_config_override = _cache_config_override_for_evaluation(task)
        model = AnalyticalModel(
            workload,
            arch,
            activation=task.activation,
            base_die=task.base_die,
            channel_level=task.channel_level,
            cache_config_override=cache_config_override,
        )
        if not model.verify_constraints(verbose=0):
            return {
                "config_id": task.config_id,
                "config_label": task.config_label,
                "config_arch_file": str(task.config_arch_file),
                "search_mode": task.search_mode,
                "activation": bool(task.activation),
                "base_die": bool(task.base_die),
                "channel_level": bool(task.channel_level),
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
        report = evaluate_workload_analytical_report(
            workload,
            arch,
            target=target,
            latency_coeff=1.0,
            mem_footprint_coeff=1.0,
            energy_coeff=1.0,
            activation=task.activation,
            base_die=task.base_die,
            channel_level=task.channel_level,
            cache_config_override=cache_config_override,
            require_constraints=False,
            refresh=False,
        )
        if report is None:
            raise RuntimeError(
                "Analytical evaluation unexpectedly returned None after constraints "
                "passed."
            )
        return {
            "config_id": task.config_id,
            "config_label": task.config_label,
            "config_arch_file": str(task.config_arch_file),
            "search_mode": task.search_mode,
            "activation": bool(task.activation),
            "base_die": bool(task.base_die),
            "channel_level": bool(task.channel_level),
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
            "config_id": task.config_id,
            "config_label": task.config_label,
            "config_arch_file": str(task.config_arch_file),
            "search_mode": task.search_mode,
            "activation": bool(task.activation),
            "base_die": bool(task.base_die),
            "channel_level": bool(task.channel_level),
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
    configs: Sequence[ActivationConfig],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows_by_config: dict[str, list[dict[str, Any]]] = {
        config.config_id: [] for config in configs
    }
    for row in evaluation_rows:
        if str(row.get("status", "")).strip() != "ok":
            continue
        rows_by_config.setdefault(str(row["config_id"]), []).append(dict(row))

    best_rows: list[dict[str, Any]] = []
    for config_order, config in enumerate(configs, start=1):
        config_rows = rows_by_config.get(config.config_id, [])
        deduped_rows = _dedupe_success_rows(config_rows)
        if not deduped_rows:
            raise RuntimeError(
                f"No successful mappings were produced for activation config "
                f"'{config.config_id}'."
            )
        best_row = min(deduped_rows, key=_best_latency_sort_key)
        best_rows.append({"config_order": int(config_order), **best_row})
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


def _build_relative_metrics_row(
    *,
    workload_token: str,
    workload_display_name: str,
    baseline_geometry_id: str,
    best_latency_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    winners_by_config = {str(row["config_id"]): dict(row) for row in best_latency_rows}
    try:
        config1 = winners_by_config["config1"]
        config2 = winners_by_config["config2"]
    except KeyError as exc:
        raise RuntimeError(
            "Expected best-latency rows for config1 and config2."
        ) from exc

    return {
        "workload_token": str(workload_token),
        "workload_display_name": str(workload_display_name),
        "baseline_geometry_id": str(baseline_geometry_id),
        "config1_state_hash": str(config1["state_hash"]),
        "config2_state_hash": str(config2["state_hash"]),
        "config1_latency_cycles": int(config1["latency_cycles"]),
        "config2_latency_cycles": int(config2["latency_cycles"]),
        "latency_ratio": float(config1["latency_cycles"]) / float(config2["latency_cycles"]),
        "config1_energy": int(config1["energy"]),
        "config2_energy": int(config2["energy"]),
        "energy_ratio": float(config1["energy"]) / float(config2["energy"]),
        "config1_area_mm2": float(config1["area_mm2"]),
        "config2_area_mm2": float(config2["area_mm2"]),
        "area_ratio": float(config1["area_mm2"]) / float(config2["area_mm2"]),
        "config1_mem_footprint_bytes": int(config1["mem_footprint_bytes"]),
        "config2_mem_footprint_bytes": int(config2["mem_footprint_bytes"]),
        "mem_footprint_ratio": float(config1["mem_footprint_bytes"])
        / float(config2["mem_footprint_bytes"]),
    }


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


def main() -> int:
    args = parse_args()
    run(
        mapping_dirs=args.mapping_dirs,
        arch_name=args.arch,
        output_root=args.output_root,
        output_dir_file=args.output_dir_file,
        target=args.target,
        workers=args.workers,
        verbose=args.verbose,
        num_traces=args.num_traces,
        max_attempts=args.max_attempts,
        max_system_sharing=args.max_system_sharing,
        inclusive_reduction_place=args.inclusive_reduction_place,
        config1_arch_file=args.config1_arch_file,
        config2_arch_file=args.config2_arch_file,
        config3_arch_file=args.config3_arch_file,
        config3_mapping_dirs=args.config3_mapping_dirs,
        config1_reuse_root=args.config1_reuse_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
