"""Shared workload evaluation helpers.

This module hosts the analytical workload evaluator and the objective-metric
helpers shared by the experiment CLIs (Pareto eval, random search,
geometry/buffer/interconnect/activation sweeps) and the final plot scripts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from pnmax.analytical_model import AnalyticalModel
from pnmax.database import Arch, LayoutCacheConfig, Workload

__all__ = [
    "SAObjectiveMetricSet",
    "evaluate_workload_analytical_report",
    "_coerce_objective_metric_set",
    "_objective_metric_set_from_metrics",
    "_compute_weighted_cost",
    "_resolve_objective_denominator",
]


@dataclass(frozen=True)
class SAObjectiveMetricSet:
    latency_cycles: int
    mem_footprint_bytes: int
    energy: int


def evaluate_workload_analytical_report(
    workload: Workload,
    arch: Arch,
    *,
    target: str | None,
    latency_coeff: float,
    mem_footprint_coeff: float,
    energy_coeff: float,
    cache_config_override: LayoutCacheConfig | Mapping[str, Any] | bool | None = None,
    activation: bool = False,
    base_die: bool = False,
    channel_level: bool = False,
    objective_start_metrics: SAObjectiveMetricSet | Mapping[str, Any] | None = None,
    objective_reference_metrics: SAObjectiveMetricSet | Mapping[str, Any] | None = None,
    require_constraints: bool = True,
    refresh: bool = True,
) -> dict[str, Any] | None:
    started = time.perf_counter()
    if refresh:
        workload.refresh()

    model = AnalyticalModel(
        workload,
        arch,
        activation=activation,
        base_die=base_die,
        channel_level=channel_level,
        cache_config_override=cache_config_override,
    )
    if require_constraints and not model.verify_constraints():
        return None

    latency_cycles = int(model.latency(target=target))
    mem_footprint_bytes = int(model.mem_footprint())
    energy = int(model.energy(target=target))
    weighted_cost = _compute_weighted_cost(
        latency_cycles=latency_cycles,
        mem_footprint_bytes=mem_footprint_bytes,
        energy=energy,
        latency_coeff=latency_coeff,
        mem_footprint_coeff=mem_footprint_coeff,
        energy_coeff=energy_coeff,
        objective_start_metrics=objective_start_metrics,
        objective_reference_metrics=objective_reference_metrics,
    )
    return {
        "status": "ok",
        "latency_cycles": latency_cycles,
        "mem_footprint_bytes": mem_footprint_bytes,
        "energy": energy,
        "weighted_cost": float(weighted_cost),
        "runtime_s": float(time.perf_counter() - started),
        "breakdown": dict(model.breakdown),
    }


def _coerce_objective_metric_set(
    raw_metrics: SAObjectiveMetricSet | Mapping[str, Any] | None,
    *,
    field_name: str,
) -> SAObjectiveMetricSet | None:
    if raw_metrics is None:
        return None
    if isinstance(raw_metrics, SAObjectiveMetricSet):
        return raw_metrics
    if not isinstance(raw_metrics, Mapping):
        raise TypeError(
            f"{field_name} must be a mapping with latency_cycles, "
            f"mem_footprint_bytes, and energy, got {type(raw_metrics).__name__}."
        )

    try:
        latency_cycles = int(raw_metrics["latency_cycles"])
        mem_footprint_bytes = int(raw_metrics["mem_footprint_bytes"])
        energy = int(raw_metrics["energy"])
    except KeyError as exc:
        raise ValueError(
            f"{field_name} must define latency_cycles, mem_footprint_bytes, and energy."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must contain integer-compatible values."
        ) from exc

    return SAObjectiveMetricSet(
        latency_cycles=latency_cycles,
        mem_footprint_bytes=mem_footprint_bytes,
        energy=energy,
    )


def _objective_metric_set_from_metrics(
    *,
    latency_cycles: int,
    mem_footprint_bytes: int,
    energy: int,
) -> SAObjectiveMetricSet:
    return SAObjectiveMetricSet(
        latency_cycles=int(latency_cycles),
        mem_footprint_bytes=int(mem_footprint_bytes),
        energy=int(energy),
    )


def _compute_weighted_cost(
    *,
    latency_cycles: int,
    mem_footprint_bytes: int,
    energy: int,
    latency_coeff: float,
    mem_footprint_coeff: float,
    energy_coeff: float,
    objective_start_metrics: SAObjectiveMetricSet | Mapping[str, Any] | None,
    objective_reference_metrics: SAObjectiveMetricSet | Mapping[str, Any] | None,
) -> float:
    start_metrics = _coerce_objective_metric_set(
        objective_start_metrics,
        field_name="objective_start_metrics",
    )
    if start_metrics is None:
        start_metrics = _objective_metric_set_from_metrics(
            latency_cycles=latency_cycles,
            mem_footprint_bytes=mem_footprint_bytes,
            energy=energy,
        )
    reference_metrics = _coerce_objective_metric_set(
        objective_reference_metrics,
        field_name="objective_reference_metrics",
    )

    latency_denominator = _resolve_objective_denominator(
        start_value=start_metrics.latency_cycles,
        reference_value=(
            reference_metrics.latency_cycles if reference_metrics is not None else None
        ),
        metric_name="latency_cycles",
    )
    mem_denominator = _resolve_objective_denominator(
        start_value=start_metrics.mem_footprint_bytes,
        reference_value=(
            reference_metrics.mem_footprint_bytes
            if reference_metrics is not None
            else None
        ),
        metric_name="mem_footprint_bytes",
    )
    energy_denominator = _resolve_objective_denominator(
        start_value=start_metrics.energy,
        reference_value=(
            reference_metrics.energy if reference_metrics is not None else None
        ),
        metric_name="energy",
    )

    latency_term = math.log1p(float(latency_cycles) / latency_denominator)
    mem_term = float(mem_footprint_bytes) / mem_denominator
    energy_term = math.log1p(float(energy) / energy_denominator)
    return (
        float(latency_coeff) * latency_term
        + float(mem_footprint_coeff) * mem_term
        + float(energy_coeff) * energy_term
    )


def _resolve_objective_denominator(
    *,
    start_value: int,
    reference_value: int | None,
    metric_name: str,
) -> float:
    denominator = float(start_value)
    if reference_value is not None:
        denominator = (float(start_value) + float(reference_value)) / 2.0
    if denominator <= 0.0:
        raise ValueError(
            f"Objective normalization denominator for {metric_name} must be > 0, "
            f"got {denominator}."
        )
    return denominator
