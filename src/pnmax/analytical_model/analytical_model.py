from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping

from pnmax.database import Arch, ArchCacheSpec, LayoutCacheConfig, Workload

from .cache_sim import (
    CacheStats,
    CacheGroupAllocation,
    CacheMode,
    GroupedCacheSimulationResult,
    simulate_grouped_cache_for_workload,
)


class TensorType(Enum):
    F = "F"
    In = "In"
    Out = "Out"


from .row_changes import TensorRowChangeEstimate, estimate_tensor_row_changes


class AnalyticalModel:
    def __init__(
        self,
        workload: Workload,
        arch: Arch,
        clamp_row: bool = False,
        activation: bool = False,
        base_die: bool = False,
        channel_level: bool = False,
        cache_config_override: LayoutCacheConfig | Mapping[str, Any] | bool | None = None,
    ):
        self.workload = workload
        self.arch = arch
        self.clamp_row = clamp_row
        self.activation = activation
        self.base_die = base_die
        self.channel_level = channel_level

        # Per-column-access width (io_width) in bits. Falls back to the historical
        # hardcoded 64b when the arch does not define specs.col_bits, so every
        # existing arch is byte-identical. getattr keeps duck-typed/stub specs
        # objects (used in some unit tests) working.
        self._col_size = getattr(self.arch.specs, "col_bits", None) or 64

        required_layout = self.workload.require_layout()
        self._streaming = required_layout.streaming
        self._sharing = required_layout.sharing
        self._bit_parallel = required_layout.bit_parallel
        self._cache_config = self.workload.resolve_cache_config(cache_config_override)
        self._cache_enabled = self._cache_config.enabled
        self._interleaving = required_layout.interleaving
        self._arch_cache = getattr(self.arch, "cache", None)
        if self._cache_enabled and self._arch_cache is None:
            raise ValueError(
                f"Architecture '{self.arch.name}' does not define a cache block; "
                "cache-enabled analytical evaluation requires one."
            )
        self._cache_mode: CacheMode | None = (
            self._arch_cache.mode if self._arch_cache is not None else None
        )
        self._validate_feature_flags()

        # precompute tensor element byte size
        self._ele_bytes_lookup: dict[TensorType, int] = {
            TensorType.F: self.workload.tensor_attrs[TensorType.F.value].bits // 8,
            TensorType.In: self.workload.tensor_attrs[TensorType.In.value].bits // 8,
            TensorType.Out: self.workload.tensor_attrs[TensorType.Out.value].bits // 8,
        }
        self._ele_bits_lookup: dict[TensorType, int] = {
            tensor_type: ele_bytes * 8
            for tensor_type, ele_bytes in self._ele_bytes_lookup.items()
        }

        # records
        self._breakdown: dict[str, Any] = {}

        self._init_lookups()

    def mem_footprint(self) -> int:
        # TODO: check the interleaving logic
        if self._mem_footprint_lookup is not None:
            return self._mem_footprint_lookup

        self._mem_footprint_lookup = 0
        if self._f_stored:
            self._mem_footprint_lookup += self._tensor_footprint(TensorType.F)
        if self._in_stored:
            self._mem_footprint_lookup += self._tensor_footprint(TensorType.In)
        if self._out_stored:
            self._mem_footprint_lookup += self._tensor_footprint(TensorType.Out)
        return self._mem_footprint_lookup

    def latency(self, target: str | None = None) -> int:
        target_key = self._normalize_metric_target(target)
        if target_key not in self._latency_lookup:
            self._evaluate_target_metrics(target_key)
        else:
            self._activate_target_metrics(target_key)
        return self._latency_lookup[target_key]

    def energy(self, target: str | None = None) -> int:
        target_key = self._normalize_metric_target(target)
        if target_key not in self._energy_lookup:
            self._evaluate_target_metrics(target_key)
        else:
            self._activate_target_metrics(target_key)
        return self._energy_lookup[target_key]

    def verify_constraints(self, verbose: int = 0) -> bool:
        failure_reasons: list[str] = []

        transformation_ok = self._transformation_constraints(failure_reasons)
        scheduling_ok = self._scheduling_constraints(failure_reasons)
        layout_ok = self._layout_constraints(failure_reasons)
        constraints_satisfied = transformation_ok and scheduling_ok and layout_ok

        if not constraints_satisfied and verbose >= 2:
            print("Constraint verification failure reasons:")
            if failure_reasons:
                for reason in failure_reasons:
                    print(f"  - {reason}")
            else:
                print(
                    "  - unavailable (constraint check returned False without details)"
                )

        return constraints_satisfied

    def max_tiles_per_row(self, tensor_type: TensorType) -> int:
        """Return the maximum number of tensor tiles that fit in one DRAM row."""
        lookup_value = self._max_tiles_per_row_lookup.get(tensor_type)
        if lookup_value is not None:
            return lookup_value

        row_bits = self.arch.specs.row_bytes * 8
        tile_bits = self._tile_size(tensor_type) * self._row_payload_bits(tensor_type)
        if tile_bits <= 0:
            raise ValueError(
                f"Tile bits must be positive when deriving row capacity for tensor {tensor_type.value}."
            )

        capacity = row_bits // tile_bits
        self._max_tiles_per_row_lookup[tensor_type] = capacity
        self._breakdown[f"{tensor_type.value} max tiles per row"] = capacity
        return capacity

    @property
    def breakdown(self) -> dict[str, Any]:
        return self._breakdown

    def _init_lookups(self) -> None:
        self._num_used_rows_lookup: dict[TensorType, int] = {}
        self._num_used_rows_pu_lookup: dict[TensorType, int] = {}

        # tensor lookups keyed by tensor type
        self._tensor_row_change_estimate_lookup: dict[
            TensorType, TensorRowChangeEstimate
        ] = {}
        self._num_tiles_lookup: dict[TensorType, int] = {}
        self._tile_size_lookup: dict[TensorType, int] = {}
        self._max_tiles_per_row_lookup: dict[TensorType, int] = {}
        self._num_tiles_system_lookup: dict[TensorType, int] = {}
        self._num_row_changes_lookup: dict[TensorType, int] = {}
        self._num_row_changes_pu_lookup: dict[TensorType, int] = {}

        self._tensor_coefficients_lookup: list[dict[str, int]] | None = None

        # calculated result lookups
        self._temporal_steps_lookup: int | None = None
        self._latency_lookup: dict[str, int] = {}
        self._energy_lookup: dict[str, int] = {}
        self._metric_breakdown_lookup: dict[str, dict[str, Any]] = {}
        self._active_metric_target: str | None = None
        self._mem_footprint_lookup: int | None = None
        self._compute_miss_rate_lookup: float | None = None
        self._streaming_miss_rate_lookup: float | None = None
        self._cache_result_lookup: GroupedCacheSimulationResult | None = None

    def _tensor_footprint(self, tensor_type: TensorType) -> int:
        """Calculate the memory footprint of a tensor"""
        num_used_rows_pu = self._num_used_rows_pu(tensor_type)
        row_bytes = self.arch.specs.row_bytes
        num_used_pus = self._num_all_used_pus()
        ret = num_used_rows_pu * row_bytes * num_used_pus
        self._breakdown[f"{tensor_type.value} footprint (bytes)"] = ret
        return ret

    def _normalize_metric_target(self, target: str | None) -> str:
        if target is None:
            return "default"

        normalized = str(target).strip()
        if normalized not in {"upmem", "attacc", "hbm_pim", "simdram"}:
            raise ValueError(f"Unknown target for analytical estimation: {target}")
        return normalized

    def _evaluate_target_metrics(self, target_key: str) -> None:
        self._clear_active_metric_breakdown()
        previous_breakdown = dict(self._breakdown)

        latency_value = int(self._target_metric_total("latency", target_key))
        energy_value = int(self._target_metric_total("energy", target_key))

        target_breakdown = {
            key: value
            for key, value in self._breakdown.items()
            if key not in previous_breakdown or previous_breakdown[key] != value
        }
        self._latency_lookup[target_key] = latency_value
        self._energy_lookup[target_key] = energy_value
        self._metric_breakdown_lookup[target_key] = target_breakdown
        self._active_metric_target = target_key

    def _activate_target_metrics(self, target_key: str) -> None:
        if self._active_metric_target == target_key:
            return
        self._clear_active_metric_breakdown()
        self._breakdown.update(self._metric_breakdown_lookup[target_key])
        self._active_metric_target = target_key

    def _clear_active_metric_breakdown(self) -> None:
        if self._active_metric_target is None:
            return
        target_breakdown = self._metric_breakdown_lookup.get(self._active_metric_target)
        if target_breakdown is not None:
            for key in target_breakdown:
                self._breakdown.pop(key, None)
        self._active_metric_target = None

    def _target_metric_total(self, metric_kind: str, target_key: str) -> float:
        if target_key == "default":
            return self._default_metric(metric_kind)
        if target_key == "upmem":
            return self._upmem_metric(metric_kind)
        if target_key == "attacc":
            return self._attacc_metric(metric_kind)
        if target_key == "hbm_pim":
            return self._hbm_pim_metric(metric_kind)
        if target_key == "simdram":
            return self._simdram_metric(metric_kind)
        raise ValueError(f"Unknown target for analytical estimation: {target_key}")

    def _default_metric(self, metric_kind: str) -> float:
        return (
            self._stream_metric(metric_kind)
            + self._compute_metric(metric_kind)
            + self._save_metric(metric_kind)
        )

    def _upmem_metric(self, metric_kind: str) -> float:
        return self._stream_metric(metric_kind) + self._compute_metric(metric_kind)

    def _attacc_metric(self, metric_kind: str) -> float:
        return self._stream_metric(
            metric_kind, consider_bus=False
        ) + self._compute_metric(metric_kind)

    def _hbm_pim_metric(self, metric_kind: str) -> float:
        return (
            self._stream_metric(metric_kind)
            + self._compute_metric(metric_kind)
            + self._save_metric(metric_kind)
        )

    def _simdram_metric(self, metric_kind: str) -> float:
        return self._stream_metric(metric_kind) + self._compute_metric(metric_kind)

    def _metric_label(self, base_name: str, metric_kind: str) -> str:
        suffix = "cycles" if metric_kind == "latency" else "energy"
        return f"{base_name} ({suffix})"

    def _validate_feature_flags(self) -> None:
        if self.base_die and self.channel_level:
            raise ValueError(
                "base_die and channel_level are mutually exclusive reduction-place "
                "toggles; enable at most one."
            )
        if self.base_die and not self.activation:
            raise ValueError("base_die requires activation to be enabled.")
        if self.channel_level and not self.activation:
            raise ValueError("channel_level requires activation to be enabled.")
        if self.base_die:
            if self.arch.unit_costs.base_die is None:
                raise ValueError(
                    f"Architecture '{self.arch.name}' must define unit_costs.base_die when base_die is enabled."
                )
            if self.arch.specs.add_tree_width is None:
                raise ValueError(
                    f"Architecture '{self.arch.name}' must define specs.add_tree_width when base_die is enabled."
                )
        if self.channel_level:
            if self.arch.unit_costs.channel_level is None:
                raise ValueError(
                    f"Architecture '{self.arch.name}' must define unit_costs.channel_level when channel_level is enabled."
                )
            if self.arch.specs.add_tree_width is None:
                raise ValueError(
                    f"Architecture '{self.arch.name}' must define specs.add_tree_width when channel_level is enabled."
                )

    def _energy_parallelism_scale(self, metric_kind: str) -> float:
        # Energy accumulates across all active PUs, while latency stays on the
        # critical path.
        if metric_kind != "energy":
            return 1.0
        return float(self._num_all_used_pus())

    def _stream_metric(self, metric_kind: str, consider_bus: bool = True) -> float:
        # TODO: fix this in a general way
        value = 0.0
        energy_scale = self._energy_parallelism_scale(metric_kind)

        if not consider_bus:
            lowest_level_pus = self._product_level(2)
            streaming_units = 0
            if self._streaming.F:
                temporal_steps = self._product_tensor_level(TensorType.F, 1)
                streaming_units += temporal_steps * self._tensor_streaming_miss_rate(
                    TensorType.F.value
                )
            if self._streaming.In:
                temporal_steps = self._product_tensor_level(TensorType.In, 1)
                streaming_units += temporal_steps * self._tensor_streaming_miss_rate(
                    TensorType.In.value
                )

            host_pu_cost = (
                self.arch.unit_costs.host_pu_transfer.latency
                if metric_kind == "latency"
                else self.arch.unit_costs.host_pu_transfer.energy
            )
            value = (
                streaming_units * lowest_level_pus * host_pu_cost * energy_scale
            )  # type: ignore
        else:
            streaming_units = 0
            temporal_steps = self._temporal_steps()

            for t_type in [TensorType.F, TensorType.In]:
                is_streaming = getattr(self._streaming, t_type.name)

                if is_streaming:
                    # num columns for a tile of a tensor
                    bits = self.workload.tensor_attrs[t_type.name].bits
                    num_cols = math.ceil(
                        self._tile_size(t_type) * bits / self._col_size
                    )

                    streaming_units += (
                        num_cols
                        * temporal_steps
                        * self._tensor_streaming_miss_rate(t_type.value)
                    )

            host_pu_cost = (
                self.arch.unit_costs.host_pu_transfer.latency
                if metric_kind == "latency"
                else self.arch.unit_costs.host_pu_transfer.energy
            )
            value = streaming_units * host_pu_cost * energy_scale
        self._breakdown[self._metric_label("streaming", metric_kind)] = float(value)
        return value

    def _compute_metric(self, metric_kind: str) -> float:
        # temporal steps
        temporal_steps = self._temporal_steps()
        energy_scale = self._energy_parallelism_scale(metric_kind)

        arithmetic_cost = (
            self.arch.unit_costs.pu_compute.latency
            if metric_kind == "latency"
            else self.arch.unit_costs.pu_compute.energy
        )
        bank_load_cost = (
            self.arch.unit_costs.bank_to_pu_load.latency
            if metric_kind == "latency"
            else self.arch.unit_costs.bank_to_pu_load.energy
        )
        bank_save_cost = (
            self.arch.unit_costs.pu_to_bank_store.latency
            if metric_kind == "latency"
            else self.arch.unit_costs.pu_to_bank_store.energy
        )

        # row_changes considering interleaving
        row_cost = (
            self.arch.unit_costs.row_change.latency
            if metric_kind == "latency"
            else self.arch.unit_costs.row_change.energy
        )
        compute_tensor_types = self._stored_compute_tensor_types()
        if self._stored_f_in_without_effective_interleaving():
            row_change_ratio = 2.0
        elif compute_tensor_types:
            row_changes = max(
                self._num_row_changes_bank(tensor_type)
                for tensor_type in compute_tensor_types
            )
            row_change_ratio = row_changes / temporal_steps
        else:
            row_change_ratio = 0.0
        self._breakdown["row change ratio"] = row_change_ratio

        # inter-bank communication due to sharing
        if self._sharing.system > 0:
            inter_bank_cost = (
                self.arch.unit_costs.bank_to_bank_transfer.latency
                if metric_kind == "latency"
                else self.arch.unit_costs.bank_to_bank_transfer.energy
            )
            num_tiles_pu = self._num_tiles_pu_sharing(TensorType.F)
            num_tiles_sys = self._num_tiles_system_sharing(TensorType.F)
            inter_bank_ratio = (num_tiles_pu - num_tiles_sys) / num_tiles_pu
            # Output partial sums shared across banks must be reduced cross-bank,
            # incurring the same bank-to-bank cost as the filter inter-bank load.
            out_tiles_pu = self._num_tiles_pu_sharing(TensorType.Out)
            out_tiles_sys = self._num_tiles_system_sharing(TensorType.Out)
            output_inter_bank_ratio = (
                (out_tiles_pu - out_tiles_sys) / out_tiles_pu if out_tiles_pu else 0.0
            )
        else:
            inter_bank_cost = 0.0
            inter_bank_ratio = 0.0
            output_inter_bank_ratio = 0.0
        self._breakdown["inter-bank ratio"] = inter_bank_ratio
        self._breakdown["output inter-bank ratio"] = output_inter_bank_ratio

        # consider cache
        compute_miss_rate = self._miss_rate()
        self._breakdown["miss rate"] = compute_miss_rate

        local_load_cost = 0.0
        row_change_load_cost = 0.0
        inter_bank_load_cost = 0.0
        for tensor_type in compute_tensor_types:
            tensor_miss_rate = self._tensor_compute_miss_rate(tensor_type.value)
            local_load_cost += tensor_miss_rate * (1 - inter_bank_ratio) * bank_load_cost
            row_change_load_cost += (
                tensor_miss_rate
                * (1 - inter_bank_ratio)
                * row_change_ratio
                * row_cost
            )
            inter_bank_load_cost += tensor_miss_rate * inter_bank_ratio * inter_bank_cost

        compute_arithmetic_cost = arithmetic_cost * temporal_steps * energy_scale
        compute_bank_save_cost = bank_save_cost * temporal_steps * energy_scale
        compute_local_bank_load_cost = local_load_cost * temporal_steps * energy_scale
        compute_row_change_load_cost = (
            row_change_load_cost * temporal_steps * energy_scale
        )
        compute_inter_bank_load_cost = (
            inter_bank_load_cost * temporal_steps * energy_scale
        )
        # Output cross-bank partial-sum reduction: symmetric to the filter
        # inter-bank load, charged per temporal step on the system-shared outputs.
        compute_output_inter_bank_cost = (
            output_inter_bank_ratio * inter_bank_cost * temporal_steps * energy_scale
        )
        compute_activation_cost = self._activation_compute_metric(metric_kind)

        value = (
            compute_arithmetic_cost
            + compute_bank_save_cost
            + compute_local_bank_load_cost
            + compute_row_change_load_cost
            + compute_inter_bank_load_cost
            + compute_output_inter_bank_cost
            + compute_activation_cost
        )
        self._breakdown[self._metric_label("compute arithmetic", metric_kind)] = (
            compute_arithmetic_cost
        )
        self._breakdown[self._metric_label("compute bank save", metric_kind)] = (
            compute_bank_save_cost
        )
        self._breakdown[self._metric_label("compute local bank load", metric_kind)] = (
            compute_local_bank_load_cost
        )
        self._breakdown[self._metric_label("compute row-change load", metric_kind)] = (
            compute_row_change_load_cost
        )
        self._breakdown[self._metric_label("compute inter-bank load", metric_kind)] = (
            compute_inter_bank_load_cost
        )
        self._breakdown[
            self._metric_label("compute output inter-bank reduce", metric_kind)
        ] = compute_output_inter_bank_cost
        if compute_activation_cost:
            self._breakdown[self._metric_label("compute activation", metric_kind)] = (
                compute_activation_cost
            )
        self._breakdown[self._metric_label("compute", metric_kind)] = value
        return value

    def _save_metric(self, metric_kind: str) -> float:
        if self._streaming.Out:
            self._breakdown[self._metric_label("save", metric_kind)] = 0.0
            return 0.0

        # Host readback pays both row-change cost and per-transfer-unit cost.
        energy_scale = self._energy_parallelism_scale(metric_kind)
        used_rows = self._num_row_changes_bank(TensorType.Out)
        row_cost = (
            self.arch.unit_costs.host_bank_row_conflict.latency
            if metric_kind == "latency"
            else self.arch.unit_costs.host_bank_row_conflict.energy
        )
        row_change_cost = used_rows * row_cost * energy_scale

        # Save/readback transfer width is based on output tile payload.
        save_tile_size = self._tile_size(TensorType.Out)
        save_tile_bits = save_tile_size * self._ele_bytes_lookup[TensorType.Out] * 8
        transfer_units_per_load = math.ceil(save_tile_bits / self._col_size)
        num_loads = self._num_tiles_pu(TensorType.Out)
        transfer_cost = (
            num_loads
            * transfer_units_per_load
            * (
                self.arch.unit_costs.host_pu_transfer.latency
                if metric_kind == "latency"
                else self.arch.unit_costs.host_pu_transfer.energy
            )
            * energy_scale
        )
        base_die_cost = self._base_die_save_metric(metric_kind)
        channel_level_cost = self._channel_level_save_metric(metric_kind)

        ret = row_change_cost + transfer_cost + base_die_cost + channel_level_cost
        self._breakdown[self._metric_label("save row-change", metric_kind)] = (
            row_change_cost
        )
        self._breakdown[self._metric_label("save transfer", metric_kind)] = (
            transfer_cost
        )
        if base_die_cost:
            self._breakdown[self._metric_label("save base-die", metric_kind)] = (
                base_die_cost
            )
        if channel_level_cost:
            self._breakdown[self._metric_label("save channel-level", metric_kind)] = (
                channel_level_cost
            )
        self._breakdown[self._metric_label("save", metric_kind)] = ret
        return ret

    def _activation_compute_metric(self, metric_kind: str) -> float:
        if (not self.activation) or self.base_die or self.channel_level:
            return 0.0

        operation_cost = (
            self.arch.unit_costs.bank_to_pu_load.latency
            + self.arch.unit_costs.pu_compute.latency
            + self.arch.unit_costs.pu_to_bank_store.latency
            if metric_kind == "latency"
            else self.arch.unit_costs.bank_to_pu_load.energy
            + self.arch.unit_costs.pu_compute.energy
            + self.arch.unit_costs.pu_to_bank_store.energy
        )
        # One load+activate+store vector op per output tile chunk; lanes bound the
        # chunk width, latency is the per-PU serial chain, and energy follows the
        # standard energy_scale convention (accumulates across active PUs).
        ops_pu = self._num_tiles_pu(TensorType.Out) * math.ceil(
            self._tile_size(TensorType.Out) / self.arch.specs.simd_lanes
        )
        return ops_pu * operation_cost * self._energy_parallelism_scale(metric_kind)

    def _base_die_save_metric(self, metric_kind: str) -> float:
        if (not self.activation) or (not self.base_die):
            return 0.0

        base_die_cost = self.arch.unit_costs.base_die
        add_tree_width = self.arch.specs.add_tree_width
        if base_die_cost is None or add_tree_width is None:
            raise ValueError(
                "base_die analytical mode requires arch.unit_costs.base_die and arch.specs.add_tree_width."
            )

        unit_cost = (
            base_die_cost.latency if metric_kind == "latency" else base_die_cost.energy
        )
        # Every used PU ships one partial per output tile it owns (Out-dim level-1
        # tiles) regardless of sharing; the add tree reduces those partials down to
        # the final output elements, retiring add_tree_width adds per pass, with
        # base_die priced per pass (one add-tree vector op).
        partial_elements = (
            self._product_tensor_level(TensorType.Out, 1)
            * self._tile_size(TensorType.Out)
            * self._num_all_used_pus()
        )
        num_output_elements = self._product_tensor(TensorType.Out)
        adds = partial_elements - num_output_elements
        if adds < 0:
            raise ValueError(
                "base_die reduction derived a negative add count: "
                f"partials={partial_elements} < finals={num_output_elements}."
            )
        self._breakdown["base-die partial elements"] = partial_elements
        self._breakdown["base-die adds"] = adds
        return math.ceil(adds / add_tree_width) * unit_cost

    def _channel_level_save_metric(self, metric_kind: str) -> float:
        if (not self.activation) or (not self.channel_level):
            return 0.0

        channel_cost = self.arch.unit_costs.channel_level
        add_tree_width = self.arch.specs.add_tree_width
        if channel_cost is None or add_tree_width is None:
            raise ValueError(
                "channel_level analytical mode requires arch.unit_costs.channel_level and arch.specs.add_tree_width."
            )

        unit_cost = (
            channel_cost.latency if metric_kind == "latency" else channel_cost.energy
        )
        # Same partial/final accounting as the base-die add-tree (every used PU ships
        # one partial per output tile it owns; the add tree retires add_tree_width adds
        # per pass). The difference from base-die: the reduction is CONFINED WITHIN each
        # channel (mapping cap sharing.system <= channel_level - 1, i.e. no cross-channel
        # reduction), so the C channels run their add-trees in parallel. Latency = a
        # single channel's passes (critical path); energy = all C channels' passes. The
        # per-pass unit cost is a base-die-class add-tree pass; the channel-count factor
        # lives in the structure (C from the hierarchy), not in the unit cost.
        partial_elements = (
            self._product_tensor_level(TensorType.Out, 1)
            * self._tile_size(TensorType.Out)
            * self._num_all_used_pus()
        )
        num_output_elements = self._product_tensor(TensorType.Out)
        adds = partial_elements - num_output_elements
        if adds < 0:
            raise ValueError(
                "channel_level reduction derived a negative add count: "
                f"partials={partial_elements} < finals={num_output_elements}."
            )
        num_channels = self._num_channels()
        passes_per_channel = math.ceil((adds / num_channels) / add_tree_width)
        self._breakdown["channel-level partial elements"] = partial_elements
        self._breakdown["channel-level adds"] = adds
        self._breakdown["channel-level parallel channels"] = num_channels
        self._breakdown["channel-level passes per channel"] = passes_per_channel
        if metric_kind == "latency":
            # all channels reduce concurrently -> only one channel on the critical path
            return passes_per_channel * unit_cost
        # every channel pays for its own passes
        return num_channels * passes_per_channel * unit_cost

    def _channel_reduction_level(self) -> int:
        """Hierarchy level whose kind is 'channel' (the parallel-reduction grain)."""
        level_kinds = self.arch.hierarchy.level_to_name
        channel_levels = [
            level
            for level, kind in level_kinds.items()
            if getattr(kind, "value", kind) == "channel"
        ]
        if not channel_levels:
            raise ValueError(
                f"channel_level reduction requires a 'channel' hierarchy level in '{self.arch.name}'."
            )
        return min(channel_levels)

    def _num_channels(self) -> int:
        """Channels reducing in parallel = loop fan-out at/above the channel level."""
        channel_level = self._channel_reduction_level()
        highest_level = max(self.workload.bounds.levels())
        if channel_level > highest_level:
            return 1
        return self._product_levels(channel_level, highest_level)

    def _miss_rate(self) -> float:
        return self._compute_miss_rate()

    def _compute_miss_rate(self) -> float:
        if self._compute_miss_rate_lookup is not None:
            return self._compute_miss_rate_lookup
        if not self._cache_effect_enabled():
            self._record_effective_cache_miss_rates(1.0, self._streaming_miss_rate())
            self._compute_miss_rate_lookup = 1.0
            return self._compute_miss_rate_lookup

        miss_rates: list[float] = []
        for tensor_type in self._stored_compute_tensor_types():
            miss_rates.append(self._tensor_compute_miss_rate(tensor_type.value))
        miss_rate = sum(miss_rates) / len(miss_rates) if miss_rates else 1.0
        self._record_effective_cache_miss_rates(miss_rate, self._streaming_miss_rate())
        self._compute_miss_rate_lookup = miss_rate
        return miss_rate

    def _tensor_compute_miss_rate(self, tensor_name: str) -> float:
        if not self._cache_effect_enabled():
            miss_rate = 1.0
        else:
            result = self._simulate_grouped_cache()
            miss_rate = self._tensor_role_miss_rate(
                result,
                tensor_name,
                role_name="compute",
            )
        self._breakdown[f"cache compute miss rate [{tensor_name}]"] = miss_rate
        return miss_rate

    def _streaming_miss_rate(self) -> float:
        if self._streaming_miss_rate_lookup is not None:
            return self._streaming_miss_rate_lookup
        if not self._cache_effect_enabled():
            self._streaming_miss_rate_lookup = 1.0
            return self._streaming_miss_rate_lookup

        result = self._simulate_grouped_cache()
        miss_rates: list[float] = []
        for tensor_type in self._streaming_compute_tensor_types():
            tensor_name = tensor_type.value
            tensor_miss_rate = self._tensor_role_miss_rate(
                result,
                tensor_name,
                role_name="streaming",
            )
            self._breakdown[f"cache streaming miss rate [{tensor_name}]"] = (
                tensor_miss_rate
            )
            miss_rates.append(tensor_miss_rate)
        miss_rate = sum(miss_rates) / len(miss_rates) if miss_rates else 1.0
        self._streaming_miss_rate_lookup = miss_rate
        self._record_effective_cache_miss_rates(self._compute_miss_rate_alias(), miss_rate)
        return miss_rate

    def _compute_miss_rate_alias(self) -> float:
        return (
            self._compute_miss_rate_lookup
            if self._compute_miss_rate_lookup is not None
            else 1.0
        )

    def _record_effective_cache_miss_rates(
        self, compute_miss_rate: float, streaming_miss_rate: float
    ) -> None:
        self._breakdown["cache effective compute miss rate"] = compute_miss_rate
        self._breakdown["cache effective streaming miss rate"] = streaming_miss_rate
        self._breakdown["cache effective load miss rate"] = compute_miss_rate

    def _tensor_streaming_miss_rate(self, tensor_name: str) -> float:
        if not self._cache_effect_enabled():
            miss_rate = 1.0
        else:
            result = self._simulate_grouped_cache()
            miss_rate = self._tensor_role_miss_rate(
                result,
                tensor_name,
                role_name="streaming",
            )
        self._breakdown[f"cache streaming miss rate [{tensor_name}]"] = miss_rate
        return miss_rate

    def _tensor_role_miss_rate(
        self,
        result: GroupedCacheSimulationResult,
        tensor_name: str,
        *,
        role_name: str,
    ) -> float:
        if self._cache_config.group_for_tensor(tensor_name) is None:
            return 1.0
        role_stats = getattr(result, "tensor_role_stats", {}).get(tensor_name, {})
        stats = role_stats.get(role_name)
        if stats is None:
            tensor_stats = getattr(result, "tensor_stats", {})
            merged_stats = tensor_stats.get(tensor_name)
            if merged_stats is None:
                return 1.0
            return float(merged_stats.miss_rate)
        return float(stats.miss_rate)

    def _simulate_grouped_cache(self) -> GroupedCacheSimulationResult:
        if not self._cache_enabled:
            raise ValueError("Cache simulation requested for a cache-disabled workload.")
        if self._cache_result_lookup is not None:
            return self._cache_result_lookup

        arch_cache = self._require_arch_cache_spec()
        allocations = self._allocate_cache_group_capacities(arch_cache)
        if not self._cache_effect_enabled():
            result = self._empty_grouped_cache_result(
                mode=arch_cache.mode,
                allocations=allocations,
            )
            self._record_cache_breakdown(result)
            self._cache_result_lookup = result
            return result
        result = simulate_grouped_cache_for_workload(
            self.workload,
            arch_cache.mode,
            allocations=allocations,
            # WRAM-mode archs don't declare vector_width_bits; derive it from the SIMD
            # width (FP16 lanes) so burst variants don't all simulate at 256 bits.
            vector_width_bits=arch_cache.vector_width_bits
            or (self.arch.specs.simd_lanes * 16),
            element_bytes=min(self._ele_bytes_lookup.values()),
        )
        self._record_cache_breakdown(result)
        self._cache_result_lookup = result
        return result

    def _empty_grouped_cache_result(
        self,
        *,
        mode: CacheMode,
        allocations: Mapping[str, CacheGroupAllocation],
    ) -> GroupedCacheSimulationResult:
        return GroupedCacheSimulationResult(
            mode=mode,
            global_stats=CacheStats(),
            tensor_stats={},
            group_stats={
                group_name: CacheStats()
                for group_name in allocations.keys()
            },
            group_allocations=dict(allocations),
            global_role_stats={
                "compute": CacheStats(),
                "streaming": CacheStats(),
            },
            tensor_role_stats={},
            group_role_stats={
                group_name: {
                    "compute": CacheStats(),
                    "streaming": CacheStats(),
                }
                for group_name in allocations.keys()
            },
            tensor_allocations={},
        )

    def _require_arch_cache_spec(self) -> ArchCacheSpec:
        if self._arch_cache is None:
            raise ValueError(
                f"Architecture '{self.arch.name}' does not define a cache block."
            )
        return self._arch_cache

    def _allocate_cache_group_capacities(
        self,
        arch_cache: ArchCacheSpec,
    ) -> dict[str, CacheGroupAllocation]:
        groups = self._cache_config.groups
        if not groups:
            return {}

        total_capacity_bytes = int(self.arch.specs.cache_bytes)
        if total_capacity_bytes <= 0:
            raise ValueError(
                f"Architecture '{self.arch.name}' must define a positive cache_bytes value for cache simulation."
            )

        weights = [group.weight if group.weight is not None else 1.0 for group in groups]
        total_weight = float(sum(weights))
        base_capacities = [
            int(total_capacity_bytes * weight / total_weight) for weight in weights
        ]
        remainder = total_capacity_bytes - sum(base_capacities)
        for index in range(remainder):
            base_capacities[index % len(base_capacities)] += 1

        vector_bytes = (
            (arch_cache.vector_width_bits or 0) // 8
            if arch_cache.mode == "regfile"
            else None
        )

        allocations: dict[str, CacheGroupAllocation] = {}
        for group, weight, capacity_bytes in zip(groups, weights, base_capacities):
            num_vector_regs: int | None = None
            if arch_cache.mode == "regfile":
                if vector_bytes is None or vector_bytes <= 0:
                    raise ValueError(
                        f"Architecture '{self.arch.name}' has invalid regfile vector width."
                    )
                num_vector_regs = capacity_bytes // vector_bytes
                if num_vector_regs <= 0:
                    raise ValueError(
                        f"Cache group '{group.name}' only received {capacity_bytes} bytes, "
                        f"which is smaller than one vector register ({vector_bytes} bytes)."
                    )
            allocations[group.name] = CacheGroupAllocation(
                name=group.name,
                tensors=group.tensors,
                weight=float(weight),
                capacity_bytes=capacity_bytes,
                num_vector_regs=num_vector_regs,
            )
        return allocations

    def _record_cache_breakdown(self, result: GroupedCacheSimulationResult) -> None:
        self._breakdown["cache mode"] = getattr(result, "mode", self._cache_mode)
        self._breakdown["cache accesses"] = result.global_stats.accesses
        self._breakdown["cache hits"] = result.global_stats.hits
        self._breakdown["cache misses"] = result.global_stats.misses
        self._breakdown["cache bytes loaded"] = result.global_stats.bytes_loaded
        self._breakdown["cache streaming accesses"] = (
            result.global_stats.streaming_accesses
        )
        self._breakdown["cache global miss rate"] = float(result.global_stats.miss_rate)
        for tensor_name, tensor_stats in sorted(result.tensor_stats.items()):
            self._breakdown[f"cache miss rate [{tensor_name}]"] = float(
                tensor_stats.miss_rate
            )

        tensor_role_stats = getattr(result, "tensor_role_stats", {})
        for tensor_name, role_stats in sorted(tensor_role_stats.items()):
            compute_stats = role_stats.get("compute")
            streaming_stats = role_stats.get("streaming")
            if compute_stats is not None:
                self._breakdown[f"cache compute miss rate [{tensor_name}]"] = float(
                    compute_stats.miss_rate
                )
            if streaming_stats is not None:
                self._breakdown[f"cache streaming miss rate [{tensor_name}]"] = float(
                    streaming_stats.miss_rate
                )

        for group_name, allocation in getattr(result, "group_allocations", {}).items():
            self._breakdown[f"cache group {group_name} capacity (bytes)"] = (
                allocation.capacity_bytes
            )
            self._breakdown[f"cache group {group_name} weight"] = allocation.weight
            self._breakdown[f"cache group {group_name} tensors"] = ",".join(
                allocation.tensors
            )
            if allocation.num_vector_regs is not None:
                self._breakdown[f"cache group {group_name} vector regs"] = (
                    allocation.num_vector_regs
                )

        tensor_allocations = getattr(result, "tensor_allocations", {})
        for tensor_name, allocation in sorted(tensor_allocations.items()):
            self._breakdown[f"cache tensor {tensor_name} capacity (bytes)"] = (
                allocation.capacity_bytes
            )
            if allocation.num_vector_regs is not None:
                self._breakdown[f"cache tensor {tensor_name} vector regs"] = (
                    allocation.num_vector_regs
                )

    def _transformation_constraints(
        self, failure_reasons: list[str] | None = None
    ) -> bool:
        """
        Return True when each loop variable's split extents multiply to its original extent.
        """
        original_extents = self.workload.problem.loop_vars
        if not original_extents:
            raise ValueError(
                f"Workload must have loop variables to check transformation constraints. Got: {original_extents}"
            )

        products = {loop_var: 1 for loop_var in original_extents}
        for level in self.workload.bounds.levels():
            level_bounds = self.workload.get_level_bounds(level)
            for loop_var, extent in level_bounds.items():
                if loop_var not in products:
                    continue
                product = products[loop_var] * extent
                if product > original_extents[loop_var]:
                    if failure_reasons is not None:
                        failure_reasons.append(
                            f"Transformation constraint failed for loop '{loop_var}' at level {level}: "
                            f"cumulative extent {product} exceeds original extent {original_extents[loop_var]}."
                        )
                    return False
                products[loop_var] = product

        for loop_var, original_extent in original_extents.items():
            if products[loop_var] != original_extent:
                if failure_reasons is not None:
                    failure_reasons.append(
                        f"Transformation constraint failed for loop '{loop_var}': "
                        f"cumulative extent {products[loop_var]} does not match original extent {original_extent}."
                    )
                return False
        return True

    def _scheduling_constraints(self, failure_reasons: list[str] | None = None) -> bool:
        """
        Return True when schedule-induced tensor footprints fit architecture limits.

        Level-0 checks SIMD lane capacity
        (`product(level-0) <= arch.specs.simd_lanes`).

        Level-1 checks:
        1. Each stored tensor tile must fit at least once in one DRAM row.
        2. Total used rows per bank across stored tensors must not exceed
        `arch.num_rows`, with intra/inter-tensor interleaving matching the
        analytical model.

        Level-2 and above checks per-level hierarchy capacity
        (`product(level) <= arch.hierarchy.organization[level]`).
        """
        if not self.workload.bounds.has_level(0):
            if failure_reasons is not None:
                failure_reasons.append(
                    "Scheduling constraint failed: missing level-0 loop bounds."
                )
            return False

        # Level-0 loops run within one PU and must fit PU concurrency.
        level0_product = self._product_level(0)
        if level0_product > self.arch.specs.simd_lanes:
            if failure_reasons is not None:
                failure_reasons.append(
                    "Scheduling constraint failed at level-0: "
                    f"product(level-0)={level0_product} exceeds simd_lanes={self.arch.specs.simd_lanes}."
                )
            return False

        # Level-1.1: each stored tensor tile must fit at least one slot in one DRAM row.
        for tensor_type, stored in (
            (TensorType.F, self._f_stored),
            (TensorType.In, self._in_stored),
            (TensorType.Out, self._out_stored),
        ):
            if not stored:
                continue

            tiles_per_row = self.max_tiles_per_row(tensor_type)
            if tiles_per_row <= 0:
                if failure_reasons is not None:
                    failure_reasons.append(
                        "Scheduling row-fit constraint failed for tensor "
                        f"{tensor_type.value}: tile_size={self._tile_size(tensor_type)} "
                        f"and row_payload_bits={self._row_payload_bits(tensor_type)} do not fit in "
                        f"row_bits={self.arch.specs.row_bytes * 8}."
                    )
                return False

        # Level-1.2: aggregate row usage in one bank and ensure it fits bank depth.
        used_rows_bank = 0
        f_rows = self._num_used_rows_bank(TensorType.F) if self._f_stored else 0
        in_rows = self._num_used_rows_bank(TensorType.In) if self._in_stored else 0

        # F/In share one striping domain under inter-tensor interleaving, so
        # combine first and apply banks_per_pu only once.
        if self._inter_tensor_interleaving:
            used_rows_bank += max(f_rows, in_rows)
        else:
            used_rows_bank += f_rows + in_rows

        if self._out_stored:
            used_rows_bank += self._num_used_rows_bank(TensorType.Out)

        if used_rows_bank > self.arch.num_rows:
            if failure_reasons is not None:
                failure_reasons.append(
                    "Scheduling bank-capacity constraint failed: "
                    f"used_rows_bank={used_rows_bank} exceeds arch.num_rows={self.arch.num_rows}."
                )
            return False

        # Level-2+: each mapped hierarchy level must fit architecture organization.
        organization = self.arch.hierarchy.organization
        for level in self.workload.bounds.levels():
            if level < 2:
                continue
            capacity = organization.get(level)
            if capacity is None:
                if failure_reasons is not None:
                    failure_reasons.append(
                        f"Scheduling hierarchy constraint failed: no architecture capacity defined for level {level}."
                    )
                return False

            level_product = self._product_level(level)
            if level_product > capacity:
                if failure_reasons is not None:
                    failure_reasons.append(
                        f"Scheduling hierarchy constraint failed at level {level}: "
                        f"product(level)={level_product} exceeds capacity={capacity}."
                    )
                return False

        return True

    def _layout_constraints(self, failure_reasons: list[str] | None = None) -> bool:
        del failure_reasons
        # TODO: verify later when arch is updated
        return True

    def _temporal_steps(self) -> int:
        """The number of temporal steps in the schedule, which is the product of level-1 loop bounds."""
        if self._temporal_steps_lookup is not None:
            return self._temporal_steps_lookup

        ret = self._product_level(1)
        self._breakdown["temporal steps"] = ret
        self._temporal_steps_lookup = ret
        return ret

    def _num_used_rows_pu(self, tensor_type: TensorType) -> int:
        """Total number of used rows of a tensor in a PU."""
        lookup_value = self._num_used_rows_pu_lookup.get(tensor_type)
        if lookup_value is not None:
            return lookup_value

        estimate = self._tensor_row_change_estimate(tensor_type)
        num_tiles_pu = self._num_tiles_pu(tensor_type)
        ret = self._used_rows_from_row_changes(
            self._rows_used_from_tiles_per_row(num_tiles_pu, estimate.tiles_per_row),
            tensor_type,
        )

        self._breakdown[f"{tensor_type.value} total used rows per pu"] = ret
        self._num_used_rows_pu_lookup[tensor_type] = ret
        return ret

    def _num_used_rows_bank(self, tensor_type: TensorType) -> int:
        """Total number of used rows of a tensor in a bank."""
        lookup_value = self._num_used_rows_lookup.get(tensor_type)
        if lookup_value is not None:
            return lookup_value

        used_rows_pu = self._rows_used_from_tiles_per_row(
            self._num_tiles_pu(tensor_type),
            self._tensor_row_change_estimate(tensor_type).tiles_per_row,
        )
        if tensor_type == TensorType.Out:
            interleaving = self._interleaving
        elif tensor_type in [TensorType.In, TensorType.F]:
            interleaving = self._intra_tensor_interleaving
        else:
            raise ValueError(f"Unknown tensor type: {tensor_type}")
        if interleaving:
            used_rows = math.ceil(used_rows_pu / self.arch.specs.banks_per_pu)
        else:
            used_rows = used_rows_pu
        ret = self._used_rows_from_row_changes(used_rows, tensor_type)

        self._breakdown[f"{tensor_type.value} total used rows"] = ret
        self._num_used_rows_lookup[tensor_type] = ret
        return ret

    def _row_payload_bits(self, tensor_type: TensorType) -> int:
        """Bits per element packed into a row slot under the current bit layout."""
        return self._ele_bits_lookup[tensor_type] if self._bit_parallel else 1

    def _used_rows_from_row_changes(
        self, row_changes: int, tensor_type: TensorType
    ) -> int:
        """Convert row changes to used rows under the current bit layout."""
        if self._bit_parallel:
            return row_changes
        return row_changes * self._ele_bits_lookup[tensor_type]

    def _num_row_changes_bank(self, tensor_type: TensorType) -> int:
        """The number of row changes of a tensor in a bank considering interleaving."""
        lookup_value = self._num_row_changes_lookup.get(tensor_type)
        if lookup_value is not None:
            return lookup_value

        row_changes_pu = self._num_row_changes_pu(tensor_type)
        if tensor_type == TensorType.Out:
            interleaving = self._interleaving
        elif tensor_type in [TensorType.In, TensorType.F]:
            interleaving = self._intra_tensor_interleaving
        else:
            raise ValueError(f"Unknown tensor type: {tensor_type}")
        banks_per_pu = self.arch.specs.banks_per_pu

        if interleaving:
            row_changes = math.ceil(row_changes_pu / banks_per_pu)
        else:
            row_changes = row_changes_pu

        self._breakdown[f"{tensor_type.value} row changes"] = row_changes
        self._num_row_changes_lookup[tensor_type] = row_changes

        return row_changes

    def _num_row_changes_pu(self, tensor_type: TensorType) -> int:
        """The number of row changes of a tensor in a PU."""
        lookup_value = self._num_row_changes_pu_lookup.get(tensor_type)
        if lookup_value is not None:
            return lookup_value

        estimate = self._tensor_row_change_estimate(tensor_type)
        row_changes = estimate.row_changes

        self._num_row_changes_pu_lookup[tensor_type] = row_changes
        self._breakdown[f"{tensor_type.value} effective tiles per row"] = (
            estimate.tiles_per_row
        )
        self._breakdown[f"{tensor_type.value} hypertile size"] = estimate.hypertile_size
        self._breakdown[f"{tensor_type.value} exact row-change simulation"] = int(
            estimate.used_exact_simulation
        )
        self._breakdown[f"{tensor_type.value} hypertile split across rows"] = int(
            estimate.hypertile_split_across_rows
        )
        self._breakdown[f"{tensor_type.value} row changes per pu"] = row_changes
        return row_changes

    def _tensor_row_change_estimate(
        self, tensor_type: TensorType
    ) -> TensorRowChangeEstimate:
        lookup_value = self._tensor_row_change_estimate_lookup.get(tensor_type)
        if lookup_value is not None:
            return lookup_value

        estimate = estimate_tensor_row_changes(self, tensor_type.value)
        self._tensor_row_change_estimate_lookup[tensor_type] = estimate
        return estimate

    def _rows_used_from_tiles_per_row(self, num_tiles: int, tiles_per_row: int) -> int:
        if num_tiles <= 0:
            return 0
        return math.ceil(num_tiles / tiles_per_row)

    def _num_tiles_pu(self, tensor_type: TensorType) -> int:
        """The number of tiles of a tensor in a PU."""
        pu_sharing = self._sharing.pu
        if (not pu_sharing) and (self._sharing.system == 0):
            # all tensors have the same number of tiles
            lookup_value = self._num_tiles_lookup.get(TensorType.F)
            if lookup_value is not None:
                return lookup_value
            ret = self._product_level(1)
            self._num_tiles_lookup[TensorType.F] = ret
            self._breakdown["num filter tiles per pu"] = ret
            return ret
        elif pu_sharing and (self._sharing.system == 0):
            return self._num_tiles_pu_sharing(tensor_type)
        elif pu_sharing and (self._sharing.system > 0):
            return self._num_tiles_system_sharing(tensor_type)
        else:
            raise ValueError("System sharing without PU sharing is not supported.")

    def _num_tiles_pu_sharing(self, tensor_type: TensorType) -> int:
        lookup_value = self._num_tiles_lookup.get(tensor_type)
        if lookup_value is not None:
            return lookup_value

        if tensor_type == TensorType.F:
            ret = self._product_tensor_level(TensorType.F, 1)
            self._breakdown["num filter tiles per pu"] = ret
        elif tensor_type == TensorType.In:
            ret = self._input_estimate(1, 1)
            self._breakdown["num input tiles per pu"] = ret
        elif tensor_type == TensorType.Out:
            ret = self._product_tensor_level(TensorType.Out, 1)
            self._breakdown["num output tiles per pu"] = ret
        else:
            raise ValueError(f"Unknown tensor type: {tensor_type}")

        self._num_tiles_lookup[tensor_type] = ret
        return ret

    def _num_tiles_system_sharing(self, tensor_type: TensorType) -> int:
        sharing_level = self._sharing.system
        num_shared_pus = self._product_levels(2, sharing_level)
        lookup_value = self._num_tiles_system_lookup.get(tensor_type)
        if lookup_value is not None:
            return lookup_value

        if tensor_type == TensorType.F:
            ret = math.ceil(
                self._product_tensor_levels(TensorType.F, 1, sharing_level)
                / num_shared_pus
            )
            self._breakdown["num filter tiles per system"] = ret
        elif tensor_type == TensorType.In:
            ret = math.ceil(self._input_estimate(1, sharing_level) / num_shared_pus)
            self._breakdown["num input tiles per system"] = ret
        elif tensor_type == TensorType.Out:
            ret = math.ceil(
                self._product_tensor_levels(TensorType.Out, 1, sharing_level)
                / num_shared_pus
            )
            self._breakdown["num output tiles per system"] = ret
        else:
            raise ValueError(f"Unknown tensor type: {tensor_type}")

        self._num_tiles_system_lookup[tensor_type] = ret
        return ret

    def _num_output_tiles_system(self) -> int:
        if self._sharing.system > 0:
            return self._num_tiles_system_sharing(TensorType.Out)

        ret = self._num_tiles_pu(TensorType.Out) * self._num_all_used_pus()
        self._breakdown["num output tiles per system"] = ret
        return ret

    def _tile_size(self, tensor_type: TensorType) -> int:
        if not self._sharing.block:
            # all tiles have the same size
            lookup_value = self._tile_size_lookup.get(TensorType.F)
            if lookup_value is not None:
                return lookup_value
            ret = self._product_level(0)
            self._tile_size_lookup[TensorType.F] = ret
            self._breakdown["filter tile size"] = ret
            return ret
        else:
            lookup_value = self._tile_size_lookup.get(tensor_type)
            if lookup_value is not None:
                return lookup_value

            if tensor_type == TensorType.F:
                ret = self._product_tensor_level(TensorType.F, 0)
                self._breakdown["filter tile size"] = ret
            elif tensor_type == TensorType.In:
                ret = self._input_estimate(0, 0)
                self._breakdown["input tile size"] = ret
            elif tensor_type == TensorType.Out:
                ret = self._product_tensor_level(TensorType.Out, 0)
                self._breakdown["output tile size"] = ret
            else:
                raise ValueError(f"Unknown tensor type: {tensor_type}")

            self._tile_size_lookup[tensor_type] = ret
            return ret

    def _input_estimate(self, start_level: int, end_level: int) -> int:
        latter = self._product_levels(start_level, end_level)
        # calculate former
        var_coeffs = self._var_coefficients()
        product_list: list[int] = []
        for l in range(start_level, end_level + 1):
            level_bounds = self.workload.get_level_bounds(l)
            level_coeffs = var_coeffs[l]
            n = level_bounds["N"]
            k = level_bounds["K"]
            c = level_bounds["C"]
            p = level_bounds["P"]
            r = level_bounds["R"]
            q = level_bounds["Q"]
            s = level_bounds["S"]
            p_coeff = level_coeffs["P"]
            r_coeff = level_coeffs["R"]
            q_coeff = level_coeffs["Q"]
            s_coeff = level_coeffs["S"]
            pr_dim = (
                int(
                    (p_coeff * (p - 1) + r_coeff * (r - 1)) / math.gcd(p_coeff, r_coeff)
                )
                + 1
            )
            qs_dim = (
                int(
                    (q_coeff * (q - 1) + s_coeff * (s - 1)) / math.gcd(q_coeff, s_coeff)
                )
                + 1
            )

            level_product = n * k * c * pr_dim * qs_dim
            product_list.append(level_product)
        former = math.prod(product_list)

        return min(former, latter)

    def _num_all_used_pus(self) -> int:
        """The number of all used PUs in the workload."""
        highest_level = max(self.workload.bounds.levels())
        return self._product_levels(2, highest_level)

    def _product_level(self, level: int) -> int:
        """The product of loop bounds at a level"""
        bounds = self.workload.get_level_bounds(level)
        return math.prod(bounds.values())

    def _product_levels(self, start_level: int, end_level: int) -> int:
        """The product of loop bounds from start_level to end_level"""
        return math.prod(
            [self._product_level(l) for l in range(start_level, end_level + 1)]
        )

    def _product_tensor(self, tensor_type: TensorType) -> int:
        """The product of loop bounds of a tensor"""
        highest_level = max(self.workload.bounds.levels())
        return self._product_tensor_levels(tensor_type, 0, highest_level)

    def _product_tensor_level(self, tensor_type: TensorType, level: int) -> int:
        """The product of loop bounds of a tensor at a level"""
        bounds = self.workload.get_tensor_level_bounds(tensor_type.value, level)
        return math.prod(bounds.values())

    def _product_tensor_levels(
        self, tensor_type: TensorType, start_level: int, end_level: int
    ) -> int:
        """The product of loop bounds of a tensor from start_level to end_level"""
        return math.prod(
            [
                self._product_tensor_level(tensor_type, l)
                for l in range(start_level, end_level + 1)
            ]
        )

    def _var_coefficients(self) -> list[dict[str, int]]:
        """Get the coefficients of each loop dimension for a tensor."""
        if self._tensor_coefficients_lookup is not None:
            return self._tensor_coefficients_lookup

        ret: list[dict[str, int]] = []
        ret.append({var: 1 for var in self.workload.problem.loop_vars.keys()})

        highest_level = max(self.workload.bounds.levels())
        for level in range(1, highest_level + 1):
            level_coeffs: dict[str, int] = {}
            last_bounds = self.workload.get_level_bounds(level - 1)
            for var, extent in last_bounds.items():
                last_coeff = ret[level - 1][var]
                level_coeffs[var] = last_coeff * extent
            ret.append(level_coeffs)
        self._tensor_coefficients_lookup = ret
        return ret

    def _stored_compute_tensor_types(self) -> tuple[TensorType, ...]:
        tensor_types: list[TensorType] = []
        if self._f_stored:
            tensor_types.append(TensorType.F)
        if self._in_stored:
            tensor_types.append(TensorType.In)
        return tuple(tensor_types)

    def _streaming_compute_tensor_types(self) -> tuple[TensorType, ...]:
        tensor_types: list[TensorType] = []
        if self._streaming.F:
            tensor_types.append(TensorType.F)
        if self._streaming.In:
            tensor_types.append(TensorType.In)
        return tuple(tensor_types)

    def _stored_f_in_without_effective_interleaving(self) -> bool:
        return (
            self._f_stored
            and self._in_stored
            and ((not self._interleaving) or self.arch.specs.banks_per_pu == 1)
        )

    def _cache_effect_enabled(self) -> bool:
        enabled = self._cache_enabled and bool(
            self._sharing.pu or self._sharing.system > 0
        )
        self._breakdown["cache performance enabled"] = enabled
        return enabled

    @property
    def _intra_tensor_interleaving(self) -> bool:
        return self._interleaving and not (self._f_stored and self._in_stored)

    @property
    def _inter_tensor_interleaving(self) -> bool:
        return self._interleaving and self._f_stored and self._in_stored

    @property
    def _f_stored(self) -> bool:
        return not self._streaming.F

    @property
    def _in_stored(self) -> bool:
        return not self._streaming.In

    @property
    def _out_stored(self) -> bool:
        return not self._streaming.Out
