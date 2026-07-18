from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from pnmax.database import Arch, Workload

if TYPE_CHECKING:
    from .analytical_model import AnalyticalModel, TensorType

TRACE_SEGMENT_LIMIT = 200
# Keep the estimator on the analytical fast path during search-time validation.
ENABLE_EXACT_HYPERTILE_ROW_CHANGE_SIMULATION = False


@dataclass(frozen=True)
class TensorRowChangeEstimate:
    tensor: str
    tiles_per_row: int
    hypertile_size: int
    row_changes: int
    used_exact_simulation: bool
    hypertile_split_across_rows: bool


@dataclass(frozen=True)
class TensorRowChangeStats:
    tensor: str
    tiles_per_row: int
    accesses: int
    unique_tiles: int
    rows_used: int
    row_changes: int
    additional_row_changes: int
    change_per_access: float


@dataclass(frozen=True)
class RowChangeReport:
    workload_name: str
    include_streaming: bool
    per_tensor: dict[str, TensorRowChangeStats]
    total_accesses: int
    total_row_changes: int
    total_additional_row_changes: int
    total_change_per_access: float
    clamp_row: bool = False
    trace_first_rows: int = 0
    tensor_traces: dict[str, "TensorAccessTrace"] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceRowTile:
    placement_index: int
    tile_indices: tuple[int, ...]
    first_touch_access: int
    access_count: int


@dataclass(frozen=True)
class TraceRow:
    row_id: int
    row_entries: int
    tiles: tuple[TraceRowTile, ...]


@dataclass(frozen=True)
class TraceRowSegment:
    row_id: int
    start_access: int
    end_access: int
    access_count: int


@dataclass(frozen=True)
class TraceTileSegment:
    row_id: int
    placement_index: int
    tile_indices: tuple[int, ...]
    start_access: int
    end_access: int
    access_count: int


@dataclass(frozen=True)
class TensorAccessTrace:
    tensor: str
    tiles_per_row: int
    rows_used: int
    rows: tuple[TraceRow, ...]
    row_stream: tuple[TraceRowSegment, ...]
    row_stream_truncated: bool
    tile_stream: tuple[TraceTileSegment, ...]
    tile_stream_truncated: bool


@dataclass
class _TrackedTile:
    placement_index: int
    tile_indices: tuple[int, ...]
    first_touch_access: int
    access_count: int = 0


@dataclass
class _RowSegmentState:
    row_id: int
    start_access: int
    end_access: int
    access_count: int


@dataclass
class _TileSegmentState:
    row_id: int
    placement_index: int
    tile_indices: tuple[int, ...]
    start_access: int
    end_access: int
    access_count: int


@dataclass
class _TraceState:
    first_rows: int
    tracked_tiles: dict[tuple[int, ...], _TrackedTile] = field(default_factory=dict)
    rows: dict[int, list[_TrackedTile]] = field(default_factory=dict)
    row_entry_counts: dict[int, int] = field(default_factory=dict)
    row_segments: list[_RowSegmentState] = field(default_factory=list)
    row_stream_truncated: bool = False
    tile_segments: list[_TileSegmentState] = field(default_factory=list)
    tile_stream_truncated: bool = False


@dataclass
class _TensorState:
    tile_to_index: dict[tuple[int, ...], int] = field(default_factory=dict)
    accesses: int = 0
    row_changes: int = 0
    previous_row: int | None = None
    trace: _TraceState | None = None


def estimate_tensor_row_changes(
    model: AnalyticalModel,
    tensor_name: str,
) -> TensorRowChangeEstimate:
    tensor_type = _tensor_type_for_name(tensor_name)
    tiles_per_row = _resolve_tiles_per_row(
        model,
        model.workload,
        tensor_name,
        clamp_row=_model_clamp_row(model),
    )
    hypertile_size = _resolve_hypertile_size(model.workload, tensor_name)
    num_tiles_pu = model._num_tiles_pu(tensor_type)
    analytical_row_changes = _rows_used_from_unique_tiles(num_tiles_pu, tiles_per_row)
    hypertile_split_across_rows = (
        hypertile_size > 1
        and analytical_row_changes > 1
        and (tiles_per_row % hypertile_size) != 0
    )

    if (
        ENABLE_EXACT_HYPERTILE_ROW_CHANGE_SIMULATION
        and hypertile_split_across_rows
    ):
        state = _simulate_tensor_state(
            model.workload,
            tensor_name,
            tiles_per_row,
        )
        row_changes = state.row_changes
        used_exact_simulation = True
    else:
        row_changes = analytical_row_changes
        used_exact_simulation = False

    return TensorRowChangeEstimate(
        tensor=tensor_name,
        tiles_per_row=tiles_per_row,
        hypertile_size=hypertile_size,
        row_changes=row_changes,
        used_exact_simulation=used_exact_simulation,
        hypertile_split_across_rows=hypertile_split_across_rows,
    )


def analyze_row_changes(
    workload: Workload,
    arch: Arch,
    *,
    include_streaming: bool = False,
    clamp_row: bool = False,
    trace_first_rows: int = 0,
) -> RowChangeReport:
    if trace_first_rows < 0:
        raise ValueError(
            f"trace_first_rows must be a non-negative integer, got {trace_first_rows}."
        )

    model = _build_analytical_model(workload, arch)
    level1_bounds = _require_level1_bounds(workload)
    tensors = _select_tensors(workload, include_streaming=include_streaming)
    if not tensors:
        raise ValueError(
            "No tensors selected for row-change analysis after streaming filtering. "
            "Use include_streaming=True to include streaming tensors."
        )

    tiles_per_row_by_tensor = {
        tensor: _resolve_tiles_per_row(
            model,
            workload,
            tensor,
            clamp_row=clamp_row,
        )
        for tensor in tensors
    }
    states = {
        tensor: _simulate_tensor_state(
            workload,
            tensor,
            tiles_per_row_by_tensor[tensor],
            level1_bounds=level1_bounds,
            trace_first_rows=trace_first_rows,
        )
        for tensor in tensors
    }

    per_tensor: dict[str, TensorRowChangeStats] = {}
    total_accesses = 0
    total_row_changes = 0
    total_additional_row_changes = 0
    tensor_traces: dict[str, TensorAccessTrace] = {}

    for tensor_name in tensors:
        tensor_stats, tensor_trace = _summarize_tensor_state(
            tensor=tensor_name,
            state=states[tensor_name],
            tiles_per_row=tiles_per_row_by_tensor[tensor_name],
        )
        per_tensor[tensor_name] = tensor_stats
        total_accesses += tensor_stats.accesses
        total_row_changes += tensor_stats.row_changes
        total_additional_row_changes += tensor_stats.additional_row_changes
        if tensor_trace is not None:
            tensor_traces[tensor_name] = tensor_trace

    total_change_per_access = (
        total_row_changes / total_accesses if total_accesses > 0 else 0.0
    )

    return RowChangeReport(
        workload_name=workload.name,
        include_streaming=include_streaming,
        per_tensor=per_tensor,
        total_accesses=total_accesses,
        total_row_changes=total_row_changes,
        total_additional_row_changes=total_additional_row_changes,
        total_change_per_access=total_change_per_access,
        clamp_row=clamp_row,
        trace_first_rows=trace_first_rows,
        tensor_traces=tensor_traces,
    )


def _build_analytical_model(
    workload: Workload,
    arch: Arch,
) -> AnalyticalModel:
    from .analytical_model import AnalyticalModel

    return AnalyticalModel(workload, arch)


def _model_clamp_row(model: AnalyticalModel) -> bool:
    return bool(getattr(model, "clamp_row", False))


def _resolve_tiles_per_row(
    model: AnalyticalModel,
    workload: Workload,
    tensor_name: str,
    *,
    clamp_row: bool,
) -> int:
    tensor_type = _tensor_type_for_name(tensor_name)
    tiles_per_row = model.max_tiles_per_row(tensor_type)
    if tiles_per_row <= 0:
        raise ValueError(
            f"Tensor {tensor_name} does not fit in a DRAM row for architecture '{model.arch.name}'."
        )
    if not clamp_row:
        return tiles_per_row

    hypertile_size = _resolve_hypertile_size(workload, tensor_name)
    clamped_tiles_per_row = (tiles_per_row // hypertile_size) * hypertile_size
    if clamped_tiles_per_row > 0:
        return clamped_tiles_per_row
    return tiles_per_row


def _resolve_hypertile_size(workload: Workload, tensor_name: str) -> int:
    level1_bounds = workload.get_level_bounds(1)
    tensor_related_dimensions = set(workload.get_tensor_level_bounds(tensor_name, 1))

    cutoff_dimension: str | None = None
    for dimension, extent in level1_bounds.items():
        if dimension in tensor_related_dimensions:
            continue
        if int(extent) > 1:
            cutoff_dimension = dimension
            break

    collect_sequential = cutoff_dimension is None
    sequential_extents: list[int] = []
    for dimension, extent in level1_bounds.items():
        if not collect_sequential:
            if dimension == cutoff_dimension:
                collect_sequential = True
            continue
        if dimension in tensor_related_dimensions:
            sequential_extents.append(int(extent))

    hypertile_size = math.prod(sequential_extents) if sequential_extents else 1
    if hypertile_size <= 0:
        raise ValueError(
            f"Tensor {tensor_name} has non-positive hypertile size {hypertile_size}."
        )
    return hypertile_size


def _require_level1_bounds(workload: Workload) -> Mapping[str, int]:
    try:
        level1_bounds = workload.get_level_bounds(1)
    except KeyError as exc:
        raise ValueError(
            f"Workload '{workload.name}' does not define level-1 bounds."
        ) from exc

    if not level1_bounds:
        raise ValueError(
            f"Workload '{workload.name}' has empty level-1 bounds; cannot analyze row changes."
        )

    for dimension, extent in level1_bounds.items():
        if int(extent) <= 0:
            raise ValueError(
                f"Level-1 bound for dimension '{dimension}' must be > 0, got {extent}."
            )
    return level1_bounds


def _select_tensors(workload: Workload, *, include_streaming: bool) -> tuple[str, ...]:
    ordered = workload.ordered_tensor_access_patterns()
    streaming_flags = workload.streaming_flags()
    selected: list[str] = []
    for pattern in ordered:
        tensor_name = pattern.tensor
        if (not include_streaming) and streaming_flags.get(tensor_name, False):
            continue
        selected.append(tensor_name)
    return tuple(selected)


def _build_eval_context(
    dimension_names: list[str], dimension_values: tuple[int, ...]
) -> dict[str, int]:
    context: dict[str, int] = {}
    for name, value in zip(dimension_names, dimension_values):
        int_value = int(value)
        context[name] = int_value
        context[name.lower()] = int_value
        context[name.upper()] = int_value
    return context


def _simulate_tensor_state(
    workload: Workload,
    tensor_name: str,
    tiles_per_row: int,
    *,
    level1_bounds: Mapping[str, int] | None = None,
    trace_first_rows: int = 0,
) -> _TensorState:
    resolved_level1_bounds = (
        level1_bounds if level1_bounds is not None else _require_level1_bounds(workload)
    )
    state = _TensorState(
        trace=_TraceState(trace_first_rows) if trace_first_rows > 0 else None
    )
    dimension_names = list(resolved_level1_bounds.keys())
    dimension_extents = [int(resolved_level1_bounds[name]) for name in dimension_names]

    for values in itertools.product(*(range(extent) for extent in dimension_extents)):
        eval_ctx = _build_eval_context(dimension_names, values)
        tile_indices = workload.evaluate_tensor_indices(tensor_name, eval_ctx)
        _record_access(
            state=state,
            tile_indices=tile_indices,
            tiles_per_row=tiles_per_row,
        )
    return state


def _rows_used_from_unique_tiles(unique_tiles: int, tiles_per_row: int) -> int:
    if unique_tiles == 0:
        return 0
    return ((unique_tiles - 1) // tiles_per_row) + 1


def _summarize_tensor_state(
    *,
    tensor: str,
    state: _TensorState,
    tiles_per_row: int,
) -> tuple[TensorRowChangeStats, TensorAccessTrace | None]:
    unique_tiles = len(state.tile_to_index)
    rows_used = _rows_used_from_unique_tiles(unique_tiles, tiles_per_row)
    additional_row_changes = state.row_changes - rows_used
    change_per_access = (
        state.row_changes / state.accesses if state.accesses > 0 else 0.0
    )
    tensor_stats = TensorRowChangeStats(
        tensor=tensor,
        tiles_per_row=tiles_per_row,
        accesses=state.accesses,
        unique_tiles=unique_tiles,
        rows_used=rows_used,
        row_changes=state.row_changes,
        additional_row_changes=additional_row_changes,
        change_per_access=change_per_access,
    )
    tensor_trace = None
    if state.trace is not None:
        tensor_trace = _finalize_tensor_trace(
            tensor=tensor,
            trace=state.trace,
            tiles_per_row=tiles_per_row,
            rows_used=rows_used,
        )
    return tensor_stats, tensor_trace


def _record_access(
    *,
    state: _TensorState,
    tile_indices: tuple[int, ...],
    tiles_per_row: int,
) -> None:
    state.accesses += 1
    access_ordinal = state.accesses
    placement_index = state.tile_to_index.get(tile_indices)
    if placement_index is None:
        placement_index = len(state.tile_to_index)
        state.tile_to_index[tile_indices] = placement_index

    row_id = placement_index // tiles_per_row
    if state.trace is not None:
        _record_trace_access(
            trace=state.trace,
            tile_indices=tile_indices,
            placement_index=placement_index,
            row_id=row_id,
            previous_row=state.previous_row,
            access_ordinal=access_ordinal,
        )
    if state.previous_row is None or row_id != state.previous_row:
        state.row_changes += 1
    state.previous_row = row_id


def _record_trace_access(
    *,
    trace: _TraceState,
    tile_indices: tuple[int, ...],
    placement_index: int,
    row_id: int,
    previous_row: int | None,
    access_ordinal: int,
) -> None:
    if row_id >= trace.first_rows:
        return

    if previous_row is None or row_id != previous_row:
        trace.row_entry_counts[row_id] = trace.row_entry_counts.get(row_id, 0) + 1

    tracked_tile = trace.tracked_tiles.get(tile_indices)
    if tracked_tile is None:
        tracked_tile = _TrackedTile(
            placement_index=placement_index,
            tile_indices=tile_indices,
            first_touch_access=access_ordinal,
        )
        trace.tracked_tiles[tile_indices] = tracked_tile
        trace.rows.setdefault(row_id, []).append(tracked_tile)
    tracked_tile.access_count += 1

    _append_row_segment(
        trace=trace,
        row_id=row_id,
        access_ordinal=access_ordinal,
    )
    _append_tile_segment(
        trace=trace,
        row_id=row_id,
        placement_index=placement_index,
        tile_indices=tile_indices,
        access_ordinal=access_ordinal,
    )


def _append_row_segment(
    *,
    trace: _TraceState,
    row_id: int,
    access_ordinal: int,
) -> None:
    if trace.row_stream_truncated:
        return

    if trace.row_segments:
        previous = trace.row_segments[-1]
        if previous.row_id == row_id and previous.end_access + 1 == access_ordinal:
            previous.end_access = access_ordinal
            previous.access_count += 1
            return

    if len(trace.row_segments) >= TRACE_SEGMENT_LIMIT:
        trace.row_stream_truncated = True
        return

    trace.row_segments.append(
        _RowSegmentState(
            row_id=row_id,
            start_access=access_ordinal,
            end_access=access_ordinal,
            access_count=1,
        )
    )


def _append_tile_segment(
    *,
    trace: _TraceState,
    row_id: int,
    placement_index: int,
    tile_indices: tuple[int, ...],
    access_ordinal: int,
) -> None:
    if trace.tile_stream_truncated:
        return

    if trace.tile_segments:
        previous = trace.tile_segments[-1]
        if (
            previous.row_id == row_id
            and previous.placement_index == placement_index
            and previous.tile_indices == tile_indices
            and previous.end_access + 1 == access_ordinal
        ):
            previous.end_access = access_ordinal
            previous.access_count += 1
            return

    if len(trace.tile_segments) >= TRACE_SEGMENT_LIMIT:
        trace.tile_stream_truncated = True
        return

    trace.tile_segments.append(
        _TileSegmentState(
            row_id=row_id,
            placement_index=placement_index,
            tile_indices=tile_indices,
            start_access=access_ordinal,
            end_access=access_ordinal,
            access_count=1,
        )
    )


def _finalize_tensor_trace(
    *,
    tensor: str,
    trace: _TraceState,
    tiles_per_row: int,
    rows_used: int,
) -> TensorAccessTrace:
    row_ids = tuple(range(min(trace.first_rows, rows_used)))
    rows = tuple(
        TraceRow(
            row_id=row_id,
            row_entries=trace.row_entry_counts.get(row_id, 0),
            tiles=tuple(
                TraceRowTile(
                    placement_index=tile.placement_index,
                    tile_indices=tile.tile_indices,
                    first_touch_access=tile.first_touch_access,
                    access_count=tile.access_count,
                )
                for tile in trace.rows.get(row_id, [])
            ),
        )
        for row_id in row_ids
        if trace.rows.get(row_id)
    )

    row_stream = tuple(
        TraceRowSegment(
            row_id=segment.row_id,
            start_access=segment.start_access,
            end_access=segment.end_access,
            access_count=segment.access_count,
        )
        for segment in trace.row_segments
    )
    tile_stream = tuple(
        TraceTileSegment(
            row_id=segment.row_id,
            placement_index=segment.placement_index,
            tile_indices=segment.tile_indices,
            start_access=segment.start_access,
            end_access=segment.end_access,
            access_count=segment.access_count,
        )
        for segment in trace.tile_segments
    )

    return TensorAccessTrace(
        tensor=tensor,
        tiles_per_row=tiles_per_row,
        rows_used=rows_used,
        rows=rows,
        row_stream=row_stream,
        row_stream_truncated=trace.row_stream_truncated,
        tile_stream=tile_stream,
        tile_stream_truncated=trace.tile_stream_truncated,
    )


def _tensor_type_for_name(tensor_name: str) -> TensorType:
    from .analytical_model import TensorType

    canonical = tensor_name.strip()
    if canonical == TensorType.In.value:
        return TensorType.In
    if canonical == TensorType.F.value:
        return TensorType.F
    if canonical == TensorType.Out.value:
        return TensorType.Out
    raise ValueError(f"Unsupported tensor for row-change analysis: {tensor_name}.")
