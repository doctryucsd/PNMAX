from __future__ import annotations

from typing import Any, Mapping, Sequence

PAIRWISE_FRONTIER_METRICS: tuple[tuple[str, str, str], ...] = (
    ("latency_mem", "latency_cycles", "mem_footprint_bytes"),
    ("latency_energy", "latency_cycles", "energy"),
    ("mem_energy", "mem_footprint_bytes", "energy"),
)


def pareto_front_nd(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    metric_names = tuple(str(metric) for metric in metrics)
    if not metric_names:
        raise ValueError("metrics must contain at least one metric name.")

    deduped_rows = _dedupe_rows_by_metric_tuple(rows, metric_names)
    if len(deduped_rows) <= 1:
        return tuple(deduped_rows)

    if len(metric_names) == 2:
        nondominated = _pareto_front_2d(deduped_rows, metric_names)
    elif len(metric_names) == 3:
        nondominated = _pareto_front_3d(deduped_rows, metric_names)
    else:
        nondominated = _pareto_front_generic(deduped_rows, metric_names)

    return tuple(
        sorted(
            nondominated,
            key=lambda row: (
                *(_metric_value_tuple(row, metric_names)),
                _weighted_cost_sort_value(row),
                str(row.get("state_hash", "")),
            ),
        )
    )


def pareto_front(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_metric: str,
    y_metric: str,
) -> tuple[dict[str, Any], ...]:
    return pareto_front_nd(
        rows,
        metrics=(x_metric, y_metric),
    )


def pairwise_frontiers(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    frontiers: dict[str, tuple[dict[str, Any], ...]] = {}
    for frontier_name, x_metric, y_metric in PAIRWISE_FRONTIER_METRICS:
        frontiers[frontier_name] = pareto_front(
            rows,
            x_metric=x_metric,
            y_metric=y_metric,
        )
    return frontiers


def augment_frontier_row(
    *,
    frontier_name: str,
    x_metric: str,
    y_metric: str,
    step_id: str,
    step_label: str,
    step_runtime_s: float,
    order: int,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(row)
    payload.update(
        {
            "frontier_name": frontier_name,
            "x_metric": x_metric,
            "y_metric": y_metric,
            "step_id": step_id,
            "step_label": step_label,
            "order": int(order),
            "total_runtime_s": float(step_runtime_s),
        }
    )
    return payload


def _dedupe_rows_by_metric_tuple(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    deduped: dict[tuple[float, ...], dict[str, Any]] = {}
    for row in rows:
        metric_key = _metric_value_tuple(row, metrics)
        if metric_key not in deduped:
            deduped[metric_key] = dict(row)
    return tuple(deduped.values())


def _metric_value_tuple(
    row: Mapping[str, Any],
    metrics: Sequence[str],
) -> tuple[float, ...]:
    return tuple(float(row[metric]) for metric in metrics)


def _pareto_front_2d(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    x_metric, y_metric = metrics
    nondominated: list[dict[str, Any]] = []
    for index, candidate in enumerate(rows):
        cand_x = float(candidate[x_metric])
        cand_y = float(candidate[y_metric])
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            other_x = float(other[x_metric])
            other_y = float(other[y_metric])
            if (
                other_x <= cand_x
                and other_y <= cand_y
                and (other_x < cand_x or other_y < cand_y)
            ):
                dominated = True
                break
        if not dominated:
            nondominated.append(dict(candidate))
    return tuple(nondominated)


def _pareto_front_3d(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    x_metric, y_metric, z_metric = metrics
    sorted_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            float(row[x_metric]),
            float(row[y_metric]),
            float(row[z_metric]),
            _weighted_cost_sort_value(row),
            str(row.get("state_hash", "")),
        ),
    )
    y_values = sorted({float(row[y_metric]) for row in sorted_rows})
    y_to_index = {value: index + 1 for index, value in enumerate(y_values)}
    fenwick = _FenwickMinTree(len(y_values))
    nondominated: list[dict[str, Any]] = []
    for row in sorted_rows:
        y_value = float(row[y_metric])
        z_value = float(row[z_metric])
        if fenwick.query(y_to_index[y_value]) <= z_value:
            continue
        nondominated.append(dict(row))
        fenwick.update(y_to_index[y_value], z_value)
    return tuple(nondominated)


def _pareto_front_generic(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    nondominated: list[dict[str, Any]] = []
    for index, candidate in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            if _dominates(other, candidate, metrics):
                dominated = True
                break
        if not dominated:
            nondominated.append(dict(candidate))
    return tuple(nondominated)


def _dominates(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
    metrics: Sequence[str],
) -> bool:
    lhs_values = _metric_value_tuple(lhs, metrics)
    rhs_values = _metric_value_tuple(rhs, metrics)
    return all(
        lhs_value <= rhs_value for lhs_value, rhs_value in zip(lhs_values, rhs_values)
    ) and any(
        lhs_value < rhs_value for lhs_value, rhs_value in zip(lhs_values, rhs_values)
    )


class _FenwickMinTree:
    def __init__(self, size: int) -> None:
        self._size = int(size)
        self._tree = [float("inf")] * (self._size + 1)

    def update(self, index: int, value: float) -> None:
        position = int(index)
        candidate = float(value)
        while position <= self._size:
            if candidate < self._tree[position]:
                self._tree[position] = candidate
            position += position & -position

    def query(self, index: int) -> float:
        position = int(index)
        best = float("inf")
        while position > 0:
            if self._tree[position] < best:
                best = self._tree[position]
            position -= position & -position
        return best


def _weighted_cost_sort_value(row: Mapping[str, Any]) -> float:
    try:
        return float(row["weighted_cost"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
