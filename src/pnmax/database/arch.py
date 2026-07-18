from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

__all__ = [
    "Arch",
    "ArchParseError",
    "ArchCacheSpec",
    "Specs",
    "Hierarchy",
    "Bandwidths",
    "UnitCosts",
    "ReductionLevel",
]


class ArchParseError(ValueError):
    """Raised when an architecture YAML payload is malformed."""


class HierLevel(str, Enum):
    """Supported hierarchy levels in architecture specs."""

    PU = "pu"
    BANK = "bank"
    BG = "bg"
    CHIP = "chip"
    PCH = "pch"
    RANK = "rank"
    CHANNEL = "channel"
    STACK = "stack"


@dataclass(frozen=True)
class Specs:
    """Architecture-level specs, normalized to bytes and Hz."""

    row_bytes: int
    bank_bytes: int
    cache_bytes: int
    banks_per_pu: int
    pu_frequency_hz: float
    simd_lanes: int
    n_lines: int | None = None
    add_tree_width: int | None = None
    col_bits: int | None = None

    @property
    def bank_size_bytes(self) -> int:
        return self.bank_bytes

    @property
    def buffer_size_bytes(self) -> int:
        return self.cache_bytes

    @property
    def pu_factor(self) -> int:
        return self.banks_per_pu

    @property
    def pu_freq_hz(self) -> float:
        return self.pu_frequency_hz

    @property
    def pu_concurrency(self) -> int:
        return self.simd_lanes


@dataclass(frozen=True)
class HierarchyLevelSpec:
    kind: HierLevel
    count: int


@dataclass(frozen=True)
class Hierarchy:
    """Logical hierarchy and organization."""

    levels: dict[int, HierarchyLevelSpec]

    @property
    def level_to_name(self) -> dict[int, HierLevel]:
        return {level: spec.kind for level, spec in self.levels.items()}

    @property
    def organization(self) -> dict[int, int]:
        return {level: spec.count for level, spec in self.levels.items()}

    @property
    def bank_level(self) -> int:
        for level, spec in self.levels.items():
            if spec.kind is HierLevel.BANK:
                return level
        raise ValueError("No 'bank' level defined in hierarchy.")

    @property
    def total_units(self) -> int:
        prod = 1
        for spec in self.levels.values():
            prod *= spec.count
        return prod

    @property
    def total_banks(self) -> int:
        return self.total_units


@dataclass(frozen=True)
class Bandwidths:
    """Interconnect bandwidths in bytes per second."""

    host_device_Bps: float


@dataclass(frozen=True)
class OperationCost:
    """One modeled operation cost in cycles and pJ."""

    latency: float
    energy: float


@dataclass(frozen=True)
class UnitCosts:
    """Per-operation latency and energy costs."""

    bank_to_pu_load: OperationCost
    pu_to_bank_store: OperationCost
    host_pu_transfer: OperationCost
    bank_to_bank_transfer: OperationCost
    host_bank_row_conflict: OperationCost
    pu_compute: OperationCost
    row_change: OperationCost
    base_die: OperationCost | None = None
    channel_level: OperationCost | None = None


@dataclass(frozen=True)
class ArchCacheSpec:
    """Architecture-defined cache shape for the analytical model."""

    mode: str
    vector_width_bits: int | None = None

    def __post_init__(self) -> None:
        normalized_mode = str(self.mode).strip().lower()
        if normalized_mode not in {"wram", "regfile"}:
            raise ValueError(f"Unsupported cache mode '{self.mode}'.")
        object.__setattr__(self, "mode", normalized_mode)

        if normalized_mode == "regfile":
            if self.vector_width_bits is None:
                raise ValueError(
                    "Regfile cache architectures must define vector_width_bits."
                )
            vector_width_bits = int(self.vector_width_bits)
            if vector_width_bits <= 0 or vector_width_bits % 8 != 0:
                raise ValueError(
                    "vector_width_bits must be a positive multiple of 8."
                )
            object.__setattr__(self, "vector_width_bits", vector_width_bits)
        elif self.vector_width_bits is not None:
            vector_width_bits = int(self.vector_width_bits)
            if vector_width_bits <= 0 or vector_width_bits % 8 != 0:
                raise ValueError(
                    "vector_width_bits must be a positive multiple of 8."
                )
            object.__setattr__(self, "vector_width_bits", vector_width_bits)


@dataclass(frozen=True)
class ReductionLevel:
    """One level of reduction in the hierarchy."""

    num: int
    concurrency: int
    bg_size: int
    _bandwidth_Bps: float
    _compute_cost_cycles: float

    def bandwidth_bytes_per_ns(self) -> float:
        return self._bandwidth_Bps / 1e9

    def compute_latency_cycles(self) -> float:
        return self._compute_cost_cycles


_BYTES_UNITS: dict[str, int] = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
}

_BANDWIDTH_UNITS: dict[str, float] = {
    unit + "/s": factor for unit, factor in _BYTES_UNITS.items()
}

_FREQ_UNITS: dict[str, float] = {
    "hz": 1.0,
    "mhz": 1e6,
    "ghz": 1e9,
}


def _parse_size_bytes(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"Size must be non-negative, got {value}.")
        return int(value)
    text = str(value).strip().lower()
    if not text:
        raise ValueError("Empty size string.")
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([kmgt]?b)", text)
    if not match:
        raise ValueError(f"Unsupported size format: {value}")
    magnitude = float(match.group(1))
    suffix = match.group(2)
    factor = _BYTES_UNITS.get(suffix)
    if factor is None:
        raise ValueError(f"Unsupported size suffix '{suffix}' in '{value}'.")
    return int(round(magnitude * factor))


def _parse_bandwidth_Bps(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"Bandwidth must be non-negative, got {value}.")
        return float(value)
    text = str(value).strip().lower()
    if not text:
        raise ValueError("Empty bandwidth string.")
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([kmgt]?b)(?:/s)?", text)
    if not match:
        raise ValueError(f"Unsupported bandwidth format: {value}")
    magnitude = float(match.group(1))
    suffix = match.group(2) + "/s"
    factor = _BANDWIDTH_UNITS.get(suffix)
    if factor is None:
        raise ValueError(f"Unsupported bandwidth suffix in '{value}'.")
    return magnitude * factor


def _parse_freq_hz(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"Frequency must be non-negative, got {value}.")
        return float(value)
    text = str(value).strip().lower()
    if not text:
        raise ValueError("Empty frequency string.")
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([mg]?hz)", text)
    if not match:
        raise ValueError(f"Unsupported frequency format: {value}")
    magnitude = float(match.group(1))
    suffix = match.group(2)
    factor = _FREQ_UNITS.get(suffix)
    if factor is None:
        raise ValueError(f"Unsupported frequency suffix in '{value}'.")
    return magnitude * factor


def _parse_specs(raw: Mapping[str, Any]) -> Specs:
    arch_name = str(raw.get("name", "<unknown>"))
    specs_raw = raw.get("specs")
    if not isinstance(specs_raw, Mapping):
        raise ValueError(f"Architecture '{arch_name}' missing 'specs' block.")

    def _require(key: str) -> Any:
        if key not in specs_raw:
            raise ValueError(f"Architecture '{arch_name}' missing specs.{key}.")
        return specs_raw[key]

    row_bytes = _parse_size_bytes(_require("row_bytes"))
    bank_bytes = _parse_size_bytes(_require("bank_bytes"))
    cache_bytes = _parse_size_bytes(_require("cache_bytes"))
    banks_per_pu = int(_require("banks_per_pu"))
    if banks_per_pu <= 0:
        raise ValueError(f"Architecture '{arch_name}' must have banks_per_pu > 0.")
    pu_frequency_hz = _parse_freq_hz(_require("pu_frequency"))
    if pu_frequency_hz <= 0:
        raise ValueError(f"Architecture '{arch_name}' must have pu_frequency > 0.")
    simd_lanes = int(_require("simd_lanes"))
    if simd_lanes <= 0:
        raise ValueError(f"Architecture '{arch_name}' must have simd_lanes > 0.")
    n_lines_raw = specs_raw.get("n_lines")
    n_lines: int | None = None
    if n_lines_raw is not None:
        if isinstance(n_lines_raw, bool):
            raise ValueError(
                f"Architecture '{arch_name}' has invalid specs.n_lines={n_lines_raw}."
            )
        if isinstance(n_lines_raw, float) and not n_lines_raw.is_integer():
            raise ValueError(
                f"Architecture '{arch_name}' has invalid specs.n_lines={n_lines_raw}."
            )
        try:
            n_lines = int(n_lines_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Architecture '{arch_name}' has invalid specs.n_lines={n_lines_raw}."
            ) from exc
        if n_lines < 0:
            raise ValueError(f"Architecture '{arch_name}' must have specs.n_lines >= 0.")

    add_tree_width_raw = specs_raw.get("add_tree_width")
    add_tree_width: int | None = None
    if add_tree_width_raw is not None:
        if isinstance(add_tree_width_raw, bool):
            raise ValueError(
                f"Architecture '{arch_name}' has invalid specs.add_tree_width={add_tree_width_raw}."
            )
        if isinstance(add_tree_width_raw, float) and not add_tree_width_raw.is_integer():
            raise ValueError(
                f"Architecture '{arch_name}' has invalid specs.add_tree_width={add_tree_width_raw}."
            )
        try:
            add_tree_width = int(add_tree_width_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Architecture '{arch_name}' has invalid specs.add_tree_width={add_tree_width_raw}."
            ) from exc
        if add_tree_width <= 0:
            raise ValueError(
                f"Architecture '{arch_name}' must have specs.add_tree_width > 0."
            )

    col_bits_raw = specs_raw.get("col_bits")
    col_bits: int | None = None
    if col_bits_raw is not None:
        if isinstance(col_bits_raw, bool):
            raise ValueError(
                f"Architecture '{arch_name}' has invalid specs.col_bits={col_bits_raw}."
            )
        if isinstance(col_bits_raw, float) and not col_bits_raw.is_integer():
            raise ValueError(
                f"Architecture '{arch_name}' has invalid specs.col_bits={col_bits_raw}."
            )
        try:
            col_bits = int(col_bits_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Architecture '{arch_name}' has invalid specs.col_bits={col_bits_raw}."
            ) from exc
        if col_bits <= 0:
            raise ValueError(
                f"Architecture '{arch_name}' must have specs.col_bits > 0."
            )

    known_keys = {
        "row_bytes",
        "bank_bytes",
        "cache_bytes",
        "banks_per_pu",
        "pu_frequency",
        "simd_lanes",
        "n_lines",
        "add_tree_width",
        "col_bits",
    }
    extra = sorted(str(key) for key in specs_raw if key not in known_keys)
    if extra:
        raise ValueError(
            f"Architecture '{arch_name}' has unsupported specs keys: {', '.join(extra)}."
        )

    return Specs(
        row_bytes=row_bytes,
        bank_bytes=bank_bytes,
        cache_bytes=cache_bytes,
        banks_per_pu=banks_per_pu,
        pu_frequency_hz=pu_frequency_hz,
        simd_lanes=simd_lanes,
        n_lines=n_lines,
        add_tree_width=add_tree_width,
        col_bits=col_bits,
    )


def _parse_level_key(key: Any, arch_name: str, block: str) -> int:
    text = str(key).strip().lower()
    match = re.fullmatch(r"l(\d+)", text)
    if not match:
        raise ValueError(
            f"Architecture '{arch_name}' has invalid key '{key}' in {block}."
        )
    return int(match.group(1))


def _parse_hierarchy(raw: Mapping[str, Any]) -> Hierarchy:
    arch_name = str(raw.get("name", "<unknown>"))
    hierarchy_raw = raw.get("hierarchy")
    if not isinstance(hierarchy_raw, Mapping):
        raise ValueError(f"Architecture '{arch_name}' missing 'hierarchy'.")

    levels: dict[int, HierarchyLevelSpec] = {}
    for key, value in hierarchy_raw.items():
        level = _parse_level_key(key, arch_name, "hierarchy")
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Architecture '{arch_name}' hierarchy.{key} must be a mapping."
            )
        kind_raw = value.get("kind")
        count_raw = value.get("count")
        if kind_raw is None:
            raise ValueError(f"Architecture '{arch_name}' missing hierarchy.{key}.kind.")
        if count_raw is None:
            raise ValueError(f"Architecture '{arch_name}' missing hierarchy.{key}.count.")
        try:
            kind = HierLevel(str(kind_raw).strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"Architecture '{arch_name}' has unknown hierarchy kind '{kind_raw}'."
            ) from exc
        count = int(count_raw)
        if count <= 0:
            raise ValueError(
                f"Architecture '{arch_name}' must have hierarchy.{key}.count > 0."
            )
        extra = sorted(str(item) for item in value if item not in {"kind", "count"})
        if extra:
            raise ValueError(
                f"Architecture '{arch_name}' hierarchy.{key} has unsupported keys: {', '.join(extra)}."
            )
        levels[level] = HierarchyLevelSpec(kind=kind, count=count)

    if not levels:
        raise ValueError(f"Architecture '{arch_name}' defines no hierarchy levels.")
    return Hierarchy(levels=levels)


def _parse_bandwidths(raw: Mapping[str, Any]) -> Bandwidths:
    arch_name = str(raw.get("name", "<unknown>"))
    bandwidth_raw = raw.get("bandwidth")
    if not isinstance(bandwidth_raw, Mapping):
        raise ValueError(f"Architecture '{arch_name}' missing 'bandwidth' block.")
    extra = sorted(str(key) for key in bandwidth_raw if key != "host_device")
    if extra:
        raise ValueError(
            f"Architecture '{arch_name}' has unsupported bandwidth keys: {', '.join(extra)}."
        )
    if "host_device" not in bandwidth_raw:
        raise ValueError(
            f"Architecture '{arch_name}' missing bandwidth.host_device."
        )
    return Bandwidths(
        host_device_Bps=_parse_bandwidth_Bps(bandwidth_raw["host_device"])
    )


def _parse_cost_value(
    arch_name: str,
    operation_name: str,
    field_name: str,
    raw_value: Any,
) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Architecture '{arch_name}' has invalid unit_costs.{operation_name}.{field_name}={raw_value}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Architecture '{arch_name}' has non-finite unit_costs.{operation_name}.{field_name}."
        )
    if value < 0:
        raise ValueError(
            f"Architecture '{arch_name}' has negative unit_costs.{operation_name}.{field_name}."
        )
    return value


def _parse_operation_cost(
    arch_name: str,
    costs_raw: Mapping[str, Any],
    operation_name: str,
) -> OperationCost:
    payload = costs_raw.get(operation_name)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Architecture '{arch_name}' missing unit_costs.{operation_name}."
        )
    extra = sorted(str(key) for key in payload if key not in {"latency", "energy"})
    if extra:
        raise ValueError(
            f"Architecture '{arch_name}' unit_costs.{operation_name} has unsupported keys: {', '.join(extra)}."
        )
    if "latency" not in payload:
        raise ValueError(
            f"Architecture '{arch_name}' missing unit_costs.{operation_name}.latency."
        )
    if "energy" not in payload:
        raise ValueError(
            f"Architecture '{arch_name}' missing unit_costs.{operation_name}.energy."
        )
    return OperationCost(
        latency=_parse_cost_value(
            arch_name, operation_name, "latency", payload["latency"]
        ),
        energy=_parse_cost_value(
            arch_name, operation_name, "energy", payload["energy"]
        ),
    )


def _parse_optional_operation_cost(
    arch_name: str,
    costs_raw: Mapping[str, Any],
    operation_name: str,
) -> OperationCost | None:
    if operation_name not in costs_raw:
        return None
    return _parse_operation_cost(arch_name, costs_raw, operation_name)


def _parse_unit_costs(raw: Mapping[str, Any]) -> UnitCosts:
    arch_name = str(raw.get("name", "<unknown>"))
    costs_raw = raw.get("unit_costs")
    if not isinstance(costs_raw, Mapping):
        raise ValueError(f"Architecture '{arch_name}' missing 'unit_costs' block.")

    required_operations = {
        "bank_to_pu_load",
        "pu_to_bank_store",
        "host_pu_transfer",
        "bank_to_bank_transfer",
        "host_bank_row_conflict",
        "pu_compute",
        "row_change",
    }
    optional_operations = {"base_die", "channel_level"}
    extra = sorted(
        str(key)
        for key in costs_raw
        if key not in required_operations and key not in optional_operations
    )
    if extra:
        raise ValueError(
            f"Architecture '{arch_name}' has unsupported unit_costs keys: {', '.join(extra)}."
        )

    return UnitCosts(
        bank_to_pu_load=_parse_operation_cost(
            arch_name, costs_raw, "bank_to_pu_load"
        ),
        pu_to_bank_store=_parse_operation_cost(
            arch_name, costs_raw, "pu_to_bank_store"
        ),
        host_pu_transfer=_parse_operation_cost(
            arch_name, costs_raw, "host_pu_transfer"
        ),
        bank_to_bank_transfer=_parse_operation_cost(
            arch_name, costs_raw, "bank_to_bank_transfer"
        ),
        host_bank_row_conflict=_parse_operation_cost(
            arch_name, costs_raw, "host_bank_row_conflict"
        ),
        pu_compute=_parse_operation_cost(arch_name, costs_raw, "pu_compute"),
        row_change=_parse_operation_cost(arch_name, costs_raw, "row_change"),
        base_die=_parse_optional_operation_cost(arch_name, costs_raw, "base_die"),
        channel_level=_parse_optional_operation_cost(
            arch_name, costs_raw, "channel_level"
        ),
    )


def _parse_cache_spec(raw: Mapping[str, Any]) -> ArchCacheSpec | None:
    arch_name = str(raw.get("name", "<unknown>"))
    cache_raw = raw.get("cache")
    if cache_raw is None:
        return None
    if not isinstance(cache_raw, Mapping):
        raise ValueError(f"Architecture '{arch_name}' cache block must be a mapping.")
    mode_raw = cache_raw.get("mode")
    if mode_raw is None:
        raise ValueError(f"Architecture '{arch_name}' cache block missing mode.")
    vector_width_raw = cache_raw.get("vector_width_bits")
    extra = sorted(str(key) for key in cache_raw if key not in {"mode", "vector_width_bits"})
    if extra:
        raise ValueError(
            f"Architecture '{arch_name}' cache block has unsupported keys: {', '.join(extra)}."
        )
    try:
        return ArchCacheSpec(mode=str(mode_raw).strip().lower(), vector_width_bits=vector_width_raw)
    except ValueError as exc:
        raise ValueError(f"Architecture '{arch_name}' cache block is invalid: {exc}") from exc


def _validate_arch(
    name: str,
    specs: Specs,
    hierarchy: Hierarchy,
    bandwidths: Bandwidths,
    cache: ArchCacheSpec | None,
) -> None:
    if specs.row_bytes <= 0:
        raise ValueError(f"Architecture '{name}' must have row_bytes > 0.")
    if specs.bank_bytes % specs.row_bytes != 0:
        raise ValueError(
            f"Architecture '{name}' must have bank_bytes divisible by row_bytes."
        )
    if hierarchy.total_units <= 0:
        raise ValueError(f"Architecture '{name}' must define at least one hierarchy unit.")
    if specs.banks_per_pu <= 0:
        raise ValueError(f"Architecture '{name}' must have banks_per_pu > 0.")
    if bandwidths.host_device_Bps < 0:
        raise ValueError(f"Architecture '{name}' bandwidth.host_device must be non-negative.")
    if cache is not None and specs.cache_bytes <= 0:
        raise ValueError(
            f"Architecture '{name}' must have cache_bytes > 0 when cache is defined."
        )


def _parse_arch_mapping(
    raw: Mapping[str, Any],
) -> tuple[str, Specs, Hierarchy, Bandwidths, UnitCosts, ArchCacheSpec | None]:
    allowed_top_level = {
        "name",
        "cache",
        "specs",
        "hierarchy",
        "bandwidth",
        "unit_costs",
        "total_area_mm2",
        # Recognized-but-ignored: provenance/metadata scalars emitted by the
        # arch generator that the analytical model does not consume. Accepting
        # them keeps generated arch files (e.g. the geometry- and buffer-sweep
        # archs, which carry a per-config pim_overhead area fraction) loadable.
        "pim_overhead",
    }
    extra = sorted(str(key) for key in raw if key not in allowed_top_level)
    if extra:
        arch_name = str(raw.get("name", "<unknown>"))
        raise ValueError(
            f"Architecture '{arch_name}' has unsupported top-level keys: {', '.join(extra)}."
        )
    specs = _parse_specs(raw)
    hierarchy = _parse_hierarchy(raw)
    bandwidths = _parse_bandwidths(raw)
    unit_costs = _parse_unit_costs(raw)
    cache = _parse_cache_spec(raw)
    name = str(raw.get("name", "<unknown>"))
    _validate_arch(name, specs, hierarchy, bandwidths, cache)
    return name, specs, hierarchy, bandwidths, unit_costs, cache


class Arch:
    """Architecture description parsed from a single YAML file."""

    def __init__(
        self,
        name: str,
        specs: Specs,
        hierarchy: Hierarchy,
        bandwidths: Bandwidths,
        unit_costs: UnitCosts,
        cache: ArchCacheSpec | None,
        raw: Mapping[str, Any],
    ) -> None:
        self._name = name
        self.specs = specs
        self.hierarchy = hierarchy
        self.bandwidths = bandwidths
        self.unit_costs = unit_costs
        self.cache = cache
        self._raw = dict(raw)
        self._level_capacities = {
            int(level): int(spec.count) for level, spec in hierarchy.levels.items()
        }

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "Arch":
        yaml_path = Path(path)
        with yaml_path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp)
        return cls.from_dict(
            raw or {}, source=str(yaml_path), default_name=yaml_path.stem
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_text: str,
        *,
        source: str | None = None,
        default_name: str = "<unknown>",
    ) -> "Arch":
        raw = yaml.safe_load(yaml_text)
        return cls.from_dict(raw or {}, source=source, default_name=default_name)

    @classmethod
    def from_dict(
        cls,
        payload: Any,
        *,
        source: str | None = None,
        default_name: str = "<unknown>",
    ) -> "Arch":
        if not isinstance(payload, Mapping):
            source_label = source or "<payload>"
            raise ArchParseError(
                f"Architecture source '{source_label}' must contain a mapping."
            )

        raw = dict(payload)
        try:
            name, specs, hierarchy, bandwidths, unit_costs, cache = _parse_arch_mapping(raw)
        except ValueError as exc:
            raise ArchParseError(str(exc)) from exc

        resolved_name = str(raw.get("name", default_name if default_name else name))
        return cls(
            name=resolved_name,
            specs=specs,
            hierarchy=hierarchy,
            bandwidths=bandwidths,
            unit_costs=unit_costs,
            cache=cache,
            raw=raw,
        )

    @property
    def row_size_bytes(self) -> int:
        return self.specs.row_bytes

    @property
    def num_rows(self) -> int:
        return self.specs.bank_bytes // self.specs.row_bytes

    @property
    def num_pus(self) -> int:
        return self.hierarchy.total_units

    @property
    def num_banks(self) -> int:
        return self.num_pus * self.specs.banks_per_pu

    @property
    def levels(self) -> int:
        if not self._level_capacities:
            return 0
        max_level = max(self._level_capacities)
        return max_level + 1

    @property
    def defined_levels(self) -> tuple[int, ...]:
        return tuple(sorted(self._level_capacities.keys()))

    @property
    def level_capacities(self) -> Mapping[int, int]:
        return dict(self._level_capacities)

    def has_level(self, level: int) -> bool:
        return int(level) in self._level_capacities

    def level_capacity(self, level: int) -> int:
        level_idx = int(level)
        try:
            return self._level_capacities[level_idx]
        except KeyError as exc:
            raise KeyError(
                f"Architecture '{self._name}' does not define hierarchy level L{level_idx}."
            ) from exc

    @property
    def simd_lanes(self) -> int:
        return int(self.specs.simd_lanes)

    @property
    def pu_concurrency(self) -> int:
        return self.simd_lanes

    def _cycle_time_ns(self) -> float:
        freq_hz = self.specs.pu_frequency_hz
        if freq_hz <= 0:
            raise ValueError(
                f"Architecture '{self._name}' must have pu_frequency_hz > 0 to convert between time and cycles."
            )
        return 1e9 / freq_hz

    def cycles_to_ns(self, cycles: float) -> float:
        if cycles <= 0:
            return 0.0
        return cycles * self._cycle_time_ns()

    def ns_to_cycles(self, time_ns: float) -> float:
        if time_ns <= 0:
            return 0.0
        return time_ns / self._cycle_time_ns()

    def bandwidth_bytes_per_ns(self) -> float:
        return self.bandwidths.host_device_Bps / 1e9

    def reduction_levels(self) -> Sequence[ReductionLevel]:
        return ()
