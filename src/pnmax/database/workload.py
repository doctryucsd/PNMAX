"""Utilities to read workload description files.

The parser focuses on the ``*.yaml`` workload files found under
``workloads/samples``.  It provides a small, well-typed object model so the
rest of the codebase can query workload metadata without dealing with YAML
details.  PyYAML is required for parsing; install it with::

    pip install PyYAML
"""

from __future__ import annotations

import ast
import math
import re
from collections import OrderedDict
from collections.abc import Sequence as AbcSequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import yaml


class WorkloadParseError(ValueError):
    """Raised when a workload file is malformed or missing mandatory fields."""


@dataclass
class LoopBound:
    """Represents a single loop bound entry such as ``P1: 7``."""

    loop_var: str
    level: int
    extent: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", _canonical_dimension_key(self.loop_var))
        object.__setattr__(self, "level", int(self.level))
        object.__setattr__(self, "extent", int(self.extent))

    @property
    def identifier(self) -> str:
        """Return the canonical dimension/level identifier, e.g. ``N0``."""
        return f"{self.loop_var}{self.level}"

    def __repr__(self) -> str:
        return f"{self.identifier}: {self.extent}"


@dataclass
class TensorAttrSpec:
    """Describes tensor metadata found under ``TensorAttr``."""

    bits: int
    role: Optional[str] = None


@dataclass
class LayoutSharing:
    """Sharing flags nested under ``Layout-Pragma``."""

    block: bool = False
    pu: bool = False
    system: int = 0


@dataclass
class LayoutStreaming:
    """Streaming pragmas for tensors in ``Layout-Pragma``."""

    In: bool = False
    F: bool = False
    Out: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return {"In": self.In, "F": self.F, "Out": self.Out}


@dataclass(frozen=True)
class LayoutCacheGroup:
    """One named cache partition and the tensors it serves."""

    name: str
    tensors: Tuple[str, ...]
    weight: Optional[float] = None

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        if not normalized_name:
            raise WorkloadParseError("Cache group name cannot be empty.")
        object.__setattr__(self, "name", normalized_name)

        seen_tensors: set[str] = set()
        normalized_tensors: list[str] = []
        for tensor in self.tensors:
            normalized_tensor = str(tensor).strip()
            if not normalized_tensor:
                raise WorkloadParseError(
                    f"Cache group '{normalized_name}' cannot contain an empty tensor name."
                )
            if normalized_tensor in seen_tensors:
                raise WorkloadParseError(
                    f"Cache group '{normalized_name}' lists tensor '{normalized_tensor}' more than once."
                )
            seen_tensors.add(normalized_tensor)
            normalized_tensors.append(normalized_tensor)
        if not normalized_tensors:
            raise WorkloadParseError(
                f"Cache group '{normalized_name}' must include at least one tensor."
            )
        object.__setattr__(self, "tensors", tuple(normalized_tensors))

        if self.weight is not None:
            try:
                normalized_weight = float(self.weight)
            except (TypeError, ValueError) as exc:
                raise WorkloadParseError(
                    f"Cache group '{normalized_name}' has invalid weight {self.weight!r}."
                ) from exc
            if not math.isfinite(normalized_weight) or normalized_weight <= 0:
                raise WorkloadParseError(
                    f"Cache group '{normalized_name}' must define a positive finite weight."
                )
            object.__setattr__(self, "weight", normalized_weight)


@dataclass(frozen=True)
class LayoutCacheConfig:
    """Resolved cache configuration for one workload layout."""

    enabled: bool
    groups: Tuple[LayoutCacheGroup, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        normalized_groups: list[LayoutCacheGroup] = []
        seen_names: set[str] = set()
        for group in self.groups:
            if not isinstance(group, LayoutCacheGroup):
                raise TypeError(
                    "LayoutCacheConfig.groups must contain LayoutCacheGroup instances."
                )
            if group.name in seen_names:
                raise WorkloadParseError(
                    f"Duplicate cache group name '{group.name}' is not allowed."
                )
            seen_names.add(group.name)
            normalized_groups.append(group)
        if not self.enabled and normalized_groups:
            raise WorkloadParseError(
                "Disabled cache configuration cannot define cache groups."
            )
        object.__setattr__(self, "groups", tuple(normalized_groups))

    def tensor_to_group(self) -> Dict[str, LayoutCacheGroup]:
        mapping: Dict[str, LayoutCacheGroup] = {}
        for group in self.groups:
            for tensor in group.tensors:
                mapping[tensor] = group
        return mapping

    def group_for_tensor(self, tensor_name: str) -> Optional[LayoutCacheGroup]:
        return self.tensor_to_group().get(str(tensor_name).strip())


@dataclass
class LayoutPragma:
    """Layout configuration for matrix tiles, cache, and streaming behavior.

    The YAML tokens ``bit-parallel``/``bit-serial`` are normalised into the
    boolean field :attr:`bit_bool`. Row-major traversal is now implicit and no
    longer configured through a pragma.
    """

    bit_bool: Optional[bool] = field(default=True, repr=False)
    cache: Optional[bool] = None
    cache_groups: Tuple[LayoutCacheGroup, ...] = field(default_factory=tuple)
    streaming: LayoutStreaming = field(default_factory=LayoutStreaming)
    sharing: Optional[LayoutSharing] = None
    interleaving: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_initialized", False)
        self.refresh()
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if getattr(self, "_initialized", False) and name in {
            "bit_bool",
            "cache",
            "cache_groups",
            "streaming",
            "sharing",
            "interleaving",
        }:
            self.refresh()

    def refresh(self) -> None:
        """Normalise streaming dictionary and ensure types remain consistent."""
        if self.bit_bool is not None and not isinstance(self.bit_bool, bool):
            raise TypeError("LayoutPragma.bit_bool must be a bool or None.")
        if self.cache is not None and not isinstance(self.cache, bool):
            raise TypeError("LayoutPragma.cache must be a bool or None.")

        object.__setattr__(
            self, "cache_groups", _coerce_cache_groups_value(self.cache_groups)
        )
        object.__setattr__(self, "streaming", _coerce_streaming_value(self.streaming))

    @property
    def bit(self) -> Optional[str]:
        """Return the bit-processing mode as a string token."""
        return self.bit_to_string()

    @bit.setter
    def bit(self, value: Union[bool, str, None]) -> None:
        self.bit_bool = _coerce_bit_value(value)

    def bit_to_string(self) -> Optional[str]:
        if self.bit_bool is None:
            return None
        return "bit-parallel" if self.bit_bool else "bit-serial"


@dataclass(frozen=True)
class RequiredLayout:
    """Fully specified layout directives required by execution backends."""

    bit_parallel: bool
    cache: bool
    cache_config: LayoutCacheConfig
    streaming: LayoutStreaming
    sharing: LayoutSharing
    interleaving: bool

    @property
    def bit(self) -> str:
        return "bit-parallel" if self.bit_parallel else "bit-serial"


@dataclass(frozen=True)
class TensorAccessPattern:
    """Single tensor access extracted from ``Compute``."""

    tensor: str
    indices: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensor", self.tensor.strip())
        object.__setattr__(
            self, "indices", tuple(index.strip() for index in self.indices)
        )


@dataclass
class ComputeSpec:
    """Parsed view of the workload ``Compute`` expression."""

    target: TensorAccessPattern
    operands: Tuple[TensorAccessPattern, ...]
    _tensor_map: Mapping[str, TensorAccessPattern] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mapping: Dict[str, TensorAccessPattern] = {self.target.tensor: self.target}
        for operand in self.operands:
            mapping.setdefault(operand.tensor, operand)
        object.__setattr__(self, "_tensor_map", mapping)

    def tensor(self, name: str) -> TensorAccessPattern:
        try:
            return self._tensor_map[name]
        except KeyError as exc:  # pragma: no cover - delegated error
            raise KeyError(f"Tensor '{name}' does not appear in Compute.") from exc

    @property
    def tensors(self) -> Mapping[str, TensorAccessPattern]:
        return self._tensor_map


@dataclass
class ProblemSpec:
    """Problem dimensions and convolution-specific stride/dilation values."""

    loop_vars: Dict[str, int]
    dilation: Optional[Tuple[int, ...]] = None
    stride: Optional[Tuple[int, ...]] = None
    N: Optional[int] = field(init=False, default=None, repr=False)
    K: Optional[int] = field(init=False, default=None, repr=False)
    P: Optional[int] = field(init=False, default=None, repr=False)
    Q: Optional[int] = field(init=False, default=None, repr=False)
    C: Optional[int] = field(init=False, default=None, repr=False)
    R: Optional[int] = field(init=False, default=None, repr=False)
    S: Optional[int] = field(init=False, default=None, repr=False)

    def extent(self, name: str) -> int:
        """Return the extent of a named dimension, raising if it is absent."""
        try:
            return self.loop_vars[name]
        except KeyError as exc:  # pragma: no cover - simple delegation
            raise KeyError(f"Unknown problem dimension '{name}'") from exc

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Normalize dimension keys and expose canonical attributes."""
        dims = {k.upper(): int(v) for k, v in self.loop_vars.items()}
        object.__setattr__(self, "dimensions", dims)

        if self.dilation is not None:
            dilation = tuple(int(x) for x in self.dilation)
            if len(dilation) != 2:
                raise WorkloadParseError(
                    "Problem.dilation must contain exactly two integers."
                )
            object.__setattr__(self, "dilation", dilation)
        if self.stride is not None:
            stride = tuple(int(x) for x in self.stride)
            if len(stride) != 2:
                raise WorkloadParseError(
                    "Problem.stride must contain exactly two integers."
                )
            object.__setattr__(self, "stride", stride)

        for base in _DIMENSION_BASES:
            object.__setattr__(self, base, dims.get(base))

    def __setattr__(self, name, value):
        if name in _DIMENSION_BASES and hasattr(self, "loop_vars"):
            updated = dict(self.loop_vars)
            updated[name] = int(value)
            object.__setattr__(self, "loop_vars", updated)
            self.refresh()
        else:
            object.__setattr__(self, name, value)
            if name in {"loop_vars", "dilation", "stride"} and hasattr(
                self, "loop_vars"
            ):
                self.refresh()


class LoopBounds:
    """Lightweight wrapper over the YAML ``Bound`` mapping."""

    def __init__(
        self, data: Optional[Mapping[int, Sequence[Mapping[str, Any]]]] = None
    ) -> None:
        self._data: OrderedDict[int, List[Dict[str, int]]] = OrderedDict()
        self._levels: Tuple[int, ...] = tuple()
        self._lookup: Dict[Tuple[str, int], int] = {}
        if data:
            temp: Dict[int, List[Dict[str, int]]] = {}
            for raw_level, entries in data.items():
                level = _coerce_bound_level_index(raw_level, None)
                sequence = _require_sequence(entries, f"Bound[{level}]", None)
                temp[level] = _normalize_bound_level(
                    sequence, context=f"Bound[{level}]"
                )
            for level in sorted(temp.keys(), reverse=True):
                self._data[level] = temp[level]
        self.refresh()

    # -- Collection protocol ----------------------------------------------------
    def __len__(self) -> int:
        return len(self._levels)

    def __iter__(self):
        return iter(self._levels)

    def has_level(self, level: int) -> bool:
        """Return True when *level* is explicitly defined in this bound hierarchy."""
        return int(level) in self._data

    def get_level(self, level: int) -> Tuple[LoopBound, ...]:
        """Return loop bounds for one level as an immutable tuple."""
        resolved = self._resolve_level_index(level)
        return tuple(self._copy_level(resolved))

    def extent(self, name: str, *, level: Optional[int] = None) -> int:
        """Return the extent for a hierarchical loop."""
        dimension, lvl = _resolve_dimension_level(name, level)
        try:
            return self._lookup[(dimension, lvl)]
        except KeyError as exc:
            raise KeyError(f"Unknown loop bound '{dimension}' at level {lvl}.") from exc

    def levels(self) -> Tuple[int, ...]:
        """Return the defined loop levels as an ordered tuple."""
        return self._levels

    def extent_items(self) -> Iterable[Tuple[str, int, int]]:
        """Iterate over (dimension, level, extent) entries in canonical ordering."""
        for level in self._levels:
            for entry in self._data[level]:
                ((dimension, extent),) = entry.items()
                yield dimension, level, extent

    def refresh(self) -> None:
        """Rebuild cached views after in-place edits."""
        if self._data:
            _validate_bound_levels(list(self._data.keys()), None)

        normalized: "OrderedDict[int, List[Dict[str, int]]]" = OrderedDict()
        lookup: Dict[Tuple[str, int], int] = {}

        for level in sorted(self._data.keys(), reverse=True):
            normalized[level] = _normalize_bound_level(
                self._data[level], context=f"Bound[{level}]"
            )
            for entry in normalized[level]:
                ((dimension, extent),) = entry.items()
                lookup[(dimension, level)] = extent

        self._data = normalized
        self._levels = tuple(normalized.keys())
        self._lookup = lookup

    def set_extent(
        self, name: str, extent: int, *, level: Optional[int] = None
    ) -> None:
        """Update or append a loop extent while preserving ordering semantics."""
        dimension, lvl = _resolve_dimension_level(name, level)
        extent_value = _require_int(extent, f"Bound[{dimension}{lvl}]", None)
        level_entries = {
            dim: val for entry in self._data.get(lvl, []) for dim, val in entry.items()
        }
        level_entries[dimension] = extent_value
        ordered = [
            {dim: level_entries[dim]}
            for dim in sorted(level_entries.keys(), key=_dimension_rank)
        ]
        self._data[lvl] = ordered
        self.refresh()

    def remove_extent(self, name: str, *, level: Optional[int] = None) -> None:
        """Remove a loop extent by name."""
        dimension, lvl = _resolve_dimension_level(name, level)
        if lvl not in self._data:
            raise KeyError(f"{dimension} at level {lvl}")
        remaining: List[Dict[str, int]] = []
        removed = False
        for entry in self._data[lvl]:
            ((entry_dim, entry_extent),) = entry.items()
            if entry_dim == dimension:
                removed = True
                continue
            remaining.append({entry_dim: entry_extent})
        if not removed:
            raise KeyError(f"{dimension} at level {lvl}")
        if remaining:
            self._data[lvl] = remaining
        else:
            del self._data[lvl]
        self.refresh()

    # -- Internal helpers -------------------------------------------------------
    def _resolve_level_index(self, key: int) -> int:
        if key >= 0 and key in self._data:
            return key
        try:
            return self._levels[key]
        except IndexError as exc:
            raise KeyError(key) from exc

    def _copy_level(self, level: int) -> List[LoopBound]:
        if level not in self._data:
            raise KeyError(level)
        return [
            LoopBound(loop_var=loop_var, level=level, extent=extent)
            for entry in self._data[level]
            for loop_var, extent in entry.items()
        ]

    def __repr__(self) -> str:
        parts: List[str] = []
        for level in self._levels:
            loop_bounds = self._copy_level(level)
            parts.append(f"level-{level}: {loop_bounds}")
        joined = "\n".join(parts)
        return f"LoopBounds\n{joined}" if joined else "LoopBounds"


@dataclass
class Workload:
    """Parsed representation of a workload description.

    The complete dimension map (problem dimensions and loop splits) is exposed
    through :attr:`dimension_values`.  Canonical dimensions live on
    :attr:`problem`, while hierarchical loop splits can be accessed via
    :attr:`bounds`.
    """

    name: str
    problem: ProblemSpec
    bounds: LoopBounds
    tensor_attrs: Mapping[str, TensorAttrSpec]
    compute: str
    layout: Optional[LayoutPragma] = None
    compute_spec: ComputeSpec = field(init=False, repr=False)
    _tensor_dimension_map: Optional[Mapping[str, Tuple[str, ...]]] = field(
        init=False, repr=False, default=None
    )
    _compiled_index_exprs: Optional[Mapping[str, Tuple[ast.AST, ...]]] = field(
        init=False, repr=False, default=None
    )

    @classmethod
    def from_file(cls, path: Path | str) -> "Workload":
        """Load a workload from a YAML file located at *path*."""
        if yaml is None:  # pragma: no cover - runtime dependency guard
            raise ImportError(
                "Workload parser requires PyYAML; install with 'pip install PyYAML'."
            )

        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if data is None:
            raise WorkloadParseError(f"{file_path} is empty.")
        return cls._from_mapping(data, source=str(file_path))

    @classmethod
    def from_yaml(cls, yaml_text: str, *, source: str | None = None) -> "Workload":
        """Parse a workload directly from a YAML string."""
        if yaml is None:  # pragma: no cover - runtime dependency guard
            raise ImportError(
                "Workload parser requires PyYAML; install with 'pip install PyYAML'."
            )

        data = yaml.safe_load(yaml_text)
        if data is None:
            raise WorkloadParseError("Cannot parse empty YAML payload.")
        return cls._from_mapping(data, source=source)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, source: str | None = None
    ) -> "Workload":
        """Build a workload from a Python mapping that follows the YAML schema."""
        return cls._from_mapping(payload, source=source)

    @classmethod
    def _from_mapping(
        cls, payload: Mapping[str, Any], *, source: str | None
    ) -> "Workload":
        parsed = _parse_workload_mapping(payload, source)
        return cls(
            name=parsed.name,
            problem=parsed.problem,
            bounds=parsed.bounds,
            tensor_attrs=parsed.tensor_attrs,
            compute=parsed.compute,
            layout=parsed.layout,
        )

    def __post_init__(self) -> None:
        self.refresh()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in {"problem", "bounds", "compute"} and all(
            hasattr(self, attr) for attr in ("problem", "bounds", "compute")
        ):
            self.refresh()

    def refresh(self) -> None:
        """Recompute cached views after in-place edits."""
        if isinstance(self.problem, ProblemSpec):
            self.problem.refresh()
        if isinstance(self.bounds, LoopBounds):
            self.bounds.refresh()
        layout = getattr(self, "layout", None)
        if isinstance(layout, LayoutPragma):
            layout.refresh()
        object.__setattr__(self, "compute_spec", _parse_compute_spec(self.compute))
        # Lazily populated when dimension/compute-expression helpers are used.
        object.__setattr__(self, "_tensor_dimension_map", None)
        object.__setattr__(self, "_compiled_index_exprs", None)
        self.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation matching the workload YAML schema."""
        self.validate()
        return _serialize_workload(self)

    def to_yaml(self) -> str:
        """Return the workload encoded as YAML."""
        if yaml is None:  # pragma: no cover - runtime dependency guard
            raise ImportError(
                "Workload parser requires PyYAML; install with 'pip install PyYAML'."
            )
        self.validate()
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def to_file(self, path: Path | str) -> None:
        """Write the workload to *path* in YAML format."""
        self.validate()
        file_path = Path(path)
        file_path.write_text(self.to_yaml(), encoding="utf-8")

    @property
    def dimension_values(self) -> Dict[str, int]:
        return _collect_dimension_values(self.problem, self.bounds)

    def validate(self) -> None:
        """Run structural checks on problem dimensions and loop hierarchy."""
        _validate_dimension_products(self.name, self.dimension_values)
        layout = self.layout
        if isinstance(layout, LayoutPragma) and layout.cache is not None:
            self.resolve_cache_config()

    def get_tensor_loop_bounds(self, tensor_name: str) -> Dict[int, Dict[str, int]]:
        """Return per-level loop extents for the requested tensor."""
        dimensions = self._tensor_dimensions(tensor_name)
        per_level: "OrderedDict[int, Dict[str, int]]" = OrderedDict()
        for level in sorted(self.bounds.levels()):
            mapping = self.get_level_bounds(level)
            per_level[level] = {dim: mapping.get(dim, 1) for dim in dimensions}
        return per_level

    def get_tensor_level_bounds(self, tensor_name: str, level: int) -> Dict[str, int]:
        """Return the loop extents for a tensor at a specific level."""
        tensor_bounds = self.get_tensor_loop_bounds(tensor_name)
        try:
            return tensor_bounds[level]
        except KeyError as exc:
            raise KeyError(
                f"Level {level} is not defined for tensor '{tensor_name}'."
            ) from exc

    def get_level_bounds(self, level: int) -> Dict[str, int]:
        """Return the canonical {dimension: extent} mapping for *level*."""
        resolved_level = _coerce_bound_level_index(level, None)
        if not self.bounds.has_level(resolved_level):
            raise KeyError(f"Loop level {resolved_level} is not defined.")
        return {
            bound.loop_var: bound.extent
            for bound in self.bounds.get_level(resolved_level)
        }

    def _tensor_dimensions(self, tensor_name: str) -> Tuple[str, ...]:
        if not isinstance(tensor_name, str):
            raise TypeError("tensor_name must be a string.")
        canonical = tensor_name.strip()
        if not canonical:
            raise ValueError("tensor_name cannot be empty.")
        tensor_dimension_map = self._ensure_tensor_dimension_map()
        try:
            return tensor_dimension_map[canonical.lower()]
        except KeyError as exc:
            raise KeyError(f"Tensor '{tensor_name}' is not supported.") from exc

    def _ensure_tensor_dimension_map(self) -> Mapping[str, Tuple[str, ...]]:
        mapping = self._tensor_dimension_map
        if mapping is None:
            mapping = _derive_tensor_dimensions(self.compute_spec)
            object.__setattr__(self, "_tensor_dimension_map", mapping)
        return mapping

    def _ensure_compiled_index_exprs(self) -> Mapping[str, Tuple[ast.AST, ...]]:
        compiled = self._compiled_index_exprs
        if compiled is None:
            compiled = _compile_tensor_index_expressions(self.compute_spec)
            object.__setattr__(self, "_compiled_index_exprs", compiled)
        return compiled

    def require_layout(self) -> RequiredLayout:
        """Return a fully specified layout contract."""
        layout = self.layout
        if layout is None:
            raise WorkloadParseError(
                f"Workload '{self.name}' missing Layout-Pragma section."
            )
        if layout.sharing is None:
            raise WorkloadParseError(
                f"Workload '{self.name}' must define Layout-Pragma.sharing."
            )
        if layout.bit_bool is None:
            raise WorkloadParseError(
                f"Workload '{self.name}' must specify Layout-Pragma.bit as 'bit-parallel' or 'bit-serial'."
            )
        if layout.cache is None:
            raise WorkloadParseError(
                f"Workload '{self.name}' must specify Layout-Pragma.cache as true/false."
            )
        interleaving = layout.interleaving
        if interleaving is None:
            raise WorkloadParseError(
                f"Workload '{self.name}' must specify Layout-Pragma.interleaving as true/false."
            )
        if not isinstance(interleaving, bool):
            raise WorkloadParseError(
                f"Layout-Pragma.interleaving must be boolean for workload '{self.name}'."
            )
        return RequiredLayout(
            bit_parallel=layout.bit_bool,
            cache=layout.cache,
            cache_config=self.resolve_cache_config(),
            streaming=layout.streaming,
            sharing=layout.sharing,
            interleaving=interleaving,
        )

    def resolve_cache_config(
        self, override: LayoutCacheConfig | Mapping[str, Any] | bool | None = None
    ) -> LayoutCacheConfig:
        """Resolve and validate cache grouping for the workload."""
        layout = self.layout
        if layout is None:
            if override is None:
                return LayoutCacheConfig(enabled=False, groups=())
            raise WorkloadParseError(
                f"Workload '{self.name}' missing Layout-Pragma section."
            )

        if override is None:
            if layout.cache is None:
                raise WorkloadParseError(
                    f"Workload '{self.name}' must specify Layout-Pragma.cache as true/false."
                )
            config = LayoutCacheConfig(
                enabled=layout.cache,
                groups=tuple(layout.cache_groups),
            )
        else:
            config = _coerce_cache_config_override(override)

        return _resolve_layout_cache_config_for_workload(
            workload=self,
            config=config,
        )

    def ordered_tensor_access_patterns(self) -> Tuple[TensorAccessPattern, ...]:
        """Return Compute tensor accesses in execution-oriented order."""
        spec = self.compute_spec
        ordered: List[TensorAccessPattern] = []
        added: set[str] = set()

        if "In" in spec.tensors:
            ordered.append(spec.tensors["In"])
            added.add("In")

        for weight_name in ("F", "Weight", "W"):
            if weight_name in spec.tensors and weight_name not in added:
                ordered.append(spec.tensors[weight_name])
                added.add(weight_name)
                break

        for operand in spec.operands:
            if operand.tensor not in added and operand.tensor != spec.target.tensor:
                ordered.append(operand)
                added.add(operand.tensor)

        if spec.target.tensor not in added:
            ordered.append(spec.target)
        return tuple(ordered)

    def streaming_flags(self) -> Dict[str, bool]:
        """Return {tensor_name: is_streaming} from Layout-Pragma."""
        if self.layout is None:
            return {}
        return self.layout.streaming.as_dict()

    def compiled_index_expressions(self, tensor_name: str) -> Tuple[ast.AST, ...]:
        """Return precompiled index-expression AST nodes for a tensor."""
        if not isinstance(tensor_name, str):
            raise TypeError("tensor_name must be a string.")
        canonical = tensor_name.strip()
        if not canonical:
            raise ValueError("tensor_name cannot be empty.")
        compiled_map = self._ensure_compiled_index_exprs()
        compiled = compiled_map.get(canonical)
        if compiled is None:
            compiled = compiled_map.get(canonical.lower())
        if compiled is None:
            raise KeyError(f"Tensor '{tensor_name}' does not appear in Compute.")
        return compiled

    def evaluate_index_expression(
        self, expression: str, variables: Mapping[str, int]
    ) -> int:
        """Evaluate one index expression under a loop-variable mapping."""
        node = _parse_index_expression_ast(expression)
        normalized = _normalize_eval_variables(variables)
        return _evaluate_ast(node, normalized)

    def evaluate_tensor_indices(
        self, tensor_name: str, variables: Mapping[str, int]
    ) -> Tuple[int, ...]:
        """Evaluate all index expressions for one tensor access."""
        compiled = self.compiled_index_expressions(tensor_name)
        normalized = _normalize_eval_variables(variables)
        return tuple(_evaluate_ast(expr, normalized) for expr in compiled)

    def require_unindp_layout(self) -> RequiredLayout:
        """Validate and return UniNDP-specific layout requirements."""
        required = self.require_layout()
        if not required.bit_parallel:
            raise WorkloadParseError(
                f"UniNDP backend does not support bit-serial layout (workload '{self.name}')."
            )
        return required

    def validate_attacc_layout(self) -> None:
        """Validate minimal layout requirements for the AttAcc backend."""
        layout = self.layout
        if layout is None:
            raise WorkloadParseError(
                f"Workload '{self.name}' is missing Layout-Pragma directives; AttAcc requires explicit layout metadata."
            )
        if layout.bit_bool is False:
            raise WorkloadParseError(
                f"Workload '{self.name}' requests bit-serial layout which is not supported by the AttAcc backend."
            )
        if layout.cache is None:
            raise WorkloadParseError(
                f"Workload '{self.name}' must specify Layout-Pragma.cache as true/false."
            )


@dataclass(frozen=True)
class _ParsedWorkload:
    """Internal container for top-level workload YAML sections."""

    name: str
    problem: ProblemSpec
    bounds: LoopBounds
    tensor_attrs: Mapping[str, TensorAttrSpec]
    compute: str
    layout: Optional[LayoutPragma]


_DEFAULT_CACHE_GROUP_NAME = "default"
_DEFAULT_CACHE_GROUP_TENSORS: Tuple[str, ...] = ("F", "In")
_BITS_RE = re.compile(r"^\s*(\d+)\s*[bB]\s*(?::\s*(\w+))?\s*$")
_COMPUTE_ASSIGN_RE = re.compile(r"^\s*(\w+)\s*\[(.*?)\]\s*([+\-*/]?=)\s*(.+)$")
_TENSOR_ACCESS_RE = re.compile(r"(\w+)\s*\[(.*?)\]")


def _parse_workload_mapping(
    payload: Mapping[str, Any], source: Optional[str]
) -> _ParsedWorkload:
    mapping = _require_mapping(payload, "workload root", source)
    name = _require_str(mapping.get("Name"), "Name", source)
    problem_raw = _require_mapping(mapping.get("Problem"), "Problem", source)
    problem = _parse_problem(problem_raw, source)
    bounds = _parse_bounds(mapping.get("Bound", []), source)
    tensor_attrs = _parse_tensor_attrs(mapping.get("TensorAttr", {}), source)
    compute = _require_str(mapping.get("Compute"), "Compute", source)
    layout_raw = mapping.get("Layout-Pragma")
    layout = _parse_layout(layout_raw, source) if layout_raw is not None else None
    return _ParsedWorkload(
        name=name,
        problem=problem,
        bounds=bounds,
        tensor_attrs=tensor_attrs,
        compute=compute,
        layout=layout,
    )


def _parse_problem(mapping: Mapping[str, Any], source: Optional[str]) -> ProblemSpec:
    dims: Dict[str, int] = {}
    dilation: Optional[Tuple[int, ...]] = None
    stride: Optional[Tuple[int, ...]] = None

    for key, value in mapping.items():
        lower = key.lower()
        if lower == "dilation":
            dilation = _parse_int_tuple(value, key, source)
        elif lower == "stride":
            stride = _parse_int_tuple(value, key, source)
        else:
            canonical = key.upper()
            if canonical not in _DIMENSION_BASES:
                raise WorkloadParseError(
                    _format_error(
                        f"Problem.{key} is not a canonical dimension (expected one of {_DIMENSION_BASES}).",
                        source,
                    )
                )
            dims[canonical] = _require_int(value, f"Problem.{key}", source)

    return ProblemSpec(loop_vars=dims, dilation=dilation, stride=stride)


def _parse_compute_spec(expression: str) -> ComputeSpec:
    expr = expression.strip()
    if not expr:
        raise WorkloadParseError("Compute expression cannot be empty.")
    match = _COMPUTE_ASSIGN_RE.match(expr)
    if not match:
        raise WorkloadParseError(f"Unsupported Compute expression: '{expression}'.")
    target_tensor, target_indices, _, rhs = match.groups()
    target = TensorAccessPattern(target_tensor, _split_index_list(target_indices))
    operands: List[TensorAccessPattern] = []
    for tensor_name, indices_str in _TENSOR_ACCESS_RE.findall(rhs):
        operands.append(
            TensorAccessPattern(tensor_name, _split_index_list(indices_str))
        )
    if not operands:
        raise WorkloadParseError(
            "Compute expression must reference at least one tensor on the right-hand side."
        )
    return ComputeSpec(target=target, operands=tuple(operands))


def _derive_tensor_dimensions(spec: ComputeSpec) -> Dict[str, Tuple[str, ...]]:
    dimensions: Dict[str, Tuple[str, ...]] = {}
    for tensor_name, access in spec.tensors.items():
        ordered_dims: List[str] = []
        for index_expr in access.indices:
            for raw_var in _extract_index_variables(index_expr):
                canonical = _canonical_dimension_key(raw_var)
                if canonical not in ordered_dims:
                    ordered_dims.append(canonical)
        base_order = [dim for dim in _DIMENSION_BASES if dim in ordered_dims]
        extras = [dim for dim in ordered_dims if dim not in _DIMENSION_BASES]
        dimensions[tensor_name.lower()] = tuple(base_order + extras)
    return dimensions


def _extract_index_variables(expression: str) -> Tuple[str, ...]:
    expr = expression.strip()
    if not expr:
        return tuple()
    tree = _parse_index_expression_ast(expr)
    names: List[str] = []
    _collect_index_names(tree, names)
    return tuple(names)


def _collect_index_names(node: ast.AST, names: List[str]) -> None:
    if isinstance(node, ast.Name):
        if node.id not in names:
            names.append(node.id)
    for child in ast.iter_child_nodes(node):
        _collect_index_names(child, names)


def _split_index_list(indices: str) -> Tuple[str, ...]:
    if not indices.strip():
        return tuple()
    items = [entry.strip() for entry in indices.split(",")]
    if any(not entry for entry in items):
        raise WorkloadParseError(f"Invalid index list '{indices}'.")
    return tuple(items)


@lru_cache(maxsize=256)
def _parse_index_expression_ast(expression: str) -> ast.AST:
    expr = expression.strip()
    if not expr:
        raise WorkloadParseError("Index expression cannot be empty.")
    try:
        return ast.parse(expr, mode="eval").body
    except SyntaxError as exc:  # pragma: no cover - delegated validation
        raise WorkloadParseError(f"Invalid index expression '{expression}'.") from exc


def _evaluate_ast(node: ast.AST, variables: Mapping[str, int]) -> int:
    if isinstance(node, ast.BinOp):
        left = _evaluate_ast(node.left, variables)
        right = _evaluate_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        raise ValueError(f"Unsupported binary operator: {ast.dump(node.op)}")
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
    if isinstance(node, ast.Name):
        direct = variables.get(node.id)
        if direct is not None:
            return int(direct)
        lower = variables.get(node.id.lower())
        if lower is not None:
            return int(lower)
        upper = variables.get(node.id.upper())
        if upper is not None:
            return int(upper)
        raise KeyError(f"Undefined loop variable '{node.id}' in index expression.")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int):
            return int(node.value)
        raise ValueError(
            f"Unsupported constant type in index expression: {node.value!r}"
        )
    if isinstance(node, ast.Num):  # pragma: no cover - compatibility
        return int(node.n)  # type: ignore[attr-defined]
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def _normalize_eval_variables(variables: Mapping[str, int]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for name, value in variables.items():
        token = str(name)
        int_value = int(value)
        normalized[token] = int_value
        normalized[token.lower()] = int_value
        normalized[token.upper()] = int_value
    return normalized


def _compile_tensor_index_expressions(
    spec: ComputeSpec,
) -> Dict[str, Tuple[ast.AST, ...]]:
    compiled: Dict[str, Tuple[ast.AST, ...]] = {}
    for tensor_name, access in spec.tensors.items():
        expressions = tuple(
            _parse_index_expression_ast(index_expr) for index_expr in access.indices
        )
        compiled[tensor_name] = expressions
        compiled[tensor_name.lower()] = expressions
    return compiled


def _parse_bounds(payload: Any, source: Optional[str]) -> LoopBounds:
    if payload is None:
        return LoopBounds()

    if isinstance(payload, Mapping):
        data = _parse_bounds_mapping(payload, source)
    else:
        data = _parse_bounds_sequence(payload, source)
    return LoopBounds(data)


def _parse_bounds_mapping(
    payload: Mapping[Any, Any], source: Optional[str]
) -> Dict[int, List[Dict[str, int]]]:
    normalized: Dict[int, List[Dict[str, int]]] = {}
    levels: List[int] = []

    for raw_level, level_entries in payload.items():
        level = _coerce_bound_level_index(raw_level, source)
        levels.append(level)
        entries_seq = _require_sequence(level_entries, f"Bound[{level}]", source)
        normalized[level] = _normalize_bound_level(
            entries_seq, context=f"Bound[{level}]"
        )

    _validate_bound_levels(levels, source)
    return normalized


def _parse_bounds_sequence(
    payload: Any, source: Optional[str]
) -> Dict[int, List[Dict[str, int]]]:
    items = _require_sequence(payload, "Bound", source)
    grouped: Dict[int, List[Dict[str, int]]] = {}

    for idx, item in enumerate(items):
        entry = _require_mapping(item, f"Bound[{idx}]", source)
        if len(entry) != 1:
            raise WorkloadParseError(
                _format_error(
                    f"Bound entries must contain a single mapping, got {entry}.", source
                )
            )
        loop_key, extent_value = next(iter(entry.items()))
        loop_token = str(loop_key)
        try:
            base, level = _split_loop_name(loop_token)
        except ValueError as exc:
            raise WorkloadParseError(
                _format_error(
                    f"Legacy Bound entries must use names like 'N0'; got '{loop_token}'.",
                    source,
                )
            ) from exc
        canonical_dim = _canonical_dimension_key(base)
        extent = _require_int(extent_value, f"Bound.{loop_token}", source)
        grouped.setdefault(level, [])
        if any(canonical_dim in entry for entry in grouped[level]):
            raise WorkloadParseError(
                _format_error(
                    f"Duplicate dimension '{canonical_dim}' found in Bound[{level}].",
                    source,
                )
            )
        grouped[level].append({canonical_dim: extent})

    _validate_bound_levels(list(grouped.keys()), source)
    for level in grouped:
        grouped[level] = _normalize_bound_level(
            grouped[level], context=f"Bound[{level}]"
        )
    return grouped


def _coerce_bound_level_index(value: Any, source: Optional[str]) -> int:
    if isinstance(value, bool):
        raise WorkloadParseError(
            _format_error(
                f"Boolean value is invalid for bound level '{value}'.", source
            )
        )
    if isinstance(value, int):
        level = value
    elif isinstance(value, str) and value.isdigit():
        level = int(value)
    else:
        raise WorkloadParseError(
            _format_error(
                f"Bound level keys must be non-negative integers, got {value!r}.",
                source,
            )
        )
    if level < 0:
        raise WorkloadParseError(
            _format_error(
                f"Bound level indices must be non-negative, got {level}.", source
            )
        )
    return level


def _validate_bound_levels(levels: Sequence[int], source: Optional[str]) -> None:
    if not levels:
        return
    min_level = min(levels)
    if min_level != 0:
        raise WorkloadParseError(
            _format_error("Bound levels must start from 0.", source)
        )
    max_level = max(levels)
    expected = set(range(0, max_level + 1))
    present = set(levels)
    missing = sorted(expected - present)
    if missing:
        raise WorkloadParseError(
            _format_error(
                f"Bound levels must be continuous from 0 to {max_level}; missing levels: {', '.join(map(str, missing)) or 'none'}.",
                source,
            )
        )


def _parse_tensor_attrs(
    payload: Any, source: Optional[str]
) -> Mapping[str, TensorAttrSpec]:
    if payload is None:
        return {}
    mapping = _require_mapping(payload, "TensorAttr", source)
    result: Dict[str, TensorAttrSpec] = {}
    for tensor, value in mapping.items():
        if isinstance(value, Mapping):
            bits = _require_int(value.get("bits"), f"TensorAttr.{tensor}.bits", source)
            role_value = value.get("role")
            role = (
                _require_str(role_value, f"TensorAttr.{tensor}.role", source)
                if role_value is not None
                else None
            )
        else:
            parsed = _BITS_RE.match(str(value))
            if not parsed:
                raise WorkloadParseError(
                    _format_error(
                        f"TensorAttr value '{value}' for tensor '{tensor}' is not of the form '16b' or mapping.",
                        source,
                    )
                )
            bits = int(parsed.group(1))
            role = parsed.group(2)
        result[tensor] = TensorAttrSpec(bits=bits, role=role)
    return result


def _parse_layout(payload: Any, source: Optional[str]) -> LayoutPragma:
    mapping = _require_mapping(payload, "Layout-Pragma", source)

    bit_raw = mapping.get("bit")
    bit_bool: Optional[bool] = None
    if bit_raw is not None:
        bit_token = _require_str(bit_raw, "Layout-Pragma.bit", source).strip().lower()
        if bit_token not in {"bit-parallel", "bit-serial"}:
            raise WorkloadParseError(
                _format_error(
                    "Layout-Pragma.bit must be either 'bit-parallel' or 'bit-serial'.",
                    source,
                )
            )
        bit_bool = bit_token == "bit-parallel"

    if "cache" not in mapping:
        raise WorkloadParseError(
            _format_error(
                "Layout-Pragma.cache must be specified as true/false.",
                source,
            )
        )
    cache = _require_bool(mapping["cache"], "Layout-Pragma.cache", source)
    cache_groups = _parse_cache_groups(
        mapping.get("cache-groups"),
        "Layout-Pragma.cache-groups",
        source,
    )

    tile_raw = mapping.get("tile")
    if tile_raw is not None or "tile" in mapping:
        raise WorkloadParseError(
            _format_error(
                "Layout-Pragma.tile is no longer supported; workloads are implicitly row-major.",
                source,
            )
        )

    streaming_raw = mapping.get("streaming", {})
    streaming_value: LayoutStreaming
    if streaming_raw is not None:
        streaming_map = _require_mapping(
            streaming_raw, "Layout-Pragma.streaming", source
        )
        expected_streams = ["In", "F", "Out"]
        if set(streaming_map.keys()) != set(expected_streams):
            raise WorkloadParseError(
                _format_error(
                    "Layout-Pragma.streaming must contain exactly the keys 'In', 'F', and 'Out'.",
                    source,
                )
            )
        streaming_value = LayoutStreaming(
            In=_require_bool(streaming_map["In"], "Layout-Pragma.streaming.In", source),
            F=_require_bool(streaming_map["F"], "Layout-Pragma.streaming.F", source),
            Out=_require_bool(
                streaming_map["Out"], "Layout-Pragma.streaming.Out", source
            ),
        )
        _validate_streaming_flags(streaming_value, source)
    else:
        streaming_value = LayoutStreaming()

    sharing_raw = mapping.get("sharing")
    if sharing_raw is None:
        sharing_raw = mapping.get("Sharing")
    sharing = None
    if sharing_raw is not None:
        sharing_map = _require_mapping(sharing_raw, "Layout-Pragma.sharing", source)
        system_raw = sharing_map.get("system", False)
        if isinstance(system_raw, bool):
            system_value = 2 if system_raw else 0
        elif system_raw is None:
            system_value = 0
        else:
            system_value = _require_int(
                system_raw, "Layout-Pragma.sharing.system", source
            )
        if system_value < 0:
            raise WorkloadParseError(
                _format_error(
                    "Layout-Pragma.sharing.system must be greater than or equal to 0.",
                    source,
                )
            )
        sharing = LayoutSharing(
            block=_require_bool(
                sharing_map.get("block", False), "Layout-Pragma.sharing.block", source
            ),
            pu=_require_bool(
                sharing_map.get("PU", False), "Layout-Pragma.sharing.PU", source
            ),
            system=system_value,
        )
        if sharing.system > 1 and not sharing.pu:
            raise WorkloadParseError(
                _format_error(
                    "Layout-Pragma.sharing.system (>1) requires PU sharing to be enabled.",
                    source,
                )
            )

    if "tiles-dim" in mapping:
        raise WorkloadParseError(
            _format_error(
                "Layout-Pragma.tiles-dim is no longer supported; row capacity is derived from the architecture and tensor tile size.",
                source,
            )
        )

    interleaving_raw = mapping.get("interleaving")
    interleaving = (
        _require_bool(interleaving_raw, "Layout-Pragma.interleaving", source)
        if interleaving_raw is not None
        else None
    )

    return LayoutPragma(
        bit_bool=bit_bool,
        cache=cache,
        cache_groups=cache_groups,
        streaming=streaming_value,
        sharing=sharing,
        interleaving=interleaving,
    )


def _parse_cache_groups(
    payload: Any,
    context: str,
    source: Optional[str],
) -> Tuple[LayoutCacheGroup, ...]:
    if payload is None:
        return tuple()

    mapping = _require_mapping(payload, context, source)
    groups: list[LayoutCacheGroup] = []
    for group_name, group_payload in mapping.items():
        name = _require_str(group_name, f"{context} group name", source)
        if isinstance(group_payload, Mapping):
            tensors_raw = group_payload.get("tensors")
            if tensors_raw is None:
                raise WorkloadParseError(
                    _format_error(
                        f"{context}.{name} must define a 'tensors' sequence.",
                        source,
                    )
                )
            tensors = _parse_cache_group_tensors(
                tensors_raw,
                f"{context}.{name}.tensors",
                source,
            )
            weight = group_payload.get("weight")
        else:
            tensors = _parse_cache_group_tensors(
                group_payload,
                f"{context}.{name}",
                source,
            )
            weight = None
        groups.append(LayoutCacheGroup(name=name, tensors=tensors, weight=weight))
    return tuple(groups)


def _parse_cache_group_tensors(
    payload: Any,
    context: str,
    source: Optional[str],
) -> Tuple[str, ...]:
    sequence = _require_sequence(payload, context, source)
    tensors: list[str] = []
    for index, tensor in enumerate(sequence):
        tensors.append(_require_str(tensor, f"{context}[{index}]", source))
    return tuple(tensors)


def _validate_streaming_flags(
    streaming: LayoutStreaming, source: Optional[str]
) -> None:
    if streaming.In and streaming.F and streaming.Out:
        raise WorkloadParseError(
            _format_error(
                "Layout-Pragma.streaming cannot enable In, F, and Out simultaneously.",
                source,
            )
        )


def _require_mapping(
    value: Any, context: str, source: Optional[str]
) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise WorkloadParseError(
        _format_error(
            f"Expected mapping for '{context}', got {type(value).__name__}.", source
        )
    )


def _require_sequence(value: Any, context: str, source: Optional[str]) -> Sequence[Any]:
    if isinstance(value, AbcSequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return value
    raise WorkloadParseError(
        _format_error(
            f"Expected sequence for '{context}', got {type(value).__name__}.", source
        )
    )


def _require_str(value: Any, context: str, source: Optional[str]) -> str:
    if isinstance(value, str):
        return value
    raise WorkloadParseError(
        _format_error(
            f"Expected string for '{context}', got {type(value).__name__}.", source
        )
    )


def _optional_str(value: Any, context: str, source: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return _require_str(value, context, source)


def _require_int(value: Any, context: str, source: Optional[str]) -> int:
    if isinstance(value, bool):
        raise WorkloadParseError(
            _format_error(
                f"Boolean value is not valid for integer field '{context}'.", source
            )
        )
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise WorkloadParseError(
        _format_error(f"Expected integer for '{context}', got {value!r}.", source)
    )


def _require_bool(value: Any, context: str, source: Optional[str]) -> bool:
    if isinstance(value, bool):
        return value
    raise WorkloadParseError(
        _format_error(
            f"Expected boolean for '{context}', got {type(value).__name__}.", source
        )
    )


def _parse_int_tuple(
    value: Any, context: str, source: Optional[str]
) -> Tuple[int, ...]:
    seq = _require_sequence(value, context, source)
    ints = tuple(_require_int(item, f"{context}[]", source) for item in seq)
    return ints


def _format_error(message: str, source: Optional[str]) -> str:
    if source:
        return f"{message} (in {source})"
    return message


_DIMENSION_BASES: Tuple[str, ...] = ("N", "K", "P", "Q", "C", "R", "S")
_BOUND_LOOP_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _collect_dimension_values(
    problem: ProblemSpec, bounds: LoopBounds
) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for base in _DIMENSION_BASES:
        if base in problem.loop_vars:
            values[base] = problem.loop_vars[base]
    for dimension, level, extent in bounds.extent_items():
        values[f"{dimension}{level}"] = extent
    return values


def _validate_dimension_products(workload_name: str, values: Mapping[str, int]) -> None:
    grouped = _group_loop_extents(values)
    for base in _DIMENSION_BASES:
        top = values.get(base)
        if top is None:
            continue
        levels = grouped.get(base, {})
        if not levels:
            continue
        max_level = max(levels)
        missing_levels = [
            level for level in range(0, max_level + 1) if level not in levels
        ]
        if missing_levels:
            missing_str = ", ".join(f"{base}{level}" for level in missing_levels)
            raise WorkloadParseError(
                f"Missing loop levels for {base} in workload '{workload_name}': {missing_str}."
            )
        product = math.prod(levels[level] for level in range(0, max_level + 1))
        if product != top:
            raise WorkloadParseError(
                f"Dimension mismatch for {base} in workload '{workload_name}': "
                f"{base} equals {top}, but the product of loop levels equals {product}."
            )


def _serialize_workload(workload: Workload) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    data["Name"] = workload.name
    data["Problem"] = _serialize_problem_spec(workload.problem)
    data["Bound"] = _serialize_bounds_sequence(workload.bounds)
    data["TensorAttr"] = _serialize_tensor_attrs_map(workload.tensor_attrs)
    data["Compute"] = workload.compute
    if workload.layout is not None:
        layout_mapping = _serialize_layout(workload.layout)
        if layout_mapping:
            data["Layout-Pragma"] = layout_mapping
    return data


def _serialize_problem_spec(problem: ProblemSpec) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    seen: set[str] = set()
    for base in _DIMENSION_BASES:
        if base in problem.loop_vars:
            data[base] = problem.loop_vars[base]
            seen.add(base)
    for key, value in problem.loop_vars.items():
        if key not in seen:
            data[key] = value
            seen.add(key)
    if problem.dilation is not None:
        data["Dilation"] = list(problem.dilation)
    if problem.stride is not None:
        data["Stride"] = list(problem.stride)
    return data


def _serialize_bounds_sequence(bounds: LoopBounds) -> Any:
    levels = bounds.levels()
    if not levels:
        return []
    serialized: Dict[int, List[Dict[str, int]]] = {}
    for level in levels:
        serialized[int(level)] = [
            {bound.loop_var: bound.extent} for bound in bounds.get_level(level)
        ]
    return serialized


def _serialize_tensor_attrs_map(
    tensor_attrs: Mapping[str, TensorAttrSpec],
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for name in sorted(tensor_attrs):
        spec = tensor_attrs[name]
        value = f"{spec.bits}b"
        if spec.role:
            value = f"{value}: {spec.role}"
        result[name] = value
    return result


def _split_loop_name(loop_name: str) -> Tuple[str, int]:
    match = _BOUND_LOOP_RE.match(loop_name)
    if not match:
        raise ValueError(loop_name)
    base, level_str = match.groups()
    return base, int(level_str)


def _group_loop_extents(values: Mapping[str, int]) -> Dict[str, Dict[int, int]]:
    grouped: Dict[str, Dict[int, int]] = {base: {} for base in _DIMENSION_BASES}
    for name, extent in values.items():
        try:
            base, level = _split_loop_name(name)
        except ValueError:
            continue
        canonical_base = base.upper()
        if canonical_base in grouped:
            grouped[canonical_base][level] = extent
    return grouped


def _dimension_rank(base: str) -> int:
    try:
        return _DIMENSION_BASES.index(base)
    except ValueError:
        return len(_DIMENSION_BASES)


def _canonical_dimension_key(key: Any) -> str:
    if not isinstance(key, str):
        raise KeyError(key)
    canonical = key.strip()
    if not canonical:
        raise KeyError(key)
    return canonical.upper()


def _normalize_bound_level(
    entries: Sequence[Any], *, context: str
) -> List[Dict[str, int]]:
    if not isinstance(entries, AbcSequence) or isinstance(
        entries, (str, bytes, bytearray)
    ):
        raise WorkloadParseError(f"{context} must be a sequence of dimension mappings.")

    normalized: Dict[str, int] = {}
    for entry in entries:
        if isinstance(entry, LoopBound):
            dimension = entry.loop_var
            extent_value = entry.extent
        elif isinstance(entry, Mapping):
            if len(entry) != 1:
                raise WorkloadParseError(
                    f"{context} entries must contain a single dimension mapping, got {entry}."
                )
            ((dimension, extent_value),) = entry.items()
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            dimension, extent_value = entry
        else:
            raise WorkloadParseError(
                f"{context} entries must be single-item mappings, (dimension, extent) pairs, or LoopBound objects."
            )
        try:
            canonical_dim = _canonical_dimension_key(dimension)
        except KeyError as exc:
            raise WorkloadParseError(
                f"{context} dimension names must be non-empty strings, got {dimension!r}."
            ) from exc
        if canonical_dim not in _DIMENSION_BASES:
            raise WorkloadParseError(
                f"{context} dimension '{canonical_dim}' is not a recognised canonical dimension ({_DIMENSION_BASES})."
            )
        if canonical_dim in normalized:
            raise WorkloadParseError(
                f"Duplicate dimension '{canonical_dim}' found in {context}."
            )
        normalized[canonical_dim] = _require_int(
            extent_value, f"{context}.{canonical_dim}", None
        )

    ordered_dims = sorted(normalized.keys(), key=_dimension_rank)
    return [{dim: normalized[dim]} for dim in ordered_dims]


def _parse_loop_reference(token: str) -> Tuple[str, Optional[int]]:
    if not isinstance(token, str):
        raise TypeError(f"Loop reference must be a string, got {type(token).__name__}.")
    match = _BOUND_LOOP_RE.match(token)
    if match:
        base, level_str = match.groups()
        return _canonical_dimension_key(base), int(level_str)
    return _canonical_dimension_key(token), None


def _resolve_dimension_level(name: str, level: Optional[int]) -> Tuple[str, int]:
    if level is not None:
        return _canonical_dimension_key(name), _coerce_bound_level_index(level, None)
    dimension, inferred_level = _parse_loop_reference(name)
    if inferred_level is None:
        raise ValueError(
            f"Loop reference '{name}' must include a level suffix (e.g. 'N0') or provide level=... explicitly."
        )
    return dimension, inferred_level


def _serialize_layout(layout: LayoutPragma) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    bit_token = layout.bit
    if bit_token is not None:
        data["bit"] = bit_token
    if layout.cache is None:
        raise WorkloadParseError(
            "Layout-Pragma.cache must be specified before serialization."
        )
    data["cache"] = layout.cache
    if layout.cache_groups:
        data["cache-groups"] = _serialize_cache_groups(layout.cache_groups)
    if layout.streaming:
        data["streaming"] = layout.streaming.as_dict()
    if layout.sharing is not None:
        data["sharing"] = {
            "block": layout.sharing.block,
            "PU": layout.sharing.pu,
            "system": layout.sharing.system,
        }
    if layout.interleaving is not None:
        data["interleaving"] = layout.interleaving
    return data


def _serialize_cache_groups(
    groups: Sequence[LayoutCacheGroup],
) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for group in groups:
        if group.weight is None:
            serialized[group.name] = list(group.tensors)
            continue
        weight: int | float = group.weight
        if isinstance(weight, float) and weight.is_integer():
            weight = int(weight)
        serialized[group.name] = {
            "tensors": list(group.tensors),
            "weight": weight,
        }
    return serialized


def _coerce_bit_value(value: Union[bool, str, None]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "bit-parallel":
            return True
        if token == "bit-serial":
            return False
    raise WorkloadParseError(
        "Layout-Pragma.bit accepts booleans or the tokens 'bit-parallel'/'bit-serial'."
    )


def _coerce_streaming_value(
    value: Union[LayoutStreaming, Mapping[str, Any], None],
) -> LayoutStreaming:
    if value is None:
        return LayoutStreaming()
    if isinstance(value, LayoutStreaming):
        streaming = value
        _validate_streaming_flags(streaming, None)
        return streaming
    if isinstance(value, Mapping):
        keys = set(value.keys())
        if keys != {"In", "F", "Out"}:
            raise WorkloadParseError(
                "Layout-Pragma.streaming must contain exactly the keys 'In', 'F', and 'Out'."
            )
        streaming = LayoutStreaming(
            In=bool(value["In"]),
            F=bool(value["F"]),
            Out=bool(value["Out"]),
        )
        _validate_streaming_flags(streaming, None)
        return streaming
    raise TypeError(
        "Layout-Pragma.streaming accepts a LayoutStreaming instance, mapping, or None."
    )


def _coerce_cache_groups_value(
    value: Union[
        None,
        Mapping[str, Any],
        Sequence[LayoutCacheGroup | Mapping[str, Any]],
    ],
) -> Tuple[LayoutCacheGroup, ...]:
    if value is None:
        return tuple()
    if isinstance(value, Mapping):
        return _parse_cache_groups(value, "Layout-Pragma.cache-groups", None)
    if isinstance(value, AbcSequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        groups: list[LayoutCacheGroup] = []
        for index, item in enumerate(value):
            if isinstance(item, LayoutCacheGroup):
                groups.append(item)
                continue
            if not isinstance(item, Mapping):
                raise WorkloadParseError(
                    "Layout-Pragma.cache-groups sequence entries must be cache-group mappings."
                )
            name = _require_str(
                item.get("name"),
                f"Layout-Pragma.cache-groups[{index}].name",
                None,
            )
            tensors = _parse_cache_group_tensors(
                item.get("tensors"),
                f"Layout-Pragma.cache-groups[{index}].tensors",
                None,
            )
            groups.append(
                LayoutCacheGroup(
                    name=name,
                    tensors=tensors,
                    weight=item.get("weight"),
                )
            )
        return tuple(groups)
    raise WorkloadParseError(
        "Layout-Pragma.cache-groups accepts a mapping or a sequence of cache-group objects."
    )


def _coerce_cache_config_override(
    override: LayoutCacheConfig | Mapping[str, Any] | bool,
) -> LayoutCacheConfig:
    if isinstance(override, bool):
        return LayoutCacheConfig(enabled=override, groups=())
    if isinstance(override, LayoutCacheConfig):
        return override
    if not isinstance(override, Mapping):
        raise TypeError(
            "cache_config_override must be a bool, LayoutCacheConfig, or mapping."
        )

    enabled_raw: Any = True
    groups_raw: Any = override
    if any(
        key in override for key in ("enabled", "cache", "groups", "cache-groups")
    ):
        enabled_raw = override.get("enabled", override.get("cache", True))
        groups_raw = override.get("groups", override.get("cache-groups"))
    enabled = _require_bool(enabled_raw, "cache_config_override.enabled", None)
    groups = _parse_cache_groups(groups_raw, "cache_config_override.groups", None)
    return LayoutCacheConfig(enabled=enabled, groups=groups)


def _resolve_layout_cache_config_for_workload(
    *,
    workload: Workload,
    config: LayoutCacheConfig,
) -> LayoutCacheConfig:
    if not config.enabled:
        if config.groups:
            raise WorkloadParseError(
                f"Workload '{workload.name}' cannot define cache groups when cache is disabled."
            )
        return LayoutCacheConfig(enabled=False, groups=())

    groups = config.groups
    if not groups:
        groups = _default_cache_groups_for_workload(workload)

    known_tensors = set(workload.compute_spec.tensors.keys())
    output_tensor = workload.compute_spec.target.tensor
    seen_tensors: dict[str, str] = {}
    for group in groups:
        for tensor in group.tensors:
            if tensor not in known_tensors:
                raise WorkloadParseError(
                    f"Workload '{workload.name}' cannot cache unknown tensor '{tensor}'."
                )
            if tensor == output_tensor:
                raise WorkloadParseError(
                    f"Workload '{workload.name}' cannot cache output tensor '{tensor}'."
                )
            previous_group = seen_tensors.get(tensor)
            if previous_group is not None:
                raise WorkloadParseError(
                    f"Workload '{workload.name}' assigns tensor '{tensor}' to cache groups '{previous_group}' and '{group.name}'."
                )
            seen_tensors[tensor] = group.name

    return LayoutCacheConfig(enabled=True, groups=tuple(groups))


def _default_cache_groups_for_workload(
    workload: Workload,
) -> Tuple[LayoutCacheGroup, ...]:
    tensors = tuple(
        tensor
        for tensor in _DEFAULT_CACHE_GROUP_TENSORS
        if tensor in workload.compute_spec.tensors
    )
    if not tensors:
        raise WorkloadParseError(
            f"Workload '{workload.name}' cannot derive default cache groups because neither F nor In appears in Compute."
        )
    return (LayoutCacheGroup(name=_DEFAULT_CACHE_GROUP_NAME, tensors=tensors),)
