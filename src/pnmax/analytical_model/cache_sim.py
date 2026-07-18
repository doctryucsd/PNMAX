"""
Fully-associative LRU cache simulator for PNMAX workloads.

This module simulates cache behavior for two specific PIM hierarchies:
1. 'regfile': A vector register file where cache lines correspond to Tiles (Level 1).
2. 'wram':   A scratchpad memory where cache lines correspond to individual Elements.

It generates a deterministic execution schedule based on the loop order defined
in the workload YAML (Levels 1 and 0 only).

For now we only support fully-associative caches with LRU replacement policy.
* Replacement Policy : LRU, [Other Methods Pending]
* Last Revision Date: 2026-01-08

Problems:
    - Cache tensor size allocation strategies
    - Tensor access order parsing and generation from workload spec not standardized.
    - Data access flag not systematically parsed and utilized.
    - Streaming access definition not standardized across workloads.
"""

from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Hashable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from pnmax.database import Workload
from pnmax.database.workload import _evaluate_ast, _parse_index_expression_ast

# For now we only support regfile and wram cache types
CacheMode = Literal["regfile", "wram"]
CacheAccessKind = Literal["compute", "streaming"]
logger = logging.getLogger(__name__)
_CACHEABLE_TENSORS: frozenset[str] = frozenset({"F", "In"})

# =============================================================================
# Replacement Policies
# =============================================================================


class CacheReplacementPolicy(ABC):
    """
    Interface for cache replacement algorithms (e.g., LRU, FIFO, Random).
    Responsible solely for tracking usage order and selecting eviction victims.
    """

    @abstractmethod
    def on_access(self, key: Hashable) -> None:
        """Called when an existing item is accessed (Hit)."""
        pass

    @abstractmethod
    def on_insert(self, key: Hashable) -> None:
        """Called when a new item is added (Miss -> Insert)."""
        pass

    @abstractmethod
    def pop_victim(self) -> Optional[Hashable]:
        """Selects, removes, and returns the key to be evicted."""
        pass


class LRUPolicy(CacheReplacementPolicy):
    """
    Least Recently Used (LRU) policy.
    Maintains a separate ordering of keys.
    """

    def __init__(self):
        # We use an OrderedDict as a queue:
        # Left (start) = LRU (Oldest), Right (end) = MRU (Newest)
        self._order: OrderedDict[Hashable, None] = OrderedDict()

    def on_access(self, key: Hashable) -> None:
        # Move to MRU position (end)
        self._order.move_to_end(key)

    def on_insert(self, key: Hashable) -> None:
        # Insert at MRU position (end)
        self._order[key] = None

    def pop_victim(self) -> Optional[Hashable]:
        # Pop the LRU item (start)
        if not self._order:
            return None
        # popitem(last=False) removes from the start
        key, _ = self._order.popitem(last=False)
        return key


# =============================================================================
# Cache Infrastructure (Storage)
# =============================================================================


@dataclass
class CacheStats:
    """Global cache statistics. Tracking hits, misses and data movement for the cache."""

    accesses: int = 0
    hits: int = 0
    misses: int = 0
    bytes_loaded: int = 0
    streaming_accesses: int = 0

    @property
    def miss_rate(self) -> float:
        # We generally exclude streaming bypasses from the "cache miss rate" metric
        # or include them depending on definition. Here we include only cacheable accesses.
        total_cacheable = self.hits + self.misses
        return self.misses / total_cacheable if total_cacheable > 0 else 0.0

    def record(self, hit: bool, cost: int) -> None:
        self.accesses += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            self.bytes_loaded += cost

    def record_streaming(self, cost: int):
        """Records a streaming bypass (counts as data loaded, but not a cache op)."""
        # TODO: Check the validity and usefulness of this metric
        self.accesses += 1
        self.streaming_accesses += 1
        self.bytes_loaded += cost


@dataclass(frozen=True)
class AccessInfo:
    """Represents a single tensor access in the schedule."""

    tensor: str
    # Global indices identify the specific element (e.g., In[1024])
    global_indices: Tuple[int, ...]  # For Element addressing
    # Tile indices identify the tile (ignoring level 0 loops)
    tile_indices: Tuple[int, ...]  # For Tile/Vector addressing
    # Flag indicating if this tensor is marked as streaming in the workload
    is_streaming: bool = False


class AbstractCache(ABC):
    """
    Base Cache Class.
    Manages:
    1. Storage (data_store)
    2. Capacity checks (current_bytes vs capacity_bytes)
    3. Statistics
    4. Delegation to ReplacementPolicy
    """

    def __init__(self, capacity_bytes: int, policy: CacheReplacementPolicy) -> None:
        self.capacity_bytes = int(capacity_bytes)
        self.current_bytes = 0
        self.policy = policy
        self.stats = CacheStats()

        # Maps Key -> Size (Policy tracks the 'Key', cache itself tracks the 'Size' & existence)
        self.data_store: Dict[Hashable, int] = {}

    @abstractmethod
    def process_timestep(
        self,
        batch: List[AccessInfo],
        tensor_stats: Dict[str, CacheStats],
        tensor_role_stats: Dict[str, Dict[str, CacheStats]],
    ) -> None:
        """
        Process a full timestep of accesses.
        Subclasses define how to handle granularity (Scalar vs Vector).
        """
        pass

    @abstractmethod
    def _map_to_key_and_size(self, access: AccessInfo) -> Tuple[Hashable, int]:
        """Translates a workload access packet to a generic cache key/size."""
        pass

    def _perform_single_access(
        self,
        access: AccessInfo,
        tensor_stat: CacheStats,
        tensor_role_stat: CacheStats,
    ) -> bool:
        """
        Internal Core Cache Logic. Returns True on Hit, False on Miss.
        Handles Lookup -> Hit/Miss -> Evict -> Insert logic.
        """
        key, size = self._map_to_key_and_size(access)

        # --- HIT ---
        if key in self.data_store:
            self.policy.on_access(key)
            self.stats.record(hit=True, cost=0)
            tensor_stat.record(hit=True, cost=0)
            tensor_role_stat.record(hit=True, cost=0)
            return True

        # --- MISS ---
        self.stats.record(hit=False, cost=size)
        tensor_stat.record(hit=False, cost=size)
        tensor_role_stat.record(hit=False, cost=size)

        # Check the loaded size
        if size > self.capacity_bytes:
            # Cannot fit into cache; throw an error fro now
            raise ValueError(
                f"Access size {size} exceeds cache capacity {self.capacity_bytes}."
            )

        # Eviction
        while self.current_bytes + size > self.capacity_bytes:
            victim_key = self.policy.pop_victim()
            if victim_key is None:
                # Cache is empty but item is too big? Should have been caught by previous check.
                # Or external error. Break to avoid infinite loop.
                break

            # Remove from storage
            victim_size = self.data_store.pop(victim_key)
            self.current_bytes -= victim_size

        # Insertion
        self.data_store[key] = size
        self.current_bytes += size
        self.policy.on_insert(key)

        return False


# =============================================================================
# Concrete Cache Implementations
# =============================================================================


class RegFileCache(AbstractCache):
    """
    Vector Register File Simulator.

    Granularity: Vector Register (Tile).
    Behavior:
    - Receives a batch of accesses for one timestep (Level 0 loop nest).
    - Deduplicates accesses that target the same Tile Key (ignoring scalar offsets).
    - This simulates loading a vector register once for a parallel instruction.
    """

    def __init__(self, num_regs: int, vector_width_bits: int):
        self.vector_bytes = vector_width_bits // 8
        capacity = num_regs * self.vector_bytes
        # We explicitly inject LRUPolicy here.
        # Ideally, this could also be passed in if we wanted FIFO RegFiles.
        super().__init__(capacity, LRUPolicy())

    def _map_to_key_and_size(self, access: AccessInfo) -> Tuple[Hashable, int]:
        # Tag: Tile Index
        return ("regfile", access.tensor, access.tile_indices), self.vector_bytes

    def process_timestep(
        self,
        batch: List[AccessInfo],
        tensor_stats: Dict[str, CacheStats],
        tensor_role_stats: Dict[str, Dict[str, CacheStats]],
    ) -> None:
        """
        Optimized processing for Vector operations.
        We deduce that all scalar ops in this batch targeting the same tile
        represent a SINGLE vector load/store.
        """
        # Track unique registers touched in this single parallel cycle
        seen_registers_this_step: Set[Hashable] = set()

        for acc in batch:
            t_stats = tensor_stats[acc.tensor]
            role_stats = _get_role_stats(tensor_role_stats, acc.tensor)
            role_stat = role_stats[_access_kind(acc)]

            # Get the Key to check for duplicates
            key, _ = self._map_to_key_and_size(acc)

            if key in seen_registers_this_step:
                continue  # Already loaded this vector in this cycle

            seen_registers_this_step.add(key)

            # Perform actual cache logic
            self._perform_single_access(acc, t_stats, role_stat)


class WRAMCache(AbstractCache):
    """
    WRAM Scratchpad Simulator.

    Granularity: Scalar Element.
    Behavior: Processes every access individually (Byte-addressable).
    """

    def __init__(
        self, capacity_bytes: int, tensor_sizes: Dict[str, int], default_size: int
    ):
        super().__init__(capacity_bytes, LRUPolicy())
        self.tensor_sizes = tensor_sizes
        self.default_size = default_size

    def _map_to_key_and_size(self, access: AccessInfo) -> Tuple[Hashable, int]:
        # Tag: Global Element Index
        size = self.tensor_sizes.get(access.tensor, self.default_size)
        return ("wram", access.tensor, access.global_indices), size

    def process_timestep(
        self,
        batch: List[AccessInfo],
        tensor_stats: Dict[str, CacheStats],
        tensor_role_stats: Dict[str, Dict[str, CacheStats]],
    ) -> None:
        """
        Standard processing for Scalar/Byte-addressable memory.
        Iterate through every single access in the batch.
        """
        for acc in batch:
            t_stats = tensor_stats[acc.tensor]
            role_stats = _get_role_stats(tensor_role_stats, acc.tensor)
            role_stat = role_stats[_access_kind(acc)]
            self._perform_single_access(acc, t_stats, role_stat)


# =============================================================================
# Cache Simulator
# =============================================================================


@dataclass
class CacheSimulationResult:
    """Final simulation report."""

    mode: CacheMode
    global_stats: CacheStats
    tensor_stats: Dict[str, CacheStats]


@dataclass(frozen=True)
class CacheGroupAllocation:
    """Resolved cache capacity for one logical cache group."""

    name: str
    tensors: Tuple[str, ...]
    weight: float
    capacity_bytes: int
    num_vector_regs: int | None = None


@dataclass
class GroupedCacheSimulationResult:
    """Final grouped-cache simulation report."""

    mode: CacheMode
    global_stats: CacheStats
    tensor_stats: Dict[str, CacheStats]
    group_stats: Dict[str, CacheStats]
    group_allocations: Dict[str, CacheGroupAllocation]
    global_role_stats: Dict[str, CacheStats] = field(default_factory=dict)
    tensor_role_stats: Dict[str, Dict[str, CacheStats]] = field(default_factory=dict)
    group_role_stats: Dict[str, Dict[str, CacheStats]] = field(default_factory=dict)
    tensor_allocations: Dict[str, CacheGroupAllocation] = field(default_factory=dict)


@dataclass(frozen=True)
class _AccessPlan:
    """Precompiled per-pattern cache access plan."""

    tensor: str
    compiled_indices: Tuple[ast.AST, ...]
    is_streaming: bool
    group_name: str


@dataclass(frozen=True)
class _PreparedSimulation:
    """Shared workload-derived state for fast cache simulation."""

    l1_keys: Tuple[str, ...]
    l0_keys: Tuple[str, ...]
    level1_bounds: OrderedDict[str, int]
    level0_bounds: OrderedDict[str, int]
    l1_strides: Dict[str, int]
    plans_by_group: Dict[str, Tuple[_AccessPlan, ...]]


def simulate_cache_for_workload(
    workload: Union[Workload, str, Path],
    mode: CacheMode,
    *,
    num_vector_regs: int = 16,
    vector_width_bits: int = 256,
    wram_capacity_bytes: int = 64 * 1024,
    element_bytes: int = 2,
) -> CacheSimulationResult:
    """
    Simulates cache behavior for the given workload and mode.

    Args:
        workload: Workload object or path to YAML.
        mode: "regfile" (tile-based) or "wram" (element-based).
        num_vector_regs: Capacity for regfile mode (in number of registers).
        vector_width_bits: Size of one register in bits (regfile mode).
        wram_capacity_bytes: Total size of WRAM in bytes.
        element_bytes: Default size of a single element if not specified in TensorAttr.
    """
    if mode == "regfile":
        return _simulate_cache_for_workload_fast_regfile(
            workload,
            num_vector_regs=num_vector_regs,
            vector_width_bits=vector_width_bits,
        )
    if mode == "wram":
        return _simulate_cache_for_workload_fast_wram(
            workload,
            wram_capacity_bytes=wram_capacity_bytes,
            element_bytes=element_bytes,
        )
    raise ValueError(f"Unsupported cache mode: {mode}")


def simulate_grouped_cache_for_workload(
    workload: Union[Workload, str, Path],
    mode: CacheMode,
    *,
    allocations: Mapping[str, CacheGroupAllocation],
    vector_width_bits: int = 256,
    element_bytes: int = 2,
) -> GroupedCacheSimulationResult:
    """Simulate one logical cache instance per named cache group."""
    if mode == "regfile":
        return _simulate_grouped_cache_for_workload_fast_regfile(
            workload,
            allocations=allocations,
            vector_width_bits=vector_width_bits,
        )
    if mode == "wram":
        return _simulate_grouped_cache_for_workload_fast_wram(
            workload,
            allocations=allocations,
            element_bytes=element_bytes,
        )
    raise ValueError(f"Unsupported cache mode: {mode}")


def _simulate_cache_for_workload_oracle(
    workload: Union[Workload, str, Path],
    mode: CacheMode,
    *,
    num_vector_regs: int = 16,
    vector_width_bits: int = 256,
    wram_capacity_bytes: int = 64 * 1024,
    element_bytes: int = 2,
) -> CacheSimulationResult:
    """Reference implementation kept as a correctness oracle during rollout."""

    workload_obj = _ensure_workload(workload)
    cacheable_tensors = _cacheable_tensors_in_workload(workload_obj)
    if not cacheable_tensors:
        return CacheSimulationResult(
            mode=mode,
            global_stats=CacheStats(),
            tensor_stats={},
        )

    capacity_bytes = (
        (num_vector_regs * (vector_width_bits // 8))
        if mode == "regfile"
        else wram_capacity_bytes
    )
    allocation = CacheGroupAllocation(
        name="__single__",
        tensors=cacheable_tensors,
        weight=1.0,
        capacity_bytes=capacity_bytes,
        num_vector_regs=num_vector_regs if mode == "regfile" else None,
    )
    grouped = _simulate_grouped_cache_for_workload_oracle(
        workload_obj,
        mode,
        allocations={"__single__": allocation},
        vector_width_bits=vector_width_bits,
        element_bytes=element_bytes,
    )
    return CacheSimulationResult(
        mode=mode,
        global_stats=grouped.global_stats,
        tensor_stats=grouped.tensor_stats,
    )


def _simulate_grouped_cache_for_workload_oracle(
    workload: Union[Workload, str, Path],
    mode: CacheMode,
    *,
    allocations: Mapping[str, CacheGroupAllocation],
    vector_width_bits: int = 256,
    element_bytes: int = 2,
) -> GroupedCacheSimulationResult:
    """Reference grouped implementation kept as a correctness oracle."""

    workload_obj = _ensure_workload(workload)
    tensor_sizes = _get_tensor_sizes(workload_obj)

    tensor_allocations = _expand_group_allocations_to_tensors(
        mode=mode,
        allocations=allocations,
        vector_width_bits=vector_width_bits,
    )
    caches: Dict[str, AbstractCache] = {}
    tensor_to_cache: Dict[str, str] = {}
    for tensor, allocation in tensor_allocations.items():
        tensor_to_cache[tensor] = tensor
        caches[tensor] = _make_cache(
            mode=mode,
            allocation=allocation,
            tensor_sizes=tensor_sizes,
            vector_width_bits=vector_width_bits,
            element_bytes=element_bytes,
        )

    per_tensor_stats: Dict[str, CacheStats] = defaultdict(CacheStats)
    per_tensor_role_stats: Dict[str, Dict[str, CacheStats]] = {}
    for timestep_batch in _iter_schedule(workload_obj):
        grouped_batch: Dict[str, List[AccessInfo]] = defaultdict(list)
        for access in timestep_batch:
            cache_name = tensor_to_cache.get(access.tensor)
            if cache_name is None:
                continue
            grouped_batch[cache_name].append(access)
        for cache_name, batch in grouped_batch.items():
            caches[cache_name].process_timestep(
                batch,
                per_tensor_stats,
                per_tensor_role_stats,
            )

    group_stats = {
        group_name: _merge_cache_stats(
            caches[tensor].stats
            for tensor in allocation.tensors
            if tensor in caches
        )
        for group_name, allocation in allocations.items()
    }
    group_role_stats = {
        group_name: _merge_role_stats(
            per_tensor_role_stats[tensor]
            for tensor in allocation.tensors
            if tensor in per_tensor_role_stats
        )
        for group_name, allocation in allocations.items()
    }
    return GroupedCacheSimulationResult(
        mode=mode,
        global_stats=_merge_cache_stats(group_stats.values()),
        tensor_stats=dict(per_tensor_stats),
        group_stats=group_stats,
        group_allocations=dict(allocations),
        global_role_stats=_merge_role_stats(per_tensor_role_stats.values()),
        tensor_role_stats={
            tensor: dict(role_stats)
            for tensor, role_stats in per_tensor_role_stats.items()
        },
        group_role_stats=group_role_stats,
        tensor_allocations=tensor_allocations,
    )


def _simulate_cache_for_workload_fast_regfile(
    workload: Union[Workload, str, Path],
    *,
    num_vector_regs: int,
    vector_width_bits: int,
) -> CacheSimulationResult:
    workload_obj = _ensure_workload(workload)
    cacheable_tensors = _cacheable_tensors_in_workload(workload_obj)
    if not cacheable_tensors:
        return CacheSimulationResult(mode="regfile", global_stats=CacheStats(), tensor_stats={})
    default_group = "__single__"
    grouped = _simulate_grouped_cache_for_workload_fast_regfile(
        workload_obj,
        allocations={
            default_group: CacheGroupAllocation(
                name=default_group,
                tensors=cacheable_tensors,
                weight=1.0,
                capacity_bytes=num_vector_regs * (vector_width_bits // 8),
                num_vector_regs=num_vector_regs,
            )
        },
        vector_width_bits=vector_width_bits,
    )
    return CacheSimulationResult(
        mode="regfile",
        global_stats=grouped.global_stats,
        tensor_stats=grouped.tensor_stats,
    )


def _simulate_grouped_cache_for_workload_fast_regfile(
    workload: Union[Workload, str, Path],
    *,
    allocations: Mapping[str, CacheGroupAllocation],
    vector_width_bits: int,
) -> GroupedCacheSimulationResult:
    workload_obj = _ensure_workload(workload)
    tensor_allocations = _expand_group_allocations_to_tensors(
        mode="regfile",
        allocations=allocations,
        vector_width_bits=vector_width_bits,
    )
    caches: Dict[str, AbstractCache] = {}
    tensor_to_cache: Dict[str, str] = {}
    for tensor, allocation in tensor_allocations.items():
        tensor_to_cache[tensor] = tensor
        caches[tensor] = _make_cache(
            mode="regfile",
            allocation=allocation,
            tensor_sizes={},
            vector_width_bits=vector_width_bits,
            element_bytes=0,
        )

    prepared = _prepare_simulation(
        workload_obj,
        tensor_to_group=tensor_to_cache,
        group_names=tuple(caches.keys()),
    )
    per_tensor_stats: Dict[str, CacheStats] = defaultdict(CacheStats)
    per_tensor_role_stats: Dict[str, Dict[str, CacheStats]] = {}
    _run_regfile_fast_path(
        prepared,
        caches=caches,
        per_tensor_stats=per_tensor_stats,
        per_tensor_role_stats=per_tensor_role_stats,
    )
    group_stats = {
        group_name: _merge_cache_stats(
            caches[tensor].stats
            for tensor in allocation.tensors
            if tensor in caches
        )
        for group_name, allocation in allocations.items()
    }
    group_role_stats = {
        group_name: _merge_role_stats(
            per_tensor_role_stats[tensor]
            for tensor in allocation.tensors
            if tensor in per_tensor_role_stats
        )
        for group_name, allocation in allocations.items()
    }
    return GroupedCacheSimulationResult(
        mode="regfile",
        global_stats=_merge_cache_stats(group_stats.values()),
        tensor_stats=dict(per_tensor_stats),
        group_stats=group_stats,
        group_allocations=dict(allocations),
        global_role_stats=_merge_role_stats(per_tensor_role_stats.values()),
        tensor_role_stats={
            tensor: dict(role_stats)
            for tensor, role_stats in per_tensor_role_stats.items()
        },
        group_role_stats=group_role_stats,
        tensor_allocations=tensor_allocations,
    )


def _simulate_cache_for_workload_fast_wram(
    workload: Union[Workload, str, Path],
    *,
    wram_capacity_bytes: int,
    element_bytes: int,
) -> CacheSimulationResult:
    workload_obj = _ensure_workload(workload)
    cacheable_tensors = _cacheable_tensors_in_workload(workload_obj)
    if not cacheable_tensors:
        return CacheSimulationResult(mode="wram", global_stats=CacheStats(), tensor_stats={})
    default_group = "__single__"
    grouped = _simulate_grouped_cache_for_workload_fast_wram(
        workload_obj,
        allocations={
            default_group: CacheGroupAllocation(
                name=default_group,
                tensors=cacheable_tensors,
                weight=1.0,
                capacity_bytes=wram_capacity_bytes,
            )
        },
        element_bytes=element_bytes,
    )
    return CacheSimulationResult(
        mode="wram",
        global_stats=grouped.global_stats,
        tensor_stats=grouped.tensor_stats,
    )


def _simulate_grouped_cache_for_workload_fast_wram(
    workload: Union[Workload, str, Path],
    *,
    allocations: Mapping[str, CacheGroupAllocation],
    element_bytes: int,
) -> GroupedCacheSimulationResult:
    workload_obj = _ensure_workload(workload)
    tensor_sizes = _get_tensor_sizes(workload_obj)
    tensor_allocations = _expand_group_allocations_to_tensors(
        mode="wram",
        allocations=allocations,
        vector_width_bits=0,
    )
    caches: Dict[str, WRAMCache] = {}
    tensor_to_cache: Dict[str, str] = {}
    for tensor, allocation in tensor_allocations.items():
        tensor_to_cache[tensor] = tensor
        caches[tensor] = WRAMCache(
            allocation.capacity_bytes,
            tensor_sizes,
            element_bytes,
        )

    prepared = _prepare_simulation(
        workload_obj,
        tensor_to_group=tensor_to_cache,
        group_names=tuple(caches.keys()),
    )
    per_tensor_stats: Dict[str, CacheStats] = defaultdict(CacheStats)
    per_tensor_role_stats: Dict[str, Dict[str, CacheStats]] = {}
    _run_wram_fast_path(
        prepared,
        caches=caches,
        per_tensor_stats=per_tensor_stats,
        per_tensor_role_stats=per_tensor_role_stats,
    )
    group_stats = {
        group_name: _merge_cache_stats(
            caches[tensor].stats
            for tensor in allocation.tensors
            if tensor in caches
        )
        for group_name, allocation in allocations.items()
    }
    group_role_stats = {
        group_name: _merge_role_stats(
            per_tensor_role_stats[tensor]
            for tensor in allocation.tensors
            if tensor in per_tensor_role_stats
        )
        for group_name, allocation in allocations.items()
    }
    return GroupedCacheSimulationResult(
        mode="wram",
        global_stats=_merge_cache_stats(group_stats.values()),
        tensor_stats=dict(per_tensor_stats),
        group_stats=group_stats,
        group_allocations=dict(allocations),
        global_role_stats=_merge_role_stats(per_tensor_role_stats.values()),
        tensor_role_stats={
            tensor: dict(role_stats)
            for tensor, role_stats in per_tensor_role_stats.items()
        },
        group_role_stats=group_role_stats,
        tensor_allocations=tensor_allocations,
    )


# =============================================================================
# Helpers (Schedule & Parsing)
# =============================================================================


def _ensure_workload(workload: Union[Workload, str, Path]) -> Workload:
    if isinstance(workload, (str, Path)):
        return Workload.from_file(Path(workload))
    return workload


def _make_cache(
    *,
    mode: CacheMode,
    allocation: CacheGroupAllocation,
    tensor_sizes: Mapping[str, int],
    vector_width_bits: int,
    element_bytes: int,
) -> AbstractCache:
    if mode == "regfile":
        num_vector_regs = allocation.num_vector_regs
        if num_vector_regs is None:
            vector_bytes = vector_width_bits // 8
            num_vector_regs = allocation.capacity_bytes // vector_bytes
        if num_vector_regs is None or num_vector_regs <= 0:
            raise ValueError(
                f"Cache group '{allocation.name}' must allocate at least one vector register."
            )
        return RegFileCache(num_vector_regs, vector_width_bits)
    if mode == "wram":
        return WRAMCache(
            allocation.capacity_bytes,
            dict(tensor_sizes),
            element_bytes,
        )
    raise ValueError(f"Unsupported cache mode: {mode}")


def _merge_cache_stats(stats_iter: Iterator[CacheStats] | List[CacheStats]) -> CacheStats:
    merged = CacheStats()
    for stats in stats_iter:
        merged.accesses += stats.accesses
        merged.hits += stats.hits
        merged.misses += stats.misses
        merged.bytes_loaded += stats.bytes_loaded
        merged.streaming_accesses += stats.streaming_accesses
    return merged


def _access_kind(access: AccessInfo) -> CacheAccessKind:
    return "streaming" if access.is_streaming else "compute"


def _get_role_stats(
    stats_map: Dict[str, Dict[str, CacheStats]], tensor: str
) -> Dict[str, CacheStats]:
    existing = stats_map.get(tensor)
    if existing is not None:
        return existing
    entry = {
        "compute": CacheStats(),
        "streaming": CacheStats(),
    }
    stats_map[tensor] = entry
    return entry


def _merge_role_stats(
    stats_iter: Iterator[Dict[str, CacheStats]] | List[Dict[str, CacheStats]],
) -> Dict[str, CacheStats]:
    merged = {
        "compute": CacheStats(),
        "streaming": CacheStats(),
    }
    for role_stats in stats_iter:
        for role_name in ("compute", "streaming"):
            stats = role_stats.get(role_name)
            if stats is None:
                continue
            target = merged[role_name]
            target.accesses += stats.accesses
            target.hits += stats.hits
            target.misses += stats.misses
            target.bytes_loaded += stats.bytes_loaded
            target.streaming_accesses += stats.streaming_accesses
    return merged


def _cacheable_tensors_in_workload(workload: Workload) -> Tuple[str, ...]:
    return tuple(
        pattern.tensor
        for pattern in workload.ordered_tensor_access_patterns()
        if pattern.tensor in _CACHEABLE_TENSORS
    )


def _expand_group_allocations_to_tensors(
    *,
    mode: CacheMode,
    allocations: Mapping[str, CacheGroupAllocation],
    vector_width_bits: int,
) -> Dict[str, CacheGroupAllocation]:
    tensor_allocations: Dict[str, CacheGroupAllocation] = {}
    vector_bytes = vector_width_bits // 8 if mode == "regfile" else None

    for group_name, allocation in allocations.items():
        tensors = tuple(allocation.tensors)
        if not tensors:
            continue

        if mode == "regfile":
            if vector_bytes is None or vector_bytes <= 0:
                raise ValueError("regfile cache requires a positive vector width.")
            total_regs = allocation.num_vector_regs
            if total_regs is None:
                total_regs = allocation.capacity_bytes // vector_bytes
            if total_regs is None or total_regs <= 0:
                raise ValueError(
                    f"Cache group '{group_name}' must allocate at least one vector register."
                )
            regs_per_tensor = total_regs // len(tensors)
            regs_remainder = total_regs % len(tensors)
            if regs_per_tensor == 0:
                raise ValueError(
                    f"Cache group '{group_name}' only has {total_regs} vector registers "
                    f"for {len(tensors)} tensors."
                )
            for index, tensor in enumerate(tensors):
                num_regs = regs_per_tensor + (1 if index < regs_remainder else 0)
                if num_regs <= 0:
                    raise ValueError(
                        f"Tensor '{tensor}' in cache group '{group_name}' received no "
                        "vector-register capacity."
                    )
                tensor_allocations[tensor] = CacheGroupAllocation(
                    name=f"{group_name}:{tensor}",
                    tensors=(tensor,),
                    weight=allocation.weight,
                    capacity_bytes=num_regs * vector_bytes,
                    num_vector_regs=num_regs,
                )
            continue

        bytes_per_tensor = allocation.capacity_bytes // len(tensors)
        bytes_remainder = allocation.capacity_bytes % len(tensors)
        if bytes_per_tensor == 0:
            raise ValueError(
                f"Cache group '{group_name}' only has {allocation.capacity_bytes} bytes "
                f"for {len(tensors)} tensors."
            )
        for index, tensor in enumerate(tensors):
            capacity_bytes = bytes_per_tensor + (1 if index < bytes_remainder else 0)
            if capacity_bytes <= 0:
                raise ValueError(
                    f"Tensor '{tensor}' in cache group '{group_name}' received no cache capacity."
                )
            tensor_allocations[tensor] = CacheGroupAllocation(
                name=f"{group_name}:{tensor}",
                tensors=(tensor,),
                weight=allocation.weight,
                capacity_bytes=capacity_bytes,
            )

    return tensor_allocations


def _get_tensor_sizes(workload: Workload) -> Dict[str, int]:
    return {
        tensor_name: (tensor_spec.bits + 7) // 8
        for tensor_name, tensor_spec in workload.tensor_attrs.items()
    }


def _get_bounds(workload: Workload, level: int) -> OrderedDict[str, int]:
    try:
        return OrderedDict(workload.get_level_bounds(level))
    except KeyError:
        return OrderedDict()


def _get_l1_strides(
    ordered_dims: List[str],
    level0_bounds: "OrderedDict[str, int]",
    *,
    coeffs: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """
    Calculates Level 1 strides for each dimension based on Level 0 bounds,
    and multiplies by coeffs if provided.
    """
    l1_strides: Dict[str, int] = {}
    for dim in ordered_dims:
        base = level0_bounds.get(dim, 1)
        if coeffs is None:
            l1_strides[dim] = base
        else:
            l1_strides[dim] = base * coeffs.get(dim, 1)
    return l1_strides


def _prepare_simulation(
    workload: Workload,
    *,
    tensor_to_group: Mapping[str, str],
    group_names: Tuple[str, ...],
) -> _PreparedSimulation:
    level1_bounds: OrderedDict[str, int] = _get_bounds(workload, level=1)
    level0_bounds: OrderedDict[str, int] = _get_bounds(workload, level=0)
    ordered_dims = list(level1_bounds.keys())
    for dim in level0_bounds.keys():
        if dim not in level1_bounds:
            raise ValueError(f"Level 0 dimension '{dim}' not found in Level 1 bounds.")

    raw_plans_by_group: Dict[str, List[_AccessPlan]] = defaultdict(list)
    streaming_config = workload.streaming_flags()
    for pattern in workload.ordered_tensor_access_patterns():
        group_name = tensor_to_group.get(pattern.tensor)
        if group_name is None:
            continue
        compiled_indices = tuple(
            _parse_index_expression_ast(index_expr)
            for index_expr in pattern.indices
        )
        raw_plans_by_group[group_name].append(
            _AccessPlan(
                tensor=pattern.tensor,
                compiled_indices=compiled_indices,
                is_streaming=streaming_config.get(pattern.tensor, False),
                group_name=group_name,
            )
        )

    plans_by_group = {
        group_name: tuple(raw_plans_by_group.get(group_name, ()))
        for group_name in group_names
    }
    return _PreparedSimulation(
        l1_keys=tuple(ordered_dims),
        l0_keys=tuple(level0_bounds.keys()),
        level1_bounds=level1_bounds,
        level0_bounds=level0_bounds,
        l1_strides=_get_l1_strides(ordered_dims, level0_bounds),
        plans_by_group=plans_by_group,
    )


def _run_regfile_fast_path(
    prepared: _PreparedSimulation,
    *,
    caches: Mapping[str, AbstractCache],
    per_tensor_stats: Dict[str, CacheStats],
    per_tensor_role_stats: Dict[str, Dict[str, CacheStats]],
) -> None:
    current_l1_vals: Dict[str, int] = {dim: 0 for dim in prepared.l1_keys}
    tile_ctx: Dict[str, int] = {}
    for dim in prepared.l1_keys:
        tile_ctx[dim] = 0
        tile_ctx[dim.lower()] = 0

    group_entries = tuple(
        (group_name, caches[group_name], plans)
        for group_name, plans in prepared.plans_by_group.items()
    )

    def _sync_tile_dim(dim: str) -> None:
        value = current_l1_vals[dim]
        tile_ctx[dim] = value
        tile_ctx[dim.lower()] = value

    def _run_level1(level_idx: int) -> None:
        if level_idx == len(prepared.l1_keys):
            for _, cache, plans in group_entries:
                if not plans:
                    continue
                timestep_batch = [
                    AccessInfo(
                        tensor=plan.tensor,
                        global_indices=(),
                        tile_indices=tuple(
                            _evaluate_ast(expr, tile_ctx)
                            for expr in plan.compiled_indices
                        ),
                        is_streaming=plan.is_streaming,
                    )
                    for plan in plans
                ]
                cache.process_timestep(
                    timestep_batch,
                    per_tensor_stats,
                    per_tensor_role_stats,
                )
            return

        dim = prepared.l1_keys[level_idx]
        for value in range(prepared.level1_bounds[dim]):
            current_l1_vals[dim] = value
            _sync_tile_dim(dim)
            _run_level1(level_idx + 1)

    _run_level1(0)


def _run_wram_fast_path(
    prepared: _PreparedSimulation,
    *,
    caches: Mapping[str, WRAMCache],
    per_tensor_stats: Dict[str, CacheStats],
    per_tensor_role_stats: Dict[str, Dict[str, CacheStats]],
) -> None:
    current_l1_vals: Dict[str, int] = {dim: 0 for dim in prepared.l1_keys}
    current_l0_vals: Dict[str, int] = {dim: 0 for dim in prepared.l0_keys}
    eval_ctx: Dict[str, int] = {}
    for dim in prepared.l1_keys:
        eval_ctx[dim] = 0
        eval_ctx[dim.lower()] = 0

    group_entries = tuple(
        (group_name, caches[group_name], plans)
        for group_name, plans in prepared.plans_by_group.items()
    )

    def _sync_global_dim(dim: str) -> None:
        value = current_l1_vals[dim] * prepared.l1_strides[dim] + current_l0_vals.get(
            dim, 0
        )
        eval_ctx[dim] = value
        eval_ctx[dim.lower()] = value

    def _process_current_point() -> None:
        for _, cache, plans in group_entries:
            for plan in plans:
                access = AccessInfo(
                    tensor=plan.tensor,
                    global_indices=tuple(
                        _evaluate_ast(expr, eval_ctx)
                        for expr in plan.compiled_indices
                    ),
                    tile_indices=(),
                    is_streaming=plan.is_streaming,
                )
                tensor_stat = per_tensor_stats[plan.tensor]
                role_stats = _get_role_stats(per_tensor_role_stats, plan.tensor)
                cache._perform_single_access(
                    access,
                    tensor_stat,
                    role_stats[_access_kind(access)],
                )

    def _run_level0(level_idx: int) -> None:
        if level_idx == len(prepared.l0_keys):
            _process_current_point()
            return

        dim = prepared.l0_keys[level_idx]
        for value in range(prepared.level0_bounds[dim]):
            current_l0_vals[dim] = value
            _sync_global_dim(dim)
            _run_level0(level_idx + 1)

    def _run_level1(level_idx: int) -> None:
        if level_idx == len(prepared.l1_keys):
            _run_level0(0)
            return

        dim = prepared.l1_keys[level_idx]
        for value in range(prepared.level1_bounds[dim]):
            current_l1_vals[dim] = value
            _sync_global_dim(dim)
            _run_level1(level_idx + 1)

    _run_level1(0)


def _iter_schedule(workload: Workload) -> Iterator[List[AccessInfo]]:
    """
    Generates the strict execution sequence for a Single PU.

    Returns:
        Iterator of LISTS. Each list contains all accesses for one Level 1 timestep.
    """
    # 1. Extract Loop Bounds
    # Workload.get_level_bounds returns canonical per-level mappings.
    level1_bounds: OrderedDict[str, int] = _get_bounds(workload, level=1)
    level0_bounds: OrderedDict[str, int] = _get_bounds(workload, level=0)

    # 2. Calculate stride for Global indexing
    ordered_dims: List[str] = list(level1_bounds.keys())
    for dim in level0_bounds.keys():
        if dim not in level1_bounds:
            raise ValueError(f"Level 0 dimension '{dim}' not found in Level 1 bounds.")
    l1_strides: Dict[str, int] = _get_l1_strides(ordered_dims, level0_bounds)

    # 3. Get Tensor Access Patterns and Layout Configurations
    tensors_access_patterns = workload.ordered_tensor_access_patterns()
    streaming_config = workload.streaming_flags()

    # 4. State Tracking
    current_l1_vals: Dict[str, int] = {d: 0 for d in ordered_dims}
    current_l0_vals: Dict[str, int] = {d: 0 for d in ordered_dims}

    # Helper to generate context for AST eval
    def get_eval_context() -> Dict[str, int]:
        ctx = {}
        for d in ordered_dims:
            # Calculate Global Logic Coordinate
            global_val = current_l1_vals[d] * l1_strides[d] + current_l0_vals[d]
            ctx[d] = global_val
            ctx[d.lower()] = global_val  # Handle case insensitivity (N vs n)
        return ctx

    def get_tile_context() -> Dict[str, int]:
        ctx = {dim: 0 for dim in level0_bounds.keys()}
        for dim, value in current_l1_vals.items():
            ctx[dim] = value
        return ctx

    # --- Recursive Loop Generators ---

    # Helper function to run Level 0 loops
    def generate_l0_batch(l0_vars: List[str]) -> List[AccessInfo]:
        """
        Recursively iterates through Level 0 loops and returns a FLAT LIST
        of all accesses generated in this batch.
        """
        if not l0_vars:
            # Innermost body: Emit Operation Accesses
            eval_ctx = get_eval_context()
            tile_ctx = get_tile_context()
            ops = []

            for pattern in tensors_access_patterns:
                name = pattern.tensor
                is_stream = streaming_config.get(name, False)

                # Evaluate Indices
                calculated_global_indices = []
                calculated_tile_indices = []

                for idx_expr in pattern.indices:
                    # Global Element Index
                    g_val = workload.evaluate_index_expression(idx_expr, eval_ctx)
                    calculated_global_indices.append(g_val)

                    # Tile Index (Level 1 only)
                    t_val = workload.evaluate_index_expression(idx_expr, tile_ctx)
                    calculated_tile_indices.append(t_val)

                ops.append(
                    AccessInfo(
                        tensor=name,
                        global_indices=tuple(calculated_global_indices),
                        tile_indices=tuple(calculated_tile_indices),
                        is_streaming=is_stream,
                    )
                )
            return ops

        # Recursive Step: Accumulate results
        batch = []
        var = l0_vars[0]
        remaining = l0_vars[1:]
        extent = level0_bounds[var]

        for i in range(extent):
            current_l0_vals[var] = i
            # Extend the batch with results from sub-loops
            batch.extend(generate_l0_batch(remaining))

        return batch

    def run_level1(
        l1_vars: List[str], l0_vars_root: List[str]
    ) -> Iterator[List[AccessInfo]]:
        if not l1_vars:
            # End of L1 chain. We are at a fixed time step.
            # Generate the FULL batch of parallel operations (L0).
            timestep_batch = generate_l0_batch(l0_vars_root)
            yield timestep_batch
            return

        var = l1_vars[0]
        remaining = l1_vars[1:]
        extent = level1_bounds[var]

        for i in range(extent):
            current_l1_vals[var] = i
            yield from run_level1(remaining, l0_vars_root)

    # 5. Start Recursion
    l1_keys = list(level1_bounds.keys())
    l0_keys = list(level0_bounds.keys())

    yield from run_level1(l1_keys, l0_keys)
