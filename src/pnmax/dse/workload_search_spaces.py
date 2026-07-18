from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pnmax.database import Arch, Workload

SPACE_ORDER: tuple[str, ...] = ("a", "b", "c", "d")
SPACE_LABELS: dict[str, str] = {
    "a": "(a) block-sharing",
    "b": "(b) + PU-level sharing",
    "c": "(c) + system-level sharing",
    "d": "(d) + cache",
}


@dataclass(frozen=True)
class SearchSpaceSpec:
    space_id: str
    label: str
    force_block_sharing: bool = True
    force_cache_enabled: bool | None = None
    force_pu_sharing: bool | None = None
    force_system_sharing_to_max_level: bool = False
    force_system_sharing_to_channel_minus_one: bool = False
    search_pu_sharing: bool = False
    search_system_sharing: bool = False
    search_cache: bool = False


@dataclass(frozen=True)
class PreparedSearchSpace:
    spec: SearchSpaceSpec
    workload: Workload
    max_system_level: int

    @property
    def search_pu_sharing(self) -> bool:
        return bool(self.spec.search_pu_sharing)

    @property
    def search_system_sharing(self) -> bool:
        return bool(self.spec.search_system_sharing)

    @property
    def search_cache(self) -> bool:
        return bool(self.spec.search_cache)


@dataclass(frozen=True)
class SearchExecutionPlan:
    requested_spaces: tuple[str, ...]
    executed_search_spaces: tuple[str, ...]
    derived_spaces: tuple[str, ...]
    visible_spaces: tuple[str, ...]
    direct_c_search: bool = False
    direct_d_search: bool = False
    direct_l4_search: bool = False

    @property
    def search_plan_mode(self) -> str:
        if self.direct_l4_search:
            return "direct-l4"
        if self.direct_d_search:
            return "direct-d"
        if self.direct_c_search:
            return "direct-c"
        return "staged-union"


SEARCH_SPACES: dict[str, SearchSpaceSpec] = {
    "a": SearchSpaceSpec(
        space_id="a",
        label=SPACE_LABELS["a"],
    ),
    "b": SearchSpaceSpec(
        space_id="b",
        label=SPACE_LABELS["b"],
        force_pu_sharing=True,
    ),
    "c": SearchSpaceSpec(
        space_id="c",
        label=SPACE_LABELS["c"],
        force_pu_sharing=True,
        force_system_sharing_to_max_level=True,
    ),
    "d": SearchSpaceSpec(
        space_id="d",
        label=SPACE_LABELS["d"],
        force_cache_enabled=True,
    ),
}

DIRECT_C_SEARCH_SPEC = SearchSpaceSpec(
    space_id="c",
    label=SPACE_LABELS["c"],
    force_block_sharing=True,
    force_cache_enabled=False,
    search_pu_sharing=True,
    search_system_sharing=True,
)

DIRECT_L4_SEARCH_SPEC = SearchSpaceSpec(
    space_id="c",
    label=SPACE_LABELS["c"],
    force_block_sharing=True,
    force_cache_enabled=False,
    # Within-channel (L4) reduction shares across PUs, so PU sharing is
    # structurally REQUIRED: the parser rejects sharing.system > 1 with PU
    # sharing disabled. Force PU sharing on (not searched) and pin system
    # sharing to the within-channel cap (channel level - 1 = L4). Pool variety
    # comes from the tiling/factor sampling, not the sharing toggles.
    force_pu_sharing=True,
    search_pu_sharing=False,
    search_system_sharing=False,
    force_system_sharing_to_channel_minus_one=True,
)

DIRECT_D_SEARCH_SPEC = SearchSpaceSpec(
    space_id="d",
    label=SPACE_LABELS["d"],
    force_block_sharing=True,
    force_cache_enabled=True,
    search_pu_sharing=True,
    search_system_sharing=True,
)


def normalize_spaces(raw_spaces: Sequence[str]) -> tuple[str, ...]:
    valid = set(SPACE_ORDER)
    normalized: list[str] = []
    seen: set[str] = set()
    for token in raw_spaces:
        space_id = str(token).strip().lower()
        if not space_id:
            continue
        if space_id not in valid:
            raise ValueError(
                f"Unsupported space '{token}'. Expected one or more of {SPACE_ORDER}."
            )
        if space_id in seen:
            continue
        seen.add(space_id)
        normalized.append(space_id)

    if not normalized:
        raise ValueError("At least one search space must be selected.")
    return tuple(space_id for space_id in SPACE_ORDER if space_id in normalized)


def plan_search_space_execution(
    raw_spaces: Sequence[str],
    *,
    direct_c_search: bool = False,
    direct_d_search: bool = False,
    direct_l4_search: bool = False,
) -> SearchExecutionPlan:
    requested_spaces = normalize_spaces(raw_spaces)
    if direct_c_search and direct_d_search:
        raise ValueError("Direct c search and direct d search cannot both be enabled.")
    if direct_l4_search and direct_c_search:
        raise ValueError("Direct l4 search and direct c search cannot both be enabled.")
    if direct_l4_search and direct_d_search:
        raise ValueError("Direct l4 search and direct d search cannot both be enabled.")
    if direct_l4_search:
        if requested_spaces != ("c",):
            raise ValueError(
                "Direct l4 search requires the requested spaces to be exactly ('c',)."
            )
        return SearchExecutionPlan(
            requested_spaces=("c",),
            executed_search_spaces=("c",),
            derived_spaces=(),
            visible_spaces=("c",),
            direct_l4_search=True,
        )
    if direct_c_search:
        if requested_spaces != ("c",):
            raise ValueError(
                "Direct c search requires the requested spaces to be exactly ('c',)."
            )
        return SearchExecutionPlan(
            requested_spaces=("c",),
            executed_search_spaces=("c",),
            derived_spaces=(),
            visible_spaces=("c",),
            direct_c_search=True,
        )
    if direct_d_search:
        if requested_spaces != ("d",):
            raise ValueError(
                "Direct d search requires the requested spaces to be exactly ('d',)."
            )
        return SearchExecutionPlan(
            requested_spaces=("d",),
            executed_search_spaces=("d",),
            derived_spaces=(),
            visible_spaces=("d",),
            direct_d_search=True,
        )

    highest_required_space = None
    if "d" in requested_spaces or "c" in requested_spaces:
        highest_required_space = "c"
    elif "b" in requested_spaces:
        highest_required_space = "b"
    elif "a" in requested_spaces:
        highest_required_space = "a"

    if highest_required_space is None:
        raise ValueError("At least one executable search space must be selected.")

    highest_stage_index = ("a", "b", "c").index(highest_required_space)
    executed_search_spaces = ("a", "b", "c")[: highest_stage_index + 1]
    derived_spaces = tuple(space_id for space_id in requested_spaces if space_id == "d")
    visible_spaces = tuple(
        space_id
        for space_id in SPACE_ORDER
        if space_id in set(executed_search_spaces) or space_id in set(derived_spaces)
    )
    return SearchExecutionPlan(
        requested_spaces=requested_spaces,
        executed_search_spaces=executed_search_spaces,
        derived_spaces=derived_spaces,
        visible_spaces=visible_spaces,
        direct_c_search=False,
        direct_d_search=False,
        direct_l4_search=False,
    )


def prepare_workload_for_space(
    base_workload: Workload,
    arch: Arch,
    space_id: str,
    *,
    all_streaming_false: bool = False,
    direct_c_search: bool = False,
    direct_d_search: bool = False,
    direct_l4_search: bool = False,
) -> PreparedSearchSpace:
    normalized_space_id = str(space_id).strip().lower()
    if direct_c_search and direct_d_search:
        raise ValueError("Direct c search and direct d search cannot both be enabled.")
    if direct_l4_search and direct_c_search:
        raise ValueError("Direct l4 search and direct c search cannot both be enabled.")
    if direct_l4_search and direct_d_search:
        raise ValueError("Direct l4 search and direct d search cannot both be enabled.")
    if direct_l4_search:
        if normalized_space_id != "c":
            raise ValueError("Direct l4 search is only supported for search space 'c'.")
        spec = DIRECT_L4_SEARCH_SPEC
    elif direct_c_search:
        if normalized_space_id != "c":
            raise ValueError("Direct c search is only supported for search space 'c'.")
        spec = DIRECT_C_SEARCH_SPEC
    elif direct_d_search:
        if normalized_space_id != "d":
            raise ValueError("Direct d search is only supported for search space 'd'.")
        spec = DIRECT_D_SEARCH_SPEC
    else:
        spec = SEARCH_SPACES.get(normalized_space_id)
    if spec is None:
        raise ValueError(f"Unknown search space '{space_id}'.")

    workload = Workload.from_dict(
        base_workload.to_dict(),
        source=f"workload-space-{spec.space_id}",
    )
    layout = workload.layout
    if layout is None:
        raise ValueError(
            f"Workload '{workload.name}' is missing Layout-Pragma required for space '{spec.space_id}'."
        )
    if layout.sharing is None:
        raise ValueError(
            f"Workload '{workload.name}' is missing Layout-Pragma.sharing required for space '{spec.space_id}'."
        )

    max_system_level = _max_system_level(workload, arch)
    if all_streaming_false:
        layout.streaming.In = False
        layout.streaming.F = False
        layout.streaming.Out = False
    if spec.force_block_sharing:
        layout.sharing.block = True
    if spec.force_pu_sharing is not None:
        layout.sharing.pu = bool(spec.force_pu_sharing)
    if spec.force_system_sharing_to_max_level:
        layout.sharing.system = int(max_system_level)
    if spec.force_system_sharing_to_channel_minus_one:
        layout.sharing.system = int(_channel_minus_one_level(arch))
    if spec.force_cache_enabled is not None:
        layout.cache = bool(spec.force_cache_enabled)
        if not spec.force_cache_enabled:
            layout.cache_groups = ()
    layout.refresh()
    workload.refresh()

    return PreparedSearchSpace(
        spec=spec,
        workload=workload,
        max_system_level=max_system_level,
    )


def _max_system_level(workload: Workload, arch: Arch) -> int:
    workload_levels = workload.bounds.levels()
    if not workload_levels:
        raise ValueError(f"Workload '{workload.name}' defines no bound levels.")
    if not arch.defined_levels:
        raise ValueError(f"Architecture '{arch.name}' defines no hierarchy levels.")
    return min(max(workload_levels), max(arch.defined_levels))


def _channel_minus_one_level(arch: Arch) -> int:
    level_kinds = arch.hierarchy.level_to_name
    channel_levels = [
        level
        for level, kind in level_kinds.items()
        if getattr(kind, "value", kind) == "channel"
    ]
    if not channel_levels:
        raise ValueError(
            f"Architecture '{arch.name}' defines no 'channel' hierarchy level."
        )
    return min(channel_levels) - 1
