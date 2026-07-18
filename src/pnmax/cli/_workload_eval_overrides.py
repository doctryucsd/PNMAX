from __future__ import annotations

from pnmax.database import Workload


def enable_workload_cache(
    workload: Workload,
    *,
    context: str,
) -> None:
    layout = workload.layout
    if layout is None:
        raise ValueError(
            f"Workload '{workload.name}' is missing Layout-Pragma required for "
            f"{context} evaluation."
        )
    layout.cache = True
    layout.refresh()
    workload.refresh()


def apply_interconnect_workload_overrides(
    workload: Workload,
    *,
    sharing_system_level: int,
    context: str,
) -> None:
    layout = workload.layout
    if layout is None:
        raise ValueError(
            f"Workload '{workload.name}' is missing Layout-Pragma required for "
            f"{context} evaluation."
        )
    if layout.sharing is None:
        raise ValueError(
            f"Workload '{workload.name}' is missing Layout-Pragma.sharing required "
            f"for {context} evaluation."
        )
    layout.cache = True
    layout.sharing.pu = True
    layout.sharing.system = int(sharing_system_level)
    layout.refresh()
    workload.refresh()
