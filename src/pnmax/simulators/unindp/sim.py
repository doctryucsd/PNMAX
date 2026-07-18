from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from pnmax.database import Arch, Workload

from .codegen import Codegen

from pnmax.paths import repo_root

PROJECT_ROOT = repo_root()
EXTERNAL_PATH = PROJECT_ROOT / "external"

if str(EXTERNAL_PATH) not in sys.path:
    sys.path.append(str(EXTERNAL_PATH))

from unindp.sim import sim

SUPPORTED_ARCHES = {"upmem", "hbm_pim"}


# UniNDP's simulator time base per architecture: one tick = one DRAM
# command-bus cycle of the simulated standard — 1.25 ns for UPMEM's DDR4-1600,
# 1.0 ns for HBM2 (see the timing tables in external/unindp/config/
# {upmem,hbm-pim}.yaml, which are denominated in these cycles). These describe
# the vendored simulator's clock, not the PNMAX architecture model; the PU
# clock used for the cycle conversion comes from the arch YAML instead.
ARCH_TICK_NS = {
    "upmem": 1.25,
    "hbm_pim": 1.0,
}


@dataclass
class UniNDPResult:
    cycles: float
    raw_cycles: float
    ticks: float
    raw_ticks: float
    sample: float
    arch: str
    workload: str
    outdir: Optional[str]
    operation_summary: dict[str, int]


def _default_pu_frequency_hz(arch: str) -> float:
    """PU clock from the experiment-default baseline arch YAML for ``arch``."""
    arch_file = (
        PROJECT_ROOT / "data" / "archs" / "lowered" / "baseline" / f"{arch}.yaml"
    )
    return float(Arch.from_yaml_file(arch_file).specs.pu_frequency_hz)


def _ticks_to_pu_cycles(
    arch: str, ticks: float, pu_frequency_hz: Optional[float] = None
) -> float:
    arch_key = arch.lower()
    tick_ns = ARCH_TICK_NS.get(arch_key)
    if tick_ns is None or ticks <= 0:
        return float(ticks)
    freq_hz = (
        float(pu_frequency_hz)
        if pu_frequency_hz
        else _default_pu_frequency_hz(arch_key)
    )
    pu_cycle_ns = 1e9 / freq_hz
    return float(ticks) * tick_ns / pu_cycle_ns


def load_workload(path: Path) -> Workload:
    """Load and validate the workload description for UPMEM simulation."""
    workload = Workload.from_file(path)
    workload.require_unindp_layout()
    return workload


def run_unindp_sim(
    workload_file: Union[str, Path],
    arch: str,
    sample_rate: float = 1.0,
    outdir: Optional[Union[str, Path]] = None,
    dbg_op_types: list[str] = [],
    single_pu: bool = False,
    verbose: int = 0,
    show_codegen_progress: bool = False,
    show_sim_progress: bool = True,
    pu_frequency_hz: Optional[float] = None,
) -> UniNDPResult:
    """
    Programmatic entry point for UniNDP UPMEM simulations.
    arch: "upmem" or "hbm_pim"
    pu_frequency_hz: PU clock for the tick -> PU-cycle conversion; pass the
        value from the arch file in use (specs.pu_frequency). Defaults to the
        target's baseline arch YAML when omitted.
    """

    if arch.lower() not in SUPPORTED_ARCHES:
        raise ValueError(
            f"Unsupported UniNDP arch '{arch}'; supported: {SUPPORTED_ARCHES}"
        )

    workload_path = Path(workload_file)
    outdir_path = Path(outdir) if outdir is not None else None

    workload = load_workload(workload_path)
    codegen = Codegen(workload)
    instructions = codegen.gen_instructions(
        arch,
        sample_rate,
        outdir_path,
        pbar=show_codegen_progress,
        dbg_op_types=dbg_op_types,
        single_pu=single_pu,
        verbose=verbose,
    )
    operation_summary = codegen.operation_summary

    # ticks
    result_ticks = sim(
        instructions,
        silent=not show_sim_progress,
        use_tqdm=show_sim_progress,
        log_commands=False,
    )
    raw_cycles = _ticks_to_pu_cycles(arch, result_ticks, pu_frequency_hz)
    report_cycles = raw_cycles / sample_rate
    report_ticks = result_ticks / sample_rate

    return UniNDPResult(
        cycles=report_cycles,
        raw_cycles=raw_cycles,
        ticks=report_ticks,
        raw_ticks=result_ticks,
        sample=sample_rate,
        arch=arch,
        workload=str(workload_path),
        outdir=str(outdir_path) if outdir_path is not None else None,
        operation_summary=operation_summary,
    )
