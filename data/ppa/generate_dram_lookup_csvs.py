#!/usr/bin/env python3
"""
Generate DRAM lookup CSVs for `generate_arch_from_template.py`.

The consumer looks up DRAM rows by:
  - mat_width = 512
  - mat_height = 512
  - n_horizontal_mat
  - banks
  - io_width = 8
  - pim_col_bytes = 8
  - pim_n_cols_per_vector = burst_len

This generator produces exactly one row per required lookup tuple for both
CACTI (DDR4 / upmem) and DreamRAM (HBM / hbm_pim), while keeping the logical
column width fixed at 64 bits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("PNMAX_ROOT", SCRIPT_DIR.parents[1])
).resolve()  # <repo>/data/ppa -> <repo>

# External characterization tools (vendored under external/; override via env).
CACTI_DIR = Path(os.environ.get("PNMAX_CACTI_DIR", REPO_ROOT / "external" / "cacti"))
DREAM_DIR = Path(
    os.environ.get("PNMAX_DREAMRAM_DIR", REPO_ROOT / "external" / "dreamram")
)

CACTI_TEMPLATE = CACTI_DIR / "2DDRAM_micron1Gb.cfg"
DREAM_TEMPLATE = DREAM_DIR / "configs" / "mem" / "sweep" / "hbm2e.json"
DREAM_TECH_CFG = DREAM_DIR / "configs" / "tech" / "scaled" / "17nm_hbm2e.json"
DREAM_BASELINE_TEMPLATE = DREAM_DIR / "configs" / "mem" / "baseline" / "hbm2e.json"
# Recorded characterization input for DreamRAM's timing calibration: the DB
# was characterized with the mem baseline's calibration.tck at 2.5 ns (a
# pre-publication calibration value; the vendored upstream config carries its
# default 2). Generation pins the recorded value by writing a driver-owned
# copy of the baseline config with ONLY calibration.tck set to 2.5 and
# pointing each per-run config at it — the vendored external/dreamram tree
# stays pristine. Restored provenance (a recorded input), nothing invented.
DREAM_CALIBRATION_TCK = 2.5
# The vendored baseline's own value, asserted before pinning: if the vendored
# tree ever changes (upstream update / re-vendoring), the run fails loudly so
# the pin is re-reviewed instead of silently compounding.
DREAM_VENDORED_CALIBRATION_TCK = 2

UPMEM_BASELINE = REPO_ROOT / "data" / "archs" / "lowered" / "baseline" / "upmem.yaml"
HBM_BASELINE = REPO_ROOT / "data" / "archs" / "lowered" / "baseline" / "hbm_pim.yaml"

CACTI_OUT = SCRIPT_DIR / "cacti_ddr4_results.csv"
DREAM_OUT = SCRIPT_DIR / "dreamram_hbm2e_results.csv"
WORK_ROOT = SCRIPT_DIR / "lookup_work" / "dram"

MAT_WIDTH = 512
MAT_HEIGHT = 512
TOTAL_CAPACITY_BITS = 1024 * 1024 * 1024  # 1 Gb
# Logical DRAM column width (bits) assumed when generating these lookup CSVs.
# Mirrors the runtime model's io-width fallback (analytical_model.py col_bits,
# which defaults to 64 when an arch does not declare specs.col_bits).
LOGICAL_COL_BITS = 64
LOGICAL_COL_BYTES = LOGICAL_COL_BITS // 8
CACTI_BURST_DEPTH = 8
# PE/RF pipeline period (ns) at AttAcc's 666 MHz PIM clock (1000 / 666 MHz).
PIM_RF_ACCESS_NS = 1000.0 / 666.0
PIM_PE_PIPE_NS = 1000.0 / 666.0
DREAM_BANKGROUPS = 4
DREAM_LDLS_MDLS = 8

CACTI_COLUMNS = [
    "run", "burst_size", "mat_width", "mat_height", "n_horizontal_mat", "banks",
    "page_size_bits", "io_width", "ndwl", "ndbl",
    "E_ACT_pJ", "E_PRE_pJ", "E_change_row_pJ",
    "E_READ_pJ", "E_READ_pJ_per_bit",
    "E_WRITE_pJ", "E_WRITE_pJ_per_bit",
    "E_bank_sink_energy_pJ", "E_bank_sink_energy_pJ_per_bit",
    "E_bank_read_issue_pJ", "E_bank_read_issue_pJ_per_bit",
    "energy_column_access_net_pJ", "energy_column_access_net_pJ_per_bit",
    "energy_column_predecoder_pJ", "energy_column_predecoder_pJ_per_bit",
    "energy_column_decoder_pJ", "energy_column_decoder_pJ_per_bit",
    "energy_column_selectline_pJ", "energy_column_selectline_pJ_per_bit",
    "energy_datapath_net_pJ", "energy_datapath_net_pJ_per_bit",
    "energy_global_data_pJ", "energy_global_data_pJ_per_bit",
    "energy_subarray_output_drv_pJ", "energy_subarray_output_drv_pJ_per_bit",
    "energy_sense_amp_pJ", "energy_sense_amp_pJ_per_bit",
    "energy_local_data_and_drv_pJ", "energy_local_data_and_drv_pJ_per_bit",
    "energy_bitlines_pJ", "energy_bitlines_pJ_per_bit",
    "tCK_ns", "tRCD_ns", "tRP_ns", "tCL_ns",
    "tCWL_ns", "tRAS_ns", "tRC_ns", "tCCD_L_ns",
    "atom_time_ns", "change_row_latency_ns",
    "area_core_mm2", "area_die_mm2",
    "pim_col_bytes", "pim_row_bytes", "pim_n_cols_per_vector",
    "pim_bank_pu_load_row_conflict_ns", "pim_bank_pu_load_row_hit_ns",
    "pim_bank_pu_load_buffer_hit_ns",
    "pim_bank_pu_store_row_conflict_ns", "pim_bank_pu_store_row_hit_ns",
    "pim_pu_compute_ns",
    "pim_host_read_row_hit_ns", "pim_host_read_row_conflict_ns",
    "pim_host_write_row_hit_ns", "pim_host_write_row_conflict_ns",
    "pim_change_row_latency_ns",
]

DREAM_COLUMNS = [
    "run", "burst_size", "mat_width", "mat_height", "n_horizontal_mat", "banks",
    "page_size_bits", "io_width", "ndwl", "ndbl",
    "E_ACT_pJ", "E_PRE_pJ", "E_change_row_pJ",
    "E_READ_pJ", "E_READ_pJ_per_bit",
    "E_bank_sink_energy_pJ", "E_bank_sink_energy_pJ_per_bit",
    "E_bank_read_issue_pJ", "E_bank_read_issue_pJ_per_bit",
    "e_set_col_pJ", "e_set_col_pJ_per_bit",
    "e_set_csl_pJ", "e_set_csl_pJ_per_bit",
    "e_set_ldl_pJ", "e_set_ldl_pJ_per_bit",
    "e_set_mdl_pJ", "e_set_mdl_pJ_per_bit",
    "tCK_ns", "tRCD_ns", "tRP_ns", "tCL_ns",
    "tCWL_ns", "tCCD_L_ns",
    "atom_time_ns", "change_row_latency_ns",
    "atom_size_bits", "page_size_bytes", "total_area_mm2",
    "pim_col_bytes", "pim_row_bytes", "pim_n_cols_per_vector",
    "pim_bank_pu_load_row_conflict_ns", "pim_bank_pu_load_row_hit_ns",
    "pim_bank_pu_load_buffer_hit_ns",
    "pim_bank_pu_store_row_conflict_ns", "pim_bank_pu_store_row_hit_ns",
    "pim_pu_compute_ns",
    "pim_host_read_row_hit_ns", "pim_host_read_row_conflict_ns",
    "pim_host_write_row_hit_ns", "pim_host_write_row_conflict_ns",
    "pim_change_row_latency_ns",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DRAM lookup CSVs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned DRAM tuples without running CACTI or DreamRAM.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_float(text: str | None) -> float | None:
    try:
        return float(str(text).strip())
    except (AttributeError, TypeError, ValueError):
        return None


def grep_float(log: str, pattern: str) -> float | None:
    for line in log.splitlines():
        if pattern in line:
            match = re.search(
                r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
                line.split(pattern, 1)[1],
            )
            if match:
                return parse_float(match.group(1))
    return None


def grep_nj(log: str, pattern: str) -> float | None:
    value = grep_float(log, pattern)
    return None if value is None else value * 1000.0


def cfg_system_freq_mhz(cfg_text: str) -> float | None:
    """System frequency (MHz) CACTI consumed for this run, read from the cfg's
    ``-system frequency (MHz) <N>`` line exactly as the vendored parser does
    (external/cacti/io.cc sscanfs the trailing unsigned int; the last
    occurrence wins). CACTI treats the frequency as an input parameter -- it
    never computes or prints one in its normal output; its only echo (io.cc,
    debug-detail gated, ``system frequency: N``) prints this same config value
    back. The shipped characterization's tCK (2.5 ns = 1/400 MHz = the fixture
    cfgs' configured frequency) shows the pre-vendoring build derived tCK from
    the same config value, so parsing the cfg CACTI ran is the faithful source.
    """
    freq: float | None = None
    for line in cfg_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-system frequency"):
            match = re.search(r"([0-9]+)\s*$", stripped)
            if match:
                freq = parse_float(match.group(1))
    return freq


def safe_add(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(values)


def variant_num_pu(variant: int) -> int:
    if variant in (2, 3):
        return 4
    if variant in (4, 5):
        return 16
    return 8


def variant_h_mat(variant: int) -> int:
    if variant in (3, 7):
        return 32
    if variant in (5, 6):
        return 8
    return 16


def variant_burst_len(variant: int) -> int:
    if variant == 8:
        return 2
    if variant == 9:
        return 8
    return 4


def variant_v_mat(arch: str, variant: int) -> int:
    if variant in (2, 6):
        return 64 if arch == "hbm_pim" else 128
    if variant in (4, 7):
        return 16 if arch == "hbm_pim" else 32
    return 32 if arch == "hbm_pim" else 64


def banks_per_pu_by_arch() -> dict[str, int]:
    return {
        "upmem": int(load_yaml(UPMEM_BASELINE)["specs"]["banks_per_pu"]),
        "hbm_pim": int(load_yaml(HBM_BASELINE)["specs"]["banks_per_pu"]),
    }


def required_lookup_tuples() -> dict[str, dict[tuple[int, int, int, int], None]]:
    per_arch: dict[str, dict[tuple[int, int, int, int], None]] = {
        "upmem": {},
        "hbm_pim": {},
    }
    banks_per_pu = banks_per_pu_by_arch()
    for arch in per_arch:
        for variant in range(1, 16):
            num_pu = variant_num_pu(variant)
            h_mat = variant_h_mat(variant)
            v_mat = variant_v_mat(arch, variant)
            burst_len = variant_burst_len(variant)
            total_banks = num_pu * banks_per_pu[arch]
            per_arch[arch][(h_mat, total_banks, v_mat, burst_len)] = None

        # PU-count sweep (breaks the iso-capacity invariant the 15 variants hold).
        # Hold the per-bank geometry fixed at the baseline (h_mat=16, v_mat=baseline,
        # burst_len=4) and scale only num_pu = {2,4,8,16,32} = {1/4,1/2,1,2,4}x of the
        # baseline 8 PUs. Total capacity then scales with num_pu instead of staying fixed.
        # NOTE: hbm_pim caps at num_pu in {4,8,16} (banks {8,16,32}) = 3 points, not 5.
        # gbuses_out = total_banks/4 must be a power of two in [2,8], pinning total_banks
        # to {8,16,32}; num_pu=2 (banks=4 -> gbuses=1) and num_pu=32 (banks=64 -> gbuses=16)
        # are out of range and layout-independent. banks=32 needs a 4x1 bankgroup layout to
        # fit the die (see dream_bankgroups); 2x2 is die-too-tall there. CACTI (upmem) has all 5.
        base_v_mat = 32 if arch == "hbm_pim" else 64
        pu_sweep_num_pu = (4, 8, 16) if arch == "hbm_pim" else (2, 4, 8, 16, 32)
        for num_pu in pu_sweep_num_pu:
            total_banks = num_pu * banks_per_pu[arch]
            per_arch[arch][(16, total_banks, base_v_mat, 4)] = None

        # Independent OFAT axes that also break the iso-capacity invariant: vary one
        # of {hmat, vmat, burst_len} at a time over {1/4,1/2,1,2,4}x of baseline,
        # holding the rest at baseline (num_pu=8 -> banks, hmat=16, vmat=baseline,
        # burst_len=4). Unlike the original variants 2-7, these are NOT paired to keep
        # capacity fixed. DreamRAM (hbm_pim) rejects the largest bank-area points, so
        # they are excluded: hmat 4x (64), vmat 2x (64) and 4x (128).
        # burst goes to 4x (burst_len=16): burst_len=16 needs simd_lanes=64, now
        # characterized in the PU post-route table (pu_postroute_22nm_cleaned.csv via
        # fpu_simd64_full.csv), so the 4x burst point can be turned into an arch.
        base_banks = 8 * banks_per_pu[arch]
        if arch == "hbm_pim":
            # 5 DreamRAM-valid points each (others rejected: h_mat 64 -> die width;
            # v_mat 64/128 -> die height). Values differ from upmem's multipliers
            # because hbm_pim's modelable range tops out lower; goal is 5 consistent
            # points for figures, not matching the 1/4..4x ladder.
            ofat_hmat = [2, 4, 8, 16, 32]
            ofat_vmat = [2, 4, 8, 16, 32]
        else:
            ofat_hmat = [4, 8, 16, 32, 64]
            ofat_vmat = [16, 32, 64, 128, 256]
        ofat_burst = [1, 2, 4, 8, 16]
        for h_mat in ofat_hmat:
            per_arch[arch][(h_mat, base_banks, base_v_mat, 4)] = None
        for v_mat in ofat_vmat:
            per_arch[arch][(16, base_banks, v_mat, 4)] = None
        for burst_len in ofat_burst:
            per_arch[arch][(16, base_banks, base_v_mat, burst_len)] = None

    # Iso-capacity arch set (build_iso_capacity_archs.py, fig14 basis): every
    # (num_pu, h_mat, v_mat, burst) combo that builder generates needs an
    # exact-match row here (generate_arch_from_template.py accepts exactly one
    # match, no fallback). Its PLANS dict is the single source of truth; the
    # deviated combos that move two axes at once (e.g. upmem pu-fixed aspect
    # extremes and pu-compensated points) are NOT expressible by the variant/
    # OFAT axes above, so they must be planned explicitly.
    import build_iso_capacity_archs as iso_archs

    for arch, plan in iso_archs.PLANS.items():
        for axis in ("iso", "burst"):
            for num_pu, h_mat, v_mat, burst_len in plan[axis]:
                per_arch[arch][
                    (h_mat, num_pu * banks_per_pu[arch], v_mat, burst_len)
                ] = None
    return per_arch


def derive_core_params(h_mat: int, total_banks: int, v_mat: int) -> dict[str, int]:
    page_size_bits = MAT_WIDTH * h_mat
    ndwl = h_mat
    ndbl = v_mat
    return {
        "mat_width": MAT_WIDTH,
        "mat_height": MAT_HEIGHT,
        "n_horizontal_mat": h_mat,
        "banks": total_banks,
        "page_size_bits": page_size_bits,
        "io_width": LOGICAL_COL_BYTES,
        "ndwl": ndwl,
        "ndbl": ndbl,
    }


def logical_pim_latencies(
    *,
    burst_len: int,
    page_size_bits: int,
    t_rp_ns: float | None,
    t_rcd_ns: float | None,
    t_cl_ns: float | None,
    t_cwl_ns: float | None,
    t_ccd_l_ns: float | None,
    logical_atom_time_ns: float | None,
) -> dict[str, float | int]:
    latencies: dict[str, float | int] = {
        "pim_col_bytes": float(LOGICAL_COL_BYTES),
        "pim_row_bytes": float(page_size_bits / 8.0),
        "pim_n_cols_per_vector": burst_len,
        "pim_bank_pu_load_buffer_hit_ns": PIM_RF_ACCESS_NS,
        "pim_pu_compute_ns": PIM_RF_ACCESS_NS + PIM_PE_PIPE_NS,
    }
    if t_ccd_l_ns is not None:
        latencies["pim_bank_pu_load_row_hit_ns"] = ((burst_len - 1) * t_ccd_l_ns) + PIM_RF_ACCESS_NS
        latencies["pim_bank_pu_store_row_hit_ns"] = ((burst_len - 1) * t_ccd_l_ns) + PIM_RF_ACCESS_NS
    if all(value is not None for value in (t_rp_ns, t_rcd_ns, t_ccd_l_ns)):
        row_conflict = t_rp_ns + t_rcd_ns + ((burst_len - 1) * t_ccd_l_ns) + PIM_RF_ACCESS_NS
        latencies["pim_bank_pu_load_row_conflict_ns"] = row_conflict
        latencies["pim_bank_pu_store_row_conflict_ns"] = row_conflict
        latencies["pim_change_row_latency_ns"] = t_rp_ns + t_rcd_ns
    if t_cl_ns is not None and logical_atom_time_ns is not None:
        latencies["pim_host_read_row_hit_ns"] = t_cl_ns + (burst_len * logical_atom_time_ns)
    if all(value is not None for value in (t_rp_ns, t_rcd_ns, t_cl_ns, logical_atom_time_ns)):
        latencies["pim_host_read_row_conflict_ns"] = (
            t_rp_ns + t_rcd_ns + t_cl_ns + (burst_len * logical_atom_time_ns)
        )
    if t_cwl_ns is not None and logical_atom_time_ns is not None:
        latencies["pim_host_write_row_hit_ns"] = t_cwl_ns + (burst_len * logical_atom_time_ns)
    if all(value is not None for value in (t_rp_ns, t_rcd_ns, t_cwl_ns, logical_atom_time_ns)):
        latencies["pim_host_write_row_conflict_ns"] = (
            t_rp_ns + t_rcd_ns + t_cwl_ns + (burst_len * logical_atom_time_ns)
        )
    return latencies


def generate_cacti_cfg(params: dict[str, int], cfg_path: Path) -> None:
    lines = CACTI_TEMPLATE.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-UCA bank count"):
            output.append(f"-UCA bank count {params['banks']}\n")
        elif stripped.startswith("-size"):
            # Intentional capacity scaling: size the die as banks x per-bank capacity
            # instead of the template's pinned 1 GiB. Per-bank capacity (bytes) =
            # MAT_WIDTH * MAT_HEIGHT * ndwl * ndbl / 8 (page = MAT_WIDTH*ndwl bits wide,
            # MAT_HEIGHT*ndbl rows deep). This makes the die (and energy/timing) scale
            # with bank count / per-bank geometry for the capacity-breaking sweeps,
            # rather than CACTI shrinking each bank to keep the total pinned at 1 GiB.
            size_bytes = (
                params["banks"] * MAT_WIDTH * MAT_HEIGHT * params["ndwl"] * params["ndbl"] // 8
            )
            output.append(f"-size (bytes) {size_bytes}\n")
        elif stripped.startswith("-IO width"):
            output.append(f"-IO width {params['cfg_io_width_bits']}\n")
        elif stripped.startswith("-page size (bits)"):
            output.append(f"-page size (bits) {params['page_size_bits']}\n")
        elif stripped.startswith("-Ndwl"):
            output.append(f"-Ndwl {params['ndwl']}\n")
        elif stripped.startswith("-Ndbl"):
            output.append(f"-Ndbl {params['ndbl']}\n")
        else:
            output.append(line)
    cfg_path.write_text("".join(output), encoding="utf-8")


def run_cacti(cfg_path: Path) -> str:
    result = subprocess.run(
        ["./cacti", "-infile", str(cfg_path)],
        cwd=CACTI_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CACTI failed for {cfg_path.name}: {result.stderr.strip()}")
    return result.stdout + result.stderr


def parse_cacti(log: str, cfg_freq_mhz: float | None = None) -> dict[str, float | None]:
    data: dict[str, float | None] = {}
    data["E_ACT_pJ"] = grep_nj(log, "Activation energy:")
    data["E_PRE_pJ"] = grep_nj(log, "Precharge energy:")
    data["E_READ_pJ"] = grep_nj(log, "Read energy:")
    data["E_WRITE_pJ"] = grep_nj(log, "Write energy:")
    data["energy_column_access_net_pJ"] = grep_nj(log, "column access bus energy (nJ):")
    data["energy_column_predecoder_pJ"] = grep_nj(log, "column predecoder energy (nJ):")
    data["energy_column_decoder_pJ"] = grep_nj(log, "column decoder energy (nJ):")
    data["energy_column_selectline_pJ"] = grep_nj(log, "column selectline energy (nJ):")
    data["energy_datapath_net_pJ"] = grep_nj(log, "datapath bus energy (nJ):")
    data["energy_global_data_pJ"] = grep_nj(log, "global dataline energy (nJ):")
    data["energy_subarray_output_drv_pJ"] = grep_nj(log, "data buffer energy (nJ):")
    data["energy_sense_amp_pJ"] = grep_nj(log, "sense amp energy (nJ):")
    data["energy_bitlines_pJ"] = grep_nj(log, "bitline energy (nJ):")
    data["energy_local_data_and_drv_pJ"] = grep_nj(log, "local dataline energy (nJ)")

    data["E_change_row_pJ"] = safe_add(data["E_ACT_pJ"], data["E_PRE_pJ"])
    # CACTI 3D DRAM models read/write as:
    #   membus_CAS + bank.mat.power_subarray_out_drv + membus_data + datapath_energy
    # The "bank sink" consumed by `host_pu_transfer` should therefore cover the
    # bank/CAS/data-path terms above, but not activation/precharge terms
    # (bitlines/sense amp) or the constant external datapath_energy remainder.
    data["E_bank_sink_energy_pJ"] = safe_add(
        data["energy_column_access_net_pJ"],
        data["energy_column_predecoder_pJ"],
        data["energy_column_decoder_pJ"],
        data["energy_column_selectline_pJ"],
        data["energy_datapath_net_pJ"],
        data["energy_global_data_pJ"],
        data["energy_subarray_output_drv_pJ"],
        data["energy_local_data_and_drv_pJ"],
    )
    data["E_bank_read_issue_pJ"] = data["E_bank_sink_energy_pJ"]

    data["tRCD_ns"] = grep_float(log, "t_RCD (Row to column command delay):")
    data["tRP_ns"] = grep_float(log, "t_RP (Row precharge latency):")
    data["tCL_ns"] = grep_float(log, "t_CAS (Column access strobe latency):")
    data["tRAS_ns"] = grep_float(log, "t_RAS (Row access strobe latency):")
    data["tRC_ns"] = grep_float(log, "t_RC (Row cycle):")
    # tCK source: the system frequency CACTI consumed. The vendored binary
    # prints no frequency in its normal output, so the cfg value (parsed by
    # cfg_system_freq_mhz) is authoritative; if this or another CACTI build
    # does echo one (pre-vendoring "system frequency (MHz):", or the vendored
    # debug-detail "system frequency:"), it must agree with the cfg.
    stdout_freq = grep_float(log, "system frequency (MHz):")
    if stdout_freq is None:
        stdout_freq = grep_float(log, "system frequency:")
    if (
        stdout_freq is not None
        and cfg_freq_mhz is not None
        and abs(stdout_freq - cfg_freq_mhz) > 1e-9
    ):
        raise RuntimeError(
            f"CACTI echoed system frequency {stdout_freq} MHz but the cfg "
            f"requested {cfg_freq_mhz} MHz."
        )
    freq_mhz = stdout_freq if stdout_freq is not None else cfg_freq_mhz
    data["tCK_ns"] = (1000.0 / freq_mhz) if freq_mhz else None
    data["tCWL_ns"] = (
        (data["tCL_ns"] - data["tCK_ns"])
        if data["tCL_ns"] is not None and data["tCK_ns"] is not None
        else None
    )
    data["tCCD_L_ns"] = (4.0 * data["tCK_ns"]) if data["tCK_ns"] is not None else None
    data["atom_time_ns"] = (
        CACTI_BURST_DEPTH * data["tCK_ns"] if data["tCK_ns"] is not None else None
    )
    data["change_row_latency_ns"] = safe_add(data["tRP_ns"], data["tRCD_ns"])
    data["area_core_mm2"] = grep_float(log, "DRAM core area:")
    data["area_die_mm2"] = grep_float(log, "DRAM area per die:")
    return data


def dream_bankgroups(total_banks: int) -> tuple[int, int]:
    """(horiz_bg, vert_bg) for a given total bank count, total = h*v*banks_per_bg.

    Default is the template 2x2 (=4 BG). total_banks=32 is die-too-tall under 2x2
    (banks_per_bg=8 stacks vertically -> status 7), but fits with a wider 4x1 layout
    (banks_per_bg=8, spread horizontally). Bankgroup split does NOT change gbuses_out
    (= total_banks/4), so this only addresses the die-dimension limit, not muxing.
    """
    if total_banks >= 32:
        return (4, 1)
    return (2, 2)


def write_dream_baseline_cfg(baseline_path: Path) -> None:
    """Write the driver-owned DreamRAM mem baseline with the recorded
    characterization calibration (calibration.tck = 2.5) pinned.

    Only that single value deviates from the vendored baseline template; the
    vendored file itself is never modified.
    """
    baseline = json.loads(DREAM_BASELINE_TEMPLATE.read_text(encoding="utf-8"))
    calibration = baseline["memconfig"].get("calibration")
    if not isinstance(calibration, dict) or "tck" not in calibration:
        raise RuntimeError(
            f"{DREAM_BASELINE_TEMPLATE} has no memconfig.calibration.tck — "
            "cannot apply the recorded characterization calibration pin."
        )
    if calibration["tck"] != DREAM_VENDORED_CALIBRATION_TCK:
        raise RuntimeError(
            f"{DREAM_BASELINE_TEMPLATE} carries calibration.tck = "
            f"{calibration['tck']!r}, expected the vendored default "
            f"{DREAM_VENDORED_CALIBRATION_TCK!r}. The vendored DreamRAM tree "
            "changed; re-review the recorded calibration pin "
            "(DREAM_CALIBRATION_TCK) before generating."
        )
    calibration["tck"] = DREAM_CALIBRATION_TCK
    baseline_path.write_text(json.dumps(baseline, indent=4), encoding="utf-8")


def write_dream_cfg(params: dict[str, int], atom_size_bits: int, cfg_path: Path) -> None:
    cfg = json.loads(DREAM_TEMPLATE.read_text(encoding="utf-8"))
    memcfg = cfg["memconfig"]
    # Point the per-run config at the driver-owned pinned baseline. DreamRAM
    # resolves the reference from its own cwd (DREAM_DIR), so emit the path
    # relative to it — keeps the derived per-run configs machine-independent.
    baseline_path = cfg_path.parent / "dreamram_baseline.json"
    write_dream_baseline_cfg(baseline_path)
    memcfg["baseline"] = os.path.relpath(baseline_path.resolve(), DREAM_DIR)
    horiz_bg, vert_bg = dream_bankgroups(params["banks"])
    memcfg["organization"]["horizontal bankgroups"] = [horiz_bg]
    memcfg["organization"]["vertical bankgroups"] = [vert_bg]
    memcfg["organization"]["banks"] = [params["banks"] // (horiz_bg * vert_bg)]
    memcfg["bank"]["subarrays"] = [params["ndbl"]]
    memcfg["bank"]["mats"] = [params["n_horizontal_mat"]]
    memcfg["mat"]["wordlines"] = [params["mat_height"]]
    memcfg["mat"]["bitlines"] = [params["mat_width"]]
    memcfg["mods"]["atom size"] = [atom_size_bits]
    cfg_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")


def run_dreamram(cfg_path: Path, run_name: str) -> Path:
    out_dir = DREAM_DIR / "data" / run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    result = subprocess.run(
        ["python3", "dreamram.py", "-m", str(cfg_path), "-t", str(DREAM_TECH_CFG), "-o", run_name],
        cwd=DREAM_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"DreamRAM failed for {run_name}: {result.stderr.strip()}")
    csv_path = out_dir / f"hbm3_{run_name}.csv"
    if not csv_path.exists():
        raise RuntimeError(f"DreamRAM did not produce {csv_path}.")
    return csv_path


def _dreamram_area_matches_request(
    row: dict[str, str],
    *,
    expected_h_mat: int,
    expected_total_banks: int,
    expected_v_mat: int,
    expected_atom_bits: int,
) -> bool:
    actual_h_mat = parse_float(row.get("mats"))
    actual_v_mat = parse_float(row.get("subarrays"))
    horiz_bg = parse_float(row.get("horiz_bg"))
    vert_bg = parse_float(row.get("vert_bg"))
    banks_per_bg = parse_float(row.get("banks"))
    actual_atom_bits = parse_float(row.get("atom_size"))
    if any(
        value is None
        for value in (actual_h_mat, actual_v_mat, horiz_bg, vert_bg, banks_per_bg, actual_atom_bits)
    ):
        return False
    actual_total_banks = horiz_bg * vert_bg * banks_per_bg
    return (
        int(actual_h_mat) == expected_h_mat
        and int(actual_v_mat) == expected_v_mat
        and int(actual_total_banks) == expected_total_banks
        and int(actual_atom_bits) == expected_atom_bits
    )


def parse_dreamram(
    csv_path: Path,
    *,
    expected_h_mat: int,
    expected_total_banks: int,
    expected_v_mat: int,
    expected_atom_bits: int,
) -> dict[str, float | None]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"DreamRAM output at {csv_path} has no data rows.")
    row = rows[1] if len(rows) >= 2 else rows[0]
    area_valid = _dreamram_area_matches_request(
        row,
        expected_h_mat=expected_h_mat,
        expected_total_banks=expected_total_banks,
        expected_v_mat=expected_v_mat,
        expected_atom_bits=expected_atom_bits,
    )

    def g(column: str) -> float | None:
        return parse_float(row.get(column))

    die_x_mm = g("die_x_mm")
    die_y_mm = g("die_y_mm")

    data: dict[str, float | None] = {
        "E_ACT_pJ": g("e_cmd_act_pj"),
        "E_PRE_pJ": g("e_cmd_pre_pj"),
        "E_READ_pJ": g("e_cmd_rd_pj"),
        "e_set_col_pJ": g("e_set_col"),
        "e_set_csl_pJ": g("e_set_csl"),
        "e_set_ldl_pJ": g("e_set_ldl"),
        "e_set_mdl_pJ": g("e_set_mdl"),
        "tCK_ns": g("core_pd_ns"),
        "tCL_ns": g("tcl"),
        "tRCD_ns": g("trcd"),
        "tRP_ns": g("trp"),
        "atom_time_ns": g("atom_time"),
        "atom_size_bits": g("atom_size"),
        "page_size_bytes": g("page_size_bytes"),
        # Use single-die footprint area, not stacked total silicon area.
        "total_area_mm2": (
            die_x_mm * die_y_mm
            if area_valid and die_x_mm is not None and die_y_mm is not None
            else (g("total_area_mmmm") if area_valid else None)
        ),
    }
    if not area_valid:
        print(
            "Warning: DreamRAM returned a geometry-mismatched row for "
            f"{csv_path.name}; leaving total_area_mm2 blank. "
            f"Expected h_mat={expected_h_mat}, banks={expected_total_banks}, "
            f"v_mat={expected_v_mat}, atom_bits={expected_atom_bits}.",
            file=sys.stderr,
        )
        # Explicitly-warned blank: exempt this cell from the silent-blank
        # guard (assert_no_silent_blanks) so the loud warning above stays the
        # single signal for it.
        data["_blank_ok"] = ("total_area_mm2",)
    data["E_change_row_pJ"] = safe_add(data["E_ACT_pJ"], data["E_PRE_pJ"])
    data["E_bank_sink_energy_pJ"] = safe_add(
        data["e_set_col_pJ"],
        data["e_set_csl_pJ"],
        data["e_set_ldl_pJ"],
        data["e_set_mdl_pJ"],
    )
    data["E_bank_read_issue_pJ"] = data["E_bank_sink_energy_pJ"]
    data["tCWL_ns"] = (
        (data["tCL_ns"] - data["tCK_ns"])
        if data["tCL_ns"] is not None and data["tCK_ns"] is not None
        else None
    )
    data["tCCD_L_ns"] = (4.0 * data["tCK_ns"]) if data["tCK_ns"] is not None else None
    data["change_row_latency_ns"] = safe_add(data["tRP_ns"], data["tRCD_ns"])
    return data


def cacti_rows_for_combo(params: dict[str, int], burst_len: int, work_dir: Path) -> list[dict[str, object]]:
    physical_burst_bits = LOGICAL_COL_BITS * burst_len
    cfg_params = dict(params)
    cfg_params["cfg_io_width_bits"] = physical_burst_bits // CACTI_BURST_DEPTH
    cfg_path = work_dir / "cacti.cfg"
    generate_cacti_cfg(cfg_params, cfg_path)
    cfg_freq_mhz = cfg_system_freq_mhz(cfg_path.read_text(encoding="utf-8"))
    log = run_cacti(cfg_path)
    parsed = parse_cacti(log, cfg_freq_mhz)
    run = (
        f"bs{physical_burst_bits}_mw{MAT_WIDTH}_mh{MAT_HEIGHT}_nhm{params['n_horizontal_mat']}"
        f"_bk{params['banks']}_v{params['ndbl']}_cols{burst_len}"
    )
    pim = logical_pim_latencies(
        burst_len=burst_len,
        page_size_bits=params["page_size_bits"],
        t_rp_ns=parsed["tRP_ns"],
        t_rcd_ns=parsed["tRCD_ns"],
        t_cl_ns=parsed["tCL_ns"],
        t_cwl_ns=parsed["tCWL_ns"],
        t_ccd_l_ns=parsed["tCCD_L_ns"],
        logical_atom_time_ns=parsed["atom_time_ns"],
    )
    row = {
        "run": run,
        "burst_size": physical_burst_bits,
        "mat_width": MAT_WIDTH,
        "mat_height": MAT_HEIGHT,
        "n_horizontal_mat": params["n_horizontal_mat"],
        "banks": params["banks"],
        "page_size_bits": params["page_size_bits"],
        "io_width": LOGICAL_COL_BYTES,
        "ndwl": params["ndwl"],
        "ndbl": params["ndbl"],
        "E_ACT_pJ": parsed["E_ACT_pJ"],
        "E_PRE_pJ": parsed["E_PRE_pJ"],
        "E_change_row_pJ": parsed["E_change_row_pJ"],
        "E_READ_pJ": parsed["E_READ_pJ"],
        "E_READ_pJ_per_bit": (
            parsed["E_READ_pJ"] / physical_burst_bits if parsed["E_READ_pJ"] is not None else ""
        ),
        "E_WRITE_pJ": parsed["E_WRITE_pJ"],
        "E_WRITE_pJ_per_bit": (
            parsed["E_WRITE_pJ"] / physical_burst_bits if parsed["E_WRITE_pJ"] is not None else ""
        ),
        "E_bank_sink_energy_pJ": parsed["E_bank_sink_energy_pJ"],
        "E_bank_sink_energy_pJ_per_bit": (
            parsed["E_bank_sink_energy_pJ"] / physical_burst_bits
            if parsed["E_bank_sink_energy_pJ"] is not None
            else ""
        ),
        "E_bank_read_issue_pJ": parsed["E_bank_read_issue_pJ"],
        "E_bank_read_issue_pJ_per_bit": (
            parsed["E_bank_read_issue_pJ"] / physical_burst_bits
            if parsed["E_bank_read_issue_pJ"] is not None
            else ""
        ),
        "energy_column_access_net_pJ": parsed["energy_column_access_net_pJ"],
        "energy_column_access_net_pJ_per_bit": (
            parsed["energy_column_access_net_pJ"] / physical_burst_bits
            if parsed["energy_column_access_net_pJ"] is not None
            else ""
        ),
        "energy_column_predecoder_pJ": parsed["energy_column_predecoder_pJ"],
        "energy_column_predecoder_pJ_per_bit": (
            parsed["energy_column_predecoder_pJ"] / physical_burst_bits
            if parsed["energy_column_predecoder_pJ"] is not None
            else ""
        ),
        "energy_column_decoder_pJ": parsed["energy_column_decoder_pJ"],
        "energy_column_decoder_pJ_per_bit": (
            parsed["energy_column_decoder_pJ"] / physical_burst_bits
            if parsed["energy_column_decoder_pJ"] is not None
            else ""
        ),
        "energy_column_selectline_pJ": parsed["energy_column_selectline_pJ"],
        "energy_column_selectline_pJ_per_bit": (
            parsed["energy_column_selectline_pJ"] / physical_burst_bits
            if parsed["energy_column_selectline_pJ"] is not None
            else ""
        ),
        "energy_datapath_net_pJ": parsed["energy_datapath_net_pJ"],
        "energy_datapath_net_pJ_per_bit": (
            parsed["energy_datapath_net_pJ"] / physical_burst_bits
            if parsed["energy_datapath_net_pJ"] is not None
            else ""
        ),
        "energy_global_data_pJ": parsed["energy_global_data_pJ"],
        "energy_global_data_pJ_per_bit": (
            parsed["energy_global_data_pJ"] / physical_burst_bits
            if parsed["energy_global_data_pJ"] is not None
            else ""
        ),
        "energy_subarray_output_drv_pJ": parsed["energy_subarray_output_drv_pJ"],
        "energy_subarray_output_drv_pJ_per_bit": (
            parsed["energy_subarray_output_drv_pJ"] / physical_burst_bits
            if parsed["energy_subarray_output_drv_pJ"] is not None
            else ""
        ),
        "energy_sense_amp_pJ": parsed["energy_sense_amp_pJ"],
        "energy_sense_amp_pJ_per_bit": (
            parsed["energy_sense_amp_pJ"] / physical_burst_bits
            if parsed["energy_sense_amp_pJ"] is not None
            else ""
        ),
        "energy_local_data_and_drv_pJ": parsed["energy_local_data_and_drv_pJ"],
        "energy_local_data_and_drv_pJ_per_bit": (
            parsed["energy_local_data_and_drv_pJ"] / physical_burst_bits
            if parsed["energy_local_data_and_drv_pJ"] is not None
            else ""
        ),
        "energy_bitlines_pJ": parsed["energy_bitlines_pJ"],
        "energy_bitlines_pJ_per_bit": (
            parsed["energy_bitlines_pJ"] / physical_burst_bits
            if parsed["energy_bitlines_pJ"] is not None
            else ""
        ),
        "tCK_ns": parsed["tCK_ns"],
        "tRCD_ns": parsed["tRCD_ns"],
        "tRP_ns": parsed["tRP_ns"],
        "tCL_ns": parsed["tCL_ns"],
        "tCWL_ns": parsed["tCWL_ns"],
        "tRAS_ns": parsed["tRAS_ns"],
        "tRC_ns": parsed["tRC_ns"],
        "tCCD_L_ns": parsed["tCCD_L_ns"],
        "atom_time_ns": parsed["atom_time_ns"],
        "change_row_latency_ns": parsed["change_row_latency_ns"],
        "area_core_mm2": parsed["area_core_mm2"],
        "area_die_mm2": parsed["area_die_mm2"],
    }
    row.update(pim)
    return [row]


def effective_hbm_atom_bits(h_mat: int) -> int:
    return max(LOGICAL_COL_BITS, h_mat * DREAM_LDLS_MDLS)


def dream_rows_for_combo(params: dict[str, int], burst_lens: list[int], work_dir: Path) -> list[dict[str, object]]:
    atom_bits = effective_hbm_atom_bits(params["n_horizontal_mat"])
    cfg_path = work_dir / "dreamram_mem.json"
    write_dream_cfg(params, atom_bits, cfg_path)
    run_name = f"lookup_h{params['n_horizontal_mat']}_b{params['banks']}_v{params['ndbl']}_a{atom_bits}"
    csv_path = run_dreamram(cfg_path, run_name)
    parsed = parse_dreamram(
        csv_path,
        expected_h_mat=params["n_horizontal_mat"],
        expected_total_banks=params["banks"],
        expected_v_mat=params["ndbl"],
        expected_atom_bits=atom_bits,
    )

    physical_atom_time_ns = parsed["atom_time_ns"]
    logical_atom_time_ns = (
        physical_atom_time_ns * (LOGICAL_COL_BITS / atom_bits)
        if physical_atom_time_ns is not None
        else None
    )

    rows: list[dict[str, object]] = []
    for burst_len in burst_lens:
        run = (
            f"bs64_eff{atom_bits}_mw{MAT_WIDTH}_mh{MAT_HEIGHT}"
            f"_nhm{params['n_horizontal_mat']}_bk{params['banks']}_v{params['ndbl']}_cols{burst_len}"
        )
        pim = logical_pim_latencies(
            burst_len=burst_len,
            page_size_bits=params["page_size_bits"],
            t_rp_ns=parsed["tRP_ns"],
            t_rcd_ns=parsed["tRCD_ns"],
            t_cl_ns=parsed["tCL_ns"],
            t_cwl_ns=parsed["tCWL_ns"],
            t_ccd_l_ns=parsed["tCCD_L_ns"],
            logical_atom_time_ns=logical_atom_time_ns,
        )
        row = {
            "run": run,
            "burst_size": atom_bits,
            "mat_width": MAT_WIDTH,
            "mat_height": MAT_HEIGHT,
            "n_horizontal_mat": params["n_horizontal_mat"],
            "banks": params["banks"],
            "page_size_bits": params["page_size_bits"],
            "io_width": LOGICAL_COL_BYTES,
            "ndwl": params["ndwl"],
            "ndbl": params["ndbl"],
            "E_ACT_pJ": parsed["E_ACT_pJ"],
            "E_PRE_pJ": parsed["E_PRE_pJ"],
            "E_change_row_pJ": parsed["E_change_row_pJ"],
            "E_READ_pJ": parsed["E_READ_pJ"],
            "E_READ_pJ_per_bit": (
                parsed["E_READ_pJ"] / atom_bits if parsed["E_READ_pJ"] is not None else ""
            ),
            "E_bank_sink_energy_pJ": parsed["E_bank_sink_energy_pJ"],
            "E_bank_sink_energy_pJ_per_bit": (
                parsed["E_bank_sink_energy_pJ"] / atom_bits
                if parsed["E_bank_sink_energy_pJ"] is not None
                else ""
            ),
            "E_bank_read_issue_pJ": parsed["E_bank_read_issue_pJ"],
            "E_bank_read_issue_pJ_per_bit": (
                parsed["E_bank_read_issue_pJ"] / atom_bits
                if parsed["E_bank_read_issue_pJ"] is not None
                else ""
            ),
            "e_set_col_pJ": parsed["e_set_col_pJ"],
            "e_set_col_pJ_per_bit": (
                parsed["e_set_col_pJ"] / atom_bits if parsed["e_set_col_pJ"] is not None else ""
            ),
            "e_set_csl_pJ": parsed["e_set_csl_pJ"],
            "e_set_csl_pJ_per_bit": (
                parsed["e_set_csl_pJ"] / atom_bits if parsed["e_set_csl_pJ"] is not None else ""
            ),
            "e_set_ldl_pJ": parsed["e_set_ldl_pJ"],
            "e_set_ldl_pJ_per_bit": (
                parsed["e_set_ldl_pJ"] / atom_bits if parsed["e_set_ldl_pJ"] is not None else ""
            ),
            "e_set_mdl_pJ": parsed["e_set_mdl_pJ"],
            "e_set_mdl_pJ_per_bit": (
                parsed["e_set_mdl_pJ"] / atom_bits if parsed["e_set_mdl_pJ"] is not None else ""
            ),
            "tCK_ns": parsed["tCK_ns"],
            "tRCD_ns": parsed["tRCD_ns"],
            "tRP_ns": parsed["tRP_ns"],
            "tCL_ns": parsed["tCL_ns"],
            "tCWL_ns": parsed["tCWL_ns"],
            "tCCD_L_ns": parsed["tCCD_L_ns"],
            "atom_time_ns": logical_atom_time_ns,
            "change_row_latency_ns": parsed["change_row_latency_ns"],
            "atom_size_bits": parsed["atom_size_bits"],
            "page_size_bytes": parsed["page_size_bytes"],
            "total_area_mm2": parsed["total_area_mm2"],
        }
        row.update(pim)
        if parsed.get("_blank_ok"):
            row["_blank_ok"] = parsed["_blank_ok"]
        rows.append(row)
    return rows


def assert_no_silent_blanks(
    rows: list[dict[str, object]], columns: list[str], csv_name: str
) -> None:
    """Fail loudly if any planned cell would be written empty.

    Guards against silent truncation (e.g. a parse pattern that no longer
    matches the vendored tool's output would otherwise blank whole columns
    without any error). Cells listed in a row's ``_blank_ok`` marker were
    already warned about explicitly at parse time and are exempt; the marker
    is stripped here so it never reaches the CSV.
    """
    if not rows:
        raise RuntimeError(f"{csv_name}: no rows were generated.")
    blank_counts: dict[str, int] = {column: 0 for column in columns}
    unexpected: list[tuple[str, str]] = []
    for row in rows:
        allowed = row.pop("_blank_ok", ())
        for column in columns:
            value = row.get(column)
            if value is None or value == "":
                blank_counts[column] += 1
                if column not in allowed:
                    unexpected.append((str(row.get("run", "<row>")), column))
    empty_columns = [
        column for column, count in blank_counts.items() if count == len(rows)
    ]
    if empty_columns or unexpected:
        details = []
        if empty_columns:
            details.append(f"columns entirely empty: {', '.join(empty_columns)}")
        if unexpected:
            shown = ", ".join(f"{run}:{column}" for run, column in unexpected[:20])
            more = "" if len(unexpected) <= 20 else f" (+{len(unexpected) - 20} more)"
            details.append(f"unexpected blank cells: {shown}{more}")
        raise RuntimeError(
            f"{csv_name}: refusing to write silently-blank data -- "
            + "; ".join(details)
        )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    required = required_lookup_tuples()

    if args.dry_run:
        for arch, tuples in required.items():
            print(arch)
            for h_mat, total_banks, v_mat, burst_len in sorted(tuples):
                print(
                    f"  h_mat={h_mat:>2} banks={total_banks:>2} burst_len={burst_len} v_mat={v_mat}"
                )
        return 0

    # lookup_work/ is a derived output tree (not shipped); create it fully.
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    cacti_grouped: dict[tuple[int, int, int, int], None] = {}
    dream_grouped: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for h_mat, total_banks, v_mat, burst_len in required["upmem"]:
        cacti_grouped[(h_mat, total_banks, v_mat, burst_len)] = None
    for h_mat, total_banks, v_mat, burst_len in required["hbm_pim"]:
        dream_grouped[(h_mat, total_banks, v_mat)].add(burst_len)

    cacti_rows: list[dict[str, object]] = []
    for h_mat, total_banks, v_mat, burst_len in sorted(cacti_grouped):
        params = derive_core_params(h_mat, total_banks, v_mat)
        work_dir = WORK_ROOT / f"cacti_h{h_mat}_b{total_banks}_v{v_mat}_c{burst_len}"
        work_dir.mkdir(exist_ok=True)
        cacti_rows.extend(cacti_rows_for_combo(params, burst_len, work_dir))

    dream_rows: list[dict[str, object]] = []
    for h_mat, total_banks, v_mat in sorted(dream_grouped):
        params = derive_core_params(h_mat, total_banks, v_mat)
        work_dir = WORK_ROOT / f"dream_h{h_mat}_b{total_banks}_v{v_mat}"
        work_dir.mkdir(exist_ok=True)
        dream_rows.extend(dream_rows_for_combo(params, sorted(dream_grouped[(h_mat, total_banks, v_mat)]), work_dir))

    cacti_rows.sort(
        key=lambda row: (
            row["n_horizontal_mat"],
            row["banks"],
            row["ndbl"],
            row["pim_n_cols_per_vector"],
        )
    )
    dream_rows.sort(
        key=lambda row: (
            row["n_horizontal_mat"],
            row["banks"],
            row["ndbl"],
            row["pim_n_cols_per_vector"],
        )
    )

    assert_no_silent_blanks(cacti_rows, CACTI_COLUMNS, CACTI_OUT.name)
    assert_no_silent_blanks(dream_rows, DREAM_COLUMNS, DREAM_OUT.name)
    write_csv(CACTI_OUT, CACTI_COLUMNS, cacti_rows)
    write_csv(DREAM_OUT, DREAM_COLUMNS, dream_rows)
    print(f"Wrote {len(cacti_rows)} rows to {CACTI_OUT}")
    print(f"Wrote {len(dream_rows)} rows to {DREAM_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
