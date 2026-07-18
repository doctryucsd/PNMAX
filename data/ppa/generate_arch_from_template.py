from __future__ import annotations

import argparse
import csv
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SWEEP_DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = SWEEP_DATA_DIR.parents[1]
DRAM_RESULTS_BY_ARCH = {
    "upmem": SWEEP_DATA_DIR / "cacti_ddr4_results.csv",
    "hbm_pim": SWEEP_DATA_DIR / "dreamram_hbm2e_results.csv",
}
SRAM_RESULTS_PATH = SWEEP_DATA_DIR / "sram_sweep_results.csv"
PU_RESULTS_PATH = SWEEP_DATA_DIR / "pu_postroute_22nm_cleaned.csv"

COL_WIDTH_BITS = 64
COL_WIDTH_BYTES = COL_WIDTH_BITS // 8
DRAM_MAT_WIDTH = 512
DRAM_MAT_HEIGHT = 512
TOTAL_CAPACITY_Mb = 1024.0
ENERGY_PER_MM_PJ = 0.3846
DELAY_PER_MM_NS = 0.2377

PU_FREQUENCY_MHZ_BY_ARCH = {
    "upmem": 400.0,
    "hbm_pim": 400.0,
}
PU_TCK_NS_BY_ARCH = {
    arch_type: 1000.0 / freq_mhz
    for arch_type, freq_mhz in PU_FREQUENCY_MHZ_BY_ARCH.items()
}
# pim_overhead scale fix: DreamRAM sizes a full HBM die = channels_per_die(2) x
# pseudochannels(2) = 4 pseudo-channels (64 banks), but the PNMAX arch (banks =
# num_pu * banks_per_pu = 16) models only ONE pseudo-channel. Since the overhead
# denominator is the full DreamRAM die, scale the per-pseudo-channel PIM logic up to the
# whole die. CACTI (upmem) sizes the die to exactly our banks, so its factor is 1.
# Source: DreamRAM configs/mem/sweep/hbm2e.json organization (channels per die,
# pseudochannels).
PSEUDO_CHANNELS_PER_DIE_BY_ARCH = {
    "upmem": 1,
    "hbm_pim": 4,
}
# DRAM-process area penalty on the PIM logic (PU + SRAM): logic/SRAM synthesized in a DRAM
# process is far less dense than the logic process they were characterized in. Applied to
# hbm_pim only (upmem gets no penalty). Both a_pu and a_rf flow into total_area_mm2 AND
# pim_overhead, so both outputs carry the penalty.
DRAM_PROCESS_PU_PENALTY_BY_ARCH = {
    "upmem": 1.0,
    "hbm_pim": 10.0,
}
DRAM_PROCESS_SRAM_PENALTY_BY_ARCH = {
    "upmem": 1.0,
    "hbm_pim": 10.0,
}
CLI_COMMON_DEFAULTS = {
    "burst_len": 4,
    "h_mat": 16,
    "num_pu": 8,
    "interconnect": "host",
}
CLI_ARCH_DEFAULTS = {
    "upmem": {
        "v_mat": 64,
        "buffer_size": "32KB",
    },
    "hbm_pim": {
        "v_mat": 32,
        "buffer_size": "512B",
    },
}

WIRE_LEN_MM_BY_ARCH_AND_INTERCONNECT = {
    "upmem": {
        "intra-bg": 3.2,
        "inter-bg": (3.2 * 4.0) + 6.4,
    },
    "hbm_pim": {
        "intra-bg": 3.2,
        "inter-bg": (1.6 * 2.0) + (3.2 * 4.0),
    },
}

LINE_WIDTH_UM: float | None = 0.088


def _coerce_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return text
    try:
        if "." not in text and "e" not in text.lower():
            return int(text)
        return float(text)
    except ValueError:
        return text


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: _coerce_value(val) for key, val in row.items()} for row in reader]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Architecture YAML at {path} must decode to a mapping.")
    return payload


def _parse_buffer_size_bytes(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        size_bytes = int(value)
        if size_bytes <= 0:
            raise ValueError(f"buffer_size must be positive, got {value}.")
        return size_bytes
    text = str(value).strip().upper()
    if not text:
        raise ValueError("Empty buffer_size string.")
    if text.endswith("KB"):
        size_bytes = int(float(text[:-2]) * 1024)
    elif text.endswith("B"):
        size_bytes = int(float(text[:-1]))
    else:
        raise ValueError(
            f"Unsupported buffer_size format: {value}. Expected integer bytes or a string ending in B or KB."
        )
    if size_bytes <= 0:
        raise ValueError(f"buffer_size must be positive, got {value}.")
    return size_bytes


def _parse_bandwidth_bits_per_ns(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        bandwidth_Bps = float(value)
    else:
        text = str(value).strip().upper()
        if text.endswith("GB/S"):
            bandwidth_Bps = float(text[:-4]) * 1e9
        elif text.endswith("MB/S"):
            bandwidth_Bps = float(text[:-4]) * 1e6
        elif text.endswith("KB/S"):
            bandwidth_Bps = float(text[:-4]) * 1e3
        elif text.endswith("B/S"):
            bandwidth_Bps = float(text[:-3])
        else:
            raise ValueError(
                f"Unsupported bandwidth format: {value}. Expected B/s, KB/s, MB/s, or GB/s."
            )
    if bandwidth_Bps <= 0:
        raise ValueError(f"bandwidth must be positive, got {value}.")
    return bandwidth_Bps * 8.0 / 1e9


def _size_to_text(size_bytes: int) -> str:
    if size_bytes % (1024**2) == 0:
        return f"{size_bytes // (1024**2)}MB"
    if size_bytes % 1024 == 0:
        return f"{size_bytes // 1024}KB"
    return f"{size_bytes}B"


def _format_decimal(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _format_freq_mhz(freq_mhz: float) -> str:
    return f"{_format_decimal(freq_mhz)}Mhz"


def _round_cost(value: float) -> float:
    return float(f"{float(value):.3f}")


def _round_area(value: float) -> float:
    return float(f"{float(value):.6f}")


def _f(row: dict[str, Any], key: str) -> float:
    raw = row.get(key)
    if raw is None or raw == "":
        raise ValueError(f"CSV row is missing required field '{key}'.")
    return float(raw)


def _iv(row: dict[str, Any], key: str) -> int:
    raw = row.get(key)
    if raw is None or raw == "":
        raise ValueError(f"CSV row is missing required field '{key}'.")
    return int(float(raw))


def _require_positive_constant(name: str, value: float | int | None) -> float:
    if value is None:
        raise ValueError(f"{name} is unset.")
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return numeric


def _supported_row_tuples(rows: list[dict[str, Any]]) -> str:
    tuples = sorted(
        {
            (
                _iv(row, "n_horizontal_mat"),
                _iv(row, "banks"),
                _iv(row, "pim_n_cols_per_vector"),
            )
            for row in rows
            if _iv(row, "mat_width") == DRAM_MAT_WIDTH
            and _iv(row, "mat_height") == DRAM_MAT_HEIGHT
            and _iv(row, "pim_col_bytes") == COL_WIDTH_BYTES
            and _iv(row, "io_width") == COL_WIDTH_BYTES
        }
    )
    if not tuples:
        return "<none>"
    return ", ".join(
        f"(h_mat={h_mat}, banks={banks}, burst_len={burst_len})"
        for h_mat, banks, burst_len in tuples
    )


def _dram_rows(arch_type: str) -> list[dict[str, Any]]:
    try:
        csv_path = DRAM_RESULTS_BY_ARCH[arch_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported architecture '{arch_type}'.") from exc
    return _load_csv(csv_path)


def _row_implied_bank_bytes(row: dict[str, Any]) -> int:
    if "page_size_bits" in row and row["page_size_bits"] != "" and "ndbl" in row and row["ndbl"] != "":
        row_bytes = int(round(_f(row, "page_size_bits") / 8.0))
        return row_bytes * _iv(row, "mat_height") * _iv(row, "ndbl")
    banks = _iv(row, "banks")
    return int(round((TOTAL_CAPACITY_Mb / float(banks)) * (1024**2) / 8.0))


def _row_implied_vmat(row: dict[str, Any]) -> int:
    if "ndbl" in row and row["ndbl"] != "":
        return _iv(row, "ndbl")
    bank_bits = _row_implied_bank_bytes(row) * 8
    denominator = (
        _iv(row, "mat_width") * _iv(row, "mat_height") * _iv(row, "n_horizontal_mat")
    )
    if denominator <= 0 or bank_bits % denominator != 0:
        raise ValueError(f"DRAM row '{row.get('run', '<unknown>')}' does not map cleanly to v_mat.")
    return bank_bits // denominator


def _row_write_energy_per_bit(row: dict[str, Any]) -> float:
    if "E_WRITE_pJ_per_bit" in row and row["E_WRITE_pJ_per_bit"] != "":
        return float(row["E_WRITE_pJ_per_bit"])
    return float(row["E_READ_pJ_per_bit"])


def _row_area_mm2(row: dict[str, Any]) -> float | None:
    for key in ("area_core_mm2", "total_area_mm2", "area_die_mm2"):
        if key not in row:
            continue
        raw = row[key]
        if raw is None or raw == "" or str(raw).strip().upper() == "NA":
            return None
        return float(raw)
    raise ValueError(f"DRAM row '{row.get('run', '<unknown>')}' has no supported area column.")


def _select_dram_row(
    arch_type: str,
    *,
    h_mat: int,
    total_banks: int,
    burst_len: int,
    expected_v_mat: int,
    expected_row_bytes: int,
    expected_bank_bytes: int,
) -> dict[str, Any]:
    rows = _dram_rows(arch_type)
    matches = [
        row
        for row in rows
        if _iv(row, "mat_width") == DRAM_MAT_WIDTH
        and _iv(row, "mat_height") == DRAM_MAT_HEIGHT
        and _iv(row, "n_horizontal_mat") == h_mat
        and _iv(row, "banks") == total_banks
        and _iv(row, "pim_col_bytes") == COL_WIDTH_BYTES
        and _iv(row, "io_width") == COL_WIDTH_BYTES
        and _iv(row, "pim_n_cols_per_vector") == burst_len
        and _row_implied_vmat(row) == expected_v_mat
    ]
    if len(matches) != 1:
        raise ValueError(
            "No exact DRAM row for "
            f"arch={arch_type}, h_mat={h_mat}, banks={total_banks}, burst_len={burst_len}, "
            f"col_width_bytes={COL_WIDTH_BYTES}. "
            f"Available exact-match tuples: {_supported_row_tuples(rows)}"
        )

    row = matches[0]
    row_page_bytes = int(round(_f(row, "page_size_bits") / 8.0))
    if row_page_bytes != expected_row_bytes:
        raise ValueError(
            f"Matched DRAM row '{row['run']}' implies row_bytes={row_page_bytes}, "
            f"expected {expected_row_bytes}."
        )
    implied_vmat = _row_implied_vmat(row)
    if implied_vmat != expected_v_mat:
        raise ValueError(
            f"Matched DRAM row '{row['run']}' implies v_mat={implied_vmat}, "
            f"expected {expected_v_mat}."
        )
    implied_bank_bytes = _row_implied_bank_bytes(row)
    if implied_bank_bytes != expected_bank_bytes:
        raise ValueError(
            f"Matched DRAM row '{row['run']}' implies bank_bytes={implied_bank_bytes}, "
            f"expected {expected_bank_bytes}."
        )
    if _iv(row, "pim_col_bytes") != _iv(row, "io_width"):
        raise ValueError(
            f"Matched DRAM row '{row['run']}' has pim_col_bytes != io_width."
        )
    return row


def _select_sram_row(buffer_size_bytes: int, io_width_bits: int) -> dict[str, Any]:
    rows = _load_csv(SRAM_RESULTS_PATH)

    def _sram_row_width_bits(row: dict[str, Any]) -> int:
        if "io_width_bit" in row and row["io_width_bit"] != "":
            return _iv(row, "io_width_bit")
        if "cacheline_bits" in row and row["cacheline_bits"] != "":
            return _iv(row, "cacheline_bits")
        raise ValueError(f"SRAM row is missing width column: {row}")

    def _sram_row_size_bytes(row: dict[str, Any]) -> int:
        size_kb = _f(row, "total_size_KB")
        return int(round(size_kb * 1024.0))

    width_matches = [
        row
        for row in rows
        if _sram_row_width_bits(row) == io_width_bits
    ]
    if not width_matches:
        available = sorted({_sram_row_width_bits(row) for row in rows})
        raise ValueError(
            f"No SRAM rows for io_width_bit={io_width_bits}. Available widths: {available}"
        )

    exact_matches = [
        row for row in width_matches if _sram_row_size_bytes(row) == buffer_size_bytes
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    return min(
        width_matches,
        key=lambda row: (
            abs(_sram_row_size_bytes(row) - buffer_size_bytes),
            _sram_row_size_bytes(row),
        ),
    )


def _select_pu_row(simd_lanes: int, tck_ns: float) -> dict[str, Any]:
    rows = _load_csv(PU_RESULTS_PATH)
    matches = [
        row
        for row in rows
        if _iv(row, "simd_lanes") == simd_lanes
        and _iv(row, "io_width_bit") == (simd_lanes * 16)
        and math.isclose(_f(row, "tck_ns"), tck_ns)
    ]
    if len(matches) != 1:
        available = sorted(
            {
                (_iv(row, "simd_lanes"), _f(row, "tck_ns"))
                for row in rows
            }
        )
        raise ValueError(
            f"No exact PU row for simd_lanes={simd_lanes}, tck_ns={tck_ns}. "
            f"Available pairs: {available}"
        )
    return matches[0]


def _cycles_from_ns(latency_ns: float, cycle_ns: float) -> int:
    if latency_ns <= 0:
        return 0
    return int(math.ceil(latency_ns / cycle_ns))


def _cost(latency_ns: float, energy_pj: float, cycle_ns: float) -> dict[str, float]:
    if energy_pj < 0:
        raise ValueError(f"Derived energy must be non-negative, got {energy_pj}.")
    return {
        "latency": _cycles_from_ns(latency_ns, cycle_ns),
        "energy": _round_cost(energy_pj),
    }


def _iter_power_of_two_candidates(limit: int) -> list[int]:
    values: list[int] = []
    candidate = 1
    while candidate <= limit:
        values.append(candidate)
        candidate *= 2
    return values


def _wire_len_mm(arch_type: str, interconnect: str) -> float:
    try:
        return WIRE_LEN_MM_BY_ARCH_AND_INTERCONNECT[arch_type][interconnect]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported interconnect '{interconnect}' for architecture '{arch_type}'."
        ) from exc


def _select_n_lines(
    *,
    arch_type: str,
    interconnect: str,
    burst_size_bits: int,
    t_bank_to_pu_load_ns: float,
    t_row_change_ns: float,
    n_pus_unit: float,
) -> tuple[int, float, float]:
    line_width_um = _require_positive_constant("LINE_WIDTH_UM", LINE_WIDTH_UM)
    wire_len_mm = _wire_len_mm(arch_type, interconnect)
    best_n_lines: int | None = None
    best_area_mm2 = 0.0
    best_latency_ns = 0.0
    best_objective: float | None = None

    for n_lines in _iter_power_of_two_candidates(burst_size_bits):
        extra_area_mm2 = n_pus_unit * (n_lines * line_width_um * wire_len_mm) / 1e3
        latency_ns = (
            t_bank_to_pu_load_ns
            + t_row_change_ns
            + (float(burst_size_bits) / float(n_lines)) * DELAY_PER_MM_NS * wire_len_mm
        )
        objective = extra_area_mm2 * latency_ns
        if (
            best_objective is None
            or objective < best_objective
            or (
                math.isclose(objective, best_objective)
                and best_n_lines is not None
                and n_lines < best_n_lines
            )
        ):
            best_n_lines = n_lines
            best_area_mm2 = extra_area_mm2
            best_latency_ns = latency_ns
            best_objective = objective

    if best_n_lines is None:
        raise ValueError("Failed to choose n_lines.")
    return best_n_lines, best_area_mm2, best_latency_ns


def _build_unit_costs_and_area(
    *,
    arch_type: str,
    interconnect: str,
    dram_row: dict[str, Any],
    sram_row: dict[str, Any],
    pu_row: dict[str, Any],
    burst_len: int,
    burst_size_bits: int,
    cycle_ns: float,
    host_pu_bw_bits_per_ns: float,
    num_pus_unit: float,
) -> tuple[dict[str, dict[str, float]], float | None, float | None, int | None]:
    t_ccd_l_ns = _f(dram_row, "tCCD_L_ns")
    t_rf_access_ns = _f(sram_row, "access_time_ns")
    t_rp_ns = _f(dram_row, "tRP_ns")
    t_rcd_ns = _f(dram_row, "tRCD_ns")
    t_host_read_ns = _f(dram_row, "pim_host_read_row_conflict_ns")
    t_host_write_ns = _f(dram_row, "pim_host_write_row_conflict_ns")
    t_row_change_ns = t_rp_ns + t_rcd_ns

    t_bank_to_pu_load_ns = ((burst_len - 1) * t_ccd_l_ns) + t_rf_access_ns
    t_pu_to_bank_store_ns = t_row_change_ns + ((burst_len - 1) * t_ccd_l_ns) + t_rf_access_ns
    # host_pu_transfer is charged by the analytical model once per 64-bit column
    # (num_cols in _stream_metric / transfer_units_per_load in _save_metric), not once
    # per burst transaction, so the burst-transaction cost is amortized over the
    # burst_len columns it carries.
    t_host_pu_transfer_ns = (
        (float(burst_size_bits) / host_pu_bw_bits_per_ns) + t_rf_access_ns
    ) / float(burst_len)
    t_host_bank_row_conflict_ns = t_host_read_ns
    t_pu_compute_ns = 2.0 * t_rf_access_ns

    chosen_n_lines: int | None = None
    extra_area_mm2 = 0.0
    if interconnect == "host":
        t_bank_to_bank_transfer_ns = t_host_read_ns + t_host_write_ns + t_bank_to_pu_load_ns
    else:
        chosen_n_lines, extra_area_mm2, t_bank_to_bank_transfer_ns = _select_n_lines(
            arch_type=arch_type,
            interconnect=interconnect,
            burst_size_bits=burst_size_bits,
            t_bank_to_pu_load_ns=t_bank_to_pu_load_ns,
            t_row_change_ns=t_row_change_ns,
            n_pus_unit=num_pus_unit,
        )

    e_bank_read_issue_pj_per_bit = _f(dram_row, "E_bank_read_issue_pJ_per_bit")
    e_rf_access_pj_per_bit = _f(sram_row, "read_energy_pJ_per_bit")
    e_pre_pj = _f(dram_row, "E_PRE_pJ")
    e_act_pj = _f(dram_row, "E_ACT_pJ")
    e_write_pj_per_bit = _row_write_energy_per_bit(dram_row)
    e_bank_sink_pj_per_bit = _f(dram_row, "E_bank_sink_energy_pJ_per_bit")
    e_pu_pj = _f(pu_row, "energy_pj")

    e_row_change_pj = e_pre_pj + e_act_pj
    e_bank_to_pu_load_pj = burst_size_bits * (e_bank_read_issue_pj_per_bit + e_rf_access_pj_per_bit)
    e_host_bank_row_conflict_pj = e_row_change_pj
    e_pu_to_bank_store_pj = e_bank_to_pu_load_pj + e_host_bank_row_conflict_pj
    e_host_pu_transfer_pj = max(
        0.0,
        (e_write_pj_per_bit - e_bank_sink_pj_per_bit + e_rf_access_pj_per_bit)
        * burst_size_bits,
    ) / float(burst_len)
    e_pu_compute_pj = (2.0 * e_rf_access_pj_per_bit * burst_size_bits) + e_pu_pj

    if interconnect == "host":
        e_bank_to_bank_transfer_pj = (
            (2.0 * e_write_pj_per_bit) + e_rf_access_pj_per_bit
        ) * burst_size_bits
    else:
        wire_len_mm = _wire_len_mm(arch_type, interconnect)
        e_bank_to_bank_transfer_pj = (
            e_bank_to_pu_load_pj
            + e_row_change_pj
            + wire_len_mm * ENERGY_PER_MM_PJ * burst_size_bits
        )

    a_dram_mm2 = _row_area_mm2(dram_row)
    a_pu_mm2 = (_f(pu_row, "area_um2") / 1e6) * DRAM_PROCESS_PU_PENALTY_BY_ARCH.get(arch_type, 1.0)
    a_rf_mm2 = _f(sram_row, "data_array_area_mm2") * DRAM_PROCESS_SRAM_PENALTY_BY_ARCH.get(arch_type, 1.0)
    num_banks_unit = float(_iv(dram_row, "banks"))
    total_area_mm2 = None
    pim_overhead = None
    if a_dram_mm2 is not None:
        pim_area_mm2 = (
            (num_pus_unit * a_pu_mm2)
            + (num_banks_unit * a_rf_mm2)
            + extra_area_mm2
        )
        total_area_mm2 = a_dram_mm2 + pim_area_mm2
        # PIM logic overhead relative to ONE DRAM die: (interconnect + SRAM + PU) / dram_area.
        # The arch models one pseudo-channel of PIM logic but the DRAM die (denominator)
        # spans PSEUDO_CHANNELS_PER_DIE of them, so scale the numerator up to the full die
        # (upmem = x1; hbm_pim = x4). Note: total_area_mm2 above is left at the modeled
        # one-pseudo-channel logic, so pim_overhead != (total_area - dram)/dram for hbm_pim.
        if a_dram_mm2 > 0:
            pch_per_die = PSEUDO_CHANNELS_PER_DIE_BY_ARCH.get(arch_type, 1)
            pim_overhead = (pim_area_mm2 * pch_per_die) / a_dram_mm2

    unit_costs = {
        "bank_to_pu_load": _cost(t_bank_to_pu_load_ns, e_bank_to_pu_load_pj, cycle_ns),
        "pu_to_bank_store": _cost(t_pu_to_bank_store_ns, e_pu_to_bank_store_pj, cycle_ns),
        "host_pu_transfer": _cost(t_host_pu_transfer_ns, e_host_pu_transfer_pj, cycle_ns),
        "bank_to_bank_transfer": _cost(
            t_bank_to_bank_transfer_ns, e_bank_to_bank_transfer_pj, cycle_ns
        ),
        "host_bank_row_conflict": _cost(
            t_host_bank_row_conflict_ns, e_host_bank_row_conflict_pj, cycle_ns
        ),
        "pu_compute": _cost(t_pu_compute_ns, e_pu_compute_pj, cycle_ns),
        "row_change": _cost(t_row_change_ns, e_row_change_pj, cycle_ns),
    }
    return (
        unit_costs,
        _round_area(total_area_mm2) if total_area_mm2 is not None else None,
        round(pim_overhead, 6) if pim_overhead is not None else None,
        chosen_n_lines,
    )


def _header_lines(
    *,
    template_path: Path,
    burst_len: int,
    buffer_size: str | int | float,
    h_mat: int,
    v_mat: int,
    num_pu: int,
    interconnect: str,
) -> str:
    # Emit the template reference repo-relative so generated headers are
    # machine-independent (builders may pass absolute paths).
    resolved_template = template_path.resolve()
    try:
        template_label = resolved_template.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        template_label = template_path.as_posix()
    return "\n".join(
        [
            f"# Generated from template: {template_label}",
            (
                "# Inputs: "
                f"burst_len={burst_len}, "
                f"buffer_size={buffer_size}, "
                f"h_mat={h_mat}, "
                f"v_mat={v_mat}, "
                f"num_pu={num_pu}, "
                f"interconnect={interconnect}"
            ),
            "# Latencies are stored in PU cycles; energies are stored in pJ.",
            "",
        ]
    )


def _write_yaml(
    *,
    path: Path,
    payload: dict[str, Any],
    template_path: Path,
    burst_len: int,
    buffer_size: str | int | float,
    h_mat: int,
    v_mat: int,
    num_pu: int,
    interconnect: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    path.write_text(
        _header_lines(
            template_path=template_path,
            burst_len=burst_len,
            buffer_size=buffer_size,
            h_mat=h_mat,
            v_mat=v_mat,
            num_pu=num_pu,
            interconnect=interconnect,
        )
        + body,
        encoding="utf-8",
    )


def _resolve_cli_defaults(args: argparse.Namespace) -> argparse.Namespace:
    template_payload = _load_yaml(args.template)
    arch_type = str(template_payload.get("name", "")).strip()
    if arch_type not in CLI_ARCH_DEFAULTS:
        raise ValueError(
            f"Unsupported template architecture '{arch_type}'. Expected one of: "
            f"{', '.join(sorted(CLI_ARCH_DEFAULTS))}."
        )

    resolved = argparse.Namespace(**vars(args))
    for key, value in CLI_COMMON_DEFAULTS.items():
        if getattr(resolved, key) is None:
            setattr(resolved, key, value)
    for key, value in CLI_ARCH_DEFAULTS[arch_type].items():
        if getattr(resolved, key) is None:
            setattr(resolved, key, value)
    return resolved


def generate_arch_file(
    template_path: Path,
    output_path: Path,
    burst_len: int,
    buffer_size: str | int | float,
    h_mat: int,
    v_mat: int,
    num_pu: int,
    interconnect: str,
) -> Path:
    if burst_len <= 0:
        raise ValueError(f"burst_len must be > 0, got {burst_len}.")
    if h_mat <= 0 or v_mat <= 0 or num_pu <= 0:
        raise ValueError("h_mat, v_mat, and num_pu must all be > 0.")
    if interconnect not in {"host", "inter-bg", "intra-bg"}:
        raise ValueError(
            f"Unsupported interconnect '{interconnect}'. Expected one of host, inter-bg, intra-bg."
        )

    template_payload = _load_yaml(template_path)
    arch_type = str(template_payload.get("name", "")).strip()
    if arch_type not in DRAM_RESULTS_BY_ARCH:
        raise ValueError(
            f"Unsupported template architecture '{arch_type}'. Expected one of: "
            f"{', '.join(sorted(DRAM_RESULTS_BY_ARCH))}."
        )

    buffer_size_bytes = _parse_buffer_size_bytes(buffer_size)
    burst_size_bits = COL_WIDTH_BITS * int(burst_len)
    if burst_size_bits % 16 != 0:
        raise ValueError(
            f"burst_size_bits={burst_size_bits} must be divisible by 16 to derive simd_lanes."
        )
    simd_lanes = burst_size_bits // 16
    row_bytes = (DRAM_MAT_WIDTH * h_mat) // 8
    bank_bytes = row_bytes * DRAM_MAT_HEIGHT * v_mat
    pu_frequency_mhz = PU_FREQUENCY_MHZ_BY_ARCH[arch_type]
    cycle_ns = 1000.0 / pu_frequency_mhz

    specs = template_payload["specs"]
    banks_per_pu = int(specs["banks_per_pu"])
    total_banks = int(num_pu) * banks_per_pu

    host_pu_bw_bits_per_ns = _parse_bandwidth_bits_per_ns(
        template_payload["bandwidth"]["host_device"]
    )
    dram_row = _select_dram_row(
        arch_type,
        h_mat=h_mat,
        total_banks=total_banks,
        burst_len=burst_len,
        expected_v_mat=v_mat,
        expected_row_bytes=row_bytes,
        expected_bank_bytes=bank_bytes,
    )
    sram_row = _select_sram_row(buffer_size_bytes, burst_size_bits)
    pu_row = _select_pu_row(simd_lanes, PU_TCK_NS_BY_ARCH[arch_type])
    num_pus_unit = float(_iv(dram_row, "banks")) / float(banks_per_pu)
    if not math.isclose(num_pus_unit, float(num_pu)):
        raise ValueError(
            f"Matched DRAM row '{dram_row['run']}' implies num_pu={num_pus_unit}, expected {num_pu}."
        )

    payload = deepcopy(template_payload)
    payload_specs = payload["specs"]
    payload_hierarchy = payload["hierarchy"]
    payload_specs["row_bytes"] = _size_to_text(row_bytes)
    payload_specs["bank_bytes"] = _size_to_text(bank_bytes)
    payload_specs["cache_bytes"] = _size_to_text(buffer_size_bytes)
    payload_specs["simd_lanes"] = simd_lanes
    payload_specs["pu_frequency"] = _format_freq_mhz(pu_frequency_mhz)
    payload_specs.pop("n_lines", None)
    payload_hierarchy["L2"]["count"] = int(num_pu)

    cache_block = payload.get("cache")
    if isinstance(cache_block, dict) and str(cache_block.get("mode", "")).strip().lower() == "regfile":
        cache_block["vector_width_bits"] = burst_size_bits

    unit_costs, total_area_mm2, pim_overhead, chosen_n_lines = _build_unit_costs_and_area(
        arch_type=arch_type,
        interconnect=interconnect,
        dram_row=dram_row,
        sram_row=sram_row,
        pu_row=pu_row,
        burst_len=burst_len,
        burst_size_bits=burst_size_bits,
        cycle_ns=cycle_ns,
        host_pu_bw_bits_per_ns=host_pu_bw_bits_per_ns,
        num_pus_unit=num_pus_unit,
    )
    if chosen_n_lines is not None:
        payload_specs["n_lines"] = int(chosen_n_lines)

    payload["unit_costs"] = unit_costs
    payload["total_area_mm2"] = total_area_mm2 if total_area_mm2 is not None else "NA"
    payload["pim_overhead"] = pim_overhead if pim_overhead is not None else "NA"

    _write_yaml(
        path=output_path,
        payload=payload,
        template_path=template_path,
        burst_len=burst_len,
        buffer_size=buffer_size,
        h_mat=h_mat,
        v_mat=v_mat,
        num_pu=num_pu,
        interconnect=interconnect,
    )
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one architecture YAML from a template and exact sweep-data matches.",
        epilog=(
            "CLI defaults are preset inputs, not guaranteed-valid exact matches under the current "
            "sweep tables. Defaults: burst_len=4, h_mat=16, num_pu=8, interconnect=host, "
            "v_mat=32 and buffer_size=512B for hbm_pim, v_mat=64 and buffer_size=32KB for upmem."
        ),
    )
    parser.add_argument("--template", type=Path, required=True, help="Template architecture YAML path.")
    parser.add_argument("--output", type=Path, required=True, help="Output architecture YAML path.")
    parser.add_argument(
        "--burst-len",
        type=int,
        help="Burst length in DRAM columns. Defaults to 4 when omitted.",
    )
    parser.add_argument(
        "--buffer-size",
        help=(
            "Buffer size in bytes or as a size string ending in B or KB. "
            "Defaults to 512B for hbm_pim and 32KB for upmem when omitted."
        ),
    )
    parser.add_argument(
        "--h-mat",
        type=int,
        help="Horizontal mat count. Defaults to 16 when omitted.",
    )
    parser.add_argument(
        "--v-mat",
        type=int,
        help="Vertical mat count. Defaults to 32 for hbm_pim and 64 for upmem when omitted.",
    )
    parser.add_argument(
        "--num-pu",
        type=int,
        help="Number of PUs (hierarchy.L2.count). Defaults to 8 when omitted.",
    )
    parser.add_argument(
        "--interconnect",
        choices=("host", "inter-bg", "intra-bg"),
        help="Interconnect mode for bank-to-bank transfer modeling. Defaults to host when omitted.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _resolve_cli_defaults(_parse_args())
    written = generate_arch_file(
        template_path=args.template,
        output_path=args.output,
        burst_len=args.burst_len,
        buffer_size=args.buffer_size,
        h_mat=args.h_mat,
        v_mat=args.v_mat,
        num_pu=args.num_pu,
        interconnect=args.interconnect,
    )
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
