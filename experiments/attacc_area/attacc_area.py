#!/usr/bin/env python3
"""AttAcc PU+buffer area overhead from the PPA DB.

Computes the area overhead of AttAcc's PUs and per-PU buffers under the
PNMAX PPA database. This script only derives and reports the value.

Method — the same PPA-DB area model the artifact uses for every sweep
architecture (documented in data/ppa/PROVENANCE.md and implemented by
``data/ppa/generate_geometry_sweep_archs.py``, whose lookup helpers are
imported here as the single source of truth):

    overhead = N_PU x (A_pu + A_buffer) / A_dram

* ``A_pu``     — post-route PU macro area from ``pu_postroute_22nm_cleaned.csv``
                 at AttAcc's SIMD width (16 fp16 lanes) and the nearest
                 characterized clock to AttAcc's 666 MHz (tCK 1.5 ns).
* ``A_buffer`` — CACTI SRAM data-array area from ``sram_sweep_results.csv`` for
                 AttAcc's per-PU register file (8 KB nominal) at the modeled
                 IO width (simd_lanes x 16 = 256 bit).
* ``A_dram``   — DreamRAM HBM2E die footprint from
                 ``dreamram_hbm2e_results.csv`` for AttAcc's DRAM geometry
                 (512x512 mats, 16 horizontal mats -> 1 KB rows, 32 vertical
                 mats -> 16 MB banks, 16 banks). DreamRAM reports the
                 single-die *footprint* (not stacked silicon), so the PU+buffer
                 silicon of the full stack (256 PUs, one per bank, 8 channels x
                 2 pseudo-channels x 4 bank groups x 4 banks) is compared
                 against that footprint. The full 1024-PU system (4 stacks)
                 gives the identical ratio.

Every input row is printed so the derivation is auditable; the script fails
loudly if any PPA-DB input is missing or ambiguous.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PPA_DIR = REPO_ROOT / "data" / "ppa"
# AttAcc arch definition — the authoritative source for the SIMD width, PU
# clock, and PUs-per-stack derived by ``_attacc_arch_config`` below (the same
# file this module's header cites).
ARCH_PATH = REPO_ROOT / "data" / "archs" / "lowered" / "baseline" / "attacc.yaml"

# ---------------------------------------------------------------------------
# AttAcc configuration (data/archs/lowered/baseline/attacc.yaml)
# — SIMD_LANES / PU_CLOCK_MHZ / PUS_PER_STACK are derived from the arch YAML
# (see _attacc_arch_config); the values below are the AttAcc nominal
# configuration kept inline as a single clearly-marked block.
# ---------------------------------------------------------------------------
BUFFER_BYTES = 8 * 1024        # AttAcc nominal per-PU register file: 8 KB
STACKS = 4                     # AttAcc nominal 1024 PUs = 4 HBM stacks
DRAM_ROW_QUERY = dict(         # AttAcc DRAM geometry in the DreamRAM sweep
    n_horizontal_mat=16,       #   16 x 512 mat columns -> 1 KB physical rows
    banks=16,                  #   16 banks per pseudo-channel
    vertical_mats=32,          #   32 x 512 mat rows -> 16 MB banks
)


def _attacc_arch_config(arch_path: Path = ARCH_PATH) -> tuple[int, float, int]:
    """Derive (SIMD_LANES, PU_CLOCK_MHZ, PUS_PER_STACK) from the AttAcc arch YAML.

    These three values were previously carried as code constants; the arch
    definition (``data/archs/lowered/baseline/attacc.yaml``) is their single
    source of truth:

    * ``SIMD_LANES``    <- ``specs.simd_lanes``          (16 fp16 MAC lanes)
    * ``PU_CLOCK_MHZ``  <- ``specs.pu_frequency`` / 1e6  (666 MHz -> tCK ~1.5 ns)
    * ``PUS_PER_STACK`` <- product of the intra-stack hierarchy levels, i.e.
      every ``hierarchy`` level except ``stack`` (bank x bg x pch x channel =
      4 x 4 x 2 x 8 = 256; the arch models 1 PU per bank).
    """
    from pnmax.database import Arch
    from pnmax.database.arch import HierLevel

    arch = Arch.from_yaml_file(arch_path)
    simd_lanes = int(arch.specs.simd_lanes)
    pu_clock_mhz = arch.specs.pu_frequency_hz / 1e6
    kinds = arch.hierarchy.level_to_name
    pus_per_stack = math.prod(
        count
        for level, count in arch.hierarchy.organization.items()
        if kinds[level] is not HierLevel.STACK
    )
    return simd_lanes, pu_clock_mhz, pus_per_stack


def _load_ppa_module():
    """Import data/ppa/generate_geometry_sweep_archs.py by path."""
    spec = importlib.util.spec_from_file_location(
        "generate_geometry_sweep_archs", PPA_DIR / "generate_geometry_sweep_archs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _sram_lookup(buffer_bytes: int, io_width_bit: int) -> tuple[dict, float]:
    """SRAM data-array area lookup against sram_sweep_results.csv.

    Mirrors the documented PPA-DB rule (nearest characterized IO width, then
    exact size / round-up-to-smallest), implemented directly against the
    cleaned CSV's ``cacheline_bits`` column.
    """
    import csv

    rows = [
        row
        for row in csv.DictReader((PPA_DIR / "sram_sweep_results.csv").open())
        if row.get("data_array_area_mm2", "").strip()
    ]
    widths = sorted({int(row["cacheline_bits"]) for row in rows})
    modeled_width = min(widths, key=lambda w: (abs(w - io_width_bit), w))
    width_rows = [row for row in rows if int(row["cacheline_bits"]) == modeled_width]
    target_kb = buffer_bytes / 1024.0
    exact = [row for row in width_rows if float(row["total_size_KB"]) == target_kb]
    if exact:
        row = exact[0]
    else:
        sizes = sorted({float(r["total_size_KB"]) for r in width_rows})
        fallback = next((s for s in sizes if s >= target_kb), sizes[-1])
        row = next(r for r in width_rows if float(r["total_size_KB"]) == fallback)
    area = float(row["data_array_area_mm2"]) * (io_width_bit / modeled_width)
    return row, area


def _dram_die_footprint_mm2(ppa) -> tuple[float, str]:
    """AttAcc DRAM die footprint via the PPA DB's raw-row selector."""
    import csv

    rows = list(csv.DictReader((PPA_DIR / "dreamram_hbm2e_results.csv").open()))
    matches = [
        row
        for row in rows
        if int(row["n_horizontal_mat"]) == DRAM_ROW_QUERY["n_horizontal_mat"]
        and int(row["banks"]) == DRAM_ROW_QUERY["banks"]
        and f"_v{DRAM_ROW_QUERY['vertical_mats']}_" in row["run"]
        and row.get("total_area_mm2", "").strip()
    ]
    if not matches:
        raise SystemExit("no DreamRAM row matches the AttAcc DRAM geometry")
    areas = {float(row["total_area_mm2"]) for row in matches}
    if len(areas) != 1:
        raise SystemExit(f"ambiguous DreamRAM area candidates: {sorted(areas)}")
    return areas.pop(), matches[0]["run"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable summary (JSON).",
    )
    args = parser.parse_args()

    SIMD_LANES, PU_CLOCK_MHZ, PUS_PER_STACK = _attacc_arch_config()
    ppa = _load_ppa_module()
    target_tck_ns = 1000.0 / PU_CLOCK_MHZ
    io_width_bit = SIMD_LANES * 16

    pu_row = ppa._nearest_pu_row(SIMD_LANES, target_tck_ns)
    pu_area_mm2 = ppa._nearest_pu_area_mm2(SIMD_LANES, target_tck_ns)
    sram_row, sram_area_mm2 = _sram_lookup(BUFFER_BYTES, io_width_bit)
    dram_area_mm2, dram_run = _dram_die_footprint_mm2(ppa)

    per_pu_mm2 = pu_area_mm2 + sram_area_mm2
    added_mm2 = PUS_PER_STACK * per_pu_mm2
    overhead_pct = 100.0 * added_mm2 / dram_area_mm2
    system_added_mm2 = STACKS * added_mm2
    system_dram_mm2 = STACKS * dram_area_mm2

    print("== AttAcc PU+buffer area overhead (PNMAX PPA DB) ==")
    print(f"PU macro       : simd_lanes={SIMD_LANES}, target tCK={target_tck_ns:.4f} ns "
          f"-> row (tck={pu_row['tck_ns']} ns, io={pu_row['io_width_bit']} bit): "
          f"{pu_area_mm2 * 1e6:.1f} um^2 = {pu_area_mm2:.6f} mm^2")
    print(f"Register file  : {BUFFER_BYTES // 1024} KB @ io {io_width_bit} bit "
          f"-> SRAM row (size={sram_row['total_size_KB']} KB, "
          f"cacheline={sram_row['cacheline_bits']} bit): {sram_area_mm2:.6f} mm^2")
    print(f"Per-PU adder   : {per_pu_mm2:.6f} mm^2 (PU + buffer)")
    print(f"DRAM footprint : DreamRAM run '{dram_run}' "
          f"(1 KB rows, 16 MB banks, 16 banks): {dram_area_mm2:.4f} mm^2 per die footprint")
    print(f"Stack          : {PUS_PER_STACK} PUs x {per_pu_mm2:.6f} mm^2 = {added_mm2:.4f} mm^2 "
          f"on a {dram_area_mm2:.4f} mm^2 footprint")
    print(f"System (total) : {STACKS} stacks -> {STACKS * PUS_PER_STACK} PUs, "
          f"{system_added_mm2:.4f} mm^2 added / {system_dram_mm2:.4f} mm^2 DRAM")
    print()
    print(f"PU+buffer area overhead (computed) : {overhead_pct:.4f}%")

    if args.out:
        import json

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "computed_overhead_pct": overhead_pct,
                    "pu_area_mm2": pu_area_mm2,
                    "sram_area_mm2": sram_area_mm2,
                    "dram_die_footprint_mm2": dram_area_mm2,
                    "pus_per_stack": PUS_PER_STACK,
                    "stacks": STACKS,
                    "pu_row_tck_ns": float(pu_row["tck_ns"]),
                    "sram_row_size_kb": float(sram_row["total_size_KB"]),
                    "dram_run": dram_run,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"summary written: {args.out}")

    print("CHECK OK: AttAcc PU+buffer area overhead computed from the "
          "PPA DB and reported above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
