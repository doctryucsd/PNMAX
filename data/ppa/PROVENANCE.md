# PPA Characterization Data (the PPA DB)

This folder is PNMAX's per-component PPA database: the characterization
results and generators from which every derived architecture YAML under
`data/archs/lowered/` is built. Under the standard flow, the
DRAM lookup CSVs and all builder-generated arch families are **derived
outputs** — `setup.sh` generates them from the vendored tools (stage
script: `data/ppa/generate_ppa.sh`), and they are not tracked in the
repository. Two characterization CSVs and the hand-authored/frozen arch sets
ship as-is (see "Shipped as-is" below).

## Generated (the standard flow)

`./data/ppa/generate_ppa.sh` (called by `setup.sh` Step 5; independently
re-runnable, ~seconds for the CSVs plus a few seconds for the builders):

1. Builds the CACTI binary under `external/cacti`.
2. `generate_dram_lookup_csvs.py` runs **CACTI** (DDR4 / upmem rows) and
   **DreamRAM** (HBM2E / hbm_pim rows) over the full lookup-tuple plan —
   the generator's variant/PU-sweep/OFAT axes plus the iso-capacity combos
   imported from `build_iso_capacity_archs.PLANS` — and distills:
   - `cacti_ddr4_results.csv` (29 rows)
   - `dreamram_hbm2e_results.csv` (21 rows; one documented blank
     `total_area_mm2` cell for a geometry-mismatched DreamRAM run, warned at
     parse time)
   Per-run tool configs and raw outputs are written under
   `lookup_work/dram/` (also derived, rebuilt every run). A
   silent-blank guard aborts the run if any planned cell comes out empty.
   The tools live under `external/cacti` and `external/dreamram`
   (overridable via `PNMAX_CACTI_DIR` / `PNMAX_DREAMRAM_DIR`).
   Recorded characterization inputs are restored at run time: the CACTI
   template carries the characterization-era technology/frequency values
   (0.022 µm / 400 MHz), and the driver pins DreamRAM's timing calibration
   (`calibration.tck = 2.5`, a recorded input value not present in the
   public upstream configs) via a driver-owned baseline copy per run — the
   vendored `external/dreamram` tree is never modified.
3. The five arch builders rebuild the derived families from the fresh CSVs:
   - `generate_arch_from_template_sweep.sh` →
     `data/archs/lowered/geometry_sweep/{upmem,hbm_pim}` (the evaluated
     geometry-variant sets: upmem variants 1–9, hbm_pim variants 1–4,7–9)
   - `build_iso_capacity_archs.py` →
     `data/archs/lowered/geometry_sweep/iso_capacity_archs` (27 files)
   - `build_buffer_archs.py` → `data/archs/lowered/buffer_sweep` (12 files)
   - `build_interconnect_archs.py` →
     `data/archs/lowered/interconnect_sweep` (8 files)
   - `make_host_congested_archs.py` →
     `data/archs/lowered/geometry_sweep_host_congested` (27 files; b2b
     latency ×16 variants of the iso-capacity set)

Generation is pure derivation (no random seeds) and is deterministic under
the pinned Python environment (`uv`-managed, Python 3.10): repeat runs
produce byte-identical CSVs, `lookup_work/` trees, and arch YAMLs.

## Shipped as-is (not generated at setup)

- `sram_sweep_results.csv` — CACTI SRAM sweep for the per-PU buffer/SRAM
  area lookup. The characterization run configs were not preserved, so this
  CSV ships as data.
- `pu_postroute_22nm_cleaned.csv` — post-route PU results: RTL synthesized
  and placed-and-routed on Nangate45, scaled to 22 nm. Requires commercial
  EDA tooling; ships as data.
- `data/archs/lowered/baseline/` — the three hand-lowered baseline archs
  (lowered from the GPAM specs in `data/archs/baseline/`; verified by the
  `gpam-lower --check` gate, not produced by the builders here).
- `data/archs/lowered/activation/` — fig15's frozen characterization
  inputs: the activation eval archs have no in-repo generator, and the
  fig15 search arch is pinned beside them
  (`activation/search_base/`) at the same vintage.

## Generators in this folder

- `generate_dram_lookup_csvs.py` — runs CACTI/DreamRAM and writes the DRAM
  lookup CSVs (step 2 above).
- `generate_arch_from_template.py` — generate one architecture YAML from a
  baseline template plus the sweep CSVs (CLI; used by all builders below).
- `generate_arch_from_template_sweep.sh` — build the
  `data/archs/lowered/geometry_sweep/{upmem,hbm_pim}` YAML sets
  (`ARCHS_OVERRIDE` / `VARIANTS_OVERRIDE` select subsets).
- `build_iso_capacity_archs.py` — build the iso-capacity set; its
  `PLANS` table is also the single source of truth for the iso lookup
  tuples the CSV driver characterizes.
- `build_buffer_archs.py` — build `data/archs/lowered/buffer_sweep/`.
- `build_interconnect_archs.py` — build
  `data/archs/lowered/interconnect_sweep/`.
- `make_host_congested_archs.py` — derive congested-host variants
  (bank-to-bank latency ×16) from existing arch YAMLs.
- `generate_geometry_sweep_archs.py` — PU post-route lookup helpers
  (`_nearest_pu_row` / `_nearest_pu_area_mm2`): nearest characterized
  `simd_lanes` row at the nearest supported `tCK`, linear scaling from the
  modeled row. Imported by `experiments/attacc_area/attacc_area.py` as the
  single source of truth for the PU-area model.

## Area model (used by the generators and attacc_area)

`total_area_mm2` in generated YAMLs is explicit metadata for area studies:

- `dram_area_mm2 + total_num_pus * (pu_area_mm2 + sram_area_mm2)`

where `dram_area_mm2` comes from the DRAM lookup row, `pu_area_mm2` from the
nearest `(simd_lanes, tCK)` row of `pu_postroute_22nm_cleaned.csv`, and
`sram_area_mm2` from `sram_sweep_results.csv` at `(cache_bytes,
simd_lanes * 16)`. Sweeps do not cover every derived SIMD width: PU energy
and area scale linearly from the closest characterized `simd_lanes` row at
the nearest supported `tCK`; SRAM area scales linearly from the closest
characterized cacheline width, and small cache blocks round up to the
smallest characterized SRAM size at the modeled IO width.

Two nearest-match retrievals engage on the characterized grid and are the
lookup rules working as documented:

- The PU post-route DB has no 666 MHz (tCK ≈ 1.5 ns) characterization
  point, so the AttAcc-clock PU-area lookup (`_nearest_pu_row`, used by
  `experiments/attacc_area/`) retrieves the nearest characterized point —
  the 400 MHz-synthesis (tCK 2.5 ns) row — by design.
- SRAM/buffer sizes below 128 B round up to the 128 B row (the smallest
  characterized SRAM size), so the 32 B and 64 B buffer-sweep variants
  carry the 128 B row's SRAM area/energy values.

## Matching rules

- `io_width * burst_length = simd_lanes * 16`
- `burst_length` comes from the filename suffix `_burst-<N>`; when absent it is `4`
- the burst-rule checks in the generator must remain consistent with the
  selected DRAM row
