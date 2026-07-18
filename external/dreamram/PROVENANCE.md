# Provenance: DreamRAM

- **Upstream:** https://github.com/harvard-acc/DreamRAM
- **Vendored branch / commit:** `DATE2026` @
  `c069ce14dfa85ce1983f3a1274a265d1e7b5494a`
- **License:** Apache 2.0 (see `LICENSE`; © 2025 Harvard University
  Architecture, Circuits & Compilers Group). Upstream `README.md` preserved.
- **Paper:** DreamRAM (DATE 2026); preprint arXiv:2512.12106.

## Why the DATE2026 branch (version identification)

PNMAX's PPA generator (`data/ppa/generate_dram_lookup_csvs.py`) references
`configs/mem/sweep/hbm2e.json` and `configs/mem/baseline/hbm2e.json`, and its
per-run configs (`dream_*/dreamram_mem.json`, derived under
`data/ppa/lookup_work/dram/`) set
`"baseline": "configs/mem/baseline/hbm2e.json"`. Those exact file names exist
**only on the `DATE2026` branch** — `main` renamed them
(`hbm2e_just_baseline.json`, etc.). Since DreamRAM's paper is DATE'26, the
DATE2026 branch is the camera-ready version that produced the artifact's
DreamRAM data. (The tool is public and redistributable.)

## What it is used for in this artifact

DreamRAM is the **HBM-class MEM PPA characterization backend**. It
analytically models the HBM2E DRAM used by the HBM-PIM baseline and geometry
sweeps, producing the `dreamram_hbm2e_results.csv` rows of the PPA DB (energy,
timing, area per DRAM organization). CACTI 7 handles the DDR4/UPMEM side; the two
together are the DRAM half of the PPA DB.

- Input configs (per run): `data/ppa/lookup_work/dram/dream_*/dreamram_mem.json`
  (sweep config, derived per run)
  + `external/dreamram/configs/tech/scaled/17nm_hbm2e.json` (tech). The driver
  additionally pins a recorded timing-calibration input via a driver-owned
  per-run baseline copy — the vendored tree is never modified; see
  `data/ppa/PROVENANCE.md`.
- Output CSV: `data/ppa/dreamram_hbm2e_results.csv` (a derived output of the
  standard generation flow; the PPA DB is keyed by DRAM geometry).
- Driver: `generate_dram_lookup_csvs.py` runs
  `dreamram.py -m <cfg> -t <tech> -o <run>` per DRAM tuple and folds the raw
  columns into the PPA schema via its `parse_dreamram` step.

## Replay verification (exact match)

At vendoring time, `dreamram.py` (this vendored DATE2026 source) was run on a
recorded `dream_*/dreamram_mem.json` per-run config (a fixture preserved in
the artifact's development history; the standard flow now derives these per
run) with `17nm_hbm2e.json`; the raw output columns were compared to the
corresponding row of the recorded `dreamram_hbm2e_results.csv`:

| PPA column ← DreamRAM raw | value (both) |
|---|---|
| `E_ACT_pJ`  ← `e_cmd_act_pj` | `458.6927951969719` |
| `E_PRE_pJ`  ← `e_cmd_pre_pj` | `106.56268121322995` |
| `E_READ_pJ` ← `e_cmd_rd_pj`  | `338.1263035869309` |
| `e_set_col_pJ` ← `e_set_col` | `11.869532578997635` |
| `e_set_csl_pJ` ← `e_set_csl` | `20.659729778162006` |
| `e_set_ldl_pJ` ← `e_set_ldl` | `1.8534209056089348` |
| `e_set_mdl_pJ` ← `e_set_mdl` | `41.31945955632401` |

**Verdict: exact match** — every column reproduces to full float precision
(16–17 significant digits). DreamRAM DATE2026 @ `c069ce14` is the exact version
that produced the artifact's HBM2E PPA data; the results are deterministic.

## Dependencies & running

`dreamram.py`'s sweep path needs only **numpy** (already a PNMAX dependency).
`plot.py` (the design-space explorer / plotting) additionally imports
`matplotlib` and `paretoset`, which are **not** required to generate the PPA
CSV and are not part of the PNMAX environment. There is no `requirements.txt`
upstream. Interface: `python dreamram.py -m <mem_sweep.json> -t <tech.json>
-o <run_label>` → writes `data/<run_label>/hbm3_<run_label>.csv`.

## Local modifications

None. Snapshot via `git archive` of the DATE2026 branch — tracked files only
(no `.git`, no generated `data/` outputs).
