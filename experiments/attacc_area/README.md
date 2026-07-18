# attacc_area — R10: AttAcc PU+buffer area overhead

Derives the area overhead of AttAcc's PUs and per-PU register-file
buffers under the PNMAX PPA database. For context, the original AttAcc paper
(ASPLOS'24) reports a **10.84%** overhead for this configuration.

```bash
./run.sh              # full = smoke: pure PPA-DB arithmetic, seconds
./run.sh --dry-run    # print the command plan
```

- Inputs: `data/ppa/pu_postroute_22nm_cleaned.csv` (post-route PU macro),
  `data/ppa/sram_sweep_results.csv` (CACTI register-file area),
  `data/ppa/dreamram_hbm2e_results.csv` (DreamRAM HBM2E die footprint).
- Method + configuration block: see `attacc_area.py` (AttAcc config:
  16-lane PU at 666 MHz, 8 KB per-PU register file, 1 PU per bank, 16 MB banks
  with 1 KB rows).
- Output: `<results-root>/attacc_area/attacc_area_summary.json` and a printed
  derivation ending in the computed overhead (`computed_overhead_pct`). The
  script only derives and reports the value; it exits non-zero if a PPA-DB
  input is missing or ambiguous.

Runtime: `< 10 s` (both modes).
