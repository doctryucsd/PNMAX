# Provenance: UniNDP

- **Upstream:** https://github.com/doctryucsd/UniNDP
  (fork of the HPCA'25 artifact https://github.com/UniNDP-hpca25-ae/UniNDP)
- **Vendored commit:** `3ac9db058b5ef7ff3d87089f50437ed1c26bd2dd`
- **License:** MIT (see `LICENSE`; © 2024 UniNDP-hpca25-ae)
- **Paper:** UniNDP: A Unified Compilation and Simulation Tool for Near DRAM
  Processing Architectures (Xie et al., HPCA 2025)

## What it is used for in this artifact

UniNDP is the primary **simulator-validation baseline** and the source of the
**UniNDP baseline mappings** in the Fig. 9 Pareto comparison:

1. **Validation (Fig. 8 / Table 4):** PNMAX's analytical latency/energy model is
   validated against UniNDP's cycle-accurate near-DRAM simulator (HBM-PIM and
   UPMEM targets). The PNMAX adapter is `src/pnmax/simulators/unindp/` and the
   CLI `run-unindp`.
2. **Fig. 9 UniNDP baseline:** UniNDP's compiler chooses a mapping for each
   kernel; that mapping is translated into a PNMAX workload YAML and re-costed on
   the same analytical model, giving the "UniNDP" points in the Pareto grid.
   See `derive_baselines.md` (recipe) and `derive_baselines.sh` (runnable demo).

## How this snapshot was produced

Exported from the PNMAX development repo's `external/unindp` at the pinned commit via
`git archive HEAD` — tracked files only (no `.git`, no `__pycache__`, no run
artifacts such as `dbg/`, `tmp_smoke*/`). The upstream `requirements.txt`
version pins (numpy 2.1.3, PyYAML 6.0.2, tqdm 4.66.4, xlrd 2.0.1,
xlsxwriter 3.2.0) are preserved as-is. `xlrd`/`xlsxwriter` are only needed by
UniNDP's spreadsheet export helpers; the compile + simulate paths PNMAX uses
require only numpy / PyYAML / tqdm (already in the PNMAX environment).

## Local modifications

None to the vendored UniNDP sources. Two **added** files (not part of upstream)
support the baseline-derivation recipe and carry a leading underscore /
descriptive name so they are obviously artifact additions:

- `derive_baselines.sh` — orchestrates compile → convert → diff.
- `derive_baseline_yaml.py` — self-contained `best_design` → PNMAX workload-YAML
  converter (pnmax-free port of `helpers/run_unindp_compile.py`'s `dump-yaml`).
- `_compile_driver.py` — namespace-package launcher for UniNDP's `compile.py`.
- `derive_baselines.md` — the derivation recipe + validation report.

## Standalone smoke test

`python _compile_driver.py -- -A upmem -W mm -N tiny -S 256 256 256 1 -WS <ws>
-O tiny -Q -K 3` compiles a 256³ GEMM and prints a `best_design` /
`best_result` / `speedup` — verified to run against the PNMAX environment
(numpy/PyYAML/tqdm).
