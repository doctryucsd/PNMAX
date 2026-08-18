# Fig. 10 baselines (UniNDP / OptiPIM / CINM)

The Fig. 10 Pareto grid (R3) compares PNMAX-searched mappings against three
external baselines. This directory holds the drivers that produce the OptiPIM and
CINM comparison points; the UniNDP baseline mappings are derived separately (see
`external/unindp/derive_baselines.md`).

> Status: these drivers are **adapted** from the originals in the PNMAX
> development repo (imports updated to the artifact's `pnmax` package;
> repo-relative, env-overridable paths). They are
> wired into `run.sh` and produce the OptiPIM/CINM comparison points as part of
> the full Fig-9 campaign (and its `--smoke` path); the resulting proxy CSVs
> feed the Pareto grid.

## OptiPIM baseline = a *searched proxy* (not the OptiPIM tool)

The OptiPIM ILP solver is **not** run. The "OptiPIM" point is a
**searched proxy**: for each of the 9 kernels on the HBM-PIM baseline arch, take
the **best of N random PNMAX mappings** (best by latency, and by the weighted
latency+footprint+energy cost), normalized to the same per-workload
`baseline.json` the grid uses.

- **N = 4096** at full scale — two independent 2048-trace random searches
  (a primary search plus a second seed, concatenated). `extract_optipim_proxy.py`
  warns if any `(workload, streaming)` cell does not total 4096 traces.
- Rationale: OptiPIM optimizes a mapping for a fixed architecture; a large random
  search over the same PNMAX mapping space is a faithful, tool-free stand-in for
  "the best mapping a strong optimizer would find", and it uses exactly the PNMAX
  cost model so the comparison is apples-to-apples. This is why the point is
  labelled a *searched proxy* in the figure.

`extract_optipim_proxy.py`
- reads the HBM-PIM random-search traces under `results/…` (env
  `PNMAX_RESULTS_ROOT`, default `<repo>/results`),
- re-costs each on the PNMAX analytical model (`data/archs/lowered/baseline/hbm_pim.yaml`,
  override `--arch-file`),
- emits `results/fig10_pareto/optipim/optipim_proxy.csv` (override `--output`).

## CINM baseline (two approaches, UPMEM subfigures)

Both use the in-repo `cinm-opt` built by `setup.sh` by default
(`external/cinm/build/bin/cinm-opt`); override with the `CINM_OPT` env var.

- **`cinm_approach_a.py` (Approach A, used for Fig. 10):** emit each kernel as a
  `cinm.compute` GEMM, run `cinm-opt --cinm-tiling`, parse CINM's tile
  decomposition, and re-cost that mapping on the PNMAX model (UPMEM-normalized).
  Kernel shapes come from the derived UniNDP-baseline `<wl>_upmem.yaml`
  (env `PNMAX_UNINDP_BASELINE_DIR`); writes `.../fig10_pareto/cinm/cinm_proxy.csv`.
- **`cinm_solution_on_our_model.py` (Approach B):** evaluate CINM's own
  `testbench/*.mlir` solutions (which encode a fixed `workgroupShape`) on the
  PNMAX model, cross-checked against the real-UPMEM HW numbers from CINM's
  artifact (hardcoded `HW_NOPT`). Imported by Approach A for its shape→mapping
  helpers; also runnable standalone.

The `cinm-opt` binary is Cinnamon `f776a07` built against public LLVM 18.1.6 — see
`external/cinm/PROVENANCE.md`.

## Environment overrides

| var | meaning | default |
|---|---|---|
| `CINM_OPT` | path to `cinm-opt` | `<repo>/external/cinm/build/bin/cinm-opt` |
| `PNMAX_UPMEM_ARCH` | UPMEM baseline arch YAML | `<repo>/data/archs/lowered/baseline/upmem.yaml` |
| `PNMAX_RESULTS_ROOT` | DSE results root | `<repo>/results` |
| `PNMAX_UNINDP_BASELINE_DIR` | derived UniNDP-baseline dir | `<repo>/results/unindp_baseline` |
| `PNMAX_CINM_OUT` | CINM proxy output dir | `<results>/fig10_pareto/cinm` |
