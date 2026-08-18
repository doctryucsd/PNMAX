# PNMAX — MICRO 2026

Source code for **PNMAX: Mapping-Architecture Co-Exploration Framework for Processing Near Memory** (MICRO 2026).

## Overview

PNMAX is a design-space-exploration framework for processing-near-memory
(PNM) systems. It co-explores workload mappings and architecture
configurations, combining a graph-based parameterized architecture model
(GPAM), a per-component PPA characterization database, and an analytical cost
model validated against the UniNDP and AttAcc simulators.

This repository contains the framework and the reproduction of all paper
results. Reproduced results may differ slightly from the published ones due
to the random seed, but the conclusions hold.

## Requirements

- Linux (tested on Ubuntu 24.04, kernel 6.x)
- Python 3.10, managed via [uv](https://docs.astral.sh/uv/): `make setup`
  bootstraps uv if missing, no manual virtualenv handling needed
- C++ toolchain (gcc/g++, CMake) for the external simulators
  (AttAcc/Ramulator2, CACTI)
- No GPU
- git, ninja and clang/clang++ for the CINM baseline in Fig. 10, which
  `make setup` builds by default (clang required — g++ rejects a CINM template;
  see `external/cinm/PROVENANCE.md`). This is a large download and a long LLVM
  build: ~5 min on the 64-core reference machine, hours on small ones. Run
  `make setup-nocinm` to skip it (Fig. 10 then renders without the CINM
  markers via `PNMAX_SKIP_CINM=1`).
- For full-scale runs a many-core machine is strongly recommended. Runs are
  pinned to 64 workers by default; the reference machine is a
  64-core/128-thread AMD Threadripper 9985WX with 768 GB RAM. Smoke runs
  (`SMOKE=1`) work on any machine.

## Setup

```sh
make setup
```

This checks/bootstraps uv, creates the Python env (`uv sync`), verifies the
GPAM specs against their lowered files, builds the external simulators,
generates the PPA DB and the derived architectures, and builds the CINM
baseline. The CINM build is a long LLVM build (~5 min on the reference
machine, hours on small ones); everything else takes a few minutes. Run
`make setup-nocinm` to skip the CINM build.

To check the whole install before the long runs, `make smoke` exercises every
pipeline end-to-end at smoke scale in about 12 minutes.

## Reproducing the paper results

Each result is one `make` target. Run `make` with no target to list them.
Figure numbers match the paper. Figs. 9 and 13 are measurement-analysis
figures with no experiment pipeline, so there is no `make fig9` or `make fig13`.

```bash
make fig10              # full scale
make fig10 SMOKE=1      # minutes-scale end-to-end check
make fig10 ARGS=--dry-run   # print the command plan, run nothing
make all               # the whole campaign in one command
```

`make all` takes **~15 h** at the default 64 workers. It runs `fig10` before
the figures that reuse its mapping pool.

| Check point | Command | Result | Full-scale runtime |
|---|---|---|---|
| 1 | `make fig10` | Fig. 10 — mapping-DSE Pareto fronts vs. UniNDP/OptiPIM/CINM baselines | ~6.5 h (builds the shared pool reused by Figs 11/12/15/16) |
| 2 | `make fig8` | Fig. 8 + Table 4 — analytical model vs. UniNDP/AttAcc simulators (R^2, speedups) | ~5 min |
| 3 | `make fig11` | Fig. 11 — impact of disabling streaming (footprint/latency) | <1 min after fig10 |
| 4 | `make fig12` | Fig. 12 — latency breakdown of best-latency HBM-PIM mappings | ~1 min after fig10 |
| 5 | `make fig14` | Fig. 14 — DRAM geometry sweeps (OPT-2.7B on UPMEM + HBM-PIM) | ~6.5 h |
| 6 | `make fig15` | Fig. 15 — buffer-size sensitivity (fc_bert-72 on HBM-PIM) | <1 min after fig10 (~10 min cold) |
| 7 | `make fig16` | Fig. 16 — system-level sharing via host vs. inter-bank interconnect | <1 min after fig10 (~30 min cold) |
| 8 | `make fig17` | Fig. 17 — reduction placement (bank/channel/base-die) | ~1.6 h |
| 9 | `make attacc_area` | AttAcc PU+buffer area overhead from the PPA DB | <10 s |

- Each target wraps `experiments/<name>/run.sh`. `ARGS="..."` forwards flags
  to it: `--workers N`, `--seed N`, `--dry-run`.
- Outputs are under `results/<experiment>/`; set `PNMAX_RESULTS_ROOT` to
  redirect the results root. `SMOKE=1` writes under `results/smoke/`.
- Figures are under `results/<experiment>/figures/figNN.pdf`.
- Why results can differ from the paper's: the paper's original DSE runs were unseeded, while artifact runs use a fixed default seed, so search-derived results sample different mappings. The conclusions do not change.

**Interrupted runs.** Every run is safe to re-run after an interruption:
completed partial results are detected and reused, and only missing or
mismatched pieces are recomputed.

**PPA DB.** The DRAM characterization CSVs
(`data/ppa/{cacti_ddr4,dreamram_hbm2e}_results.csv`) and the derived
sweep-arch families under `data/archs/lowered/` are generated
deterministically by `make setup`.

## Using PNMAX for your own DSE

The PNMAX toolchain is general: it can co-explore your own mappings and PNM
architectures.

**Workloads.** A workload is one kernel in one YAML file. Its fields:

- `Problem` — the loop extents, in convolution notation: `N` batch, `K`
  output channels, `P`/`Q` output size, `C` input channels, `R`/`S` filter
  size. GEMM/GEMV = conv with `P=Q=R=S=1`.
- `Bound` — the scheduling, as loop factors per level. Level `0` is the
  per-step concurrency slice. Level `1` is the sequential loop. Higher
  levels are spatial splits. Each dim's factors must multiply to its
  `Problem` extent.
- `TensorAttr` — the element width of each tensor, e.g. `16b`.
- `Compute` — the einsum statement over the tensors.
- `Layout-Pragma` — the mapping knobs. `cache` uses the per-PU buffer. `streaming` specifies if a tensor is streamed from
  the host instead of pre-staged in DRAM. `sharing` sets the block, PU,
  and system sharing levels. `interleaving` interleaves the stored tensors
  across a PU's banks (if multiple).

Samples: `data/workloads/samples/`. The nine paper kernels:
`data/workloads/{fc,conv2d,b_fc}/`. Multi-layer end-to-end models:
`data/workloads/end_to_end/`.

**Architectures (GPAM).** An architecture is a GPAM (Graph-based
Parameterized Architecture Model) spec. A spec is two YAML files: a topology
and a technology database. Both are required: the topology defines the
structure and names its technology database (top-level `tech:` key), which
carries the costs. The three baselines are under
`data/archs/baseline/`. The analytical model does not read GPAM specs
directly: `gpam-lower` generates the runtime YAML from a spec. The runtime
YAMLs are under `data/archs/lowered/`.

Two ways to build a variant:

- Copy a GPAM pair, edit it, and lower it:
  `gpam-lower --arch my.yaml --out my_lowered.yaml`.
- Generate a parameter delta with
  `data/ppa/generate_arch_from_template.py`. Knobs: bank geometry, PU count,
  burst length, buffer size, interconnect placement. Costs are filled from
  the PPA DB. `setup.sh` builds the sweep families under
  `data/archs/lowered/` this way.

Look at `data/archs/README.md` and `data/ppa/PROVENANCE.md` for details.

**Search and evaluate.**  `run-workload-space-random-search` is an example mapping explorer that samples legal mappings for a problem/arch pair. You can also use your own mapping explorer.
`run-workload-space-pareto-eval` is an example mapping-architecture evaluation.

## Repository layout

```
.
├── README.md              # this file
├── Makefile               # reviewer entry point: make setup / make figNN / make all
├── LICENSE                # MIT
├── pyproject.toml         # uv project (Python 3.10); the gpam-* and run-* CLIs
│                          # are declared here
├── uv.lock                # locked dependency set (generated by `uv lock`)
├── setup.sh               # one-time bootstrap: uv env, external simulator builds,
│                          # PPA characterization + derived-arch generation
├── src/pnmax/             # the PNMAX framework
├── data/
│   ├── workloads/         # kernel + end-to-end workload YAMLs
│   ├── archs/             # GPAM specs + lowered flat YAMLs (sweep families
│   │                      # generated by setup.sh)
│   └── ppa/               # PPA DB: characterization drivers + CSVs (DRAM CSVs
│                          # generated by setup.sh; SRAM/post-route CSVs ship)
├── external/              # vendored third-party tools (unindp, attacc, cacti,
│                          # dreamram, cinm), each with LICENSE + PROVENANCE.md
├── experiments/           # one push-button run.sh per paper result
│   └── _lib/              # shared run.sh helpers (banners, timers, seeds, --smoke)
├── plot/                  # per-figure plot scripts
└── results/               # run outputs (gitignored; created on demand)
```

## License

MIT — see [LICENSE](LICENSE). Vendored tools under `external/` retain their
upstream licenses (see each `external/*/LICENSE` and `PROVENANCE.md`).
