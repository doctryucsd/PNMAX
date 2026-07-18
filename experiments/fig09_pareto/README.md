# fig09_pareto — Fig. 9: mapping-space Pareto-front grid (R3)

Produces the Pareto-front grid: 9 kernels x {UPMEM, HBM-PIM} x DSE spaces
(a)-(d), with the UniNDP-baseline point and the OptiPIM searched-proxy / CINM
compiler-tiling comparison markers.

```bash
./run.sh              # full scale
./run.sh --smoke      # 1 kernel, 8 traces/space — end-to-end check in ~20 s
./run.sh --dry-run    # print the command plan
```

Phases

1. Derive the 18 UniNDP baseline mappings (vendored UniNDP compile +
   `best_design` conversion; `results/unindp_baseline/`).
2. Seeded random search of spaces a-d per (arch, kernel, streaming) into the
   shared pool `results/workload_space_random_search/` (2048 traces per space
   delta at full scale). Completed cells are reused by the
   fig10/fig11/fig13/fig14 buttons.
3. Pareto evaluation per kernel into `results/workload_space_pareto_eval/`.
4. Baselines: OptiPIM searched proxy (best of 2x2048 random HBM-PIM mappings,
   second pool at seed+1) and CINM compiler tiling (`cinm-opt`, built by
   default by `./setup.sh`; full mode requires it unless `PNMAX_SKIP_CINM=1`).
5. The Fig. 9 grid (latency vs footprint), rendered as `figures/fig09.pdf`.

Runtimes: smoke `~35 s` measured (cold); full `~6.5 h` measured at 64 workers
(baselines ~5 min, search pool ~80 min, Pareto evals ~5 h; a re-run with the
pool present completes in ~20 min).

Baseline driver internals: see `baselines/README.md`.
