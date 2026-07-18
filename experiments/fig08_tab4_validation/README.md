# fig08_tab4_validation — Fig. 8 + Table 4: model-vs-simulator validation (R1/R2)

Validates the PNMAX analytical model against the UniNDP simulator (HBM-PIM,
UPMEM) and the AttAcc simulator (latency + energy): scatter plots (Fig. 8)
plus the Table-4 wall-time/R^2 fields.

```bash
./run.sh              # full scale: 2000 validated mappings per UniNDP
                      # arch, ~1013 for AttAcc (its restricted mapping space)
./run.sh --smoke      # 8-mapping sets — end-to-end check in seconds
./run.sh --dry-run    # print the command plan
```

Phases: seeded random-mapping set generation (2048 traces per UniNDP arch
from the 9 evaluation kernels; AttAcc full-mode GEMV set) -> simulator vs
analytical validation runs (`--sample_rate 0.05 --single_pu`; requires the
AttAcc ramulator2 build from `./setup.sh`) -> the 2x2 validation figure.

Outputs under `<results-root>/fig08_tab4_validation/`: `validation/*.json`
(the per-arch summary fields carry the Table-4 wall-time/R^2 quantities)
and `figures/fig08.pdf`.

Runtimes: smoke `~5 s` measured; full `~5 min` measured at 64 workers
(UniNDP validation 75 s upmem + 134 s hbm_pim; mapping-set generation ~30 s).
