# fig16_interconnect — Fig. 16: via-host vs inter-bank sharing (R8)

Grouped bars: each kernel's best (b)-space mapping re-evaluated under the
via-host (congested, 16x-degraded) and inter-bank interconnect sharing
configs, normalized to the no-sharing reference (latency + energy per config,
config-independent footprint). Arch basis:
`data/archs/lowered/interconnect_sweep/**` with `--host-cost congested`
(the center file's b2b already bakes in the 16x congested-host degradation).

```bash
./run.sh              # full scale
./run.sh --smoke      # 8-trace (b)-pools, all 9 kernels (the evaluator
                      # needs every cell) — first run ~5-6 min (18 UniNDP
                      # baseline compiles, cached), warm ~30 s
./run.sh --dry-run    # print the command plan
```

Phases: UniNDP baselines -> (b)-space search pools for 9 kernels x
{UPMEM, HBM-PIM} (spaces a b; shared with the Fig 10 pool) -> the figure's
own evaluation loop + render (`plot/bspace_interconnect_sharing_bars.py
--host-cost congested --no-intra-bg --drop-no-duplication --metric all`) into
`figures/fig16.{pdf,csv}`.

Note: `--drop-no-duplication` removes kernel groups whose best (b) mapping
needs no inter-bank duplication, so the surviving group set depends on the
searched pool (a group can be absent when its best mapping happens to need
no duplication).

Runtimes: smoke `~5-6 min` cold (18 UniNDP baseline compiles dominate; ~30 s
warm); full `<1 min` with the fig10 pool present measured (evaluation + render
~22 s at 64 workers); ~30 min standalone-cold (18 (b)-pool cells + baselines).
