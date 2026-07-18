# fig15_reduction — Fig. 15: reduction placement (R9)

Pareto fronts of bank- vs channel- vs base-die-level reduction placement for
bmm_llama256-3 on HBM-PIM, normalized to the AttAcc-style point (best-latency
base-die mapping). Encodes the two-stage inclusive semantics in the
driver: bank = config1 (PU) pool; channel = config1 re-cost under
`channel_level` ∪ the L4-capped channel pool (PU-union step); base-die =
composed config1 ∪ channel ∪ config2 pool.

```bash
./run.sh              # full scale (2048-trace pools)
./run.sh --smoke      # 8-trace pools — end-to-end check in ~3 min
./run.sh --dry-run    # print the command plan
```

Phases: UniNDP baseline -> direct-c + direct-l4 mapping pools on the pinned
search base (`lowered/activation/search_base/hbm_pim__pu-8__hmat-16__vmat-32`,
frozen at the activation eval archs' characterization vintage — see the README
in that directory)
-> stage A plain activation eval (config1 search) -> stage B inclusive eval
(config1 reuse, `--max-system-sharing 4`) -> channel PU-union re-cost
(`puunion_recost.py`) -> Pareto plot (`figures/fig15.pdf`).

Arch basis: `data/archs/lowered/activation/{hbm_pim__pu-8__hmat-16__
vmat-32,base_die,channel_level}.yaml`.

Runtimes: smoke `~3 min` measured; full `~1.6 h` measured at 64 workers
(pools ~40 s, activation + inclusive evals ~72 min, PU-union re-cost ~22 min).
