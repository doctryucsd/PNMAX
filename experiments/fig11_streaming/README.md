# fig11_streaming — Fig. 11: no-streaming latency/footprint trade-off (R4)

Percent change in latency (x) and memory footprint (y) of each Pareto front's
best-latency mapping when streaming is disabled, faceted by architecture.
Shares the Fig 10 search pool + Pareto evals (reused when already present, so
running this after `fig10_pareto` only renders the figure).

```bash
./run.sh              # full scale
./run.sh --smoke      # 1 kernel, 8 traces/space — end-to-end check
./run.sh --dry-run    # print the command plan
```

Outputs: `<results-root>/fig11_streaming/figures/fig11.pdf`.

Runtimes: smoke `~1 s` after fig10's smoke pool (`~50 s` cold);
full `<1 min` after fig10 measured (pool fully reused; render only —
standalone-cold it first rebuilds the fig10 inputs, ~6.5 h).
