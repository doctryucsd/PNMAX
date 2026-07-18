# fig10_streaming — Fig. 10: no-streaming latency/footprint trade-off (R4)

Percent change in latency (x) and memory footprint (y) of each Pareto front's
best-latency mapping when streaming is disabled, faceted by architecture.
Shares the Fig 9 search pool + Pareto evals (reused when already present, so
running this after `fig09_pareto` only renders the figure).

```bash
./run.sh              # full scale
./run.sh --smoke      # 1 kernel, 8 traces/space — end-to-end check
./run.sh --dry-run    # print the command plan
```

Outputs: `<results-root>/fig10_streaming/figures/fig10.pdf`.

Runtimes: smoke `~1 s` after fig09's smoke pool (`~50 s` cold);
full `<1 min` after fig09 measured (pool fully reused; render only —
standalone-cold it first rebuilds the fig09 inputs, ~6.5 h).
