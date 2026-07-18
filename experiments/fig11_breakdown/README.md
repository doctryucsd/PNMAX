# fig11_breakdown — Fig. 11: HBM-PIM latency breakdown (R5)

Stacked latency-component bars of each kernel's best-latency HBM-PIM mapping
per DSE space (a)-(d), normalized to (a). Shares the Fig 9 HBM-PIM search
pool + Pareto evals (reused when already present).

```bash
./run.sh              # full scale
./run.sh --smoke      # 1 kernel, 8 traces/space — end-to-end check
./run.sh --dry-run    # print the command plan
```

Outputs: `<results-root>/fig11_breakdown/figures/fig11.pdf`.

Runtimes: smoke `~1 s` after fig09's smoke pool (`~30 s` cold);
full `~1 min` after fig09 measured (HBM-PIM pool reused; render only —
standalone-cold it rebuilds the HBM-PIM half of the fig09 inputs, ~3 h).
