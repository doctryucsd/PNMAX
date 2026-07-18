# fig13_buffer — Fig. 13: buffer-size sensitivity (R7)

Latency of selected fc_bert-72 mappings on HBM-PIM across register-file sizes
128B..2KB, normalized to the 512B variant.

```bash
./run.sh              # full scale (2048-trace pool, 30x3 sampled sets)
./run.sh --smoke      # 8-trace pool, 4x2 sets — end-to-end check in seconds
./run.sh --dry-run    # print the command plan
```

Phases: UniNDP baseline + fc_bert-72 mapping pool (shared with Fig 9) ->
random-set latency evaluation over the 5-variant buffer subset
(the generated `lowered/buffer_sweep/hbm_pim` tree also carries 32B/64B
variants, which the driver excludes via a symlinked subset) -> figure
(`figures/fig13.pdf`).

Per-variant area sits alongside latency in `<run dir>/plot_rows.csv` (run dir
recorded in `results/fig13_buffer/latest_run_dir.txt`).

Runtimes: smoke `~2 s` with the pool present (`~1 min` cold);
full `<1 min` with the fig09 pool present measured (5-variant eval ~20 s at
64 workers); ~10 min standalone-cold (one a-d search cell + baseline).
