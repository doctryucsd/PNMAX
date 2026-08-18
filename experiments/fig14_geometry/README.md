# fig14_geometry — Fig. 14: fixed-capacity geometry sweeps (R6)

DRAM-geometry sensitivity for OPT-2.7B end-to-end on UPMEM + HBM-PIM:
(a) PU tile size, (b) bank-group count with inverse row size, (c) bank-group
count with inverse row count — each rung's best-latency mappings re-cost on
the 16x bases and normalized to the 1x geometry.

```bash
./run.sh              # full scale (2048 traces per (geometry, layer))
./run.sh --smoke      # 2 geometries/family + 1 extreme rung, 8 traces
                      # (~3 min; missing rungs render as gaps)
./run.sh --dry-run    # print the command plan
```

Phases

1. Reconstruct `nn_models/<model>/layer_params.csv` from the shipped
   end-to-end workload manifests into a `PNMAX_ROOT` shadow root
   (layer_params is not shipped).
2. `run-geometry-sweep-end-to-end-workload-space` for both arch families over
   `data/archs/lowered/geometry_sweep/**` (search basis; the sweep also
   covers the other two end-to-end models — OPT feeds Fig 14).
3. The extreme rungs the main sweep never searched (/4, x4, burst-1/16):
   UPMEM extremes on the `geometry_sweep_host_congested/iso_capacity_archs`
   basis, HBM-PIM extremes on the plain `geometry_sweep/iso_capacity_archs`
   basis (spaces c, direct-c).
4. `plot/geometry_sweep_abc_opt.py` re-cost + render: latency/energy on
   `lowered/geometry_sweep_host_congested/iso_capacity_archs/**`, area on
   `lowered/geometry_sweep/iso_capacity_archs/**`. The line chart lands as
   `figures/fig14.pdf`.

Runtimes: smoke `~3 min` measured; full `~6.5 h` measured at 64 workers
(e2e sweeps ~4 h upmem + ~2 h hbm_pim, extremes ~16 min, re-cost + render ~14 min).
