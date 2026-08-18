#!/usr/bin/env bash
# experiments/fig16_interconnect/run.sh — Fig. 16: system-level sharing via
# host vs. inter-bank interconnect (grouped bars; latency + energy + footprint
# normalized to the no-sharing reference).
#
# Usage: ./run.sh [--smoke] [--dry-run] [--workers N] [--seed N]
#   default : full scale (2048-trace (b)-space pools, all 9 kernels)
#   --smoke : minutes-long end-to-end check (8-trace pools, all 9 kernels —
#             the evaluator requires every kernel cell)
# Writes only under the results root (shared pool + fig16_interconnect/).
#
# Pipeline: (1) UniNDP baselines + streaming-mode (b)-space search pools for
# all 9 kernels x {UPMEM, HBM-PIM}, (2) the figure's own evaluation loop
# (best (b) mapping per cell re-evaluated under host-x16 / inter-bg /
# no-sharing interconnect configs) + render.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../_lib/common.sh
. "${SCRIPT_DIR}/../_lib/common.sh"

pnmax_init "fig16_interconnect" "$@"
phase_banner "fig16_interconnect — Fig. 16: via-host vs inter-bank sharing"

# shellcheck source=../_lib/pipeline.sh
. "${SCRIPT_DIR}/../_lib/pipeline.sh"

# ---------------------------------------------------------------------------
# Experiment defaults (single block). Arch basis:
# Fig 16 evaluates data/archs/lowered/interconnect_sweep/** with
# --host-cost congested (the center file's b2b already bakes in the 16x
# congested-host degradation = 1152) — those paths are hardcoded in
# plot/bspace_interconnect_sharing_bars.py.
# The mapping pools are searched on the BASE baselines (shared with Fig 10).
# ---------------------------------------------------------------------------
FULL_NUM_TRACES=2048
SMOKE_NUM_TRACES=8
PLOT_FLAGS=(--host-cost congested --no-intra-bg --drop-no-duplication --metric all)
ARCHS=(upmem hbm_pim)

if [ "${PNMAX_SMOKE}" = "1" ]; then
  NUM_TRACES="${SMOKE_NUM_TRACES}"
else
  NUM_TRACES="${FULL_NUM_TRACES}"
fi

# ---------------------------------------------------------------------------
phase_banner "phase 1/2 — UniNDP baselines + (b)-space search pools (9 kernels x 2 archs)"
# ---------------------------------------------------------------------------
pairs=()
for arch in "${ARCHS[@]}"; do
  for token in "${WORKLOAD_TOKENS[@]}"; do
    pairs+=("${token}:${arch}")
  done
done
step_start "UniNDP baseline derivation (${#pairs[@]} pairs)"
ensure_unindp_baselines "${pairs[@]}"
step_end

step_start "(b)-space search pools"
for arch in "${ARCHS[@]}"; do
  pool_arch=$(arch_pool_dir "${arch}")
  for token in "${WORKLOAD_TOKENS[@]}"; do
    ensure_search_cell \
      "${arch}" "$(arch_base_file "${arch}")" \
      "${BASELINE_DIR}/${token}_${arch}.yaml" \
      "${SEARCH_POOL_ROOT}/no_streaming_false/${pool_arch}/${token}" \
      "${NUM_TRACES}" false a b
  done
done
step_end

# ---------------------------------------------------------------------------
phase_banner "phase 2/2 — interconnect/sharing evaluation + bars (full: ~1 min)"
# ---------------------------------------------------------------------------
step_start "best-(b) selection + config re-evaluation + render"
run_py python "${REPO_ROOT}/plot/bspace_interconnect_sharing_bars.py" \
  "${PLOT_FLAGS[@]}" \
  --search-root "${SEARCH_POOL_ROOT}/no_streaming_false" \
  --workers "${PNMAX_WORKERS}" \
  --output "${RESULTS_DIR}/figures/fig16.pdf"
step_end

phase_banner "fig16_interconnect done — figures: ${RESULTS_DIR}/figures"
