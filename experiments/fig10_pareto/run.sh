#!/usr/bin/env bash
# experiments/fig10_pareto/run.sh — Fig. 10: mapping-space Pareto-front grid,
# 9 kernels x {UPMEM, HBM-PIM} x DSE spaces (a)-(d) vs the UniNDP / OptiPIM /
# CINM baselines.
#
# Usage: ./run.sh [--smoke] [--dry-run] [--workers N] [--seed N]
#   default : full scale (2048 traces per DSE space per cell; the
#             OptiPIM proxy pool totals 4096 traces per (kernel, streaming))
#   --smoke : minutes-long end-to-end check (1 kernel, 8 traces per space)
# Writes only under the results root (shared pool + fig10_pareto/).
#
# Pipeline: (1) derive UniNDP baseline mappings from the vendored UniNDP,
# (2) seeded random search of DSE spaces a-d per (arch, kernel, streaming),
# (3) Pareto evaluation, (4) OptiPIM searched-proxy + CINM compiler baselines,
# (5) render the Pareto grid.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../_lib/common.sh
. "${SCRIPT_DIR}/../_lib/common.sh"

pnmax_init "fig10_pareto" "$@"
phase_banner "fig10_pareto — Fig. 10: Pareto-front grid vs UniNDP/OptiPIM/CINM"

# shellcheck source=../_lib/pipeline.sh
. "${SCRIPT_DIR}/../_lib/pipeline.sh"

# ---------------------------------------------------------------------------
# Experiment defaults (single block). Arch files: figs 9-11 run on
# the BASE lowered baselines (pipeline.sh ARCH_FILE_*). Counts are the
# full-scale budgets; PNMAX_SEED+1 is the OptiPIM proxy's second independent
# search.
# ---------------------------------------------------------------------------
FULL_NUM_TRACES=2048        # per DSE-space delta per (arch, kernel, streaming)
SMOKE_NUM_TRACES=8
ARCHS=(upmem hbm_pim)
CINM_OPT="${CINM_OPT:-${REPO_ROOT}/external/cinm/build/bin/cinm-opt}"
export PNMAX_CINM_OUT="${RESULTS_DIR}/cinm"

if [ "${PNMAX_SMOKE}" = "1" ]; then
  NUM_TRACES="${SMOKE_NUM_TRACES}"
  WORKLOAD_TOKENS=(bert-3)   # one kernel keeps the smoke pool in minutes
else
  NUM_TRACES="${FULL_NUM_TRACES}"
  # Up-front precondition (skipped under --dry-run): fail before hours of
  # search, not at the CINM phase.
  if [ "${PNMAX_DRY_RUN}" != "1" ] && [ ! -x "${CINM_OPT}" ] \
    && [ "${PNMAX_SKIP_CINM:-0}" != "1" ]; then
    die "cinm-opt not found at ${CINM_OPT}.
  Fig. 10 includes CINM baseline points; ./setup.sh builds the vendored CINM
  by default. Re-run ./setup.sh (without --without-cinm), or re-run with
  PNMAX_SKIP_CINM=1 to render the figure without the CINM overlay."
  fi
fi

# ---------------------------------------------------------------------------
phase_banner "phase 1/4 — UniNDP baselines + DSE search pool + Pareto evals"
# ---------------------------------------------------------------------------
# Full scale at 64 workers (campaign-measured): baselines ~5 min, search pool
# ~80 min, Pareto evals ~5 h — ~6.5 h end-to-end. The pool is shared with the
# fig11/fig12/fig15/fig16 buttons and completed cells are reused.
ensure_fig10_inputs "${NUM_TRACES}" "${ARCHS[@]}"

# ---------------------------------------------------------------------------
phase_banner "phase 2/4 — OptiPIM searched proxy (best of 2x${NUM_TRACES} random mappings)"
# ---------------------------------------------------------------------------
step_start "second-seed HBM-PIM (a)-space searches (seed $((PNMAX_SEED + 1)))"
export PNMAX_SEARCH_SEED=$((PNMAX_SEED + 1))
for mode in false true; do
  for token in "${WORKLOAD_TOKENS[@]}"; do
    ensure_search_cell \
      hbm_pim "${ARCH_FILE_HBM}" \
      "${BASELINE_DIR}/${token}_hbm_pim.yaml" \
      "${RESULTS_ROOT}/optipim_proxy/random_search_seed2/no_streaming_${mode}/hbm_pim/${token}" \
      "${NUM_TRACES}" "${mode}" a
  done
done
unset PNMAX_SEARCH_SEED
step_end

step_start "proxy extraction (re-cost + best-of-N)"
run_py python "${SCRIPT_DIR}/baselines/extract_optipim_proxy.py" \
  --arch-file "${ARCH_FILE_HBM}" \
  --workers "${PNMAX_WORKERS}" \
  --output "${RESULTS_DIR}/optipim_proxy.csv"
step_end

# ---------------------------------------------------------------------------
phase_banner "phase 3/4 — CINM compiler-tiling baseline (UPMEM)"
# ---------------------------------------------------------------------------
if [ -x "${CINM_OPT}" ]; then
  step_start "cinm-opt tiling + re-cost"
  run_py python "${SCRIPT_DIR}/baselines/cinm_approach_a.py"
  step_end
else
  echo "CINM baseline SKIPPED: ${CINM_OPT} not built" \
    "($([ "${PNMAX_SMOKE}" = "1" ] && echo 'smoke mode tolerates this' || echo 'PNMAX_SKIP_CINM=1 set'));" \
    "the grid renders without CINM markers."
fi

# ---------------------------------------------------------------------------
phase_banner "phase 4/4 — render the Pareto grid"
# ---------------------------------------------------------------------------
step_start "pareto grid (latency-mem, Fig. 10)"
run_py python "${REPO_ROOT}/plot/workload_space_pareto_grid.py" \
  "${PARETO_EVAL_ROOT}" \
  --output-dir "${RESULTS_DIR}/figures" \
  --pairs latency_mem \
  --optipim-proxy-csv "${RESULTS_DIR}/optipim_proxy.csv" \
  --cinm-proxy-csv "${PNMAX_CINM_OUT}/cinm_proxy.csv"
run_cmd mv -f "${RESULTS_DIR}/figures/workload_space_pareto_grid_latency_mem.pdf" \
  "${RESULTS_DIR}/figures/fig10.pdf"
step_end

phase_banner "fig10_pareto done — figures: ${RESULTS_DIR}/figures"
