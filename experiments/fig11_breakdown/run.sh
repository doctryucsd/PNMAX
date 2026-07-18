#!/usr/bin/env bash
# experiments/fig11_breakdown/run.sh — Fig. 11: latency breakdown of the
# best-latency HBM-PIM mappings per DSE space (normalized stacked bars).
#
# Usage: ./run.sh [--smoke] [--dry-run] [--workers N] [--seed N]
#   default : full scale (shares the Fig 9 HBM-PIM search pool +
#             Pareto evals; completed cells are reused)
#   --smoke : minutes-long end-to-end check (1 kernel, 8 traces per space)
# Writes only under the results root (shared pool + fig11_breakdown/).
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../_lib/common.sh
. "${SCRIPT_DIR}/../_lib/common.sh"

pnmax_init "fig11_breakdown" "$@"
phase_banner "fig11_breakdown — Fig. 11: HBM-PIM latency breakdown per DSE space"

# shellcheck source=../_lib/pipeline.sh
. "${SCRIPT_DIR}/../_lib/pipeline.sh"

# ---------------------------------------------------------------------------
# Experiment defaults (single block; arch files via pipeline.sh).
# The figure only uses HBM-PIM runs.
# ---------------------------------------------------------------------------
FULL_NUM_TRACES=2048
SMOKE_NUM_TRACES=8
ARCHS=(hbm_pim)

if [ "${PNMAX_SMOKE}" = "1" ]; then
  NUM_TRACES="${SMOKE_NUM_TRACES}"
  WORKLOAD_TOKENS=(bert-3)
else
  NUM_TRACES="${FULL_NUM_TRACES}"
fi

# ---------------------------------------------------------------------------
phase_banner "phase 1/2 — ensure the Fig 9 HBM-PIM search pool + Pareto evals"
# ---------------------------------------------------------------------------
ensure_fig9_inputs "${NUM_TRACES}" "${ARCHS[@]}"

# ---------------------------------------------------------------------------
phase_banner "phase 2/2 — render the latency-breakdown bars"
# ---------------------------------------------------------------------------
step_start "latency breakdown (normalized stacked bars)"
run_py python "${REPO_ROOT}/plot/latency_breakdown.py" \
  "${PARETO_EVAL_ROOT}" \
  --output-dir "${RESULTS_DIR}/figures" \
  --normalized-only
run_cmd mv -f \
  "${RESULTS_DIR}/figures/workload_space_pareto_latency_breakdown_hbm_pim_streaming_only_normalized_to_a.pdf" \
  "${RESULTS_DIR}/figures/fig11.pdf"
step_end

phase_banner "fig11_breakdown done — figures: ${RESULTS_DIR}/figures"
