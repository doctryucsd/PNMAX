#!/usr/bin/env bash
# experiments/fig11_streaming/run.sh — Fig. 11: latency/footprint trade-off of
# DISABLING streaming (percent change of each front's best-latency mapping,
# faceted by architecture).
#
# Usage: ./run.sh [--smoke] [--dry-run] [--workers N] [--seed N]
#   default : full scale (shares the Fig 10 search pool + Pareto evals;
#             completed cells are reused, so after fig10_pareto only the
#             render remains)
#   --smoke : minutes-long end-to-end check (1 kernel, 8 traces per space)
# Writes only under the results root (shared pool + fig11_streaming/).
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../_lib/common.sh
. "${SCRIPT_DIR}/../_lib/common.sh"

pnmax_init "fig11_streaming" "$@"
phase_banner "fig11_streaming — Fig. 11: no-streaming latency/footprint trade-off"

# shellcheck source=../_lib/pipeline.sh
. "${SCRIPT_DIR}/../_lib/pipeline.sh"

# ---------------------------------------------------------------------------
# Experiment defaults (single block; arch files via pipeline.sh).
# ---------------------------------------------------------------------------
FULL_NUM_TRACES=2048
SMOKE_NUM_TRACES=8
ARCHS=(upmem hbm_pim)

if [ "${PNMAX_SMOKE}" = "1" ]; then
  NUM_TRACES="${SMOKE_NUM_TRACES}"
  WORKLOAD_TOKENS=(bert-3)
else
  NUM_TRACES="${FULL_NUM_TRACES}"
fi

# ---------------------------------------------------------------------------
phase_banner "phase 1/2 — ensure the Fig 10 search pool + Pareto evals"
# ---------------------------------------------------------------------------
ensure_fig10_inputs "${NUM_TRACES}" "${ARCHS[@]}"

# ---------------------------------------------------------------------------
phase_banner "phase 2/2 — render the streaming trade-off scatter"
# ---------------------------------------------------------------------------
step_start "streaming trade-off scatter"
run_py python "${REPO_ROOT}/plot/streaming_tradeoff_scatter.py" \
  --input-dir "${PARETO_EVAL_ROOT}" \
  --output "${RESULTS_DIR}/figures/fig11.pdf"
step_end

phase_banner "fig11_streaming done — figures: ${RESULTS_DIR}/figures"
