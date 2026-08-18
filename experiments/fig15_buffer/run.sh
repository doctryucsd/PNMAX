#!/usr/bin/env bash
# experiments/fig15_buffer/run.sh — Fig. 15: buffer-size sensitivity of
# selected fc_bert-72 mappings on HBM-PIM (128B..2KB register-file variants).
#
# Usage: ./run.sh [--smoke] [--dry-run] [--workers N] [--seed N]
#   default : full scale (2048-trace search pool; 30x3 sampled sets)
#   --smoke : minutes-long end-to-end check (8 traces, 4x2 sampled sets)
# Writes only under the results root (shared pool + fig15_buffer/).
#
# Pipeline: (1) UniNDP baseline + fc_bert-72 mapping pool (shared with Fig 10),
# (2) buffer-sweep random-set latency evaluation over the 5 buffer variants,
# (3) render the selected-mappings figure.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../_lib/common.sh
. "${SCRIPT_DIR}/../_lib/common.sh"

pnmax_init "fig15_buffer" "$@"
phase_banner "fig15_buffer — Fig. 15: buffer-size sensitivity (fc_bert-72 on HBM-PIM)"

# shellcheck source=../_lib/pipeline.sh
. "${SCRIPT_DIR}/../_lib/pipeline.sh"

# ---------------------------------------------------------------------------
# Experiment defaults (single block). Arch basis:
# data/archs/lowered/buffer_sweep/hbm_pim (b2b x1 variants), restricted to
# the figure's 5 buffer sizes (128B..2KB, midpoint 512B) — the generated
# family also carries 32B/64B variants that Fig 15 does not include, so the
# driver materializes a 5-variant subset.
# ---------------------------------------------------------------------------
TOKEN=bert-72                       # workload display name: fc_bert-72
BUFFER_ARCH_SRC="${REPO_ROOT}/data/archs/lowered/buffer_sweep/hbm_pim"
BUFFER_VARIANTS=(
  hbm_pim__pu-8__hmat-16__vmat-32_buffer-128B.yaml
  hbm_pim__pu-8__hmat-16__vmat-32_buffer-256B.yaml
  hbm_pim__pu-8__hmat-16__vmat-32.yaml
  hbm_pim__pu-8__hmat-16__vmat-32_buffer-1KB.yaml
  hbm_pim__pu-8__hmat-16__vmat-32_buffer-2KB.yaml
)
BUFFER_ARCH_ROOT="${RESULTS_DIR}/arch_subset"
FULL_NUM_TRACES=2048
FULL_SAMPLE_COUNT=30                # full scale: 30 sampled sets
FULL_SET_SIZE=3                     #   of 3 mappings each
SMOKE_NUM_TRACES=8
SMOKE_SAMPLE_COUNT=4
SMOKE_SET_SIZE=2

if [ "${PNMAX_SMOKE}" = "1" ]; then
  NUM_TRACES="${SMOKE_NUM_TRACES}"
  SAMPLE_COUNT="${SMOKE_SAMPLE_COUNT}"
  SET_SIZE="${SMOKE_SET_SIZE}"
else
  NUM_TRACES="${FULL_NUM_TRACES}"
  SAMPLE_COUNT="${FULL_SAMPLE_COUNT}"
  SET_SIZE="${FULL_SET_SIZE}"
fi

POOL_CELL="${SEARCH_POOL_ROOT}/no_streaming_false/$(arch_pool_dir hbm_pim)/${TOKEN}"
EVAL_ROOT="${RESULTS_DIR}/random_set_eval"

# ---------------------------------------------------------------------------
phase_banner "phase 1/3 — UniNDP baseline + fc_bert-72 mapping pool"
# ---------------------------------------------------------------------------
step_start "UniNDP baseline (bert-72 on hbm_pim)"
ensure_unindp_baselines "${TOKEN}:hbm_pim"
step_end
step_start "mapping pool (spaces a-d, shared with Fig 10)"
ensure_search_cell \
  hbm_pim "${ARCH_FILE_HBM}" \
  "${BASELINE_DIR}/${TOKEN}_hbm_pim.yaml" \
  "${POOL_CELL}" "${NUM_TRACES}" false a b c d
step_end

# ---------------------------------------------------------------------------
phase_banner "phase 2/3 — buffer-sweep random-set latency evaluation"
# ---------------------------------------------------------------------------
step_start "5-variant arch subset"
fresh_dir "${BUFFER_ARCH_ROOT}/hbm_pim"
for variant in "${BUFFER_VARIANTS[@]}"; do
  run_cmd ln -s "${BUFFER_ARCH_SRC}/${variant}" "${BUFFER_ARCH_ROOT}/hbm_pim/${variant}"
done
step_end

step_start "random-set eval over 5 buffer variants (full: ~1 min)"
run_dir_file="${RESULTS_DIR}/latest_run_dir.txt"
run_py run-buffer-sweep-workload-space-random-set-latency-eval \
  --mapping-dir "${POOL_CELL}" \
  --arch hbm_pim \
  --arch-root "${BUFFER_ARCH_ROOT}" \
  --baseline-root "${BASELINE_DIR}" \
  --output-root "${EVAL_ROOT}" \
  --output-dir-file "${run_dir_file}" \
  --workers "${PNMAX_WORKERS}" \
  --sample-count "${SAMPLE_COUNT}" \
  --set-size "${SET_SIZE}" \
  --seed "${PNMAX_SEED}" \
  --verbose 0
step_end

if [ "${PNMAX_DRY_RUN}" = "1" ]; then
  RUN_DIR="<eval run dir from ${run_dir_file}>"
else
  RUN_DIR=$(tr -d '\r\n' < "${run_dir_file}")
  require_dir "${RUN_DIR}" "the buffer-sweep evaluation did not report its run dir"
fi

# ---------------------------------------------------------------------------
phase_banner "phase 3/3 — render Fig. 15"
# ---------------------------------------------------------------------------
step_start "selected-mappings latency figure"
run_py python "${REPO_ROOT}/plot/buffer_sweep_workload_space_random_set_latency.py" \
  "${RUN_DIR}" \
  --output-dir "${RESULTS_DIR}/figures"
run_cmd mv -f "${RESULTS_DIR}/figures/selected_mappings.pdf" \
  "${RESULTS_DIR}/figures/fig15.pdf"
step_end

phase_banner "fig15_buffer done — figures: ${RESULTS_DIR}/figures"
