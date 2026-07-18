#!/usr/bin/env bash
# experiments/fig15_reduction/run.sh — Fig. 15: reduction placement
# (bank / channel / base-die) Pareto fronts for bmm_llama256-3 on HBM-PIM,
# normalized to the AttAcc-style point.
#
# Usage: ./run.sh [--smoke] [--dry-run] [--workers N] [--seed N]
#   default : full scale (2048-trace pools)
#   --smoke : minutes-long end-to-end check (8-trace pools)
# Writes only under the results root (fig15_reduction/ + unindp_baseline/).
#
# Pipeline (two-stage inclusive semantics, encoded in the driver):
# (1) UniNDP baseline; (2) llama256-3 mapping pool on the baseline geometry
# arch (spaces c, direct-c) + an L4-capped pool (direct-l4); (3) inclusive
# reduction-place evaluation (config1 = bank/PU search + re-cost, config2 =
# base-die over the composed pool, config3 = channel pool capped at L4);
# (4) channel PU-union re-cost (channel = config1 mappings re-cost under
# channel_level ∪ config3 pool); (5) Pareto plot.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../_lib/common.sh
. "${SCRIPT_DIR}/../_lib/common.sh"

pnmax_init "fig15_reduction" "$@"
phase_banner "fig15_reduction — Fig. 15: reduction placement (bank/channel/base-die)"

# shellcheck source=../_lib/pipeline.sh
. "${SCRIPT_DIR}/../_lib/pipeline.sh"

# ---------------------------------------------------------------------------
# Experiment defaults (single block). Arch basis:
# Fig 15 uses data/archs/lowered/activation/{hbm_pim__pu-8__hmat-16
# __vmat-32,base_die,channel_level}.yaml (config1/2/3) with mapping pools
# searched on the pinned copy under lowered/activation/search_base/ (frozen
# characterization inputs at the eval archs' vintage — see the README there;
# the generated geometry_sweep family belongs to fig12).
# ---------------------------------------------------------------------------
TOKEN=llama256-3
SEARCH_ARCH_FILE="${REPO_ROOT}/data/archs/lowered/activation/search_base/hbm_pim__pu-8__hmat-16__vmat-32.yaml"
CONFIG1_ARCH="${REPO_ROOT}/data/archs/lowered/activation/hbm_pim__pu-8__hmat-16__vmat-32.yaml"
CONFIG2_ARCH="${REPO_ROOT}/data/archs/lowered/activation/base_die.yaml"
CONFIG3_ARCH="${REPO_ROOT}/data/archs/lowered/activation/channel_level.yaml"
MAX_SYSTEM_SHARING=4      # channel pool capped at L4
FULL_NUM_TRACES=2048
SMOKE_NUM_TRACES=8

if [ "${PNMAX_SMOKE}" = "1" ]; then
  NUM_TRACES="${SMOKE_NUM_TRACES}"
else
  NUM_TRACES="${FULL_NUM_TRACES}"
fi

GEOMETRY_POOL="${RESULTS_DIR}/geometry_search/no_streaming_false/hbm_pim/hbm_pim__pu-8__hmat-16__vmat-32/${TOKEN}"
L4_POOL="${RESULTS_DIR}/l4_pool/hbm_pim/hbm_pim__pu-8__hmat-16__vmat-32/${TOKEN}"
INCLUSIVE_ROOT="${RESULTS_DIR}/pareto_eval_channel_inclusive"
PUUNION_ROOT="${RESULTS_DIR}/pareto_eval_channel_inclusive_PUunion/hbm_pim"

# _fig15_search <cell> [extra flags...] — direct-c / direct-l4 pool searches
# on the baseline geometry arch (spaces c only).
_fig15_search() {
  local cell=$1
  shift
  fresh_dir "${cell}"
  run_py run-workload-space-random-search \
    --workload "${BASELINE_DIR}/${TOKEN}_hbm_pim.yaml" \
    --arch hbm_pim \
    --arch-file "${SEARCH_ARCH_FILE}" \
    --outdir "${cell}" \
    --num-traces "${NUM_TRACES}" \
    --max-attempts 1000000000 \
    --spaces c \
    --workers "${PNMAX_WORKERS}" \
    --batch-attempts 1 \
    --seed "${PNMAX_SEED}" \
    "$@"
}

# ---------------------------------------------------------------------------
phase_banner "phase 1/5 — UniNDP baseline (llama256-3 on hbm_pim)"
# ---------------------------------------------------------------------------
step_start "UniNDP baseline"
ensure_unindp_baselines "${TOKEN}:hbm_pim"
step_end

# ---------------------------------------------------------------------------
phase_banner "phase 2/5 — mapping pools on the baseline geometry arch"
# ---------------------------------------------------------------------------
step_start "direct-c pool (config1/config2 source)"
_fig15_search "${GEOMETRY_POOL}" --direct-c-search
step_end
step_start "direct-l4 pool (channel config3 source)"
_fig15_search "${L4_POOL}" --direct-l4-search
step_end

# ---------------------------------------------------------------------------
phase_banner "phase 3/5 — reduction-place evaluations (full: ~1.2 h at 64 workers)"
# ---------------------------------------------------------------------------
# The inclusive semantics are two-stage by construction: a plain
# activation evaluation first (it SEARCHES the config1/PU mapping set and
# evaluates config1+config2), then the inclusive run, which re-scores that
# config1 set and composes the inclusive config2/config3 pools.
step_start "stage A: plain activation eval (config1 search + config1/2)"
plain_dir_file="${RESULTS_DIR}/latest_plain_run_dir.txt"
run_py run-activation-workload-space-latency-eval \
  --mapping-dirs "${GEOMETRY_POOL}" \
  --arch hbm_pim \
  --output-root "${RESULTS_DIR}/activation_eval" \
  --output-dir-file "${plain_dir_file}" \
  --config1-arch-file "${CONFIG1_ARCH}" \
  --config2-arch-file "${CONFIG2_ARCH}" \
  --num-traces "${NUM_TRACES}" \
  --max-attempts 1000000000 \
  --workers "${PNMAX_WORKERS}" \
  --verbose 0
# (no --seed flag: the internal config1 search reads the exported PNMAX_SEED)
step_end

if [ "${PNMAX_DRY_RUN}" = "1" ]; then
  PLAIN_RUN_DIR="<stage A run dir from ${plain_dir_file}>"
else
  PLAIN_RUN_DIR=$(tr -d '\r\n' < "${plain_dir_file}")
  require_dir "${PLAIN_RUN_DIR}" "stage A did not report its run dir"
fi

step_start "stage B: inclusive eval (config1 reuse + config2/config3 pools)"
run_dir_file="${RESULTS_DIR}/latest_run_dir.txt"
run_py run-activation-workload-space-latency-eval \
  --mapping-dirs "${GEOMETRY_POOL}" \
  --arch hbm_pim \
  --output-root "${INCLUSIVE_ROOT}" \
  --output-dir-file "${run_dir_file}" \
  --inclusive-reduction-place \
  --config1-reuse-root "${PLAIN_RUN_DIR}" \
  --config1-arch-file "${CONFIG1_ARCH}" \
  --config2-arch-file "${CONFIG2_ARCH}" \
  --config3-arch-file "${CONFIG3_ARCH}" \
  --config3-mapping-dirs "${L4_POOL}" \
  --max-system-sharing "${MAX_SYSTEM_SHARING}" \
  --num-traces "${NUM_TRACES}" \
  --max-attempts 1000000000 \
  --workers "${PNMAX_WORKERS}" \
  --verbose 0
step_end

if [ "${PNMAX_DRY_RUN}" = "1" ]; then
  RUN_DIR="<stage B run dir from ${run_dir_file}>"
  CONFIG1_TRACES="${PLAIN_RUN_DIR}/random_search/config1/${TOKEN}/spaces/d"
else
  RUN_DIR=$(tr -d '\r\n' < "${run_dir_file}")
  require_dir "${RUN_DIR}" "the inclusive evaluation did not report its run dir"
  CONFIG1_TRACES=$(find "${PLAIN_RUN_DIR}/random_search/config1" -type d -name '[a-d]' -path '*/spaces/*' | sort | tail -1)
  require_dir "${CONFIG1_TRACES}" "no config1 trace dir under ${PLAIN_RUN_DIR}/random_search/config1"
fi

# ---------------------------------------------------------------------------
phase_banner "phase 4/5 — channel PU-union re-cost"
# ---------------------------------------------------------------------------
step_start "config1 mappings re-cost under channel_level + pool union"
run_py python "${SCRIPT_DIR}/puunion_recost.py" \
  --token "${TOKEN}" \
  --config1-trace-dir "${CONFIG1_TRACES}" \
  --inclusive-csv "${RUN_DIR}/evaluations.csv" \
  --channel-arch "${CONFIG3_ARCH}" \
  --out-root "${PUUNION_ROOT}" \
  --workers "${PNMAX_WORKERS}"
step_end

# ---------------------------------------------------------------------------
phase_banner "phase 5/5 — render Fig. 15"
# ---------------------------------------------------------------------------
step_start "reduction-place Pareto front (lat-mem, Fig. 15)"
run_py python "${REPO_ROOT}/plot/reduction_place_pareto.py" \
  --input-root "${INCLUSIVE_ROOT}/hbm_pim" \
  --channel-input-root "${PUUNION_ROOT}" \
  --output-dir "${RESULTS_DIR}/figures" \
  --pairs lat-mem
run_cmd mv -f "${RESULTS_DIR}/figures/reduction_place_pareto_${TOKEN}_lat-mem.pdf" \
  "${RESULTS_DIR}/figures/fig15.pdf"
step_end

phase_banner "fig15_reduction done — figures: ${RESULTS_DIR}/figures"
