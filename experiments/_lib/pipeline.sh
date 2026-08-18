#!/usr/bin/env bash
# experiments/_lib/pipeline.sh — shared pipeline steps for the workload-space
# experiment family (Figs 10, 11, 12, 15, 16, 17). Source AFTER common.sh +
# pnmax_init.
#
# The DSE experiments share one mapping-search pool and one Pareto-eval
# tree under the results root so buttons can reuse each other's completed
# cells instead of re-searching:
#
#   <results-root>/unindp_baseline/<token>_<arch>.yaml     derived baselines
#   <results-root>/workload_space_random_search/
#       no_streaming_{false,true}/{upmem,hbm-pim}/<token>/spaces/{a..d}/
#   <results-root>/workload_space_pareto_eval/<arch>/<display>/<timestamp>/
#
# NOTE the legacy 'hbm-pim' (hyphen) spelling of the search-pool arch dir: the
# final plot scripts (plot/bspace_interconnect_sharing_bars.py) and the
# OptiPIM-proxy extractor hardcode it.
#
# Reuse rule (fresh-outdir contract): a search cell is reused only when its
# manifest.json exists and records the same requested spaces / per-space trace
# budget / streaming mode; otherwise the cell directory is deleted and searched
# fresh (the search CLI silently RESUMES into non-empty dirs, which would break
# determinism).

# ---- shared paths -----------------------------------------------------------
BASELINE_DIR="${RESULTS_ROOT}/unindp_baseline"
SEARCH_POOL_ROOT="${RESULTS_ROOT}/workload_space_random_search"
PARETO_EVAL_ROOT="${RESULTS_ROOT}/workload_space_pareto_eval"
export PNMAX_UNINDP_BASELINE_DIR="${BASELINE_DIR}"

# Per-pipeline experiment-default architecture files — the BASE
# (non-x16) baselines are what the Fig 8/10-12/15/17 buttons consume.
ARCH_FILE_HBM="${REPO_ROOT}/data/archs/lowered/baseline/hbm_pim.yaml"
ARCH_FILE_UPMEM="${REPO_ROOT}/data/archs/lowered/baseline/upmem.yaml"
ARCH_FILE_ATTACC="${REPO_ROOT}/data/archs/lowered/baseline/attacc.yaml"

# The 9 evaluation kernels (workload tokens) and the search-pool arch dir names.
WORKLOAD_TOKENS=(alexnet-4 bert-3 bert-72 llama128-6 llama256-3 llama512-3 resnet50-2 resnet50-53 vgg16-0)

# arch_pool_dir <arch> — legacy search-pool directory name for an arch.
arch_pool_dir() {
  case "$1" in
    hbm_pim) printf 'hbm-pim' ;;
    upmem) printf 'upmem' ;;
    *) die "arch_pool_dir: unknown arch '$1'" ;;
  esac
}

# arch_base_file <arch> — default arch YAML for an arch.
arch_base_file() {
  case "$1" in
    hbm_pim) printf '%s' "${ARCH_FILE_HBM}" ;;
    upmem) printf '%s' "${ARCH_FILE_UPMEM}" ;;
    *) die "arch_base_file: unknown arch '$1'" ;;
  esac
}

# ensure_unindp_baselines <token:arch> [...] — derive the requested UniNDP
# baselines into ${BASELINE_DIR} (reuses existing files; UniNDP compile is
# deterministic).
ensure_unindp_baselines() {
  run_py python "${REPO_ROOT}/experiments/_lib/derive_unindp_baselines.py" \
    --output-dir "${BASELINE_DIR}" \
    --work-dir "${RESULTS_ROOT}/unindp_baseline_work" \
    --pairs "$@"
}

# _search_cell_complete <cell-dir> <num-traces> <streaming-mode> <seed> <workers> <spaces...>
# Return 0 when the cell's manifest matches the request (reusable).
_search_cell_complete() {
  local cell=$1 num=$2 mode=$3 seed=$4 workers=$5
  shift 5
  local spaces="$*"
  [ -f "${cell}/manifest.json" ] || return 1
  python3 - "$cell" "$num" "$mode" "$seed" "$workers" $spaces <<'EOF'
import json, sys
cell, num, mode, seed, workers = (
    sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
)
spaces = sys.argv[6:]
try:
    m = json.load(open(f"{cell}/manifest.json"))
except Exception:
    raise SystemExit(1)
ok = (
    m.get("num_traces_per_space_delta") == num
    and m.get("all_streaming_false") == (mode == "true")
    and m.get("seed") == seed
    and m.get("workers") == workers
    and set(spaces).issubset(set(m.get("requested_spaces", [])))
)
raise SystemExit(0 if ok else 1)
EOF
}

# ensure_search_cell <arch> <arch-file> <workload-yaml> <cell-dir> \
#                    <num-traces> <streaming true|false> <spaces...>
# Populate one search-pool cell (skip when a matching complete cell exists).
ensure_search_cell() {
  local arch=$1 arch_file=$2 workload=$3 cell=$4 num=$5 mode=$6
  shift 6
  local spaces=("$@")

  if [ "${PNMAX_DRY_RUN}" != "1" ] \
    && _search_cell_complete "${cell}" "${num}" "${mode}" "${PNMAX_SEARCH_SEED:-${PNMAX_SEED}}" "${PNMAX_WORKERS}" "${spaces[@]}"; then
    printf 'search cell reused: %s\n' "${cell}"
    return 0
  fi
  fresh_dir "${cell}"
  local extra=()
  if [ "${mode}" = "true" ]; then
    extra+=(--all-streaming-false)
  fi
  # PNMAX_SEARCH_SEED lets a driver run an independent second search
  # (e.g. the OptiPIM proxy's seed+1 pool) without changing the base seed.
  # No --no-resume: multi-space runs feed later stages from earlier ones and
  # would trip it; fresh_dir above already guarantees a fresh outdir.
  run_py run-workload-space-random-search \
    --workload "${workload}" \
    --arch "${arch}" \
    --arch-file "${arch_file}" \
    --outdir "${cell}" \
    --num-traces "${num}" \
    --max-attempts 1000000000 \
    --spaces "${spaces[@]}" \
    --workers "${PNMAX_WORKERS}" \
    --batch-attempts 1 \
    --seed "${PNMAX_SEARCH_SEED:-${PNMAX_SEED}}" \
    "${extra[@]}"

  # Integrity gate: every requested space must have materialized traces —
  # fail here (with the cell path) rather than in a downstream plot.
  if [ "${PNMAX_DRY_RUN}" != "1" ]; then
    local space
    for space in "${spaces[@]}"; do
      if ! ls "${cell}/spaces/${space}"/trace_*.yaml >/dev/null 2>&1; then
        die "search cell ${cell} finished without traces for space '${space}'
  (seed ${PNMAX_SEARCH_SEED:-${PNMAX_SEED}}, ${num} traces requested) — the cell is incomplete; re-run this button."
      fi
    done
  fi
}

# ensure_pool_cells <num-traces> <arch...> — populate the standard pool
# (both streaming modes, spaces a-d, all 9 kernels) for the given archs.
ensure_pool_cells() {
  local num=$1
  shift
  local arch token mode cell pool_arch
  for arch in "$@"; do
    pool_arch=$(arch_pool_dir "${arch}")
    for token in "${WORKLOAD_TOKENS[@]}"; do
      for mode in false true; do
        cell="${SEARCH_POOL_ROOT}/no_streaming_${mode}/${pool_arch}/${token}"
        ensure_search_cell \
          "${arch}" "$(arch_base_file "${arch}")" \
          "${BASELINE_DIR}/${token}_${arch}.yaml" \
          "${cell}" "${num}" "${mode}" a b c d
      done
    done
  done
}

# latest_pareto_eval_dir <arch> <token> — newest eval run dir for a kernel
# (empty when none). Eval output dirs are named by workload display name
# (e.g. fc_bert-72), which ends with the token.
latest_pareto_eval_dir() {
  local arch=$1 token=$2 d latest=""
  for d in "${PARETO_EVAL_ROOT}/${arch}"/*"${token}"/*/; do
    [ -f "${d}/summary.json" ] || continue
    latest="${d}"
  done
  printf '%s' "${latest%/}"
}

# _eval_fingerprint_reusable <eval-dir> <seed> <workers> <trace-budget>
# Reuse policy for a cached Pareto eval: an ABSENT eval_fingerprint.json means
# reuse (grandfathers the full-campaign eval tree, which predates fingerprints);
# a PRESENT fingerprint must match seed/workers/trace budget to be reusable, so
# e.g. a seed-43 rerun over a seed-42 root rebuilds rather than mixing pools.
_eval_fingerprint_reusable() {
  local dir=$1 seed=$2 workers=$3 num=$4
  [ -f "${dir}/eval_fingerprint.json" ] || return 0
  python3 - "${dir}/eval_fingerprint.json" "$seed" "$workers" "$num" <<'EOF'
import json, sys
fp, seed, workers, num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
try:
    d = json.load(open(fp))
except Exception:
    raise SystemExit(0)  # unreadable fingerprint: do not block reuse
ok = (d.get("seed") == seed and d.get("workers") == workers
      and d.get("trace_budget") == num)
raise SystemExit(0 if ok else 1)
EOF
}

# _write_eval_fingerprint <eval-dir> <seed> <workers> <trace-budget>
# Record the parameters a fresh eval was produced under, beside its summary.json.
_write_eval_fingerprint() {
  local dir=$1 seed=$2 workers=$3 num=$4
  [ -n "${dir}" ] || return 0
  python3 - "${dir}/eval_fingerprint.json" "$seed" "$workers" "$num" <<'EOF'
import json, sys
out, seed, workers, num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(out, "w") as f:
    json.dump({"seed": seed, "workers": workers, "trace_budget": num}, f, indent=2)
EOF
}

# ensure_pareto_eval <arch> <token> <num-traces> — evaluate one kernel's search
# cells into the shared Pareto-eval tree (skips when a REUSABLE eval run exists;
# see _eval_fingerprint_reusable).
ensure_pareto_eval() {
  local arch=$1 token=$2 num=$3
  local pool_arch
  pool_arch=$(arch_pool_dir "${arch}")
  local false_dir="${SEARCH_POOL_ROOT}/no_streaming_false/${pool_arch}/${token}"
  local true_dir="${SEARCH_POOL_ROOT}/no_streaming_true/${pool_arch}/${token}"

  if [ "${PNMAX_DRY_RUN}" != "1" ]; then
    local existing
    existing=$(latest_pareto_eval_dir "${arch}" "${token}")
    if [ -n "${existing}" ] \
      && _eval_fingerprint_reusable "${existing}" "${PNMAX_SEED}" "${PNMAX_WORKERS}" "${num}"; then
      printf 'pareto eval reused: %s\n' "${existing}"
      return 0
    fi
  fi
  run_py run-workload-space-pareto-eval \
    --workload-dirs "${false_dir}" "${true_dir}" \
    --arch "${arch}" \
    --arch-file "$(arch_base_file "${arch}")" \
    --baseline-workload "${BASELINE_DIR}/${token}_${arch}.yaml" \
    --output-root "${PARETO_EVAL_ROOT}" \
    --workers "${PNMAX_WORKERS}" \
    --verbose 0
  if [ "${PNMAX_DRY_RUN}" != "1" ]; then
    _write_eval_fingerprint "$(latest_pareto_eval_dir "${arch}" "${token}")" \
      "${PNMAX_SEED}" "${PNMAX_WORKERS}" "${num}"
  fi
}

# ensure_fig10_inputs <num-traces> <arch...> — baselines + pool + eval for the
# given archs (the shared front half of Figs 10/11/12).
ensure_fig10_inputs() {
  local num=$1
  shift
  local arch token pairs=()
  for arch in "$@"; do
    for token in "${WORKLOAD_TOKENS[@]}"; do
      pairs+=("${token}:${arch}")
    done
  done
  step_start "UniNDP baseline derivation (${#pairs[@]} pairs)"
  ensure_unindp_baselines "${pairs[@]}"
  step_end
  step_start "mapping-space random search pool"
  ensure_pool_cells "${num}" "$@"
  step_end
  step_start "Pareto evaluation"
  for arch in "$@"; do
    for token in "${WORKLOAD_TOKENS[@]}"; do
      ensure_pareto_eval "${arch}" "${token}" "${num}"
    done
  done
  step_end
}
