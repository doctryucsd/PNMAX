#!/usr/bin/env bash
# experiments/fig08_tab4_validation/run.sh — Fig. 8 + Table 4: analytical model
# vs. simulator validation (UniNDP for HBM-PIM/UPMEM, AttAcc for AttAcc).
#
# Usage: ./run.sh [--smoke] [--dry-run] [--workers N] [--seed N]
#   default : full scale (2000 mappings/arch vs UniNDP, ~1013 vs AttAcc)
#   --smoke : minutes-long end-to-end check (tiny mapping sets)
# Writes only under <results-root>/fig08_tab4_validation/.
#
# Pipeline: (1) generate the deterministic random-mapping validation sets,
# (2) run each mapping through simulator + analytical model, (3) render the
# 2x2 validation figure (the Table 4 fields live in the validation JSONs).
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../_lib/common.sh
. "${SCRIPT_DIR}/../_lib/common.sh"

pnmax_init "fig08_tab4_validation" "$@"
phase_banner "fig08_tab4_validation — Fig. 8 + Table 4: model-vs-simulator validation"

# ---------------------------------------------------------------------------
# Experiment defaults (single block; see docs). Arch files: figs 8-11
# consume the BASE lowered baselines.
# ---------------------------------------------------------------------------
ARCH_FILE_HBM="${REPO_ROOT}/data/archs/lowered/baseline/hbm_pim.yaml"
ARCH_FILE_UPMEM="${REPO_ROOT}/data/archs/lowered/baseline/upmem.yaml"
ARCH_FILE_ATTACC="${REPO_ROOT}/data/archs/lowered/baseline/attacc.yaml"
SAMPLE_RATE=0.05          # UniNDP instruction-sampling rate (full-scale default)
FULL_NUM_TRACES=2048      # generated mappings per UniNDP arch (full-scale budget)
FULL_NUM_WORKLOADS=2000   # validated mappings per UniNDP arch (Table 4)
SMOKE_NUM_TRACES=8
ATTACC_MODE_FULL=full     # run-random-search-attacc full mode (~1013 unique)
SMOKE_ATTACC_TRACES=8

if [ "${PNMAX_SMOKE}" = "1" ]; then
  NUM_TRACES="${SMOKE_NUM_TRACES}"
  NUM_WORKLOADS="${SMOKE_NUM_TRACES}"
  # One kernel template per arch keeps the smoke search in seconds.
  UNINDP_WORKLOADS=("${REPO_ROOT}/data/workloads/b_fc/bert-3-unindp.yaml")
else
  NUM_TRACES="${FULL_NUM_TRACES}"
  NUM_WORKLOADS="${FULL_NUM_WORKLOADS}"
  UNINDP_WORKLOADS=(
    "${REPO_ROOT}/data/workloads/b_fc/bert-3-unindp.yaml"
    "${REPO_ROOT}/data/workloads/b_fc/llama256-3-unindp.yaml"
    "${REPO_ROOT}/data/workloads/b_fc/llama512-3-unindp.yaml"
    "${REPO_ROOT}/data/workloads/conv2d/alexnet-4-unindp.yaml"
    "${REPO_ROOT}/data/workloads/conv2d/resnet50-2-unindp.yaml"
    "${REPO_ROOT}/data/workloads/conv2d/vgg16-0-unindp.yaml"
    "${REPO_ROOT}/data/workloads/fc/bert-72-unindp.yaml"
    "${REPO_ROOT}/data/workloads/fc/llama128-6-unindp.yaml"
    "${REPO_ROOT}/data/workloads/fc/resnet50-53-unindp.yaml"
  )
fi

TRACES_DIR="${RESULTS_DIR}/validation_traces"
VALIDATION_DIR="${RESULTS_DIR}/validation"
RAMULATOR_DIR="${REPO_ROOT}/external/attacc/ramulator2"
RAMULATOR_BIN="${RAMULATOR_DIR}/build/ramulator2"
GENERATOR_SCRIPT="${REPO_ROOT}/external/attacc/pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py"
require_file "${RAMULATOR_BIN}" "build the AttAcc simulator first: ./setup.sh"

# ---------------------------------------------------------------------------
phase_banner "phase 1/3 — generate validation mapping sets (seeded random search)"
# ---------------------------------------------------------------------------
for arch in hbm_pim upmem; do
  case "${arch}" in
    hbm_pim) arch_file="${ARCH_FILE_HBM}" ;;
    upmem) arch_file="${ARCH_FILE_UPMEM}" ;;
  esac
  step_start "mapping set: ${arch} (${NUM_TRACES} traces)"
  fresh_dir "${TRACES_DIR}/${arch}"
  run_py run-random-search \
    --workload "${UNINDP_WORKLOADS[@]}" \
    --arch "${arch_file}" \
    --outdir "${TRACES_DIR}/${arch}" \
    --num-traces "${NUM_TRACES}" \
    --max-attempts 10000000000 \
    --workers "${PNMAX_WORKERS}" \
    --batch-attempts 128 \
    --seed "${PNMAX_SEED}"
  step_end
done

step_start "mapping set: attacc"
fresh_dir "${TRACES_DIR}/attacc"
attacc_gen_args=(
  --workload "${REPO_ROOT}/data/workloads/samples/attacc_gemv.yaml"
  --outdir "${TRACES_DIR}/attacc"
  --mode "${ATTACC_MODE_FULL}"
  --workers "${PNMAX_WORKERS}"
  --seed "${PNMAX_SEED}"
)
if [ "${PNMAX_SMOKE}" = "1" ]; then
  attacc_gen_args+=(--num-traces "${SMOKE_ATTACC_TRACES}")
fi
run_py run-random-search-attacc "${attacc_gen_args[@]}"
step_end

# ---------------------------------------------------------------------------
phase_banner "phase 2/3 — simulator vs analytical validation runs"
# ---------------------------------------------------------------------------
if [ "${PNMAX_DRY_RUN}" != "1" ]; then
  mkdir -p "${VALIDATION_DIR}"
fi

step_start "UniNDP validation: upmem (full: ~1.5 min at 64 workers)"
run_py run-unindp-validation \
  --workload_dir "${TRACES_DIR}/upmem" \
  --num_workloads "${NUM_WORKLOADS}" \
  --arch_file "${ARCH_FILE_UPMEM}" \
  --target upmem \
  --sample_rate "${SAMPLE_RATE}" \
  --single_pu \
  --workers "${PNMAX_WORKERS}" \
  --output_json "${VALIDATION_DIR}/upmem.json" \
  --verbose 0
step_end

step_start "UniNDP validation: hbm_pim (full: ~2.5 min at 64 workers)"
run_py run-unindp-validation \
  --workload_dir "${TRACES_DIR}/hbm_pim" \
  --num_workloads "${NUM_WORKLOADS}" \
  --arch_file "${ARCH_FILE_HBM}" \
  --target hbm_pim \
  --sample_rate "${SAMPLE_RATE}" \
  --single_pu \
  --workers "${PNMAX_WORKERS}" \
  --output_json "${VALIDATION_DIR}/hbm_pim.json" \
  --verbose 0
step_end

step_start "AttAcc validation (latency + energy; full: ~15 s)"
if [ "${PNMAX_DRY_RUN}" = "1" ]; then
  attacc_count="<count of generated attacc traces>"
else
  attacc_count=$(find "${TRACES_DIR}/attacc" -maxdepth 1 -name 'trace_*.yaml' | wc -l)
  [ "${attacc_count}" -ge 1 ] || die "no attacc validation traces were generated"
fi
run_py run-attacc-validation \
  --workload_dir "${TRACES_DIR}/attacc" \
  --num_workloads "${attacc_count}" \
  --arch_file "${ARCH_FILE_ATTACC}" \
  --target attacc \
  --ramulator_dir "${RAMULATOR_DIR}" \
  --ramulator_bin "${RAMULATOR_BIN}" \
  --generator_script "${GENERATOR_SCRIPT}" \
  --workers "${PNMAX_WORKERS}" \
  --output_json "${VALIDATION_DIR}/attacc.json" \
  --verbose 0
# The energy panel reads attacc_energy.json; one modern validation run carries
# both the latency fields and the traffic-derived energy fields.
run_cmd cp -f "${VALIDATION_DIR}/attacc.json" "${VALIDATION_DIR}/attacc_energy.json"
step_end

# ---------------------------------------------------------------------------
phase_banner "phase 3/3 — render Fig. 8"
# ---------------------------------------------------------------------------
step_start "combined validation figure (2x2)"
run_py python "${REPO_ROOT}/plot/combined_validation_lat_energy.py" \
  "${VALIDATION_DIR}" \
  --output-2x2 "${RESULTS_DIR}/figures/fig08.pdf" \
  --skip-4x1
step_end

phase_banner "fig08_tab4_validation done — figures: ${RESULTS_DIR}/figures"
