#!/usr/bin/env bash
# experiments/fig12_geometry/run.sh — Fig. 12: fixed-capacity DRAM geometry
# sweeps ((a) PU tile size, (b) BG x row-size inverse, (c) BG x #rows inverse)
# for OPT-2.7B end-to-end on UPMEM + HBM-PIM.
#
# Usage: ./run.sh [--smoke] [--dry-run] [--workers N] [--seed N]
#   default : full scale (2048 traces per (geometry, layer); the
#             end-to-end sweep also covers the other two end-to-end models)
#   --smoke : minutes-long end-to-end check (2 geometries/family, 1 extreme
#             rung, 8 traces; missing rungs render as gaps)
# Writes only under the results root (fig12_geometry/).
#
# Pipeline: (1) reconstruct nn_models/layer_params.csv from the end-to-end
# workload manifests into a PNMAX_ROOT shadow root (layer_params is not
# shipped); (2) geometry-sweep end-to-end random searches for
# both arch families; (3) the /4, x4 and burst extreme rungs (searched
# separately); (4) re-cost every searched OPT mapping on the Fig 12 cost
# bases and render.
#
# Cost bases: searches run on lowered/geometry_sweep/**;
# latency/energy re-cost basis = lowered/geometry_sweep_host_congested/
# iso_capacity_archs/**; area basis = lowered/geometry_sweep/iso_capacity_archs/**
# (encoded in plot/geometry_sweep_abc_opt.py).
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../_lib/common.sh
. "${SCRIPT_DIR}/../_lib/common.sh"

pnmax_init "fig12_geometry" "$@"
phase_banner "fig12_geometry — Fig. 12: fixed-capacity geometry sweeps (OPT-2.7B)"

# shellcheck source=../_lib/pipeline.sh
. "${SCRIPT_DIR}/../_lib/pipeline.sh"

# ---------------------------------------------------------------------------
# Experiment defaults (single block).
# ---------------------------------------------------------------------------
FULL_NUM_TRACES=2048
SMOKE_NUM_TRACES=8
GEOMETRY_ROOT="${REPO_ROOT}/data/archs/lowered/geometry_sweep"        # search basis
EXTREME_LAT_BASIS="${REPO_ROOT}/data/archs/lowered/geometry_sweep_host_congested/iso_capacity_archs"
EXTREME_AREA_BASIS="${REPO_ROOT}/data/archs/lowered/geometry_sweep/iso_capacity_archs"
E2E_ROOT="${RESULTS_DIR}/geometry_sweep_end_to_end_workload_space"
EXTREMES_ROOT="${RESULTS_DIR}/geometry_sweep_abc_opt_extremes/random_search/opt-2.7B-2048"
SHADOW_ROOT="${RESULTS_DIR}/shadow_root"
OPT_TOKENS=(bmm_n32_c2048_k80 bmm_n32_c80_k2048 fc_n1_c10240_k2560 fc_n1_c2560_k10240 fc_n1_c2560_k2560)
# Extreme rungs never searched by the main sweep: the UPMEM extremes are
# searched on the congested-host iso_capacity basis, the HBM-PIM extremes on
# the plain iso_capacity basis.
UPMEM_EXTREMES=(
  upmem__pu-2__hmat-64__vmat-64 upmem__pu-32__hmat-4__vmat-64
  upmem__pu-2__hmat-16__vmat-256 upmem__pu-32__hmat-16__vmat-16
  upmem__pu-8__hmat-16__vmat-64_burst-1 upmem__pu-8__hmat-16__vmat-64_burst-16
)
HBM_EXTREMES=(
  hbm_pim__pu-16__hmat-8__vmat-32
  hbm_pim__pu-8__hmat-16__vmat-32_burst-1 hbm_pim__pu-8__hmat-16__vmat-32_burst-16
)

if [ "${PNMAX_SMOKE}" = "1" ]; then
  NUM_TRACES="${SMOKE_NUM_TRACES}"
  RECOST_SAMPLES=8
  UPMEM_EXTREMES=()
  HBM_EXTREMES=(hbm_pim__pu-8__hmat-16__vmat-32_burst-1)
else
  NUM_TRACES="${FULL_NUM_TRACES}"
  RECOST_SAMPLES=0   # 0 = re-cost all searched traces (full-scale default)
fi

# ---------------------------------------------------------------------------
phase_banner "phase 1/4 — reconstruct layer_params.csv (shadow PNMAX_ROOT)"
# ---------------------------------------------------------------------------
step_start "nn_models reconstruction from end_to_end manifests"
run_py python "${REPO_ROOT}/experiments/_lib/reconstruct_layer_params.py" \
  --out "${SHADOW_ROOT}" --shadow-root
# The end-to-end pipeline hard-codes its model dirs as <repo>/data/nn_models;
# the shadow root redirects only that lookup (everything else symlinks back).
export PNMAX_ROOT="${SHADOW_ROOT}"
step_end

# ---------------------------------------------------------------------------
phase_banner "phase 2/4 — geometry-sweep end-to-end searches (full: ~4 h upmem + ~2 h hbm_pim at 64 workers)"
# ---------------------------------------------------------------------------
if [ "${PNMAX_SMOKE}" = "1" ]; then
  step_start "smoke geometry subset (2 geometries per family)"
  fresh_dir "${RESULTS_DIR}/geometry_subset"
  for fam_pair in "upmem:upmem__pu-8__hmat-16__vmat-64:upmem__pu-4__hmat-32__vmat-64" \
                  "hbm_pim:hbm_pim__pu-8__hmat-16__vmat-32:hbm_pim__pu-4__hmat-32__vmat-32"; do
    IFS=':' read -r fam base rung <<<"${fam_pair}"
    run_cmd mkdir -p "${RESULTS_DIR}/geometry_subset/${fam}"
    for stem in "${base}" "${rung}"; do
      run_cmd ln -sf "${GEOMETRY_ROOT}/${fam}/${stem}.yaml" \
        "${RESULTS_DIR}/geometry_subset/${fam}/${stem}.yaml"
    done
  done
  EFFECTIVE_GEOMETRY_ROOT="${RESULTS_DIR}/geometry_subset"
  step_end
else
  EFFECTIVE_GEOMETRY_ROOT="${GEOMETRY_ROOT}"
fi

for fam in upmem hbm_pim; do
  step_start "end-to-end sweep: ${fam} (all end-to-end models; OPT feeds Fig 12)"
  # --workload-root is an OUTPUT dir: the pipeline regenerates the per-layer
  # workload YAMLs + manifest there from layer_params.csv. Keep it under the
  # results root — pointing it at data/workloads/end_to_end would rewrite the
  # shipped manifests in place. (The shipped copies stay read-only inputs for
  # the extreme-rung searches below.)
  run_py run-geometry-sweep-end-to-end-workload-space \
    --arch-family "${fam}" \
    --geometry-root "${EFFECTIVE_GEOMETRY_ROOT}" \
    --workload-root "${RESULTS_DIR}/end_to_end_workloads" \
    --output-root "${E2E_ROOT}" \
    --num-traces "${NUM_TRACES}" \
    --search-workers "${PNMAX_WORKERS}" \
    --eval-workers "${PNMAX_WORKERS}" \
    --max-attempts 1000000000 \
    --seed "${PNMAX_SEED}"
  step_end
done

# ---------------------------------------------------------------------------
phase_banner "phase 3/4 — extreme geometry rungs (/4, x4, burst-1/16)"
# ---------------------------------------------------------------------------
step_start "extreme-rung searches (spaces c, direct-c)"
_fig12_extreme_cell() {
  local arch=$1 arch_file=$2 stem=$3 token=$4
  local cell="${EXTREMES_ROOT}/${stem}/${token}"
  if [ "${PNMAX_DRY_RUN}" != "1" ] \
    && _search_cell_complete "${cell}" "${NUM_TRACES}" false "${PNMAX_SEED}" "${PNMAX_WORKERS}" c; then
    printf 'search cell reused: %s\n' "${cell}"
    return 0
  fi
  fresh_dir "${cell}"
  run_py run-workload-space-random-search \
    --workload "${REPO_ROOT}/data/workloads/end_to_end/opt-2.7B-2048/${token}.yaml" \
    --arch "${arch}" \
    --arch-file "${arch_file}" \
    --outdir "${cell}" \
    --num-traces "${NUM_TRACES}" \
    --max-attempts 1000000000 \
    --spaces c \
    --direct-c-search \
    --workers "${PNMAX_WORKERS}" \
    --batch-attempts 1 \
    --seed "${PNMAX_SEED}"
}
for stem in "${UPMEM_EXTREMES[@]}"; do
  for token in "${OPT_TOKENS[@]}"; do
    _fig12_extreme_cell upmem "${EXTREME_LAT_BASIS}/upmem/${stem}.yaml" "${stem}" "${token}"
  done
done
for stem in "${HBM_EXTREMES[@]}"; do
  for token in "${OPT_TOKENS[@]}"; do
    _fig12_extreme_cell hbm_pim "${EXTREME_AREA_BASIS}/hbm_pim/${stem}.yaml" "${stem}" "${token}"
  done
done
step_end

# ---------------------------------------------------------------------------
phase_banner "phase 4/4 — re-cost + render"
# ---------------------------------------------------------------------------
step_start "abc_opt re-cost + line figure (full: ~15 min)"
recost_args=(--e2e-root "${E2E_ROOT}" --extremes-root "${EXTREMES_ROOT}"
             --output-dir "${RESULTS_DIR}/figures" --chart line)
if [ "${RECOST_SAMPLES}" != "0" ]; then
  recost_args+=(--samples "${RECOST_SAMPLES}")
fi
run_py python "${REPO_ROOT}/plot/geometry_sweep_abc_opt.py" "${recost_args[@]}"
run_cmd mv -f "${RESULTS_DIR}/figures/geometry_sweep_abc_opt_line.pdf" \
  "${RESULTS_DIR}/figures/fig12.pdf"
step_end

phase_banner "fig12_geometry done — figures: ${RESULTS_DIR}"
