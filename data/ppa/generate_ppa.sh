#!/usr/bin/env bash
# data/ppa/generate_ppa.sh — generate the derived PPA characterization
# outputs. This is the STANDARD artifact flow: the DRAM lookup CSVs and every
# builder-generated architecture family are DERIVED data (gitignored), built
# here from the vendored tools + checked-in generators. setup.sh runs this
# script; it can also be run on its own at any time:
#
#     ./data/ppa/generate_ppa.sh
#
# Stages (each fails the script loudly on error):
#   (1) build the CACTI binary        (external/cacti, ~seconds)
#   (2) DRAM characterization         generate_dram_lookup_csvs.py
#         -> data/ppa/cacti_ddr4_results.csv
#         -> data/ppa/dreamram_hbm2e_results.csv
#         -> data/ppa/lookup_work/dram/   (per-run tool cfgs + raw outputs)
#   (3) derived architecture families  (from the fresh CSVs)
#         -> data/archs/lowered/geometry_sweep/{upmem,hbm_pim}
#         -> data/archs/lowered/geometry_sweep/iso_capacity_archs
#         -> data/archs/lowered/buffer_sweep
#         -> data/archs/lowered/interconnect_sweep
#         -> data/archs/lowered/geometry_sweep_host_congested
#
# Any previous derived state is wiped first, so repeat runs are
# byte-deterministic under the pinned Python environment. Generation is
# pure derivation — no random seeds are involved at this stage.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

PPA_DIR="data/ppa"
LOWERED_DIR="data/archs/lowered"

banner() { printf '\n\033[1m---- PPA gen: %s ----\033[0m\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
  if [ -x "${HOME}/.local/bin/uv" ]; then
    PATH="${HOME}/.local/bin:${PATH}"
  else
    echo "ERROR: uv not found — run ./setup.sh first (it bootstraps uv + the Python env)." >&2
    exit 1
  fi
fi

expect_count() {
  # expect_count <dir> <glob-count-expected> — recursive *.yaml count check.
  local dir="$1" expected="$2" actual
  actual=$(find "${dir}" -name '*.yaml' -type f | wc -l)
  if [ "${actual}" != "${expected}" ]; then
    echo "ERROR: ${dir} holds ${actual} arch YAMLs, expected ${expected}." >&2
    exit 1
  fi
  echo "  ${dir}: ${actual}/${expected} files OK"
}

# ---------------------------------------------------------------------------
banner "stage 0/3 — wipe previous derived outputs"
# ---------------------------------------------------------------------------
rm -f "${PPA_DIR}/cacti_ddr4_results.csv" "${PPA_DIR}/dreamram_hbm2e_results.csv"
rm -rf "${PPA_DIR}/lookup_work"
rm -rf "${LOWERED_DIR}/geometry_sweep" \
       "${LOWERED_DIR}/geometry_sweep_host_congested" \
       "${LOWERED_DIR}/buffer_sweep" \
       "${LOWERED_DIR}/interconnect_sweep"
echo "derived CSVs, lookup_work/ and derived arch families removed"

# ---------------------------------------------------------------------------
banner "stage 1/3 — build CACTI (external/cacti)"
# ---------------------------------------------------------------------------
for tool in make g++; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "ERROR: CACTI build needs ${tool}." >&2; exit 1; }
done
make -C external/cacti -j "$(nproc)"
[ -x external/cacti/cacti ] || { echo "ERROR: external/cacti/cacti binary missing after build." >&2; exit 1; }
echo "CACTI built: external/cacti/cacti"

# ---------------------------------------------------------------------------
banner "stage 2/3 — DRAM characterization (CACTI + DreamRAM -> lookup CSVs)"
# ---------------------------------------------------------------------------
uv run python "${PPA_DIR}/generate_dram_lookup_csvs.py"
for csv in "${PPA_DIR}/cacti_ddr4_results.csv" "${PPA_DIR}/dreamram_hbm2e_results.csv"; do
  [ -s "${csv}" ] || { echo "ERROR: ${csv} missing/empty after characterization." >&2; exit 1; }
done
# DreamRAM writes per-run outputs inside its own vendored tree; the parsed
# results now live in the CSVs + lookup_work/, so clear the tool-side residue.
rm -rf external/dreamram/data

# ---------------------------------------------------------------------------
banner "stage 3/3 — derived architecture families (5 builders)"
# ---------------------------------------------------------------------------
# geometry_sweep/{upmem,hbm_pim}: the evaluated geometry-variant sets.
# fig12's search enumerates these directories, so exactly the evaluated
# variants are generated: upmem variants 1-9; hbm_pim variants 1-4,7-9
# (variants 5-6 are not part of the evaluated hbm_pim geometry set, and
# variants 10-15 ship under their own buffer_/interconnect_sweep families).
echo "[builder 1/5] geometry_sweep (generate_arch_from_template_sweep.sh)"
ARCHS_OVERRIDE="upmem" VARIANTS_OVERRIDE="1 2 3 4 5 6 7 8 9" \
  bash "${PPA_DIR}/generate_arch_from_template_sweep.sh"
ARCHS_OVERRIDE="hbm_pim" VARIANTS_OVERRIDE="1 2 3 4 7 8 9" \
  bash "${PPA_DIR}/generate_arch_from_template_sweep.sh"
expect_count "${LOWERED_DIR}/geometry_sweep/upmem" 9
expect_count "${LOWERED_DIR}/geometry_sweep/hbm_pim" 7

echo "[builder 2/5] geometry_sweep/iso_capacity_archs (build_iso_capacity_archs.py)"
uv run python "${PPA_DIR}/build_iso_capacity_archs.py"
expect_count "${LOWERED_DIR}/geometry_sweep/iso_capacity_archs" 27

echo "[builder 3/5] buffer_sweep (build_buffer_archs.py)"
uv run python "${PPA_DIR}/build_buffer_archs.py"
expect_count "${LOWERED_DIR}/buffer_sweep" 12

echo "[builder 4/5] interconnect_sweep (build_interconnect_archs.py)"
uv run python "${PPA_DIR}/build_interconnect_archs.py"
expect_count "${LOWERED_DIR}/interconnect_sweep" 8

echo "[builder 5/5] geometry_sweep_host_congested (make_host_congested_archs.py)"
uv run python "${PPA_DIR}/make_host_congested_archs.py" --force \
  "${LOWERED_DIR}/geometry_sweep/iso_capacity_archs" \
  "${LOWERED_DIR}/geometry_sweep_host_congested/iso_capacity_archs"
expect_count "${LOWERED_DIR}/geometry_sweep_host_congested" 27

banner "PPA generation complete"
echo "DRAM lookup CSVs + derived arch families built under data/."
