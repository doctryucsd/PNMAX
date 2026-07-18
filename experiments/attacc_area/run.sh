#!/usr/bin/env bash
# experiments/attacc_area/run.sh — AttAcc PU+buffer area overhead
# derived from the PPA DB.
#
# Usage: ./run.sh [--smoke] [--dry-run] [--workers N] [--seed N]
# Pure PPA-DB arithmetic (no search, no simulator): --smoke and the default
# run the identical, seconds-long computation.
# Writes only under <results-root>/attacc_area/.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../_lib/common.sh
. "${SCRIPT_DIR}/../_lib/common.sh"

pnmax_init "attacc_area" "$@"
phase_banner "attacc_area — AttAcc PU+buffer area overhead from the PPA DB"

step_start "area derivation (expect < 10 s)"
run_py python "${SCRIPT_DIR}/attacc_area.py" \
  --out "${RESULTS_DIR}/attacc_area_summary.json"
step_end

phase_banner "attacc_area done — summary: ${RESULTS_DIR}/attacc_area_summary.json"
