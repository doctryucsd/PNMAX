#!/usr/bin/env bash
# derive_baselines.sh — reproduce a PNMAX UniNDP-baseline mapping YAML (D7).
#
# Runs the vendored UniNDP compiler for one workload x architecture, then
# translates UniNDP's chosen partition into a PNMAX nested-loop workload YAML.
# Fully self-contained: uses external/unindp/_compile_driver.py (namespace-package
# launcher for compile.py) and external/unindp/derive_baseline_yaml.py (log ->
# workload-YAML converter, no pnmax dependency).
#
# Usage:
#   derive_baselines.sh <name> <arch> <M> <K> <N> <B> <template.yaml> <out.yaml> [reference.yaml]
#     name        workload token (e.g. bert-3), used for log/output naming
#     arch        hbm_pim | upmem | aim | aim8 | dimmining
#     M K N B     UniNDP mm workload size (input rows, reduction, output cols, batch)
#     template    a workload YAML supplying static fields (TensorAttr/Compute/Layout-Pragma)
#     out         path to write the derived baseline YAML
#     reference   (optional) reference YAML to diff the result against
#
# Env:
#   WORKDIR   scratch dir for UniNDP compile artefacts (default: mktemp under TMPDIR)
#   TOPK      UniNDP quicksearch top-k (default 30, matches the paper run)
#   PYTHON    python interpreter (default: python3)
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON="${PYTHON:-python3}"
TOPK="${TOPK:-30}"

if [ "$#" -lt 8 ]; then
  sed -n '3,24p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

NAME="$1"; ARCH="$2"; M="$3"; K="$4"; N="$5"; B="$6"; TEMPLATE="$7"; OUT="$8"
REFERENCE="${9:-}"
WORKDIR="${WORKDIR:-$(mktemp -d "${TMPDIR:-/tmp}/unindp_derive.XXXXXX")}"
mkdir -p "${WORKDIR}"
# compile.py reads ./config/*.yaml and writes ./<ws>/... relative to cwd; run it
# from WORKDIR with a config symlink so nothing lands in the vendored snapshot.
ln -sfn "${SCRIPT_DIR}/config" "${WORKDIR}/config"

# Absolutise template/out/reference before we cd into WORKDIR.
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "$(pwd)" "$1" ;; esac; }
TEMPLATE=$(abspath "${TEMPLATE}"); OUT=$(abspath "${OUT}")
[ -n "${REFERENCE}" ] && REFERENCE=$(abspath "${REFERENCE}")

echo "==== UniNDP compile: ${NAME} on ${ARCH}  (-S ${M} ${K} ${N} ${B}) ===="
( cd "${WORKDIR}" && "${PYTHON}" "${SCRIPT_DIR}/_compile_driver.py" -- \
  -A "${ARCH}" -W mm -N "${NAME}" -S "${M}" "${K}" "${N}" "${B}" \
  -WS . -O "${NAME}_${ARCH}" -Q -K "${TOPK}" )

LOG="${WORKDIR}/${NAME}_${ARCH}/log/_${NAME}.log"
if [ ! -f "${LOG}" ]; then
  echo "ERROR: UniNDP compile log not found at ${LOG}" >&2
  exit 1
fi

echo "==== Convert best_design -> PNMAX workload YAML ===="
"${PYTHON}" "${SCRIPT_DIR}/derive_baseline_yaml.py" \
  --log-path "${LOG}" \
  --template-path "${TEMPLATE}" \
  --output-path "${OUT}" \
  --name "${NAME}" \
  --workload-size "${M}" "${K}" "${N}" "${B}" \
  --design best --best-index 1

echo "Derived: ${OUT}"

if [ -n "${REFERENCE}" ]; then
  echo "==== Diff vs reference: ${REFERENCE} ===="
  if diff -u "${REFERENCE}" "${OUT}"; then
    echo "RESULT: identical to reference."
  else
    echo "RESULT: differs from reference (see diff above)."
  fi
fi
