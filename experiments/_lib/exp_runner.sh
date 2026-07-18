#!/usr/bin/env bash
# Generic serialized experiment runner for long sweep campaigns.
#
# Runs the jobs in <queue_file> strictly one-at-a-time. Because each individual
# job is configured to use a bounded worker count (<=100), sequential execution
# keeps the whole runner within that core budget (never run more than one heavy
# sweep at once).
#
# Usage: experiments/_lib/exp_runner.sh <queue_file> <state_dir>
#   queue_file : one job per line, "<name>\t<command>". Lines that are blank or
#                start with '#' are ignored. <command> is run via `bash -c`.
#   state_dir  : where logs/<name>.log, status.log, CURRENT and DONE are written.
#
# A lock file (default: <state_dir>/runner.lock, override with PNMAX_EXP_LOCK)
# guarantees a single active runner per state dir.
#
# Designed to be launched detached (setsid/nohup) so it survives the session
# ending; a takeover session reads state_dir/status.log to learn progress.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PNMAX_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}" || exit 1

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <queue_file> <state_dir>" >&2
  exit 2
fi

QUEUE="$1"
STATE="$2"

if [[ ! -f "${QUEUE}" ]]; then
  echo "queue file not found: ${QUEUE}" >&2
  exit 2
fi

mkdir -p "${STATE}/logs"
STATUS="${STATE}/status.log"

# Single-runner guard. The lock lives inside the state dir by default so
# concurrent runners on unrelated state dirs never collide.
LOCK_FILE="${PNMAX_EXP_LOCK:-${STATE}/runner.lock}"
exec 9> "${LOCK_FILE}"
if ! flock -n 9; then
  echo "another exp_runner already holds ${LOCK_FILE}; refusing to start" >&2
  exit 3
fi

rm -f "${STATE}/DONE"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

mapfile -t LINES < <(grep -vE '^[[:space:]]*(#|$)' "${QUEUE}")
total=${#LINES[@]}

echo "[$(ts)] exp_runner START queue=${QUEUE} pid=$$ total_jobs=${total}" >> "${STATUS}"

i=0
ok=0
fail=0
for line in "${LINES[@]}"; do
  i=$((i + 1))
  name="${line%%$'\t'*}"
  cmd="${line#*$'\t'}"
  log="${STATE}/logs/${name}.log"

  echo "${name}" > "${STATE}/CURRENT"
  echo "[$(ts)] START ${i}/${total} ${name}" >> "${STATUS}"
  start=$(date +%s)
  bash -c "${cmd}" > "${log}" 2>&1
  rc=$?
  end=$(date +%s)

  if [[ ${rc} -eq 0 ]]; then
    ok=$((ok + 1))
    echo "[$(ts)] DONE  ${i}/${total} ${name} elapsed=$((end - start))s" >> "${STATUS}"
  else
    fail=$((fail + 1))
    echo "[$(ts)] FAIL  ${i}/${total} ${name} exit=${rc} elapsed=$((end - start))s log=${log}" >> "${STATUS}"
  fi
done

rm -f "${STATE}/CURRENT"
echo "[$(ts)] exp_runner ALL DONE total=${total} ok=${ok} fail=${fail}" >> "${STATUS}"
echo "ok=${ok} fail=${fail} total=${total}" > "${STATE}/DONE"
