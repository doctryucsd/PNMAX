#!/usr/bin/env bash
# experiments/_lib/common.sh — shared helpers sourced by every experiments/*/run.sh.
#
# Usage (top of a run.sh):
#   . "${SCRIPT_DIR}/../_lib/common.sh"
#   pnmax_init "<experiment-name>" "$@"
#
# Standard flags parsed by pnmax_init (every button accepts exactly these):
#   --smoke        minutes-long end-to-end code check (tiny counts, isolated
#                  results universe under <results-root>/smoke/)
#                  (env: a pre-set PNMAX_SMOKE=1 behaves exactly like --smoke;
#                  the variable is exported to every child CLI)
#   --dry-run      walk the plan without executing anything. Default output =
#                  phase banners + step lines only; --dry-run --verbose adds
#                  one compact line per planned command;
#                  PNMAX_DRY_RUN_FULL=1 prints full shell-quoted commands
#   --verbose      raise output verbosity: echo every executed command
#                  ('+ ...' lines); the default output is phase banners,
#                  step start/finish lines, and tool output
#                  (env: a pre-set PNMAX_VERBOSE=1 behaves exactly like
#                  --verbose; the variable is exported to every child CLI)
#   --workers N    worker-process count (default: 64 — a fixed literal, never
#                  derived from nproc; sampled mapping sets depend on this
#                  value, so keep it at the default to reproduce a run;
#                  changing --seed or --workers requires a fresh results
#                  root — see README.md "Determinism & reuse")
#   --seed N       base seed (default: PNMAX_SEED env, else 42)
#   -h | --help    usage
#
# After pnmax_init:
#   REPO_ROOT      absolute path of the artifact repository root
#   RESULTS_ROOT   effective results universe:
#                    ${PNMAX_RESULTS_ROOT:-<repo>/results}          (full mode)
#                    ${PNMAX_RESULTS_ROOT:-<repo>/results}/smoke    (--smoke)
#                  (re-exported as PNMAX_RESULTS_ROOT so every child CLI and
#                  plot script resolves the same root)
#   RESULTS_DIR    ${RESULTS_ROOT}/<experiment-name>   (created if missing)
#   WORK_DIR       ${RESULTS_DIR}/work — cwd for run_cmd, so any stray
#                  cwd-relative output a tool produces lands under the
#                  results root, never in the repository
#   PNMAX_SEED     deterministic base seed (flag > env > 42)
#   PNMAX_WORKERS  pinned worker count (flag > env > 64)
#   PNMAX_SMOKE    exported =1 iff --smoke
#   PNMAX_DRY_RUN  exported =1 iff --dry-run
#   PNMAX_VERBOSE  exported =1 iff --verbose (or pre-set in the environment)
#
# Helpers: phase_banner, step_start/step_end, die, run_cmd, run_py,
#          fresh_dir, require_file, require_dir.

# Fail fast: -e exit on error, -u unset variables are errors, pipefail makes a
# pipeline fail on any stage, -E propagates the ERR trap into functions.
set -Eeuo pipefail

# Repo root is two levels above this file (<repo>/experiments/_lib/common.sh).
_PNMAX_LIB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${_PNMAX_LIB_DIR}/../.." && pwd)
export REPO_ROOT

# Styled output only when stdout is a terminal.
if [ -t 1 ]; then
  _PNMAX_BOLD=$'\033[1m'
  _PNMAX_DIM=$'\033[2m'
  _PNMAX_RED=$'\033[1;31m'
  _PNMAX_RESET=$'\033[0m'
else
  _PNMAX_BOLD=""
  _PNMAX_DIM=""
  _PNMAX_RED=""
  _PNMAX_RESET=""
fi

# die "message" [exit-code] — print a diagnostic to stderr and exit (default 1).
die() {
  printf '%s[%s] ERROR:%s %s\n' \
    "${_PNMAX_RED}" "${PNMAX_EXPERIMENT:-pnmax}" "${_PNMAX_RESET}" "$1" >&2
  exit "${2:-1}"
}

# ERR trap — on any failing command, report status, location, and the command,
# then exit with the same status so callers see the real failure code.
_pnmax_on_error() {
  local status=$1 src=$2 line=$3 cmd=$4
  printf '%s[%s] FAILED%s (exit %s) at %s:%s while running: %s\n' \
    "${_PNMAX_RED}" "${PNMAX_EXPERIMENT:-pnmax}" "${_PNMAX_RESET}" \
    "${status}" "${src}" "${line}" "${cmd}" >&2
  printf 'Partial outputs (if any): %s\n' "${RESULTS_DIR:-<results dir not set>}" >&2
  exit "${status}"
}
trap '_pnmax_on_error "$?" "${BASH_SOURCE[0]:-$0}" "${LINENO}" "${BASH_COMMAND}"' ERR

# phase_banner "text" — bold, timestamped banner marking a major phase.
phase_banner() {
  _pnmax_dry_flush
  printf '\n%s==== [%s] %s ====%s\n' \
    "${_PNMAX_BOLD}" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" "${_PNMAX_RESET}"
}

# step_start "label" — begin a timed step (pair with step_end).
step_start() {
  _PNMAX_STEP_LABEL=$1
  _PNMAX_STEP_T0=$(date +%s)
  printf '[%s] step started: %s\n' "$(date '+%H:%M:%S')" "${_PNMAX_STEP_LABEL}"
}

# step_end — finish the current step and print its wall-clock duration.
step_end() {
  _pnmax_dry_flush
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - ${_PNMAX_STEP_T0:-now}))
  printf '[%s] step finished: %s (%ds)\n' \
    "$(date '+%H:%M:%S')" "${_PNMAX_STEP_LABEL:-<unnamed>}" "${elapsed}"
}

# pnmax_usage — standard usage text (driver docstring comes from run.sh header).
pnmax_usage() {
  cat <<'EOF'
Usage: run.sh [--smoke] [--dry-run] [--workers N] [--seed N]
  --smoke      quick end-to-end code check (tiny counts, isolated under
               <results-root>/smoke/); default is full scale
  --dry-run    walk the plan, execute nothing (default: phase + step lines
               only; add --verbose for per-command lines;
               PNMAX_DRY_RUN_FULL=1 for full shell-quoted commands)
  --verbose    echo every executed command
               (default output: banners, step lines, tool output)
  --workers N  worker processes (default 64; fixed literal, never nproc —
               sampled mapping sets depend on this value)
  --seed N     base seed (default: PNMAX_SEED env, else 42)
EOF
}

# pnmax_init "<experiment-name>" [args...] — parse the standard flags, export
# the deterministic seed + worker pin, and resolve/create RESULTS_DIR/WORK_DIR.
pnmax_init() {
  [ $# -ge 1 ] || die "pnmax_init: missing experiment name"
  PNMAX_EXPERIMENT=$1
  shift
  export PNMAX_EXPERIMENT

  export PNMAX_SMOKE="${PNMAX_SMOKE:-0}"
  export PNMAX_VERBOSE="${PNMAX_VERBOSE:-0}"
  export PNMAX_DRY_RUN=0
  local _workers_flag="" _seed_flag=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --smoke)
        export PNMAX_SMOKE=1
        ;;
      --dry-run)
        export PNMAX_DRY_RUN=1
        ;;
      --verbose)
        export PNMAX_VERBOSE=1
        ;;
      --workers)
        [ $# -ge 2 ] || die "--workers requires a value"
        _workers_flag=$2
        shift
        ;;
      --workers=*)
        _workers_flag=${1#--workers=}
        ;;
      --seed)
        [ $# -ge 2 ] || die "--seed requires a value"
        _seed_flag=$2
        shift
        ;;
      --seed=*)
        _seed_flag=${1#--seed=}
        ;;
      -h | --help)
        pnmax_usage
        exit 0
        ;;
      *)
        die "unknown argument '$1' (supported: --smoke --dry-run --verbose --workers N --seed N)"
        ;;
    esac
    shift
  done

  # Determinism contract: every stochastic component reads PNMAX_SEED
  # (resolution order: --seed flag > PNMAX_SEED env > 42).
  export PNMAX_SEED="${_seed_flag:-${PNMAX_SEED:-42}}"

  # Worker pin: a fixed literal default (64), never nproc-derived — the
  # multiprocess samplers shard work by worker index, so the sampled mapping
  # sets depend on this value. Override changes results; document when used.
  export PNMAX_WORKERS="${_workers_flag:-${PNMAX_WORKERS:-64}}"
  case "${PNMAX_WORKERS}" in
    '' | *[!0-9]*) die "--workers must be a positive integer, got '${PNMAX_WORKERS}'" ;;
  esac

  # Results universe. Intermediate/run outputs live under PNMAX_RESULTS_ROOT
  # when set (e.g. a larger scratch volume); the default keeps a reviewer run
  # self-contained inside the repository. Smoke runs are isolated under a
  # dedicated smoke/ sub-root so they can never contaminate a full campaign.
  RESULTS_ROOT="${PNMAX_RESULTS_ROOT:-${REPO_ROOT}/results}"
  if [ "${PNMAX_SMOKE}" = "1" ]; then
    RESULTS_ROOT="${RESULTS_ROOT}/smoke"
  fi
  export RESULTS_ROOT
  # Re-export so child CLIs (pnmax.paths.results_root) and plot scripts
  # (PNMAX_RESULTS_ROOT env) resolve exactly the same universe.
  export PNMAX_RESULTS_ROOT="${RESULTS_ROOT}"

  RESULTS_DIR="${RESULTS_ROOT}/${PNMAX_EXPERIMENT}"
  export RESULTS_DIR
  WORK_DIR="${RESULTS_DIR}/work"
  export WORK_DIR
  if [ "${PNMAX_DRY_RUN}" != "1" ]; then
    mkdir -p "${RESULTS_DIR}" "${WORK_DIR}"
  fi

  local mode="full"
  if [ "${PNMAX_SMOKE}" = "1" ]; then mode="smoke"; fi
  if [ "${PNMAX_DRY_RUN}" = "1" ]; then mode="${mode} (DRY RUN — command plan only)"; fi
  printf 'mode: %s | seed: %s | workers: %s | results: %s\n' \
    "${mode}" "${PNMAX_SEED}" "${PNMAX_WORKERS}" "${RESULTS_DIR}"
}

# _pnmax_dry_flush — close a collapsed command group (compact dry-run mode).
_pnmax_dry_flush() {
  if [ "${_PNMAX_DRY_GROUP_N:-0}" -gt 0 ] && [ "${PNMAX_VERBOSE:-0}" = "1" ]; then
    printf '%s  ... +%d more %s call(s) of the same shape%s\n' \
      "${_PNMAX_DIM}" "${_PNMAX_DRY_GROUP_N}" "${_PNMAX_DRY_GROUP_KEY}" "${_PNMAX_RESET}"
  fi
  _PNMAX_DRY_GROUP_KEY=""
  _PNMAX_DRY_GROUP_N=0
}

# _pnmax_print_cmd — echo of a command vector. Real runs echo the exact,
# shell-quoted, copy-pasteable command only at raised verbosity
# (--verbose / PNMAX_VERBOSE=1); by default they print nothing here, so the
# run output stays at banners, step lines, and tool output.
# Dry runs always show the plan: full per-command form with
# PNMAX_DRY_RUN_FULL=1, otherwise a COMPACT plan — plumbing commands
# (rm/mkdir/ln) are hidden, the `uv run --project <root>` wrapper and the
# repo/results path prefixes are stripped, and consecutive calls of the same
# command collapse into a single line plus a count.
_pnmax_print_cmd() {
  if [ "${PNMAX_DRY_RUN}" != "1" ]; then
    if [ "${PNMAX_VERBOSE:-0}" = "1" ]; then
      printf '%s+' "${_PNMAX_DIM}"
      printf ' %q' "$@"
      printf '%s\n' "${_PNMAX_RESET}"
    fi
    return 0
  fi
  if [ "${PNMAX_DRY_RUN_FULL:-0}" = "1" ]; then
    printf '%s+' "${_PNMAX_DIM}"
    printf ' %q' "$@"
    printf '%s\n' "${_PNMAX_RESET}"
    return 0
  fi
  case "${1:-}" in rm | mkdir | ln) return 0 ;; esac
  if [ "${1:-}" = "uv" ] && [ "${2:-}" = "run" ] && [ "${3:-}" = "--project" ]; then
    shift 4
  fi
  if [ "${1:-}" = "python" ] || [ "${1:-}" = "python3" ]; then
    local script
    script=$(basename "${2}")
    shift 2
    set -- "${script}" "$@"
  fi
  if [ "${1}" = "${_PNMAX_DRY_GROUP_KEY:-}" ]; then
    _PNMAX_DRY_GROUP_N=$(( ${_PNMAX_DRY_GROUP_N:-0} + 1 ))
    return 0
  fi
  _pnmax_dry_flush
  _PNMAX_DRY_GROUP_KEY=$1
  if [ "${PNMAX_VERBOSE:-0}" != "1" ]; then
    # Default dry-run: phases + step lines only; commands print nothing.
    return 0
  fi
  local tok
  printf '%s+' "${_PNMAX_DIM}"
  for tok in "$@"; do
    tok=${tok//"${RESULTS_ROOT:-/__unset__}/"/results/}
    tok=${tok//"${REPO_ROOT}/"/}
    printf ' %s' "${tok}"
  done
  printf '%s\n' "${_PNMAX_RESET}"
}

# run_cmd cmd [args...] — print the exact command, then execute it from
# WORK_DIR (unless --dry-run). All artifact commands go through this so
# --dry-run prints a complete, copy-pasteable command plan.
run_cmd() {
  _pnmax_print_cmd "$@"
  if [ "${PNMAX_DRY_RUN}" = "1" ]; then
    return 0
  fi
  (cd "${WORK_DIR}" && "$@")
}

# run_py args... — run_cmd wrapper for `uv run` against the artifact project
# (works from any cwd; WORK_DIR keeps stray relative outputs in results/).
run_py() {
  run_cmd uv run --project "${REPO_ROOT}" "$@"
}

# fresh_dir <dir> — guarantee an empty output directory (delete + recreate).
# Search CLIs silently RESUME into non-empty directories, which breaks the
# fresh-run determinism contract — always route their outdirs through this.
# Refuses to delete anything outside the results universe.
fresh_dir() {
  local dir=$1
  case "${dir}" in
    "${RESULTS_ROOT}"/*) ;;
    *) die "fresh_dir: refusing to touch '${dir}' (outside ${RESULTS_ROOT})" ;;
  esac
  _pnmax_print_cmd rm -rf "${dir}"
  if [ "${PNMAX_DRY_RUN}" = "1" ]; then
    return 0
  fi
  rm -rf "${dir}"
  mkdir -p "${dir}"
}

# require_file <path> <hint> — fail early with a actionable message.
require_file() {
  [ "${PNMAX_DRY_RUN}" = "1" ] && return 0
  [ -f "$1" ] || die "missing required file: $1
  hint: $2"
}

# require_dir <path> <hint>
require_dir() {
  [ "${PNMAX_DRY_RUN}" = "1" ] && return 0
  [ -d "$1" ] || die "missing required directory: $1
  hint: $2"
}
