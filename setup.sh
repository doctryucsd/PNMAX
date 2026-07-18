#!/usr/bin/env bash
# setup.sh — one-time environment bootstrap for the PNMAX artifact.
#
# Usage: ./setup.sh [--without-cinm] [--check-gpam-only]
#   --without-cinm     skip the CINM baseline build (Step 6). CINM is built by
#                      default; this opt-out avoids its large LLVM download and
#                      multi-hour build when the CINM points in Fig. 9 are not
#                      needed (render Fig. 9 without them via PNMAX_SKIP_CINM=1)
#   --check-gpam-only  run only the fast GPAM equivalence gate (uv env + check)
#                      and skip the external tool build + PPA generation
#                      (quick verification; produces no derived data)
#
# Steps: (1) uv bootstrap  (2) Python env via `uv sync`  (3) GPAM equivalence
# gate  (4) external tool build (AttAcc Ramulator2 — required)  (5) PPA
# characterization + derived-arch generation (builds CACTI, runs the
# CACTI/DreamRAM characterization, builds the derived architecture
# families; see data/ppa/generate_ppa.sh)  (6) CINM baseline build (default;
# skip with --without-cinm).
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${REPO_ROOT}"

# Pinned public llvm/llvm-project commit for the CINM baseline (LLVM 18.1.6 +
# the linalg::ReduceOp verifier fix). Verified fetchable by SHA from the
# canonical public repo; see external/cinm/PROVENANCE.md.
LLVM_COMMIT="6f89431c3d4de87df6d76cf7ffa73bfa881607b7"
LLVM_REPO="https://github.com/llvm/llvm-project.git"

WITH_CINM=1
CHECK_GPAM_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --without-cinm) WITH_CINM=0 ;;
    --with-cinm) WITH_CINM=1 ;;  # accepted no-op: CINM is built by default
    --check-gpam-only) CHECK_GPAM_ONLY=1 ;;
    -h | --help)
      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "setup.sh: unknown argument '${arg}' (supported: --without-cinm, --check-gpam-only)" >&2
      exit 1
      ;;
  esac
done

banner() { printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }

require_tools() {
  local missing=0 tool
  for tool in "$@"; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      echo "  missing required tool: ${tool}" >&2
      missing=1
    fi
  done
  return "${missing}"
}

# gpam_equivalence_check — the GPAM architecture-interface gate. The two-file
# GPAM specs in data/archs/baseline/ are the authoritative architecture
# definitions; `gpam-lower` lowers them to the flat data/archs/lowered/baseline/*
# files that every experiment loads. This verifies the lowering still
# matches the calibrated flat references byte-semantically (Tier-1 YAML +
# Tier-2 parsed-Arch), for all three baselines. Fast (<10 s): needs only the
# uv env, no external builds. Fails loudly (shows the field-level diff).
gpam_equivalence_check() {
  local a out failed=0
  for a in hbm_pim upmem attacc; do
    if ! out=$(uv run gpam-lower \
        --arch "data/archs/baseline/${a}.yaml" \
        --check "data/archs/lowered/baseline/${a}.yaml" 2>&1); then
      echo "  ERROR: GPAM equivalence check FAILED for ${a}:" >&2
      printf '%s\n' "${out}" | sed 's/^/    /' >&2
      failed=1
    fi
  done
  if [ "${failed}" = "1" ]; then
    echo "GPAM equivalence gate FAILED: a baseline GPAM spec no longer lowers to its" >&2
    echo "flat reference under data/archs/lowered/baseline/. Reproduce with:" >&2
    echo "  uv run gpam-lower --arch data/archs/baseline/<arch>.yaml \\" >&2
    echo "    --check data/archs/lowered/baseline/<arch>.yaml" >&2
    exit 1
  fi
  echo "GPAM equivalence gate OK: hbm_pim, upmem, attacc lower to their flat baselines (gpam-lower --check, 3/3 exit 0)."
}

banner "Step 1/6: uv bootstrap"
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "${HOME}/.local/bin/uv" ]; then
    # uv is installed but not on PATH in this shell.
    PATH="${HOME}/.local/bin:${PATH}"
  else
    echo "uv not found. Install it with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "then open a new shell (or 'source \$HOME/.local/bin/env') and re-run ./setup.sh" >&2
    exit 1
  fi
fi
echo "using uv $(uv --version | awk '{print $2}') at $(command -v uv)"

banner "Step 2/6: Python environment (uv sync)"
uv sync
uv run python -c "import pnmax; print('pnmax', pnmax.__version__, 'import OK')"

banner "Step 3/6: GPAM architecture-interface equivalence gate"
gpam_equivalence_check

if [ "${CHECK_GPAM_ONLY}" = "1" ]; then
  banner "GPAM-only check finished"
  echo "Skipped external tool builds + PPA generation (--check-gpam-only)."
  exit 0
fi

banner "Step 4/6: External tool build (AttAcc Ramulator2)"

# --- AttAcc / Ramulator2 (REQUIRED for the Fig. 8 / Table 4 validation) --------
# The vendored external/attacc/ramulator2 is the prepared source (Ramulator2
# b7c70275 + AttAcc's PIM extensions); build it in place. Ext deps (yaml-cpp,
# spdlog, argparse) are fetched by CMake FetchContent at configure time.
echo "Building AttAcc Ramulator2 (required)..."
if ! require_tools cmake make g++; then
  echo "ERROR: AttAcc build needs cmake, make and a C++ compiler (g++)." >&2
  exit 1
fi
ATTACC_RAM="${REPO_ROOT}/external/attacc/ramulator2"
cmake -S "${ATTACC_RAM}" -B "${ATTACC_RAM}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${ATTACC_RAM}/build" -j "$(nproc)"
if [ -x "${ATTACC_RAM}/build/ramulator2" ]; then
  echo "AttAcc ramulator2 built: ${ATTACC_RAM}/build/ramulator2"
else
  echo "ERROR: AttAcc build finished but ${ATTACC_RAM}/build/ramulator2 is missing." >&2
  exit 1
fi

# --- PPA characterization + derived-arch generation (REQUIRED) -----------------
# The DRAM lookup CSVs (data/ppa/{cacti_ddr4,dreamram_hbm2e}_results.csv) and
# every derived architecture family under data/archs/lowered/ are DERIVED
# outputs: they are not shipped and are generated here as the standard flow
# (CACTI build + CACTI/DreamRAM characterization + the 5 arch builders). The
# stage script is independently re-runnable: ./data/ppa/generate_ppa.sh
banner "Step 5/6: PPA characterization + derived-arch generation"
"${REPO_ROOT}/data/ppa/generate_ppa.sh"

# --- CINM baseline (default; Step 6; skip with --without-cinm) ------------------
if [ "${WITH_CINM}" = "1" ]; then
  banner "Step 6/6: CINM baseline"
  if ! require_tools git cmake ninja clang clang++; then
    echo "ERROR: the default CINM build needs git, cmake, ninja and clang/clang++." >&2
    echo "       Install them, or pass --without-cinm to skip the CINM baseline." >&2
    exit 1
  fi

  LLVM_SRC="${REPO_ROOT}/external/cinm/third-party/llvm"
  LLVM_BUILD="${LLVM_SRC}/build"

  # (a) Fetch the pinned public LLVM at LLVM_COMMIT (shallow, by SHA).
  if [ ! -f "${LLVM_SRC}/llvm/CMakeLists.txt" ]; then
    echo "Fetching public llvm-project @ ${LLVM_COMMIT} (shallow)..."
    mkdir -p "${LLVM_SRC}"
    git -C "${LLVM_SRC}" init -q
    if ! git -C "${LLVM_SRC}" remote get-url origin >/dev/null 2>&1; then
      git -C "${LLVM_SRC}" remote add origin "${LLVM_REPO}"
    fi
    git -C "${LLVM_SRC}" fetch --depth 1 origin "${LLVM_COMMIT}"
    git -C "${LLVM_SRC}" checkout -q FETCH_HEAD
  else
    echo "LLVM sources already present at ${LLVM_SRC} — skipping fetch."
  fi

  # (b) Build LLVM + MLIR (Release, shared libs, assertions) with clang.
  #     WARNING: multi-hour build; needs plenty of RAM/disk.
  echo "Configuring + building LLVM/MLIR (this takes a long time)..."
  cmake -S "${LLVM_SRC}/llvm" -B "${LLVM_BUILD}" -G Ninja \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
  cmake --build "${LLVM_BUILD}"

  # (c) Build cinm-opt against that LLVM (clang required — g++ rejects a cinm
  #     injected-class-name template; see external/cinm/PROVENANCE.md).
  echo "Configuring + building cinm-opt..."
  cmake -S "${REPO_ROOT}/external/cinm" -B "${REPO_ROOT}/external/cinm/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_DIR="${LLVM_BUILD}/lib/cmake/llvm" \
    -DMLIR_DIR="${LLVM_BUILD}/lib/cmake/mlir" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build "${REPO_ROOT}/external/cinm/build" --target cinm-opt
  CINM_OPT_BIN="${REPO_ROOT}/external/cinm/build/bin/cinm-opt"
  if [ -x "${CINM_OPT_BIN}" ]; then
    echo "cinm-opt built: ${CINM_OPT_BIN}"
  else
    echo "ERROR: CINM build finished but ${CINM_OPT_BIN} is missing." >&2
    exit 1
  fi
else
  banner "Step 6/6: CINM baseline"
  echo "skipped (--without-cinm): the CINM points in Fig. 9 will be absent —"
  echo "  render Fig. 9 without them via PNMAX_SKIP_CINM=1."
fi

banner "Setup finished"
echo "Next: run the smoke suite, e.g. experiments/fig08_tab4_validation/run.sh --smoke"
