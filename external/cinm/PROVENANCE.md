# Provenance: Cinnamon (CINM)

- **Upstream:** https://github.com/doctryucsd/Cinnamon
  (fork of https://github.com/tud-ccc/Cinnamon)
- **Vendored commit:** `f776a0799429c6dd59444d5b9bfd877fe7b1c078`
- **License:** MIT (see `LICENSE`; © 2024 Chair for Compiler Construction,
  TU Dresden). The `f776a07` tree predates the project adding a LICENSE file;
  the project's MIT license (identical in `tud-ccc/Cinnamon` and the later
  `doctryucsd/Cinnamon` commits) is added here to preserve it.
- **Paper:** CINM (Cinnamon): A Compilation Infrastructure for Compute-In-Memory
  and Compute-Near-Memory (arXiv:2301.07486).

## ⚠️ Which Cinnamon commit — and why not the PNMAX development repo's submodule HEAD

**FLAG for review.** The PNMAX development repo's `external/cinm` submodule is
pinned at `0d132a3`, but this artifact vendors `f776a07` instead. Rationale:

- The reference Fig. 9 CINM baseline (`experiments/fig09_pareto/`) was
  produced with the `cinm-opt` from a
  local `cinm-jun2024` checkout (the path the development repo's original
  drivers hardcoded;
  this artifact's drivers resolve `cinm-opt` repo-relatively, `CINM_OPT` to
  override). That checkout is Cinnamon **`f776a07`** built against
  **LLVM 18.1.6**.
- The development repo's submodule was later bumped to `0d132a3` (a descendant of `f776a07`)
  **without updating its `third-party/llvm`**, which stayed pinned at 18.1.6
  (`6f89431`, below). `0d132a3` uses newer MLIR APIs (e.g. the member form
  `MemRefType::getStridesAndOffset`) and **requires LLVM 20.1** — it does *not*
  compile against the pinned 18.1.6, so the submodule + its own LLVM pin are
  mutually inconsistent.
- `f776a07` is the commit that (a) the drivers actually use, (b) matches the
  pinned public LLVM 18.1.6, and (c) builds + runs cleanly here. It is a direct
  ancestor of `0d132a3`.

If the newer `0d132a3` is ever preferred, `setup.sh --with-cinm` would need to
fetch LLVM 20.1 instead; the drivers and Fig. 9 comparison are unchanged.

## LLVM pin — public, no custom patches

- **Pinned commit:** `6f89431c3d4de87df6d76cf7ffa73bfa881607b7`
  (from `git -C external/cinm/third-party/llvm rev-parse HEAD` in the PNMAX
  development repo).
- This is a genuine **public `llvm/llvm-project`** commit: it is fetchable by SHA
  from `https://github.com/llvm/llvm-project.git` (`git fetch --depth 1 <url>
  6f89431…` succeeds), and GitHub's compare API reports it as exactly **1 commit
  ahead of `llvmorg-18.1.6`** — that one commit is
  `[mlir] Fix #93973 - linalg::ReduceOp verifier crash` (Clément Fournier,
  2024-06-01), which is itself in the public repo.
- Cinnamon's `build-llvm.sh` conveniently clones the `oowekyala/llvm-project`
  fork's `tilefirst-llvm` branch (whose HEAD, `195baaa`, has since advanced), but
  the **pinned/checked-out commit `6f89431` is unmodified public LLVM** — the
  local checkout is clean (no working-tree patches) and the SHA resolves in
  `llvm/llvm-project`. So `setup.sh --with-cinm` fetches from the canonical
  public `llvm/llvm-project` and does **not** depend on the fork.

## What it is used for in this artifact

CINM is a Fig. 9 baseline (`experiments/fig09_pareto/`): the drivers in
`experiments/fig09_pareto/baselines/` emit each kernel as a `cinm.compute` GEMM,
run `cinm-opt --cinm-tiling` to obtain CINM's compiler tiling, and re-cost that
mapping on the PNMAX analytical model (UPMEM-normalized). Inputs used: the
`testbench/*.mlir` shapes (present here) and hardcoded UPMEM HW-reference numbers
in the driver (documented as originating from `artifact/plot/exp-fig-11.txt`,
which is not needed at runtime).

## Build (validated fast path)

`setup.sh --with-cinm` fetches + builds the pinned public LLVM into
`external/cinm/third-party/llvm/`, then builds `cinm-opt` into
`external/cinm/build/` (both gitignored). The full from-scratch LLVM build is
long on small machines (~5 min on a 64-core machine) and is exercised via ./setup.sh --with-cinm.

The **cinm side** was validated here by building against the existing local LLVM
18.1.6 build (out-of-tree, sources untouched):

- **Requires clang.** `g++` rejects an injected-class-name template in
  `CinmToCnm.cpp` (`ConvertElementWiseToCnm<CinmOp>(...)`); `clang++` accepts it.
  `setup.sh --with-cinm` therefore configures cinm with clang.
- `cmake -G Ninja -DLLVM_DIR=… -DMLIR_DIR=… -DCMAKE_CXX_COMPILER=clang++` then
  `ninja cinm-opt` → **~11.5 s**, producing `build/bin/cinm-opt` (28 MB).
- Smoke: `cinm-opt --version` → LLVM 18.1.6; `cinm-opt --cinm-tiling <gemm.mlir>`
  emits the tiled MLIR the drivers parse (exit 0).

## Local modifications

None to the Cinnamon sources. Snapshot via `git archive` of `f776a07` — tracked
files only (no `.git`, no `build/`, and the `third-party/llvm` submodule is
excluded; it is fetched by `setup.sh --with-cinm`). `LICENSE` added as noted.

`README.md` reconciliation (2026-07-17): the snapshot as originally
exported carried the earlier jun2024-era upstream README (blob `61992d0`, the
short ".env / LLVM_BUILD_DIR" setup notes — genuine upstream content, but not
the pinned commit's) rather than `f776a07`'s README. It has been replaced with
the byte-exact `README.md` of the pinned commit (upstream blob `ae444bc`), so
every shipped file now matches `f776a07` verbatim. Nothing in this artifact
uses the README's setup flow — builds go through `setup.sh --with-cinm`.
