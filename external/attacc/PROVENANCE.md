# Provenance: AttAcc simulator (Ramulator2-based)

## AttAcc wrapper

- **Upstream:** https://github.com/qianxu1998/attacc_simulator
  (fork of the ASPLOS'24 release https://github.com/scale-snu/attacc_simulator)
- **Vendored commit:** `94d1d3e647857ec3153c193c8412dece5c8e869b`
- **License:** MIT (see `LICENSE`; © 2024 Scalable Computer Architecture
  Laboratory (SCAL), Seoul National University)
- **Paper:** AttAcc! Unleashing the Power of PIM for Batched Transformer-based
  Generative Model Inference (ASPLOS 2024)

## Bundled Ramulator2 (`ramulator2/`)

- **Upstream:** https://github.com/CMU-SAFARI/ramulator2
- **Base commit:** `b7c70275f04126c647edb989270cc429776955d1`
  (the commit `set_pim_ramulator.sh` pins)
- **License:** MIT (see `ramulator2/LICENSE`; © 2023 SAFARI Research Group at
  ETH Zurich and Carnegie Mellon University)
- **State:** the vendored `ramulator2/` is the **prepared** source — Ramulator2
  at `b7c70275` with AttAcc's PIM extensions applied, exactly as
  `set_pim_ramulator.sh` does:
  1. copy AttAcc's PIM sources from `pim_ramulator_src/` (HBM3-PIM device,
     PIM controllers/schedulers, linear mappers, trace recorder, AttAcc bank/BG/
     buffer configs) into the Ramulator2 tree;
  2. apply the 22 patches in `pim_ramulator_src/patches/*.patch`.

  Vendoring the prepared source (rather than a submodule + build-time patching)
  keeps the artifact self-contained and makes `setup.sh` a plain `cmake && make`
  with no git / patch-idempotency concerns. The unmodified upstream diff is fully
  recoverable: base = `b7c70275`, changes = `pim_ramulator_src/` + its patches.

## What it is used for in this artifact

AttAcc is the **HBM-PIM attention-accelerator validation baseline**
(`experiments/fig08_tab4_validation/` — the Fig. 8 + Table 4 latency/energy
validation — and the `experiments/attacc_area/` area estimate). PNMAX's
analytical model is
validated against AttAcc's Ramulator2-based cycle model. The PNMAX adapter is
`src/pnmax/simulators/attacc/` and the CLIs `run-attacc-analytical` /
`attacc-validation`, which invoke the built `ramulator2` binary.

## Build

`setup.sh` builds Ramulator2 into `external/attacc/ramulator2/build/`
(gitignored). Verified on this machine:

- Toolchain: g++ 13.3.0, cmake 3.28.3.
- `cmake ..` + `make -j$(nproc)` → **~10 s** wall (64-core Threadripper).
- Produced binary: `external/attacc/ramulator2/build/ramulator2`.
- Smoke test: `gen_trace_attacc_bank.py` → `ramulator2 -f attacc_bank.yaml`
  runs the PIMDRAM model and reports `memory_system_cycles` (bank-level PIM
  MAC/softmax requests) — exit 0.

### Build-time external dependencies (CMake FetchContent)

Ramulator2's `CMakeLists.txt` fetches three header-focused libraries from GitHub
at configure time into `ext/` (gitignored), matching upstream behaviour:

| lib | repo | pinned tag |
|---|---|---|
| yaml-cpp | github.com/jbeder/yaml-cpp | `yaml-cpp-0.7.0` |
| spdlog   | github.com/gabime/spdlog   | `v1.11.0` |
| argparse | github.com/p-ranav/argparse | `v2.9` |

These pinned fetches are the only network access the AttAcc build needs.

## Local modifications

None beyond the `set_pim_ramulator.sh` preparation described above (which is
AttAcc's own upstream recipe). Checked-in prebuilt binaries from the source
snapshot (`ramulator.out`, `libramulator.so`, `ramulator2/ramulator2`) and all
`build/` / `ext/` trees were **excluded** from vendoring — they are rebuilt by
`setup.sh`.
