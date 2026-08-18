# Deriving the UniNDP baseline mappings

The PNMAX Fig. 10 Pareto grid compares PNMAX-searched mappings against the
**UniNDP** compiler's own best mapping for each of the 9 kernels on each of the
two baseline architectures (UPMEM, HBM-PIM). Those 18 UniNDP baselines are
**derived at run time** by compiling each kernel with the vendored UniNDP and
translating UniNDP's chosen partition into a PNMAX nested-loop workload YAML —
they are *not* shipped as static data.

The PNMAX development repo carries a reference copy of the 18 derived files at
`workloads/unindp_baseline/<token>_<arch>.yaml`; those serve only as the
**validation reference** for the converter (see "Validation" below) and are
not part of this repository — the example paths below that cite them are the
historical validation invocation, not a runnable artifact command.

## Pipeline

Two stages:

1. **Compile** — run UniNDP's `compile.py` for the kernel's GEMM shape on the
   target architecture. UniNDP performs its own mapping search and writes a log
   containing a `best_design:` line (its chosen partition + SIMD factors).
2. **Convert** — translate that `best_design` into a PNMAX `Bound` (nested-loop
   schedule) + derived `Problem`, merged with a template YAML that supplies the
   static `TensorAttr` / `Compute` / `Layout-Pragma` fields.

Both stages are wrapped by `derive_baselines.sh` in this directory:

```bash
external/unindp/derive_baselines.sh <name> <arch> <M> <K> <N> <B> \
    <template.yaml> <out.yaml> [reference.yaml]
```

- `_compile_driver.py` — namespace-package launcher for `compile.py` (UniNDP
  mixes top-level and `from ..` relative imports, so it must run as a
  sub-package; the launcher wires this up exactly like the artifact's
  `helpers/run_unindp_compile.py`). Runs from a scratch cwd with a `config`
  symlink so no artefacts land in the vendored tree.
- `derive_baseline_yaml.py` — self-contained `best_design` → workload-YAML
  converter. It is a faithful port of the `dump-yaml` logic in
  `helpers/run_unindp_compile.py`, minus the final
  `pnmax.database.Workload.from_dict` schema check (so the derivation runs
  without importing the `pnmax` package).

### Where the canonical converter lives

The artifact's in-tree converter is **`helpers/run_unindp_compile.py`** (ported
from the PNMAX development repo by the framework-import task). It has three sub-commands:
`compile` (runs UniNDP), `summary` (pretty-prints a design), and `dump-yaml`
(the log→YAML conversion). It imports `pnmax.database.Workload` to validate the
emitted mapping, so it depends on the `pnmax` package. `derive_baselines.sh`
here intentionally does **not** depend on `pnmax` so the derivation can be
demonstrated/executed standalone; the in-tree helper is the production path used
by the Fig. 10 driver.

## UniNDP `mm` workload size ⇄ PNMAX Problem

UniNDP compiles every GEMM/MV as `-W mm -S M K N B` where
`M`=input rows, `K`=reduction, `N`=output columns, `B`=batch. The converter maps
UniNDP's partition factors onto PNMAX dims as:

| UniNDP factor | PNMAX dim |
|---|---|
| M (input rows)   | **K** |
| K (reduction)    | **C** |
| N (output cols)  | **P** |
| B (batch)        | **N** |

So to reproduce a reference `<token>_<arch>.yaml` from a workload template with
PNMAX Problem `(N, K, P, Q, C, R, S)`, invoke UniNDP with the TRUE kernel dims
(UniNDP pads per architecture itself, e.g. bert-3 reduction 768→780 on UPMEM,
vgg16-0 reduction 27→32 on HBM-PIM / 27→28 on UPMEM):

- fc/bmm kernels (`Q=R=S=1`): `M = N·K` (batch/head dim folds into rows),
  `K = C`, `N = P`, `B = 1`;
- conv2d kernels: `M = N·K·P·Q` (im2col rows), `K = C·R·S`, `N = 1`, `B = 1`.

Per-kernel sizes (identical for both `hbm_pim` and `upmem`):

| token | display name | `-S M K N B` |
|---|---|---|
| alexnet-4   | conv2d_alexnet-4  | `43520 2304 1 1` (reference input: M padded +256 over the im2col 43264; documented override) |
| bert-3      | bmm_bert-3        | `768 768 128 1` |
| bert-72     | fc_bert-72        | `768 768 1 1` |
| llama128-6  | fc_llama128-6     | `14336 4096 1 1` |
| llama256-3  | bmm_llama256-3    | `4096 128 256 1` |
| llama512-3  | bmm_llama512-3    | `4096 128 512 1` |
| resnet50-2  | conv2d_resnet50-2 | `200704 576 1 1` |
| resnet50-53 | fc_resnet50-53    | `1000 2048 1 1` |
| vgg16-0     | conv2d_vgg16-0    | `3211264 27 1 1` |

The production deriver cross-checks this table against each template's Problem
at run time and refuses to compile on a mismatch (typo guard).

The compile flags used for the reference baselines are UniNDP quicksearch with
`-Q -K 30` (predictor-aided search, keep top-30); `PO2=false`,
`ALLOW_UNDER_UTILIZE=false`.

## Example

```bash
external/unindp/derive_baselines.sh \
    bmm_bert-3 hbm_pim 768 768 128 1 \
    workloads/unindp_baseline/bert-3_hbm_pim.yaml \    # template (static fields)
    results/unindp_baseline/bert-3_hbm_pim.yaml \      # derived output
    workloads/unindp_baseline/bert-3_hbm_pim.yaml      # optional: diff reference
```

## SIMD-lanes clamp (intra-PU split)

UniNDP's `mm` compiler can select an intra-PU split whose element count
(`simd_k · simd_l`) exceeds the physical SIMD lane count of the modeled PU
(e.g. 16×16 = 256 elements on a 16-lane HBM-PIM PU). The PNMAX mapping format
prices Bound level 0 as one SIMD issue, so transplanting an oversubscribed
split verbatim would cost the baseline optimistically. The converter therefore
rebalances such splits to the largest lanes-legal split (halving the larger
even factor until legal, e.g. 16×16 → 4×4 at 16 lanes; displaced factors move
into the level-1 temporal loop). The reference baseline mappings carry the same
normalization. The production deriver passes each architecture's
`specs.simd_lanes` automatically; the standalone converter takes
`--simd-lanes`.

## Validation status (executed)

The full derivation (compile → convert → clamp) was executed for **all 18
(kernel, architecture) pairs** and compared against the reference
baselines:

- **Effective `Problem` — identical** on all 18 (identity `Dilation`/`Stride`
  keys inherited from the conv2d templates excluded).
- **`Bound` (all levels, including the intra-PU L1/L0 split) — identical** on
  all 18 after the SIMD-lanes clamp.
- **TensorAttr / Layout-Pragma — identical** (template-supplied).

The converter transcribes UniNDP's `best_design` into the PNMAX schema
exactly; with the SIMD-lanes clamp the derivation reproduces the reference
baseline mappings deterministically.

> Runtime note: the production Fig. 10 driver invokes the in-tree
> `helpers/run_unindp_compile.py` (which additionally validates the emitted YAML
> through `pnmax.database.Workload`). That path depends on the `pnmax` package;
> the standalone scripts here do not and are what was executed for this report.
