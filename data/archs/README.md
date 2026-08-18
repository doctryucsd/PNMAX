# `data/archs/` — architecture specifications

PNMAX architectures are authored as **GPAM** (Graph-based Parameterized
Architecture Model) *topology + technology-database* pairs and then **lowered**
to a flat, single-file YAML that the runtime analytical model consumes. This
directory holds both forms.

## Layout

| Path | Contents |
|---|---|
| `baseline/` | The three authoritative GPAM baselines — `upmem`, `hbm_pim`, `attacc` — each a `<arch>.yaml` topology plus a `<arch>_tech_db.yaml` technology database. |
| `geometry_exemplar/` | One sweep variant authored end-to-end as a GPAM pair, demonstrating that generated sweep points flow through the same GPAM → `gpam-lower` contract as the baselines. |
| `examples/` | Standalone GPAM samples (e.g. `hbm_pim`, `simdram`) for reference and for exercising `gpam-info` / `gpam-lower`; not consumed by the experiment buttons. |
| `lowered/` | The flat, single-file arch YAMLs actually loaded at runtime — the shipped `baseline/` and `activation/` sets plus the generated sweep families (`geometry_sweep/`, `geometry_sweep_host_congested/`, `buffer_sweep/`, `interconnect_sweep/`). |

## Two schemas, two docs

- **Flat (lowered) schema** — the format the runtime parser
  (`src/pnmax/database/arch.py`) accepts: see
  [`lowered/FORMAT.md`](lowered/FORMAT.md).
- **Authoring GPAM specs** (topology + tech_db, `gpam-lower`, `gpam-info`,
  the `--check`/`--out` contract) — see
  [Authoring GPAM specs](#authoring-gpam-specs) below; the DSE walkthrough is
  in the top-level [`README.md`](../../README.md#using-pnmax-for-your-own-dse).

The generated sweep families under `lowered/` are DERIVED outputs: parameter
deltas of the GPAM baselines materialized through the PPA characterization DB,
generated as the standard flow by `setup.sh` (stage script
`data/ppa/generate_ppa.sh`) and not tracked in the repository.
`data/ppa/PROVENANCE.md` documents the recipes, and every generated YAML
records its inputs in its header comment. Shipped as-is (not generated at setup):
`lowered/baseline/` (hand-lowered from the GPAM baselines) and
`lowered/activation/` (frozen characterization inputs with no in-repo
generator, including the pinned fig17 search base in
`lowered/activation/search_base/`).

## Authoring GPAM specs

The GPAM specs are the authoritative architecture definitions; the flat
files under `lowered/` are generated from them by the `gpam-lower` pass.
Costs live in the `<arch>_tech_db.yaml`, structure in the topology YAML.
Keep the replication ranges of nodes and edges consistent — `gpam-lower`
rejects inconsistent graphs. Bank-to-bank transfer must be authored as a
direct bank↔bank edge or through a dedicated bus/xbar/router hub; routing
it through the host node is not a supported topology (the host link models
the separate per-column host↔PU transfer).

```sh
# Lower a GPAM spec to a flat arch (writes the file; exit 0 == lowering ok)
uv run gpam-lower --arch data/archs/baseline/hbm_pim.yaml \
                  --out  /tmp/my_archs/hbm_pim_relowered.yaml

# Verify a checked-in lowered file still matches its GPAM spec (exit 0 == match)
uv run gpam-lower --arch data/archs/baseline/hbm_pim.yaml \
                  --check data/archs/lowered/baseline/hbm_pim.yaml
```

`gpam-lower` exit codes: `0` ok, `2` spec load error, `3` lowering error,
`4` `--check` mismatch (a missing or unreadable reference also counts as a
mismatch). `--out` is written whenever lowering itself succeeds, even if a
concurrent `--check` reports a mismatch. `setup.sh` runs the `--check` form
for all three baselines; run just that gate with
`./setup.sh --check-gpam-only`.

The `geometry_exemplar/` pair is one sweep variant authored end-to-end as a
GPAM spec, with the generator-mode values for that geometry in its tech_db:
it lowers through the same GPAM → `gpam-lower` contract as the baselines
while its numbers come from the PPA DB. The builder route records extra
derivation metadata the GPAM spec does not carry (a `pim_overhead` field and
its own `total_area_mm2` summary), so the byte-level `--check` contract
applies only between the baselines and `lowered/baseline/`.
