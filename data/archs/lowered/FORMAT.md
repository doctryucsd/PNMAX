# Lowered Architecture YAML Schema

This directory holds architecture specs in PNMAX's flat, "lowered" format —
the single-file YAML consumed directly by the runtime parser
(`src/pnmax/database/arch.py`). GPAM graph specs (`data/archs/`, topology +
tech_db files) are lowered into this format by the `gpam` front-end.

- Baselines live in `data/archs/lowered/baseline/<arch_id>.yaml`.
- Geometry sweeps live in `data/archs/lowered/geometry_sweep/<arch_id>/<arch_id>__banks-<n>__hmat-<n>__vmat-<n>.yaml`.
- Canonical architecture ids are `upmem`, `hbm_pim`, and `attacc`.

Only the fields documented here are part of the supported schema. Legacy keys such
as `hierarchy_names`, `organization`, `bank_size`, `buffer_size`, `pu_factor`,
`pu_freq`, `pu_concurrency`, `t_rcd`, `t_rp`, `t_ras`, `pu_flops`, and flat
`*_energy` cost keys are rejected by the parser.

## Example

```yaml
name: upmem
cache:
  mode: wram
specs:
  row_bytes: 1KB
  bank_bytes: 32MB
  cache_bytes: 32KB
  banks_per_pu: 2
  pu_frequency: 350Mhz
  simd_lanes: 16
hierarchy:
  L2: { kind: pu, count: 8 }
  L3: { kind: chip, count: 8 }
  L4: { kind: rank, count: 2 }
  L5: { kind: channel, count: 20 }
bandwidth:
  host_device: 25.6GB/s
unit_costs:
  bank_to_pu_load: { latency: 2.375, energy: 178 }
  pu_to_bank_store: { latency: 2.375, energy: 178 }
  host_pu_transfer: { latency: 4, energy: 178 }
  bank_to_bank_transfer: { latency: 100, energy: 0 }
  host_bank_row_conflict: { latency: 31, energy: 534 }
  pu_compute: { latency: 4.75, energy: 59 }
  row_change: { latency: 25, energy: 534 }
```

## Conventions

- Operation names use either `<src>_to_<dst>_<action>` for transfers or
  `<resource>_<event>` for penalties and state changes.
- `unit_costs.<operation>.latency` is always measured in PU cycles.
- `unit_costs.<operation>.energy` is always measured in pJ per modeled event.
- Every field description below follows one format: meaning, type/unit, whether it
  is required, and which runtime component consumes it.
- `total_area_mm2` is optional metadata and is not consumed by runtime.

## Top-Level Fields

- `name`: Architecture id, `string`, required, consumed by parser and reporting.
- `cache`: Cache interpretation block, `mapping`, optional, consumed by the
  analytical cache model.
- `specs`: Core capacity and execution parameters, `mapping`, required, consumed by
  `arch.py` and the analytical model.
- `hierarchy`: Logical hierarchy levels, `mapping`, required, consumed by
  `arch.py`, the analytical model, and DSE legality checks.
- `bandwidth`: Host/device bandwidth block, `mapping`, required, consumed by
  `arch.py` and analytical streaming calculations.
- `unit_costs`: Nested latency/energy operation table, `mapping`, required,
  consumed by `arch.py` and analytical runtime cost evaluation.
- `total_area_mm2`: Sweep metadata for area studies, `number`, optional, not
  consumed by runtime.

## Cache Block

- `cache.mode`: Cache interpretation mode, `string` in `{wram, regfile}`,
  required when `cache` exists, consumed by the analytical cache simulator.
- `cache.vector_width_bits`: Register-file lane width, `integer bits`, required
  for `regfile`, consumed by the analytical cache simulator.

## Specs Block

- `specs.row_bytes`: DRAM row payload size, `size string or integer bytes`,
  required, consumed by `arch.py`, analytical row-fit checks, and memory
  footprint calculations.
- `specs.bank_bytes`: Capacity of one bank, `size string or integer bytes`,
  required, consumed by `arch.py` to derive `num_rows`.
- `specs.cache_bytes`: Total per-PU cache budget interpreted by `cache.mode`,
  `size string or integer bytes`, required, consumed by the analytical cache
  simulator.
- `specs.banks_per_pu`: Number of banks attached to one PU, `integer`, required,
  consumed by analytical row-usage and row-change calculations.
- `specs.pu_frequency`: PU clock frequency, `frequency string or numeric Hz`,
  required, consumed by `arch.py` for cycle/time conversions.
- `specs.simd_lanes`: Maximum level-0 parallel lanes within one PU, `integer`,
  required, consumed by scheduling legality checks and DSE capacity checks.
- `specs.n_lines`: Chosen interconnect line count for generated interconnect
  variants, `integer`, optional metadata consumed only by parser validation and
  reporting; it does not affect runtime behavior.
- `specs.add_tree_width`: Optional activation-tree width for base-die analytical
  modeling, `integer > 0`, optional, consumed only when base-die mode is enabled
  in the analytical model.

## Hierarchy Block

Each hierarchy level is keyed as `L<number>` and must contain exactly:

- `kind`: Logical resource kind, `string` in `{pu, bank, bg, chip, pch, rank, channel, stack}`,
  required, consumed by parser metadata and downstream reporting.
- `count`: Number of units at that level, `integer`, required, consumed by
  `arch.py`, analytical capacity checks, and DSE legality checks.

## Bandwidth Block

- `bandwidth.host_device`: Aggregate host/device bandwidth, `bandwidth string or numeric B/s`,
  required, consumed by analytical streaming cost calculations.

## Unit Costs

Every operation entry must have exactly:

- `latency`: Operation latency, `number of PU cycles`, required, consumed by
  analytical latency estimation.
- `energy`: Operation energy, `number of pJ`, required, consumed by analytical
  energy estimation.

The required operation names are:

- `bank_to_pu_load`: Bank-to-PU local load cost.
- `pu_to_bank_store`: PU-to-bank local store cost.
- `host_pu_transfer`: Host-to-PU transfer cost used by streaming/save transfers.
- `bank_to_bank_transfer`: Cross-bank transfer cost.
- `host_bank_row_conflict`: Additional row-conflict penalty on host-bank access.
- `pu_compute`: PU arithmetic cost.
- `row_change`: Bank row-change penalty.

Optional operation names:

- `base_die`: Optional base-die activation-tree cost used only when base-die
  mode is enabled in the analytical model.
