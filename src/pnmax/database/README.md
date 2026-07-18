# Parser Utilities

This package contains the typed YAML parsers used by PNMAX:

- `workload.py`: workload schema parsing and validation
- `arch.py`: architecture schema parsing and derived runtime metrics

Use imports from `pnmax.database`.

## Workload Parser

```python
from pathlib import Path
from pnmax.database import Workload

workload = Workload.from_file(Path("workloads/samples/conv1d.yaml"))
```

Key helpers:

- `workload.require_layout()` returns the resolved layout pragma.
- `workload.require_unindp_layout()` applies UniNDP-specific checks.
- `workload.validate()` checks structural consistency.
- `workload.to_dict()` and `workload.to_yaml()` serialize the current payload.

## Architecture Parser

```python
from pathlib import Path
from pnmax.database import Arch

arch = Arch.from_yaml_file(Path("data/archs/baseline/upmem.yaml"))
```

Parse failures raise `ArchParseError`.

The canonical flat (lowered) architecture schema this parser accepts is
documented in `data/archs/lowered/FORMAT.md` (see `data/archs/README.md` for
how the flat files relate to their GPAM sources). The parser accepts only
these top-level blocks:

- `name`
- `cache`
- `specs`
- `hierarchy`
- `bandwidth`
- `unit_costs`
- optional `total_area_mm2`

### Derived Accessors

- Capacity: `row_size_bytes`, `num_rows`, `num_banks`, `num_pus`
- Scheduling limits: `simd_lanes`, `level_capacity(level)`, `level_capacities`
- Bandwidth: `bandwidth_bytes_per_ns()`
- Unit costs: `unit_costs.<operation>.latency` and `unit_costs.<operation>.energy`
- Optional metadata for activation modeling: `specs.add_tree_width` and
  `unit_costs.base_die`

### Cache Interpretation

If an architecture defines a `cache` block, the analytical model interprets
`specs.cache_bytes` as one total cache budget per PU:

```yaml
cache:
  mode: regfile
  vector_width_bits: 256
```

`mode: wram` uses the same `specs.cache_bytes` capacity but does not require a
vector width.
