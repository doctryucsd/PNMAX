from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any

from tqdm import tqdm
import yaml

from pnmax.database import Workload

from .operations import PU, Addr, Load, Operation, Stream, ops2instructions


from pnmax.paths import repo_root

PROJECT_ROOT = repo_root()
EXTERNAL_PATH = PROJECT_ROOT / "external"
CONFIG_DIR = EXTERNAL_PATH / "unindp" / "config"
ARCH_CONFIG_BY_TARGET = {
    "upmem": CONFIG_DIR / "upmem.yaml",
    "hbm_pim": CONFIG_DIR / "hbm-pim.yaml",
}


class TensorType(Enum):
    F = "F"
    In = "In"
    Out = "Out"


@dataclass
class Compute:
    f_idx: int
    in_idx: int
    out_idx: int
    bg_idx: int
    pu_idx: int
    temporal_step: int


class Codegen:
    def __init__(self, workload: Workload) -> None:
        # register workload
        self.workload = workload
        self._layout = self.workload.require_unindp_layout()

        #  Compute strides once to avoid O(N) math in loops
        # level_strides[level_idx][loop_var]
        self._level_strides = self._precompute_strides()

        # tile information
        self._tiles_per_row = 1
        self._row_bytes_by_arch: dict[str, int] = {}
        self._interleaving = self._layout.interleaving
        self._f_tile_size, self._in_tile_size, self._out_tile_size = (
            self._get_tile_sizes()
        )
        self._f_tile_size_bytes = self._f_tile_size * (
            self.workload.tensor_attrs["F"].bits // 8
        )
        self._in_tile_size_bytes = self._in_tile_size * (
            self.workload.tensor_attrs["In"].bits // 8
        )
        self._out_tile_size_bytes = self._out_tile_size * (
            self.workload.tensor_attrs["Out"].bits // 8
        )

    def gen_instructions(
        self,
        arch: str,
        sample_rate: float,
        outdir_path: Path | None = None,
        pbar: bool = False,
        dbg_op_types: list[str] = [],
        single_pu: bool = False,
        verbose: int = 0,
    ) -> list[tuple[int, list[int], list[Any]]]:
        arch = arch.lower()
        self._tiles_per_row = self._resolve_tiles_per_row(arch)
        computes = self._gen_computes(sample_rate, pbar, single_pu)
        # Sort computes by temporal step
        computes.sort(key=lambda c: (c.temporal_step, c.bg_idx, c.pu_idx))

        if verbose >= 2:
            print("Generated Computes:")
            for compute in computes:
                print(f"  {compute}")
            print()

        # get tile information
        tile_locations = self._get_tiles(computes, single_pu)

        # generate operations
        ops = self._gen_operations(computes, tile_locations, arch)
        if dbg_op_types:
            dbg_op_types_lower = {op_type.lower() for op_type in dbg_op_types}
            ops = [
                op for op in ops if op.__class__.__name__.lower() in dbg_op_types_lower
            ]
        self.operation_summary = self._get_op_summary(ops)

        if verbose >= 2:
            print("Generated Operations:")
            for op in ops:
                print(f"  {op}")
            print()

        instructions, row_changes = ops2instructions(ops, arch)

        if verbose >= 1:
            print(f"Row Changes: {row_changes}")

        if verbose >= 2:
            print("Generated Instructions:")
            for instruction in instructions[0][2]:
                print(f"  {instruction}")
            print()

        if outdir_path is not None:
            # dump operations and instructions to file for debugging
            outdir_path.mkdir(parents=True, exist_ok=True)
            with open(outdir_path / "operations.txt", "w") as f:
                f.write("Generated Operations:\n")
                for op in ops:
                    f.write(f"{op}\n")
                f.write("\n")
                f.write("Generated Instructions:\n")
                for instruction in instructions[0][2]:
                    f.write(f"{instruction}\n")

        return instructions

    def _get_op_summary(self, ops):
        summary: dict[str, int] = {}
        summary_get = summary.get
        for op in ops:
            op_name = op.__class__.__name__
            summary[op_name] = summary_get(op_name, 0) + 1
        return summary

    def _resolve_tiles_per_row(self, arch: str) -> int:
        stored_tensor_types: list[TensorType] = []
        if not self._layout.streaming.F:
            stored_tensor_types.append(TensorType.F)
        if not self._layout.streaming.Out:
            stored_tensor_types.append(TensorType.Out)
        if not stored_tensor_types:
            raise ValueError("UniNDP codegen requires at least one stored tensor.")

        capacities = [
            self._max_tiles_per_row(tensor_type, arch)
            for tensor_type in stored_tensor_types
        ]
        if any(capacity <= 0 for capacity in capacities):
            raise ValueError(
                "UniNDP codegen encountered a stored tensor that does not fit in a DRAM row."
            )
        return min(capacities)

    def _max_tiles_per_row(self, tensor_type: TensorType, arch: str) -> int:
        row_bits = self._row_bytes_from_arch(arch) * 8
        ele_bits = self.workload.tensor_attrs[tensor_type.value].bits
        tile_bits = self._tensor_tile_size(tensor_type) * ele_bits
        if tile_bits <= 0:
            raise ValueError(
                f"Tile bits must be positive when deriving row capacity for tensor {tensor_type.value}."
            )
        return row_bits // tile_bits

    def _row_bytes_from_arch(self, arch: str) -> int:
        cached = self._row_bytes_by_arch.get(arch)
        if cached is not None:
            return cached

        arch_path = ARCH_CONFIG_BY_TARGET.get(arch)
        if arch_path is None:
            raise ValueError(f"Unsupported UniNDP arch '{arch}'.")

        payload = yaml.safe_load(arch_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid UniNDP arch file: {arch_path}")

        num_columns = int(payload["co"])
        column_width_bits = int(payload["co_w"])
        row_bits = num_columns * column_width_bits
        if row_bits % 8 != 0:
            raise ValueError(
                f"UniNDP arch row width must be byte-aligned, got {row_bits} bits in {arch_path}."
            )
        row_bytes = row_bits // 8
        self._row_bytes_by_arch[arch] = row_bytes
        return row_bytes

    def _precompute_strides(self) -> list[dict[str, int]]:
        """
        Computes the positional weight (stride) for each level.
        Logic: Stride(Dim, Level) = Product of Extents(Dim) in all levels < Level.
        """
        strides: list[dict[str, int]] = []
        # Start with a base stride of 1 for all canonical dimensions at Level 0
        # (Level 0 is the innermost execution, its offsets directly index elements)
        current_cumulative_stride = {
            dim.lower(): 1 for dim in self.workload.problem.loop_vars
        }

        # We iterate through levels in increasing order (0, 1, 2...)
        # Note: workload.bounds.levels() provides defined levels (e.g., (3, 2, 1, 0))
        defined_levels = sorted(self.workload.bounds.levels())

        for level in defined_levels:
            # 1. Store the stride for the current level's offsets
            strides.append(current_cumulative_stride.copy())

            # 2. Update the cumulative stride for the NEXT level
            # We get the mapping of {dim: extent} for this specific level
            # We iterate through canonical bounds for this level.
            level_mapping = {}
            for bound in self.workload.bounds.get_level(level):
                level_mapping[bound.loop_var.lower()] = bound.extent

            for dim in current_cumulative_stride:
                # Multiply by the extent found at this level (default to 1 if not present)
                current_cumulative_stride[dim] *= level_mapping.get(dim, 1)

        return strides

    def _get_tile_sizes(self) -> tuple[int, int, int]:
        if self._layout.sharing.block:
            return (
                self._tensor_tile_size(TensorType.F),
                self._tensor_tile_size(TensorType.In),
                self._tensor_tile_size(TensorType.Out),
            )
        else:
            bounds = self.workload.get_level_bounds(0)
            total_size = math.prod([bound for bound in bounds.values()])
            return total_size, total_size, total_size

    def _tensor_tile_size(self, tensor_type: TensorType) -> int:
        if tensor_type in (TensorType.F, TensorType.Out):
            bounds = self.workload.get_tensor_level_bounds(tensor_type.name, 0)
            return math.prod([bound for bound in bounds.values()])
        if tensor_type == TensorType.In:
            # Deduplicate computed indices to account for access expressions that overlap.
            unique_indices: set[tuple[int, ...]] = set()
            loop_bounds = self.workload.bounds.get_level(0)
            ranges = [range(loop_bound.extent) for loop_bound in loop_bounds]
            loop_vars = [loop_bound.loop_var.lower() for loop_bound in loop_bounds]
            base_vector = [0 for _ in loop_vars]
            for offsets in product(*ranges):
                var_map = {
                    lv: base + off
                    for lv, base, off in zip(loop_vars, base_vector, offsets)
                }
                indices = self.workload.evaluate_tensor_indices(
                    tensor_type.value, var_map
                )
                unique_indices.add(indices)
            return len(unique_indices)
        raise ValueError(f"Unsupported tensor type: {tensor_type}")

    def _gen_operations(
        self,
        computes: list[Compute],
        tile_locations: dict[TensorType, list[Addr]],
        arch: str,
    ) -> list[Operation]:
        compute_ops = self._gen_compute_ops(computes, tile_locations)
        if arch == "hbm_pim":
            save_ops = self._gen_save_ops(tile_locations)
        else:
            save_ops = []
        return compute_ops + save_ops

    def _gen_save_ops(
        self, tile_locations: dict[TensorType, list[Addr]]
    ) -> list[Operation]:
        if self._layout.streaming.Out:
            return []
        ops = []
        out_locations = tile_locations[TensorType.Out]
        for addr in out_locations:
            ops.append(
                Load(
                    addr,
                    self._out_tile_size_bytes,
                    interleaving=self._interleaving,
                    tiles_per_row=self._tiles_per_row,
                )
            )

        return ops

    def _gen_compute_ops(
        self, computes: list[Compute], tile_locations: dict[TensorType, list[Addr]]
    ) -> list[Operation]:
        ret: list[Operation] = []
        n = len(computes)

        for i, compute in enumerate(computes):
            # input streaming
            in_op = Stream(compute.bg_idx, compute.pu_idx, self._in_tile_size_bytes)
            ret.append(in_op)

            # only add PU op at the *end* of a series of same bg_idx and temporal_step
            next_compute = computes[i + 1] if i + 1 < n else None
            if next_compute is not None:
                if (
                    next_compute.bg_idx == compute.bg_idx
                    and next_compute.temporal_step == compute.temporal_step
                ):
                    continue

            f_addr = tile_locations[TensorType.F][compute.f_idx]
            out_addr = tile_locations[TensorType.Out][compute.out_idx]
            pu_op = PU(
                f_addr,
                out_addr,
                self._f_tile_size,
                self._out_tile_size,
                interleaving=self._interleaving,
                tiles_per_row=self._tiles_per_row,
            )
            ret.append(pu_op)

        return ret

    def _get_tiles(
        self, computes: list[Compute], single_pu: bool = False
    ) -> dict[TensorType, list[Addr]]:
        """
        Only F and Out tensors are stored in the banks.
        """
        if not computes:
            return {TensorType.F: [], TensorType.In: [], TensorType.Out: []}

        if single_pu:
            if not all(c.bg_idx == 0 and c.pu_idx == 0 for c in computes):
                raise AssertionError(
                    "single_pu requires all computes to target bg_idx=0, pu_idx=0."
                )

        max_f = -1
        max_out = -1
        max_bg = -1
        max_pu = -1
        for compute in computes:
            f_idx = compute.f_idx
            out_idx = compute.out_idx
            bg_idx = compute.bg_idx
            pu_idx = compute.pu_idx
            if f_idx > max_f:
                max_f = f_idx
            if out_idx > max_out:
                max_out = out_idx
            if bg_idx > max_bg:
                max_bg = bg_idx
            if pu_idx > max_pu:
                max_pu = pu_idx

        f_locations = [(-1, -1)] * (max_f + 1)
        out_locations = [(-1, -1)] * (max_out + 1)

        for compute in computes:
            f_idx = compute.f_idx
            if f_locations[f_idx][0] < 0:
                f_locations[f_idx] = (compute.bg_idx, compute.pu_idx)
            out_idx = compute.out_idx
            if out_locations[out_idx][0] < 0:
                out_locations[out_idx] = (compute.bg_idx, compute.pu_idx)

        def assign_addrs(
            locations: list[tuple[int, int]], tile_size: int
        ) -> list[Addr]:
            if not locations:
                return []
            # Track per-PU row/offset so tiles pack contiguously in rows.
            used_rows = [[0] * (max_pu + 1) for _ in range(max_bg + 1)]
            tiles_in_row = [[0] * (max_pu + 1) for _ in range(max_bg + 1)]
            addrs = [Addr(-1, -1, -1, -1, -1)] * len(locations)
            max_tiles = self._tiles_per_row
            for tile_id, (bg_idx, pu_idx) in enumerate(locations):
                if bg_idx < 0:
                    continue
                tile_offset = tiles_in_row[bg_idx][pu_idx]
                # Start a new row when the per-row capacity is exceeded.
                if tile_offset >= max_tiles:
                    tiles_in_row[bg_idx][pu_idx] = 0
                    used_rows[bg_idx][pu_idx] += 1
                    tile_offset = 0
                row_id = used_rows[bg_idx][pu_idx]
                byte_id = tile_offset * tile_size
                addrs[tile_id] = Addr(bg_idx, pu_idx, row_id, byte_id, tile_size)
                tiles_in_row[bg_idx][pu_idx] = tile_offset + 1
            return addrs

        return {
            TensorType.F: assign_addrs(f_locations, self._f_tile_size_bytes),
            TensorType.In: [],
            TensorType.Out: assign_addrs(out_locations, self._out_tile_size_bytes),
        }

    def _gen_computes(
        self, sample_rate: float, pbar: bool, single_pu: bool
    ) -> list[Compute]:
        if single_pu:
            l3_ranges = [range(1) for _ in self.workload.problem.loop_vars]
            l2_ranges = [range(1) for _ in self.workload.problem.loop_vars]
        else:
            l3_ranges = self._get_ranges(3)
            l2_ranges = self._get_ranges(2)
        l1_ranges = self._get_ranges(1)

        computes: list[Compute] = []
        # tensortype -> bg_idx -> pu_idx -> (indices) -> [tile_ids]
        tile_dict: dict[
            TensorType, dict[int, dict[int, dict[tuple[int, ...], list[int]]]]
        ] = {
            TensorType.F: defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
            TensorType.In: defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
            TensorType.Out: defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        }
        f_counter, in_counter, out_counter = 0, 0, 0
        pu_sharing = self._layout.sharing.pu

        def get_tile_id(
            t_type: TensorType,
            key: tuple[int, ...],
            bg_idx: int,
            pu_idx: int,
            counter: int,
        ) -> tuple[int, int]:
            if not pu_sharing:
                return counter, counter + 1
            pu_map = tile_dict[t_type][bg_idx][pu_idx]
            if key in pu_map:
                return pu_map[key][0], counter
            pu_map[key].append(counter)
            return counter, counter + 1

        # set iteration counts and progress bar
        total_iterations = (
            len(list(product(*l3_ranges)))
            * len(list(product(*l2_ranges)))
            * len(list(product(*l1_ranges)))
        )
        sample_iterations = max(1, int(total_iterations * sample_rate))
        progress = (
            tqdm(total=sample_iterations, desc="Generating Computes") if pbar else None
        )
        iter_counter: int = 0
        stop: bool = False
        # --- NESTED TRAVERSAL ---
        for bg_idx, l3_offsets in enumerate(product(*l3_ranges)):
            l3_dict = self._get_offset_dict(l3_offsets, 3)
            if stop:
                break

            for pu_idx, l2_offsets in enumerate(product(*l2_ranges)):
                l2_dict = self._get_offset_dict(l2_offsets, 2)
                if stop:
                    break

                for step, l1_offsets in enumerate(product(*l1_ranges)):
                    l1_dict = self._get_offset_dict(l1_offsets, 1)
                    start_indices = self._get_start_indices(l3_dict, l2_dict, l1_dict)

                    # Use per-tensor access expressions so dedup respects tensor indexing.
                    f_key = self.workload.evaluate_tensor_indices(
                        TensorType.F.value, start_indices
                    )
                    in_key = self.workload.evaluate_tensor_indices(
                        TensorType.In.value, start_indices
                    )
                    out_key = self.workload.evaluate_tensor_indices(
                        TensorType.Out.value, start_indices
                    )
                    f_id, f_counter = get_tile_id(
                        TensorType.F, f_key, bg_idx, pu_idx, f_counter
                    )
                    in_id, in_counter = get_tile_id(
                        TensorType.In, in_key, bg_idx, pu_idx, in_counter
                    )
                    out_id, out_counter = get_tile_id(
                        TensorType.Out, out_key, bg_idx, pu_idx, out_counter
                    )
                    compute = Compute(f_id, in_id, out_id, bg_idx, pu_idx, step)
                    computes.append(compute)

                    if progress is not None:
                        progress.update(1)
                    iter_counter += 1
                    if iter_counter >= sample_iterations:
                        stop = True
                        break

        if progress is not None:
            progress.close()

        return computes

    def _get_start_indices(
        self, l3_off: dict[str, int], l2_off: dict[str, int], l1_off: dict[str, int]
    ) -> dict[str, int]:
        """Calculates global start indices using pre-calculated strides."""
        start_indices = {}
        for var in self.workload.problem.loop_vars:
            v = var.lower()
            # Index = Sum of (Offset * Stride) for each level
            start_indices[v] = (
                l3_off.get(v, 0) * self._level_strides[2][v]
                + l2_off.get(v, 0) * self._level_strides[1][v]
                + l1_off.get(v, 0) * self._level_strides[0][v]
            )
        return start_indices

    def _get_ranges(self, level: int) -> list[range]:
        ranges: list[range] = []
        if not self.workload.bounds.has_level(level):
            for _ in self.workload.problem.loop_vars.keys():
                ranges.append(range(1))
            return ranges

        level_bounds = self.workload.bounds.get_level(level)
        for bound in level_bounds:
            ranges.append(range(bound.extent))
        return ranges

    def _get_offset_dict(
        self, level_offsets: tuple[int, ...], level: int
    ) -> dict[str, int]:
        offset_dict: dict[str, int] = {}
        if not self.workload.bounds.has_level(level):
            for loop_var in self.workload.problem.loop_vars.keys():
                offset_dict[loop_var.lower()] = 1
            return offset_dict

        level_bounds = self.workload.bounds.get_level(level)
        assert len(level_offsets) == len(level_bounds), (
            f"Expected {len(level_bounds)} offsets, got {len(level_offsets)}."
        )
        for bound, offset in zip(level_bounds, level_offsets):
            offset_dict[bound.loop_var.lower()] = offset
        return offset_dict
