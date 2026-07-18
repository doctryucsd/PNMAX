from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

# import path
from pnmax.paths import repo_root

PROJECT_ROOT = repo_root()
EXTERNAL_PATH = PROJECT_ROOT / "external"

if str(EXTERNAL_PATH) not in sys.path:
    sys.path.append(str(EXTERNAL_PATH))

from unindp.tools import LEVEL, OPTYPE, SimConfig

# arch file paths
CONFIG_DIR = EXTERNAL_PATH / "unindp" / "config"
UPMEM_ARCH_PATH = str(CONFIG_DIR / "upmem.yaml")
HBM_PIM_ARCH_PATH = str(CONFIG_DIR / "hbm-pim.yaml")


def bg_id2ch_ra_de(bg_id: int, arch: str) -> Tuple[int, int, int]:
    if arch == "upmem":
        de_id, ra_id = divmod(bg_id, SimConfig.ra)  # type: ignore
        return (0, ra_id, de_id)
    elif arch == "hbm_pim":
        return (bg_id, 0, 0)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


def num2mask(num: int) -> List[bool]:
    return [True] * num


def id2mask(id: int, num_units: int) -> List[bool]:
    assert 0 <= id < num_units, f"id {id} out of range [0, {num_units})"
    mask = [False] * num_units
    mask[id] = True
    return mask


def size2col(size: int) -> int:
    """size in bytes"""
    return math.ceil((size * 8) / SimConfig.co_w)  # type: ignore


def byte_id2col_offset(byte_id: int) -> int:
    return (byte_id * 8) // SimConfig.co_w  # type: ignore


def _num_banks_per_pu() -> int:
    num_banks = SimConfig.bg * SimConfig.ba  # type: ignore
    return num_banks // SimConfig.de_pu[0]  # type: ignore


def old_row_id2bank_row(row_id: int) -> Tuple[int, int]:
    """Legacy row-interleaved mapping from row_id to (bank_id, row_in_bank)."""
    num_banks_per_pu = _num_banks_per_pu()
    bank_id = row_id % num_banks_per_pu
    row_in_bank = row_id // num_banks_per_pu
    return (bank_id, row_in_bank)


def addr2bank_row_col(
    row_id: int,
    byte_id: int,
    tile_size_bytes: int,
    tiles_per_row: int,
    interleaving: bool,
) -> Tuple[int, int, int]:
    """Map logical address (row_id, byte_id) to physical (bank_id, row_in_bank, col_offset)."""
    if not interleaving:
        bank_id, row_in_bank = old_row_id2bank_row(row_id)
        return (bank_id, row_in_bank, byte_id2col_offset(byte_id))

    if tiles_per_row <= 0:
        raise ValueError(
            f"tiles_per_row must be positive for interleaving; got {tiles_per_row}."
        )
    if tile_size_bytes <= 0:
        raise ValueError(
            f"tile_size_bytes must be positive for interleaving; got {tile_size_bytes}."
        )

    tile_in_row, intra_tile_byte = divmod(byte_id, tile_size_bytes)
    if tile_in_row >= tiles_per_row:
        raise ValueError(
            f"byte_id={byte_id} resolves to tile_in_row={tile_in_row}, "
            f"which exceeds tiles_per_row={tiles_per_row}."
        )

    num_banks_per_pu = _num_banks_per_pu()
    global_tile_idx = row_id * tiles_per_row + tile_in_row
    bank_id = global_tile_idx % num_banks_per_pu

    tile_seq_in_bank = global_tile_idx // num_banks_per_pu
    row_in_bank = tile_seq_in_bank // tiles_per_row
    slot_in_row = tile_seq_in_bank % tiles_per_row

    mapped_byte_id = slot_in_row * tile_size_bytes + intra_tile_byte
    col_offset = byte_id2col_offset(mapped_byte_id)
    return (bank_id, row_in_bank, col_offset)


def _is_sys_host_read(inst: Tuple[Any, ...]) -> bool:
    return inst[0] == LEVEL.SYS and inst[1] == OPTYPE.host_read


def _is_de_pu(inst: Tuple[Any, ...]) -> bool:
    return inst[0] == LEVEL.DE and inst[1] == OPTYPE.pu


def _is_de_buf2bk(inst: Tuple[Any, ...]) -> bool:
    return inst[0] == LEVEL.DE and inst[1] == OPTYPE.buf2bk


def _auto_precharge_index(inst: Tuple[Any, ...]) -> int | None:
    if _is_de_pu(inst) or _is_sys_host_read(inst):
        return 9
    if _is_de_buf2bk(inst):
        return 8
    return None


def _utilized_pu_banks(
    pu_num: int, pu_mask: List[bool], relative_bank_id: int
) -> List[int]:
    banks_per_pu = (SimConfig.ba * SimConfig.bg) // pu_num  # type: ignore
    banks: List[int] = []
    for pu_id, enabled in enumerate(pu_mask):
        if pu_id >= pu_num:
            break
        if enabled:
            banks.append(pu_id * banks_per_pu + relative_bank_id)
    return banks


def _inst_bank_rows(
    inst: Tuple[Any, ...],
) -> List[Tuple[Tuple[int, int, int, int], int]]:
    """
    Return [(bank_key, row_id), ...] touched by an instruction.
    bank_key is normalized as (ch_id, ra_id, de_id, bank_id).
    """
    if _is_sys_host_read(inst):
        ch_id, ra_id, de_mask = inst[2], inst[3], inst[4]
        bank_id, row_id = inst[5], inst[6]
        return [
            ((ch_id, ra_id, de_id, bank_id), row_id)
            for de_id, enabled in enumerate(de_mask)
            if enabled
        ]

    if _is_de_pu(inst):
        ch_id, ra_id, de_id = inst[2], inst[3], inst[4]
        pu_num, pu_mask = inst[5]
        op1_bank, op1_row, _ = inst[6]
        op2_bank, op2_row, _ = inst[7]

        rel_bank_rows: List[Tuple[int, int]] = [(op1_bank, op1_row)]
        if op1_bank != op2_bank:
            rel_bank_rows.append((op2_bank, op2_row))

        bank_rows: List[Tuple[Tuple[int, int, int, int], int]] = []
        for rel_bank_id, row_id in rel_bank_rows:
            for bank_id in _utilized_pu_banks(pu_num, pu_mask, rel_bank_id):
                bank_rows.append(((ch_id, ra_id, de_id, bank_id), row_id))
        return bank_rows

    if _is_de_buf2bk(inst):
        ch_id, ra_id, de_id = inst[2], inst[3], inst[4]
        pu_num, pu_mask = inst[5]
        rel_bank_id, row_id, _ = inst[6]
        return [
            ((ch_id, ra_id, de_id, bank_id), row_id)
            for bank_id in _utilized_pu_banks(pu_num, pu_mask, rel_bank_id)
        ]

    return []


def _apply_auto_precharge_policy(instructions: List[Any]) -> int:
    """
    Keep rows open only when the next access to the same physical bank hits the same row.

    Returns the maximum number of distinct rows used by any single bank.
    """
    next_row_by_bank: dict[Tuple[int, int, int, int], int] = {}
    rows_by_bank: dict[Tuple[int, int, int, int], set[int]] = {}

    for i in range(len(instructions) - 1, -1, -1):
        inst = instructions[i]
        auto_precharge_idx = _auto_precharge_index(inst)
        if auto_precharge_idx is None:
            continue

        bank_rows = _inst_bank_rows(inst)
        if not bank_rows:
            continue

        keep_open = True
        for bank_key, row_id in bank_rows:
            next_row = next_row_by_bank.get(bank_key)
            if next_row is None or next_row != row_id:
                keep_open = False
                break

        auto_precharge = not keep_open
        if inst[auto_precharge_idx] != auto_precharge:
            new_inst = list(inst)
            new_inst[auto_precharge_idx] = auto_precharge
            instructions[i] = tuple(new_inst)

        for bank_key, row_id in bank_rows:
            next_row_by_bank[bank_key] = row_id
            if bank_key not in rows_by_bank:
                rows_by_bank[bank_key] = set()
            rows_by_bank[bank_key].add(row_id)

    if not rows_by_bank:
        return 0
    return max(len(rows) for rows in rows_by_bank.values())


@dataclass
class Addr:
    bg_id: int
    pu_id: int
    row_id: int
    byte_id: int
    tile_size_bytes: int

    def __repr__(self) -> str:
        return (
            "Addr("
            f"bg_id={self.bg_id}, "
            f"pu_id={self.pu_id}, "
            f"row_id={self.row_id}, "
            f"column_id={self.byte_id}, "
            f"tile_size_bytes={self.tile_size_bytes})"
        )


class Operation(ABC):
    @abstractmethod
    def instructions(self, arch: str) -> List[Any]: ...

    @abstractmethod
    def __repr__(self) -> str: ...


class Save(Operation):
    def __init__(self, addr: Addr, size: int, info: str | None = None) -> None:
        """size in bytes"""
        self.addr = addr
        self.size = size
        self.info = info

    def instructions(self, arch: str) -> List[Any]:
        raise NotImplementedError("Save instruction method not implemented yet.")

    def __repr__(self) -> str:
        return f"Save(addr={self.addr}, size={self.size}, info={self.info})"


class Stream(Operation):
    def __init__(
        self, bg_id: int, pu_id: int, size: int, info: str | None = None
    ) -> None:
        """size in bytes"""
        self.bg_id = bg_id
        self.pu_id = pu_id
        self.size = size
        self.info = info

    def instructions(self, arch: str) -> List[Any]:
        ch_id, ra_id, de_id = bg_id2ch_ra_de(self.bg_id, arch)
        # Only simulate one channel
        if ch_id > 0:
            return []
        de_mask = id2mask(de_id, SimConfig.de)  # type: ignore
        pu_mask = id2mask(self.pu_id, SimConfig.de_pu[0])  # type: ignore
        col_offset = 0
        col_num = size2col(self.size)

        instr = (
            LEVEL.SYS,
            OPTYPE.host_write_pu_inbuf,
            ch_id,
            ra_id,
            de_mask,
            pu_mask,
            col_offset,
            col_num,
        )

        return [instr]

    def __repr__(self) -> str:
        return f"Stream(bg_id={self.bg_id}, pu_id={self.pu_id}, size={self.size}, info={self.info})"


class PU(Operation):
    def __init__(
        self,
        f_addr: Addr,
        out_addr: Addr,
        f_size: int,
        out_size: int,
        interleaving: bool = False,
        tiles_per_row: int = 1,
        info: str | None = None,
    ) -> None:
        """size in bytes"""
        assert f_addr.bg_id == out_addr.bg_id, (
            f"PU operation must be within the same BG: f_addr.bg_id={f_addr.bg_id}, out_addr.bg_id={out_addr.bg_id}"
        )
        assert f_addr.pu_id == out_addr.pu_id, (
            f"PU operation must be within the same PU: f_addr.pu_id={f_addr.pu_id}, out_addr.pu_id={out_addr.pu_id}"
        )
        self.f_addr = f_addr
        self.out_addr = out_addr
        self.f_size = f_size
        self.out_size = out_size
        self.interleaving = interleaving
        self.tiles_per_row = tiles_per_row
        self.info = info

    def instructions(self, arch: str) -> List[Any]:
        ch_id, ra_id, de_id = bg_id2ch_ra_de(self.f_addr.bg_id, arch)
        ret: List[Any] = []

        # Only simulate one channel
        if ch_id > 0:
            return []

        pu_mask = num2mask(SimConfig.de_pu[0])  # type: ignore
        mask_tuple = (SimConfig.de_pu[0], pu_mask)  # type: ignore
        ld_in_instr = (LEVEL.DE, OPTYPE.buf2reg, ch_id, ra_id, de_id, mask_tuple, 0)  # type: ignore
        ret.append(ld_in_instr)

        op1_addr = addr2bank_row_col(
            self.f_addr.row_id,
            self.f_addr.byte_id,
            self.f_addr.tile_size_bytes,
            self.tiles_per_row,
            self.interleaving,
        )
        op1_bank, _, _ = op1_addr
        op2_addr = (op1_bank, 0, 0)

        col_num = max(size2col(self.f_size), size2col(self.out_size))
        pu_instr = (
            LEVEL.DE,
            OPTYPE.pu,
            ch_id,
            ra_id,
            de_id,
            mask_tuple,
            op1_addr,
            op2_addr,
            col_num,
            True,
        )
        ret.append(pu_instr)

        flush_instr = (LEVEL.DE, OPTYPE.reg2buf, ch_id, ra_id, de_id, mask_tuple, 0)
        ret.append(flush_instr)

        # Store output from buffer to DRAM
        if arch == "hbm_pim":
            dest_addr = addr2bank_row_col(
                self.out_addr.row_id,
                self.out_addr.byte_id,
                self.out_addr.tile_size_bytes,
                self.tiles_per_row,
                self.interleaving,
            )
            st_out_instr = (
                LEVEL.DE,
                OPTYPE.buf2bk,
                ch_id,
                ra_id,
                de_id,
                mask_tuple,
                dest_addr,
                [False, 0, 0],
                True,
            )
            ret.append(st_out_instr)

        return ret

    def __repr__(self) -> str:
        return f"PU(f_addr={self.f_addr}, out_addr={self.out_addr}, f_size={self.f_size}, out_size={self.out_size} info={self.info})"


class Load(Operation):
    def __init__(
        self,
        addr: Addr,
        size: int,
        interleaving: bool = False,
        tiles_per_row: int = 1,
        info: str | None = None,
    ) -> None:
        """size in bytes"""
        self.addr = addr
        self.size = size
        self.interleaving = interleaving
        self.tiles_per_row = tiles_per_row
        self.info = info

    def instructions(self, arch: str) -> List[Any]:
        ch_id, ra_id, de_id = bg_id2ch_ra_de(self.addr.bg_id, arch)

        # Only simulate one channel
        if ch_id > 0:
            return []

        de_mask = id2mask(de_id, SimConfig.de)  # type: ignore
        pu_bank_id, pu_row_id, col_offset = addr2bank_row_col(
            self.addr.row_id,
            self.addr.byte_id,
            self.addr.tile_size_bytes,
            self.tiles_per_row,
            self.interleaving,
        )
        bank_id = self.addr.pu_id * 2 + pu_bank_id
        col_num = size2col(self.size)
        instr = (
            LEVEL.SYS,
            OPTYPE.host_read,
            ch_id,
            ra_id,
            de_mask,
            bank_id,
            pu_row_id,
            col_offset,
            col_num,
            True,
        )

        return [instr]

    def __repr__(self) -> str:
        return f"Load(addr={self.addr}, size={self.size}, info={self.info})"


class Comment(Operation):
    def __init__(self, text: str) -> None:
        self.text = text

    def instructions(self, arch: str) -> List[Any]:
        return []

    def __repr__(self) -> str:
        return f"Comment(text={self.text})"


def ops2instructions(
    ops: List[Operation], arch: str
) -> Tuple[List[Tuple[int, List[int], List[Any]]], int]:
    instructions: List[Any] = []
    if arch == "upmem":
        SimConfig.read_from_yaml(UPMEM_ARCH_PATH)
    elif arch == "hbm_pim":
        SimConfig.read_from_yaml(HBM_PIM_ARCH_PATH)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    for op in ops:
        instructions.extend(op.instructions(arch))
    row_changes = _apply_auto_precharge_policy(instructions)
    return [(0, [], instructions)], row_changes
