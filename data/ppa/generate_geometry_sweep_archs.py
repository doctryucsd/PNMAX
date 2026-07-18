"""PU post-route lookup helpers for the PPA database.

This module holds the PU-macro lookup rules shared by the PPA-DB area model:
nearest characterized ``simd_lanes`` row at the nearest supported ``tCK``,
with linear scaling from the modeled row (see ``data/ppa/PROVENANCE.md``).
``experiments/attacc_area/attacc_area.py`` imports ``_nearest_pu_row`` /
``_nearest_pu_area_mm2`` as the single source of truth for that model.

The historical geometry-sweep YAML generation path that used to live here
(driven by curated ``geometry_sweep_*.csv`` snapshots) was removed: every
architecture family under ``data/archs/lowered/`` that has a generator is
generated from the RAW characterization CSVs by the builders listed in
``data/ppa/PROVENANCE.md`` (entry point: ``data/ppa/generate_ppa.sh``).
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any


__all__ = [
    "PU_RESULTS_PATH",
    "_nearest_pu_row",
    "_nearest_pu_area_mm2",
]


SWEEP_DATA_DIR = Path(__file__).resolve().parent
PU_RESULTS_PATH = SWEEP_DATA_DIR / "pu_postroute_22nm_cleaned.csv"


def _coerce_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return text
    try:
        if "." not in text and "e" not in text.lower():
            return int(text)
        return float(text)
    except ValueError:
        return text


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: _coerce_value(val) for key, val in row.items()} for row in reader]


@lru_cache(maxsize=None)
def _pu_rows() -> list[dict[str, Any]]:
    return _load_csv(PU_RESULTS_PATH)


def _nearest_pu_row(simd_lanes: int, target_tck_ns: float) -> dict[str, Any]:
    available_simd_lanes = sorted({int(row["simd_lanes"]) for row in _pu_rows()})
    modeled_simd_lanes = min(
        available_simd_lanes,
        key=lambda candidate: (abs(candidate - int(simd_lanes)), candidate),
    )
    candidates = [
        row
        for row in _pu_rows()
        if int(row["simd_lanes"]) == modeled_simd_lanes
    ]
    if not candidates:
        raise ValueError(f"No PU energy row for simd_lanes={simd_lanes}.")
    return min(
        candidates,
        key=lambda row: (abs(float(row["tck_ns"]) - target_tck_ns), float(row["tck_ns"])),
    )


def _nearest_pu_area_mm2(simd_lanes: int, target_tck_ns: float) -> float:
    modeled_row = _nearest_pu_row(simd_lanes, target_tck_ns)
    modeled_simd_lanes = int(modeled_row["simd_lanes"])
    modeled_area_mm2 = float(modeled_row["area_um2"]) / 1_000_000.0
    return modeled_area_mm2 * (float(simd_lanes) / modeled_simd_lanes)
