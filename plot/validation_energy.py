#!/usr/bin/env python3
"""AttAcc energy validation scatter: PNMAX modeled energy vs AttAcc reference energy.

Presentation mirrors the execution-time validation panels
(``plot/combined_validation.py``): each
axis is normalized to its own maximum, a least-squares fit line is drawn, and the
Pearson R^2 is reported. Per-axis-max normalization removes absolute scale, so the
plot shows the *trend* (does modeled energy track the reference) rather than the
magnitude — exactly as the cycles validation does.

The AttAcc simulator (Ramulator2) emits only traffic counts (mac/mvgb/mvsb/wrgb),
*not* energy. The reference energy is therefore re-derived here from those counts,
mirroring AttAcc's own energy model:

  external/attacc/src/ramulator_wrapper.py  (Ramulator.run/output: traffic derivation)
  external/attacc/src/devices.py            (PIM.get_time_and_energy, score MATMUL)
  external/attacc/src/config.py             (ENERGY_TABLE['PIM'][PIMType.BA])

PNMAX modeled energy is read from each result's ``analytical.energy`` field
(``AnalyticalModel.energy(target="attacc")``) produced by the validation re-run.

Usage:
    uv run python plot/validation_energy.py exp_results/validation/attacc_energy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- AttAcc reference energy constants (external/attacc/src/config.py) ---------
# ENERGY_TABLE['PIM'][PIMType.BA]:
_MEM = (0.11 + 0.44) * 8  # 4.4 pJ/byte  DRAM cell access (ACT/PRE + RD/WR)
_ALU = 0.32  # pJ per (flop/2)
_IO = [0.3, 0.5, 1.23, 1.01]  # si, tsv, giomux, bgmux  pJ/byte
# Aggregation factors (ramulator_wrapper.py / devices.py):
_NUM_HBM = 5  # default n_hbm in Ramulator wrapper
_NUM_ATTACC = 8  # default num_attacc in devices.py
_BA_MULT = 2 * 2 * 4 * 4  # pCH * Rank * BG * BA  (PIMType.BA mem_acc multiplier)
_PREFETCH = 32  # bytes per request (read granularity)


def attacc_reference_energy(simulator: dict) -> float:
    """Re-derive AttAcc reference energy (pJ) from simulator traffic counts.

    Mirrors ramulator_wrapper.Ramulator.run() traffic derivation and
    devices.PIM.get_time_and_energy() for the score MATMUL layer (PIMType.BA).
    """
    mac = float(simulator.get("mac", 0) or 0)
    mvgb = float(simulator.get("mvgb", 0) or 0)
    mvsb = float(simulator.get("mvsb", 0) or 0)
    wrgb = float(simulator.get("wrgb", 0) or 0)

    si_io = wrgb * _PREFETCH
    tsv_io = (wrgb + mvsb + mvgb) * _PREFETCH
    giomux_io = (wrgb + mvsb + mvgb) * _PREFETCH
    bgmux_io = (wrgb + mvsb + mvgb) * _PREFETCH
    mem_acc = mac * _PREFETCH * _BA_MULT

    traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
    traffic = [t * _NUM_HBM for t in traffic]

    io_energy = sum(traffic[i] * _IO[i] for i in range(4))
    cell_energy = traffic[-1] * _MEM
    dram_energy = cell_energy + io_energy

    trace = simulator.get("trace_params", {}) or {}
    nhead = float(trace.get("nhead", 0) or 0)
    seqlen = float(trace.get("seqlen", 0) or 0)
    dhead = float(trace.get("dhead", 0) or 0)
    flops = 2.0 * nhead * seqlen * dhead
    cal_energy = flops / 2.0 * _ALU

    return (dram_energy + cal_energy) * _NUM_ATTACC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot AttAcc energy validation scatter: AttAcc reference energy (x) vs "
            "PNMAX modeled energy (y), each normalized to its own max, with a "
            "least-squares fit line and R^2 (mirrors plot/combined_validation.py)."
        )
    )
    parser.add_argument(
        "validation_json",
        type=Path,
        help="Path to the AttAcc validation JSON (e.g. attacc_energy.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. Defaults to <json_stem>_energy.pdf next to the JSON.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument("--title-fontsize", type=float, default=16, help="Title font size.")
    parser.add_argument("--label-fontsize", type=float, default=14, help="Axis label font size.")
    parser.add_argument("--tick-fontsize", type=float, default=12, help="Tick label font size.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the PDF.",
    )
    return parser.parse_args()


def _extract_points(report: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (reference, modeled) energy arrays (pJ)."""
    reference: list[float] = []
    modeled: list[float] = []
    for item in report.get("results", []):
        if not isinstance(item, dict):
            continue
        simulator = item.get("simulator", {})
        analytical = item.get("analytical", {})
        if not isinstance(simulator, dict) or not isinstance(analytical, dict):
            continue
        if simulator.get("status") != "ok" or analytical.get("status") != "ok":
            continue
        energy = analytical.get("energy")
        if energy is None:
            continue
        try:
            ref = attacc_reference_energy(simulator)
            mod = float(energy)
        except (TypeError, ValueError):
            continue
        if ref <= 0.0 or mod <= 0.0:
            continue
        reference.append(ref)
        modeled.append(mod)
    return np.asarray(reference, dtype=float), np.asarray(modeled, dtype=float)


def _r2(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr * corr)


def main() -> int:
    args = parse_args()
    report = json.loads(args.validation_json.read_text(encoding="utf-8"))
    ref, mod = _extract_points(report)
    if ref.size == 0:
        raise SystemExit(
            "No usable (energy + traffic) points found. Re-run the AttAcc "
            "validation so analytical.energy is present in the JSON."
        )

    # R^2 is scale-invariant, so it is identical on raw or per-axis-normalized data.
    r2 = _r2(ref, mod)

    # Normalize each axis to its own maximum (mirrors plot/combined_validation.py).
    x_max = float(np.max(ref))
    y_max = float(np.max(mod))
    x_plot = ref / x_max if x_max != 0 else ref
    y_plot = mod / y_max if y_max != 0 else mod

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.scatter(x_plot, y_plot, s=18, alpha=0.75)

    if x_plot.size >= 2 and np.unique(x_plot).size >= 2:
        slope, intercept = np.polyfit(x_plot, y_plot, 1)
        x_line = np.array([x_plot.min(), x_plot.max()], dtype=float)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="crimson", linewidth=2.0)

    r2_txt = "N/A" if r2 is None else f"{r2:.4f}"
    ax.set_title(f"AttAcc energy ($R^2$ = {r2_txt})", fontsize=args.title_fontsize)
    ax.set_xlabel("AttAcc reference energy (norm.)", fontsize=args.label_fontsize)
    ax.set_ylabel("PNMAX modeled energy (norm.)", fontsize=args.label_fontsize)
    ax.tick_params(axis="both", labelsize=args.tick_fontsize)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()

    output_path = (
        args.output.resolve()
        if args.output is not None
        else args.validation_json.with_name(
            f"{args.validation_json.stem}_energy.pdf"
        ).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    print(f"N={ref.size}  R^2={r2_txt}")
    print(f"Saved AttAcc energy validation scatter to {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
