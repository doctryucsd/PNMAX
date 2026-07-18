#!/usr/bin/env python3
"""Combined latency + energy validation figure.

Stitches the three latency validation panels (UPMEM, HBM-PIM, AttAcc — simulator
cycles vs analytical cycles) together with the AttAcc energy validation panel
(AttAcc energy vs analytical energy; x is the AttAcc reference energy re-derived
from traffic counts, y is the modeled energy) into one image, in two layout
variants:

  * 2x2 grid ("attacc below"): top row = UPMEM latency, HBM-PIM latency;
    bottom row = AttAcc latency, AttAcc energy.
  * 4x1 row: UPMEM latency, HBM-PIM latency, AttAcc latency, AttAcc energy.

Both layouts plot the SAME four panels; only the grid shape differs.

Latency panels mirror ``plot/combined_validation.py`` exactly (per-axis-max
normalization, crimson least-squares fit, R^2 from each JSON's ``summary.r2``).
The energy panel mirrors ``plot/validation_energy.py`` exactly — the AttAcc
reference energy is NOT a stored field; it is re-derived from simulator traffic
counts by ``attacc_reference_energy`` (imported from ``plot.validation_energy``).

Because the energy panel's axes mean energy (not cycles), it gets its own axis
labels; a single shared ``supxlabel``/``supylabel`` would be wrong.

Usage:
    uv run python plot/combined_validation_lat_energy.py exp_results/validation
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse the latency helpers from plot/combined_validation.py and the AttAcc
# reference-energy derivation from plot/validation_energy.py. Both are sibling
# modules in plot/; load them by path so this works whether or not plot/ is a
# package and regardless of the cwd.
_PLOT_DIR = Path(__file__).resolve().parent


def _load_sibling(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _PLOT_DIR / filename)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load sibling module {filename}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass (which looks the class's module up in
    # sys.modules) resolves correctly during import of the sibling module.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_combined = _load_sibling("_pnmax_combined_validation", "combined_validation.py")
_energy = _load_sibling("_pnmax_validation_energy", "validation_energy.py")

# Latency helpers (verbatim reuse; source: plot/combined_validation.py).
_extract_points = _combined._extract_points
_normalize = _combined._normalize
_compute_r2 = _combined._compute_r2
# Energy reference-energy derivation (verbatim reuse; source: plot/validation_energy.py).
attacc_reference_energy = _energy.attacc_reference_energy
_extract_energy_points = _energy._extract_points

LATENCY_ARCHS: tuple[str, ...] = ("upmem", "hbm_pim", "attacc")
ARCH_LABELS: dict[str, str] = {
    "upmem": "UPMEM",
    "hbm_pim": "HBM-PIM",
    "attacc": "AttAcc",
}

# Marker styling shared with the source scripts.
_MARKER_SIZE = 18
_MARKER_ALPHA = 0.75
_FIT_COLOR = "crimson"
_FIT_WIDTH = 2.0


@dataclass(frozen=True)
class Panel:
    title: str
    x_vals: np.ndarray  # normalized to own max
    y_vals: np.ndarray  # normalized to own max
    r2: float | None
    xlabel: str
    ylabel: str


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_r2(report: dict, x: np.ndarray, y: np.ndarray) -> float | None:
    summary = report.get("summary", {})
    summary_r2 = summary.get("r2") if isinstance(summary, dict) else None
    if summary_r2 is None:
        return _compute_r2(x, y)
    return float(summary_r2)


def _build_latency_panel(validation_dir: Path, arch: str) -> Panel:
    report_path = validation_dir / f"{arch}.json"
    if not report_path.is_file():
        raise ValueError(f"Missing validation JSON report: {report_path}")
    report = _load_report(report_path)
    x_vals, y_vals = _extract_points(report)
    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError(f"No valid data points found in report results: {report_path}")
    r2 = _summary_r2(report, x_vals, y_vals)
    print(f"[latency] {arch}: N={x_vals.size}  R^2={r2}")
    return Panel(
        title=f"{ARCH_LABELS[arch]} latency",
        x_vals=_normalize(x_vals),
        y_vals=_normalize(y_vals),
        r2=r2,
        xlabel="Simulator cycles (norm.)",
        ylabel="Analytical cycles (norm.)",
    )


def _build_energy_panel(validation_dir: Path) -> Panel:
    report_path = validation_dir / "attacc_energy.json"
    if not report_path.is_file():
        raise ValueError(f"Missing validation JSON report: {report_path}")
    report = _load_report(report_path)
    ref, mod = _extract_energy_points(report)
    if ref.size == 0 or mod.size == 0:
        raise ValueError(f"No usable (energy + traffic) points found in: {report_path}")
    # NOTE: do NOT use summary.r2 here — that field is the *cycles* (latency) R^2,
    # not the energy R^2. Compute the energy R^2 from the (reference, modeled)
    # energy points, matching plot/validation_energy.py.
    r2 = _compute_r2(ref, mod)
    print(f"[energy] attacc_energy: N={ref.size}  R^2={r2}")
    return Panel(
        title="AttAcc energy",
        x_vals=_normalize(ref),
        y_vals=_normalize(mod),
        r2=r2,
        xlabel="AttAcc energy (norm.)",
        ylabel="Analytical energy (norm.)",
    )


def _build_panels(validation_dir: Path) -> list[Panel]:
    panels = [_build_latency_panel(validation_dir, arch) for arch in LATENCY_ARCHS]
    panels.append(_build_energy_panel(validation_dir))
    return panels


def _draw_panel(
    ax,
    panel: Panel,
    label_fontsize: float,
    tick_fontsize: float,
    title_fontsize: float,
    box_aspect: float,
    twox2_labels: bool = False,
) -> None:
    ax.scatter(panel.x_vals, panel.y_vals, s=_MARKER_SIZE, alpha=_MARKER_ALPHA)

    if panel.x_vals.size >= 2 and np.unique(panel.x_vals).size >= 2:
        slope, intercept = np.polyfit(panel.x_vals, panel.y_vals, 1)
        x_line = np.array([panel.x_vals.min(), panel.x_vals.max()], dtype=float)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=_FIT_COLOR, linewidth=_FIT_WIDTH)

    r2_txt = "N/A" if panel.r2 is None else f"{panel.r2:.4f}"
    ax.set_title(f"{panel.title} ($R^2$ = {r2_txt})", fontsize=title_fontsize, pad=4)
    # 2x2-only label tweaks: abbreviate "Analytical" and label the energy panel's x
    # axis "Simulator energy" (consistent with the latency panels' "Simulator cycles").
    xlabel = panel.xlabel
    ylabel = panel.ylabel
    if twox2_labels:
        xlabel = xlabel.replace("AttAcc energy", "Simulator energy")
        ylabel = ylabel.replace("Analytical", "Ana.")
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    # box_aspect = height / width; < 1 makes the panel wider (bigger x:y).
    ax.set_box_aspect(box_aspect)
    ax.set_xlim(0.0, max(float(np.max(panel.x_vals)) * 1.02, 1e-9))
    ax.set_ylim(0.0, max(float(np.max(panel.y_vals)) * 1.02, 1e-9))


def _render(
    panels: list[Panel],
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
    output_path: Path,
    dpi: int,
    label_fontsize: float,
    tick_fontsize: float,
    title_fontsize: float,
    box_aspect: float,
    show: bool,
    twox2_labels: bool = False,
) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, panel in zip(axes_flat, panels, strict=True):
        _draw_panel(
            ax, panel, label_fontsize, tick_fontsize, title_fontsize, box_aspect,
            twox2_labels=twox2_labels,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved combined lat+energy validation plot to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the 3 latency validation panels (UPMEM, HBM-PIM, AttAcc) with "
            "the AttAcc energy validation panel into one image, in 2x2 and 4x1 "
            "layout variants."
        )
    )
    parser.add_argument(
        "validation_dir",
        type=Path,
        help="Directory with upmem.json, hbm_pim.json, attacc.json, attacc_energy.json.",
    )
    parser.add_argument(
        "--output-2x2",
        type=Path,
        default=None,
        help=(
            "Output PDF for the 2x2 'attacc below' layout. "
            "Defaults to <validation_dir>/validation_lat_energy_2x2.pdf."
        ),
    )
    parser.add_argument(
        "--output-4x1",
        type=Path,
        default=None,
        help=(
            "Output PDF for the 4x1 row layout. "
            "Defaults to <validation_dir>/validation_lat_energy_4x1.pdf."
        ),
    )
    parser.add_argument(
        "--skip-4x1",
        action="store_true",
        help="Render only the 2x2 layout (Fig. 8); skip the 4x1 variant.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument("--label-fontsize", type=float, default=24, help="Axis label font size (4x1).")
    parser.add_argument("--tick-fontsize", type=float, default=19, help="Tick label font size (4x1).")
    parser.add_argument("--title-fontsize", type=float, default=26, help="Panel title font size (4x1).")
    parser.add_argument(
        "--label-fontsize-2x2", type=float, default=22, help="Axis label font size (2x2)."
    )
    parser.add_argument(
        "--tick-fontsize-2x2", type=float, default=18, help="Tick label font size (2x2)."
    )
    parser.add_argument(
        "--title-fontsize-2x2", type=float, default=24, help="Panel title font size (2x2)."
    )
    parser.add_argument(
        "--box-aspect-2x2",
        type=float,
        default=0.42,
        help="Per-panel height/width for the 2x2 layout (<1 = wider panels / bigger x:y).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot windows in addition to saving the PDFs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_dir = args.validation_dir.resolve()

    panels = _build_panels(validation_dir)

    out_2x2 = (
        args.output_2x2.resolve()
        if args.output_2x2 is not None
        else (validation_dir / "validation_lat_energy_2x2.pdf").resolve()
    )
    out_4x1 = (
        args.output_4x1.resolve()
        if args.output_4x1 is not None
        else (validation_dir / "validation_lat_energy_4x1.pdf").resolve()
    )

    # 2x2 "attacc below": top = UPMEM lat, HBM-PIM lat; bottom = AttAcc lat, AttAcc energy.
    # panels order is [upmem lat, hbm_pim lat, attacc lat, attacc energy], which fills
    # a 2x2 grid row-major into exactly that arrangement.
    _render(
        panels,
        nrows=2,
        ncols=2,
        figsize=(13.0, 8.0),
        output_path=out_2x2,
        dpi=args.dpi,
        label_fontsize=args.label_fontsize_2x2,
        tick_fontsize=args.tick_fontsize_2x2,
        title_fontsize=args.title_fontsize_2x2,
        box_aspect=args.box_aspect_2x2,
        show=args.show,
        twox2_labels=True,
    )

    # 4x1 row: same four panels in a single row.
    if not args.skip_4x1:
        _render(
            panels,
            nrows=1,
            ncols=4,
            figsize=(16.0, 4.2),
            output_path=out_4x1,
            dpi=args.dpi,
            label_fontsize=args.label_fontsize,
            tick_fontsize=args.tick_fontsize,
            title_fontsize=args.title_fontsize,
            box_aspect=1.0,
            show=args.show,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
