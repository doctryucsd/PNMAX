#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARCH_ORDER: tuple[str, ...] = ("upmem", "hbm_pim", "attacc")
ARCH_LABELS: dict[str, str] = {
    "upmem": "UPMEM",
    "hbm_pim": "HBM-PIM",
    "attacc": "AttAcc",
}


@dataclass(frozen=True)
class PlotSeries:
    arch: str
    title: str
    x_vals: np.ndarray
    y_vals: np.ndarray
    r2: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a combined 3-arch validation scatter: simulator ticks (x) "
            "vs analytical cycles (y)."
        )
    )
    parser.add_argument(
        "validation_dir",
        type=Path,
        help="Directory containing upmem.json, hbm_pim.json, and attacc.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. Defaults to <validation_dir>/validation.pdf.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the image.",
    )
    return parser.parse_args()


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_points(report: dict) -> tuple[np.ndarray, np.ndarray]:
    simulator_ticks: list[float] = []
    analytical_cycles: list[float] = []

    for item in report.get("results", []):
        if not isinstance(item, dict):
            continue

        simulator = item.get("simulator", {})
        analytical = item.get("analytical", {})
        if not isinstance(simulator, dict) or not isinstance(analytical, dict):
            continue

        if simulator.get("status") != "ok" or analytical.get("status") != "ok":
            continue

        ticks = simulator.get("ticks")
        cycles = analytical.get("cycles")
        if ticks is None or cycles is None:
            continue

        try:
            simulator_ticks.append(float(ticks))
            analytical_cycles.append(float(cycles))
        except (TypeError, ValueError):
            continue

    return (
        np.asarray(simulator_ticks, dtype=float),
        np.asarray(analytical_cycles, dtype=float),
    )


def _compute_r2(x_vals: np.ndarray, y_vals: np.ndarray) -> float | None:
    if x_vals.size < 2 or y_vals.size < 2:
        return None
    corr = np.corrcoef(x_vals, y_vals)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr * corr)


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    max_value = float(np.max(values))
    if max_value == 0.0:
        return values
    return values / max_value


def _build_plot_series(validation_dir: Path, arch: str) -> PlotSeries:
    report_path = validation_dir / f"{arch}.json"
    if not report_path.is_file():
        raise ValueError(f"Missing validation JSON report: {report_path}")

    report = _load_report(report_path)
    x_vals, y_vals = _extract_points(report)
    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError(f"No valid data points found in report results: {report_path}")

    summary = report.get("summary", {})
    summary_r2 = summary.get("r2") if isinstance(summary, dict) else None
    if summary_r2 is None:
        r2 = _compute_r2(x_vals, y_vals)
    else:
        r2 = float(summary_r2)

    return PlotSeries(
        arch=arch,
        title=ARCH_LABELS[arch],
        x_vals=_normalize(x_vals),
        y_vals=_normalize(y_vals),
        r2=r2,
    )


def _plot_series(ax, series: PlotSeries) -> None:
    ax.scatter(series.x_vals, series.y_vals, s=18, alpha=0.75)

    if series.x_vals.size >= 2 and np.unique(series.x_vals).size >= 2:
        slope, intercept = np.polyfit(series.x_vals, series.y_vals, 1)
        x_line = np.array([series.x_vals.min(), series.x_vals.max()], dtype=float)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="crimson", linewidth=2.0)

    if series.r2 is None:
        ax.set_title(f"{series.title} (R^2 = N/A)", fontsize=14, pad=2)
    else:
        ax.set_title(f"{series.title} (R^2 = {series.r2:.4f})", fontsize=14, pad=2)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_box_aspect(1)


def main() -> int:
    args = parse_args()
    validation_dir = args.validation_dir.resolve()
    series_list = [_build_plot_series(validation_dir, arch) for arch in ARCH_ORDER]

    x_max = max(float(np.max(series.x_vals)) for series in series_list)
    y_max = max(float(np.max(series.y_vals)) for series in series_list)

    fig, axes = plt.subplots(
        1,
        len(series_list),
        figsize=(9.0, 4.2),
        sharex=True,
        sharey=True,
    )
    for ax, series in zip(np.atleast_1d(axes), series_list, strict=True):
        _plot_series(ax, series)
        ax.set_xlim(0.0, x_max * 1.02)
        ax.set_ylim(0.0, y_max * 1.02)

    fig.supxlabel("Simulator cycles (norm.)", fontsize=16, y=0.1)
    fig.supylabel("Analytical cycles (norm.)", fontsize=16, x=0.01)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.02, top=1.02)


    output_path = (
        args.output.resolve()
        if args.output is not None
        else (validation_dir / "validation.pdf").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved combined validation plot to {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
