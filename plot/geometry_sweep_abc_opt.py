#!/usr/bin/env python3
"""Combined (a)/(b)/(c) geometry figure on the ORIGINAL Fig-11 workload: OPT-2.7B
end-to-end (not the 9-layer geomean), on the corrected 16x basis.

We REUSE the existing end-to-end random-search mappings (no new search) and only
RE-COST them against the corrected 16x arch files, then sum per layer to an
end-to-end best-latency metric per geometry -- exactly the original Fig-11 method
(best-latency mapping per variant, summed over the model's layers). Only the
single-step rungs the original had are available (the /4, x4, burst-1, burst-16
extremes were never searched for OPT), so each panel shows 3 rungs.

Output: results/fig14_geometry/geometry_sweep_abc_opt_{line,bar}.pdf
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_sweep_abc_combined as abc  # noqa: E402

# Bump every shared font a little, scoped to this (abc_opt) figure only. The
# abc_combined source is untouched, so running it standalone is unaffected.
_FS_BUMP = 1.5
for _fs in ("FS_TITLE", "FS_LABEL", "FS_TICK", "FS_ANNOT", "FS_LEGEND"):
    setattr(abc, _fs, getattr(abc, _fs) * _FS_BUMP)

# OPT figure uses fixed canonical slots so the baseline (1x) is always centered,
# and a missing variant simply leaves a gap (e.g. HBM-PIM has no /4, x4, or (b) x2).
OPT_BC_KEYS = (0.25, 0.5, 1.0, 2.0, 4.0)
OPT_BC_LABELS = ("/4", "/2", "1x", "x2", "x4")
OPT_LANE_KEYS = (4, 8, 16, 32, 64)
OPT_LANE_LABELS = ("/4", "/2", "1x", "x2", "x4")  # ratios vs baseline (16 = 1x)

from pnmax.database import Arch, Workload  # noqa: E402
from pnmax.dse.workload_eval import evaluate_workload_analytical_report  # noqa: E402

# Cache of the re-cost results (normalized per-rung metrics) so cosmetic changes
# (labels, no point numbers, styling) can re-PLOT without re-costing (--replot).
# CACHE_PATH is defined just after OUT_STEM below (it depends on it).


def save_cache(bc_all, burst_by_arch):
    obj = {"bc_all": bc_all,
           "burst_by_arch": {fam: [bn, cnt] for fam, (bn, cnt) in burst_by_arch.items()}}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(obj))
    print(f"cached re-cost results -> {CACHE_PATH}")


def load_cache():
    obj = json.loads(CACHE_PATH.read_text())
    bc_all = {}
    for fam, d in obj["bc_all"].items():
        norm = {p: {float(k): v for k, v in d["normalized"].get(p, {}).items()}
                for p in ("b", "c")}
        bc_all[fam] = {"normalized": norm, "workload_count": d["workload_count"]}
    burst_by_arch = {fam: ({int(k): v for k, v in bn.items()}, cnt)
                     for fam, (bn, cnt) in obj["burst_by_arch"].items()}
    return bc_all, burst_by_arch

REPO_ROOT = Path(os.environ.get("PNMAX_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_ROOT = Path(os.environ.get("PNMAX_RESULTS_ROOT", str(REPO_ROOT / "results")))
DEFAULT_RESULTS_DIR = RESULTS_ROOT / "fig14_geometry"

# OPT-2.7B layer multiplicities are read from the shipped end-to-end workload
# manifest (the authority on the model's layer decomposition), in its
# unique_workloads order -- the order the end-to-end sum has always used.
OPT_MANIFEST_PATH = (
    REPO_ROOT / "data" / "workloads" / "end_to_end" / "opt-2.7B-2048" / "manifest.json"
)


def load_opt_layer_mult(manifest_path=OPT_MANIFEST_PATH):
    """token -> multiplicity from the manifest's unique_workloads rows,
    cross-checked against its aggregate multiplicities map."""
    manifest = json.loads(Path(manifest_path).read_text())
    mult = {str(w["token"]): int(w["multiplicity"])
            for w in manifest["unique_workloads"]}
    declared = {str(k): int(v) for k, v in manifest["multiplicities"].items()}
    if not mult or declared != mult:
        raise ValueError(
            f"unique_workloads multiplicities {mult} disagree with the "
            f"multiplicities map {declared} in {manifest_path}")
    return mult


OPT_LAYER_MULT = load_opt_layer_mult()

# End-to-end search roots (mappings to reuse) per arch family: the end-to-end
# pipeline run holds the original single-step variants; the extremes run holds
# the /4,x4,burst extremes. best_latency_for_layer searches them in order.
# Filled in main() via _resolve_rs_roots() from --e2e-root/--extremes-root.
RS_ROOTS: dict[str, list[str]] = {}
DEFAULT_E2E_ROOT = DEFAULT_RESULTS_DIR / "geometry_sweep_end_to_end_workload_space"
DEFAULT_EXTREMES_ROOT = (
    DEFAULT_RESULTS_DIR
    / "geometry_sweep_abc_opt_extremes"
    / "random_search"
    / "opt-2.7B-2048"
)


def _resolve_rs_roots(e2e_root: Path, extremes_root: Path) -> dict[str, list[str]]:
    """Per arch family: latest e2e run's OPT random-search dir + the extremes dir."""
    roots: dict[str, list[str]] = {}
    for fam in ("upmem", "hbm_pim"):
        fam_dir = Path(e2e_root) / fam
        run_dirs = (
            sorted(d for d in fam_dir.glob("[0-9]*_[0-9]*") if d.is_dir())
            if fam_dir.is_dir()
            else []
        )
        fam_roots = [
            str(run_dir / "random_search" / "opt-2.7B-2048") for run_dir in run_dirs[-1:]
        ]
        fam_roots.append(str(extremes_root))
        roots[fam] = fam_roots
    return roots


# 16x corrected basis (cacti-fixed area + b2b-transfer latency x16).
# Latency/energy are re-cost against these arch files.
ARCH_DIR = REPO_ROOT / "data" / "archs" / "lowered" / "geometry_sweep_host_congested" / "iso_capacity_archs"
# AREA ONLY is read from the (re-derived) plain geometry_sweep tree, decoupled from
# the latency/energy basis above. UPMEM area is identical between the two trees;
# this differs only for HBM-PIM. Latency/energy remain on the congested-host
# 16x basis.
AREA_DIR = REPO_ROOT / "data" / "archs" / "lowered" / "geometry_sweep" / "iso_capacity_archs"
OUT_STEM = "geometry_sweep_abc_opt"
CACHE_PATH = DEFAULT_RESULTS_DIR / f"_recost_cache_{OUT_STEM}.json"

# rung -> geometry stem.  ratio key (b/c) or SIMD-lane key (a) -> stem.
LADDER = {
    "upmem": {
        "baseline": "upmem__pu-8__hmat-16__vmat-64",
        "b": {0.25: "upmem__pu-2__hmat-64__vmat-64", 0.5: "upmem__pu-4__hmat-32__vmat-64",
              2.0: "upmem__pu-16__hmat-8__vmat-64", 4.0: "upmem__pu-32__hmat-4__vmat-64"},
        "c": {0.25: "upmem__pu-2__hmat-16__vmat-256", 0.5: "upmem__pu-4__hmat-16__vmat-128",
              2.0: "upmem__pu-16__hmat-16__vmat-32", 4.0: "upmem__pu-32__hmat-16__vmat-16"},
        "a": {4: "upmem__pu-8__hmat-16__vmat-64_burst-1", 8: "upmem__pu-8__hmat-16__vmat-64_burst-2",
              32: "upmem__pu-8__hmat-16__vmat-64_burst-8", 64: "upmem__pu-8__hmat-16__vmat-64_burst-16"},
    },
    "hbm_pim": {
        "baseline": "hbm_pim__pu-8__hmat-16__vmat-32",
        "b": {0.5: "hbm_pim__pu-4__hmat-32__vmat-32", 2.0: "hbm_pim__pu-16__hmat-8__vmat-32"},
        "c": {0.5: "hbm_pim__pu-4__hmat-16__vmat-64", 2.0: "hbm_pim__pu-16__hmat-16__vmat-16"},
        "a": {4: "hbm_pim__pu-8__hmat-16__vmat-32_burst-1", 8: "hbm_pim__pu-8__hmat-16__vmat-32_burst-2",
              32: "hbm_pim__pu-8__hmat-16__vmat-32_burst-8", 64: "hbm_pim__pu-8__hmat-16__vmat-32_burst-16"},
    },
}


def arch_file(arch_family, stem):
    return Path(ARCH_DIR) / arch_family / f"{stem}.yaml"


def best_latency_for_layer(rs_roots, stem, token, arch, target, samples):
    """Re-cost every searched mapping for (geometry, layer) and return the
    best-latency mapping's (latency, energy). Mappings are looked up across the
    given roots (original single-step run + the new extremes run)."""
    traces = []
    for root in rs_roots:
        traces = sorted(glob.glob(f"{root}/{stem}/{token}/spaces/c/trace_*.yaml"))
        if traces:
            break
    if samples:
        traces = traces[:samples]
    best = None
    for tp in traces:
        w = Workload.from_yaml(Path(tp).read_text())
        rep = evaluate_workload_analytical_report(
            w, arch, target=target, latency_coeff=1.0, mem_footprint_coeff=1.0,
            energy_coeff=1.0, require_constraints=False)
        if rep is None:
            continue
        lat = rep["latency_cycles"]
        if best is None or lat < best[0]:
            best = (lat, rep["energy"])
    return best


def end_to_end(rs_roots, stem, arch, target, samples):
    """Sum best-latency layer metrics over OPT-2.7B's layers (with multiplicity)."""
    tot_lat = tot_en = 0
    for token, mult in OPT_LAYER_MULT.items():
        bl = best_latency_for_layer(rs_roots, stem, token, arch, target, samples)
        if bl is None:
            # Geometry not searched for this layer (e.g. an HBM-PIM variant the
            # original run skipped due to the DreamRAM modeling issue) -> no point.
            return None
        tot_lat += mult * bl[0]
        tot_en += mult * bl[1]
    return tot_lat, tot_en


def arch_area(arch_family, stem):
    p = Path(AREA_DIR) / arch_family / f"{stem}.yaml"
    return float(yaml.safe_load(p.read_text())["total_area_mm2"])


def build_arch_data(arch_family, samples):
    rs_roots = RS_ROOTS[arch_family]
    target = arch_family
    L = LADDER[arch_family]
    base_stem = L["baseline"]
    base_arch = Arch.from_yaml_file(str(arch_file(arch_family, base_stem)))
    base_lat, base_en = end_to_end(rs_roots, base_stem, base_arch, target, samples)
    base_area = arch_area(arch_family, base_stem)
    print(f"[{arch_family}] baseline end-to-end lat={base_lat} en={base_en} area={base_area:.3f}")

    def norm(stem):
        a = Arch.from_yaml_file(str(arch_file(arch_family, stem)))
        res = end_to_end(rs_roots, stem, a, target, samples)
        if res is None:
            return None
        lat, en = res
        return {"latency": lat / base_lat, "energy": en / base_en,
                "area": arch_area(arch_family, stem) / base_area}

    bc_norm = {"b": {1.0: {"latency": 1.0, "energy": 1.0, "area": 1.0}},
               "c": {1.0: {"latency": 1.0, "energy": 1.0, "area": 1.0}}}
    for panel in ("b", "c"):
        for ratio, stem in L[panel].items():
            v = norm(stem)
            if v is None:
                print(f"  {arch_family} ({panel}) {ratio}x -> {stem}: SKIPPED (no mappings)")
                continue
            bc_norm[panel][ratio] = v
            print(f"  {arch_family} ({panel}) {ratio}x -> {stem}: {v}")

    burst_norm = {16: {"latency": 1.0, "energy": 1.0, "area": 1.0}}
    for lanes, stem in L["a"].items():
        v = norm(stem)
        if v is None:
            print(f"  {arch_family} (a) lanes={lanes} -> {stem}: SKIPPED (no mappings)")
            continue
        burst_norm[lanes] = v
        print(f"  {arch_family} (a) lanes={lanes} -> {stem}: {v}")

    return {"normalized": bc_norm, "workload_count": 1}, (burst_norm, 1)


def build_opt_figure(bc_all, burst_by_arch, chart, output_dir, dpi):
    # Subplot x:y aspect = (width/2):(height/3); widen here for a bigger x:y.
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11.0))
    row_specs = [
        ("a", OPT_LANE_KEYS, OPT_LANE_LABELS, "(a) PU tile size",
         lambda arch: burst_by_arch[arch][0]),
        ("b", OPT_BC_KEYS, OPT_BC_LABELS, "(b) BG; R. size inverse",
         lambda arch: bc_all.get(arch, {}).get("normalized", {}).get("b", {})),
        ("c", OPT_BC_KEYS, OPT_BC_LABELS, "(c) BG; #R inverse",
         lambda arch: bc_all.get(arch, {}).get("normalized", {}).get("c", {})),
    ]
    any_data = False
    for r, (panel, keys, labels, xlabel, getter) in enumerate(row_specs):
        for c, arch in enumerate(abc.ARCHES):
            ax = axes[r][c]
            source = getter(arch)
            if source:
                any_data = True
            abc.plot_cell(ax, keys, labels, source, chart, xlabel, spread=True)
            if r == 0:
                ax.set_title(abc.ARCH_TITLES[arch], fontsize=abc.FS_TITLE * 1.3,
                             fontweight="bold")
            if c == 0:
                ax.set_ylabel("norm. to 1x", fontsize=abc.FS_LABEL)
    if not any_data:
        plt.close(fig)
        return None
    handles = [
        Line2D([0], [0], color=abc.METRIC_COLORS[m], marker=abc.METRIC_MARKERS[m],
               lw=2.2, label=abc.METRICS[m][1])
        for m in abc.METRIC_PLOT_ORDER
    ]
    fig.tight_layout()
    # Add the legend AFTER tight_layout so placing it high does not make
    # tight_layout reserve extra room and inflate the inter-row gaps. It overlays
    # the empty top-right region of the bottom-right panel instead.
    axes[2][1].legend(handles=handles, fontsize=abc.FS_LEGEND, loc="upper right",
                      bbox_to_anchor=(1.0, 1.15), borderpad=0.3, labelspacing=0.3,
                      handlelength=1.3, handletextpad=0.4)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{OUT_STEM}_{chart}.pdf"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=0,
                    help="cap traces re-costed per layer (0 = all 2048)")
    ap.add_argument("--e2e-root", type=Path, default=DEFAULT_E2E_ROOT,
                    help="geometry-sweep end-to-end pipeline output root")
    ap.add_argument("--extremes-root", type=Path, default=DEFAULT_EXTREMES_ROOT,
                    help="abc-opt extremes random-search root")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--chart", choices=("bar", "line", "both"), default="both")
    ap.add_argument("--replot", action="store_true",
                    help="skip the re-cost; load cached results and just re-plot (instant)")
    args = ap.parse_args()

    RS_ROOTS.update(_resolve_rs_roots(args.e2e_root, args.extremes_root))

    if args.replot and CACHE_PATH.is_file():
        print(f"--replot: loading cached results from {CACHE_PATH} (no re-cost)")
        bc_all, burst_by_arch = load_cache()
    else:
        bc_all, burst_by_arch = {}, {}
        for fam in abc.ARCHES:
            bc, burst = build_arch_data(fam, args.samples)
            bc_all[fam] = bc
            burst_by_arch[fam] = burst
        save_cache(bc_all, burst_by_arch)

    charts = ("bar", "line") if args.chart == "both" else (args.chart,)
    for chart in charts:
        out = build_opt_figure(bc_all, burst_by_arch, chart, args.output_dir, args.dpi)
        print(out.resolve() if out else f"{chart}: no data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
