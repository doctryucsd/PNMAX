#!/usr/bin/env python3
"""Approach A — run OUR Fig-9 DNN layers through CINM (Cinnamon), extract the
compiler tiling, re-cost it on the PNMAX analytical model (UPMEM baseline-normalized),
and write the CINM proxy CSV that feeds the "CINM" marker on the UPMEM
subfigures of the Fig. 9 Pareto grid.

Each layer's shape comes from the derived UniNDP-baseline workload
``<wl>_upmem.yaml`` (never invented); each layer is expressed as a
``cinm.compute`` GEMM with ``workgroupShape=array<i64: D, T, 1>`` (derived from
the loaded arch, see below); ``cinm-opt --cinm-tiling`` produces the compiler tile
decomposition; we parse the tile sizes and re-cost the resulting mapping on our
model, normalized to the SAME ``baseline.json`` the Fig-9 grid uses (mirroring
extract_optipim_proxy.py).

Why D=20, T=128: ``data/archs/lowered/baseline/upmem.yaml`` hierarchy = pu:8 * chip:8 *
rank:2 * channel:20 = 2560 DPUs; T = DPUs/DIMM = pu*chip*rank = 8*8*2 = 128, D = #DIMMs =
channel = 20, so the workgroup = full arch DPU count (|WG| = 20*128 = 2560). Both come
from ``workgroup_geometry(arch)`` (the arch hierarchy), never from code constants.

The CINM->PNMAX mapping correspondence (spatial over output M,K; reduction C temporal;
K factor <= simd on L0) is reused from ``cinm_solution_on_our_model.py`` (Approach B).

Artifact adaptation (PNMAX AE):
- imports use the renamed ``pnmax`` package;
- ``CINM_OPT`` resolves repo-relatively to the in-repo cinm build
  (``external/cinm/build/bin/cinm-opt``, built by ``setup.sh`` by default),
  overridable via the ``CINM_OPT`` env var;
- input/output roots resolve repo-relatively and are env-overridable.
Wired into ``run.sh`` and exercised by the full Fig-9 campaign — see README.md.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import subprocess
from pathlib import Path

from pnmax.database import Arch
from pnmax.dse.workload_eval import evaluate_workload_analytical_report

# Reuse the Approach-B correspondence helpers verbatim.
from cinm_solution_on_our_model import _load_arch, build, workgroup_geometry  # noqa: E402

# experiments/fig09_pareto/baselines/ -> repo root is 3 levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]

# In-repo cinm-opt (built by setup.sh by default); override with CINM_OPT=...
CINM_OPT = os.environ.get(
    "CINM_OPT", str(REPO_ROOT / "external" / "cinm" / "build" / "bin" / "cinm-opt")
)
ARCH_PATH = os.environ.get(
    "PNMAX_UPMEM_ARCH", str(REPO_ROOT / "data" / "archs" / "lowered" / "baseline" / "upmem.yaml")
)
# Derived UniNDP-baseline workloads supplying each kernel's shape. Override
# with PNMAX_UNINDP_BASELINE_DIR to point at the derived-baseline output dir.
UNINDP_BASELINE_DIR = Path(
    os.environ.get(
        "PNMAX_UNINDP_BASELINE_DIR", str(REPO_ROOT / "results" / "unindp_baseline")
    )
)
# Output + DSE-results roots (the Fig-9 DSE runs produce these under the results root).
RESULTS_ROOT = Path(os.environ.get("PNMAX_RESULTS_ROOT", str(REPO_ROOT / "results")))
OUT_DIR = Path(
    os.environ.get("PNMAX_CINM_OUT", str(RESULTS_ROOT / "fig09_pareto" / "cinm"))
)
MLIR_IN = OUT_DIR / "mlir_in"
MLIR_OUT = OUT_DIR / "mlir_out"
PROXY_CSV = OUT_DIR / "cinm_proxy.csv"
PARETO_EVAL_ROOT = RESULTS_ROOT / "workload_space_pareto_eval" / "upmem"

# Fig-9 UPMEM workloads + their pareto_eval directory prefix (op type).
# token -> (pareto_eval_subdir, source workload yaml basename)
WORKLOADS = {
    "alexnet-4": ("conv2d_alexnet-4", "alexnet-4_upmem.yaml"),
    "bert-3": ("bmm_bert-3", "bert-3_upmem.yaml"),
    "bert-72": ("fc_bert-72", "bert-72_upmem.yaml"),
    "llama128-6": ("fc_llama128-6", "llama128-6_upmem.yaml"),
    "llama256-3": ("bmm_llama256-3", "llama256-3_upmem.yaml"),
    "llama512-3": ("bmm_llama512-3", "llama512-3_upmem.yaml"),
    "resnet50-2": ("conv2d_resnet50-2", "resnet50-2_upmem.yaml"),
    "resnet50-53": ("fc_resnet50-53", "resnet50-53_upmem.yaml"),
    "vgg16-0": ("conv2d_vgg16-0", "vgg16-0_upmem.yaml"),
}


def shape_from_workload(yaml_path: str) -> tuple[int, int, int]:
    """Return (M, C, K) GEMM shape from a PNMAX source workload.

    Compute is Out[n,p,q,k] += In[n,p+r,q+s,c]*F[k,r,s,c]; with R=S=Q=1 each layer is
    GEMM A(M x C) . B(C x K) -> (M x K) with M = N*P*Q (output rows), reduction C.
    """
    import yaml as _yaml

    raw = _yaml.safe_load(open(yaml_path))["Problem"]
    N, K, P, Q, C = raw["N"], raw["K"], raw["P"], raw["Q"], raw["C"]
    M = N * P * Q
    return M, C, K


def emit_mlir(token: str, M: int, C: int, K: int, d: int, t: int) -> Path:
    """Write a cinm.compute GEMM MLIR for this shape; return its path."""
    fname = re.sub(r"[^a-zA-Z0-9_]", "_", token)
    body = f"""module {{
    func.func @{fname}(%A: tensor<{M}x{C}xi32>, %B: tensor<{C}x{K}xi32>) -> tensor<{M}x{K}xi32> {{
        %r0 = cinm.compute attributes {{ workgroupShape=array<i64: {d}, {t}, 1> }} -> tensor<{M}x{K}xi32> {{
            %r = cinm.op.gemm %A, %B: (tensor<{M}x{C}xi32>, tensor<{C}x{K}xi32>) -> tensor<{M}x{K}xi32>
            cinm.yield %r : tensor<{M}x{K}xi32>
        }}
        func.return %r0 : tensor<{M}x{K}xi32>
    }}
}}
"""
    path = MLIR_IN / f"{token}.mlir"
    path.write_text(body)
    return path


def run_tiling(in_path: Path, token: str) -> str:
    """Run cinm-opt --cinm-tiling; save and return the output MLIR text."""
    res = subprocess.run(
        [CINM_OPT, "--cinm-tiling", str(in_path)],
        capture_output=True,
        text=True,
    )
    out = res.stdout
    (MLIR_OUT / f"{token}.tiled.mlir").write_text(out)
    if res.returncode != 0:
        (MLIR_OUT / f"{token}.tiling.stderr.txt").write_text(res.stderr)
    return out


def parse_tiling(out_mlir: str, M: int, C: int, K: int) -> dict[str, int]:
    """Extract CINM's tile decomposition from --cinm-tiling output.

    CINM's GEMM tiling (lib/Dialect/Cinm/IR/CinmTilingImplementations.cpp): the parallel
    output (M,K) is iterated with steps (p0, p1) = parallelClusterSize, and the per-launch
    tile <p0 x p1> (reduced over the C dim in tiles of r) is scattered onto the workgroup.
    So spatial DPU parallelism = p0 * p1; temporal = ceil(M/p0)*ceil(K/p1) over output +
    the C reduction.  The affine.for steps in emitted order are: M (step p0), K (step p1),
    C-reduce (step r).  If the gemm is `cinm.notile` (whole output <= workgroup), the
    output fits in one launch: p0=M, p1=K, r=C.

    Returns p0 (M_tile), p1 (K_tile), r (C_reduce_tile).
    """
    # affine.for ... = 0 to BOUND step STEP   (step omitted => step 1).
    steps = re.findall(
        r"affine\.for\s+\S+\s*=\s*0\s+to\s+(\d+)(?:\s+step\s+(\d+))?",
        out_mlir,
    )
    notile = ("cinm.notile" in out_mlir) and not steps
    if notile:
        return {
            "p0": M, "p1": K, "r": C, "notile": True,
            "loop_steps": "notile",
        }
    if len(steps) < 3:
        raise ValueError(
            f"Expected >=3 affine.for loops (M,K,C); got {len(steps)}: {steps}"
        )
    # Emitted order: M-loop, K-loop, C-reduce-loop.
    (mb, ms), (kb, ks), (cb, cs) = steps[0], steps[1], steps[2]
    p0 = int(ms) if ms else 1
    p1 = int(ks) if ks else 1
    r = int(cs) if cs else int(cb)
    return {
        "p0": p0, "p1": p1, "r": r, "notile": False,
        "loop_steps": ";".join(f"{b}/{s or '1'}" for b, s in steps),
    }


def baseline_for(token: str, subdir: str) -> dict | None:
    """Load the same UPMEM baseline.json metrics the Fig-9 grid normalizes against."""
    root = PARETO_EVAL_ROOT / subdir
    runs = sorted(
        d for d in glob.glob(str(root / "*")) if (Path(d) / "baseline.json").is_file()
    )
    if not runs:
        return None
    payload = json.load(open(Path(runs[-1]) / "baseline.json"))
    return payload["metrics"]


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def recost(arch: Arch, token: str, M: int, C: int, K: int, tile: dict) -> dict | None:
    """Re-cost CINM's *own* tiling decision on our analytical model.

    Faithful to CINM's GEMM tiling (CinmTilingImplementations.cpp + CinmToCnm.cpp): the
    per-launch tile <p0 x p1> of the OUTPUT (p0 rows of N=M, p1 cols of K) is scattered
    across the workgroup -> spatial DPU parallelism = p0 * p1. The output-tile loops
    (ceil(M/p0) x ceil(K/p1)) and the C reduction are temporal (L1). A SIMD factor of the
    reduction goes on L0 (<= simd_lanes), matching how our model uses SIMD inside a DPU.

    PNMAX dims: N = output rows (M), K = output cols, C = reduction.
    Spatial levels = L2..L5 (caps = arch.hierarchy.organization). When p0/p1 exceed the
    available level capacity (e.g. p1=2560 > prod of caps), the excess stays temporal.
    """
    org = arch.hierarchy.organization
    simd = arch.specs.simd_lanes
    spatial_levels = sorted(l for l in org if l >= 2)  # [2,3,4,5]

    p0, p1 = tile["p0"], tile["p1"]
    lv: dict[int, dict[str, int]] = {l: {} for l in range(0, max(spatial_levels) + 1)}
    remaining_cap = {l: org[l] for l in spatial_levels}

    # Our model tiles the iteration space exactly: product of a dim's level-factors must
    # equal the problem dim. CINM wants ~p0 output-rows / ~p1 output-cols in parallel;
    # we realize the largest DIVISOR of the problem dim that is <= CINM's tile and fits
    # the available spatial-level capacity. Any unplaceable / non-dividing remainder
    # stays temporal (L1), so dims are conserved exactly. (Faithful: same spatial target,
    # rounded to a legal divisor — CINM itself pads partial tiles.)
    def place(dim: str, problem_dim: int, want: int) -> int:
        """Place spatial factors of `problem_dim` up to ~`want`; return product placed."""
        placed = 1
        budget = max(1, want)
        for l in spatial_levels:
            if placed >= budget:
                break
            cap = remaining_cap[l]
            rem_problem = problem_dim // placed  # remaining divisible part
            best = 1
            for d in sorted(_divisors(rem_problem), reverse=True):
                if 1 < d <= cap and placed * d <= budget:
                    best = d
                    break
            if best > 1:
                lv[l][dim] = lv[l].get(dim, 1) * best
                remaining_cap[l] //= best
                placed *= best
        return placed

    n_spatial = place("N", M, p0)
    k_spatial = place("K", K, p1)
    realized = n_spatial * k_spatial

    # SIMD factor of reduction C onto L0 (largest divisor of C that is <= simd).
    simd_c = 1
    for d in sorted(_divisors(C), reverse=True):
        if d <= simd:
            simd_c = d
            break

    # Temporal (L1): the exact remaining (divisor-conserving) part of each dim.
    n_temporal = M // n_spatial
    k_temporal = K // k_spatial
    c_temporal = C // simd_c

    lv[0] = {"C": simd_c}
    l1: dict[str, int] = {}
    if c_temporal > 1:
        l1["C"] = c_temporal
    if n_temporal > 1:
        l1["N"] = n_temporal
    if k_temporal > 1:
        l1["K"] = k_temporal
    lv[1] = l1

    wl = build(f"cinm_{token}", M, K, C, lv)
    metrics = evaluate_workload_analytical_report(
        wl,
        arch,
        target="upmem",
        latency_coeff=1.0,
        mem_footprint_coeff=1.0,
        energy_coeff=1.0,
        require_constraints=True,
        refresh=True,
    )
    if metrics is None:
        return None
    metrics["_realized_pus"] = realized
    return metrics


def main() -> int:
    arch = _load_arch()
    d, t = workgroup_geometry(arch)
    print(
        f"arch={ARCH_PATH}  hierarchy={arch.hierarchy.organization}  "
        f"simd={arch.specs.simd_lanes}  max_PUs={arch.hierarchy.total_units}  "
        f"workgroup=D{d}xT{t}={d*t}  cinm_opt={CINM_OPT}\n"
    )
    MLIR_IN.mkdir(parents=True, exist_ok=True)
    MLIR_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    hdr = (
        f"{'token':14} {'M':>8} {'C':>6} {'K':>9} {'CINM_tile(MxKredC)':22} "
        f"{'realPU':>6} {'lat_cyc':>10} {'norm_lat':>9} {'norm_mem':>9} {'norm_en':>8}"
    )
    print(hdr)
    for token, (subdir, yname) in WORKLOADS.items():
        ypath = str(UNINDP_BASELINE_DIR / yname)
        if os.environ.get("PNMAX_SMOKE") == "1" and not os.path.exists(ypath):
            # Smoke runs derive only the smoke workload subset (bert-3); skip
            # kernels whose baseline was not built. Full runs keep failing
            # loudly on a genuinely missing baseline.
            print(f"{token:14} SKIP (smoke: no derived baseline {yname})")
            continue
        M, C, K = shape_from_workload(ypath)
        in_path = emit_mlir(token, M, C, K, d, t)
        out = run_tiling(in_path, token)
        tile = parse_tiling(out, M, C, K)
        metrics = recost(arch, token, M, C, K, tile)
        baseline = baseline_for(token, subdir)
        if metrics is None:
            print(f"{token:14} {M:>8} {C:>6} {K:>9} CONSTRAINT-FAIL")
            continue
        lat = metrics["latency_cycles"]
        mem = metrics["mem_footprint_bytes"]
        en = metrics["energy"]
        if baseline:
            nl = lat / baseline["latency_cycles"]
            nm = mem / baseline["mem_footprint_bytes"]
            ne = en / baseline["energy"]
        else:
            nl = nm = ne = ""
        tilestr = f"{tile['p0']}x{tile['p1']}r{tile['r']}{'(notile)' if tile['notile'] else ''}"
        nl_s = f"{nl:.4f}" if isinstance(nl, float) else "n/a"
        nm_s = f"{nm:.4f}" if isinstance(nm, float) else "n/a"
        ne_s = f"{ne:.4f}" if isinstance(ne, float) else "n/a"
        print(
            f"{token:14} {M:>8} {C:>6} {K:>9} {tilestr:22} "
            f"{metrics['_realized_pus']:>6} {lat:>10} {nl_s:>9} {nm_s:>9} {ne_s:>8}"
        )
        rows.append(
            {
                "workload": token,
                "streaming": "false",
                "criterion": "cinm_tiling",
                "wg_D": d,
                "wg_T": t,
                "M": M,
                "C": C,
                "K": K,
                "cinm_M_tile": tile["p0"],
                "cinm_K_tile": tile["p1"],
                "cinm_C_reduce_tile": tile["r"],
                "cinm_notile": tile["notile"],
                "cinm_loop_steps": tile["loop_steps"],
                "realized_pus": metrics["_realized_pus"],
                "latency_cycles": lat,
                "mem_footprint_bytes": mem,
                "energy": en,
                "weighted_cost": metrics.get("weighted_cost", ""),
                "norm_latency": nl,
                "norm_mem": nm,
                "norm_energy": ne,
                "mlir_in": str(in_path),
                "mlir_out": str(MLIR_OUT / f"{token}.tiled.mlir"),
            }
        )

    fieldnames = [
        "workload", "streaming", "criterion", "wg_D", "wg_T", "M", "C", "K",
        "cinm_M_tile", "cinm_K_tile", "cinm_C_reduce_tile", "cinm_notile",
        "cinm_loop_steps", "realized_pus", "latency_cycles", "mem_footprint_bytes",
        "energy", "weighted_cost", "norm_latency", "norm_mem", "norm_energy",
        "mlir_in", "mlir_out",
    ]
    PROXY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PROXY_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {PROXY_CSV}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
