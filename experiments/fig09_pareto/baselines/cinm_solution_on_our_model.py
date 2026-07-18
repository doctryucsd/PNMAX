#!/usr/bin/env python3
"""Evaluate CINM (Cinnamon) testbench *solutions* on the PNMAX analytical model.

Approach (no UPMEM hardware): the CINM testbench MLIRs
(``external/cinm/testbench/*.mlir``) encode both the problem (op + tensor shapes)
AND CINM's mapping decision (``workgroupShape=array<i64: D, T, 1>`` = a grid of
D*T parallel UPMEM DPUs, with T = DPUs-per-DIMM — derived from the arch
hierarchy via ``workgroup_geometry``, 128 for baseline UPMEM — and D = #DIMMs
per benchmark variant). We translate
each PNMAX-supported op (gemm / gemv / conv-as-im2col-gemm) into a PNMAX workload +
mapping pragma, replicating CINM's parallelism choice, then evaluate on our
analytical model under the baseline UPMEM arch.

Sources (verified):
- shapes + workgroupShape: external/cinm/testbench/{1mm,mv,conv,2mm,3mm}.mlir
- HW reference (real UPMEM, ms): external/cinm/artifact/plot/exp-fig-11.txt
  columns per benchmark = [4d-nopt, 4d-opt, 8d-nopt, 8d-opt, 16d-nopt, 16d-opt]
  (column meaning from external/cinm/testbench/get_results.py + plot-fig-11.py)
- cycle->ms uses the codebase's own Arch.cycles_to_ns (350 MHz baseline UPMEM).

workgroupShape -> PNMAX mapping correspondence:
- PNMAX Bound levels: L0 = SIMD lanes (<= simd_lanes), L1 = temporal steps,
  L2..Lmax = the spatial hierarchy (pu/chip/rank/channel). _num_all_used_pus =
  product of levels 2..max.  (analytical_model.py)
- CINM's workgroup of |WG| = D*T DPUs => spatial parallelism over the GEMM OUTPUT
  dims (N,K), because CinmToCnm scatters the output across |WG| DPUs
  (lib/Conversion/CinmToCnm/CinmToCnm.cpp). The reduction dim C is temporal (L1).

Artifact adaptation (PNMAX AE): imports use the renamed ``pnmax`` package; the
baseline arch path resolves repo-relatively so the script is cwd-independent.
Wired into ``run.sh`` and exercised by the full Fig-9 campaign — see README.md.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

from pnmax.database import Arch, Workload
from pnmax.dse.workload_eval import evaluate_workload_analytical_report

# experiments/fig09_pareto/baselines/ -> repo root is 3 levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
ARCH_PATH = os.environ.get(
    "PNMAX_UPMEM_ARCH", str(REPO_ROOT / "data" / "archs" / "lowered" / "baseline" / "upmem.yaml")
)


def _load_arch() -> Arch:
    return Arch.from_yaml_file(ARCH_PATH)


def workgroup_geometry(arch: Arch) -> tuple[int, int]:
    """(D, T) of the CINM UPMEM workgroup, derived from the arch hierarchy.

    T = DPUs per DIMM = product of the intra-DIMM spatial levels (pu * chip *
    rank, L2..L4); D = #DIMMs = the top spatial level (channel, L5).  Baseline
    ``upmem.yaml``: T = 8*8*2 = 128 (the testbench workgroupShape's second
    element), D = 20, so |WG| = D*T = 2560 = every DPU in the arch.
    """
    org = arch.hierarchy.organization
    spatial = sorted(l for l in org if l >= 2)
    return org[spatial[-1]], math.prod(org[l] for l in spatial[:-1])


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def map_gemm(arch: Arch, N: int, K: int, C: int, target_pus: int):
    """Greedily distribute output (N,K) across spatial levels (respecting caps)
    up to target_pus DPUs; reduction C -> temporal L1; a K factor (<=simd) -> L0."""
    org = arch.hierarchy.organization
    simd = arch.specs.simd_lanes
    spatial = sorted(l for l in org if l >= 2)
    cap = {l: org[l] for l in spatial}

    target = min(target_pus, N * K, math.prod(cap.values()))
    Nrem, Krem = N, K
    lv: dict[int, dict[str, int]] = {l: {} for l in spatial}
    realized = 1
    for l in spatial:
        budget = target // realized
        c = 1
        for d in sorted(_divisors(Nrem), reverse=True):
            if 1 < d <= cap[l] and d <= budget:
                c = d
                lv[l]["N"] = d
                Nrem //= d
                break
        if c == 1:
            for d in sorted(_divisors(Krem), reverse=True):
                if 1 < d <= cap[l] and d <= budget:
                    c = d
                    lv[l]["K"] = d
                    Krem //= d
                    break
        realized *= c
    simd_k = 1
    for d in sorted(_divisors(Krem), reverse=True):
        if d <= simd:
            simd_k = d
            break
    Krem //= simd_k
    lv[1] = {"C": C}
    if Nrem > 1:
        lv[1]["N"] = Nrem
    if Krem > 1:
        lv[1]["K"] = Krem
    lv[0] = {"K": simd_k}
    for l in range(max(lv) + 1):
        lv.setdefault(l, {})
    return lv, realized


def build(name: str, N: int, K: int, C: int, lv: dict[int, dict[str, int]]) -> Workload:
    dims = ("N", "K", "P", "Q", "C", "R", "S")
    return Workload.from_dict(
        {
            "Name": name,
            "Problem": {"N": N, "K": K, "P": 1, "Q": 1, "C": C, "R": 1, "S": 1},
            "Bound": {
                l: [{d: lv[l].get(d, 1)} for d in dims] for l in lv
            },
            "TensorAttr": {"In": "16b", "F": "16b", "Out": "16b"},
            "Compute": "Out[n,p,q,k] += In[n,p+r,q+s,c] * F[k,r,s,c]",
            "Layout-Pragma": {
                "cache": False,
                "bit": "bit-parallel",
                "streaming": {"In": True, "F": False, "Out": False},
                "sharing": {"block": True, "PU": False, "system": 0},
                "interleaving": True,
            },
        }
    )


def evaluate(arch: Arch, wl: Workload):
    return evaluate_workload_analytical_report(
        wl,
        arch,
        target="upmem",
        latency_coeff=1.0,
        mem_footprint_coeff=1.0,
        energy_coeff=1.0,
        require_constraints=True,
        refresh=True,
    )


# (N, C, K) per dimm count from the testbench MLIRs (the true problem shapes),
# cross-checked at startup by check_nopt_against_testbench(): the value is the
# live *nopt* func where upstream keeps one (1mm d4/d8, mv d8, conv all), the
# live *opt* func where the nopt is commented out (mv d4/d16), and the
# commented-out nopt where upstream disabled BOTH variants (1mm d16 — its
# shapes still document the benchmark config the HW reference table covers).
# 1mm: cinm.op.gemm A(N x C) . B(C x K) -> (N x K)
NOPT = {
    "1mm": {4: (8, 1024, 256), 8: (8, 1024, 128), 16: (8, 1024, 64)},
    # mv: cinm.op.gemv A(N x C) . B(C) -> (N); K=1
    "mv": {4: (4096, 2048, 1), 8: (16384, 512, 1), 16: (16384, 128, 1)},
    # conv: im2col gemm A(NP x C) . B(C x K) -> (NP x K)
    "conv": {4: (25088, 16, 256), 8: (25088, 8, 256), 16: (25088, 4, 256)},
}
# Testbench MLIRs backing NOPT, checked by check_nopt_against_testbench(). The
# source is keyed strictly by FILE: conv.mlir's funcs are also named
# ``@mm_dimm*`` (like 1mm.mlir's), so a func-name prefix cannot disambiguate.
TESTBENCH_DIR = REPO_ROOT / "external" / "cinm" / "testbench"
NOPT_MLIR = {"1mm": "1mm.mlir", "mv": "mv.mlir", "conv": "conv.mlir"}
# 2mm / 3mm = 1mm repeated 2x / 3x (same shape, 2/3 gemm ops in the compute block);
# under our model (independent layers summed) latency = K_layers * 1mm.
REPEAT = {"2mm": ("1mm", 2), "3mm": ("1mm", 3)}

# HW reference (ms), real UPMEM, nopt columns from plot/exp-fig-11.txt.
HW_NOPT = {
    "1mm": {4: 180.471, 8: 90.353667, 16: 45.349667},
    "2mm": {4: 360.908, 8: 180.699, 16: 90.682},
    "3mm": {4: 541.395, 8: 271.093333, 16: 136.064},
    "conv": {4: 6430.966333, 8: 3455.250333, 16: 1960.521667},
    "mv": {4: 477.512, 8: 232.491, 16: 116.4792},
}

# A func.func decl line (live or //-commented) carries the DIMM count, the
# variant, and the %A (NxC) / %B (CxK gemm | C gemv) tensor shapes; the
# workgroupShape (D, T, 1) sits in the func body two lines down.
_FUNC_RE = re.compile(
    r"func\.func\s+@(?P<name>\w+_dimm(?P<d>\d+)_(?P<variant>nopt|opt))\s*\(\s*"
    r"%A:\s*tensor<(?P<a>[^>]+)>\s*,\s*%B:\s*tensor<(?P<b>[^>]+)>"
)
_WG_RE = re.compile(r"workgroupShape\s*=\s*array<i64:\s*(\d+),\s*(\d+),\s*1>")


def _uncomment(line: str) -> str:
    """Strip a single leading ``//`` marker (upstream disables a func variant by
    line-commenting its whole body; the shapes inside still document the config)."""
    stripped = line.lstrip()
    return stripped[2:] if stripped.startswith("//") else line


def _tensor_dims(spec: str) -> list[int]:
    """``"8x1024xi32"`` -> ``[8, 1024]``; ``"2048xi32"`` -> ``[2048]`` (the
    trailing element type is dropped)."""
    return [int(p) for p in spec.split("x")[:-1]]


def _parse_testbench_shapes(path: Path) -> dict[tuple[int, str], dict]:
    """Parse every ``@..._dimm<D>_<nopt|opt>`` func in a testbench MLIR, INCLUDING
    the //-commented (upstream-disabled) ones, whose shapes still document the
    benchmark config. Returns records keyed ``(D, variant)`` with fields
    ``N, C, K, wg_d, wg_t, live, name``; ``live`` is False when the func's
    declaration line is commented out.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    # Index every func declaration line and whether it is live (not commented).
    decls: list[tuple[int, bool]] = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        commented = stripped.startswith("//")
        body = stripped[2:].lstrip() if commented else stripped
        if body.startswith("func.func") and "_dimm" in body:
            decls.append((i, not commented))
    records: dict[tuple[int, str], dict] = {}
    for idx, (start, live) in enumerate(decls):
        end = decls[idx + 1][0] if idx + 1 < len(decls) else len(lines)
        # A commented func's whole body is line-commented; uncomment before parse.
        chunk = "\n".join(_uncomment(l) for l in lines[start:end])
        fm = _FUNC_RE.search(chunk)
        wm = _WG_RE.search(chunk)
        if fm is None or wm is None:
            raise ValueError(f"{path.name}: unparseable func chunk at line {start + 1}")
        a_dims = _tensor_dims(fm.group("a"))
        b_dims = _tensor_dims(fm.group("b"))
        records[(int(fm.group("d")), fm.group("variant"))] = {
            "N": a_dims[0],
            "C": a_dims[1],
            "K": b_dims[1] if len(b_dims) == 2 else 1,
            "wg_d": int(wm.group(1)),
            "wg_t": int(wm.group(2)),
            "live": live,
            "name": fm.group("name"),
        }
    return records


def check_nopt_against_testbench(expected_t: int) -> None:
    """Fail fast if the NOPT table drifts from the vendored testbench MLIRs, or if
    the DPUs-per-DIMM (T) baked into every workgroupShape is not ``expected_t``.

    Per ``(bench, D)`` the backing func is selected by preference
    live-nopt -> live-opt -> commented-nopt: upstream disables a variant by
    line-commenting it, so where the nopt func is dead we fall back to the live
    opt func, and where BOTH are dead (1mm d16) to the commented nopt whose
    shapes still document the benchmark config the HW reference table covers.
    """
    for bench, dm in NOPT.items():
        mlir = NOPT_MLIR[bench]
        records = _parse_testbench_shapes(TESTBENCH_DIR / mlir)
        for D, shape in dm.items():
            nopt = records.get((D, "nopt"))
            opt = records.get((D, "opt"))
            if nopt is not None and nopt["live"]:
                rec = nopt
            elif opt is not None and opt["live"]:
                rec = opt
            elif nopt is not None:
                rec = nopt
            else:
                raise ValueError(
                    f"{bench} d{D}: no func @*_dimm{D}_(nopt|opt) in {mlir} "
                    f"backing NOPT value {shape}"
                )
            got = (rec["N"], rec["C"], rec["K"])
            if got != shape:
                raise ValueError(
                    f"{bench} d{D} (func @{rec['name']} in {mlir}): parsed shape "
                    f"(N,C,K)={got} != NOPT value {shape}"
                )
            if rec["wg_d"] != D:
                raise ValueError(
                    f"{bench} d{D} (func @{rec['name']} in {mlir}): workgroupShape "
                    f"D={rec['wg_d']} != DIMM key {D}"
                )
            if rec["wg_t"] != expected_t:
                raise ValueError(
                    f"{bench} d{D} (func @{rec['name']} in {mlir}): workgroupShape "
                    f"T={rec['wg_t']} != expected T {expected_t}"
                )


def main() -> None:
    arch = _load_arch()
    # T = DPUs per DIMM, derived from the arch hierarchy (128 for baseline
    # UPMEM); the guard also cross-checks the NOPT table vs the testbench MLIRs.
    _, t = workgroup_geometry(arch)
    check_nopt_against_testbench(t)
    freq = arch.specs.pu_frequency_hz
    print(f"arch={ARCH_PATH}  hierarchy={arch.hierarchy.organization}  "
          f"simd={arch.specs.simd_lanes}  freq={freq/1e6:.0f}MHz  "
          f"max_PUs={arch.hierarchy.total_units}\n")
    hdr = (f"{'bench':5} {'D':>2} {'wgDPU':>5} {'realPU':>6} {'shape(NxCxK)':16} "
           f"{'lat_cyc':>9} {'our_ms':>8} {'HW_ms':>9} {'HW/our':>7} {'energy_pJ':>13}")
    print(hdr)
    base_ms: dict[tuple[str, int], float] = {}
    for bench, dm in NOPT.items():
        for D, (N, C, K) in dm.items():
            lv, real = map_gemm(arch, N, K, C, D * t)
            wl = build(f"{bench}_nopt_d{D}", N, K, C, lv)
            m = evaluate(arch, wl)
            if m is None:
                print(f"{bench:5} {D:>2} {D*t:>5} {'-':>6} {f'{N}x{C}x{K}':16} CONSTRAINT-FAIL")
                continue
            ms = arch.cycles_to_ns(m["latency_cycles"]) / 1e6
            base_ms[(bench, D)] = ms
            hw = HW_NOPT[bench][D]
            print(f"{bench:5} {D:>2} {D*t:>5} {real:>6} {f'{N}x{C}x{K}':16} "
                  f"{m['latency_cycles']:>9} {ms:>8.4f} {hw:>9.2f} {hw/ms:>7.0f} {m['energy']:>13}")
    for bench, (src, mult) in REPEAT.items():
        for D in (4, 8, 16):
            ms = base_ms[(src, D)] * mult
            hw = HW_NOPT[bench][D]
            print(f"{bench:5} {D:>2} {'(='+str(mult)+'x'+src+')':>12} "
                  f"{'':16} {'':>9} {ms:>8.4f} {hw:>9.2f} {hw/ms:>7.0f}")


if __name__ == "__main__":
    main()
