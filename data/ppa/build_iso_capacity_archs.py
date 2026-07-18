#!/usr/bin/env python3
"""Build the flat iso_capacity_archs/ geometry sweep (27 YAMLs) by shelling out to the
arch generator CLI (so per-arch buffer/interconnect defaults resolve correctly).

FULL iso-capacity invariant: P = num_pu*h_mat*v_mat is held at the baseline value on
EVERY file (upmem 8192=512MB, hbm_pim 4096=256MB). Each capacity axis swept by factor f
is compensated by scaling ONE of the other two axes by 1/f (third stays baseline) -> each
deviated file moves exactly two of {pu,hmat,vmat}. The 3 per-axis sweeps collapse to the
unique deviated set listed in "iso" (dedup by filename). burst axis is capacity-neutral.
hbm_pim drops 7 of 12 deviated configs (pu in {4,8,16} only; DreamRAM status-7 die limit)."""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GEN = SCRIPT_DIR / "generate_arch_from_template.py"
BASELINE = SCRIPT_DIR.parent / "archs" / "lowered" / "baseline"
OUT_DIR = SCRIPT_DIR.parent / "archs" / "lowered" / "geometry_sweep" / "iso_capacity_archs"

# (num_pu, h_mat, v_mat, burst_len). Center shared across axes; dedup by filename.
PLANS = {
    "upmem": {  # P = pu*h*v = 8192 (512 MB) on every file
        "center": (8, 16, 64, 4),
        # center + 12 deviated configs (each: one axis xm, one axis /m, third baseline; m in {2,4})
        "iso": [
            (8, 16, 64, 4),                                              # center
            (2, 16, 256, 4), (2, 64, 64, 4),                            # pu 1/4 (comp v, h)
            (4, 16, 128, 4), (4, 32, 64, 4),                            # pu 1/2 (comp v, h)
            (8, 4, 256, 4), (8, 8, 128, 4), (8, 32, 32, 4), (8, 64, 16, 4),  # aspect (pu fixed)
            (16, 8, 64, 4), (16, 16, 32, 4),                            # pu 2x (comp h, v)
            (32, 4, 64, 4), (32, 16, 16, 4),                            # pu 4x (comp h, v)
        ],
        "burst": [(8, 16, 64, b) for b in (1, 2, 4, 8, 16)],
    },
    "hbm_pim": {  # P = pu*h*v = 4096 (256 MB) on every file
        "center": (8, 16, 32, 4),
        # center + 5 DreamRAM-feasible deviated (7 of 12 rejected: pu in {4,8,16} only -> gbus;
        # h64/v8, h8/v64, h4/v128 -> status-7 die-size limit)
        "iso": [
            (8, 16, 32, 4),                                             # center
            (4, 16, 64, 4), (4, 32, 32, 4),                            # pu 1/2 (comp v, h)
            (8, 32, 16, 4),                                            # aspect 2x (pu fixed)
            (16, 8, 32, 4), (16, 16, 16, 4),                          # pu 2x (comp h, v)
        ],
        "burst": [(8, 16, 32, b) for b in (1, 2, 4, 8, 16)],
    },
}


def fname(arch, pu, h, v, burst):
    name = f"{arch}__pu-{pu}__hmat-{h}__vmat-{v}"
    if burst != 4:
        name += f"_burst-{burst}"
    return name + ".yaml"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = [], []
    for arch, plan in PLANS.items():
        template = BASELINE / f"{arch}.yaml"
        arch_dir = OUT_DIR / arch  # per-arch subdirs: iso_capacity_archs/{upmem,hbm_pim}/
        arch_dir.mkdir(parents=True, exist_ok=True)
        combos = {}  # filename -> tuple (dedup center)
        for axis in ("iso", "burst"):
            for (pu, h, v, b) in plan[axis]:
                combos[fname(arch, pu, h, v, b)] = (pu, h, v, b)
        for name, (pu, h, v, b) in sorted(combos.items()):
            out = arch_dir / name
            cmd = [
                "python3", str(GEN),
                "--template", str(template),
                "--output", str(out),
                "--num-pu", str(pu),
                "--h-mat", str(h),
                "--v-mat", str(v),
                "--burst-len", str(b),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and out.exists():
                ok.append(name)
            else:
                fail.append((name, (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else "no output"))
        print(f"{arch}: {sum(1 for n in ok if n.startswith(arch))} written")
    print(f"\nTOTAL written: {len(ok)}")
    if fail:
        print(f"FAILED: {len(fail)}")
        for name, err in fail:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
