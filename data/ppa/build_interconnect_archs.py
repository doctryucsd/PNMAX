#!/usr/bin/env python3
"""Rebuild interconnect_sweep/{upmem,hbm_pim} (a derived arch family):
  center (no suffix) = host with bank_to_bank_transfer.latency x16 (post-generation edit)
  host-x1            = plain host (the x1 reference)
  inter-bg / intra-bg = generator --interconnect variants
The family is generated from the current DRAM lookup CSVs as part of the
standard flow (data/ppa/generate_ppa.sh). Output dir = argv[1] if given,
else the real interconnect_sweep/ tree."""
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GEN = SCRIPT_DIR / "generate_arch_from_template.py"
BASELINE = SCRIPT_DIR.parent / "archs" / "lowered" / "baseline"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR.parent / "archs" / "lowered" / "interconnect_sweep"

HOST_X16 = 16  # congested-host variant: base "host" file edited to x16 bank-to-bank latency
GEOM = {"upmem": dict(h=16, v=64), "hbm_pim": dict(h=16, v=32)}


def gen(arch, h, v, interconnect, out):
    cmd = ["python3", str(GEN), "--template", str(BASELINE / f"{arch}.yaml"), "--output", str(out),
           "--num-pu", "8", "--h-mat", str(h), "--v-mat", str(v), "--burst-len", "4",
           "--interconnect", interconnect]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"gen failed {out.name}: {(r.stderr or r.stdout).strip()[-200:]}")


def scale_bank_to_bank_latency(path, factor):
    txt = path.read_text()
    new = re.sub(r"(bank_to_bank_transfer:\s*\n\s*latency:\s*)([\d.]+)",
                 lambda m: f"{m.group(1)}{int(round(float(m.group(2)) * factor))}", txt, count=1)
    if new == txt:
        raise RuntimeError(f"could not find bank_to_bank_transfer latency in {path.name}")
    path.write_text(new)


def main():
    n = 0
    for arch, g in GEOM.items():
        h, v = g["h"], g["v"]
        d = OUT_DIR / arch
        d.mkdir(parents=True, exist_ok=True)
        base = f"{arch}__pu-8__hmat-{h}__vmat-{v}"
        # center = host x16
        center = d / f"{base}.yaml"
        gen(arch, h, v, "host", center)
        scale_bank_to_bank_latency(center, HOST_X16)
        # host-x1 = plain host
        gen(arch, h, v, "host", d / f"{base}_interconnect-host-x1.yaml")
        # bank-group variants
        gen(arch, h, v, "inter-bg", d / f"{base}_interconnect-inter-bg.yaml")
        gen(arch, h, v, "intra-bg", d / f"{base}_interconnect-intra-bg.yaml")
        n += 4
        print(f"{arch}: 4 written")
    print(f"TOTAL {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
