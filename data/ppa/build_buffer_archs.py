#!/usr/bin/env python3
"""Rebuild buffer_sweep/{upmem,hbm_pim} by driving generate_arch_from_template.py.
Reproduces the exact combos (center geometry, buffer-size axis). The old SRAM ×10 patch
is NOT reapplied (the generator produces ×1 SRAM); PU ×10 for hbm comes from the generator.
Output dir = argv[1] if given, else the real buffer_sweep/."""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GEN = SCRIPT_DIR / "generate_arch_from_template.py"
BASELINE = SCRIPT_DIR.parent / "archs" / "lowered" / "baseline"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR.parent / "archs" / "lowered" / "buffer_sweep"

# (geometry center, default buffer, swept buffer set). Default buffer -> no filename suffix.
PLANS = {
    "upmem":   dict(h=16, v=64, default="32KB", buffers=["8KB", "16KB", "32KB", "64KB", "128KB"]),
    "hbm_pim": dict(h=16, v=32, default="512B", buffers=["32B", "64B", "128B", "256B", "512B", "1KB", "2KB"]),
}


def fname(arch, h, v, buf, default):
    name = f"{arch}__pu-8__hmat-{h}__vmat-{v}"
    if buf != default:
        name += f"_buffer-{buf}"
    return name + ".yaml"


def main():
    ok, fail = [], []
    for arch, p in PLANS.items():
        template = BASELINE / f"{arch}.yaml"
        arch_dir = OUT_DIR / arch
        arch_dir.mkdir(parents=True, exist_ok=True)
        for buf in p["buffers"]:
            out = arch_dir / fname(arch, p["h"], p["v"], buf, p["default"])
            cmd = ["python3", str(GEN), "--template", str(template), "--output", str(out),
                   "--num-pu", "8", "--h-mat", str(p["h"]), "--v-mat", str(p["v"]),
                   "--burst-len", "4", "--buffer-size", buf]
            r = subprocess.run(cmd, capture_output=True, text=True)
            (ok if (r.returncode == 0 and out.exists()) else fail).append(out.name)
            if r.returncode != 0:
                print(f"  FAIL {out.name}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
        print(f"{arch}: {sum(1 for n in ok if n.startswith(arch))} written")
    print(f"TOTAL {len(ok)}; FAILED {len(fail)}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
