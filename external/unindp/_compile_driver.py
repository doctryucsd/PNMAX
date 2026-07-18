#!/usr/bin/env python3
"""Standalone launcher for UniNDP's ``compile.py``.

UniNDP mixes top-level imports (``from sim import sim`` in ``compile.py``) with
package-relative imports (``from ..tools import *`` in ``sim/*.py``), so it must
run as a namespace sub-package.  This launcher reproduces the import wiring used
by the PNMAX artifact's ``helpers/run_unindp_compile.py`` and then execs
``compile.py`` with whatever arguments follow ``--``.

compile.py reads its arch config from ``./config/*.yaml`` and writes its log/csv
under ``./<workspace>/<outdir>/`` — both relative to the *current working
directory*.  Run this launcher from a writable scratch dir that contains a
``config`` symlink (or copy) so nothing is written into the vendored tree; see
``derive_baselines.sh``.

Usage (from the scratch cwd):
    python /path/to/_compile_driver.py -- -A hbm_pim -W mm -N bert-3 \
        -S 128 768 768 1 -WS . -O bert-3_hbm_pim -Q -K 30
"""
from __future__ import annotations

import importlib
import os
import runpy
import sys

UNINDP_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(UNINDP_DIR)  # so `unindp` is importable as a package


def main(argv: list[str]) -> None:
    if "--" in argv:
        compile_args = argv[argv.index("--") + 1:]
    else:
        compile_args = argv
    # UniNDP maps arch name "hbm_pim" (PNMAX spelling) to its own "hbm-pim".
    rewritten = list(compile_args)
    for i, arg in enumerate(rewritten[:-1]):
        if arg in {"-A", "--architecture"} and rewritten[i + 1] == "hbm_pim":
            rewritten[i + 1] = "hbm-pim"

    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    if UNINDP_DIR not in sys.path:
        sys.path.insert(0, UNINDP_DIR)
    # Bind the relative-import parent so `from ..tools` / `from sim import sim` resolve.
    sys.modules["tools"] = importlib.import_module("unindp.tools")
    sys.modules["sim"] = importlib.import_module("unindp.sim")

    # NB: do NOT chdir — the caller sets cwd to a scratch dir with ./config so
    # compile.py's outputs stay out of the vendored snapshot.
    sys.argv = ["compile.py", *rewritten]
    runpy.run_path(os.path.join(UNINDP_DIR, "compile.py"), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
