#!/usr/bin/env python3
"""Derive the UniNDP-baseline workload mappings (production path).

The Fig. 10 Pareto grid, the Fig. 15 buffer sweep, and the Fig. 17 reduction
study all consume "UniNDP baseline" workload YAMLs: for each evaluation kernel and
target architecture, UniNDP's own compiler picks its best mapping and that
mapping is transcribed into a PNMAX nested-loop workload. These files are NOT
shipped as static data — this script derives them at run time from the
vendored UniNDP snapshot:

1. compile  — run ``external/unindp/compile.py`` (via the namespace launcher
   ``external/unindp/_compile_driver.py``) for the kernel's GEMM shape with the
   quicksearch settings ``-Q -K 30``, from a scratch cwd under the
   results root so nothing is written into the vendored tree;
2. convert  — translate the log's ``best_design`` into a PNMAX ``Bound`` +
   derived ``Problem`` (``external/unindp/derive_baseline_yaml.py``), merged
   with the kernel's ``*-unindp.yaml`` template (static TensorAttr / Compute /
   Layout-Pragma fields); intra-PU splits larger than the architecture's
   ``specs.simd_lanes`` are rebalanced to a lanes-legal split (the reference
   baseline mappings carry the same normalization — UniNDP's compile-level
   split is not always realizable as one SIMD issue);
3. validate — round-trip the emitted mapping through
   ``pnmax.database.Workload`` (the step the standalone converter omits).

UniNDP's compile search is deterministic (no RNG on the quicksearch path), so
existing outputs are reused unless ``--force`` is passed.

Usage:
    uv run python experiments/_lib/derive_unindp_baselines.py \
        --output-dir <results>/unindp_baseline [--pairs bert-3:hbm_pim ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UNINDP_DIR = REPO_ROOT / "external" / "unindp"
COMPILE_DRIVER = UNINDP_DIR / "_compile_driver.py"
CONVERTER = UNINDP_DIR / "derive_baseline_yaml.py"

ARCHS = ("hbm_pim", "upmem")

# Kernel table (see external/unindp/derive_baselines.md): UniNDP compiles every
# GEMM/MV as ``-W mm -S M K N B`` with M=input rows, K=reduction, N=output
# columns, B=batch; the converter maps them onto PNMAX dims (M->K, K->C,
# N->P, B->N). Shapes are identical for both architectures (UniNDP pads the
# reduction dim per architecture during compile). Template files supply the
# static TensorAttr/Compute/Layout-Pragma fields.
#   token: (display_name, (M, K, N, B), template repo-relative path)
KERNELS: dict[str, tuple[str, tuple[int, int, int, int], str]] = {
    "alexnet-4": ("conv2d_alexnet-4", (43520, 2304, 1, 1), "data/workloads/conv2d/alexnet-4-unindp.yaml"),
    "bert-3": ("bmm_bert-3", (768, 768, 128, 1), "data/workloads/b_fc/bert-3-unindp.yaml"),
    "bert-72": ("fc_bert-72", (768, 768, 1, 1), "data/workloads/fc/bert-72-unindp.yaml"),
    "llama128-6": ("fc_llama128-6", (14336, 4096, 1, 1), "data/workloads/fc/llama128-6-unindp.yaml"),
    "llama256-3": ("bmm_llama256-3", (4096, 128, 256, 1), "data/workloads/b_fc/llama256-3-unindp.yaml"),
    "llama512-3": ("bmm_llama512-3", (4096, 128, 512, 1), "data/workloads/b_fc/llama512-3-unindp.yaml"),
    "resnet50-2": ("conv2d_resnet50-2", (200704, 576, 1, 1), "data/workloads/conv2d/resnet50-2-unindp.yaml"),
    "resnet50-53": ("fc_resnet50-53", (1000, 2048, 1, 1), "data/workloads/fc/resnet50-53-unindp.yaml"),
    "vgg16-0": ("conv2d_vgg16-0", (3211264, 27, 1, 1), "data/workloads/conv2d/vgg16-0-unindp.yaml"),
}

QUICKSEARCH_TOPK = 30  # UniNDP quicksearch setting: -Q -K 30

# Kernels whose compiled -S deviates from the template-implied true dims.
# alexnet-4: the original run compiled M = 43520 = 13*13*256 + 256 (one extra
# 256-row block over the true im2col row count) — verified against the
# original reference baseline mappings, which carry K: 43520 on both
# architectures. Keep that value; the guard below accepts it.
REFERENCE_S_OVERRIDES: dict[str, tuple[int, int, int, int]] = {
    "alexnet-4": (43520, 2304, 1, 1),
}


def _arch_simd_lanes(arch: str) -> int:
    """Physical SIMD lane count from the experiment-default arch YAML."""
    import yaml

    arch_file = REPO_ROOT / "data" / "archs" / "lowered" / "baseline" / f"{arch}.yaml"
    payload = yaml.safe_load(arch_file.read_text(encoding="utf-8"))
    return int(payload["specs"]["simd_lanes"])


def _sizes_from_template(template_path: Path) -> tuple[int, int, int, int]:
    """Recompute the UniNDP GEMM size (M, K, N, B) from the template Problem.

    conv2d kernels (spatial Q/R/S extents > 1) lower im2col-style:
    M = N*K*P*Q output rows, reduction = C*R*S.  fc/bmm kernels (Q=R=S=1)
    map M = N*K output rows (the batch/head dim N folds into M, mirroring
    UniNDP's own B-folding), reduction = C, with the column count P as
    UniNDP's N.  Guards the hand-maintained KERNELS table against shape
    typos — UniNDP must be fed TRUE dims (it pads per architecture itself).
    """
    import yaml

    problem = yaml.safe_load(Path(template_path).read_text(encoding="utf-8"))["Problem"]

    def dim(name: str) -> int:
        return int(problem.get(name, 1))

    if dim("Q") == 1 and dim("R") == 1 and dim("S") == 1:
        return (dim("N") * dim("K"), dim("C"), dim("P"), 1)
    return (dim("N") * dim("K") * dim("P") * dim("Q"), dim("C") * dim("R") * dim("S"), 1, 1)


def _load_converter():
    spec = importlib.util.spec_from_file_location("derive_baseline_yaml", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _compile_one(
    token: str,
    arch: str,
    sizes: tuple[int, int, int, int],
    workdir: Path,
    *,
    quiet: bool = False,
) -> Path:
    """Run UniNDP compile for one (kernel, arch); return the log path."""
    workdir.mkdir(parents=True, exist_ok=True)
    config_link = workdir / "config"
    if not config_link.exists():
        config_link.symlink_to(UNINDP_DIR / "config")

    out_name = f"{token}_{arch}"
    log_path = workdir / out_name / "log" / f"_{token}.log"
    m, k, n, b = sizes
    cmd = [
        sys.executable,
        str(COMPILE_DRIVER),
        "--",
        "-A", arch,
        "-W", "mm",
        "-N", token,
        "-S", str(m), str(k), str(n), str(b),
        "-WS", ".",
        "-O", out_name,
        "-Q",
        "-K", str(QUICKSEARCH_TOPK),
    ]
    print(f"[unindp-baseline] compile {token} on {arch}: -S {m} {k} {n} {b}", flush=True)
    if quiet:
        # Parallel mode: keep stdout readable, capture per-pair output.
        stdout_log = workdir / "compile_stdout.log"
        with stdout_log.open("w") as sink:
            subprocess.run(cmd, cwd=workdir, check=True, stdout=sink, stderr=sink)
    else:
        subprocess.run(cmd, cwd=workdir, check=True)
    if not log_path.is_file():
        raise FileNotFoundError(f"UniNDP compile log not found: {log_path}")
    return log_path


def _convert_one(
    converter,
    *,
    token: str,
    arch: str,
    display_name: str,
    sizes: tuple[int, int, int, int],
    log_path: Path,
    template_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter.dump_yaml(
        log_path=str(log_path),
        template_path=str(template_path),
        output_path=str(output_path),
        name=display_name,
        workload_size=sizes,
        design_kind="best",
        best_index=1,
        simd_lanes=_arch_simd_lanes(arch),
    )
    # Production-path extra vs the standalone converter: validate the emitted
    # mapping through the PNMAX workload schema.
    from pnmax.database import Workload

    Workload.from_file(output_path)
    print(f"[unindp-baseline] wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the derived <token>_<arch>.yaml baselines.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Scratch dir for UniNDP compile artefacts (default: <output-dir>/_work).",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help=(
            "Subset of '<token>:<arch>' pairs to derive "
            "(default: all 9 kernels x {hbm_pim, upmem})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-derive even when the output file already exists.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=9,
        help=(
            "Parallel UniNDP compiles (default 9, one per kernel — a fixed "
            "literal; compiles are independent single-thread processes, so "
            "parallelism cannot change their outputs)."
        ),
    )
    args = parser.parse_args()

    if args.pairs:
        pairs = []
        for item in args.pairs:
            token, _, arch = item.partition(":")
            if token not in KERNELS or arch not in ARCHS:
                raise SystemExit(f"unknown pair '{item}' (want <token>:<arch>)")
            pairs.append((token, arch))
    else:
        pairs = [(token, arch) for token in KERNELS for arch in ARCHS]

    converter = _load_converter()
    work_root = args.work_dir or (args.output_dir / "_work")
    todo: list[tuple[str, str]] = []
    skipped = 0
    for token, arch in pairs:
        output_path = args.output_dir / f"{token}_{arch}.yaml"
        if output_path.is_file() and not args.force:
            print(f"[unindp-baseline] reuse existing {output_path}")
            skipped += 1
            continue
        template_path = REPO_ROOT / KERNELS[token][2]
        if not template_path.is_file():
            raise SystemExit(f"template not found: {template_path}")
        expected_sizes = _sizes_from_template(template_path)
        table_sizes = tuple(KERNELS[token][1])
        if table_sizes != expected_sizes and table_sizes != REFERENCE_S_OVERRIDES.get(token):
            raise SystemExit(
                f"KERNELS table for '{token}' requests -S {table_sizes}, but the "
                f"template Problem implies -S {expected_sizes} — fix the table "
                f"(UniNDP must be fed the true dims; it pads per architecture), "
                f"or record a documented REFERENCE_S_OVERRIDES entry."
            )
        todo.append((token, arch))

    # Compile stage — embarrassingly parallel across (kernel, arch) pairs.
    logs: dict[tuple[str, str], Path] = {}
    jobs = max(1, args.jobs)
    if todo:
        from concurrent.futures import ThreadPoolExecutor

        def _compile_job(pair: tuple[str, str]) -> tuple[tuple[str, str], Path]:
            token, arch = pair
            sizes = KERNELS[token][1]
            log = _compile_one(
                token, arch, sizes, work_root / f"{token}_{arch}", quiet=jobs > 1
            )
            return pair, log

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for pair, log in pool.map(_compile_job, todo):
                logs[pair] = log

    # Convert + validate stage — fast, kept serial and in a stable order.
    for token, arch in todo:
        display_name, sizes, _ = KERNELS[token]
        _convert_one(
            converter,
            token=token,
            arch=arch,
            display_name=display_name,
            sizes=sizes,
            log_path=logs[(token, arch)],
            template_path=REPO_ROOT / KERNELS[token][2],
            output_path=args.output_dir / f"{token}_{arch}.yaml",
        )

    print(
        f"[unindp-baseline] done: {len(todo)} derived, {skipped} reused -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
