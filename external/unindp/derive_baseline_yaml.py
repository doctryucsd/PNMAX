#!/usr/bin/env python3
"""Convert a UniNDP ``compile.py`` log into a PNMAX baseline workload YAML.

This is a self-contained port of the conversion logic in the PNMAX artifact's
``helpers/run_unindp_compile.py`` (``dump-yaml`` sub-command).  It is bundled
here so that ``derive_baselines.sh`` can run the full UniNDP-baseline derivation
*without* importing the (possibly in-flight) ``pnmax`` package — the only thing
the in-tree helper adds on top of this file is a final
``pnmax.database.Workload.from_dict`` schema-validation of the emitted mapping.

Recipe context (D7): the 18 files in ``workloads/unindp_baseline/*.yaml`` are the
*reference* baselines.  Each is produced by (1) running UniNDP's compiler for the
workload's GEMM shape on the target architecture and (2) translating UniNDP's
chosen ``best_design`` partition into a PNMAX nested-loop ``Bound``.  This script
performs step (2).
"""
from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Any, Sequence

import yaml

CANONICAL_DIMS = ("N", "K", "P", "Q", "C", "R", "S")


def _parse_design(text: str):
    text = re.sub(r"<LEVEL\.([A-Z]+): \d+>", r"'LEVEL.\1'", text)
    return ast.literal_eval(text)


def parse_compile_log(log_path: str) -> dict[str, Any]:
    baseline = None
    best_designs = None
    best_result = None
    speedup = None
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("baseline strategy:"):
                baseline = _parse_design(line.split(":", 1)[1].strip())
            elif line.startswith("best_design:"):
                best_designs = _parse_design(line.split(":", 1)[1].strip())
            elif line.startswith("best_result:"):
                best_result = line.split(":", 1)[1].strip()
            elif line.startswith("speedup:"):
                speedup = line.split(":", 1)[1].strip()
    if best_designs is None:
        best_design_list: list[Any] = []
    elif isinstance(best_designs, list):
        best_design_list = list(best_designs)
    else:
        best_design_list = [best_designs]
    return {
        "baseline": baseline,
        "best_designs": best_design_list,
        "best_result": best_result,
        "speedup": speedup,
    }


def _normalize_mm_workload_size(workload_size: Sequence[int]) -> tuple[int, int, int, int]:
    if len(workload_size) != 4:
        raise ValueError(f"workload_size must have 4 ints for mm; got {workload_size}.")
    m_size, k_size, n_size, b_size = (int(item) for item in workload_size)
    return (m_size * b_size, k_size, n_size, 1)


def _make_bound_level(*, n: int, k: int, p: int, c: int) -> list[dict[str, int]]:
    return [
        {"N": int(n)}, {"K": int(k)}, {"P": int(p)}, {"Q": 1},
        {"C": int(c)}, {"R": 1}, {"S": 1},
    ]


def _split_factors(partition):
    m_split = k_split = l_split = b_split = 1
    for level_partition in partition:
        m_split *= level_partition[0]
        k_split *= level_partition[1]
        l_split *= level_partition[2]
        b_split *= level_partition[3]
    return m_split, k_split, l_split, b_split


def _clamp_simd_split(simd_k, simd_l, simd_lanes) -> tuple[int, int]:
    """Clamp an intra-PU ``(simd_k, simd_l)`` split to the PU's SIMD width.

    UniNDP's mm compiler can select an intra-PU split whose element count
    (``simd_k * simd_l``) exceeds the physical SIMD lane count of the modeled
    PU (e.g. 16x16 = 256 elements on a 16-lane HBM-PIM PU).  The PNMAX mapping
    format prices Bound level 0 as one SIMD issue, so an oversubscribed split
    would be costed optimistically.  Rebalance by halving the larger even
    factor until the split is lanes-legal; the displaced factors are absorbed
    into the level-1 temporal loop by the bound arithmetic in
    ``_build_bound_payload``.  The reference baseline mappings apply the
    same normalization (bmm kernels on HBM-PIM: 16x16 -> 4x4).
    """
    if not simd_lanes:
        return int(simd_k), int(simd_l)
    lanes = int(simd_lanes)
    k, l = int(simd_k), int(simd_l)
    while k * l > lanes:
        if k >= l and k % 2 == 0:
            k //= 2
        elif l % 2 == 0:
            l //= 2
        elif k % 2 == 0:
            k //= 2
        else:
            break  # odd factors on both sides: nothing halves cleanly
    return k, l


def _build_bound_payload(partition, simd_k, simd_l, mm_size) -> dict[int, list[dict[str, int]]]:
    m_total, k_total, l_total, b_total = (int(item) for item in mm_size)
    div_m, div_k, div_l, div_b = _split_factors(partition)
    m_after = math.ceil(m_total / div_m)
    k_after = math.ceil(k_total / div_k)
    l_after = math.ceil(l_total / div_l)
    b_after = math.ceil(b_total / div_b)

    bounds: dict[int, list[dict[str, int]]] = {}
    partition_level_count = len(partition)
    for index, level_partition in enumerate(partition):
        m_factor, k_factor, l_factor, b_factor = level_partition
        level = partition_level_count + 1 - index
        bounds[level] = _make_bound_level(n=b_factor, k=m_factor, p=l_factor, c=k_factor)
    bounds[1] = _make_bound_level(
        n=b_after, k=m_after,
        p=math.ceil(l_after / int(simd_l)), c=math.ceil(k_after / int(simd_k)),
    )
    bounds[0] = _make_bound_level(n=1, k=1, p=int(simd_l), c=int(simd_k))
    return bounds


def _build_problem_payload(bounds: dict[int, list[dict[str, int]]]) -> dict[str, int]:
    problem = {dimension: 1 for dimension in CANONICAL_DIMS}
    for level in sorted(bounds.keys(), reverse=True):
        for entry in bounds[level]:
            ((dimension, extent),) = entry.items()
            problem[dimension] *= int(extent)
    return problem


def dump_yaml(*, log_path, template_path, output_path, name, workload_size,
              design_kind="best", best_index=1, simd_lanes=None) -> Path:
    parsed_log = parse_compile_log(log_path)
    if design_kind == "baseline":
        selected_design = parsed_log["baseline"]
        if selected_design is None:
            raise ValueError(f"No baseline strategy in log: {log_path}")
    else:
        if best_index <= 0:
            raise ValueError("best_index must be >= 1")
        selected_design = parsed_log["best_designs"][best_index - 1]

    template_payload = yaml.safe_load(Path(template_path).read_text(encoding="utf-8"))
    if not isinstance(template_payload, dict):
        raise ValueError(f"Template YAML must be a mapping: {template_path}")

    internal_mm_size = _normalize_mm_workload_size(workload_size)
    _, _, partition, simd_k, _, simd_l, _ = selected_design
    simd_k, simd_l = _clamp_simd_split(simd_k, simd_l, simd_lanes)
    bound_payload = _build_bound_payload(partition, simd_k, simd_l, internal_mm_size)
    problem_payload = _build_problem_payload(bound_payload)

    template_problem = template_payload.get("Problem")
    if isinstance(template_problem, dict):
        for extra_key in ("Dilation", "Stride"):
            if extra_key in template_problem:
                problem_payload[extra_key] = template_problem[extra_key]

    output_payload = dict(template_payload)
    output_payload["Name"] = name
    output_payload["Problem"] = problem_payload
    output_payload["Bound"] = bound_payload

    text = yaml.safe_dump(output_payload, sort_keys=False, default_flow_style=False)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--template-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workload-size", nargs=4, type=int,
                        metavar=("M", "K", "N", "B"), required=True)
    parser.add_argument("--design", choices=("best", "baseline"), default="best")
    parser.add_argument("--best-index", type=int, default=1)
    parser.add_argument(
        "--simd-lanes", type=int, default=None,
        help="Physical SIMD lane count of the target PU; intra-PU splits "
             "larger than this are rebalanced to a lanes-legal split "
             "(see _clamp_simd_split).",
    )
    args = parser.parse_args()
    out = dump_yaml(
        log_path=args.log_path, template_path=args.template_path,
        output_path=args.output_path, name=args.name,
        workload_size=args.workload_size, design_kind=args.design,
        best_index=args.best_index, simd_lanes=args.simd_lanes,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
