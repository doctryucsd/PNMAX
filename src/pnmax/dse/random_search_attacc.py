from __future__ import annotations

import copy
import hashlib
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from pnmax.paths import repo_root, results_root
MAX_PROBLEM_SIZE = 524_288
DEFAULT_BASE_WORKLOAD = repo_root() / "data" / "workloads" / "samples" / "attacc_gemv.yaml"
DEFAULT_OUTPUT_DIR = results_root() / "validation_traces" / "attacc"
DEFAULT_K1_CANDIDATES: tuple[int, ...] = tuple(2**idx for idx in range(12))


@dataclass(frozen=True)
class AttaccRandomSearchConfig:
    workload_file: Path | str = DEFAULT_BASE_WORKLOAD
    out_dir: Path | str = DEFAULT_OUTPUT_DIR
    num_traces: int = 100
    workers: int = 1
    seed: int | None = None
    dry_run: bool = False
    k1_candidates: Sequence[int] = DEFAULT_K1_CANDIDATES


@dataclass(frozen=True)
class AttaccRandomSearchResult:
    output_dir: Path
    workload_file: Path
    seed: int
    requested: int
    unique_limit: int
    effective: int
    generated: int
    capped: bool
    dry_run: bool
    removed_existing: int
    generated_files: tuple[Path, ...]


def load_attacc_workload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected mapping payload in {path}, got {type(payload).__name__}."
        )
    return payload


def _flatten_bound_level(entry: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for dim_dict in entry:
        if not isinstance(dim_dict, Mapping):
            raise ValueError(
                f"Bound level entries must be mappings, got {type(dim_dict).__name__}."
            )
        for key, value in dim_dict.items():
            merged[str(key)] = int(value)
    return merged


def _resolve_level1_entry(
    bound: Mapping[Any, Any],
) -> tuple[int | str, Sequence[Mapping[str, Any]]]:
    if 1 in bound:
        level1 = bound[1]
        level1_key: int | str = 1
    elif "1" in bound:
        level1 = bound["1"]
        level1_key = "1"
    else:
        raise KeyError("Bound level 1 not found in workload config.")

    if not isinstance(level1, Sequence):
        raise ValueError(
            f"Bound level 1 must be a sequence, got {type(level1).__name__}."
        )
    return level1_key, level1


def compute_problem_kc_from_bounds(cfg: Mapping[str, Any]) -> tuple[int, int]:
    bound = cfg.get("Bound")
    if not isinstance(bound, Mapping):
        raise ValueError("Workload must contain a Bound mapping.")

    prod_k = 1
    prod_c = 1
    for entry in bound.values():
        if not isinstance(entry, Sequence):
            raise ValueError("Each Bound level entry must be a sequence.")
        flattened = _flatten_bound_level(entry)
        prod_k *= int(flattened.get("K", 1))
        prod_c *= int(flattened.get("C", 1))
    return prod_k, prod_c


def adjust_level1_kc(
    cfg: Mapping[str, Any], new_k1: int, new_c1: int
) -> dict[str, Any]:
    if new_k1 <= 0 or new_c1 <= 0:
        raise ValueError(f"new_k1 and new_c1 must be positive, got {new_k1}, {new_c1}.")

    new_cfg = copy.deepcopy(dict(cfg))
    bound = new_cfg.get("Bound")
    if not isinstance(bound, Mapping):
        raise ValueError("Workload must contain a Bound mapping.")

    level1_key, level1 = _resolve_level1_entry(bound)
    mutable_level1: list[dict[str, int]] = []
    for dim_dict in level1:
        if not isinstance(dim_dict, Mapping):
            raise ValueError("Bound level entries must be mappings.")
        updated = {str(key): int(value) for key, value in dim_dict.items()}
        if "K" in updated:
            updated["K"] = int(new_k1)
        if "C" in updated:
            updated["C"] = int(new_c1)
        mutable_level1.append(updated)

    bound[level1_key] = mutable_level1

    problem = new_cfg.get("Problem")
    if not isinstance(problem, Mapping):
        raise ValueError("Workload must contain a Problem mapping.")

    problem_k, problem_c = compute_problem_kc_from_bounds(new_cfg)
    problem["K"] = int(problem_k)
    problem["C"] = int(problem_c)
    new_cfg["Problem"] = problem

    return new_cfg


def _extract_fixed_kc_factors(cfg: Mapping[str, Any]) -> tuple[int, int]:
    orig_k, orig_c = compute_problem_kc_from_bounds(cfg)

    bound = cfg.get("Bound")
    if not isinstance(bound, Mapping):
        raise ValueError("Workload must contain a Bound mapping.")

    _, level1 = _resolve_level1_entry(bound)
    flattened_l1 = _flatten_bound_level(level1)
    k1_orig = int(flattened_l1.get("K", 1))
    c1_orig = int(flattened_l1.get("C", 1))

    if k1_orig <= 0 or c1_orig <= 0:
        raise ValueError(f"Level-1 K/C must be positive, got K={k1_orig}, C={c1_orig}.")

    if orig_k % k1_orig != 0 or orig_c % c1_orig != 0:
        raise ValueError(
            "Inconsistent factorization between Problem and Bound level 1: "
            f"Problem.K={orig_k}, K1={k1_orig}, Problem.C={orig_c}, C1={c1_orig}."
        )

    return orig_k // k1_orig, orig_c // c1_orig


def enumerate_feasible_level1_pairs(
    cfg: Mapping[str, Any],
    *,
    max_problem_size: int = MAX_PROBLEM_SIZE,
    k1_candidates: Sequence[int] = DEFAULT_K1_CANDIDATES,
) -> list[tuple[int, int]]:
    if max_problem_size <= 1:
        raise ValueError(f"max_problem_size must be > 1, got {max_problem_size}.")

    k_other, c_other = _extract_fixed_kc_factors(cfg)
    denom = k_other * c_other
    if denom <= 0:
        raise ValueError(f"Non-positive K/C fixed denominator: {denom}.")

    pairs: list[tuple[int, int]] = []
    for k1 in sorted({int(candidate) for candidate in k1_candidates}):
        if k1 <= 0:
            continue
        max_c1 = (max_problem_size - 1) // (denom * k1)
        if max_c1 < 1:
            continue
        for c1 in range(1, max_c1 + 1):
            pairs.append((k1, c1))

    return pairs


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _trace_output_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"trace_{index:06d}.yaml"


def _clear_existing_traces(out_dir: Path) -> int:
    removed = 0
    for existing in out_dir.glob("trace_*.yaml"):
        existing.unlink()
        removed += 1
    return removed


def _split_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total == 0:
        return []

    base = total // parts
    remainder = total % parts
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(parts):
        count = base + (1 if idx < remainder else 0)
        if count <= 0:
            continue
        ranges.append((start, count))
        start += count
    return ranges


def _write_trace_chunk(
    base_cfg: Mapping[str, Any],
    selected_pairs: Sequence[tuple[int, int]],
    out_dir: Path,
    start_idx: int,
) -> tuple[Path, ...]:
    written: list[Path] = []
    for local_idx, (k1, c1) in enumerate(selected_pairs):
        payload = adjust_level1_kc(base_cfg, k1, c1)
        out_path = _trace_output_path(out_dir, start_idx + local_idx)
        yaml_text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        out_path.write_text(yaml_text, encoding="utf-8")
        written.append(out_path)
    return tuple(written)


def run_attacc_random_search(
    config: AttaccRandomSearchConfig,
) -> AttaccRandomSearchResult:
    workload_path = Path(config.workload_file)
    out_dir = Path(config.out_dir)
    requested = int(config.num_traces)
    workers = max(1, int(config.workers))

    if requested < 0:
        raise ValueError(f"num_traces must be >= 0, got {requested}.")

    base_cfg = load_attacc_workload(workload_path)
    feasible_pairs = enumerate_feasible_level1_pairs(
        base_cfg,
        max_problem_size=MAX_PROBLEM_SIZE,
        k1_candidates=config.k1_candidates,
    )

    unique_limit = len(feasible_pairs)
    effective = min(requested, unique_limit)
    capped = requested > unique_limit

    # Fallback seed derives from the workload FILENAME (not its absolute path)
    # so seedless library calls stay machine- and checkout-independent.
    seed = (
        int(config.seed)
        if config.seed is not None
        else _stable_seed("attacc-random-search", workload_path.name)
    )

    rng = random.Random(seed)
    rng.shuffle(feasible_pairs)
    selected = feasible_pairs[:effective]

    out_dir.mkdir(parents=True, exist_ok=True)
    removed_existing = 0
    generated_files: tuple[Path, ...] = ()

    if not config.dry_run:
        removed_existing = _clear_existing_traces(out_dir)

        if effective > 0:
            if workers <= 1:
                generated_files = _write_trace_chunk(base_cfg, selected, out_dir, 0)
            else:
                ranges = _split_ranges(effective, min(workers, effective))
                ctx = mp.get_context("spawn")
                try:
                    generated: list[Path] = []
                    with ProcessPoolExecutor(
                        max_workers=len(ranges), mp_context=ctx
                    ) as executor:
                        futures = [
                            executor.submit(
                                _write_trace_chunk,
                                base_cfg,
                                selected[start : start + count],
                                out_dir,
                                start,
                            )
                            for start, count in ranges
                        ]
                        for future in futures:
                            generated.extend(future.result())
                    generated_files = tuple(sorted(generated))
                except (PermissionError, OSError):
                    generated_files = _write_trace_chunk(base_cfg, selected, out_dir, 0)

    return AttaccRandomSearchResult(
        output_dir=out_dir,
        workload_file=workload_path,
        seed=seed,
        requested=requested,
        unique_limit=unique_limit,
        effective=effective,
        generated=effective if not config.dry_run else 0,
        capped=capped,
        dry_run=bool(config.dry_run),
        removed_existing=removed_existing,
        generated_files=generated_files,
    )
