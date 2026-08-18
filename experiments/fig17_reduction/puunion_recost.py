#!/usr/bin/env python3
"""Fig 17: build the INCLUSIVE channel-level pool (PU-union re-cost).

Reduction-place inclusivity is an EVAL-side property (see
plot/reduction_place_pareto.py): the channel level pools the more-local bank
(config1 / PU) mappings re-cost under the channel_level architecture, unioned
with the channel-native (config3) pool from the inclusive evaluation run.
This script performs that union re-cost for one workload:

1. re-cost every config1 trace under ``channel_level.yaml``
   (``activation=True, channel_level=True``, constraints required);
2. union the successful rows with the inclusive run's config3 rows
   (dedup by ``(config_id, state_hash)``, config3 first);
3. write ``<out-root>/<token>/<timestamp>/evaluations.csv`` in the exact
   column set ``reduction_place_pareto.read_evaluations`` requires.

Ported from a one-off channel-inclusive re-cost script in the PNMAX
development repo, with parameterized paths and pnmax imports.
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from pnmax.database import Arch, Workload

# _state_hash must be the exact function the pareto-eval pipeline uses when it
# writes evaluations.csv (the PUunion re-cost joins on those state_hash values).
from pnmax.cli.run_workload_space_pareto_eval import _state_hash
from pnmax.dse.workload_eval import evaluate_workload_analytical_report

OUT_COLUMNS = (
    "config_id",
    "status",
    "state_hash",
    "latency_cycles",
    "mem_footprint_bytes",
    "energy",
)

_CHANNEL_ARCH: Arch | None = None
_CHANNEL_ARCH_FILE: str = ""


def _init_worker(channel_arch_file: str) -> None:
    global _CHANNEL_ARCH_FILE
    _CHANNEL_ARCH_FILE = channel_arch_file


def _get_channel_arch() -> Arch:
    global _CHANNEL_ARCH
    if _CHANNEL_ARCH is None:
        _CHANNEL_ARCH = Arch.from_yaml(Path(_CHANNEL_ARCH_FILE).read_text())
    return _CHANNEL_ARCH


def recost_trace(trace_path_str: str) -> dict | None:
    """Re-cost one config1 trace under the channel-level arch (picklable)."""
    arch = _get_channel_arch()
    try:
        workload = Workload.from_yaml(Path(trace_path_str).read_text())
    except Exception:
        return None
    try:
        rep = evaluate_workload_analytical_report(
            workload,
            arch,
            target="hbm_pim",
            latency_coeff=1,
            mem_footprint_coeff=1,
            energy_coeff=1,
            activation=True,
            channel_level=True,
            require_constraints=True,
        )
    except Exception:
        return None
    if rep is None:
        return None
    lat = rep.get("latency_cycles")
    mem = rep.get("mem_footprint_bytes")
    energy = rep.get("energy")
    if lat is None or mem is None or energy is None:
        return None
    try:
        state_hash = _state_hash(workload.to_dict())
    except Exception:
        return None
    return {
        "config_id": "config3",
        "status": "ok",
        "state_hash": state_hash,
        "latency_cycles": int(lat),
        "mem_footprint_bytes": int(mem),
        "energy": int(energy),
    }


def read_config3_rows(inclusive_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with inclusive_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("config_id", "").strip() != "config3":
                continue
            if row.get("status", "").strip() != "ok":
                continue
            rows.append(
                {
                    "config_id": "config3",
                    "status": "ok",
                    "state_hash": row.get("state_hash", "").strip(),
                    "latency_cycles": int(float(row["latency_cycles"])),
                    "mem_footprint_bytes": int(float(row["mem_footprint_bytes"])),
                    "energy": int(float(row["energy"])),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--token", required=True, help="Workload token (e.g. llama256-3).")
    parser.add_argument("--config1-trace-dir", required=True, type=Path)
    parser.add_argument("--inclusive-csv", required=True, type=Path)
    parser.add_argument("--channel-arch", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    traces = sorted(str(p) for p in args.config1_trace_dir.glob("trace_*.yaml"))
    if not traces:
        raise SystemExit(f"no config1 traces under {args.config1_trace_dir}")

    t0 = time.time()
    ok = failed = done = 0
    recost_rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(args.channel_arch),),
    ) as executor:
        futures = [executor.submit(recost_trace, trace) for trace in traces]
        for future in as_completed(futures):
            row = future.result()
            if row is None:
                failed += 1
            else:
                ok += 1
                recost_rows.append(row)
            done += 1
            if done % 256 == 0 or done == len(traces):
                print(
                    f"[{args.token}] PU-union re-cost {done}/{len(traces)} "
                    f"ok={ok} failed={failed} elapsed_s={time.time() - t0:.0f}",
                    flush=True,
                )
    # Deterministic union order: config3 pool first, then re-cost rows sorted
    # by state_hash (as_completed order is nondeterministic).
    recost_rows.sort(key=lambda row: row["state_hash"])
    config3_rows = read_config3_rows(args.inclusive_csv)
    seen: set[tuple[str, str]] = set()
    union_rows: list[dict] = []
    for row in [*config3_rows, *recost_rows]:
        key = (row["config_id"], row["state_hash"])
        if key in seen:
            continue
        seen.add(key)
        union_rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / args.token / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "evaluations.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(union_rows)
    print(
        f"[{args.token}] union pool: config3={len(config3_rows)} "
        f"recost_ok={ok} recost_failed={failed} total={len(union_rows)} -> {out_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
