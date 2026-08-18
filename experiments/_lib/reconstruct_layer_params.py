#!/usr/bin/env python3
"""Reconstruct nn_models/<model>/layer_params.csv at run time.

The artifact deliberately does not ship ``data/nn_models/`` — the end-to-end
workload manifests (``data/workloads/end_to_end/<model>/manifest.json``) embed
every original CSV row verbatim in their ``layers[].raw_row`` fields. The
end-to-end pipelines (Fig. 14 geometry sweep) hard-code their model dirs as
``<repo>/data/nn_models/<model>``, so this script both

1. writes ``<out>/data/nn_models/<model>/layer_params.csv`` for every model
   manifest found, and
2. (``--shadow-root`` mode) turns ``<out>`` into a *shadow repository root*:
   ``<out>/data/<child>`` symlinks back to the real repo data dirs,
   ``<out>/results`` symlinks to the results root, and ``<out>/pyproject.toml``
   to the real project file. Exporting ``PNMAX_ROOT=<out>`` then redirects only
   the missing ``data/nn_models`` lookups while everything else resolves to the
   real repository — no repository file is created or modified.

Usage:
    uv run python experiments/_lib/reconstruct_layer_params.py \
        --out <results>/fig14_geometry/shadow_root --shadow-root
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_WORKLOADS = REPO_ROOT / "data" / "workloads" / "end_to_end"


def reconstruct_model_csv(manifest_path: Path, dest_csv: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = manifest.get("layers", [])
    rows = []
    for layer in layers:
        raw_row = layer.get("raw_row")
        if not isinstance(raw_row, list) or not raw_row:
            raise ValueError(
                f"{manifest_path}: layer {layer.get('layer_index')} has no raw_row"
            )
        rows.append(",".join(str(item) for item in raw_row))
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    dest_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def _replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            raise SystemExit(f"refusing to replace real directory {link}")
        link.unlink()
    link.symlink_to(target)


def build_shadow_root(out: Path, results_root: Path) -> None:
    """Make <out> usable as PNMAX_ROOT (see module docstring)."""
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted((REPO_ROOT / "data").iterdir()):
        if child.name == "nn_models":
            continue
        _replace_symlink(data_dir / child.name, child)
    # repo_root()-anchored callers may also resolve results/ and the project
    # marker through PNMAX_ROOT.
    _replace_symlink(out / "pyproject.toml", REPO_ROOT / "pyproject.toml")
    _replace_symlink(out / "results", results_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output root (layer_params.csv goes to <out>/data/nn_models/<model>/).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model subset (default: every dir under data/workloads/end_to_end).",
    )
    parser.add_argument(
        "--shadow-root",
        action="store_true",
        help="Also symlink the real data/ children so <out> works as PNMAX_ROOT.",
    )
    args = parser.parse_args()

    models = args.models or sorted(
        p.name for p in E2E_WORKLOADS.iterdir() if (p / "manifest.json").is_file()
    )
    if not models:
        raise SystemExit(f"no model manifests under {E2E_WORKLOADS}")

    for model in models:
        manifest_path = E2E_WORKLOADS / model / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"missing manifest: {manifest_path}")
        dest = args.out / "data" / "nn_models" / model / "layer_params.csv"
        n = reconstruct_model_csv(manifest_path, dest)
        print(f"[layer-params] {model}: {n} rows -> {dest}")

    if args.shadow_root:
        results_root = Path(
            os.environ.get("PNMAX_RESULTS_ROOT", str(REPO_ROOT / "results"))
        )
        build_shadow_root(args.out, results_root)
        print(f"[layer-params] shadow root ready: export PNMAX_ROOT={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
