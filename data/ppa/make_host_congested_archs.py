#!/usr/bin/env python3
"""Emit congested-host variants of arch YAMLs (bank_to_bank_transfer.latency x16).

The congested-host transform models a 16x-slower host bank-to-bank
interconnect: it multiplies ONLY ``unit_costs.bank_to_bank_transfer.latency``
by 16 and leaves everything else (energy, geometry, all other costs, comments)
byte-for-byte unchanged, then prepends a provenance header comment. This
matches the ``data/archs/lowered/geometry_sweep_host_congested/``
convention (whose files carry exactly this header).

The transform is done as a text edit (not a YAML round-trip) so the generator
header comments and original formatting are preserved.

Usage (uv run python, no bare python on PATH):

    # single file -> single file
    uv run python data/ppa/make_host_congested_archs.py IN.yaml OUT.yaml

    # single file -> directory (keeps the input filename)
    uv run python data/ppa/make_host_congested_archs.py IN.yaml OUTDIR/

    # directory -> directory (recurses *.yaml, mirrors the tree)
    uv run python data/ppa/make_host_congested_archs.py INDIR/ OUTDIR/

Options:
    --factor N      multiplier for the b2b latency (default: 16)
    --glob PATTERN  glob used when INPUT is a directory (default: *.yaml)
    --dry-run       report what would be written without writing
    --force         overwrite existing outputs (default: refuse)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Two layouts occur in this repo's arch YAMLs:
#  (1) BLOCK style (generated archs):  bank_to_bank_transfer:\n    latency: 72
#  (2) FLOW style (hand-written baselines):
#        bank_to_bank_transfer: { latency: 630, energy: 916.642 }
_B2B_KEY_RE = re.compile(r"^(\s*)bank_to_bank_transfer\s*:\s*$")
_LATENCY_RE = re.compile(r"^(\s*)latency\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$")
# Flow-style: capture the latency number embedded in the inline mapping.
_B2B_FLOW_RE = re.compile(
    r"^(\s*bank_to_bank_transfer\s*:\s*\{[^}]*?latency\s*:\s*)"
    r"([0-9]+(?:\.[0-9]+)?)"
    r"([^}]*\}\s*)$"
)


class TransformError(RuntimeError):
    """Raised when the b2b latency line cannot be located/transformed."""


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _source_label(src: Path) -> str:
    """Repo-relative label for the provenance header (machine-independent)."""
    try:
        return src.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return src.as_posix()


def _format_number(old_text: str, new_value: float) -> str:
    """Render new_value, keeping an integer look when the source was integral."""
    if "." not in old_text and float(new_value).is_integer():
        return str(int(round(new_value)))
    # Preserve a reasonable float rendering (trim trailing zeros).
    return ("%f" % new_value).rstrip("0").rstrip(".")


def transform_text(text: str, *, factor: int, source_label: str) -> str:
    lines = text.splitlines(keepends=True)
    in_b2b_block = False
    b2b_indent = ""
    transformed: list[str] = []
    old_value_text: str | None = None
    new_value_text: str | None = None

    for line in lines:
        if not in_b2b_block:
            # Flow-style: the latency lives on the bank_to_bank_transfer line itself.
            flow_match = _B2B_FLOW_RE.match(line.rstrip("\n"))
            if flow_match and old_value_text is None:
                newline = "\n" if line.endswith("\n") else ""
                old_value_text = flow_match.group(2)
                new_value = float(old_value_text) * factor
                new_value_text = _format_number(old_value_text, new_value)
                transformed.append(
                    flow_match.group(1) + new_value_text + flow_match.group(3) + newline
                )
                continue
            match = _B2B_KEY_RE.match(line)
            if match:
                in_b2b_block = True
                b2b_indent = match.group(1)
            transformed.append(line)
            continue

        # Inside the bank_to_bank_transfer block: the latency line is the target.
        latency_match = _LATENCY_RE.match(line)
        if latency_match and old_value_text is None:
            indent = latency_match.group(1)
            # The latency line must be more-indented than the b2b key line.
            if len(indent) > len(b2b_indent):
                old_value_text = latency_match.group(2)
                new_value = float(old_value_text) * factor
                new_value_text = _format_number(old_value_text, new_value)
                transformed.append(f"{indent}latency: {new_value_text}\n")
                in_b2b_block = False
                continue
        # A dedent back to (or above) the b2b key indent ends the block.
        stripped = line.strip()
        if stripped and not line.startswith(b2b_indent + " "):
            in_b2b_block = False
        transformed.append(line)

    if old_value_text is None or new_value_text is None:
        raise TransformError(
            "could not locate unit_costs.bank_to_bank_transfer.latency"
        )

    header = (
        f"# Derived from {source_label} with bank_to_bank_transfer.latency "
        f"x{factor} ({old_value_text} -> {new_value_text}); energy unchanged.\n"
    )
    return header + "".join(transformed)


def _iter_inputs(input_path: Path, glob: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob(glob))
    raise FileNotFoundError(f"input not found: {input_path}")


def _resolve_output(
    input_path: Path, output_path: Path, src: Path
) -> Path:
    if input_path.is_file():
        # OUT may be a file or a directory.
        if output_path.is_dir() or str(output_path).endswith("/"):
            return output_path / src.name
        return output_path
    # Directory mode: mirror the relative tree under OUT.
    rel = src.relative_to(input_path)
    return output_path / rel


def run(
    *,
    input_path: Path,
    output_path: Path,
    factor: int,
    glob: str,
    dry_run: bool,
    force: bool,
) -> int:
    sources = _iter_inputs(input_path, glob)
    if not sources:
        print(f"no inputs matched under {input_path} (glob {glob})", file=sys.stderr)
        return 1

    written = 0
    for src in sources:
        out = _resolve_output(input_path, output_path, src)
        source_label = _source_label(src)
        try:
            new_text = transform_text(
                src.read_text(encoding="utf-8"),
                factor=factor,
                source_label=source_label,
            )
        except TransformError as exc:
            print(f"SKIP {src}: {exc}", file=sys.stderr)
            return 2
        if out.exists() and not force and not dry_run:
            print(f"refusing to overwrite {out} (use --force)", file=sys.stderr)
            return 3
        if dry_run:
            print(f"[dry-run] {src} -> {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(new_text, encoding="utf-8")
        print(f"wrote {out}")
        written += 1

    if not dry_run:
        print(f"done: {written} file(s) written (factor x{factor}).")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="input arch YAML file or directory")
    parser.add_argument("output", type=Path, help="output file or directory")
    parser.add_argument("--factor", type=int, default=16, help="b2b latency multiplier (default 16)")
    parser.add_argument("--glob", type=str, default="*.yaml", help="glob for directory inputs (default *.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.factor <= 0:
        print("--factor must be positive", file=sys.stderr)
        return 1
    return run(
        input_path=args.input,
        output_path=args.output,
        factor=args.factor,
        glob=args.glob,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
