"""gpam-info — human-readable summary of a GPAM architecture spec.

Usage:
    gpam-info --arch data/archs/baseline/hbm_pim.yaml [--corner ss] [--verbose]
    (equivalently: python -m pnmax.gpam.cli --arch ...)

Prints node counts by kind, the hierarchy-depth table, edge counts by
link class, and a few resolved (tech-DB-merged, corner-scaled) attributes.
"""

import argparse
import sys
from collections import Counter

from ruamel.yaml.error import YAMLError

from pnmax.gpam.graph import NodeKind
from pnmax.gpam.loader import load_arch

# node attributes that are bookkeeping, not architecture parameters
_SKIP_ATTRS = ("kind", "hier_id", "hier_depth", "parent")
_MAX_ATTRS_SHOWN = 6


def _summarize(arch, arch_path: str, corner) -> None:
    g = arch.g

    print(f"architecture : {arch.meta.get('name', '?')}")
    print(f"spec file    : {arch_path}")
    print(f"corner       : {corner if corner else '<nominal>'}")
    if "clock" in arch.meta:
        print(f"clock        : {arch._fmt(arch.meta['clock'])}")
    if "comment" in arch.meta:
        print(f"comment      : {arch.meta['comment']}")
    print()

    # ── nodes by kind ────────────────────────────────────────────────────
    print(f"nodes ({g.number_of_nodes()} total)")
    for kind in NodeKind:
        nids = arch.node_ids_of_kind(kind)
        if nids:
            preview = ", ".join(nids[:3]) + (", ..." if len(nids) > 3 else "")
            print(f"  {kind.value:<8} {len(nids):>5}   [{preview}]")
    print()

    # ── hierarchy ────────────────────────────────────────────────────────
    depths = arch.hier_depths
    print(f"hierarchy (max depth {max(depths)})")
    for depth in sorted(depths):
        kinds = ", ".join(k.value for k in depths[depth])
        count = len(arch.node_ids_of_kind_at_depth(depth))
        print(f"  depth {depth}: {count:>5} node(s)   kinds: {kinds}")
    print()

    # ── edges by link class ──────────────────────────────────────────────
    by_link = Counter(link for _, _, link in g.edges(data="link"))
    print(f"edges ({g.number_of_edges()} total, by link class)")
    for link_cls, count in sorted(by_link.items(), key=lambda kv: str(kv[0])):
        print(f"  {str(link_cls):<16} {count:>5}")
    print()

    # ── sample resolved attributes (first node of each kind) ────────────
    print("resolved attributes (first node of each kind)")
    for kind in NodeKind:
        nids = arch.node_ids_of_kind(kind)
        if not nids:
            continue
        nid = nids[0]
        attrs = {k: v for k, v in g.nodes[nid].items() if k not in _SKIP_ATTRS}
        shown = list(attrs.items())[:_MAX_ATTRS_SHOWN]
        pretty = ", ".join(f"{k}={arch._fmt(v)}" for k, v in shown)
        more = "" if len(attrs) <= _MAX_ATTRS_SHOWN else ", ..."
        print(f"  {nid:<10} ({kind.value}): {pretty}{more}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="gpam-info",
        description="Summarize a GPAM architecture spec (topology YAML + "
                    "its referenced tech-DB).")
    parser.add_argument("--arch", required=True,
                        help="path to the GPAM topology YAML; its 'tech:' "
                             "reference is resolved relative to that file")
    parser.add_argument("--corner", default=None,
                        help="PVT corner name from the tech-DB "
                             "(e.g. ss, ff, slow_900mV)")
    parser.add_argument("--verbose", action="store_true",
                        help="additionally dump every node and edge")
    args = parser.parse_args(argv)

    try:
        arch = load_arch(args.arch, corner=args.corner)
    except (OSError, ValueError, KeyError, YAMLError) as exc:
        # ruamel parse errors (malformed YAML) are not ValueErrors —
        # report them as a clean one-line message instead of a traceback.
        message = " ".join(str(exc).split())
        print(f"gpam-info: error: {message}", file=sys.stderr)
        return 2

    _summarize(arch, args.arch, args.corner)
    if args.verbose:
        print()
        arch.print_nodes(verbose=True)
        arch.print_edges(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
