"""gpam-lower — lower a GPAM graph spec to the flat PNMAX architecture YAML.

Turns a GPAM topology YAML (+ referenced tech-DB), parsed by the vendored
:mod:`pnmax.gpam.loader`, into the flat single-file architecture YAML consumed
by the strict runtime parser (:mod:`pnmax.database.arch`,
``data/archs/lowered/FORMAT.md``).

The emission grammar mirrors data/ppa/generate_arch_from_template.py
(``_size_to_text``, ``_format_freq_mhz``, ``_round_cost``).

Key semantics:

* ``meta.clock`` is the PU clock; cycle-valued latencies are read as PU cycles
  directly (``q.to("cycle").magnitude``), never round-tripped through ns.
  Time-valued latencies are converted ``x clock`` and reported via notes.
* The modeled transaction is one SIMD vector load:
  ``transaction_bytes = simd_lanes * 16 / 8``.  Per-byte link energies are
  charged per event over that transaction (32 B for the baselines); the host
  link is charged per 64-bit column (8 B).
* Transfer aggregates are composed by SUMMING per-edge hop costs along
  explicit paths, so the placement of an aggregate on either hop of a
  MEM<->BUF<->PU path is equivalence-neutral.  ``bandwidth: INF`` links
  transfer for free; the store path must be authored as explicit reverse
  edges — a missing store path is an error, never assumed symmetric.
* Hierarchy levels come from the PU/MEM ``hier_id`` path segments, leaf = L2
  upward.  BUF/HOST/BUS/XBAR segments never form levels; a segment token
  (trailing digits stripped) must be a legal flat-parser level kind.  The
  leaf segment falls back to the node kind (PU -> ``pu``, MEM -> ``bank``)
  when its token is not itself a legal kind (e.g. upmem's ``dpu3``).  If PUs
  are 1:1-paired with leaf MEMs at the same depth (attacc), the leaf level
  kind is taken from the MEM segment.
* The bank<->bank path must resolve to a SINGLE cost signature: a direct
  MEM->MEM edge, or a MEM->hub->MEM path through a dedicated bus/xbar/router
  hub. Its absence (like a missing BUF node) makes the flat schema
  unsatisfiable and is reported as E_SCHEMA; two disagreeing signatures are
  E_TOPO_AMBIGUOUS_PATH. Routing bank-to-bank through the HOST node is NOT a
  supported topology: the host link is reserved for and modeled as the
  per-64-bit-column host_pu_transfer path (``_host_link`` requires the host's
  out-edges to be uniform), so the baseline architectures route bank-to-bank
  via the device-root bus, never via the host.

Error taxonomy (``LowerError.code``): E_TOPO_NO_HOST / E_TOPO_NO_PU /
E_TOPO_NO_MEM / E_TOPO_NO_LOAD_PATH / E_TOPO_NO_STORE_PATH / E_TOPO_RAGGED /
E_TOPO_AMBIGUOUS_PATH / E_ATTR_MISSING / E_ATTR_RAW_STRING / E_UNIT /
E_KIND_UNKNOWN / E_SCHEMA.

CLI::

    gpam-lower --arch SPEC [--corner NAME] [--out PATH] [--name ID]
               [--check REFERENCE.yaml] [--print] [--verbose]

Exit codes: 0 ok / 2 load error / 3 lower error / 4 check mismatch.
``--check`` treats a missing or unreadable reference as a mismatch (exit 4),
not a load error. ``--out`` is written whenever lowering itself succeeds —
including when a concurrent ``--check`` reports a mismatch — so the freshly
lowered file is always produced for inspection.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import networkx as nx
import yaml
from pint.errors import DimensionalityError
from ruamel.yaml.error import YAMLError

from pnmax.database.arch import Arch, ArchParseError, HierLevel
from pnmax.gpam.graph import ArchGraph, NodeKind, ureg
from pnmax.gpam.loader import load_arch

__all__ = ["LowerError", "lower_arch", "lower_file", "main"]

# Hierarchy level kinds accepted by the strict flat parser (arch.py:28-38).
_LEVEL_KINDS = frozenset(level.value for level in HierLevel)

# Node kinds whose hier segments never form hierarchy levels.
_NON_LEVEL_KINDS = frozenset(
    {NodeKind.BUS, NodeKind.XBAR, NodeKind.HOST, NodeKind.RTR, NodeKind.BUF}
)

# Hub kinds a bank<->bank transfer may be routed through.
_B2B_HUB_KINDS = frozenset({NodeKind.BUS, NodeKind.XBAR, NodeKind.HOST, NodeKind.RTR})

_INF_SENTINELS = {"INF", "INFINITY"}


class LowerError(ValueError):
    """Raised when a GPAM graph cannot be lowered to the flat schema.

    ``code`` is a stable, machine-readable error class (see module docstring
    for the full taxonomy).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ──────────────────────────────────────────────────────────────────────────
#  Emission grammar (mirrors data/ppa/generate_arch_from_template.py)
# ──────────────────────────────────────────────────────────────────────────


def _size_to_text(size_bytes: int) -> str:
    # Mirror of generate_arch_from_template.py:160-165 (1024-based).
    if size_bytes % (1024**2) == 0:
        return f"{size_bytes // (1024**2)}MB"
    if size_bytes % 1024 == 0:
        return f"{size_bytes // 1024}KB"
    return f"{size_bytes}B"


def _format_freq_mhz(mhz: float) -> str:
    # Same rendering as generate_arch_from_template.py:172-173 for the values
    # in play (300 -> "300Mhz"); :g drops a trailing ".0" like _format_decimal.
    return f"{mhz:g}Mhz"


def _round_cost(value: float) -> float:
    # Mirror of generate_arch_from_template.py:176-177 (3-dp round).
    return float(f"{float(value):.3f}")


def _round_area(value: float) -> float:
    # Mirror of generate_arch_from_template.py:180-181 (6-dp round).
    return float(f"{float(value):.6f}")


def _bandwidth_to_text(bps: float) -> str:
    """Render bytes/s in the flat 1024-based grammar (670MB/s, 1.4GB/s)."""
    units = (("GB/s", 1024**3), ("MB/s", 1024**2), ("KB/s", 1024), ("B/s", 1))
    for unit, factor in units:
        value = bps / factor
        if value >= 1 and float(value).is_integer():
            return f"{int(value)}{unit}"
    for unit, factor in units:
        value = bps / factor
        if value >= 1:
            return f"{value:g}{unit}"
    return f"{bps:g}B/s"


# ──────────────────────────────────────────────────────────────────────────
#  Attribute conversion (validated by attribute NAME, never dimensionality —
#  pint's byte reduces to dimensionless)
# ──────────────────────────────────────────────────────────────────────────


def _reject_raw_string(value: Any, where: str) -> None:
    # Unparseable pint tokens (e.g. "1KB") silently survive tech-DB
    # parsing as raw strings; hard-error instead of passing garbage through.
    if isinstance(value, str):
        raise LowerError(
            "E_ATTR_RAW_STRING",
            f"{where}: value {value!r} survived tech-DB parsing as a raw "
            f"string (unparseable pint token); use pint units such as "
            f"KiB/MiB/GiB(/s), cycle, ns, pJ",
        )


def _as_int(value: Any, where: str) -> int:
    _reject_raw_string(value, where)
    if isinstance(value, bool):
        raise LowerError("E_UNIT", f"{where}: expected an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise LowerError("E_UNIT", f"{where}: expected an integer, got {value!r}")
        return int(value)
    if isinstance(value, ureg.Quantity):
        if not value.dimensionless:
            raise LowerError(
                "E_UNIT", f"{where}: expected a bare integer, got {value!r}"
            )
        magnitude = float(value.magnitude)
        if not magnitude.is_integer():
            raise LowerError("E_UNIT", f"{where}: expected an integer, got {value!r}")
        return int(magnitude)
    raise LowerError("E_UNIT", f"{where}: expected an integer, got {value!r}")


def _as_bytes(value: Any, where: str) -> int:
    _reject_raw_string(value, where)
    if isinstance(value, bool):
        raise LowerError("E_UNIT", f"{where}: expected a size, got {value!r}")
    if isinstance(value, (int, float)):
        byte_count = float(value)  # bare number = bytes
    elif isinstance(value, ureg.Quantity):
        try:
            byte_count = float(value.to("byte").magnitude)
        except DimensionalityError as exc:
            raise LowerError(
                "E_UNIT", f"{where}: not convertible to bytes: {value!r}"
            ) from exc
    else:
        raise LowerError("E_UNIT", f"{where}: expected a size, got {value!r}")
    if byte_count < 0 or abs(byte_count - round(byte_count)) > 1e-6:
        raise LowerError(
            "E_UNIT", f"{where}: size must be a non-negative whole byte count, got {value!r}"
        )
    return int(round(byte_count))


def _as_cycles(value: Any, where: str, clock: Any, notes: list[str]) -> float:
    """Latency attribute -> PU cycles (cycle magnitudes read directly)."""
    _reject_raw_string(value, where)
    if isinstance(value, bool):
        raise LowerError("E_UNIT", f"{where}: expected a latency, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)  # bare number = cycles (graph._as_time convention)
    if isinstance(value, ureg.Quantity):
        if value.dimensionless:
            return float(value.to("cycle").magnitude)
        try:
            cycles = float((value * clock).to("cycle").magnitude)
        except DimensionalityError as exc:
            raise LowerError(
                "E_UNIT", f"{where}: latency must be cycles or a time, got {value!r}"
            ) from exc
        notes.append(
            f"{where}: time-valued latency {value} converted to {cycles:g} PU "
            f"cycles via meta.clock"
        )
        return cycles
    raise LowerError("E_UNIT", f"{where}: expected a latency, got {value!r}")


def _as_pj(value: Any, where: str) -> float:
    _reject_raw_string(value, where)
    if isinstance(value, bool):
        raise LowerError("E_UNIT", f"{where}: expected an energy, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)  # bare number = pJ
    if isinstance(value, ureg.Quantity):
        try:
            return float(value.to("pJ").magnitude)
        except DimensionalityError as exc:
            raise LowerError(
                "E_UNIT", f"{where}: not convertible to pJ: {value!r}"
            ) from exc
    raise LowerError("E_UNIT", f"{where}: expected an energy, got {value!r}")


def _node_attr(graph: ArchGraph, nid: str, attr: str) -> Any:
    data = graph.g.nodes[nid]
    if attr not in data:
        raise LowerError(
            "E_ATTR_MISSING", f"node '{nid}': required attribute '{attr}' is missing"
        )
    return data[attr]


def _uniform_attr(
    graph: ArchGraph,
    nids: list[str],
    attr: str,
    conv,
    *,
    role: str,
    required: bool = True,
):
    """Convert ``attr`` on every node in ``nids``; require one uniform value.

    ``required=False``: absent everywhere -> None; present on only a strict
    subset -> E_TOPO_RAGGED.
    """
    present: dict[str, Any] = {}
    missing: list[str] = []
    for nid in nids:
        data = graph.g.nodes[nid]
        if attr in data:
            present[nid] = conv(data[attr], f"{role} node '{nid}' attribute '{attr}'")
        else:
            missing.append(nid)
    if not present:
        if required:
            raise LowerError(
                "E_ATTR_MISSING",
                f"{role} node '{missing[0]}': required attribute '{attr}' is missing",
            )
        return None
    if missing:
        if required:
            raise LowerError(
                "E_ATTR_MISSING",
                f"{role} node '{missing[0]}': required attribute '{attr}' is missing",
            )
        raise LowerError(
            "E_TOPO_RAGGED",
            f"attribute '{attr}' is present on {len(present)} {role} node(s) but "
            f"missing on {len(missing)} (e.g. '{missing[0]}')",
        )
    distinct = set(present.values())
    if len(distinct) > 1:
        shown = ", ".join(sorted(repr(v) for v in distinct)[:4])
        raise LowerError(
            "E_TOPO_RAGGED",
            f"attribute '{attr}' is not uniform across {role} nodes: {shown}",
        )
    return next(iter(present.values()))


# ──────────────────────────────────────────────────────────────────────────
#  Per-edge hop costs (replicates ArchGraph.hop_cost()'s INF and per-byte
#  branches, but composes in PU cycles / pJ so cycle-valued startup
#  latencies never round-trip through wall-clock time)
# ──────────────────────────────────────────────────────────────────────────


def _edge_latency_cycles(
    data: dict, where: str, nbytes_for_bw: int | None, clock: Any, notes: list[str]
) -> float:
    t0 = data.get("t0")
    latency = 0.0 if t0 is None else _as_cycles(t0, f"{where} startup_latency", clock, notes)
    if nbytes_for_bw is None:
        return latency
    bw = data.get("bw")
    if bw is None:
        raise LowerError("E_ATTR_MISSING", f"{where}: link has no bandwidth")
    if isinstance(bw, str):
        if bw.strip().upper() in _INF_SENTINELS:
            return latency  # INF bandwidth = free hop, bytes/bw contributes 0
        raise LowerError("E_ATTR_RAW_STRING", f"{where}: unparsable bandwidth {bw!r}")
    if not isinstance(bw, ureg.Quantity):
        raise LowerError(
            "E_UNIT",
            f"{where}: bandwidth must carry pint units (e.g. MiB/s) or be INF, got {bw!r}",
        )
    try:
        latency += float((nbytes_for_bw * ureg.byte / bw * clock).to("cycle").magnitude)
    except DimensionalityError as exc:
        raise LowerError("E_UNIT", f"{where}: bad bandwidth units: {bw!r}") from exc
    return latency


def _edge_energy_pj(data: dict, where: str, nbytes: int) -> float:
    e_B = data.get("e_B")
    if e_B is None:
        return 0.0
    _reject_raw_string(e_B, f"{where} energy_per_byte")
    if isinstance(e_B, bool):
        raise LowerError("E_UNIT", f"{where}: bad energy_per_byte {e_B!r}")
    if isinstance(e_B, (int, float)):
        return float(e_B) * nbytes  # bare number = pJ per byte by convention
    if not isinstance(e_B, ureg.Quantity):
        raise LowerError("E_UNIT", f"{where}: bad energy_per_byte {e_B!r}")
    per_byte_spelled = any(
        ("byte" in unit or "bit" in unit) and exp < 0
        for unit, exp in e_B.units._units.items()
    )
    try:
        if per_byte_spelled:  # spelled "pJ/byte"
            return float((nbytes * ureg.byte * e_B).to("pJ").magnitude)
        # plain energy ("0.8 pJ"), per byte by convention (graph.py:225-234)
        return float(e_B.to("pJ").magnitude) * nbytes
    except DimensionalityError as exc:
        raise LowerError("E_UNIT", f"{where}: bad energy_per_byte units {e_B!r}") from exc


def _edge_hop_cost(
    graph: ArchGraph, u: str, v: str, nbytes: int, clock: Any, notes: list[str]
) -> tuple[float, float]:
    data = graph.g[u][v][0]
    where = f"edge {u} -> {v}"
    return (
        _edge_latency_cycles(data, where, nbytes, clock, notes),
        _edge_energy_pj(data, where, nbytes),
    )


def _edge_link(graph: ArchGraph, u: str, v: str) -> str:
    return str(graph.g[u][v][0].get("link"))


# ──────────────────────────────────────────────────────────────────────────
#  Path discovery (load / store / host / bank<->bank)
# ──────────────────────────────────────────────────────────────────────────


def _load_paths(graph: ArchGraph, pu: str) -> list[list[tuple[str, str]]]:
    """All MEM ->(BUF ->)* pu simple paths; intermediates must be BUF nodes."""
    g = graph.g
    paths: list[list[tuple[str, str]]] = []

    def _backward(node: str, suffix: list[tuple[str, str]], seen: frozenset) -> None:
        for pred in g.predecessors(node):
            if pred in seen:
                continue
            kind = g.nodes[pred]["kind"]
            if kind is NodeKind.MEM:
                paths.append([(pred, node)] + suffix)
            elif kind is NodeKind.BUF:
                _backward(pred, [(pred, node)] + suffix, seen | {pred})

    _backward(pu, [], frozenset({pu}))
    return paths


def _store_paths(graph: ArchGraph, pu: str) -> list[list[tuple[str, str]]]:
    """All pu ->(BUF ->)* MEM simple paths; intermediates must be BUF nodes."""
    g = graph.g
    paths: list[list[tuple[str, str]]] = []

    def _forward(node: str, prefix: list[tuple[str, str]], seen: frozenset) -> None:
        for succ in g.successors(node):
            if succ in seen:
                continue
            kind = g.nodes[succ]["kind"]
            if kind is NodeKind.MEM:
                paths.append(prefix + [(node, succ)])
            elif kind is NodeKind.BUF:
                _forward(succ, prefix + [(node, succ)], seen | {succ})

    _forward(pu, [], frozenset({pu}))
    return paths


def _path_cost_uniform(
    graph: ArchGraph,
    paths_by_pu: dict[str, list[list[tuple[str, str]]]],
    nbytes: int,
    clock: Any,
    notes: list[str],
    *,
    what: str,
) -> tuple[float, float]:
    """Sum per-edge hop costs; require ONE link-class signature + cost overall."""
    by_signature: dict[tuple[str, ...], tuple[float, float]] = {}
    for pu, paths in paths_by_pu.items():
        for path in paths:
            signature = tuple(_edge_link(graph, u, v) for u, v in path)
            latency = 0.0
            energy = 0.0
            for u, v in path:
                hop_latency, hop_energy = _edge_hop_cost(graph, u, v, nbytes, clock, notes)
                latency += hop_latency
                energy += hop_energy
            cost = (latency, energy)
            previous = by_signature.get(signature)
            if previous is not None and previous != cost:
                raise LowerError(
                    "E_TOPO_AMBIGUOUS_PATH",
                    f"{what} paths with link classes {list(signature)} disagree on "
                    f"cost ({previous} vs {cost}); check per-edge attr overrides",
                )
            by_signature[signature] = cost
    if len(by_signature) > 1:
        shown = sorted(list(sig) for sig in by_signature)
        raise LowerError(
            "E_TOPO_AMBIGUOUS_PATH",
            f"two distinct {what} paths with different link classes: {shown}",
        )
    return next(iter(by_signature.values()))


def _host_link(
    graph: ArchGraph, clock: Any, notes: list[str]
) -> tuple[str, float, float, float]:
    """-> (host_nid, host_device_Bps, transfer_latency_cycles, transfer_energy_pJ).

    host_pu_transfer is charged per 64-bit column: latency = startup only,
    energy = energy_per_byte x 8 B.
    """
    hosts = graph.node_ids_of_kind(NodeKind.HOST)
    if not hosts:
        raise LowerError("E_TOPO_NO_HOST", "no host node in the graph")
    if len(hosts) > 1:
        raise LowerError(
            "E_TOPO_AMBIGUOUS_PATH", f"multiple host nodes: {sorted(hosts)}"
        )
    host = hosts[0]
    out_edges = list(graph.g.out_edges(host))
    if not out_edges:
        raise LowerError(
            "E_TOPO_NO_HOST", f"host node '{host}' has no outgoing (downstream) link"
        )
    resolved: set[tuple[str, float, float, float]] = set()
    for u, v in out_edges:
        data = graph.g[u][v][0]
        where = f"edge {u} -> {v}"
        bw = data.get("bw")
        if bw is None:
            raise LowerError("E_ATTR_MISSING", f"{where}: host link has no bandwidth")
        if isinstance(bw, str):
            if bw.strip().upper() in _INF_SENTINELS:
                raise LowerError(
                    "E_UNIT",
                    f"{where}: bandwidth.host_device must be finite, got {bw!r}",
                )
            raise LowerError("E_ATTR_RAW_STRING", f"{where}: unparsable bandwidth {bw!r}")
        if not isinstance(bw, ureg.Quantity):
            raise LowerError(
                "E_UNIT", f"{where}: host bandwidth must carry pint units, got {bw!r}"
            )
        try:
            bps = float(bw.to("byte/second").magnitude)
        except DimensionalityError as exc:
            raise LowerError("E_UNIT", f"{where}: bad bandwidth units {bw!r}") from exc
        latency = _edge_latency_cycles(data, where, None, clock, notes)
        energy = _edge_energy_pj(data, where, 8)
        resolved.add((_edge_link(graph, u, v), bps, latency, energy))
    if len(resolved) > 1:
        raise LowerError(
            "E_TOPO_AMBIGUOUS_PATH",
            f"host node '{host}' has non-uniform downstream links: {sorted(resolved)}",
        )
    _, bps, latency, energy = resolved.pop()
    return host, bps, latency, energy


def _b2b_cost(
    graph: ArchGraph, mems: list[str], nbytes: int, clock: Any, notes: list[str]
) -> tuple[float, float]:
    """bank<->bank transfer: MEM->MEM direct, or MEM->hub->MEM with a single
    BUS/XBAR/HOST/ROUTER intermediate.  One link-class signature required."""
    g = graph.g
    found: dict[tuple[str, ...], tuple[float, float]] = {}

    def _record(signature: tuple[str, ...], cost: tuple[float, float]) -> None:
        previous = found.get(signature)
        if previous is not None and previous != cost:
            raise LowerError(
                "E_TOPO_AMBIGUOUS_PATH",
                f"bank-to-bank paths with link classes {list(signature)} disagree "
                f"on cost ({previous} vs {cost})",
            )
        found[signature] = cost

    fanout_cache: dict[str, dict[str, tuple[float, float]]] = {}

    def _hub_fanout(hub: str) -> dict[str, tuple[float, float]]:
        if hub in fanout_cache:
            return fanout_cache[hub]
        table: dict[str, tuple[float, float]] = {}
        for _, dst in g.out_edges(hub):
            if g.nodes[dst]["kind"] is not NodeKind.MEM:
                continue
            link = _edge_link(graph, hub, dst)
            cost = _edge_hop_cost(graph, hub, dst, nbytes, clock, notes)
            previous = table.get(link)
            if previous is not None and previous != cost:
                raise LowerError(
                    "E_TOPO_AMBIGUOUS_PATH",
                    f"link class '{link}' from '{hub}' has non-uniform per-edge "
                    f"costs on its MEM fan-out",
                )
            table[link] = cost
        fanout_cache[hub] = table
        return table

    for mem in mems:
        for _, dst in g.out_edges(mem):
            dst_kind = g.nodes[dst]["kind"]
            if dst_kind is NodeKind.MEM:
                if dst == mem:
                    continue
                _record(
                    (_edge_link(graph, mem, dst),),
                    _edge_hop_cost(graph, mem, dst, nbytes, clock, notes),
                )
            elif dst_kind in _B2B_HUB_KINDS:
                first_link = _edge_link(graph, mem, dst)
                first_cost = _edge_hop_cost(graph, mem, dst, nbytes, clock, notes)
                for second_link, second_cost in _hub_fanout(dst).items():
                    _record(
                        (first_link, second_link),
                        (
                            first_cost[0] + second_cost[0],
                            first_cost[1] + second_cost[1],
                        ),
                    )
    if not found:
        raise LowerError(
            "E_SCHEMA",
            "no bank-to-bank path found (need MEM->MEM or MEM->{bus,xbar,host,"
            "router}->MEM edges with a dedicated link class); required for "
            "unit_costs.bank_to_bank_transfer",
        )
    if len(found) > 1:
        shown = sorted(list(sig) for sig in found)
        raise LowerError(
            "E_TOPO_AMBIGUOUS_PATH",
            f"multiple distinct bank-to-bank path signatures: {shown}",
        )
    return next(iter(found.values()))


# ──────────────────────────────────────────────────────────────────────────
#  Hierarchy derivation
# ──────────────────────────────────────────────────────────────────────────


def _strip_index(token: str) -> str:
    return re.sub(r"\d+$", "", token).strip().lower()


def _per_parent_count(paths: list[list[str]], position: int) -> int:
    groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for path in paths:
        groups[tuple(path[:position])].add(path[position])
    counts = {len(children) for children in groups.values()}
    if len(counts) != 1:
        raise LowerError(
            "E_TOPO_RAGGED",
            f"non-uniform child multiplicity at hierarchy position {position}: "
            f"{sorted(counts)}",
        )
    return counts.pop()


def _derive_hierarchy(
    graph: ArchGraph, pus: list[str], mems: list[str], pairwise: bool
) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    """Derive the flat ``hierarchy`` block from PU (and, for the pairwise
    attacc case, MEM) hier_id paths.

    Returns ``({"L2": {kind, count}, ...}, {path_position: level_kind})``.
    """
    g = graph.g
    node_kind_at = {data["hier_id"]: data["kind"] for _, data in g.nodes(data=True)}

    pu_paths = [g.nodes[pu]["hier_id"].split("/") for pu in pus]
    depths = {len(path) for path in pu_paths}
    if len(depths) != 1:
        raise LowerError(
            "E_TOPO_RAGGED", f"PU nodes sit at different hierarchy depths: {sorted(depths)}"
        )
    depth = depths.pop()

    # Leaf level: kind from the PU segment, or from the MEM segment when PUs
    # are 1:1-paired with leaf MEMs at the same depth (attacc rule).
    if pairwise:
        leaf_paths = [g.nodes[mem]["hier_id"].split("/") for mem in mems]
        fallback_kind = "bank"
    else:
        leaf_paths = pu_paths
        fallback_kind = "pu"
    leaf_tokens = {_strip_index(path[-1]) for path in leaf_paths}
    if len(leaf_tokens) != 1:
        raise LowerError(
            "E_TOPO_RAGGED", f"mixed leaf segment names: {sorted(leaf_tokens)}"
        )
    leaf_token = leaf_tokens.pop()
    # The leaf segment is the device node itself: fall back to its node kind
    # when the token is not a legal level kind (e.g. upmem's "dpu3" -> pu).
    leaf_kind = leaf_token if leaf_token in _LEVEL_KINDS else fallback_kind

    # (kind, count) list, leaf first, walking toward the root.
    levels: list[tuple[int, str, int]] = [
        (depth - 1, leaf_kind, _per_parent_count(leaf_paths, depth - 1))
    ]
    for position in range(depth - 2, -1, -1):
        tokens = {_strip_index(path[position]) for path in pu_paths}
        if len(tokens) != 1:
            raise LowerError(
                "E_TOPO_RAGGED",
                f"mixed segment names at hierarchy position {position}: {sorted(tokens)}",
            )
        token = tokens.pop()
        prefixes = {"/".join(path[: position + 1]) for path in pu_paths}
        named_kinds = {
            node_kind_at[prefix] for prefix in prefixes if prefix in node_kind_at
        }
        if named_kinds and named_kinds <= _NON_LEVEL_KINDS:
            # Segment names BUF/HOST/BUS/XBAR/router node(s): never a level
            # (covers the skipped bus/xbar root segment).
            continue
        if token in _LEVEL_KINDS:
            levels.append((position, token, _per_parent_count(pu_paths, position)))
            continue
        raise LowerError(
            "E_KIND_UNKNOWN",
            f"hierarchy segment '{sorted(prefixes)[0]}' (token '{token}') is neither "
            f"a legal level kind ({sorted(_LEVEL_KINDS)}) nor a bus/xbar/host node",
        )

    hierarchy: dict[str, dict[str, Any]] = {}
    position_kinds: dict[int, str] = {}
    for offset, (position, kind, count) in enumerate(levels):
        hierarchy[f"L{2 + offset}"] = {"kind": kind, "count": int(count)}
        position_kinds[position] = kind
    return hierarchy, position_kinds


# ──────────────────────────────────────────────────────────────────────────
#  Specialized add-tree units (activation variants)
# ──────────────────────────────────────────────────────────────────────────


def _special_units(
    graph: ArchGraph,
    special_ids: list[str],
    position_kinds: dict[int, str],
    clock: Any,
    notes: list[str],
) -> tuple[int, str, tuple[float, float]] | None:
    """-> (add_tree_width, unit-cost key, (latency, energy)) or None.

    The unit-cost key is selected by the unit's hier depth: directly under the
    root segment -> ``base_die``; at the channel level -> ``channel_level``.
    """
    if not special_ids:
        return None
    keys: set[str] = set()
    widths: set[int] = set()
    costs: set[tuple[float, float]] = set()
    for nid in special_ids:
        data = graph.g.nodes[nid]
        special = data.get("special")
        if not isinstance(special, str) or special.strip().lower() != "add_tree":
            raise LowerError(
                "E_KIND_UNKNOWN",
                f"PU '{nid}': unknown special unit kind {special!r} (supported: add_tree)",
            )
        widths.add(_as_int(_node_attr(graph, nid, "add_tree_width"),
                           f"PU node '{nid}' attribute 'add_tree_width'"))
        latency = _as_cycles(
            _node_attr(graph, nid, "compute_latency"),
            f"PU node '{nid}' attribute 'compute_latency'", clock, notes,
        )
        energy = _as_pj(
            _node_attr(graph, nid, "compute_energy"),
            f"PU node '{nid}' attribute 'compute_energy'",
        )
        costs.add((latency, energy))
        segments = data["hier_id"].split("/")
        parent_position = len(segments) - 2
        if parent_position < 0:
            raise LowerError(
                "E_KIND_UNKNOWN",
                f"add-tree unit '{nid}' has no parent hierarchy segment",
            )
        if position_kinds.get(parent_position) == "channel":
            keys.add("channel_level")
        elif parent_position == 0:
            keys.add("base_die")
        else:
            raise LowerError(
                "E_KIND_UNKNOWN",
                f"add-tree unit '{nid}' (hier '{data['hier_id']}') sits neither "
                f"directly under the root segment (base die) nor at the channel level",
            )
    if len(keys) > 1 or len(widths) > 1 or len(costs) > 1:
        raise LowerError(
            "E_TOPO_RAGGED",
            f"add-tree units disagree (keys {sorted(keys)}, widths {sorted(widths)}, "
            f"costs {sorted(costs)})",
        )
    return widths.pop(), keys.pop(), costs.pop()


# ──────────────────────────────────────────────────────────────────────────
#  Area
# ──────────────────────────────────────────────────────────────────────────


def _as_area(value: Any, where: str) -> float:
    _reject_raw_string(value, where)
    if isinstance(value, bool):
        raise LowerError("E_UNIT", f"{where}: bad area {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, ureg.Quantity):
        return float(value.magnitude)
    raise LowerError("E_UNIT", f"{where}: bad area {value!r}")


def _total_area_mm2(graph: ArchGraph, pus: list[str], bufs: list[str]) -> float | None:
    """meta.total_area_mm2 override, else dram_die_area_mm2 + sum of PU/BUF
    area_mm2 attrs; OMITTED entirely when neither meta key exists (attacc)."""
    meta = graph.meta if isinstance(graph.meta, dict) else {}
    if "total_area_mm2" in meta:
        try:
            return float(meta["total_area_mm2"])
        except (TypeError, ValueError) as exc:
            raise LowerError(
                "E_UNIT", f"meta.total_area_mm2 is not a number: {meta['total_area_mm2']!r}"
            ) from exc
    if "dram_die_area_mm2" not in meta:
        return None
    try:
        total = float(meta["dram_die_area_mm2"])
    except (TypeError, ValueError) as exc:
        raise LowerError(
            "E_UNIT", f"meta.dram_die_area_mm2 is not a number: {meta['dram_die_area_mm2']!r}"
        ) from exc
    for nid in list(pus) + list(bufs):
        value = graph.g.nodes[nid].get("area_mm2")
        if value is not None:
            total += _as_area(value, f"node '{nid}' attribute 'area_mm2'")
    return _round_area(total)


# ──────────────────────────────────────────────────────────────────────────
#  lower_arch
# ──────────────────────────────────────────────────────────────────────────


def lower_arch(arch: ArchGraph, *, notes: list[str] | None = None) -> dict[str, Any]:
    """Lower a parsed GPAM ``ArchGraph`` to the flat arch payload dict.

    ``notes`` (optional) collects human-readable remarks (e.g. time-valued
    latency conversions) for ``--verbose``.
    """
    notes = notes if notes is not None else []
    g = arch.g

    # PU clock: required — cycle-valued latencies are PU cycles.
    clock = arch.clock
    if clock is None:
        raise LowerError(
            "E_UNIT",
            "meta.clock is missing or unparsable; the PU clock is required "
            "(cycle-valued latencies are PU cycles)",
        )
    try:
        mhz = float(clock.to("MHz").magnitude)
    except DimensionalityError as exc:
        raise LowerError("E_UNIT", f"meta.clock has bad units: {clock!r}") from exc
    if mhz <= 0:
        raise LowerError("E_UNIT", f"meta.clock must be positive, got {clock!r}")

    mems = sorted(arch.node_ids_of_kind(NodeKind.MEM))
    all_pus = sorted(arch.node_ids_of_kind(NodeKind.PU))
    special_pus = [nid for nid in all_pus if g.nodes[nid].get("special") is not None]
    pus = [nid for nid in all_pus if g.nodes[nid].get("special") is None]
    bufs = sorted(arch.node_ids_of_kind(NodeKind.BUF))
    if not pus:
        raise LowerError("E_TOPO_NO_PU", "no (non-special) PU node in the graph")
    if not mems:
        raise LowerError("E_TOPO_NO_MEM", "no MEM node in the graph")
    if not bufs:
        raise LowerError(
            "E_SCHEMA",
            "no BUF node in the graph; the flat schema requires specs.cache_bytes "
            "(BUF capacity) and cache.mode",
        )

    # ── specs from node attributes (uniform across nodes of a role) ────────
    row_bytes = _uniform_attr(arch, mems, "row_bytes", _as_bytes, role="MEM")
    bank_bytes = _uniform_attr(arch, mems, "capacity", _as_bytes, role="MEM")
    simd_lanes = _uniform_attr(arch, pus, "simd_lanes", _as_int, role="PU")
    if simd_lanes <= 0:
        raise LowerError("E_UNIT", f"simd_lanes must be positive, got {simd_lanes}")
    # The modeled transaction: one SIMD vector load.
    transaction_bytes = simd_lanes * 16 // 8

    if row_bytes <= 0:
        raise LowerError("E_UNIT", f"row_bytes must be positive, got {row_bytes}")
    if bank_bytes % row_bytes != 0:
        raise LowerError(
            "E_SCHEMA",
            f"bank capacity {bank_bytes} B is not divisible by row_bytes "
            f"{row_bytes} B (arch.py invariant)",
        )

    # ── cache block from BUF nodes ──────────────────────────────────────────
    cache_bytes = _uniform_attr(arch, bufs, "capacity", _as_bytes, role="BUF")

    def _as_mode(value: Any, where: str) -> str:
        if not isinstance(value, str):
            raise LowerError(
                "E_UNIT", f"{where}: expected a cache mode string (wram|regfile), got {value!r}"
            )
        return value
    cache_mode = _uniform_attr(arch, bufs, "mode", _as_mode, role="BUF")
    vector_width_bits = _uniform_attr(
        arch, bufs, "vector_width_bits", _as_int, role="BUF",
        required=(str(cache_mode).strip().lower() == "regfile"),
    )
    if vector_width_bits is not None and vector_width_bits != simd_lanes * 16:
        raise LowerError(
            "E_SCHEMA",
            f"cache.vector_width_bits {vector_width_bits} is inconsistent with "
            f"simd_lanes {simd_lanes} (expected simd_lanes*16 = {simd_lanes * 16})",
        )

    # ── load paths: banks_per_pu + bank_to_pu_load ─────────────────────────
    load_paths_by_pu: dict[str, list[list[tuple[str, str]]]] = {}
    banks_per_pu_values: set[int] = set()
    for pu in pus:
        paths = _load_paths(arch, pu)
        if not paths:
            raise LowerError(
                "E_TOPO_NO_LOAD_PATH",
                f"PU '{pu}' has no MEM->(BUF->)PU load path (zero-match "
                f"wildcards emit nothing silently — check topology edges)",
            )
        load_paths_by_pu[pu] = paths
        banks_per_pu_values.add(len({path[0][0] for path in paths}))
    if len(banks_per_pu_values) != 1:
        raise LowerError(
            "E_TOPO_RAGGED",
            f"banks_per_pu (per-PU MEM in-degree over the load path) is not "
            f"uniform across PUs: {sorted(banks_per_pu_values)}",
        )
    banks_per_pu = banks_per_pu_values.pop()
    load_latency, load_energy = _path_cost_uniform(
        arch, load_paths_by_pu, transaction_bytes, clock, notes, what="load"
    )

    # ── store paths (explicit reverse edges — never assumed) ───────────────
    store_paths_by_pu: dict[str, list[list[tuple[str, str]]]] = {}
    for pu in pus:
        paths = _store_paths(arch, pu)
        if not paths:
            raise LowerError(
                "E_TOPO_NO_STORE_PATH",
                f"PU '{pu}' has no PU->(BUF->)MEM store path; the parser is "
                f"forward-only, so the store path must be authored as explicit "
                f"reverse edges with their own link class",
            )
        store_paths_by_pu[pu] = paths
    store_latency, store_energy = _path_cost_uniform(
        arch, store_paths_by_pu, transaction_bytes, clock, notes, what="store"
    )

    # ── host link + bank<->bank ─────────────────────────────────────────────
    host, host_device_bps, host_latency, host_energy = _host_link(arch, clock, notes)
    b2b_latency, b2b_energy = _b2b_cost(arch, mems, transaction_bytes, clock, notes)

    # ── structural completeness ─────────────────────────────────────────────
    reachable = nx.descendants(g, host) | {host}
    for pu in pus:
        if pu not in reachable:
            raise LowerError(
                "E_TOPO_NO_HOST",
                f"PU '{pu}' has no host path (unreachable from host '{host}')",
            )
    for mem in mems:
        if mem not in reachable:
            raise LowerError(
                "E_TOPO_NO_MEM",
                f"MEM '{mem}' is unreachable from host '{host}'",
            )

    # ── verbatim node-attribute unit costs ──────────────────────────────────
    def _cycles_conv(value: Any, where: str) -> float:
        return _as_cycles(value, where, clock, notes)

    row_change_latency = _uniform_attr(arch, mems, "row_change_latency", _cycles_conv, role="MEM")
    row_change_energy = _uniform_attr(arch, mems, "row_change_energy", _as_pj, role="MEM")
    conflict_latency = _uniform_attr(
        arch, mems, "host_row_conflict_latency", _cycles_conv, role="MEM"
    )
    conflict_energy = _uniform_attr(
        arch, mems, "host_row_conflict_energy", _as_pj, role="MEM"
    )
    compute_latency = _uniform_attr(arch, pus, "compute_latency", _cycles_conv, role="PU")
    compute_energy = _uniform_attr(arch, pus, "compute_energy", _as_pj, role="PU")

    # ── hierarchy ───────────────────────────────────────────────────────────
    pu_depths = {g.nodes[pu]["hier_id"].count("/") for pu in pus}
    mem_depths = {g.nodes[mem]["hier_id"].count("/") for mem in mems}
    pairwise = (
        banks_per_pu == 1
        and len(mems) == len(pus)
        and len(pu_depths) == 1
        and pu_depths == mem_depths
    )
    hierarchy, position_kinds = _derive_hierarchy(arch, pus, mems, pairwise)

    # ── optional emissions ──────────────────────────────────────────────────
    buses = sorted(arch.node_ids_of_kind(NodeKind.BUS))
    n_lines_buses = [nid for nid in buses if "n_lines" in g.nodes[nid]]
    n_lines = (
        _uniform_attr(arch, n_lines_buses, "n_lines", _as_int, role="BUS")
        if n_lines_buses
        else None
    )
    col_bits = _uniform_attr(arch, mems, "col_bits", _as_int, role="MEM", required=False)
    special = _special_units(arch, special_pus, position_kinds, clock, notes)
    total_area = _total_area_mm2(arch, all_pus, bufs)

    # ── assemble payload (baseline file key order) ─────────────────────────
    cache: dict[str, Any] = {"mode": cache_mode}
    if vector_width_bits is not None:
        cache["vector_width_bits"] = int(vector_width_bits)

    specs: dict[str, Any] = {
        "row_bytes": _size_to_text(row_bytes),
        "bank_bytes": _size_to_text(bank_bytes),
        "cache_bytes": _size_to_text(cache_bytes),
        "banks_per_pu": int(banks_per_pu),
        "pu_frequency": _format_freq_mhz(mhz),
        "simd_lanes": int(simd_lanes),
    }
    if n_lines is not None:
        specs["n_lines"] = int(n_lines)
    if special is not None:
        specs["add_tree_width"] = int(special[0])
    if col_bits is not None:
        specs["col_bits"] = int(col_bits)

    unit_costs: dict[str, Any] = {
        "bank_to_pu_load": {
            "latency": _round_cost(load_latency), "energy": _round_cost(load_energy)
        },
        "pu_to_bank_store": {
            "latency": _round_cost(store_latency), "energy": _round_cost(store_energy)
        },
        "host_pu_transfer": {
            "latency": _round_cost(host_latency), "energy": _round_cost(host_energy)
        },
        "bank_to_bank_transfer": {
            "latency": _round_cost(b2b_latency), "energy": _round_cost(b2b_energy)
        },
        "host_bank_row_conflict": {
            "latency": _round_cost(conflict_latency), "energy": _round_cost(conflict_energy)
        },
        "pu_compute": {
            "latency": _round_cost(compute_latency), "energy": _round_cost(compute_energy)
        },
        "row_change": {
            "latency": _round_cost(row_change_latency), "energy": _round_cost(row_change_energy)
        },
    }
    if special is not None:
        _, key, (special_latency, special_energy) = special
        unit_costs[key] = {
            "latency": _round_cost(special_latency),
            "energy": _round_cost(special_energy),
        }

    payload: dict[str, Any] = {
        "name": str(arch.meta.get("name", "arch")) if isinstance(arch.meta, dict) else "arch",
        "cache": cache,
        "specs": specs,
        "hierarchy": hierarchy,
        "bandwidth": {"host_device": _bandwidth_to_text(host_device_bps)},
        "unit_costs": unit_costs,
    }
    if total_area is not None:
        payload["total_area_mm2"] = total_area
    return payload


# ──────────────────────────────────────────────────────────────────────────
#  Self-check, rendering, lower_file
# ──────────────────────────────────────────────────────────────────────────


def _self_check(payload: dict[str, Any]) -> None:
    """Never emit anything the strict flat parser rejects."""
    try:
        Arch.from_dict(payload)
    except ArchParseError as exc:
        raise LowerError(
            "E_SCHEMA", f"lowered payload rejected by the strict flat parser: {exc}"
        ) from exc


def _render(payload: dict[str, Any], topo: Path, corner: str | None) -> str:
    header = (
        f"# Lowered by gpam-lower from {Path(topo).as_posix()} "
        f"(corner={corner if corner else 'nominal'})\n"
    )
    return header + yaml.safe_dump(payload, sort_keys=False)


def lower_file(
    topo: str | Path,
    *,
    corner: str | None = None,
    out: str | Path | None = None,
    name: str | None = None,
) -> Path:
    """Lower a GPAM topology file to a flat arch YAML file; returns the path."""
    topo_path = Path(topo)
    graph = load_arch(topo_path, corner)
    notes: list[str] = []
    payload = lower_arch(graph, notes=notes)
    if name is not None:
        payload["name"] = str(name)
    _self_check(payload)
    out_path = (
        Path(out) if out is not None
        else topo_path.with_name(f"{payload['name']}.lowered.yaml")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(payload, topo_path, corner), encoding="utf-8")
    return out_path


# ──────────────────────────────────────────────────────────────────────────
#  --check: Tier-1 (semantic YAML) + Tier-2 (parsed Arch) equivalence
# ──────────────────────────────────────────────────────────────────────────


def _tier1_diff(generated: Any, reference: Any, path: str = "") -> list[str]:
    label = path[:-1] if path else "<root>"
    if isinstance(generated, dict) and isinstance(reference, dict):
        diffs: list[str] = []
        for key in sorted(set(generated) | set(reference), key=str):
            sub = f"{path}{key}"
            if key not in reference:
                diffs.append(f"tier1: {sub}: present in generated, missing in reference")
            elif key not in generated:
                diffs.append(f"tier1: {sub}: missing in generated, present in reference")
            else:
                diffs.extend(_tier1_diff(generated[key], reference[key], f"{sub}."))
        return diffs
    if isinstance(generated, list) and isinstance(reference, list):
        if len(generated) != len(reference):
            return [
                f"tier1: {label}: list length {len(generated)} != {len(reference)}"
            ]
        diffs = []
        for index, (gen_item, ref_item) in enumerate(zip(generated, reference)):
            diffs.extend(_tier1_diff(gen_item, ref_item, f"{path}[{index}]."))
        return diffs
    # Scalars: Python equality — floats exact, 10.0 == 10 acceptable.
    if generated != reference:
        return [f"tier1: {label}: generated {generated!r} != reference {reference!r}"]
    return []


def _tier2_diff(payload: dict[str, Any], reference: Path) -> list[str]:
    try:
        generated = Arch.from_dict(payload)
    except ArchParseError as exc:
        return [f"tier2: generated payload failed to parse: {exc}"]
    try:
        expected = Arch.from_yaml_file(reference)
    except (ArchParseError, OSError, yaml.YAMLError) as exc:
        return [f"tier2: reference '{reference}' failed to parse: {exc}"]

    diffs: list[str] = []
    for field in fields(generated.specs):
        gen_value = getattr(generated.specs, field.name)
        ref_value = getattr(expected.specs, field.name)
        if gen_value != ref_value:
            diffs.append(
                f"tier2: specs.{field.name}: generated {gen_value!r} != reference {ref_value!r}"
            )
    for level in sorted(set(generated.hierarchy.levels) | set(expected.hierarchy.levels)):
        gen_level = generated.hierarchy.levels.get(level)
        ref_level = expected.hierarchy.levels.get(level)
        if gen_level != ref_level:
            diffs.append(
                f"tier2: hierarchy.L{level}: generated {gen_level!r} != reference {ref_level!r}"
            )
    if generated.bandwidths != expected.bandwidths:
        diffs.append(
            f"tier2: bandwidth.host_device: generated "
            f"{generated.bandwidths.host_device_Bps!r} != reference "
            f"{expected.bandwidths.host_device_Bps!r}"
        )
    for field in fields(generated.unit_costs):
        gen_value = getattr(generated.unit_costs, field.name)
        ref_value = getattr(expected.unit_costs, field.name)
        if gen_value != ref_value:
            diffs.append(
                f"tier2: unit_costs.{field.name}: generated {gen_value!r} != "
                f"reference {ref_value!r}"
            )
    if generated.cache != expected.cache:
        diffs.append(
            f"tier2: cache: generated {generated.cache!r} != reference {expected.cache!r}"
        )
    return diffs


def _check_against(payload: dict[str, Any], reference: Path) -> list[str]:
    try:
        reference_raw = yaml.safe_load(reference.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"tier1: cannot read reference '{reference}': {exc}"]
    diffs = _tier1_diff(payload, reference_raw)
    diffs.extend(_tier2_diff(payload, reference))
    return diffs


# ──────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gpam-lower",
        description="Lower a GPAM architecture spec (topology YAML + its "
                    "referenced tech-DB) to the flat PNMAX arch YAML consumed "
                    "by the runtime parser.",
    )
    parser.add_argument("--arch", required=True,
                        help="path to the GPAM topology YAML; its 'tech:' "
                             "reference is resolved relative to that file")
    parser.add_argument("--corner", default=None,
                        help="PVT corner name from the tech-DB")
    parser.add_argument("--out", default=None,
                        help="output path for the lowered flat YAML")
    parser.add_argument("--name", default=None,
                        help="override the emitted 'name' field (default: meta.name)")
    parser.add_argument("--check", default=None, metavar="REFERENCE.yaml",
                        help="lower in-memory and compare against a reference "
                             "flat YAML (Tier-1 semantic YAML equality + "
                             "Tier-2 parsed-Arch equality); exit 4 on mismatch")
    parser.add_argument("--print", action="store_true", dest="print_",
                        help="print the lowered YAML to stdout")
    parser.add_argument("--verbose", action="store_true",
                        help="print lowering notes (e.g. time-valued latency "
                             "conversions) to stderr")
    args = parser.parse_args(argv)

    if not (args.out or args.print_ or args.check):
        parser.error("nothing to do: pass --out PATH, --print, and/or --check REF")

    try:
        graph = load_arch(args.arch, corner=args.corner)
    except (OSError, ValueError, KeyError, YAMLError) as exc:
        # One-line message even for ruamel parse errors (multi-line repr).
        message = " ".join(str(exc).split())
        print(f"gpam-lower: error: {message}", file=sys.stderr)
        return 2

    notes: list[str] = []
    if args.corner:
        # The corner shorthand regexes cover *_latency but not
        # compute_energy / row_change_energy / bandwidth.
        notes.append(
            f"corner '{args.corner}': the scale_latency/scale_energy shorthand "
            f"does not cover compute_energy, row_change_energy, or bandwidth; "
            f"tech-DBs relying on shorthand must use an explicit scale: block "
            f"for those attributes"
        )
    try:
        payload = lower_arch(graph, notes=notes)
        if args.name is not None:
            payload["name"] = str(args.name)
        _self_check(payload)
    except LowerError as exc:
        print(f"gpam-lower: error[{exc.code}]: {exc}", file=sys.stderr)
        return 3

    if args.verbose:
        for note in notes:
            print(f"gpam-lower: note: {note}", file=sys.stderr)

    text = _render(payload, Path(args.arch), args.corner)
    exit_code = 0
    if args.check:
        diffs = _check_against(payload, Path(args.check))
        if diffs:
            for diff in diffs:
                print(diff)
            print(
                f"gpam-lower: check FAILED: {len(diffs)} difference(s) vs {args.check}",
                file=sys.stderr,
            )
            exit_code = 4
        else:
            print(f"gpam-lower: check OK vs {args.check}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    if args.print_:
        sys.stdout.write(text)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
