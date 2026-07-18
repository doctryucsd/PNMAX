"""
Immutable, unit-safe representation of a digital/near-memory PIM architecture.

Vendored from OptiPIM experimental/GPA @ c775742 (arch/graph.py) for the PNMAX
artifact.  Local changes are marked with "PNMAX:" comments:
  * unit definitions ported to pint >= 0.21 syntax ("X = []" instead of
    "X = 1 * dimensionless" and numeric factors instead of prefix names,
    which modern pint no longer resolves);
  * Edge carries its tech-DB link-class name (``link``);
  * ArchGraph parses meta['clock'] so cycle-denominated latencies can be
    converted to wall-clock time;
  * hop_cost(): startup latencies given in cycles are converted via the meta
    clock (pint cannot add cycles to seconds), "INF"-bandwidth links transfer
    in zero time, and per-byte energy is applied per *byte* explicitly
    (multiplying a byte Quantity into a plain-energy Quantity would silently
    count bits, because pint's byte reduces to a dimensionless bit);
  * pu_cost()/hop_cost() accept plain numbers for ops/bytes.

Key changes (upstream notes)
-----------
1.  Every node now carries
        • nid        – unique *global* identifier  (id_pattern in YAML)
        • hier_id    – hierarchical path, '/'-separated  ("tsv/die3/bg0/bank1")
   Both are preserved even if users rename nodes later.
2.  Added helper `children_of(path)` to fetch sub-trees by prefix.
3.  Edge objects unchanged – YAML already addresses nodes by **global** nid.
"""
import enum
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import networkx as nx
import pint

# ---------------------------------------------------------------------------
#  Units <-- Change here if you want to use different units
# ---------------------------------------------------------------------------
ureg = pint.UnitRegistry()

# PNMAX: pint >= 0.21 dropped support for the upstream definition style
# ("cycle = 1 * dimensionless" parses but breaks dimensional analysis with
# KeyError(''); "giga" is a prefix, not a unit usable as a factor).
# "X = []" defines a dimensionless base unit, which is what was intended.
# pint ships a built-in `cycle` (= turn = 2*pi*radian), so redefining it
# raises a RedefinitionWarning that we deliberately silence.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # 1) one CPU/DRAM cycle  (dimensionless)
    ureg.define("cycle = [] = cy = cycles")

    # 2) one "operation"  (dimensionless)
    ureg.define("op = [] = ops")

# 3) allow GOp/s, MOp/s, KOp/s
ureg.define("Gop = 1e9 * op")
ureg.define("Mop = 1e6 * op")
ureg.define("Kop = 1e3 * op")

# ---------------------------------------------------------------------------
#  Enums and helper types
# ---------------------------------------------------------------------------
class NodeKind(str, enum.Enum):
    MEM = "mem"
    BUF = "buffer"
    PU  = "pu"
    BUS = "bus"
    XBAR= "xbar"
    HOST= "host"      # optional, not used in cost kernels (yet)
    RTR = "router"    # optional generic switch

@dataclass(frozen=True)
class Node:
    nid:      str
    kind:     NodeKind
    hier_id:  str
    attrs:    Dict[str, pint.Quantity] = field(default_factory=dict)

@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    bw:  pint.Quantity                   # [B/s]
    t0:  Optional[pint.Quantity] = None  # startup latency
    e_B: Optional[pint.Quantity] = None  # energy per byte
    sharing_group: Optional[str] = None  # contention domain
    link: Optional[str] = None           # PNMAX: tech-DB link-class name

# --------------------------------------------------------------------------- #
#  ArchGraph
# --------------------------------------------------------------------------- #
class ArchGraph:
    """
    Wrapper around networkx.MultiDiGraph that keeps pint.Quantity attributes
    *and* exposes a few fast cost primitives.
    """
    # ────────────────────────────────────────────────────────────────────── #
    def __init__(self,
                 nodes: Dict[str, Node],           # nid  → Node
                 edges: Dict[Tuple[str,str], Edge],# (src,dst) → Edge
                 meta : Dict[str, Any]):
        self.g = nx.MultiDiGraph()
        self.meta = meta

        # PNMAX: single clock domain from meta – used to convert
        # cycle-denominated latencies to wall-clock time in hop_cost().
        clk = meta.get("clock") if isinstance(meta, dict) else None
        if isinstance(clk, str):
            try:
                clk = ureg(clk)
            except Exception:
                clk = None
        self.clock: Optional[pint.Quantity] = (
            clk if isinstance(clk, ureg.Quantity) else None)

        # Add nodes to the graph
        for n in nodes.values():
            self.g.add_node(n.nid,
                            kind=n.kind,
                            hier_id=n.hier_id,
                            **n.attrs)
        # Add edges to the graph
        for e in edges.values():
            self.g.add_edge(e.src, e.dst,
                            bw=e.bw, t0=e.t0, e_B=e.e_B,
                            sharing_group=e.sharing_group,
                            link=e.link)  # PNMAX: keep link-class name

        # Get the hierarchy depth of each hardware kind
        self.hier_depths: Dict[int, List[NodeKind]] = self._scan_arch()

    # -----------------------------------------------------------------------
    #  Convenience helpers
    # -----------------------------------------------------------------------
    def children_of(self, path: str) -> List[str]:
        """Return all node-IDs whose hier_id starts with `path/`."""
        return [nid for nid, d in self.g.nodes(data=True)
                     if d["hier_id"].startswith(path)]

    # Fast queries in the visualizer / pipeline analysis
    def node_ids_of_kind(self, kind: NodeKind) -> List[str]:
        return [n for n, d in self.g.nodes(data=True) if d["kind"] == kind]

    # Return hierarchy depth -> kinds map
    def _scan_arch(self) -> Dict[int, List[NodeKind]]:
        """Return depth → kinds-present list (ordered by priority)."""
        mp: DefaultDict[int, List[NodeKind]] = defaultdict(list)
        for nid, d in self.g.nodes(data=True):
            depth = d["hier_id"].count("/")

            # Add hier_depth to node attr
            d["hier_depth"] = depth

            k = d["kind"]
            if k not in mp[depth]:
                mp[depth].append(k)
        return mp

    # Return the desired attribute of a node
    def get_node_attr(self, node_id: str, attr_name: str, default=None):
        return self.g.nodes[node_id].get(attr_name, default)

    def get_edge_attr(self, src: str, dst: str, attr_name: str, default=None, multi=0):
        """
        Return attribute *key* on edge (u→v)  (default: multigraph slot 0).
        arch.edge_attr('bank3','rank_bus','bw') -> Quantity
        """
        return self.g[src][dst][multi].get(attr_name, default)

    def node_ids_of_kind_at_depth(self, hier_depth: int, kind: Optional[NodeKind] = None):
        """
            Get nodes at specified depth with certain node type
        """
        if (kind != None):
            return [n for n, d in self.g.nodes(data=True) if d["hier_depth"] == hier_depth and d["kind"] == kind]
        else:
            return [n for n, d in self.g.nodes(data=True) if d["hier_depth"] == hier_depth]


    # --------------------------------------------------------------------- #
    #  Cost kernels
    # --------------------------------------------------------------------- #

    def _as_time(self, q) -> pint.Quantity:
        """
        PNMAX: normalise a latency attribute to wall-clock time.
        Cycle counts (dimensionless) are converted via meta['clock'];
        real time quantities pass through; None counts as zero.
        """
        if q is None:
            return 0 * ureg.ns
        if isinstance(q, str):
            q = ureg(q)
        if not isinstance(q, ureg.Quantity):
            q = q * ureg.cycle
        if q.dimensionless:                        # cycles (or a bare number)
            if self.clock is None:
                raise ValueError(
                    "cycle-denominated latency requires a 'clock:' entry "
                    "under meta: in the topology YAML")
            return (q.to("cycle").magnitude / self.clock).to("ns")
        return q.to("ns")

    def hop_cost(self, src: str, dst: str, bytes_: pint.Quantity):
        link = self.g[src][dst][0]
        bw   = link["bw"]

        if not isinstance(bytes_, ureg.Quantity):  # PNMAX: plain number = bytes
            bytes_ = bytes_ * ureg.byte

        # PNMAX: "INF" bandwidth (e.g. in-situ bit-line coupling) = free hop
        if isinstance(bw, str):
            if bw.strip().upper() in ("INF", "INFINITY"):
                transfer = 0 * ureg.ns
            else:
                raise ValueError(
                    f"edge {src} -> {dst}: unparsable bandwidth {bw!r}")
        else:
            transfer = (bytes_ / bw).to("ns")

        t = transfer + self._as_time(link["t0"])   # PNMAX: cycles → time

        # PNMAX: tech-DBs give energy_per_byte as a plain energy ("0.8 pJ")
        # that is per byte by convention – apply it per byte explicitly.
        # An explicit per-byte unit ("0.8 pJ/byte") is also honoured.
        e_B = link["e_B"] or 0 * ureg.pJ
        if isinstance(e_B, ureg.Quantity) and any(
                ("byte" in u or "bit" in u) and exp < 0
                for u, exp in e_B.units._units.items()):
            e = bytes_ * e_B                       # spelled "pJ/byte"
        else:
            e = bytes_.to("byte").magnitude * e_B  # plain "pJ", per byte
        return t.to("ns"), e.to("nJ")

    def pu_cost(self, pu_id: str, ops: pint.Quantity):
        n    = self.g.nodes[pu_id]
        phi  = n["peak_gops"].to("op/s")
        e_op = n["energy_per_op"].to("joule")
        if not isinstance(ops, ureg.Quantity):     # PNMAX: plain number = ops
            ops = ops * ureg.op
        t = (ops / phi).to("ns")
        e = ops * e_op
        return t.to("ns"), e.to("nJ")

    # -----------------------------------------------------------------------
    #  Pretty-printing helpers  (for quick CLI inspection)
    # -----------------------------------------------------------------------
    def _fmt(self, q):
        """
        Human-readable formatting for either pint.Quantity or anything else.
        • Keeps units compact (25.6 GB/s).
        • Won’t crash on dimensionless or raw scalars.
        """
        if isinstance(q, pint.Quantity):
            try:
                return f"{q.to_compact():~P}"       # nice "GiB/s" style
            except Exception:                       # dimensionless or parse bug
                try:
                    return f"{q:~P}"                # PNMAX: keep units in fallback
                except Exception:
                    return str(q.magnitude)
        return str(q)


    # ----- public: quick glance --------------------------------------------
    def print_summary(self):
        kinds = {k: len(self.node_ids_of_kind(k)) for k in NodeKind}
        kinds_str = ", ".join(f"{k.value}:{v}" for k, v in kinds.items() if v)
        print(f"Arch '{self.meta.get('name','?')}'  "
              f"[nodes: {kinds_str}; edges: {self.g.number_of_edges()}]")
        if "clock" in self.meta:
            print(f"  clock: {self._fmt(self.meta['clock'])}")
        print()

    def print_nodes(self, verbose=False):
        print("=== Nodes ===")
        for nid, d in self.g.nodes(data=True):
            kind = d["kind"].value
            main = f"{nid:<8}  ({kind})"
            if not verbose:
                # show only most interesting fields
                essential = {k: d[k] for k in ("row_bytes", "latency", "energy_per_op", "hier_id", "parent")
                             if k in d}
                rest = ", ".join(f"{k}={self._fmt(v)}" for k, v in essential.items())
                print(f"{main}  {rest}")
            else:
                rest = ", ".join(f"{k}={self._fmt(v)}" for k, v in d.items()
                                 if k != "kind")
                print(f"{main}  {rest}")
        print()

    def print_edges(self, verbose=False):
        print("=== Edges ===")
        for u, v, d in self.g.edges(data=True):
            main = f"{u:>8} -> {v:<8}"
            bw   = self._fmt(d['bw'])
            if not verbose:
                print(f"{main}  bw={bw}")
            else:
                extras = ", ".join(f"{k}={self._fmt(vv)}"
                                   for k, vv in d.items() if k != "bw")
                print(f"{main}  bw={bw}, {extras}")
        print()
