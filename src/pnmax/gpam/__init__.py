"""GPAM front-end: graph-based parameterized architecture model (parser + lowering).

Parser vendored from OptiPIM experimental/GPA @ c775742 (arch/loader.py,
arch/graph.py); see the module docstrings for the list of local changes.
The authoritative GPAM YAML format specification is the edge-mode
cheat-sheet at the top of :mod:`pnmax.gpam.loader`.

Public API
----------
load_arch(topo_yaml, corner=None) -> ArchGraph
    Parse a GPAM topology YAML plus its referenced tech-DB into an ArchGraph.
ArchGraph
    networkx.MultiDiGraph wrapper with pint-quantity attributes and the
    hop_cost()/pu_cost() primitives.
Node, Edge, NodeKind
    Building blocks (NodeKind: mem / buffer / pu / bus / xbar / host / router).
ureg
    The shared pint.UnitRegistry (defines cycle, op, Gop/Mop/Kop).
"""

from pnmax.gpam.graph import ArchGraph, Edge, Node, NodeKind, ureg
from pnmax.gpam.loader import load_arch

__all__ = ["ArchGraph", "Edge", "Node", "NodeKind", "load_arch", "ureg"]
