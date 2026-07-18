"""
YAML  ➜  ArchGraph
Supports:
  • macros.REPLICATE with {vars: …}
  • edge modes cartesian | pairwise | range | template
  • PVT *corner* scaling (scale & override blocks in tech-DB)

Vendored from OptiPIM experimental/GPA @ c775742 (arch/loader.py) for the
PNMAX artifact.  Local changes are marked with "PNMAX:" comments:
  * package-relative import (pnmax.gpam.graph instead of cwd-relative
    arch.graph);
  * corner tables may live under `corner:` or `corners:` in the tech-DB, and
    the `scale_latency:` / `scale_energy:` shorthand (see simdram_tech_db.yaml)
    is supported alongside explicit `scale:` / `override:` blocks;
  * corner scaling parses unit strings ("14 cycle") before multiplying —
    upstream silently skipped every attribute that carried a unit;
  * clear errors for a missing `tech:` reference, unknown corners, unknown
    node classes, unknown link classes / missing bandwidth, and edges that
    reference non-existent node ids.
"""

# ──────────────────────────────────────────────────────────────────────────
#  Edge-mode cheat-sheet  (YAML → concrete (src , dst) pairs)
#  ------------------------------------------------------------------------
#  Every edge item in YAML may contain:
#
#     src:     pattern using
#                • explicit nid          (bank17)
#                • wildcard ‘*’          (bank*)
#                • Python-expr braces    (bank{b+1})
#
#     dst:     same as src
#
#     mode:    cartesian | pairwise | range | template
#              (default: cartesian)
#
#     range:   {var: "lo:hi"}  or  {var: [list]}
#              Only used by range / template modes.
#
#  Global node-IDs (nid) are the *only* thing that appears in `src` / `dst`.
#  The hierarchical path (`hier_id`) is irrelevant for edge expansion.
#
#  Step 1 – wildcard match
#  -----------------------
#      bank*     →   bank0 bank1 …  (all existing nids that match)
#
#  Step 2 – mode-specific rule
#  ---------------------------
#
#  ①  cartesian   (default)
#      Connect EVERY src to EVERY dst.
#
#        - src: bank*
#          dst: rank_bus
#          link: bank2bus
#          mode: cartesian
#          # → (bank0,rank_bus) … (bank7,rank_bus)      |src|×|dst|
#
#  ②  pairwise
#      Keep only pairs whose **trailing digits are equal**.
#      Perfect for 1-to-1 links like bank{i} ↔ pum{i}.
#
#        - src: bank*
#          dst: pum*
#          link: bank2pum
#          mode: pairwise
#          # → (bank0,pum0) (bank1,pum1) …
#
#  ③  range
#      Exactly ONE side contains a single {var} placeholder.
#      The loader substitutes every value given in `range:` and
#      pairs that concrete node with ALL matches on the opposite side.
#
#        - src: bank0/mat{m}
#          dst: switch0
#          link: mat2sw
#          mode: range
#          range: {m: 0:3}
#          # → mat0→sw0  mat1→sw0  mat2→sw0  mat3→sw0
#
#        - src: die*
#          dst: tsv{z}
#          link: die2tsv
#          mode: range
#          range: {z: [0,1]}
#          # → (die0,tsv0) (die1,tsv0) … (dieN,tsv1)
#
#  ④  template
#      BOTH src and dst may carry placeholders {expr}.  The loader builds
#      the full Cartesian product of all variables listed under `range`,
#      then `eval()`s each {...} in that local environment (built-ins off!).
#
#        - src: die{d}/bank{b}
#          dst: die{d}/router
#          link: bank2router
#          mode: template
#          range: {d: 0:3, b: 0:7}
#
#          env = {'d':0,'b':0}  →  die0/bank0  → die0/router
#          env = {'d':0,'b':1}  →  die0/bank1  → die0/router
#          ...
#          env = {'d':3,'b':7}  →  die3/bank7  → die3/router
#
#      Arithmetic is allowed inside braces:
#          bank{b+1}, vault{d*2}, …
#
#  Notes
#  -----
#  • 'range' values can be "lo:hi" strings or explicit lists:
#        {z: 0:3}  ≡  {z: [0,1,2,3]}
#
#  • Security – every {expr} is evaluated with
#        eval(expr, {"__builtins__": None}, env)
#    so no file-system or network access is possible.
#
#  • Edge attributes (bandwidth, energy_per_byte, …) come from the
#    link-class table in tech-DB, but can be overridden per edge item
#    via an `attr:` block.
# ──────────────────────────────────────────────────────────────────────────

import copy, re, itertools
from ruamel.yaml import YAML
from pathlib import Path
from pint.errors import UndefinedUnitError, DimensionalityError
from typing import Dict, Any, List, Tuple, Optional

from pnmax.gpam.graph import Node, Edge, NodeKind, ArchGraph, ureg  # PNMAX

yaml = YAML(typ="safe")

# ──────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────

def _Q(val):
    """
    Convert "12.5GB/s" → Quantity, but leave strings like "bus" or "INF"
    unchanged.  Works on Python 3.8.
    """
    if not isinstance(val, str) or "**" in val:
        return val                        # numbers, lists, dicts stay as-is

    text = val.strip()

    # pattern to accept arbitrary integer or floating point numbers.
    if not re.match(r"^[-+]?(\d+(\.\d*)?|\.\d+)", text):
        return val

    try:
        return ureg(text)
    except (UndefinedUnitError, DimensionalityError):
        # Could be an unknown token like "GOp/s" or "INF" — leave untouched
        return val

def _deep_merge(base: Dict, new: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    for key, value in new.items():
        if isinstance(value, dict) and key in base:
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base

# ────────────────────────────────────────────────────────────────────────── #
#  PVT corner scaling
# ────────────────────────────────────────────────────────────────────────── #
def _scale_numeric(tree: Any, factors: Dict[str, float], key_factor=None):
    """
    Multiply matching attributes in-place.  `factors` maps exact attribute
    names to factors; `key_factor` (PNMAX) is an optional fallback callable
    used for the scale_latency / scale_energy shorthand.
    """
    if not factors and key_factor is None:      # nothing to do
        return
    if isinstance(tree, dict):
        for k, v in tree.items():
            f = factors.get(k)
            if f is None and key_factor is not None:
                f = key_factor(k)
            if f is not None:
                # PNMAX: parse unit strings ("14 cycle", "64GB/s") first —
                # upstream only scaled bare numbers and silently skipped
                # every value that carried a unit.
                q = _Q(v) if isinstance(v, str) else v
                if isinstance(q, (int, float, ureg.Quantity)):
                    tree[k] = q * f
                    continue
            _scale_numeric(v, factors, key_factor)
    elif isinstance(tree, list):
        for v in tree:
            _scale_numeric(v, factors, key_factor)

# PNMAX: attribute-name predicates for the corner shorthand.
#   scale_latency → timing attrs: t_rcd / t_rp / t_ras / … and *latency*
#   scale_energy  → energy attrs: energy_per_access / energy_per_op / …
_LATENCY_KEY = re.compile(r"(^|_)latency($|_)|^t_[a-z0-9]+$")
_ENERGY_KEY  = re.compile(r"^energy(_|$)")

def _shorthand_key_factor(crn: dict):
    """
    PNMAX: support the `scale_latency:` / `scale_energy:` corner shorthand
    used by sample tech-DBs, e.g.
        corners:
          slow_900mV:  {scale_latency: 1.1, scale_energy: 1.25}
    Returns a key→factor callable, or None if no shorthand is present.
    """
    lat = crn.get("scale_latency")
    en  = crn.get("scale_energy")
    unknown = [k for k in crn
               if k.startswith("scale_") and k not in ("scale_latency",
                                                       "scale_energy")]
    if unknown:
        raise ValueError(
            f"unsupported corner shorthand key(s) {unknown}; use an explicit "
            f"'scale: {{attr: factor}}' block instead")
    if lat is None and en is None:
        return None

    def key_factor(k):
        if not isinstance(k, str):
            return None
        if lat is not None and _LATENCY_KEY.search(k):
            return lat
        if en is not None and _ENERGY_KEY.match(k):
            return en
        return None

    return key_factor

# ──────────────────────────────────────────────────────────────────────────
#  Macro helpers
# ──────────────────────────────────────────────────────────────────────────
def _expand_repl(nodes: dict, repl: dict) -> dict:
    """
    Supports both legacy 1-D  bank[0:7]  and new vars: {i:0:7,j:0:3}
    Returns nid -> (node_body_dict, env_dict)
    """
    for key, spec in repl.items():

        # ── NEW vars: path -------------------------------------------------
        if isinstance(spec, dict) and "vars" in spec:
            ranges = {k: _expand_range_dict({k: v})[k] for k, v in spec["vars"].items()}
            keys, lists = zip(*ranges.items())
            for tup in itertools.product(*lists):
                env = dict(zip(keys, tup))
                id_pat = spec.get("id_pattern", None)
                if id_pat:
                    nid = _render_tmpl(id_pat, env)
                else:
                    # For both legacy and new single index macros, i.e. "bank{b}"
                    suffix = "".join(str(env[k]) for k in keys)
                    nid = f"{key}{suffix}"
                # Add hier_pattern, hier_id to the returned body
                body = copy.deepcopy(spec.get("of", {}))
                for extra_key in ("hier_pattern", "hier_id"):
                    if extra_key in spec:
                        body[extra_key] = spec[extra_key]
                nodes[nid] = (body, env)
            continue

        # ── LEGACY name[lo:hi] path ---------------------------------------
        m = re.match(r"(\w+)\[(\d+):(\d+)]$", key)
        if not m:
            raise ValueError(f"Invalid REPLICATE key: {key}")
        base, lo, hi = m.groups()
        for i in range(int(lo), int(hi) + 1):
            env = {"idx": i}
            nid = f"{base}{i}"
            nodes[nid] = (spec["of"], env)

    return nodes

def _match(pattern: str, names) -> list:
    rx = re.compile("^" + pattern.replace("*", ".*") + "$")
    return [n for n in names if rx.match(n)]

def _idx(name: str):
    m = re.search(r"(\d+)$", name)
    return m.group(1) if m else None

# ──────────────────────────────────────────────────────────────────────────
#  range/template helpers
# ──────────────────────────────────────────────────────────────────────────
_rng_re = re.compile(r"(\d+):(\d+)")   # matches "0:7"

def _expand_range_dict(rdict: dict) -> Dict[str, List[int]]:
    """
    {i: 0:3, j: 1:2}  →  {'i': [0,1,2,3], 'j':[1,2]}
    Accepts either slice strings or explicit list/tuple.
    """
    out = {}
    for k, v in rdict.items():
        if isinstance(v, str) and _rng_re.fullmatch(v):
            lo, hi = (int(x) for x in v.split(":"))
            out[k] = list(range(lo, hi + 1))
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            raise ValueError(f"range entry {k}:{v} must be 'lo:hi' or list")
    return out

def _render_tmpl(s: str, env: dict) -> str:
    """
    Replace each {...} with eval(expr, env)  (builtins disabled).
      "die{z}/bank{b+1}"  with env={'z':2,'b':3} → "die2/bank4"
    """
    def repl(m):
        expr = m.group(1)
        return str(eval(expr, {"__builtins__": None}, env))
    return re.sub(r"\{([^{}]+)\}", repl, s)

# ──────────────────────────────────────────────────────────────────────────
#  Loader entry-point
# ──────────────────────────────────────────────────────────────────────────

def load_arch(topo_path: "str | Path",
              corner: Optional[str] = None) -> ArchGraph:
    """
    Parse *both* the user topology YAML and its referenced tech-DB,
    expand macros, apply PVT corner scaling, and build an ArchGraph.

    The `tech:` reference in the topology file is resolved relative to the
    topology file's directory.
    """
    # 0 ▸ read YAML files ---------------------------------------------------
    topo_p = Path(topo_path)
    topo   = yaml.load(topo_p.read_text())
    tech_ref = topo.get("tech")
    if not tech_ref:  # PNMAX: clear error instead of KeyError
        raise ValueError(
            f"{topo_p}: topology must reference a tech-DB via a top-level "
            f"'tech:' key")
    tech   = yaml.load((topo_p.parent / tech_ref).read_text())

    # 1 ▸ merge tech → topo (+ corner) -------------------------------------
    data = _deep_merge(copy.deepcopy(tech["defaults"]), topo)

    corner_spec = corner or topo.get("corner")
    if corner_spec:
        # PNMAX: accept both `corner:` and `corners:` table spellings
        # (the sample tech-DBs use one each).
        corner_tbl = {**(tech.get("corner") or {}), **(tech.get("corners") or {})}
        crn = corner_tbl.get(corner_spec)
        if crn is None:
            known = ", ".join(sorted(corner_tbl)) or "<none>"
            raise ValueError(
                f"corner '{corner_spec}' not found in tech-DB "
                f"(known corners: {known})")
        # 1.a absolute overrides
        data = _deep_merge(data, crn.get("override", {}))
        # 1.b multiplicative scaling (explicit block and/or shorthand)
        _scale_numeric(data, dict(crn.get("scale", {})),
                       _shorthand_key_factor(crn))

    # 2 ▸ expand REPLICATE macros  →  nodes_raw  (nid → (body, env)|body) --
    nodes_raw: Dict[str, Any] = _expand_repl(
        data.setdefault("nodes", {}),
        data.get("macros", {}).get("REPLICATE", {})
    )

    # include explicit nodes placed straight under 'nodes:'
    for nid, val in list(data["nodes"].items()):
        if nid not in nodes_raw:
            nodes_raw[nid] = val

    cls_tbl = data.get("node_class", {})            # class templates

    # 3 ▸ create Node objects ------------------------------------------------
    node_objs: Dict[str, Node] = {}
    for nid, val in nodes_raw.items():

        # ── val may be  (body, env)  OR  body --------------------------------
        if isinstance(val, tuple):
            body, env = val
            body = copy.deepcopy(body)              # never mutate original
        else:
            body, env = copy.deepcopy(val), {}

        # ── mandatory fields -------------------------------------------------
        node_type = body.pop("type", None)
        if node_type is None:
            raise ValueError(f"node '{nid}' is missing mandatory 'type:' field")
        kind = NodeKind(node_type)

        cls = body.pop("class", None)
        if cls:
            if cls not in cls_tbl:  # PNMAX: clear error instead of KeyError
                known = ", ".join(sorted(cls_tbl)) or "<none>"
                raise ValueError(
                    f"node '{nid}': unknown class '{cls}' — not found in the "
                    f"tech-DB node_class table (known classes: {known})")
            body = _deep_merge(copy.deepcopy(cls_tbl[cls]), body)

        # ── hierarchy --------------------------------------------------------
        hier_id = body.pop("hier_id", None)
        if not hier_id:
            pat = body.pop("hier_pattern", None)
            hier_id = _render_tmpl(pat, env) if pat else nid
        parent = "/".join(hier_id.split("/")[:-1]) or None

        body["parent"]  = parent

        # convert all numeric strings → pint.Quantity  (via _Q)
        node_objs[nid] = Node(nid, kind, hier_id, {k: _Q(v) for k, v in body.items()})

    # 4 ▸ build Edge objects --------------------------------------------------
    link_tbl = data.get("link", {})
    edge_objs: Dict[Tuple[str, str], Edge] = {}

    for e in data.get("edges", []):
        mode   = e.get("mode", "cartesian")
        link_cls = e.get("link")                    # PNMAX: keep class name

        # merge link-class defaults + per-edge override
        attrs  = copy.deepcopy(link_tbl.get(link_cls, {}))
        attrs  = _deep_merge(attrs, e.get("attr", {}))
        attrs  = {k: _Q(v) for k, v in attrs.items()}

        # PNMAX: fail early with a readable message (upstream died later
        # with a bare KeyError('bandwidth')).
        if "bandwidth" not in attrs:
            raise ValueError(
                f"edge {e.get('src')} -> {e.get('dst')}: link class "
                f"'{link_cls}' is not in the tech-DB link table (and no "
                f"per-edge 'attr:' supplies 'bandwidth')")

        def _emit(pairs):
            for s, d in pairs:
                # PNMAX: catch range/template typos early — networkx would
                # otherwise silently create attribute-less ghost nodes.
                if s not in node_objs or d not in node_objs:
                    raise ValueError(
                        f"edge {s} -> {d}: unknown node id "
                        f"(check wildcard/range/template expansion)")
                edge_objs[(s, d)] = Edge(
                    s, d,
                    attrs["bandwidth"],
                    attrs.get("startup_latency"),
                    attrs.get("energy_per_byte"),
                    attrs.get("sharing_group"),
                    link_cls,                       # PNMAX
                )

        # wildcards for cartesian / pairwise covered here
        if mode in ("cartesian", "pairwise"):
            srcs = _match(e["src"], node_objs)
            dsts = _match(e["dst"], node_objs)

        if mode == "cartesian":
            _emit(itertools.product(srcs, dsts))

        elif mode == "pairwise":
            _emit((s, d) for s in srcs for d in dsts if _idx(s) == _idx(d))

        elif mode == "range":            # one placeholder + range dict
            rng = _expand_range_dict(e["range"])
            var = next(iter(rng))
            for v in rng[var]:
                env = {var: v}
                if "{" in e["src"]:      # placeholder is in src
                    src = _render_tmpl(e["src"], env)
                    dsts = _match(e["dst"], node_objs)
                    _emit((src, d) for d in dsts)
                else:                    # placeholder in dst
                    dst = _render_tmpl(e["dst"], env)
                    srcs = _match(e["src"], node_objs)
                    _emit((s, dst) for s in srcs)

        elif mode == "template":         # full Cartesian of all vars
            rng = _expand_range_dict(e["range"])
            keys, lists = zip(*rng.items())
            for tup in itertools.product(*lists):
                env = dict(zip(keys, tup))
                src = _render_tmpl(e["src"], env)
                dst = _render_tmpl(e["dst"], env)
                _emit([(src, dst)])

        else:
            raise ValueError(f"Unknown edge mode: {mode}")

    # 5 ▸ hand meta through, ensure 'name' present ---------------------------
    meta = data.get("meta", {})
    meta.setdefault("name", Path(topo_path).stem)

    return ArchGraph(node_objs, edge_objs, meta)
