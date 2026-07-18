"""Deterministic seeding policy for every stochastic component.

Artifact determinism contract: the same command produces byte-identical
outputs on any machine. Every CLI that samples (random search, simulated
annealing, pool sampling, startpoint selection) resolves its seed as:

1. an explicit ``--seed`` flag, when passed;
2. the ``PNMAX_SEED`` environment variable (exported by the experiment
   drivers), when set;
3. the fixed default ``42``.

Per-candidate / per-worker seeds are derived stably from the resolved base
seed (see the ``_stable_seed`` / ``_derived_seed`` helpers in the DSE
modules), so worker counts and scheduling order cannot change results.
"""

from __future__ import annotations

import os

DEFAULT_SEED = 42


def default_seed() -> int:
    """Return the base seed: ``PNMAX_SEED`` env override or 42."""
    env = os.environ.get("PNMAX_SEED")
    if env is None or not env.strip():
        return DEFAULT_SEED
    try:
        return int(env)
    except ValueError as exc:
        raise ValueError(
            f"PNMAX_SEED must be an integer, got {env!r}."
        ) from exc
