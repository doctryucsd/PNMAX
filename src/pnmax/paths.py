"""Repository-root anchoring for repo-relative default paths.

CLI defaults that reference repository data (``data/...``) or run outputs
(``results/...``) are anchored to the repository root so the commands work
regardless of the caller's working directory. The root is resolved as:

1. the ``PNMAX_ROOT`` environment variable, when set;
2. the repository that contains this package (source / editable installs);
3. the current working directory, as a last resort.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the PNMAX repository root as an absolute path."""
    env = os.environ.get("PNMAX_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/pnmax/paths.py -> src/pnmax -> src -> <repo root>
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd().resolve()


def results_root() -> Path:
    """Return the root directory for run outputs (intermediate results).

    Resolved as ``PNMAX_RESULTS_ROOT`` when set (an absolute path to keep
    heavy run outputs on a different volume), otherwise ``<repo>/results``
    so a default artifact run stays self-contained inside the repository.
    """
    env = os.environ.get("PNMAX_RESULTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "results"
