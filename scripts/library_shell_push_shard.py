#!/usr/bin/env python3
"""Pick one shard of Tests/UI/test_library_shell.py's nodes (TASK-39153).

The ``library-shell-dev`` job in ``.github/workflows/derived-artifacts.yml``
runs the whole file on pushes to dev, split by node across a matrix: the file
is about 100-150 serial minutes, far over one 20-minute job. Each shard
collects the file (``pytest --collect-only -q``), passes that output here, and
runs the node ids this prints, one per line.

Nodes are dealt round-robin in collection order, so the parametrized cases of
one slow test land on different shards. The ids listed in
``scripts/library_shell_push_deselect.txt`` (each under a comment saying why)
are left out; a listed id the collection no longer has is an error, so the
list cannot go stale silently.

Stdlib only: it never imports the app.

Usage:
    python scripts/library_shell_push_shard.py INDEX TOTAL COLLECTED_FILE

Exits 0 with the shard's ids on stdout, 1 on a stale deselect entry or an
empty shard.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = "Tests/UI/test_library_shell.py"
DESELECT_PATH = REPO_ROOT / "scripts" / "library_shell_push_deselect.txt"


def read_deselected(path: Path = DESELECT_PATH) -> list[str]:
    """Return the deselected node ids, ignoring comments and blank lines.

    Args:
        path: The deselect list, one node id per line.

    Returns:
        The listed node ids in file order.
    """
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def collected_nodes(collect_output: str) -> list[str]:
    """Return the node ids in ``pytest --collect-only -q`` output, in order.

    Args:
        collect_output: The collection's stdout.

    Returns:
        Every line that names a node of the target file.
    """
    return [
        line.rstrip("\n")
        for line in collect_output.splitlines()
        if line.startswith(f"{TARGET}::")
    ]


def shard(nodes: list[str], deselected: list[str], index: int, total: int) -> list[str]:
    """Deal the kept nodes round-robin and return shard ``index`` of ``total``.

    Args:
        nodes: Collected node ids, in collection order.
        deselected: Node ids to leave out; each must be collected.
        index: Zero-based shard index.
        total: Number of shards.

    Returns:
        The shard's node ids, in collection order.

    Raises:
        ValueError: A deselected id is not collected, or the index is out of range.
    """
    if not 0 <= index < total:
        raise ValueError(f"shard index {index} is outside 0..{total - 1}")
    stale = [node for node in deselected if node not in nodes]
    if stale:
        raise ValueError("deselected but not collected: " + ", ".join(stale))
    skip = set(deselected)
    kept = [node for node in nodes if node not in skip]
    return kept[index::total]


def main(argv: list[str] | None = None) -> int:
    """Print one shard's node ids.

    Args:
        argv: ``[INDEX, TOTAL, COLLECTED_FILE]``; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 on a stale deselect entry or an empty shard.
    """
    index, total, collected = argv if argv is not None else sys.argv[1:]
    nodes = collected_nodes(Path(collected).read_text(encoding="utf-8"))
    try:
        selected = shard(nodes, read_deselected(), int(index), int(total))
    except ValueError as error:
        print(f"library_shell_push_shard: {error}", file=sys.stderr)
        return 1
    if not selected:
        print("library_shell_push_shard: empty shard", file=sys.stderr)
        return 1
    print("\n".join(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
