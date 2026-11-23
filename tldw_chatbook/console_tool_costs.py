"""TASK-33621.4: the Console's saved agent tool costs.

Owner decision 2026-11-23 ("persist across restarts"): the largest measured
agent tool cost per tool configuration is kept in the profile's data folder,
so an agent chat reopened after a restart can say Ready instead of "may pause
to compact" before its first send. Keys are digests of the configuration
signature (``context_spend._tool_cost_key``), never its config content.

The Console calls this module off the UI loop only, and imports it lazily, so
it stays off the boot path. It is a small top-level module so the UI test
suite can give every test its own file without loading Console code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

FILE_NAME = "console_agent_tool_costs.json"
#: The newest entries kept; older configurations fall off first.
LIMIT = 32


def tool_cost_path() -> Path:
    """The file in the profile's data folder."""
    from .config import get_user_data_dir
    from .Utils.path_validation import safe_join_path

    return safe_join_path(get_user_data_dir(), FILE_NAME)


def load_costs(path: Path | None = None) -> dict[str, int]:
    """Read the saved costs; anything missing or malformed reads as empty.

    Args:
        path: The file; the profile's data folder by default.

    Returns:
        Digest -> tokens, oldest first, at most ``LIMIT`` rows.
    """
    try:
        data = json.loads((path or tool_cost_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    current = isinstance(data, dict) and data.get("version") == 1
    rows = data.get("costs") if current else None
    costs: dict[str, int] = {}
    for row in rows if isinstance(rows, list) else ():
        if (
            isinstance(row, list)
            and len(row) == 2
            and isinstance(row[0], str)
            and type(row[1]) is int
            and row[1] >= 0
        ):
            costs[row[0]] = row[1]
    return dict(list(costs.items())[-LIMIT:])


def save_costs(costs: Mapping[str, int], path: Path | None = None) -> None:
    """Merge ``costs`` into the file, replacing it atomically.

    Two app instances on one profile each hold their own copy, so the file is
    re-read first and the larger figure per key is kept (review 2026-11-23: a
    plain overwrite let one instance drop or lower the other's figure). The
    keys in ``costs`` become the newest; the oldest fall off past ``LIMIT``.
    """
    from .Utils.atomic_file_ops import atomic_write_text

    target = path or tool_cost_path()
    merged = load_costs(target)
    for key, value in costs.items():
        merged[key] = max(int(value), merged.pop(key, -1))
    rows = [[key, value] for key, value in list(merged.items())[-LIMIT:]]
    payload = json.dumps({"version": 1, "costs": rows})
    atomic_write_text(target, payload, private=True, privacy_safe_log=True)
