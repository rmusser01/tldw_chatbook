"""Shrink-only ratchet on what ``import tldw_chatbook.Agents.agent_runtime`` loads.

The module documents itself as pure, but today it imports chains that
end at textual, httpx, sqlite3 and the DB layer. TASK-36944.10 makes it pure;
until then this pins the footprint so it cannot grow (TASK-36942.2).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PIN = Path(__file__).with_name("agent_runtime_import_footprint.txt")
PROBE = (
    "import sys, json\n"
    "import tldw_chatbook.Agents.agent_runtime\n"
    "mods = [m for m in sys.modules if m.startswith('tldw_chatbook')]\n"
    "mods += [m for m in ('textual', 'httpx', 'sqlite3') if m in sys.modules]\n"
    "print('FOOTPRINT=' + json.dumps(sorted(mods)))\n"
)


def _pinned() -> set[str]:
    return {
        line.strip()
        for line in PIN.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def test_agent_runtime_import_footprint_only_shrinks():
    # Inherit os.environ: the scratch HOME/XDG/TLDW_CONFIG_PATH must reach the child.
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True, text=True, env=dict(os.environ), timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("FOOTPRINT="))
    loaded = set(json.loads(line[len("FOOTPRINT="):]))
    pinned = _pinned()

    grew = sorted(loaded - pinned)
    assert not grew, (
        "Importing tldw_chatbook.Agents.agent_runtime now loads modules that are "
        f"not pinned (the runtime must not gain imports): {grew}"
    )
    stale = sorted(pinned - loaded)
    assert not stale, (
        "These pinned entries are no longer loaded; good. Remove them from "
        f"{PIN.name} so the ratchet tightens: {stale}"
    )
