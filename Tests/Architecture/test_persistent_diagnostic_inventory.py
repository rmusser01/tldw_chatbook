from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_production_diagnostic_inventory_and_sink_topology_are_unchanged() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_persistent_diagnostic_inventory.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_persistent_metadata_marker_cannot_be_forged_outside_its_owner() -> None:
    marker = "_tldw_metadata_only_record"
    allowed = {
        "tldw_chatbook/Utils/persistent_diagnostics.py",
    }
    offenders = []
    for path in (REPO_ROOT / "tldw_chatbook").rglob("*.py"):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in allowed:
            continue
        if marker in path.read_text(encoding="utf-8"):
            offenders.append(relative)
    assert offenders == [], (
        "persistent metadata admission marker used outside its sole owner: "
        + ", ".join(offenders)
    )
