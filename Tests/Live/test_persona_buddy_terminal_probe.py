"""Failure-evidence contract for the bounded Persona Buddy PTY probe."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_injected_child_failure_persists_atomic_parent_evidence(tmp_path: Path) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    report = tmp_path / "failure-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--inject-child-failure",
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["category"] == "persona_buddy_terminal_child_exit"
    assert payload["phase"] == "first:startup"
    assert payload["parent_return_code"] == 1
    assert payload["child_return_code"] == 1
    assert "Traceback (most recent call last)" in payload["diagnostic_tail"]
    assert "persona_buddy_injected_child_failure" in payload["diagnostic_tail"]
    assert len(payload["diagnostic_tail"].encode("utf-8")) <= 16_000
    assert "persona-buddy-terminal-" not in payload["diagnostic_tail"]
    assert payload["checks"] and not any(payload["checks"].values())
    assert set(payload["check_statuses"].values()) == {"not_run"}

    artifact = Path(payload["diagnostic_artifact"])
    assert artifact.is_file()
    assert artifact.parent == report.parent
    assert "persona_buddy_injected_child_failure" in artifact.read_text(
        encoding="utf-8"
    )
    assert payload["category"] in completed.stderr
    assert str(artifact) in completed.stderr
