"""Failure-evidence contract for the bounded Persona Buddy PTY probe."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_CAPTURE_NAMES = {
    "normal.ansi",
    "alert.ansi",
    "folded.ansi",
    "constrained.ansi",
}
_VISUAL_CHECKS = {
    "pet_only_normal",
    "fixed_alert_replaces_pet",
    "real_folded_thumbnail",
    "constrained_two_icons",
}


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_probe_persists_four_exact_visual_state_captures(tmp_path: Path) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    report = tmp_path / "report.json"
    captures = tmp_path / "captures"

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--report",
            str(report),
            "--capture-dir",
            str(captures),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert all(payload["checks"][name] for name in _VISUAL_CHECKS)
    assert set(payload["regions"]) == {
        "normal",
        "alert",
        "folded",
        "constrained",
    }
    for region in payload["regions"].values():
        assert set(region) == {"x", "y", "width", "height"}
        assert all(type(value) is int for value in region.values())
        assert region["width"] > 0
        assert region["height"] > 0

    artifacts = {path.name for path in captures.iterdir()}
    assert artifacts == _CAPTURE_NAMES
    assert all((captures / name).stat().st_size > 0 for name in artifacts)
    assert "Traceback" not in completed.stdout
    assert "Traceback" not in completed.stderr
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "/Users/",
        "/private/",
        "provider",
        "prompt",
        "assistant",
        "tool_arguments",
        "tool_result",
    ):
        assert forbidden not in serialized


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_success_requires_a_caller_owned_capture_directory(tmp_path: Path) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")

    completed = subprocess.run(
        [sys.executable, str(probe), "--report", str(tmp_path / "report.json")],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "--capture-dir" in completed.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_injected_child_failure_persists_atomic_parent_evidence(tmp_path: Path) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    report = tmp_path / "failure-report.json"
    captures = tmp_path / "captures"
    captures.mkdir()
    report.write_text('{"status":"STALE"}', encoding="utf-8")
    for name in _CAPTURE_NAMES:
        (captures / name).write_text("stale", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--inject-child-failure",
            "--report",
            str(report),
            "--capture-dir",
            str(captures),
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
    assert not any((captures / name).exists() for name in _CAPTURE_NAMES)


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_probe_rejects_cli_paths_outside_workspace_and_temp_roots() -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--child",
            "/etc/persona-buddy-preferences.json",
            "/etc/persona-buddy-report.json",
            "--inject-child-failure",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "persona_buddy_probe_path_invalid" in completed.stderr
    assert "/etc/" not in completed.stderr
