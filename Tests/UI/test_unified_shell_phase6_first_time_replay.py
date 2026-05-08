"""Phase 6.1 first-time user replay contract."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")
PHASE_6_README = Path("Docs/superpowers/qa/unified-shell/phase-6/README.md")
PHASE_6_FIRST_TIME_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-6/2026-05-05-phase-6-1-first-time-user-replay.md"
)
PHASE_6_PARENT_TASK = Path("backlog/tasks/task-7 - Phase-6-Audit-Replay-And-Closeout.md")
PHASE_6_FIRST_TIME_TASK = Path(
    "backlog/tasks/task-7.1 - Phase-6.1-Replay-first-time-user-walkthrough.md"
)

EXPECTED_NAV = [
    ("nav-home", "Home"),
    ("nav-console", "Console"),
    ("nav-library", "Library"),
    ("nav-artifacts", "Artifacts"),
    ("nav-personas", "Personas"),
    ("nav-watchlists_collections", "Watchlists"),
    ("nav-schedules", "Schedules"),
    ("nav-workflows", "Workflows"),
    ("nav-mcp", "MCP"),
    ("nav-acp", "ACP"),
    ("nav-skills", "Skills"),
    ("nav-settings", "Settings"),
]


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _status_line(text: str) -> str:
    match = re.search(r"^Status:\s*(.+)$", text, re.MULTILINE)
    assert match is not None
    return match.group(1).strip()


def _phase_evidence_row(text: str, phase_name: str) -> list[str]:
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        columns = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if columns and columns[0] == phase_name:
            return columns
    raise AssertionError(f"{phase_name} evidence row not found")


def _phase_overview_row(text: str, phase_title: str) -> list[str]:
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        columns = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if columns and columns[0] == phase_title:
            return columns
    raise AssertionError(f"{phase_title} overview row not found")


def _phase_six_first_time_metadata(text: str) -> dict:
    match = re.search(
        r"<!-- PHASE_6_1_FIRST_TIME_REPLAY_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_6_1_FIRST_TIME_REPLAY_METADATA:END -->",
        text,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _screen_text(app) -> str:
    pieces: list[str] = []
    for widget in app.screen.query(Static):
        pieces.append(str(widget.renderable))
    for widget in app.screen.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(pieces)


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s")


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


@pytest.mark.asyncio
async def test_first_time_shell_replay_exposes_home_console_and_orientation_paths() -> None:
    """Verify first-time launch exposes the shell's primary orientation paths."""
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            nav_buttons = list(app.screen.query(MainNavigationBar).first().query(Button))
            assert [(button.id, str(button.label).strip()) for button in nav_buttons] == EXPECTED_NAV

            home_text = _screen_text(app)
            assert "Dashboard, notifications, status, active work, and next actions." in home_text
            assert "Set up Console model" in home_text
            assert "More: Ctrl+P" in home_text

            for button_id, current_tab, screen_name, required_copy in (
                (
                    "nav-console",
                    "chat",
                    "ChatScreen",
                    ("Live work sources", "Watchlists: Connected"),
                ),
                (
                    "nav-library",
                    "library",
                    "LibraryScreen",
                    ("Library", "Import/Export Sources", "Search/RAG"),
                ),
                (
                    "nav-personas",
                    "personas",
                    "PersonasScreen",
                    ("Personas", "behavior profiles", "Attach to Console"),
                ),
                (
                    "nav-skills",
                    "skills",
                    "SkillsScreen",
                    ("Skills", "Agent Skills", "SKILL.md"),
                ),
            ):
                app.screen.query_one(f"#{button_id}", Button).press()
                await _wait_until(
                    pilot,
                    lambda current_tab=current_tab, screen_name=screen_name: (
                        app.current_tab == current_tab and app.screen.__class__.__name__ == screen_name
                    ),
                )
                screen_text = _screen_text(app)
                for copy in required_copy:
                    assert copy in screen_text


def test_phase_six_first_time_replay_evidence_and_tracking_are_current() -> None:
    """Verify Phase 6.1 evidence and tracking stay aligned with the roadmap."""
    evidence = _text(PHASE_6_FIRST_TIME_EVIDENCE)
    readme = _text(PHASE_6_README)
    roadmap = _text(ROADMAP)
    parent_task = _text(PHASE_6_PARENT_TASK)
    first_time_task = _text(PHASE_6_FIRST_TIME_TASK)
    metadata = _phase_six_first_time_metadata(evidence)

    assert "/Users/" not in evidence
    assert metadata["task"] == "TASK-7.1"
    assert metadata["parent_task"] == "TASK-7"
    assert metadata["persona"] == "first-time-user"
    assert metadata["decision"] == "first_time_walkthrough_recorded"
    assert metadata["entry_path"] == "clean-first-run-home"
    assert metadata["verified_routes"] == ["home", "console", "library", "personas", "skills"]
    assert metadata["orientation_paths"] == ["library", "personas", "skills"]
    assert metadata["final_focused_replay_result"]["failed"] == 0
    assert metadata["final_focused_replay_result"]["passed"] > 0

    for section in (
        "## Environment",
        "## Entry Path",
        "## Steps Attempted",
        "## Visual Usability Notes",
        "## Keyboard Path Result",
        "## Functional Result",
        "## Defect Severity",
        "## First-Time Onboarding Gaps",
        "## Deferred Service-Depth Work",
        "## Residual Risk",
    ):
        assert section in evidence
    assert "running Textual app" in evidence
    assert "Tests/UI/test_unified_shell_phase6_first_time_replay.py" in evidence

    assert _status_line(readme) == "verified"
    assert PHASE_6_FIRST_TIME_EVIDENCE.name in readme
    assert "TASK-7.1" in readme

    normalized_status = _status_line(roadmap).lower().replace("-", " ")
    assert re.search(r"phase\s+5\s+verified", normalized_status)
    assert re.search(r"phase\s+6\s+verified", normalized_status)
    assert _phase_evidence_row(roadmap, "Phase 6")[2] == "verified"
    assert str(PHASE_6_FIRST_TIME_EVIDENCE).replace("\\", "/") in roadmap
    assert "Phase 6.1: Replay first-time user walkthrough - `TASK-7.1`" in roadmap
    phase_six_overview = _phase_overview_row(roadmap, "Phase 6: Audit Replay And Closeout")
    assert phase_six_overview[2] == "verified"
    assert "TASK-7" in phase_six_overview[3]
    assert "TASK-7.1" in phase_six_overview[3]
    assert "service-depth and live-path risks remain tracked" in phase_six_overview[5]

    assert "status: Done" in parent_task
    assert "TASK-7.1" in parent_task
    assert "- [x] #1 First-time user walkthrough is replayed against the running app." in parent_task
    assert "- [x] #2 Power-user workflows are replayed against the running app." in parent_task
    assert "- [x] #3 Nielsen heuristic closeout documents remaining defects and residual risks." in parent_task

    assert "status: Done" in first_time_task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in first_time_task
    assert "Implementation Notes" in first_time_task
