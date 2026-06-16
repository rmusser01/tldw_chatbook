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
                    (
                        "Personas",
                        "Behavior profiles for chat and agents",
                        "characters, personas, prompts, dictionaries, and lore",
                        "Attach to Console",
                    ),
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

