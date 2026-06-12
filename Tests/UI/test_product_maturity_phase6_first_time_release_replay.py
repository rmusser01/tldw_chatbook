"""Product maturity Phase 6.2 first-time release replay contract."""

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
EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-2-first-time-user-release-replay.md"
)
QA_README = Path("Docs/superpowers/qa/product-maturity/phase-6/README.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
TASK = Path("backlog/tasks/task-13.2 - Phase-6.2-Full-first-time-user-release-replay.md")
LOCAL_PATH_PREFIXES = (
    "/Users/",
    "/home/",
    "/var/home/",
    "/private/var/folders/",
    "C:\\Users\\",
    "C:/Users/",
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _assert_no_local_path_prefixes(text: str) -> None:
    leaked_prefixes = [prefix for prefix in LOCAL_PATH_PREFIXES if prefix in text]
    assert not leaked_prefixes, f"evidence contains local filesystem prefix(es): {leaked_prefixes}"


def _screen_text(app) -> str:
    pieces: list[str] = []
    for widget in app.screen.query(Static):
        pieces.append(str(widget.renderable))
    for widget in app.screen.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(pieces)


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


def _prepare_clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_var, relative_path in (
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "xdg-config"),
        ("XDG_DATA_HOME", "xdg-data"),
        ("XDG_CACHE_HOME", "xdg-cache"),
    ):
        target = tmp_path / relative_path
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(target))


def _build_clean_first_run_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"
    return app


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


def _phase_overview_row(markdown: str, phase_title: str) -> list[str]:
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if columns and columns[0] == phase_title:
            return columns
    raise AssertionError(f"{phase_title!r} row not found")


def _metadata(text: str) -> dict:
    match = re.search(
        r"<!-- PHASE_6_2_FIRST_TIME_RELEASE_REPLAY_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_6_2_FIRST_TIME_RELEASE_REPLAY_METADATA:END -->",
        text,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


@pytest.mark.asyncio
async def test_release_first_time_replay_exposes_home_console_library_and_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_first_run_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 42)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            nav_buttons = list(app.screen.query(MainNavigationBar).first().query(Button))
            nav = [(button.id, str(button.label).strip()) for button in nav_buttons]
            for expected_nav in (
                ("nav-home", "Home"),
                ("nav-console", "Console"),
                ("nav-library", "Library"),
                ("nav-settings", "Settings"),
            ):
                assert expected_nav in nav

            home_text = _screen_text(app)
            assert "Dashboard, notifications, status, active work, and next actions." in home_text
            assert "Set up Console model" in home_text
            assert "Ctrl+P" in home_text

            for button_id, current_tab, screen_name, required_copy in (
                (
                    "nav-console",
                    "chat",
                    "ChatScreen",
                    ("Console", "Live work sources", "Model: not selected"),
                ),
                (
                    "nav-library",
                    "library",
                    "LibraryScreen",
                    ("Library", "Import/Export Sources", "Search/RAG"),
                ),
                (
                    "nav-settings",
                    "settings",
                    "SettingsScreen",
                    ("Settings", "Global preferences", "Appearance"),
                ),
            ):
                app.screen.query_one(f"#{button_id}", Button).press()
                await _wait_until(
                    pilot,
                    lambda current_tab=current_tab, screen_name=screen_name: (
                        app.current_tab == current_tab and app.screen.__class__.__name__ == screen_name
                    ),
                )
                # Some destinations (Settings categories) populate a beat
                # after the screen switch; wait for the copy, then assert.
                await _wait_until(
                    pilot,
                    lambda required_copy=required_copy: all(
                        copy in _screen_text(app) for copy in required_copy
                    ),
                )
                screen_text = _screen_text(app)
                for copy in required_copy:
                    assert copy in screen_text


