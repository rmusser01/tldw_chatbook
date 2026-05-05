"""Product maturity Phase 1.2 first-run walkthrough contract."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-2-first-run-walkthrough.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path("backlog/tasks/task-8.2 - Product-Maturity-Phase-1.2-Clean-First-Run-Launch-And-Configuration-Walkthrough.md")


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


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


def _prepare_clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    paths = {
        "HOME": tmp_path / "home",
        "XDG_CONFIG_HOME": tmp_path / "xdg-config",
        "XDG_DATA_HOME": tmp_path / "xdg-data",
        "XDG_CACHE_HOME": tmp_path / "xdg-cache",
    }
    for env_var, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(path))
    return {env_var: str(path) for env_var, path in paths.items()}


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


@pytest.mark.asyncio
async def test_clean_first_run_launches_home_and_exposes_setup_orientation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_first_run_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            nav_buttons = list(app.screen.query(MainNavigationBar).first().query(Button))
            nav_ids = [button.id for button in nav_buttons]
            assert "nav-home" in nav_ids
            assert "nav-console" in nav_ids
            assert "nav-library" in nav_ids
            assert "nav-settings" in nav_ids

            home_text = _screen_text(app)
            assert "Dashboard, notifications, status, active work, and next actions." in home_text
            assert "Set up Console model" in home_text
            assert "Start in Console" not in home_text
            assert "More: Ctrl+P" in home_text

            for button_id, current_tab, screen_name, required_copy in (
                (
                    "nav-console",
                    "chat",
                    "ChatScreen",
                    ("Console", "Live work sources"),
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
                    ("Settings", "global preferences", "Appearance"),
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


@pytest.mark.parametrize("size", [(100, 32), (180, 50)])
@pytest.mark.asyncio
async def test_clean_first_run_home_survives_supported_terminal_sizes(
    size: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_first_run_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=size) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            home_text = _screen_text(app)
            assert app.current_tab == "home"
            assert app.screen.__class__.__name__ == "HomeScreen"
            assert "Set up Console model" in home_text
            assert "More: Ctrl+P" in home_text


def test_phase_one_two_evidence_records_clean_first_run_walkthrough() -> None:
    evidence = _text(EVIDENCE)

    assert "/Users/" not in evidence
    for required_text in (
        "## Clean-Run Setup",
        "Fresh HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "running Textual app",
        "Home",
        "Console",
        "Library",
        "Settings",
        "usable, not merely rendered",
        "TASK-8.2",
    ):
        assert required_text in evidence


def test_phase_one_two_tracking_and_task_closeout_are_current() -> None:
    tracker = _text(TRACKER)
    readme = _text(PHASE_1_README)
    task = _text(TASK)

    assert "Phase 1.2" in tracker
    assert "TASK-8.2" in tracker
    assert EVIDENCE.name in tracker
    assert EVIDENCE.name in readme
    assert "Phase 1.2 clean first-run status: verified" in readme
    assert "status: Done" in task
    for acceptance_criterion in range(1, 6):
        assert f"- [x] #{acceptance_criterion}" in task
    assert "Implementation Notes" in task
