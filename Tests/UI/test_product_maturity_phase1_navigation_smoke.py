"""Product maturity Phase 1.3 top-level navigation smoke contract."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-3-navigation-smoke.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path("backlog/tasks/task-8.3 - Product-Maturity-Phase-1.3-Top-Level-Navigation-Smoke-Walkthrough.md")
TOP_LEVEL_DESTINATION_IDS = tuple(destination.destination_id for destination in SHELL_DESTINATION_ORDER)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _screen_text(app) -> str:
    content = app.screen.query_one("#screen-content")
    pieces: list[str] = []
    for widget in content.query(Static):
        pieces.append(str(widget.renderable))
    for widget in content.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(pieces)


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


def _prepare_clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_var, path_name in (
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "xdg-config"),
        ("XDG_DATA_HOME", "xdg-data"),
        ("XDG_CACHE_HOME", "xdg-cache"),
    ):
        path = tmp_path / path_name
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(path))


def _build_clean_navigation_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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


async def _click_visible_widget_bottom_row(pilot, app, widget_id: str) -> None:
    """Click the visible bottom row of a widget by absolute screen coordinates."""
    widget = app.screen.query_one(widget_id)
    region = widget.region
    await pilot.click(offset=(region.x + max(1, region.width // 2), region.y + max(0, region.height - 1)))


@pytest.mark.asyncio
async def test_destination_body_text_helper_excludes_navigation_chrome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_navigation_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            screen_text = _screen_text(app)
            assert "Set up Console model" in screen_text
            assert "Ctrl+P" not in screen_text


@pytest.mark.asyncio
async def test_clean_run_top_level_navigation_reaches_every_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_navigation_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            nav_bar = app.screen.query_one(MainNavigationBar)
            nav_ids = tuple(button.id.removeprefix("nav-") for button in nav_bar.query(Button))
            assert nav_ids == TOP_LEVEL_DESTINATION_IDS
            assert "Ctrl+P" in str(app.screen.query_one("#nav-overflow-hint", Static).renderable)

            reached: dict[str, str] = {}
            for destination in SHELL_DESTINATION_ORDER:
                expected_screen_name, expected_tab, expected_screen_class = app._resolve_screen_navigation_target(
                    destination.primary_route
                )
                assert expected_screen_class is not None, destination.primary_route

                await pilot.click(f"#nav-{destination.destination_id}")
                await _wait_until(
                    pilot,
                    lambda expected_tab=expected_tab, expected_screen_class=expected_screen_class: (
                        app.current_tab == expected_tab
                        and app.screen.__class__.__name__ == expected_screen_class.__name__
                    ),
                )

                screen_text = _screen_text(app)
                assert screen_text.strip(), destination.destination_id
                reached[destination.destination_id] = expected_screen_name

            assert set(reached) == set(TOP_LEVEL_DESTINATION_IDS)


@pytest.mark.asyncio
async def test_top_level_navigation_activates_visible_tab_border_from_cached_console_screen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_navigation_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            await pilot.click("#nav-console")
            await _wait_until(
                pilot,
                lambda: app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen",
            )

            await _click_visible_widget_bottom_row(pilot, app, "#nav-settings")
            await _wait_until(
                pilot,
                lambda: app.current_tab == "settings" and app.screen.__class__.__name__ == "SettingsScreen",
            )

            await pilot.click("#nav-console")
            await _wait_until(
                pilot,
                lambda: app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen",
            )

            await pilot.click("#nav-settings")
            await _wait_until(
                pilot,
                lambda: app.current_tab == "settings" and app.screen.__class__.__name__ == "SettingsScreen",
            )


