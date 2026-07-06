"""Product maturity Phase 1.4 keyboard and focus sweep contract."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.app import TabNavigationProvider, TldwCli
from tldw_chatbook.UI.Navigation.shell_destinations import (
    SHELL_DESTINATION_ORDER,
    get_shell_destination,
    resolve_shell_route,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-4-keyboard-focus.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path("backlog/tasks/task-8.4 - Product-Maturity-Phase-1.4-Keyboard-And-Focus-Sweep.md")
TOP_LEVEL_DESTINATION_IDS = tuple(destination.destination_id for destination in SHELL_DESTINATION_ORDER)
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


def _build_clean_keyboard_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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
async def test_clean_run_tab_order_reaches_nav_and_primary_setup_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_keyboard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            if not isinstance(app.focused, Button):
                await pilot.press("tab")

            focus_ids: list[str] = []
            expected_focus_ids = [f"nav-{destination_id}" for destination_id in TOP_LEVEL_DESTINATION_IDS]
            for expected_focus_id in expected_focus_ids:
                await _wait_until(
                    pilot,
                    lambda: isinstance(app.focused, Button) and app.focused.id == expected_focus_id,
                )
                focused = app.focused
                assert isinstance(focused, Button)
                assert focused.id is not None
                focus_ids.append(focused.id)

                await pilot.press("tab")

            for _ in range(24):
                if isinstance(app.focused, Button) and app.focused.id == "home-primary-action":
                    break
                await pilot.press("tab")
            await _wait_until(
                pilot,
                lambda: isinstance(app.focused, Button) and app.focused.id == "home-primary-action",
            )
            focused = app.focused
            assert isinstance(focused, Button)
            focus_ids.append(focused.id or "")

            assert focus_ids == [*expected_focus_ids, "home-primary-action"]
            nav_hint = str(app.screen.query_one("#nav-overflow-hint", Static).renderable)
            assert "More" in nav_hint
            assert "Ctrl+P" in nav_hint


@pytest.mark.asyncio
async def test_command_palette_keyboard_fallback_exposes_top_level_product_model() -> None:
    ctrl_p_bindings = [binding for binding in TldwCli.BINDINGS if "ctrl+p" in str(binding.key).lower()]
    assert ctrl_p_bindings
    assert any("command_palette" in str(binding.action) for binding in ctrl_p_bindings)

    palette_destination_ids = tuple(
        resolve_shell_route(TabNavigationProvider.route_for_tab(tab_id)).destination_id
        for tab_id in TabNavigationProvider.navigation_tab_ids()
    )
    assert palette_destination_ids == TOP_LEVEL_DESTINATION_IDS

    provider = TabNavigationProvider(MagicMock())
    hits = []
    async for hit in provider.search("Tab Navigation"):
        hits.append(hit)

    hit_text_and_help = [(str(hit.text), str(hit.help or "")) for hit in hits]

    for destination_id in TOP_LEVEL_DESTINATION_IDS:
        destination = get_shell_destination(destination_id)
        matching_hits = [
            (text, help_text)
            for text, help_text in hit_text_and_help
            if destination.accessible_label in text or destination.accessible_label in help_text
        ]

        assert matching_hits, f"missing command-palette hit for {destination.destination_id}"
        assert any("Tab Navigation: Switch to" in text for text, _ in matching_hits)
        assert any(destination.purpose in help_text for _, help_text in matching_hits)

