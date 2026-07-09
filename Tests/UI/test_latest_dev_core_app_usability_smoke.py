"""Latest-dev core app usability smoke gate."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from unittest.mock import patch

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from Tests.UI.test_product_maturity_phase1_core_loop import _core_loop_payload
from Tests.UI.test_product_maturity_phase1_first_run import (
    _assert_no_local_path_prefixes,
    _build_clean_first_run_app,
    _test_cli_setting,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Constants import (
    TAB_CHAT,
    TAB_HOME,
    TAB_LIBRARY,
    TAB_MCP,
    TAB_SETTINGS,
    TAB_SKILLS,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar


RAW_OBJECT_REPR = re.compile(r"<[\w.]+ object at 0x[0-9a-fA-F]+>")
BROKEN_TEXT_PATTERNS = (
    "Traceback",
    "Unhandled exception",
    "No screens installed",
    "Unable to mount",
    "Internal Error",
)
CORE_FIRST_USE_ROUTES = (
    (TAB_HOME, TAB_HOME, "HomeScreen", ("Home", "Set up Console model")),
    (TAB_CHAT, TAB_CHAT, "ChatScreen", ("Console", "Live work sources")),
    (TAB_LIBRARY, TAB_LIBRARY, "LibraryScreen", ("Library", "Search/RAG")),
    (TAB_SKILLS, TAB_SKILLS, "SkillsScreen", ("Skills", "Agent Skills")),
    (TAB_MCP, TAB_MCP, "MCPScreen", ("MCP", "Unified MCP")),
    (TAB_SETTINGS, TAB_SETTINGS, "SettingsScreen", ("Settings", "Providers & Models")),
)


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
    context: str,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s for {context}")


def _content_text(app) -> str:
    try:
        content = app.screen.query_one("#screen-content")
    except NoMatches:
        return ""
    pieces: list[str] = []
    for widget in content.query(Static):
        pieces.append(str(widget.renderable))
    for widget in content.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(piece for piece in pieces if piece.strip())


def _assert_core_render_is_healthy(app, *, route: str) -> None:
    text = _content_text(app)
    assert text.strip(), f"{route} rendered empty content"
    for broken_text in BROKEN_TEXT_PATTERNS:
        assert broken_text not in text
    assert RAW_OBJECT_REPR.search(text) is None
    try:
        _assert_no_local_path_prefixes(text)
    except AssertionError as exc:
        raise AssertionError(f"{route} leaked a local path in rendered content:\n{text}") from exc

    svg = app.export_screenshot(title=f"Latest dev core smoke {route}", simplify=True)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert len(svg) > 1_000
    for broken_text in BROKEN_TEXT_PATTERNS:
        assert broken_text not in svg
    assert RAW_OBJECT_REPR.search(svg) is None
    try:
        _assert_no_local_path_prefixes(svg)
    except AssertionError as exc:
        raise AssertionError(f"{route} leaked a local path in exported screenshot") from exc


@pytest.mark.asyncio
async def test_latest_dev_core_first_use_routes_exclude_sync_and_persona(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    app = _build_clean_first_run_app(monkeypatch, tmp_path)
    visited_routes: list[str] = []

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == TAB_HOME and app.screen.__class__.__name__ == "HomeScreen",
                context="initial Home route",
            )

            for route, expected_tab, expected_screen_name, required_copy in CORE_FIRST_USE_ROUTES:
                if route != TAB_HOME:
                    await app.handle_screen_navigation(NavigateToScreen(route))
                    await _wait_until(
                        pilot,
                        lambda expected_tab=expected_tab, expected_screen_name=expected_screen_name: (
                            app.current_tab == expected_tab
                            and app.screen.__class__.__name__ == expected_screen_name
                        ),
                        context=f"{route} navigation",
                    )

                await _wait_until(
                    pilot,
                    lambda required_copy=required_copy: all(
                        copy in _content_text(app) for copy in required_copy
                    ),
                    context=f"{route} expected copy",
                )
                _assert_core_render_is_healthy(app, route=route)
                visited_routes.append(route)

            assert "personas" not in visited_routes
            assert "sync" not in visited_routes


@pytest.mark.asyncio
async def test_home_console_model_setup_routes_to_settings_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    app = _build_clean_first_run_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == TAB_HOME and app.screen.__class__.__name__ == "HomeScreen",
                context="initial Home route",
            )

            app.screen.query_one("#home-primary-action", Button).press()
            await _wait_until(
                pilot,
                lambda: (
                    app.current_tab == TAB_SETTINGS
                    and app.screen.__class__.__name__ == "SettingsScreen"
                ),
                context="Home model setup Settings route",
            )
            await _wait_until(
                pilot,
                lambda: (
                    getattr(app.screen, "active_category", None)
                    == SettingsCategoryId.PROVIDERS_MODELS.value
                    and "Provider readiness" in _content_text(app)
                ),
                context="Settings Providers & Models category",
            )
            await _wait_until(
                pilot,
                lambda: bool(app.screen.query("#settings-provider-value SelectOverlay")),
                context="Settings provider Select overlay mounted",
            )

            _assert_core_render_is_healthy(app, route="home model setup")


@pytest.mark.asyncio
async def test_latest_dev_library_to_console_staged_context_smoke() -> None:
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = TAB_HOME
    payload = _core_loop_payload()

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == TAB_HOME,
                context="Home initial route",
            )

            app.open_chat_with_handoff(payload)
            await _wait_until(
                pilot,
                lambda: app.current_tab == TAB_CHAT and app.screen.__class__.__name__ == "ChatScreen",
                context="Console route after Library/RAG handoff",
            )
            await _wait_until(
                pilot,
                lambda: "Sources: 1 staged" in _content_text(app),
                context="staged source count",
            )

            text = _content_text(app)
            assert "RAG: on" in text
            assert "Live work: Transcript chunk: Agentic terminal design" in text
            assert "Evidence: 1/1 available" in text

            composer = app.screen.query_one("#console-native-composer", ConsoleComposerBar)
            assert payload.suggested_prompt in composer.draft_text()
            _assert_core_render_is_healthy(app, route="library to console")
