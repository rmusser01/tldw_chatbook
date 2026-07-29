from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import time
from typing import Any

import pytest
from textual.widgets import Input, TabbedContent

import tldw_chatbook.app as app_module
from Tests.reactive_ownership_contract import RETIRED_TLDW_REACTIVES
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_INGEST,
    LIBRARY_NAV_CONTEXT_MODE,
    TAB_CHAT,
    TAB_EVALS,
    TAB_LIBRARY,
    TAB_LLM,
    TAB_MCP,
    TAB_MEDIA,
    TAB_PERSONAS,
    TAB_SEARCH,
    TAB_SETTINGS,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_INGEST_MEDIA,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen
from tldw_chatbook.UI.Screens.home_screen import HomeScreen
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen
from tldw_chatbook.UI.Screens.media_screen import MediaScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.search_screen import SearchScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_APP_TEST_ROOT = PROJECT_ROOT / "Tests" / "ProductionApp"
LEGACY_HARNESS_MODULES = frozenset(
    {
        "Tests.textual_test_harness",
        "Tests.textual_test_utils",
    }
)
ROUTE_SPECS = (
    ("llm", TAB_LLM, LLMScreen),
    ("chat", TAB_CHAT, ChatScreen),
    ("personas", TAB_PERSONAS, PersonasScreen),
    ("library", TAB_LIBRARY, LibraryScreen),
    ("media", TAB_MEDIA, MediaScreen),
    ("search", TAB_SEARCH, SearchScreen),
    ("ingest", TAB_LIBRARY, LibraryScreen),
    ("mcp", TAB_MCP, MCPScreen),
    ("evals", TAB_EVALS, EvalsScreen),
    ("settings", TAB_SETTINGS, SettingsScreen),
)
CHAT_SESSION_TITLE = "TASK-906 Console owner"
PERSONAS_QUERY = "TASK-906 personas owner"
LIBRARY_QUERY = "TASK-906 library owner"
MEDIA_QUERY = "TASK-906 media owner"
SEARCH_QUERY = "TASK-906 search owner"
SETTINGS_QUERY = "TASK-906 settings owner"
SECRET_SNAPSHOT_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credentials",
        "headers",
        "password",
        "secret",
        "token",
    }
)
CONTENT_SNAPSHOT_KEYS = frozenset(
    {
        "content",
        "draft",
        "pinned_prefill",
        "system_prompt",
        "user_prompt",
    }
)
DEFAULT_SCREEN_WAIT_SECONDS = 30.0
SCREEN_WAIT_SECONDS_ENV = "TLDW_TEST_SCREEN_WAIT_SECONDS"


def _production_app(monkeypatch: pytest.MonkeyPatch) -> TldwCli:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)
    app = TldwCli()
    app.app_config["_first_run"] = False
    return app


def _screen_wait_seconds() -> float:
    """Return the validated real-app readiness deadline for this test run."""
    raw_value = os.environ.get(
        SCREEN_WAIT_SECONDS_ENV,
        str(DEFAULT_SCREEN_WAIT_SECONDS),
    )
    try:
        seconds = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{SCREEN_WAIT_SECONDS_ENV} must be a positive finite number"
        ) from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{SCREEN_WAIT_SECONDS_ENV} must be a positive finite number")
    return seconds


async def _wait_until(pilot: Any, predicate, failure: str) -> None:
    deadline = time.monotonic() + _screen_wait_seconds()
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError(failure)


async def _wait_for_screen(
    app: TldwCli,
    pilot: Any,
    screen_type: type,
    canonical_tab: str,
    *,
    previous_screen: object | None = None,
):
    await _wait_until(
        pilot,
        lambda: (
            type(app.screen) is screen_type
            and app.current_tab == canonical_tab
            and app.screen is not previous_screen
            and app.screen.is_mounted
        ),
        f"production TldwCli did not route to exact {screen_type.__name__}",
    )
    return app.screen


def _navigation_message(route: str) -> NavigateToScreen:
    if route == "ingest":
        return NavigateToScreen(
            route,
            {LIBRARY_NAV_CONTEXT_INGEST: True},
        )
    return NavigateToScreen(route)


async def _exercise_route(route: str, screen: object, pilot: Any) -> None:
    if route == "llm":
        assert type(screen) is LLMScreen
        await _wait_until(
            pilot,
            lambda: (
                screen.llm_window is not None
                and screen.llm_window.is_mounted
                and screen.llm_window.active_view == "llama-cpp"
            ),
            "production Models body did not finish mounting",
        )
        screen.llm_window.active_view = "ollama"
        await pilot.pause()
        assert screen.llm_window.active_view == "ollama"
    elif route == "chat":
        assert type(screen) is ChatScreen
        store = screen._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        store.rename_session(session_id, CHAT_SESSION_TITLE)
        assert store.ensure_session().title == CHAT_SESSION_TITLE
    elif route == "personas":
        assert type(screen) is PersonasScreen
        screen.query_one("#personas-library-search", Input).value = PERSONAS_QUERY
        await pilot.pause()
        assert screen.state.search_query == PERSONAS_QUERY
    elif route == "library":
        assert type(screen) is LibraryScreen
        screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_MODE: "notes"})
        screen._library_rag_query = LIBRARY_QUERY
        await pilot.pause()
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
    elif route == "media":
        assert type(screen) is MediaScreen
        await _wait_until(
            pilot,
            lambda: screen.media_window is not None,
            "production Media owner did not mount",
        )
        screen.media_window.active_media_type = "all-media"
        screen.media_window.runtime_state.active_media_type = "all-media"
        screen.media_window.search_panel.search_term = MEDIA_QUERY
        assert screen.media_window.search_panel.search_term == MEDIA_QUERY
    elif route == "search":
        assert type(screen) is SearchScreen
        screen.query_one("#search-query-input", Input).value = SEARCH_QUERY
        screen.query_one("#search-tabs", TabbedContent).active = "history-tab"
        await pilot.pause()
        assert screen.query_one("#search-query-input", Input).value == SEARCH_QUERY
    elif route == "ingest":
        assert type(screen) is LibraryScreen
        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
    elif route == "mcp":
        assert type(screen) is MCPScreen
        assert screen.workbench is not None
        screen.action_mcp_mode("permissions")
        await pilot.pause()
        assert screen.workbench.active_mode == "permissions"
    elif route == "evals":
        assert type(screen) is EvalsScreen
        screen.select(kind="classic", id=None)
        await pilot.pause()
        assert screen._selection.kind == "classic"
    elif route == "settings":
        assert type(screen) is SettingsScreen
        screen._submit_category_search(SETTINGS_QUERY)
        await pilot.pause()
        assert screen.category_search_query == SETTINGS_QUERY
    else:
        raise AssertionError(f"unhandled maturity route: {route}")


async def _assert_restored_route(route: str, screen: object, pilot: Any) -> None:
    if route == "llm":
        assert type(screen) is LLMScreen
        await _wait_until(
            pilot,
            lambda: screen.llm_window is not None,
            "restored Models body did not mount",
        )
    elif route == "chat":
        assert type(screen) is ChatScreen
        assert (
            screen._ensure_console_chat_store().ensure_session().title
            == CHAT_SESSION_TITLE
        )
    elif route == "personas":
        assert type(screen) is PersonasScreen
        assert screen.state.search_query == PERSONAS_QUERY
    elif route == "library":
        assert type(screen) is LibraryScreen
        assert screen._library_rag_query == LIBRARY_QUERY
        assert screen._library_selected_row_id in {
            LIBRARY_ROW_BROWSE_NOTES,
            LIBRARY_ROW_INGEST_MEDIA,
        }
    elif route == "media":
        assert type(screen) is MediaScreen
        await _wait_until(
            pilot,
            lambda: (
                screen.media_window is not None
                and screen.media_window.active_media_type == "all-media"
                and screen.media_window.search_panel.search_term == MEDIA_QUERY
            ),
            "fresh Media owner did not restore its snapshot",
        )
    elif route == "search":
        assert type(screen) is SearchScreen
        assert screen.query_one("#search-query-input", Input).value == SEARCH_QUERY
        assert screen.query_one("#search-tabs", TabbedContent).active == "history-tab"
    elif route == "ingest":
        assert type(screen) is LibraryScreen
        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
    elif route == "mcp":
        assert type(screen) is MCPScreen
        assert screen.workbench is not None
        await _wait_until(
            pilot,
            lambda: screen.workbench.active_mode == "permissions",
            "fresh MCP owner did not restore its view state",
        )
    elif route == "evals":
        assert type(screen) is EvalsScreen
    elif route == "settings":
        assert type(screen) is SettingsScreen
        assert screen.category_search_query == SETTINGS_QUERY
    else:
        raise AssertionError(f"unhandled restored maturity route: {route}")


def _snapshot_violations(value: object, path: str = "$") -> list[str]:
    if value is None or type(value) in {bool, int, float, str}:
        return []
    if type(value) is dict:
        violations: list[str] = []
        for key, child in value.items():
            if not isinstance(key, str):
                violations.append(f"{path}: non-string key {type(key).__name__}")
                continue
            if key in RETIRED_TLDW_REACTIVES:
                violations.append(f"{path}.{key}: retired root name")
            if key in SECRET_SNAPSHOT_KEYS and not _empty_snapshot_value(child):
                violations.append(f"{path}.{key}: secret-bearing value")
            if key in CONTENT_SNAPSHOT_KEYS and not _empty_snapshot_value(child):
                violations.append(f"{path}.{key}: prompt or generated content")
            violations.extend(_snapshot_violations(child, f"{path}.{key}"))
        return violations
    if type(value) in {list, tuple, set, frozenset}:
        return [
            violation
            for index, child in enumerate(value)
            for violation in _snapshot_violations(child, f"{path}[{index}]")
        ]
    return [f"{path}: non-primitive {type(value).__module__}.{type(value).__name__}"]


def _empty_snapshot_value(value: object) -> bool:
    return (
        value is None
        or value == ""
        or (type(value) in {dict, list, tuple, set, frozenset} and not value)
    )


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _surrogate_test_violations(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[tuple[int, str]] = []
    substitute_classes: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            base_names = {_dotted_name(base).rsplit(".", 1)[-1] for base in node.bases}
            if any(base_name.endswith(("App", "Screen")) for base_name in base_names):
                substitute_classes.add(node.name)
                violations.append((node.lineno, f"surrogate class {node.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if module in LEGACY_HARNESS_MODULES:
                violations.append((node.lineno, f"legacy harness import {module}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in LEGACY_HARNESS_MODULES:
                    violations.append(
                        (node.lineno, f"legacy harness import {alias.name}")
                    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _dotted_name(node.func)
        short_name = call_name.rsplit(".", 1)[-1]
        if short_name in {"SimpleNamespace", "MagicMock"}:
            violations.append((node.lineno, f"app substitute call {short_name}"))
        if (
            call_name == "object.__new__"
            and node.args
            and _dotted_name(node.args[0]).rsplit(".", 1)[-1] == "TldwCli"
        ):
            violations.append((node.lineno, "object.__new__(TldwCli)"))
        if call_name.startswith("TldwCli."):
            violations.append((node.lineno, f"unbound call {call_name}"))
        if short_name in substitute_classes:
            violations.append((node.lineno, f"surrogate constructor {short_name}"))

    return violations


def test_production_app_tests_contain_no_surrogate_application_patterns() -> None:
    """Keep production ownership coverage on real routes or pure functions."""
    violations = {
        str(path.relative_to(PROJECT_ROOT)): path_violations
        for path in sorted(PRODUCTION_APP_TEST_ROOT.glob("*.py"))
        if (path_violations := _surrogate_test_violations(path))
    }

    assert violations == {}


def test_snapshot_guard_accepts_only_reviewed_builtin_containers() -> None:
    """Keep the memory-only snapshot allowlist explicit and recursive."""
    assert (
        _snapshot_violations(
            {
                "scope": {"notes"},
                "nested": (1, [None, frozenset({"media"})]),
            }
        )
        == []
    )
    assert _snapshot_violations({"owner": object()}) == [
        "$.owner: non-primitive builtins.object"
    ]
    assert _snapshot_violations({"draft": "private prompt"}) == [
        "$.draft: prompt or generated content"
    ]
    assert _snapshot_violations({"api_key": "private key"}) == [
        "$.api_key: secret-bearing value"
    ]


def test_screen_wait_seconds_validates_the_ci_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept positive finite wait overrides and reject invalid values."""
    monkeypatch.delenv("TLDW_TEST_SCREEN_WAIT_SECONDS", raising=False)
    assert _screen_wait_seconds() == DEFAULT_SCREEN_WAIT_SECONDS

    monkeypatch.setenv("TLDW_TEST_SCREEN_WAIT_SECONDS", "12.5")
    assert _screen_wait_seconds() == 12.5

    for invalid in ("0", "-1", "nan", "inf", "not-a-number"):
        monkeypatch.setenv("TLDW_TEST_SCREEN_WAIT_SECONDS", invalid)
        with pytest.raises(ValueError, match="TLDW_TEST_SCREEN_WAIT_SECONDS"):
            _screen_wait_seconds()


@pytest.mark.asyncio
async def test_registered_routes_use_fresh_production_owners_and_safe_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every changed route twice through the normal production app."""
    for route, canonical_tab, screen_type in ROUTE_SPECS:
        assert resolve_screen_target(route) == (
            "library" if route == "ingest" else route,
            canonical_tab,
            screen_type,
        )

    app = _production_app(monkeypatch)
    original_screens: dict[str, object] = {}

    async with app.run_test(size=(180, 55)) as pilot:
        for route, canonical_tab, screen_type in ROUTE_SPECS:
            app.post_message(_navigation_message(route))
            screen = await _wait_for_screen(
                app,
                pilot,
                screen_type,
                canonical_tab,
            )
            original_screens[route] = screen
            await _exercise_route(route, screen, pilot)
            assert all(not hasattr(app, name) for name in RETIRED_TLDW_REACTIVES)

        app.post_message(NavigateToScreen("home"))
        await _wait_for_screen(app, pilot, HomeScreen, "home")

        for route, canonical_tab, screen_type in ROUTE_SPECS:
            app.post_message(_navigation_message(route))
            screen = await _wait_for_screen(
                app,
                pilot,
                screen_type,
                canonical_tab,
                previous_screen=original_screens[route],
            )
            await _assert_restored_route(route, screen, pilot)
            assert all(not hasattr(app, name) for name in RETIRED_TLDW_REACTIVES)

        runtime_identity = app._current_runtime_identity()
        snapshot_violations: dict[str, list[str]] = {}
        for canonical_tab in {spec[1] for spec in ROUTE_SPECS}:
            snapshot = app.screen_state_store.restore(
                canonical_tab,
                runtime_identity,
            )
            if snapshot is None:
                continue
            violations = _snapshot_violations(snapshot)
            if violations:
                snapshot_violations[canonical_tab] = violations

        assert snapshot_violations == {}
