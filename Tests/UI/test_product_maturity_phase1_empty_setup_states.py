"""Product maturity Phase 1.6 empty/error/setup-state coverage contract."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    RaisingLibraryNotesScopeService,
    RaisingSkillsScopeService,
    RaisingWatchlistsScopeService,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticReadItLaterScopeService,
    _active_destination_screen,
    _visible_text,
    _wait_for_library_snapshot,
    _wait_for_personas_snapshot,
    _wait_for_skills_snapshot,
    _wait_for_wc_snapshot,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Views.RAGSearch import search_rag_window as search_rag_module
from tldw_chatbook.UI.Views.RAGSearch.search_rag_window import SearchRAGWindow


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-6-empty-error-setup-states.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path("backlog/tasks/task-8.6 - Product-Maturity-Phase-1.6-Empty-Error-Setup-State-Coverage.md")
LOCAL_PATH_PREFIXES = (
    "/Users/",
    "/home/",
    "/var/home/",
    "/private/var/folders/",
    "C:\\Users\\",
    "C:/Users/",
)


class _WidgetHost(App):
    def __init__(self, widget) -> None:
        super().__init__()
        self.widget_under_test = widget

    def compose(self) -> ComposeResult:
        yield self.widget_under_test


class _FakeAppInstance:
    def __init__(self) -> None:
        self.notifications = []

    def notify(self, message, *args, **kwargs) -> None:
        self.notifications.append((message, kwargs))

    def get_authoritative_runtime_source(self) -> str:
        return "local"


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _assert_no_local_path_prefixes(text: str) -> None:
    leaked_prefixes = [prefix for prefix in LOCAL_PATH_PREFIXES if prefix in text]
    assert not leaked_prefixes, f"evidence contains local filesystem prefix(es): {leaked_prefixes}"


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


def _test_chat_window_setting(section: str, key: str, default=None):
    if section == "chat_defaults" and key == "enable_tabs":
        return False
    return _test_cli_setting(section, key, default)


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
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[chat_defaults]\n"
        'provider = "OpenAI"\n'
        'model = "gpt-4o"\n'
        "enable_tabs = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))


def _build_clean_setup_state_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"
    app.providers_models = {}
    return app


def _screen_text(app) -> str:
    pieces: list[str] = []
    for widget in app.screen.query(Static):
        pieces.append(str(widget.renderable))
    for widget in app.screen.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(piece for piece in pieces if piece.strip())


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
        await pilot.pause()
        await asyncio.sleep(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s for {context}")


@pytest.mark.asyncio
async def test_clean_run_setup_and_runtime_blockers_expose_recovery_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_setup_state_app(monkeypatch, tmp_path)

    with (
        patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting),
        patch("tldw_chatbook.UI.Chat_Window_Enhanced.get_cli_setting", side_effect=_test_chat_window_setting),
    ):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
                context="home initial setup state",
            )

            home_text = _screen_text(app)
            assert "Model: Blocked" in home_text
            assert "Set up Console model" in home_text
            assert "Console needs a working model before live AI tasks." in home_text

            await app.handle_screen_navigation(NavigateToScreen("chat"))
            await _wait_until(
                pilot,
                lambda: app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen",
                context="console setup route",
            )
            # In a clean run the default model may be preselected, but the
            # session settings action and setup card remain the
            # recovery/control surfaces.
            await _wait_until(
                pilot,
                lambda: "Choose model" in _screen_text(app)
                or "Open Settings" in _screen_text(app),
                context="console provider setup controls",
            )
            assert (
                app.screen._console_provider_blocker_copy()
                == "Provider setup needed: choose a model"
            )
            # The shared Workbench recovery banner stays hidden — the setup
            # card's action button is the recovery/control surface now
            # (Phase 2 spec, section 2).
            recovery_callout = app.screen.query("#workbench-recovery-callout")
            assert recovery_callout and recovery_callout[0].display is False
            recovery_action = app.screen.query_one("#workbench-recovery-action", Button)
            assert recovery_action.display is False
            card_action = app.screen.query_one("#console-empty-choose-model", Button)
            assert card_action.display is True
            assert str(card_action.label) == "Choose model"
            assert not list(app.screen.query("#console-open-provider-settings"))
            assert "More: Ctrl+P" in _screen_text(app)

            await app.handle_screen_navigation(NavigateToScreen("acp"))
            await _wait_until(
                pilot,
                lambda: app.current_tab == "acp" and app.screen.__class__.__name__ == "ACPScreen",
                context="acp runtime blocker",
            )
            acp_text = _screen_text(app)
            acp_launch = app.screen.query_one("#acp-launch-agent", Button)
            assert "Runtime not configured" in acp_text
            assert "Why: no ACP-compatible runtime is configured." in acp_text
            assert "Next: Configure ACP runtime setup in ACP before launch." in acp_text
            assert "Owner: ACP runtime." in acp_text
            assert acp_launch.disabled is True
            assert "Configure an ACP-compatible runtime in ACP before launching an ACP agent." in str(acp_launch.tooltip)


@pytest.mark.asyncio
async def test_optional_dependency_missing_state_exposes_owner_and_setup_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(search_rag_module, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setitem(search_rag_module.DEPENDENCIES_AVAILABLE, "embeddings_rag", False)
    monkeypatch.setattr(
        "tldw_chatbook.Utils.widget_helpers.alert_embeddings_not_available",
        lambda widget: None,
    )

    widget = SearchRAGWindow(_FakeAppInstance())
    app = _WidgetHost(widget)

    async with app.run_test() as pilot:
        await pilot.pause()

        recovery = widget.query_one("#search-rag-dependency-missing", Static)
        recovery_text = str(recovery.renderable)
        search_input = widget.query_one("#search-query-input", Input)
        search_button = widget.query_one("#search-button", Button)

        assert "Dependency missing" in recovery_text
        assert "Unavailable: Search/RAG queries." in recovery_text
        assert "Why: Missing optional dependencies: embeddings_rag." in recovery_text
        assert 'pip install -e ".[embeddings_rag]"' in recovery_text
        assert 'pip install "tldw_chatbook[embeddings_rag]"' in recovery_text
        assert "Recovery: Settings > RAG." in recovery_text
        assert "Owner: Library Search/RAG." in recovery_text
        assert search_input.disabled is True
        assert search_button.disabled is True
        assert 'pip install -e ".[embeddings_rag]"' in str(search_button.tooltip)
        assert 'pip install "tldw_chatbook[embeddings_rag]"' in str(search_button.tooltip)


@pytest.mark.parametrize(
    ("route", "button_selector", "expected_copy", "setup"),
    [
        (
            "library",
            "#library-use-in-console",
            "Library source services unavailable; retry Library later.",
            "library-error",
        ),
        (
            "watchlists_collections",
            "#wc-attach-to-console",
            "Watchlists services unavailable; retry Watchlists later.",
            "wc-error",
        ),
        (
            "skills",
            "#skills-attach-to-console",
            "Skills service unavailable; retry Skills later.",
            "skills-error",
        ),
    ],
)
@pytest.mark.asyncio
async def test_service_unavailable_states_disable_false_console_handoffs(
    route: str,
    button_selector: str,
    expected_copy: str,
    setup: str,
) -> None:
    app = _build_test_app()
    if setup == "library-error":
        app.notes_scope_service = RaisingLibraryNotesScopeService()
        app.media_reading_scope_service = StaticLibraryMediaScopeService([])
        app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
        wait_for_snapshot = _wait_for_library_snapshot
    elif setup == "wc-error":
        app.watchlist_scope_service = RaisingWatchlistsScopeService()
        app.collections_feeds_scope_service = StaticReadItLaterScopeService([])
        wait_for_snapshot = _wait_for_wc_snapshot
    elif setup == "skills-error":
        app.skills_scope_service = RaisingSkillsScopeService()
        wait_for_snapshot = _wait_for_skills_snapshot
    else:
        raise AssertionError(f"unexpected setup: {setup}")

    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await wait_for_snapshot(screen, pilot)
        button = screen.query_one(button_selector, Button)

        assert expected_copy in _visible_text(screen)
        assert button.disabled is True
        assert "unavailable" in str(button.tooltip).lower()


@pytest.mark.asyncio
async def test_personas_default_state_disables_false_console_handoff() -> None:
    """Personas starts local-first; Console attach stays blocked until selection."""
    app = _build_test_app()
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_personas_snapshot(screen, pilot)
        button = screen.query_one("#personas-attach-to-console", Button)

        visible_text = _visible_text(screen)
        assert "Console blocked: select an item" in visible_text
        assert button.disabled is True
