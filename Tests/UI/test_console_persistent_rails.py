"""Mounted Console persistent rail first-start contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat import console_chat_store as console_chat_store_module
from tldw_chatbook.Chat.console_chat_models import ConsoleWorkspaceContext
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import TabState


def _is_displayed(widget) -> bool:
    current = widget
    while current is not None:
        if current.display is False or current.styles.display == "none":
            return False
        current = getattr(current, "parent", None)
    return True


def _assert_selector_hidden_or_absent(screen, selector: str) -> None:
    matches = list(screen.query(selector))
    assert not matches or all(not _is_displayed(widget) for widget in matches)


def _static_text(widget) -> str:
    renderable = getattr(widget, "renderable", "")
    return getattr(renderable, "plain", str(renderable))


async def _wait_for_badge(screen, pilot, selector: str, expected: str) -> str:
    for _ in range(20):
        matches = list(screen.query(selector))
        for widget in matches:
            if _is_displayed(widget):
                text = _static_text(widget)
                if expected in text:
                    return text
        await pilot.pause(0.05)
    visible_text = " ".join(
        _static_text(widget) for widget in screen.query(selector) if _is_displayed(widget)
    )
    raise AssertionError(
        f"{selector} did not include {expected!r}; visible badge text={visible_text!r}"
    )


async def _wait_for_displayed(screen, pilot, selector: str):
    for _ in range(20):
        widget = screen.query_one(selector)
        if _is_displayed(widget):
            return widget
        await pilot.pause(0.05)
    raise AssertionError(f"{selector} was not displayed")


async def _wait_for_hidden(screen, pilot, selector: str) -> None:
    for _ in range(20):
        if not any(_is_displayed(widget) for widget in screen.query(selector)):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"{selector} was still displayed")


async def _wait_for_main_column_width_change(
    screen,
    pilot,
    *,
    original_width: int,
    direction: str,
) -> int:
    for _ in range(20):
        width = screen.query_one("#console-main-column").region.width
        if direction == "increase" and width > original_width:
            return width
        if direction == "decrease" and width < original_width:
            return width
        await pilot.pause(0.05)
    raise AssertionError(
        f"main column width did not {direction}; original={original_width}, "
        f"current={screen.query_one('#console-main-column').region.width}"
    )


async def _wait_for_native_console_session(screen, pilot):
    for _ in range(20):
        store = getattr(screen, "_console_chat_store", None)
        if store is not None and store.active_session_id is not None:
            return store.ensure_session(workspace_id=store.workspace_context.active_workspace_id)
        await pilot.pause(0.05)
    raise AssertionError("native Console store did not expose an active session")


class _FixedUuid:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


def test_generated_console_stylesheet_includes_rail_rules():
    stylesheet = Path("tldw_chatbook/css/tldw_cli_modular.tcss")
    css = stylesheet.read_text(encoding="utf-8")

    for selector in (
        "#console-right-rail",
        ".console-rail-handle",
        ".console-rail-header",
        ".console-rail-collapse-button",
    ):
        assert selector in css


@pytest.mark.asyncio
async def test_console_first_start_renders_left_rail_and_right_handle():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        assert _is_displayed(console.query_one("#console-left-rail"))
        assert _is_displayed(console.query_one("#console-staged-context-tray"))
        assert _is_displayed(console.query_one("#console-workspace-context"))
        _assert_selector_hidden_or_absent(console, "#console-context-rail-handle")
        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        _assert_selector_hidden_or_absent(console, "#console-run-inspector-state")
        _assert_selector_hidden_or_absent(
            console,
            "#console-live-work-source-readiness",
        )
        assert _is_displayed(console.query_one("#console-inspector-rail-handle"))
        assert "Inspector" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_first_start_does_not_create_rail_state_config_on_read():
    app = _build_test_app()
    console_config = app.app_config.setdefault("console", {})
    console_config.pop("rail_state", None)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

    assert "rail_state" not in console_config


@pytest.mark.asyncio
async def test_console_first_start_right_handle_is_focusable():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-open")

        for _ in range(80):
            focused = console.focused
            if getattr(focused, "id", None) == "console-inspector-rail-open":
                assert isinstance(focused, Button)
                return
            await pilot.press("tab")

        focused_id = getattr(console.focused, "id", None)
        raise AssertionError(
            "console-inspector-rail-open was not reachable by tab; "
            f"focused={focused_id!r}"
        )


@pytest.mark.asyncio
async def test_console_context_rail_collapse_hides_left_rail_and_expands_main_column():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-collapse")
        first_start_width = console.query_one("#console-main-column").region.width

        await pilot.click("#console-context-rail-collapse")

        await _wait_for_hidden(console, pilot, "#console-left-rail")
        await _wait_for_hidden(console, pilot, "#console-staged-context-tray")
        await _wait_for_hidden(console, pilot, "#console-workspace-context")
        assert _is_displayed(console.query_one("#console-context-rail-handle"))
        assert (
            await _wait_for_main_column_width_change(
                console,
                pilot,
                original_width=first_start_width,
                direction="increase",
            )
        ) > first_start_width


@pytest.mark.asyncio
async def test_console_inspector_rail_open_restores_right_rail_and_narrows_main_column():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-open")
        right_collapsed_width = console.query_one("#console-main-column").region.width

        await pilot.click("#console-inspector-rail-open")

        await _wait_for_displayed(console, pilot, "#console-right-rail")
        assert _is_displayed(console.query_one("#console-run-inspector-state"))
        assert _is_displayed(console.query_one("#console-live-work-source-readiness"))
        await _wait_for_hidden(console, pilot, "#console-inspector-rail-handle")
        assert (
            await _wait_for_main_column_width_change(
                console,
                pilot,
                original_width=right_collapsed_width,
                direction="decrease",
            )
        ) < right_collapsed_width


@pytest.mark.asyncio
async def test_console_rail_state_persists_by_workspace_session_key(monkeypatch):
    app = _build_test_app()
    app.app_config = {"console": {"rail_state": {}}}
    saved_settings = []

    def fake_save_setting(section, key, value):
        saved_settings.append((section, key, value))
        return True

    monkeypatch.setattr(
        chat_screen_module,
        "save_setting_to_cli_config",
        fake_save_setting,
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        session = await _wait_for_native_console_session(console, pilot)
        await _wait_for_selector(console, pilot, "#console-context-rail-collapse")
        await pilot.click("#console-context-rail-collapse")
        await _wait_for_hidden(console, pilot, "#console-left-rail")
        await pilot.click("#console-inspector-rail-open")
        await _wait_for_displayed(console, pilot, "#console-right-rail")

    rail_state = app.app_config["console"]["rail_state"]
    expected_key = f"console_rail_state:global:{session.id}"
    assert rail_state[expected_key] == {
        "left_open": False,
        "right_open": True,
    }
    assert saved_settings[-1] == (
        "console.rail_state",
        expected_key,
        {"left_open": False, "right_open": True},
    )

    app.console_rail_session_id = session.id
    remounted_host = ConsoleHarness(app)
    async with remounted_host.run_test(size=(180, 48)) as pilot:
        console = remounted_host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")

        _assert_selector_hidden_or_absent(console, "#console-left-rail")
        assert _is_displayed(console.query_one("#console-context-rail-handle"))
        assert _is_displayed(console.query_one("#console-right-rail"))
        _assert_selector_hidden_or_absent(console, "#console-inspector-rail-handle")


@pytest.mark.asyncio
async def test_console_rail_state_uses_workspace_session_specific_keys(monkeypatch):
    app = _build_test_app()
    app.app_config = {
        "console": {
            "rail_state": {
                "console_rail_state:workspace-a:session-a": {
                    "left_open": False,
                    "right_open": True,
                }
            }
        }
    }

    def workspace_a(self):
        return ConsoleWorkspaceContext(active_workspace_id="workspace-a")

    def workspace_b(self):
        return ConsoleWorkspaceContext(active_workspace_id="workspace-b")

    monkeypatch.setattr(ChatScreen, "_current_console_workspace_context", workspace_a)
    monkeypatch.setattr(
        ChatScreen,
        "_current_console_session_id",
        lambda self: "session-a",
        raising=False,
    )
    host_a = ConsoleHarness(app)
    async with host_a.run_test(size=(180, 48)) as pilot:
        console = host_a.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")

        _assert_selector_hidden_or_absent(console, "#console-left-rail")
        assert _is_displayed(console.query_one("#console-right-rail"))

    monkeypatch.setattr(ChatScreen, "_current_console_workspace_context", workspace_b)
    monkeypatch.setattr(
        ChatScreen,
        "_current_console_session_id",
        lambda self: "session-b",
        raising=False,
    )
    host_b = ConsoleHarness(app)
    async with host_b.run_test(size=(180, 48)) as pilot:
        console = host_b.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        assert _is_displayed(console.query_one("#console-left-rail"))
        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        assert _is_displayed(console.query_one("#console-inspector-rail-handle"))


@pytest.mark.asyncio
async def test_console_session_preference_copies_to_durable_conversation_key(monkeypatch):
    app = _build_test_app()
    app.app_config = {
        "console": {
            "rail_state": {
                "console_rail_state:global:session-1": {
                    "left_open": False,
                    "right_open": True,
                }
            }
        }
    }

    monkeypatch.setattr(
        chat_screen_module,
        "save_setting_to_cli_config",
        lambda section, key, value: True,
        raising=False,
    )
    monkeypatch.setattr(
        console_chat_store_module,
        "uuid4",
        lambda: _FixedUuid("session-1"),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        session = await _wait_for_native_console_session(console, pilot)
        assert session.id == "session-1"
        session.persisted_conversation_id = "conv-1"
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")

    rail_state = app.app_config["console"]["rail_state"]
    assert rail_state["console_rail_state:global:session-1"] == {
        "left_open": False,
        "right_open": True,
    }
    assert rail_state["console_rail_state:global:conv-1"] == {
        "left_open": False,
        "right_open": True,
    }


@pytest.mark.asyncio
async def test_console_rail_key_prefers_native_session_over_legacy_conversation(monkeypatch):
    app = _build_test_app()
    app.app_config = {"console": {"rail_state": {}}}

    monkeypatch.setattr(
        chat_screen_module,
        "save_setting_to_cli_config",
        lambda section, key, value: True,
        raising=False,
    )
    monkeypatch.setattr(
        console_chat_store_module,
        "uuid4",
        lambda: _FixedUuid("session-native"),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        console.chat_state.add_tab(
            TabState(
                tab_id="legacy",
                title="Legacy",
                conversation_id="legacy-conv",
                is_active=True,
            )
        )
        console.chat_state.active_tab_id = "legacy"
        session = await _wait_for_native_console_session(console, pilot)
        assert session.persisted_conversation_id is None

        await pilot.click("#console-context-rail-collapse")
        await _wait_for_hidden(console, pilot, "#console-left-rail")

        rail_state = app.app_config["console"]["rail_state"]
        assert rail_state["console_rail_state:global:session-native"] == {
            "left_open": False,
            "right_open": False,
        }
        assert "console_rail_state:global:legacy-conv" not in rail_state

        session.persisted_conversation_id = "native-conv"
        console._sync_console_rail_visibility(console._current_console_rail_state())

    rail_state = app.app_config["console"]["rail_state"]
    assert rail_state["console_rail_state:global:native-conv"] == {
        "left_open": False,
        "right_open": False,
    }
    assert "console_rail_state:global:legacy-conv" not in rail_state


@pytest.mark.asyncio
async def test_console_rail_fallback_migration_read_path_does_not_create_empty_state(
    monkeypatch,
):
    app = _build_test_app()
    console_config = app.app_config.setdefault("console", {})
    console_config.pop("rail_state", None)

    monkeypatch.setattr(
        console_chat_store_module,
        "uuid4",
        lambda: _FixedUuid("session-no-fallback"),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        session = await _wait_for_native_console_session(console, pilot)
        session.persisted_conversation_id = "conv-no-fallback"
        console._current_console_rail_state()

    assert "rail_state" not in console_config


@pytest.mark.asyncio
async def test_console_provider_blocked_badge_does_not_auto_open_inspector():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4.1-2025-04-14"},
        "api_settings": {"openai": {"api_key": ""}},
        "console": {
            "rail_state": {
                "console_rail_state:global:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        },
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        assert "blocked" in await _wait_for_badge(
            console,
            pilot,
            "#console-inspector-rail-badge",
            "blocked",
        )
        assert _is_displayed(console.query_one("#console-provider-recovery-strip"))
        settings_button = console.query_one("#console-open-provider-settings", Button)
        assert _is_displayed(settings_button)
        assert settings_button.disabled is False


@pytest.mark.asyncio
async def test_console_provider_ready_with_missing_model_uses_model_recovery_copy(
    monkeypatch,
):
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": ""},
        "console": {
            "rail_state": {
                "console_rail_state:global:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = ""
    monkeypatch.setattr(
        ChatScreen,
        "_effective_console_provider_model",
        lambda self: ("llama_cpp", ""),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        inspector_state = console._build_console_inspector_state(None)
        provider_row = next(
            row for row in inspector_state.rows if row.label == "Provider"
        )
        assert provider_row.status == "blocked"
        assert "Select a model before sending." in provider_row.recovery
        assert "is ready" not in provider_row.recovery


@pytest.mark.asyncio
async def test_console_failed_badge_takes_priority_over_provider_blocked():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4.1-2025-04-14"},
        "api_settings": {"openai": {"api_key": ""}},
        "console": {
            "rail_state": {
                "console_rail_state:global:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        },
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    app.console_run_status_override = "failed"
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        badge = await _wait_for_badge(
            console,
            pilot,
            "#console-inspector-rail-badge",
            "failed",
        )
        assert "blocked" not in badge
        _assert_selector_hidden_or_absent(console, "#console-right-rail")


@pytest.mark.asyncio
async def test_console_pending_approval_badge_does_not_auto_open_inspector():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "console": {
            "rail_state": {
                "console_rail_state:global:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        }
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.console_pending_approval_count = 1
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        assert "1 approval" in await _wait_for_badge(
            console,
            pilot,
            "#console-inspector-rail-badge",
            "1 approval",
        )
        _assert_selector_hidden_or_absent(console, "#console-right-rail")


@pytest.mark.asyncio
async def test_console_tool_badge_when_no_higher_priority_inspector_badge():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "console": {
            "rail_state": {
                "console_rail_state:global:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        }
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.console_tool_count = 2
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        assert "tools" in await _wait_for_badge(
            console,
            pilot,
            "#console-inspector-rail-badge",
            "tools",
        )
        _assert_selector_hidden_or_absent(console, "#console-right-rail")


@pytest.mark.asyncio
async def test_console_left_staged_context_badge_does_not_auto_open_context():
    app = _build_test_app()
    app.console_rail_session_id = "badge-session"
    app.app_config = {
        "console": {
            "rail_state": {
                "console_rail_state:global:badge-session": {
                    "left_open": False,
                    "right_open": False,
                }
            }
        }
    }
    app.pending_console_launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="RAG result",
        payload={"query": "badge sync"},
        status="ready",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")

        _assert_selector_hidden_or_absent(console, "#console-left-rail")
        badge = await _wait_for_badge(
            console,
            pilot,
            "#console-context-rail-badge",
            "staged",
        )
        assert badge in {"1 staged", "staged"}


@pytest.mark.asyncio
async def test_console_badge_state_update_after_mount_does_not_auto_open_inspector():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "console": {
            "rail_state": {
                "console_rail_state:global:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        }
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")
        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        _assert_selector_hidden_or_absent(console, "#console-inspector-rail-badge")

        app.console_pending_approval_count = 1
        console._sync_console_control_bar()
        await pilot.pause(0.05)

        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        assert "1 approval" in await _wait_for_badge(
            console,
            pilot,
            "#console-inspector-rail-badge",
            "1 approval",
        )
