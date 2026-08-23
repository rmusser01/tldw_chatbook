import asyncio
import gc
import threading
import time
import weakref
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from loguru import logger as loguru_logger
from textual import events

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.geometry import Region
from textual.widgets import Button, Input, OptionList, Select, Static, TextArea

import tldw_chatbook.UI.Console_Modules.session as session_module
import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text as _screen_visible_text,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
    build_console_settings_summary_state,
    build_default_console_session_settings,
    validate_console_session_settings,
)
from tldw_chatbook.Chat.local_server_discovery import LocalModelProbeResult
from tldw_chatbook.config import (
    API_MODELS_BY_PROVIDER,
    DEFAULT_CONFIG_FROM_TOML,
    RuntimeConfigSnapshot,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    MergedModelEntry,
)
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleFirstChatIntent,
    HandoffChannel,
)
from tldw_chatbook.UI.Screens import provider_model_resolution
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
    ChatScreen,
)
from tldw_chatbook.Widgets.Console import (
    console_settings_summary as settings_summary_module,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS,
    MODAL_BODY_MIN_HEIGHT,
    MODAL_CONTROL_HEIGHT,
    MODEL_DISCOVER_BUTTON_ID,
    MODEL_DISCOVER_STATUS_ID,
    PROVIDER_CHOICE_NO_EFFECT_SUFFIX,
    ConsoleSettingsInput,
    ConsoleSettingsModal,
    ConsoleSettingsResult,
    _is_local_thinking_provider,
    _settings_screen_region,
)
from tldw_chatbook.Widgets.Console.console_settings_summary import (
    ConsoleSettingsSummary,
)
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)
from tldw_chatbook.Widgets.Console.console_system_prompt_modal import (
    APPLY_BUTTON_ID as SYSTEM_PROMPT_APPLY_BUTTON_ID,
    TEXT_AREA_ID as SYSTEM_PROMPT_TEXT_AREA_ID,
)
from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker


class SummaryHarness(ConsolidatedCSSApp):
    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(self, state: ConsoleSettingsSummaryState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield ConsoleSettingsSummary(self.state)


def test_console_settings_screen_region_prefers_absolute_region() -> None:
    absolute_region = Region(10, 20, 30, 1)
    widget = SimpleNamespace(
        region=Region(1, 2, 30, 1),
        screen_region=absolute_region,
    )

    assert _settings_screen_region(widget) == absolute_region


def test_console_settings_screen_region_falls_back_to_mounted_region() -> None:
    mounted_region = Region(3, 4, 30, 1)
    widget = SimpleNamespace(region=mounted_region)

    assert _settings_screen_region(widget) == mounted_region


class ModalHarness(ConsolidatedCSSApp):
    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.app_config = {
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
                "openai": {"api_key": "test-key"},
            },
        }
        self.saved_settings: ConsoleSessionSettings | None = None
        self.saved_result: ConsoleSettingsResult | None = None

    def capture_saved_settings(self, result: ConsoleSettingsResult | None) -> None:
        self.saved_result = result
        self.saved_settings = result.settings if result is not None else None


class StyledModalHarness(ModalHarness):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


class StyledConsoleHarness(ConsoleHarness):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


class FakeConsoleModelDiscoveryScope:
    def __init__(self, entries: tuple[MergedModelEntry, ...]) -> None:
        self.entries = entries
        self.merge_calls = []

    async def merge_saved_and_discovered_models(self, **kwargs):
        self.merge_calls.append(kwargs)
        return self.entries


class FailingConsoleModelDiscoveryScope:
    async def merge_saved_and_discovered_models(self, **kwargs):
        raise RuntimeError("merge failed")


class EmptyConsoleModelSnapshotScope:
    async def merge_saved_and_discovered_models(self, **_kwargs):
        return ()

    async def has_discovered_model_snapshot(self, **_kwargs):
        return True


def _visible_text(app: App[None]) -> str:
    return " ".join(str(widget.renderable) for widget in app.screen.query(Static))


def _summary_text(console) -> str:
    summary = console.query_one("#console-settings-summary")
    return " ".join(
        getattr(widget.renderable, "plain", str(widget.renderable))
        for widget in summary.query(Static)
        if widget.display and hasattr(widget, "renderable")
    )


def test_groq_console_default_uses_current_catalog_model() -> None:
    groq_settings = DEFAULT_CONFIG_FROM_TOML["api_settings"]["groq"]

    assert groq_settings["model"] == "llama-3.3-70b-versatile"
    assert groq_settings["model"] in API_MODELS_BY_PROVIDER["Groq"]
    assert groq_settings["model"] not in {"llama3-70b-8192", "llama3-8b-8192"}


def test_console_remote_defaults_use_smoke_verified_models() -> None:
    expected_defaults = {
        "anthropic": ("Anthropic", "claude-sonnet-5"),
        "cohere": ("Cohere", "command-a-03-2025"),
        "google": ("Google", "gemini-2.5-flash"),
        "huggingface": ("HuggingFace", "openai/gpt-oss-120b"),
    }

    for config_key, (catalog_key, expected_model) in expected_defaults.items():
        provider_settings = DEFAULT_CONFIG_FROM_TOML["api_settings"][config_key]

        assert provider_settings["model"] == expected_model
        assert expected_model in API_MODELS_BY_PROVIDER[catalog_key]


async def _wait_for_console_settings_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if (
            host.screen_stack
            and host.screen_stack[-1].query("#console-settings-modal")
            and host.screen_stack[-1].query("#console-settings-provider")
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console settings modal did not open")


async def _visible_console_settings_button(console: ChatScreen, pilot) -> Button:
    """Open the inspector rail and return the actionable settings summary button."""
    rail_state = replace(
        console._current_console_rail_state(),
        right_open=True,
    )
    console._sync_console_rail_visibility(rail_state)
    assert rail_state.right_open is True
    await _wait_for_selector(console, pilot, "#console-settings-open")
    for _ in range(40):
        button = console.query_one("#console-settings-open", Button)
        if button.display and button.region.width > 0 and button.region.height > 0:
            return button
        await pilot.pause(0.05)
    button = console.query_one("#console-settings-open", Button)
    raise AssertionError(
        "Console settings button is not visible/actionable: "
        f"display={button.display!r} region={button.region!r}"
    )


async def _wait_for_console_top_screen(host: ConsoleHarness, console, pilot) -> None:
    for _ in range(40):
        if host.screen_stack and host.screen_stack[-1] is console:
            return
        await pilot.pause(0.05)
    raise AssertionError("Console settings modal did not dismiss")


async def _wait_for_focused_id(host: App[None], pilot, widget_id: str) -> None:
    for _ in range(40):
        focused_id = getattr(host.focused, "id", None)
        if focused_id == widget_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"Expected focus on {widget_id!r}, found {getattr(host.focused, 'id', None)!r}"
    )


async def _press_new_console_tab(console, store, pilot) -> str:
    previous_session_id = store.active_session_id
    console.query_one("#console-new-chat-tab", Button).press()
    for _ in range(40):
        active_session_id = store.active_session_id
        if active_session_id is not None and active_session_id != previous_session_id:
            return active_session_id
        await pilot.pause(0.05)
    raise AssertionError("New Console tab did not activate")


def _first_chat_config(provider: str = "openai", model: str = "model-a") -> dict:
    return {
        "chat_defaults": {"provider": provider, "model": model},
        "api_settings": {
            provider: {
                "api_key": "test-only-key",
                "model": model,
            }
        },
    }


def _pending_first_chat(app) -> ConsoleFirstChatIntent | None:
    claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    if claim is None:
        return None
    value = claim.value
    app.pending_handoffs.release(claim)
    return value if isinstance(value, ConsoleFirstChatIntent) else None


def _first_chat_owner(console: ChatScreen) -> ConsoleSessionController:
    return console._session


def _first_chat_session_snapshot(session: ConsoleChatSession) -> dict[str, object]:
    """Capture all session values without comparing holder object identity."""

    snapshot = {
        item.name: deepcopy(getattr(session, item.name))
        for item in fields(ConsoleChatSession)
        if item.name not in {"rag_scope_holder", "todo_store"}
    }
    snapshot["rag_scope_holder"] = deepcopy(session.rag_scope_holder.scope)
    snapshot["todo_store"] = deepcopy(session.todo_store.export_snapshot())
    return snapshot


@pytest.fixture(autouse=True)
def _first_chat_generation_guard_uses_session_snapshot(monkeypatch):
    """Keep synthetic Console snapshots internally consistent in this suite.

    The config module's real publication lock is covered in
    ``Tests/test_config_delete_settings.py``. Console tests intentionally use
    arbitrary generations, so their final guard must observe the same injected
    snapshot as the rest of the consumer transaction.
    """

    def guarded(expected_generation: int, action) -> bool:
        if (
            session_module.get_runtime_config_snapshot().generation
            != expected_generation
        ):
            return False
        return action() is True

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        guarded,
        raising=False,
    )


def _mounted_first_chat_projection(console: ChatScreen) -> dict[str, object]:
    """Capture first-chat-owned mounted state without retaining widgets."""

    store = console._ensure_console_chat_store()
    controller = console._ensure_console_chat_controller()
    control_bar = console.query_one("#console-control-bar")
    composer = console.query_one("#console-native-composer")
    tabs = tuple(
        (
            str(tab.id),
            str(getattr(tab, "label", "")),
            tuple(sorted(tab.classes)),
        )
        for tab in console.query(".console-session-tab")
    )
    return {
        "active_session_id": store.active_session_id,
        "controller": (
            controller.provider,
            controller.model,
            controller.configured_model,
            controller.system_prompt,
        ),
        "control_scalars": (
            console._console_control_provider,
            console._console_control_model,
        ),
        "summary": _summary_text(console),
        "control_state": deepcopy(control_bar.state),
        "provider_label": str(
            console.query_one("#console-provider-label", Static).renderable
        ),
        "model_label": str(
            console.query_one("#console-model-label", Static).renderable
        ),
        "tabs": tabs,
        "composer_draft": composer.draft_text(),
        "focus_id": getattr(console.app.focused, "id", None),
    }


async def _wait_for_first_chat_projection(
    console: ChatScreen,
    pilot,
    expected: dict[str, object],
) -> None:
    for _ in range(80):
        if _mounted_first_chat_projection(console) == expected:
            return
        await pilot.pause(0.05)
    assert _mounted_first_chat_projection(console) == expected


def test_console_store_can_reserve_an_exact_first_chat_session_id() -> None:
    settings = build_default_console_session_settings(_first_chat_config())
    store = ConsoleChatStore()

    session = store.create_session(
        session_id="first-chat-session",
        settings=settings,
        canonical_settings_baseline=settings,
    )

    assert session.id == "first-chat-session"
    with pytest.raises(ValueError, match="already exists"):
        store.create_session(session_id=session.id, settings=settings)


def test_first_chat_target_eligibility_query_is_read_only() -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    defaults = build_default_console_session_settings(
        _first_chat_config("openai", "old-model")
    )
    session = store.create_session(
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    session_before = _first_chat_session_snapshot(session)
    active_before = store.active_session_id
    controls_before = (
        console._console_control_provider,
        console._console_control_model,
        console._console_chat_controller,
    )

    assert (
        _first_chat_owner(console).eligible_console_first_chat_session_id()
        == session.id
    )
    assert store.active_session_id == active_before
    assert _first_chat_session_snapshot(session) == session_before
    assert (
        console._console_control_provider,
        console._console_control_model,
        console._console_chat_controller,
    ) == controls_before

    store.set_session_draft(session.id, "user draft")
    changed_before = _first_chat_session_snapshot(session)
    assert _first_chat_owner(console).eligible_console_first_chat_session_id() is None
    assert _first_chat_session_snapshot(session) == changed_before


@pytest.mark.parametrize("user_provenance", ["custom-workspace", "renamed-back"])
def test_first_chat_eligibility_rejects_empty_user_owned_session_without_mutation(
    user_provenance: str,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    old_defaults = build_default_console_session_settings(
        _first_chat_config("openai", "old-model")
    )
    workspace_id = "workspace-user" if user_provenance == "custom-workspace" else None
    store = ConsoleChatStore(
        workspace_context=ConsoleWorkspaceContext(
            active_workspace_id=workspace_id or "global"
        )
    )
    console._console_chat_store = store
    user_session = store.create_session(
        title="Chat 1",
        workspace_id=workspace_id,
        settings=old_defaults,
        canonical_settings_baseline=old_defaults,
    )
    if user_provenance == "renamed-back":
        store.rename_session(user_session.id, "User planning")
        store.rename_session(user_session.id, "Chat 1")
    before = _first_chat_session_snapshot(user_session)

    assert _first_chat_owner(console).eligible_console_first_chat_session_id() is None
    assert store.active_session_id == user_session.id
    assert len(store.sessions()) == 1
    preserved = next(item for item in store.sessions() if item.id == user_session.id)
    assert _first_chat_session_snapshot(preserved) == before


def test_first_chat_eligibility_rejects_user_created_default_named_session() -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    user_settings = build_default_console_session_settings(
        _first_chat_config("openai", "user-model")
    )
    user_session = store.create_session(title="Chat 1", settings=user_settings)
    before = _first_chat_session_snapshot(user_session)

    assert _first_chat_owner(console).eligible_console_first_chat_session_id() is None
    assert store.active_session_id == user_session.id
    assert len(store.sessions()) == 1
    preserved = next(item for item in store.sessions() if item.id == user_session.id)
    assert _first_chat_session_snapshot(preserved) == before


def test_session_owner_refuses_session_switch_and_config_generation_races(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    snapshot = RuntimeConfigSnapshot(23, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    target = store.create_session(
        session_id="existing-first-chat-target",
        settings=build_default_console_session_settings(snapshot.values),
        canonical_settings_baseline=build_default_console_session_settings(
            snapshot.values
        ),
    )
    intent = ConsoleFirstChatIntent(target.id, "openai", "model-a", 23)
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)
    competing = store.create_session(
        settings=build_default_console_session_settings(snapshot.values),
        canonical_settings_baseline=build_default_console_session_settings(
            snapshot.values
        ),
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.active_session_id == competing.id
    assert _pending_first_chat(app) == intent

    store.switch_session(target.id)
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshot(24, snapshot.values),
        raising=False,
    )
    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.active_session_id == target.id
    assert _pending_first_chat(app) == intent


def test_first_chat_consumer_activates_once_and_acknowledges_exact_target(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    snapshot = RuntimeConfigSnapshot(31, _first_chat_config("llama_cpp", "local-a"))
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    prior_settings = build_default_console_session_settings(
        _first_chat_config("openai", "prior-model")
    )
    prior = store.create_session(
        settings=prior_settings,
        canonical_settings_baseline=prior_settings,
    )
    store.set_session_draft(prior.id, "keep this draft")
    intent = ConsoleFirstChatIntent(
        "first-run-future-session", "llama_cpp", "local-a", snapshot.generation
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    owner = _first_chat_owner(console)
    real_apply = owner._apply_first_chat_control_selection_fn
    presentation = MagicMock(side_effect=real_apply)
    restore_focus = MagicMock()
    owner._apply_first_chat_control_selection_fn = presentation
    owner._restore_first_chat_focus_fn = restore_focus

    assert owner._screen_mounted_accessor() is False
    assert owner.consume_pending_console_first_chat_intent() is True
    presentation.assert_called_once_with("llama_cpp", "local-a")
    restore_focus.assert_not_called()
    assert store.active_session_id == "first-run-future-session"
    assert store.session_settings("first-run-future-session").provider == "llama_cpp"
    assert store.session_settings("first-run-future-session").model == "local-a"
    assert store.session_draft(prior.id) == "keep this draft"
    assert store.session_settings(prior.id) == prior_settings
    assert console._console_control_provider == "llama_cpp"
    assert console._console_control_model == "local-a"
    assert _pending_first_chat(app) is None
    assert owner.consume_pending_console_first_chat_intent() is False


def test_first_chat_consumer_refuses_absent_nonreserved_target(monkeypatch) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    snapshot = RuntimeConfigSnapshot(37, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent("deleted-target", "openai", "model-a", 37)
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.sessions() == []
    assert _pending_first_chat(app) == intent


def test_first_chat_reserved_target_concurrent_id_claim_is_not_overwritten(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    snapshot = RuntimeConfigSnapshot(39, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent("reserved-target", "openai", "model-a", 39)
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    original_create = store.create_session
    competing = ConsoleSessionSettings(
        provider="openai",
        model="competing-model",
        source="user",
    )

    def create_with_concurrent_claim(**kwargs):
        if kwargs.get("session_id") == intent.session_id:
            original_create(session_id=intent.session_id, settings=competing)
        return original_create(**kwargs)

    monkeypatch.setattr(store, "create_session", create_with_concurrent_claim)

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.session_settings(intent.session_id) == competing
    assert _pending_first_chat(app) == intent


def test_first_chat_reserved_target_never_adopts_preexisting_pristine_id(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    snapshot = RuntimeConfigSnapshot(41, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent("reserved-target", "openai", "model-a", 41)
    competing = build_default_console_session_settings(
        _first_chat_config("openai", "restored-model")
    )
    store.create_session(
        session_id=intent.session_id,
        settings=competing,
        canonical_settings_baseline=competing,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.session_settings(intent.session_id) == competing
    assert _pending_first_chat(app) == intent


def test_first_chat_reserved_create_preserves_concurrent_active_session_switch(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Prior user work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="prior-user-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "preserve prior draft")
    competing = store.create_session(
        title="Competing user work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="competing-user-model",
            source="user",
        ),
    )
    store.set_session_draft(competing.id, "preserve competing draft")
    store.switch_session(prior.id)
    user_sessions_before = {
        item.id: _first_chat_session_snapshot(item) for item in store.sessions()
    }
    snapshot = RuntimeConfigSnapshot(42, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent(
        "reserved-active-session-race",
        "openai",
        "model-a",
        snapshot.generation,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    original_create = store.create_session

    def create_then_select_competing_session(**kwargs):
        created = original_create(**kwargs)
        if kwargs.get("session_id") == intent.session_id:
            store.switch_session(competing.id)
        return created

    monkeypatch.setattr(
        store,
        "create_session",
        create_then_select_competing_session,
    )
    owner = _first_chat_owner(console)
    real_apply = owner._apply_first_chat_control_selection_fn
    presentation = MagicMock(side_effect=real_apply)
    acknowledgement = MagicMock(
        wraps=app.pending_handoffs.acknowledge_current,
    )
    owner._apply_first_chat_control_selection_fn = presentation
    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        acknowledgement,
    )

    assert owner.consume_pending_console_first_chat_intent() is False
    assert all(item.id != intent.session_id for item in store.sessions())
    assert {
        item.id: _first_chat_session_snapshot(item) for item in store.sessions()
    } == user_sessions_before
    assert store.session_draft(prior.id) == "preserve prior draft"
    assert store.session_draft(competing.id) == "preserve competing draft"
    assert store.active_session_id == competing.id
    assert _pending_first_chat(app) == intent
    presentation.assert_called_once_with(None, None)
    acknowledgement.assert_not_called()


def test_first_chat_generation_change_during_reserved_create_rolls_back(
    monkeypatch,
) -> None:
    app = _build_test_app()
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )
    console = ChatScreen(app)
    console._console_control_provider = "prior-control-provider"
    console._console_control_model = "prior-control-model"
    store = ConsoleChatStore()
    console._console_chat_store = store
    user_settings = ConsoleSessionSettings(
        provider="openai",
        model="user-model",
        source="user",
    )
    user_session = store.create_session(
        title="User session",
        settings=user_settings,
    )
    store.set_session_draft(user_session.id, "preserve exactly")
    user_before = _first_chat_session_snapshot(user_session)
    current = [RuntimeConfigSnapshot(43, _first_chat_config())]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    intent = ConsoleFirstChatIntent("reserved-race", "openai", "model-a", 43)
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    original_create = store.create_session

    def create_then_advance_generation(**kwargs):
        created = original_create(**kwargs)
        if kwargs.get("session_id") == intent.session_id:
            current[0] = RuntimeConfigSnapshot(44, current[0].values)
        return created

    monkeypatch.setattr(store, "create_session", create_then_advance_generation)
    owner = _first_chat_owner(console)
    real_apply = owner._apply_first_chat_control_selection_fn
    projections: list[tuple[object, object]] = []

    def apply_and_record(provider, model) -> None:
        projections.append((provider, model))
        real_apply(provider, model)

    owner._apply_first_chat_control_selection_fn = apply_and_record

    assert owner.consume_pending_console_first_chat_intent() is False
    assert projections == [("prior-control-provider", "prior-control-model")]
    assert all(item.id != intent.session_id for item in store.sessions())
    assert store.active_session_id == user_session.id
    preserved = next(item for item in store.sessions() if item.id == user_session.id)
    assert _first_chat_session_snapshot(preserved) == user_before
    assert _pending_first_chat(app) == intent
    assert len(notifications) == 1


def test_first_chat_generation_change_during_refresh_restores_exact_target(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    old_defaults = build_default_console_session_settings(
        _first_chat_config("openai", "old-model")
    )
    target = store.create_session(
        settings=old_defaults,
        canonical_settings_baseline=old_defaults,
    )
    target_before = _first_chat_session_snapshot(target)
    current = [RuntimeConfigSnapshot(47, _first_chat_config())]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    intent = ConsoleFirstChatIntent(target.id, "openai", "model-a", 47)
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)
    original_refresh = store.refresh_pristine_session_settings

    def refresh_then_advance_generation(*args, **kwargs):
        refreshed = original_refresh(*args, **kwargs)
        current[0] = RuntimeConfigSnapshot(48, current[0].values)
        return refreshed

    monkeypatch.setattr(
        store,
        "refresh_pristine_session_settings",
        refresh_then_advance_generation,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.active_session_id == target.id
    restored = next(item for item in store.sessions() if item.id == target.id)
    assert _first_chat_session_snapshot(restored) == target_before
    assert _pending_first_chat(app) == intent


def test_first_chat_generation_publish_at_ack_rolls_back_reserved_creation(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Prior user work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="prior-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "preserve before ack")
    sessions_before = [_first_chat_session_snapshot(item) for item in store.sessions()]
    current = [RuntimeConfigSnapshot(67, _first_chat_config())]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    intent = ConsoleFirstChatIntent("ack-publish-new", "openai", "model-a", 67)
    app.pending_handoffs.stage_reserved_console_first_chat(intent)

    def publish_before_guarded_ack(_generation, _acknowledge) -> bool:
        current[0] = RuntimeConfigSnapshot(68, current[0].values)
        return False

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        publish_before_guarded_ack,
        raising=False,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.active_session_id == prior.id
    assert [
        _first_chat_session_snapshot(item) for item in store.sessions()
    ] == sessions_before
    assert _pending_first_chat(app) == intent


def test_first_chat_generation_publish_at_ack_restores_existing_refresh(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    old_defaults = build_default_console_session_settings(
        _first_chat_config("openai", "old-model")
    )
    target = store.create_session(
        settings=old_defaults,
        canonical_settings_baseline=old_defaults,
    )
    target_before = _first_chat_session_snapshot(target)
    current = [RuntimeConfigSnapshot(69, _first_chat_config())]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    intent = ConsoleFirstChatIntent(target.id, "openai", "model-a", 69)
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)

    def publish_before_guarded_ack(_generation, _acknowledge) -> bool:
        current[0] = RuntimeConfigSnapshot(70, current[0].values)
        return False

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        publish_before_guarded_ack,
        raising=False,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert store.active_session_id == target.id
    restored = next(item for item in store.sessions() if item.id == target.id)
    assert _first_chat_session_snapshot(restored) == target_before
    assert _pending_first_chat(app) == intent


def test_first_chat_replacement_and_session_switch_during_create_roll_back_old_target(
    monkeypatch,
) -> None:
    app = _build_test_app()
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    user_settings = ConsoleSessionSettings(
        provider="openai",
        model="user-model",
        source="user",
    )
    selected = store.create_session(title="Selected work", settings=user_settings)
    store.set_session_draft(selected.id, "selected draft")
    selected_before = _first_chat_session_snapshot(selected)
    snapshot = RuntimeConfigSnapshot(49, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    old_intent = ConsoleFirstChatIntent("old-reserved", "openai", "model-a", 49)
    replacement = ConsoleFirstChatIntent(
        "replacement-reserved",
        "openai",
        "model-a",
        49,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(old_intent)
    original_create = store.create_session

    def create_then_replace_and_reselect(**kwargs):
        created = original_create(**kwargs)
        if kwargs.get("session_id") == old_intent.session_id:
            store.switch_session(selected.id)
            app.pending_handoffs.stage_reserved_console_first_chat(replacement)
        return created

    monkeypatch.setattr(store, "create_session", create_then_replace_and_reselect)

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert all(item.id != old_intent.session_id for item in store.sessions())
    assert store.active_session_id == selected.id
    assert _first_chat_session_snapshot(selected) == selected_before
    assert _pending_first_chat(app) == replacement
    assert notifications == []
    assert _first_chat_owner(console)._first_chat_handoff_notified_revision is None


def test_first_chat_guarded_ack_replacement_rolls_back_original_claim(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Guarded-ack prior work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="guarded-ack-prior-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "preserve guarded-ack draft")
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(50, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    original = ConsoleFirstChatIntent(
        "guarded-ack-original-target",
        "openai",
        "model-a",
        snapshot.generation,
    )
    replacement = replace(
        original,
        session_id="guarded-ack-replacement-target",
    )
    app.pending_handoffs.stage_reserved_console_first_chat(original)

    def stage_replacement_before_acknowledgement(
        _expected_generation,
        acknowledge,
    ) -> bool:
        app.pending_handoffs.stage_reserved_console_first_chat(replacement)
        return acknowledge()

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        stage_replacement_before_acknowledgement,
        raising=False,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert all(item.id != original.session_id for item in store.sessions())
    assert all(item.id != replacement.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before
    assert store.session_draft(prior.id) == "preserve guarded-ack draft"
    assert _pending_first_chat(app) == replacement


def test_first_chat_current_claim_fence_blocks_replacement_target_projection(
    monkeypatch,
) -> None:
    app = _build_test_app()
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )
    console = ChatScreen(app)
    console._console_control_provider = "prior-control-provider"
    console._console_control_model = "prior-control-model"
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Current-claim prior work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="current-claim-prior-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "preserve current-claim draft")
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(52, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    original = ConsoleFirstChatIntent(
        "current-claim-original-target",
        "openai",
        "model-a",
        snapshot.generation,
    )
    replacement = replace(
        original,
        session_id="current-claim-replacement-target",
    )
    app.pending_handoffs.stage_reserved_console_first_chat(original)
    original_create = store.create_session

    def create_then_stage_replacement(**kwargs):
        created = original_create(**kwargs)
        if kwargs.get("session_id") == original.session_id:
            app.pending_handoffs.stage_reserved_console_first_chat(replacement)
        return created

    monkeypatch.setattr(
        store,
        "create_session",
        create_then_stage_replacement,
    )
    owner = _first_chat_owner(console)
    real_apply = owner._apply_first_chat_control_selection_fn
    projections: list[tuple[object, object]] = []

    def apply_and_record(provider, model) -> None:
        projections.append((provider, model))
        real_apply(provider, model)

    owner._apply_first_chat_control_selection_fn = apply_and_record

    assert owner.consume_pending_console_first_chat_intent() is False
    assert projections == [("prior-control-provider", "prior-control-model")]
    assert all(item.id != original.session_id for item in store.sessions())
    assert all(item.id != replacement.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before
    assert store.session_draft(prior.id) == "preserve current-claim draft"
    assert _pending_first_chat(app) == replacement
    assert notifications == []


def test_first_chat_failed_acknowledgement_rolls_back_and_requeues(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="User work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="user-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "keep")
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(51, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent("ack-race", "openai", "model-a", 51)
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    events: list[tuple[object, ...]] = []
    owner = _first_chat_owner(console)
    real_apply = owner._apply_first_chat_control_selection_fn
    real_rollback = store.rollback_created_pristine_session
    real_release = app.pending_handoffs.release

    def apply_and_record(provider, model) -> None:
        events.append(("project", provider, model))
        real_apply(provider, model)

    def rollback_and_record(session_id, **kwargs) -> bool:
        events.append(("store-rollback", session_id))
        return real_rollback(session_id, **kwargs)

    def release_and_record(claim) -> bool:
        events.append(("release", claim.revision))
        return real_release(claim)

    owner._apply_first_chat_control_selection_fn = apply_and_record
    monkeypatch.setattr(
        store,
        "rollback_created_pristine_session",
        rollback_and_record,
    )
    monkeypatch.setattr(app.pending_handoffs, "release", release_and_record)
    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        lambda _claim: False,
    )

    assert owner.consume_pending_console_first_chat_intent() is False
    assert [event[0] for event in events] == [
        "project",
        "store-rollback",
        "project",
        "release",
    ]
    assert events[0][1:] == ("openai", "model-a")
    assert events[1][1] == intent.session_id
    assert events[2][1:] == (None, None)
    assert all(item.id != intent.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before
    assert _pending_first_chat(app) == intent


def test_first_chat_ack_exception_rolls_back_create_and_survives_release_error(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="User work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="user-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "keep exact")
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(73, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent(
        "ack-exception-create-target",
        "openai",
        "model-a",
        snapshot.generation,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    real_acknowledge = app.pending_handoffs.acknowledge_current
    real_release = app.pending_handoffs.release
    secret = "PRIVATE_ACK_EXCEPTION_TEXT"
    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        session_module.logger,
        "warning",
        lambda *args, **_kwargs: warnings.append(args),
    )

    def fail_acknowledge(_claim) -> bool:
        raise RuntimeError(secret)

    def fail_release(_claim) -> bool:
        raise RuntimeError("PRIVATE_RELEASE_EXCEPTION_TEXT")

    monkeypatch.setattr(app.pending_handoffs, "acknowledge_current", fail_acknowledge)
    monkeypatch.setattr(app.pending_handoffs, "release", fail_release)

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert all(item.id != intent.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before
    rendered_warnings = repr(warnings)
    assert secret not in rendered_warnings
    assert "PRIVATE_RELEASE_EXCEPTION_TEXT" not in rendered_warnings
    assert intent.session_id not in rendered_warnings

    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        real_acknowledge,
    )
    monkeypatch.setattr(app.pending_handoffs, "release", real_release)
    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is True
    )
    assert store.active_session_id == intent.session_id
    assert _pending_first_chat(app) is None


def test_first_chat_exception_log_is_metadata_only() -> None:
    records: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: records.append(str(message)),
        level="WARNING",
    )
    try:
        ConsoleSessionController._log_first_chat_handoff_exception(
            "guarded-acknowledgement",
            RuntimeError("SECRET-FIRST-CHAT-EXCEPTION"),
        )
    finally:
        loguru_logger.remove(sink_id)

    rendered = "".join(records)
    assert "guarded-acknowledgement" in rendered
    assert "RuntimeError" in rendered
    assert "SECRET-FIRST-CHAT-EXCEPTION" not in rendered


def test_first_chat_ack_exception_restores_refresh_and_retries(monkeypatch) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior_settings = build_default_console_session_settings(
        _first_chat_config("openai", "prior-model")
    )
    target = store.create_session(
        session_id="ack-exception-refresh-target",
        settings=prior_settings,
        canonical_settings_baseline=prior_settings,
    )
    target_before = _first_chat_session_snapshot(target)
    snapshot = RuntimeConfigSnapshot(75, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent(
        target.id,
        "openai",
        "model-a",
        snapshot.generation,
    )
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)
    real_acknowledge = app.pending_handoffs.acknowledge_current
    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        lambda _claim: (_ for _ in ()).throw(RuntimeError("PRIVATE_REFRESH")),
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    restored = next(item for item in store.sessions() if item.id == target.id)
    assert _first_chat_session_snapshot(restored) == target_before

    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        real_acknowledge,
    )
    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is True
    )
    assert store.session_settings(target.id).model == "model-a"
    assert _pending_first_chat(app) is None


def test_first_chat_config_guard_exception_rolls_back_and_retries(monkeypatch) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Prior",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="prior-model",
            source="user",
        ),
    )
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(77, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent(
        "config-guard-exception-target",
        "openai",
        "model-a",
        snapshot.generation,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    real_guard = session_module.run_if_runtime_config_generation_current
    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("PRIVATE_GUARD")),
        raising=False,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert all(item.id != intent.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        real_guard,
        raising=False,
    )
    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is True
    )
    assert _pending_first_chat(app) is None


def test_first_chat_ack_exception_after_replacement_preserves_replacement(
    monkeypatch,
) -> None:
    app = _build_test_app()
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    prior = store.create_session(
        title="Prior replacement work",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="prior-model",
            source="user",
        ),
    )
    store.set_session_draft(prior.id, "preserve replacement draft")
    prior_before = _first_chat_session_snapshot(prior)
    snapshot = RuntimeConfigSnapshot(79, _first_chat_config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    original = ConsoleFirstChatIntent(
        "ack-exception-old-target",
        "openai",
        "model-a",
        snapshot.generation,
    )
    replacement = replace(original, session_id="ack-exception-replacement-target")
    app.pending_handoffs.stage_reserved_console_first_chat(original)
    real_acknowledge = app.pending_handoffs.acknowledge_current

    def replace_then_raise(_claim) -> bool:
        app.pending_handoffs.stage_reserved_console_first_chat(replacement)
        raise RuntimeError("PRIVATE_REPLACEMENT_ACK")

    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        replace_then_raise,
    )

    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is False
    )
    assert all(item.id != original.session_id for item in store.sessions())
    assert store.active_session_id == prior.id
    assert _first_chat_session_snapshot(prior) == prior_before
    assert _pending_first_chat(app) == replacement

    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        real_acknowledge,
    )
    assert (
        _first_chat_owner(console).consume_pending_console_first_chat_intent() is True
    )
    assert store.active_session_id == replacement.session_id
    assert _pending_first_chat(app) is None


def test_first_chat_failed_notification_tracking_is_bounded_to_latest_revision(
    monkeypatch,
) -> None:
    app = _build_test_app()
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )
    console = ChatScreen(app)
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshot(999, _first_chat_config()),
        raising=False,
    )
    latest_revision = 0

    for index in range(128):
        latest_revision = app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_FIRST_CHAT,
            ConsoleFirstChatIntent(
                f"missing-{index}",
                "openai",
                "model-a",
                index + 1,
            ),
        )
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )

    assert len(notifications) == 128
    assert (
        _first_chat_owner(console)._first_chat_handoff_notified_revision
        == latest_revision
    )


@pytest.mark.asyncio
async def test_mounted_first_chat_preserves_restored_and_concurrent_sessions(
    monkeypatch,
) -> None:
    app = _build_test_app()
    snapshot = RuntimeConfigSnapshot(
        53,
        _first_chat_config("llama_cpp", "mounted-local"),
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        restored_settings = ConsoleSessionSettings(
            provider="openai",
            model="restored-model",
            source="user",
        )
        restored = ConsoleChatSession(
            id="restored-user-session",
            title="Restored work",
            workspace_id="workspace-restored",
            persisted_conversation_id="conversation-restored",
            settings=restored_settings,
            draft="restored draft",
            has_user_work=True,
        )
        concurrent = ConsoleChatSession(
            id="concurrent-user-session",
            title="Concurrent work",
            settings=replace(restored_settings, model="concurrent-model"),
            draft="concurrent draft",
            has_user_work=True,
        )
        store.restore_state(
            sessions=[restored, concurrent],
            active_session_id=concurrent.id,
        )
        before = [_first_chat_session_snapshot(item) for item in store.sessions()]
        intent = ConsoleFirstChatIntent(
            "mounted-first-chat",
            "llama_cpp",
            "mounted-local",
            snapshot.generation,
        )
        app.pending_handoffs.stage_reserved_console_first_chat(intent)

        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is True
        )
        await pilot.pause()
        assert store.active_session_id == intent.session_id
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()[:2]
        ] == before
        assert store.session_settings(intent.session_id).provider == "llama_cpp"
        assert store.session_settings(intent.session_id).model == "mounted-local"
        assert _pending_first_chat(app) is None

        sessions_after_success = [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ]
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ] == sessions_after_success


@pytest.mark.asyncio
async def test_mounted_first_chat_replacement_ack_exception_restores_prior_ui(
    monkeypatch,
) -> None:
    app = _build_test_app()
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )
    snapshot = RuntimeConfigSnapshot(
        59,
        _first_chat_config("llama_cpp", "replacement-race-model"),
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        prior_settings = ConsoleSessionSettings(
            provider="openai",
            model="prior-mounted-model",
            source="user",
            system_prompt="Preserve this system prompt.",
        )
        prior = ConsoleChatSession(
            id="prior-mounted-replacement",
            title="Prior mounted work",
            settings=prior_settings,
            draft="preserve mounted draft",
            has_user_work=True,
        )
        store.restore_state(sessions=[prior], active_session_id=prior.id)
        await console._sync_native_console_chat_ui()
        console._focus_console_composer_if_needed(force=True)
        await pilot.pause()
        sessions_before = [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ]
        mounted_before = _mounted_first_chat_projection(console)
        old_intent = ConsoleFirstChatIntent(
            "mounted-old-target",
            "llama_cpp",
            "replacement-race-model",
            snapshot.generation,
        )
        replacement = replace(old_intent, session_id="mounted-replacement-target")
        app.pending_handoffs.stage_reserved_console_first_chat(old_intent)
        real_acknowledge_current = getattr(
            app.pending_handoffs,
            "acknowledge_current",
            None,
        )

        def replace_immediately_before_ack(_claim) -> bool:
            app.pending_handoffs.stage_reserved_console_first_chat(replacement)
            raise RuntimeError("PRIVATE_MOUNTED_REPLACEMENT")

        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge_current",
            replace_immediately_before_ack,
            raising=False,
        )

        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )
        await _wait_for_first_chat_projection(console, pilot, mounted_before)
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ] == sessions_before
        assert all(item.id != old_intent.session_id for item in store.sessions())
        assert store.active_session_id == prior.id
        assert _pending_first_chat(app) == replacement
        assert notifications == []

        assert real_acknowledge_current is not None
        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge_current",
            real_acknowledge_current,
        )
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is True
        )
        await pilot.pause()
        assert store.active_session_id == replacement.session_id
        assert _pending_first_chat(app) is None


@pytest.mark.asyncio
async def test_mounted_first_chat_generation_publish_at_ack_restores_reserved_ui(
    monkeypatch,
) -> None:
    app = _build_test_app()
    current = [
        RuntimeConfigSnapshot(
            61,
            _first_chat_config("llama_cpp", "failed-ack-model"),
        )
    ]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        prior_settings = ConsoleSessionSettings(
            provider="openai",
            model="prior-failed-ack-model",
            source="user",
            system_prompt="Keep the mounted controller state.",
        )
        prior = ConsoleChatSession(
            id="prior-mounted-failed-ack",
            title="Prior failed-ack work",
            settings=prior_settings,
            draft="keep the composer exact",
            has_user_work=True,
        )
        store.restore_state(sessions=[prior], active_session_id=prior.id)
        await console._sync_native_console_chat_ui()
        console._focus_console_composer_if_needed(force=True)
        await pilot.pause()
        sessions_before = [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ]
        mounted_before = _mounted_first_chat_projection(console)
        intent = ConsoleFirstChatIntent(
            "mounted-failed-ack-target",
            "llama_cpp",
            "failed-ack-model",
            current[0].generation,
        )
        app.pending_handoffs.stage_reserved_console_first_chat(intent)

        def publish_at_guarded_ack(_generation, _acknowledge) -> bool:
            current[0] = RuntimeConfigSnapshot(62, current[0].values)
            return False

        monkeypatch.setattr(
            session_module,
            "run_if_runtime_config_generation_current",
            publish_at_guarded_ack,
            raising=False,
        )

        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )
        await _wait_for_first_chat_projection(console, pilot, mounted_before)
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ] == sessions_before
        assert all(item.id != intent.session_id for item in store.sessions())
        assert store.active_session_id == prior.id
        assert _pending_first_chat(app) == intent


@pytest.mark.asyncio
async def test_mounted_first_chat_generation_publish_at_ack_restores_refresh_ui(
    monkeypatch,
) -> None:
    app = _build_test_app()
    current = [
        RuntimeConfigSnapshot(
            63,
            _first_chat_config("llama_cpp", "refresh-ack-model"),
        )
    ]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        prior_settings = build_default_console_session_settings(
            _first_chat_config("openai", "prior-refresh-model")
        )
        target = ConsoleChatSession(
            id="mounted-refresh-ack-target",
            title="Chat 1",
            settings=prior_settings,
            canonical_settings_baseline=prior_settings,
        )
        store.restore_state(sessions=[target], active_session_id=target.id)
        await console._sync_native_console_chat_ui()
        console._focus_console_composer_if_needed(force=True)
        await pilot.pause()
        sessions_before = [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ]
        mounted_before = _mounted_first_chat_projection(console)
        intent = ConsoleFirstChatIntent(
            target.id,
            "llama_cpp",
            "refresh-ack-model",
            current[0].generation,
        )
        app.pending_handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)

        def publish_at_guarded_ack(_generation, _acknowledge) -> bool:
            current[0] = RuntimeConfigSnapshot(64, current[0].values)
            return False

        monkeypatch.setattr(
            session_module,
            "run_if_runtime_config_generation_current",
            publish_at_guarded_ack,
            raising=False,
        )

        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is False
        )
        await _wait_for_first_chat_projection(console, pilot, mounted_before)
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ] == sessions_before
        assert store.active_session_id == target.id
        assert _pending_first_chat(app) == intent


@pytest.mark.asyncio
async def test_mounted_first_chat_ack_exception_during_mount_is_retryable(
    monkeypatch,
) -> None:
    app = _build_test_app()
    snapshot = RuntimeConfigSnapshot(
        81,
        _first_chat_config("llama_cpp", "mount-exception-model"),
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    intent = ConsoleFirstChatIntent(
        "mounted-on-mount-exception-target",
        "llama_cpp",
        "mount-exception-model",
        snapshot.generation,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    real_acknowledge = app.pending_handoffs.acknowledge_current
    monkeypatch.setattr(
        app.pending_handoffs,
        "acknowledge_current",
        lambda _claim: (_ for _ in ()).throw(RuntimeError("PRIVATE_MOUNT")),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        await pilot.pause()
        assert all(item.id != intent.session_id for item in store.sessions())
        assert _pending_first_chat(app) == intent

        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge_current",
            real_acknowledge,
        )
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is True
        )
        await pilot.pause()
        assert store.active_session_id == intent.session_id
        assert _pending_first_chat(app) is None


@pytest.mark.asyncio
async def test_mounted_first_chat_ack_exception_during_resume_restores_ui(
    monkeypatch,
) -> None:
    app = _build_test_app()
    snapshot = RuntimeConfigSnapshot(
        83,
        _first_chat_config("llama_cpp", "resume-exception-model"),
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        store = console._ensure_console_chat_store()
        prior = ConsoleChatSession(
            id="mounted-resume-prior",
            title="Resume prior",
            settings=ConsoleSessionSettings(
                provider="openai",
                model="resume-prior-model",
                source="user",
                system_prompt="Preserve resume UI.",
            ),
            draft="preserve resume composer",
            has_user_work=True,
        )
        store.restore_state(sessions=[prior], active_session_id=prior.id)
        await console._sync_native_console_chat_ui()
        console._focus_console_composer_if_needed(force=True)
        await pilot.pause()
        prior_focused_widget = console.app.focused
        assert prior_focused_widget is not None
        assert prior_focused_widget.is_mounted is True
        sessions_before = [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ]
        mounted_before = _mounted_first_chat_projection(console)
        intent = ConsoleFirstChatIntent(
            "mounted-on-resume-exception-target",
            "llama_cpp",
            "resume-exception-model",
            snapshot.generation,
        )
        app.pending_handoffs.stage_reserved_console_first_chat(intent)
        real_acknowledge = app.pending_handoffs.acknowledge_current
        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge_current",
            lambda _claim: (_ for _ in ()).throw(RuntimeError("PRIVATE_RESUME")),
        )
        owner = _first_chat_owner(console)
        real_restore_focus = owner._restore_first_chat_focus_fn
        restore_focus = MagicMock(side_effect=real_restore_focus)
        owner._restore_first_chat_focus_fn = restore_focus

        console.on_screen_resume()
        await _wait_for_first_chat_projection(console, pilot, mounted_before)
        for _ in range(80):
            if restore_focus.call_count:
                break
            await pilot.pause(0.05)
        restore_focus.assert_called_once_with(prior_focused_widget)
        assert console.app.focused is prior_focused_widget
        assert [
            _first_chat_session_snapshot(item) for item in store.sessions()
        ] == sessions_before
        assert all(item.id != intent.session_id for item in store.sessions())
        assert _pending_first_chat(app) == intent

        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge_current",
            real_acknowledge,
        )
        assert (
            _first_chat_owner(console).consume_pending_console_first_chat_intent()
            is True
        )
        await pilot.pause()
        assert store.active_session_id == intent.session_id
        assert _pending_first_chat(app) is None


async def _click_console_session_tab(console, store, pilot, session_id: str) -> None:
    await pilot.click(f"#console-session-tab-{session_id}")
    for _ in range(40):
        if store.active_session_id == session_id:
            await pilot.pause()
            return
        await pilot.pause(0.05)
    console._ensure_console_chat_controller().switch_session(session_id)
    await console._sync_native_console_chat_ui()
    if store.active_session_id == session_id:
        await pilot.pause()
        return
    raise AssertionError(f"Console tab {session_id!r} did not activate")


def _select_values(select: Select) -> set[str]:
    options = getattr(select, "options", None)
    if options is None:
        options = getattr(select, "_options", [])
    values: set[str] = set()
    for option in options:
        value = getattr(option, "value", None)
        if value is None and isinstance(option, tuple) and len(option) >= 2:
            value = option[1]
        if value is not None and value is not Select.NULL:
            values.add(str(value))
    return values


def _merged_model(
    model_id: str,
    *,
    source: str = "saved",
    capability_status: str = "known",
    persisted: bool = True,
) -> MergedModelEntry:
    return MergedModelEntry(
        provider="openai",
        provider_list_key="openai",
        model_id=model_id,
        display_name=model_id,
        source=source,
        capability_status=capability_status,
        persisted=persisted,
    )


@pytest.mark.asyncio
async def test_console_settings_summary_renders_rows_and_button() -> None:
    state = ConsoleSettingsSummaryState(
        provider_row="Provider: llama.cpp",
        model_row="Model: model-a",
        context_row="Context: 12 / 4k",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Assistant: General",
        readiness_label="Ready",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        text = _visible_text(app)
        assert "Session Settings" in text
        assert "Provider: llama.cpp" in text
        assert "Model: model-a" in text
        assert "Context: 12 / 4k" in text
        assert "Sampling: T 0.70, P 0.95" in text
        assert "Assistant: General" in text
        header = app.query_one("#console-settings-header", Horizontal)
        title = app.query_one("#console-settings-title", Static)
        button = app.query_one("#console-settings-open", Button)
        assert title.parent is header
        assert button.parent is header
        assert title.region.y == button.region.y
        assert str(button.label) == "Configure"
        assert button.tooltip == "Configure Console settings"


@pytest.mark.asyncio
async def test_console_settings_summary_uses_direct_choose_model_action_when_setup_blocked() -> (
    None
):
    state = ConsoleSettingsSummaryState(
        provider_row="Provider: llama.cpp",
        model_row="Model: Missing",
        context_row="Context: unavailable",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Assistant: General",
        readiness_label="Missing model",
        action_label="Choose Model",
        action_tooltip="Choose a model for this Console session",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        text = _visible_text(app)
        assert "Provider: llama.cpp" in text
        assert "Model: Missing" in text
        button = app.query_one("#console-settings-open", Button)
        assert str(button.label) == "Choose Model"
        assert button.tooltip == "Choose a model for this Console session"


@pytest.mark.asyncio
async def test_console_settings_summary_treats_missing_provider_row_as_blank() -> None:
    state = ConsoleSettingsSummaryState(
        provider_row=None,  # type: ignore[arg-type]
        model_row="Model: model-a",
        context_row="Context: 12 / 4k",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Assistant: General",
        readiness_label="Ready",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        provider_row = app.query_one("#console-settings-provider-row", Static)
        assert str(provider_row.renderable) == ""
        assert "None" not in _visible_text(app)

        updated_state = ConsoleSettingsSummaryState(
            provider_row=None,  # type: ignore[arg-type]
            model_row="Model: model-b",
            context_row="Context: 20 / 4k",
            sampling_row="Sampling: T 0.20, P 0.90",
            identity_row="Persona: Analyst",
            readiness_label="Ready",
        )
        app.query_one(ConsoleSettingsSummary).sync_state(updated_state)
        await pilot.pause()

        assert str(provider_row.renderable) == ""
        assert "None" not in _visible_text(app)


def test_console_settings_summary_button_sizing_uses_named_constants() -> None:
    assert not hasattr(settings_summary_module, "CONSOLE_SETTINGS_SUMMARY_MAX_HEIGHT")
    assert settings_summary_module.CONSOLE_SETTINGS_BUTTON_HORIZONTAL_PADDING == 2
    assert settings_summary_module.CONSOLE_SETTINGS_BUTTON_MIN_WIDTH == 9
    assert settings_summary_module.CONSOLE_SETTINGS_BUTTON_MAX_WIDTH == 14
    assert settings_summary_module.CONSOLE_SETTINGS_ROW_HEIGHT == 1


@pytest.mark.asyncio
async def test_console_settings_header_is_external_and_one_row_body_uses_one_line() -> (
    None
):
    state = ConsoleSettingsSummaryState(
        provider_row="",
        model_row="Model: only visible row",
        context_row="",
        sampling_row="",
        identity_row="",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        summary = app.query_one(ConsoleSettingsSummary)
        header = summary.query_one("#console-settings-header", Horizontal)
        body = summary.query_one(
            "#console-bounded-section-session-settings", ConsoleBoundedSection
        )
        for _ in range(4):
            await pilot.pause()

        assert header.parent is summary
        assert body.parent is summary
        assert list(summary.children) == [header, body]
        assert body.desired_content_lines == 1
        assert body.viewport.content_region.height == 1
        assert body.hint.display is False


@pytest.mark.asyncio
async def test_console_settings_body_uses_exact_twenty_line_content_ceiling() -> None:
    state = ConsoleSettingsSummaryState(
        model_row="Model: visible",
        context_row="",
        sampling_row="",
        identity_row="",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 30)) as pilot:
        body = app.query_one(
            "#console-bounded-section-session-settings", ConsoleBoundedSection
        )
        await body.viewport.remove_children()
        content = Static("\n".join(f"row {index}" for index in range(20)))
        await body.viewport.mount(content)
        body.request_reconcile()
        for _ in range(4):
            await pilot.pause()
        assert body.viewport.content_region.height == 20
        assert body.hint.display is False

        content.update("\n".join(f"row {index}" for index in range(21)))
        content.refresh(layout=True)
        body.request_reconcile()
        for _ in range(4):
            await pilot.pause()
        assert body.viewport.content_region.height == 20
        assert body.hint.display is True
        assert body.hint.region.height == 1


def test_console_settings_modal_sizing_uses_named_constants() -> None:
    assert MODAL_BODY_MIN_HEIGHT == 0
    assert MODAL_CONTROL_HEIGHT == 3
    assert f"min-height: {MODAL_BODY_MIN_HEIGHT};" in ConsoleSettingsModal.DEFAULT_CSS
    assert f"height: {MODAL_CONTROL_HEIGHT};" in ConsoleSettingsModal.DEFAULT_CSS
    assert f"min-height: {MODAL_CONTROL_HEIGHT};" in ConsoleSettingsModal.DEFAULT_CSS


def test_pending_launch_inspector_auto_open_docstring_is_google_style() -> None:
    docstring = ChatScreen._apply_pending_launch_inspector_auto_open.__doc__

    assert docstring is not None
    assert "Args:" in docstring
    assert "Returns:" in docstring


def test_summary_state_appends_non_ready_readiness_to_model_row() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="WIP", detail="Provider not wired yet.", native_send_supported=False
        ),
    )

    assert state.provider_row == "Provider: llama_cpp"
    assert state.model_row == "Model: model-a (WIP)"
    assert state.readiness_label == "WIP"


def test_default_console_session_settings_prefers_provider_model_profile() -> None:
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.9,
            "top_p": 0.8,
            "streaming": False,
        },
        "api_settings": {
            "openai": {
                "temperature": 0.7,
                "top_p": 0.95,
                "streaming": False,
                "model_defaults": {
                    "gpt-4.1": {
                        "temperature": 0.2,
                        "top_p": 0.88,
                        "min_p": 0.04,
                        "top_k": 40,
                        "max_tokens": 1234,
                        "seed": 101,
                        "presence_penalty": 0.2,
                        "frequency_penalty": 0.3,
                        "reasoning_effort": "high",
                        "reasoning_summary": "auto",
                        "verbosity": "high",
                        "streaming": True,
                    },
                },
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.provider == "openai"
    assert settings.model == "gpt-4.1"
    assert settings.temperature == 0.2
    assert settings.top_p == 0.88
    assert settings.min_p == 0.04
    assert settings.top_k == 40
    assert settings.max_tokens == 1234
    assert settings.seed == 101
    assert settings.presence_penalty == 0.2
    assert settings.frequency_penalty == 0.3
    assert settings.reasoning_effort == "high"
    assert settings.reasoning_summary == "auto"
    assert settings.verbosity == "high"
    assert settings.streaming is True


def test_default_console_session_settings_prefers_chat_defaults_over_provider_scalars() -> (
    None
):
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.9,
            "top_p": 0.8,
            "streaming": False,
        },
        "api_settings": {
            "openai": {
                "temperature": 0.7,
                "top_p": 0.95,
                "streaming": True,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.temperature == 0.9
    assert settings.top_p == 0.8
    assert settings.streaming is False


def test_default_console_settings_delegates_canonical_model_and_endpoint_resolution() -> (
    None
):
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI-Compatible",
            "model": "chat-model",
            "temperature": 0.31,
        },
        "api_settings": {
            "openai": {
                "model": "provider-model",
                "api_base_url": "https://api.example.test/v1",
                "temperature": 0.79,
            }
        },
    }

    settings = build_default_console_session_settings(app_config)

    assert settings.provider == "openai"
    assert settings.model == "chat-model"
    assert settings.base_url == "https://api.example.test/v1"
    assert settings.temperature == 0.31
    assert app_config["chat_defaults"]["provider"] == "OpenAI-Compatible"


def test_console_session_settings_accepts_documented_effort_values() -> None:
    app_config = {
        "api_settings": {
            "openai": {"api_key": "test-key", "model": "gpt-5.1"},
            "anthropic": {"api_key": "test-key", "model": "claude-opus-4-8"},
        }
    }
    openai_settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-5.1",
        reasoning_effort="none",
    )
    anthropic_settings = ConsoleSessionSettings(
        provider="anthropic",
        model="claude-opus-4-8",
        thinking_effort="max",
    )

    assert (
        validate_console_session_settings(openai_settings, app_config=app_config) == []
    )
    assert (
        validate_console_session_settings(anthropic_settings, app_config=app_config)
        == []
    )


def test_default_console_session_settings_reads_enable_streaming_as_compatibility_fallback() -> (
    None
):
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "enable_streaming": False,
        },
        "api_settings": {
            "openai": {
                "streaming": True,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.streaming is False


def test_default_console_session_settings_prefers_canonical_streaming_over_enable_streaming() -> (
    None
):
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "streaming": True,
            "enable_streaming": False,
        },
        "api_settings": {
            "openai": {
                "streaming": False,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.streaming is True


def test_default_console_session_settings_uses_global_fallbacks_when_profile_is_absent() -> (
    None
):
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.33,
            "top_p": 0.81,
            "max_tokens": 2048,
            "streaming": False,
        },
        "api_settings": {
            "openai": {},
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.temperature == 0.33
    assert settings.top_p == 0.81
    assert settings.max_tokens == 2048
    assert settings.streaming is False


def test_default_console_session_settings_skips_blank_model_profile_values() -> None:
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.9,
            "top_p": 0.8,
            "streaming": False,
        },
        "api_settings": {
            "openai": {
                "temperature": 0.7,
                "top_p": 0.95,
                "streaming": True,
                "model_defaults": {
                    "gpt-4.1": {
                        "temperature": "",
                        "top_p": " ",
                    },
                },
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.temperature == 0.9
    assert settings.top_p == 0.8


def test_summary_state_keeps_missing_model_row_compact() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model=None),
        ConsoleSettingsContextEstimate(
            used_tokens=None, token_limit=None, label="unknown"
        ),
        ConsoleSettingsReadiness(
            label="Missing model",
            detail="Select a model before sending.",
            native_send_supported=False,
        ),
    )

    assert state.provider_row == "Provider: llama_cpp"
    assert state.model_row == "Model: Missing"
    assert state.readiness_label == "Missing model"
    assert state.action_label == "Choose Model"
    assert state.action_tooltip == "Choose a model for this Console session"


def test_summary_state_exposes_safe_credential_source() -> None:
    """Show safe env/config credential sources without exposing secret values."""
    env_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="OpenAI is ready. API key found via env:OPENAI_API_KEY.",
            native_send_supported=True,
        ),
    )
    config_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet-4-20250514"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="Anthropic is ready. API key found via config:api_settings.anthropic.api_key.",
            native_send_supported=True,
        ),
    )

    assert env_state.credential_row == "Credential: env OPENAI_API_KEY"
    assert (
        config_state.credential_row
        == "Credential: config api_settings.anthropic.api_key"
    )


def test_summary_state_handles_empty_credential_source_names() -> None:
    """Collapse empty env/config credential-source identifiers without padding."""
    env_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="OpenAI is ready. API key found via env:   .",
            native_send_supported=True,
        ),
    )
    config_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet-4-20250514"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="Anthropic is ready. API key found via config:   .",
            native_send_supported=True,
        ),
    )

    assert env_state.credential_row == "Credential: env"
    assert config_state.credential_row == "Credential: config"


def test_summary_state_ignores_warning_lines_after_credential_source() -> None:
    """Keep appended readiness warnings out of the credential summary row."""
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail=(
                "OpenAI is ready. API key found via env:OPENAI_API_KEY.\n"
                "Model warning: selected model may not support native tools."
            ),
            native_send_supported=True,
        ),
    )

    assert state.credential_row == "Credential: env OPENAI_API_KEY"


def test_summary_state_appends_optional_sampling_fields_only_when_set() -> None:
    without_optional = build_console_settings_summary_state(
        ConsoleSessionSettings(
            provider="llama_cpp", model="model-a", temperature=0.7, top_p=0.95
        ),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready", detail="Ready.", native_send_supported=True
        ),
    )
    with_optional = build_console_settings_summary_state(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            temperature=0.7,
            top_p=0.95,
            min_p=0.05,
            top_k=40,
            max_tokens=512,
        ),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready", detail="Ready.", native_send_supported=True
        ),
    )

    assert without_optional.sampling_row == "Sampling: T 0.70, P 0.95"
    assert (
        with_optional.sampling_row
        == "Sampling: T 0.70, P 0.95, min_p 0.05, top_k 40, max_tokens 512"
    )


def test_summary_state_normalizes_unknown_context_label() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        ConsoleSettingsContextEstimate(
            used_tokens=None, token_limit=None, label="Context: unknown"
        ),
        ConsoleSettingsReadiness(
            label="Ready", detail="Ready.", native_send_supported=True
        ),
    )

    assert state.context_row == "Context: unavailable"


def test_summary_state_renders_character_or_generic_assistant_identity() -> None:
    character = build_console_settings_summary_state(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            character_label="Ada",
        ),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready", detail="Ready.", native_send_supported=True
        ),
    )
    generic = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        ConsoleSettingsContextEstimate(
            used_tokens=12, token_limit=4096, label="12 / 4k"
        ),
        ConsoleSettingsReadiness(
            label="Ready", detail="Ready.", native_send_supported=True
        ),
    )

    assert character.identity_row == "Character: Ada"
    assert generic.identity_row == "Assistant: General"


def test_summary_state_projects_character_identity_to_one_line_without_mutating_settings() -> (
    None
):
    raw_name = "Nyx\n\tAdmin\x00[/bold]"
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        character_label=raw_name,
    )

    summary = build_console_settings_summary_state(
        settings,
        ConsoleSettingsContextEstimate(
            used_tokens=12,
            token_limit=4096,
            label="12 / 4k",
        ),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="Ready.",
            native_send_supported=True,
        ),
    )

    assert summary.identity_row == "Character: Nyx Admin?[/bold]"
    assert settings.character_label == raw_name


def test_legacy_identity_settings_helper_ignores_unknown_keys_without_mutation() -> (
    None
):
    source = {
        "provider": "llama_cpp",
        "model": "model-a",
        "persona_label": "Legacy A",
        "user_profile_label": "Legacy B",
    }
    source_before = dict(source)
    restored = ConsoleSessionController._restore_console_settings(source)

    assert restored is not None
    assert source == source_before
    assert not hasattr(restored, "user_profile_label")
    serialized = ConsoleSessionController._serialize_console_settings(restored)
    assert serialized is not None
    assert {"persona_label", "user_profile_label"}.isdisjoint(serialized)


def test_choose_model_action_label_normalization() -> None:
    assert ChatScreen._is_console_choose_model_action(" Choose Model ")
    assert ChatScreen._is_console_choose_model_action("choose model")
    assert ChatScreen._is_console_choose_model_action("CHOOSE MODEL")
    assert not ChatScreen._is_console_choose_model_action("Configure")


@pytest.mark.asyncio
async def test_console_model_resolution_includes_runtime_discovered_models() -> None:
    scope = FakeConsoleModelDiscoveryScope(
        (
            _merged_model("gpt-4.1", source="persisted_discovered"),
            _merged_model(
                "gpt-5",
                source="runtime_discovered",
                capability_status="unknown",
                persisted=False,
            ),
        )
    )
    options = await provider_model_resolution.resolve_provider_model_options(
        {"openai": ["gpt-4.1"]},
        scope,
        provider="OpenAI",
    )

    assert [option.model_id for option in options] == ["gpt-4.1", "gpt-5"]
    assert (
        options[1].warning
        == "Capabilities unknown until saved or verified; text chat is assumed."
    )
    assert scope.merge_calls == [
        {
            "mode": "local",
            "provider": "openai",
        }
    ]


@pytest.mark.asyncio
async def test_console_model_resolution_failure_logs_provider_context(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.providers_models = {"openai": ["gpt-4.1"]}
    app.llm_provider_catalog_scope_service = FailingConsoleModelDiscoveryScope()
    console = ChatScreen(app)
    logged = []

    def fake_exception(message, *args, **kwargs):
        logged.append((message, args, kwargs))

    monkeypatch.setattr(chat_screen_module.logger, "exception", fake_exception)

    models = await console._providers_models_for_console_settings(
        "OpenAI",
        current_model="gpt-5",
    )

    assert models == {"openai": ["gpt-4.1"]}
    assert logged == [
        (
            "Unable to resolve Console runtime-discovered models for provider=%s model=%s",
            ("openai", "gpt-5"),
            {},
        )
    ]


@pytest.mark.asyncio
async def test_console_settings_model_resolution_preserves_configured_alternatives() -> (
    None
):
    app = _build_test_app()
    app.providers_models = {
        "local_llamacpp": ["uat-local-model", "uat-alt-local-model"],
    }
    app.llm_provider_catalog_scope_service = FakeConsoleModelDiscoveryScope(
        (
            _merged_model(
                "uat-local-model",
                source="runtime_discovered",
                capability_status="known",
                persisted=False,
            ),
        )
    )
    console = ChatScreen(app)

    models = await console._providers_models_for_console_settings(
        "local_llamacpp",
        current_model="uat-local-model",
    )

    assert models["local_llamacpp"] == ["uat-local-model", "uat-alt-local-model"]


@pytest.mark.asyncio
async def test_console_settings_model_resolution_keeps_empty_cloud_snapshot_authoritative() -> (
    None
):
    app = _build_test_app()
    app.providers_models = {"anthropic": ["retired-model"]}
    app.llm_provider_catalog_scope_service = EmptyConsoleModelSnapshotScope()
    console = ChatScreen(app)

    models = await console._providers_models_for_console_settings(
        "anthropic",
        current_model=None,
    )

    assert models["anthropic"] == []


@pytest.mark.asyncio
async def test_console_settings_modal_cancel_discards_draft() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    app.saved_settings = ConsoleSessionSettings(provider="openai", model="should-clear")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a", "model-b"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    used_tokens=10,
                    token_limit=4096,
                    label="10 / 4k",
                ),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        await pilot.click("#console-settings-cancel")

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_escape_dismisses_none() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    app.saved_settings = ConsoleSessionSettings(provider="openai", model="should-clear")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        await pilot.press("escape")

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_save_returns_validated_settings() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleSettingsModal(
            settings=settings,
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a", "model-b"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        )
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one(
            "#console-settings-provider-model-section"
        )
        assert "Choose a model to enable sending." not in str(readiness.renderable)
        assert (
            provider_model_section.has_class("console-settings-primary-section")
            is False
        )
        app.screen.query_one("#console-settings-temperature", Input).value = "0.42"
        app.screen.query_one("#console-settings-top-p", Input).value = "0.88"
        app.screen.query_one(
            "#console-settings-user-display-name", Input
        ).value = "  Captain Rowan  "
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "model-a"
    assert app.saved_settings.temperature == 0.42
    assert app.saved_settings.top_p == 0.88
    assert app.saved_result is not None
    assert app.saved_result.user_display_name_override == "Captain Rowan"
    assert not hasattr(app.saved_result.settings, "user_display_name_override")


@pytest.mark.asyncio
async def test_console_settings_modal_renders_current_chat_identity() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                user_display_name_override="Captain Rowan",
                global_user_display_name="Default Name",
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        identity = app.screen.query_one("#console-settings-user-display-name", Input)
        assert identity.value == "Captain Rowan"
        assert identity.placeholder == "Default Name"
        assert "Chat identity" in _visible_text(app)
        assert "Leave blank to use the global default." in _visible_text(app)


def test_local_thinking_provider_detection_covers_execution_key_aliases() -> None:
    assert _is_local_thinking_provider("llama_cpp") is True
    assert _is_local_thinking_provider("local_llamacpp") is True
    assert _is_local_thinking_provider("local_llamafile") is True
    assert _is_local_thinking_provider("local_llm") is True
    assert _is_local_thinking_provider("vllm") is True
    assert _is_local_thinking_provider("local_vllm") is True
    assert _is_local_thinking_provider("local_mlx_lm") is True
    # Readiness aliases resolve to their custom-openai execution keys.
    assert _is_local_thinking_provider("custom") is True
    assert _is_local_thinking_provider("custom_2") is True
    assert _is_local_thinking_provider("anthropic") is False
    assert _is_local_thinking_provider("openai") is False
    assert _is_local_thinking_provider(None) is False


@pytest.mark.asyncio
async def test_console_settings_modal_local_provider_marks_no_effect_choices() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()

        thinking = app.screen.query_one("#console-settings-thinking-effort", Input)
        summary = app.screen.query_one("#console-settings-reasoning-summary", Input)
        verbosity = app.screen.query_one("#console-settings-verbosity", Input)
        effort = app.screen.query_one("#console-settings-reasoning-effort", Input)
        # Local providers consume only the reasoning-effort level; the other
        # provider-specific choice inputs say so right in the placeholder.
        assert thinking.placeholder.endswith(PROVIDER_CHOICE_NO_EFFECT_SUFFIX)
        assert summary.placeholder.endswith(PROVIDER_CHOICE_NO_EFFECT_SUFFIX)
        assert verbosity.placeholder.endswith(PROVIDER_CHOICE_NO_EFFECT_SUFFIX)
        assert PROVIDER_CHOICE_NO_EFFECT_SUFFIX not in effort.placeholder


@pytest.mark.asyncio
async def test_console_settings_modal_remote_provider_keeps_thinking_hint_plain() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="anthropic", model="claude-3-5-sonnet-latest"
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()

        thinking = app.screen.query_one("#console-settings-thinking-effort", Input)
        assert PROVIDER_CHOICE_NO_EFFECT_SUFFIX not in thinking.placeholder


@pytest.mark.asyncio
async def test_console_settings_modal_provider_switch_refreshes_choice_hints() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(
                settings,
                app,
                providers_models={"openai": ["gpt-4.1"], "llama_cpp": ["model-a"]},
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        thinking = app.screen.query_one("#console-settings-thinking-effort", Input)
        assert PROVIDER_CHOICE_NO_EFFECT_SUFFIX not in thinking.placeholder

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        provider_select.value = "llama_cpp"
        await pilot.pause()

        assert thinking.placeholder.endswith(PROVIDER_CHOICE_NO_EFFECT_SUFFIX)


@pytest.mark.asyncio
async def test_console_settings_modal_blank_name_returns_separate_none_override() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(
                settings,
                app,
                user_display_name_override="Captain Rowan",
                global_user_display_name="Default Name",
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-user-display-name", Input).value = "   "
        await pilot.click("#console-settings-save")

    assert app.saved_result is not None
    assert app.saved_result.user_display_name_override is None
    assert app.saved_result.settings == app.saved_settings
    assert not hasattr(app.saved_result.settings, "user_display_name_override")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invalid_name", "expected_error"),
    [
        ("界" * 25, "Display name must fit within 48 terminal cells."),
        ("Captain\x07Rowan", "Display name cannot contain control characters."),
        ("Captain\u202eRowan", "Display name cannot contain control characters."),
    ],
)
async def test_console_settings_modal_invalid_name_prevents_dismissal(
    invalid_name, expected_error
) -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        app.screen.query_one(
            "#console-settings-user-display-name", Input
        ).value = invalid_name
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal")
        assert expected_error in str(
            app.screen.query_one("#console-settings-error", Static).renderable
        )

    assert app.saved_result is None


@pytest.mark.asyncio
async def test_console_settings_validation_error_clears_on_edit() -> None:
    """TASK-363: a validation error must clear as soon as the user edits any
    field, not linger stale until the next Save."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleSettingsModal(
            settings=settings,
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        )
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()

        temperature = app.screen.query_one("#console-settings-temperature", Input)
        temperature.value = ""
        await pilot.click("#console-settings-save")
        await pilot.pause()

        error = app.screen.query_one("#console-settings-error", Static)
        assert "Temperature is required" in str(error.renderable)

        # Editing the field invalidates the stale summary immediately.
        temperature.value = "0.5"
        await pilot.pause()
        assert str(error.renderable).strip() == ""


@pytest.mark.asyncio
async def test_console_settings_error_summary_is_visually_distinct() -> None:
    """TASK-363: the validation summary must read as an error (bold, error
    colour), not near-body-text salience."""
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleSettingsModal(
            settings=settings,
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        )
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()

        error = app.screen.query_one("#console-settings-error", Static)
        assert "bold" in str(error.styles.text_style)


@pytest.mark.asyncio
async def test_console_settings_modal_single_model_uses_readonly_value_not_dead_dropdown() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        model_custom = app.screen.query_one("#console-settings-model-custom", Button)

        assert model_select.display is False
        assert model_select.disabled is True
        assert model_input.display is True
        assert model_input.disabled is True
        assert model_input.value == "model-a"
        assert model_custom.display is True
        assert model_custom.disabled is False

        model_custom.press()
        await pilot.pause()
        assert model_input.display is True
        assert model_input.disabled is False
        assert model_custom.label == "Model list"

        model_custom.press()
        await pilot.pause()
        assert model_select.display is False
        assert model_input.display is True
        assert model_input.disabled is True
        assert model_input.value == "model-a"
        assert getattr(app.focused, "id", None) == "model-search-picker-input"


@pytest.mark.asyncio
async def test_console_settings_modal_saves_replaced_temperature_input() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        temperature=0.60,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        temperature = app.screen.query_one("#console-settings-temperature", Input)
        body = app.screen.query_one("#console-settings-body")
        body.scroll_to_widget(temperature)
        await pilot.pause()

        await pilot.click(temperature)
        await pilot.press("ctrl+a")
        await pilot.press("0")
        await pilot.press(".")
        await pilot.press("7")
        assert temperature.value == "0.7"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.temperature == 0.7


@pytest.mark.asyncio
async def test_console_settings_modal_replaces_focused_sampling_input() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        temperature=0.60,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        temperature = app.screen.query_one("#console-settings-temperature", Input)
        body = app.screen.query_one("#console-settings-body")
        body.scroll_to_widget(temperature)
        await pilot.pause()

        await pilot.click(temperature)
        await pilot.press("0")
        await pilot.press(".")
        await pilot.press("7")
        await pilot.press("2")

        assert temperature.value == "0.72"


@pytest.mark.parametrize(
    ("field_id", "attribute", "backspace_count", "typed_suffix", "expected"),
    [
        ("console-settings-temperature", "temperature", 0, "1", 0.71),
        ("console-settings-top-p", "top_p", 1, "6", 0.96),
        ("console-settings-min-p", "min_p", 1, "6", 0.06),
        ("console-settings-top-k", "top_k", 1, "1", 41),
        ("console-settings-max-tokens", "max_tokens", 1, "5", 65),
    ],
)
@pytest.mark.asyncio
async def test_console_settings_modal_accepts_keyboard_edited_sampling_inputs(
    field_id: str,
    attribute: str,
    backspace_count: int,
    typed_suffix: str,
    expected: float | int,
) -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        temperature=0.70,
        top_p=0.95,
        min_p=0.05,
        top_k=40,
        max_tokens=64,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        target_input = app.screen.query_one(f"#{field_id}", Input)
        body = app.screen.query_one("#console-settings-body")
        body.scroll_to_widget(target_input)
        await pilot.pause()
        await pilot.click(target_input)
        await pilot.press("end")
        for _ in range(backspace_count):
            await pilot.press("backspace")
        await pilot.press(typed_suffix)
        assert str(expected) in target_input.value

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert getattr(app.saved_settings, attribute) == expected


@pytest.mark.asyncio
async def test_console_settings_modal_body_is_scrollable_container_for_overflow_controls() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-4.1",
        temperature=0.70,
        top_p=0.95,
        seed=17,
        presence_penalty=0.4,
        frequency_penalty=0.5,
        reasoning_effort="high",
        reasoning_summary="auto",
        verbosity="medium",
        thinking_effort="low",
        thinking_budget_tokens=2048,
    )

    async with app.run_test(size=(140, 32)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        body = app.screen.query_one("#console-settings-body")
        assert isinstance(body, ScrollableContainer)


@pytest.mark.asyncio
async def test_console_settings_modal_preserves_provider_specific_generation_controls() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-4.1",
        temperature=0.70,
        top_p=0.95,
        seed=17,
        presence_penalty=0.4,
        frequency_penalty=0.5,
        reasoning_effort="high",
        reasoning_summary="auto",
        verbosity="medium",
        thinking_effort="low",
        thinking_budget_tokens=2048,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        for selector in (
            "#console-settings-seed",
            "#console-settings-presence-penalty",
            "#console-settings-frequency-penalty",
            "#console-settings-reasoning-effort",
            "#console-settings-reasoning-summary",
            "#console-settings-verbosity",
            "#console-settings-thinking-effort",
            "#console-settings-thinking-budget-tokens",
        ):
            input_widget = app.screen.query_one(selector, Input)
            body = app.screen.query_one("#console-settings-body")
            body.scroll_to_widget(input_widget)
            await pilot.pause()

            assert input_widget.display is True
            assert input_widget.disabled is False
            assert input_widget.value
            assert input_widget.content_region.height >= 1

        app.screen.query_one("#console-settings-seed", Input).value = "23"
        app.screen.query_one("#console-settings-presence-penalty", Input).value = "0.6"
        app.screen.query_one("#console-settings-frequency-penalty", Input).value = "0.7"
        app.screen.query_one(
            "#console-settings-reasoning-effort", Input
        ).value = "medium"
        app.screen.query_one(
            "#console-settings-reasoning-summary", Input
        ).value = "concise"
        app.screen.query_one("#console-settings-verbosity", Input).value = "high"
        app.screen.query_one(
            "#console-settings-thinking-effort", Input
        ).value = "medium"
        app.screen.query_one(
            "#console-settings-thinking-budget-tokens", Input
        ).value = "4096"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.seed == 23
    assert app.saved_settings.presence_penalty == 0.6
    assert app.saved_settings.frequency_penalty == 0.7
    assert app.saved_settings.reasoning_effort == "medium"
    assert app.saved_settings.reasoning_summary == "concise"
    assert app.saved_settings.verbosity == "high"
    assert app.saved_settings.thinking_effort == "medium"
    assert app.saved_settings.thinking_budget_tokens == 4096


@pytest.mark.asyncio
async def test_console_settings_modal_normalizes_provider_specific_choices() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-4.1",
        temperature=0.70,
        top_p=0.95,
        reasoning_effort="medium",
        reasoning_summary="concise",
        verbosity="low",
        thinking_effort="medium",
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        app.screen.query_one(
            "#console-settings-reasoning-effort", Input
        ).value = " HIGH "
        app.screen.query_one(
            "#console-settings-reasoning-summary", Input
        ).value = " AUTO "
        app.screen.query_one("#console-settings-verbosity", Input).value = " Medium "
        app.screen.query_one("#console-settings-thinking-effort", Input).value = " LOW "
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.reasoning_effort == "high"
    assert app.saved_settings.reasoning_summary == "auto"
    assert app.saved_settings.verbosity == "medium"
    assert app.saved_settings.thinking_effort == "low"


@pytest.mark.asyncio
async def test_console_settings_modal_shows_inherited_provider_endpoint() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", base_url=None
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        assert base_url_input.display is True
        assert base_url_input.disabled is False
        assert base_url_input.value == "http://127.0.0.1:9099"


@pytest.mark.asyncio
async def test_console_settings_modal_prefers_api_base_url_alias_over_default_api_url() -> (
    None
):
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = {
        "api_url": "http://localhost:8080/completion",
        "api_base_url": "http://127.0.0.1:9099/v1",
    }
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", base_url=None
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        assert base_url_input.value == "http://127.0.0.1:9099"
        assert "Provider blocked" not in str(readiness.renderable)
        assert "localhost:8080" not in str(readiness.renderable)


@pytest.mark.asyncio
async def test_console_settings_modal_replaces_stale_lower_priority_endpoint_with_alias() -> (
    None
):
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = {
        "api_url": "http://localhost:8080/completion",
        "api_base_url": "http://127.0.0.1:9099/v1",
    }
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://localhost:8080",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        assert base_url_input.value == "http://127.0.0.1:9099"
        assert "Provider blocked" not in str(readiness.renderable)
        assert "localhost:8080" not in str(readiness.renderable)


@pytest.mark.asyncio
async def test_console_settings_modal_focus_mode_uses_ready_copy_when_model_selected() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one(
            "#console-settings-provider-model-section"
        )
        assert (
            str(readiness.renderable) == "llama_cpp is ready. No API key is required."
        )
        assert (
            provider_model_section.has_class("console-settings-primary-section")
            is False
        )


@pytest.mark.asyncio
async def test_console_settings_modal_clears_setup_copy_when_dropdown_model_is_available() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="custom", model=None)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"custom": ["freeform-model"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one(
            "#console-settings-provider-model-section"
        )
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        readiness_copy = str(readiness.renderable)
        assert "Choose a model to enable sending." not in readiness_copy
        assert "not wired yet" not in readiness_copy
        assert "custom is ready" in str(readiness.renderable)
        assert model_select.disabled is False
        assert model_select.value == "freeform-model"
        assert (
            provider_model_section.has_class("console-settings-primary-section")
            is False
        )


@pytest.mark.asyncio
async def test_console_settings_modal_setup_copy_preserves_blocking_readiness_detail() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model=None,
        base_url="ftp://127.0.0.1:9099",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        readiness = app.screen.query_one("#console-settings-readiness", Static)
        readiness_copy = str(readiness.renderable)
        assert "Choose a model to enable sending." in readiness_copy
        assert "Provider blocked: invalid llama.cpp base URL." in readiness_copy


@pytest.mark.asyncio
async def test_console_settings_modal_invalid_temperature_stays_open_and_renders_error() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-temperature", Input).value = "3.0"
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal") is not None
        assert "Temperature must be between 0 and 2." in str(
            app.screen.query_one("#console-settings-error", Static).renderable
        )

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_blank_temperature_stays_open_and_renders_error() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", temperature=0.7
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-temperature", Input).value = ""
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal") is not None
        assert "Temperature is required." in str(
            app.screen.query_one("#console-settings-error", Static).renderable
        )

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_blank_top_p_stays_open_and_renders_error() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a", top_p=0.95)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-top-p", Input).value = ""
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal") is not None
        assert "Top P is required." in str(
            app.screen.query_one("#console-settings-error", Static).renderable
        )

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_save_disabled_when_cannot_save() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=False,
            )
        )
        await pilot.pause()

        assert app.screen.query_one("#console-settings-save", Button).disabled is True


@pytest.mark.asyncio
async def test_console_settings_modal_has_stable_body_error_and_footer_regions() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(100, 32)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        body = app.screen.query_one("#console-settings-body")
        error = app.screen.query_one("#console-settings-error", Static)
        actions = app.screen.query_one("#console-settings-actions")
        temperature = app.screen.query_one("#console-settings-temperature", Input)

        assert "console-settings-body" in body.classes
        assert "console-settings-error-summary" in error.classes
        assert "console-settings-modal-actions" in actions.classes
        assert "console-settings-control" in temperature.classes
        assert error.region.y < body.region.y < actions.region.y


@pytest.mark.asyncio
async def test_console_settings_modal_inputs_keep_visible_content_row_when_unfocused() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        for selector in (
            "#console-settings-base-url",
            "#console-settings-temperature",
            "#console-settings-top-p",
            "#console-settings-max-tokens",
        ):
            input_widget = app.screen.query_one(selector, Input)

            assert input_widget.value
            assert input_widget.content_region.height >= 1


@pytest.mark.asyncio
async def test_console_settings_modal_renders_context_and_single_identity_row() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        character_label="Ada",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    10,
                    4096,
                    "10 / 4k",
                    staged_source_count=2,
                    staged_context_summary="2 staged sources",
                ),
                can_save=True,
            )
        )
        await pilot.pause()

        text = _visible_text(app)
        assert "Current" in str(
            app.screen.query_one("#console-settings-context-current", Static).renderable
        )
        assert "10 / 4k tokens" in text
        assert "2 staged sources" in str(
            app.screen.query_one("#console-settings-context-sources", Static).renderable
        )
        assert "Estimate only; no truncation changes in this version." in str(
            app.screen.query_one("#console-settings-context-note", Static).renderable
        )
        assert "Character: Ada" in str(
            app.screen.query_one(
                "#console-settings-identity-current", Static
            ).renderable
        )
        assert not app.screen.query("#console-settings-persona-readonly")
        assert not app.screen.query("#console-settings-character-readonly")
        assert "User Profile" not in text
        assert "As:" not in text
        assert not app.screen.query("#console-settings-persona-input")
        assert not app.screen.query("#console-settings-character-input")


@pytest.mark.asyncio
async def test_console_settings_modal_provider_select_lists_all_configured_providers() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "openai": ["gpt-4"],
                    "custom": [],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        provider_values = _select_values(
            app.screen.query_one("#console-settings-provider", Select)
        )
        assert {"custom", "llama_cpp", "openai"}.issubset(provider_values)


@pytest.mark.asyncio
async def test_console_settings_modal_uses_model_dropdown_without_configured_models() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="custom", model="freeform-model")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"custom": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.display is True
        assert model_select.disabled is False
        assert model_select.value == "freeform-model"
        assert "freeform-model" in _select_values(model_select)
        assert model_input.display is False
        assert model_input.disabled is True
        assert model_input.value == "freeform-model"


@pytest.mark.asyncio
async def test_console_settings_modal_uses_first_model_when_initial_model_missing() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model=None)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.disabled is False
        assert model_select.value == "gpt-4.1"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "gpt-4.1"


@pytest.mark.asyncio
async def test_console_settings_modal_keyboard_selects_model_from_dropdown() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1", "gpt-5"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_select.focus()
        await pilot.press("enter")
        assert model_select.expanded is True

        await pilot.press("down")
        await pilot.press("enter")
        assert model_select.expanded is False
        assert model_select.value == "gpt-5"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "gpt-5"


@pytest.mark.asyncio
async def test_console_settings_modal_keyboard_selects_provider_and_refreshes_models() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert provider_select.value == "llama_cpp"
        assert model_select.value == "model-a"

        provider_select.focus()
        await pilot.press("enter")
        assert provider_select.expanded is True

        await pilot.press("down")
        await pilot.press("enter")
        assert provider_select.expanded is False
        assert provider_select.value == "local_llamacpp"
        assert model_select.disabled is False
        assert model_select.value == "local-model"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "local_llamacpp"
    assert app.saved_settings.model == "local-model"


@pytest.mark.asyncio
async def test_console_settings_modal_tabs_to_model_picker_after_provider_change() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        picker = app.screen.query_one(
            "#console-settings-model-picker", ModelSearchPicker
        )

        provider_select.focus()
        provider_select.value = "groq"
        await pilot.pause()

        assert (
            app.screen.query_one("#console-settings-model-legacy-adapter").display
            is False
        )
        assert model_select.value == "llama-3.3-70b-versatile"
        assert picker.value == "llama-3.3-70b-versatile"

        await pilot.press("tab")
        await _wait_for_focused_id(app, pilot, "model-search-picker-input")
        await pilot.press("8")
        await pilot.pause()

        assert app.screen.query_one("#model-search-picker-results", OptionList).display


@pytest.mark.asyncio
async def test_console_settings_modal_reopens_provider_select_after_input_edit() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        temperature = app.screen.query_one("#console-settings-temperature", Input)
        provider_select = app.screen.query_one("#console-settings-provider", Select)

        temperature.focus()
        temperature.value = "0.22"
        await pilot.pause()

        provider_select.focus()
        await pilot.press("enter")

        assert provider_select.expanded is True


@pytest.mark.asyncio
async def test_console_settings_modal_opens_provider_select_click_after_input_edit() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        temperature = app.screen.query_one(
            "#console-settings-temperature", ConsoleSettingsInput
        )
        provider_select = app.screen.query_one("#console-settings-provider", Select)

        await pilot.click("#console-settings-temperature")
        temperature.value = "0.72"
        await pilot.pause()
        await pilot.click("#console-settings-provider")

        assert provider_select.expanded is True


@pytest.mark.asyncio
async def test_console_settings_modal_opens_screen_routed_select_click_after_input_edit() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        temperature = app.screen.query_one(
            "#console-settings-temperature", ConsoleSettingsInput
        )
        provider_select = app.screen.query_one("#console-settings-provider", Select)

        temperature.focus()
        temperature.value = "0.72"
        await pilot.pause()

        provider_region = _settings_screen_region(provider_select)
        click = events.Click(
            app.screen,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_region.x + provider_region.width - 1,
            screen_y=provider_region.y,
        )

        app.screen.on_click(click)

        assert provider_select.expanded is True


@pytest.mark.asyncio
async def test_console_settings_input_releases_mouse_capture_after_click_to_replace() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        temperature = app.screen.query_one(
            "#console-settings-temperature", ConsoleSettingsInput
        )
        temperature.capture_mouse()

        assert app.mouse_captured is temperature

        temperature.on_click()

        assert app.mouse_captured is None
        assert temperature.selected_text == temperature.value


@pytest.mark.asyncio
async def test_console_settings_modal_opens_provider_select_from_redirected_input_click(
    monkeypatch,
) -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        temperature = app.screen.query_one(
            "#console-settings-temperature", ConsoleSettingsInput
        )
        provider_select = app.screen.query_one("#console-settings-provider", Select)
        temperature.capture_mouse()
        temperature.value = "0.22"

        provider_screen_region = provider_select.region.translate((10, 0))
        monkeypatch.setattr(
            Select,
            "screen_region",
            property(
                lambda widget: (
                    provider_screen_region
                    if widget is provider_select
                    else widget.region
                )
            ),
            raising=False,
        )
        click = events.Click(
            temperature,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_screen_region.x + provider_screen_region.width - 1,
            screen_y=provider_screen_region.y,
        )

        temperature.on_click(click)

        assert app.mouse_captured is None
        assert provider_select.expanded is True


@pytest.mark.asyncio
async def test_console_settings_modal_ignores_plain_select_click_without_redirected_input() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        provider_region = _settings_screen_region(provider_select)
        click = events.Click(
            provider_select,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_region.x + provider_region.width - 1,
            screen_y=provider_region.y,
        )

        app.screen.on_click(click)

        assert app.mouse_captured is None
        assert provider_select.expanded is False


@pytest.mark.asyncio
async def test_console_settings_modal_ignores_screen_routed_select_click_without_input_focus() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        cancel_button = app.screen.query_one("#console-settings-cancel", Button)
        cancel_button.focus()
        await pilot.pause()
        provider_region = _settings_screen_region(provider_select)
        click = events.Click(
            app.screen,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_region.x + provider_region.width - 1,
            screen_y=provider_region.y,
        )

        app.screen.on_click(click)

        assert getattr(app.focused, "id", None) == "console-settings-cancel"
        assert app.mouse_captured is None
        assert provider_select.expanded is False


@pytest.mark.asyncio
async def test_console_settings_modal_preserves_missing_registry_model_for_current_provider() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="custom-openai-model")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.disabled is False
        assert model_select.value == "custom-openai-model"
        assert {"custom-openai-model", "gpt-4.1"}.issubset(_select_values(model_select))
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "custom-openai-model"


@pytest.mark.asyncio
async def test_console_settings_modal_allows_manual_model_when_registry_has_stale_options() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="anthropic", model="claude-3-haiku-20240307"
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"anthropic": ["claude-3-haiku-20240307"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        custom_button = app.screen.query_one("#console-settings-model-custom", Button)
        assert model_select.display is True
        assert model_input.display is False
        assert custom_button.display is True

        await pilot.click("#console-settings-model-custom")
        await pilot.pause()

        assert model_select.display is False
        assert model_input.display is True
        assert model_input.disabled is False
        model_input.value = "claude-haiku-4-5-20251001"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "anthropic"
    assert app.saved_settings.model == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_console_settings_modal_uses_shared_picker_and_saves_search_result() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1", "gpt-5"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        picker = app.screen.query_one(
            "#console-settings-model-picker", ModelSearchPicker
        )
        legacy_adapter = app.screen.query_one("#console-settings-model-legacy-adapter")
        assert picker.display is True
        assert legacy_adapter.display is False

        search_input = picker.query_one("#model-search-picker-input", Input)
        search_input.value = "gpt-5"
        await pilot.pause()
        results = picker.query_one("#model-search-picker-results", OptionList)
        option = results.get_option_at_index(0)
        results.post_message(OptionList.OptionSelected(results, option, 0))
        await pilot.pause()
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.model == "gpt-5"


@pytest.mark.asyncio
async def test_console_settings_modal_refreshes_readiness_after_returning_to_model_list() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        app.screen.query_one("#console-settings-model-custom", Button).press()
        picker = app.screen.query_one(
            "#console-settings-model-picker", ModelSearchPicker
        )
        for _ in range(500):
            if picker.custom_mode:
                break
            await pilot.pause(0.01)
        assert picker.custom_mode is True
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one(
            "#console-settings-provider-model-section"
        )
        model_input.value = ""
        # Debounced (task-15476): let the production `Input.Changed`
        # handler settle instead of forcing `_sync_readiness_display()`
        # directly, which raced the (now-delayed) handler-driven update.
        await pilot.pause(CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS + 0.1)

        assert model_input.value == ""
        assert picker.value is None
        assert "Choose a model to enable sending." in str(readiness.renderable)
        assert (
            provider_model_section.has_class("console-settings-primary-section") is True
        )

        app.screen._toggle_manual_model_input()
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.display is True
        assert model_select.value == "model-a"
        assert (
            str(readiness.renderable) == "llama_cpp is ready. No API key is required."
        )
        assert (
            provider_model_section.has_class("console-settings-primary-section")
            is False
        )


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_uses_configured_provider_model() -> (
    None
):
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = {
        "api_url": "http://127.0.0.1:9099",
        "model": "gemma-local-config-model",
    }
    settings = ConsoleSessionSettings(
        provider="custom",
        model="custom-model-beta",
        base_url="http://localhost:1234/v1/chat/completions",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "Custom": ["custom-model-alpha", "custom-model-beta"],
                    "Llama_cpp": ["None"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        assert model_select.display is True
        assert model_select.disabled is False
        assert model_select.value == "gemma-local-config-model"
        assert model_input.display is False
        assert model_input.disabled is True
        assert model_input.value == "gemma-local-config-model"
        assert base_url_input.value == "http://127.0.0.1:9099"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "gemma-local-config-model"
    assert app.saved_settings.base_url == "http://127.0.0.1:9099"


@pytest.mark.parametrize(
    ("provider_settings", "expected_model"),
    (
        (
            {
                "api_url": "http://127.0.0.1:9099",
                "api_model": "gemma-api-model",
            },
            "gemma-api-model",
        ),
        (
            {
                "api_url": "http://127.0.0.1:9099",
                "model": "None",
                "api_model": "null",
                "default_model": "gemma-default-model",
            },
            "gemma-default-model",
        ),
    ),
)
@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_uses_model_alias_fallbacks(
    provider_settings: dict[str, str],
    expected_model: str,
) -> None:
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = provider_settings
    settings = ConsoleSessionSettings(
        provider="custom",
        model="custom-model-beta",
        base_url="http://localhost:1234/v1/chat/completions",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "Custom": ["custom-model-alpha", "custom-model-beta"],
                    "Llama_cpp": ["None"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.display is True
        assert model_select.value == expected_model
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == expected_model


@pytest.mark.asyncio
async def test_console_settings_modal_can_select_runtime_discovered_model_with_warning() -> (
    None
):
    app = _build_test_app()
    app.providers_models = {"openai": ["gpt-4.1"]}
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": "test-key"}}
    app.llm_provider_catalog_scope_service = FakeConsoleModelDiscoveryScope(
        (
            _merged_model("gpt-4.1"),
            _merged_model(
                "gpt-5",
                source="runtime_discovered",
                capability_status="unknown",
                persisted=False,
            ),
        )
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 60)) as pilot:
        console = host.screen_stack[-1]
        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)

        model_select = modal_screen.query_one("#console-settings-model-select", Select)
        assert {"gpt-4.1", "gpt-5"}.issubset(_select_values(model_select))

        model_select.value = "gpt-5"
        await pilot.pause()
        await pilot.click("#console-settings-save")
        await _wait_for_console_top_screen(host, console, pilot)
        await _visible_console_settings_button(console, pilot)
        for _ in range(40):
            summary_text = _summary_text(console)
            if "Model: gpt-5 (Capabilities unknown)" in summary_text:
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError(
                f"Console summary did not show discovered-model warning: {summary_text}"
            )

        _settings, readiness = console._active_console_settings_readiness()
        assert readiness.native_send_supported is True


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_to_no_models_allows_freeform_model_entry() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "custom": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "custom"
        await pilot.pause()

        picker = app.screen.query_one("#console-settings-model-picker")
        picker_input = picker.query_one("#model-search-picker-input", Input)
        picker_status = picker.query_one("#model-search-picker-status", Static)
        custom_button = app.screen.query_one("#console-settings-model-custom", Button)
        assert picker.value is None
        assert "No models reported" in str(picker_status.renderable)
        assert custom_button.display is True
        assert custom_button.disabled is False

        await pilot.click("#console-settings-model-custom")
        picker_input.value = "freeform-model"
        await pilot.pause()
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "custom"
    assert app.saved_settings.model == "freeform-model"


@pytest.mark.asyncio
async def test_console_settings_modal_accepts_keyboard_edited_freeform_model_input() -> (
    None
):
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "koboldcpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "koboldcpp"
        await pilot.pause()

        await pilot.click("#console-settings-model-custom")
        model_input = app.screen.query_one("#model-search-picker-input", Input)
        assert model_input.placeholder == "Choose or search models"

        await pilot.click(model_input)
        for character in "local-model":
            await pilot.press(character)
        assert model_input.value == "local-model"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "koboldcpp"
    assert app.saved_settings.model == "local-model"


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_uses_target_provider_model() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "openai"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.display is False
        assert model_select.disabled is True
        assert model_select.value == "gpt-4.1"
        assert model_input.display is True
        assert model_input.disabled is True
        assert model_input.value == "gpt-4.1"
        assert "model-a" not in _select_values(model_select)
        picker = app.screen.query_one(
            "#console-settings-model-picker", ModelSearchPicker
        )
        assert picker.value == "gpt-4.1"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "gpt-4.1"


@pytest.mark.asyncio
async def test_console_settings_modal_provider_round_trip_ignores_none_model_sentinel() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="koboldcpp",
        model=None,
        base_url="http://localhost:5001/api/v1/generate",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "koboldcpp": ["None"],
                    "Llama_cpp": ["None"],
                    "llama_cpp": ["model-a"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.disabled is False
        assert model_select.value == "model-a"
        assert "None" not in _select_values(model_select)
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "model-a"


@pytest.mark.asyncio
async def test_console_settings_modal_existing_none_model_sentinel_is_not_saved() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="None",
        base_url="http://127.0.0.1:9099",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "Llama_cpp": ["None"],
                    "llama_cpp": ["model-a"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.value == "model-a"
        assert "None" not in _select_values(model_select)
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.model == "model-a"


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_does_not_carry_base_url_to_non_url_provider() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "openai"
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        assert base_url_input.disabled is True or base_url_input.display is False
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.base_url is None


@pytest.mark.asyncio
async def test_console_settings_modal_restores_freeform_model_after_provider_round_trip() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="custom", model="freeform-model")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"custom": [], "llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()
        assert (
            app.screen.query_one("#console-settings-model-select", Select).value
            == "model-a"
        )

        app.screen.query_one("#console-settings-provider", Select).value = "custom"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.display is True
        assert model_select.disabled is False
        assert model_select.value == "freeform-model"
        assert model_input.display is False
        assert model_input.disabled is True
        assert model_input.value == "freeform-model"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "custom"
    assert app.saved_settings.model == "freeform-model"


@pytest.mark.asyncio
async def test_console_inspector_hosts_staged_context_above_source_readiness() -> None:
    """Task-400: the Context section tops the Inspector, not the left rail.

    The tray is the FIRST child of the Inspector rail body so it is visible
    without scrolling and reads above the run inspector's "Source Readiness"
    section; the bottom "Live work sources" card keeps its pre-move slot
    after the run-inspector block.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        staged_context = console.query_one("#console-staged-context-tray")
        settings = console.query_one("#console-settings-summary")
        rail_body = console.query_one("#console-inspector-rail-body")
        run_inspector = console.query_one("#console-run-inspector")
        readiness = console.query_one("#console-live-work-source-readiness")
        live_work = console.query_one("#console-live-work-section")
        project_status = console.query_one("#console-project-instruction-status")
        left_rail = console.query_one("#console-left-rail")

        # DOM order: tray first, then the run-inspector block (which renders
        # the Source Readiness section), then the bottom readiness card.
        assert settings.parent.id == "console-run-inspector"
        assert staged_context.parent is rail_body
        assert readiness in live_work.query("*")
        children = list(rail_body.children)
        assert children.index(project_status) == 0
        assert children.index(project_status) < children.index(staged_context)
        assert children.index(staged_context) < children.index(run_inspector)
        assert children.index(run_inspector) < children.index(live_work)

        # The left rail no longer hosts a Context section (header, body, or
        # tray): only Session, Model, Agent, and Details remain.
        assert not list(left_rail.query("#console-staged-context-tray"))
        assert not list(console.query("#console-rail-section-header-context"))
        assert not list(console.query("#console-rail-section-body-context"))

        # With the Inspector opened, the tray measures at the top of the rail
        # body, above the run inspector's "Source Readiness" heading (the
        # section the user sees) and above the bottom readiness card.
        await pilot.click("#console-inspector-rail-open")
        readiness_heading = console.query_one(
            "#console-inspector-source-readiness-heading"
        )
        for _ in range(40):
            if staged_context.region.height > 0 and readiness_heading.region.height > 0:
                break
            await pilot.pause(0.05)
        assert project_status.region.y == rail_body.region.y
        assert project_status.region.y < staged_context.region.y
        assert staged_context.region.y < readiness_heading.region.y
        assert staged_context.region.y < readiness.region.y


@pytest.mark.asyncio
async def test_console_left_rail_body_scrolls_below_fixed_header_without_settings_summary() -> (
    None
):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(100, 32)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        left_rail = console.query_one("#console-left-rail")
        header = console.query_one(".console-rail-header")
        body = console.query_one("#console-left-rail-body")
        conversations_body = console.query_one(
            "#console-rail-section-body-conversations"
        )
        settings = console.query_one("#console-settings-summary")
        workspace_context = console.query_one("#console-workspace-context")

        assert header.region.height == 1
        assert body.region.y >= header.region.y + header.region.height
        assert body.region.height <= left_rail.region.height - header.region.height
        assert settings.parent.id == "console-run-inspector"
        # TASK-14810 keeps the durable conversation browser in its dedicated
        # Conversations disclosure section while the whole section stack
        # remains inside the fixed-header rail scroller.
        assert workspace_context.parent is conversations_body
        conversations_section = body.query_one(
            "#console-bounded-section-conversations", ConsoleBoundedSection
        )
        assert conversations_section.parent is body
        assert conversations_body.parent is conversations_section.viewport
        viewport_width = conversations_section.viewport.region.width
        assert workspace_context.region.width <= viewport_width
        assert viewport_width - workspace_context.region.width <= 2


@pytest.mark.asyncio
async def test_console_settings_modal_save_updates_active_summary_only() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        store.replace_session_settings(
            first.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()

        second_id = await _press_new_console_tab(console, store, pilot)
        store.replace_session_settings(
            second_id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.dismiss(
            ConsoleSettingsResult(
                settings=ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
                user_display_name_override=None,
            )
        )
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await _visible_console_settings_button(console, pilot)

        summary_text = _summary_text(console)
        assert "Provider: openai" in summary_text
        assert "Model: gpt-4.1" in summary_text
        assert store.session_settings(second_id).provider == "openai"
        assert store.session_settings(first.id).provider == "llama_cpp"

        await _click_console_session_tab(console, store, pilot, first.id)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await _visible_console_settings_button(console, pilot)

        summary_text = _summary_text(console)
        assert "Provider: llama_cpp" in summary_text
        assert "Model: model-a" in summary_text


@pytest.mark.asyncio
async def test_console_settings_modal_result_stays_bound_to_opening_session() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
        "user_display_name": "User",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        store.replace_session_settings(
            first.id,
            ConsoleSessionSettings(
                provider="llama_cpp", model="model-a", system_prompt="First prompt"
            ),
        )
        second_id = await _press_new_console_tab(console, store, pilot)
        store.replace_session_settings(
            second_id,
            ConsoleSessionSettings(
                provider="llama_cpp", model="model-a", system_prompt="Second prompt"
            ),
        )
        await console._sync_native_console_chat_ui()

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        store.switch_session(first.id)
        modal_screen.dismiss(
            ConsoleSettingsResult(
                settings=ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
                user_display_name_override="Captain Rowan",
            )
        )
        await _wait_for_console_top_screen(host, console, pilot)
        await pilot.pause()

        assert store.active_session_id == first.id
        assert store.session_settings(first.id) == ConsoleSessionSettings(
            provider="llama_cpp", model="model-a", system_prompt="First prompt"
        )
        second = next(
            session for session in store.sessions() if session.id == second_id
        )
        assert store.session_settings(second_id) == ConsoleSessionSettings(
            provider="openai",
            model="gpt-4.1",
            system_prompt="Second prompt",
            source="user",
        )
        assert second.user_display_name_override == "Captain Rowan"


@pytest.mark.asyncio
async def test_console_settings_save_preserves_omitted_system_prompt_and_source() -> (
    None
):
    """The real general-settings draft omits prompt ownership entirely."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        session.assistant_kind = "character"
        session.character_name = "Alraune"
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        )
        store.seed_character_roleplay(
            session.id,
            system_template="Protect {{user}}.",
            greeting_template="",
            global_default="User",
        )
        system_prompt_writes: list[str | None] = []
        roleplay_writes: list[str | None] = []
        store.persistence = SimpleNamespace(
            update_conversation_system_prompt=lambda **kwargs: (
                system_prompt_writes.append(kwargs["system_prompt"]) or True
            ),
            update_conversation_roleplay_context=lambda **kwargs: (
                roleplay_writes.append(kwargs["character_system_template"]) or True
            ),
        )
        session.persisted_conversation_id = "conv-1"
        assert session.character_system_template == "Protect {{user}}."

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.query_one("#console-settings-temperature", Input).value = "0.5"
        await pilot.click("#console-settings-save")
        await _wait_for_console_top_screen(host, console, pilot)
        await pilot.pause()

        settings = store.session_settings(session.id)
        assert settings.temperature == 0.5
        assert settings.system_prompt == "Protect User."
        assert session.character_system_template == "Protect {{user}}."
        assert system_prompt_writes == []
        assert roleplay_writes == []


def test_console_settings_result_applies_name_override_without_losing_prompt_source(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Default Name"}
    notifications: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append(
            (message, kwargs.get("severity"))
        ),
    )
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.ensure_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    )
    session.assistant_kind = "character"
    session.character_name = "Alraune"
    store.seed_character_roleplay(
        session.id,
        system_template="Protect {{user}}.",
        greeting_template="",
        global_default="Default Name",
    )
    session.persisted_conversation_id = "conv-1"
    store.persistence = SimpleNamespace(
        update_conversation_system_prompt=lambda **_kwargs: True,
        update_conversation_roleplay_context=lambda **_kwargs: False,
    )
    monkeypatch.setattr(console, "_sync_console_identity_surfaces", lambda: None)
    monkeypatch.setattr(
        console,
        "run_worker",
        lambda coroutine, **_kwargs: coroutine.close(),
    )

    console._apply_console_settings_result(
        ConsoleSettingsResult(
            settings=ConsoleSessionSettings(
                provider="llama_cpp", model="model-a", temperature=0.5
            ),
            user_display_name_override="Captain Rowan",
        )
    )

    assert store.session_settings(session.id).temperature == 0.5
    assert store.session_settings(session.id).system_prompt == "Protect Captain Rowan."
    assert session.character_system_template == "Protect {{user}}."
    assert session.user_display_name_override == "Captain Rowan"
    assert (
        "Name changed for this session, but it may not survive reopening.",
        "warning",
    ) in notifications


@pytest.mark.asyncio
async def test_console_global_name_refresh_coalesces_and_respects_session_override(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Default One"}
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    inherited = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        assistant_kind="character",
        character_name="Alraune",
    )
    store.seed_character_roleplay(
        inherited.id,
        system_template="Protect {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Default One",
    )
    overridden = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        assistant_kind="character",
        character_name="Alraune",
    )
    store.seed_character_roleplay(
        overridden.id,
        system_template="Protect {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Default One",
    )
    store.set_session_user_display_name_override(
        overridden.id,
        "Captain Rowan",
        global_default="Default One",
    )
    queued = []
    surface_syncs = []
    monkeypatch.setattr(
        console,
        "run_worker",
        lambda coroutine, **_kwargs: queued.append(coroutine),
    )
    monkeypatch.setattr(
        console,
        "_sync_console_identity_surfaces",
        lambda: surface_syncs.append(store.active_session_id),
    )

    app.app_config["chat_defaults"]["user_display_name"] = "Default Two"
    store.switch_session(inherited.id)
    assert console._dispatch_active_console_roleplay_refresh() is True
    assert surface_syncs == [inherited.id]
    assert console._dispatch_active_console_roleplay_refresh() is False
    assert queued == []

    assert store.presentation_context(inherited.id, "Default Two").user_name == (
        "Default Two"
    )
    assert store.session_settings(inherited.id).system_prompt == "Protect Default Two."

    store.switch_session(overridden.id)
    assert console._dispatch_active_console_roleplay_refresh() is True
    assert surface_syncs[-1] == overridden.id
    assert queued == []
    assert store.presentation_context(overridden.id, "Default Two").user_name == (
        "Captain Rowan"
    )
    assert store.session_settings(overridden.id).system_prompt == (
        "Protect Captain Rowan."
    )

    store.set_session_user_display_name_override(
        overridden.id,
        None,
        global_default="Default Two",
    )
    assert store.presentation_context(overridden.id, "Default Two").user_name == (
        "Default Two"
    )
    assert store.session_settings(overridden.id).system_prompt == "Protect Default Two."
    assert surface_syncs == [
        inherited.id,
        overridden.id,
    ]


def test_console_identity_refresh_request_dispatches_without_transcript_tick(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Default One"}
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        assistant_kind="character",
        character_name="Alraune",
    )
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Protect {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Default One",
    )
    assert greeting is not None
    monkeypatch.setattr(
        console,
        "_sync_console_identity_surfaces",
        console._sync_console_chat_core_state,
    )

    app.app_config["chat_defaults"]["user_display_name"] = "Default Two"
    assert console.request_console_identity_refresh(1) is True
    assert console.request_console_identity_refresh(1) is False

    assert store.session_settings(session.id).system_prompt == "Protect Default Two."
    assert store.get_message(greeting.id).content == "Hello Default Two."


@pytest.mark.asyncio
async def test_real_inactive_console_tab_activation_dispatches_identity_refresh(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
        "user_display_name": "Default One",
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        first.assistant_kind = "character"
        first.character_name = "Alraune"
        greeting = store.seed_character_roleplay(
            first.id,
            system_template="Protect {{user}}.",
            greeting_template="Hello {{user}}.",
            global_default="Default One",
        )
        assert greeting is not None
        await _press_new_console_tab(console, store, pilot)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console,
            pilot,
            f"#console-session-tab-{first.id}",
        )
        dispatches = []
        original_dispatch = console._dispatch_active_console_roleplay_refresh

        def audited_dispatch():
            result = original_dispatch()
            dispatches.append(
                (
                    store.active_session_id,
                    console._global_chat_display_name(),
                    result,
                )
            )
            return result

        monkeypatch.setattr(
            console,
            "_dispatch_active_console_roleplay_refresh",
            audited_dispatch,
        )

        app.app_config["chat_defaults"]["user_display_name"] = "Default Two"
        await _click_console_session_tab(console, store, pilot, first.id)
        for _ in range(40):
            if store.session_settings(first.id).system_prompt == "Protect Default Two.":
                break
            await pilot.pause(0.05)

        assert (first.id, "Default Two", True) in dispatches
        assert store.session_settings(first.id).system_prompt == "Protect Default Two."
        assert store.get_message(greeting.id).content == "Hello Default Two."


@pytest.mark.asyncio
async def test_console_roleplay_refresh_serializes_blocked_b_then_c_without_stale_win(
    monkeypatch,
) -> None:
    class BlockingPersistence:
        def __init__(self) -> None:
            self.system_started = threading.Event()
            self.release_system = threading.Event()
            self.durable_system = "Speak with User."
            self.durable_greeting = "Hello User."
            self.writer_threads: list[int] = []

        def create_message(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, *, conversation_id, system_prompt):
            self.writer_threads.append(threading.get_ident())
            if system_prompt == "Speak with Bravo.":
                self.system_started.set()
                assert self.release_system.wait(5)
            self.durable_system = system_prompt
            return True

        def update_message_content(self, **kwargs):
            self.writer_threads.append(threading.get_ident())
            self.durable_greeting = kwargs["content"]
            return True

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "User"}
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    persistence = BlockingPersistence()
    store.persistence = persistence
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            system_prompt="Speak with User.",
        ),
        assistant_kind="character",
        character_name="Alraune",
    )
    session.persisted_conversation_id = "conv-1"
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="User",
    )
    assert greeting is not None
    controller = console._ensure_console_chat_controller()
    queued = []
    owner_thread = threading.get_ident()
    prepare_threads: list[int] = []
    original_prepare = store.prepare_session_roleplay_projection_refresh

    def audited_prepare(*args, **kwargs):
        prepare_threads.append(threading.get_ident())
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        store, "prepare_session_roleplay_projection_refresh", audited_prepare
    )
    monkeypatch.setattr(
        console,
        "run_worker",
        lambda coroutine, **_kwargs: queued.append(coroutine),
    )
    monkeypatch.setattr(
        console,
        "_sync_console_identity_surfaces",
        console._sync_console_chat_core_state,
    )

    app.app_config["chat_defaults"]["user_display_name"] = "Bravo"
    assert console._dispatch_active_console_roleplay_refresh() is True
    assert store.session_settings(session.id).system_prompt == "Speak with Bravo."
    assert store.get_message(greeting.id).content == "Hello Bravo."
    provider_system = controller._provider_messages_for_session(session.id)[0][
        "content"
    ]
    assert provider_system.startswith("Speak with Bravo.")
    assert provider_system.endswith("Hello Bravo.")
    assert await asyncio.to_thread(persistence.system_started.wait, 5)
    assert len(queued) == 1
    waiter = asyncio.create_task(queued.pop(0)())
    waiter.cancel()
    await asyncio.sleep(0)
    assert waiter.done() is True

    names = [f"Commander {index}" for index in range(25)]
    for name in names:
        app.app_config["chat_defaults"]["user_display_name"] = name
        assert console._dispatch_active_console_roleplay_refresh() is True
    assert console._dispatch_active_console_roleplay_refresh() is False
    assert store.session_settings(session.id).system_prompt == (
        "Speak with Commander 24."
    )
    assert store.get_message(greeting.id).content == "Hello Commander 24."
    provider_system = controller._provider_messages_for_session(session.id)[0][
        "content"
    ]
    assert provider_system.startswith("Speak with Commander 24.")
    assert provider_system.endswith("Hello Commander 24.")
    assert queued == []
    drain = console._console_roleplay_persistence_task
    assert drain is not None
    assert console._console_roleplay_active_plan is not None
    assert console._console_roleplay_pending_plan is not None
    assert (
        console._console_roleplay_pending_plan.generation == session.identity_revision
    )
    assert persistence.durable_system == "Speak with User."

    persistence.release_system.set()
    await drain

    assert console._console_roleplay_persistence_task is None
    assert console._console_roleplay_active_plan is None
    assert console._console_roleplay_pending_plan is None
    assert persistence.durable_system == "Speak with Commander 24."
    assert persistence.durable_greeting == "Hello Commander 24."
    assert prepare_threads == [owner_thread] * 26
    assert persistence.writer_threads
    assert all(thread_id != owner_thread for thread_id in persistence.writer_threads)


@pytest.mark.asyncio
async def test_console_roleplay_refresh_skips_plan_stale_before_writer() -> None:
    class RecordingPersistence:
        def __init__(self) -> None:
            self.system_writes: list[str | None] = []
            self.message_writes: list[str] = []

        def create_message(self, **_kwargs):
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            self.system_writes.append(kwargs["system_prompt"])
            return True

        def update_message_content(self, **kwargs):
            self.message_writes.append(kwargs["content"])
            return True

    app = _build_test_app()
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    persistence = RecordingPersistence()
    store.persistence = persistence
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", model="model-a", system_prompt="Speak with Alpha."
        ),
        assistant_kind="character",
        character_name="Alraune",
    )
    session.persisted_conversation_id = "conv-1"
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Alpha",
    )
    assert greeting is not None
    persistence.system_writes.clear()
    persistence.message_writes.clear()
    plan_b = store.prepare_session_roleplay_projection_refresh(
        session.id, global_default="Bravo"
    )
    plan_c = store.prepare_session_roleplay_projection_refresh(
        session.id, global_default="Cecelia"
    )
    assert plan_b is not None and plan_c is not None
    console._sync_console_identity_surfaces = lambda: None

    await console._refresh_console_roleplay_projections(plan_b)
    assert persistence.system_writes == []
    assert persistence.message_writes == []

    await console._refresh_console_roleplay_projections(plan_c)
    assert persistence.system_writes == ["Speak with Cecelia."]
    assert persistence.message_writes == ["Hello Cecelia."]


@pytest.mark.asyncio
async def test_cancelled_unmounted_drain_finishes_latest_plan(
    monkeypatch,
) -> None:
    class BlockingPersistence:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.durable_system = "Speak with Alpha."
            self.durable_greeting = "Hello Alpha."

        def create_message(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            self.started.set()
            assert self.release.wait(5)
            self.durable_system = kwargs["system_prompt"]
            return True

        def update_message_content(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return True

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Alpha"}
    console = ChatScreen(app)
    queued = []
    monkeypatch.setattr(
        console, "run_worker", lambda coroutine, **_kwargs: queued.append(coroutine)
    )
    monkeypatch.setattr(console, "_sync_console_identity_surfaces", lambda: None)
    store = console._ensure_console_chat_store()
    persistence = BlockingPersistence()
    store.persistence = persistence
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", system_prompt="Speak with Alpha."
        ),
        assistant_kind="character",
        character_name="Alraune",
    )
    session.persisted_conversation_id = "conv-1"
    store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Alpha",
    )

    app.app_config["chat_defaults"]["user_display_name"] = "Cecelia"
    assert console._dispatch_active_console_roleplay_refresh() is True
    assert await asyncio.to_thread(persistence.started.wait, 5)
    drain = console._console_roleplay_persistence_task
    assert drain is not None
    drain.cancel()
    await asyncio.sleep(0)
    assert drain.done() is False
    persistence.release.set()
    await drain

    assert console._console_roleplay_persistence_task is None
    assert console._console_roleplay_pending_plan is None
    assert persistence.durable_system == "Speak with Cecelia."
    assert persistence.durable_greeting == "Hello Cecelia."


@pytest.mark.asyncio
async def test_mounted_console_cancel_latest_waiter_keeps_durable_c() -> None:
    class BlockingPersistence:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.durable_system = "Speak with Alpha."
            self.durable_greeting = "Hello Alpha."

        def create_message(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            value = kwargs["system_prompt"]
            if value == "Speak with Bravo.":
                self.started.set()
                assert self.release.wait(5)
            self.durable_system = value
            return True

        def update_message_content(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return True

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Alpha"}
    host = ConsoleHarness(app)
    persistence = BlockingPersistence()

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.persistence = persistence
        session.assistant_kind = "character"
        session.character_name = "Alraune"
        session.persisted_conversation_id = "conv-1"
        greeting = store.seed_character_roleplay(
            session.id,
            system_template="Speak with {{user}}.",
            greeting_template="Hello {{user}}.",
            global_default="Alpha",
        )
        assert greeting is not None

        app.app_config["chat_defaults"]["user_display_name"] = "Bravo"
        assert console._dispatch_active_console_roleplay_refresh() is True
        assert await asyncio.to_thread(persistence.started.wait, 5)
        app.app_config["chat_defaults"]["user_display_name"] = "Cecelia"
        assert console._dispatch_active_console_roleplay_refresh() is True
        console.workers.cancel_group(console, "console-roleplay-refresh")
        await pilot.pause(0.05)
        persistence.release.set()
        for _ in range(100):
            if console._console_roleplay_persistence_task is None:
                break
            await pilot.pause(0.02)

        assert console._console_roleplay_persistence_task is None
        assert console._console_roleplay_active_plan is None
        assert console._console_roleplay_pending_plan is None
        assert persistence.durable_system == "Speak with Cecelia."
        assert persistence.durable_greeting == "Hello Cecelia."


@pytest.mark.asyncio
async def test_mounted_console_unmount_times_out_hung_refresh_and_repairs_on_resume(
    monkeypatch: pytest.MonkeyPatch,
):
    # This test owns the roleplay writer lifecycle. Task 5's independent
    # Changed Files thread worker may outlive the popped screen in Textual's
    # executor context, so keep it outside this weakref/GC assertion.
    monkeypatch.setattr(
        ChatScreen,
        "_console_changed_files_section_enabled",
        staticmethod(lambda: False),
    )

    class HungFirstWritePersistence:
        """One shared-store double: the FIRST system-prompt write blocks.

        task-16815 fixture correction: the Console runtime/store became
        app-owned (task-15860), so two co-mounted ChatScreens share ONE
        store -- the original per-screen persistence pair aliased to a
        single store and the repair write bound to the hung double
        (stack-verified 2026-08-16). One double now serves both roles:
        the hung screen's refresh write blocks until released; every
        later write (the app-level repair force-persist) records. The
        contract under test is unchanged: unmount bounds a stuck writer,
        and the repair persists the latest identity even while the
        original write is still blocked.
        """

        def __init__(self) -> None:
            self.durable_system = "Speak with Alpha."
            self.durable_greeting = "Hello Alpha."
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()
            self.first_system_write_seen = False

        def create_message(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return "msg-base"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            first_write = not self.first_system_write_seen
            self.first_system_write_seen = True
            if first_write:
                self.started.set()
                try:
                    assert self.release.wait(10)
                    # The released write still applies its side effect, as
                    # the original HungPersistence did via super() -- a
                    # completed write must persist what it carried (Qodo
                    # review, PR #1726).
                    self.durable_system = kwargs["system_prompt"]
                    return True
                finally:
                    self.finished.set()
            self.durable_system = kwargs["system_prompt"]
            return True

        def update_message_content(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return True

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Alpha"}
    app.app_config.setdefault("console", {})[
        "roleplay_refresh_teardown_timeout_seconds"
    ] = 0.05
    host = ConsoleHarness(app)
    hung_persistence = HungFirstWritePersistence()

    async with host.run_test(size=(160, 48)) as pilot:
        resumed = host.screen_stack[-1]
        await _wait_for_selector(resumed, pilot, "#console-settings-summary")
        resumed_store = resumed._ensure_console_chat_store()
        resumed_store.persistence = hung_persistence
        resumed_session = resumed_store.ensure_session()
        resumed_session.settings = ConsoleSessionSettings(
            provider="llama_cpp", system_prompt="Speak with Alpha."
        )
        resumed_session.assistant_kind = "character"
        resumed_session.character_name = "Alraune"
        resumed_session.persisted_conversation_id = "conv-base"
        resumed_store.seed_character_roleplay(
            resumed_session.id,
            system_template="Speak with {{user}}.",
            greeting_template="Hello {{user}}.",
            global_default="Alpha",
        )

        hung = ChatScreen(app)
        await host.push_screen(hung)
        await _wait_for_selector(hung, pilot, "#console-settings-summary")
        hung_store = hung._ensure_console_chat_store()
        hung_session = hung_store.ensure_session()
        hung_session.settings = ConsoleSessionSettings(
            provider="llama_cpp", system_prompt="Speak with Alpha."
        )
        hung_session.assistant_kind = "character"
        hung_session.character_name = "Alraune"
        hung_session.persisted_conversation_id = "conv-hung"
        hung_store.seed_character_roleplay(
            hung_session.id,
            system_template="Speak with {{user}}.",
            greeting_template="Hello {{user}}.",
            global_default="Alpha",
        )

        app.app_config["chat_defaults"]["user_display_name"] = "Cecelia"
        assert hung._dispatch_active_console_roleplay_refresh() is True
        assert await asyncio.to_thread(hung_persistence.started.wait, 5)
        writer_thread = hung._console_roleplay_writer_thread
        assert writer_thread is not None
        assert writer_thread.daemon is True
        old_screen = weakref.ref(hung)
        event_loop = asyncio.get_running_loop()
        loop_errors: list[dict[str, object]] = []
        previous_exception_handler = event_loop.get_exception_handler()
        event_loop.set_exception_handler(
            lambda _loop, context: loop_errors.append(context)
        )
        try:
            started_at = asyncio.get_running_loop().time()
            await host.pop_screen()

            elapsed = asyncio.get_running_loop().time() - started_at
            assert elapsed < 0.5
            assert app._console_roleplay_repair_generation == 1
            assert app._console_roleplay_repair_global_name == "Cecelia"
            for _ in range(100):
                if (
                    getattr(
                        app,
                        "_console_roleplay_repair_consumed_generation",
                        0,
                    )
                    == 1
                    and hung_persistence.durable_system == "Speak with Cecelia."
                    and hung_persistence.durable_greeting == "Hello Cecelia."
                ):
                    break
                await pilot.pause(0.01)
            assert app._console_roleplay_repair_consumed_generation == 1
            assert host.screen_stack[-1] is resumed
            assert hung_persistence.durable_system == "Speak with Cecelia."
            assert hung_persistence.durable_greeting == "Hello Cecelia."

            del hung, hung_store, hung_session
            for _ in range(50):
                gc.collect()
                if old_screen() is None:
                    break
                await pilot.pause(0.01)
            assert old_screen() is None
        finally:
            hung_persistence.release.set()
            assert await asyncio.to_thread(hung_persistence.finished.wait, 5)
            await pilot.pause(0.05)
            event_loop.set_exception_handler(previous_exception_handler)
        assert loop_errors == []


def test_mounted_hung_roleplay_writer_does_not_delay_event_loop_close():
    class HungPersistence:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def create_message(self, **_kwargs):
            return "msg-hung"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **_kwargs):
            self.started.set()
            assert self.release.wait(10)
            return True

        def update_message_content(self, **_kwargs):
            return True

    persistence = HungPersistence()
    state: dict[str, object] = {}

    async def exercise() -> None:
        app = _build_test_app()
        app.app_config["chat_defaults"] = {"user_display_name": "Alpha"}
        app.app_config.setdefault("console", {})[
            "roleplay_refresh_teardown_timeout_seconds"
        ] = 0.05
        host = ConsoleHarness(app)
        async with host.run_test(size=(160, 48)) as pilot:
            base = host.screen_stack[-1]
            await _wait_for_selector(base, pilot, "#console-settings-summary")
            hung = ChatScreen(app)
            await host.push_screen(hung)
            await _wait_for_selector(hung, pilot, "#console-settings-summary")
            store = hung._ensure_console_chat_store()
            store.persistence = persistence
            session = store.ensure_session()
            session.settings = ConsoleSessionSettings(
                provider="llama_cpp", system_prompt="Speak with Alpha."
            )
            session.assistant_kind = "character"
            session.character_name = "Alraune"
            session.persisted_conversation_id = "conv-hung"
            store.seed_character_roleplay(
                session.id,
                system_template="Speak with {{user}}.",
                greeting_template="Hello {{user}}.",
                global_default="Alpha",
            )
            app.app_config["chat_defaults"]["user_display_name"] = "Cecelia"
            assert hung._dispatch_active_console_roleplay_refresh() is True
            for _ in range(500):
                if persistence.started.is_set():
                    break
                await pilot.pause(0.01)
            assert persistence.started.is_set()
            writer_thread = hung._console_roleplay_writer_thread
            assert writer_thread is not None
            state["writer_thread"] = writer_thread
            state["screen_ref"] = weakref.ref(hung)
            state["shutdown_started_at"] = time.monotonic()
            await host.pop_screen()
            del hung, store, session

    try:
        asyncio.run(exercise())
        state["close_elapsed"] = time.monotonic() - float(state["shutdown_started_at"])
        for _ in range(50):
            gc.collect()
            screen_ref = state.get("screen_ref")
            if callable(screen_ref) and screen_ref() is None:
                break
    finally:
        persistence.release.set()
        writer_thread = state.get("writer_thread")
        if isinstance(writer_thread, threading.Thread):
            writer_thread.join(5)

    assert float(state["close_elapsed"]) < 1.5
    writer_thread = state["writer_thread"]
    assert isinstance(writer_thread, threading.Thread)
    assert writer_thread.daemon is True
    screen_ref = state["screen_ref"]
    assert callable(screen_ref)
    assert screen_ref() is None


@pytest.mark.asyncio
async def test_roleplay_repair_marker_retries_partial_then_consumes(monkeypatch):
    class PartialPersistence:
        def __init__(self) -> None:
            self.fail_messages = True
            self.durable_system = "Speak with Alpha."
            self.durable_greeting = "Hello Alpha."

        def create_message(self, **kwargs):
            self.durable_greeting = kwargs["content"]
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            self.durable_system = kwargs["system_prompt"]
            return True

        def update_message_content(self, **kwargs):
            if self.fail_messages:
                return False
            self.durable_greeting = kwargs["content"]
            return True

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Cecelia"}
    app._console_roleplay_repair_generation = 1
    app._console_roleplay_repair_global_name = "Cecelia"
    notifications: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append(
            (message, kwargs.get("severity"))
        ),
    )
    console = ChatScreen(app)
    persistence = PartialPersistence()
    store = console._ensure_console_chat_store()
    store.persistence = persistence
    session = store.ensure_session()
    session.settings = ConsoleSessionSettings(
        provider="llama_cpp", system_prompt="Speak with Cecelia."
    )
    session.assistant_kind = "character"
    session.character_name = "Alraune"
    session.persisted_conversation_id = "conv-1"
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Cecelia",
    )
    assert greeting is not None
    monkeypatch.setattr(console, "_sync_console_identity_surfaces", lambda: None)
    queued = []
    monkeypatch.setattr(
        console,
        "run_worker",
        lambda coroutine, **_kwargs: queued.append(coroutine),
    )

    assert console._consume_pending_console_roleplay_repair() is True
    await queued.pop(0)()
    assert getattr(app, "_console_roleplay_repair_consumed_generation", 0) == 0
    assert console._console_roleplay_repair_inflight_generation == 0
    assert len([note for note in notifications if note[1] == "warning"]) == 1

    persistence.fail_messages = False
    assert console._consume_pending_console_roleplay_repair() is True
    await queued.pop(0)()
    assert app._console_roleplay_repair_consumed_generation == 1
    assert persistence.durable_system == "Speak with Cecelia."
    assert persistence.durable_greeting == "Hello Cecelia."
    assert len([note for note in notifications if note[1] == "warning"]) == 1


@pytest.mark.asyncio
async def test_console_global_name_refresh_failure_notifies_once(monkeypatch) -> None:
    class RefusingPersistence:
        def __init__(self) -> None:
            self.system_writes: list[str | None] = []
            self.message_writes: list[str] = []

        def create_message(self, **_kwargs):
            return "msg-1"

        def update_conversation_roleplay_context(self, **_kwargs):
            return True

        def update_conversation_system_prompt(self, **kwargs):
            self.system_writes.append(kwargs["system_prompt"])
            return False

        def update_message_content(self, **kwargs):
            self.message_writes.append(kwargs["content"])
            return False

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"user_display_name": "Default Name"}
    notifications: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append(
            (message, kwargs.get("severity"))
        ),
    )
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    persistence = RefusingPersistence()
    store.persistence = persistence
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            system_prompt="Protect Default Name.",
        ),
        assistant_kind="character",
        character_name="Alraune",
    )
    session.persisted_conversation_id = "conv-1"
    greeting = store.seed_character_roleplay(
        session.id,
        system_template="Protect {{user}}.",
        greeting_template="Hello {{user}}.",
        global_default="Default Name",
    )
    assert greeting is not None
    persistence.system_writes.clear()
    persistence.message_writes.clear()
    controller = console._ensure_console_chat_controller()
    queued = []
    estimate_calls = []

    def estimate(messages, provider, model, **kwargs):
        estimate_calls.append((messages, provider, model, kwargs["system_prompt"]))
        return ConsoleSettingsContextEstimate(
            used_tokens=17,
            token_limit=4096,
            label="17 / 4096 tokens",
        )

    monkeypatch.setattr(chat_screen_module, "build_console_context_estimate", estimate)
    monkeypatch.setattr(
        console,
        "run_worker",
        lambda coroutine, **_kwargs: queued.append(coroutine),
    )
    monkeypatch.setattr(
        console,
        "_sync_console_identity_surfaces",
        console._sync_console_chat_core_state,
    )

    app.app_config["chat_defaults"]["user_display_name"] = "Captain Rowan"
    assert console._dispatch_active_console_roleplay_refresh() is True
    assert console._dispatch_active_console_roleplay_refresh() is False
    assert store.session_settings(session.id).system_prompt == "Protect Captain Rowan."
    assert store.get_message(greeting.id).content == "Hello Captain Rowan."
    provider_system = controller._provider_messages_for_session(session.id)[0][
        "content"
    ]
    assert provider_system.startswith("Protect Captain Rowan.")
    assert provider_system.endswith("Hello Captain Rowan.")
    estimate_result = console._active_console_settings_context_estimate()
    assert estimate_result.used_tokens == 17
    assert estimate_calls[-1][0][-1]["content"] == "Hello Captain Rowan."
    assert estimate_calls[-1][3] == "Protect Captain Rowan."
    await queued.pop(0)()

    expected = (
        "Your chat name is active, but updated character templates may not survive "
        "reopening."
    )
    assert notifications.count((expected, "warning")) == 1
    assert persistence.system_writes == ["Protect Captain Rowan."]
    assert persistence.message_writes == ["Hello Captain Rowan."]
    assert store.session_settings(session.id).system_prompt == "Protect Captain Rowan."
    assert store.get_message(greeting.id).content == "Hello Captain Rowan."
    provider_system = controller._provider_messages_for_session(session.id)[0][
        "content"
    ]
    assert provider_system.startswith("Protect Captain Rowan.")
    assert provider_system.endswith("Hello Captain Rowan.")
    assert store.active_session_id == session.id


@pytest.mark.asyncio
async def test_system_prompt_editor_clears_character_template_source() -> None:
    """The dedicated prompt editor owns explicit prompt replacement."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        session.assistant_kind = "character"
        session.character_name = "Alraune"
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        )
        store.seed_character_roleplay(
            session.id,
            system_template="Protect {{user}}.",
            greeting_template="",
            global_default="User",
        )

        console.run_worker(
            console._open_console_system_prompt_editor(), exclusive=False
        )
        await pilot.pause(0.2)
        modal = host.screen_stack[-1]
        modal.query_one(
            f"#{SYSTEM_PROMPT_TEXT_AREA_ID}", TextArea
        ).text = "Manual prompt."
        modal.query_one(f"#{SYSTEM_PROMPT_APPLY_BUTTON_ID}", Button).press()
        await pilot.pause(0.2)

        assert store.session_settings(session.id).system_prompt == "Manual prompt."
        assert session.character_system_template is None


def test_system_prompt_command_clears_character_template_through_store(
    monkeypatch,
) -> None:
    """The `/system` apply path must retain the store's provenance revocation."""
    app = _build_test_app()
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.ensure_session()
    session.assistant_kind = "character"
    session.character_name = "Alraune"
    store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
    )
    store.seed_character_roleplay(
        session.id,
        system_template="Protect {{user}}.",
        greeting_template="",
        global_default="User",
    )
    monkeypatch.setattr(console, "_sync_console_chat_core_state", lambda: None)
    monkeypatch.setattr(console, "_sync_console_settings_summary", lambda: None)
    monkeypatch.setattr(console, "_sync_console_control_bar", lambda: None)

    console._session._apply_console_session_system_prompt("Manual slash prompt.")

    assert store.session_settings(session.id).system_prompt == "Manual slash prompt."
    assert session.character_system_template is None


@pytest.mark.asyncio
async def test_console_settings_are_isolated_between_native_tabs() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        store.replace_session_settings(
            first.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()

        second_id = await _press_new_console_tab(console, store, pilot)
        store.replace_session_settings(
            second_id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()
        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.dismiss(
            ConsoleSettingsResult(
                settings=ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
                user_display_name_override=None,
            )
        )
        await _wait_for_console_top_screen(host, console, pilot)
        await _click_console_session_tab(console, store, pilot, first.id)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert console._build_console_provider_selection().provider == "llama_cpp"
        await _click_console_session_tab(console, store, pilot, second_id)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        assert console._build_console_provider_selection().provider == "openai"


@pytest.mark.asyncio
async def test_console_native_tab_click_switches_without_programmatic_fallback() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id
        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{first_id}")

        first_tab = console.query_one(f"#console-session-tab-{first_id}", Button)
        assert await pilot.click(first_tab, offset=(1, 0))
        for _ in range(10):
            if store.active_session_id == first_id:
                break
            await pilot.pause(0.05)

        assert store.active_session_id == first_id
        assert store.active_session_id != second_id


@pytest.mark.asyncio
async def test_console_workspace_conversation_row_switches_native_tab() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id
        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-1")

        first_conversation = console.query_one(
            "#console-workspace-conversation-1", Button
        )
        assert (
            getattr(first_conversation, "conversation_id", None) == f"native:{first_id}"
        )
        first_conversation.press()
        for _ in range(10):
            if store.active_session_id == first_id:
                break
            await pilot.pause(0.05)

        assert store.active_session_id == first_id
        assert store.active_session_id != second_id


@pytest.mark.asyncio
async def test_console_provider_selection_includes_generation_controls() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(
                provider="openai",
                model="gpt-4.1",
                seed=17,
                presence_penalty=0.4,
                frequency_penalty=0.5,
                reasoning_effort="high",
                reasoning_summary="auto",
                verbosity="medium",
                thinking_effort="low",
                thinking_budget_tokens=2048,
            ),
        )
        await console._sync_native_console_chat_ui()

        selection = console._build_console_provider_selection()

    assert selection.seed == 17
    assert selection.presence_penalty == 0.4
    assert selection.frequency_penalty == 0.5
    assert selection.reasoning_effort == "high"
    assert selection.reasoning_summary == "auto"
    assert selection.verbosity == "medium"
    assert selection.thinking_effort == "low"
    assert selection.thinking_budget_tokens == 2048


@pytest.mark.asyncio
async def test_console_settings_modal_cancel_keeps_original_summary() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()
        await _visible_console_settings_button(console, pilot)
        original_summary = _summary_text(console)

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.dismiss(None)
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert _summary_text(console) == original_summary
        assert store.session_settings(session.id).provider == "llama_cpp"


@pytest.mark.asyncio
async def test_console_settings_modal_save_disabled_during_active_run() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"}
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        await console._sync_native_console_chat_ui()
        controller = console._ensure_console_chat_controller()
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
        )

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)

        assert modal_screen.query_one("#console-settings-save", Button).disabled is True


@pytest.mark.asyncio
async def test_console_settings_save_clears_stale_terminal_run_status() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "custom": {
            "api_url": "http://localhost:1234/v1/chat/completions",
            "model": "custom-model-beta",
        },
    }
    app.providers_models = {
        "llama_cpp": ["model-a"],
        "custom": ["custom-model-alpha", "custom-model-beta"],
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        )
        await console._sync_native_console_chat_ui()

        controller = console._ensure_console_chat_controller()
        stale_copy = "Provider blocked: old llama.cpp failure."
        controller._set_run_state(ConsoleRunState.blocked(stale_copy))
        console._sync_console_mode_bar()
        assert stale_copy in str(
            console.query_one("#console-mode-bar", Static).renderable
        )

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.dismiss(
            ConsoleSessionSettings(
                provider="custom",
                model="custom-model-beta",
                base_url="http://localhost:1234/v1/chat/completions",
            )
        )
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert console._build_console_provider_selection().provider == "custom"
        assert controller.run_state.status is ConsoleRunStatus.IDLE
        assert stale_copy not in str(
            console.query_one("#console-mode-bar", Static).renderable
        )


@pytest.mark.asyncio
async def test_console_send_blocker_uses_saved_unsupported_session_provider() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="wip_provider", model="test-model"),
        )
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer")
        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        for _ in range(40):
            if "Provider blocked" in _screen_visible_text(console):
                break
            await pilot.pause(0.05)

        assert (
            "Provider blocked: 'wip_provider' is not available in Console yet."
            in _screen_visible_text(console)
        )


@pytest.mark.asyncio
async def test_console_missing_model_opens_console_settings_from_summary() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _visible_console_settings_button(console, pilot)
        # The shared Workbench recovery banner stays hidden — the setup
        # card's action button carries this recovery instead (Phase 2 spec,
        # section 2).
        await _wait_for_selector(console, pilot, "#console-setup-modal-action")

        recovery_button = console.query_one("#console-setup-modal-action", Button)
        assert str(recovery_button.label) == "Choose model"
        assert recovery_button.display is True

        recovery_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        await _wait_for_focused_id(host, pilot, "model-search-picker-input")

        assert (
            modal_screen.query_one("#console-settings-provider", Select).value
            == "llama_cpp"
        )
        assert modal_screen.query_one(ModelSearchPicker).value == "model-a"
        readiness = modal_screen.query_one("#console-settings-readiness", Static)
        provider_model_section = modal_screen.query_one(
            "#console-settings-provider-model-section"
        )
        assert (
            str(readiness.renderable) == "llama_cpp is ready. No API key is required."
        )
        assert (
            provider_model_section.has_class("console-settings-primary-section")
            is False
        )

        await pilot.click("#console-settings-save")
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await _visible_console_settings_button(console, pilot)

        text = _screen_visible_text(console)
        assert "Model: model-a" in _summary_text(console)
        assert "Setup required: choose a model in Console Settings." not in text
        assert console._console_send_blocked_reason() == ""


@pytest.mark.asyncio
async def test_console_llamacpp_saved_missing_model_blocks_before_send() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id, ConsoleSessionSettings(provider="llama_cpp", model=None)
        )
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer")
        composer.load_draft("hello")
        send_button = console.query_one("#console-send-message", Button)
        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.1)

        # TASK-2154.6 (FR-04): Send is now genuinely disabled while setup
        # blocks, so the press above is a no-op by design; the persistent
        # reason strip (plus the kept tooltip) is the pre-click affordance.
        assert send_button.disabled is True
        reason = console.query_one("#console-send-disabled-reason")
        assert reason.styles.display == "block"
        assert reason.renderable.plain == "Send blocked — choose a model to continue"
        assert (
            send_button.tooltip == "Choose a model in Console Settings before sending."
        )
        assert (
            "Console send blocked: Select a model before sending."
            not in _screen_visible_text(console)
        )
        assert (
            "Setup required: choose a model in Console Settings."
            not in _screen_visible_text(console)
        )
        assert composer.draft_text() == "hello"


def test_console_default_settings_keep_configured_model_without_legacy_model() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1",
            "model": "configured-model",
        },
    }
    screen = ChatScreen(app)

    settings = screen._session._default_console_session_settings()

    assert settings.provider == "llama_cpp"
    assert settings.model == "configured-model"


def test_console_settings_summary_uses_effective_config_endpoint_for_llamacpp_defaults() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1",
            "model": "configured-model",
        },
    }
    screen = ChatScreen(app)

    summary_state = screen._build_console_settings_summary_state()

    assert summary_state.endpoint_row == "Endpoint: http://127.0.0.1:9099"


def test_console_readiness_uses_saved_session_settings_over_stale_global_provider() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099/v1", "model": "local-model"},
        "openai": {},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="llama_cpp", model="local-model")
    )

    control_state = screen._build_console_control_state(None)
    inspector_state = screen._build_console_inspector_state(None)
    provider_row = next(row for row in inspector_state.rows if row.label == "Provider")

    assert screen._console_provider_blocker_copy() == ""
    assert control_state.provider_label == "Provider: llama_cpp"
    assert control_state.model_label == "Model: local-model"
    assert provider_row.value == "ready"
    assert provider_row.recovery == ""


def test_console_control_state_reads_persona_label_without_storing_it_on_session(
    monkeypatch,
) -> None:
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    monkeypatch.setattr(
        screen._session,
        "_active_native_console_session",
        lambda: SimpleNamespace(
            assistant_kind="persona",
            assistant_name="Guide",
            assistant_id="persona-7",
        ),
    )

    state = screen._build_console_control_state(None)

    assert state.assistant_label == "Persona: Guide"
    assert session.assistant_kind == "generic"
    assert session.assistant_id == "console"
    assert session.assistant_authority_id is None
    assert "assistant_kind" in session.__dataclass_fields__
    assert "assistant_id" in session.__dataclass_fields__
    assert "assistant_name" not in session.__dataclass_fields__


def test_console_saved_openai_with_key_shows_ready_readiness() -> None:
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="openai", model="gpt-4.1")
    )

    summary_state = screen._build_console_settings_summary_state()
    inspector_state = screen._build_console_inspector_state(None)
    provider_row = next(row for row in inspector_state.rows if row.label == "Provider")
    blocker_copy = screen._console_provider_blocker_copy()

    assert summary_state.readiness_label == "Ready"
    assert provider_row.value == "ready"
    assert provider_row.recovery == ""
    assert blocker_copy == ""
    assert screen._console_send_blocked_reason() == ""


def test_console_missing_key_recovery_action_is_provider_specific() -> None:
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY", "model": "gpt-4.1"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="openai", model="gpt-4.1")
    )

    label, target, tooltip = screen._console_provider_recovery_action()

    assert (
        screen._console_provider_blocker_copy()
        == "Provider setup needed: OpenAI missing API key"
    )
    assert label == CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL
    assert target == "settings"
    assert tooltip == "Configure OpenAI API and API key in Settings"
    assert screen._console_provider_recovery_field() == "api_key"
    assert (
        screen._console_setup_blocked_reason()
        == "Add API key in Settings > Providers & Models before sending."
    )


def test_console_unsaved_generic_endpoint_blocks_inspector_with_endpoint_details() -> (
    None
):
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "ollama": {"api_url": "http://127.0.0.1:11434", "model": "llama3"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(
            provider="ollama",
            model="llama3",
            base_url="http://127.0.0.1:9999/v1",
        ),
    )

    inspector_state = screen._build_console_inspector_state(None)
    provider_row = next(row for row in inspector_state.rows if row.label == "Provider")
    label, target, tooltip = screen._console_provider_recovery_action()

    assert provider_row.value == "blocked"
    assert "Selected endpoint: http://127.0.0.1:9999/v1" in provider_row.recovery
    assert "Saved endpoint: http://127.0.0.1:11434" in provider_row.recovery
    assert "save the endpoint in Settings" in screen._console_provider_blocker_copy()
    assert label == "Configure endpoint"
    assert target == "settings"
    assert tooltip == "Save the Ollama endpoint in Settings"
    assert screen._console_provider_recovery_field() == "endpoint"
    assert (
        screen._console_setup_blocked_reason()
        == "Save provider endpoint in Settings > Providers & Models before sending."
    )


def test_console_no_provider_recovery_action_and_card_step_are_provider_actions() -> (
    None
):
    """FR-05/FR-07: no provider at all -> provider action, no empty '' copy."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {}
    app.app_config["api_settings"] = {}
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="", model=None)
    )

    label, target, _tooltip = screen._console_provider_recovery_action()
    card_state = screen._build_console_setup_card_state()
    _settings, readiness = screen._active_console_settings_readiness()

    assert (
        screen._console_provider_blocker_copy()
        == "Provider setup needed: choose a provider"
    )
    assert label == "Choose provider"
    assert target == "console"
    assert screen._console_provider_recovery_field() == ""
    assert readiness.label == "Unknown"
    assert "Select a provider" in readiness.detail
    assert "''" not in readiness.detail
    assert card_state.mode == "card"
    step_one, step_two, _step_three = card_state.steps
    assert step_one.state == "active"
    assert step_one.label == "Choose a supported provider"
    assert "''" not in step_one.label
    assert step_two.state == "pending"


def test_console_missing_key_no_model_recovery_action_is_provider_action() -> None:
    """FR-05: with provider blocked AND model missing, the provider blocker wins."""
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="openai", model=None)
    )

    label, target, _tooltip = screen._console_provider_recovery_action()
    card_state = screen._build_console_setup_card_state()
    _settings, readiness = screen._active_console_settings_readiness()

    assert readiness.label == "Missing key"
    assert readiness.native_send_supported is False
    assert (
        screen._console_provider_blocker_copy()
        == "Provider setup needed: OpenAI missing API key"
    )
    assert label == CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL
    assert target == "settings"
    assert screen._console_provider_recovery_field() == "api_key"
    step_one, step_two, _step_three = card_state.steps
    assert step_one.state == "active"
    assert step_one.label == "Connect a provider (API key or local server)"
    assert step_two.state == "pending"


def test_console_provider_ready_missing_model_keeps_choose_model_action() -> None:
    """FR-05 regression: provider ready + model missing -> Choose model, step 1 done."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="llama_cpp", model=None)
    )

    label, target, _tooltip = screen._console_provider_recovery_action()
    card_state = screen._build_console_setup_card_state()
    _settings, readiness = screen._active_console_settings_readiness()

    assert readiness.label == "Missing model"
    assert readiness.native_send_supported is False
    assert (
        screen._console_provider_blocker_copy()
        == "Provider setup needed: choose a model"
    )
    assert label == "Choose model"
    assert target == "console"
    assert screen._console_provider_recovery_field() == ""
    assert (
        screen._console_send_blocked_reason()
        == "Console send blocked: Select a model before sending."
    )
    step_one, step_two, _step_three = card_state.steps
    assert step_one.state == "done"
    assert step_one.label == "Provider ready"
    assert step_two.state == "active"
    assert step_two.label == "Pick a model"


def test_console_unsaved_endpoint_no_model_recovery_action_is_configure_endpoint() -> (
    None
):
    """FR-05: unsaved endpoint + no model -> Configure endpoint, step 1 active."""
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "ollama": {"api_url": "http://127.0.0.1:11434"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(
            provider="ollama",
            model=None,
            base_url="http://127.0.0.1:9999/v1",
        ),
    )

    label, target, _tooltip = screen._console_provider_recovery_action()
    card_state = screen._build_console_setup_card_state()

    assert "save the endpoint in Settings" in screen._console_provider_blocker_copy()
    assert label == "Configure endpoint"
    assert target == "settings"
    assert screen._console_provider_recovery_field() == "endpoint"
    step_one, step_two, _step_three = card_state.steps
    assert step_one.state == "active"
    assert step_one.label == "Save the provider's server address (endpoint)"
    assert step_two.state == "pending"


def test_console_invalid_endpoint_no_model_recovery_action_is_configure_endpoint() -> (
    None
):
    """FR-05: invalid endpoint + no model -> Configure endpoint, step 1 active."""
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "ollama": {"api_url": "http://127.0.0.1:11434"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(
            provider="ollama",
            model=None,
            base_url="not-a-url",
        ),
    )

    label, target, _tooltip = screen._console_provider_recovery_action()
    card_state = screen._build_console_setup_card_state()
    _settings, readiness = screen._active_console_settings_readiness()

    assert readiness.label == "Invalid URL"
    assert "invalid base URL" in screen._console_provider_blocker_copy()
    assert label == "Configure endpoint"
    assert target == "settings"
    assert screen._console_provider_recovery_field() == "endpoint"
    step_one, step_two, _step_three = card_state.steps
    assert step_one.state == "active"
    assert step_one.label == "Save the provider's server address (endpoint)"
    assert step_two.state == "pending"


def test_console_saved_llamacpp_missing_model_summary_is_not_ready_without_fallback() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="llama_cpp", model=None)
    )

    summary_state = screen._build_console_settings_summary_state()

    assert summary_state.readiness_label != "Ready"
    assert summary_state.provider_row == "Provider: llama_cpp"
    assert summary_state.model_row == "Model: Missing"
    assert (
        screen._console_send_blocked_reason()
        == "Console send blocked: Select a model before sending."
    )


def test_console_saved_llamacpp_missing_model_summary_ready_with_configured_fallback() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "configured-model",
        },
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="llama_cpp", model=None)
    )

    summary_state = screen._build_console_settings_summary_state()

    assert summary_state.readiness_label == "Ready"
    assert "Select a model before sending" not in summary_state.model_row


@pytest.mark.asyncio
async def test_console_new_native_tab_receives_default_settings_snapshot() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1",
            "model": "configured-model",
        },
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id

        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert second_id != first_id
        settings = store.session_settings(second_id)
        assert settings is not None
        assert settings.provider == "llama_cpp"
        assert settings.model == "configured-model"


@pytest.mark.asyncio
async def test_console_new_native_tab_inherits_active_session_settings_snapshot() -> (
    None
):
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "local-model",
        },
    }
    app.providers_models = {
        "openai": ["gpt-4.1"],
        "local_llamacpp": ["local-model"],
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id
        active_settings = ConsoleSessionSettings(
            provider="local_llamacpp",
            model="local-model",
            base_url="http://127.0.0.1:9099",
            temperature=0.2,
            top_p=0.8,
            streaming=False,
        )
        store.replace_session_settings(first_id, active_settings)
        await console._sync_native_console_chat_ui()

        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert second_id != first_id
        assert store.session_settings(second_id) == active_settings


@pytest.mark.asyncio
async def test_console_model_switch_inherits_selected_model_default_profile() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model_defaults": {
                "gpt-4.1": {"temperature": 0.2, "top_p": 0.8, "streaming": True},
                "gpt-4.1-mini": {"temperature": 0.45, "top_p": 0.9, "streaming": False},
            },
        },
    }
    app.providers_models = {"openai": ["gpt-4.1", "gpt-4.1-mini"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()

        initial_settings = store.session_settings(session.id)
        assert initial_settings is not None
        assert initial_settings.model == "gpt-4.1"
        assert initial_settings.temperature == 0.2

        console._sync_compact_shell_controls(model="gpt-4.1-mini")
        await pilot.pause()

        updated_settings = store.session_settings(session.id)
        assert updated_settings is not None
        assert updated_settings.model == "gpt-4.1-mini"
        assert updated_settings.temperature == 0.45
        assert updated_settings.top_p == 0.9
        assert updated_settings.streaming is False


@pytest.mark.asyncio
async def test_console_model_switch_preserves_explicit_session_overrides() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model_defaults": {
                "gpt-4.1": {"temperature": 0.2, "top_p": 0.8, "streaming": True},
                "gpt-4.1-mini": {"temperature": 0.45, "top_p": 0.9, "streaming": False},
            },
        },
    }
    app.providers_models = {"openai": ["gpt-4.1", "gpt-4.1-mini"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()

        console._sync_compact_shell_controls(temperature="0.33")
        console._sync_compact_shell_controls(model="gpt-4.1-mini")
        await pilot.pause()

        updated_settings = store.session_settings(session.id)
        assert updated_settings is not None
        assert updated_settings.model == "gpt-4.1-mini"
        assert updated_settings.temperature == 0.33
        assert updated_settings.top_p == 0.9
        assert updated_settings.streaming is False


# --- task-177: readiness must follow Settings saves without an app restart ---


def _disk_loaded_snapshot(**overrides) -> dict:
    """Snapshot shaped like a real ``load_settings()`` boot config."""
    snapshot = {
        "general": {},
        "logging": {},
        "splash_screen": {},
        "api_settings": {"openai": {"api_key": ""}},
    }
    snapshot.update(overrides)
    return snapshot


def test_provider_readiness_config_refreshes_disk_loaded_snapshot(monkeypatch) -> None:
    app = _build_test_app()
    app.app_config = _disk_loaded_snapshot()
    console = ChatScreen(app)
    fresh = _disk_loaded_snapshot(api_settings={"openai": {"api_key": "sk-fresh"}})
    monkeypatch.setattr(chat_screen_module, "load_settings", lambda: fresh)

    assert console._provider_readiness_app_config() is fresh


def test_provider_readiness_config_honors_injected_test_snapshot(monkeypatch) -> None:
    """Fakes without the disk-loaded marker sections stay authoritative."""
    app = _build_test_app()
    app.app_config = {"api_settings": {"openai": {"api_key": "injected"}}}
    console = ChatScreen(app)

    def _fail_load_settings():
        raise AssertionError(
            "load_settings must not be consulted for injected snapshots"
        )

    monkeypatch.setattr(chat_screen_module, "load_settings", _fail_load_settings)

    assert console._provider_readiness_app_config() is app.app_config


def test_provider_readiness_config_falls_back_when_load_settings_fails(
    monkeypatch,
) -> None:
    app = _build_test_app()
    app.app_config = _disk_loaded_snapshot()
    console = ChatScreen(app)

    def _boom():
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(chat_screen_module, "load_settings", _boom)

    assert console._provider_readiness_app_config() is app.app_config


def test_console_readiness_unblocks_after_provider_save_without_restart(
    monkeypatch, tmp_path
) -> None:
    """Save a provider key via the config API after boot; readiness must see it.

    Mirrors the live UAT failure: Settings saved the key, the config module
    cache reloaded, but Console kept reading the boot-time ``app_config``
    snapshot until restart.
    """
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    config_path = tmp_path / "console-readiness-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        app = _build_test_app()
        # Boot-time snapshot: disk-loaded shape, but captured before the save.
        app.app_config = _disk_loaded_snapshot()
        console = ChatScreen(app)
        settings = ConsoleSessionSettings(provider="openai", model="gpt-4o")

        readiness_before = build_console_settings_readiness(
            settings,
            app_config=console._provider_readiness_app_config(),
            environ={},
        )
        assert readiness_before.native_send_supported is False

        # The Settings screen save path: config API write + cache reload.
        assert config_module.save_setting_to_cli_config(
            "api_settings.openai", "api_key", "sk-saved-after-boot"
        )

        readiness_after = build_console_settings_readiness(
            settings,
            app_config=console._provider_readiness_app_config(),
            environ={},
        )
        assert readiness_after.native_send_supported is True
        assert readiness_after.label == "Ready"
        # The stale snapshot alone would still be blocked - proving the fresh
        # read (not the snapshot) unblocked readiness.
        readiness_stale = build_console_settings_readiness(
            settings,
            app_config=app.app_config,
            environ={},
        )
        assert readiness_stale.native_send_supported is False
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)


# --- task-178: settings modal persistence affordance, boolean streaming, focus artifact ---


def _basic_modal(
    settings: ConsoleSessionSettings, app: "ModalHarness", **kwargs
) -> ConsoleSettingsModal:
    return ConsoleSettingsModal(
        settings=settings,
        app_config=app.app_config,
        providers_models=kwargs.pop("providers_models", {"llama_cpp": ["model-a"]}),
        context_estimate=kwargs.pop(
            "context_estimate", ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
        ),
        can_save=kwargs.pop("can_save", True),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_console_settings_modal_streaming_is_boolean_toggle() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", streaming=False
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        toggle = app.screen.query_one("#console-settings-streaming", Button)
        assert str(toggle.label) == "Off"

        toggle.press()
        await pilot.pause()
        assert str(toggle.label) == "On"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.streaming is True


@pytest.mark.asyncio
async def test_console_settings_modal_enumerated_inputs_list_accepted_values() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        # llama.cpp is a local thinking provider: only the reasoning-effort
        # level is consumed, so the other choice inputs carry the no-effect
        # suffix.
        placeholders = {
            "console-settings-reasoning-effort": "none, minimal, low, medium, high, xhigh",
            "console-settings-reasoning-summary": (
                "auto, concise, detailed, none" + PROVIDER_CHOICE_NO_EFFECT_SUFFIX
            ),
            "console-settings-verbosity": (
                "low, medium, high" + PROVIDER_CHOICE_NO_EFFECT_SUFFIX
            ),
            "console-settings-thinking-effort": (
                "off, low, medium, high, xhigh, max" + PROVIDER_CHOICE_NO_EFFECT_SUFFIX
            ),
        }
        for input_id, expected in placeholders.items():
            assert app.screen.query_one(f"#{input_id}", Input).placeholder == expected


@pytest.mark.asyncio
async def test_console_settings_modal_scope_line_names_session_and_default_scopes() -> (
    None
):
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        CONSOLE_SETTINGS_SCOPE_COPY,
    )

    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        scope = app.screen.query_one("#console-settings-scope", Static)
        assert str(scope.renderable) == CONSOLE_SETTINGS_SCOPE_COPY
        assert "conversation" in CONSOLE_SETTINGS_SCOPE_COPY.lower()
        assert "model defaults" in CONSOLE_SETTINGS_SCOPE_COPY.lower()
        assert (
            str(app.screen.query_one("#console-settings-save-default", Button).label)
            == "Save model defaults"
        )
        response_control = app.screen.query_one("#console-settings-max-tokens", Input)
        response_label = response_control.parent.query_one(
            ".console-settings-modal-label",
            Static,
        )
        assert str(response_label.renderable) == "Response max tokens"


@pytest.mark.asyncio
async def test_console_settings_modal_save_as_default_writes_through_config(
    monkeypatch,
) -> None:
    from tldw_chatbook.Widgets.Console import console_settings_modal as modal_module

    captured: list[dict] = []

    def fake_save(sections):
        captured.append(sections)
        return True

    monkeypatch.setattr(modal_module, "save_settings_to_cli_config", fake_save)
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
        temperature=0.6,
        streaming=False,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        await pilot.click("#console-settings-save-default")

    assert app.saved_settings is not None
    assert app.saved_settings.model == "model-a"
    assert len(captured) == 1
    sections = captured[0]
    provider_section = sections["api_settings.llama_cpp"]
    assert provider_section["model"] == "model-a"
    # llama_cpp already persists its endpoint under api_url in ModalHarness config.
    assert provider_section["api_url"] == "http://127.0.0.1:9099"
    # TASK-342: sampling values land in the Console-saved-defaults section the
    # boot builder ranks above chat_defaults; writing them into api_settings
    # was inert (chat_defaults deliberately outranks it, f14d22dc3).
    assert "temperature" not in provider_section
    saved_section = sections["console.provider_defaults.llama_cpp"]
    assert saved_section["temperature"] == 0.6
    # Streaming persists on the canonical chat_defaults key (bridged legacy key),
    # and the provider itself becomes the default (PR #606 review finding:
    # chat_defaults.provider is the ONLY source of the default provider).
    # The model is written here as well as into api_settings: chat_defaults.model
    # is what `resolve_effective_provider_model` feeds to the session builder as
    # an explicit override, so omitting it left a stale model winning in every
    # new session (roleplay UAT: character "Chat now" silently reverted to the
    # model onboarding had auto-picked).
    assert sections["chat_defaults"] == {
        "streaming": False,
        "provider": "llama_cpp",
        "model": "model-a",
    }
    # Never persist None-valued optionals.
    assert "min_p" not in saved_section
    assert "seed" not in saved_section


@pytest.mark.asyncio
async def test_console_settings_legacy_alias_is_passive_until_canonical_default_save(
    monkeypatch,
) -> None:
    from tldw_chatbook.Widgets.Console import console_settings_modal as modal_module

    captured: list[dict[str, dict[str, object]]] = []
    canonical_calls = []
    real_canonical_mutation = modal_module.build_canonical_chat_defaults_mutation

    def fake_save(sections: dict[str, dict[str, object]]) -> bool:
        captured.append(sections)
        return True

    def recording_canonical_mutation(effective):
        canonical_calls.append(effective)
        return real_canonical_mutation(effective)

    monkeypatch.setattr(modal_module, "save_settings_to_cli_config", fake_save)
    monkeypatch.setattr(
        modal_module,
        "build_canonical_chat_defaults_mutation",
        recording_canonical_mutation,
    )
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="OpenAI-Compatible",
        model="pocket-tts",
        streaming=False,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(
                settings,
                app,
                providers_models={"OpenAI-Compatible": ["pocket-tts"]},
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        assert captured == []
        assert canonical_calls == []
        await pilot.click("#console-settings-save-default")

    assert len(captured) == 1
    assert len(canonical_calls) == 1
    assert canonical_calls[0].provider == "openai"
    assert canonical_calls[0].model == "pocket-tts"
    assert captured[0]["chat_defaults"] == {
        "streaming": False,
        "provider": "openai",
        "model": "pocket-tts",
    }
    assert captured[0]["api_settings.openai"]["model"] == "pocket-tts"


@pytest.mark.asyncio
async def test_console_settings_modal_save_as_default_failure_keeps_modal_open(
    monkeypatch,
) -> None:
    from tldw_chatbook.Widgets.Console import console_settings_modal as modal_module
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        CONSOLE_SETTINGS_SAVE_DEFAULT_FAILED_COPY,
    )

    monkeypatch.setattr(
        modal_module, "save_settings_to_cli_config", lambda sections: False
    )
    app = ModalHarness()
    app.saved_settings = ConsoleSessionSettings(provider="openai", model="sentinel")
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        await pilot.click("#console-settings-save-default")
        await pilot.pause()
        error = app.screen.query_one("#console-settings-error", Static)
        assert str(error.renderable) == CONSOLE_SETTINGS_SAVE_DEFAULT_FAILED_COPY
        # Modal stays open (dismiss would pop it and fire the callback).
        assert isinstance(app.screen, ConsoleSettingsModal)
        await pilot.click("#console-settings-cancel")

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_body_scroll_container_is_not_focusable() -> None:
    """The focused scroll body painted stray focus-outline fragments ("|")
    through the section margins with the real app CSS; keeping it out of the
    focus chain removes the artifact and lands first focus on a real control."""
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            _basic_modal(settings, app), callback=app.capture_saved_settings
        )
        await pilot.pause()
        body = app.screen.query_one("#console-settings-body", ScrollableContainer)
        assert body.can_focus is False
        assert app.focused is not body


# --- task-177 live regression: REAL journey (boot -> Settings save -> Console) ---


def _build_live_config_test_app():
    """Real TldwCli booted against the REAL (test-sandboxed) config file.

    Unlike ``_build_test_app`` this does NOT stub ``load_settings`` /
    ``get_cli_setting``: ``app.app_config`` is the genuine template config from
    the sandbox ``TLDW_CONFIG_PATH``, so the disk-loaded snapshot path (and the
    stale-snapshot bug it guards against) is exercised end to end.
    """
    import tempfile
    from contextlib import ExitStack
    from unittest.mock import MagicMock, patch

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState

    user_data_dir = Path(
        tempfile.mkdtemp(prefix="tldw-chatbook-live-config-test-")
    ).resolve(strict=True)

    def fake_runtime_policy(app):
        context = SimpleNamespace(
            state=RuntimeSourceState(active_source="local", server_configured=True),
            persist=lambda: None,
        )
        app.runtime_policy = context
        app._publish_runtime_policy_projection(context.state)
        return context

    with ExitStack() as stack:
        stack.enter_context(
            patch("tldw_chatbook.app.get_chachanotes_db_lazy", return_value=None)
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.ServerNotesWorkspaceService.from_config",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.ServerCharacterPersonaService.from_config",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.PersonaActorPackCoordinator.recover",
                return_value=SimpleNamespace(blocked_intent_ids=()),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_notes_service",
                lambda self, _user: setattr(self, "notes_service", None),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_prompts_service",
                lambda self: setattr(self, "prompts_service_initialized", False),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_providers_models",
                lambda self: setattr(self, "providers_models", {}),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_media_db",
                lambda self: (
                    setattr(self, "media_db", None),
                    setattr(self, "_media_types_for_ui", ["All Media"]),
                ),
            )
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.load_runtime_policy_for_app",
                side_effect=fake_runtime_policy,
            )
        )
        for db_path_getter in (
            "get_notifications_db_path",
            "get_research_db_path",
            "get_writing_db_path",
        ):
            stack.enter_context(
                patch(f"tldw_chatbook.app.{db_path_getter}", return_value=":memory:")
            )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.get_subscriptions_db_path",
                return_value=user_data_dir / "subscriptions.sqlite",
            )
        )
        stack.enter_context(
            patch("tldw_chatbook.app.get_user_data_dir", return_value=user_data_dir)
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.get_workspaces_db_path",
                return_value=user_data_dir / "workspaces.sqlite",
            )
        )
        return TldwCli()


async def _wait_for_screen(app, pilot, screen_type_name: str, *, attempts: int = 250):
    for _ in range(attempts):
        if type(app.screen).__name__ == screen_type_name:
            return app.screen
        await pilot.pause(0.02)
    raise AssertionError(
        f"Never reached {screen_type_name}; current screen: {type(app.screen).__name__}"
    )


@pytest.mark.asyncio
async def test_real_journey_settings_save_unblocks_console_without_restart(
    monkeypatch,
) -> None:
    """Live-UAT regression: boot -> blocked Console -> Settings save -> Console.

    Mirrors the exact live failure: the Settings adapter saves
    chat_defaults.provider/model + the llama.cpp endpoint (config caches reload),
    the user clicks the Console nav tab (fresh ChatScreen composes, prior screen
    state restores), and the setup card must NOT still be blocking.
    """
    from tldw_chatbook import config as config_module
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
    from tldw_chatbook.Widgets.Console import ConsoleSetupModal

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    # Prime the sandbox template config and keep the boot fast/deterministic.
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    assert config_module.save_setting_to_cli_config("splash_screen", "enabled", False)
    assert config_module.save_setting_to_cli_config(
        "first_run", "setup_completed", True
    )
    config_module.load_settings(force_reload=True)

    app = _build_live_config_test_app()
    # Sanity: the boot snapshot must look disk-loaded (markers present) so the
    # fresh-config branch is the one under test.
    assert ChatScreen._console_config_snapshot_is_disk_loaded(app.app_config)

    async with app.run_test(size=(180, 50)) as pilot:
        # 1) First-run landing: Console blocked on the template OpenAI default.
        app.post_message(NavigateToScreen("chat"))
        console = await _wait_for_screen(app, pilot, "ChatScreen")
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        assert console._build_console_setup_card_state().mode == "card"

        # 2) Leave Console (screen state, including session settings, is saved).
        app.post_message(NavigateToScreen("home"))
        await _wait_for_screen(app, pilot, "HomeScreen")

        # 3) The real Settings save path (same three values as the live run).
        adapter = SettingsConfigAdapter()
        assert adapter.save_values(
            "chat_defaults",
            {"provider": "llama_cpp", "model": "Qwen3-Coder-Test.gguf"},
        )
        assert adapter.save_values(
            "api_settings.llama_cpp",
            {"api_url": "http://127.0.0.1:9099"},
        )

        # 4) Back to Console: a fresh ChatScreen composes and restores state.
        app.post_message(NavigateToScreen("chat"))
        console = await _wait_for_screen(app, pilot, "ChatScreen")
        await _wait_for_selector(console, pilot, "#console-setup-modal")

        card_state = console._build_console_setup_card_state()
        assert card_state.mode != "card", (
            "Setup card still blocking after a provider save; "
            f"steps={[(step.state, step.label) for step in card_state.steps]}"
        )
        settings, readiness = console._active_console_settings_readiness()
        assert settings.provider == "llama_cpp"
        assert readiness.native_send_supported is True

        # The blocking modal must clear once guidance syncs.
        for _ in range(100):
            modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
            if not modal.is_blocking:
                break
            await pilot.pause(0.02)
        assert not console.query_one(
            "#console-setup-modal", ConsoleSetupModal
        ).is_blocking


def test_console_stale_default_refresh_respects_user_marked_settings() -> None:
    """Blocked derived defaults refresh; explicit user selections never do."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "local-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"},
        "openai": {"api_key": ""},
    }
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.ensure_session()

    user_choice = ConsoleSessionSettings(
        provider="openai", model="gpt-4o", source="user"
    )
    store.replace_session_settings(session.id, user_choice)
    assert console._session._ensure_active_console_session_settings() == user_choice

    # A user-work marker is intentionally durable, so exercise the untouched
    # stale-default case in a separate canonical session.
    stale_derived = ConsoleSessionSettings(provider="openai", model="gpt-4o")
    session = store.create_session(
        settings=stale_derived,
        canonical_settings_baseline=stale_derived,
    )
    refreshed = console._session._ensure_active_console_session_settings()
    assert refreshed.provider == "llama_cpp"
    assert refreshed.source == "derived"


def test_console_stale_default_refresh_respects_applied_system_prompt() -> None:
    """A stale-default refresh must not overwrite an applied `/system` prompt.

    The user-work provenance introduced on ``dev`` treats `/system` as an
    explicit session choice. Automatic refresh therefore leaves the whole
    settings snapshot intact, including its provider and prompt.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "local-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"},
        "openai": {"api_key": ""},
    }
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.ensure_session()

    # Blocked derived defaults (openai, no key) -- as if snapshotted on a
    # fresh, never-configured session -- with a `/system` prompt applied
    # before any message was sent.
    stale_derived = ConsoleSessionSettings(provider="openai", model="gpt-4o")
    store.replace_session_settings(
        session.id,
        stale_derived,
        mark_user_work=False,
        canonical_settings_baseline=stale_derived,
    )
    store.set_session_system_prompt(session.id, "Be concise.")

    refreshed = console._session._ensure_active_console_session_settings()

    assert refreshed.provider == "openai"
    assert refreshed.system_prompt == "Be concise."
    # The store itself must carry the preserved prompt forward too, not just
    # the returned snapshot.
    assert store.session_settings(session.id).system_prompt == "Be concise."


# --- task-188/191: provider display names + Discover models -----------------


def _select_labels(select: Select) -> set[str]:
    options = getattr(select, "options", None)
    if options is None:
        options = getattr(select, "_options", [])
    labels: set[str] = set()
    for option in options:
        prompt = getattr(option, "prompt", None)
        if prompt is None and isinstance(option, tuple) and option:
            prompt = option[0]
        if prompt is not None:
            labels.add(str(getattr(prompt, "plain", prompt)))
    return labels


@pytest.mark.asyncio
async def test_console_settings_modal_provider_labels_use_catalog_display_names() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        labels = _select_labels(provider_select)
        values = _select_values(provider_select)

    # Labels render shared-catalog display names; values stay raw config keys.
    assert "llama.cpp" in labels
    assert "OpenAI" in labels
    assert "Ollama" in labels
    assert "llama_cpp" not in labels
    assert {"llama_cpp", "openai", "ollama"}.issubset(values)


class _RecordingProber:
    def __init__(self, result: LocalModelProbeResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, base_url: str, provider_key: str) -> LocalModelProbeResult:
        self.calls.append((base_url, provider_key))
        return self.result


async def _wait_for_discover_status(app, pilot, fragment: str) -> Static:
    status = app.screen.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static)
    for _ in range(60):
        if fragment in str(status.renderable):
            return status
        await pilot.pause(0.05)
    raise AssertionError(
        f"discover status never showed {fragment!r}; last: {str(status.renderable)!r}"
    )


@pytest.mark.asyncio
async def test_console_settings_modal_discover_models_success_swaps_input_for_select() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model=None)
    prober = _RecordingProber(
        LocalModelProbeResult(
            ok=True,
            base_url="http://127.0.0.1:9099",
            model_ids=("srv-a", "srv-b"),
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                model_prober=prober,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).press()
        await _wait_for_discover_status(
            app, pilot, "Found 2 models at http://127.0.0.1:9099."
        )

        assert prober.calls == [("http://127.0.0.1:9099", "llama_cpp")]
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.display is True
        assert model_select.disabled is False
        assert _select_values(model_select) == {"srv-a", "srv-b"}
        assert model_select.value == "srv-a"
        # Free-text fallback stays available after discovery.
        model_custom = app.screen.query_one("#console-settings-model-custom", Button)
        assert model_custom.display is True
        assert model_custom.disabled is False

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.model == "srv-a"


@pytest.mark.asyncio
async def test_console_settings_modal_discover_models_failure_shows_inline_copy() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model=None)
    prober = _RecordingProber(
        LocalModelProbeResult(
            ok=False,
            base_url="http://127.0.0.1:9099",
            detail="No models endpoint at http://127.0.0.1:9099.",
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                model_prober=prober,
            )
        )
        await pilot.pause()

        discover = app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        discover.press()
        await _wait_for_discover_status(
            app, pilot, "No models endpoint at http://127.0.0.1:9099."
        )

        # Honest inline line, button usable again, manual entry still works.
        assert discover.disabled is False
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_input.display is True
        assert model_input.disabled is False


@pytest.mark.asyncio
async def test_console_settings_modal_discover_button_only_for_url_based_providers() -> (
    None
):
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"], "llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        discover = app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        assert discover.display is False
        assert discover.disabled is True

        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()
        assert discover.display is True
        assert discover.disabled is False


@pytest.mark.asyncio
async def test_console_settings_modal_discover_rejects_invalid_endpoint_url() -> None:
    """PR #608 review: user-entered endpoint must pass shared URL validation
    before any network probe; the prober must never be called."""
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        MODEL_DISCOVER_INVALID_URL_COPY,
    )

    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model=None,
        base_url="http://[not-a-valid-url",
    )
    prober = _RecordingProber(
        LocalModelProbeResult(ok=True, base_url="", model_ids=("x",))
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                model_prober=prober,
            )
        )
        await pilot.pause()

        discover = app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        discover.press()
        await _wait_for_discover_status(app, pilot, MODEL_DISCOVER_INVALID_URL_COPY)

    assert prober.calls == []


# --- Roleplay UAT regression: Save as default must not leave a stale model ---
# Live repro (origin/dev @ f384a2807): onboarding auto-selected a wrong model and
# wrote it to [chat_defaults].model. Correcting the model in Console Settings and
# pressing "Save as default" wrote the new model ONLY to
# [api_settings.<provider>].model, leaving [chat_defaults].model stale. Because
# `resolve_effective_provider_model` reads chat_defaults.model and passes it as an
# explicit override into `build_default_console_session_settings` (where it
# outranks api_settings), every NEW session -- new tab, character "Chat now",
# app relaunch -- silently reverted to the old model.


def test_save_as_default_persists_model_to_chat_defaults() -> None:
    """The chosen model must land in chat_defaults, the section Console reads."""
    modal = ConsoleSettingsModal(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="good-model"),
        app_config={},
        providers_models={"llama_cpp": ["good-model"]},
        context_estimate=ConsoleSettingsContextEstimate(
            used_tokens=10, token_limit=16384, label="10 / 16k"
        ),
        can_save=True,
    )
    sections = modal._default_persist_sections(
        ConsoleSessionSettings(provider="llama_cpp", model="good-model")
    )

    assert sections["chat_defaults"]["model"] == "good-model"


def test_save_as_default_model_agrees_across_config_sections() -> None:
    """chat_defaults and api_settings must not disagree about the active model."""
    modal = ConsoleSettingsModal(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="good-model"),
        app_config={},
        providers_models={"llama_cpp": ["good-model"]},
        context_estimate=ConsoleSettingsContextEstimate(
            used_tokens=10, token_limit=16384, label="10 / 16k"
        ),
        can_save=True,
    )
    sections = modal._default_persist_sections(
        ConsoleSessionSettings(provider="llama_cpp", model="good-model")
    )

    assert (
        sections["chat_defaults"]["model"]
        == sections["api_settings.llama_cpp"]["model"]
    )


# --- Roleplay UAT: model discovery looked like it did nothing ---
# Live repro (origin/dev @ f384a2807): pressing "Discover models" produced no
# visible change. The status line ("Found 1 model at http://127.0.0.1:9099.")
# was composed BELOW the unrelated Base URL field, four rows from the button
# that produced it, and the discovered model was not selected -- the
# known-broken model stayed in the box. It read as a dead button.


def test_discovery_status_renders_next_to_the_discover_button() -> None:
    """Feedback must sit with the control that produced it, not below another field."""
    source = (
        Path(chat_screen_module.__file__).resolve().parents[2]
        / "Widgets"
        / "Console"
        / "console_settings_modal.py"
    )
    text = source.read_text()

    status_pos = text.index("id=MODEL_DISCOVER_STATUS_ID,")
    base_url_pos = text.index('id="console-settings-base-url"')

    assert status_pos < base_url_pos, (
        "discovery status is composed after the Base URL row, so it renders "
        "detached from the button that produced it"
    )


@pytest.mark.asyncio
async def test_discovery_selects_the_model_when_exactly_one_is_found() -> None:
    """One discovered model must be selected, not left for the user to notice.

    Leaving the previous (often wrong) model selected after a successful
    discovery is what let a TTS model stay active on a chat endpoint.
    """
    app = ModalHarness()
    modal = ConsoleSettingsModal(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", model="stale-model", base_url="http://127.0.0.1:9099"
        ),
        app_config=app.app_config,
        providers_models={"llama_cpp": ["stale-model"]},
        context_estimate=ConsoleSettingsContextEstimate(
            used_tokens=10, token_limit=16384, label="10 / 16k"
        ),
        can_save=True,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._apply_model_discovery_result(
            "llama_cpp",
            LocalModelProbeResult(
                ok=True,
                base_url="http://127.0.0.1:9099",
                model_ids=("only-real-model",),
            ),
        )
        await pilot.pause()

        assert modal._current_model_value() == "only-real-model"
        picker = modal.query_one(ModelSearchPicker)
        picker.focus_input()
        await pilot.pause()
        await pilot.press("o", "n", "l", "y")
        await pilot.pause()
        results = modal.query_one("#model-search-picker-results", OptionList)
        assert [str(option.prompt) for option in results.options] == ["only-real-model"]
