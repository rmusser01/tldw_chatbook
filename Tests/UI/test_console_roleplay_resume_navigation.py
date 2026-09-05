"""Focused contracts for Roleplay's ID-only Console resume navigation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from textual.worker import Worker, WorkerState

import tldw_chatbook.UI.Console_Modules.session as session_module
import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_console_workspace_controller import _conversation_tree_payload
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID
from tldw_chatbook.config import RuntimeConfigSnapshot
from tldw_chatbook.Constants import (
    CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleFirstChatIntent,
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID


def test_resume_navigation_context_captures_only_normalized_local_id() -> None:
    screen = ChatScreen.__new__(ChatScreen)

    screen.apply_navigation_context(
        {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "  abc  "}
    )

    assert screen._pending_resume_local_conversation_id == "abc"


@pytest.mark.parametrize("value", [None, 7, "", "   ", "x" * 257])
def test_resume_navigation_context_ignores_invalid_ids(value: object) -> None:
    screen = ChatScreen.__new__(ChatScreen)

    screen.apply_navigation_context(
        {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: value}
    )

    assert screen._pending_resume_local_conversation_id is None


def test_resume_navigation_context_is_capture_only_before_mount() -> None:
    screen = ChatScreen.__new__(ChatScreen)

    def unexpected_side_effect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("navigation capture touched Console runtime state")

    screen.query_one = unexpected_side_effect
    screen._ensure_console_chat_store = unexpected_side_effect
    screen._sync_native_console_chat_ui = unexpected_side_effect
    screen._restore_console_workbench_focus = unexpected_side_effect
    screen.run_worker = unexpected_side_effect
    screen.set_timer = unexpected_side_effect

    screen.apply_navigation_context(
        {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "conversation-7"}
    )

    assert screen._pending_resume_local_conversation_id == "conversation-7"


def test_resume_navigation_context_has_class_level_unset_default() -> None:
    screen = ChatScreen.__new__(ChatScreen)

    assert screen._pending_resume_local_conversation_id is None
    assert screen._resume_navigation_startup_in_progress is False


def _async_spy(events: list[str], label: str):
    async def callback(*_args: object, **_kwargs: object) -> None:
        events.append(label)

    callback.__name__ = label
    return callback


class _MountedNavigationConsoleHarness(ConsolidatedCSSApp):
    """Apply navigation context before mounting a real Console screen."""

    def __init__(
        self,
        app_instance: object,
        *,
        conversation_id: str | None,
        configure: Callable[[ChatScreen], None],
    ) -> None:
        super().__init__()
        self.app_instance = app_instance
        self.conversation_id = conversation_id
        self.configure = configure
        self.chat_screen: ChatScreen | None = None

    async def on_mount(self) -> None:
        screen = ChatScreen(self.app_instance)
        self.chat_screen = screen
        if self.conversation_id is not None:
            screen.apply_navigation_context(
                {
                    CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: (
                        self.conversation_id
                    )
                }
            )
        self.configure(screen)
        await self.push_screen(screen)


async def _wait_until(
    pilot,
    predicate: Callable[[], bool],
    *,
    timeout: float = 3,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for mounted Console lifecycle state")


def _configure_ready_console(app: object) -> None:
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "local-model",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "local-model",
        }
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"


def _install_first_chat_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    *,
    acknowledge: bool,
) -> RuntimeConfigSnapshot:
    snapshot = RuntimeConfigSnapshot(
        91,
        {
            "chat_defaults": {
                "provider": "llama_cpp",
                "model": "local-model",
            },
            "api_settings": {
                "llama_cpp": {
                    "api_url": "http://127.0.0.1:9099",
                    "model": "local-model",
                }
            },
        },
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
    )

    def guarded_acknowledgement(generation: int, action: Callable[[], bool]) -> bool:
        if not acknowledge or generation != snapshot.generation:
            return False
        return action() is True

    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        guarded_acknowledgement,
    )
    return snapshot


def _instrument_first_chat_presentation(
    screen: ChatScreen,
    *,
    observations: list[tuple[bool, bool]],
    presentation_events: list[str],
    lifecycle_events: list[str] | None = None,
) -> None:
    owner = screen._session
    original_consume = owner.consume_pending_console_first_chat_intent

    def consume(*, defer_presentation: bool = False) -> bool:
        if lifecycle_events is not None:
            lifecycle_events.append("first-chat")
        observations.append(
            (
                screen._resume_navigation_startup_in_progress,
                defer_presentation,
            )
        )
        if defer_presentation:
            return original_consume(defer_presentation=True)
        return original_consume()

    def presentation_snapshot() -> tuple[None, None, object]:
        presentation_events.append("presentation-snapshot")
        return None, None, object()

    async def native_sync() -> None:
        presentation_events.append("rollback-native-sync")

    owner.consume_pending_console_first_chat_intent = consume
    owner._first_chat_presentation_snapshot_fn = presentation_snapshot
    owner._apply_first_chat_control_selection_fn = (
        lambda _provider, _model: presentation_events.append("control-selection")
    )
    owner._sync_chat_core_state_fn = lambda: presentation_events.append("core-sync")
    owner._sync_settings_summary_fn = lambda: presentation_events.append(
        "settings-sync"
    )
    owner._sync_control_bar_fn = lambda: presentation_events.append("control-sync")
    owner._sync_native_console_chat_ui_fn = native_sync
    owner._restore_first_chat_focus_fn = lambda _token: presentation_events.append(
        "rollback-focus"
    )


@pytest.mark.asyncio
async def test_mounted_resume_orders_consumers_once_and_suppresses_competitors() -> None:
    app = _build_test_app()
    _configure_ready_console(app)
    events: list[str] = []
    calls: dict[str, int] = {}
    timers: list[tuple[float, str]] = []
    workers: list[Worker[object]] = []
    opener_ids: list[str] = []
    first_chat_observations: list[tuple[bool, bool]] = []

    def record(label: str) -> None:
        calls[label] = calls.get(label, 0) + 1
        events.append(label)

    def configure(screen: ChatScreen) -> None:
        def first_chat(*, defer_presentation: bool = False) -> None:
            first_chat_observations.append(
                (
                    screen._resume_navigation_startup_in_progress,
                    defer_presentation,
                )
            )
            record("first-chat")

        async def chat_handoff(**_kwargs: object) -> None:
            record("chat-handoff")

        def roleplay_repair() -> bool:
            record("roleplay-repair")
            return True

        async def prompt_insert() -> None:
            record("prompt-insert")

        def provider_intent() -> bool:
            record("provider-intent")
            return True

        async def fleet_completion() -> None:
            record("fleet-completion")

        async def opener(conversation_id: str) -> bool:
            opener_ids.append(conversation_id)
            record("resume-selected-conversation")
            record("final-presentation-focus")
            return True

        async def native_sync() -> None:
            record("intermediate-native-sync")

        def restore_focus() -> None:
            record("intermediate-focus")

        def reconcile() -> None:
            record("registry-reconcile")

        def identity_refresh() -> bool:
            record("intermediate-identity-refresh")
            return False

        def active_roleplay_refresh(**_kwargs: object) -> bool:
            record("intermediate-roleplay-refresh")
            return False

        original_set_timer = screen.set_timer

        def recording_set_timer(delay, callback, **kwargs):
            timers.append((delay, getattr(callback, "__name__", "")))
            return original_set_timer(delay, callback, **kwargs)

        original_run_worker = screen.run_worker

        def recording_run_worker(work, **kwargs):
            worker = original_run_worker(work, **kwargs)
            workers.append(worker)
            return worker

        screen._session.consume_pending_console_first_chat_intent = first_chat
        screen._consume_pending_chat_handoff = chat_handoff
        screen._consume_pending_console_roleplay_repair = roleplay_repair
        screen._prompts._consume_pending_console_prompt_insert = prompt_insert
        screen.consume_pending_console_provider_intent = provider_intent
        screen._fleet.consume_pending_console_fleet_completion = fleet_completion
        screen._workspace.open_console_workspace_conversation = opener
        screen._sync_native_console_chat_ui = native_sync
        screen._restore_console_workbench_focus = restore_focus
        screen._workspace._reconcile_console_session_with_registry = reconcile
        screen._consume_pending_console_identity_refresh = identity_refresh
        screen._dispatch_active_console_roleplay_refresh = active_roleplay_refresh
        screen.set_timer = recording_set_timer
        screen.run_worker = recording_run_worker

    host = _MountedNavigationConsoleHarness(
        app,
        conversation_id="  resume-target  ",
        configure=configure,
    )

    async with host.run_test(size=(160, 48)) as pilot:
        await _wait_until(pilot, lambda: "final-presentation-focus" in events)
        screen = host.chat_screen
        assert screen is not None

    competing = {
        "chat-handoff",
        "roleplay-repair",
        "prompt-insert",
        "provider-intent",
        "fleet-completion",
    }
    assert not competing.intersection(name for delay, name in timers if delay == 0.15)
    assert set(timers) >= {
        (0.1, "_restore_collapsible_states"),
        (0.05, "sync_task_resume_state"),
        (0.15, "_sync_console_dictation_availability"),
        (0.3, "_maybe_start_console_fleet_survivor_tick"),
        (0.15, "_start_resume_navigation_startup"),
    }
    assert events == [
        "first-chat",
        "chat-handoff",
        "roleplay-repair",
        "prompt-insert",
        "provider-intent",
        "fleet-completion",
        "resume-selected-conversation",
        "final-presentation-focus",
    ]
    assert all(calls[label] == 1 for label in competing)
    assert calls["first-chat"] == 1
    assert first_chat_observations == [(True, True)]
    assert calls["resume-selected-conversation"] == 1
    assert "intermediate-native-sync" not in events
    assert "intermediate-focus" not in events
    assert "registry-reconcile" not in events
    assert opener_ids == ["resume-target"]
    assert screen._pending_resume_local_conversation_id is None
    assert screen._resume_navigation_startup_in_progress is False
    ordered_workers = [
        worker
        for worker in workers
        if worker.group == "console-resume-navigation-startup"
    ]
    assert len(ordered_workers) == 1
    assert ordered_workers[0].state is WorkerState.SUCCESS


@pytest.mark.asyncio
async def test_mounted_global_resume_outranks_registry_active_workspace() -> None:
    """An explicit global resume creates no named-workspace bootstrap tab."""
    app = _build_test_app()
    _configure_ready_console(app)
    store = app.console_runtime.ensure_chat_store()
    workspace = app.workspace_registry_service.create_workspace(
        workspace_id="workspace-startup",
        name="Startup Workspace",
    )
    app.workspace_registry_service.set_active_workspace(workspace.workspace_id)
    app.chat_conversation_scope_service = SimpleNamespace(
        get_conversation_tree=lambda conversation_id, **_kwargs: (
            _conversation_tree_payload(conversation_id)
        )
    )
    host = _MountedNavigationConsoleHarness(
        app,
        conversation_id="resume-global",
        configure=lambda _screen: None,
    )

    async with host.run_test(size=(160, 48)) as pilot:
        await _wait_until(
            pilot,
            lambda: any(
                session.persisted_conversation_id == "resume-global"
                and session.id == store.active_session_id
                for session in store.sessions()
            ),
        )

        active_session = store.ensure_session()
        assert active_session.persisted_conversation_id == "resume-global"
        assert active_session.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID
        assert len(store.sessions()) == 1
        assert (
            app.workspace_registry_service.get_active_workspace().workspace_id
            == DEFAULT_WORKSPACE_ID
        )


@pytest.mark.asyncio
async def test_mounted_missing_resume_falls_back_to_registry_active_workspace() -> None:
    """A failed ID-only resume still leaves a usable ordinary Console tab."""
    app = _build_test_app()
    _configure_ready_console(app)
    store = app.console_runtime.ensure_chat_store()
    workspace = app.workspace_registry_service.create_workspace(
        workspace_id="workspace-startup",
        name="Startup Workspace",
    )
    app.workspace_registry_service.set_active_workspace(workspace.workspace_id)
    load_attempts: list[str] = []

    def missing_tree(conversation_id: str, **_kwargs: object) -> dict[str, object]:
        load_attempts.append(conversation_id)
        return {}

    app.chat_conversation_scope_service = SimpleNamespace(
        get_conversation_tree=missing_tree,
    )
    host = _MountedNavigationConsoleHarness(
        app,
        conversation_id="missing-resume",
        configure=lambda _screen: None,
    )

    async with host.run_test(size=(160, 48)) as pilot:
        await _wait_until(
            pilot,
            lambda: bool(load_attempts)
            and host.chat_screen is not None
            and not host.chat_screen._resume_navigation_startup_in_progress,
        )

        assert load_attempts == ["missing-resume"]
        screen = host.chat_screen
        assert screen is not None
        active_session_id = store.active_session_id
        assert active_session_id is not None
        tab_selector = f"#console-session-tab-{active_session_id}"
        await _wait_for_selector(screen, pilot, tab_selector)
        active_session = store.ensure_session()
        assert active_session.workspace_id == workspace.workspace_id
        assert len(store.sessions()) == 1
        assert screen.query_one(tab_selector)


@pytest.mark.asyncio
async def test_mounted_resume_settles_first_chat_once_without_intermediate_presentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    _configure_ready_console(app)
    snapshot = _install_first_chat_snapshot(monkeypatch, acknowledge=True)
    intent = ConsoleFirstChatIntent(
        "ordered-first-chat",
        "llama_cpp",
        "local-model",
        snapshot.generation,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    observations: list[tuple[bool, bool]] = []
    presentation_events: list[str] = []
    lifecycle_events: list[str] = []

    def configure(screen: ChatScreen) -> None:
        _instrument_first_chat_presentation(
            screen,
            observations=observations,
            presentation_events=presentation_events,
            lifecycle_events=lifecycle_events,
        )

        async def chat_handoff(**_kwargs: object) -> None:
            lifecycle_events.append("chat-handoff")

        async def prompt_insert() -> None:
            lifecycle_events.append("prompt-insert")

        async def opener(_conversation_id: str) -> bool:
            lifecycle_events.append("resume-final-presentation")
            composer = screen.query_one("#console-native-composer")
            composer.can_focus = True
            composer.focus()
            lifecycle_events.append("resume-final-focus")
            return True

        async def intermediate_native_sync() -> None:
            lifecycle_events.append("intermediate-native-sync")

        screen._consume_pending_chat_handoff = chat_handoff
        screen._consume_pending_console_roleplay_repair = lambda: False
        screen._prompts._consume_pending_console_prompt_insert = prompt_insert
        screen.consume_pending_console_provider_intent = lambda: False
        screen._fleet.consume_pending_console_fleet_completion = lambda: False
        screen._workspace.open_console_workspace_conversation = opener
        screen._sync_native_console_chat_ui = intermediate_native_sync
        screen._restore_console_workbench_focus = lambda: lifecycle_events.append(
            "intermediate-focus"
        )
        screen._consume_pending_console_identity_refresh = (
            lambda: lifecycle_events.append("intermediate-identity-refresh") or False
        )
        screen._dispatch_active_console_roleplay_refresh = (
            lambda **_kwargs: lifecycle_events.append(
                "intermediate-roleplay-refresh"
            )
            or False
        )

    host = _MountedNavigationConsoleHarness(
        app,
        conversation_id="resume-target",
        configure=configure,
    )

    async with host.run_test(size=(160, 48)) as pilot:
        await _wait_until(pilot, lambda: "resume-final-focus" in lifecycle_events)
        screen = host.chat_screen
        assert screen is not None
        await pilot.pause()
        store = screen._ensure_console_chat_store()

        assert host.focused is screen.query_one("#console-native-composer")
        assert store.active_session_id == intent.session_id
        assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_FIRST_CHAT)

    assert observations == [(True, True)]
    assert presentation_events == []
    assert lifecycle_events == [
        "first-chat",
        "chat-handoff",
        "prompt-insert",
        "resume-final-presentation",
        "resume-final-focus",
    ]


@pytest.mark.asyncio
async def test_mounted_resume_releases_transient_first_chat_without_rollback_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    _configure_ready_console(app)
    snapshot = _install_first_chat_snapshot(monkeypatch, acknowledge=False)
    intent = ConsoleFirstChatIntent(
        "transient-first-chat",
        "llama_cpp",
        "local-model",
        snapshot.generation,
    )
    app.pending_handoffs.stage_reserved_console_first_chat(intent)
    observations: list[tuple[bool, bool]] = []
    presentation_events: list[str] = []
    lifecycle_events: list[str] = []
    workers: list[Worker[object]] = []

    def configure(screen: ChatScreen) -> None:
        _instrument_first_chat_presentation(
            screen,
            observations=observations,
            presentation_events=presentation_events,
            lifecycle_events=lifecycle_events,
        )
        original_run_worker = screen.run_worker

        def recording_run_worker(work, **kwargs):
            worker = original_run_worker(work, **kwargs)
            workers.append(worker)
            return worker

        async def opener(_conversation_id: str) -> bool:
            lifecycle_events.append("resume-final-presentation")
            composer = screen.query_one("#console-native-composer")
            composer.can_focus = True
            composer.focus()
            lifecycle_events.append("resume-final-focus")
            return True

        async def intermediate_native_sync() -> None:
            lifecycle_events.append("intermediate-native-sync")

        screen._consume_pending_chat_handoff = _async_spy(
            lifecycle_events,
            "chat-handoff",
        )
        screen._consume_pending_console_roleplay_repair = lambda: False
        screen._prompts._consume_pending_console_prompt_insert = _async_spy(
            lifecycle_events,
            "prompt-insert",
        )
        screen.consume_pending_console_provider_intent = lambda: False
        screen._fleet.consume_pending_console_fleet_completion = lambda: False
        screen._workspace.open_console_workspace_conversation = opener
        screen._sync_native_console_chat_ui = intermediate_native_sync
        screen._restore_console_workbench_focus = lambda: lifecycle_events.append(
            "intermediate-focus"
        )
        screen._consume_pending_console_identity_refresh = (
            lambda: lifecycle_events.append("intermediate-identity-refresh") or False
        )
        screen._dispatch_active_console_roleplay_refresh = (
            lambda **_kwargs: lifecycle_events.append(
                "intermediate-roleplay-refresh"
            )
            or False
        )
        screen.run_worker = recording_run_worker

    host = _MountedNavigationConsoleHarness(
        app,
        conversation_id="resume-target",
        configure=configure,
    )

    async with host.run_test(size=(160, 48)) as pilot:
        await _wait_until(pilot, lambda: "resume-final-focus" in lifecycle_events)
        screen = host.chat_screen
        assert screen is not None
        await pilot.pause(0.05)
        store = screen._ensure_console_chat_store()

        assert host.focused is screen.query_one("#console-native-composer")
        assert all(session.id != intent.session_id for session in store.sessions())
        assert app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_FIRST_CHAT)

    assert observations == [(True, True)]
    assert presentation_events == []
    assert not [
        worker for worker in workers if worker.group == "console-first-chat-rollback"
    ]
    assert lifecycle_events == [
        "first-chat",
        "chat-handoff",
        "prompt-insert",
        "resume-final-presentation",
        "resume-final-focus",
    ]


@pytest.mark.asyncio
async def test_mounted_resume_never_focuses_setup_modal_before_final_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    worker_started = asyncio.Event()
    release_worker = asyncio.Event()
    focus_events: list[str] = []
    screen_holder: list[ChatScreen] = []
    in_final_opener = False
    original_focus = ConsoleSetupModal.focus_primary_action

    def track_modal_focus(modal: ConsoleSetupModal) -> None:
        screen = screen_holder[0]
        if screen._resume_navigation_startup_in_progress:
            focus_events.append("final" if in_final_opener else "intermediate")
        original_focus(modal)

    monkeypatch.setattr(ConsoleSetupModal, "focus_primary_action", track_modal_focus)

    def configure(screen: ChatScreen) -> None:
        screen_holder.append(screen)

        async def chat_handoff(**_kwargs: object) -> None:
            worker_started.set()
            await release_worker.wait()

        async def opener(_conversation_id: str) -> bool:
            nonlocal in_final_opener
            in_final_opener = True
            try:
                modal = screen.query_one(
                    "#console-setup-modal",
                    ConsoleSetupModal,
                )
                modal.focus_primary_action()
            finally:
                in_final_opener = False
            return True

        screen._consume_pending_chat_handoff = chat_handoff
        screen._consume_pending_console_roleplay_repair = lambda: False
        screen._prompts._consume_pending_console_prompt_insert = _async_spy([], "prompt")
        screen.consume_pending_console_provider_intent = lambda: False
        screen._fleet.consume_pending_console_fleet_completion = lambda: False
        screen._workspace.open_console_workspace_conversation = opener

    host = _MountedNavigationConsoleHarness(
        app,
        conversation_id="resume-target",
        configure=configure,
    )

    async with host.run_test(size=(160, 48)) as pilot:
        screen = host.chat_screen
        assert screen is not None
        await _wait_for_selector(screen, pilot, "#console-setup-modal")
        await asyncio.wait_for(worker_started.wait(), timeout=2)
        await pilot.pause(0.05)
        composer = screen.query_one("#console-native-composer")
        composer.can_focus = True
        composer.focus()
        await pilot.pause()
        assert host.focused is composer

        screen._sync_console_transcript_guidance()
        await pilot.pause()
        assert composer.can_focus is False
        assert host.focused is not composer
        assert host.focused is None
        await pilot.press("x")
        assert composer.draft_text() == ""
        assert focus_events == []

        release_worker.set()
        await _wait_until(
            pilot,
            lambda: not screen._resume_navigation_startup_in_progress,
        )
        assert focus_events == ["final"]


@pytest.mark.asyncio
async def test_resume_navigation_continues_after_chat_handoff_release() -> None:
    screen = ChatScreen.__new__(ChatScreen)
    handoffs = PendingHandoffStore()
    handoffs.stage(
        HandoffChannel.CHAT,
        ChatHandoffPayload(
            source="Library",
            item_type="note",
            title="Pending note",
            body="body",
        ),
    )
    events: list[str] = []

    async def release_handoff(_payload: ChatHandoffPayload) -> bool:
        events.append("chat-handoff-released")
        raise RuntimeError("not ready")

    async def opener(conversation_id: str) -> bool:
        events.append(f"resume:{conversation_id}")
        return True

    screen.app_instance = SimpleNamespace(pending_handoffs=handoffs)
    screen._handoff_consumption_in_progress = False
    screen._session = SimpleNamespace(
        _start_character_console_session=release_handoff,
        consume_pending_console_first_chat_intent=lambda **_kwargs: False,
    )
    screen._stage_handoff_as_console_live_work = lambda _payload: None
    screen._consume_pending_console_roleplay_repair = lambda: False
    screen._prompts._consume_pending_console_prompt_insert = _async_spy(events, "prompt")
    screen.consume_pending_console_provider_intent = lambda: False
    screen._fleet = SimpleNamespace(
        consume_pending_console_fleet_completion=lambda: False,
    )
    screen._workspace = SimpleNamespace(
        open_console_workspace_conversation=opener,
    )
    screen._pending_resume_local_conversation_id = "resume-target"
    screen._resume_navigation_startup_in_progress = True

    await screen._consume_resume_navigation_startup()

    assert handoffs.has_pending(HandoffChannel.CHAT)
    assert events == ["chat-handoff-released", "prompt", "resume:resume-target"]


@pytest.mark.asyncio
async def test_resume_navigation_propagates_logged_chat_handoff_acquisition_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingHandoffStore:
        def claim(self, channel: HandoffChannel) -> None:
            assert channel is HandoffChannel.CHAT
            raise RuntimeError("private acquisition failure")

    warnings: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        chat_screen_module,
        "logger",
        SimpleNamespace(
            warning=lambda message, *args: warnings.append((message, args)),
        ),
    )
    opener_calls: list[str] = []
    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = SimpleNamespace(pending_handoffs=FailingHandoffStore())
    screen._handoff_consumption_in_progress = False
    screen._session = SimpleNamespace(
        consume_pending_console_first_chat_intent=lambda **_kwargs: False,
    )
    screen._consume_pending_console_roleplay_repair = lambda: False
    screen._prompts._consume_pending_console_prompt_insert = _async_spy([], "prompt")
    screen.consume_pending_console_provider_intent = lambda: False
    screen._fleet = SimpleNamespace(
        consume_pending_console_fleet_completion=lambda: False,
    )

    async def opener(conversation_id: str) -> bool:
        opener_calls.append(conversation_id)
        return True

    screen._workspace = SimpleNamespace(
        open_console_workspace_conversation=opener,
    )
    screen._pending_resume_local_conversation_id = "resume-target"
    screen._resume_navigation_startup_in_progress = True

    with pytest.raises(RuntimeError, match="private acquisition failure"):
        await screen._consume_resume_navigation_startup()

    assert opener_calls == []
    assert screen._pending_resume_local_conversation_id is None
    assert screen._resume_navigation_startup_in_progress is False
    assert warnings == [
        (
            "Chat handoff acquisition failed "
            "(channel={}, exception_category={})",
            ("chat", "RuntimeError"),
        )
    ]


@pytest.mark.asyncio
async def test_mounted_resume_worker_is_cancelled_and_timers_stop_on_unmount() -> None:
    app = _build_test_app()
    _configure_ready_console(app)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    events: list[str] = []
    timers: list[object] = []
    workers: list[Worker[object]] = []

    def configure(screen: ChatScreen) -> None:
        async def pending_handoff(**_kwargs: object) -> None:
            events.append("chat-handoff")
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("cancelled")
                cancelled.set()
                raise

        async def opener(_conversation_id: str) -> bool:
            events.append("resume")
            return True

        original_set_timer = screen.set_timer

        def recording_set_timer(delay, callback, **kwargs):
            timer = original_set_timer(delay, callback, **kwargs)
            timers.append(timer)
            return timer

        original_run_worker = screen.run_worker

        def recording_run_worker(work, **kwargs):
            worker = original_run_worker(work, **kwargs)
            workers.append(worker)
            return worker

        screen._consume_pending_chat_handoff = pending_handoff
        screen._consume_pending_console_roleplay_repair = lambda: False
        screen._prompts._consume_pending_console_prompt_insert = _async_spy(events, "prompt")
        screen.consume_pending_console_provider_intent = lambda: False
        screen._fleet.consume_pending_console_fleet_completion = lambda: False
        screen._workspace.open_console_workspace_conversation = opener
        screen.set_timer = recording_set_timer
        screen.run_worker = recording_run_worker

    host = _MountedNavigationConsoleHarness(
        app,
        conversation_id="resume-target",
        configure=configure,
    )

    async with host.run_test(size=(160, 48)) as pilot:
        await asyncio.wait_for(started.wait(), timeout=2)
        screen = host.chat_screen
        assert screen is not None
        resume_worker = next(
            worker
            for worker in workers
            if worker.group == "console-resume-navigation-startup"
        )

        await host.pop_screen()
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        await pilot.pause()

        assert resume_worker.is_cancelled
        assert resume_worker.state is WorkerState.CANCELLED
        assert not screen._timers
        assert all(timer._task is None for timer in timers)

    assert events == ["chat-handoff", "cancelled"]
    assert screen._pending_resume_local_conversation_id is None
    assert screen._resume_navigation_startup_in_progress is False


@pytest.mark.asyncio
async def test_mounted_no_resume_keeps_blocking_modal_focus_transfer() -> None:
    app = _build_test_app()
    host = _MountedNavigationConsoleHarness(
        app,
        conversation_id=None,
        configure=lambda _screen: None,
    )

    async with host.run_test(size=(160, 48)) as pilot:
        screen = host.chat_screen
        assert screen is not None
        await _wait_for_selector(screen, pilot, "#console-setup-modal")
        composer = screen.query_one("#console-native-composer")
        composer.can_focus = True
        composer.focus()
        await pilot.pause()
        assert host.focused is composer

        screen._sync_console_transcript_guidance()
        await pilot.pause()

        assert composer.can_focus is False
        assert host.focused is screen.query_one("#console-setup-modal-action")


@pytest.mark.asyncio
async def test_mounted_no_resume_keeps_ordinary_startup_sync_timers_and_focus() -> None:
    app = _build_test_app()
    _configure_ready_console(app)
    timers: list[tuple[float, str]] = []
    lifecycle_calls: list[str] = []
    first_chat_observations: list[tuple[bool, bool]] = []

    def configure(screen: ChatScreen) -> None:
        original_first_chat = (
            screen._session.consume_pending_console_first_chat_intent
        )

        def first_chat(*, defer_presentation: bool = False) -> bool:
            first_chat_observations.append(
                (
                    screen._resume_navigation_startup_in_progress,
                    defer_presentation,
                )
            )
            if defer_presentation:
                return original_first_chat(defer_presentation=True)
            return original_first_chat()

        original_set_timer = screen.set_timer

        def recording_set_timer(delay, callback, **kwargs):
            timers.append((delay, getattr(callback, "__name__", "")))
            return original_set_timer(delay, callback, **kwargs)

        original_native_sync = screen._sync_native_console_chat_ui

        async def native_sync() -> None:
            lifecycle_calls.append("native-sync")
            await original_native_sync()

        original_reconcile = screen._workspace._reconcile_console_session_with_registry

        def reconcile() -> None:
            lifecycle_calls.append("registry-reconcile")
            original_reconcile()

        original_focus = screen._restore_console_workbench_focus

        def restore_focus() -> None:
            lifecycle_calls.append("focus")
            original_focus()

        screen.set_timer = recording_set_timer
        screen._session.consume_pending_console_first_chat_intent = first_chat
        screen._sync_native_console_chat_ui = native_sync
        screen._workspace._reconcile_console_session_with_registry = reconcile
        screen._restore_console_workbench_focus = restore_focus

    host = _MountedNavigationConsoleHarness(
        app,
        conversation_id=None,
        configure=configure,
    )

    async with host.run_test(size=(160, 48)) as pilot:
        screen = host.chat_screen
        assert screen is not None
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        await _wait_until(
            pilot,
            lambda: {
                "native-sync",
                "registry-reconcile",
                "focus",
            }.issubset(lifecycle_calls),
        )
        await pilot.pause(0.25)

        assert host.focused is screen.query_one("#console-native-composer")
        assert screen._resume_navigation_startup_in_progress is False

    assert set(timers) >= {
        (0.15, "_consume_pending_chat_handoff"),
        (0.15, "_consume_pending_console_roleplay_repair"),
        (0.15, "_consume_pending_console_prompt_insert"),
        (0.15, "consume_pending_console_provider_intent"),
        (0.15, "consume_pending_console_fleet_completion"),
        (0.2, "restore_focus"),
    }
    assert lifecycle_calls.count("native-sync") >= 1
    assert lifecycle_calls.count("registry-reconcile") >= 1
    assert lifecycle_calls.count("focus") >= 1
    assert first_chat_observations == [(False, False), (False, False)]
