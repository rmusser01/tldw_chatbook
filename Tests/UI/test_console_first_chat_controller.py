"""Isolated first-chat handoff coverage for ``ConsoleSessionController``."""

from __future__ import annotations

import gc
import warnings
from functools import partial
from types import SimpleNamespace

import pytest

import tldw_chatbook.UI.Console_Modules.session as session_module
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_default_console_session_settings,
)
from tldw_chatbook.config import RuntimeConfigSnapshot
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleFirstChatIntent,
    HandoffChannel,
    PendingHandoffStore,
)


class _FakeScreen:
    """Framework-service surface used by Session without mounting Textual."""

    def __init__(self) -> None:
        self.is_mounted = False
        self.workers: list[tuple[object, dict[str, object]]] = []

    def run_worker(self, work, **kwargs):
        self.workers.append((work, kwargs))
        return object()


def _config(model: str = "model-a") -> dict[str, object]:
    return {
        "chat_defaults": {"provider": "openai", "model": model},
        "api_settings": {
            "openai": {
                "api_key": "test-only-key",
                "model": model,
            }
        },
    }


def _controller(
    *,
    screen: _FakeScreen | None = None,
    store: ConsoleChatStore | None = None,
    events: list[object] | None = None,
) -> ConsoleSessionController:
    """Build the real controller with narrow in-process dependencies."""

    screen = screen or _FakeScreen()
    events = events if events is not None else []
    app = SimpleNamespace(
        pending_handoffs=PendingHandoffStore(),
        notify=lambda message, **kwargs: events.append(("notify", message, kwargs)),
    )

    async def sync_native() -> None:
        events.append("native")

    return ConsoleSessionController(
        screen,
        app_instance=app,
        chat_store_accessor=lambda: store,
        current_chat_store_accessor=lambda: store,
        ensure_console_chat_controller=lambda: None,
        composer_accessor=lambda: None,
        effective_console_provider_model=lambda: (None, None),
        provider_readiness_app_config=dict,
        build_provider_selection=lambda _session_id: None,
        rag_source_types_accessor=lambda: (),
        rag_top_k_accessor=lambda: 0,
        sync_native_console_chat_ui=sync_native,
        sync_chat_core_state=lambda: events.append("core"),
        sync_temporary_chip=lambda: None,
        sync_settings_summary=lambda: events.append("summary"),
        sync_control_bar=lambda: events.append("bar"),
        sync_command_popup=lambda: None,
        note_follow_intent=lambda: None,
        focus_composer_if_needed=lambda **_kwargs: None,
        invalidate_persisted_rows_cache=lambda: None,
        mark_conversation_row_broken=lambda _conversation_id: None,
        refresh_effective_scope_and_sync=lambda _session: None,
        set_active_workspace_for_session=lambda _session_id: None,
        resume_workspace_conversation=lambda *_args, **_kwargs: None,
        workspace_initial_session_title=lambda _workspace_id: "Chat 1",
        merge_workspace_rows=lambda native_rows, _rows: native_rows,
        session_id_for_workspace_conversation=lambda _row_key: None,
        ensure_console_image_view=lambda: (None, None),
        visual_identity_db_accessor=lambda: None,
        reaction_preview_coordinator_accessor=lambda: None,
        refresh_character_avatar=lambda **_kwargs: None,
        read_control_values=lambda: ("prior-provider", "prior-model"),
        set_control_values=lambda provider, model: events.append(
            ("controls", provider, model)
        ),
        capture_focus_token=lambda: "opaque-focus-token",
        restore_focus_token=lambda token: events.append(("focus", token)),
    )


def test_first_chat_eligibility_does_not_create_a_missing_store() -> None:
    controller = _controller(store=None)

    assert controller.eligible_console_first_chat_session_id() is None


@pytest.mark.parametrize("state", ["untouched", "edited", "workspace"])
def test_first_chat_eligibility_respects_pristine_global_ownership(state: str) -> None:
    store = ConsoleChatStore()
    defaults = ConsoleSessionSettings(
        provider="openai",
        model="model-a",
        source="derived",
    )
    session = store.create_session(
        workspace_id="workspace-a" if state == "workspace" else "global",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    if state == "edited":
        store.set_session_draft(session.id, "user work")
    controller = _controller(store=store)

    assert controller.eligible_console_first_chat_session_id() == (
        session.id if state == "untouched" else None
    )


def test_first_chat_replacement_during_creation_rolls_back_exact_old_claim(
    monkeypatch,
) -> None:
    store = ConsoleChatStore()
    prior_settings = ConsoleSessionSettings(
        provider="openai",
        model="prior-model",
        source="user",
    )
    prior = store.create_session(settings=prior_settings)
    controller = _controller(store=store)
    snapshot = RuntimeConfigSnapshot(
        17,
        _config(),
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
    )
    original = ConsoleFirstChatIntent("old-target", "openai", "model-a", 17)
    replacement = ConsoleFirstChatIntent("replacement-target", "openai", "model-a", 17)
    handoffs = controller.app_instance.pending_handoffs
    handoffs.stage_reserved_console_first_chat(original)
    real_create = store.create_session
    real_switch = store.switch_session
    switch_calls: list[str] = []

    def create_then_replace(**kwargs):
        created = real_create(**kwargs)
        if kwargs.get("session_id") == original.session_id:
            handoffs.stage_reserved_console_first_chat(replacement)
        return created

    monkeypatch.setattr(store, "create_session", create_then_replace)
    monkeypatch.setattr(
        store,
        "switch_session",
        lambda session_id: (
            switch_calls.append(session_id),
            real_switch(session_id),
        )[1],
    )

    assert controller.consume_pending_console_first_chat_intent() is False
    assert switch_calls == []
    assert store.active_session_id == prior.id
    assert all(item.id != original.session_id for item in store.sessions())
    claim = handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None
    assert claim.value == replacement
    assert handoffs.release(claim) is True


def test_stale_config_generation_releases_for_retry_and_deduplicates_warning(
    monkeypatch,
) -> None:
    events: list[object] = []
    store = ConsoleChatStore()
    controller = _controller(store=store, events=events)
    current = RuntimeConfigSnapshot(22, _config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current,
    )
    handoffs = controller.app_instance.pending_handoffs
    intent = ConsoleFirstChatIntent("stale-target", "openai", "model-a", 21)
    handoffs.stage_reserved_console_first_chat(intent)
    create_calls: list[dict[str, object]] = []
    real_create = store.create_session

    def record_create(**kwargs):
        create_calls.append(kwargs)
        return real_create(**kwargs)

    monkeypatch.setattr(store, "create_session", record_create)

    assert controller.consume_pending_console_first_chat_intent() is False
    assert controller.consume_pending_console_first_chat_intent() is False
    assert store.sessions() == []
    assert create_calls == []
    assert len([event for event in events if event[0] == "notify"]) == 1

    replacement = ConsoleFirstChatIntent("new-stale-target", "openai", "model-a", 20)
    handoffs.stage_reserved_console_first_chat(replacement)
    assert controller.consume_pending_console_first_chat_intent() is False
    assert len([event for event in events if event[0] == "notify"]) == 2
    claim = handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None
    assert claim.value == replacement
    assert handoffs.release(claim) is True


def test_generation_change_after_reserved_create_rolls_back_exact_session(
    monkeypatch,
) -> None:
    store = ConsoleChatStore()
    prior_settings = ConsoleSessionSettings(
        provider="openai",
        model="prior-model",
        source="user",
    )
    prior = store.create_session(settings=prior_settings)
    store.set_session_draft(prior.id, "preserve")
    controller = _controller(store=store)
    current = [RuntimeConfigSnapshot(31, _config())]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
    )
    intent = ConsoleFirstChatIntent("created-target", "openai", "model-a", 31)
    handoffs = controller.app_instance.pending_handoffs
    handoffs.stage_reserved_console_first_chat(intent)
    real_create = store.create_session

    def create_then_advance(**kwargs):
        created = real_create(**kwargs)
        if kwargs.get("session_id") == intent.session_id:
            current[0] = RuntimeConfigSnapshot(32, current[0].values)
        return created

    monkeypatch.setattr(store, "create_session", create_then_advance)

    assert controller.consume_pending_console_first_chat_intent() is False
    assert store.active_session_id == prior.id
    assert store.session_draft(prior.id) == "preserve"
    assert store.session_settings(prior.id) == prior_settings
    assert all(item.id != intent.session_id for item in store.sessions())
    claim = handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None and claim.value == intent
    assert handoffs.release(claim) is True


def test_generation_change_after_pristine_refresh_restores_exact_snapshot(
    monkeypatch,
) -> None:
    store = ConsoleChatStore()
    old_defaults = build_default_console_session_settings(_config("old-model"))
    target = store.create_session(
        settings=old_defaults,
        canonical_settings_baseline=old_defaults,
    )
    old_updated_at = target.updated_at
    controller = _controller(store=store)
    current = [RuntimeConfigSnapshot(41, _config())]
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
    )
    intent = ConsoleFirstChatIntent(target.id, "openai", "model-a", 41)
    handoffs = controller.app_instance.pending_handoffs
    handoffs.stage(HandoffChannel.CONSOLE_FIRST_CHAT, intent)
    real_refresh = store.refresh_pristine_session_settings

    def refresh_then_advance(*args, **kwargs):
        refreshed = real_refresh(*args, **kwargs)
        current[0] = RuntimeConfigSnapshot(42, current[0].values)
        return refreshed

    monkeypatch.setattr(
        store,
        "refresh_pristine_session_settings",
        refresh_then_advance,
    )

    assert controller.consume_pending_console_first_chat_intent() is False
    restored = next(item for item in store.sessions() if item.id == target.id)
    assert restored.settings == old_defaults
    assert restored.canonical_settings_baseline == old_defaults
    assert restored.updated_at == old_updated_at
    claim = handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None and claim.value == intent
    assert handoffs.release(claim) is True


def test_acknowledgement_exception_rolls_back_retries_and_logs_metadata_only(
    monkeypatch,
) -> None:
    events: list[object] = []
    store = ConsoleChatStore()
    prior = store.create_session(
        settings=ConsoleSessionSettings(
            provider="openai",
            model="prior-model",
            source="user",
        )
    )
    controller = _controller(store=store, events=events)
    snapshot = RuntimeConfigSnapshot(51, _config())
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
    )
    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        lambda _generation, action: action(),
    )
    intent = ConsoleFirstChatIntent(
        "PRIVATE_SESSION_ID", "openai", "model-a", snapshot.generation
    )
    handoffs = controller.app_instance.pending_handoffs
    handoffs.stage_reserved_console_first_chat(intent)
    real_acknowledge = handoffs.acknowledge_current
    real_release = handoffs.release
    captured: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        session_module.logger,
        "warning",
        lambda *args, **_kwargs: captured.append(args),
    )

    def reject_ack(_claim):
        raise RuntimeError("PRIVATE_ACK_EXCEPTION")

    def reject_release(_claim):
        raise RuntimeError("PRIVATE_RELEASE_EXCEPTION")

    monkeypatch.setattr(handoffs, "acknowledge_current", reject_ack)
    monkeypatch.setattr(handoffs, "release", reject_release)

    assert controller.consume_pending_console_first_chat_intent() is False
    assert store.active_session_id == prior.id
    assert all(item.id != intent.session_id for item in store.sessions())
    rendered = repr(captured)
    assert "PRIVATE_ACK_EXCEPTION" not in rendered
    assert "PRIVATE_RELEASE_EXCEPTION" not in rendered
    assert intent.session_id not in rendered
    assert "guarded-acknowledgement" in rendered
    assert "claim-release" in rendered
    assert "RuntimeError" in rendered

    monkeypatch.setattr(handoffs, "acknowledge_current", real_acknowledge)
    monkeypatch.setattr(handoffs, "release", real_release)
    assert controller.consume_pending_console_first_chat_intent() is True
    assert store.active_session_id == intent.session_id
    assert handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT) is None


@pytest.mark.asyncio
async def test_rollback_projects_scalars_then_restores_focus_after_native_sync() -> (
    None
):
    events: list[object] = []
    screen = _FakeScreen()
    screen.is_mounted = True
    controller = _controller(screen=screen, store=ConsoleChatStore(), events=events)

    controller._resync_mounted_console_after_first_chat_rollback(
        prior_control_provider="prior-provider",
        prior_control_model="prior-model",
        prior_focus_token="opaque-focus-token",
    )

    assert events == [
        ("controls", "prior-provider", "prior-model"),
        "core",
        "summary",
        "bar",
    ]
    assert len(screen.workers) == 1
    work, options = screen.workers[0]
    assert isinstance(work, partial)
    assert options == {
        "group": "console-first-chat-rollback",
        "exit_on_error": False,
    }

    await work()

    assert events[-2:] == ["native", ("focus", "opaque-focus-token")]


@pytest.mark.asyncio
async def test_rollback_does_not_restore_focus_after_unmount_during_sync() -> None:
    events: list[object] = []
    screen = _FakeScreen()
    screen.is_mounted = True
    controller = _controller(screen=screen, store=ConsoleChatStore(), events=events)

    async def sync_then_unmount() -> None:
        events.append("native")
        screen.is_mounted = False

    controller._sync_native_console_chat_ui_fn = sync_then_unmount
    controller._resync_mounted_console_after_first_chat_rollback(
        prior_control_provider=None,
        prior_control_model=None,
        prior_focus_token="opaque-focus-token",
    )
    work, _options = screen.workers[0]

    await work()

    assert events[-1] == "native"
    assert ("focus", "opaque-focus-token") not in events


def test_worker_rejection_does_not_allocate_an_unawaited_rollback_coroutine() -> None:
    events: list[object] = []
    screen = _FakeScreen()
    screen.is_mounted = True

    def reject_worker(_work, **_kwargs):
        raise RuntimeError("worker rejected")

    screen.run_worker = reject_worker
    controller = _controller(screen=screen, store=ConsoleChatStore(), events=events)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="worker rejected"):
            controller._resync_mounted_console_after_first_chat_rollback(
                prior_control_provider=None,
                prior_control_model=None,
                prior_focus_token="opaque-focus-token",
            )
        gc.collect()

    assert not [
        warning for warning in caught if "was never awaited" in str(warning.message)
    ]
