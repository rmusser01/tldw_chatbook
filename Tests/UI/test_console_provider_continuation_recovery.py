"""Keyboard and privacy coverage for Console continuation recovery."""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

from tldw_chatbook.Agents.agent_models import ContinuationEventContext, ToolBatchReady
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    ProviderContinuationRecoveryCallout,
    ProviderContinuationTranscriptRegion,
    provider_continuation_recovery_state,
)


def _message(
    *, call_state: str = "pending", remote: bool = False
) -> ConsoleChatMessage:
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k2",
        api_base_url="https://api.moonshot.ai/v1",
        state="active",
        rounds=(
            ContinuationRound(
                "PRIVATE_REASONING_CANARY",
                ("PRIVATE_REASONING_BLOCK",),
                (
                    ContinuationCall(
                        "PRIVATE_CALL_ID",
                        "calculator",
                        '{"secret":"PRIVATE_ARGUMENT_CANARY"}',
                        call_state,  # type: ignore[arg-type]
                    ),
                ),
            ),
        ),
    )
    return ConsoleChatMessage(
        id="assistant-owner",
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        provider_continuation=checkpoint,
        provider_continuation_remote=remote,
        provider_continuation_message_version=1,
    )


class _RecoveryApp(App):
    def __init__(
        self,
        message: ConsoleChatMessage,
        *,
        replay_available: bool = True,
    ) -> None:
        super().__init__()
        self.message = message
        self.replay_available = replay_available
        self.actions: list[tuple[str, str, int]] = []

    def compose(self) -> ComposeResult:
        yield ProviderContinuationRecoveryCallout(
            state=provider_continuation_recovery_state(
                self.message,
                replay_available=self.replay_available,
            ),
            on_action=self._record,
        )

    async def _record(self, action: str, message_id: str, version: int) -> bool:
        self.actions.append((action, message_id, version))
        return True


async def test_pending_recovery_is_private_wrapped_and_keyboard_reachable() -> None:
    app = _RecoveryApp(_message())
    async with app.run_test(size=(38, 16)) as pilot:
        await pilot.pause()
        rendered = "\n".join(
            str(widget.render())
            for widget in app.screen.query("ProviderContinuationRecoveryCallout *")
            if hasattr(widget, "render")
        )
        assert "Interrupted tool run" in rendered
        assert "may not have finished" in rendered
        assert "PRIVATE_" not in rendered
        assert app.actions == []

        await pilot.press("tab")
        await pilot.press("shift+tab")
        await pilot.press("enter")
        await pilot.pause()
        assert app.actions == [("resume", "assistant-owner", 1)]


async def test_executing_recovery_blocks_resume_and_explains_why() -> None:
    app = _RecoveryApp(_message(call_state="executing"))
    async with app.run_test(size=(42, 16)) as pilot:
        await pilot.pause()
        assert not app.screen.query_one("#console-continuation-resume", Button).display
        detail = str(app.screen.query_one("#console-continuation-impact").render())
        assert "may already have run" in detail
        assert app.actions == []


async def test_remote_recovery_offers_take_over_not_resume() -> None:
    app = _RecoveryApp(_message(remote=True))
    async with app.run_test(size=(46, 18)) as pilot:
        await pilot.pause()
        assert not app.screen.query_one("#console-continuation-resume", Button).display
        assert len(app.screen.query("#console-continuation-take-over")) == 1
        warning = str(app.screen.query_one("#console-continuation-impact").render())
        assert "other device may still be running" in warning
        assert "exactly-once" not in warning


async def test_unavailable_replay_is_honest_and_discard_remains_enabled() -> None:
    app = _RecoveryApp(_message(), replay_available=False)
    async with app.run_test(size=(46, 18)) as pilot:
        await pilot.pause()
        resume = app.screen.query_one("#console-continuation-resume", Button)
        discard = app.screen.query_one("#console-continuation-discard", Button)
        impact = str(app.screen.query_one("#console-continuation-impact").render())
        assert resume.display
        assert resume.disabled
        assert not discard.disabled
        assert "replay support" in impact
        assert "provider integration" in impact


async def test_quarantined_generation_offers_reload_with_accurate_copy() -> None:
    message = _message()
    message.provider_continuation = None
    message.provider_continuation_warning = (
        "Canonical generation is unavailable; reload required."
    )
    message.provider_continuation_message_version = None
    message.generation_projection_quarantined = True
    message.generation_projection_quarantine_version = 3
    app = _RecoveryApp(message)

    async with app.run_test(size=(48, 18)) as pilot:
        await pilot.pause()
        title = str(
            app.screen.query_one("#console-continuation-title", Static).render()
        )
        status = str(
            app.screen.query_one("#console-continuation-status", Static).render()
        )
        reload_button = app.screen.query_one("#console-continuation-reload", Button)
        assert "Generation unavailable" in title
        assert "unchanged" not in status
        assert "No action is required" not in status
        assert reload_button.display
        assert not reload_button.disabled

        reload_button.focus()
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        assert app.actions == [("reload", "assistant-owner", 3)]


class _RegionApp(App):
    def __init__(
        self,
        message_builder,
        *,
        replay_available: bool = True,
        on_action=None,
        owner_live=None,
    ) -> None:
        super().__init__()
        self.message_builder = message_builder
        self.replay_available = replay_available
        self.on_action = on_action or (lambda *_args: True)
        self.owner_live = owner_live or (lambda _message: False)

    def compose(self) -> ComposeResult:
        yield ProviderContinuationTranscriptRegion(
            session_surface_builder=lambda: Static("Visible transcript"),
            recovery_message_builder=self.message_builder,
            recovery_replay_available_builder=lambda: self.replay_available,
            recovery_owner_live_builder=self.owner_live,
            on_recovery_action=self.on_action,
        )


class _BlockingRecovery:
    def __init__(self, *, succeeds: bool) -> None:
        self.succeeds = succeeds
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.actions: list[tuple[str, str, int]] = []

    async def __call__(self, action: str, message_id: str, version: int) -> bool:
        self.actions.append((action, message_id, version))
        self.started.set()
        await self.release.wait()
        return self.succeeds


class _CancelOnceRecovery:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str, int]] = []

    async def __call__(self, action: str, message_id: str, version: int) -> bool:
        self.actions.append((action, message_id, version))
        if len(self.actions) == 1:
            raise asyncio.CancelledError
        return False


async def test_cancelled_recovery_releases_busy_guard_and_remains_recoverable() -> None:
    message = _message()
    recovery = _CancelOnceRecovery()
    app = _RegionApp(lambda: message, on_action=recovery)
    async with app.run_test(size=(48, 18)) as pilot:
        await pilot.pause()
        region = app.screen.query_one(ProviderContinuationTranscriptRegion)
        callout = app.screen.query_one(ProviderContinuationRecoveryCallout)
        resume = callout.query_one("#console-continuation-resume", Button)
        resume.focus()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        region.sync_recovery()
        await pilot.pause()
        assert not callout._busy
        assert not resume.disabled
        assert recovery.actions == [("resume", "assistant-owner", 1)]
        rendered = "\n".join(
            str(widget.render())
            for widget in callout.query("*")
            if hasattr(widget, "render")
        )
        assert "PRIVATE_" not in rendered

        discard = callout.query_one("#console-continuation-discard", Button)
        discard.focus()
        await pilot.pause()
        assert app.focused is discard
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        assert recovery.actions == [
            ("resume", "assistant-owner", 1),
            ("discard", "assistant-owner", 1),
        ]


async def test_abandoned_dispatch_releases_busy_guard_before_later_restore(
    monkeypatch,
) -> None:
    message = _message()
    checkpoint = message.provider_continuation
    actions: list[tuple[str, str, int]] = []

    async def recover(action: str, message_id: str, version: int) -> bool:
        actions.append((action, message_id, version))
        return False

    app = _RegionApp(lambda: message, on_action=recover)
    async with app.run_test(size=(48, 18)) as pilot:
        await pilot.pause()
        region = app.screen.query_one(ProviderContinuationTranscriptRegion)
        callout = app.screen.query_one(ProviderContinuationRecoveryCallout)
        original_dispatch = callout._dispatch
        dispatch_started = asyncio.Event()
        release_dispatch = asyncio.Event()

        async def delayed_dispatch(action: str) -> None:
            dispatch_started.set()
            await release_dispatch.wait()
            await original_dispatch(action)

        monkeypatch.setattr(callout, "_dispatch", delayed_dispatch)
        resume = callout.query_one("#console-continuation-resume", Button)
        resume.focus()
        await pilot.press("enter")
        await asyncio.wait_for(dispatch_started.wait(), timeout=1)

        message.provider_continuation = None
        region.sync_recovery()
        release_dispatch.set()
        await pilot.app.workers.wait_for_complete()
        assert actions == []
        assert not callout._busy

        message.provider_continuation = checkpoint
        region.sync_recovery()
        await pilot.pause()
        assert callout.display
        assert not resume.disabled
        rendered = "\n".join(
            str(widget.render())
            for widget in callout.query("*")
            if hasattr(widget, "render")
        )
        assert "PRIVATE_" not in rendered

        discard = callout.query_one("#console-continuation-discard", Button)
        discard.focus()
        await pilot.pause()
        assert app.focused is discard
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        assert actions == [("discard", "assistant-owner", 1)]


@pytest.mark.parametrize("transition", ["same", "changed", "removed"])
@pytest.mark.parametrize("succeeds", [True, False])
async def test_sync_during_recovery_never_releases_inflight_action(
    transition: str,
    succeeds: bool,
) -> None:
    message = _message()
    checkpoint = message.provider_continuation
    blocker = _BlockingRecovery(succeeds=succeeds)
    app = _RegionApp(lambda: message, on_action=blocker)
    async with app.run_test(size=(48, 18)) as pilot:
        await pilot.pause()
        region = app.screen.query_one(ProviderContinuationTranscriptRegion)
        callout = app.screen.query_one(ProviderContinuationRecoveryCallout)
        resume = callout.query_one("#console-continuation-resume", Button)
        resume.focus()
        await pilot.press("enter")
        await asyncio.wait_for(blocker.started.wait(), timeout=1)

        if transition == "changed":
            message.provider_continuation_remote = True
        elif transition == "removed":
            message.provider_continuation = None
        region.sync_recovery()
        await pilot.pause()

        retry = callout.query_one(
            "#console-continuation-take-over"
            if transition == "changed"
            else "#console-continuation-resume",
            Button,
        )
        assert retry.disabled
        retry.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert blocker.actions == [("resume", "assistant-owner", 1)]
        rendered = "\n".join(
            str(widget.render())
            for widget in callout.query("*")
            if hasattr(widget, "render")
        )
        assert "PRIVATE_" not in rendered

        blocker.release.set()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        if transition == "changed":
            assert callout.display
            assert not callout.query_one(
                "#console-continuation-take-over", Button
            ).disabled
        elif transition == "removed" or succeeds:
            assert not callout.display
        else:
            assert callout.display

        message.provider_continuation = checkpoint
        message.provider_continuation_remote = False
        region.sync_recovery()
        await pilot.pause()
        resume = callout.query_one("#console-continuation-resume", Button)
        assert not resume.disabled
        resume.focus()
        await pilot.pause()
        assert app.focused is resume
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        assert len(blocker.actions) == 2


async def test_invalid_private_hydration_shows_safe_warning_without_actions() -> None:
    database = CharactersRAGDB(":memory:", "console-continuation-invalid-ui")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(database))
        session = store.create_session(title="Invalid recovery")
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="keep this",
            persist=True,
        )
        owner = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Visible content survives",
            persist=True,
        )
        store.persist_provider_continuation_event(
            ToolBatchReady(
                ContinuationEventContext(owner.id, "run", "primary", "persistent"),
                _message().provider_continuation,
                None,
            )
        )
        original_getter = database.get_messages_for_conversation

        def invalid_private_rows(conversation_id, *, limit=100_000):
            rows = original_getter(conversation_id, limit=limit)
            return [
                {
                    **row,
                    "provider_continuation_json": (
                        "PRIVATE_INVALID_CANARY"
                        if str(row["id"]) == owner.persisted_message_id
                        else None
                    ),
                }
                for row in rows
            ]

        database.get_messages_for_conversation = invalid_private_rows
        restored = ConsoleChatStore(persistence=ChatPersistenceService(database))
        loaded = restored.restore_persisted_session(
            title="Invalid recovery",
            workspace_id=None,
            persisted_conversation_id=session.persisted_conversation_id,
            all_nodes=store.messages_for_session(session.id),
            active_leaf_persisted_id=owner.persisted_message_id,
        )
        hydrated = restored.messages_for_session(loaded.id)[-1]
        assert hydrated.content == "Visible content survives"

        app = _RegionApp(lambda: hydrated)
        async with app.run_test(size=(42, 16)) as pilot:
            await pilot.pause()
            rendered = "\n".join(
                str(widget.render())
                for widget in app.screen.query("ProviderContinuationRecoveryCallout *")
                if hasattr(widget, "render")
            )
            assert "Exact tool continuation was discarded." in rendered
            assert "PRIVATE_INVALID_CANARY" not in rendered
            assert not any(button.display for button in app.screen.query(Button))
    finally:
        database.close_connection()


async def test_failed_recovery_uses_specific_message_warning() -> None:
    message = _message()

    async def fail_recovery(*_args) -> bool:
        message.provider_continuation_warning = "Pinned provider settings no longer match. Restore those settings or Discard."
        return False

    app = _RegionApp(lambda: message, on_action=fail_recovery)
    async with app.run_test(size=(48, 18)) as pilot:
        await pilot.pause()
        resume = app.screen.query_one("#console-continuation-resume", Button)
        resume.focus()
        await pilot.press("enter")
        await pilot.pause()
        rendered = str(app.screen.query_one("#console-continuation-impact").render())
        assert "Pinned provider settings no longer match" in rendered
        assert "Reload the conversation" not in rendered


async def test_failed_recovery_without_new_warning_reenables_safe_actions() -> None:
    async def raise_recovery(*_args) -> bool:
        raise RuntimeError("private provider failure")

    app = _RegionApp(lambda: _message(), on_action=raise_recovery)
    async with app.run_test(size=(48, 18)) as pilot:
        await pilot.pause()
        resume = app.screen.query_one("#console-continuation-resume", Button)
        resume.focus()
        await pilot.press("enter")
        await pilot.pause()

        discard = app.screen.query_one("#console-continuation-discard", Button)
        assert not discard.disabled
        assert "Reload the conversation" in str(
            app.screen.query_one("#console-continuation-status").render()
        )


async def test_real_region_hides_live_owner_then_exposes_interruption() -> None:
    """Only an owner no longer registered to a live run is recoverable."""
    database = CharactersRAGDB(":memory:", "console-continuation-live-ui")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(database))
        session = store.create_session(title="Live owner")
        owner = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Visible preface",
            persist=True,
        )
        store.persist_provider_continuation_event(
            ToolBatchReady(
                ContinuationEventContext(owner.id, "run", "primary", "persistent"),
                _message().provider_continuation,
                None,
            )
        )
        controller = ConsoleChatController(
            store=store,
            provider_gateway=object(),
            agent_bridge=object(),
        )
        controller._active_assistant_message_ids[session.id] = owner.id
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Agent running."),
            session_id=session.id,
        )
        app = _RegionApp(
            lambda: store.provider_continuation_recovery_message(),
            owner_live=lambda message: controller.provider_continuation_owner_is_live(
                message.id
            ),
            on_action=controller.recover_provider_continuation,
        )
        async with app.run_test(size=(48, 18)) as pilot:
            await pilot.pause()
            region = app.screen.query_one(ProviderContinuationTranscriptRegion)
            callout = app.screen.query_one(ProviderContinuationRecoveryCallout)
            assert not callout.display

            controller._active_assistant_message_ids.pop(session.id)
            region.sync_recovery()
            await pilot.pause()
            assert callout.display
            discard = callout.query_one("#console-continuation-discard", Button)
            discard.focus()
            await pilot.pause()
            assert app.focused is discard
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            assert store.get_message(owner.id).provider_continuation is None
    finally:
        database.close_connection()


async def test_real_screen_sync_mounts_updates_and_clears_recovery_callout() -> None:
    database = CharactersRAGDB(":memory:", "console-continuation-reactive-ui")
    app_instance = _build_test_app()
    app_instance.chachanotes_db = database
    app_instance.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "local-model",
    }
    app_instance.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "local-model",
        }
    }
    host = ConsoleHarness(app_instance)
    try:
        async with host.run_test(size=(92, 32)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-main-column")
            callout = console.query_one(ProviderContinuationRecoveryCallout)
            assert not callout.display

            store = console._ensure_console_chat_store()
            session_id = store.active_session_id
            assert session_id is not None
            store.append_message(
                session_id,
                role=ConsoleMessageRole.USER,
                content="Use a tool",
                persist=True,
            )
            owner = store.append_message(
                session_id,
                role=ConsoleMessageRole.ASSISTANT,
                content="Visible preface",
                persist=True,
            )
            store.persist_provider_continuation_event(
                ToolBatchReady(
                    ContinuationEventContext(
                        owner.id,
                        "run",
                        "primary",
                        "persistent",
                    ),
                    _message().provider_continuation,
                    None,
                )
            )
            controller = console._ensure_console_chat_controller()
            controller._active_assistant_message_ids[session_id] = owner.id
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STREAMING, "Agent running."),
                session_id=session_id,
            )
            await console._sync_native_console_chat_ui()
            assert not callout.display

            controller._active_assistant_message_ids.pop(session_id)
            await console._sync_native_console_chat_ui()
            assert callout.display
            assert "Interrupted tool run" in str(
                callout.query_one("#console-continuation-title").render()
            )

            discard = callout.query_one("#console-continuation-discard", Button)
            discard.focus()
            await pilot.pause()
            assert console.focused is discard
            await pilot.press("enter")
            await pilot.pause()
            assert not callout.display
            assert store.get_message(owner.id).provider_continuation is None
            assert console.focused is not None
            assert console.focused is not discard
    finally:
        database.close_connection()
