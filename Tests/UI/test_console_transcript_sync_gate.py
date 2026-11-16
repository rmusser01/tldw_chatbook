"""The transcript gate retains only an original completed widget publication."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_hook_review_send_freeze import _until
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


@asynccontextmanager
async def _screen(store, *, recovery=None):
    """Complete the original pipeline against an actually mounted transcript."""
    from textual.app import ComposeResult

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from Tests.UI.test_console_turn_activity_line import _sync_stub
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState
    from tldw_chatbook.Chat.console_roleplay_identity import ConsolePresentationContext
    from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
        ProviderContinuationTranscriptRegion,
    )
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    class GateApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            if recovery is None:
                yield ConsoleTranscript(id="console-native-transcript")
            else:
                yield ProviderContinuationTranscriptRegion(
                    session_surface_builder=lambda: ConsoleTranscript(
                        id="console-native-transcript"
                    ),
                    recovery_message_builder=lambda: recovery["continuation"],
                    on_recovery_action=lambda *_: False,
                    trace_recovery_state_builder=lambda: recovery["trace"],
                )

    app = GateApp()
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        transcript = app.query_one(ConsoleTranscript)
        assert transcript.is_mounted and transcript.is_attached
        screen, _unused_transcript = _sync_stub("")
        screen.query_one = app.query_one
        screen._closing = screen._closed = False
        screen._console_preparing_display_task = None
        screen._console_send_ack = None
        screen._console_chat_store = store
        store._sessions = {}
        screen._ensure_console_chat_store.return_value = store
        state = ConsoleRunState()
        screen._console_chat_controller = SimpleNamespace(
            run_state_for=lambda _sid: state,
            has_pending_approval_round=lambda _sid: False,
            pending_round_kinds=lambda _sid: (),
            context_control_inputs=lambda *_args, **_kwargs: (None, None, None),
        )
        screen._console_presentation_context.return_value = ConsolePresentationContext()
        screen._show_model_thinking.return_value = False
        screen._current_console_conversation_id.return_value = None
        screen._current_console_run_status_value.return_value = "idle"
        screen._rendered_console_terminal_receipts.return_value = ()
        screen._library_outcome.sync_transcript.return_value = {}
        screen._message._pending_console_delete_message_id = None
        screen._message._console_delete_scope = None
        fetches = []
        screen._message._native_console_messages.side_effect = lambda: (
            fetches.append(1),
            [],
        )[1]
        screen._change_review_projection = SimpleNamespace(
            project=lambda messages: messages, revision_key=lambda: ("conv", 1)
        )
        screen._console_transcript_region_or_none.return_value = (
            app.query_one(ProviderContinuationTranscriptRegion)
            if recovery is not None
            else None
        )
        screen._console_transcript_sync_ticks = 0
        screen._console_transcript_sync_token = None
        screen._last_native_transcript_session_id = None
        yield screen, fetches


def _publication_waits_for_refresh(task, transcript):
    """Identify this publication's original refresh and exact lock waiter."""
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    waiting = getattr(task, "_fut_waiter", None)
    if waiting is None or not any(
        waiting is waiter for waiter in (transcript._refresh_lock._waiters or ())
    ):
        return False
    current = task.get_coro()
    while current is not None:
        frame = getattr(current, "cr_frame", None)
        if (
            frame is not None
            and frame.f_code is ConsoleTranscript.refresh_messages.__code__
            and frame.f_locals.get("self") is transcript
        ):
            return True
        current = getattr(current, "cr_await", None)
    return False


async def _run(screen):
    await ChatScreen._sync_native_console_transcript(screen)
    # No swallowed exception or manufactured token: these are written only
    # after the original mounted-widget pipeline reaches its normal end.
    assert screen._last_native_transcript_refresh_key is not None
    assert (
        screen._last_native_transcript_session_id
        == screen._console_chat_store.active_session_id
    )


@pytest.mark.asyncio
async def test_unchanged_token_skips_the_message_pipeline():
    store = SimpleNamespace(
        active_session_id="s1", display_projection_revision=lambda _sid: 7
    )
    async with _screen(store) as (screen, fetches):
        await _run(screen)
        assert len(fetches) == 1, "first tick must run the pipeline"
        await _run(screen)
        await _run(screen)
        assert len(fetches) == 1, "unchanged token must skip the pipeline"


@pytest.mark.asyncio
async def test_token_bump_reopens_the_pipeline():
    state = {"rev": 7}
    store = SimpleNamespace(
        active_session_id="s1", display_projection_revision=lambda _sid: state["rev"]
    )
    async with _screen(store) as (screen, fetches):
        await _run(screen)
        state["rev"] += 1
        await _run(screen)
        assert len(fetches) == 2, "a store change must re-run the pipeline"


@pytest.mark.asyncio
async def test_forced_pass_runs_every_fifteenth_tick():
    store = SimpleNamespace(
        active_session_id="s1", display_projection_revision=lambda _sid: 7
    )
    async with _screen(store) as (screen, fetches):
        for _ in range(15):
            await _run(screen)
        assert len(fetches) == 2, "ticks 1 and 15 run; ticks 2-14 skip"


@pytest.mark.asyncio
async def test_session_switch_reopens_the_pipeline():
    store = SimpleNamespace(
        active_session_id="s1", display_projection_revision=lambda _sid: 7
    )
    async with _screen(store) as (screen, fetches):
        await _run(screen)
        store.active_session_id = "s2"
        await _run(screen)
        assert len(fetches) == 2, "switching sessions must re-run the pipeline"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["trace", "continuation"])
async def test_recovery_changes_are_visible_while_message_pipeline_is_gated(kind):
    from Tests.UI.test_console_provider_continuation_recovery import _message
    from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
        ProviderContinuationRecoveryCallout,
        TraceCallRecoveryCallout,
        TraceCallRecoveryState,
    )

    current = {"trace": None, "continuation": None}
    store = SimpleNamespace(
        active_session_id="s1", display_projection_revision=lambda _sid: 7
    )
    async with _screen(store, recovery=current) as (screen, fetches):
        region = screen._console_transcript_region_or_none()
        card = region.query_one(
            TraceCallRecoveryCallout
            if kind == "trace"
            else ProviderContinuationRecoveryCallout
        )
        assert region.is_mounted and not card.display
        await _run(screen)
        assert fetches == [1]
        current[kind] = (
            TraceCallRecoveryState("held-send") if kind == "trace" else _message()
        )
        await _run(screen)
        assert card.display
        current[kind] = None
        await _run(screen)
        assert not card.display
        assert fetches == [1], "recovery must not reopen the message pipeline"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["error", "cancel"])
async def test_abandoned_publication_keeps_the_previous_completed_gate(outcome):
    state = {"rev": 7}
    store = SimpleNamespace(
        active_session_id="s1", display_projection_revision=lambda _sid: state["rev"]
    )
    async with _screen(store) as (screen, _fetches):
        await _run(screen)
        prior = screen._console_transcript_sync_token
        assert prior is not None
        state["rev"] += 1
        if outcome == "error":

            def fail(_messages):
                raise RuntimeError("controlled downstream view failure")

            screen._change_review_projection.project = fail
            with pytest.raises(
                RuntimeError, match="controlled downstream view failure"
            ):
                await ChatScreen._sync_native_console_transcript(screen)
            assert screen._console_transcript_sync_token is prior
        else:
            transcript = screen.query_one("#console-native-transcript")
            screen._native_console_transcript_fingerprint.return_value = ("changed",)
            await transcript._refresh_lock.acquire()
            task = asyncio.create_task(
                ChatScreen._sync_native_console_transcript(screen)
            )
            try:
                assert await _until(
                    lambda: _publication_waits_for_refresh(task, transcript), 5
                )
                assert screen._console_transcript_sync_token is prior
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                assert task.cancelled()
                assert screen._console_transcript_sync_token is prior
            finally:
                if not task.done():
                    task.cancel()
                transcript._refresh_lock.release()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_citation_publication_during_first_refresh_reopens_the_gate():
    """The original count producer cannot be hidden by an older render finishing."""
    from types import MethodType
    from unittest.mock import AsyncMock

    from Tests.UI.test_console_citation_sources import (
        _FakeRepository,
        _active_result,
        _message,
        _trace,
    )

    store = SimpleNamespace(
        active_session_id="s1", display_projection_revision=lambda _sid: 7
    )
    async with _screen(store) as (screen, fetches):
        message = _message("citation-answer", persisted_message_id="saved-answer")
        messages = [message]
        screen._message._native_console_messages.side_effect = lambda: (
            fetches.append(1), messages
        )[1]
        screen._message._console_citation_message_body.side_effect = (
            lambda current: current.content
        )
        repository = _FakeRepository(_active_result(_trace()))
        screen.app_instance = SimpleNamespace(
            citation_trace_repository=repository, chachanotes_db=None
        )
        for name in (
            "_console_citation_message_body",
            "_console_citation_signature",
            "_console_citation_repository_readiness",
            "_apply_console_citation_counts",
        ):
            setattr(screen, name, MethodType(getattr(ChatScreen, name), screen))
        screen._read_console_citation_counts = ChatScreen._read_console_citation_counts
        signature = screen._console_citation_signature(messages)
        screen._console_citation_input_signature = signature
        screen._console_citation_request_generation = 1
        screen._console_citation_resolved_signatures = {}
        # The producer requests FULL while the original publication is held.
        # Its existing coalescer is a separate mounted-journey obligation.
        screen._sync_native_console_chat_ui = AsyncMock(return_value=False)
        transcript = screen.query_one("#console-native-transcript")
        assert screen._console_transcript_sync_token is None
        await transcript._refresh_lock.acquire()
        released = False
        task = asyncio.create_task(ChatScreen._sync_native_console_transcript(screen))
        try:
            assert await _until(
                lambda: _publication_waits_for_refresh(task, transcript), 5
            )
            await ChatScreen._discover_console_citation_counts(
                screen, repository, signature, 1
            )
            assert repository.calls == [("saved-answer", message.content)]
            assert repository.verified_results == [repository.result]
            assert screen._console_citation_counts == {"citation-answer": 2}
            assert screen._console_transcript_sync_token is None
            screen._sync_native_console_chat_ui.assert_awaited_once_with()
            transcript._refresh_lock.release()
            released = True
            await task
            assert screen._console_transcript_sync_token is None, (
                "an older publication swallowed the original citation invalidation"
            )
            before_replay = len(fetches)
            await _run(screen)
            assert len(fetches) == before_replay + 1
            sources = transcript.query_one("#console-citation-sources-citation-answer")
            assert sources.is_mounted and sources.is_attached
            assert str(sources.label) == "Cited sources (2)"
        finally:
            if not task.done():
                task.cancel()
            if not released:
                transcript._refresh_lock.release()
            await asyncio.gather(task, return_exceptions=True)
