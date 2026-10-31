"""A DONE run must finalize ``complete`` when shutdown or close lands late.

Console UAT 2026-10-17 finding 2 (``qa/console-uat-2026-10-17/``): assistant
replies whose run had already finished ended with
``assistant_generation_state='stopped'`` instead of ``'complete'`` when the
app instance shut down soon after -- content fully persisted, terminal state
wrong. ``shutdown()``/session close reuse ``_signal_stop`` (indistinguishable
from a user Stop) and cancel the submit task; the cancellation then wins over
the READY run outcome and a stopped mark is written.

Mirroring ``test_console_agent_swap.py``'s post-outcome-window tests (task-227
AC3): a user Stop in that window must KEEP marking ``stopped`` (existing
spec), while an app-shutdown or session-close stop must let the finished
reply finalize ``complete``. The stop source is threaded through
``_signal_stop`` exactly as the stop callers pass it.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome

from Tests.Chat.test_console_agent_swap import _Gateway

# bootstrap_profile: keep the collection-time config profile so send
# admission's guarded hooks-config read does not fail closed under the
# per-test sandbox redirect (same remedy as test_console_agent_swap.py).
pytestmark = pytest.mark.bootstrap_profile

REPLY_TEXT = "the finished reply, fully streamed"


def _controller(tmp_path: Path, name: str) -> tuple:
    gateway = _Gateway([[REPLY_TEXT]])
    store = ConsoleChatStore(
        persistence=ChatPersistenceService(
            CharactersRAGDB(str(tmp_path / f"{name}-chacha.sqlite"), client_id="t")
        )
    )
    db = AgentRunsDB(tmp_path / f"{name}-runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    return controller, store


def _assistant_for_active_session(store) -> str:
    messages = store.messages_for_session(store.active_session_id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    return assistant.id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "expected_state"),
    (("shutdown", "complete"), ("session_close", "complete"), ("user", "stopped")),
)
async def test_stop_source_in_post_done_window_decides_terminal_state(
    tmp_path: Path, stop_reason: str, expected_state: str
) -> None:
    """A stop landing after RUN_DONE finalizes per its source (TASK-35311).

    Mirrors ``test_finalize_after_already_stopped_is_a_benign_noop``'s
    window: the bridge has returned RUN_DONE, the stop lands before
    ``_finalize_agent_reply`` runs. A user Stop keeps the existing
    ``stopped`` contract; shutdown and session close must not stamp
    ``stopped`` on a reply that provably finished.
    """
    controller, store = _controller(tmp_path, f"window-{stop_reason}")

    def stopped_then_done_run_reply(*, assistant_message_id, **_kwargs):
        # Thread-safe window injection (run_reply executes on a worker
        # thread): set the run's own permanent cancel event directly and
        # record the stop source, mirroring what _signal_stop's callers
        # hand it -- same style as test_console_agent_swap's task-227
        # post-outcome-window tests. The session is created by the send
        # itself, so resolve it from the store at run time.
        session_id = store.active_session_id
        # The real bridge streams the reply into the message before it
        # returns; the finalizer's success path trusts that streamed
        # content (final_text is only its expectation), so stream here too.
        controller.store.append_stream_chunk(assistant_message_id, REPLY_TEXT)
        event = controller._active_cancel_events.get(session_id)
        if event is not None:
            event.set()
        getattr(controller, "_record_active_stop_reason", lambda *_a, **_k: None)(
            session_id, stop_reason
        )
        if stop_reason == "user":
            # The user-Stop path pre-marks the message stopped before
            # finalize runs (the CancelledError branch's _mark_stream_
            # stopped); finalize then treats it as a benign no-op.
            store.mark_message_stopped(assistant_message_id)
        return "run-test", RunOutcome(status=RUN_DONE, steps=[], final_text=REPLY_TEXT)

    controller._agent_bridge.run_reply = stopped_then_done_run_reply
    result = await asyncio.wait_for(controller.submit_draft("hi"), timeout=30)
    assert result.accepted is True

    assistant_id = _assistant_for_active_session(store)
    stored = store.get_message(assistant_id)
    assert stored.assistant_generation_state == expected_state, (
        f"a {stop_reason} stop after RUN_DONE must finalize "
        f"{expected_state!r}, got {stored.assistant_generation_state!r}"
    )
    assert controller._active_stop_reasons == {}
    assert controller._active_provider_workers == {}
    if expected_state == "complete":
        assert REPLY_TEXT in (stored.content or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reasons", "expected_state"),
    ((("shutdown",), "complete"), (("user", "shutdown"), "stopped")),
)
async def test_shutdown_cancellation_at_the_worker_await_finalizes_complete(
    tmp_path: Path,
    stop_reasons: tuple[str, ...],
    expected_state: str,
) -> None:
    """The UAT's exact window: shutdown's signal-then-cancel lands while the
    submit task is suspended on the provider-worker await whose result is
    already READY (Console UAT 2026-10-17 finding 2).

    Drives ``shutdown()``'s documented per-session sequence directly --
    ``_signal_stop(reason="shutdown")`` then ``task.cancel()`` -- so the
    window is deterministic instead of racing shutdown's drain phases.
    """
    controller, store = _controller(tmp_path, "worker-await-cancel")
    chat_start = controller._chat_start
    real_worker = chat_start.run_provider_worker
    worker_done = asyncio.Event()
    release = asyncio.Event()

    async def gated_worker(session_id, operation):
        result = await real_worker(session_id, operation)
        worker_done.set()
        await release.wait()  # the controller's await, result ready
        return result

    chat_start.run_provider_worker = gated_worker

    task = asyncio.create_task(controller.submit_draft("hi"))
    await asyncio.wait_for(worker_done.wait(), timeout=30)
    assistant_id = _assistant_for_active_session(store)
    session_id = store.active_session_id

    for reason in stop_reasons:
        controller._signal_stop(session_id=session_id, reason=reason)
    task.cancel()
    await asyncio.sleep(0)  # let the cancellation land at the gated await
    release.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=15)

    stored = store.get_message(assistant_id)
    assert stored.assistant_generation_state == expected_state, (
        f"{stop_reasons!r} at the worker await must leave the DONE reply "
        f"{expected_state}, got {stored.assistant_generation_state!r}"
    )
    assert REPLY_TEXT in (stored.content or "")


@pytest.mark.asyncio
async def test_shutdown_after_done_retains_citation_selection_ownership():
    from Tests.Chat.test_console_local_citation_boundary import (
        _controlled_citation_repair,
    )

    controller, store, gateway, _bridge, initial_body, _ = _controlled_citation_repair(
        agent=True
    )
    real_worker = controller._chat_start.run_provider_worker
    worker_done = asyncio.Event()
    release = asyncio.Event()

    async def gated_worker(session_id, operation):
        result = await real_worker(session_id, operation)
        worker_done.set()
        await release.wait()
        return result

    controller._chat_start.run_provider_worker = gated_worker
    task = asyncio.create_task(controller.submit_draft("question"))
    try:
        await asyncio.wait_for(worker_done.wait(), timeout=30)
        assistant_id = _assistant_for_active_session(store)
        repair = controller._active_citation_repair_sessions[store.active_session_id]
        await controller.shutdown()
        await task
        assistant = store.get_message(assistant_id)
        assert assistant.status == "complete"
        assert assistant.content == initial_body
        assert repair.selection_committed
        assert gateway.calls == []
        assert all(
            row.role is not ConsoleMessageRole.SYSTEM
            for row in store.messages_for_session(store.active_session_id)
        )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_failure_retires_retained_worker(tmp_path):
    controller, store = _controller(tmp_path, "provider-failure")

    def fail(**kwargs):
        raise RuntimeError("provider failed")

    controller._agent_bridge.run_reply = fail
    await controller.submit_draft("hi")
    assert store.get_message(_assistant_for_active_session(store)).status == "failed"
    assert controller._active_provider_workers == {}
