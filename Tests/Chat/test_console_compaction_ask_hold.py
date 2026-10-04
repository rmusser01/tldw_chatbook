"""TASK-34350: the Ask compaction hold, end to end in a live session.

Owner ruling (2026-10-03): a send over the compaction threshold under the
default Ask policy prompts to compact instead of dead-ending. Before this, the
send was accepted and then blocked inside the stream: the reply was marked
Failed, a System row said "Review and approve compaction before sending
again" with nothing to approve, and a durable turn surfaced as "Response
accepted; waiting for dispatch" (TASK-33621.4).

The hold now happens before the turn is committed, as a paused preparation:
nothing is persisted, nothing is marked Failed, and there is no dispatch
checkpoint to recover. Compact and send, Send without compacting and Cancel
act on that exact held send. Same harness as the live-session suite: real DB,
store, controller send, context repository, compaction service and request
preparation; only the provider network is doubled.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Chat.test_console_compaction_live_session import (
    _OVERRIDES,
    _LiveProviderGateway,
    _active_memory_count,
    _live_controller,
    _system_rows,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)

pytestmark = pytest.mark.bootstrap_profile

_ASK = replace(_OVERRIDES, compaction_mode=ContextCompactionMode.ASK)


async def _send_until_held(controller, gateway, *, limit: int = 12):
    """Send ordinary turns until one is held; return (draft, result)."""

    for index in range(limit):
        draft = f"question-{index}: explain step {index} in detail."
        streams = gateway.stream_calls
        result = await controller.submit_draft(draft, session_id="session-1")
        if gateway.stream_calls == streams:
            return draft, result
    raise AssertionError("the custom budget was never crossed")


def _user_texts(store) -> list[str]:
    return [
        message.content
        for message in store.messages_for_session("session-1")
        if message.role is ConsoleMessageRole.USER
    ]


def _failed_rows(store) -> list[str]:
    return [
        message.id
        for message in store.messages_for_session("session-1")
        if message.status == "failed"
    ]


@pytest.mark.asyncio
async def test_an_ask_hold_pauses_the_send_before_it_is_committed(
    tmp_path: Path,
) -> None:
    _db, store, controller, gateway = _live_controller(tmp_path, overrides=_ASK)
    system_before = len(_system_rows(store))

    draft, result = await _send_until_held(controller, gateway)

    assert result.accepted is False
    assert result.preparation_id is not None
    held = store.preparation_for_session("session-1")
    assert held is not None
    assert held.preparation_id == result.preparation_id
    assert held.state is ConsoleTurnPreparationState.PAUSED
    assert held.pause_kind is ConsolePreparationPauseKind.CONTEXT_COMPACTION
    # Nothing Failed, no dispatch checkpoint, no dead-end System row.
    assert _failed_rows(store) == []
    assert store.dispatch_recovery_for_session("session-1") is None
    assert len(_system_rows(store)) == system_before
    assert gateway.auxiliary_calls == 0
    hold = controller.context_compaction_hold(result.preparation_id)
    assert hold is not None
    assert hold.used_tokens >= hold.trigger_tokens > 0
    assert hold.budget_tokens >= hold.trigger_tokens
    assert hold.estimated is False
    assert "compaction threshold" in result.visible_copy


@pytest.mark.asyncio
async def test_compact_and_send_summarizes_then_sends_the_held_message_once(
    tmp_path: Path,
) -> None:
    db, store, controller, gateway = _live_controller(tmp_path, overrides=_ASK)
    draft, held = await _send_until_held(controller, gateway)
    streams = gateway.stream_calls

    result = await controller.compact_and_send(held.preparation_id)

    assert result.accepted is True
    assert gateway.auxiliary_calls == 1
    assert _active_memory_count(db) == 1
    assert gateway.stream_calls == streams + 1
    assert _user_texts(store).count(draft) == 1
    assert _failed_rows(store) == []
    assert store.preparation_for_session("session-1") is None


@pytest.mark.asyncio
async def test_send_without_compacting_sends_with_no_summary_call(
    tmp_path: Path,
) -> None:
    db, store, controller, gateway = _live_controller(tmp_path, overrides=_ASK)
    draft, held = await _send_until_held(controller, gateway)
    streams = gateway.stream_calls

    result = await controller.send_without_compacting(held.preparation_id)

    assert result.accepted is True
    assert gateway.auxiliary_calls == 0
    assert _active_memory_count(db) == 0
    assert gateway.stream_calls == streams + 1
    assert _user_texts(store).count(draft) == 1
    assert _failed_rows(store) == []


@pytest.mark.asyncio
async def test_cancel_sends_nothing_and_returns_the_message_to_the_composer(
    tmp_path: Path,
) -> None:
    db, store, controller, gateway = _live_controller(tmp_path, overrides=_ASK)
    draft, held = await _send_until_held(controller, gateway)
    streams = gateway.stream_calls

    controller.cancel_library_preparation(held.preparation_id)

    assert gateway.stream_calls == streams
    assert gateway.auxiliary_calls == 0
    assert _active_memory_count(db) == 0
    assert draft not in _user_texts(store)
    assert store.session_draft("session-1") == draft
    assert store.preparation_for_session("session-1") is None
    assert controller.context_compaction_hold(held.preparation_id) is None


@pytest.mark.asyncio
async def test_a_failed_compaction_keeps_the_hold_and_says_why(
    tmp_path: Path,
) -> None:
    _db, store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary=""), overrides=_ASK
    )
    _draft, held = await _send_until_held(controller, gateway)
    streams = gateway.stream_calls

    result = await controller.compact_and_send(held.preparation_id)

    assert result.accepted is False
    assert "Compaction failed and nothing changed" in result.visible_copy
    assert gateway.stream_calls == streams
    still = store.preparation_for_session("session-1")
    assert still is not None
    assert still.pause_kind is ConsolePreparationPauseKind.CONTEXT_COMPACTION
    assert _failed_rows(store) == []



def _runtime_owned(tmp_path: Path, *, gateway: _LiveProviderGateway | None = None):
    """The production shape: ConsoleRuntime owns the controller and custody."""

    from types import SimpleNamespace

    from Tests.Chat.test_console_compaction_live_session import (
        _MODEL,
        _OPEN_DATABASES,
    )
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.prompt_history import PromptHistory
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "runtime-hold.sqlite", client_id="task34350")
    _OPEN_DATABASES.append(db)
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(session_id="session-1", title="Chat 1")
    store.set_session_context_policy_overrides(session.id, _ASK)
    live_gateway = gateway or _LiveProviderGateway()
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    app = SimpleNamespace(
        app_config={},
        _conversation_send_inflight={},
        local_chat_conversation_service=ChatConversationService(db),
    )
    runtime = ConsoleRuntime(app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    controller = runtime.ensure_chat_controller(
        store=store,
        provider_gateway=live_gateway,
        provider="openai",
        model=_MODEL,
    )
    controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
    return db, store, controller, runtime, live_gateway


async def _admit_and_wait(runtime, controller, draft: str):
    """Admit one turn exactly as the composer does, then await its task."""

    from uuid import uuid4

    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

    request = ConsoleTurnCustodyRequest(
        turn_id=str(uuid4()),
        session_id="session-1",
        draft=draft,
        configuration=controller.resolve_runtime_turn_configuration_snapshot(
            "session-1"
        ),
        staged_evidence_launch=runtime.snapshot_console_staged_evidence()[0],
    )
    return await runtime.wait_for_turn(runtime.accept_turn(request))


@pytest.mark.asyncio
async def test_a_runtime_held_send_is_not_also_a_turn_recovery(tmp_path: Path) -> None:
    """One held send, one surface: no "Unsent turn needs attention" twin."""

    _db, store, controller, runtime, gateway = _runtime_owned(tmp_path)
    held = None
    for index in range(12):
        streams = gateway.stream_calls
        result = await _admit_and_wait(
            runtime, controller, f"question-{index}: explain step {index} in detail."
        )
        if gateway.stream_calls == streams:
            held = result
            break
    assert held is not None, "the custom budget was never crossed"

    assert held.accepted is False
    assert controller.context_compaction_hold(held.preparation_id) is not None
    assert runtime.recoveries_for_session("session-1") == ()
    assert _failed_rows(store) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["compact_and_send", "send_without_compacting"])
async def test_a_runtime_held_send_resumes_and_replies(
    tmp_path: Path, action: str
) -> None:
    _db, store, controller, runtime, gateway = _runtime_owned(tmp_path)
    held = draft = None
    for index in range(12):
        draft = f"question-{index}: explain step {index} in detail."
        streams = gateway.stream_calls
        result = await _admit_and_wait(runtime, controller, draft)
        if gateway.stream_calls == streams:
            held = result
            break
    assert held is not None, "the custom budget was never crossed"
    streams = gateway.stream_calls

    result = await getattr(controller, action)(held.preparation_id)

    assert result.accepted is True
    assert gateway.stream_calls == streams + 1
    assert _user_texts(store).count(draft) == 1
    assert _failed_rows(store) == []
    assert runtime.recoveries_for_session("session-1") == ()


# --- The cannot-fit alert is decided before commit too ---------------------
# Live 2026-10-04: with Max tokens equal to the model's (estimated) window,
# the alert was right but arrived after the turn was accepted, so the screen
# also showed "Response accepted; waiting for dispatch." with Retry / Discard
# and blocked the composer -- the TASK-33621.4 panel, burying the alert.


@pytest.mark.asyncio
@pytest.mark.parametrize("first_message", [True, False])
async def test_a_send_that_cannot_fit_is_refused_before_commit_with_the_alert(
    tmp_path: Path, first_message: bool
) -> None:
    gateway = _LiveProviderGateway()
    db, store, controller, _gateway = _live_controller(
        tmp_path, gateway=gateway, overrides=_ASK
    )
    if not first_message:
        await controller.submit_draft("question-0: hello.", session_id="session-1")
    # 600-token window: Max tokens (120) plus the 512-token minimum margin
    # leave no input capacity; compacting cannot help.
    gateway.context_window = 600
    streams = gateway.stream_calls
    persisted_before = len(_user_texts(store))
    draft = "This message cannot fit."

    result = await controller.submit_draft(draft, session_id="session-1")

    assert result.accepted is False
    assert result.visible_copy.startswith("Your message was not sent:")
    assert "Max tokens" in result.visible_copy
    assert gateway.stream_calls == streams
    assert store.dispatch_recovery_for_session("session-1") is None
    assert store.preparation_for_session("session-1") is None
    persisted = [
        message.content
        for message in store.messages_for_session("session-1")
        if message.role is ConsoleMessageRole.USER and message.status != "failed"
    ]
    assert draft not in persisted
    assert len(persisted) == persisted_before


@pytest.mark.asyncio
async def test_a_runtime_send_that_cannot_fit_keeps_the_message_recoverable(
    tmp_path: Path,
) -> None:
    _db, store, controller, runtime, gateway = _runtime_owned(tmp_path)
    gateway.context_window = 600
    draft = "This message cannot fit."

    with pytest.raises(RuntimeError, match="refused before durable acceptance"):
        await _admit_and_wait(runtime, controller, draft)

    assert store.dispatch_recovery_for_session("session-1") is None
    recoveries = runtime.recoveries_for_session("session-1")
    assert [entry.draft for entry in recoveries] == [draft]
    assert gateway.stream_calls == 0


# --- Qodo review on PR #3003 ------------------------------------------------


@pytest.mark.asyncio
async def test_send_without_compacting_holds_even_if_the_policy_turned_automatic(
    tmp_path: Path,
) -> None:
    """Qodo #2: the answer is "do not compact this send", whatever the mode."""
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
    )

    db, store, controller, gateway = _live_controller(tmp_path, overrides=_ASK)
    draft, held = await _send_until_held(controller, gateway)
    store.set_session_context_policy_overrides(
        "session-1", replace(_ASK, compaction_mode=ContextCompactionMode.AUTOMATIC)
    )
    streams = gateway.stream_calls

    result = await controller.send_without_compacting(held.preparation_id)

    assert result.accepted is True
    assert gateway.auxiliary_calls == 0
    assert _active_memory_count(db) == 0
    assert gateway.stream_calls == streams + 1
    assert _user_texts(store).count(draft) == 1


@pytest.mark.asyncio
async def test_an_abandoned_resumed_hold_leaves_no_answered_id_behind(
    tmp_path: Path,
) -> None:
    """Qodo #6: an answered hold that never reaches the stream must not leak."""
    _db, store, controller, gateway = _live_controller(tmp_path, overrides=_ASK)
    _draft, held = await _send_until_held(controller, gateway)

    async def destination_changed(_selection):
        raise RuntimeError("provider unavailable")

    gateway.resolve_for_send = destination_changed  # the resumed send re-resolves
    await controller.send_without_compacting(held.preparation_id)
    paused = store.preparation_for_session("session-1")
    assert paused is not None
    controller.cancel_library_preparation(paused.preparation_id)

    assert held.preparation_id not in controller._compaction_hold_answered
    assert store.preparation_for_session("session-1") is None


class _RecordingGateway(_LiveProviderGateway):
    """Record the prepared request that reaches the provider."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sent: list = []

    async def stream_chat(self, resolution, prepared, **kwargs):
        self.sent.append(prepared)
        async for chunk in super().stream_chat(resolution, prepared, **kwargs):
            yield chunk


@pytest.mark.asyncio
async def test_send_without_compacting_past_the_ceiling_drops_older_turns_to_fit(
    tmp_path: Path,
) -> None:
    """Qodo #3: an over-ceiling chat sent uncompacted is windowed, not sent whole.

    "Send without compacting" means what compaction mode Off means for that
    one send: the real request preparation drops whole older turns until the
    request fits the input ceiling.
    """
    gateway = _RecordingGateway()
    _db, store, controller, _gateway = _live_controller(
        tmp_path, gateway=gateway, overrides=_ASK
    )
    draft, held = await _send_until_held(controller, gateway)
    assert controller.context_compaction_hold(held.preparation_id) is not None
    # Shrink the window so the chat is past the input ceiling (1,500 - Max
    # tokens 120 - 512 margin = 868) while the held message alone still fits.
    gateway.context_window = 1_500

    result = await controller.send_without_compacting(held.preparation_id)

    assert result.accepted is True
    prepared = gateway.sent[-1]
    assert prepared.known_overflow is False
    assert prepared.dropped_units, "older turns were not windowed out"
    assert prepared.accounting.total_input_tokens <= (
        prepared.capacity.effective_input_ceiling_tokens
    )
    assert draft in str(prepared.messages_payload)


def _context_hook_engine(tmp_path: Path, context_chars: int):
    """A real RunHooksEngine whose UserPromptSubmit hook adds model context."""
    import sys

    from Tests.Agents.hook_test_utils import trusted_hook_engine
    from tldw_chatbook.Agents.run_hooks import load_hooks_config

    config = load_hooks_config(
        {
            "hooks": {
                "hook": [
                    {
                        "event": "UserPromptSubmit",
                        "command": [
                            sys.executable,
                            "-c",
                            f"print('hook context ' * {context_chars // 13})",
                        ],
                    }
                ]
            }
        }
    )
    return trusted_hook_engine(
        config_provider=lambda: config, cwd_provider=lambda: str(tmp_path)
    )


@pytest.mark.asyncio
async def test_hook_context_that_cannot_fit_is_refused_before_commit(
    tmp_path: Path,
) -> None:
    """Qodo #1: the check also runs after hooks add model-visible context."""
    gateway = _RecordingGateway(context_window=1_200)
    _db, store, controller, _gateway = _live_controller(
        tmp_path, gateway=gateway, overrides=_ASK
    )
    await controller.submit_draft("question-0: hello.", session_id="session-1")
    # 1,200 - Max tokens 120 - 512 margin leaves 568 input tokens. History
    # and the draft fit (the early check passes); the hook's context, capped
    # at HOOK_IO_BUDGET_CHARS (~615 tokens), does not.
    controller._ensure_run_hooks = lambda: _context_hook_engine(tmp_path, 4_000)
    streams = gateway.stream_calls
    draft = "A short question the hook makes too large."

    result = await controller.submit_draft(draft, session_id="session-1")

    assert result.accepted is False
    assert result.visible_copy.startswith("Your message was not sent:")
    assert gateway.stream_calls == streams
    assert store.dispatch_recovery_for_session("session-1") is None
    assert store.preparation_for_session("session-1") is None
    assert draft not in [
        message.content
        for message in store.messages_for_session("session-1")
        if message.role is ConsoleMessageRole.USER and message.status != "failed"
    ]



@pytest.mark.asyncio
async def test_a_hold_after_hooks_resumes_without_repeating_its_side_effects(
    tmp_path: Path,
) -> None:
    """Qodo #1/#4: hook context that pushes a send past the Ask trigger holds
    it before commit. That rare late hold resumes without re-appending notes
    or retrieval events (skills and retrieval ran once already)."""
    gateway = _RecordingGateway(context_window=32_000)
    _db, store, controller, _gateway = _live_controller(
        tmp_path, gateway=gateway, overrides=_ASK
    )
    for index in range(2):
        await controller.submit_draft(f"question-{index}.", session_id="session-1")
    # 1,700-token window, measured with the real probe: without the hook the
    # trigger is 843 against 468 tokens of history (no hold); the hook's
    # context cuts the window-capped budget to 448, trigger 358 (Ask).
    gateway.context_window = 1_700
    controller._ensure_run_hooks = lambda: _context_hook_engine(tmp_path, 4_000)
    streams = gateway.stream_calls
    draft = "question-2: one more, please."

    held = await controller.submit_draft(draft, session_id="session-1")

    assert held.accepted is False, held.visible_copy
    assert controller.context_compaction_hold(held.preparation_id) is not None
    assert held.preparation_id in controller._compaction_hold_after_effects
    assert store.dispatch_recovery_for_session("session-1") is None
    system_rows_at_hold = _system_rows(store)

    result = await controller.send_without_compacting(held.preparation_id)

    assert result.accepted is True
    assert gateway.stream_calls == streams + 1
    assert _user_texts(store).count(draft) == 1
    new_rows = _system_rows(store)[len(system_rows_at_hold):]
    assert len(new_rows) == len(set(new_rows)), new_rows
    assert held.preparation_id not in controller._compaction_hold_after_effects
