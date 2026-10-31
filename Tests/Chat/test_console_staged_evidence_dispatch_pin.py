"""TASK-34352: staged evidence is pinned when a Console turn is dispatched.

Every production send is admitted through ``ConsoleRuntime.accept_turn``,
which snapshots the runtime's staged launch into the custody request and
hands the controller explicit capture and release hooks. These tests drive
that path with the runtime-owned controller in a fresh real profile and pin
two facts:

* evidence staged at dispatch is the evidence the turn uses, and evidence
  staged afterwards stays staged for the next message;
* a turn dispatched with *nothing* staged that pauses (here, a temporary
  chat's Capture-On card) and is then resumed does not pick up evidence the
  user staged for their next message while the card was open. Before the
  fix the resumed turn took the live capture route and consumed it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from Tests.Chat.test_console_first_send_atomicity import _CheckpointObservingGateway
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
from tldw_chatbook.Chat.console_chat_controller import ConsoleSessionSettings
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

EARLIER = "Evidence staged before the send: ORCHID-7731."
LATER = "Evidence staged for the NEXT message: HELIX-0042."


class _DurableCaptureGateway(_CheckpointObservingGateway):
    """The production gateway reports durable capture when storage is wired."""

    supports_durable_capture = True


def _runtime(tmp_path: Path, *, ephemeral: bool):
    db = CharactersRAGDB(tmp_path / "controller.sqlite", client_id="task-34352")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(
        session_id="session-1",
        title="Chat 1",
        ephemeral=ephemeral,
        settings=ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
    )
    app = SimpleNamespace(app_config={}, _conversation_send_inflight={})
    runtime = ConsoleRuntime(app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    controller = runtime.ensure_chat_controller(
        store=store,
        provider_gateway=_DurableCaptureGateway(db),
        provider="llama_cpp",
        model="test-model",
    )
    controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
    return runtime, store, controller, session.id


def _record_provider_entries(controller, monkeypatch) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    async def stream(**kwargs: Any) -> ConsoleSubmitResult:
        hook = kwargs.get("before_provider_dispatch")
        if callable(hook):
            await hook()
        entries.append(kwargs)
        return ConsoleSubmitResult(True, True)

    monkeypatch.setattr(controller, "_stream_assistant_response", stream)
    return entries


def _library_search(monkeypatch) -> list[Any]:
    """Answer the Library search boundary with each launch's own evidence."""
    from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events

    searched: list[Any] = []

    async def search(_app: Any, staged: Any, *, user_message: str) -> Any:
        searched.append(staged)
        if staged is None:
            return SimpleNamespace(context=None)
        return SimpleNamespace(
            context=staged.evidence,
            citation_repair_contract=SimpleNamespace(allowed_ordinals=(1,)),
        )

    monkeypatch.setattr(
        chat_rag_events, "capture_console_staged_evidence_for_chat", search
    )
    return searched


def _admit(runtime, controller, session_id: str, draft: str) -> str:
    """Admit one turn exactly as the composer does (``wiring.py``)."""
    request = ConsoleTurnCustodyRequest(
        turn_id=str(uuid4()),
        session_id=session_id,
        draft=draft,
        configuration=controller.resolve_runtime_turn_configuration_snapshot(
            session_id
        ),
        staged_evidence_launch=runtime.snapshot_console_staged_evidence()[0],
    )
    return runtime.accept_turn(request)


def _sent_text(entry: dict[str, Any]) -> str:
    return "\n".join(
        str(row.get("content", "")) for row in entry["provider_messages"]
    )


@pytest.mark.asyncio
@private_profile_test
async def test_a_composer_send_uses_the_evidence_staged_at_dispatch(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _store, controller, session_id = _runtime(tmp_path, ephemeral=False)
    entries = _record_provider_entries(controller, monkeypatch)
    _library_search(monkeypatch)
    earlier = SimpleNamespace(evidence=EARLIER)
    later = SimpleNamespace(evidence=LATER)
    runtime.stage_console_staged_evidence(earlier)

    turn_id = _admit(runtime, controller, session_id, "what does it say?")
    # The user stages evidence for their next message before this turn runs.
    runtime.stage_console_staged_evidence(later)
    await runtime.wait_for_turn(turn_id)

    assert len(entries) == 1
    sent = _sent_text(entries[0])
    assert EARLIER in sent
    assert LATER not in sent
    assert runtime.snapshot_console_staged_evidence()[0] is later


@pytest.mark.asyncio
@private_profile_test
async def test_a_resumed_turn_does_not_take_evidence_staged_while_it_was_paused(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, store, controller, session_id = _runtime(tmp_path, ephemeral=True)
    entries = _record_provider_entries(controller, monkeypatch)
    searched = _library_search(monkeypatch)

    # Nothing is staged when the user sends. A temporary chat with Capture on
    # (the shipped default) pauses on the Save / Send without capture card.
    # A paused send is not durably accepted, so the custody task reports it.
    with pytest.raises(RuntimeError, match="refused before durable acceptance"):
        await runtime.wait_for_turn(
            _admit(runtime, controller, session_id, "summarize our chat")
        )
    paused = store.preparation_for_session(session_id)
    assert paused is not None
    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.TEMPORARY_CAPTURE
    assert entries == []

    # While the card is open, the user stages evidence for their NEXT message.
    later = SimpleNamespace(evidence=LATER)
    runtime.stage_console_staged_evidence(later)

    result = await controller.send_without_capture(paused.preparation_id)

    assert result.accepted is True
    assert len(entries) == 1
    assert LATER not in _sent_text(entries[0])
    assert later not in searched
    assert runtime.snapshot_console_staged_evidence()[0] is later
