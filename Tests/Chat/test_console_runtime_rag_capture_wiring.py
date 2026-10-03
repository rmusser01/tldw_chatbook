"""TASK-33940.4: runtime-owned sends must not call the live RAG capture seam.

ConsoleRuntime wires the controller with ``rag_capture_provider`` set to its
three-argument ``_capture_frozen_console_staged_rag(draft, turn_context,
launch)``. Every custodied send without staged evidence fell through to the
LIVE path, which calls ``provider(draft, turn_context)``; the TypeError was
swallowed and logged as "Console RAG capture unavailable;
reason=capture_provider_failure", with a false "Retrieval failed" trace pair,
on every send (review finding G2-37). A custodied turn already froze its
evidence decision at admission, so it must take the frozen route instead.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from Tests.Chat.test_console_first_send_atomicity import _CheckpointObservingGateway
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

EVIDENCE = "Staged evidence: ORCHID-7731 is the marker."


def _runtime_controller(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "controller.sqlite", client_id="task-33940-4")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    store.create_session(session_id="session-1", title="Chat 1")
    runtime = ConsoleRuntime(type("App", (), {"app_config": {}})())
    controller = runtime.ensure_chat_controller(
        store=store,
        provider_gateway=_CheckpointObservingGateway(db),
        provider="llama_cpp",
        model="test-model",
    )
    controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
    return runtime, store, controller


def _record_provider_entries(controller, monkeypatch) -> list[dict[str, Any]]:
    """Record each provider dispatch; not every route has a pre-dispatch hook."""
    entries: list[dict[str, Any]] = []

    async def stream(**kwargs: Any) -> ConsoleSubmitResult:
        hook = kwargs.get("before_provider_dispatch")
        if callable(hook):
            await hook()
        entries.append(kwargs)
        return ConsoleSubmitResult(True, True)

    monkeypatch.setattr(controller, "_stream_assistant_response", stream)
    return entries


def _configuration() -> ConsoleTurnConfigurationSnapshot:
    return ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-1",
        provider_selection=ConsoleProviderSelection(provider="llama_cpp"),
    )


def _warnings(records: list[str]):
    sink = controller_module.logger.add(
        lambda message: records.append(message.record["message"]), level="WARNING"
    )
    return sink


@pytest.mark.asyncio
@private_profile_test
async def test_runtime_send_without_staged_evidence_never_calls_live_capture(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _store, controller = _runtime_controller(tmp_path)
    entries = _record_provider_entries(controller, monkeypatch)
    records: list[str] = []
    sink = _warnings(records)
    try:
        result = await controller.submit_draft(
            "hello",
            session_id="session-1",
            configuration=_configuration(),
            staged_evidence_launch=None,
            staged_evidence_capture=runtime._capture_frozen_console_staged_rag,
            staged_evidence_release=lambda *_args: None,
        )
    finally:
        controller_module.logger.remove(sink)

    assert result.accepted is True
    assert len(entries) == 1
    assert not any("RAG capture unavailable" in line for line in records), records


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("draft", ["what does it say?", "/explain the staged note"])
async def test_runtime_send_with_staged_evidence_attaches_and_releases_it(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    draft: str,
) -> None:
    _runtime, _store, controller = _runtime_controller(tmp_path)
    entries = _record_provider_entries(controller, monkeypatch)
    launch = object()
    captured: list[Any] = []
    released: list[tuple[Any, Any]] = []

    async def capture(draft_text: str, _turn_context: Any, staged: Any) -> Any:
        captured.append(staged)
        return SimpleNamespace(context=EVIDENCE)

    result = await controller.submit_draft(
        draft,
        session_id="session-1",
        configuration=_configuration(),
        staged_evidence_launch=launch,
        staged_evidence_capture=capture,
        staged_evidence_release=lambda staged, outcome: released.append(
            (staged, outcome)
        ),
    )

    assert result.accepted is True
    assert captured == [launch]
    assert len(entries) == 1
    sent = "\n".join(
        str(row.get("content", "")) for row in entries[0]["provider_messages"]
    )
    assert EVIDENCE in sent
    assert [staged for staged, _outcome in released] == [launch]
