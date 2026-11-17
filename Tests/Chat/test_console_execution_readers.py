"""Stable presentation readers observe current values without owning a store."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsolePendingDecisionProjection
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_control_dispatch import ConsoleProjection
from tldw_chatbook.Chat.console_execution_process import ConsoleExecutionClient
from tldw_chatbook.Chat.console_execution_projection_values import (
    decode_session_projection,
    encode_session_projection,
)
from tldw_chatbook.Chat.console_execution_projection_view import (
    ConsoleProjectionUnavailable,
    ConsoleProjectionView,
)


def _publication(session, *, revision=0, messages=(), decision=None, state=None):
    row = decode_session_projection(encode_session_projection(session))
    frame = ConsoleProjection(
        0,
        row.id,
        row.incarnation_id,
        revision,
        messages,
        state or ConsoleRunState(ConsoleRunStatus.IDLE, "Ready"),
        decision,
    )
    return ConsoleProjectionView(
        0,
        (row,),
        row.id,
        ((row.id, (len(messages), 0)),),
        (frame,),
    )


def _reader_client(publication):
    # An inert holder exercises value reads only; no process/profile or native
    # authority is created or claimed by these tests.
    holder = SimpleNamespace(console_execution_client=None)
    client = ConsoleExecutionClient(holder)
    holder.console_execution_client = client
    client.projection_view = publication
    return client


def test_held_store_and_session_follow_actual_replaced_publications():
    session = ConsoleChatSession(id="session", draft="before")
    client = _reader_client(_publication(session))
    controller, store = client.controller_reader, client.store_reader
    held = store.session(session.id)
    assert held is store.sessions()[0]
    assert held.persisted_conversation_id is None

    session.persisted_conversation_id = "persisted-conversation"
    session.draft = "after"
    message = ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "saved answer")
    client.projection_view = _publication(session, revision=2, messages=(message,))
    assert controller.store is store
    assert store.session(session.id) is held
    assert held.persisted_conversation_id == "persisted-conversation"
    assert held.draft == "after"
    assert store.display_projection_revision(held.id) == 2
    copied = store.messages_for_session(held.id)[0]
    assert copied is not message and copied.content == message.content
    copied.content = "local rendering edit"
    assert store.messages_for_session(held.id)[0].content == "saved answer"

    original = client.projection_view
    replacement = replace(original.session_rows[0], incarnation_id="replacement")
    client.projection_view = replace(
        original,
        session_rows=(replacement,),
        projections=(replace(original.projections[0], incarnation_id="replacement"),),
    )
    assert store.sessions()[0] is not held
    with pytest.raises(ConsoleProjectionUnavailable):
        _ = held.persisted_conversation_id
    assert len(store._sessions) == 1


def test_reader_requires_current_source_and_original_client_holder():
    session = ConsoleChatSession(id="session")
    client = _reader_client(_publication(session))
    held = client.store_reader.session(session.id)
    client._source_revision += 1
    with pytest.raises(ConsoleProjectionUnavailable, match="publication"):
        _ = held.id
    client._source_revision -= 1
    client._app.console_execution_client = None
    with pytest.raises(ConsoleProjectionUnavailable, match="owner"):
        _ = held.id


def test_approval_observer_rejects_superseded_head_and_preserves_run_status():
    session = ConsoleChatSession(id="session")
    payload = {
        "round_id": "round",
        "calls": [{"call_id": "call", "tool_name": "fs_read"}],
    }
    head = ConsolePendingDecisionProjection(
        "approval", session.id, "round", None, payload
    )
    state = ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming")
    client = _reader_client(_publication(session, decision=head, state=state))
    reader = client.controller_reader
    observed = reader._parked_approval_payloads
    assert reader._head_round_payload(observed, session.id) is payload
    assert reader._head_round_payload(observed, "other") is None
    assert reader.run_state is state
    client.projection_view = _publication(session)
    assert reader.run_state.status is ConsoleRunStatus.IDLE
    assert reader._parked_approval_payloads is None
    with pytest.raises(ConsoleProjectionUnavailable, match="approval"):
        reader._head_round_payload(observed, session.id)


def test_history_reader_refuses_replacement_before_scheduling(monkeypatch):
    client = _reader_client(_publication(ConsoleChatSession(id="session")))
    called = []
    monkeypatch.setattr(
        client, "read_run_history", lambda request: called.append(request)
    )
    with pytest.raises(ConsoleProjectionUnavailable, match="history reader"):
        client.controller_reader._agent_bridge._db.list_runs("conversation")
    assert called == []
