"""A named custom endpoint's traced request records what was sent (TASK-37762).

A named OpenAI-compatible custom endpoint (``custom-ep:<slug>``) runs through
the strict hosted engine under the ``custom-hosted`` execution key. With
trace capture on (the default), every such call used to record its traced
request surface as one ``provider_overlay`` omission with reason
``alignment_mismatch`` instead of the saved user revision a Custom slot call
records, so the trace never held what was sent. The same gap blocked the
one-time retry without streaming (TASK-37421): the retry's own trace call
could not replace that omission, and the run ended "Blocked -- capture
failed".

These tests send through the real Console controller, agent bridge, gateway,
``chat_api_call`` and adapters (the hosted engine for ``custom-ep:box``, the
legacy handler for the Custom slot), with the real
``ConsoleTraceBoundaryFactory`` writing the trace into a real SQLite
database. Only the server is local: a loopback OpenAI-compatible endpoint
that can refuse ``"stream": true`` or stream nothing.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from Tests.Chat.test_console_project_instruction_traces import _project_console
from Tests.Chat.test_console_refused_stream_retry import (
    _REPLY,
    _USAGE,
    _resolve,
    _server,
)

#: The adapters read provider settings through the guarded config loader;
#: bootstrap_profile keeps the collection-time sandbox profile (still
#: isolated from the user's).
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.loopback_network]

_PROVIDERS = ("custom", "custom-ep:box")


def _surface_chain(cursor, node_id: str | None) -> list[tuple[object, ...]]:
    """Walk one call's surface from its head node back to the first node."""
    rows: list[tuple[object, ...]] = []
    seen: set[str] = set()
    while node_id is not None and node_id not in seen:
        seen.add(node_id)
        row = cursor.execute(
            "SELECT n.predecessor_node_id, n.component_kind, n.reference_kind,"
            " n.omission_reason_code, r.source_message_id, r.normalized_role"
            " FROM console_trace_surface_nodes AS n"
            " LEFT JOIN console_trace_semantic_revisions AS r"
            " ON r.revision_id = n.semantic_revision_id"
            " WHERE n.node_id = ?",
            (node_id,),
        ).fetchone()
        assert row is not None, node_id
        rows.append(tuple(row[1:]))
        node_id = row[0]
    return rows


def _traced_calls(app) -> list[dict[str, object]]:
    """Each trace call of the run with its surface, in call order."""
    with app.db.transaction() as cursor:
        calls = cursor.execute(
            "SELECT call_sequence, state, route_identity, integrity_state,"
            " omission_reason_code, surface_node_id, usage_json"
            " FROM console_trace_calls ORDER BY call_sequence"
        ).fetchall()
        return [
            {
                "sequence": row[0],
                "state": row[1],
                "route": row[2],
                "integrity": row[3],
                "omission": row[4],
                "surface": _surface_chain(cursor, row[5]),
                "usage": json.loads(row[6]) if row[6] else None,
            }
            for row in calls
        ]


def _assert_surface_holds_saved_user_turn(call: dict, user_message_id: str) -> None:
    """The traced request is the saved user revision, not an omission."""
    surface = call["surface"]
    assert surface, call
    assert all(node[1] != "omission" for node in surface), surface
    assert ("message", "revision", None, user_message_id, "user") in surface, surface
    assert call["integrity"] == "complete", call
    assert call["omission"] is None, call


async def _send(app, monkeypatch, provider: str, base_url: str, *, streaming: bool):
    helper, selected = await asyncio.to_thread(
        _resolve, provider, base_url, streaming=streaming
    )
    await helper.aclose()

    async def resolve(_selection):
        return selected

    monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
    result = await app.controller.submit_draft("Hello", session_id=app.session.id)
    assert result.accepted
    user = next(
        row
        for row in app.store.messages_for_session(app.session.id)
        if row.content == "Hello"
    )
    return selected, app.store.get_message(result.assistant_message_id), user


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", _PROVIDERS)
async def test_custom_endpoint_traced_request_records_the_saved_user_revision(
    tmp_path, monkeypatch, provider
):
    """AC#1/#3: a custom-ep call's traced surface is what a Custom call's is."""
    with _server() as server:
        async with _project_console(tmp_path, monkeypatch) as app:
            selected, assistant, user = await _send(
                app, monkeypatch, provider, server.base_url, streaming=False
            )
            assert selected.execution_key == (
                "custom-hosted" if provider == "custom-ep:box" else "custom-openai-api"
            )
            assert assistant.content == _REPLY, app.reservation_errors
            assert server.stream_flags == [False]
            calls = _traced_calls(app)

    assert app.reservation_errors == []
    assert [(call["state"], call["route"]) for call in calls] == [
        ("complete", "agent_first")
    ]
    _assert_surface_holds_saved_user_turn(calls[0], user.persisted_message_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("server_mode", ["refused", "empty"])
async def test_refused_or_empty_stream_retry_is_traced_with_capture_on(
    tmp_path, monkeypatch, provider, server_mode
):
    """AC#2: one user turn, one reply, two traced calls on the saved turn.

    The refused (or empty) streamed call settles as an error with no usage;
    the retry without streaming is its own traced call on the same saved
    user revision and carries the reply's usage once.
    """
    with _server(empty_stream=server_mode == "empty") as server:
        async with _project_console(tmp_path, monkeypatch) as app:
            _selected, assistant, user = await _send(
                app, monkeypatch, provider, server.base_url, streaming=True
            )
            assert assistant.status == "complete", (
                assistant.content,
                app.reservation_errors,
                server.stream_flags,
            )
            assert assistant.content == _REPLY
            assert server.stream_flags == [True, False]
            rows = app.store.messages_for_session(app.session.id)
            calls = _traced_calls(app)

    assert app.reservation_errors == []
    # One user turn and one persisted reply: no System failure row.
    assert [row.content for row in rows] == ["Hello", _REPLY]
    assert [(call["state"], call["route"]) for call in calls] == [
        ("error", "agent_first"),
        # A Console send is an agent run: its retry keeps the run's tool
        # chain (the fallback reservation projects onto tool_loop).
        ("complete", "tool_loop"),
    ]
    assert calls[0]["usage"] is None
    assert calls[1]["usage"] == _USAGE
    for call in calls:
        _assert_surface_holds_saved_user_turn(call, user.persisted_message_id)
