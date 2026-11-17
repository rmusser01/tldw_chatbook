"""TASK-41310: Capture On around a native ``spawn_subagent`` call.

The real ``ConsoleChatController``, ``ConsoleChatStore``, agent bridge,
provider gateway and ``ConsoleTraceBoundaryFactory`` run over real SQLite.
Only the HTTP socket is a double. It answers in the OpenAI streaming shape:
the parent's first reply is one native ``spawn_subagent`` call
(``delta.tool_calls``, ``finish_reason: tool_calls``).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.owned_http_client,
    pytest.mark.asyncio,
]

MODEL = "synthetic-qwen"
ENDPOINT = {
    "display_name": "Spawn trace fixture",
    "family": "openai_compatible",
    "base_url": "http://127.0.0.1:8010/v1",
    "models": [MODEL],
}
DRAFT = "Delegate a task, then finish."


def _sse(delta: dict, finish: str) -> bytes:
    chunk = {"id": "s", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
    events = [
        {**chunk, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            **chunk,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return (body + "data: [DONE]\n\n").encode()


async def _spawn_turn(
    tmp_path,
    monkeypatch,
    task: str,
    *,
    child_tool=False,
    tamper_parent=False,
) -> SimpleNamespace:
    """Send one Capture-On turn whose model spawns a child; return the facts."""
    import httpx
    import urllib3

    from Tests import network_guard
    from tldw_chatbook import config
    from tldw_chatbook.Agents.run_context import current_run_actor
    from tldw_chatbook.Chat import console_chat_controller, console_send_diagnostics
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
    from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    facts = SimpleNamespace(wire=[], refusals=[], boundary_errors=[])
    # The child re-resolves the inherited custom endpoint from app config,
    # which conftest redirects into the test root; refuse outside it.
    assert config.get_cli_config_path().is_relative_to(
        Path(os.environ["TLDW_TEST_CONFIG_ROOT"])
    )
    old_endpoint = config.get_cli_setting("custom_endpoints", "spawntrace", {})
    config.save_setting_to_cli_config("custom_endpoints", "spawntrace", ENDPOINT)
    original_preparation = console_chat_controller.ConsoleTurnPreparation

    def capture_on(**kwargs):
        kwargs["capture_mode"] = ConsoleTraceCaptureMode.CAPTURE_ON
        return original_preparation(**kwargs)

    monkeypatch.setattr(console_chat_controller, "ConsoleTurnPreparation", capture_on)
    record = console_send_diagnostics.record_send_stage

    def recording(phase, status="entered", *, error=None, **fields):
        if "surface_refusal_kind" in fields:
            facts.refusals.append(fields["surface_refusal_kind"])
        return record(phase, status, error=error, **fields)

    monkeypatch.setattr(console_send_diagnostics, "record_send_stage", recording)
    original_boundary = ConsoleTraceBoundaryFactory.__call__

    def boundary(factory, *args, **kwargs):
        actor = current_run_actor()
        try:
            return original_boundary(factory, *args, **kwargs)
        except Exception as exc:
            facts.boundary_errors.append(
                (actor and actor.kind, type(exc).__name__, str(exc))
            )
            raise

    if tamper_parent:
        from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

        original_delta = ConsoleTraceService.prepare_current_surface_delta

        def changed_parent_history(service, cursor, **kwargs):
            # The parent's own follow-up arrives with its first user row
            # changed; its root surface must refuse it as before.
            if kwargs["route_identity"] == "tool_loop" and (
                current_run_actor().kind == "primary"
            ):
                values = list(kwargs["values"])
                index = next(
                    i
                    for i, value in enumerate(values)
                    if hasattr(value, "get") and value.get("role") == "user"
                )
                values[index] = {**values[index], "content": "a different question"}
                kwargs["values"] = tuple(values)
            return original_delta(service, cursor, **kwargs)

        monkeypatch.setattr(
            ConsoleTraceService, "prepare_current_surface_delta", changed_parent_history
        )
    monkeypatch.setattr(ConsoleTraceBoundaryFactory, "__call__", boundary)

    async def metadata(transport, request):
        assert request.method == "GET" and request.url.path == "/v1/models"
        return httpx.Response(
            200, json={"data": [{"id": MODEL, "context_length": 32768}]}
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", metadata)

    def physical(pool, connection, method, url, *args, **kwargs):
        assert (pool.host, pool.port, url) == (
            "127.0.0.1",
            8010,
            "/v1/chat/completions",
        )
        request = json.loads(kwargs.get("body", args[0] if args else None))
        kind = getattr(current_run_actor(), "kind", None)
        roles = [message["role"] for message in request["messages"]]
        facts.wire.append((kind, roles))
        if kind == "primary" and "tool" not in roles:
            call = {
                "index": 0,
                "id": "call_spawn_1",
                "type": "function",
                "function": {
                    "name": "spawn_subagent",
                    "arguments": json.dumps({"task": task}),
                },
            }
            body = _sse({"tool_calls": [call]}, "tool_calls")
        elif child_tool and kind != "primary" and len(roles) == 2:
            # A fenced call: the child's own tool round without a provider
            # continuation (native child rounds on this endpoint fail before
            # any trace, Capture Off too; see the task notes).
            from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN

            call = json.dumps({"name": "get_current_datetime", "arguments": {}})
            body = _sse({"content": f"{FENCE_OPEN}\n{call}\n```"}, "stop")
        else:
            text = "parent final" if kind == "primary" else "child done"
            body = _sse({"content": text}, "stop")
        return urllib3.response.HTTPResponse(
            body=io.BytesIO(body),
            status=200,
            headers={"Content-Type": "text/event-stream"},
            preload_content=False,
        )

    def denied(*args, **kwargs):
        raise AssertionError("network prohibited")

    chat = CharactersRAGDB(str(tmp_path / "chat.db"), client_id="spawntrace")
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"custom_endpoints": {"spawntrace": ENDPOINT}},
        environ={},
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(chat),
    )
    store = ConsoleChatStore(persistence=ChatPersistenceService(chat))
    runs = AgentRunsDB(tmp_path / "runs.db", client_id="spawntrace")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="custom-ep:spawntrace",
        model=MODEL,
        agent_bridge=ConsoleAgentBridge(
            agent_runs_db=runs, store=store, provider_gateway=gateway
        ),
        agent_runtime_enabled=True,
    )
    session = store.ensure_session()
    assert not network_guard.is_allowed()
    try:
        with (
            patch.object(socket, "getaddrinfo", denied),
            patch.object(
                urllib3.connectionpool.HTTPConnectionPool, "_make_request", physical
            ),
        ):
            facts.result = await controller.submit_draft(DRAFT, session_id=session.id)
            # The child is a fleet child: wait until every run has settled.
            for _ in range(300):
                facts.runs = sorted(
                    (row["agent_kind"], row["status"])
                    for row in runs.list_runs(session.persisted_conversation_id)
                )
                if any(kind == "subagent" for kind, _ in facts.runs) and all(
                    status != "running" for _, status in facts.runs
                ):
                    break
                await asyncio.sleep(0.05)
        facts.calls = [
            tuple(row)
            for row in chat.get_connection().execute(
                "SELECT route_identity, state, omission_reason_code "
                "FROM console_trace_calls ORDER BY created_at, call_sequence"
            )
        ]
        # (on the owner's root segment, owner, turn, run, route, state)
        facts.owned_calls = [
            tuple(row)
            for row in chat.get_connection().execute(
                "SELECT call.segment_id = owner.root_segment_id, call.owner_id,"
                " call.turn_id, call.run_id, call.route_identity, call.state"
                " FROM console_trace_calls AS call JOIN console_trace_owners AS owner"
                " ON owner.owner_id = call.owner_id"
                " ORDER BY call.created_at, call.call_sequence"
            )
        ]
        facts.links = [
            tuple(row)
            for row in chat.get_connection().execute(
                "SELECT segment_id, owner_id, turn_id, run_id"
                " FROM console_trace_child_segments"
            )
        ]
        facts.omitted_nodes = (
            chat.get_connection()
            .execute(
                "SELECT COUNT(*) FROM console_trace_surface_nodes "
                "WHERE reference_kind = 'omission'"
            )
            .fetchone()[0]
        )
        return facts
    finally:
        await controller.shutdown()
        config.save_setting_to_cli_config(
            "custom_endpoints", "spawntrace", old_endpoint
        )
        runs.close()
        await gateway.aclose()
        chat.close_connection()


async def test_parent_follow_up_after_native_spawn_keeps_capture(tmp_path, monkeypatch):
    """The parent's request after the spawn result is admitted and captured."""
    facts = await _spawn_turn(tmp_path, monkeypatch, "Say child done.")
    why = str(vars(facts))
    assert facts.result.accepted and facts.result.provider_started, why
    primary = [roles for kind, roles in facts.wire if kind == "primary"]
    assert primary == [["system", "user"], ["system", "user", "assistant", "tool"]], why
    assert facts.refusals == [], why
    assert facts.omitted_nodes == 0, why
    assert ("tool_loop", "complete", None) in facts.calls, why
    assert all(call[1:] == ("complete", None) for call in facts.calls), why


@pytest.mark.parametrize(
    ("task", "child_tool"),
    [("Say child done.", False), (DRAFT, False), ("Say child done.", True)],
    ids=["own-task", "task-equals-user-message", "child-tool-loop"],
)
async def test_child_request_after_native_spawn_reaches_provider(
    tmp_path, monkeypatch, task, child_tool
):
    """Capture On traces a spawned child in its own segment of the parent's turn."""
    facts = await _spawn_turn(tmp_path, monkeypatch, task, child_tool=child_tool)
    why = str(vars(facts))
    child_wire = [roles for kind, roles in facts.wire if kind == "subagent"]
    assert child_wire and child_wire[0][-1] == "user", why
    assert len(child_wire) == (2 if child_tool else 1), why
    assert ("subagent", "done") in facts.runs, why
    assert facts.boundary_errors == [] and facts.refusals == [], why
    assert facts.omitted_nodes == 0, why
    parent = [call for call in facts.owned_calls if call[0]]
    child = [call for call in facts.owned_calls if not call[0]]
    assert [call[4:] for call in parent] == [
        ("agent_first", "complete"),
        ("tool_loop", "complete"),
    ], why
    assert [call[4:] for call in child] == [("agent_first", "complete")] + (
        [("tool_loop", "complete")] if child_tool else []
    ), why
    # Same owner and turn as the parent; its own run, in its own linked segment.
    (owner_id, turn_id, parent_run) = parent[0][1:4]
    assert {call[1:3] for call in child} == {(owner_id, turn_id)}, why
    assert {call[3] for call in child} != {parent_run}, why
    assert [link[1:] for link in facts.links] == [(owner_id, turn_id, child[0][3])], why


async def test_parent_surface_still_refuses_a_changed_history_beside_a_child(
    tmp_path, monkeypatch
):
    """A child's segment never widens what the parent's own surface admits."""
    facts = await _spawn_turn(
        tmp_path, monkeypatch, "Say child done.", tamper_parent=True
    )
    why = str(vars(facts))
    assert [kind for kind, _ in facts.wire].count("primary") == 1, why
    assert [error[0] for error in facts.boundary_errors] == ["primary"], why
    # The child still went out, traced in its own segment (its settlement
    # waits on the parent's turn, which this refusal ended).
    assert [call[4] for call in facts.owned_calls if not call[0]] == ["agent_first"], (
        why
    )
    assert len(facts.links) == 1, why
    assert ("subagent", "done") in facts.runs, why
