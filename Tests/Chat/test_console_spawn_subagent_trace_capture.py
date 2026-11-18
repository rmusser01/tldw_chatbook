"""TASK-41310: Capture On around a native ``spawn_subagent`` call.

TASK-41651 adds the child's own native tool round (Capture On and Off) and
TASK-41652 the chat's prompt mix-ins.

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
MIXIN = "MIXIN-BODY: answer as a pirate."


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


async def _enable_mixin(tmp_path, monkeypatch, store, controller, session):
    """Enable one Before mix-in on the chat (as the prompt mix-in journeys do)."""
    from tldw_chatbook.Chat.console_prompt_mixin_service import (
        ConsolePromptMixinService,
    )
    from tldw_chatbook.Chat.console_prompt_mixins import MixinSourceRef
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Prompt_Management.prompt_scope_service import (
        LocalPromptService,
        PromptScopeService,
    )

    prompts = PromptsDatabase(tmp_path / "prompts.sqlite", client_id="spawntrace")
    scope = PromptScopeService(
        local_service=LocalPromptService(prompts), server_service=None
    )
    service = ConsolePromptMixinService(
        store=store, prompt_scope_service=scope, persistence=store.persistence
    )
    monkeypatch.setattr(controller, "prompt_mixin_service", lambda: service)
    record = await scope.save_prompt(
        mode="local",
        name="Pirate",
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition={
            "kind": "text_mixin",
            "schema_version": 3,
            "body": MIXIN,
            "placement": "before",
        },
    )
    await service.enable_reference(
        session.id,
        source=MixinSourceRef(
            scope.mixin_source_identity(mode="local")[0], "local", record["uuid"]
        ),
        target_incarnation_id=store._sessions[session.id].incarnation_id,
        expected_revision=0,
    )
    return prompts


async def _spawn_turn(
    tmp_path,
    monkeypatch,
    task: str,
    *,
    child_tool=False,
    tamper_parent=False,
    tasks=None,
    tamper_child_route=False,
    child_native=False,
    capture="capture_on",
    mixin=False,
    fenced_spawn=False,
    parent_openai=False,
    trace_view=False,
) -> SimpleNamespace:
    """Send one turn (Capture On unless ``capture`` says otherwise) whose model
    spawns a child; return the facts."""
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

    facts = SimpleNamespace(wire=[], refusals=[], boundary_errors=[], tools=[])
    facts.applied, facts.persisted, facts.sent = [], [], []
    # The child re-resolves the inherited custom endpoint from app config,
    # which conftest redirects into the test root; refuse outside it.
    assert config.get_cli_config_path().is_relative_to(
        Path(os.environ["TLDW_TEST_CONFIG_ROOT"])
    )
    old_endpoint = config.get_cli_setting("custom_endpoints", "spawntrace", {})
    config.save_setting_to_cli_config("custom_endpoints", "spawntrace", ENDPOINT)
    original_preparation = console_chat_controller.ConsoleTurnPreparation

    def capture_on(**kwargs):
        kwargs["capture_mode"] = ConsoleTraceCaptureMode(capture)
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
    if tamper_child_route == "missing":
        from tldw_chatbook.Chat.console_agent_bridge import (
            ConsoleAgentTraceRequestFactory,
        )

        original_build = ConsoleAgentTraceRequestFactory.build

        def parent_shaped(factory, *args, **kwargs):
            # A child request built as if it were the parent's: no turn named.
            return original_build(factory, *args, **{**kwargs, "child": False})

        monkeypatch.setattr(ConsoleAgentTraceRequestFactory, "build", parent_shaped)
    elif tamper_child_route:
        from tldw_chatbook.Chat.console_agent_bridge import (
            ConsoleAgentTraceRequestFactory,
        )
        from tldw_chatbook.Chat.console_trace_models import new_opaque_id

        original_turn = ConsoleAgentTraceRequestFactory._turn_revision_id
        asked = []

        def drifting_turn(factory):
            # The route is built with the real turn; the dispatch-time
            # derivation then disagrees.
            asked.append(1)
            return original_turn(factory) if len(asked) == 1 else new_opaque_id()

        monkeypatch.setattr(
            ConsoleAgentTraceRequestFactory, "_turn_revision_id", drifting_turn
        )
    monkeypatch.setattr(ConsoleTraceBoundaryFactory, "__call__", boundary)
    from tldw_chatbook.LLM_Calls import hosted_provider_engine

    original_apply = hosted_provider_engine._apply_continuations

    def applied(record, messages, checkpoints, resolution):
        # The continuation checkpoints each actor's request carried, as
        # (state, call ids).
        kind = getattr(current_run_actor(), "kind", None)
        facts.applied.append(
            (
                kind,
                [
                    (each.state, tuple(c.call_id for r in each.rounds for c in r.calls))
                    for each in checkpoints
                ],
            )
        )
        return original_apply(record, messages, checkpoints, resolution)

    monkeypatch.setattr(hosted_provider_engine, "_apply_continuations", applied)
    original_persist = ConsoleChatStore.persist_provider_continuation_event

    def persisted(store, event, **kwargs):
        facts.persisted.append(event.context.agent_kind)
        return original_persist(store, event, **kwargs)

    monkeypatch.setattr(
        ConsoleChatStore, "persist_provider_continuation_event", persisted
    )

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
        facts.sent.append((kind, request["messages"]))
        facts.tools.append(
            (
                kind,
                [
                    (tool.get("function") or tool).get("name")
                    for tool in request.get("tools") or ()
                ],
            )
        )
        if fenced_spawn and kind == "primary" and len(roles) == 2:
            from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN

            call = json.dumps({"name": "spawn_subagent", "arguments": {"task": task}})
            body = _sse({"content": f"{FENCE_OPEN}\n{call}\n```"}, "stop")
        elif kind == "primary" and "tool" not in roles and not fenced_spawn:
            calls = [
                {
                    "index": index,
                    "id": f"call_spawn_{index + 1}",
                    "type": "function",
                    "function": {
                        "name": "spawn_subagent",
                        "arguments": json.dumps(
                            {
                                "task": each,
                                **({"agent": "worker"} if parent_openai else {}),
                            }
                        ),
                    },
                }
                for index, each in enumerate(tasks or (task,))
            ]
            body = _sse({"tool_calls": calls}, "tool_calls")
        elif child_native and kind != "primary" and "tool" not in roles:
            # The child's own native tool round (a provider continuation).
            call = {
                "index": 0,
                "id": "call_child_1",
                "type": "function",
                "function": {"name": "get_current_datetime", "arguments": "{}"},
            }
            body = _sse({"tool_calls": [call]}, "tool_calls")
        elif child_tool and kind != "primary" and len(roles) == 2:
            # A fenced call: the child's own tool round without a provider
            # continuation (the native round is test_child_native_tool_round_*).
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
    app_config = {"custom_endpoints": {"spawntrace": ENDPOINT}}
    if parent_openai:
        # A parent on an endpoint without a continuation protocol (legacy
        # OpenAI handler) whose child is routed to the custom endpoint.
        app_config["api_settings"] = {
            "openai": {
                "api_key_env_var": "OPENAI_API_KEY",
                "api_base_url": ENDPOINT["base_url"],
            }
        }
    gateway = ConsoleProviderGateway(
        config_provider=lambda: app_config,
        environ={"OPENAI_API_KEY": "test-spawntrace-key"} if parent_openai else {},
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(chat),
    )
    store = ConsoleChatStore(persistence=ChatPersistenceService(chat))
    runs = AgentRunsDB(tmp_path / "runs.db", client_id="spawntrace")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="openai" if parent_openai else "custom-ep:spawntrace",
        model=MODEL,
        agent_bridge=ConsoleAgentBridge(
            agent_runs_db=runs, store=store, provider_gateway=gateway
        ),
        agent_runtime_enabled=True,
    )
    if parent_openai:
        from tldw_chatbook.Agents.agent_models import AgentDefinition

        runs.create_agent_definition(
            AgentDefinition(
                name="worker",
                instructions="Do the task.",
                provider="custom-ep:spawntrace",
                model=MODEL,
            )
        )
    session = store.ensure_session()
    if mixin:
        prompts = await _enable_mixin(tmp_path, monkeypatch, store, controller, session)
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
                if sum(kind == "subagent" for kind, _ in facts.runs) >= len(
                    tasks or (task,)
                ) and all(status != "running" for _, status in facts.runs):
                    break
                await asyncio.sleep(0.05)
        facts.child_errors = [
            step.get("summary")
            for row in runs.list_runs(session.persisted_conversation_id)
            if row["agent_kind"] == "subagent"
            for step in (runs.get_run(row["id"]) or {}).get("steps") or ()
            if step.get("kind") == "error"
        ]

        if trace_view:
            from tldw_chatbook.UI.Console_Modules.review_selection import (
                _build_trajectory_snapshot,
            )

            # The trace view's own snapshot, read before the databases close.
            facts.snapshot = _build_trajectory_snapshot(
                store, session.persisted_conversation_id, agent_runs_db=runs
            )
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
        if mixin:
            prompts.close_connection()
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


@pytest.mark.parametrize(
    "tasks",
    [("Say child done.", "Say it again."), ("Say child done.", "Say child done.")],
    ids=["two-siblings", "respawn-same-task-text"],
)


async def test_sibling_children_each_get_their_own_segment_under_one_turn(
    tmp_path, monkeypatch, tasks
):
    """Two children of one turn never share a segment, even for equal task text."""
    facts = await _spawn_turn(tmp_path, monkeypatch, tasks[0], tasks=tasks)
    why = str(vars(facts))
    assert facts.boundary_errors == [] and facts.refusals == [], why
    assert [kind for kind, _ in facts.runs].count("subagent") == 2, why
    parent = [call for call in facts.owned_calls if call[0]]
    child = [call for call in facts.owned_calls if not call[0]]
    (owner_id, turn_id) = parent[0][1:3]
    assert len(child) == 2 and {call[1:3] for call in child} == {(owner_id, turn_id)}, why
    assert len({call[3] for call in child}) == 2, why
    assert len({link[0] for link in facts.links}) == 2, why
    assert {link[1:3] for link in facts.links} == {(owner_id, turn_id)}, why


async def test_a_child_cannot_spawn_a_grandchild(tmp_path, monkeypatch):
    """Nested children are structurally unsupported: the child is never offered
    the spawn tool, so there is no grandchild call to trace."""
    facts = await _spawn_turn(tmp_path, monkeypatch, "Say child done.")
    why = str(vars(facts))
    parent_tools = [tools for kind, tools in facts.tools if kind == "primary"]
    child_tools = [tools for kind, tools in facts.tools if kind == "subagent"]
    assert child_tools and parent_tools, why
    assert all("spawn_subagent" in tools for tools in parent_tools[:1]), why
    assert all("spawn_subagent" not in tools for tools in child_tools), why


@pytest.mark.parametrize("tamper", [True, "missing"], ids=["drifted", "missing"])
async def test_a_child_route_naming_another_turn_is_refused_before_dispatch(
    tmp_path, monkeypatch, tamper
):
    """The adapter re-derives the spawning turn and refuses a drifted or
    missing one, through the real ``chat_call_impl`` path."""
    facts = await _spawn_turn(
        tmp_path, monkeypatch, "Say child done.", tamper_child_route=tamper
    )
    why = str(vars(facts))
    assert [kind for kind, _ in facts.wire].count("subagent") == 0, why
    assert not any(status == "done" for kind, status in facts.runs if kind == "subagent"), why


@pytest.mark.parametrize("capture", ["capture_on", "capture_off"])
async def test_child_native_tool_round_completes(tmp_path, monkeypatch, capture):
    """TASK-41651: a child's own native tool call (a provider continuation)
    completes its round, Capture On and Off."""
    facts = await _spawn_turn(
        tmp_path, monkeypatch, "Say child done.", child_native=True, capture=capture
    )
    why = str(vars(facts))
    child_wire = [roles for kind, roles in facts.wire if kind == "subagent"]
    assert child_wire == [
        ["system", "user"],
        ["system", "user", "assistant", "tool"],
    ], why
    assert ("subagent", "done") in facts.runs and facts.child_errors == [], why
    assert facts.boundary_errors == [] and facts.refusals == [], why
    # The round's continuation reaches the provider, not dropped, and stays
    # the child's: the host's persistence (and trace custody) never sees it.
    assert facts.persisted and set(facts.persisted) == {"primary"}, why
    assert [states for kind, states in facts.applied if kind == "subagent"] == [
        [],
        [("active", ("call_child_1",))],
    ], why
    # The primary's requests carry only its own checkpoint, never the child's.
    assert ("primary", [("active", ("call_spawn_1",))]) in facts.applied, why
    assert all(
        ids == ("call_spawn_1",)
        for kind, states in facts.applied
        if kind == "primary"
        for _, ids in states
    ), why
    if capture == "capture_on":
        child = [call[4:] for call in facts.owned_calls if not call[0]]
        assert child == [("agent_first", "complete"), ("tool_loop", "complete")], why


@pytest.mark.parametrize("fenced_spawn", [False, True], ids=["native", "fenced"])
@pytest.mark.parametrize("capture", ["capture_on", "capture_off"])
async def test_a_spawned_child_never_gets_the_chats_mix_ins(
    tmp_path, monkeypatch, capture, fenced_spawn
):
    """TASK-41652 / ADR-286 rules 5-6: a mix-in composes only the user's current
    message, reused through the parent's tool continuations; a spawned child's
    requests (a machine-authored task) never carry it."""
    facts = await _spawn_turn(
        tmp_path,
        monkeypatch,
        "Say child done.",
        mixin=True,
        capture=capture,
        fenced_spawn=fenced_spawn,
    )
    why = str(vars(facts))
    assert facts.boundary_errors == [] and facts.refusals == [], why
    assert ("subagent", "done") in facts.runs, why
    composed = MIXIN + "\n\n" + DRAFT
    primary = [messages for kind, messages in facts.sent if kind == "primary"]
    assert len(primary) == 2, why
    for messages in primary:
        users = [m["content"] for m in messages if m["role"] == "user"]
        assert users[0] == composed and MIXIN not in json.dumps(users[1:]), why
    child = [messages for kind, messages in facts.sent if kind == "subagent"]
    assert child and MIXIN not in json.dumps(child), why


@pytest.mark.parametrize("capture", ["capture_on", "capture_off"])
async def test_child_native_round_when_the_parent_endpoint_has_no_continuations(
    tmp_path, monkeypatch, capture
):
    """TASK-41651: the parent's endpoint has no continuation protocol, so the
    adapter has no owner key; a child routed to an endpoint that has one still
    completes its native round."""
    facts = await _spawn_turn(
        tmp_path,
        monkeypatch,
        "Say child done.",
        child_native=True,
        capture=capture,
        parent_openai=True,
    )
    why = str(vars(facts))
    child_wire = [roles for kind, roles in facts.wire if kind == "subagent"]
    assert child_wire == [
        ["system", "user"],
        ["system", "user", "assistant", "tool"],
    ], why
    assert ("subagent", "done") in facts.runs and facts.child_errors == [], why
    assert facts.boundary_errors == [] and facts.refusals == [], why
    assert [states for kind, states in facts.applied if kind == "subagent"] == [
        [],
        [("active", ("call_child_1",))],
    ], why
    # The primary's requests carry only its own checkpoint, never the child's.
    assert all(
        ids == ("call_spawn_1",)
        for kind, states in facts.applied
        if kind == "primary"
        for _, ids in states
    ), why
    assert set(facts.persisted) <= {"primary"}, why
    if capture == "capture_on":
        child = [call[4:] for call in facts.owned_calls if not call[0]]
        assert child == [("agent_first", "complete"), ("tool_loop", "complete")], why


async def test_a_service_that_prepares_continuations_needs_a_child_owner_key(tmp_path):
    """TASK-41651: without the key a child's native round would only refuse
    later ("Active provider continuation requires an owner"), so refuse now."""
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    runs = AgentRunsDB(tmp_path / "runs.db", client_id="spawntrace")
    try:
        with pytest.raises(ValueError, match="child_continuation_owner_key"):
            AgentService(
                db=runs,
                registry=ToolCatalogRegistry(),
                chat_call=lambda **_: {},
                prepare_provider_continuation_request=True,
            )
        # A host that prepares nothing needs no key.
        AgentService(db=runs, registry=ToolCatalogRegistry(), chat_call=lambda **_: {})
    finally:
        runs.close()


async def test_trace_view_lists_child_calls_under_the_spawning_turn(
    tmp_path, monkeypatch
):
    """TASK-41650: the mounted Trace lists each child's calls under the turn
    that spawned it, labelled as a child's, never as the parent's."""
    from textual.app import App
    from textual.widgets import DataTable

    from tldw_chatbook.UI.Screens.trajectory_screen import TrajectoryScreen

    tasks = ("Say child done.", "Say it again.")
    facts = await _spawn_turn(
        tmp_path, monkeypatch, tasks[0], tasks=tasks, child_tool=True, trace_view=True
    )
    why = str(vars(facts))
    child = [call for call in facts.owned_calls if not call[0]]
    assert len(child) == 4 and facts.boundary_errors == [], why
    turn_id = child[0][2]

    async with App().run_test(size=(200, 60)) as pilot:
        screen = TrajectoryScreen(facts.snapshot)
        await pilot.app.push_screen(screen)
        await pilot.pause()
        table = screen.query_one("#trajectory-table", DataTable)
        rows = [
            (row.key.value, [str(cell).strip() for cell in table.get_row(row.key)])
            for row in table.ordered_rows
        ]
        # `a` (next child agent) stops on a child call.
        first = next(i for i, (_, cells) in enumerate(rows) if cells[1] == "Child call")
        table.move_cursor(row=first - 1)
        await pilot.press("a")
        await pilot.pause()
        landed = dict(rows)[screen._cursor_key()]

    headers = [i for i, (key, _) in enumerate(rows) if key.startswith("turn")]
    # One user turn, one turn header: the children's own agent runs join the
    # spawning turn instead of opening a phantom one.
    assert [rows[i][0] for i in headers] == [f"turn:{turn_id}"], rows
    start = [key for key, _ in rows].index(f"turn:{turn_id}")
    end = next((i for i in headers if i > start), len(rows))
    child_rows = [cells for _, cells in rows if cells[1] == "Child call"]
    # Every child call sits inside the spawning turn's block, none elsewhere,
    # so the parent's own two calls are never shown as a child's.
    assert child_rows == [
        cells for _, cells in rows[start:end] if cells[1] == "Child call"
    ], rows
    assert sorted(cells[2] for cells in child_rows) == [
        "↳ sub-agent 1 · call 1 · agent first",
        "↳ sub-agent 1 · call 2 · tool loop",
        "↳ sub-agent 2 · call 1 · agent first",
        "↳ sub-agent 2 · call 2 · tool loop",
    ], rows
    assert {cells[3] for cells in child_rows} == {"Complete"}, rows
    # In / Cache / Out, as the parent's rows show them (the double reports
    # prompt 10 and completion 5 on every reply).
    assert {tuple(cells[4:7]) for cells in child_rows} == {("10", "0", "5")}, rows
    assert landed[1] == "Child call", landed
