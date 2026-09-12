"""Local tool calls retain their own canonical thinking through the live loop."""

from dataclasses import replace

from Tests.Chat.test_console_agent_bridge import (
    _bridge_with_gateway,
    _ChunkGateway,
    _native_calls,
    _native_resolution,
    _run,
)
from tldw_chatbook.Chat.console_provider_gateway import ProviderThinkingDelta
from tldw_chatbook.Chat.console_thinking_capture import ThinkingCapture


def _delta(text):
    return ProviderThinkingDelta(
        text=text,
        provider="llama_cpp",
        model="reasoner",
        protocol="chat_completions",
        source_format="reasoning_content",
    )


def test_aggregate_tool_thinking_cannot_become_final_answer_thinking():
    capture = ThinkingCapture(assistant_owner_id="answer")
    capture.observe(_delta("tool thought"))
    capture.observe_tool()
    capture.observe_answer("final answer without thinking")
    envelope = capture.settle("complete").envelope
    assert envelope.blocks[0].source_format == "reasoning_content:tool_call"


def test_live_native_loop_passes_exact_call_envelope_before_preparation(tmp_path):
    class Gateway(_ChunkGateway):
        def __init__(self):
            super().__init__(
                [
                    [
                        _delta("first tool"),
                        _native_calls("calculator", {"expression": "1+1"}),
                    ],
                    [
                        _delta("second tool"),
                        _native_calls("calculator", {"expression": "2+2"}, "c2"),
                    ],
                    ["The answer is 4."],
                ]
            )
            self.prepared = []

        def prepare_chat_request(self, resolution, messages, **kwargs):
            self.prepared.append((messages, kwargs["thinking_sidecar"]))
            return messages

    gateway = Gateway()
    bridge, db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    outcome = _run(bridge, store, session, aid, resolution=_native_resolution())
    assert outcome.final_text == "The answer is 4."
    assert len(gateway.prepared) == 2
    rows, sidecars = gateway.prepared[-1]
    assert [s.envelope.blocks[0].text for s in sidecars] == [
        "first tool",
        "second tool",
    ]
    assert all(
        s.envelope.blocks[0].source_format == "reasoning_content" for s in sidecars
    )
    assert all(s.envelope.blocks[0].status == "complete" for s in sidecars)
    for row, sidecar in zip([r for r in rows if r.get("tool_calls")], sidecars):
        assert sidecar.owner_message_id in row.values()
        assert "_tldw_call_thinking" not in row
    assert all(
        block.source_format.endswith(":tool_call")
        for block in store.get_message(aid).thinking.blocks
    )
    db.close()


def test_local_native_capability_is_independent_of_reasoning_off():
    from tldw_chatbook.Agents.native_tools import provider_supports_native_tools
    from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy

    enabled = ReasoningReplayPolicy("off", "override", native_tools=True)
    disabled = replace(enabled, mode="all", native_tools=False)
    assert provider_supports_native_tools("local_llamacpp", reasoning_replay=enabled)
    assert not provider_supports_native_tools(
        "local_llamacpp", reasoning_replay=disabled
    )


def test_agent_config_pins_immutable_reasoning_policy():
    from dataclasses import FrozenInstanceError

    import pytest

    from tldw_chatbook.Agents.agent_models import AgentConfig
    from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy

    policy = ReasoningReplayPolicy("current", "override", native_tools=True)
    config = AgentConfig(
        model="org/exact-model", system_prompt="", reasoning_replay=policy
    )
    assert config.reasoning_replay is policy
    with pytest.raises(FrozenInstanceError):
        config.reasoning_replay.mode = "off"


def test_runtime_steering_retains_current_exchange_annotation():
    from Tests.Agents.test_agent_runtime import CFG, make_deps
    from tldw_chatbook.Agents.agent_models import ModelTurn
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop
    from tldw_chatbook.Chat.local_reasoning import EXCHANGE_CONTINUATION_KEY

    deps = make_deps([ModelTurn(text="done")])
    deps.drain_mailbox = lambda: [("user", "keep going")]
    seen = []
    deps.call_model = lambda rows, schemas: seen.extend(rows) or ModelTurn(text="done")
    run_agent_loop(
        config=CFG,
        initial_messages=[{"role": "user", "content": "task"}],
        active_schemas=[],
        deps=deps,
    )
    assert seen[-1].get(EXCHANGE_CONTINUATION_KEY) is True


def test_agent_budget_counts_exact_model_projected_reasoning():
    from tldw_chatbook.Agents.agent_service import _count_model_messages
    from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy

    capture = ThinkingCapture(assistant_owner_id="tool")
    capture.observe(
        ProviderThinkingDelta(
            text="a long thought " * 40,
            provider="llama_cpp",
            model="org/exact-model",
            protocol="chat_completions",
            source_format="reasoning_content",
        )
    )
    envelope = capture.settle("complete").envelope
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "call", "_tldw_call_thinking": envelope},
    ]
    policy = ReasoningReplayPolicy("all", "override")
    included = _count_model_messages(
        messages,
        "org/exact-model",
        "llama_cpp",
        reasoning_replay=policy,
        tokenizer_model="exact-model",
    )
    excluded = _count_model_messages(
        messages,
        "org/exact-model",
        "llama_cpp",
        reasoning_replay=replace(policy, mode="off"),
        tokenizer_model="exact-model",
    )
    assert included > excluded + 100


import pytest


@pytest.mark.parametrize(
    "conversation_policy,expected",
    [("auto", None), ("include", "first tool"), ("exclude", None)],
)
def test_local_live_native_payload_respects_conversation_override(
    tmp_path, conversation_policy, expected
):
    import asyncio

    from tldw_chatbook.Chat.console_prepared_request import (
        PreparedProviderRequest,
        thaw_json,
    )
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy

    real_gateway = ConsoleProviderGateway(config_provider=dict, environ={})

    class Gateway(_ChunkGateway):
        prepare_chat_request = real_gateway.prepare_chat_request

        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            if isinstance(messages, PreparedProviderRequest):
                tools = [thaw_json(tool) for tool in messages.tools]
                messages = [thaw_json(row) for row in messages.messages]
            async for chunk in super().stream_chat(
                resolution, messages, tools=tools, **kwargs
            ):
                yield chunk

    gateway = Gateway(
        [
            [_delta("first tool"), _native_calls("calculator", {"expression": "1+1"})],
            ["The answer is 2."],
        ]
    )
    bridge, db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    resolution = replace(
        _native_resolution(),
        provider="llama_cpp",
        execution_key="llama_cpp",
        model="reasoner",
        local_structured_thinking=True,
        reasoning_replay=ReasoningReplayPolicy("off", "override", native_tools=True),
    )
    try:
        outcome = _run(
            bridge,
            store,
            session,
            aid,
            resolution=resolution,
            model=resolution.model,
            thinking_policy=conversation_policy,
        )
        assert outcome.final_text == "The answer is 2."
        assert gateway.tools_seen[0]
        tool_owner = next(
            row for row in gateway.messages_seen[-1] if row.get("tool_calls")
        )
        assert tool_owner.get("reasoning_content") == expected
        assert all(
            not key.startswith("_tldw_")
            for row in gateway.messages_seen[-1]
            for key in row
        )
    finally:
        db.close()
        asyncio.run(real_gateway.aclose())


@pytest.mark.parametrize("child_model", ["test-model", "child-model"])
def test_child_inherits_policy_and_only_replays_its_own_calls(
    tmp_path, monkeypatch, child_model
):
    from tldw_chatbook.Agents import agent_service
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME, AgentDefinition
    from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter
    from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy

    real_setting = agent_service._setting
    monkeypatch.setattr(
        agent_service,
        "_setting",
        lambda key, default: (
            1
            if key == agent_service.MAX_LIVE_SUBAGENTS_KEY
            else real_setting(key, default)
        ),
    )
    policy = ReasoningReplayPolicy("all", "override", native_tools=True)

    configs = []
    real_config = agent_service.AgentConfig

    def capture_config(**kwargs):
        config = real_config(**kwargs)
        configs.append(config)
        return config

    monkeypatch.setattr(agent_service, "AgentConfig", capture_config)

    class Gateway(_ChunkGateway):
        def prepare_chat_request(self, resolution, messages, **kwargs):
            expected = policy if resolution.model == "test-model" else None
            assert resolution.reasoning_replay is expected
            assert resolution.local_structured_thinking is (expected is policy)
            thoughts = [
                sidecar.envelope.blocks[0].text
                for sidecar in kwargs["thinking_sidecar"]
            ]
            if _StreamingModelAdapter._is_subagent(messages):
                assert thoughts == ["child tool"]
            else:
                assert thoughts == ["parent spawn"]
            return messages

    gateway = Gateway(
        [
            [
                _delta("parent spawn"),
                _native_calls(
                    SPAWN_TOOL_NAME, {"task": "calculate two", "agent": "worker"}
                ),
            ],
            [
                _delta("child tool"),
                _native_calls("calculator", {"expression": "1+1"}, "child-call"),
            ],
            ["child done"],
            ["parent done"],
        ]
    )
    bridge, db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    db.create_agent_definition(
        AgentDefinition(name="worker", instructions="Calculate.", model=child_model)
    )
    try:
        outcome = _run(
            bridge,
            store,
            session,
            aid,
            resolution=replace(
                _native_resolution(),
                model="test-model",
                reasoning_replay=policy,
                local_structured_thinking=True,
            ),
        )
        assert outcome.final_text == "parent done"
        assert gateway.calls == 4
        assert configs
        assert all(
            config.reasoning_replay is (policy if child_model == "test-model" else None)
            for config in configs
        )
        assert [block.text for block in store.get_message(aid).thinking.blocks] == [
            "parent spawn"
        ]
    finally:
        db.close()


def test_active_local_call_thinking_keeps_trace_owner_through_real_admission(
    tmp_path,
):
    from Tests.Chat.test_console_agent_bridge import (
        RUN_DONE,
        CharactersRAGDB,
        ConsoleProviderGateway,
        ConsoleRequestRoute,
        ConsoleTraceCallBoundary,
        ConsoleTraceCaptureMode,
        ConsoleTraceRepository,
        ConsoleTraceService,
        FrozenTracePolicy,
        SavedRevisionTraceProvenance,
        SurfaceDeltaAdmission,
        TraceCallIdentity,
        TraceCallState,
        TraceProvenanceSource,
        _test_resolution,
        build_console_request,
        new_opaque_id,
    )
    from tldw_chatbook.Chat.console_trace_provenance import TraceTransformKind
    from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy

    trace_db = CharactersRAGDB(tmp_path / "trace.sqlite", "reasoning-agent-trace")
    repository = ConsoleTraceRepository()
    service = ConsoleTraceService(repository)
    conversation_id = trace_db.add_conversation({"title": "agent trace"})
    assert conversation_id is not None
    message_id = trace_db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "hi",
        }
    )
    assert message_id is not None
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )
    with trace_db.transaction() as cursor:
        revision_row = cursor.execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ?
                 ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id,),
        ).fetchone()
        assert revision_row is not None
        saved_user = SavedRevisionTraceProvenance(str(revision_row[0]))
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
        repository.ensure_policy(cursor, policy)

    routes: list[ConsoleRequestRoute] = []
    provenances = []
    capture_policies = []
    adapter_requests = []
    adapter_entries = 0

    boundary_errors = []

    def prepare_boundary(request, _resolution, route):
        sequence = len(routes)
        provenance = request.provenance
        assert provenance is not None
        preparation_identity = new_opaque_id()
        with trace_db.transaction() as cursor:
            tail = repository.get_surface_tail(cursor, segment.segment_id)
            prefix_length = 0 if tail is None else tail.sequence + 1
            delta = tuple(provenance.messages_payload[prefix_length:])
            admission = SurfaceDeltaAdmission(
                owner_id=owner.owner_id,
                segment_id=segment.segment_id,
                predecessor_surface_head_id=(None if tail is None else tail.node_id),
                route_identity=route.value,
                preparation_identity=preparation_identity,
                descriptors=delta,
            )
            surface_boundary = service.prepare_surface_provenance(
                cursor,
                None,
                provenance=provenance,
                admission=admission,
                values=tuple(request.messages_payload),
            )
        routes.append(route)
        provenances.append(provenance)
        assert request.semantic.provenance is not None
        capture_policies.append(request.semantic.provenance.capture_policy)
        return ConsoleTraceCallBoundary(
            service=service,
            database=trace_db,
            identity=TraceCallIdentity(
                owner_id=owner.owner_id,
                segment_id=segment.segment_id,
                turn_id="turn-1",
                run_id="run-1",
                call_sequence=sequence,
                idempotency_key=new_opaque_id(),
                policy_id=policy.policy_id,
            ),
            admission=admission,
            occurred_at_factory=lambda: "2026-08-29T20:00:00Z",
            surface_boundary=surface_boundary,
        )

    def boundary_factory(request, resolution, route):
        try:
            return prepare_boundary(request, resolution, route)
        except Exception as exc:
            boundary_errors.append(exc)
            raise

    def adapter(**kwargs):
        nonlocal adapter_entries
        calls = repository.read_calls(
            trace_db.get_connection().cursor(), owner.owner_id
        )
        assert calls[-1].state is TraceCallState.DISPATCH_STARTED
        adapter_requests.append(tuple(kwargs["messages_payload"]))
        adapter_entries += 1
        assert kwargs["tools"]
        message = {"content": "42"}
        if adapter_entries == 1:
            message = {
                "content": "",
                "reasoning_content": "EXACT-LOCAL-TOOL-THOUGHT",
                "tool_calls": list(
                    _native_calls("calculator", {"expression": "6*7"}).tool_calls
                ),
            }
        return {"choices": [{"message": message}]}

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=adapter,
        trace_call_boundary_factory=boundary_factory,
    )
    bridge, runs_db, store, session, aid = _bridge_with_gateway(
        tmp_path / "agent", gateway
    )
    try:
        outcome = _run(
            bridge,
            store,
            session,
            aid,
            resolution=_test_resolution(
                provider="local_vllm",
                execution_key="local_vllm",
                base_url="http://127.0.0.1:9099/v1",
                model="reasoner",
                streaming=False,
                local_structured_thinking=True,
                reasoning_replay=ReasoningReplayPolicy(
                    "all", "override", native_tools=True
                ),
            ),
            model="reasoner",
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            propagate_trace_call_persistence_errors=True,
            trace_request=build_console_request(
                [{"role": "user", "content": "hi"}],
                message_provenance=(saved_user,),
                memory_provenance=(),
                mandatory_provenance=(),
                tool_provenance=(),
                capture_policy=policy,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            ),
        )
        calls = repository.read_calls(
            trace_db.get_connection().cursor(), owner.owner_id
        )
        assert outcome.status == RUN_DONE, outcome.steps
        assert adapter_entries == 2
        assert routes == [
            ConsoleRequestRoute.AGENT_FIRST,
            ConsoleRequestRoute.TOOL_LOOP,
        ]
        assert [call.call_sequence for call in calls] == [0, 1]
        assert all(call.state is TraceCallState.COMPLETE for call in calls)
        links = [
            repository.get_response_link(
                trace_db.get_connection().cursor(), call.call_id
            )
            for call in calls
        ]
        assert all(link is not None and link.link_kind == "artifact" for link in links)
        assert capture_policies == [policy, policy]
        assert provenances[0].messages_payload == (saved_user,)
        assert provenances[1].messages_payload[0] == saved_user
        assert len(provenances[1].thinking) == 1
        attachment = provenances[1].thinking[0]
        assert attachment.transform is TraceTransformKind.THINKING_ATTACHMENT
        assert attachment.inputs[0].source is TraceProvenanceSource.TOOL_CALL
        rewritten_owner = provenances[1].messages_payload[1]
        assert rewritten_owner.transform is TraceTransformKind.MESSAGE_REWRITE
        assert rewritten_owner.inputs[0] == attachment.inputs[0]
        assert attachment in rewritten_owner.inputs
        assert (
            provenances[1].messages_payload[2].source
            is TraceProvenanceSource.TOOL_RESULT
        )
        assert [len(request) for request in adapter_requests] == [1, 3]
        assert adapter_requests[1][1]["role"] == "assistant"
        assert adapter_requests[1][2]["role"] == "tool"
        assert adapter_requests[1][1]["reasoning_content"] == "EXACT-LOCAL-TOOL-THOUGHT"
        assert all(
            not key.startswith("_tldw_")
            for request in adapter_requests
            for row in request
            for key in row
        )
        assert store.get_message(aid).content == "42"
        assert (
            store.get_message(aid).thinking.blocks[0].source_format
            == "reasoning_content:tool_call"
        )
    finally:
        runs_db.close()
        trace_db.close_connection()
        if boundary_errors:
            raise boundary_errors[0]


def test_fallback_cannot_inherit_another_local_servers_native_approval(
    tmp_path, monkeypatch
):
    from Tests.Agents.test_agent_service import CFG, _service_with, provider_reply
    from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
    from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if kwargs["api_endpoint"] == "llama_cpp":
            raise ChatProviderError("quota", status_code=402)
        return provider_reply("fallback complete")

    db = AgentRunsDB(tmp_path / "fallback.db", client_id="fallback-policy")
    service = _service_with(db, chat)
    monkeypatch.setattr(service, "_provider_is_ready", lambda _provider: True)
    try:
        _, outcome = service.run_turn(
            conversation_id="fallback",
            messages=[{"role": "user", "content": "task"}],
            config=replace(
                CFG,
                provider="llama_cpp",
                fallback_providers=("local_vllm",),
                reasoning_replay=ReasoningReplayPolicy(
                    "all", "primary override", native_tools=True
                ),
            ),
            api_endpoint="llama_cpp",
            should_cancel=lambda: False,
        )
        assert outcome.final_text == "fallback complete"
        assert len(calls) == 2
        assert calls[0]["tools"]
        assert not calls[1].get("tools")
        assert "tool_call" in calls[1]["messages_payload"][0]["content"]
    finally:
        db.close()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("mode", ["current", "all"])
@pytest.mark.parametrize("child_model", ["reasoner", "child-model"])
def test_resumed_child_retains_its_final_call_thinking_only(
    tmp_path, monkeypatch, native, mode, child_model
):
    from Tests.Agents.test_agent_service import fence, provider_reply
    from Tests.Agents.test_fleet_continuation import (
        RESUME_CFG,
        _await_retained,
        _finished_child,
    )
    from Tests.Agents.test_fleet_continuation import _run as run_fleet
    from Tests.Agents.test_fleet_runtime import make_fleet_service
    from tldw_chatbook.Agents import agent_service
    from tldw_chatbook.Agents.agent_models import (
        SEND_TO_AGENT_TOOL_NAME,
        SPAWN_TOOL_NAME,
        WAIT_AGENTS_TOOL_NAME,
        AgentDefinition,
    )
    from tldw_chatbook.Chat.console_agent_bridge import _StreamingProviderResponse
    from tldw_chatbook.Chat.local_reasoning import (
        ReasoningReplayPolicy,
        project_reasoning_history,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    configs = []
    real_config = agent_service.AgentConfig

    def capture_config(**kwargs):
        config = real_config(**kwargs)
        configs.append(config)
        return config

    monkeypatch.setattr(agent_service, "AgentConfig", capture_config)
    capture = ThinkingCapture(assistant_owner_id="child-final")
    capture.observe(replace(_delta("CHILD-FINAL-THOUGHT"), model=child_model))
    envelope = capture.settle("complete").envelope
    holder = {}
    db = AgentRunsDB(tmp_path / "retained.db", client_id="retained-thinking")
    db.create_agent_definition(
        AgentDefinition(name="worker", instructions="Calculate.", model=child_model)
    )
    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "calculate", "agent": "worker"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "parent done",
            lambda: fence(
                SEND_TO_AGENT_TOOL_NAME, {"id": holder["id"], "message": "check again"}
            ),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "parent checked",
        ],
        {"calculate": ["child done", "child checked"]},
    )
    chat._reply = lambda item: _StreamingProviderResponse(
        provider_reply(item), None, envelope if item == "child done" else None
    )
    policy = ReasoningReplayPolicy(mode, "override", native_tools=True)
    config = replace(
        RESUME_CFG, model="reasoner", native_tools=native, reasoning_replay=policy
    )
    try:
        _, first = run_fleet(service, config=config)
        assert first.status == "done"
        child = _finished_child(coordinator)
        holder["id"] = child.handle_id
        _await_retained(coordinator, child.handle_id)
        retained = coordinator.get_retained(child.handle_id)
        _, second = run_fleet(service, config=config)
        assert second.status == "done"
        assert len(configs) == 2
        assert all(
            cfg.reasoning_replay is (policy if child_model == "reasoner" else None)
            for cfg in configs
        )
        assert retained.messages[-1].get("_tldw_call_thinking") is envelope
        resumed_rows = chat.child_calls["calculate"][1]["messages_payload"]
        projected = project_reasoning_history(
            resumed_rows, provider="llama_cpp", model=child_model, policy=policy
        )
        assert [
            row["reasoning_content"] for row in projected if "reasoning_content" in row
        ] == ["CHILD-FINAL-THOUGHT"]
        assert all("_tldw_call_thinking" not in row for row in projected)
        resumed_handle = next(
            handle
            for handle in coordinator.snapshot()
            if handle.handle_id != child.handle_id
        )
        _await_retained(coordinator, resumed_handle.handle_id)
        assert (
            "_tldw_call_thinking"
            not in coordinator.get_retained(resumed_handle.handle_id).messages[-1]
        )
    finally:
        db.close()


def test_retained_transcript_cap_counts_canonical_thinking_text():
    from Tests.Agents.test_fleet_continuation import _coord, _finished_handle

    coordinator = _coord(retained_transcript_max_chars=2000)
    handle = _finished_handle(coordinator)
    capture = ThinkingCapture(assistant_owner_id="large-final")
    capture.observe(_delta("large thought " * 1000))
    envelope = capture.settle("complete").envelope
    rows = [
        {
            "role": "assistant",
            "content": "small answer",
            "_tldw_call_thinking": envelope,
        }
    ]
    assert coordinator.retain_transcript(handle.handle_id, rows) is False
    assert coordinator.get_retained(handle.handle_id) is None
