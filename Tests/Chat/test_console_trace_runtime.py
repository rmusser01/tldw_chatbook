from __future__ import annotations

import sqlite3
from contextlib import AsyncExitStack
from types import SimpleNamespace

import httpx
import pytest

from tldw_chatbook.Chat.console_prepared_request import (
    CONTINUATION_OWNER_KEY,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    ProviderArtifactTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceSource,
    request_route_provenance,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRound,
    ProviderContinuationCheckpoint,
    continuation_owner_group,
    dump_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def make_database():
    """Create runtime-test databases and close every instance at teardown."""

    databases: list[CharactersRAGDB] = []

    def create(path, client_id: str) -> CharactersRAGDB:
        database = CharactersRAGDB(path, client_id)
        databases.append(database)
        return database

    yield create
    for database in reversed(databases):
        database.close()


@pytest.fixture
async def make_gateway():
    """Create gateways whose owned clients close on every test exit path."""
    async with AsyncExitStack() as resources:

        def create(**kwargs: object) -> ConsoleProviderGateway:
            gateway = ConsoleProviderGateway(**kwargs)
            resources.push_async_callback(gateway.aclose)
            return gateway

        yield create


def _saved_message(
    database: CharactersRAGDB,
    conversation_id: str,
    content: str,
    *,
    sender: str = "user",
) -> tuple[str, SavedRevisionTraceProvenance]:
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": sender,
            "content": content,
        }
    )
    assert message_id is not None
    with database.transaction() as cursor:
        row = cursor.execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id,),
        ).fetchone()
    assert row is not None
    return message_id, SavedRevisionTraceProvenance(str(row[0]))


def _semantic_request(
    messages: list[dict[str, str]],
    descriptors: list[SavedRevisionTraceProvenance],
    policy: FrozenTracePolicy,
    *,
    route: ConsoleRequestRoute = ConsoleRequestRoute.FRESH,
    actor_id: str | None = None,
    chain_id: str | None = None,
):
    return build_console_request(
        messages,
        message_provenance=tuple(descriptors),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        metadata_provenance=(
            request_route_provenance(
                route,
                actor_id=actor_id,
                chain_id=chain_id,
            ),
        ),
        capture_policy=policy,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )


@pytest.mark.asyncio
async def test_console_runtime_wires_production_boundary_for_durable_database(
    tmp_path,
    make_database,
) -> None:
    database = make_database(tmp_path / "trace-runtime-wiring.sqlite", "trace-wiring")
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=database))

    gateway = runtime.ensure_provider_gateway(config_provider=lambda: {})
    try:
        assert gateway.supports_durable_capture is True
        lazy_factory = gateway._trace_call_boundary_factory
        assert callable(lazy_factory)
        assert runtime.chat_store is not None
        repository = runtime.chat_store.persistence.console_trace_repository
        assert lazy_factory._repository is repository
        assert isinstance(
            lazy_factory._get_delegate(),
            ConsoleTraceBoundaryFactory,
        )
        assert lazy_factory._get_delegate().repository is repository
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "cold",
        "equal_policy_fresh_id",
        "credential_value",
        "wrong_policy",
        "wrong_pii_enabled",
        "wrong_ruleset_revision",
        "wrong_response_revision",
        "incomplete_terminal",
        "unrelated_reservation",
        "non_tool_artifact",
        "changed_prefix",
        "changed_assistant_envelope",
        "extra_new_item",
        "unsupported_route",
        "foreign_assistant",
        "rollback_replace",
        "rollback_append",
        "rollback_bind",
    ],
)
async def test_completed_tool_turn_compound_admission_checks_durable_proof(
    tmp_path,
    make_database,
    make_gateway,
    monkeypatch,
    scenario,
):
    """Only the completed run's exact saved answer may collapse its tool suffix."""
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader

    database = make_database(tmp_path / "completed-tool.sqlite", "completed-tool")
    conversation = database.add_conversation({"title": "completed tool"})
    user_id, user = _saved_message(database, conversation, "calculate")
    answer_id, answer = _saved_message(database, conversation, "42", sender="assistant")
    next_text = (
        "Bearer sk-proj-" + "x" * 48 if scenario == "credential_value" else "next"
    )
    _, next_user = _saved_message(database, conversation, next_text)
    policy = FrozenTracePolicy(
        new_opaque_id(), "credentials-v1", False,
        new_opaque_id() if scenario == "wrong_pii_enabled" else None,
    )
    actor, chain = new_opaque_id(), new_opaque_id()
    factory = ConsoleTraceBoundaryFactory(database)
    boundaries, adapter_entries = [], []

    def boundary(request, resolution, route):
        result = factory(request, resolution, route)
        boundaries.append(result)
        return result

    def adapter(**kwargs):
        adapter_entries.append(kwargs["messages_payload"])
        return {"choices": [{"message": {"content": "42"}}]}

    gateway = make_gateway(
        chat_api_call_fn=adapter, trace_call_boundary_factory=boundary
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="openai",
        model="gpt-test",
        execution_key="openai",
        base_url="https://api.openai.com/v1",
        streaming=False,
    )

    def prepare(messages, descriptors, route, selected_policy=policy):
        routed_actor = (
            actor
            if route in {ConsoleRequestRoute.AGENT_FIRST, ConsoleRequestRoute.TOOL_LOOP}
            else None
        )
        routed_chain = chain if routed_actor is not None else None
        return gateway.prepare_chat_request(
            resolution,
            _semantic_request(
                messages,
                descriptors,
                selected_policy,
                route=route,
                actor_id=routed_actor,
                chain_id=routed_chain,
            ),
            route=route,
            route_actor_id=routed_actor,
            route_chain_id=routed_chain,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )

    async def dispatch(prepared, route, canonical_id=None):
        signals = ConsoleProviderStreamSignals()
        if canonical_id is not None:

            def settle(handoff):
                if scenario != "incomplete_terminal":
                    assert handoff.settle(canonical_id)
                return True

            signals.bind_trace_settlement_sink(settle)
        return [
            item
            async for item in gateway.stream_chat(
                resolution,
                prepared,
                route=route,
                route_actor_id=actor,
                route_chain_id=chain,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                signals=signals,
            )
        ]

    first_messages = [{"role": "user", "content": "calculate"}]
    assert await dispatch(
        prepare(first_messages, [user], ConsoleRequestRoute.AGENT_FIRST),
        ConsoleRequestRoute.AGENT_FIRST,
    ) == ["42"]
    source = (
        TraceProvenanceSource.ACTIVE_REQUEST
        if scenario == "non_tool_artifact"
        else TraceProvenanceSource.TOOL_RESULT
    )
    tool_count = 2
    tool_messages = [
        {"role": "tool", "content": f"result-{index}", "tool_call_id": f"call-{index}"}
        for index in range(tool_count)
    ]
    tool_descriptors = [ProviderArtifactTraceProvenance(source, policy)] * tool_count
    if scenario == "non_tool_artifact":
        tool_messages = [
            {"role": "assistant", "content": f"ordinary-{index}"}
            for index in range(tool_count)
        ]
    assert await dispatch(
        prepare(
            first_messages + tool_messages,
            [user] + tool_descriptors,
            ConsoleRequestRoute.TOOL_LOOP,
        ),
        ConsoleRequestRoute.TOOL_LOOP,
        answer_id,
    ) == ["42"]
    terminal = boundaries[-1].reserve()
    repository = factory.repository
    with database.transaction() as cursor:
        terminal = repository.get_call(cursor, terminal.call_id)
        assert terminal.state is (
            TraceCallState.RESPONSE_STARTED
            if scenario == "incomplete_terminal"
            else TraceCallState.COMPLETE
        )
        link = repository.get_response_link(cursor, terminal.call_id)
        if scenario == "incomplete_terminal":
            assert link is None
        else:
            assert link.semantic_revision_id == answer.revision_id
            assert link.verification_outcome == "verified_equal"
    reader = ConsoleTraceNativeReader(database)
    original = reader.read_calls(user_id)
    assert len(original) == (1 if scenario == "incomplete_terminal" else 2)
    incoming = first_messages + [
        {"role": "assistant", "content": "42"},
        {"role": "user", "content": next_text},
    ]
    descriptors = [user, answer, next_user]
    next_policy = policy
    route = ConsoleRequestRoute.AGENT_FIRST
    chain = new_opaque_id()
    if scenario == "cold":
        factory = ConsoleTraceBoundaryFactory(database)
    elif scenario == "equal_policy_fresh_id":
        next_policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    elif scenario == "wrong_policy":
        next_policy = FrozenTracePolicy(new_opaque_id(), "credentials-v2", False, None)
    elif scenario == "wrong_pii_enabled":
        next_policy = FrozenTracePolicy(
            new_opaque_id(), "credentials-v1", True, policy.pii_ruleset_revision_id
        )
    elif scenario == "wrong_ruleset_revision":
        next_policy = FrozenTracePolicy(
            new_opaque_id(), "credentials-v1", False, new_opaque_id()
        )
    elif scenario == "wrong_response_revision":
        _, descriptors[1] = _saved_message(
            database, conversation, "42", sender="assistant"
        )
    elif scenario == "foreign_assistant":
        foreign = database.add_conversation({"title": "foreign"})
        _, descriptors[1] = _saved_message(database, foreign, "42", sender="assistant")
    elif scenario == "changed_prefix":
        incoming[0] = {"role": "user", "content": "changed"}
    elif scenario == "changed_assistant_envelope":
        incoming[1] = {"role": "assistant", "content": "42", "name": "different"}
    elif scenario == "extra_new_item":
        _, extra = _saved_message(database, conversation, "extra")
        descriptors.insert(2, extra)
        incoming.insert(2, {"role": "user", "content": "extra"})
    elif scenario == "unsupported_route":
        route = ConsoleRequestRoute.REGENERATE
    elif scenario == "unrelated_reservation":
        with database.transaction() as cursor:
            unrelated = repository.reserve_call(
                cursor,
                owner_id=terminal.owner_id,
                segment_id=terminal.segment_id,
                turn_id=user_id,
                run_id=new_opaque_id(),
                call_sequence=0,
                idempotency_key=new_opaque_id(),
                policy_id=policy.policy_id,
            )
            event_tail = repository.get_event_tail(cursor, terminal.segment_id)
            repository.append_event(
                cursor,
                segment_id=terminal.segment_id,
                sequence=event_tail.sequence + 1,
                event_type="call_boundary",
                call_id=unrelated.call_id,
            )
    prepared = prepare(incoming, descriptors, route, next_policy)
    with database.transaction() as cursor:
        before_calls = tuple(
            tuple(row) for row in cursor.execute("SELECT * FROM console_trace_calls")
        )
        before_nodes = tuple(
            tuple(row)
            for row in cursor.execute("SELECT * FROM console_trace_surface_nodes")
        )
    if scenario.startswith("rollback_"):
        from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError

        target = repository if scenario == "rollback_bind" else factory.service
        operation = {
            "rollback_replace": "_replace_surface",
            "rollback_append": "_append_descriptor",
            "rollback_bind": "bind_call",
        }[scenario]
        real_operation = getattr(type(target), operation)
        hits = []

        def fail_after_write(instance, *args, **kwargs):
            result = real_operation(instance, *args, **kwargs)
            if instance is not target:
                return result
            hits.append(True)
            if scenario != "rollback_append" or len(hits) == 2:
                raise RuntimeError("synthetic precommit failure")
            return result

        monkeypatch.setattr(type(target), operation, fail_after_write)
        with pytest.raises(TraceCallPersistenceError):
            await dispatch(prepared, route)
        assert len(hits) == (2 if scenario == "rollback_append" else 1)
        assert len(adapter_entries) == 2
        with database.transaction() as cursor:
            assert (
                tuple(
                    tuple(row)
                    for row in cursor.execute(
                        "SELECT * FROM console_trace_surface_nodes"
                    )
                )
                == before_nodes
            )
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM console_trace_surface_replacements"
                ).fetchone()[0]
                == 0
            )
            reserved = repository.get_call(cursor, boundaries[-1].reserve().call_id)
            assert reserved.state is TraceCallState.RESERVED
            assert reserved.surface_node_id is None
            assert reserved.request_header_id is None
        assert reader.read_calls(user_id) == original
    elif scenario in {"valid", "cold", "equal_policy_fresh_id", "credential_value"}:
        assert await dispatch(prepared, route) == ["42"]
        assert [dict(row) for row in adapter_entries[-1]] == [
            {"role": "user", "content": "calculate"},
            {"role": "assistant", "content": "42"},
            {"role": "user", "content": next_text},
        ]
        with database.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM console_trace_surface_replacements"
                ).fetchone()[0]
                == 1
            )
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM console_trace_surface_nodes"
                ).fetchone()[0]
                == 5
            )
        assert reader.read_calls(user_id) == original
    else:
        with pytest.raises(ValueError):
            factory(prepared, resolution, route)
        assert len(adapter_entries) == 2
        with database.transaction() as cursor:
            assert (
                tuple(
                    tuple(row)
                    for row in cursor.execute("SELECT * FROM console_trace_calls")
                )
                == before_calls
            )
            assert (
                tuple(
                    tuple(row)
                    for row in cursor.execute(
                        "SELECT * FROM console_trace_surface_nodes"
                    )
                )
                == before_nodes
            )


@pytest.mark.asyncio
async def test_production_factory_persists_append_only_calls_through_real_gateway(
    tmp_path,
    make_database,
    make_gateway,
) -> None:
    database = make_database(tmp_path / "trace-runtime.sqlite", "trace-runtime")
    conversation_id = database.add_conversation({"title": "runtime trace"})
    assert conversation_id is not None
    _first_id, first = _saved_message(database, conversation_id, "first")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    calls: list[dict[str, object]] = []

    def adapter(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    factory = ConsoleTraceBoundaryFactory(database)
    gateway = make_gateway(
        chat_api_call_fn=adapter,
        trace_call_boundary_factory=factory,
    )
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        ready=True,
        execution_key="openai",
        api_key="secret",
        streaming=False,
    )

    messages = [{"role": "user", "content": "first"}]
    descriptors = [first]
    for content in (None, "second"):
        if content is not None:
            _message_id, descriptor = _saved_message(
                database,
                conversation_id,
                content,
            )
            messages.append({"role": "user", "content": content})
            descriptors.append(descriptor)
        prepared = gateway.prepare_chat_request(
            resolution,
            _semantic_request(messages, descriptors, policy),
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
        output = [
            item
            async for item in gateway.stream_chat(
                resolution,
                prepared,
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]
        assert output == ["ok"]

    with database.transaction() as cursor:
        assert cursor.execute("SELECT COUNT(*) FROM console_trace_calls").fetchone()[0] == 2
        assert cursor.execute("SELECT COUNT(*) FROM console_trace_surface_nodes").fetchone()[0] == 2
        assert cursor.execute("SELECT COUNT(*) FROM console_trace_owners").fetchone()[0] == 1
        assert cursor.execute("SELECT COUNT(*) FROM message_exchanges").fetchone()[0] == 0
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "missing",
        "foreign_turn",
        "changed_policy",
        "changed_actor",
        "stale_surface",
        "same_surface_new_run",
        "ambiguous",
    ],
)
async def test_tool_loop_uses_recorded_chain_and_rejects_changed_ownership(
    tmp_path,
    make_database,
    make_gateway,
    scenario,
):
    database = make_database(tmp_path / "tool-chain.sqlite", "tool-chain")
    conversation_id = database.add_conversation({"title": "tool chain"})
    user_id, revision = _saved_message(database, conversation_id, "calculate")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor, chain = new_opaque_id(), new_opaque_id()
    factory = ConsoleTraceBoundaryFactory(database)
    gateway = make_gateway(
        chat_api_call_fn=lambda **kwargs: {"choices": [{"message": {"content": "ok"}}]},
        trace_call_boundary_factory=factory,
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="openai",
        model="gpt-test",
        execution_key="openai",
        base_url="https://api.openai.com/v1",
        streaming=False,
    )
    messages = [{"role": "user", "content": "calculate"}]
    initial = _semantic_request(
        messages,
        [revision],
        policy,
        route=ConsoleRequestRoute.AGENT_FIRST,
        actor_id=actor,
        chain_id=chain,
    )
    prepared = gateway.prepare_chat_request(
        resolution,
        initial,
        route=ConsoleRequestRoute.AGENT_FIRST,
        route_actor_id=actor,
        route_chain_id=chain,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    assert [
        item
        async for item in gateway.stream_chat(
            resolution,
            prepared,
            route=ConsoleRequestRoute.AGENT_FIRST,
            route_actor_id=actor,
            route_chain_id=chain,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
    ] == ["ok"]
    with database.transaction() as cursor:
        origin_run_id = cursor.execute(
            "SELECT run_id FROM console_trace_calls WHERE call_sequence = 0"
        ).fetchone()[0]
    if scenario == "missing":
        chain = new_opaque_id()
    elif scenario == "foreign_turn":
        other = database.add_conversation({"title": "foreign"})
        _, revision = _saved_message(database, other, "calculate")
    elif scenario == "changed_policy":
        policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    elif scenario == "changed_actor":
        actor = new_opaque_id()
    elif scenario == "ambiguous":
        other = database.add_conversation({"title": "colliding chain"})
        other_id, _ = _saved_message(database, other, "other turn")
        with database.transaction() as cursor:
            segment = factory.repository.create_segment(cursor)
            owner = factory.repository.attach_owner(
                cursor, conversation_id=other, root_segment_id=segment.segment_id
            )
            factory.repository.reserve_call(
                cursor,
                owner_id=owner.owner_id,
                segment_id=segment.segment_id,
                turn_id=other_id,
                run_id=origin_run_id,
                call_sequence=0,
                idempotency_key=new_opaque_id(),
                policy_id=policy.policy_id,
            )
    elif scenario == "stale_surface":
        _, later = _saved_message(database, conversation_id, "new turn")
        next_request = _semantic_request(
            messages + [{"role": "user", "content": "new turn"}],
            [revision, later],
            policy,
        )
        next_prepared = gateway.prepare_chat_request(
            resolution,
            next_request,
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
        assert [
            item
            async for item in gateway.stream_chat(
                resolution,
                next_prepared,
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ] == ["ok"]
    elif scenario == "same_surface_new_run":
        newer_chain = new_opaque_id()
        newer_request = _semantic_request(
            messages,
            [revision],
            policy,
            route=ConsoleRequestRoute.AGENT_FIRST,
            actor_id=actor,
            chain_id=newer_chain,
        )
        newer_prepared = gateway.prepare_chat_request(
            resolution,
            newer_request,
            route=ConsoleRequestRoute.AGENT_FIRST,
            route_actor_id=actor,
            route_chain_id=newer_chain,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
        assert [
            item
            async for item in gateway.stream_chat(
                resolution,
                newer_prepared,
                route=ConsoleRequestRoute.AGENT_FIRST,
                route_actor_id=actor,
                route_chain_id=newer_chain,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ] == ["ok"]
    tool_result = {"role": "tool", "content": "323", "tool_call_id": "call-1"}
    loop_request = _semantic_request(
        messages + [tool_result],
        [
            revision,
            ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_RESULT, policy),
        ],
        policy,
        route=ConsoleRequestRoute.TOOL_LOOP,
        actor_id=actor,
        chain_id=chain,
    )
    loop_prepared = gateway.prepare_chat_request(
        resolution,
        loop_request,
        route=ConsoleRequestRoute.TOOL_LOOP,
        route_actor_id=actor,
        route_chain_id=chain,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    with database.transaction() as cursor:
        before = cursor.execute("SELECT COUNT(*) FROM console_trace_calls").fetchone()[
            0
        ]
    if scenario == "valid":
        # A new factory must recover the chain from SQLite, not a process cache.
        boundary = ConsoleTraceBoundaryFactory(database)(
            loop_prepared, resolution, ConsoleRequestRoute.TOOL_LOOP
        )
        assert boundary.identity.turn_id == user_id
        assert boundary.identity.run_id == origin_run_id
        assert boundary.identity.call_sequence == 1
    else:
        with pytest.raises(ValueError, match="trace_tool_chain_unavailable"):
            factory(loop_prepared, resolution, ConsoleRequestRoute.TOOL_LOOP)
        with database.transaction() as cursor:
            assert (
                cursor.execute("SELECT COUNT(*) FROM console_trace_calls").fetchone()[0]
                == before
            )


@pytest.mark.asyncio
async def test_dual_write_legacy_capture_reuses_normalized_call_identity(
    tmp_path,
    make_database,
    make_gateway,
) -> None:
    database = make_database(tmp_path / "trace-dual-write.sqlite", "trace-dual")
    conversation_id = database.add_conversation({"title": "dual write"})
    assert conversation_id is not None
    _message_id, revision = _saved_message(database, conversation_id, "question")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    factory = ConsoleTraceBoundaryFactory(database)
    signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)

    gateway = make_gateway(
        chat_api_call_fn=lambda **_kwargs: {
            "choices": [{"message": {"content": "answer"}}]
        },
        trace_call_boundary_factory=factory,
    )
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        ready=True,
        execution_key="openai",
        api_key="secret",
        streaming=False,
    )
    prepared = gateway.prepare_chat_request(
        resolution,
        _semantic_request(
            [{"role": "user", "content": "question"}],
            [revision],
            policy,
        ),
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )

    assert [
        item
        async for item in gateway.stream_chat(
            resolution,
            prepared,
            signals=signals,
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
    ] == ["answer"]

    with database.transaction() as cursor:
        normalized = factory.repository.read_conversation_call_lineage(
            cursor,
            conversation_id,
        )
    legacy = signals.exchange_captures()
    assert len(normalized) == len(legacy) == 1
    assert (legacy[0].run_tag, legacy[0].seq) == (
        normalized[0].run_id,
        normalized[0].call_sequence,
    )


@pytest.mark.asyncio
async def test_dual_write_llamacpp_fallback_reuses_both_normalized_identities(
    tmp_path,
    make_database,
) -> None:
    database = make_database(
        tmp_path / "trace-dual-write-fallback.sqlite",
        "trace-dual-fallback",
    )
    conversation_id = database.add_conversation({"title": "dual fallback"})
    assert conversation_id is not None
    _message_id, revision = _saved_message(database, conversation_id, "question")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    factory = ConsoleTraceBoundaryFactory(database)
    signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)

    def respond(request: httpx.Request) -> httpx.Response:
        if b'"stream":true' in request.content:
            return httpx.Response(200, text="")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "fallback"}}]},
        )

    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        base_url="http://localhost:8080",
        model="local-model",
        ready=True,
        execution_key="llama_cpp",
        streaming=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gateway = ConsoleProviderGateway(
            http_client=client,
            trace_call_boundary_factory=factory,
        )
        prepared = gateway.prepare_chat_request(
            resolution,
            _semantic_request(
                [{"role": "user", "content": "question"}],
                [revision],
                policy,
            ),
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )

        assert [
            item
            async for item in gateway.stream_chat(
                resolution,
                prepared,
                signals=signals,
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ] == ["fallback"]

    with database.transaction() as cursor:
        normalized = factory.repository.read_conversation_call_lineage(
            cursor,
            conversation_id,
        )
    legacy = signals.exchange_captures()
    assert len(normalized) == len(legacy) == 2
    assert {(item.run_tag, item.seq) for item in legacy} == {
        (item.run_id, item.call_sequence) for item in normalized
    }


@pytest.mark.asyncio
async def test_production_factory_persists_one_item_bounded_replacement(
    tmp_path,
    make_database,
    make_gateway,
) -> None:
    database = make_database(tmp_path / "trace-replacement.sqlite", "trace-replace")
    conversation_id = database.add_conversation({"title": "replacement trace"})
    assert conversation_id is not None
    saved = [
        _saved_message(database, conversation_id, content)[1]
        for content in ("old-1", "old-2", "keep")
    ]
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)

    def adapter(**_kwargs):
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = make_gateway(
        chat_api_call_fn=adapter,
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(database),
    )
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        ready=True,
        execution_key="openai",
        api_key="secret",
        streaming=False,
    )

    async def dispatch(messages, descriptors) -> None:
        prepared = gateway.prepare_chat_request(
            resolution,
            _semantic_request(messages, descriptors, policy),
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
        assert [
            item
            async for item in gateway.stream_chat(
                resolution,
                prepared,
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ] == ["ok"]

    await dispatch(
        [
            {"role": "user", "content": "old-1"},
            {"role": "user", "content": "old-2"},
            {"role": "user", "content": "keep"},
        ],
        saved,
    )
    await dispatch(
        [
            {"role": "user", "content": "summary"},
            {"role": "user", "content": "keep"},
        ],
        [
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.ACTIVE_REQUEST,
                policy,
            ),
            saved[-1],
        ],
    )

    with database.transaction() as cursor:
        replacement = cursor.execute(
            """SELECT start_sequence, end_sequence
                 FROM console_trace_surface_replacements"""
        ).fetchone()
        assert replacement is not None
        assert tuple(replacement) == (0, 1)
        assert cursor.execute("SELECT COUNT(*) FROM console_trace_surface_nodes").fetchone()[0] == 4


def test_production_factory_uses_latest_message_revision_for_turn_identity(
    tmp_path,
    make_database,
) -> None:
    database = make_database(tmp_path / "trace-turn-owner.sqlite", "trace-turn")
    conversation_id = database.add_conversation({"title": "turn owner"})
    assert conversation_id is not None
    prior_id, _prior_revision = _saved_message(
        database,
        conversation_id,
        "prior",
        sender="assistant",
    )
    current_id, current_revision = _saved_message(database, conversation_id, "current")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=(ContinuationRound("prior", ("reasoning",), ()),),
    )
    continuation = continuation_owner_group(
        {"id": prior_id, "role": "assistant", "content": "prior"},
        checkpoint,
    )
    assert database.update_provider_continuation(
        message_id=prior_id,
        expected_message_version=1,
        provider_continuation_json=dump_provider_continuation_json(checkpoint),
        content="prior",
    )
    with database.transaction() as cursor:
        prior_row = cursor.execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1""",
            (prior_id,),
        ).fetchone()
    assert prior_row is not None
    prior_revision = SavedRevisionTraceProvenance(str(prior_row[0]))
    semantic = build_console_request(
        [
            {
                "role": "assistant",
                "content": "prior",
                CONTINUATION_OWNER_KEY: prior_id,
            },
            {"role": "user", "content": "current"},
        ],
        continuation_groups=(continuation,),
        message_provenance=(prior_revision, current_revision),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        metadata_provenance=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        capture_policy=policy,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider="openai",
        model="gpt-test",
        capacity=resolve_request_capacity(context_window_tokens=None),
        apply_safety_window=False,
    )

    boundary = ConsoleTraceBoundaryFactory(database)(
        prepared,
        None,
        ConsoleRequestRoute.FRESH,
    )

    assert boundary.identity.turn_id == current_id


def test_production_factory_rejects_unsaved_active_message_instead_of_stale_turn(
    tmp_path,
    make_database,
) -> None:
    database = make_database(tmp_path / "trace-unsaved-active.sqlite", "trace-unsaved")
    conversation_id = database.add_conversation({"title": "unsaved active turn"})
    assert conversation_id is not None
    _prior_id, prior_revision = _saved_message(database, conversation_id, "prior")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    semantic = build_console_request(
        [
            {"role": "user", "content": "prior"},
            {"role": "user", "content": "active but unsaved"},
        ],
        message_provenance=(
            prior_revision,
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.ACTIVE_REQUEST,
                policy,
            ),
        ),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        metadata_provenance=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        capture_policy=policy,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider="openai",
        model="gpt-test",
        capacity=resolve_request_capacity(context_window_tokens=None),
        apply_safety_window=False,
    )

    with pytest.raises(ValueError, match="trace_turn_unavailable"):
        ConsoleTraceBoundaryFactory(database)(
            prepared,
            None,
            ConsoleRequestRoute.FRESH,
        )


def test_production_factory_batches_revision_owner_lookup_for_long_traces(
    tmp_path,
    make_database,
) -> None:
    database = make_database(tmp_path / "trace-long-owner.sqlite", "trace-long")
    conversation_id = database.add_conversation({"title": "long trace"})
    assert conversation_id is not None
    saved = [
        _saved_message(database, conversation_id, f"message-{index}")
        for index in range(257)
    ]
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    semantic = _semantic_request(
        [
            {"role": "user", "content": f"message-{index}"}
            for index in range(len(saved))
        ],
        [descriptor for _message_id, descriptor in saved],
        policy,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider="openai",
        model="gpt-test",
        capacity=resolve_request_capacity(context_window_tokens=None),
        apply_safety_window=False,
    )
    connection = database.get_connection()
    prior_limit = connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 256)
    try:
        boundary = ConsoleTraceBoundaryFactory(database)(
            prepared,
            None,
            ConsoleRequestRoute.FRESH,
        )
    finally:
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, prior_limit)

    assert boundary.identity.turn_id == saved[-1][0]


def test_recreated_factory_continues_durable_chain_sequence(
    tmp_path,
    make_database,
) -> None:
    database = make_database(tmp_path / "trace-chain-sequence.sqlite", "trace-chain")
    conversation_id = database.add_conversation({"title": "chain sequence"})
    assert conversation_id is not None
    _message_id, revision = _saved_message(database, conversation_id, "turn")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor_id = new_opaque_id()
    chain_id = new_opaque_id()
    semantic = _semantic_request(
        [{"role": "user", "content": "turn"}],
        [revision],
        policy,
        route=ConsoleRequestRoute.AGENT_FIRST,
        actor_id=actor_id,
        chain_id=chain_id,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider="openai",
        model="gpt-test",
        capacity=resolve_request_capacity(context_window_tokens=None),
        apply_safety_window=False,
    )

    first = ConsoleTraceBoundaryFactory(database)(
        prepared,
        None,
        ConsoleRequestRoute.AGENT_FIRST,
    )
    first.reserve()
    second = ConsoleTraceBoundaryFactory(database)(
        prepared,
        None,
        ConsoleRequestRoute.AGENT_FIRST,
    )

    assert first.identity.call_sequence == 0
    assert second.identity.call_sequence == 1
    second.reserve()


def test_recreated_factories_atomically_reserve_distinct_chain_sequences(
    tmp_path,
    make_database,
) -> None:
    database = make_database(tmp_path / "trace-chain-race.sqlite", "trace-chain-race")
    conversation_id = database.add_conversation({"title": "chain race"})
    assert conversation_id is not None
    _message_id, revision = _saved_message(database, conversation_id, "turn")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    chain_id = new_opaque_id()
    semantic = _semantic_request(
        [{"role": "user", "content": "turn"}],
        [revision],
        policy,
        route=ConsoleRequestRoute.AGENT_FIRST,
        actor_id=new_opaque_id(),
        chain_id=chain_id,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider="openai",
        model="gpt-test",
        capacity=resolve_request_capacity(context_window_tokens=None),
        apply_safety_window=False,
    )

    first = ConsoleTraceBoundaryFactory(database)(
        prepared,
        None,
        ConsoleRequestRoute.AGENT_FIRST,
    )
    second = ConsoleTraceBoundaryFactory(database)(
        prepared,
        None,
        ConsoleRequestRoute.AGENT_FIRST,
    )

    assert (first.identity.call_sequence, second.identity.call_sequence) == (0, 1)
    assert first.reserve().call_id != second.reserve().call_id


@pytest.mark.asyncio
async def test_production_factory_accepts_active_ancestor_revisions_on_nested_fork(
    tmp_path,
    make_database,
    make_gateway,
) -> None:
    database = make_database(tmp_path / "trace-fork-runtime.sqlite", "trace-fork")
    source_id = database.add_conversation({"title": "source"})
    child_id = database.add_conversation({"title": "child"})
    assert source_id is not None and child_id is not None
    source_message_id, source_revision = _saved_message(database, source_id, "source")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)

    def adapter(**_kwargs):
        return {"choices": [{"message": {"content": "ok"}}]}

    factory = ConsoleTraceBoundaryFactory(database)
    gateway = make_gateway(
        chat_api_call_fn=adapter,
        trace_call_boundary_factory=factory,
    )
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        ready=True,
        execution_key="openai",
        api_key="secret",
        streaming=False,
    )

    async def dispatch(messages, descriptors) -> None:
        prepared = gateway.prepare_chat_request(
            resolution,
            _semantic_request(messages, descriptors, policy),
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
        chunks = [
            item
            async for item in gateway.stream_chat(
                resolution,
                prepared,
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]
        assert chunks == ["ok"]

    await dispatch([{"role": "user", "content": "source"}], [source_revision])
    with database.transaction(immediate=True) as cursor:
        boundary = factory.repository.capture_fork_boundary(
            cursor,
            conversation_id=source_id,
            included_turn_ids=(source_message_id,),
        )
        assert boundary is not None
        child_owner = factory.repository.attach_fork_owner(
            cursor,
            conversation_id=child_id,
            boundary=boundary,
        )
    child_message_id, child_revision = _saved_message(database, child_id, "child")

    await dispatch(
        [
            {"role": "user", "content": "source"},
            {"role": "user", "content": "child"},
        ],
        [source_revision, child_revision],
    )

    with database.transaction() as cursor:
        calls = factory.repository.read_conversation_call_lineage(cursor, child_id)
        assert [call.turn_id for call in calls] == [source_message_id, child_message_id]
        assert calls[-1].owner_id == child_owner.owner_id

    grandchild_id = database.add_conversation({"title": "grandchild"})
    assert grandchild_id is not None
    with database.transaction(immediate=True) as cursor:
        nested_boundary = factory.repository.capture_fork_boundary(
            cursor,
            conversation_id=child_id,
            included_turn_ids=(child_message_id,),
        )
        assert nested_boundary is not None
        grandchild_owner = factory.repository.attach_fork_owner(
            cursor,
            conversation_id=grandchild_id,
            boundary=nested_boundary,
        )
    grandchild_message_id, grandchild_revision = _saved_message(
        database,
        grandchild_id,
        "grandchild",
    )

    await dispatch(
        [
            {"role": "user", "content": "source"},
            {"role": "user", "content": "child"},
            {"role": "user", "content": "grandchild"},
        ],
        [source_revision, child_revision, grandchild_revision],
    )

    with database.transaction() as cursor:
        calls = factory.repository.read_conversation_call_lineage(
            cursor,
            grandchild_id,
        )
        assert [call.turn_id for call in calls] == [
            source_message_id,
            child_message_id,
            grandchild_message_id,
        ]
        assert calls[-1].owner_id == grandchild_owner.owner_id

    summary = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST,
        policy,
    )
    await dispatch(
        [
            {"role": "user", "content": "summary"},
            {"role": "user", "content": "child"},
            {"role": "user", "content": "grandchild"},
        ],
        [summary, child_revision, grandchild_revision],
    )

    with database.transaction() as cursor:
        replacements = factory.repository.read_surface_replacements(
            cursor,
            grandchild_owner.root_segment_id,
        )
        assert len(replacements) == 1
        assert replacements[0].replacement.start_sequence == 0
        assert replacements[0].replacement.end_sequence == 0
