from __future__ import annotations

import sqlite3
from types import SimpleNamespace

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
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
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


def test_console_runtime_wires_production_boundary_for_durable_database(tmp_path) -> None:
    database = CharactersRAGDB(tmp_path / "trace-runtime-wiring.sqlite", "trace-wiring")
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=database))

    gateway = runtime.ensure_provider_gateway(config_provider=lambda: {})

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


@pytest.mark.asyncio
async def test_production_factory_persists_append_only_calls_through_real_gateway(
    tmp_path,
) -> None:
    database = CharactersRAGDB(tmp_path / "trace-runtime.sqlite", "trace-runtime")
    conversation_id = database.add_conversation({"title": "runtime trace"})
    assert conversation_id is not None
    _first_id, first = _saved_message(database, conversation_id, "first")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    calls: list[dict[str, object]] = []

    def adapter(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    factory = ConsoleTraceBoundaryFactory(database)
    gateway = ConsoleProviderGateway(
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
    assert factory._chain_sequences == {}
    assert len(calls) == 2
    await gateway.aclose()


@pytest.mark.asyncio
async def test_production_factory_persists_one_item_bounded_replacement(
    tmp_path,
) -> None:
    database = CharactersRAGDB(tmp_path / "trace-replacement.sqlite", "trace-replace")
    conversation_id = database.add_conversation({"title": "replacement trace"})
    assert conversation_id is not None
    saved = [
        _saved_message(database, conversation_id, content)[1]
        for content in ("old-1", "old-2", "keep")
    ]
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)

    def adapter(**_kwargs):
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = ConsoleProviderGateway(
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
    await gateway.aclose()


def test_production_factory_uses_latest_message_revision_for_turn_identity(
    tmp_path,
) -> None:
    database = CharactersRAGDB(tmp_path / "trace-turn-owner.sqlite", "trace-turn")
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


def test_production_factory_batches_revision_owner_lookup_for_long_traces(
    tmp_path,
) -> None:
    database = CharactersRAGDB(tmp_path / "trace-long-owner.sqlite", "trace-long")
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


def test_recreated_factory_continues_durable_chain_sequence(tmp_path) -> None:
    database = CharactersRAGDB(tmp_path / "trace-chain-sequence.sqlite", "trace-chain")
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


@pytest.mark.asyncio
async def test_production_factory_accepts_active_ancestor_revisions_on_nested_fork(
    tmp_path,
) -> None:
    database = CharactersRAGDB(tmp_path / "trace-fork-runtime.sqlite", "trace-fork")
    source_id = database.add_conversation({"title": "source"})
    child_id = database.add_conversation({"title": "child"})
    assert source_id is not None and child_id is not None
    source_message_id, source_revision = _saved_message(database, source_id, "source")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)

    def adapter(**_kwargs):
        return {"choices": [{"message": {"content": "ok"}}]}

    factory = ConsoleTraceBoundaryFactory(database)
    gateway = ConsoleProviderGateway(
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
    await gateway.aclose()
