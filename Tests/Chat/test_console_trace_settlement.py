"""Independent provider-response settlement and cold trace recovery."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import threading

import pytest

from tldw_chatbook.Chat import console_trace_settlement as settlement_module
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.Chat.console_prepared_request import build_console_request
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    SavedRevisionTraceProvenance,
    request_route_provenance,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_runtime import recover_console_trace_calls
from tldw_chatbook.Chat.console_semantic_revision import SemanticRevisionCoordinator
from tldw_chatbook.Chat.console_trace_settlement import (
    ConsoleTraceSettlementCoordinator,
    MAX_TRACE_RESPONSE_BYTES,
    TraceSettlementRequest,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(
        str(tmp_path / "trace-settlement.sqlite"),
        "trace-settlement-test",
    )
    yield database
    database.close_connection()


def _call(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    *,
    sequence: int = 0,
    state: TraceCallState = TraceCallState.DISPATCH_STARTED,
    conversation_id: str | None = None,
) -> tuple[str, str, str]:
    conversation_id = conversation_id or db.add_conversation({"title": "settlement"})
    assert conversation_id is not None
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "request",
        }
    )
    assert message_id is not None
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    with db.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
        repository.ensure_policy(cursor, policy)
        revision_row = cursor.execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ? AND live_message_id = ?
                 ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id, message_id),
        ).fetchone()
        assert revision_row is not None
        node = repository.append_surface_node(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="message",
            reference=SemanticRevisionRef(str(revision_row[0])),
        )
        header = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="fresh",
            endpoint_identity="public-api",
            generation_parameters={},
            adapter_defaults={},
            response_format={},
            reasoning_controls={},
            components=(),
        )
        call = repository.reserve_call(
            cursor,
            owner_id=owner.owner_id,
            segment_id=segment.segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=sequence,
            idempotency_key=new_opaque_id(),
            policy_id=policy.policy_id,
        )
        call = repository.bind_call(
            cursor,
            call_id=call.call_id,
            surface_node_id=node.node_id,
            request_header_id=header.header_id,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="fresh",
        )
        if state is TraceCallState.DISPATCH_STARTED:
            call = repository.advance_call_state(
                cursor,
                call_id=call.call_id,
                target=state,
                occurred_at="2026-08-30T01:00:00Z",
            )
        elif state is TraceCallState.RESPONSE_STARTED:
            call = repository.advance_call_state(
                cursor,
                call_id=call.call_id,
                target=TraceCallState.DISPATCH_STARTED,
                occurred_at="2026-08-30T01:00:00Z",
            )
            call = repository.advance_call_state(
                cursor,
                call_id=call.call_id,
                target=state,
                occurred_at="2026-08-30T01:00:01Z",
            )
    return conversation_id, segment.segment_id, call.call_id


def _request(
    call_id: str,
    *,
    outcome: TraceCallState = TraceCallState.COMPLETE,
    response: object | None = None,
    usage: dict[str, object] | None = None,
    canonical_message_id: str | None = None,
) -> TraceSettlementRequest:
    return TraceSettlementRequest(
        call_id=call_id,
        outcome=outcome,
        response_envelope=(
            {"role": "assistant", "content": "answer"}
            if response is None and outcome is not TraceCallState.ERROR
            else response
        ),
        usage=usage,
        response_started_at="2026-08-30T01:00:01Z",
        settled_at="2026-08-30T01:00:02Z",
        canonical_message_id=canonical_message_id,
    )


def _captured_request(
    gateway: ConsoleProviderGateway,
    resolution: ConsoleProviderResolution,
):
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    semantic = build_console_request(
        [{"role": "user", "content": "question"}],
        message_provenance=(SavedRevisionTraceProvenance(new_opaque_id()),),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        metadata_provenance=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        capture_policy=policy,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    return gateway.prepare_chat_request(
        resolution,
        semantic,
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )


class _CoordinatorBoundary:
    def __init__(
        self,
        coordinator: ConsoleTraceSettlementCoordinator,
        database: CharactersRAGDB,
        call_id: str,
    ) -> None:
        self.coordinator = coordinator
        self.database = database
        self.call_id = call_id
        self.requests: list[TraceSettlementRequest] = []
        self.results: list[bool] = []

    def reserve(self) -> None:
        return None

    def mark_dispatch_started(self, _bundle, _provenance) -> None:
        return None

    def mark_response_started(self) -> None:
        self.coordinator.mark_response_started(
            self.database,
            call_id=self.call_id,
            occurred_at="2026-08-30T01:00:01Z",
        )

    def settle_response(self, response, outcome, usage=None) -> None:
        request = TraceSettlementRequest(
            call_id=self.call_id,
            outcome=outcome,
            response_envelope=response,
            usage=usage,
            response_started_at="2026-08-30T01:00:01Z",
            settled_at="2026-08-30T01:00:02Z",
        )
        self.requests.append(request)
        self.results.append(self.coordinator.submit(self.database, request))


def test_response_started_complete_usage_and_duplicate_settlement_are_idempotent(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, segment_id, call_id = _call(db, repository)
    request = _request(
        call_id,
        usage={"prompt_tokens": 3, "completion_tokens": 2},
    )

    started = coordinator.mark_response_started(
        db, call_id=call_id, occurred_at=request.response_started_at
    )
    first = coordinator.settle(db, request)
    repeated = coordinator.settle(db, request)

    assert started.state is TraceCallState.RESPONSE_STARTED
    assert first == repeated
    assert first.state is TraceCallState.COMPLETE
    assert first.usage == {"prompt_tokens": 3, "completion_tokens": 2}
    cursor = db.get_connection().cursor()
    link = repository.get_response_link(cursor, call_id)
    assert link is not None and link.link_kind == "artifact"
    events = repository.read_events(cursor, segment_id)
    assert [event.event_type for event in events] == [
        "response_selection",
        "call_outcome",
        "usage",
    ]


def test_post_terminal_conflicting_submission_is_rejected_without_queue_poison(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository, max_pending=2)
    _conversation_id, segment_id, call_id = _call(db, repository)
    first = _request(call_id)

    assert coordinator.submit(db, first) is True
    cursor = db.get_connection().cursor()
    original_link = repository.get_response_link(cursor, call_id)
    original_events = repository.read_events(cursor, segment_id)
    original_artifact_count = cursor.execute(
        "SELECT COUNT(*) FROM console_trace_artifacts"
    ).fetchone()[0]

    conflicting = replace(
        first,
        response_envelope={"role": "assistant", "content": "different"},
    )
    assert coordinator.submit(db, conflicting) is False
    assert coordinator.pending_count == 0
    assert coordinator.retry_pending() == 0
    assert repository.get_response_link(cursor, call_id) == original_link
    assert repository.read_events(cursor, segment_id) == original_events
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == original_artifact_count
    )


def test_queued_retry_drops_conflict_after_another_coordinator_settles_call(
    db: CharactersRAGDB,
) -> None:
    class BlockingRepository(ConsoleTraceRepository):
        blocked = True

        def store_response_link(self, *args, **kwargs):
            if self.blocked:
                raise RuntimeError("PRIVATE-RESPONSE-CANARY")
            return super().store_response_link(*args, **kwargs)

    repository = BlockingRepository()
    queued = ConsoleTraceSettlementCoordinator(repository, max_pending=2)
    external = ConsoleTraceSettlementCoordinator(repository, max_pending=2)
    _conversation_id, segment_id, call_id = _call(db, repository)
    first_signal = _request(call_id)
    durable_signal = replace(
        first_signal,
        response_envelope={"role": "assistant", "content": "external"},
    )

    assert queued.submit(db, first_signal) is False
    assert queued.pending_count == 1
    repository.blocked = False
    assert external.submit(db, durable_signal) is True
    cursor = db.get_connection().cursor()
    durable_link = repository.get_response_link(cursor, call_id)
    durable_events = repository.read_events(cursor, segment_id)

    assert queued.retry_pending() == 0
    assert queued.pending_count == 0
    assert repository.get_response_link(cursor, call_id) == durable_link
    assert repository.read_events(cursor, segment_id) == durable_events


@pytest.mark.asyncio
async def test_generic_cancel_after_dispatch_before_response_settles_unknown_once(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, segment_id, call_id = _call(db, repository)
    boundary = _CoordinatorBoundary(coordinator, db, call_id)
    entered = threading.Event()
    release = threading.Event()

    def blocked_response(**_kwargs):
        entered.set()
        assert release.wait(timeout=3)
        if False:
            yield "unreachable"

    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        ready=True,
        execution_key="openai",
        streaming=True,
    )
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=blocked_response,
        trace_call_boundary_factory=lambda _request, _resolution, _route: boundary,
    )
    prepared = _captured_request(gateway, resolution)
    response = gateway.stream_chat(
        resolution,
        prepared,
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    pending = asyncio.create_task(anext(response))
    assert await asyncio.to_thread(entered.wait, 3)

    pending.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await response.aclose()

    call = repository.get_call(db.get_connection().cursor(), call_id)
    assert call is not None
    assert call.state is TraceCallState.DISPATCH_UNKNOWN
    assert call.response_started_at is None
    assert repository.get_response_link(db.get_connection().cursor(), call_id) is None
    assert coordinator.pending_count == 0
    assert boundary.results == [True]
    assert coordinator.submit(db, boundary.requests[0]) is True
    assert coordinator.pending_count == 0
    assert [
        event.event_type
        for event in repository.read_events(db.get_connection().cursor(), segment_id)
    ] == ["call_outcome"]


@pytest.mark.asyncio
async def test_direct_cancel_after_dispatch_before_response_settles_unknown_once(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, _segment_id, call_id = _call(db, repository)
    boundary = _CoordinatorBoundary(coordinator, db, call_id)
    entered = asyncio.Event()

    async def blocked_stream(_self, **kwargs):
        await kwargs["before_adapter"]()
        entered.set()
        await asyncio.Future()
        yield "unreachable"

    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        base_url="http://127.0.0.1:9099",
        model="gpt-test",
        ready=True,
        execution_key="llama_cpp",
        streaming=True,
    )
    gateway = ConsoleProviderGateway(
        trace_call_boundary_factory=lambda _request, _resolution, _route: boundary,
    )
    gateway.stream_llamacpp_chat = blocked_stream.__get__(gateway)
    prepared = _captured_request(gateway, resolution)
    response = gateway.stream_chat(
        resolution,
        prepared,
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    pending = asyncio.create_task(anext(response))
    await entered.wait()

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await response.aclose()

    call = repository.get_call(db.get_connection().cursor(), call_id)
    assert call is not None
    assert call.state is TraceCallState.DISPATCH_UNKNOWN
    assert call.response_started_at is None
    assert repository.get_response_link(db.get_connection().cursor(), call_id) is None
    assert coordinator.pending_count == 0
    assert boundary.results == [True]
    assert coordinator.submit(db, boundary.requests[0]) is True
    assert coordinator.pending_count == 0


@pytest.mark.parametrize(
    "outcome",
    [TraceCallState.STOPPED, TraceCallState.ERROR, TraceCallState.INTERRUPTED],
)
def test_response_started_accepts_each_terminal_outcome(
    db: CharactersRAGDB,
    outcome: TraceCallState,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, _segment_id, call_id = _call(
        db, repository, state=TraceCallState.RESPONSE_STARTED
    )

    settled = coordinator.settle(db, _request(call_id, outcome=outcome))

    assert settled.state is outcome
    assert settled.outcome == outcome.value


def test_terminal_retry_fingerprints_response_omission_and_rejects_genuine_none(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, _segment_id, call_id = _call(db, repository)
    recursive: list[object] = []
    recursive.append(recursive)
    omitted = replace(
        _request(call_id),
        response_envelope={"content": recursive},
    )

    first = coordinator.settle(db, omitted)
    repeated = coordinator.settle(db, omitted)

    assert repeated == first
    assert first.integrity_state == "incomplete"
    assert first.omission_reason_code == "credential_sanitizer_unavailable"
    with pytest.raises(ValueError, match="settlement_integrity_conflict"):
        coordinator.settle(db, replace(omitted, response_envelope=None))


def test_terminal_retry_fingerprints_usage_omission_and_rejects_genuine_none(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, _segment_id, call_id = _call(db, repository)
    recursive: list[object] = []
    recursive.append(recursive)
    omitted = replace(
        _request(call_id),
        usage={"provider_detail": recursive},
    )

    first = coordinator.settle(db, omitted)
    repeated = coordinator.settle(db, omitted)

    assert repeated == first
    assert first.integrity_state == "incomplete"
    assert first.omission_reason_code == "usage_credential_sanitizer_unavailable"
    with pytest.raises(ValueError, match="settlement_integrity_conflict"):
        coordinator.settle(db, replace(omitted, usage=None))


@pytest.mark.asyncio
async def test_gateway_invalid_unicode_usage_is_omitted_without_stranding_response(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, _segment_id, call_id = _call(db, repository)
    boundary = _CoordinatorBoundary(coordinator, db, call_id)
    surrogate = "\ud800"
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        ready=True,
        execution_key="openai",
        streaming=False,
    )
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"detail": surrogate},
        },
        trace_call_boundary_factory=lambda _request, _resolution, _route: boundary,
    )
    prepared = _captured_request(gateway, resolution)

    output = [
        item
        async for item in gateway.stream_chat(
            resolution,
            prepared,
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
    ]

    assert output == ["answer"]
    call = repository.get_call(db.get_connection().cursor(), call_id)
    assert call is not None
    assert call.state is TraceCallState.COMPLETE
    assert call.usage is None
    assert call.integrity_state == "incomplete"
    assert call.omission_reason_code == "usage_canonicalization_unavailable"
    assert (
        repository.get_response_link(db.get_connection().cursor(), call_id) is not None
    )
    assert coordinator.pending_count == 0
    assert surrogate not in repr(boundary.requests)
    assert coordinator.submit(db, boundary.requests[0]) is True
    assert coordinator.pending_count == 0


def test_invalid_unicode_response_fails_closed_to_content_free_omission(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, _segment_id, call_id = _call(db, repository)
    surrogate = "\ud800"
    request = replace(
        _request(call_id),
        response_envelope={"role": "assistant", "content": surrogate},
    )

    assert coordinator.submit(db, request) is True

    call = repository.get_call(db.get_connection().cursor(), call_id)
    assert call is not None
    assert call.state is TraceCallState.COMPLETE
    assert call.integrity_state == "incomplete"
    assert call.omission_reason_code == "response_canonicalization_unavailable"
    assert repository.get_response_link(db.get_connection().cursor(), call_id) is None
    assert coordinator.pending_count == 0
    assert surrogate not in repr(request)
    assert coordinator.submit(db, request) is True
    assert coordinator.pending_count == 0


def test_duplicate_revision_link_settlement_survives_canonical_message_edit(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    conversation_id, _segment_id, call_id = _call(db, repository)
    assistant_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "answer",
        }
    )
    assert assistant_id is not None
    request = _request(
        call_id,
        canonical_message_id=assistant_id,
    )
    first = coordinator.settle(db, request)
    link = repository.get_response_link(db.get_connection().cursor(), call_id)
    assert link is not None and link.semantic_revision_id is not None
    revision_id = link.semantic_revision_id

    semantic = SemanticRevisionCoordinator(db, repository=repository)
    with db.transaction(immediate=True) as cursor:
        semantic.mutate_message(
            cursor,
            message_id=assistant_id,
            creation_reason="message_edit",
            mutate=lambda scoped: scoped.execute(
                "UPDATE messages SET content = 'edited' WHERE id = ?",
                (assistant_id,),
            ),
        )

    cursor = db.get_connection().cursor()
    revision = repository.get_semantic_revision(cursor, revision_id)
    binding = repository.get_revision_policy_binding(
        cursor,
        revision_id=revision_id,
        policy_id=first.policy_id,
    )
    assert revision is not None and revision.live_message_id is None
    assert binding is not None and binding.artifact_id is not None
    historical = repository.get_artifact(cursor, binding.artifact_id)
    assert historical is not None
    assert json.loads(historical.sanitized_bytes)["content"] == "answer"
    assert coordinator.submit(db, request) is True
    assert coordinator.pending_count == 0
    assert repository.get_call(cursor, call_id).state is TraceCallState.COMPLETE  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="settlement_response_conflict"):
        coordinator.settle(
            db,
            replace(
                request,
                response_envelope={"role": "assistant", "content": "different"},
            ),
        )


def test_provider_error_without_response_settles_directly_from_dispatch_started(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, _segment_id, call_id = _call(db, repository)

    settled = coordinator.settle(
        db,
        _request(call_id, outcome=TraceCallState.ERROR, response=None),
    )

    assert settled.state is TraceCallState.ERROR
    assert settled.response_started_at is None
    assert repository.get_response_link(db.get_connection().cursor(), call_id) is None


def test_exact_saved_assistant_links_revision_but_mismatch_stores_one_artifact(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    persistence = ChatPersistenceService(db)
    conversation_id, _segment_id, exact_call_id = _call(db, repository)
    assistant_id = persistence.create_message(
        conversation_id=conversation_id,
        sender="assistant",
        content="answer",
        image_data=None,
        image_mime_type=None,
    )

    assert persistence.settle_provider_response_trace(
        coordinator=coordinator,
        request=_request(exact_call_id),
        canonical_message_id=assistant_id,
    )
    exact = repository.get_call(db.get_connection().cursor(), exact_call_id)
    assert exact is not None
    cursor = db.get_connection().cursor()
    exact_link = repository.get_response_link(cursor, exact.call_id)
    assert exact_link is not None
    assert exact_link.link_kind == "revision"
    assert exact_link.verification_outcome == "verified_equal"
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 0
    )

    exact_record = repository.get_call(cursor, exact_call_id)
    assert exact_record is not None
    with db.transaction(immediate=True) as transaction:
        call = repository.reserve_call(
            transaction,
            owner_id=exact_record.owner_id,
            segment_id=exact_record.segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=1,
            idempotency_key=new_opaque_id(),
            policy_id=exact_record.policy_id,
        )
        call = repository.bind_call(
            transaction,
            call_id=call.call_id,
            surface_node_id=str(exact_record.surface_node_id),
            request_header_id=str(exact_record.request_header_id),
            provider_name="openai",
            model_name="gpt-test",
            route_identity="fresh",
        )
        repository.advance_call_state(
            transaction,
            call_id=call.call_id,
            target=TraceCallState.DISPATCH_STARTED,
            occurred_at="2026-08-30T01:00:00Z",
        )
    mismatch = coordinator.settle(
        db,
        _request(
            call.call_id,
            response={"role": "assistant", "content": "different"},
            canonical_message_id=assistant_id,
        ),
    )
    mismatch_link = repository.get_response_link(cursor, mismatch.call_id)
    assert mismatch_link is not None
    assert mismatch_link.link_kind == "artifact"
    assert mismatch_link.verification_outcome == "sanitized_artifact"
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 1
    )


def test_response_remains_trace_owned_when_canonical_persistence_is_unavailable(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, _segment_id, call_id = _call(db, repository)

    persistence = ChatPersistenceService(db)
    assert persistence.settle_provider_response_trace(
        coordinator=coordinator,
        request=_request(call_id),
        canonical_message_id=None,
    )
    settled = repository.get_call(db.get_connection().cursor(), call_id)
    assert settled is not None

    link = repository.get_response_link(db.get_connection().cursor(), settled.call_id)
    assert link is not None and link.link_kind == "artifact"
    artifact = repository.get_artifact(
        db.get_connection().cursor(), str(link.artifact_id)
    )
    assert artifact is not None
    assert json.loads(artifact.sanitized_bytes) == {
        "content": "answer",
        "role": "assistant",
    }


@pytest.mark.asyncio
async def test_gateway_handoff_links_exact_store_persistence_without_artifact(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    persistence = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
        persist=True,
    )
    assert session.persisted_conversation_id is not None
    _conversation_id, _segment_id, call_id = _call(
        db,
        repository,
        conversation_id=session.persisted_conversation_id,
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )

    class Boundary:
        def reserve(self) -> None:
            return None

        def mark_dispatch_started(self, _bundle, _provenance) -> None:
            return None

        def mark_response_started(self) -> None:
            coordinator.mark_response_started(
                db,
                call_id=call_id,
                occurred_at="2026-08-30T01:00:01Z",
            )

        def prepare_response_settlement(self, response, outcome, usage=None):
            return coordinator.prepare_handoff(
                db,
                _request(
                    call_id,
                    outcome=outcome,
                    response=response,
                    usage=usage,
                ),
            )

    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        ready=True,
        execution_key="openai",
        streaming=False,
    )
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: {
            "choices": [{"message": {"content": "answer"}}]
        },
        trace_call_boundary_factory=lambda _request, _resolution, _route: Boundary(),
    )
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    semantic = build_console_request(
        [{"role": "user", "content": "question"}],
        message_provenance=(SavedRevisionTraceProvenance(new_opaque_id()),),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        metadata_provenance=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        capture_policy=policy,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    prepared = gateway.prepare_chat_request(
        resolution,
        semantic,
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    signals = ConsoleProviderStreamSignals()
    signals.bind_trace_settlement_sink(
        lambda handoff: store.register_provider_trace_settlement(
            assistant.id,
            handoff,
        )
    )

    output = [
        item
        async for item in gateway.stream_chat(
            resolution,
            prepared,
            signals=signals,
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
    ]
    for item in output:
        assert isinstance(item, str)
        store.append_stream_chunk(assistant.id, item)
    completed = store.mark_message_complete(assistant.id)

    assert completed.persisted_message_id is not None
    cursor = db.get_connection().cursor()
    link = repository.get_response_link(cursor, call_id)
    assert link is not None and link.link_kind == "revision"
    assert link.artifact_id is None
    assert store.pending_provider_trace_settlement_count(assistant.id) == 0


def test_store_handoff_mismatch_and_persistence_failure_stay_trace_owned(
    db: CharactersRAGDB,
) -> None:
    class FailingAssistantPersistence(ChatPersistenceService):
        fail_assistant = False

        def create_message(self, **kwargs):
            if self.fail_assistant and kwargs.get("sender") == "assistant":
                raise RuntimeError("assistant persistence unavailable")
            return super().create_message(**kwargs)

    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    persistence = FailingAssistantPersistence(db)
    for body, fail in (("edited answer", False), ("answer", True)):
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session()
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="question",
            persist=True,
        )
        assert session.persisted_conversation_id is not None
        _conversation_id, _segment_id, call_id = _call(
            db,
            repository,
            conversation_id=session.persisted_conversation_id,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        handoff = coordinator.prepare_handoff(db, _request(call_id))
        store.register_provider_trace_settlement(assistant.id, handoff)
        store.register_provider_trace_settlement(assistant.id, handoff)
        assert store.pending_provider_trace_settlement_count(assistant.id) == 1
        store.append_stream_chunk(assistant.id, body)
        persistence.fail_assistant = fail

        if fail:
            with pytest.raises(RuntimeError, match="assistant persistence unavailable"):
                store.mark_message_complete(assistant.id)
        else:
            store.mark_message_complete(assistant.id)

        link = repository.get_response_link(db.get_connection().cursor(), call_id)
        assert link is not None and link.link_kind == "artifact"
        assert store.pending_provider_trace_settlement_count(assistant.id) == 0
        persistence.fail_assistant = False


@pytest.mark.parametrize("cleanup", ["close", "restore", "teardown"])
def test_store_cleanup_trace_owns_pending_stream_handoff_idempotently(
    db: CharactersRAGDB,
    cleanup: str,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
        persist=True,
    )
    assert session.persisted_conversation_id is not None
    _conversation_id, _segment_id, call_id = _call(
        db,
        repository,
        conversation_id=session.persisted_conversation_id,
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(assistant.id, "partial")
    handoff = coordinator.prepare_handoff(
        db,
        _request(
            call_id,
            response={
                "role": "assistant",
                "content": "partial",
                "api_key": "sk-live-cleanup-canary",
            },
        ),
    )
    store.register_provider_trace_settlement(assistant.id, handoff)
    assert store.pending_provider_trace_settlement_count(assistant.id) == 1

    if cleanup == "close":
        store.close_session(session.id)
        store.end_app_runtime()
        store.end_app_runtime()
    elif cleanup == "restore":
        store.restore_state(sessions=[])
        store.restore_state(sessions=[])
    else:
        store.end_app_runtime()
        store.end_app_runtime()

    cursor = db.get_connection().cursor()
    call = repository.get_call(cursor, call_id)
    assert call is not None and call.state is TraceCallState.COMPLETE
    link = repository.get_response_link(cursor, call_id)
    assert link is not None and link.link_kind == "artifact"
    artifact = repository.get_artifact(cursor, str(link.artifact_id))
    assert artifact is not None
    assert json.loads(artifact.sanitized_bytes) == {
        "content": "partial",
        "role": "assistant",
    }
    assert b"sk-live-cleanup-canary" not in artifact.sanitized_bytes
    assert store.pending_provider_trace_settlement_count(assistant.id) == 0


@pytest.mark.parametrize("cleanup", ["close", "restore", "teardown"])
def test_store_cleanup_fences_handoff_registration_racing_after_drain(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: str,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
        persist=True,
    )
    assert session.persisted_conversation_id is not None
    _conversation_id, _segment_id, call_id = _call(
        db,
        repository,
        conversation_id=session.persisted_conversation_id,
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    handoff = coordinator.prepare_handoff(db, _request(call_id))
    drained = threading.Event()
    release_cleanup = threading.Event()
    original_drain = store._settle_provider_trace_settlements_for_messages

    def pause_after_drain(message_ids) -> None:
        original_drain(message_ids)
        drained.set()
        assert release_cleanup.wait(5)

    monkeypatch.setattr(
        store,
        "_settle_provider_trace_settlements_for_messages",
        pause_after_drain,
    )
    cleanup_errors: list[BaseException] = []

    def cleanup_store() -> None:
        try:
            if cleanup == "close":
                store.close_session(session.id)
            elif cleanup == "restore":
                store.restore_state(sessions=[])
            else:
                store.end_app_runtime()
        except BaseException as exc:  # pragma: no cover - asserted below
            cleanup_errors.append(exc)

    cleanup_thread = threading.Thread(target=cleanup_store)
    cleanup_thread.start()
    assert drained.wait(5)

    store.register_provider_trace_settlement(assistant.id, handoff)
    release_cleanup.set()
    cleanup_thread.join(5)

    assert not cleanup_thread.is_alive()
    assert cleanup_errors == []
    call = repository.get_call(db.get_connection().cursor(), call_id)
    assert call is not None and call.state is TraceCallState.COMPLETE
    assert store.pending_provider_trace_settlement_count(assistant.id) == 0


def test_sanitizer_failure_seals_content_free_incomplete_outcome(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, _segment_id, call_id = _call(db, repository)
    recursive: list[object] = []
    recursive.append(recursive)

    settled = coordinator.settle(
        db,
        _request(call_id, response={"content": recursive}),
    )

    assert settled.state is TraceCallState.COMPLETE
    assert settled.integrity_state == "incomplete"
    assert settled.omission_reason_code == "credential_sanitizer_unavailable"
    cursor = db.get_connection().cursor()
    assert repository.get_response_link(cursor, call_id) is None
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 0
    )


def test_oversized_response_is_replaced_by_one_bounded_labeled_artifact(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _conversation_id, _segment_id, call_id = _call(db, repository)

    settled = coordinator.settle(
        db,
        _request(call_id, response={"content": "x" * MAX_TRACE_RESPONSE_BYTES}),
    )

    assert settled.integrity_state == "incomplete"
    assert settled.omission_reason_code == "response_size_limit"
    cursor = db.get_connection().cursor()
    link = repository.get_response_link(cursor, call_id)
    assert link is not None and link.link_kind == "artifact"
    artifact = repository.get_artifact(cursor, str(link.artifact_id))
    assert artifact is not None
    assert len(artifact.sanitized_bytes) <= MAX_TRACE_RESPONSE_BYTES
    assert json.loads(artifact.sanitized_bytes) == {
        "byte_length": MAX_TRACE_RESPONSE_BYTES + len('{"content":""}'),
        "omitted": True,
        "reason": "response_size_limit",
    }


def test_queued_settlement_drops_oversized_response_and_usage_values(
    db: CharactersRAGDB,
) -> None:
    class FailingRepository(ConsoleTraceRepository):
        def store_response_link(self, *args, **kwargs):
            raise RuntimeError("force bounded retry")

    repository = FailingRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository, max_pending=2)
    _conversation_id, _segment_id, call_id = _call(db, repository)
    response_canary = "QUEUED-RESPONSE-CANARY"
    usage_canary = "QUEUED-USAGE-CANARY"
    request = _request(
        call_id,
        response={
            "content": "x" * MAX_TRACE_RESPONSE_BYTES + response_canary,
        },
        usage={
            "provider_detail": "y" * settlement_module.MAX_TRACE_USAGE_BYTES
            + usage_canary,
        },
    )

    assert coordinator.submit(db, request) is False
    assert coordinator.pending_count == 1
    _database, prepared = next(iter(coordinator._pending.values()))
    assert prepared.response_bytes is not None
    assert len(prepared.response_bytes) <= MAX_TRACE_RESPONSE_BYTES
    assert response_canary.encode() not in prepared.response_bytes
    assert prepared.response_omission == "response_size_limit"
    assert prepared.usage is None
    assert prepared.usage_omission == "usage_size_limit"
    assert usage_canary not in repr(prepared)


def test_failed_post_dispatch_seal_queues_and_retries_without_rolling_back_result(
    db: CharactersRAGDB,
) -> None:
    class FailOnceRepository(ConsoleTraceRepository):
        failed = False

        def store_response_link(self, *args, **kwargs):
            if not self.failed:
                self.failed = True
                raise RuntimeError("PRIVATE-RESPONSE-CANARY")
            return super().store_response_link(*args, **kwargs)

    repository = FailOnceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository, max_pending=4)
    _conversation_id, _segment_id, call_id = _call(db, repository)
    request = _request(call_id)

    assert coordinator.submit(db, request) is False
    assert coordinator.pending_count == 1
    assert (
        repository.get_call(db.get_connection().cursor(), call_id).state
        is TraceCallState.DISPATCH_STARTED
    )  # type: ignore[union-attr]

    assert coordinator.retry_pending() == 1
    assert coordinator.pending_count == 0
    assert (
        repository.get_call(db.get_connection().cursor(), call_id).state
        is TraceCallState.COMPLETE
    )  # type: ignore[union-attr]


def test_exact_pending_submission_retries_after_repository_fault_clears(
    db: CharactersRAGDB,
) -> None:
    class BlockingRepository(ConsoleTraceRepository):
        blocked = True

        def store_response_link(self, *args, **kwargs):
            if self.blocked:
                raise RuntimeError("PRIVATE-RESPONSE-CANARY")
            return super().store_response_link(*args, **kwargs)

    repository = BlockingRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository, max_pending=2)
    _conversation_id, segment_id, call_id = _call(db, repository)
    request = _request(call_id)

    assert coordinator.submit(db, request) is False
    assert coordinator.pending_count == 1
    repository.blocked = False

    assert coordinator.submit(db, request) is True
    assert coordinator.pending_count == 0
    cursor = db.get_connection().cursor()
    settled = repository.get_call(cursor, call_id)
    assert settled is not None and settled.state is TraceCallState.COMPLETE
    events = repository.read_events(cursor, segment_id)
    event_ids = tuple(event.event_id for event in events)
    artifact_count = cursor.execute(
        "SELECT COUNT(*) FROM console_trace_artifacts"
    ).fetchone()[0]

    assert coordinator.submit(db, request) is True
    assert (
        tuple(event.event_id for event in repository.read_events(cursor, segment_id))
        == event_ids
    )
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == artifact_count
    )


def test_concurrent_exact_pending_retries_have_one_settlement_owner(
    db: CharactersRAGDB,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class PausingRepository(ConsoleTraceRepository):
        blocked = True
        pause = False

        def store_response_link(self, *args, **kwargs):
            if self.blocked:
                raise RuntimeError("PRIVATE-RESPONSE-CANARY")
            if self.pause:
                entered.set()
                assert release.wait(timeout=3)
            return super().store_response_link(*args, **kwargs)

    repository = PausingRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository, max_pending=2)
    _conversation_id, segment_id, call_id = _call(db, repository)
    request = _request(call_id)
    assert coordinator.submit(db, request) is False
    repository.blocked = False
    repository.pause = True
    results: list[bool] = []

    worker = threading.Thread(
        target=lambda: results.append(coordinator.submit(db, request))
    )
    worker.start()
    try:
        assert entered.wait(timeout=3)
        assert coordinator.submit(db, request) is False
    finally:
        release.set()
        worker.join(timeout=3)

    assert not worker.is_alive()
    assert results == [True]
    assert coordinator.pending_count == 0
    cursor = db.get_connection().cursor()
    settled = repository.get_call(cursor, call_id)
    assert settled is not None and settled.state is TraceCallState.COMPLETE
    events = repository.read_events(cursor, segment_id)
    event_types = [event.event_type for event in events]
    assert event_types.count("response_selection") == 1
    assert event_types.count("call_outcome") == 1
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 1
    )


def test_next_settlement_opportunistically_retries_pending_seals(
    db: CharactersRAGDB,
) -> None:
    class FailOnceRepository(ConsoleTraceRepository):
        failed = False

        def store_response_link(self, *args, **kwargs):
            if not self.failed:
                self.failed = True
                raise RuntimeError("PRIVATE-RESPONSE-CANARY")
            return super().store_response_link(*args, **kwargs)

    repository = FailOnceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository, max_pending=4)
    first_call = _call(db, repository, sequence=0)[2]
    second_call = _call(db, repository, sequence=1)[2]

    assert coordinator.submit(db, _request(first_call)) is False
    assert coordinator.submit(db, _request(second_call)) is True

    assert coordinator.pending_count == 0
    cursor = db.get_connection().cursor()
    assert repository.get_call(cursor, first_call).state is TraceCallState.COMPLETE  # type: ignore[union-attr]
    assert repository.get_call(cursor, second_call).state is TraceCallState.COMPLETE  # type: ignore[union-attr]


def test_settlement_retry_queue_is_bounded_and_idempotent(
    db: CharactersRAGDB,
) -> None:
    class FailingRepository(ConsoleTraceRepository):
        def store_response_link(self, *args, **kwargs):
            raise RuntimeError("PRIVATE-RESPONSE-CANARY")

    repository = FailingRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository, max_pending=2)
    calls = [_call(db, repository, sequence=index)[2] for index in range(3)]

    assert [coordinator.submit(db, _request(call_id)) for call_id in calls] == [
        False,
        False,
        False,
    ]
    assert coordinator.pending_count == 2
    assert coordinator.dropped_count == 1
    coordinator.submit(db, _request(calls[-1]))
    assert coordinator.pending_count == 2
    assert coordinator.dropped_count == 1


def test_conflicting_signal_cannot_replace_first_queued_settlement(
    db: CharactersRAGDB,
) -> None:
    class BlockingRepository(ConsoleTraceRepository):
        blocked = True

        def store_response_link(self, *args, **kwargs):
            if self.blocked:
                raise RuntimeError("PRIVATE-RESPONSE-CANARY")
            return super().store_response_link(*args, **kwargs)

    repository = BlockingRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository, max_pending=2)
    _conversation_id, _segment_id, call_id = _call(db, repository)

    assert coordinator.submit(db, _request(call_id)) is False
    repository.blocked = False
    assert (
        coordinator.submit(
            db,
            _request(call_id, outcome=TraceCallState.ERROR, response=None),
        )
        is False
    )

    assert coordinator.retry_pending() == 1
    settled = repository.get_call(db.get_connection().cursor(), call_id)
    assert settled is not None
    assert settled.state is TraceCallState.COMPLETE


def test_cold_restart_recovers_open_calls_monotonically_and_idempotently(
    tmp_path,
) -> None:
    path = tmp_path / "trace-restart.sqlite"
    repository = ConsoleTraceRepository()
    first = CharactersRAGDB(str(path), "trace-restart-first")
    _conversation_id, _segment_id, reserved_id = _call(
        first, repository, sequence=0, state=TraceCallState.RESERVED
    )
    _conversation_id, _segment_id, dispatched_id = _call(
        first, repository, sequence=1, state=TraceCallState.DISPATCH_STARTED
    )
    _conversation_id, _segment_id, response_id = _call(
        first, repository, sequence=2, state=TraceCallState.RESPONSE_STARTED
    )
    _conversation_id, _segment_id, terminal_id = _call(
        first, repository, sequence=3, state=TraceCallState.RESPONSE_STARTED
    )
    ConsoleTraceSettlementCoordinator(repository).settle(first, _request(terminal_id))
    first.close_connection()

    reopened = CharactersRAGDB(str(path), "trace-restart-second")
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    recovered = recover_console_trace_calls(
        reopened,
        occurred_at="2026-08-30T02:00:00Z",
        repository=repository,
    )
    assert {record.call_id: record.state for record in recovered} == {
        reserved_id: TraceCallState.NOT_DISPATCHED,
        dispatched_id: TraceCallState.DISPATCH_UNKNOWN,
        response_id: TraceCallState.INTERRUPTED,
    }
    assert (
        coordinator.recover_open_calls(reopened, occurred_at="2026-08-30T02:00:01Z")
        == ()
    )
    assert (
        repository.get_call(reopened.get_connection().cursor(), terminal_id).state
        is TraceCallState.COMPLETE
    )  # type: ignore[union-attr]
    reopened.close_connection()
