"""Durable pre-dispatch lifecycle boundaries for Console provider calls."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from tldw_chatbook.Chat.console_prepared_request import build_console_request
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_final_values import (
    ProviderRequestShadowBundle,
    SurfaceDeltaAdmission,
    build_verified_surface_delta,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    OmittedTraceProvenance,
    ProviderRequestProvenance,
    SavedRevisionTraceProvenance,
    TraceOmissionReason,
    TraceProvenanceSource,
    request_route_provenance,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_service import (
    ConsoleTraceCallBoundary,
    ConsoleTraceService,
    TraceCallIdentity,
    TraceCallPersistenceError,
)
from tldw_chatbook.Chat.console_trace_settlement import (
    ConsoleTraceSettlementCoordinator,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db() -> CharactersRAGDB:
    database = CharactersRAGDB(":memory:", "console-trace-call-lifecycle-test")
    yield database
    database.close_connection()


def _identity(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> TraceCallIdentity:
    conversation_id = db.add_conversation({"title": "trace lifecycle"})
    assert conversation_id is not None
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )
    with db.transaction() as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
        repository.ensure_policy(cursor, policy)
    return TraceCallIdentity(
        owner_id=owner.owner_id,
        segment_id=segment.segment_id,
        turn_id="turn-1",
        run_id="run-1",
        call_sequence=0,
        idempotency_key=new_opaque_id(),
        policy_id=policy.policy_id,
    )


def test_capture_on_reservation_commits_only_content_free_call_identity(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    service = ConsoleTraceService(repository)
    identity = _identity(db, repository)

    reserved = service.reserve_call(db, identity)

    cursor = db.get_connection().cursor()
    durable = repository.get_call(cursor, reserved.call_id)
    assert durable is not None
    assert durable.state is TraceCallState.RESERVED
    assert (
        durable.owner_id,
        durable.segment_id,
        durable.turn_id,
        durable.run_id,
        durable.call_sequence,
        durable.idempotency_key,
        durable.policy_id,
    ) == (
        identity.owner_id,
        identity.segment_id,
        identity.turn_id,
        identity.run_id,
        identity.call_sequence,
        identity.idempotency_key,
        identity.policy_id,
    )
    assert durable.surface_node_id is None
    assert durable.request_header_id is None
    assert durable.provider_name is None
    assert durable.dispatch_started_at is None
    for table in (
        "console_trace_artifacts",
        "console_trace_request_headers",
        "console_trace_surface_nodes",
        "console_trace_events",
    ):
        assert cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


class _AmbiguousCommitDatabase:
    """Raise once after delegating a successful transaction commit."""

    def __init__(self, database: CharactersRAGDB) -> None:
        self.database = database
        self.transaction_entries = 0

    @contextmanager
    def transaction(self, *, immediate: bool = False):
        self.transaction_entries += 1
        with self.database.transaction(immediate=immediate) as cursor:
            yield cursor
        if self.transaction_entries == 1:
            raise RuntimeError("AMBIGUOUS-COMMIT-CANARY")

    def get_connection(self):
        return self.database.get_connection()


def test_ambiguous_reservation_commit_reconciles_by_idempotency_without_duplicate(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    service = ConsoleTraceService(repository)
    identity = _identity(db, repository)
    ambiguous = _AmbiguousCommitDatabase(db)

    reserved = service.reserve_call(ambiguous, identity)

    calls = repository.read_calls(db.get_connection().cursor(), identity.owner_id)
    assert calls == (reserved,)
    assert ambiguous.transaction_entries == 2
    assert "CANARY" not in repr(reserved)


def test_reservation_reconciliation_does_not_confirm_absent_on_logical_conflict(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    service = ConsoleTraceService(repository)
    identity = _identity(db, repository)
    reserved = service.reserve_call(db, identity)
    conflicting = TraceCallIdentity(
        owner_id=identity.owner_id,
        segment_id=identity.segment_id,
        turn_id=identity.turn_id,
        run_id=identity.run_id,
        call_sequence=identity.call_sequence,
        idempotency_key=new_opaque_id(),
        policy_id=identity.policy_id,
    )

    with pytest.raises(TraceCallPersistenceError) as raised:
        service.reserve_call(db, conflicting)

    assert raised.value.reservation_status == "unknown"
    assert repository.read_calls(db.get_connection().cursor(), identity.owner_id) == (
        reserved,
    )


def _unavailable_request(
    identity: TraceCallIdentity,
) -> tuple[
    ProviderRequestProvenance,
    ProviderRequestShadowBundle,
    SurfaceDeltaAdmission,
]:
    omission = OmittedTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST,
        TraceOmissionReason.SANITIZER_FAILED,
    )
    provenance = ProviderRequestProvenance(
        messages=(omission,),
        messages_payload=(omission,),
        metadata=(request_route_provenance(ConsoleRequestRoute.FRESH),),
    )
    preparation_identity = new_opaque_id()
    bundle = ProviderRequestShadowBundle(
        available=False,
        omission_reason=TraceOmissionReason.SANITIZER_FAILED,
        preparation_identity=preparation_identity,
    )
    admission = SurfaceDeltaAdmission(
        owner_id=identity.owner_id,
        segment_id=identity.segment_id,
        predecessor_surface_head_id=None,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=preparation_identity,
        descriptors=(omission,),
    )
    return provenance, bundle, admission


def test_persisted_incomplete_boundary_binds_and_commits_dispatch_started(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    service = ConsoleTraceService(repository)
    identity = _identity(db, repository)
    reserved = service.reserve_call(db, identity)
    provenance, bundle, admission = _unavailable_request(identity)
    surface_delta = build_verified_surface_delta(
        provenance,
        bundle,
        admission=admission,
    )

    started = service.bind_and_mark_dispatch(
        db,
        call_id=reserved.call_id,
        owner_id=identity.owner_id,
        segment_id=identity.segment_id,
        provenance=provenance,
        bundle=bundle,
        surface_delta=surface_delta,
        occurred_at="2026-08-29T18:00:00Z",
    )

    assert started.state is TraceCallState.DISPATCH_STARTED
    assert started.surface_node_id is not None
    assert started.request_header_id is not None
    assert started.dispatch_started_at == "2026-08-29T18:00:00Z"
    assert started.integrity_state == "incomplete"
    assert started.omission_reason_code == "sanitizer_failed"
    cursor = db.get_connection().cursor()
    node = repository.get_surface_node(cursor, started.surface_node_id)
    assert node is not None and node.reference_kind == "omission"
    assert node.omission_reason_code == "sanitizer_failed"
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 0
    )


def test_call_boundary_owns_reserve_then_verified_bundle_dispatch_sequence(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    service = ConsoleTraceService(repository)
    identity = _identity(db, repository)
    provenance, bundle, admission = _unavailable_request(identity)
    boundary = ConsoleTraceCallBoundary(
        service=service,
        database=db,
        identity=identity,
        admission=admission,
        occurred_at_factory=lambda: "2026-08-29T18:01:00Z",
    )

    reserved = boundary.reserve()
    started = boundary.mark_dispatch_started(bundle, provenance)
    unknown = boundary.mark_dispatch_unknown()

    assert reserved.state is TraceCallState.RESERVED
    assert started.call_id == reserved.call_id
    assert started.state is TraceCallState.DISPATCH_STARTED
    assert started.dispatch_started_at == "2026-08-29T18:01:00Z"
    assert unknown.call_id == reserved.call_id
    assert unknown.state is TraceCallState.DISPATCH_UNKNOWN
    assert (
        repository.get_call(db.get_connection().cursor(), reserved.call_id) == unknown
    )


def test_call_boundary_retries_pending_exact_settlement_until_confirmed(
    db: CharactersRAGDB,
) -> None:
    class SequencedSubmissionService(ConsoleTraceService):
        __slots__ = ("results", "submissions")

        def __init__(self, repository: ConsoleTraceRepository) -> None:
            super().__init__(repository)
            self.results = iter((False, False, True))
            self.submissions: list[dict[str, object]] = []

        def submit_settlement(self, _database, **kwargs):
            self.submissions.append(kwargs)
            return next(self.results)

    repository = ConsoleTraceRepository()
    service = SequencedSubmissionService(repository)
    identity = _identity(db, repository)
    provenance, bundle, admission = _unavailable_request(identity)
    boundary = ConsoleTraceCallBoundary(
        service=service,
        database=db,
        identity=identity,
        admission=admission,
        occurred_at_factory=lambda: "2026-08-29T18:01:10Z",
    )
    boundary.reserve()
    boundary.mark_dispatch_started(bundle, provenance)
    response = {"role": "assistant", "content": "answer"}

    assert boundary.settle_response(response, TraceCallState.COMPLETE) is False
    assert (
        boundary.settle_response(
            {"role": "assistant", "content": "conflict"},
            TraceCallState.COMPLETE,
        )
        is False
    )
    assert len(service.submissions) == 1
    assert boundary.settle_response(response, TraceCallState.COMPLETE) is False
    assert boundary.settle_response(response, TraceCallState.COMPLETE) is True
    assert len(service.submissions) == 3
    assert (
        boundary.settle_response(
            {"role": "assistant", "content": "conflict"},
            TraceCallState.COMPLETE,
        )
        is False
    )
    assert boundary.settle_response(response, TraceCallState.COMPLETE) is True
    assert len(service.submissions) == 3


def test_call_boundary_retries_after_bounded_queue_eviction_without_duplicates(
    db: CharactersRAGDB,
) -> None:
    class BlockingResponseRepository(ConsoleTraceRepository):
        blocked = True

        def store_response_link(self, *args, **kwargs):
            if self.blocked:
                raise RuntimeError("PRIVATE-RESPONSE-CANARY")
            return super().store_response_link(*args, **kwargs)

    repository = BlockingResponseRepository()
    service = ConsoleTraceService(repository)
    service._settlement = ConsoleTraceSettlementCoordinator(  # noqa: SLF001
        repository,
        max_pending=1,
    )

    def started_boundary(label: str) -> ConsoleTraceCallBoundary:
        identity = _identity(db, repository)
        provenance, bundle, admission = _unavailable_request(identity)
        boundary = ConsoleTraceCallBoundary(
            service=service,
            database=db,
            identity=identity,
            admission=admission,
            occurred_at_factory=lambda: f"2026-08-29T18:01:{label}Z",
        )
        boundary.reserve()
        boundary.mark_dispatch_started(bundle, provenance)
        return boundary

    first = started_boundary("20")
    second = started_boundary("21")
    first_response = {"role": "assistant", "content": "first"}
    second_response = {"role": "assistant", "content": "second"}

    assert first.settle_response(first_response, TraceCallState.COMPLETE) is False
    assert second.settle_response(second_response, TraceCallState.COMPLETE) is False
    assert service._settlement.pending_count == 1  # noqa: SLF001
    assert service._settlement.dropped_count == 1  # noqa: SLF001

    repository.blocked = False
    assert first.settle_response(first_response, TraceCallState.COMPLETE) is True

    cursor = db.get_connection().cursor()
    first_call = repository.get_call(cursor, first._started.call_id)  # noqa: SLF001
    second_call = repository.get_call(cursor, second._started.call_id)  # noqa: SLF001
    assert first_call is not None and first_call.state is TraceCallState.COMPLETE
    assert second_call is not None and second_call.state is TraceCallState.COMPLETE
    assert service._settlement.pending_count == 0  # noqa: SLF001
    first_events = repository.read_events(cursor, first_call.segment_id)
    first_event_ids = tuple(event.event_id for event in first_events)
    first_event_types = [event.event_type for event in first_events]
    assert first_event_types.count("response_selection") == 1
    assert first_event_types.count("call_outcome") == 1
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 2
    )

    assert first.settle_response(first_response, TraceCallState.COMPLETE) is True
    assert (
        tuple(
            event.event_id
            for event in repository.read_events(cursor, first_call.segment_id)
        )
        == first_event_ids
    )
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 2
    )


def test_call_boundary_cancels_reserved_call_as_not_dispatched_idempotently(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    service = ConsoleTraceService(repository)
    identity = _identity(db, repository)
    _provenance, _bundle, admission = _unavailable_request(identity)
    boundary = ConsoleTraceCallBoundary(
        service=service,
        database=db,
        identity=identity,
        admission=admission,
        occurred_at_factory=lambda: "2026-08-29T18:01:30Z",
    )

    reserved = boundary.reserve()
    assert boundary.reservation_status == "established"
    first = boundary.mark_not_dispatched()
    repeated = boundary.mark_not_dispatched()

    assert first.call_id == reserved.call_id
    assert first.state is TraceCallState.NOT_DISPATCHED
    assert repeated == first
    assert repository.get_call(db.get_connection().cursor(), reserved.call_id) == first


def test_call_boundary_reports_confirmed_absent_reservation_after_failed_reconcile(
    db: CharactersRAGDB,
) -> None:
    class ReservationFailingRepository(ConsoleTraceRepository):
        def reserve_call(self, *args, **kwargs):
            raise RuntimeError("PRIVATE-RESERVATION-FAILURE")

    repository = ReservationFailingRepository()
    service = ConsoleTraceService(repository)
    identity = _identity(db, repository)
    _provenance, _bundle, admission = _unavailable_request(identity)
    boundary = ConsoleTraceCallBoundary(
        service=service,
        database=db,
        identity=identity,
        admission=admission,
        occurred_at_factory=lambda: "2026-08-29T18:01:31Z",
    )

    with pytest.raises(TraceCallPersistenceError) as raised:
        boundary.reserve()

    assert raised.value.boundary is boundary
    assert boundary.reservation_status == "not_established"
    assert (
        repository.get_call_by_idempotency_key(
            db.get_connection().cursor(), identity.idempotency_key
        )
        is None
    )


def test_reservation_reports_unknown_when_reconciliation_cannot_read(
    db: CharactersRAGDB,
) -> None:
    class ReservationFailingRepository(ConsoleTraceRepository):
        def reserve_call(self, *args, **kwargs):
            raise RuntimeError("PRIVATE-RESERVATION-FAILURE")

    class ReconciliationUnavailableDatabase:
        def __init__(self) -> None:
            self.entries = 0

        @contextmanager
        def transaction(self, *, immediate: bool = False):
            self.entries += 1
            if self.entries == 2:
                raise RuntimeError("PRIVATE-RECONCILIATION-FAILURE")
            with db.transaction(immediate=immediate) as cursor:
                yield cursor

    repository = ReservationFailingRepository()
    service = ConsoleTraceService(repository)
    identity = _identity(db, repository)

    with pytest.raises(TraceCallPersistenceError) as raised:
        service.reserve_call(ReconciliationUnavailableDatabase(), identity)

    assert raised.value.reservation_status == "unknown"


class _FailingRepository(ConsoleTraceRepository):
    """Inject one content-free failure at a caller-owned transaction step."""

    def __init__(self, fail_at: str) -> None:
        self.fail_at = fail_at

    def _check(self, step: str) -> None:
        if self.fail_at == step:
            raise RuntimeError("PRIVATE-TRACE-BODY-CREDENTIAL-CANARY")

    def append_surface_node(self, *args, **kwargs):
        self._check("surface")
        return super().append_surface_node(*args, **kwargs)

    def create_or_reuse_request_header(self, *args, **kwargs):
        self._check("header")
        return super().create_or_reuse_request_header(*args, **kwargs)

    def bind_call(self, *args, **kwargs):
        self._check("bind")
        return super().bind_call(*args, **kwargs)

    def advance_call_state(self, *args, **kwargs):
        self._check("dispatch_started")
        return super().advance_call_state(*args, **kwargs)


@pytest.mark.parametrize("fail_at", ["surface", "header", "bind", "dispatch_started"])
def test_pre_dispatch_failure_matrix_rolls_back_boundary_and_leaves_reserved(
    db: CharactersRAGDB,
    fail_at: str,
) -> None:
    repository = _FailingRepository(fail_at)
    service = ConsoleTraceService(repository)
    identity = _identity(db, repository)
    provenance, bundle, admission = _unavailable_request(identity)
    boundary = ConsoleTraceCallBoundary(
        service=service,
        database=db,
        identity=identity,
        admission=admission,
        occurred_at_factory=lambda: "2026-08-29T18:02:00Z",
    )
    reserved = boundary.reserve()

    with pytest.raises(TraceCallPersistenceError) as raised:
        boundary.mark_dispatch_started(bundle, provenance)

    assert "CANARY" not in str(raised.value)
    cursor = db.get_connection().cursor()
    durable = repository.get_call(cursor, reserved.call_id)
    assert durable is not None and durable.state is TraceCallState.RESERVED
    assert durable.surface_node_id is None
    assert durable.request_header_id is None
    for table in (
        "console_trace_request_headers",
        "console_trace_surface_nodes",
        "console_trace_events",
    ):
        assert cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_real_sqlite_adapter_observes_dispatch_started_commit(
    tmp_path,
) -> None:
    database = CharactersRAGDB(
        str(tmp_path / "trace-call-boundary.db"),
        "console-trace-call-boundary-test",
    )
    with database.transaction() as cursor:
        cursor.execute("CREATE TABLE task11_boundary(state TEXT NOT NULL)")
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )
    semantic = build_console_request(
        [{"role": "user", "content": "not persisted as trace content"}],
        message_provenance=(SavedRevisionTraceProvenance(new_opaque_id()),),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        metadata_provenance=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        capture_policy=policy,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1",
        ready=True,
        execution_key="openai",
        streaming=False,
    )

    class SqliteBoundary:
        def reserve(self) -> None:
            with database.transaction(immediate=True) as cursor:
                cursor.execute("INSERT INTO task11_boundary VALUES ('reserved')")

        def mark_dispatch_started(self, _bundle, _provenance) -> None:
            with database.transaction(immediate=True) as cursor:
                cursor.execute("UPDATE task11_boundary SET state = 'dispatch_started'")

    def adapter(**_kwargs):
        state = (
            database.get_connection()
            .execute("SELECT state FROM task11_boundary")
            .fetchone()[0]
        )
        assert state == "dispatch_started"
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=adapter,
        trace_call_boundary_factory=lambda _request, _resolution, _route: (
            SqliteBoundary()
        ),
    )
    prepared = gateway.prepare_chat_request(
        resolution,
        semantic,
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )

    result = [
        item
        async for item in gateway.stream_chat(
            resolution,
            prepared,
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
    ]

    assert result == ["ok"]
    database.close_connection()
