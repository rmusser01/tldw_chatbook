"""Reference-machine latency gate for normalized Console provider-call writes."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from types import FrameType
from typing import TypedDict
import warnings

from loguru import logger
import pytest
from textual.app import App

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_prepared_request import (
    PreparedProviderRequest,
    build_console_request,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_trace_final_values import (
    ProviderRequestShadowBundle,
    SurfaceDeltaAdmission,
    verify_provider_request_shadow,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    ProviderRequestProvenance,
    SavedRevisionTraceProvenance,
    request_route_provenance,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_service import (
    ConsoleTraceCallBoundary,
    ConsoleTraceService,
    TRACE_CRITICAL_WRITE_WAL_AUTOCHECKPOINT_PAGES,
    TraceCallIdentity,
)
from tldw_chatbook.Chat.console_trace_settlement import (
    ConsoleTraceSettlementCoordinator,
    TraceSettlementRequest,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


FIXTURE_PATH = Path(__file__).with_name("fixtures") / (
    "console_trace_reference_machine.json"
)
ARTIFACT_NAME = "console_trace_call_latency.json"


@dataclass(frozen=True, slots=True)
class _PreparedCall:
    boundary: ConsoleTraceCallBoundary
    bundle: ProviderRequestShadowBundle
    provenance: ProviderRequestProvenance


class _RunResult(TypedDict):
    run_index: int
    reservation_dispatch_started_ns: list[int]
    reservation_dispatch_started_summary: dict[str, float | int]
    reservation_only_ns: list[int]
    reservation_only_summary: dict[str, float | int]
    dispatch_started_only_ns: list[int]
    dispatch_started_only_summary: dict[str, float | int]
    reservation_dispatch_started_process_cpu_ns: list[int]
    reservation_dispatch_started_thread_cpu_ns: list[int]
    settlement_ns: list[int]
    settlement_summary: dict[str, float | int]
    settlement_process_cpu_ns: list[int]
    settlement_thread_cpu_ns: list[int]
    sqlite_settings: dict[str, object]
    wal_file_bytes_before_settlement: int
    wal_file_bytes_after_settlement: int
    wal_file_bytes_after_close: int
    close_ns: int


class _TraceSettlementApp(App[None]):
    def __init__(self, database: CharactersRAGDB) -> None:
        super().__init__()
        self.chachanotes_db = database


class _BlockingTraceHandoff:
    def __init__(self, call_id: str, release: threading.Event) -> None:
        self.call_id = call_id
        self.release = release
        self.started = threading.Event()
        self.settlement_threads: list[int] = []

    def settle(self, _canonical_message_id: str | None) -> None:
        self.settlement_threads.append(threading.get_ident())
        self.started.set()
        assert self.release.wait(5), "test must release blocked settlement"


class _RecordingTraceHandoff:
    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        self.canonical_message_ids: list[str | None] = []
        self.settlement_threads: list[int] = []

    def settle(self, canonical_message_id: str | None) -> None:
        self.canonical_message_ids.append(canonical_message_id)
        self.settlement_threads.append(threading.get_ident())


class _ThreadRecordingSettlementCoordinator(ConsoleTraceSettlementCoordinator):
    def __init__(self, repository: ConsoleTraceRepository) -> None:
        super().__init__(repository)
        self.settlement_threads: list[int] = []

    def _settle_prepared(self, database, prepared):
        self.settlement_threads.append(threading.get_ident())
        return super()._settle_prepared(database, prepared)


class _GatewayTraceBoundary:
    def __init__(
        self,
        coordinator: ConsoleTraceSettlementCoordinator,
        database: CharactersRAGDB,
        call_id: str,
    ) -> None:
        self.coordinator = coordinator
        self.database = database
        self.call_id = call_id

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

    def prepare_response_settlement(self, response, outcome, usage=None):
        return self.coordinator.prepare_handoff(
            self.database,
            TraceSettlementRequest(
                call_id=self.call_id,
                outcome=outcome,
                response_envelope=response,
                usage=usage,
                response_started_at="2026-08-30T01:00:01Z",
                settled_at="2026-08-30T01:00:02Z",
            ),
        )


def _gateway_for_started_call(
    coordinator: ConsoleTraceSettlementCoordinator,
    database: CharactersRAGDB,
    call_id: str,
) -> tuple[
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    PreparedProviderRequest,
]:
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
        trace_call_boundary_factory=lambda _request, _resolution, _route: (
            _GatewayTraceBoundary(coordinator, database, call_id)
        ),
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
    return gateway, resolution, prepared


async def _consume_gateway(
    gateway: ConsoleProviderGateway,
    resolution: ConsoleProviderResolution,
    prepared: PreparedProviderRequest,
    signals: ConsoleProviderStreamSignals,
) -> list[object]:
    return [
        item
        async for item in gateway.stream_chat(
            resolution,
            prepared,
            signals=signals,
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
    ]


def _hardware_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": _physical_memory_bytes(),
        "model_identifier": "unknown",
        "cpu_model": platform.processor() or "unknown",
    }
    if sys.platform != "darwin":
        return metadata
    try:
        completed = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return metadata
    fields = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator:
            fields[key] = value.strip()
    metadata["model_identifier"] = fields.get("Model Identifier", "unknown")
    metadata["cpu_model"] = fields.get("Chip", metadata["cpu_model"])
    return metadata


def _physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, TypeError, ValueError):
        return None


def _filesystem_type(path: Path) -> str:
    if sys.platform == "darwin":
        return _darwin_filesystem_type(path)
    try:
        return subprocess.run(
            ["stat", "-f", "-c", "%T", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _darwin_filesystem_type(path: Path) -> str:
    """Resolve the volume format through ``df`` because BSD stat returns ``/``."""

    try:
        df_result = subprocess.run(
            ["df", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        device = df_result.stdout.splitlines()[-1].split()[0]
        mount_result = subprocess.run(
            ["mount"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (IndexError, OSError, subprocess.SubprocessError):
        return "unknown"
    prefix = f"{device} on "
    for line in mount_result.stdout.splitlines():
        if line.startswith(prefix) and "(" in line:
            return line.rsplit("(", 1)[1].split(",", 1)[0].rstrip(")")
    return "unknown"


def _environment_metadata(tmp_path: Path) -> dict[str, object]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "sqlite_version": sqlite3.sqlite_version,
        "filesystem": _filesystem_type(tmp_path),
        "hardware": _hardware_metadata(),
    }


def _sqlite_settings(database: CharactersRAGDB) -> dict[str, object]:
    cursor = database.get_connection().cursor()
    return {
        name: cursor.execute(f"PRAGMA {name}").fetchone()[0]
        for name in (
            "journal_mode",
            "synchronous",
            "page_size",
            "cache_size",
            "wal_autocheckpoint",
        )
    }


def _file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _prepare_call(
    database: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    service: ConsoleTraceService,
    *,
    sequence: int,
) -> _PreparedCall:
    conversation_id = database.add_conversation({"title": "trace latency fixture"})
    assert conversation_id is not None
    request_content = f"representative request {sequence:04d}"
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": request_content,
        }
    )
    assert message_id is not None
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    with database.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
        repository.ensure_policy(cursor, policy)
        revision = cursor.execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ? AND live_message_id = ?
                 ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id, message_id),
        ).fetchone()
        assert revision is not None
        descriptor = SavedRevisionTraceProvenance(str(revision[0]))
        provenance = ProviderRequestProvenance(
            messages=(descriptor,),
            messages_payload=(descriptor,),
            metadata=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        )
        preparation_identity = new_opaque_id()
        admission = SurfaceDeltaAdmission(
            owner_id=owner.owner_id,
            segment_id=segment.segment_id,
            predecessor_surface_head_id=None,
            route_identity=ConsoleRequestRoute.FRESH.value,
            preparation_identity=preparation_identity,
            descriptors=(descriptor,),
        )
        surface_boundary = service.prepare_surface_provenance(
            cursor,
            None,
            provenance=provenance,
            admission=admission,
            values=({"role": "user", "content": request_content},),
        )
    actual_kwargs = {
        "api_endpoint": "openai",
        "model": "gpt-test",
        "temp": 0.2,
        "max_tokens": 256,
        "response_format": {"type": "text"},
        "reasoning_effort": "medium",
        **surface_boundary._provider_request_surface_values(),  # noqa: SLF001
    }
    bundle = verify_provider_request_shadow(
        actual_kwargs=actual_kwargs,
        expected_kwargs=dict(actual_kwargs),
        provenance=surface_boundary.provenance,
        project_handler_kwargs=lambda kwargs: kwargs,
        endpoint_identity="https://api.example.invalid/v1",
        preparation_identity=preparation_identity,
        surface_boundary=surface_boundary,
    )
    identity = TraceCallIdentity(
        owner_id=owner.owner_id,
        segment_id=segment.segment_id,
        turn_id=f"turn-{sequence}",
        run_id=f"run-{sequence}",
        call_sequence=0,
        idempotency_key=new_opaque_id(),
        policy_id=policy.policy_id,
    )
    return _PreparedCall(
        ConsoleTraceCallBoundary(
            service=service,
            database=database,
            identity=identity,
            admission=admission,
            occurred_at_factory=lambda: "2026-08-30T01:00:00Z",
            surface_boundary=surface_boundary,
        ),
        bundle,
        surface_boundary.provenance,
    )


def _p95(samples: list[int]) -> float:
    # Pinned release-gate formula; changing method or index changes the gate.
    return statistics.quantiles(samples, n=100, method="inclusive")[94]


def _summary(samples: list[int]) -> dict[str, float | int]:
    return {"p95_ns": _p95(samples), "max_ns": max(samples)}


def _environment_mismatches(
    fixture: dict[str, object],
    actual: dict[str, object],
) -> list[str]:
    expected = fixture["environment"]
    assert isinstance(expected, dict)
    mismatches: list[str] = []
    for key in (
        "python_implementation",
        "python_version",
        "sqlite_version",
        "filesystem",
    ):
        if actual.get(key) != expected.get(key):
            mismatches.append(key)
    expected_hardware = expected["hardware"]
    actual_hardware = actual["hardware"]
    assert isinstance(expected_hardware, dict)
    assert isinstance(actual_hardware, dict)
    for key in (
        "architecture",
        "logical_cpu_count",
        "memory_bytes",
        "model_identifier",
        "cpu_model",
    ):
        if actual_hardware.get(key) != expected_hardware.get(key):
            mismatches.append(f"hardware.{key}")
    return mismatches


def _enforce_reference_environment_policy(mismatches: list[str]) -> bool:
    """Return whether thresholds apply, failing closed unless explicitly waived."""

    if not mismatches:
        return True
    if os.environ.get("TLDW_TRACE_LATENCY_ALLOW_NON_REFERENCE") == "1":
        warnings.warn(
            "Trace latency correctness is running without release thresholds on "
            f"an explicitly allowed non-reference environment: {mismatches}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    raise AssertionError(
        "reference environment mismatch; set "
        f"TLDW_TRACE_LATENCY_ALLOW_NON_REFERENCE=1 for correctness-only: {mismatches}"
    )


def test_reference_database_uses_pinned_sqlite_settings(tmp_path: Path) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["trace_operation_policy"]["latency_critical_wal_autocheckpoint"] == (
        TRACE_CRITICAL_WRITE_WAL_AUTOCHECKPOINT_PAGES
    )
    database = CharactersRAGDB(
        str(tmp_path / "sqlite-settings.sqlite"), "trace-latency"
    )
    try:
        assert _sqlite_settings(database) == fixture["sqlite_settings"]
    finally:
        database.close_connection()


def test_trace_transactions_scope_checkpoint_policy_and_restore_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = CharactersRAGDB(
        str(tmp_path / "trace-checkpoint-scope.sqlite"),
        "trace-checkpoint-scope",
    )
    try:
        assert _sqlite_settings(database)["wal_autocheckpoint"] == 1000
        repository = ConsoleTraceRepository()
        service = ConsoleTraceService(repository)
        prepared = _prepare_call(database, repository, service, sequence=0)
        observed: list[int] = []
        original_transaction = database.transaction

        @contextmanager
        def recording_transaction(*, immediate: bool = False):
            with original_transaction(immediate=immediate) as cursor:
                observed.append(
                    int(cursor.execute("PRAGMA wal_autocheckpoint").fetchone()[0])
                )
                yield cursor

        monkeypatch.setattr(database, "transaction", recording_transaction)
        reserved = prepared.boundary.reserve()
        dispatched = prepared.boundary.mark_dispatch_started(
            prepared.bundle,
            prepared.provenance,
        )
        settled = ConsoleTraceSettlementCoordinator(repository).settle(
            database,
            TraceSettlementRequest(
                call_id=dispatched.call_id,
                outcome=TraceCallState.COMPLETE,
                response_envelope={"role": "assistant", "content": "answer"},
                usage={"prompt_tokens": 4, "completion_tokens": 1},
                response_started_at="2026-08-30T01:00:01Z",
                settled_at="2026-08-30T01:00:02Z",
            ),
        )

        assert reserved.call_id == dispatched.call_id
        assert settled.state is TraceCallState.COMPLETE
        assert observed == [TRACE_CRITICAL_WRITE_WAL_AUTOCHECKPOINT_PAGES] * 2 + [1000]
        assert _sqlite_settings(database)["wal_autocheckpoint"] == 1000

        conversation_id = database.add_conversation({"title": "ordinary write"})
        assert conversation_id is not None
        assert _sqlite_settings(database)["wal_autocheckpoint"] == 1000
    finally:
        database.close_connection()


def test_trace_checkpoint_policy_restores_default_after_reservation_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = CharactersRAGDB(
        str(tmp_path / "trace-checkpoint-failure.sqlite"),
        "trace-checkpoint-failure",
    )
    try:
        repository = ConsoleTraceRepository()
        service = ConsoleTraceService(repository)
        prepared = _prepare_call(database, repository, service, sequence=0)
        observed: list[int] = []
        original_transaction = database.transaction

        @contextmanager
        def recording_transaction(*, immediate: bool = False):
            with original_transaction(immediate=immediate) as cursor:
                observed.append(
                    int(cursor.execute("PRAGMA wal_autocheckpoint").fetchone()[0])
                )
                yield cursor

        monkeypatch.setattr(database, "transaction", recording_transaction)

        def fail_reservation(_cursor: sqlite3.Cursor, **_kwargs: object) -> None:
            raise RuntimeError("forced reservation failure")

        monkeypatch.setattr(repository, "reserve_call", fail_reservation)
        with pytest.raises(RuntimeError):
            prepared.boundary.reserve()

        assert observed == [TRACE_CRITICAL_WRITE_WAL_AUTOCHECKPOINT_PAGES] * 2
        assert _sqlite_settings(database)["wal_autocheckpoint"] == 1000
    finally:
        database.close_connection()


def test_trace_checkpoint_policy_is_connection_local_across_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = CharactersRAGDB(
        str(tmp_path / "trace-checkpoint-thread.sqlite"),
        "trace-checkpoint-thread",
    )
    repository = ConsoleTraceRepository()
    service = ConsoleTraceService(repository)
    prepared = _prepare_call(database, repository, service, sequence=0)
    transaction_entered = threading.Event()
    release_transaction = threading.Event()
    observed: list[int] = []
    original_transaction = database.transaction

    @contextmanager
    def recording_transaction(*, immediate: bool = False):
        with original_transaction(immediate=immediate) as cursor:
            observed.append(
                int(cursor.execute("PRAGMA wal_autocheckpoint").fetchone()[0])
            )
            transaction_entered.set()
            assert release_transaction.wait(2)
            yield cursor

    monkeypatch.setattr(database, "transaction", recording_transaction)
    errors: list[BaseException] = []

    def reserve_on_worker() -> None:
        try:
            prepared.boundary.reserve()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            database.close_connection()

    worker = threading.Thread(target=reserve_on_worker)
    worker.start()
    try:
        assert transaction_entered.wait(1)
        assert observed == [TRACE_CRITICAL_WRITE_WAL_AUTOCHECKPOINT_PAGES]
        assert _sqlite_settings(database)["wal_autocheckpoint"] == 1000
    finally:
        release_transaction.set()
        worker.join(3)
        database.close_connection()

    assert not worker.is_alive()
    assert errors == []


def test_off_ui_settlement_owns_checkpoint_policy_with_a_long_reader(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "trace-checkpoint-long-reader.sqlite"
    database = CharactersRAGDB(str(database_path), "trace-checkpoint-long-reader")
    repository = ConsoleTraceRepository()
    call_id = _started_call(database, repository)
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    reader = sqlite3.connect(database_path)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM console_trace_calls").fetchone()
    settlement_done = threading.Event()
    release_worker = threading.Event()
    observed: list[int] = []
    errors: list[BaseException] = []

    def settle_on_worker() -> None:
        try:
            connection = database.get_connection()
            connection.execute("PRAGMA wal_autocheckpoint=1")
            settled = coordinator.settle(
                database,
                TraceSettlementRequest(
                    call_id=call_id,
                    outcome=TraceCallState.COMPLETE,
                    response_envelope={"role": "assistant", "content": "answer"},
                    usage={"prompt_tokens": 4, "completion_tokens": 1},
                    response_started_at="2026-08-30T01:00:01Z",
                    settled_at="2026-08-30T01:00:02Z",
                ),
            )
            assert settled.state is TraceCallState.COMPLETE
            observed.append(
                int(connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0])
            )
            settlement_done.set()
            assert release_worker.wait(2)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
            settlement_done.set()
        finally:
            database.close_connection()

    worker = threading.Thread(target=settle_on_worker)
    worker.start()
    try:
        assert settlement_done.wait(1)
        assert errors == []
        assert observed == [1]
        assert _sqlite_settings(database)["wal_autocheckpoint"] == 1000
    finally:
        reader.close()
        release_worker.set()
        worker.join(3)
        database.close_connection()

    assert not worker.is_alive()
    assert errors == []


def test_non_reference_environment_requires_explicit_correctness_only_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TLDW_TRACE_LATENCY_ALLOW_NON_REFERENCE", raising=False)
    with pytest.raises(AssertionError, match="reference environment mismatch"):
        _enforce_reference_environment_policy(["python_version"])

    monkeypatch.setenv("TLDW_TRACE_LATENCY_ALLOW_NON_REFERENCE", "1")
    with pytest.warns(RuntimeWarning, match="correctness is running"):
        assert not _enforce_reference_environment_policy(["python_version"])


def _started_call(
    database: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> str:
    conversation_id = database.add_conversation({"title": "trace latency"})
    assert conversation_id is not None
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "representative request",
        }
    )
    assert message_id is not None
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    with database.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
        repository.ensure_policy(cursor, policy)
        revision = cursor.execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ? AND live_message_id = ?
                 ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id, message_id),
        ).fetchone()
        assert revision is not None
        node = repository.append_surface_node(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="message",
            reference=SemanticRevisionRef(str(revision[0])),
        )
        header = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="fresh",
            endpoint_identity="https://api.example.invalid/v1",
            generation_parameters={"temperature": 0.2, "max_tokens": 256},
            adapter_defaults={"stream": True},
            response_format={"type": "text"},
            reasoning_controls={"effort": "medium"},
            components=(),
        )
        call = repository.reserve_call(
            cursor,
            owner_id=owner.owner_id,
            segment_id=segment.segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=0,
            idempotency_key=new_opaque_id(),
            policy_id=policy.policy_id,
        )
        repository.bind_call(
            cursor,
            call_id=call.call_id,
            surface_node_id=node.node_id,
            request_header_id=header.header_id,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="fresh",
        )
        repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.DISPATCH_STARTED,
            occurred_at="2026-08-30T01:00:00Z",
        )
    return call.call_id


@pytest.mark.asyncio
async def test_production_store_schedules_real_trace_settlement_off_ui_thread(
    tmp_path: Path,
) -> None:
    """The app runtime must not settle SQLite work on Textual's loop thread."""

    database = CharactersRAGDB(str(tmp_path / "off-thread.sqlite"), "trace-latency")
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    settlement_threads: list[int] = []
    target_code = ConsoleTraceSettlementCoordinator._settle_prepared.__code__

    def profile(frame: FrameType, event: str, _arg: object) -> None:
        if event == "call" and frame.f_code is target_code:
            settlement_threads.append(threading.get_ident())

    threading.setprofile(profile)
    try:
        app = _TraceSettlementApp(database)
        runtime = ConsoleRuntime(app)
        async with app.run_test():
            ui_thread = threading.get_ident()
            store = runtime.ensure_chat_store()
            call_id = _started_call(database, repository)
            handoff = coordinator.prepare_handoff(
                database,
                TraceSettlementRequest(
                    call_id=call_id,
                    outcome=TraceCallState.COMPLETE,
                    response_envelope={"role": "assistant", "content": "answer"},
                    usage={"prompt_tokens": 4, "completion_tokens": 1},
                    response_started_at="2026-08-30T01:00:01Z",
                    settled_at="2026-08-30T01:00:02Z",
                ),
            )
            store.register_provider_trace_settlement("missing-message", handoff)
            await runtime.dispose()
            await runtime.dispose()
    finally:
        threading.setprofile(None)

    settled = repository.get_call(database.get_connection().cursor(), call_id)
    database.close_connection()
    assert settled is not None and settled.state is TraceCallState.COMPLETE
    assert settlement_threads
    assert all(thread_id != ui_thread for thread_id in settlement_threads)


@pytest.mark.asyncio
async def test_runtime_dispose_keeps_textual_ui_responsive_while_settlement_drains(
    tmp_path: Path,
) -> None:
    """A blocked trace worker must not block Textual's event-loop ticker."""

    database = CharactersRAGDB(str(tmp_path / "dispose-ticker.sqlite"), "trace-latency")
    release = threading.Event()
    handoff = _BlockingTraceHandoff("blocked-call", release)
    release_timer = threading.Timer(0.3, release.set)
    app = _TraceSettlementApp(database)
    runtime = ConsoleRuntime(app)
    try:
        async with app.run_test():
            ui_thread = threading.get_ident()
            store = runtime.ensure_chat_store()
            store.register_provider_trace_settlement("missing-message", handoff)
            assert await asyncio.to_thread(handoff.started.wait, 1)

            ticked = asyncio.Event()
            started = time.perf_counter()

            async def tick() -> None:
                await asyncio.sleep(0.05)
                ticked.set()

            tick_task = asyncio.create_task(tick())
            release_timer.start()
            dispose_tasks = [
                asyncio.create_task(runtime.dispose()),
                asyncio.create_task(runtime.dispose()),
            ]
            await ticked.wait()
            tick_elapsed = time.perf_counter() - started
            release.set()
            await asyncio.wait_for(asyncio.gather(*dispose_tasks), timeout=1)
            await tick_task
            await asyncio.wait_for(runtime.dispose(), timeout=1)

            assert tick_elapsed < 0.2
            assert handoff.settlement_threads
            assert all(
                thread_id != ui_thread for thread_id in handoff.settlement_threads
            )
    finally:
        release.set()
        release_timer.cancel()
        database.close_connection()


@pytest.mark.asyncio
async def test_trace_handoff_registered_after_runtime_close_is_not_run_on_ui_thread(
    tmp_path: Path,
) -> None:
    """A closed scheduler rejects ownership instead of settling inline."""

    database = CharactersRAGDB(str(tmp_path / "closed-handoff.sqlite"), "trace-latency")
    release = threading.Event()
    release.set()
    handoff = _BlockingTraceHandoff("late-call", release)
    app = _TraceSettlementApp(database)
    runtime = ConsoleRuntime(app)
    try:
        async with app.run_test():
            store = runtime.ensure_chat_store()
            await runtime.dispose()

            with pytest.raises(RuntimeError, match="settlement scheduler is closed"):
                store.register_provider_trace_settlement("missing-message", handoff)

            assert handoff.settlement_threads == []
    finally:
        database.close_connection()


def test_trace_settlement_scheduler_has_one_visible_bounded_work_queue() -> None:
    """One drain job owns deduplicated handoffs while its worker is stalled."""

    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore(settle_provider_traces_off_thread=True)
    executor = store._provider_trace_settlement_executor  # noqa: SLF001
    assert executor is not None
    original_submit = executor.submit
    submit_count = 0

    def counting_submit(*args, **kwargs):
        nonlocal submit_count
        submit_count += 1
        return original_submit(*args, **kwargs)

    executor.submit = counting_submit  # type: ignore[method-assign]
    release = threading.Event()
    first = _BlockingTraceHandoff("call-0", release)
    try:
        store.register_provider_trace_settlement("missing-message", first)
        assert first.started.wait(1)
        accepted = [
            _BlockingTraceHandoff(f"call-{index}", release) for index in range(1, 64)
        ]
        for handoff in accepted:
            store.register_provider_trace_settlement("missing-message", handoff)
            store.register_provider_trace_settlement("missing-message", handoff)
        rejected = _BlockingTraceHandoff("call-64", release)

        assert submit_count == 1
        assert store.pending_provider_trace_settlement_work_count() == 64
        with pytest.raises(RuntimeError, match="settlement queue is full"):
            store.register_provider_trace_settlement("missing-message", rejected)
        assert rejected.settlement_threads == []
        assert store.pending_provider_trace_settlement_work_count() == 64
    finally:
        release.set()
        store.end_app_runtime()

    assert store.pending_provider_trace_settlement_work_count() == 0
    assert all(handoff.settlement_threads for handoff in [first, *accepted])
    assert rejected.settlement_threads == []


def test_pending_trace_handoff_transfer_is_atomic_with_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing between pending removal and scheduler admission must not lose work."""

    store = ConsoleChatStore(
        persistence=object(),
        settle_provider_traces_off_thread=True,
    )
    session = store.create_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=False,
    )
    handoff = _RecordingTraceHandoff("race-call")
    store.register_provider_trace_settlement(assistant.id, handoff)

    transfer_entered = threading.Event()
    release_transfer = threading.Event()
    original_submit = store._submit_provider_trace_settlement  # noqa: SLF001

    def pause_before_scheduler_admission(*args, **kwargs) -> None:
        transfer_entered.set()
        assert release_transfer.wait(5)
        original_submit(*args, **kwargs)

    monkeypatch.setattr(
        store,
        "_submit_provider_trace_settlement",
        pause_before_scheduler_admission,
    )
    settlement_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def settle_pending() -> None:
        try:
            store._settle_provider_trace_settlements(assistant.id, None)  # noqa: SLF001
        except BaseException as exc:  # pragma: no cover - asserted below
            settlement_errors.append(exc)

    def close_store() -> None:
        try:
            store.end_app_runtime()
        except BaseException as exc:  # pragma: no cover - asserted below
            close_errors.append(exc)

    settlement_thread = threading.Thread(target=settle_pending)
    close_thread = threading.Thread(target=close_store)
    settlement_thread.start()
    assert transfer_entered.wait(1)
    close_thread.start()
    # On the broken two-lock transfer, close completes while admission is
    # paused. With atomic ownership, it waits on the store lock until release.
    assert not store._provider_trace_settlement_executor_close_complete.wait(  # noqa: SLF001
        0.2
    )
    release_transfer.set()
    settlement_thread.join(5)
    close_thread.join(5)

    assert not settlement_thread.is_alive()
    assert not close_thread.is_alive()
    assert settlement_errors == []
    assert close_errors == []
    assert handoff.canonical_message_ids == [None]
    assert handoff.settlement_threads
    assert all(
        thread_id != threading.get_ident() for thread_id in handoff.settlement_threads
    )
    assert store.pending_provider_trace_settlement_count(assistant.id) == 0
    assert store.pending_provider_trace_settlement_work_count() == 0
    assert store._provider_trace_settlement_owned_call_ids == set()  # noqa: SLF001


def test_executor_submit_failure_retains_handoff_for_off_thread_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed drain submission must leave one explicit teardown owner."""

    store = ConsoleChatStore(
        persistence=object(),
        settle_provider_traces_off_thread=True,
    )
    session = store.create_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=False,
    )
    handoff = _RecordingTraceHandoff("submit-failure-call")
    store.register_provider_trace_settlement(assistant.id, handoff)
    executor = store._provider_trace_settlement_executor  # noqa: SLF001
    assert executor is not None

    monkeypatch.setattr(
        executor,
        "submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("forced submit failure")
        ),
    )
    settlement_errors: list[BaseException] = []

    def settle_pending() -> None:
        try:
            store._settle_provider_trace_settlements(assistant.id, None)  # noqa: SLF001
        except BaseException as exc:  # pragma: no cover - asserted below
            settlement_errors.append(exc)

    settlement_thread = threading.Thread(target=settle_pending)
    settlement_thread.start()
    settlement_thread.join(5)

    teardown_errors: list[BaseException] = []

    def close_store() -> None:
        try:
            store.end_app_runtime()
        except BaseException as exc:  # pragma: no cover - asserted below
            teardown_errors.append(exc)

    teardown_thread = threading.Thread(target=close_store)
    teardown_thread.start()
    teardown_thread.join(5)

    assert not settlement_thread.is_alive()
    assert not teardown_thread.is_alive()
    assert settlement_errors == []
    assert teardown_errors == []
    assert handoff.canonical_message_ids == [None]
    assert handoff.settlement_threads
    assert all(
        thread_id != threading.get_ident() for thread_id in handoff.settlement_threads
    )
    assert store.pending_provider_trace_settlement_count(assistant.id) == 0
    assert store.pending_provider_trace_settlement_work_count() == 0
    assert store._provider_trace_settlement_owned_call_ids == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_async_store_registration_backpressures_at_the_hard_cap() -> None:
    """The 65th caller waits for one owned slot without spawning drain tasks."""

    store = ConsoleChatStore(settle_provider_traces_off_thread=True)
    release = threading.Event()
    handoffs = [
        _BlockingTraceHandoff(f"cap-call-{index}", release) for index in range(65)
    ]
    try:
        store.register_provider_trace_settlement("missing-message", handoffs[0])
        assert await asyncio.to_thread(handoffs[0].started.wait, 1)
        for handoff in handoffs[1:64]:
            store.register_provider_trace_settlement("missing-message", handoff)

        tasks_before = asyncio.all_tasks()
        registration = asyncio.create_task(
            store.register_provider_trace_settlement_async(
                "missing-message",
                handoffs[64],
            )
        )
        await asyncio.sleep(0.05)

        assert not registration.done()
        assert asyncio.all_tasks() - tasks_before == {registration}
        assert store.pending_provider_trace_settlement_work_count() == 64
        assert len(store._provider_trace_settlement_owned_call_ids) == 64  # noqa: SLF001

        release.set()
        await asyncio.wait_for(registration, timeout=2)
        await asyncio.to_thread(store.end_app_runtime)

        assert handoffs[64].settlement_threads
        assert all(
            thread_id != threading.get_ident()
            for thread_id in handoffs[64].settlement_threads
        )
        assert store.pending_provider_trace_settlement_work_count() == 0
        assert store._provider_trace_settlement_owned_call_ids == set()  # noqa: SLF001
    finally:
        release.set()
        await asyncio.to_thread(store.end_app_runtime)


@pytest.mark.asyncio
async def test_same_assistant_sixty_fifth_handoff_cannot_deadlock_terminal_save() -> (
    None
):
    """A full deferred owner promotes its oldest artifact so call 65 can finish."""

    store = ConsoleChatStore(
        persistence=object(),
        settle_provider_traces_off_thread=True,
    )
    session = store.create_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=False,
    )
    handoffs = [_RecordingTraceHandoff(f"same-owner-{index}") for index in range(65)]
    try:
        for handoff in handoffs[:64]:
            await store.register_provider_trace_settlement_async(
                assistant.id,
                handoff,
            )

        await asyncio.wait_for(
            store.register_provider_trace_settlement_async(
                assistant.id,
                handoffs[64],
            ),
            timeout=1,
        )

        assert handoffs[0].canonical_message_ids == [None]
        assert handoffs[0].settlement_threads
        assert handoffs[0].settlement_threads[0] != threading.get_ident()
        assert store.pending_provider_trace_settlement_count(assistant.id) == 64
        assert store.pending_provider_trace_settlement_work_count() == 0
        assert len(store._provider_trace_settlement_owned_call_ids) == 64  # noqa: SLF001

        canonical_message_id = "canonical-assistant-message"
        store._settle_provider_trace_settlements(  # noqa: SLF001
            assistant.id,
            canonical_message_id,
        )
        await asyncio.to_thread(store.end_app_runtime)

        assert all(
            handoff.canonical_message_ids == [canonical_message_id]
            for handoff in handoffs[1:]
        )
        assert all(
            thread_id != threading.get_ident()
            for handoff in handoffs
            for thread_id in handoff.settlement_threads
        )
        assert store.pending_provider_trace_settlement_count(assistant.id) == 0
        assert store.pending_provider_trace_settlement_work_count() == 0
        assert store._provider_trace_settlement_owned_call_ids == set()  # noqa: SLF001
    finally:
        await asyncio.to_thread(store.end_app_runtime)


@pytest.mark.asyncio
async def test_deferred_capacity_progress_wakes_same_and_cross_owner_waiters() -> None:
    """One promotion must not strand concurrent or cross-owner registrations."""

    store = ConsoleChatStore(
        persistence=object(),
        settle_provider_traces_off_thread=True,
    )
    session = store.create_session()
    owner_a = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=False,
    )
    owner_b = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=False,
    )
    release = threading.Event()
    promoted = _BlockingTraceHandoff("concurrent-owner-0", release)
    recorded_deferred = [
        _RecordingTraceHandoff(f"concurrent-owner-{index}") for index in range(1, 64)
    ]
    deferred = [promoted, *recorded_deferred]
    same_owner = _RecordingTraceHandoff("same-owner-waiter")
    cross_owner = _RecordingTraceHandoff("cross-owner-waiter")
    registrations: list[asyncio.Task[None]] = []
    try:
        for handoff in deferred:
            await store.register_provider_trace_settlement_async(
                owner_a.id,
                handoff,
            )

        registrations = [
            asyncio.create_task(
                store.register_provider_trace_settlement_async(
                    owner_a.id,
                    same_owner,
                )
            ),
            asyncio.create_task(
                store.register_provider_trace_settlement_async(
                    owner_b.id,
                    cross_owner,
                )
            ),
        ]
        assert await asyncio.to_thread(promoted.started.wait, 1)
        for _ in range(1000):
            if len(store._provider_trace_settlement_capacity_waiters) == 2:  # noqa: SLF001
                break
            await asyncio.sleep(0.001)
        assert len(store._provider_trace_settlement_capacity_waiters) == 2  # noqa: SLF001

        release.set()
        for _ in range(2000):
            if all(registration.done() for registration in registrations):
                break
            await asyncio.sleep(0.001)
        state = (
            store.pending_provider_trace_settlement_count(owner_a.id),
            store.pending_provider_trace_settlement_count(owner_b.id),
            store.pending_provider_trace_settlement_work_count(),
            len(store._provider_trace_settlement_owned_call_ids),  # noqa: SLF001
            len(store._provider_trace_settlement_capacity_waiters),  # noqa: SLF001
            store._provider_trace_settlement_worker_active,  # noqa: SLF001
            [registration.done() for registration in registrations],
            recorded_deferred[0].canonical_message_ids,
            same_owner.call_id in store._provider_trace_settlement_owned_call_ids,  # noqa: SLF001
            cross_owner.call_id in store._provider_trace_settlement_owned_call_ids,  # noqa: SLF001
        )
        assert all(registration.done() for registration in registrations), repr(state)
        await asyncio.gather(*registrations)

        assert promoted.settlement_threads
        assert recorded_deferred[0].canonical_message_ids == [None]
        assert store.pending_provider_trace_settlement_count(owner_a.id) == 63
        assert store.pending_provider_trace_settlement_count(owner_b.id) == 1
        assert store.pending_provider_trace_settlement_work_count() == 0
        assert len(store._provider_trace_settlement_owned_call_ids) == 64  # noqa: SLF001

        store._settle_provider_trace_settlements(  # noqa: SLF001
            owner_a.id,
            "canonical-owner-a",
        )
        store._settle_provider_trace_settlements(  # noqa: SLF001
            owner_b.id,
            "canonical-owner-b",
        )
        await asyncio.to_thread(store.end_app_runtime)

        assert all(
            handoff.canonical_message_ids == ["canonical-owner-a"]
            for handoff in [*recorded_deferred[1:], same_owner]
        )
        assert cross_owner.canonical_message_ids == ["canonical-owner-b"]
        assert store._provider_trace_settlement_owned_call_ids == set()  # noqa: SLF001
    finally:
        release.set()
        for registration in registrations:
            if not registration.done():
                registration.cancel()
        await asyncio.gather(*registrations, return_exceptions=True)
        await asyncio.to_thread(store.end_app_runtime)


@pytest.mark.asyncio
async def test_cancelled_capacity_wait_retains_off_thread_caller_ownership() -> None:
    """Cancellation during backpressure must not discard the sanitized handoff."""

    store = ConsoleChatStore(settle_provider_traces_off_thread=True)
    release = threading.Event()
    blockers = [
        _BlockingTraceHandoff(f"cancel-cap-{index}", release) for index in range(64)
    ]
    handoff = _RecordingTraceHandoff("cancelled-caller-owned")
    signals = ConsoleProviderStreamSignals()

    async def publish(value: object) -> None:
        await store.register_provider_trace_settlement_async(
            "missing-message",
            value,
        )

    signals.bind_trace_settlement_sink(publish)
    publish_task: asyncio.Task[bool] | None = None
    try:
        store.register_provider_trace_settlement("missing-message", blockers[0])
        assert await asyncio.to_thread(blockers[0].started.wait, 1)
        for blocker in blockers[1:]:
            store.register_provider_trace_settlement("missing-message", blocker)

        publish_task = asyncio.create_task(
            signals.new_usage_call().publish_trace_settlement(handoff)
        )
        await asyncio.sleep(0.05)
        assert not publish_task.done()
        publish_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await publish_task

        assert handoff.canonical_message_ids == [None]
        assert handoff.settlement_threads
        assert all(
            thread_id != threading.get_ident()
            for thread_id in handoff.settlement_threads
        )
        assert len(store._provider_trace_settlement_owned_call_ids) == 64  # noqa: SLF001
    finally:
        release.set()
        if publish_task is not None and not publish_task.done():
            publish_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await publish_task
        await asyncio.to_thread(store.end_app_runtime)


@pytest.mark.asyncio
async def test_full_store_gateway_handoff_waits_for_slot_and_settles_once(
    tmp_path: Path,
) -> None:
    """A real gateway must backpressure at 64 instead of settling on its caller."""

    database = CharactersRAGDB(str(tmp_path / "gateway-full.sqlite"), "trace-latency")
    repository = ConsoleTraceRepository()
    coordinator = _ThreadRecordingSettlementCoordinator(repository)
    call_id = _started_call(database, repository)
    gateway, resolution, prepared = _gateway_for_started_call(
        coordinator,
        database,
        call_id,
    )
    store = ConsoleChatStore(settle_provider_traces_off_thread=True)
    release = threading.Event()
    blockers = [
        _BlockingTraceHandoff(f"gateway-cap-{index}", release) for index in range(64)
    ]
    signals = ConsoleProviderStreamSignals()

    async def publish(handoff: object) -> None:
        await store.register_provider_trace_settlement_async(
            "missing-message",
            handoff,
        )

    signals.bind_trace_settlement_sink(publish)
    gateway_task: asyncio.Task[list[object]] | None = None
    try:
        store.register_provider_trace_settlement("missing-message", blockers[0])
        assert await asyncio.to_thread(blockers[0].started.wait, 1)
        for blocker in blockers[1:]:
            store.register_provider_trace_settlement("missing-message", blocker)

        gateway_task = asyncio.create_task(
            _consume_gateway(gateway, resolution, prepared, signals)
        )
        await asyncio.sleep(0.05)

        assert not gateway_task.done()
        assert store.pending_provider_trace_settlement_work_count() == 64
        assert len(store._provider_trace_settlement_owned_call_ids) == 64  # noqa: SLF001

        release.set()
        assert await asyncio.wait_for(gateway_task, timeout=3) == ["answer"]
        await asyncio.to_thread(store.end_app_runtime)

        settled = repository.get_call(database.get_connection().cursor(), call_id)
        assert settled is not None and settled.state is TraceCallState.COMPLETE
        assert len(coordinator.settlement_threads) == 1
        assert coordinator.settlement_threads[0] != threading.get_ident()
        assert store.pending_provider_trace_settlement_work_count() == 0
        assert store._provider_trace_settlement_owned_call_ids == set()  # noqa: SLF001
    finally:
        release.set()
        if gateway_task is not None and not gateway_task.done():
            await asyncio.wait_for(gateway_task, timeout=3)
        await asyncio.to_thread(store.end_app_runtime)
        database.close_connection()


@pytest.mark.asyncio
async def test_post_close_gateway_handoff_uses_awaited_off_thread_owner(
    tmp_path: Path,
) -> None:
    """A producer-fence violation stays caller-owned without UI-thread SQLite."""

    database = CharactersRAGDB(
        str(tmp_path / "gateway-post-close.sqlite"),
        "trace-latency",
    )
    repository = ConsoleTraceRepository()
    coordinator = _ThreadRecordingSettlementCoordinator(repository)
    call_id = _started_call(database, repository)
    gateway, resolution, prepared = _gateway_for_started_call(
        coordinator,
        database,
        call_id,
    )
    store = ConsoleChatStore(settle_provider_traces_off_thread=True)
    await asyncio.to_thread(store.end_app_runtime)
    signals = ConsoleProviderStreamSignals()

    async def publish(handoff: object) -> None:
        await store.register_provider_trace_settlement_async(
            "missing-message",
            handoff,
        )

    signals.bind_trace_settlement_sink(publish)
    try:
        assert await _consume_gateway(gateway, resolution, prepared, signals) == [
            "answer"
        ]

        settled = repository.get_call(database.get_connection().cursor(), call_id)
        assert settled is not None and settled.state is TraceCallState.COMPLETE
        assert len(coordinator.settlement_threads) == 1
        assert coordinator.settlement_threads[0] != threading.get_ident()
        assert store.pending_provider_trace_settlement_work_count() == 0
        assert store._provider_trace_settlement_owned_call_ids == set()  # noqa: SLF001
    finally:
        database.close_connection()


@pytest.mark.benchmark
def test_console_trace_call_persistence_latency_reference_gate(tmp_path: Path) -> None:
    """Gate real reservation/dispatch and settlement transactions on the fixture."""

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    warmup_count = fixture["samples"]["warmup_per_operation_per_database"]
    measured_count = fixture["samples"]["measured_per_operation_per_database"]
    database_count = fixture["samples"]["fresh_database_count"]
    assert (warmup_count, measured_count, database_count) == (100, 1_000, 5)
    environment = _environment_metadata(tmp_path)
    mismatches = _environment_mismatches(fixture, environment)
    expected_sqlite = fixture["sqlite_settings"]
    trace_operation_policy = fixture["trace_operation_policy"]
    runs: list[_RunResult] = []

    logger.disable("tldw_chatbook")
    try:
        for run_index in range(database_count):
            database_path = tmp_path / f"trace-latency-{run_index}.sqlite"
            wal_path = Path(f"{database_path}-wal")
            database = CharactersRAGDB(
                str(database_path),
                f"trace-latency-{run_index}",
            )
            closed = False
            try:
                sqlite_settings = _sqlite_settings(database)
                assert sqlite_settings == expected_sqlite
                repository = ConsoleTraceRepository()
                service = ConsoleTraceService(repository)
                coordinator = ConsoleTraceSettlementCoordinator(repository)
                reservation_samples: list[int] = []
                reservation_only_samples: list[int] = []
                dispatch_only_samples: list[int] = []
                reservation_process_cpu_samples: list[int] = []
                reservation_thread_cpu_samples: list[int] = []
                warmup_call_ids: list[str] = []
                for index in range(warmup_count):
                    prepared = _prepare_call(
                        database,
                        repository,
                        service,
                        sequence=index,
                    )
                    reserved = prepared.boundary.reserve()
                    dispatched = prepared.boundary.mark_dispatch_started(
                        prepared.bundle,
                        prepared.provenance,
                    )
                    assert reserved.call_id == dispatched.call_id
                    assert dispatched.state is TraceCallState.DISPATCH_STARTED
                    warmup_call_ids.append(dispatched.call_id)
                for call_id in warmup_call_ids:
                    settled = coordinator.settle(
                        database,
                        TraceSettlementRequest(
                            call_id=call_id,
                            outcome=TraceCallState.COMPLETE,
                            response_envelope={
                                "role": "assistant",
                                "content": "representative response",
                            },
                            usage={"prompt_tokens": 32, "completion_tokens": 8},
                            response_started_at="2026-08-30T01:00:01Z",
                            settled_at="2026-08-30T01:00:02Z",
                        ),
                    )
                    assert settled.state is TraceCallState.COMPLETE

                started_call_ids: list[str] = []
                for index in range(measured_count):
                    # Prepare immediately before the real operation, as the
                    # Console does. Retaining 1,100 complete boundary graphs
                    # made an unrelated whole-heap pause land inside an arbitrary
                    # timed dispatch and measured fixture retention, not the call.
                    prepared = _prepare_call(
                        database,
                        repository,
                        service,
                        sequence=warmup_count + index,
                    )
                    started_ns = time.perf_counter_ns()
                    started_process_cpu_ns = time.process_time_ns()
                    started_thread_cpu_ns = time.thread_time_ns()
                    reserved = prepared.boundary.reserve()
                    reserved_ns = time.perf_counter_ns()
                    dispatched = prepared.boundary.mark_dispatch_started(
                        prepared.bundle,
                        prepared.provenance,
                    )
                    dispatched_thread_cpu_ns = time.thread_time_ns()
                    dispatched_process_cpu_ns = time.process_time_ns()
                    dispatched_ns = time.perf_counter_ns()
                    assert reserved.call_id == dispatched.call_id
                    assert dispatched.state is TraceCallState.DISPATCH_STARTED
                    started_call_ids.append(dispatched.call_id)
                    reservation_samples.append(dispatched_ns - started_ns)
                    reservation_only_samples.append(reserved_ns - started_ns)
                    dispatch_only_samples.append(dispatched_ns - reserved_ns)
                    reservation_process_cpu_samples.append(
                        dispatched_process_cpu_ns - started_process_cpu_ns
                    )
                    reservation_thread_cpu_samples.append(
                        dispatched_thread_cpu_ns - started_thread_cpu_ns
                    )

                wal_file_bytes_before_settlement = _file_bytes(wal_path)
                settlement_samples: list[int] = []
                settlement_process_cpu_samples: list[int] = []
                settlement_thread_cpu_samples: list[int] = []
                for call_id in started_call_ids:
                    started_ns = time.perf_counter_ns()
                    started_process_cpu_ns = time.process_time_ns()
                    started_thread_cpu_ns = time.thread_time_ns()
                    settled = coordinator.settle(
                        database,
                        TraceSettlementRequest(
                            call_id=call_id,
                            outcome=TraceCallState.COMPLETE,
                            response_envelope={
                                "role": "assistant",
                                "content": "representative response",
                            },
                            usage={"prompt_tokens": 32, "completion_tokens": 8},
                            response_started_at="2026-08-30T01:00:01Z",
                            settled_at="2026-08-30T01:00:02Z",
                        ),
                    )
                    settled_thread_cpu_ns = time.thread_time_ns()
                    settled_process_cpu_ns = time.process_time_ns()
                    elapsed_ns = time.perf_counter_ns() - started_ns
                    assert settled.state is TraceCallState.COMPLETE
                    settlement_samples.append(elapsed_ns)
                    settlement_process_cpu_samples.append(
                        settled_process_cpu_ns - started_process_cpu_ns
                    )
                    settlement_thread_cpu_samples.append(
                        settled_thread_cpu_ns - started_thread_cpu_ns
                    )
                wal_file_bytes_after_settlement = _file_bytes(wal_path)

                assert len(reservation_samples) == measured_count
                assert len(reservation_only_samples) == measured_count
                assert len(dispatch_only_samples) == measured_count
                assert len(reservation_process_cpu_samples) == measured_count
                assert len(reservation_thread_cpu_samples) == measured_count
                assert len(settlement_samples) == measured_count
                assert len(settlement_process_cpu_samples) == measured_count
                assert len(settlement_thread_cpu_samples) == measured_count
                close_started_ns = time.perf_counter_ns()
                database.close_connection()
                close_ns = time.perf_counter_ns() - close_started_ns
                closed = True
                wal_file_bytes_after_close = _file_bytes(wal_path)
                runs.append(
                    {
                        "run_index": run_index,
                        "reservation_dispatch_started_ns": reservation_samples,
                        "reservation_dispatch_started_summary": _summary(
                            reservation_samples
                        ),
                        "reservation_only_ns": reservation_only_samples,
                        "reservation_only_summary": _summary(reservation_only_samples),
                        "dispatch_started_only_ns": dispatch_only_samples,
                        "dispatch_started_only_summary": _summary(
                            dispatch_only_samples
                        ),
                        "reservation_dispatch_started_process_cpu_ns": (
                            reservation_process_cpu_samples
                        ),
                        "reservation_dispatch_started_thread_cpu_ns": (
                            reservation_thread_cpu_samples
                        ),
                        "settlement_ns": settlement_samples,
                        "settlement_summary": _summary(settlement_samples),
                        "settlement_process_cpu_ns": settlement_process_cpu_samples,
                        "settlement_thread_cpu_ns": settlement_thread_cpu_samples,
                        "sqlite_settings": sqlite_settings,
                        "wal_file_bytes_before_settlement": (
                            wal_file_bytes_before_settlement
                        ),
                        "wal_file_bytes_after_settlement": (
                            wal_file_bytes_after_settlement
                        ),
                        "wal_file_bytes_after_close": wal_file_bytes_after_close,
                        "close_ns": close_ns,
                    }
                )
            finally:
                if not closed:
                    database.close_connection()
    finally:
        logger.enable("tldw_chatbook")

    all_reservations = [
        sample for run in runs for sample in run["reservation_dispatch_started_ns"]
    ]
    all_settlements = [sample for run in runs for sample in run["settlement_ns"]]
    threshold_gate_applied = not mismatches
    report = {
        "fixture_version": fixture["fixture_version"],
        "environment": environment,
        "environment_match": not mismatches,
        "environment_mismatches": mismatches,
        "p95_formula": fixture["p95_formula"],
        "diagnostic_clocks": fixture["diagnostic_clocks"],
        "operation_order": fixture["operation_order"],
        "samples": fixture["samples"],
        "sqlite_settings": expected_sqlite,
        "trace_operation_policy": trace_operation_policy,
        "threshold_gate_applied": threshold_gate_applied,
        "thresholds_ns": fixture["thresholds_ns"],
        "runs": runs,
        "aggregate": {
            "reservation_dispatch_started": _summary(all_reservations),
            "settlement": _summary(all_settlements),
        },
    }
    artifact_path = tmp_path / ARTIFACT_NAME
    artifact_path.write_text(
        json.dumps(report, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"trace latency artifact: {artifact_path}")

    _enforce_reference_environment_policy(mismatches)
    if not threshold_gate_applied:
        return

    thresholds = fixture["thresholds_ns"]
    reservation_p95_limit = thresholds["reservation_p95"]
    reservation_max_limit = thresholds["reservation_max"]
    settlement_p95_limit = thresholds["settlement_p95"]
    for run in runs:
        assert (
            run["reservation_dispatch_started_summary"]["p95_ns"]
            <= reservation_p95_limit
        )
        assert (
            run["reservation_dispatch_started_summary"]["max_ns"]
            <= reservation_max_limit
        )
        assert run["settlement_summary"]["p95_ns"] <= settlement_p95_limit
    assert _p95(all_reservations) <= reservation_p95_limit
    assert max(all_reservations) <= reservation_max_limit
    assert _p95(all_settlements) <= settlement_p95_limit
