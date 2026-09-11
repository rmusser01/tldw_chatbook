"""Durable-owner qualification for cancelled speculative voice attempts."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
import traceback
from uuid import uuid4

from loguru import logger
import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    freeze_provisional_capture_eligibility,
)
from tldw_chatbook.Chat.console_exchange_export import (
    TraceExportProfile,
    project_exchange_export,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.Chat.console_speculative_voice import (
    AttemptDispatchPrepared,
    AttemptOutputDelta,
)
from tldw_chatbook.Chat.console_speculative_voice_session import (
    PreparedSpeculativeVoiceAttempt,
    SpeculativeVoiceAttemptEffects,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.console_voice_attempts import (
    AttemptCleanupOutcome,
    VoiceAttemptRequest,
)
from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy
from tldw_chatbook.Chat.console_voice_trace_gateway import (
    ProvisionalTraceUnavailable,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.Chat.trajectory_export import build_trajectory_export
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Logging_Config import PrivateRotatingFileHandler, RichLogHandler
from tldw_chatbook.Notifications.notification_presentation import (
    NotificationPresentationStore,
)
from tldw_chatbook.Utils.persistent_diagnostics import (
    PersistentDiagnosticFilter,
    persist_event,
)


INVENTORY_PATH = (
    Path(__file__).parents[2]
    / "Docs/Development/TTS/speculative-voice-durable-owner-inventory.md"
)
OWNER_IDS = frozenset(
    {
        "console_messages",
        "terminal_marks",
        "trace_capture",
        "provider_usage",
        "tools_approvals",
        "citations",
        "notifications",
        "replay_trajectory",
        "chatbook_exports",
        "temporary_chat",
        "persistent_files",
        "runtime_logs",
        "in_app_logs",
        "exception_diagnostics",
    }
)

USER_POISON = "VOICE-USER-POISON-23175"
ASSISTANT_POISON = "VOICE-ASSISTANT-POISON-23175"
POISONS = (USER_POISON, ASSISTANT_POISON)


def test_durable_owner_inventory_matches_the_qualification_probe() -> None:
    owner_ids = frozenset(
        re.findall(r"^\| `([^`]+)` \|", INVENTORY_PATH.read_text(), re.MULTILINE)
    )

    assert owner_ids == OWNER_IDS


def _turn_context() -> ConsoleTurnExecutionContext:
    policy = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        policy_revision=1,
        source="durable",
    )
    scope = ConsoleLibraryItemScopeSnapshot((), (), True)
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id="voice-owner-session",
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="gpt-test",
        ),
        library_policy_maximum=policy,
        library_scope_maximum=scope,
    )
    return ConsoleTurnExecutionContext(
        configuration=configuration,
        library_authority=ConsoleTurnLibraryAuthority(
            policy=policy,
            direct_library_tools=True,
            source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
            scope_snapshot=scope,
            provider_intent=ConsoleProviderIntent("openai", "gpt-test", None),
            attempt_id="voice-owner-authority",
        ),
        resolved_destination=ConsoleResolvedDestination(
            provider="openai",
            model="gpt-test",
            endpoint_identity="https://api.openai.com",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    )


class _CancellationGateway:
    """Exercise real signals and trace abandonment, then fail after cancellation."""

    def __init__(self, trace_owner: ConsoleProviderGateway) -> None:
        self._trace_owner = trace_owner
        self.signals: list[ConsoleProviderStreamSignals] = []

    async def stream_chat(
        self,
        _resolution: object,
        _prepared: object,
        *,
        tools: object,
        signals: ConsoleProviderStreamSignals,
    ):
        del tools
        self.signals.append(signals)
        call = signals.new_usage_call()
        call.record_usage_payload(
            {"input_tokens": 1, "provisional_content": USER_POISON}
        )
        call.begin_exchange(
            provider="openai",
            model="gpt-test",
            endpoint=None,
            request={"content": USER_POISON},
            omitted_keys=(),
        )
        call.record_exchange_content(ASSISTANT_POISON)
        call.close_exchange(status="complete")
        call.close_usage_call()
        yield f"{ASSISTANT_POISON}."
        try:
            await asyncio.Future()
        except asyncio.CancelledError as exc:
            raise RuntimeError(ASSISTANT_POISON) from exc

    def abandon_provisional_voice_trace(self, trace_attempt: object) -> None:
        self._trace_owner.abandon_provisional_voice_trace(trace_attempt)


class _BlockingSynthesizer:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def synthesize_hands_free(self, *, text: str) -> object:
        assert ASSISTANT_POISON in text
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


class _Transport:
    clock_generation = 0

    def __init__(self) -> None:
        self.abort_count = 0

    async def abort_output(self) -> None:
        self.abort_count += 1

    def fence_output(self) -> None:
        self.abort_count += 1

    def queue_render(self, _pcm16: bytes) -> None:
        raise AssertionError("cancelled synthesis must not render audio")


class _PreviewOwner:
    def __init__(self) -> None:
        self.current: object | None = None
        self.projected = asyncio.Event()

    def project(self, value: object) -> None:
        self.current = value
        self.projected.set()

    def clear(self) -> None:
        self.current = None


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        rendered = self.format(record)
        if record.exc_info:
            rendered += "\n" + "".join(traceback.format_exception(*record.exc_info))
        self.lines.append(rendered)


class _RichLogWidget:
    is_mounted = False
    app = None

    def write(self, _message: str) -> None:
        raise AssertionError("the qualification probe reads the real handler queue")


@dataclass(slots=True)
class _LogCapture:
    path: Path
    persistent: PrivateRotatingFileHandler
    runtime: _ListHandler
    rich: RichLogHandler
    loguru_lines: list[str]


@pytest.fixture
def voice_owner_logs(tmp_path: Path):
    """Install every local log owner before the cancelled attempt starts."""

    log_path = tmp_path / "app-data" / "logs" / "chatbook.log"
    persistent = PrivateRotatingFileHandler(log_path)
    persistent.addFilter(PersistentDiagnosticFilter())
    persistent.setFormatter(logging.Formatter("%(name)s %(message)s"))
    runtime = _ListHandler()
    rich = RichLogHandler(_RichLogWidget())  # type: ignore[arg-type]
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(persistent)
    root_logger.addHandler(runtime)
    root_logger.addHandler(rich)
    loguru_lines: list[str] = []
    sink_id = logger.add(loguru_lines.append, format="{message}", diagnose=False)
    try:
        yield _LogCapture(log_path, persistent, runtime, rich, loguru_lines)
    finally:
        logger.remove(sink_id)
        root_logger.removeHandler(rich)
        root_logger.removeHandler(runtime)
        root_logger.removeHandler(persistent)
        root_logger.setLevel(old_level)
        rich.close()
        runtime.close()
        persistent.close()


def _sqlite_text(db: CharactersRAGDB, tables: set[str] | None = None) -> str:
    connection = db.get_connection()
    names = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        if tables is None or str(row[0]) in tables
    ]
    values: list[str] = []
    for name in names:
        quoted = name.replace('"', '""')
        try:
            rows = connection.execute(f'SELECT * FROM "{quoted}"').fetchall()
        except Exception:  # pragma: no cover - virtual table internals vary by SQLite
            continue
        values.append(name)
        values.extend(repr(tuple(row)) for row in rows)
    return "\n".join(values)


def _assert_live_clean(
    owners: dict[str, str],
    controls: dict[str, str],
) -> None:
    assert owners.keys() == controls.keys() == OWNER_IDS
    for owner_id in sorted(OWNER_IDS):
        rendered = owners[owner_id]
        assert controls[owner_id] in rendered, f"{owner_id} probe is not live"
        for poison in POISONS:
            assert poison not in rendered, f"{owner_id} retained cancelled content"


@pytest.mark.asyncio
async def test_cancelled_attempt_is_absent_from_every_default_durable_owner(
    tmp_path: Path,
    voice_owner_logs: _LogCapture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "app-data" / "voice-owner.sqlite"
    db_path.parent.mkdir(exist_ok=True)
    db = CharactersRAGDB(db_path, "voice-owner-probe")
    persistence = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=persistence)
    trace_gateway = ConsoleProviderGateway(http_client=object())  # type: ignore[arg-type]
    cancellation_gateway = _CancellationGateway(trace_gateway)
    synthesizer = _BlockingSynthesizer()
    transport = _Transport()
    preview = _PreviewOwner()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=trace_gateway,
        agent_runtime_enabled=False,
    )

    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-test",
        ready=True,
        execution_key="openai",
    )
    trace_attempt = trace_gateway.begin_provisional_voice_trace(
        policy=FrozenTracePolicy(str(uuid4()), "credentials-v1", False, None),
        promotion_id=str(uuid4()),
        attempt_id=str(uuid4()),
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True,
            session_is_saved=True,
        ),
    )
    assert trace_attempt is not None
    prepared_request = trace_gateway.prepare_chat_request(
        resolution,
        [{"role": "user", "content": USER_POISON}],
    )
    prepared = PreparedSpeculativeVoiceAttempt(
        request=VoiceAttemptRequest(
            attempt_epoch=1,
            resolution=resolution,
            prepared=prepared_request,
            exchange_capture_enabled=True,
            provisional_trace_attempt=trace_attempt,
        ),
        frozen_session_context=_turn_context(),
    )
    effects: SpeculativeVoiceAttemptEffects

    async def submit_event(event: object) -> None:
        if isinstance(event, AttemptDispatchPrepared):
            effects.start_prepared_attempt(event.attempt_epoch)
        elif isinstance(event, AttemptOutputDelta):
            effects.publish_preview(event.attempt_epoch, event.text)

    effects = SpeculativeVoiceAttemptEffects(
        submit_event=submit_event,
        prepare_attempt=lambda **_kwargs: asyncio.sleep(0, result=prepared),
        gateway=cancellation_gateway,
        synthesizer=synthesizer,
        transport=transport,
        promotion=lambda **_kwargs: pytest.fail("cancelled attempt was promoted"),
        promotion_owner=object(),
        dispatch_supervisor=VoiceDispatchSupervisor(),
        project_preview=preview.project,
        clear_preview=preview.clear,
        submit_accepted_voice_turn=lambda *_args: pytest.fail(
            "cancelled attempt entered ordinary submission"
        ),
    )

    loop = asyncio.get_running_loop()
    loop_exception_contexts: list[dict[str, object]] = []
    prior_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, context: loop_exception_contexts.append(dict(context))
    )

    effects.dispatch_attempt(
        turn_id="voice-poison-turn",
        attempt_epoch=1,
        transcript=USER_POISON,
    )
    async with asyncio.timeout(1):
        await preview.projected.wait()
    assert getattr(preview.current, "user_text") == USER_POISON
    assert getattr(preview.current, "assistant_text") == f"{ASSISTANT_POISON}."
    assert USER_POISON not in repr(preview.current)
    assert ASSISTANT_POISON not in repr(preview.current)

    effects.fence_attempt(1)
    await effects.abort_output(1)
    outcome = await effects.cancel_attempt(1)
    effects.clear_preview(1)
    await effects.close()

    assert outcome is AttemptCleanupOutcome.CLEAN
    assert transport.abort_count == 1
    assert preview.current is None
    assert cancellation_gateway.signals
    for signals in cancellation_gateway.signals:
        assert signals.usage_payloads() == []
        assert signals.exchange_captures() == []
    assert trace_gateway.provisional_trace_registry.retained_bytes == 0
    with pytest.raises(ProvisionalTraceUnavailable):
        trace_gateway.seal_provisional_voice_trace(trace_attempt)

    # Establish one real, ordinary durable conversation as the positive control.
    console_control = "CONSOLE-CONTROL-23175"
    session = store.create_session(title="voice owner control")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content=console_control,
        persist=True,
    )
    assert message.persisted_message_id is not None
    saved_session = next(item for item in store.sessions() if item.id == session.id)
    assert saved_session.persisted_conversation_id is not None

    marks_control = "MARKS-CONTROL-23175"
    marks_service = ConversationLocalMarksService(db)
    with db.transaction(immediate=True) as cursor:
        marks_service.set_console_terminal_with_cursor(
            cursor,
            marks_control,
            str(uuid4()),
            "complete",
            created_at="2026-08-31T00:00:00Z",
            updated_at="2026-08-31T00:00:00Z",
        )

    usage_control = "USAGE-CONTROL-23175"
    assert db.update_message_usage_local(
        message.persisted_message_id,
        json.dumps({"control": usage_control}),
    )

    trajectory_control = "TRAJECTORY-CONTROL-23175"
    connection = db.get_connection()
    row = connection.execute(
        "SELECT seq FROM message_trajectory_metadata WHERE message_id = ? LIMIT 1",
        (message.persisted_message_id,),
    ).fetchone()
    if row is None:
        connection.execute(
            """INSERT INTO message_trajectory_metadata(
                   message_id, conversation_id, turn_id, seq, event_kind, provider
               ) VALUES (?, ?, ?, 1, 'user', ?)""",
            (
                message.persisted_message_id,
                saved_session.persisted_conversation_id,
                message.persisted_message_id,
                trajectory_control,
            ),
        )
    else:
        connection.execute(
            "UPDATE message_trajectory_metadata SET provider = ? WHERE message_id = ?",
            (trajectory_control, message.persisted_message_id),
        )

    citation_control = "CITATION-CONTROL-23175"
    profile_id = connection.execute(
        "SELECT profile_id FROM rag_identity_context WHERE context_name = 'default'"
    ).fetchone()[0]
    connection.execute(
        """INSERT INTO rag_citation_traces(
               profile_id, trace_id, schema_version, request_id, generation_id,
               origin_scope_id, origin, lifecycle, completeness_at_seal,
               selected_attempt_id, policy_version, aggregate_json,
               visibility_state, created_at, sealed_at
           ) VALUES (?, ?, 1, ?, ?, ?, 'local', 'sealed', 'complete', ?, ?, ?,
                     'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
        (
            profile_id,
            citation_control,
            str(uuid4()),
            str(uuid4()),
            profile_id,
            str(uuid4()),
            "voice-owner-probe-v1",
            json.dumps({"control": citation_control}),
        ),
    )
    connection.commit()

    approval_control = "APPROVAL-CONTROL-23175"
    controller.add_pending_round(approval_control, approval_control)

    notification_control = "NOTIFICATION-CONTROL-23175"
    notifications = NotificationPresentationStore()
    notifications.mark_failed(
        notification_control,
        delivery_error=notification_control,
    )

    trace_control = str(uuid4())
    control_trace = trace_gateway.begin_provisional_voice_trace(
        policy=FrozenTracePolicy(str(uuid4()), "credentials-v1", False, None),
        promotion_id=trace_control,
        attempt_id=str(uuid4()),
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True,
            session_is_saved=True,
        ),
    )
    assert control_trace is not None

    export_control = "EXPORT-CONTROL-23175"
    export_signals = ConsoleProviderStreamSignals(
        exchange_capture_enabled=True,
        capture_detail=CaptureDetail.FULL,
    )
    export_call = export_signals.new_usage_call()
    export_call.begin_exchange(
        provider="openai",
        model="gpt-test",
        endpoint=None,
        request={"control": export_control},
        omitted_keys=(),
    )
    export_call.record_exchange_content(export_control)
    export_call.close_exchange(status="complete")
    export_call.close_usage_call()
    exchange_export = project_exchange_export(
        export_signals.exchange_captures()[0],
        TraceExportProfile.FULL_TRACE,
    ).json_text
    trajectory_export = json.dumps(
        build_trajectory_export(db, saved_session.persisted_conversation_id),
        default=str,
    )

    temporary_control = "TEMPORARY-CONTROL-23175"
    temporary_session = store.create_session(
        title="temporary owner control",
        ephemeral=True,
    )
    store.append_message(
        temporary_session.id,
        role=ConsoleMessageRole.USER,
        content=temporary_control,
    )
    temporary_projection = json.dumps(
        [asdict(item) for item in store.messages_for_session(temporary_session.id)],
        default=str,
    )
    promoted_temporary_id = store.promote_ephemeral_session(temporary_session.id)
    assert promoted_temporary_id is not None
    temporary_save_later_projection = json.dumps(
        build_trajectory_export(db, promoted_temporary_id),
        default=str,
    )

    persistent_control = "voice_owner_control_23175"
    runtime_control = "RUNTIME-CONTROL-23175"
    in_app_control = "IN-APP-CONTROL-23175"
    exception_control = "EXCEPTION-CONTROL-23175"
    persist_event("voiceprobe", persistent_control, status="ok")
    logging.getLogger("voice-owner-probe").warning(runtime_control)
    logger.warning(runtime_control)
    print(runtime_control)
    logging.getLogger("voice-owner-probe").warning(in_app_control)
    loop.call_exception_handler(
        {
            "message": exception_control,
            "exception": RuntimeError(exception_control),
        }
    )
    await asyncio.sleep(0)
    voice_owner_logs.persistent.flush()
    loop.set_exception_handler(prior_exception_handler)
    terminal_output = capfd.readouterr()

    rich_lines: list[str] = []
    while not voice_owner_logs.rich.log_queue.empty():
        rich_lines.append(voice_owner_logs.rich.log_queue.get_nowait())

    try:
        raise RuntimeError(exception_control)
    except RuntimeError as exc:
        exception_text = repr(exc) + "\n" + traceback.format_exc()

    sqlite_all = _sqlite_text(db)
    owner_texts = {
        "console_messages": sqlite_all,
        "terminal_marks": _sqlite_text(db, {"conversation_local_marks"}),
        "trace_capture": repr(control_trace),
        "provider_usage": _sqlite_text(db, {"messages", "console_trace_calls"}),
        "tools_approvals": repr(
            (
                controller._pending_approvals,
                controller._pending_approval_rounds,
                controller._parked_approval_payloads,
            )
        ),
        "citations": _sqlite_text(
            db,
            {"rag_citation_traces", "rag_message_trace_owners"},
        ),
        "notifications": repr(notifications.get(notification_control)),
        "replay_trajectory": trajectory_export,
        "chatbook_exports": trajectory_export + "\n" + exchange_export,
        "temporary_chat": (
            temporary_projection + "\n" + temporary_save_later_projection
        ),
        "persistent_files": voice_owner_logs.path.read_text(),
        "runtime_logs": "\n".join(
            voice_owner_logs.runtime.lines + voice_owner_logs.loguru_lines
        )
        + terminal_output.out
        + terminal_output.err,
        "in_app_logs": "\n".join(rich_lines),
        "exception_diagnostics": exception_text + "\n" + repr(loop_exception_contexts),
    }
    controls = {
        "console_messages": console_control,
        "terminal_marks": marks_control,
        "trace_capture": trace_control,
        "provider_usage": usage_control,
        "tools_approvals": approval_control,
        "citations": citation_control,
        "notifications": notification_control,
        "replay_trajectory": trajectory_control,
        "chatbook_exports": export_control,
        "temporary_chat": temporary_control,
        "persistent_files": persistent_control,
        "runtime_logs": runtime_control,
        "in_app_logs": in_app_control,
        "exception_diagnostics": exception_control,
    }
    _assert_live_clean(owner_texts, controls)

    # The all-table query and the raw SQLite file family are separate probes:
    # the latter catches content in freelists/WAL pages that no live row exposes.
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for path in db_path.parent.glob(f"{db_path.name}*"):
        raw = path.read_bytes()
        for poison in POISONS:
            assert poison.encode() not in raw, f"{path.name} retained cancelled content"

    assert sum(runtime_control in line for line in voice_owner_logs.runtime.lines) >= 1
    assert sum(runtime_control in line for line in voice_owner_logs.loguru_lines) >= 1
    trace_gateway.abandon_provisional_voice_trace(control_trace)
    db.close_connection()
