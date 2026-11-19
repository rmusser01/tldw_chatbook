"""Frozen report sources remain scoped and owned through timeout/cancellation."""

import asyncio
from collections import deque
import logging
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Logging_Config import LogsBufferHandler
from tldw_chatbook.UI.Navigation.screen_state_store import RuntimeIdentity
from tldw_chatbook.Utils.bug_report import BugReportEvidence, BugReportItem
from tldw_chatbook.UI.bug_report_capture import (
    BugReportCollector,
    capture_bug_report_context,
)

pytestmark = pytest.mark.bootstrap_profile


def _app(with_session=False):
    session = SimpleNamespace(
        id="session-1",
        persisted_conversation_id="conversation-1",
        workspace_id="workspace-1",
        title="PRIVATE_TITLE",
    )
    store = SimpleNamespace(
        active_session_id=session.id,
        sessions=lambda: [session],
        active_session_epoch=lambda: 1,
        persistence=SimpleNamespace(db=None),
    )
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(chat_store=store if with_session else None),
        _current_runtime_identity=lambda: RuntimeIdentity("local"),
        _log_buffer=deque(maxlen=10000),
        _log_records=deque(maxlen=10000),
        _bug_report_collectors=None,
        _current_logs_window=None,
        _current_log_widget=None,
        current_tab="logs",
        screen=None,
        is_web=False,
    )
    app._persistent_log_handler = LogsBufferHandler(app)
    return app


def test_log_snapshot_is_bounded_sanitized_and_immutable_during_append():
    app = _app()
    for index in range(8):
        app._persistent_log_handler.handle(
            logging.LogRecord(
                "diagnostic",
                logging.ERROR,
                __file__,
                1,
                f"entry-{index} api_key=synthetic-secret",
                (),
                None,
            )
        )
    watermark, buffered, rows = app._persistent_log_handler.snapshot_report_tail(3)
    assert watermark == 8 and buffered == 8 and len(rows) == 3
    assert "entry-5" in rows[0][2] and "entry-7" in rows[-1][2]
    assert "synthetic-secret" not in str(rows)
    app._persistent_log_handler.handle(
        logging.LogRecord("diagnostic", logging.INFO, __file__, 1, "later", (), None)
    )
    assert len(rows) == 3 and "later" not in str(rows)


def test_context_does_not_initialize_console_and_suppresses_identity_repr():
    app = _app()
    context = capture_bug_report_context(app)
    assert context.conversation_id is None
    assert app.console_runtime.chat_store is None
    selected = capture_bug_report_context(_app(True))
    assert selected.conversation_id == "conversation-1"
    assert "PRIVATE_TITLE" not in repr(selected)


async def test_collection_without_console_still_produces_log_and_environment_evidence():
    app = _app()
    app._persistent_log_handler.handle(
        logging.LogRecord(
            "diagnostic", logging.WARNING, __file__, 1, "captured failure", (), None
        )
    )
    collector = BugReportCollector(app, capture_bug_report_context(app))
    captured = await collector.collect()
    assert captured.evidence.tools == ()
    assert "captured failure" in captured.evidence.logs[0].text
    assert captured.evidence.logs[0].observed_times
    assert captured.evidence.environment
    assert "Application-wide" in " ".join(captured.evidence.inventory)
    assert app.console_runtime.chat_store is None


async def test_environment_failure_does_not_discard_log_evidence(monkeypatch):
    app = _app()
    app._persistent_log_handler.handle(
        logging.LogRecord(
            "diagnostic", logging.ERROR, __file__, 1, "retained failure", (), None
        )
    )
    collector = BugReportCollector(app, capture_bug_report_context(app))

    def failed_environment():
        raise OSError("PRIVATE_ENVIRONMENT_ERROR")

    monkeypatch.setattr(collector, "_environment", failed_environment)
    capture = await collector.collect()
    assert "retained failure" in capture.evidence.logs[0].text
    assert "PRIVATE_ENVIRONMENT_ERROR" not in str(capture.evidence.inventory)


async def test_source_deadline_keeps_read_owned_until_retirement(monkeypatch):
    app = _app(True)
    collector = BugReportCollector(app, capture_bug_report_context(app))
    collector.presentation_timeout = 0.03
    entered, release = threading.Event(), threading.Event()
    calls = []

    def stalled():
        calls.append(True)
        entered.set()
        release.wait(3)
        return BugReportEvidence(trace=(BugReportItem("trace", "late trace"),)), ()

    monkeypatch.setattr(collector, "_read_trace", stalled)
    try:
        captured = await collector.collect()
        assert entered.is_set() and not captured.evidence.trace
        assert "trace" in collector.pending_sources
        await collector.collect()
        assert len(calls) == 1  # timeout did not orphan a read or launch an overlap
        collector.cancel()
    finally:
        release.set()
        await collector.wait_retired()
    assert collector.pending_sources == ()


async def test_workspace_change_rejects_late_capture(monkeypatch):
    app = _app(True)
    collector = BugReportCollector(app, capture_bug_report_context(app))

    def changed():
        app._current_runtime_identity = lambda: RuntimeIdentity(
            "server", "different-server"
        )
        return BugReportEvidence(trace=(BugReportItem("trace", "WRONG_CONTEXT"),)), ()

    monkeypatch.setattr(collector, "_read_trace", changed)
    capture = await collector.collect()
    assert not capture.valid
    assert not capture.evidence.trace


async def test_cancelled_retirement_wait_does_not_abandon_backing_read(monkeypatch):
    app = _app()
    collector = BugReportCollector(app, capture_bug_report_context(app))
    collector.presentation_timeout = 0.02
    release = threading.Event()
    monkeypatch.setattr(collector, "_read_logs", lambda: release.wait(3))
    try:
        await collector._source("logs", collector._read_logs)
        waiter = asyncio.create_task(collector.wait_retired())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert collector.pending_sources == ("logs",)
    finally:
        release.set()
        await collector.wait_retired()


async def test_reopened_collector_waits_for_previous_source_retirement(monkeypatch):
    app = _app()
    first = BugReportCollector(app, capture_bug_report_context(app))
    second = BugReportCollector(app, capture_bug_report_context(app))
    first.presentation_timeout = 0.02
    release = threading.Event()
    calls = []
    try:
        await first._source("logs", lambda: release.wait(3))
        first.cancel()
        value = await second._source("logs", lambda: calls.append(True))
        assert value is None and calls == []
    finally:
        release.set()
        await first.wait_retired()
    assert not app._bug_report_collectors


async def test_tool_toggle_uses_frozen_event_refs(monkeypatch):
    app = _app(True)
    collector = BugReportCollector(app, capture_bug_report_context(app))
    refs = ("frozen-source",)
    monkeypatch.setattr(collector, "_read_trace", lambda: (BugReportEvidence(), refs))
    capture = await collector.collect()
    received = []

    def tools(selected):
        received.append(selected)
        return (BugReportItem("tools", "safe selected tool"),)

    monkeypatch.setattr(collector, "_read_tools", tools)
    opted_in = await collector.collect_tools(capture)
    assert received == [refs]
    assert opted_in.evidence.tools[0].text == "safe selected tool"
    assert capture.evidence.tools == ()


async def test_aliasing_does_not_rewrite_short_words_or_identifier_prefixes(
    monkeypatch,
):
    app = _app(True)
    app._persistent_log_handler.handle(
        logging.LogRecord(
            "diagnostic",
            logging.ERROR,
            __file__,
            1,
            "a failure for run-abcdefgh123456789 and run-abcdefgh123456789-extra",
            (),
            None,
        )
    )
    collector = BugReportCollector(app, capture_bug_report_context(app))

    def trace():
        collector._alias("a")
        collector._alias("run-abcdefgh123456789", "run")
        return BugReportEvidence(), ()

    monkeypatch.setattr(collector, "_read_trace", trace)
    capture = await collector.collect()
    text = capture.evidence.logs[0].text
    assert "a failure for run-0002" in text
    assert "run-abcdefgh123456789-extra" in text


async def test_default_capture_reads_existing_file_database_metadata_only(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(
        tmp_path / "report-source.sqlite", "bug-report-existing-owner"
    )
    try:
        cid = database.add_conversation({"title": "PRIVATE_TITLE"})
        mid = database.add_message(
            {
                "conversation_id": cid,
                "sender": "assistant",
                "content": "PRIVATE_MESSAGE",
            }
        )
        with database.transaction() as cursor:
            cursor.execute(
                "INSERT INTO message_trajectory_metadata(message_id, conversation_id, turn_id, seq, event_kind, payload_json) VALUES (?, ?, ?, 0, 'tool_result', ?)",
                (mid, cid, mid, '{"result":"PRIVATE_PAYLOAD"}'),
            )
        app = _app(True)
        store = app.console_runtime.chat_store
        store.sessions()[0].persisted_conversation_id = cid
        store.persistence.db = database

        def forbidden(*_args, **_kwargs):
            raise AssertionError("unbounded reader or storage bootstrap called")

        monkeypatch.setattr(database, "get_messages_for_conversation", forbidden)
        monkeypatch.setattr(database, "get_trajectory_rows", forbidden)
        monkeypatch.setattr(database, "_initialize_schema", forbidden)
        collector = BugReportCollector(app, capture_bug_report_context(app))
        capture = await collector.collect()
        assert capture.valid and capture.evidence.trace
        assert "tool_result" in capture.evidence.trace[0].text
        assert "PRIVATE_MESSAGE" not in str(capture.evidence.trace)
        assert "PRIVATE_PAYLOAD" not in str(capture.evidence.trace)
        assert not capture.evidence.tools
        await collector.wait_retired()
    finally:
        database.close_connection()


async def test_populated_log_alias_projection_does_not_run_on_ui_loop(monkeypatch):
    import tldw_chatbook.UI.bug_report_capture as capture_module

    app = _app()
    collector = BugReportCollector(app, capture_bug_report_context(app))
    identifiers = tuple(f"00000000-0000-4000-8000-{index:012d}" for index in range(750))
    evidence = BugReportEvidence(
        logs=tuple(
            BugReportItem(
                "logs", f"failure {identifiers[index]} " + "diagnostic " * 100
            )
            for index in range(500)
        )
    )
    monkeypatch.setattr(collector, "_read_logs", lambda: evidence)

    def trace():
        for identifier in identifiers:
            collector._alias(identifier)
        return BugReportEvidence(), ()

    monkeypatch.setattr(collector, "_read_trace", trace)
    loop_thread = threading.get_ident()
    observed = set()
    original_replace = capture_module.replace

    def checked_replace(value, **changes):
        if isinstance(value, BugReportItem) and "text" in changes:
            observed.add(threading.get_ident())
            assert (
                threading.get_ident() != loop_thread
            ), "log projection blocks the UI loop"
        return original_replace(value, **changes)

    monkeypatch.setattr(capture_module, "replace", checked_replace)
    captured = await collector.collect()
    assert len(captured.evidence.logs) == 500 and observed
    assert "event-0001" in captured.evidence.logs[0].text
    assert identifiers[0] not in captured.evidence.logs[0].text
