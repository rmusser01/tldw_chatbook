"""TASK-31977: support evidence must survive the real disk and share sinks."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_durable_commit_diagnostics import (
    SECRET_IDENTIFIER,
    _controller_with_failing_commit,
)
from Tests.test_logs_share_path_privacy import _Collector
from tldw_chatbook.Logging_Config import PrivateRotatingFileHandler
from tldw_chatbook.UI.Logs_Window import LogsWindow
from tldw_chatbook.Utils.persistent_diagnostics import PersistentDiagnosticFilter
from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor


@pytest.fixture
def sinks(tmp_path):
    path = tmp_path / "diagnostic.log"
    handler = PrivateRotatingFileHandler(path, maxBytes=2_000_000, backupCount=1)
    handler.addFilter(PersistentDiagnosticFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    with _Collector() as app:
        root.addHandler(handler)
        copied = []
        window = SimpleNamespace(
            app_instance=app,
            app=SimpleNamespace(
                copy_to_clipboard=copied.append, notify=lambda *a, **k: None
            ),
        )
        try:
            yield SimpleNamespace(path=path, window=window, copied=copied)
        finally:
            root.removeHandler(handler)
            handler.close()


def assert_export(sinks, *tokens):
    LogsWindow._on_copy_all(sinks.window)
    for text in (sinks.path.read_text(), sinks.copied[-1]):
        for token in tokens:
            assert token in text
        assert SECRET_IDENTIFIER not in text
        assert "PRIVATE-DRAFT-31977" not in text
    return sinks.path.read_text()


async def test_pre_trace_commit_failure_reaches_file_and_copy_all(sinks):
    controller, _store = _controller_with_failing_commit()
    result = await controller.submit_draft("PRIVATE-DRAFT-31977")
    assert not result.accepted
    text = assert_export(
        sinks,
        "event=console_send_stage",
        "phase=controller_submit",
        "phase=durable_commit",
        "status=failed",
        "error_category=validation",
        "exception_type=ValueError",
        "app_version=",
        "python_version=",
        "capture_enabled=false",
    )
    assert "phase=provider_entry" not in text


async def test_responsive_refresh_churn_is_exported_once_per_episode(sinks):
    monitor = UIResponsivenessMonitor()
    try:
        for _ in range(100):
            monitor.record_worker_started("console-sync")
            monitor.record_worker_finished("console-sync")
        monitor.record_heartbeat_delta(0.01)
        for _ in range(100):
            monitor.record_worker_started("console-sync")
            monitor.record_worker_finished("console-sync")
        monitor.record_heartbeat_delta(0.01)
        for _ in range(100):
            if "event=ui_refresh_churn" in sinks.path.read_text():
                break
            await asyncio.sleep(0.01)
        assert monitor.snapshot().stalled is False
        text = assert_export(sinks, "event=ui_refresh_churn", "operation=console_sync")
        assert text.count("event=ui_refresh_churn") == 1
    finally:
        close = getattr(monitor, "close", None)
        if close is not None:
            await asyncio.to_thread(close)


async def test_failure_is_retained_when_file_threshold_is_warning(sinks):
    for handler in logging.getLogger().handlers:
        if isinstance(
            handler, PrivateRotatingFileHandler
        ) and handler.baseFilename == str(sinks.path):
            handler.setLevel(logging.WARNING)
    controller, _store = _controller_with_failing_commit()
    await controller.submit_draft("PRIVATE-DRAFT-31977")
    assert "phase=durable_commit" in sinks.path.read_text()
    assert "status=failed" in sinks.path.read_text()


async def test_diagnostic_sink_failure_does_not_change_send_result(monkeypatch):
    from tldw_chatbook.Utils import persistent_diagnostics

    def failed_sink(*args, **kwargs):
        raise OSError("PRIVATE-DRAFT-31977")

    monkeypatch.setattr(persistent_diagnostics, "persist_event", failed_sink)
    controller, _store = _controller_with_failing_commit()
    result = await controller.submit_draft("PRIVATE-DRAFT-31977")
    assert not result.accepted
    assert result.visible_copy == "Couldn't save the prepared turn. Retry or cancel."


async def test_monitor_queue_is_bounded_and_cannot_block_ui(monkeypatch):
    import threading

    from tldw_chatbook.Utils import persistent_diagnostics

    entered = threading.Event()
    release = threading.Event()
    threads = []

    def slow_sink(*args, **kwargs):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(persistent_diagnostics, "persist_event", slow_sink)
    monitor = UIResponsivenessMonitor()
    try:
        monitor.record_diagnostic("console", "console_send_stage", phase="ui_submit")
        assert await asyncio.to_thread(entered.wait, 1)
        for _ in range(1000):
            monitor.record_diagnostic(
                "console", "console_send_stage", phase="ui_submit"
            )
        assert monitor._stall_queue.qsize() <= monitor._STALL_QUEUE_DEPTH
        assert monitor._diagnostic_dropped > 0
        assert threading.get_ident() not in threads
    finally:
        release.set()
        await asyncio.to_thread(monitor.close)


async def test_churn_rearms_only_after_quiet_window_and_ignores_normal_polling(sinks):
    monitor = UIResponsivenessMonitor()
    try:
        for _ in range(5):
            monitor.record_worker_started("console-sync")
        monitor.record_heartbeat_delta(0)
        assert "ui_refresh_churn" not in sinks.path.read_text()
        for _ in range(10):
            monitor.record_refresh("screen_recompose")
        monitor.record_heartbeat_delta(0)
        monitor.record_heartbeat_delta(0)
        for _ in range(10):
            monitor.record_refresh("screen_recompose")
        monitor.record_heartbeat_delta(0)
        await asyncio.to_thread(monitor.close)
        text = assert_export(sinks, "operation=screen_recompose")
        assert text.count("event=ui_refresh_churn") == 2
    finally:
        await asyncio.to_thread(monitor.close)


async def test_chained_sqlite_failure_retains_code_without_query_text(sinks):
    import sqlite3

    from tldw_chatbook.Chat.console_send_diagnostics import (
        record_send_stage,
        send_diagnostic_scope,
    )

    connection = sqlite3.connect(":memory:")
    try:
        async with send_diagnostic_scope("controller_submit"):
            try:
                try:
                    connection.execute(
                        'SELECT "PRIVATE-DRAFT-31977" FROM private_table'
                    )
                except sqlite3.OperationalError:
                    raise RuntimeError("PRIVATE-DRAFT-31977") from None
            except RuntimeError as error:
                record_send_stage("trace_reservation", "failed", error=error)
        assert_export(
            sinks,
            "error_category=database",
            "sqlite_code=1",
            "exception_type=OperationalError",
        )
    finally:
        connection.close()


async def test_sequential_queued_submissions_get_independent_diagnostic_budgets(sinks):
    import re

    from tldw_chatbook.Chat.console_send_diagnostics import (
        record_send_stage,
        send_diagnostic_scope,
    )

    async with send_diagnostic_scope("ui_submit"):
        for index in range(8):
            async with send_diagnostic_scope("controller_submit") as diagnostic:
                for _ in range(10):
                    record_send_stage("provider_entry")
                if index == 7:
                    record_send_stage(
                        "trace_reservation",
                        "failed",
                        error=ValueError("PRIVATE-DRAFT-31977"),
                    )
                # Isolate per-attempt budgets from the separately tested bounded queue.
                await asyncio.to_thread(diagnostic.monitor._stall_queue.join)
    text = assert_export(sinks, "status=failed", "phase=trace_reservation")
    starts = [
        line
        for line in text.splitlines()
        if "phase=controller_submit" in line and "status=entered" in line
    ]
    assert len({re.search(r"attempt_token=(\w+)", line)[1] for line in starts}) == 8


async def test_failure_keeps_runtime_and_capture_context_at_warning_root(sinks):
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.WARNING)
    try:
        controller, _store = _controller_with_failing_commit()
        await controller.submit_draft("PRIVATE-DRAFT-31977")
        assert_export(
            sinks,
            "status=failed",
            "app_version=",
            "python_version=",
            "sqlite_version=",
            "capture_enabled=false",
        )
    finally:
        root.setLevel(previous)


async def test_validation_timeout_survives_warning_logging(monkeypatch, sinks):
    controller, _store = _controller_with_failing_commit()

    async def stalled_resolution(selection):
        await asyncio.sleep(10)

    monkeypatch.setattr(
        controller.provider_gateway, "resolve_for_send", stalled_resolution
    )
    monkeypatch.setattr(controller, "PROVIDER_VALIDATION_TIMEOUT_SECONDS", 0.01)
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.WARNING)
    try:
        result = await controller.submit_draft("PRIVATE-DRAFT-31977")
        assert not result.accepted
        assert_export(
            sinks,
            "phase=provider_resolution",
            "status=failed",
            "error_category=timeout",
        )
    finally:
        root.setLevel(previous)


async def test_known_trace_validation_reason_is_identifiable_without_raw_text(sinks):
    from tldw_chatbook.Chat.console_send_diagnostics import (
        record_send_stage,
        send_diagnostic_scope,
    )

    async with send_diagnostic_scope("controller_submit"):
        record_send_stage(
            "trace_reservation", "failed", error=ValueError("trace_owner_unavailable")
        )
    assert_export(sinks, "error_category=trace_owner_unavailable")
