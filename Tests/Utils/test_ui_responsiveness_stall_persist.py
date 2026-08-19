"""Stall persistence guardrails for UIResponsivenessMonitor (TASK-18908).

The 2026-08 lag reports arrived with no evidence. These tests pin the
contract that a heartbeat breach persists exactly one diagnostics record
carrying the lag and the timer/worker context, re-arms after recovery, and
can never itself crash the loop it measures.

Two layers:

* Contract tests patch ``persist_event`` to observe dispatch behaviour
  (edge-trigger, re-arm, failure isolation) without touching the sink.
* Integration tests (``TestStallPersistenceIntegration``) run the REAL
  schema validation and formatting path end to end: the monitor's drain
  thread emits through ``log_persistent_metadata`` into a captured stdlib
  logger, so a schema rejection -- a field the persistent sink refuses --
  fails the test instead of being silently swallowed (the gap the PR review
  caught: an earlier revision shipped fields the schema did not admit,
  hidden by fully-mocked tests).
"""

from __future__ import annotations

import logging
import time

import pytest

from tldw_chatbook.Utils import persistent_diagnostics
from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor


def _patch_persist(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture persist_event calls (dispatch-layer contract tests only)."""
    persisted: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.Utils.persistent_diagnostics.persist_event",
        lambda component, event, **fields: persisted.append(
            {"component": component, "event": event, **fields}
        ),
    )
    return persisted


class TestStallDispatchContract:
    """Dispatch behaviour with the sink patched out."""

    def test_stall_breach_persists_one_record(self, monkeypatch) -> None:
        monitor = UIResponsivenessMonitor(stall_threshold_ms=250)
        persisted = _patch_persist(monkeypatch)

        monitor.record_timer_created("footer-token-periodic")
        monitor.record_worker_started("console-transcript-sync")
        monitor.record_heartbeat_delta(0.9)  # 900 ms drift -> breach
        _drain(monitor, expected=1)

        assert len(persisted) == 1
        record = persisted[0]
        assert record["component"] == "ui"
        assert record["event"] == "event_loop_stall"
        assert record["lag_ms"] == 900
        assert record["threshold_ms"] == 250
        assert record["active_timers"] == ["footer-token-periodic"]
        assert record["active_workers"] == ["console-transcript-sync"]

        # Level-triggered silence: a still-wedged loop writes ONE record.
        monitor.record_heartbeat_delta(1.5)
        _drain(monitor, expected=1)  # no new record dispatched
        assert len(persisted) == 1

        assert monitor.snapshot().stalled is True
        assert monitor.snapshot().max_heartbeat_lag_ms == 1500

    def test_healthy_heartbeat_re_arms_persistence(self, monkeypatch) -> None:
        """Production recovery path: a BELOW-threshold heartbeat re-arms.

        No manual baseline reset -- the same sequence the app's 1s heartbeat
        timer produces in a real stall-then-recover-then-stall-again session.
        """
        monitor = UIResponsivenessMonitor(stall_threshold_ms=250)
        persisted = _patch_persist(monkeypatch)

        monitor.record_heartbeat_delta(0.5)  # stall
        _drain(monitor, expected=1)
        assert len(persisted) == 1

        monitor.record_heartbeat_delta(0.05)  # healthy heartbeat: recovery
        monitor.record_heartbeat_delta(0.6)  # a NEW incident after recovery
        _drain(monitor, expected=2)
        assert len(persisted) == 2

    def test_below_threshold_never_persists(self, monkeypatch) -> None:
        monitor = UIResponsivenessMonitor(stall_threshold_ms=250)
        persisted = _patch_persist(monkeypatch)
        monitor.record_heartbeat_delta(0.05)
        monitor.record_heartbeat_delta(0.2)
        _drain(monitor, expected=0)
        assert persisted == []
        assert monitor.snapshot().max_heartbeat_lag_ms == 200

    def test_persist_failure_never_raises(self, monkeypatch) -> None:
        monitor = UIResponsivenessMonitor(stall_threshold_ms=250)

        def exploding_persist(component, event, **fields):
            raise RuntimeError("sink unavailable")

        monkeypatch.setattr(
            "tldw_chatbook.Utils.persistent_diagnostics.persist_event",
            exploding_persist,
        )
        monitor.record_heartbeat_delta(1.0)  # must not raise
        _drain(monitor)
        assert monitor.snapshot().max_heartbeat_lag_ms == 1000

    def test_disabled_monitor_records_nothing(self, monkeypatch) -> None:
        monitor = UIResponsivenessMonitor(enabled=False)
        persisted = _patch_persist(monkeypatch)
        monitor.record_heartbeat_delta(5.0)
        _drain(monitor, expected=0)
        assert persisted == []


class TestStallPersistenceIntegration:
    """End-to-end through the REAL schema validation and record formatting."""

    @pytest.fixture()
    def captured(self, monkeypatch: pytest.MonkeyPatch):
        """Capture the stdlib record the real sink would receive.

        persist_event routes to the ``tldw_chatbook.diagnostics.ui`` logger
        (``component`` becomes the suffix); the capture handler attaches to
        THAT logger so the whole validation + formatting path runs for real.
        """
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger("tldw_chatbook.diagnostics.ui")
        # The root logger's WARNING default would drop INFO records before
        # any handler sees them; the real sink configures INFO, so the
        # capture must too.
        monkeypatch.setattr(logger, "handlers", [_Capture()])
        monkeypatch.setattr(logger, "propagate", False)
        monkeypatch.setattr(logger, "level", logging.INFO)
        return records

    def test_stall_record_passes_real_schema_validation(self, captured) -> None:
        """The emitted record must survive the persistent sink's own gate.

        Regression for the review finding: an earlier revision shipped
        lag_ms/threshold_ms/mounts/removes/active_timers/active_workers
        through a fully-mocked persist_event, and the real schema rejected
        every one of them -- the drain thread swallowed the ValueError and
        no record ever reached the log.
        """
        monitor = UIResponsivenessMonitor(stall_threshold_ms=250)
        monitor.record_timer_created("footer-token-periodic")
        monitor.record_worker_started("console-transcript-sync")
        monitor.record_mounts("console", mounted=3, removed=1)
        monitor.record_heartbeat_delta(1.2)
        _drain(monitor, expected=1)

        assert len(captured) == 1
        message = captured[0].getMessage()
        assert "event=event_loop_stall" in message
        assert "lag_ms=1200" in message
        assert "threshold_ms=250" in message
        assert "mounts=3" in message
        assert "removes=1" in message
        assert "active_timers=footer-token-periodic" in message
        assert "active_workers=console-transcript-sync" in message

    def test_schema_rejects_nothing_the_monitor_sends(self, captured) -> None:
        """Direct admission check: every monitor field is in the schema."""
        allowed = (
            persistent_diagnostics._INTEGER_FIELDS | persistent_diagnostics._LIST_FIELDS
        )
        sent = {
            "lag_ms",
            "threshold_ms",
            "mounts",
            "removes",
            "active_timers",
            "active_workers",
        }
        missing = sent - allowed
        assert not missing, f"monitor sends fields the sink rejects: {missing}"


def _drain(
    monitor: UIResponsivenessMonitor, expected: int = 1, timeout: float = 2.0
) -> None:
    """Wait until the drain thread has processed ``expected`` records.

    Synchronizes on the monitor's processed-record counter rather than a
    sleep, so the tests are deterministic and fast in the common case.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if monitor._stall_records_processed >= expected:
            return
        time.sleep(0.005)
    pytest.fail(
        f"drain thread processed {monitor._stall_records_processed} of "
        f"{expected} stall records within {timeout}s"
    )
