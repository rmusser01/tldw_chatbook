"""Stall persistence guardrails for UIResponsivenessMonitor (TASK-18908).

The 2026-08 lag reports arrived with no evidence. These tests pin the
contract that a heartbeat breach persists exactly one diagnostics record
carrying the lag and the timer/worker context, re-arms after recovery,
and can never itself crash the loop it measures.
"""

from __future__ import annotations

from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor


def test_stall_breach_persists_one_record(monkeypatch) -> None:
    monitor = UIResponsivenessMonitor(stall_threshold_ms=250)
    persisted: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.Utils.persistent_diagnostics.persist_event",
        lambda component, event, **fields: persisted.append(
            {"component": component, "event": event, **fields}
        ),
    )

    monitor.record_timer_created("footer-token-periodic")
    monitor.record_worker_started("console-transcript-sync")
    monitor.record_heartbeat_delta(0.9)  # 900 ms drift -> breach

    assert len(persisted) == 1
    record = persisted[0]
    assert record["component"] == "ui"
    assert record["event"] == "event_loop_stall"
    assert record["lag_ms"] == 900
    assert record["threshold_ms"] == 250
    assert record["active_timers"] == "footer-token-periodic"
    assert record["active_workers"] == "console-transcript-sync"

    # Level-triggered silence: a still-wedged loop writes ONE record.
    monitor.record_heartbeat_delta(1.5)
    assert len(persisted) == 1

    assert monitor.snapshot().stalled is True
    assert monitor.snapshot().max_heartbeat_lag_ms == 1500


def test_recovery_re_arms_persistence(monkeypatch) -> None:
    monitor = UIResponsivenessMonitor(stall_threshold_ms=250)
    persisted: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.Utils.persistent_diagnostics.persist_event",
        lambda component, event, **fields: persisted.append({"event": event}),
    )

    monitor.record_heartbeat_delta(0.5)
    assert len(persisted) == 1
    monitor.reset_heartbeat_baseline()
    monitor.record_heartbeat_delta(0.6)  # a NEW incident after recovery
    assert len(persisted) == 2


def test_below_threshold_never_persists(monkeypatch) -> None:
    monitor = UIResponsivenessMonitor(stall_threshold_ms=250)
    persisted: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.Utils.persistent_diagnostics.persist_event",
        lambda component, event, **fields: persisted.append({"event": event}),
    )
    monitor.record_heartbeat_delta(0.05)
    monitor.record_heartbeat_delta(0.2)
    assert persisted == []
    assert monitor.snapshot().max_heartbeat_lag_ms == 200


def test_persist_failure_never_raises(monkeypatch) -> None:
    monitor = UIResponsivenessMonitor(stall_threshold_ms=250)

    def exploding_persist(component, event, **fields):
        raise RuntimeError("sink unavailable")

    monkeypatch.setattr(
        "tldw_chatbook.Utils.persistent_diagnostics.persist_event",
        exploding_persist,
    )
    monitor.record_heartbeat_delta(1.0)  # must not raise
    assert monitor.snapshot().max_heartbeat_lag_ms == 1000


def test_disabled_monitor_records_nothing(monkeypatch) -> None:
    monitor = UIResponsivenessMonitor(enabled=False)
    persisted: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.Utils.persistent_diagnostics.persist_event",
        lambda component, event, **fields: persisted.append({"event": event}),
    )
    monitor.record_heartbeat_delta(5.0)
    assert persisted == []
