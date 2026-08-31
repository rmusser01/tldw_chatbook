"""The Prometheus listener must not bind unless the user asked for it.

TASK-25914. Before this, ``init_metrics_server`` bound a socket whenever
``prometheus_client`` was importable, so installing the ``dev`` or ``debugging``
extra silently opened an unauthenticated HTTP listener. Dependency presence is
not consent.

These tests force ``PROMETHEUS_AVAILABLE`` on. Without that the gate under test
is masked -- the function would return ``False`` because the optional dependency
is absent, not because the gate works, and every assertion here would pass for
the wrong reason.
"""

from __future__ import annotations

import logging

import pytest

import tldw_chatbook.Metrics.metrics as prometheus_metrics


class _RecordingBinder:
    """Stands in for ``prometheus_client.start_http_server``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, port, addr=None, **kwargs):
        self.calls.append({"port": port, "addr": addr, **kwargs})


@pytest.fixture
def binder(monkeypatch: pytest.MonkeyPatch) -> _RecordingBinder:
    recorder = _RecordingBinder()
    monkeypatch.setattr(prometheus_metrics, "PROMETHEUS_AVAILABLE", True)
    monkeypatch.setattr(
        prometheus_metrics, "start_http_server", recorder, raising=False
    )
    return recorder


def _with_config(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    resolved = {"enabled": False, "port": 8000, "bind_address": "127.0.0.1"}
    resolved.update(overrides)
    monkeypatch.setattr(
        prometheus_metrics, "_metrics_server_config", lambda: dict(resolved)
    )


def test_listener_does_not_bind_when_disabled(monkeypatch, binder):
    _with_config(monkeypatch, enabled=False)

    started = prometheus_metrics.init_metrics_server()

    assert started is False
    assert binder.calls == [], "no socket may be bound while metrics are disabled"


def test_listener_binds_when_explicitly_enabled(monkeypatch, binder):
    _with_config(monkeypatch, enabled=True, port=9123)

    started = prometheus_metrics.init_metrics_server()

    assert started is True
    assert len(binder.calls) == 1
    assert binder.calls[0]["port"] == 9123


def test_listener_binds_to_configured_address(monkeypatch, binder):
    _with_config(monkeypatch, enabled=True, bind_address="127.0.0.1")

    prometheus_metrics.init_metrics_server()

    assert binder.calls[0]["addr"] == "127.0.0.1"


def test_metrics_disabled_by_default(monkeypatch):
    """The shipped default must be off, not merely settable to off."""
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: default,
    )

    assert prometheus_metrics._metrics_server_config()["enabled"] is False


def test_bind_address_defaults_to_loopback(monkeypatch):
    """prometheus_client defaults to 0.0.0.0; we must not inherit that."""
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: default,
    )

    assert prometheus_metrics._metrics_server_config()["bind_address"] == "127.0.0.1"


def test_started_listener_reports_address_and_port(monkeypatch, binder, caplog):
    _with_config(monkeypatch, enabled=True, port=9123, bind_address="127.0.0.1")

    with caplog.at_level(logging.INFO):
        prometheus_metrics.init_metrics_server()

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "9123" in logged
    assert "127.0.0.1" in logged


def test_collection_still_runs_while_listener_disabled(monkeypatch, binder):
    """Disabling the listener must not disable metric collection itself.

    Scope note: with ``prometheus_client`` absent from the test environment the
    metric classes are no-op stand-ins, so this asserts the call path stays
    reachable and raises nothing -- not that a value lands in a registry. That
    stronger claim was verified out-of-band against the real dependency and is
    recorded in TASK-25914's implementation notes (counter 3.0, histogram count
    1.0, listener off).
    """
    _with_config(monkeypatch, enabled=False)
    assert prometheus_metrics.init_metrics_server() is False

    prometheus_metrics.log_counter("task_25914_probe", value=1)
    prometheus_metrics.log_histogram("task_25914_probe_seconds", value=0.25)


def test_metrics_port_env_var_overrides_configured_port(monkeypatch):
    """METRICS_PORT kept working as a port override (it predates this task).

    It deliberately does NOT enable the listener: AC#1 requires an explicit
    config setting for that. The unimplemented 2026-08-12 launch-diagnostics
    plan proposed env-alone-opts-in; this task's criteria say otherwise, so the
    env var moves the port and nothing else.
    """
    monkeypatch.setenv("METRICS_PORT", "9555")
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: default,
    )

    resolved = prometheus_metrics._metrics_server_config()

    assert resolved["port"] == 9555
    assert resolved["enabled"] is False, "env var must not enable the listener"
