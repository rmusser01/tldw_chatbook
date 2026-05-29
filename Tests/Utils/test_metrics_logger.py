"""Focused coverage for normal metric log-volume controls."""

from __future__ import annotations

from loguru import logger

from tldw_chatbook.Metrics import metrics_logger


def _capture_metric_output(action) -> list[str]:
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="METRIC")
    try:
        action()
    finally:
        logger.remove(sink_id)
    return messages


def test_metric_logging_disabled_by_default(monkeypatch) -> None:
    """Normal runs should not emit high-volume METRIC log lines."""

    monkeypatch.delenv("TLDW_METRICS_LOGGING", raising=False)

    messages = _capture_metric_output(
        lambda: metrics_logger.log_counter("unit_metric_disabled")
    )

    assert messages == []


def test_metric_logging_enabled_by_env(monkeypatch) -> None:
    """Explicit metric logging opt-in should preserve existing METRIC output."""

    monkeypatch.setenv("TLDW_METRICS_LOGGING", "1")

    messages = _capture_metric_output(
        lambda: metrics_logger.log_counter("unit_metric_enabled")
    )

    assert any("unit_metric_enabled" in message for message in messages)
