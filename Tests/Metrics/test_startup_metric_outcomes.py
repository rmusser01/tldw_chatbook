from __future__ import annotations

import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_chatbook.Metrics.Otel_Metrics as otel_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_otel_import_is_silent_when_optional_dependency_is_absent() -> None:
    script = """
import sys
sys.modules["opentelemetry"] = None
import tldw_chatbook.Metrics.Otel_Metrics
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0
    assert "OpenTelemetry" not in result.stdout
    assert "OpenTelemetry" not in result.stderr


@pytest.fixture(autouse=True)
def reset_otel_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otel_metrics, "_initialization_result", None, raising=False)
    monkeypatch.setattr(otel_metrics, "_meter", None)


def test_otel_unavailable_initializes_once_across_threads(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", False)

    with caplog.at_level(logging.INFO):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(lambda _: otel_metrics.init_metrics(), range(16))
            )

    assert results == [False] * 16
    messages = [record.getMessage() for record in caplog.records]
    unavailable = [
        message
        for message in messages
        if "OpenTelemetry metrics are unavailable" in message
    ]
    assert len(unavailable) == 1
    assert "tldw_chatbook[debugging]" in unavailable[0]
    assert not any("initialized" in message.lower() for message in messages)


def test_otel_available_initializes_once_across_repeated_calls(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = {
        "reader": 0,
        "provider": 0,
        "set_provider": 0,
        "get_meter": 0,
        "instrument": 0,
    }
    resource_attributes: dict[str, str] = {}

    def fake_resource(*, attributes: dict[str, str]) -> object:
        resource_attributes.update(attributes)
        return object()

    def fake_reader() -> object:
        calls["reader"] += 1
        return object()

    def fake_provider(*, resource: object, metric_readers: list[object]) -> object:
        assert resource is not None
        assert len(metric_readers) == 1
        calls["provider"] += 1
        return object()

    def fake_set_provider(provider: object) -> None:
        assert provider is not None
        calls["set_provider"] += 1

    def fake_get_meter(name: str) -> object:
        assert name == "app.metrics.library"
        calls["get_meter"] += 1
        return object()

    class FakeSystemMetricsInstrumentor:
        def instrument(self) -> None:
            calls["instrument"] += 1

    monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(otel_metrics, "SERVICE_NAME", "service.name", raising=False)
    monkeypatch.setattr(
        otel_metrics, "SERVICE_VERSION", "service.version", raising=False
    )
    monkeypatch.setattr(otel_metrics, "Resource", fake_resource, raising=False)
    monkeypatch.setattr(
        otel_metrics, "PrometheusMetricReader", fake_reader, raising=False
    )
    monkeypatch.setattr(otel_metrics, "MeterProvider", fake_provider, raising=False)
    monkeypatch.setattr(
        otel_metrics,
        "SystemMetricsInstrumentor",
        FakeSystemMetricsInstrumentor,
        raising=False,
    )
    monkeypatch.setattr(
        otel_metrics,
        "metrics",
        SimpleNamespace(
            set_meter_provider=fake_set_provider,
            get_meter=fake_get_meter,
        ),
        raising=False,
    )
    monkeypatch.setenv("OTEL_SERVICE_NAME", "SERVICE-NAME-SENTINEL")

    with caplog.at_level(logging.INFO):
        assert otel_metrics.init_metrics() is True
        assert otel_metrics.init_metrics() is True

    assert calls == {
        "reader": 1,
        "provider": 1,
        "set_provider": 1,
        "get_meter": 1,
        "instrument": 1,
    }
    assert resource_attributes["service.name"] == "SERVICE-NAME-SENTINEL"
    messages = [record.getMessage() for record in caplog.records]
    assert messages.count("OpenTelemetry metrics initialized.") == 1
    assert "SERVICE-NAME-SENTINEL" not in "\n".join(messages)


def test_otel_setup_failure_propagates_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_resource(*, attributes: dict[str, str]) -> object:
        assert attributes
        raise RuntimeError("SETUP-FAILURE-SENTINEL")

    monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(otel_metrics, "SERVICE_NAME", "service.name", raising=False)
    monkeypatch.setattr(
        otel_metrics, "SERVICE_VERSION", "service.version", raising=False
    )
    monkeypatch.setattr(otel_metrics, "Resource", fail_resource, raising=False)

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError, match="SETUP-FAILURE-SENTINEL"):
            otel_metrics.init_metrics()

    assert otel_metrics._initialization_result is None
    assert "OpenTelemetry metrics initialized." not in [
        record.getMessage() for record in caplog.records
    ]

    monkeypatch.setattr(
        otel_metrics,
        "Resource",
        lambda *, attributes: object(),
        raising=False,
    )
    monkeypatch.setattr(
        otel_metrics, "PrometheusMetricReader", lambda: object(), raising=False
    )
    monkeypatch.setattr(
        otel_metrics,
        "MeterProvider",
        lambda *, resource, metric_readers: object(),
        raising=False,
    )
    monkeypatch.setattr(
        otel_metrics,
        "SystemMetricsInstrumentor",
        lambda: SimpleNamespace(instrument=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(
        otel_metrics,
        "metrics",
        SimpleNamespace(
            set_meter_provider=lambda provider: None,
            get_meter=lambda name: object(),
        ),
        raising=False,
    )

    with caplog.at_level(logging.INFO):
        assert otel_metrics.init_metrics() is True
    assert otel_metrics._initialization_result is True
