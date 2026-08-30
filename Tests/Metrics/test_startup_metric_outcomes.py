from __future__ import annotations

import logging
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_chatbook.Metrics.Otel_Metrics as otel_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]


class _SetOnceMetrics:
    def __init__(self, provider: object | None = None) -> None:
        self.provider = provider
        self.set_calls: list[object] = []
        self.get_provider_calls = 0
        self.legacy_get_meter_calls = 0

    def set_meter_provider(self, provider: object) -> None:
        self.set_calls.append(provider)
        if self.provider is None:
            self.provider = provider

    def get_meter_provider(self) -> object | None:
        self.get_provider_calls += 1
        return self.provider

    def get_meter(self, name: str) -> object:
        self.legacy_get_meter_calls += 1
        return self.provider.get_meter(name)  # type: ignore[union-attr]


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
    assert result.stdout == ""
    assert result.stderr == ""


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
        "resource": 0,
        "reader": 0,
        "provider": 0,
        "provider_get_meter": 0,
        "instrument": 0,
        "uninstrument": 0,
        "provider_shutdown": 0,
        "reader_shutdown": 0,
    }
    resource_attributes: dict[str, str] = {}
    providers: list[FakeProvider] = []

    def fake_resource(*, attributes: dict[str, str]) -> object:
        calls["resource"] += 1
        resource_attributes.update(attributes)
        return object()

    class FakeReader:
        def __init__(self) -> None:
            calls["reader"] += 1

        def shutdown(self) -> None:
            calls["reader_shutdown"] += 1

    class FakeProvider:
        def __init__(self, *, resource: object, metric_readers: list[object]) -> None:
            assert resource is not None
            assert len(metric_readers) == 1
            calls["provider"] += 1
            self.metric_readers = metric_readers
            self.meter = object()
            providers.append(self)

        def get_meter(self, name: str) -> object:
            assert name == "app.metrics.library"
            calls["provider_get_meter"] += 1
            return self.meter

        def shutdown(self) -> None:
            calls["provider_shutdown"] += 1
            for reader in self.metric_readers:
                reader.shutdown()

    class FakeSystemMetricsInstrumentor:
        def instrument(self, *, meter_provider: object | None = None) -> None:
            assert meter_provider is providers[0]
            calls["instrument"] += 1

        def uninstrument(self) -> None:
            calls["uninstrument"] += 1

    metrics_api = _SetOnceMetrics()
    monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(otel_metrics, "SERVICE_NAME", "service.name", raising=False)
    monkeypatch.setattr(
        otel_metrics, "SERVICE_VERSION", "service.version", raising=False
    )
    monkeypatch.setattr(otel_metrics, "Resource", fake_resource, raising=False)
    monkeypatch.setattr(
        otel_metrics, "PrometheusMetricReader", FakeReader, raising=False
    )
    monkeypatch.setattr(otel_metrics, "MeterProvider", FakeProvider, raising=False)
    monkeypatch.setattr(
        otel_metrics,
        "SystemMetricsInstrumentor",
        FakeSystemMetricsInstrumentor,
        raising=False,
    )
    monkeypatch.setattr(
        otel_metrics,
        "metrics",
        metrics_api,
        raising=False,
    )
    monkeypatch.setenv("OTEL_SERVICE_NAME", "SERVICE-NAME-SENTINEL")
    monkeypatch.delenv("OTEL_SERVICE_VERSION", raising=False)

    with caplog.at_level(logging.INFO):
        assert otel_metrics.init_metrics() is True
        assert otel_metrics.init_metrics() is True

    assert calls == {
        "resource": 1,
        "reader": 1,
        "provider": 1,
        "provider_get_meter": 1,
        "instrument": 1,
        "uninstrument": 0,
        "provider_shutdown": 0,
        "reader_shutdown": 0,
    }
    assert metrics_api.set_calls == [providers[0]]
    assert metrics_api.get_provider_calls == 1
    assert metrics_api.legacy_get_meter_calls == 0
    assert metrics_api.provider is providers[0]
    assert otel_metrics._meter is providers[0].meter
    assert resource_attributes == {
        "service.name": "SERVICE-NAME-SENTINEL",
        "service.version": "0.1.0",
    }
    messages = [record.getMessage() for record in caplog.records]
    assert messages.count("OpenTelemetry metrics initialized.") == 1
    assert "SERVICE-NAME-SENTINEL" not in "\n".join(messages)


def test_otel_setup_failure_propagates_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = {
        "resource": 0,
        "reader": 0,
        "provider": 0,
        "provider_get_meter": 0,
        "instrument": 0,
        "uninstrument": 0,
        "provider_shutdown": 0,
        "reader_shutdown": 0,
    }
    providers: list[FakeProvider] = []
    instrumented_providers: list[object | None] = []
    fail_instrumentation = True

    def fake_resource(*, attributes: dict[str, str]) -> object:
        assert attributes
        calls["resource"] += 1
        return object()

    class FakeReader:
        def __init__(self) -> None:
            calls["reader"] += 1

        def shutdown(self) -> None:
            calls["reader_shutdown"] += 1

    class FakeProvider:
        def __init__(self, *, resource: object, metric_readers: list[object]) -> None:
            assert resource is not None
            assert len(metric_readers) == 1
            calls["provider"] += 1
            self.metric_readers = metric_readers
            self.meter = object()
            providers.append(self)

        def get_meter(self, name: str) -> object:
            assert name == "app.metrics.library"
            calls["provider_get_meter"] += 1
            return self.meter

        def shutdown(self) -> None:
            calls["provider_shutdown"] += 1
            for reader in self.metric_readers:
                reader.shutdown()
            if self is providers[0]:
                raise RuntimeError("CLEANUP-FAILURE-SENTINEL")

    class FakeSystemMetricsInstrumentor:
        def instrument(self, *, meter_provider: object | None = None) -> None:
            calls["instrument"] += 1
            instrumented_providers.append(meter_provider)
            if fail_instrumentation:
                raise RuntimeError("SETUP-FAILURE-SENTINEL")

        def uninstrument(self) -> None:
            calls["uninstrument"] += 1

    metrics_api = _SetOnceMetrics()
    monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(otel_metrics, "SERVICE_NAME", "service.name", raising=False)
    monkeypatch.setattr(
        otel_metrics, "SERVICE_VERSION", "service.version", raising=False
    )
    monkeypatch.setattr(otel_metrics, "Resource", fake_resource, raising=False)
    monkeypatch.setattr(
        otel_metrics, "PrometheusMetricReader", FakeReader, raising=False
    )
    monkeypatch.setattr(otel_metrics, "MeterProvider", FakeProvider, raising=False)
    monkeypatch.setattr(
        otel_metrics,
        "SystemMetricsInstrumentor",
        FakeSystemMetricsInstrumentor,
        raising=False,
    )
    monkeypatch.setattr(
        otel_metrics,
        "metrics",
        metrics_api,
        raising=False,
    )

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError, match="SETUP-FAILURE-SENTINEL"):
            otel_metrics.init_metrics()

    assert calls == {
        "resource": 1,
        "reader": 1,
        "provider": 1,
        "provider_get_meter": 1,
        "instrument": 1,
        "uninstrument": 1,
        "provider_shutdown": 1,
        "reader_shutdown": 1,
    }
    assert metrics_api.provider is None
    assert metrics_api.set_calls == []
    assert metrics_api.get_provider_calls == 0
    assert metrics_api.legacy_get_meter_calls == 0
    assert instrumented_providers == [providers[0]]
    assert otel_metrics._initialization_result is None
    assert otel_metrics._meter is None
    assert "OpenTelemetry metrics initialized." not in [
        record.getMessage() for record in caplog.records
    ]

    fail_instrumentation = False

    with caplog.at_level(logging.INFO):
        assert otel_metrics.init_metrics() is True
    assert calls == {
        "resource": 2,
        "reader": 2,
        "provider": 2,
        "provider_get_meter": 2,
        "instrument": 2,
        "uninstrument": 1,
        "provider_shutdown": 1,
        "reader_shutdown": 1,
    }
    assert providers[0] is not providers[1]
    assert metrics_api.provider is providers[1]
    assert metrics_api.set_calls == [providers[1]]
    assert metrics_api.get_provider_calls == 1
    assert metrics_api.legacy_get_meter_calls == 0
    assert instrumented_providers == providers
    assert otel_metrics._initialization_result is True
    assert otel_metrics._meter is providers[1].meter
    assert [record.getMessage() for record in caplog.records].count(
        "OpenTelemetry metrics initialized."
    ) == 1


def test_otel_provider_construction_failure_cleans_up_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader_shutdowns = 0
    metrics_api = _SetOnceMetrics()

    class FakeReader:
        def shutdown(self) -> None:
            nonlocal reader_shutdowns
            reader_shutdowns += 1
            raise RuntimeError("CLEANUP-FAILURE-SENTINEL")

    class FailingProvider:
        def __init__(self, *, resource: object, metric_readers: list[object]) -> None:
            assert resource is not None
            assert len(metric_readers) == 1
            raise RuntimeError("PROVIDER-FAILURE-SENTINEL")

    monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(otel_metrics, "SERVICE_NAME", "service.name", raising=False)
    monkeypatch.setattr(
        otel_metrics, "SERVICE_VERSION", "service.version", raising=False
    )
    monkeypatch.setattr(
        otel_metrics,
        "Resource",
        lambda *, attributes: object(),
        raising=False,
    )
    monkeypatch.setattr(
        otel_metrics, "PrometheusMetricReader", FakeReader, raising=False
    )
    monkeypatch.setattr(otel_metrics, "MeterProvider", FailingProvider, raising=False)
    monkeypatch.setattr(
        otel_metrics,
        "SystemMetricsInstrumentor",
        lambda: SimpleNamespace(instrument=lambda **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(otel_metrics, "metrics", metrics_api, raising=False)

    with pytest.raises(RuntimeError, match="PROVIDER-FAILURE-SENTINEL"):
        otel_metrics.init_metrics()

    assert reader_shutdowns == 1
    assert metrics_api.set_calls == []
    assert otel_metrics._initialization_result is None
    assert otel_metrics._meter is None


def test_otel_preexisting_provider_fails_closed_and_cleans_up_local_setup(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = {
        "provider_get_meter": 0,
        "instrument": 0,
        "uninstrument": 0,
        "provider_shutdown": 0,
        "reader_shutdown": 0,
    }
    existing_meter = object()

    class ExistingProvider:
        def get_meter(self, name: str) -> object:
            assert name == "app.metrics.library"
            return existing_meter

    existing_provider = ExistingProvider()
    metrics_api = _SetOnceMetrics(existing_provider)
    providers: list[FakeProvider] = []
    instrumented_providers: list[object | None] = []

    class FakeReader:
        def shutdown(self) -> None:
            calls["reader_shutdown"] += 1

    class FakeProvider:
        def __init__(self, *, resource: object, metric_readers: list[object]) -> None:
            assert resource is not None
            self.metric_readers = metric_readers
            self.meter = object()
            providers.append(self)

        def get_meter(self, name: str) -> object:
            assert name == "app.metrics.library"
            calls["provider_get_meter"] += 1
            return self.meter

        def shutdown(self) -> None:
            calls["provider_shutdown"] += 1
            for reader in self.metric_readers:
                reader.shutdown()

    class FakeSystemMetricsInstrumentor:
        def instrument(self, *, meter_provider: object | None = None) -> None:
            calls["instrument"] += 1
            instrumented_providers.append(meter_provider)

        def uninstrument(self) -> None:
            calls["uninstrument"] += 1

    monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(otel_metrics, "SERVICE_NAME", "service.name", raising=False)
    monkeypatch.setattr(
        otel_metrics, "SERVICE_VERSION", "service.version", raising=False
    )
    monkeypatch.setattr(
        otel_metrics,
        "Resource",
        lambda *, attributes: object(),
        raising=False,
    )
    monkeypatch.setattr(
        otel_metrics, "PrometheusMetricReader", FakeReader, raising=False
    )
    monkeypatch.setattr(otel_metrics, "MeterProvider", FakeProvider, raising=False)
    monkeypatch.setattr(
        otel_metrics,
        "SystemMetricsInstrumentor",
        FakeSystemMetricsInstrumentor,
        raising=False,
    )
    monkeypatch.setattr(otel_metrics, "metrics", metrics_api, raising=False)

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError, match="ownership"):
            otel_metrics.init_metrics()

    assert calls == {
        "provider_get_meter": 1,
        "instrument": 1,
        "uninstrument": 1,
        "provider_shutdown": 1,
        "reader_shutdown": 1,
    }
    assert metrics_api.provider is existing_provider
    assert metrics_api.set_calls == [providers[0]]
    assert metrics_api.get_provider_calls == 1
    assert metrics_api.legacy_get_meter_calls == 0
    assert instrumented_providers == [providers[0]]
    assert otel_metrics._initialization_result is None
    assert otel_metrics._meter is None
    assert "OpenTelemetry metrics initialized." not in [
        record.getMessage() for record in caplog.records
    ]


def test_otel_concurrent_start_publishes_only_complete_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {
        "reader": 0,
        "provider": 0,
        "provider_get_meter": 0,
        "instrument": 0,
    }
    metrics_api = _SetOnceMetrics()
    providers: list[FakeProvider] = []
    instrumented_providers: list[object | None] = []
    workers_ready = threading.Barrier(8)
    instrumentation_started = threading.Event()
    release_instrumentation = threading.Event()
    get_meter_waiting = threading.Event()

    class FakeReader:
        def __init__(self) -> None:
            calls["reader"] += 1

        def shutdown(self) -> None:
            raise AssertionError("successful setup must not shut down its reader")

    class FakeProvider:
        def __init__(self, *, resource: object, metric_readers: list[object]) -> None:
            assert resource is not None
            assert len(metric_readers) == 1
            calls["provider"] += 1
            self.meter = object()
            providers.append(self)

        def get_meter(self, name: str) -> object:
            assert name == "app.metrics.library"
            calls["provider_get_meter"] += 1
            return self.meter

        def shutdown(self) -> None:
            raise AssertionError("successful setup must not shut down its provider")

    class FakeSystemMetricsInstrumentor:
        def instrument(self, *, meter_provider: object | None = None) -> None:
            calls["instrument"] += 1
            instrumented_providers.append(meter_provider)
            instrumentation_started.set()
            assert release_instrumentation.wait(timeout=5)

        def uninstrument(self) -> None:
            raise AssertionError("successful setup must not uninstrument")

    def call_init_metrics() -> bool:
        workers_ready.wait(timeout=5)
        return otel_metrics.init_metrics()

    def capture_waiting_warning(*_args: object, **_kwargs: object) -> None:
        get_meter_waiting.set()

    monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(otel_metrics, "SERVICE_NAME", "service.name", raising=False)
    monkeypatch.setattr(
        otel_metrics, "SERVICE_VERSION", "service.version", raising=False
    )
    monkeypatch.setattr(
        otel_metrics,
        "Resource",
        lambda *, attributes: object(),
        raising=False,
    )
    monkeypatch.setattr(
        otel_metrics, "PrometheusMetricReader", FakeReader, raising=False
    )
    monkeypatch.setattr(otel_metrics, "MeterProvider", FakeProvider, raising=False)
    monkeypatch.setattr(
        otel_metrics,
        "SystemMetricsInstrumentor",
        FakeSystemMetricsInstrumentor,
        raising=False,
    )
    monkeypatch.setattr(otel_metrics, "metrics", metrics_api, raising=False)
    monkeypatch.setattr(
        otel_metrics,
        "logging",
        SimpleNamespace(info=logging.info, warning=capture_waiting_warning),
    )

    with ThreadPoolExecutor(max_workers=9) as executor:
        init_futures = [executor.submit(call_init_metrics) for _ in range(8)]
        assert instrumentation_started.wait(timeout=5)
        try:
            assert otel_metrics._meter is None
            assert otel_metrics._initialization_result is None
            assert metrics_api.provider is None
            assert metrics_api.set_calls == []

            meter_future = executor.submit(otel_metrics._get_meter)
            assert get_meter_waiting.wait(timeout=5)
            assert not meter_future.done()
        finally:
            release_instrumentation.set()

        assert [future.result(timeout=5) for future in init_futures] == [True] * 8
        assert meter_future.result(timeout=5) is providers[0].meter

    assert calls == {
        "reader": 1,
        "provider": 1,
        "provider_get_meter": 1,
        "instrument": 1,
    }
    assert metrics_api.provider is providers[0]
    assert metrics_api.set_calls == [providers[0]]
    assert metrics_api.get_provider_calls == 1
    assert metrics_api.legacy_get_meter_calls == 0
    assert instrumented_providers == [providers[0]]
    assert otel_metrics._meter is providers[0].meter
    assert otel_metrics._initialization_result is True
