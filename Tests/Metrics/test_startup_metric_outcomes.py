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
import tldw_chatbook.Metrics.metrics as prometheus_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeReader:
    def __init__(self, harness: _FakeOtelHarness) -> None:
        self.harness = harness

    def shutdown(self) -> None:
        self.harness.calls["reader_shutdown"] += 1
        if self.harness.fail_reader_shutdown:
            raise RuntimeError("CLEANUP-FAILURE-SENTINEL")


class _FakeProvider:
    def __init__(
        self,
        harness: _FakeOtelHarness,
        resource: object,
        metric_readers: list[_FakeReader],
    ) -> None:
        assert resource is not None
        assert len(metric_readers) == 1
        self.harness = harness
        self.metric_readers = metric_readers
        self.meter = object()

    def get_meter(self, name: str) -> object:
        assert name == "app.metrics.library"
        self.harness.calls["provider_get_meter"] += 1
        return self.meter

    def shutdown(self) -> None:
        self.harness.calls["provider_shutdown"] += 1
        for reader in self.metric_readers:
            reader.shutdown()
        if (
            self.harness.fail_first_provider_shutdown
            and self is self.harness.providers[0]
        ):
            raise RuntimeError("CLEANUP-FAILURE-SENTINEL")


class _FakeInstrumentor:
    def __init__(self, harness: _FakeOtelHarness) -> None:
        self.harness = harness
        self.is_instrumented_by_opentelemetry = False

    def instrument(self, *, meter_provider: object | None = None) -> None:
        self.harness.calls["instrument"] += 1
        self.harness.instrumented_providers.append(meter_provider)
        if self.is_instrumented_by_opentelemetry:
            return
        if self.harness.instrumentation_started is not None:
            self.harness.instrumentation_started.set()
        if self.harness.release_instrumentation is not None:
            assert self.harness.release_instrumentation.wait(timeout=5)
        if self.harness.fail_instrumentation:
            raise RuntimeError("SETUP-FAILURE-SENTINEL")
        if not self.harness.silent_instrumentation_decline:
            self.is_instrumented_by_opentelemetry = True

    def uninstrument(self) -> None:
        self.harness.calls["uninstrument"] += 1
        if self.is_instrumented_by_opentelemetry:
            self.is_instrumented_by_opentelemetry = False


class _FakeOtelHarness:
    def __init__(self) -> None:
        self.calls = {
            "resource": 0,
            "reader": 0,
            "provider": 0,
            "provider_get_meter": 0,
            "instrument": 0,
            "uninstrument": 0,
            "provider_shutdown": 0,
            "reader_shutdown": 0,
        }
        self.resource_attributes: dict[str, str] = {}
        self.providers: list[_FakeProvider] = []
        self.instrumented_providers: list[object | None] = []
        self.instrumentor = _FakeInstrumentor(self)
        self.provider: object | None = None
        self.set_calls: list[object] = []
        self.get_provider_calls = 0
        self.legacy_get_meter_calls = 0
        self.fail_provider_construction = False
        self.fail_instrumentation = False
        self.silent_instrumentation_decline = False
        self.fail_first_provider_shutdown = False
        self.fail_reader_shutdown = False
        self.instrumentation_started: threading.Event | None = None
        self.release_instrumentation: threading.Event | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", True)
        monkeypatch.setattr(otel_metrics, "SERVICE_NAME", "service.name", raising=False)
        monkeypatch.setattr(
            otel_metrics, "SERVICE_VERSION", "service.version", raising=False
        )
        monkeypatch.setattr(otel_metrics, "Resource", self.make_resource, raising=False)
        monkeypatch.setattr(
            otel_metrics,
            "PrometheusMetricReader",
            self.make_reader,
            raising=False,
        )
        monkeypatch.setattr(
            otel_metrics, "MeterProvider", self.make_provider, raising=False
        )
        monkeypatch.setattr(
            otel_metrics,
            "SystemMetricsInstrumentor",
            lambda: self.instrumentor,
            raising=False,
        )
        monkeypatch.setattr(otel_metrics, "metrics", self, raising=False)

    def make_resource(self, *, attributes: dict[str, str]) -> object:
        self.calls["resource"] += 1
        self.resource_attributes.update(attributes)
        return object()

    def make_reader(self) -> _FakeReader:
        self.calls["reader"] += 1
        return _FakeReader(self)

    def make_provider(
        self,
        *,
        resource: object,
        metric_readers: list[_FakeReader],
    ) -> _FakeProvider:
        self.calls["provider"] += 1
        if self.fail_provider_construction:
            raise RuntimeError("PROVIDER-FAILURE-SENTINEL")
        provider = _FakeProvider(self, resource, metric_readers)
        self.providers.append(provider)
        return provider

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


@pytest.fixture
def otel_harness(monkeypatch: pytest.MonkeyPatch) -> _FakeOtelHarness:
    harness = _FakeOtelHarness()
    harness.install(monkeypatch)
    return harness


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
    otel_harness: _FakeOtelHarness,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "SERVICE-NAME-SENTINEL")
    monkeypatch.delenv("OTEL_SERVICE_VERSION", raising=False)

    with caplog.at_level(logging.INFO):
        assert otel_metrics.init_metrics() is True
        assert otel_metrics.init_metrics() is True

    assert otel_harness.calls == {
        "resource": 1,
        "reader": 1,
        "provider": 1,
        "provider_get_meter": 1,
        "instrument": 1,
        "uninstrument": 0,
        "provider_shutdown": 0,
        "reader_shutdown": 0,
    }
    assert otel_harness.set_calls == [otel_harness.providers[0]]
    assert otel_harness.get_provider_calls == 1
    assert otel_harness.legacy_get_meter_calls == 0
    assert otel_harness.provider is otel_harness.providers[0]
    assert otel_metrics._meter is otel_harness.providers[0].meter
    assert otel_harness.instrumentor.is_instrumented_by_opentelemetry is True
    assert otel_harness.resource_attributes == {
        "service.name": "SERVICE-NAME-SENTINEL",
        "service.version": "0.1.0",
    }
    messages = [record.getMessage() for record in caplog.records]
    assert messages.count("OpenTelemetry metrics initialized.") == 1
    assert "SERVICE-NAME-SENTINEL" not in "\n".join(messages)


def test_otel_setup_failure_propagates_and_allows_retry(
    otel_harness: _FakeOtelHarness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    otel_harness.fail_instrumentation = True
    otel_harness.fail_first_provider_shutdown = True

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError, match="SETUP-FAILURE-SENTINEL"):
            otel_metrics.init_metrics()

    assert otel_harness.calls == {
        "resource": 1,
        "reader": 1,
        "provider": 1,
        "provider_get_meter": 1,
        "instrument": 1,
        "uninstrument": 0,
        "provider_shutdown": 1,
        "reader_shutdown": 1,
    }
    assert otel_harness.provider is None
    assert otel_harness.set_calls == []
    assert otel_harness.get_provider_calls == 0
    assert otel_harness.legacy_get_meter_calls == 0
    assert otel_harness.instrumented_providers == [otel_harness.providers[0]]
    assert otel_harness.instrumentor.is_instrumented_by_opentelemetry is False
    assert otel_metrics._initialization_result is None
    assert otel_metrics._meter is None
    assert "OpenTelemetry metrics initialized." not in [
        record.getMessage() for record in caplog.records
    ]

    otel_harness.fail_instrumentation = False

    with caplog.at_level(logging.INFO):
        assert otel_metrics.init_metrics() is True

    assert otel_harness.calls == {
        "resource": 2,
        "reader": 2,
        "provider": 2,
        "provider_get_meter": 2,
        "instrument": 2,
        "uninstrument": 0,
        "provider_shutdown": 1,
        "reader_shutdown": 1,
    }
    assert otel_harness.providers[0] is not otel_harness.providers[1]
    assert otel_harness.provider is otel_harness.providers[1]
    assert otel_harness.set_calls == [otel_harness.providers[1]]
    assert otel_harness.get_provider_calls == 1
    assert otel_harness.instrumented_providers == otel_harness.providers
    assert otel_harness.instrumentor.is_instrumented_by_opentelemetry is True
    assert otel_metrics._initialization_result is True
    assert otel_metrics._meter is otel_harness.providers[1].meter
    assert [record.getMessage() for record in caplog.records].count(
        "OpenTelemetry metrics initialized."
    ) == 1


def test_otel_provider_construction_failure_cleans_up_reader(
    otel_harness: _FakeOtelHarness,
) -> None:
    otel_harness.fail_provider_construction = True
    otel_harness.fail_reader_shutdown = True

    with pytest.raises(RuntimeError, match="PROVIDER-FAILURE-SENTINEL"):
        otel_metrics.init_metrics()

    assert otel_harness.calls == {
        "resource": 1,
        "reader": 1,
        "provider": 1,
        "provider_get_meter": 0,
        "instrument": 0,
        "uninstrument": 0,
        "provider_shutdown": 0,
        "reader_shutdown": 1,
    }
    assert otel_harness.set_calls == []
    assert otel_metrics._initialization_result is None
    assert otel_metrics._meter is None


def test_otel_preexisting_provider_fails_closed_and_cleans_up_local_setup(
    otel_harness: _FakeOtelHarness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    existing_provider = object()
    otel_harness.provider = existing_provider

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError, match="ownership"):
            otel_metrics.init_metrics()

    assert otel_harness.calls == {
        "resource": 1,
        "reader": 1,
        "provider": 1,
        "provider_get_meter": 1,
        "instrument": 1,
        "uninstrument": 1,
        "provider_shutdown": 1,
        "reader_shutdown": 1,
    }
    assert otel_harness.provider is existing_provider
    assert otel_harness.set_calls == [otel_harness.providers[0]]
    assert otel_harness.get_provider_calls == 1
    assert otel_harness.legacy_get_meter_calls == 0
    assert otel_harness.instrumented_providers == [otel_harness.providers[0]]
    assert otel_harness.instrumentor.is_instrumented_by_opentelemetry is False
    assert otel_metrics._initialization_result is None
    assert otel_metrics._meter is None
    assert "OpenTelemetry metrics initialized." not in [
        record.getMessage() for record in caplog.records
    ]


def test_otel_preinstrumented_singleton_fails_without_uninstrumenting_owner(
    otel_harness: _FakeOtelHarness,
) -> None:
    otel_harness.instrumentor.is_instrumented_by_opentelemetry = True

    with pytest.raises(RuntimeError, match="already instrumented"):
        otel_metrics.init_metrics()

    assert otel_harness.calls == {
        "resource": 1,
        "reader": 1,
        "provider": 1,
        "provider_get_meter": 1,
        "instrument": 0,
        "uninstrument": 0,
        "provider_shutdown": 1,
        "reader_shutdown": 1,
    }
    assert otel_harness.instrumentor.is_instrumented_by_opentelemetry is True
    assert otel_harness.set_calls == []
    assert otel_metrics._initialization_result is None
    assert otel_metrics._meter is None


def test_otel_silent_instrumentation_decline_fails_without_uninstrumenting(
    otel_harness: _FakeOtelHarness,
) -> None:
    otel_harness.silent_instrumentation_decline = True

    with pytest.raises(RuntimeError, match="did not activate"):
        otel_metrics.init_metrics()

    assert otel_harness.calls == {
        "resource": 1,
        "reader": 1,
        "provider": 1,
        "provider_get_meter": 1,
        "instrument": 1,
        "uninstrument": 0,
        "provider_shutdown": 1,
        "reader_shutdown": 1,
    }
    assert otel_harness.instrumentor.is_instrumented_by_opentelemetry is False
    assert otel_harness.set_calls == []
    assert otel_metrics._initialization_result is None
    assert otel_metrics._meter is None


def test_otel_concurrent_start_publishes_only_complete_state(
    otel_harness: _FakeOtelHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers_ready = threading.Barrier(8)
    instrumentation_started = threading.Event()
    release_instrumentation = threading.Event()
    get_meter_waiting = threading.Event()
    otel_harness.instrumentation_started = instrumentation_started
    otel_harness.release_instrumentation = release_instrumentation

    def call_init_metrics() -> bool:
        workers_ready.wait(timeout=5)
        return otel_metrics.init_metrics()

    def capture_waiting_warning(*_args: object, **_kwargs: object) -> None:
        get_meter_waiting.set()

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
            assert otel_harness.provider is None
            assert otel_harness.set_calls == []

            meter_future = executor.submit(otel_metrics._get_meter)
            assert get_meter_waiting.wait(timeout=5)
            assert not meter_future.done()
        finally:
            release_instrumentation.set()

        assert [future.result(timeout=5) for future in init_futures] == [True] * 8
        assert meter_future.result(timeout=5) is otel_harness.providers[0].meter

    assert otel_harness.calls == {
        "resource": 1,
        "reader": 1,
        "provider": 1,
        "provider_get_meter": 1,
        "instrument": 1,
        "uninstrument": 0,
        "provider_shutdown": 0,
        "reader_shutdown": 0,
    }
    assert otel_harness.provider is otel_harness.providers[0]
    assert otel_harness.set_calls == [otel_harness.providers[0]]
    assert otel_harness.get_provider_calls == 1
    assert otel_harness.legacy_get_meter_calls == 0
    assert otel_harness.instrumented_providers == [otel_harness.providers[0]]
    assert otel_harness.instrumentor.is_instrumented_by_opentelemetry is True
    assert otel_metrics._meter is otel_harness.providers[0].meter
    assert otel_metrics._initialization_result is True


def _enable_metrics(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    """Opt the listener in.

    Since TASK-25914 the listener is gated on ``[metrics] enabled``, so a test
    that wants to reach the binding path has to say so. Without this the
    function short-circuits at the gate and these tests would pass while
    exercising nothing.
    """
    resolved = {"enabled": True, "port": 8000, "bind_address": "127.0.0.1"}
    resolved.update(overrides)
    monkeypatch.setattr(
        prometheus_metrics, "_metrics_server_config", lambda: dict(resolved)
    )


def test_prometheus_unavailable_returns_false_and_reports_info(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_metrics(monkeypatch)
    monkeypatch.setattr(prometheus_metrics, "PROMETHEUS_AVAILABLE", False)

    with caplog.at_level(logging.INFO):
        assert prometheus_metrics.init_metrics_server() is False

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Prometheus metrics listener is enabled in config but the optional "
        "dependency is missing. Install tldw_chatbook[debugging] to use it."
    ]


def test_disabled_listener_is_silent_at_info(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A feature the user never enabled must not narrate itself at startup."""
    monkeypatch.setattr(
        prometheus_metrics,
        "_metrics_server_config",
        lambda: {"enabled": False, "port": 8000, "bind_address": "127.0.0.1"},
    )
    monkeypatch.setattr(prometheus_metrics, "PROMETHEUS_AVAILABLE", False)

    with caplog.at_level(logging.INFO):
        assert prometheus_metrics.init_metrics_server() is False

    assert [record.getMessage() for record in caplog.records] == []


def test_prometheus_available_returns_true_without_real_listener(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[tuple[int, str]] = []

    def record_start(port: int, addr: str = "0.0.0.0") -> None:
        calls.append((port, addr))

    _enable_metrics(monkeypatch)
    monkeypatch.setattr(prometheus_metrics, "PROMETHEUS_AVAILABLE", True)
    monkeypatch.setattr(prometheus_metrics, "start_http_server", record_start)

    with caplog.at_level(logging.INFO):
        assert prometheus_metrics.init_metrics_server(8123) is True

    assert calls == [(8123, "127.0.0.1")]
    assert [record.getMessage() for record in caplog.records] == [
        "Prometheus metrics listener started on 127.0.0.1:8123 (unauthenticated "
        "-- bind address is configurable via [metrics] bind_address)"
    ]


def test_prometheus_server_start_failure_propagates_without_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = RuntimeError("PROMETHEUS-START-FAILURE-SENTINEL")

    def fail_start(port: int, addr: str = "0.0.0.0") -> None:
        assert port == 8124
        raise failure

    _enable_metrics(monkeypatch)
    monkeypatch.setattr(prometheus_metrics, "PROMETHEUS_AVAILABLE", True)
    monkeypatch.setattr(prometheus_metrics, "start_http_server", fail_start)

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError) as caught:
            prometheus_metrics.init_metrics_server(8124)

    assert caught.value is failure
    assert [record.getMessage() for record in caplog.records] == []
