# Local metrics and telemetry

This document describes the three separate metrics paths, the opt-in exposition listener, and the privacy/cardinality patterns. All telemetry in this app is local-scrape-only by default; there is no remote exporter.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Prometheus path (the de-facto API) | `Metrics/metrics.py` | `log_counter`, `log_histogram`, `log_gauge`, `timeit`, `log_resource_usage`, `init_metrics_server`; thread-safe dynamic registry; no-op dummies when `prometheus_client` is missing |
| OTel path | `Metrics/Otel_Metrics.py` | `init_metrics` (idempotent, fail-closed with full cleanup), mirrored `log_counter` / `log_histogram` / `timeit`; the only reader wired is Prometheus |
| Loguru path | `Metrics/metrics_logger.py` | custom `METRIC` log level, JSON-ish lines gated by `TLDW_METRICS_LOGGING`; async-aware `timeit` (cancellation records "cancelled") |
| Legacy | `Metrics/logger_config.py` | console-only loguru setup; file sinks deliberately disabled (privacy boundary) |
| Boot wiring | `app_entry.py` | `init_metrics_server()` and `init_otel_metrics()` behind opt-in gates, both failure-tolerant (logs exception type only, boot continues) |

Note: `Metrics/__init__.py` is empty — imports must name the submodule.

## The three paths and who actually uses them

1. **`Metrics.metrics` (prometheus_client)** — what feature code calls: ~76 modules import `log_counter`/`log_histogram` from here. Instruments accumulate in-process in a dynamic registry.
2. **`Metrics.Otel_Metrics`** — initialized at boot (MeterProvider + PrometheusMetricReader + system instrumentor) but its instrument functions have essentially no feature call sites on this tree. The only OTel exporter wired is the Prometheus reader; **there is no OTLP/remote exporter**.
3. **`Metrics/metrics_logger`** — a separate loguru-based path emitting `METRIC`-level structured lines when enabled; consumed by evals/chat/scheduling handlers for run logs, not by the registry.

## Instrumented examples (verified)

`app_startup_phase_duration_seconds{phase=…}` (startup phases incl. theme registration), `chat_api_call_attempt{api_endpoint}` (provider dispatch), `watchlist_checks` / `watchlist_check_duration{status,subscription_type,url_errors,shadow}`, `scheduler_dispatch_late{task_type,cause∈{away,busy,stalled}}`.

## Exposition (dataflow)

`app_entry` → gate `[metrics] enabled` → availability check → `start_http_server(port, addr)` on the configured bind address (default loopback 9090). Dependency presence alone never binds (pinned by test); a non-loopback bind logs a loud warning because the endpoint is unauthenticated by nature. `METRICS_PORT` overrides the port but never enables the listener.

## Privacy and cardinality patterns

- Endpoint labels are collapsed through a safe-endpoint mapper: unrecognized provider endpoints become the constant `<unrecognised>` in metrics, while the log line keeps only a length + redaction note — metrics and diagnostics deliberately diverge so neither leaks credentials or unbounded label values.
- Watchlist `url_errors` is a boolean label, not a count (a per-URL label would explode cardinality).
- Boot failures log `exception_type={}` placeholders, never payloads; persistent-log redaction is owned by `Logging_Config.py`; the legacy metrics file sinks are disabled.
- Both metric modules carry explicit low-cardinality-only contracts.

## Config keys

`[metrics] enabled` (default **false**), `[metrics] bind_address` (default `127.0.0.1`), `[metrics] port` (default 9090); env `METRICS_PORT` (port override only), `TLDW_METRICS_LOGGING` (loguru metric lines; off by default), `OTEL_SERVICE_NAME` / `OTEL_SERVICE_VERSION`. All bool coercion fails closed (a quoted `"false"` must not enable anything).

## Failure behavior

Missing dependencies degrade to no-op dummy instruments with a debug log; every `log_*` call wraps its own errors — metrics can never break the app. OTel init is idempotent with a double-checked lock and full cleanup + re-raise on partial failure.

## Governance note

There is **no dedicated metrics/telemetry ADR** — the rationale lives in code comments and tests (the opt-in listener gate, the endpoint-redaction divergence, the boolean-label decision). `Docs/Design/Local_Telemetry/OTel-Metrics.md` is partially stale: its `[telemetry] …` keys and `OTEL_METRICS_ENABLED` do not exist in code; the config keys above are authoritative. Tests: `Tests/Metrics/test_metrics_server_gate.py`, `test_startup_metric_outcomes.py`, `Tests/Utils/test_metrics_logger.py`, `Tests/Metrics/test_logger_config_privacy.py`.

## Verified gotchas

1. The de-facto metrics API is `Metrics.metrics`; adding instruments via `Otel_Metrics` today would create a second, essentially unread path.
2. No dashboards and no Logs-screen integration exist — exposition is scrape-only when enabled.
3. The OTel design doc's config section is aspirational; do not copy its keys.
