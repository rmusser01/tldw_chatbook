# Startup Diagnostic Clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make startup output distinguish optional capability absence, unverified security posture, recoverable cache rejection, and real initialization failures without duplicates or sensitive interpolation.

**Architecture:** Keep each subsystem's existing boundary authoritative: the task-loader import owns the once-per-process HuggingFace capability notice, each metrics initializer owns its boolean outcome, and current privacy/cache owners keep their warning and deduplication scope. Add only one lock-protected OpenTelemetry result sentinel, route recognized remote dataset IDs through existing typed errors, and change bounded copy at the existing emitters.

**Tech Stack:** Python 3.11+ logging/warnings, Loguru, `threading.Lock`, pytest, subprocess isolation

**Backlog:** `TASK-24532`

**ADR required:** no

**ADR path:** N/A

**Reason:** Severity, ownership, wording, and existing error routing change without altering persistence, privacy policy, runtime-policy decisions, cache schema, dependencies, or startup entry-point boundaries.

---

## File Map

- Modify `tldw_chatbook/Evals/task_loader.py`: make expected HuggingFace absence informational and actionable at the import boundary.
- Modify `tldw_chatbook/Evals/dataset_loader.py`: route recognized remote IDs through a typed missing-dependency error.
- Modify `tldw_chatbook/Evals/eval_runner.py`: mirror the same public typed outcome in the production runner's dataset loader.
- Modify `tldw_chatbook/Metrics/Otel_Metrics.py`: silence import, serialize first initialization, return a stable boolean, and own one static outcome message.
- Modify `tldw_chatbook/Metrics/metrics.py`: return a boolean and own Prometheus normal outcome messages.
- Modify `tldw_chatbook/app.py`: remove duplicate/unconditional success messages and bound unexpected metrics errors to exception type.
- Modify `tldw_chatbook/DB/private_sqlite.py`: clarify that platform verification is unavailable while database work continues unverified.
- Modify `tldw_chatbook/runtime_policy/source_state.py`: clarify the same posture at the existing operation-scoped warning boundary.
- Modify `tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py`: retain count-only warning severity while stating partial-load and discovery recovery behavior.
- Modify `Tests/Evals/test_task_loader.py`: isolate and assert the once-per-process informational import notice.
- Modify `Tests/Evals/test_eval_runner.py`: assert both public dataset-loader paths return the typed actionable missing-dependency outcome.
- Create `Tests/Metrics/test_startup_metric_outcomes.py`: isolate import behavior and test OpenTelemetry/Prometheus outcome ownership without real providers or listeners.
- Modify `Tests/App/test_startup_init_hygiene.py`: execute the alternate module entry block's exact metrics statements with sentinel failures to pin removal of unconditional success logs and type-only unexpected failures.
- Modify `Tests/DB/test_private_sqlite.py`: pin severity, wording, deduplication, and sentinel exclusion.
- Modify `Tests/RuntimePolicy/test_runtime_policy_private_store.py`: pin operation continuation, warning severity, and sentinel exclusion.
- Modify `Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py`: pin count-only partial-load/recovery copy and sentinel exclusion.
- Modify `backlog/tasks/task-24532 - Clarify-and-deduplicate-startup-diagnostics.md`: record final evidence and close the task.

No central registry, persistent ledger, new dependency, shared dataset abstraction, or installed-entry-point telemetry side effect is planned.

## Pre-implementation formatter baseline

Before changing production or tests, run the final formatter command from Task
5 without the not-yet-created metrics test. Expected inherited baseline: exit 1
naming exactly these three files and reporting the other twelve as formatted:

```text
Tests/App/test_startup_init_hygiene.py
tldw_chatbook/Evals/eval_runner.py
tldw_chatbook/app.py
```

Do not bulk-format those large legacy files. Keep task-owned hunks
formatter-consistent, format the new metrics test before its first commit, and
require the final formatter output to retain exactly this three-file baseline
with no newly dirty file. Record the inherited deviation rather than claiming a
green whole-file formatter gate.

### Task 1: Make optional evaluation diagnostics accurate at import and feature use

**Files:**
- Modify: `Tests/Evals/test_task_loader.py:1-30`
- Modify: `Tests/Evals/test_eval_runner.py:1-35`
- Modify: `tldw_chatbook/Evals/task_loader.py:20-40`
- Modify: `tldw_chatbook/Evals/dataset_loader.py:55-90, 329-340`
- Modify: `tldw_chatbook/Evals/eval_runner.py:185-220, 446-456`

- [ ] **Step 1: Add an isolated task-loader import diagnostic test**

Add standard-library imports `subprocess`, `sys`, and `Path` to `test_task_loader.py`, define the repository root once, and add:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_missing_datasets_import_notice_is_once_and_informational() -> None:
    script = """
import sys
sys.modules["datasets"] = None
from loguru import logger
logger.remove()
logger.add(sys.stdout, format="{level}|{message}")
import tldw_chatbook.Evals.task_loader
import tldw_chatbook.Evals.task_loader
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    notices = [
        line for line in result.stdout.splitlines()
        if "HuggingFace evaluation datasets" in line
    ]
    assert len(notices) == 1
    assert notices[0].startswith("INFO|")
    assert "pip install datasets" in notices[0]
    assert "WARNING|" not in result.stdout
```

Subprocess isolation is mandatory because ordinary test collection imports `task_loader` before an in-process Loguru sink can observe its import boundary.

- [ ] **Step 2: Add public typed-error tests for both dataset loader implementations**

In `test_eval_runner.py`, import both modules rather than patching only private helpers:

```python
from tldw_chatbook.Evals import dataset_loader as standalone_dataset_loader
from tldw_chatbook.Evals import eval_runner as eval_runner_module
from tldw_chatbook.Evals.eval_errors import DatasetLoadingError
```

Add one parameterized public-boundary test:

```python
@pytest.mark.parametrize(
    ("module", "loader"),
    [
        (standalone_dataset_loader, standalone_dataset_loader.DatasetLoader),
        (eval_runner_module, eval_runner_module.DatasetLoader),
    ],
)
def test_remote_dataset_without_optional_dependency_is_actionable(
    monkeypatch, module, loader
):
    monkeypatch.setattr(module, "HF_DATASETS_AVAILABLE", False)
    config = TaskConfig(
        name="remote",
        description="remote",
        task_type="question_answer",
        dataset_name="owner/dataset",
        metric="exact_match",
    )

    with pytest.raises(DatasetLoadingError) as caught:
        loader.load_dataset_samples(config)

    assert caught.value.context.message == "HuggingFace dataset support is unavailable"
    assert caught.value.context.suggestion == "Install it with: pip install datasets"
    assert "Unexpected error" not in str(caught.value)
    assert "Cannot determine dataset type" not in str(caught.value)
```

Retain or add controls showing an existing local path still uses `_load_local_dataset` and an invalid non-path/non-`owner/dataset` name still returns the existing cannot-determine-type error.

- [ ] **Step 3: Run the evaluation tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Evals/test_task_loader.py::test_missing_datasets_import_notice_is_once_and_informational
../../.venv/bin/python -m pytest -q \
  Tests/Evals/test_eval_runner.py \
  -k "remote_dataset_without_optional_dependency"
```

Expected before the fix: the subprocess captures a warning, and both public loaders return a generic route/import outcome instead of the typed actionable error.

- [ ] **Step 4: Make the import notice informational and static**

In `task_loader.py`, keep the existing import guard and Python import-cache deduplication, changing only the emit call:

```python
logger.info(
    "HuggingFace evaluation datasets are unavailable. "
    "Install with: pip install datasets"
)
```

Do not add a once registry or emit from `dataset_loader.py`/`eval_runner.py` imports.

- [ ] **Step 5: Route recognized remote IDs through typed feature-use guards**

In both public `load_dataset_samples` methods, change:

```python
elif HF_DATASETS_AVAILABLE and "/" in dataset_name:
```

to:

```python
elif "/" in dataset_name:
```

Then replace each private helper's missing-dependency `ImportError` with the existing typed error:

```python
raise DatasetLoadingError(
    ErrorContext(
        category=ErrorCategory.DATASET_LOADING,
        severity=ErrorSeverity.ERROR,
        message="HuggingFace dataset support is unavailable",
        suggestion="Install it with: pip install datasets",
        is_retryable=False,
    )
)
```

Duplicate these few lines in the two existing loaders; do not introduce a shared abstraction for two independent legacy implementations.

- [ ] **Step 6: Run focused Evals tests and verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Evals/test_task_loader.py \
  Tests/Evals/test_eval_runner.py
```

Expected: both files pass; task-loader import output is informational once and feature use remains actionable.

- [ ] **Step 7: Commit the Evals slice**

```bash
git add -- \
  tldw_chatbook/Evals/task_loader.py \
  tldw_chatbook/Evals/dataset_loader.py \
  tldw_chatbook/Evals/eval_runner.py \
  Tests/Evals/test_task_loader.py \
  Tests/Evals/test_eval_runner.py
git diff --cached --check
git commit -m "fix: clarify optional evaluation dependency outcomes"
```

### Task 2: Make OpenTelemetry initialization silent-on-import and idempotent

**Files:**
- Create: `Tests/Metrics/test_startup_metric_outcomes.py`
- Modify: `tldw_chatbook/Metrics/Otel_Metrics.py:20-105`

- [ ] **Step 1: Add an isolated silent-import test**

Create the test module with standard imports and this subprocess proof:

```python
from __future__ import annotations

import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_chatbook.Metrics.Otel_Metrics as otel_metrics
import tldw_chatbook.Metrics.metrics as prometheus_metrics

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
    )
    assert result.returncode == 0
    assert "OpenTelemetry" not in result.stdout
    assert "OpenTelemetry" not in result.stderr
```

- [ ] **Step 2: Add a fixture that resets only the planned module state**

```python
@pytest.fixture(autouse=True)
def reset_otel_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otel_metrics, "_initialization_result", None)
    monkeypatch.setattr(otel_metrics, "_meter", None)
```

Do not mutate the real process-global OpenTelemetry provider in tests.

- [ ] **Step 3: Add concurrent unavailable and repeated available outcome tests**

```python
def test_otel_unavailable_initializes_once_across_threads(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", False)
    with caplog.at_level(logging.INFO):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: otel_metrics.init_metrics(), range(16)))

    assert results == [False] * 16
    messages = [record.getMessage() for record in caplog.records]
    unavailable = [m for m in messages if "OpenTelemetry metrics are unavailable" in m]
    assert len(unavailable) == 1
    assert "tldw_chatbook[debugging]" in unavailable[0]
    assert not any("initialized" in message.lower() for message in messages)
```

For the available path, explicitly set `OTEL_AVAILABLE=True`, then stub every
optional-import global with `raising=False` so the test works in a plain
`.[dev]` environment where those names were never imported:

```python
monkeypatch.setattr(otel_metrics, "OTEL_AVAILABLE", True)
monkeypatch.setattr(otel_metrics, "SERVICE_NAME", "service.name", raising=False)
monkeypatch.setattr(otel_metrics, "SERVICE_VERSION", "service.version", raising=False)
monkeypatch.setattr(otel_metrics, "Resource", fake_resource, raising=False)
monkeypatch.setattr(
    otel_metrics,
    "PrometheusMetricReader",
    fake_reader,
    raising=False,
)
monkeypatch.setattr(otel_metrics, "MeterProvider", fake_provider, raising=False)
monkeypatch.setattr(
    otel_metrics,
    "SystemMetricsInstrumentor",
    fake_instrumentor,
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
```

Set `OTEL_SERVICE_NAME` to `SERVICE-NAME-SENTINEL`, call `init_metrics()` twice,
and assert:

```python
assert otel_metrics.init_metrics() is True
assert otel_metrics.init_metrics() is True
assert calls == {
    "reader": 1,
    "provider": 1,
    "set_provider": 1,
    "get_meter": 1,
    "instrument": 1,
}
messages = [record.getMessage() for record in caplog.records]
assert messages.count("OpenTelemetry metrics initialized.") == 1
assert "SERVICE-NAME-SENTINEL" not in "\n".join(messages)
```

- [ ] **Step 4: Run the OpenTelemetry tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Metrics/test_startup_metric_outcomes.py -k "otel"
```

Expected before the fix: import emits a warning, unavailable calls return `None` and duplicate notices, and available repeat calls replace provider/instrumentation state.

- [ ] **Step 5: Add the minimal lock-protected result sentinel and authoritative messages**

Delete the import-time warning. Beside the existing instrument lock/state, add:

```python
_initialization_lock = threading.Lock()
_initialization_result: bool | None = None
```

Change the public signature and guard:

```python
def init_metrics() -> bool:
    """Initialize OpenTelemetry once and return whether it is available."""
    global _meter, _initialization_result
    if _initialization_result is not None:
        return _initialization_result

    with _initialization_lock:
        if _initialization_result is not None:
            return _initialization_result
        if not OTEL_AVAILABLE:
            logging.info(
                "OpenTelemetry metrics are unavailable. "
                "Install tldw_chatbook[debugging] to enable them."
            )
            _initialization_result = False
            return False

        # Keep the existing Resource/reader/provider/meter/instrumentation setup here.
        logging.info("OpenTelemetry metrics initialized.")
        _initialization_result = True
        return True
```

Keep setup inside the lock. If setup raises, leave `_initialization_result` as `None` so the caller receives the real failure and a later explicit attempt may retry. Keep service name/version in the SDK resource but never log them.

- [ ] **Step 6: Run the OpenTelemetry tests and verify GREEN**

Run the same `-k "otel"` command from Step 4.

Expected: all selected tests pass without touching a real provider, exporter, instrumentation hook, or network listener.

- [ ] **Step 7: Commit the OpenTelemetry slice**

```bash
git add -- tldw_chatbook/Metrics/Otel_Metrics.py Tests/Metrics/test_startup_metric_outcomes.py
git diff --cached --check
git commit -m "fix: own OpenTelemetry initialization outcomes"
```

### Task 3: Make Prometheus authoritative and remove caller overclaims

**Files:**
- Modify: `Tests/Metrics/test_startup_metric_outcomes.py`
- Modify: `Tests/App/test_startup_init_hygiene.py`
- Modify: `tldw_chatbook/Metrics/metrics.py:240-258`
- Modify: `tldw_chatbook/app.py:15675-15705`

- [ ] **Step 1: Add unavailable and available Prometheus initializer tests**

Use `caplog` and a stubbed server function:

```python
def test_prometheus_unavailable_returns_false_and_reports_info(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(prometheus_metrics, "PROMETHEUS_AVAILABLE", False)
    with caplog.at_level(logging.INFO):
        assert prometheus_metrics.init_metrics_server(8000) is False
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Prometheus metrics are unavailable. "
        "Install tldw_chatbook[debugging] to enable them."
    ]


def test_prometheus_available_returns_true_without_real_listener(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(prometheus_metrics, "PROMETHEUS_AVAILABLE", True)
    monkeypatch.setattr(prometheus_metrics, "start_http_server", calls.append)
    with caplog.at_level(logging.INFO):
        assert prometheus_metrics.init_metrics_server(8123) is True
    assert calls == [8123]
    assert [record.getMessage() for record in caplog.records] == [
        "Prometheus metrics server started on port 8123"
    ]
```

- [ ] **Step 2: Execute the alternate startup block's exact metrics statements with sentinel failures**

In `test_startup_init_hygiene.py`, use its existing `ast` import to isolate the
two `try` statements containing `init_metrics_server` and
`init_otel_metrics` from the module's `if __name__ == "__main__"` body. Compile
only those exact source nodes into a test module, then execute them with inert
dependencies and sentinel-bearing failures:

```python
def test_alternate_startup_metrics_failures_are_type_only() -> None:
    source = (REPO_ROOT / "tldw_chatbook/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    metrics_tries = [
        node
        for node in main_block.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in {"init_metrics_server", "init_otel_metrics"}
            for call in ast.walk(node)
        )
    ]
    assert len(metrics_tries) == 2

    class FakeLogger:
        def __init__(self) -> None:
            self.infos: list[str] = []
            self.warnings: list[str] = []

        def info(self, message: str, *args: object) -> None:
            self.infos.append(message.format(*args))

        def warning(self, message: str, *args: object) -> None:
            self.warnings.append(message.format(*args))

    exception_sentinel = "METRICS-EXCEPTION-SENTINEL-7d31"

    def fail_metrics(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(exception_sentinel)

    executable = ast.fix_missing_locations(
        ast.Module(body=metrics_tries, type_ignores=[])
    )
    failure_logger = FakeLogger()
    failure_namespace = {
        "os": SimpleNamespace(environ={"METRICS_PORT": "8000"}),
        "init_metrics_server": fail_metrics,
        "init_otel_metrics": fail_metrics,
        "loguru_logger": failure_logger,
    }
    code = compile(executable, "<startup-metrics>", "exec")
    exec(code, failure_namespace)

    assert failure_logger.infos == []
    assert len(failure_logger.warnings) == 2
    assert all(
        "exception_type=RuntimeError" in message
        for message in failure_logger.warnings
    )
    assert exception_sentinel not in "\n".join(failure_logger.warnings)

    success_logger = FakeLogger()
    success_namespace = {
        "os": SimpleNamespace(environ={"METRICS_PORT": "8000"}),
        "init_metrics_server": lambda **_kwargs: True,
        "init_otel_metrics": lambda: True,
        "loguru_logger": success_logger,
    }
    exec(code, success_namespace)

    assert success_logger.infos == []
    assert success_logger.warnings == []
```

This executable sentinel test catches `{exc}`, `str(exc)`, `repr(exc)`, and
traceback/message logging, proves failure remains warning severity, and makes
unconditional caller success reachable in a separate successful execution.
It does so without booting the TUI or adding a production startup helper. Keep
the existing `SimpleNamespace` import in this test module.

- [ ] **Step 3: Run the Prometheus/caller tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Metrics/test_startup_metric_outcomes.py \
  -k "prometheus"
../../.venv/bin/python -m pytest -q \
  Tests/App/test_startup_init_hygiene.py \
  -k "metrics"
```

Expected before the fix: initializer returns `None`, unavailable is warning severity, and the caller still emits unconditional success and arbitrary exception text.

- [ ] **Step 4: Return authoritative Prometheus outcomes**

Change only `init_metrics_server`:

```python
def init_metrics_server(port: int = 8000) -> bool:
    """Start the Prometheus server and report whether it is available."""
    if not PROMETHEUS_AVAILABLE:
        logging.info(
            "Prometheus metrics are unavailable. "
            "Install tldw_chatbook[debugging] to enable them."
        )
        return False
    start_http_server(port)
    logging.info("Prometheus metrics server started on port %s", port)
    return True
```

Do not catch server-start failures here; they continue to reach the application warning path.

- [ ] **Step 5: Remove duplicate success and sanitize unexpected failures in `app.py`**

Keep both initialization attempts, delete both caller success logs, and replace exception interpolation with bounded type-only diagnostics:

```python
except Exception as exc:
    loguru_logger.warning(
        "Prometheus metrics initialization failed (exception_type={}).",
        type(exc).__name__,
    )
```

```python
except Exception as exc:
    loguru_logger.warning(
        "OpenTelemetry metrics initialization failed (exception_type={}).",
        type(exc).__name__,
    )
```

Do not add OpenTelemetry initialization to `main_cli_runner`; the installed `tldw-cli` path remains unchanged.

- [ ] **Step 6: Run the Prometheus/caller tests and verify GREEN**

Run the same command from Step 3.

Expected: all selected tests pass, no real listener is bound, and only initializer-owned normal outcomes remain.

- [ ] **Step 7: Commit the metrics caller-ownership slice**

```bash
git add -- \
  tldw_chatbook/Metrics/metrics.py \
  tldw_chatbook/app.py \
  Tests/Metrics/test_startup_metric_outcomes.py \
  Tests/App/test_startup_init_hygiene.py
git diff --cached --check
git commit -m "fix: centralize startup metrics outcomes"
```

### Task 4: Clarify unverified posture and recoverable cache warnings

**Files:**
- Modify: `Tests/DB/test_private_sqlite.py:2350-2445`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_private_store.py:220-275`
- Modify: `Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py:410-450`
- Modify: `tldw_chatbook/DB/private_sqlite.py:536-546`
- Modify: `tldw_chatbook/runtime_policy/source_state.py:158-175`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py:360-372`

- [ ] **Step 1: Strengthen the SQLite warning contract without weakening deduplication**

Update the existing per-owner and thread-safety tests to assert the warning remains `SQLitePrivacyUnverifiedWarning` and contains all of:

```python
"permission verification is unavailable"
"database operation continues"
"unverified privacy posture"
```

Add path and credential sentinel values to the warning setup and assert neither appears. Retain the current two-owner count, thread race, and warning-as-error retry tests unchanged.

- [ ] **Step 2: Strengthen runtime-policy operation continuation and secrecy assertions**

In `test_runtime_policy_store_reports_windows_posture_as_unverified`, continue asserting two warnings (load and save), then assert every message contains:

```python
"permission verification is unavailable"
"continues"
"unverified_platform"
```

Retain the existing assertion that no message claims privacy, and extend the adjacent sentinel test to prove path/state/credential sentinels remain absent.

- [ ] **Step 3: Strengthen model-cache partial-load and recovery assertions**

Extend `test_disk_cache_diagnostics_are_bounded_and_secret_free`:

```python
assert "count=1" in text
assert "accepted entries remain available" in text
assert "discovery may refresh missing models" in text
assert [item.model_id for item in cache.list("Custom", "later")] == ["safe-model"]
```

The hostile endpoint/cache content sentinel must remain absent and the message must remain at warning severity.

- [ ] **Step 4: Run the three warning contracts and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/DB/test_private_sqlite.py \
  -k "unverified"
../../.venv/bin/python -m pytest -q \
  Tests/RuntimePolicy/test_runtime_policy_private_store.py \
  -k "unverified or diagnostics"
../../.venv/bin/python -m pytest -q \
  Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py \
  -k "diagnostics_are_bounded"
```

Expected before the copy change: severity/count/dedup controls pass, while explicit continuation and recovery-language assertions fail.

- [ ] **Step 5: Change only bounded emitter copy**

Use static SQLite copy:

```python
"SQLite permission verification is unavailable on this platform; "
"database operation continues with an unverified privacy posture"
```

Use bounded runtime-policy copy with the already-normalized operation and posture only:

```python
"Runtime policy permission verification is unavailable; "
"operation={} continues with posture={}.",
operation,
result.status.value,
```

Use count-only cache copy:

```python
logger.warning(
    "Rejected model catalog cache entries (count={}); accepted entries "
    "remain available and discovery may refresh missing models.",
    rejected,
)
```

Do not interpolate paths, owners, rejected records, service names, or exception text.

- [ ] **Step 6: Run the warning tests and verify GREEN**

Run the same focused command from Step 4.

Expected: all selected cases pass with existing warning categories and deduplication scopes preserved.

- [ ] **Step 7: Commit the posture/cache copy slice**

```bash
git add -- \
  tldw_chatbook/DB/private_sqlite.py \
  tldw_chatbook/runtime_policy/source_state.py \
  tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py \
  Tests/DB/test_private_sqlite.py \
  Tests/RuntimePolicy/test_runtime_policy_private_store.py \
  Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py
git diff --cached --check
git commit -m "fix: clarify degraded startup posture"
```

### Task 5: Verify the complete diagnostic slice and close `TASK-24532`

**Files:**
- Test: all files listed above
- Modify: `backlog/tasks/task-24532 - Clarify-and-deduplicate-startup-diagnostics.md`

- [ ] **Step 1: Run the approved focused test files**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Evals/test_task_loader.py \
  Tests/Evals/test_eval_runner.py \
  Tests/Metrics/test_startup_metric_outcomes.py \
  Tests/App/test_startup_init_hygiene.py \
  Tests/DB/test_private_sqlite.py \
  Tests/RuntimePolicy/test_runtime_policy_private_store.py \
  Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py
```

Expected: all selected tests pass. Do not run the full repository suite without explicit user opt-in.

- [ ] **Step 2: Run scoped lint and compilation**

Run:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Evals/task_loader.py \
  tldw_chatbook/Evals/dataset_loader.py \
  tldw_chatbook/Evals/eval_runner.py \
  tldw_chatbook/Metrics/Otel_Metrics.py \
  tldw_chatbook/Metrics/metrics.py \
  tldw_chatbook/app.py \
  tldw_chatbook/DB/private_sqlite.py \
  tldw_chatbook/runtime_policy/source_state.py \
  tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py \
  Tests/Evals/test_task_loader.py \
  Tests/Evals/test_eval_runner.py \
  Tests/Metrics/test_startup_metric_outcomes.py \
  Tests/App/test_startup_init_hygiene.py \
  Tests/DB/test_private_sqlite.py \
  Tests/RuntimePolicy/test_runtime_policy_private_store.py \
  Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Evals/task_loader.py \
  tldw_chatbook/Evals/dataset_loader.py \
  tldw_chatbook/Evals/eval_runner.py \
  tldw_chatbook/Metrics/Otel_Metrics.py \
  tldw_chatbook/Metrics/metrics.py \
  tldw_chatbook/app.py \
  tldw_chatbook/DB/private_sqlite.py \
  tldw_chatbook/runtime_policy/source_state.py \
  tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py \
  Tests/Evals/test_task_loader.py \
  Tests/Evals/test_eval_runner.py \
  Tests/Metrics/test_startup_metric_outcomes.py \
  Tests/App/test_startup_init_hygiene.py \
  Tests/DB/test_private_sqlite.py \
  Tests/RuntimePolicy/test_runtime_policy_private_store.py \
  Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Evals \
  tldw_chatbook/Metrics \
  tldw_chatbook/runtime_policy/source_state.py \
  tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py
git diff --check
```

Expected: Ruff check, compilation, and `git diff --check` exit 0. Ruff format
retains exactly the inherited three-file baseline recorded before
implementation; every other touched file, including the new metrics test,
passes. If broad legacy lint debt appears in unchanged lines, run the
equivalent scoped checks on changed files/lines and document the baseline
rather than silently claiming a clean full-file result.

- [ ] **Step 3: Audit changed diagnostic strings for sentinels and dynamic interpolation**

Run:

```bash
git diff --unified=0 | rg "logger|logging|warnings\.warn|exception_type|HuggingFace|OpenTelemetry|Prometheus|unverified|cache entries"
```

Confirm normal outcomes have one owner, optional absence is informational, posture/cache integrity remains warning severity, exception text is type-only, and no path/credential/service/cache content is newly interpolated.

- [ ] **Step 4: Self-review entry-point boundaries and state resets**

Confirm:

- `python -m tldw_chatbook.app` still attempts both metrics initializers;
- the installed `tldw-cli` path still does not gain OpenTelemetry initialization;
- unavailable and successful OpenTelemetry calls each settle one stable result;
- a setup exception leaves the result unset and propagates to the sanitized caller warning; and
- no cache, privacy, or runtime-policy decision changed with its copy.

- [ ] **Step 5: Complete and verify the Backlog record**

Check all acceptance criteria, add concise Implementation Notes with per-slice commits, exact test results, static-check results, the inherited three-file formatter baseline, modified files, and `ADR required: no`, then run:

```bash
backlog task edit 24532 -s Done
backlog task 24532 --plain
```

Verify the CLI reports the expected `task-24532` path and all criteria are checked before committing.

- [ ] **Step 6: Commit task closeout**

```bash
git add -- "backlog/tasks/task-24532 - Clarify-and-deduplicate-startup-diagnostics.md"
git diff --cached --check
git commit -m "docs: close startup diagnostic clarity task"
```
