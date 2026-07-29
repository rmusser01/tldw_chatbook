# Persistent Operational Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the persistent app log carry six operational events, so a crash or restart leaves behind enough to reconstruct what the app was doing.

**Architecture:** A `persist_event()` wrapper in `Utils/persistent_diagnostics.py` is the single admitted path; it uses stdlib `logging` deliberately, because the persistent marker does not survive the Loguru forwarder and must not. One new `component` token field is added to the existing schema. Six call sites emit events, each from exactly one existing hook. No existing logging call is modified.

**Tech Stack:** Python 3.11+, stdlib `logging`, pytest. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-07-28-persistent-operational-diagnostics-design.md`

## Global Constraints

- **Never make the marker survive Loguru.** Propagating `_tldw_metadata_only_record` through `_forward_loguru_to_standard` would let any `logger.bind(_tldw_metadata_only_record=True).info(secret)` bypass the schema. Verified: the Loguru path currently writes nothing, and that is correct.
- **ADR-029's exclusion list is preserved exactly.** No prompt, message body, provider request/response payload, API key or key fragment, tool argument value, or tool result value becomes persistable. Only `component` is added to the admitted fields.
- **No message text, no tracebacks, no paths** in any emitted event. `exception_type` is a class name only.
- **Do not modify any existing `logger.*` call site.** Every change is additive.
- **The existing privacy tests must pass unchanged**: `Tests/test_persistent_diagnostic_boundary.py`, `Tests/test_logging_private_files.py`, `Tests/test_persistent_diagnostic_sentinel_matrix.py`, `Tests/test_remaining_diagnostic_sentinel_matrix.py`.
- **Merge gate:** this branch requires sign-off from the ADR-029 privacy work's owner before merge. It adds one admitted field and six admitted events to a boundary that work owns.
- Run tests with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`. Use **plain output** — `-q` suppresses the failure summary in this repo and has hidden real failures three times.

---

### Task 1: `component` field and the `persist_event` wrapper

**Files:**
- Modify: `tldw_chatbook/Utils/persistent_diagnostics.py:34-48` (add `component` to `_TOKEN_FIELDS`), `:177` (`__all__`)
- Test: `Tests/Utils/test_persist_event.py` (create)

**Interfaces:**
- Consumes: existing `log_persistent_metadata(target_logger, level, event, **metadata)` and `PersistentDiagnosticFilter`.
- Produces: `persist_event(component: str, event: str, *, level: int = logging.INFO, **fields: Any) -> None`, imported by every later task as `from tldw_chatbook.Utils.persistent_diagnostics import persist_event`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Utils/test_persist_event.py`:

```python
"""TASK-1240: persist_event is the single admitted path to the persistent log."""

from __future__ import annotations

import logging

import pytest

from tldw_chatbook.Utils.persistent_diagnostics import (
    PersistentDiagnosticFilter,
    persist_event,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sink(tmp_path):
    """A real file handler behind the real filter, as the app installs it."""
    path = tmp_path / "app.log"
    handler = logging.FileHandler(path)
    handler.addFilter(PersistentDiagnosticFilter())
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.INFO)
    yield path, handler
    handler.flush()
    root.removeHandler(handler)
    root.setLevel(previous)
    handler.close()


def test_persist_event_reaches_the_sink(sink):
    path, handler = sink
    persist_event("scheduling", "scheduler_configured", item_count=2, status="ok")
    handler.flush()
    written = path.read_text()
    assert "event=scheduler_configured" in written
    assert "component=scheduling" in written
    assert "item_count=2" in written


def test_ordinary_logging_is_still_rejected(sink):
    """The boundary must not widen: only marked records are admitted."""
    path, handler = sink
    logging.getLogger("tldw_chatbook.diagnostics.scheduling").info(
        "an ordinary line that must not persist"
    )
    handler.flush()
    assert path.read_text() == ""


def test_unknown_fields_are_still_rejected(sink):
    """The schema is the guarantee; persist_event must not bypass it."""
    with pytest.raises(ValueError):
        persist_event("app", "app_started", prompt="secret user text")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_persist_event.py --timeout=120`

Expected: FAIL — `ImportError: cannot import name 'persist_event'`.

- [ ] **Step 3: Add `component` to the token schema**

In `tldw_chatbook/Utils/persistent_diagnostics.py`, inside the `_TOKEN_FIELDS = frozenset({...})` literal, add `"component",` immediately after `"transport",`:

```python
        "transport",
        # TASK-1240. Names the subsystem an operational event came from
        # (`scheduling`, `app`, `logging`). Code-side identifiers only, held to
        # the same token regex as every other field here. Chosen to match the
        # label vocabulary in Metrics/metrics_logger.py rather than inventing a
        # second dialect for the same idea.
        "component",
```

- [ ] **Step 4: Add the wrapper**

In the same file, immediately after the `log_persistent_metadata` function definition and before `class PersistentDiagnosticFilter`:

```python
def persist_event(
    component: str,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Record one operational event in the persistent log.

    Uses stdlib logging deliberately. The persistent marker does not survive
    `Logging_Config._forward_loguru_to_standard`, which rebuilds `extra` from
    scratch -- and it must not: if the marker crossed that boundary, any code
    could write `logger.bind(_tldw_metadata_only_record=True).info(secret)` and
    bypass this module's schema entirely.

    Since the rest of the codebase uses `from loguru import logger`, reaching for
    the usual idiom at a persist site would silently write nothing. This wrapper
    exists so no call site has to know that.

    The logger is namespaced `tldw_chatbook.diagnostics.*` rather than the
    caller's module: naming it after the module would interleave persisted
    events with that module's descriptive records and expose them to any
    per-logger level configuration aimed at it. The prefix still satisfies
    `_is_chatbook_record`.
    """

    log_persistent_metadata(
        logging.getLogger(f"tldw_chatbook.diagnostics.{component}"),
        level,
        event,
        component=component,
        **fields,
    )
```

- [ ] **Step 5: Export it**

In the `__all__` list at the bottom of the file, add `"persist_event",` after `"log_persistent_metadata",`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_persist_event.py --timeout=120`

Expected: PASS, 4 tests.

- [ ] **Step 7: Verify the boundary did not widen**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/test_persistent_diagnostic_boundary.py Tests/test_logging_private_files.py Tests/test_persistent_diagnostic_sentinel_matrix.py Tests/test_remaining_diagnostic_sentinel_matrix.py --timeout=300`

Expected: PASS, unchanged from before this task.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Utils/persistent_diagnostics.py Tests/Utils/test_persist_event.py
git commit -m "feat: add persist_event and the component field (TASK-1240)"
```

---

### Task 2: Prove the Loguru path stays closed

**Files:**
- Test: `Tests/Utils/test_persist_event.py` (modify — add one test)

**Interfaces:**
- Consumes: `persist_event` from Task 1; `Logging_Config._forward_loguru_to_standard`.
- Produces: nothing new. This task exists because the closed Loguru path is a security property, not incidental behaviour, and nothing currently asserts it.

- [ ] **Step 1: Write the failing test**

Append to `Tests/Utils/test_persist_event.py`:

```python
def test_a_loguru_record_carrying_the_marker_is_still_rejected(sink):
    """The marker must not survive the Loguru forwarder.

    If it did, any code could write
    `logger.bind(_tldw_metadata_only_record=True).info(secret)` and bypass the
    schema entirely. `_forward_loguru_to_standard` rebuilds `extra` from
    scratch, which drops it -- this asserts that stays true.
    """
    from loguru import logger as loguru_logger

    from tldw_chatbook.Logging_Config import _forward_loguru_to_standard

    path, handler = sink
    loguru_logger.remove()
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="TRACE")
    try:
        loguru_logger.bind(_tldw_metadata_only_record=True).info(
            "event=forged component=attacker"
        )
    finally:
        loguru_logger.remove(sink_id)
    handler.flush()
    assert "forged" not in path.read_text()
```

- [ ] **Step 2: Run it to confirm it passes against current code**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_persist_event.py::test_a_loguru_record_carrying_the_marker_is_still_rejected --timeout=120`

Expected: PASS. This test pins existing correct behaviour rather than driving new code, so it passes immediately — that is intended.

- [ ] **Step 3: Mutation-check it**

Temporarily edit `tldw_chatbook/Logging_Config.py` in `_forward_loguru_to_standard`, changing

```python
    extra = {_SOURCE_PATH_FIELD: str(record["file"].path)}
```

to also carry the marker:

```python
    extra = {_SOURCE_PATH_FIELD: str(record["file"].path), "_tldw_metadata_only_record": True}
```

Run the test again. Expected: **FAIL** — this proves the test detects the bypass. Then revert the edit with `git checkout -- tldw_chatbook/Logging_Config.py` and re-run to confirm PASS.

- [ ] **Step 4: Commit**

```bash
git add Tests/Utils/test_persist_event.py
git commit -m "test: pin the closed Loguru path as a security property (TASK-1240)"
```

---

### Task 3: `persistent_sink_installed`, and making an empty log unambiguous

**Files:**
- Modify: `tldw_chatbook/Logging_Config.py:305-310`
- Test: `Tests/test_logging_private_files.py` (modify — add one test)

**Interfaces:**
- Consumes: `persist_event` from Task 1.
- Produces: the first line in a healthy log file. Later tasks rely on the log being non-empty from install onward.

- [ ] **Step 1: Write the failing test**

Append to `Tests/test_logging_private_files.py`:

```python
def test_successful_install_writes_its_own_first_event(tmp_path, monkeypatch):
    """TASK-1240: an empty log must mean "the sink did not install".

    `_configure_private_file_logging` catches Exception, warns, and returns
    False, so a permissions or path problem yields an empty file forever -- the
    same silent-failure class this task exists to fix. Emitting one event the
    moment the sink installs makes the two states distinguishable.
    """
    import logging as stdlib_logging

    from tldw_chatbook.Logging_Config import _configure_private_file_logging

    log_path = tmp_path / "tldw_cli_app.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: log_path
    )
    root_logger = stdlib_logging.getLogger("tldw_chatbook.test_install_event")
    root_logger.setLevel(stdlib_logging.INFO)

    assert _configure_private_file_logging(root_logger) is True
    for handler in root_logger.handlers:
        handler.flush()

    written = log_path.read_text()
    assert "event=persistent_sink_installed" in written
    assert "component=logging" in written
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/test_logging_private_files.py::test_successful_install_writes_its_own_first_event --timeout=120`

Expected: FAIL — the log contains no `event=persistent_sink_installed`.

- [ ] **Step 3: Emit the event after the handler is attached**

In `tldw_chatbook/Logging_Config.py`, replace lines 305-310:

```python
        root_logger.addHandler(file_handler)
        root_logger.info(
            "Private rotating file logging installed at level %s.",
            logging.getLevelName(file_log_level),
        )
        return True
```

with:

```python
        root_logger.addHandler(file_handler)
        root_logger.info(
            "Private rotating file logging installed at level %s.",
            logging.getLevelName(file_log_level),
        )
        # TASK-1240. Written the moment the sink is live, so an empty file means
        # "the sink did not install" rather than "nothing has happened yet".
        # This function swallows install failures (it warns and returns False),
        # so without this line those two states are indistinguishable.
        persist_event("logging", "persistent_sink_installed", status="ok")
        return True
```

- [ ] **Step 4: Add the import**

At the top of `tldw_chatbook/Logging_Config.py`, alongside the existing `from tldw_chatbook.Utils.persistent_diagnostics import (...)` block, add `persist_event` to the imported names.

- [ ] **Step 5: Run the test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/test_logging_private_files.py --timeout=300`

Expected: PASS, including all pre-existing tests in that file.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Logging_Config.py Tests/test_logging_private_files.py
git commit -m "feat: emit persistent_sink_installed so an empty log is unambiguous (TASK-1240)"
```

---

### Task 4: App lifecycle events

**Files:**
- Modify: `tldw_chatbook/app.py:6601` (`on_mount`), `tldw_chatbook/app.py:7601` (`on_unmount`)
- Test: `Tests/App/test_app_lifecycle_events.py` (create)

**Interfaces:**
- Consumes: `persist_event` from Task 1.
- Produces: `event=app_started` and `event=app_stopping` in the log.

- [ ] **Step 1: Write the failing test**

Create `Tests/App/test_app_lifecycle_events.py`:

```python
"""TASK-1240: the app records that it started and that it stopped cleanly."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_mounting_the_app_records_app_started(monkeypatch):
    """Boot the real app and assert the event fired.

    Asserted by capturing the call rather than scanning source: this repo has
    been burned by name-matching guards that pass against unwired code.
    """
    from Tests.UI.test_screen_navigation import _build_test_app

    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append((component, event)),
    )

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

    assert ("app", "app_started") in recorded
    assert ("app", "app_stopping") in recorded
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/App/test_app_lifecycle_events.py --timeout=300`

Expected: FAIL — `AttributeError: module 'tldw_chatbook.app' has no attribute 'persist_event'`.

- [ ] **Step 3: Import the helper in `app.py`**

Add to the imports near the other `tldw_chatbook.Utils` imports:

```python
from tldw_chatbook.Utils.persistent_diagnostics import persist_event
```

- [ ] **Step 4: Emit `app_started`**

In `TldwCli.on_mount` (line 6601), immediately after `mount_start = time.perf_counter()`:

```python
        # TASK-1240. Anchors a session in the persistent log; its absence dates
        # a crash to before this point.
        persist_event("app", "app_started")
```

- [ ] **Step 5: Emit `app_stopping`**

In `TldwCli.on_unmount` (line 7601), immediately after `logging.info("--- App Unmounting ---")`:

```python
        # TASK-1240. Distinguishes a clean exit from a kill: a log whose last
        # line is app_started ended abruptly.
        persist_event("app", "app_stopping")
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/App/test_app_lifecycle_events.py --timeout=300`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/app.py Tests/App/test_app_lifecycle_events.py
git commit -m "feat: record app_started and app_stopping (TASK-1240)"
```

---

### Task 5: `worker_failed` from the one central hook

**Files:**
- Modify: `tldw_chatbook/app.py:7989` (`on_worker_state_changed`)
- Test: `Tests/App/test_worker_failure_event.py` (create)

**Interfaces:**
- Consumes: `persist_event` from Task 1, imported in Task 4.
- Produces: `event=worker_failed` carrying `operation` and `exception_type`.

There is deliberately **no** `worker_started` event. The app has 398 `run_worker` call sites and 118 `@work` decorators, and this hook fires on every transition, so a start event would emit a line per keystroke-triggered search and per timer tick — in the file, the terminal and the Logs screen at once. Failures are rare and diagnostic; starts are neither.

- [ ] **Step 1: Write the failing test**

Create `Tests/App/test_worker_failure_event.py`:

```python
"""TASK-1240: a worker that dies leaves a trace naming its exception type."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.worker import WorkerState

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_worker_error_records_worker_failed(monkeypatch):
    from textual.worker import Worker

    from Tests.UI.test_screen_navigation import _build_test_app

    recorded: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append(
            {"component": component, "event": event, **fields}
        ),
    )

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        recorded.clear()
        worker = MagicMock(spec=Worker)
        worker.name = "scheduler_worker"
        worker.group = "scheduling"
        worker.error = ValueError("boom")
        event = Worker.StateChanged(worker, WorkerState.ERROR)
        await app.on_worker_state_changed(event)
        await pilot.pause()

    failures = [r for r in recorded if r["event"] == "worker_failed"]
    assert failures, f"no worker_failed recorded, got {recorded}"
    assert failures[-1]["exception_type"] == "ValueError"
    assert failures[-1]["operation"] == "scheduler_worker"
    # The message must not travel: "boom" is caller-supplied text.
    assert "boom" not in str(failures[-1])


@pytest.mark.asyncio
async def test_successful_worker_records_nothing(monkeypatch):
    """Only failures persist. A start/success event per transition would emit a
    line per keystroke-triggered search across 500+ worker sites."""
    from textual.worker import Worker

    from Tests.UI.test_screen_navigation import _build_test_app

    recorded: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append({"event": event}),
    )

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        recorded.clear()
        worker = MagicMock(spec=Worker)
        worker.name = "some_worker"
        worker.group = "misc"
        worker.error = None
        await app.on_worker_state_changed(
            Worker.StateChanged(worker, WorkerState.SUCCESS)
        )
        await pilot.pause()

    assert not [r for r in recorded if r["event"] in {"worker_failed", "worker_started"}]
```

- [ ] **Step 2: Run the tests to verify the first fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/App/test_worker_failure_event.py --timeout=300`

Expected: `test_worker_error_records_worker_failed` FAILS with "no worker_failed recorded"; `test_successful_worker_records_nothing` passes trivially.

- [ ] **Step 3: Emit from the central hook**

In `TldwCli.on_worker_state_changed` (line 7989), immediately after the existing `self.loguru_logger.debug(...)` state-change line and before `handled = await self.worker_handler_registry.handle_event(event)`:

```python
        # TASK-1240. One hook already sees every worker transition, so failures
        # are recorded without touching any of the 398 run_worker call sites.
        # Only ERROR persists: a start or success event here would emit a line
        # per keystroke-triggered search and per timer tick.
        if event.state is WorkerState.ERROR:
            error = getattr(event.worker, "error", None)
            persist_event(
                "app",
                "worker_failed",
                level=logging.ERROR,
                operation=str(worker_name or "unknown"),
                exception_type=type(error).__name__ if error is not None else "unknown",
            )
```

- [ ] **Step 4: Add the `WorkerState` import**

`app.py` already imports from `textual.worker`. Ensure `WorkerState` is among the imported names; if not, add it to that import statement.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/App/test_worker_failure_event.py --timeout=300`

Expected: PASS, 2 tests.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/app.py Tests/App/test_worker_failure_event.py
git commit -m "feat: record worker_failed from the central worker hook (TASK-1240)"
```

---

### Task 6: `unhandled_exception`

**Files:**
- Modify: `tldw_chatbook/app.py` (add `_handle_exception` override to `TldwCli`, class begins line 3234)
- Test: `Tests/App/test_unhandled_exception_event.py` (create)

**Interfaces:**
- Consumes: `persist_event` from Task 1, imported in Task 4.
- Produces: `event=unhandled_exception` carrying `exception_type`.

`TldwCli` does not currently override `_handle_exception`. Textual's implementation sets `self._return_code = 1` and re-raises for test frameworks, so the override **must** call `super()` and must not swallow.

- [ ] **Step 1: Write the failing test**

Create `Tests/App/test_unhandled_exception_event.py`:

```python
"""TASK-1240: a crash names its exception type in the persistent log."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_unhandled_exception_is_recorded(monkeypatch):
    from Tests.UI.test_screen_navigation import _build_test_app

    recorded: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append(
            {"component": component, "event": event, **fields}
        ),
    )

    app = _build_test_app()
    try:
        app._handle_exception(RuntimeError("secret detail"))
    except Exception:
        # Textual's implementation re-raises; that behaviour must be preserved.
        pass

    crashes = [r for r in recorded if r["event"] == "unhandled_exception"]
    assert crashes, f"no unhandled_exception recorded, got {recorded}"
    assert crashes[-1]["exception_type"] == "RuntimeError"
    assert "secret detail" not in str(crashes[-1])


def test_the_override_still_delegates_to_textual():
    """Must not swallow: Textual sets the return code and re-raises for tests."""
    from Tests.UI.test_screen_navigation import _build_test_app

    app = _build_test_app()
    try:
        app._handle_exception(RuntimeError("boom"))
    except Exception:
        pass
    assert app._return_code == 1
```

- [ ] **Step 2: Run the tests to verify the first fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/App/test_unhandled_exception_event.py --timeout=300`

Expected: `test_unhandled_exception_is_recorded` FAILS with "no unhandled_exception recorded".

- [ ] **Step 3: Add the override**

Add to the `TldwCli` class, immediately before `async def on_unmount` (line 7601):

```python
    def _handle_exception(self, error: Exception) -> None:
        """Record the crash type, then let Textual do what it always did.

        TASK-1240. Names the exception class only -- never the message, which is
        caller-supplied text and may quote user or model content. Calls super()
        unconditionally: Textual sets the return code here and re-raises for
        test frameworks, and swallowing that would turn a crash into a hang.
        """
        try:
            persist_event(
                "app",
                "unhandled_exception",
                level=logging.ERROR,
                exception_type=type(error).__name__,
            )
        except Exception:
            # Diagnostics must never be the reason a crash handler fails.
            pass
        super()._handle_exception(error)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/App/test_unhandled_exception_event.py --timeout=300`

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/app.py Tests/App/test_unhandled_exception_event.py
git commit -m "feat: record unhandled_exception without its message (TASK-1240)"
```

---

### Task 7: `scheduler_configured`

**Files:**
- Modify: `tldw_chatbook/Scheduling/scheduler/loop.py:44` (`report_configuration`)
- Test: `Tests/Scheduling/test_scheduler_observability.py` (modify — add one test)

**Interfaces:**
- Consumes: `persist_event` from Task 1.
- Produces: `event=scheduler_configured` carrying `item_count` and `status`.

This is the TASK-1210 case made durable: the scheduler already reports its wiring to the live log, and this puts the same fact on disk.

- [ ] **Step 1: Write the failing test**

Append to `Tests/Scheduling/test_scheduler_observability.py`:

```python
@pytest.mark.asyncio
async def test_configuration_is_recorded_in_the_persistent_log(monkeypatch):
    """TASK-1240: the wiring that TASK-1210 needed an import trace to discover
    is now one line on disk."""
    from tldw_chatbook.Scheduling.scheduler import loop as loop_module

    recorded: list[dict] = []
    monkeypatch.setattr(
        loop_module,
        "persist_event",
        lambda component, event, **fields: recorded.append(
            {"component": component, "event": event, **fields}
        ),
    )

    loop = SchedulerLoop(
        _tasks_db(), handlers={"reminder": AsyncMock()}, poll_interval=0
    )
    loop.queue.push(_due_watchlist_task())
    loop.report_configuration()

    events = [r for r in recorded if r["event"] == "scheduler_configured"]
    assert events, f"no scheduler_configured recorded, got {recorded}"
    assert events[-1]["component"] == "scheduling"
    assert events[-1]["item_count"] == 1
    assert events[-1]["status"] == "unhandled_types"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Scheduling/test_scheduler_observability.py --timeout=300`

Expected: FAIL — `AttributeError: module ... has no attribute 'persist_event'`.

- [ ] **Step 3: Import the helper**

At the top of `tldw_chatbook/Scheduling/scheduler/loop.py`, after the existing `from tldw_chatbook.Metrics.metrics_logger import log_counter` line:

```python
from tldw_chatbook.Utils.persistent_diagnostics import persist_event
```

- [ ] **Step 4: Emit at the end of `report_configuration`**

At the end of `report_configuration`, after the `if orphaned:` block:

```python
        # TASK-1240. The same fact the log line above states, put on disk:
        # discovering that watchlist checks never ran (TASK-1210) took a runtime
        # import trace and a seeded database probe, and should have taken this.
        persist_event(
            "scheduling",
            "scheduler_configured",
            item_count=len(registered),
            status="unhandled_types" if orphaned else "ok",
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Scheduling/ --timeout=600`

Expected: PASS, including all pre-existing scheduler tests.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Scheduling/scheduler/loop.py Tests/Scheduling/test_scheduler_observability.py
git commit -m "feat: record scheduler_configured on disk (TASK-1240)"
```

---

### Task 8: The end-to-end guard, and it must not be satisfiable by the install line

**Files:**
- Test: `Tests/test_persistent_log_is_not_empty.py` (create)

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: the regression guard for TASK-1240 itself.

The existing tests all pass today against a zero-byte log, because they assert a handler is attached rather than that anything was written. This is the test that would have caught the original defect — and it must assert **named events**, because `persistent_sink_installed` alone would satisfy a bare "non-empty" check even with everything else broken.

- [ ] **Step 1: Write the failing test**

Create `Tests/test_persistent_log_is_not_empty.py`:

```python
"""TASK-1240: the persistent log must actually contain records.

`tldw_cli_app.log` was zero bytes on every profile from 1df0c4cb4 onward,
because `PersistentDiagnosticFilter` admits only records marked by
`log_persistent_metadata()` and that function had no production callers. Every
existing test passed throughout: they assert a handler is attached, which was
true the whole time.
"""

from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.unit


def test_a_booted_app_writes_named_events_to_the_persistent_log(
    tmp_path, monkeypatch
):
    """Install the real sink, run the real emitters, read the real file."""
    from tldw_chatbook.Logging_Config import _configure_private_file_logging
    from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
    from tldw_chatbook.Utils.persistent_diagnostics import persist_event

    log_path = tmp_path / "tldw_cli_app.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: log_path
    )
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    try:
        assert _configure_private_file_logging(root) is True
        persist_event("app", "app_started")
        for handler in root.handlers:
            handler.flush()
        written = log_path.read_text()
    finally:
        root.setLevel(previous_level)

    assert written, "the persistent log is empty after a real install"

    # Non-empty alone is not enough: `persistent_sink_installed` is written the
    # moment the sink installs, so a bare emptiness check would pass with every
    # other event broken.
    assert "event=persistent_sink_installed" in written
    assert "event=app_started" in written

    events = {
        line.split("event=", 1)[1].split(" ", 1)[0]
        for line in written.splitlines()
        if "event=" in line
    }
    assert events - {"persistent_sink_installed"}, (
        "the only event in the log is the sink's own install line"
    )


def test_no_message_text_reaches_the_persistent_log(tmp_path, monkeypatch):
    """The boundary this task must not widen."""
    from tldw_chatbook.Logging_Config import _configure_private_file_logging
    from tldw_chatbook.Utils.persistent_diagnostics import persist_event

    log_path = tmp_path / "tldw_cli_app.log"
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path", lambda: log_path
    )
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    try:
        assert _configure_private_file_logging(root) is True
        logging.getLogger("tldw_chatbook.someplace").info(
            "a user's prompt: PRIVATE-SENTINEL-VALUE"
        )
        persist_event("app", "app_started")
        for handler in root.handlers:
            handler.flush()
        written = log_path.read_text()
    finally:
        root.setLevel(previous_level)

    assert "PRIVATE-SENTINEL-VALUE" not in written
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/test_persistent_log_is_not_empty.py --timeout=300`

Expected: PASS, 2 tests.

- [ ] **Step 3: Mutation-check the guard**

Temporarily revert Task 3's emit by commenting out the `persist_event("logging", "persistent_sink_installed", status="ok")` line in `Logging_Config.py`, and run the guard again.

Expected: `test_a_booted_app_writes_named_events_to_the_persistent_log` **FAILS** on the missing `event=persistent_sink_installed` assertion. Restore the line with `git checkout -- tldw_chatbook/Logging_Config.py` and confirm PASS.

- [ ] **Step 4: Commit**

```bash
git add Tests/test_persistent_log_is_not_empty.py
git commit -m "test: guard that the persistent log is not empty (TASK-1240)"
```

---

### Task 9: Close the task and record the ADR-029 amendment for sign-off

**Files:**
- Modify: `backlog/tasks/task-1240 - A-fresh-profile-writes-a-zero-byte-app-log.md`
- Modify: `backlog/decisions/029-local-private-data-boundary.md`

**Interfaces:**
- Consumes: the completed Tasks 1-8.
- Produces: nothing code-side.

The ADR amendment is written here but **must not be merged without the privacy work's owner signing off** — this branch adds one admitted field and six admitted events to a boundary that work owns.

- [ ] **Step 1: Run the full affected suite with plain output**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Utils/ Tests/Scheduling/ Tests/App/ \
  Tests/test_persistent_log_is_not_empty.py \
  Tests/test_persistent_diagnostic_boundary.py \
  Tests/test_logging_private_files.py \
  Tests/test_persistent_diagnostic_sentinel_matrix.py \
  Tests/test_remaining_diagnostic_sentinel_matrix.py \
  --timeout=1200
```

Do **not** add `-q`: it suppresses the failure summary in this repo and has hidden real failures three times. Compare the failure set against the same command on a clean `origin/dev` worktree; it must be identical.

- [ ] **Step 2: Amend ADR-029**

Add to `backlog/decisions/029-local-private-data-boundary.md`, after the diagnostics paragraph:

```markdown
## Amendment (2026-07-28, TASK-1240) — pending owner sign-off

"Metadata-only with respect to user and model content" is clarified to permit a
fixed set of operational events. Six are admitted: `app_started`, `app_stopping`,
`persistent_sink_installed`, `worker_failed`, `scheduler_configured`, and
`unhandled_exception`. They carry only fields from the existing schema plus
`component`, a code-side subsystem identifier.

The exclusion list is unchanged: no prompt, message body, provider request or
response payload, key fragment, tool argument value, or tool result value is
persistable. `exception_type` is a class name; exception messages remain excluded.

This restores the design's stated goal of keeping persistent diagnostics useful.
Before it, the sink admitted nothing at all, because `log_persistent_metadata()`
had no production callers and every ordinary log record was rejected.
```

- [ ] **Step 3: Close the task**

In `backlog/tasks/task-1240 - A-fresh-profile-writes-a-zero-byte-app-log.md`, set `status: In Progress`, tick every acceptance criterion, and add an `## Implementation Notes` section stating: what shipped (six events, `persist_event`, the `component` field), that the Loguru path stays deliberately closed and is now pinned by a mutation-checked test, and that the guard asserts named events because `persistent_sink_installed` alone would satisfy a bare non-empty check.

Leave the status at `In Progress` rather than `Done` until the ADR sign-off lands.

- [ ] **Step 4: Commit**

```bash
git add backlog/
git commit -m "docs: amend ADR-029 for operational events, pending owner sign-off (TASK-1240)"
```

---

## Self-review

**Spec coverage.** §1 `persist_event` → Task 1; the closed Loguru path → Task 2; §2 `component` → Task 1; §3's six events → Tasks 3-7 (`persistent_sink_installed`, `app_started`/`app_stopping`, `worker_failed`, `unhandled_exception`, `scheduler_configured`); §4 install-failure surfacing → Task 3; Testing → Tasks 1-8; Governance → Task 9. The spec's "no `worker_started`" rationale is carried in Task 5 and asserted by `test_successful_worker_records_nothing`.

**Placeholders.** None: every code step carries the literal text to insert, and every test step names the command and the expected result.

**Type consistency.** `persist_event(component, event, *, level, **fields)` is defined in Task 1 and called with that exact signature in Tasks 3-7. Field names used (`item_count`, `status`, `operation`, `exception_type`) all exist in `_ALLOWED_FIELDS`; `component` is added in Task 1 before any caller uses it. `WorkerState` and `Worker.error` were verified against the installed Textual, as was `App._handle_exception(self, error: Exception) -> None`.
