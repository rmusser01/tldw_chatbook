"""PERF-03 (TASK-33262): a real level gate and one redaction per record.

The 2026-09-27 audit measured the shipped pipeline at ~7-8 us per *dropped*
``logger.debug`` (the loguru forwarder sat at TRACE, so loguru's early level
check never fired and ``opt(lazy=True)`` guards were evaluated anyway) and
~340 us per INFO record (redacted three times -- ``shouldRollover``, the file
``emit`` and the Logs buffer -- then written and flushed on the emitting
thread, the event loop included).

These tests pin the behaviour, not the timings: the forwarder follows the
stdlib threshold, each record reaches the sanitizer exactly once however many
redacting sinks it feeds, and a secret is still masked in every sink --
including across the truncation cap and in a logger name. Writes stay
synchronous on the emitting thread: an earlier off-thread writer was removed
in review (it saved ~80 us per INFO record, at the cost of losing the last
lines before a forced exit and of drain races at close and at a pause).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook import Logging_Config
from tldw_chatbook.Logging_Config import (
    LogsBufferHandler,
    PrivateRotatingFileHandler,
    _private_file_formatter,
    sync_loguru_forward_level,
)
from tldw_chatbook.Utils.log_sanitizer import (
    MAX_REDACTED_LINE_CHARS,
    REDACTION_MARKER,
)

pytestmark = pytest.mark.unit

#: Assembled at import so no committed line carries a contiguous token shape
#: (secret scanners match the literal; see Tests/test_logs_share_path_privacy).
SECRET = "".join(("sk", "-perf03SECRETsentinelKEYnotreal01"))


class AppStub:
    """The two stores and the live-feed slots ``LogsBufferHandler`` fills."""

    def __init__(self) -> None:
        self._log_buffer: deque[str] = deque(maxlen=500)
        self._log_records: deque[tuple[str, str, str]] = deque(maxlen=500)
        self._current_logs_window = None
        self._current_log_widget = None


class FakeLogsWindow:
    """Records what the live feed appends; seeded through no record yet."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._seeded_through = 0

    def append_record(self, level: str, name: str, message: str) -> None:
        self.calls.append((level, name, message))


@pytest.fixture
def redactions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every text handed to the sanitizer by the logging pipeline."""
    calls: list[str] = []
    real = Logging_Config.redact_log_line

    def counting(text, *args, **kwargs):
        calls.append(text)
        return real(text, *args, **kwargs)

    monkeypatch.setattr(Logging_Config, "redact_log_line", counting)
    return calls


@pytest.fixture
def perf_logger():
    """A private logger, so no test here touches the root logger's handlers."""
    logger = logging.Logger("tldw_chatbook.tests.perf03", level=logging.DEBUG)
    yield logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _file_handler(tmp_path: Path) -> PrivateRotatingFileHandler:
    # maxBytes > 0 is what made RotatingFileHandler.shouldRollover format
    # (and so redact) every record a second time.
    handler = PrivateRotatingFileHandler(
        tmp_path / "logs" / "app.log",
        maxBytes=10_000_000,
        backupCount=1,
        encoding="utf-8",
    )
    handler.setFormatter(_private_file_formatter())
    return handler


def _read(handler: PrivateRotatingFileHandler) -> str:
    handler.flush()
    return Path(handler.baseFilename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Single-pass redaction
# ---------------------------------------------------------------------------


def _warm(logger: logging.Logger, redactions: list[str]) -> None:
    """Make the file non-empty: 3.12's shouldRollover skips an empty file."""
    logger.info("warm-up record")
    redactions.clear()


def test_file_sink_redacts_each_record_once(tmp_path, perf_logger, redactions):
    """shouldRollover and emit used to format -- and redact -- the record twice."""
    handler = _file_handler(tmp_path)
    perf_logger.addHandler(handler)
    _warm(perf_logger, redactions)

    perf_logger.info("provider call api_key=%s failed", SECRET)

    written = _read(handler)
    assert SECRET not in written
    assert REDACTION_MARKER in written
    assert len(redactions) == 1, redactions


def test_file_and_logs_buffer_share_one_redaction(tmp_path, perf_logger, redactions):
    """Two redacting sinks, two formats, one sanitizer pass per record."""
    handler = _file_handler(tmp_path)
    app = AppStub()
    perf_logger.addHandler(handler)
    perf_logger.addHandler(LogsBufferHandler(app))
    _warm(perf_logger, redactions)

    perf_logger.info("provider call api_key=%s failed", SECRET)

    assert len(redactions) == 1, redactions
    written = _read(handler)
    buffered = "\n".join(app._log_buffer)
    for text in (written, buffered, app._log_records[-1][2]):
        assert SECRET not in text
        assert REDACTION_MARKER in text
    # The Logs screen parses this layout (Logs_Window._styled_line).
    _stamp, name, level, message = app._log_records[-1][2].split(" - ", 3)
    assert (name, level) == ("tldw_chatbook.tests.perf03", "INFO")
    assert message.startswith("provider call api_key=")
    assert list(app._log_buffer) == [row[2] for row in app._log_records]


def test_exception_text_is_redacted_once_in_both_sinks(
    tmp_path, perf_logger, redactions
):
    """The secret lives in ``str(exc)``: the traceback block must be covered."""
    handler = _file_handler(tmp_path)
    app = AppStub()
    perf_logger.addHandler(handler)
    perf_logger.addHandler(LogsBufferHandler(app))
    _warm(perf_logger, redactions)

    try:
        raise ValueError(f"connect failed token={SECRET}")
    except ValueError:
        perf_logger.exception("provider probe failed")

    assert len(redactions) == 1, redactions
    for text in (_read(handler), "\n".join(app._log_buffer)):
        assert SECRET not in text
        assert "ValueError" in text
        assert "Traceback (most recent call last)" in text


def test_secret_straddling_the_cap_reaches_no_sink(tmp_path, perf_logger):
    """Walk the secret across the truncation cap of the redacted body.

    Truncation now applies to the caller-controlled body (message, exception,
    stack), so that is where a straddling key has to be proven safe.
    """
    handler = _file_handler(tmp_path)
    app = AppStub()
    perf_logger.addHandler(handler)
    perf_logger.addHandler(LogsBufferHandler(app))

    for start in range(
        MAX_REDACTED_LINE_CHARS - len(SECRET), MAX_REDACTED_LINE_CHARS + 1
    ):
        perf_logger.info("%s %s tail %s", "x" * (start - 1), SECRET, "y " * 1500)

    surfaces = "\n".join((_read(handler), "\n".join(app._log_buffer)))
    assert "truncated, " in surfaces
    assert SECRET not in surfaces
    assert SECRET[:12] not in surfaces


# ---------------------------------------------------------------------------
# Logs buffer, logger names, forwarder hand-off
# ---------------------------------------------------------------------------


async def test_logs_buffer_stores_at_emit_and_displays_on_the_loop(redactions):
    """A worker-thread record is buffered at once; only the widget waits for the loop."""
    app = AppStub()
    window = FakeLogsWindow()
    app._current_logs_window = window
    handler = LogsBufferHandler(app)  # built on the running loop
    display_threads: list[str] = []
    real_display = handler._display

    def display_spy(*entry):
        display_threads.append(threading.current_thread().name)
        real_display(*entry)

    handler._display = display_spy
    logger = logging.Logger("tldw_chatbook.tests.perf03.buffer", level=logging.INFO)
    logger.addHandler(handler)
    Logging_Config._safe_logger_name(logger.name)  # names are sanitized once, cached
    redactions.clear()
    try:
        worker = threading.Thread(
            target=logger.error, args=("provider call api_key=%s failed", SECRET)
        )
        worker.start()
        worker.join(5)
        # Copy all sees the record before the loop has run a single callback.
        assert len(app._log_buffer) == 1
        assert not window.calls
        for _ in range(5):
            await asyncio.sleep(0)
    finally:
        logger.removeHandler(handler)

    assert display_threads == [threading.current_thread().name]
    assert len(redactions) == 1
    for text in (app._log_records[0][2], app._log_buffer[0], window.calls[0][2]):
        assert SECRET not in text
        assert REDACTION_MARKER in text


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("tldw_chatbook.api_key=" + SECRET, id="secret"),
        pytest.param("tldw_chatbook.evil\nFORGED - CRITICAL - line", id="newline"),
    ],
)
def test_a_logger_name_is_sanitized_in_both_sinks(tmp_path, name):
    """Only the body is redacted per record, so the name needs its own pass."""
    logger = logging.Logger(name, level=logging.INFO)
    handler = _file_handler(tmp_path)
    app = AppStub()
    logger.addHandler(handler)
    logger.addHandler(LogsBufferHandler(app))
    try:
        logger.info("hello")
    finally:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

    written = Path(handler.baseFilename).read_text(encoding="utf-8")
    for text in (written, "\n".join(app._log_buffer), app._log_records[0][1]):
        assert SECRET not in text
        assert "\nFORGED" not in text
    assert len(written.splitlines()) == 1
    # Other handlers keep the original record's name.
    assert logger.name == name


def test_a_record_both_forwarders_see_reaches_stdlib_once(stdlib_levels):
    """sync_loguru_forward_level adds the new sink before removing the old one."""
    root, _package = stdlib_levels
    root.setLevel(logging.INFO)
    seen: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    collector = _Collect()
    root.addHandler(collector)
    old = loguru_logger.add(Logging_Config._forward_loguru_to_standard, level="INFO")
    new = loguru_logger.add(Logging_Config._forward_loguru_to_standard, level="INFO")
    try:
        loguru_logger.info("during the swap")
        loguru_logger.info("during the swap")  # a second, distinct call
    finally:
        loguru_logger.remove(old)
        loguru_logger.remove(new)
        root.removeHandler(collector)

    assert seen == ["during the swap", "during the swap"]


async def test_every_sink_masks_a_forwarded_loguru_secret(tmp_path):
    """loguru -> stdlib forwarder -> file, Copy-all buffer, records, live view."""
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    handler = _file_handler(tmp_path)
    app = AppStub()
    window = FakeLogsWindow()
    app._current_logs_window = window
    buffer_handler = LogsBufferHandler(app)
    root.addHandler(handler)
    root.addHandler(buffer_handler)
    sink_id = loguru_logger.add(
        Logging_Config._forward_loguru_to_standard,
        level="INFO",
        diagnose=False,
        backtrace=True,
    )
    try:
        loguru_logger.error(
            "provider call failed api_key={} at /Users/privateperson/x.pdf", SECRET
        )
        for _ in range(5):
            await asyncio.sleep(0)
    finally:
        loguru_logger.remove(sink_id)
        root.removeHandler(buffer_handler)
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()

    surfaces = {
        "file": Path(handler.baseFilename).read_text(encoding="utf-8"),
        "copy-all buffer": "\n".join(app._log_buffer),
        "records": "\n".join(row[2] for row in app._log_records),
        "live view": "\n".join(call[2] for call in window.calls),
    }
    for name, text in surfaces.items():
        assert "provider call failed" in text, name
        assert SECRET not in text, name
        assert "privateperson" not in text, name
        assert REDACTION_MARKER in text, name


# ---------------------------------------------------------------------------
# The level gate
# ---------------------------------------------------------------------------


@pytest.fixture
def stdlib_levels(monkeypatch: pytest.MonkeyPatch):
    root = logging.getLogger()
    package = logging.getLogger("tldw_chatbook")
    saved = (root.level, package.level)
    monkeypatch.setattr(Logging_Config, "_loguru_forward_sink", None)
    yield root, package
    sink = Logging_Config._loguru_forward_sink
    if sink is not None:
        try:
            loguru_logger.remove(sink[0])
        except ValueError:
            pass
    root.setLevel(saved[0])
    package.setLevel(saved[1])


def _install_forwarder() -> int:
    """Install the forwarder the way configure_application_logging records it."""
    level = Logging_Config._loguru_forward_level()
    sink_id = loguru_logger.add(
        Logging_Config._forward_loguru_to_standard,
        level=level,
        diagnose=False,
        backtrace=True,
    )
    Logging_Config._loguru_forward_sink = (sink_id, level)
    return sink_id


def _forwarder_level() -> int:
    sink_id, _level = Logging_Config._loguru_forward_sink
    return loguru_logger._core.handlers[sink_id].levelno


def test_forwarder_level_follows_the_stdlib_threshold(stdlib_levels):
    root, package = stdlib_levels
    root.setLevel(logging.INFO)
    package.setLevel(logging.NOTSET)
    _install_forwarder()
    assert _forwarder_level() == logging.INFO

    # TRACE is forwarded as DEBUG, so a DEBUG threshold must keep TRACE.
    root.setLevel(logging.DEBUG)
    sync_loguru_forward_level()
    assert _forwarder_level() == 0

    # app.py's "Disable debug logging for performance" now reaches loguru.
    root.setLevel(logging.INFO)
    package.setLevel(logging.INFO)
    sync_loguru_forward_level()
    assert _forwarder_level() == logging.INFO

    root.setLevel(logging.WARNING)
    package.setLevel(logging.NOTSET)
    sync_loguru_forward_level()
    assert _forwarder_level() == logging.WARNING
    assert len(
        [
            handler
            for handler in loguru_logger._core.handlers.values()
            if "_forward_loguru_to_standard" in str(handler)
        ]
    ) == 1, "re-levelling must replace the forwarder, not add a second one"


def test_sync_never_resurrects_a_removed_forwarder(stdlib_levels):
    root, _package = stdlib_levels
    root.setLevel(logging.INFO)
    sink_id = _install_forwarder()
    loguru_logger.remove(sink_id)
    before = set(loguru_logger._core.handlers)

    root.setLevel(logging.DEBUG)
    sync_loguru_forward_level()

    assert set(loguru_logger._core.handlers) == before
    assert Logging_Config._loguru_forward_sink is None


_PRODUCTION_WIRING_SCRIPT = """
import json, logging, sys, time
from loguru import logger
from textual.css.query import QueryError
import tldw_chatbook.Logging_Config as lc

class EarlyApp:
    app_config = {"general": {"log_level": "INFO"}}
    _rich_log_handler = None
    def query_one(self, *args, **kwargs):
        raise QueryError("no UI during early logging")

lc.configure_application_logging(EarlyApp())
evaluated = []
logger.opt(lazy=True).debug("lazy {}", lambda: evaluated.append(1) or "x")
started = time.perf_counter()
for _ in range(20000):
    logger.debug("dropped")
dropped_us = (time.perf_counter() - started) / 20000 * 1e6
logger.info("provider call api_key={} failed", sys.argv[1])
for handler in logging.getLogger().handlers:
    handler.flush()
print("RESULT " + json.dumps({
    "lazy_evaluated": len(evaluated),
    "dropped_debug_us": dropped_us,
    "log": lc.get_cli_log_file_path().read_text(encoding="utf-8"),
}))
"""


def test_production_wiring_drops_debug_before_evaluating_lazy_arguments(tmp_path):
    """The real ``configure_application_logging``, in a fresh sandboxed process.

    Out-of-process because production removes every loguru sink first: in this
    session other sinks sit at DEBUG, and loguru evaluates lazy arguments
    against the minimum level across all sinks.
    """
    home = tmp_path / "home"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LOGURU_", "TLDW_", "PYTEST_", "XDG_"))
    }
    environment.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        XDG_CACHE_HOME=str(home / ".cache"),
        TLDW_CONFIG_PATH=str(home / ".config" / "tldw_cli" / "config.toml"),
        TLDW_TEST_MODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
    )
    completed = subprocess.run(
        [sys.executable, "-c", _PRODUCTION_WIRING_SCRIPT, SECRET],
        capture_output=True,
        check=False,
        text=True,
        env=environment,
        cwd=tmp_path,
        timeout=120,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, completed.stdout + completed.stderr
    result = json.loads(lines[-1][len("RESULT ") :])

    assert result["lazy_evaluated"] == 0
    # ~0.15 us measured with the gate, ~7-8 us without; 2 us leaves CI headroom.
    assert result["dropped_debug_us"] < 2.0, result["dropped_debug_us"]
    assert "provider call api_key=" in result["log"]
    assert SECRET not in result["log"]
    assert REDACTION_MARKER in result["log"]


# ---------------------------------------------------------------------------
# Volume: per-row INFO, worker transitions, @timeit
# ---------------------------------------------------------------------------


@pytest.fixture
def info_messages():
    """Messages loguru emits at INFO or above while the fixture is active."""
    seen: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: seen.append(message.record["message"]), level="INFO"
    )
    yield seen
    loguru_logger.remove(sink_id)


def test_chachanotes_row_writes_log_below_info(tmp_path, info_messages):
    """Bulk imports wrote one INFO (~0.34 ms, ~200 B of log) per row."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(tmp_path / "chachanotes.db", client_id="perf03")
    try:
        info_messages.clear()
        conversation_id = database.add_conversation({"title": "perf03"})
        database.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "hello",
            }
        )
        note_id = database.add_note("perf03 note", "body")
        database.update_note(note_id, {"content": "body 2"}, expected_version=1)
    finally:
        database.close_connection()

    per_row = [
        message
        for message in info_messages
        if message.startswith(("Added ", "Updated "))
    ]
    assert not per_row, per_row


def test_unhandled_app_worker_transitions_do_not_warn(info_messages):
    """TASK-31806 by construction: only a failure nobody handles warns."""
    from unittest.mock import MagicMock

    from textual.worker import Worker, WorkerState

    from tldw_chatbook.Event_Handlers.worker_handlers.base_handler import (
        WorkerHandlerRegistry,
    )

    registry = WorkerHandlerRegistry(app=MagicMock())
    worker = MagicMock(spec=Worker)
    worker.name = "fire_and_forget"
    worker.group = "perf03-unregistered-group"

    for state in (WorkerState.PENDING, WorkerState.RUNNING, WorkerState.SUCCESS):
        handled = asyncio.run(
            registry.handle_event(Worker.StateChanged(worker, state))
        )
        assert handled is False
    assert not [m for m in info_messages if "No handler found" in m], info_messages

    asyncio.run(registry.handle_event(Worker.StateChanged(worker, WorkerState.ERROR)))
    warned = [m for m in info_messages if "No handler found" in m]
    assert len(warned) == 1 and "fire_and_forget" in warned[0], info_messages


def test_timeit_times_the_awaited_coroutine(monkeypatch, info_messages):
    """@timeit on an async def used to time coroutine *creation* (~0 s)."""
    import inspect

    from tldw_chatbook.Metrics import metrics_logger

    recorded: list[tuple[str, float]] = []
    monkeypatch.setattr(
        metrics_logger,
        "_log_metric",
        lambda name, kind, value, labels=None: recorded.append((name, value)),
    )

    @metrics_logger.timeit("perf03_async")
    async def slow(value):
        await asyncio.sleep(0.05)
        return value

    @metrics_logger.timeit("perf03_sync")
    def fast(value):
        return value

    assert inspect.iscoroutinefunction(slow)
    assert asyncio.run(slow(7)) == 7
    assert fast(3) == 3
    timings = dict(recorded)
    assert timings["perf03_async"] >= 0.04, timings
    assert "perf03_sync" in timings
    assert not [m for m in info_messages if "finished in" in m], info_messages



def test_timeit_labels_a_cancelled_coroutine_cancelled(monkeypatch):
    """A cancelled task is not a success (Qodo, #2904); cancellation still propagates."""
    from tldw_chatbook.Metrics import metrics_logger

    statuses: list[str] = []
    monkeypatch.setattr(
        metrics_logger,
        "_log_metric",
        lambda name, kind, value, labels=None: statuses.append((labels or {}).get("status")),
    )

    @metrics_logger.timeit("perf03_cancelled")
    async def waits_forever():
        await asyncio.Event().wait()

    async def scenario():
        task = asyncio.create_task(waits_forever())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert "cancelled" in statuses and "success" not in statuses, statuses


class FakeLoop:
    """Collects callbacks the handler defers to "the app loop"."""

    def __init__(self) -> None:
        self.callbacks: list[tuple[Callable[..., object], tuple[object, ...]]] = []

    def is_closed(self) -> bool:
        """Report an open loop, so the handler defers instead of dropping.

        Returns:
            Always False.
        """
        return False

    def call_soon_threadsafe(self, callback: Callable[..., object], *args: object) -> None:
        """Queue ``callback(*args)`` for the test to run later.

        Args:
            callback: What the handler wants run on the loop.
            *args: Its arguments.
        """
        self.callbacks.append((callback, args))


def test_a_record_seeded_before_its_deferred_display_is_shown_once():
    """A worker record is buffered at once but displayed later; if the Logs
    window seeds from the buffer in between, the deferred display must not
    add it again (Qodo, #2904)."""
    app = AppStub()
    handler = Logging_Config.LogsBufferHandler(app)
    loop = FakeLoop()
    handler._loop = loop  # emit "from a worker thread": display is deferred

    record = logging.LogRecord("tldw.worker", logging.INFO, __file__, 1, "from a worker", None, None)
    handler.emit(record)
    assert len(app._log_records) == 1 and loop.callbacks

    window = FakeLogsWindow()  # the Logs screen mounts and seeds, as load_from_app does
    with app._log_records_lock:
        window._seeded_through = app._log_records_seq
    app._current_logs_window = window
    for callback, args in loop.callbacks:
        callback(*args)
    assert window.calls == [], "the seeded record was displayed a second time"

    loop.callbacks.clear()
    handler.emit(logging.LogRecord("tldw.worker", logging.INFO, __file__, 2, "after mount", None, None))
    for callback, args in loop.callbacks:
        callback(*args)
    assert [call[2].endswith("after mount") for call in window.calls] == [True]
