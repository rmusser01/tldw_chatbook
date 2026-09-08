"""The real app collector preserves diagnostics across the view and copy paths.

TASK-32047 amends ADR-029 to mask credentials and recognizable PII while keeping
ordinary diagnostic messages. Test the collector the app actually installs,
including bounded storage, both live-view feeds and the clipboard actions.
"""

from __future__ import annotations

import logging
from collections import deque

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.UI.Logs_Window import LogRecord, LogsWindow, MAX_LOG_RECORDS
from tldw_chatbook.Utils.persistent_diagnostics import log_persistent_metadata

pytestmark = pytest.mark.unit


def _synthetic(*parts: str) -> str:
    """Assemble a synthetic credential at import time from fragments.

    No committed line here may contain a contiguous string in a real token
    shape (TASK-19555 Qodo round, rule 497144): secret scanners match the
    literal, and GitHub push protection rejects the whole branch rather than
    the file. Splitting the detector-bearing prefix is what makes the fixture
    shippable; the assembled value is byte-identical to what the redactor is
    asked to handle.

    Args:
        *parts: Fragments to join, split at the detector prefix.

    Returns:
        The joined synthetic credential.
    """
    return "".join(parts)


API_KEY_SENTINEL = _synthetic("sk", "-19555PRIVATEsentinelKEYnotreal01")
CONTENT_SENTINEL = "19555-PRIVATE-NOTE-TITLE-divorce-papers"


class _AppStub:
    """Bare object for binding ``TldwCli._setup_buffered_logging``."""


class _Collector:
    """Install the real app collector on the root logger, then tear it down."""

    def __enter__(self) -> _AppStub:
        from tldw_chatbook.Logging_Config import _forward_loguru_to_standard
        from tldw_chatbook.app import TldwCli

        self._sinks_before = set(loguru_logger._core.handlers)
        loguru_logger.add(
            _forward_loguru_to_standard,
            level="TRACE",
            diagnose=False,
            backtrace=True,
        )
        self._root = logging.getLogger()
        self._old_level = self._root.level
        self._root.setLevel(logging.DEBUG)
        self.stub = _AppStub()
        TldwCli._setup_buffered_logging(self.stub)
        return self.stub

    def __exit__(self, *exc_info) -> None:
        for sink_id in set(loguru_logger._core.handlers) - self._sinks_before:
            try:
                loguru_logger.remove(sink_id)
            except ValueError:
                pass
        handler = getattr(self.stub, "_persistent_log_handler", None)
        if handler is not None:
            self._root.removeHandler(handler)
        self._root.setLevel(self._old_level)


def _buffer_text(stub: _AppStub) -> str:
    """The exact payload ``LogsWindow._on_copy_all`` puts on the clipboard."""
    return "\n".join(stub._log_buffer)


def _records_text(stub: _AppStub) -> str:
    """Everything the in-app Logs view renders and can copy."""
    return "\n".join(message for _level, _name, message in stub._log_records)


# ---------------------------------------------------------------------------
# Credentials and user identity: refused everywhere on the in-app path.
# ---------------------------------------------------------------------------


def test_api_key_never_reaches_the_in_app_collector() -> None:
    """A key logged at INFO stays out of the buffer, the records, and the view."""
    with _Collector() as stub:
        loguru_logger.info("calling provider with api_key={}", API_KEY_SENTINEL)
        loguru_logger.error("Authorization: Bearer {}", API_KEY_SENTINEL)

        assert API_KEY_SENTINEL not in _buffer_text(stub)
        assert API_KEY_SENTINEL not in _records_text(stub)
        # The redaction is a substitution, not a drop: the records survive.
        assert "***REDACTED***" in _records_text(stub)


def test_home_directory_username_never_reaches_the_in_app_collector() -> None:
    """Paths keep their shape; the OS account name is not an identity leak."""
    with _Collector() as stub:
        loguru_logger.info("attachment saved to /Users/privateperson/Notes/x.pdf")
        loguru_logger.info("cache dir /home/privateperson/.cache/tldw")

        rendered = _records_text(stub)
        assert "privateperson" not in rendered
        assert "privateperson" not in _buffer_text(stub)
        # Still debuggable: the path below the home root is untouched.
        assert "~/Notes/x.pdf" in rendered
        assert "~/.cache/tldw" in rendered


# ---------------------------------------------------------------------------
# The share path: "Copy all" bulk-exports what the user has never read.
# ---------------------------------------------------------------------------


def test_copy_all_share_artifact_preserves_diagnostic_text() -> None:
    """Copying the session must retain the same useful text as the live view."""
    with _Collector() as stub:
        loguru_logger.info("Created note: {}", CONTENT_SENTINEL)

        # The viewer stays rich -- that is the point of the Logs screen.
        assert CONTENT_SENTINEL in _records_text(stub)
        assert CONTENT_SENTINEL in _buffer_text(stub)
        assert list(stub._log_buffer) == [row[2] for row in stub._log_records]


def test_copy_all_share_artifact_keeps_triage_metadata() -> None:
    """Redaction is not deletion: level, logger, and exception type survive."""
    with _Collector() as stub:
        try:
            raise TimeoutError(CONTENT_SENTINEL)
        except TimeoutError:
            logging.getLogger("tldw_chatbook.RAG_Search.demo").exception(
                "search failed for %s", CONTENT_SENTINEL
            )

        share = _buffer_text(stub)
        assert CONTENT_SENTINEL in share
        assert "tldw_chatbook.RAG_Search.demo" in share
        assert "ERROR" in share
        assert "TimeoutError:" in share
        assert "Traceback (most recent call last)" in share


def test_copy_all_keeps_failure_context_and_masks_credentials_and_pii() -> None:
    """The actual copy action must retain the error without its personal values."""
    copied = []

    class Clipboard:
        def copy_to_clipboard(self, text):
            copied.append(text)

        def notify(self, *_args, **_kwargs):
            pass

    with _Collector() as stub:
        logging.getLogger("tldw_chatbook.Chat.console_trace_service").warning(
            "trace validation failed: expected append; provider=llama_cpp "
            "model=qwen3.7-27b api_key=%s email=elise@example.test phase=trace_reservation",
            API_KEY_SENTINEL,
        )

        class Window:
            app_instance = stub
            app = Clipboard()

        LogsWindow._on_copy_all(Window())
        assert "trace validation failed: expected append" in copied[0]
        assert "provider=llama_cpp model=qwen3.7-27b" in copied[0]
        assert "phase=trace_reservation" in copied[0]
        assert API_KEY_SENTINEL not in copied[0]
        assert "elise@example.test" not in copied[0]


def test_copy_visible_describes_the_credential_and_pii_policy() -> None:
    """The clipboard confirmation agrees with the retained diagnostic text."""
    notifications: list[str] = []
    copied: list[str] = []

    class _App:
        def copy_to_clipboard(self, text: str) -> None:
            copied.append(text)

        def notify(self, message: str, **_kwargs: object) -> None:
            notifications.append(message)

    class _Window:
        app = _App()

        @staticmethod
        def _visible_records() -> list[LogRecord]:
            return [LogRecord("INFO", "test", "visible diagnostic")]

    LogsWindow._on_copy_visible(_Window())

    assert copied == ["visible diagnostic"]
    assert notifications == [
        "Copied 1 visible log lines. Recognized credentials and PII are masked."
    ]


def test_schema_validated_metadata_records_survive_the_share_artifact() -> None:
    """ADR-029 metadata events are admitted verbatim, exactly as to the file."""
    with _Collector() as stub:
        log_persistent_metadata(
            logging.getLogger("tldw_chatbook.diagnostics.app"),
            logging.INFO,
            "operation_complete",
            operation="rag_search",
            status="success",
            duration_ms=12,
        )

        share = _buffer_text(stub)
        assert "event=operation_complete" in share
        assert "operation=rag_search" in share
        assert "status=success" in share
        assert "duration_ms=12" in share


# ---------------------------------------------------------------------------
# The buffer itself.
# ---------------------------------------------------------------------------


def test_session_log_buffer_is_bounded() -> None:
    """An unbounded session buffer is a memory *and* a disclosure surface."""
    with _Collector() as stub:
        assert isinstance(stub._log_buffer, deque)
        assert stub._log_buffer.maxlen == MAX_LOG_RECORDS
        # Both in-app stores retain the same window, so "Copy all" cannot
        # export more history than the screen admits to keeping.
        assert stub._log_records.maxlen == MAX_LOG_RECORDS


def test_oversized_lines_are_truncated_before_they_are_stored() -> None:
    """The buffer bounds line COUNT; without this it did not bound line SIZE."""
    with _Collector() as stub:
        loguru_logger.info("body " + "chunk " * 20_000)

        stored = _records_text(stub)
        assert "truncated, " in stored
        assert len(max(stub._log_records, key=lambda r: len(r[2]))[2]) < 3_000


def test_a_credential_astride_the_truncation_boundary_reaches_no_surface() -> None:
    """The unit-level guarantee, re-proved through the real handler.

    Truncating before redaction sliced a straddling key into a fragment too
    short for any `_STANDALONE_CREDENTIALS` pattern, and the fragment then
    reached the live view and "Copy visible logs".

    A SWEEP, not an anecdote, and one whose alignment is MEASURED rather than
    assumed: the handler's formatter prepends a timestamp and a logger name,
    so a hand-computed padding length does not reliably land the key across
    the cap. Two earlier drafts of this test guessed and passed against the
    broken implementation. The prefix width is read off a probe record, then
    every straddle position is walked.
    """
    from tldw_chatbook.Utils.log_sanitizer import MAX_REDACTED_LINE_CHARS

    with _Collector() as stub:
        window = _FakeLogsWindow()
        stub._current_logs_window = window

        loguru_logger.info("PROBE")
        prefix_width = len(stub._log_records[-1][2]) - len("PROBE")
        assert 0 < prefix_width < MAX_REDACTED_LINE_CHARS

        # Anti-vacuity control. Without it the sweep below could assert
        # nothing at all -- an earlier draft used a `%s` template, which
        # loguru renders literally, so it logged no key and passed happily
        # against the broken implementation.
        loguru_logger.info("control {}", API_KEY_SENTINEL)
        assert "***REDACTED***" in stub._log_records[-1][2]

        # Walk the key's start position across the cap, so at least one
        # emission is cut through the middle of it.
        for start in range(
            MAX_REDACTED_LINE_CHARS - len(API_KEY_SENTINEL),
            MAX_REDACTED_LINE_CHARS + 1,
        ):
            padding = start - prefix_width - 1
            assert padding > 0
            # Braces, not %s: loguru formats with str.format, so a %s template
            # emits the template and silently drops the payload. An earlier
            # draft did that and the test passed while logging nothing.
            loguru_logger.info(
                "{} {} tail {}", "x" * padding, API_KEY_SENTINEL, "y " * 2_000
            )

        # ...and the sweep really did produce oversized lines to cut.
        assert any("truncated," in message for _l, _n, message in stub._log_records)

        surfaces = "\n".join(
            (
                _records_text(stub),
                _buffer_text(stub),
                "\n".join(message for _l, _n, message in window.calls),
            )
        )
        assert API_KEY_SENTINEL not in surfaces
        # No leading slice of it either -- the fragment is the whole point.
        assert API_KEY_SENTINEL[:12] not in surfaces


# ---------------------------------------------------------------------------
# The LIVE FEED. `emit` fills two stores and then hands the line to whichever
# on-screen surface is mounted. Pinning the stores does not pin the feed:
# a review mutation that passed the UNREDACTED line to `append_record` left
# the whole suite green, while putting live credentials into
# `LogsWindow._records` -- which is exactly what `_on_copy_visible` copies to
# the clipboard. The stores and the feed are separate seams; both need a pin.
# ---------------------------------------------------------------------------


class _FakeLogsWindow:
    """Stands in for the mounted `LogsWindow` the handler feeds live."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def append_record(self, level: str, name: str, message: str) -> None:
        self.calls.append((level, name, message))


class _FakeRichLog:
    """Stands in for the legacy `_current_log_widget` fallback."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, message: str) -> None:
        self.lines.append(message)


def test_live_logs_window_feed_receives_the_redacted_line() -> None:
    """What reaches the mounted widget is what `_on_copy_visible` copies."""
    with _Collector() as stub:
        window = _FakeLogsWindow()
        stub._current_logs_window = window
        loguru_logger.error(
            "provider call failed api_key={} at /Users/privateperson/x.pdf",
            API_KEY_SENTINEL,
        )

        assert window.calls, "the handler never fed the mounted window"
        fed = "\n".join(message for _level, _name, message in window.calls)
        assert API_KEY_SENTINEL not in fed
        assert "privateperson" not in fed
        assert "***REDACTED***" in fed


def test_legacy_rich_log_feed_receives_the_redacted_line() -> None:
    """The fallback branch of the same `emit` is a clipboard path too."""
    with _Collector() as stub:
        widget = _FakeRichLog()
        stub._current_logs_window = None
        stub._current_log_widget = widget
        loguru_logger.error("legacy path api_key={}", API_KEY_SENTINEL)

        assert widget.lines, "the handler never fed the legacy widget"
        assert API_KEY_SENTINEL not in "\n".join(widget.lines)
