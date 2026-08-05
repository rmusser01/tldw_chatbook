# test_recompose_capture_guard.py
# Description: Regression coverage for RecomposeCaptureGuard's exception logging.
"""
PR #905 review (finding 3): ``RecomposeCaptureGuard`` used to log capture
release/sweep failures with ``logger.debug(..., exc_info=True)``. Loguru does
not honor the stdlib ``exc_info`` kwarg -- it is bound as an opaque "extra"
field instead of triggering traceback formatting -- so the traceback was
silently dropped, defeating the whole point of logging it. The fix uses
loguru's own mechanism, ``logger.opt(exception=True)``.
"""

from __future__ import annotations

import io

import pytest
from loguru import logger

from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


@pytest.fixture
def loguru_sink():
    """Capture loguru output into an in-memory buffer for one test."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{message}")
    try:
        yield sink
    finally:
        logger.remove(handler_id)


class _FailingApp:
    """Minimal stand-in whose capture_mouse always raises, to hit the log line."""

    def capture_mouse(self, widget) -> None:
        raise RuntimeError("capture_mouse boom")


class _GuardHost(RecomposeCaptureGuard):
    """Bare host exercising the mixin's logging without any real Textual app."""

    def __init__(self) -> None:
        self.app = _FailingApp()


def test_release_own_capture_failure_logs_traceback_via_loguru_opt(loguru_sink) -> None:
    """``_release_own_capture_if_any`` must attach a real traceback on failure.

    Regression for the exc_info=True bug: before the fix this assertion is
    RED because loguru's stdlib-style ``exc_info=True`` kwarg is bound as an
    inert "extra" field rather than formatting a traceback, so the emitted
    line never contains ``RuntimeError`` or ``capture_mouse boom`` -- the
    failure is logged with its cause silently discarded.
    """
    host = _GuardHost()

    # _capture_is_within_self must see a captured widget that is "within"
    # self for the guarded capture_mouse(None) call (and its except branch)
    # to run at all.
    host._capture_is_within_self = lambda captured: True  # type: ignore[method-assign]

    host._release_own_capture_if_any(context="before recompose")

    output = loguru_sink.getvalue()
    assert "mouse-capture release before recompose skipped" in output
    assert "RuntimeError" in output
    assert "capture_mouse boom" in output
    assert "Traceback" in output
