"""TASK-1240: a crash names its exception type in the persistent log.

Every test here calls `_handle_exception` from **inside a live `except:` block**,
which is how production reaches it: textual calls it from `Worker._run`'s
`except Exception as error:` and from the message pump's own handler. That is not
cosmetic. `App._handle_exception` -> `_fatal_error()` builds a
`rich.traceback.Traceback()` with no arguments, which raises
``ValueError: Value for 'trace' required if not called in except: block`` when no
exception is being handled.

An earlier version of this file instead called `_handle_exception` bare and wrapped
it in `try/except Exception: pass`, commented "Textual's implementation re-raises;
that behaviour must be preserved." That comment is wrong -- textual's
`_handle_exception` sets `_return_code`, stores the exception on `self._exception`
for the pilot harness to re-raise *later*, and returns normally. The swallow was
real rather than dead code, though: it was absorbing rich's `ValueError` above,
which means the tests were exercising a path production never takes and would have
silently tolerated the override raising anything at all. Raising for real, inside
an `except` block, removes both problems.
"""

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
        raise RuntimeError("secret detail")
    except RuntimeError as error:
        app._handle_exception(error)

    crashes = [r for r in recorded if r["event"] == "unhandled_exception"]
    assert crashes, f"no unhandled_exception recorded, got {recorded}"
    assert crashes[-1]["exception_type"] == "RuntimeError"
    assert "secret detail" not in str(crashes[-1])


def test_worker_failed_wrapper_is_unwrapped(monkeypatch):
    """The type recorded must be the worker's, not textual's wrapper.

    When a worker raises and `exit_on_error` is true -- the default --
    `Worker._run` sets `WorkerState.ERROR` (posting `StateChanged`
    *asynchronously*) and then calls `app._handle_exception(WorkerFailed(...))`
    *synchronously*, from inside its own `except` block. So this override fires
    first and, without unwrapping, would persist `exception_type=WorkerFailed`
    for every worker crash in the app -- identical for all of them -- while
    `_fatal_error()` -> `_close_messages_no_wait()` races the queued
    `StateChanged`, so the `worker_failed` event carrying the real type and
    `operation` may never be delivered at all. A crashed session's log could
    then read `event=unhandled_exception exception_type=WorkerFailed` and
    nothing else.

    A hand-built `RuntimeError` (as the sibling above uses) cannot see this: it
    is never wrapped, which is why this was invisible.
    """
    from textual.worker import WorkerFailed

    from Tests.UI.test_screen_navigation import _build_test_app

    class DistinctiveWorkerError(RuntimeError):
        pass

    recorded: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append(
            {"component": component, "event": event, **fields}
        ),
    )

    app = _build_test_app()
    # Mirrors textual/worker.py's `except Exception as error:` block exactly.
    try:
        raise DistinctiveWorkerError("secret detail")
    except DistinctiveWorkerError as error:
        app._handle_exception(WorkerFailed(error))

    crashes = [r for r in recorded if r["event"] == "unhandled_exception"]
    assert crashes, f"no unhandled_exception recorded, got {recorded}"
    assert crashes[-1]["exception_type"] == "DistinctiveWorkerError", (
        "the useless wrapper type was persisted instead of the real one"
    )
    # `WorkerFailed.__init__` builds its message as
    # f"Worker raised exception: {error!r}", so the wrapper's own text quotes
    # the underlying message -- nothing of it may travel.
    assert "secret detail" not in str(crashes[-1])


def test_the_override_still_delegates_to_textual(monkeypatch):
    """Must not swallow: textual sets the return code from this call."""
    from Tests.UI.test_screen_navigation import _build_test_app

    # Patched for symmetry with the siblings above. Unpatched, this test ran
    # the real `persist_event` against whatever sinks the session happened to
    # have installed, which is not what it is testing.
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event", lambda *args, **kwargs: None
    )

    app = _build_test_app()
    try:
        raise RuntimeError("boom")
    except RuntimeError as error:
        app._handle_exception(error)
    assert app._return_code == 1
