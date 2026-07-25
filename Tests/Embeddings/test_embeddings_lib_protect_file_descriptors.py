"""task-634 round 3: protect_file_descriptors() must never close the real,
process-shared stdout/stderr file descriptors (1/2).

Root cause (live re-UAT, faulthandler all-threads dump): Textual's App.run()
wraps its ENTIRE message loop in
``with redirect_stdout(self._capture_stdout): with
redirect_stderr(self._capture_stderr): ...`` (textual/app.py ~3491-3492) --
this reassigns the GLOBAL, process-wide ``sys.stdout``/``sys.stderr`` to
non-fd-backed capture objects for the whole app session, on every thread.
When a background worker (e.g. the RAG Backfill worker loading a HuggingFace
embedding model via ``_HuggingFaceEmbedder``) enters
``protect_file_descriptors()``, ``sys.stdout.fileno()``/``sys.stderr.fileno()``
raise (the capture objects have no real fd), so the except-branch creates
``sys.stdout = os.fdopen(1, "w")`` / ``sys.stderr = os.fdopen(2, "w")`` --
brand-new ``TextIOWrapper`` objects that, by ``os.fdopen``'s default
``closefd=True``, OWN and will CLOSE the shared, real fd 1/2 when garbage
collected.

The function's own "close any temporary files we created" cleanup
(``if sys.stdout != original_stdout: sys.stdout.close()``) is DEAD CODE: it
runs immediately AFTER ``sys.stdout = original_stdout`` on the line above it,
so the comparison is always False. That reassignment is exactly what drops
the last reference to the temporary ``os.fdopen(1, "w")``/``os.fdopen(2,
"w")`` wrappers, and CPython's reference-counting GC finalizes them
IMMEDIATELY and synchronously right there -- closing the real fd 1/2 before
the dead check even has a chance to matter. This closes the SAME file
descriptors Textual's own output ``WriterThread`` writes to via
``sys.__stderr__`` (captured once, early, at driver init and never
re-resolved) -- the next compositor write raises ``OSError: [Errno 9] Bad
file descriptor`` inside ``WriterThread.run()``'s unguarded ``write()``/
``flush()`` calls, silently killing that daemon thread (Python's default
``threading.excepthook`` just logs to the now-also-broken stderr). The next
screen refresh's ``queue.put()`` (Textual's bounded 30-slot output queue,
nothing left to drain it) then blocks forever -- the exact
``queue.py:140 in put`` -> ``threading.Condition.wait()`` freeze reproduced
in every live re-UAT round on this bug.

Empirically verified directly (see task-634's Implementation Notes, Round 3)
with a throwaway script mimicking Textual's non-fd-backed capture streams:
calling the real ``protect_file_descriptors()`` under those conditions
closed the process's real fd 1 AND fd 2 (confirmed via both
``os.fstat()`` and ``os.write()`` failing with ``OSError(9, 'Bad file
descriptor')`` immediately afterward).

These tests exercise the REAL fd numbers 1/2 the function hardcodes (there's
no way to test the actual bug otherwise), but protect the test session's own
I/O: fd 1/2 are ``os.dup()``-backed up before the test and unconditionally
``os.dup2()``-restored in a ``finally``, regardless of whether the bug
reproduces.
"""
import os
import sys
from contextlib import contextmanager

import pytest

from tldw_chatbook.Embeddings.Embeddings_Lib import protect_file_descriptors


class _NonFdBackedStream:
    """Stand-in for Textual's App._capture_stdout/_capture_stderr: a plain
    object with no real file descriptor, matching what sys.stdout/sys.stderr
    actually are for the entire duration of App.run() (see
    textual/app.py's `with redirect_stdout(self._capture_stdout): with
    redirect_stderr(self._capture_stderr): await run_process_messages()`).
    """

    def __init__(self):
        self.written = []

    def write(self, text):
        self.written.append(text)

    def flush(self):
        pass

    # Deliberately no fileno().


@contextmanager
def _protected_real_stdio():
    """Back up the process's real fd 1/2 and unconditionally restore them
    afterward, regardless of what protect_file_descriptors() does to them --
    so this test suite's own output survives even if the bug under test
    reproduces."""
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def _fd_usable(fd: int) -> bool:
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False


def test_protect_file_descriptors_does_not_close_real_stdio_when_sys_streams_are_non_fd_backed(
    monkeypatch,
):
    """RED for task-634 round 3: simulate Textual's redirected, non-fd-backed
    sys.stdout/sys.stderr (the state they're in for the app's entire
    lifetime) and confirm protect_file_descriptors() leaves the real
    process fd 1/2 open and writable afterward."""
    monkeypatch.setattr(sys, "stdout", _NonFdBackedStream())
    monkeypatch.setattr(sys, "stderr", _NonFdBackedStream())

    with _protected_real_stdio():
        with protect_file_descriptors():
            # Confirm we actually hit the except-branch (fileno() raised) --
            # otherwise this test would trivially pass without exercising
            # anything.
            assert not isinstance(sys.stdout, _NonFdBackedStream)

        assert _fd_usable(1), (
            "protect_file_descriptors() closed the real fd 1 (stdout) -- "
            "this is exactly what kills Textual's WriterThread"
        )
        assert _fd_usable(2), (
            "protect_file_descriptors() closed the real fd 2 (stderr) -- "
            "this is exactly what kills Textual's WriterThread"
        )
        # A write must actually succeed, not just fstat() -- fstat() alone
        # doesn't prove the fd is still open for I/O on every platform.
        os.write(1, b"")
        os.write(2, b"")


def test_protect_file_descriptors_restores_original_sys_streams(monkeypatch):
    """The original (non-fd-backed) sys.stdout/sys.stderr objects must be
    back in place after the context manager exits, regardless of the
    fd-safety fix -- unrelated behavior this fix must not disturb."""
    fake_stdout = _NonFdBackedStream()
    fake_stderr = _NonFdBackedStream()
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    with _protected_real_stdio():
        with protect_file_descriptors():
            pass

        assert sys.stdout is fake_stdout
        assert sys.stderr is fake_stderr


def test_protect_file_descriptors_is_a_noop_when_streams_are_already_fd_backed(tmp_path):
    """When sys.stdout/sys.stderr already have valid, real file descriptors
    (the common non-Textual case, e.g. a plain script or pytest itself), the
    except-branch must never trigger and nothing should be touched."""
    with _protected_real_stdio():
        # pytest's own sys.stdout/stderr at collection time are typically
        # capture-wrapped too (its own capsys machinery) -- use a real,
        # honest-to-goodness fd-backed file to make this deterministic.
        real_file = open(tmp_path / "out.txt", "w")
        try:
            import tldw_chatbook.Embeddings.Embeddings_Lib as lib

            original_fdopen = os.fdopen
            fdopen_calls = []

            def _tracking_fdopen(*args, **kwargs):
                fdopen_calls.append((args, kwargs))
                return original_fdopen(*args, **kwargs)

            import unittest.mock as mock

            with mock.patch.object(os, "fdopen", side_effect=_tracking_fdopen):
                with mock.patch.object(sys, "stdout", real_file):
                    with mock.patch.object(sys, "stderr", real_file):
                        with lib.protect_file_descriptors():
                            pass

            assert fdopen_calls == [], (
                "protect_file_descriptors() replaced already-valid, "
                "fd-backed streams -- it should be a no-op in this case"
            )
        finally:
            real_file.close()
