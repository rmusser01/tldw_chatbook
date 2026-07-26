"""task-640: consolidated tests for tldw_chatbook.Utils.fd_protection.

Supersedes the old Tests/Embeddings/test_embeddings_lib_protect_file_
descriptors.py (task-641 round 3) now that protect_file_descriptors() has
been consolidated (task-640 item 5) out of Embeddings/Embeddings_Lib.py,
Local_Ingestion/transcription_service.py, and TTS/backends/higgs.py into
this one shared module -- all four now import the SAME function object (see
``test_all_call_sites_share_the_same_function_object`` below).

Root cause of the underlying bug (kept from the superseded file; still the
reason these tests exercise the REAL fd numbers 1/2 the function hardcodes):
Textual's App.run() wraps its ENTIRE message loop in
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
collected. ``closefd=False`` is the fix; see the function's own docstring
for the full root-cause writeup.

These tests protect the test session's own I/O: fd 1/2 are ``os.dup()``-
backed up before each test and unconditionally ``os.dup2()``-restored in a
``finally``, regardless of whether the bug reproduces.
"""
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from tldw_chatbook.Utils.fd_protection import (
    protect_file_descriptors,
    _fd_protection_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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


class _NegativeFilenoStream:
    """task-640 item 6: some stream implementations return -1 from
    fileno() instead of raising (rather than lacking fileno() entirely,
    like _NonFdBackedStream above). protect_file_descriptors()'s detection
    must still treat this as "not a real usable fd" via the downstream
    os.fstat(-1) -> OSError('[Errno 9] Bad file descriptor') and take the
    same except-branch fallback path.
    """

    def __init__(self):
        self.written = []

    def write(self, text):
        self.written.append(text)

    def flush(self):
        pass

    def fileno(self):
        return -1


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
    """RED for task-641 round 3: simulate Textual's redirected, non-fd-backed
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


def test_protect_file_descriptors_handles_fileno_returning_negative_one(monkeypatch):
    """task-640 item 6: fileno() returning -1 (not raising) must take the
    same safe fallback path as the AttributeError/no-fileno() case above --
    os.fstat(-1) raises OSError, which the except-branch also catches."""
    monkeypatch.setattr(sys, "stdout", _NegativeFilenoStream())
    monkeypatch.setattr(sys, "stderr", _NegativeFilenoStream())

    with _protected_real_stdio():
        with protect_file_descriptors():
            assert not isinstance(sys.stdout, _NegativeFilenoStream)

        assert _fd_usable(1), (
            "protect_file_descriptors() closed the real fd 1 (stdout) after "
            "a fileno()->-1 stream -- the os.fstat(-1) OSError path must "
            "still fall back safely, not close the real fd"
        )
        assert _fd_usable(2), (
            "protect_file_descriptors() closed the real fd 2 (stderr) after "
            "a fileno()->-1 stream -- the os.fstat(-1) OSError path must "
            "still fall back safely, not close the real fd"
        )
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
            import tldw_chatbook.Utils.fd_protection as fd_protection

            original_fdopen = os.fdopen
            fdopen_calls = []

            def _tracking_fdopen(*args, **kwargs):
                fdopen_calls.append((args, kwargs))
                return original_fdopen(*args, **kwargs)

            import unittest.mock as mock

            with mock.patch.object(os, "fdopen", side_effect=_tracking_fdopen):
                with mock.patch.object(sys, "stdout", real_file):
                    with mock.patch.object(sys, "stderr", real_file):
                        with fd_protection.protect_file_descriptors():
                            pass

            assert fdopen_calls == [], (
                "protect_file_descriptors() replaced already-valid, "
                "fd-backed streams -- it should be a no-op in this case"
            )
        finally:
            real_file.close()


# --- task-641 round-3 review: the finally block must close ONLY the
# wrapper(s) protect_file_descriptors() itself created -- never whatever
# happens to be sitting in sys.stdout/sys.stderr when the context manager
# exits. If code inside the protected `yield` (e.g. a nested library call)
# reassigns sys.stdout/sys.stderr to some OTHER real, fd-owning stream and
# leaves it there (the reviewer's example: `sys.stdout = sys.__stdout__`),
# the earlier "close whatever sys.stdout currently holds" cleanup would
# close THAT foreign stream -- for something like sys.__stdout__, that
# closes the real fd 1 and reproduces the exact WriterThread-killing freeze
# through a different door than the original round-3 bug. ---


def test_finally_closes_only_self_created_wrapper_not_a_foreign_stream_left_behind(
    monkeypatch, tmp_path
):
    """RED for the round-3 review finding: a foreign, fd-owning stream a
    nested call leaves in sys.stdout must survive untouched, while the
    wrapper protect_file_descriptors() itself created (no longer referenced
    by sys.stdout by the time `finally` runs) must still be closed."""
    monkeypatch.setattr(sys, "stdout", _NonFdBackedStream())
    monkeypatch.setattr(sys, "stderr", _NonFdBackedStream())

    foreign_stdout = open(tmp_path / "foreign_stdout.txt", "w")
    foreign_stderr = open(tmp_path / "foreign_stderr.txt", "w")

    with _protected_real_stdio():
        try:
            our_wrapper_out = None
            our_wrapper_err = None
            with protect_file_descriptors():
                # protect_file_descriptors() has replaced sys.stdout/
                # sys.stderr with its own fdopen(1/2, closefd=False)
                # wrappers -- capture them, then simulate a nested library
                # call (e.g. inside AutoTokenizer/AutoModel.from_pretrained)
                # reassigning sys.stdout/sys.stderr to some OTHER stream it
                # owns and never restoring it.
                our_wrapper_out = sys.stdout
                our_wrapper_err = sys.stderr
                sys.stdout = foreign_stdout
                sys.stderr = foreign_stderr

            assert not foreign_stdout.closed, (
                "protect_file_descriptors() closed a foreign stream a "
                "nested call left in sys.stdout -- it must only ever close "
                "wrappers it created itself"
            )
            assert not foreign_stderr.closed, (
                "protect_file_descriptors() closed a foreign stream a "
                "nested call left in sys.stderr -- it must only ever close "
                "wrappers it created itself"
            )
            assert our_wrapper_out is not None and our_wrapper_out.closed, (
                "protect_file_descriptors() failed to close its OWN "
                "temporary wrapper because sys.stdout no longer pointed "
                "at it by the time the context manager exited"
            )
            assert our_wrapper_err is not None and our_wrapper_err.closed, (
                "protect_file_descriptors() failed to close its OWN "
                "temporary wrapper because sys.stderr no longer pointed "
                "at it by the time the context manager exited"
            )
        finally:
            foreign_stdout.close()
            foreign_stderr.close()


def test_finally_never_closes_sys_dunder_stdout_left_behind_by_nested_code():
    """The reviewer's exact scenario: a nested call reassigns sys.stdout to
    sys.__stdout__ itself (the real, process-shared stream object) and
    leaves it there. protect_file_descriptors() must never close it --
    that would be the WriterThread freeze through a different door.

    Deliberately run in an isolated SUBPROCESS rather than in-process:
    closing sys.__stdout__/sys.__stderr__'s underlying TextIOWrapper
    objects is NOT something os.dup()/os.dup2() fd-number restoration (as
    used by the other tests in this file) can undo -- once the wrapper
    OBJECT itself is marked closed, every later write through
    sys.__stdout__/sys.__stderr__ specifically raises `ValueError: I/O
    operation on closed file`, regardless of what real fd number 1/2 later
    point at. Reproducing this in-process was confirmed (during this fix's
    own development) to corrupt the running pytest process's own stderr
    reporting for the remainder of the session. A subprocess contains the
    blast radius entirely to a throwaway child process.
    """
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})

class _NonFdBackedStream:
    def write(self, text):
        pass
    def flush(self):
        pass

sys.stdout = _NonFdBackedStream()
sys.stderr = _NonFdBackedStream()

from tldw_chatbook.Utils.fd_protection import protect_file_descriptors

with protect_file_descriptors():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# protect_file_descriptors()'s own finally block already restored
# sys.stdout/sys.stderr back to our no-op _NonFdBackedStream by this point
# (correct behavior -- it must always restore the TRUE original) -- so
# print()/sys.stdout here would go nowhere. Write through sys.__stdout__/
# sys.__stderr__ directly instead: if protect_file_descriptors() closed
# them, this raises ValueError("I/O operation on closed file") and the
# subprocess exits non-zero.
sys.__stdout__.write("stdout-still-alive\\n")
sys.__stdout__.flush()
sys.__stderr__.write("stderr-still-alive\\n")
sys.__stderr__.flush()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        result.returncode == 0
        and "stdout-still-alive" in result.stdout
        and "stderr-still-alive" in result.stderr
    ), (
        "protect_file_descriptors() closed sys.__stdout__/sys.__stderr__ "
        f"left behind by nested code (subprocess returncode="
        f"{result.returncode}, stdout={result.stdout!r}, "
        f"stderr={result.stderr!r})"
    )


# --- task-640 items 4/5: the module-level lock and the consolidation it
# depends on. ---


def test_lock_is_held_for_the_full_context_manager_body(monkeypatch):
    """task-640 item 4: the module-level lock must be held for the ENTIRE
    CM body, including the protected `yield` region -- not just the
    sys.stdout/stderr/os.environ setup -- so a second concurrent caller can
    never start its own save/reassign sequence while one is already in
    flight. Proven directly against the lock object protect_file_
    descriptors() actually uses (white-box, but the most deterministic way
    to prove mutual exclusion without a flaky timing-based test)."""
    monkeypatch.setattr(sys, "stdout", _NonFdBackedStream())
    monkeypatch.setattr(sys, "stderr", _NonFdBackedStream())

    with _protected_real_stdio():
        with protect_file_descriptors():
            # While inside the yielded region, a second thread attempting
            # to acquire the SAME lock must block (return False on a short
            # timeout) -- proving the lock is actually held right now, not
            # just released after the sys.stdout/stderr swap completes.
            acquired_elsewhere = []

            def _try_from_other_thread():
                got = _fd_protection_lock.acquire(timeout=0.2)
                acquired_elsewhere.append(got)
                if got:
                    _fd_protection_lock.release()

            t = threading.Thread(target=_try_from_other_thread)
            t.start()
            t.join(timeout=2)

            assert acquired_elsewhere == [False], (
                "a second thread acquired protect_file_descriptors()'s lock "
                "while a call was still inside its protected region -- the "
                "lock does not cover the full context-manager body"
            )

    # After the context manager has fully exited, the lock must be free
    # again (no leak).
    reacquired = _fd_protection_lock.acquire(timeout=1)
    assert reacquired, "the lock was not released after protect_file_descriptors() exited"
    _fd_protection_lock.release()


def test_protect_file_descriptors_serializes_concurrent_callers(monkeypatch):
    """task-640 item 4, end-to-end: two threads calling protect_file_
    descriptors() concurrently must run their protected regions one at a
    time, never interleaved."""
    monkeypatch.setattr(sys, "stdout", _NonFdBackedStream())
    monkeypatch.setattr(sys, "stderr", _NonFdBackedStream())

    events = []
    events_lock = threading.Lock()
    a_started = threading.Event()
    a_may_exit = threading.Event()

    def _record(entry):
        with events_lock:
            events.append(entry)

    def thread_a():
        with protect_file_descriptors():
            _record(("a", "enter"))
            a_started.set()
            # Hold the protected region open so thread B has a real chance
            # to race in if the lock did not actually serialize them.
            a_may_exit.wait(timeout=2)
            _record(("a", "exit"))

    def thread_b():
        assert a_started.wait(timeout=2), "thread A never entered"
        with protect_file_descriptors():
            _record(("b", "enter"))
            _record(("b", "exit"))

    with _protected_real_stdio():
        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        # Give thread B a real opportunity to (incorrectly) race in while A
        # is still holding its region open.
        time.sleep(0.1)
        a_may_exit.set()
        ta.join(timeout=5)
        tb.join(timeout=5)

    assert events == [("a", "enter"), ("a", "exit"), ("b", "enter"), ("b", "exit")], (
        f"thread B's protected region overlapped thread A's: {events!r}"
    )


def test_all_call_sites_share_the_same_function_object():
    """task-640 item 5 regression guard: Embeddings_Lib.py, transcription_
    service.py, higgs.py, and chatterbox.py's dynamic imports must all
    resolve to the SAME function object as tldw_chatbook.Utils.
    fd_protection.protect_file_descriptors -- not independent re-copies.
    A future accidental re-duplication (a local `def protect_file_
    descriptors():` reappearing in any of these modules) would fail this
    identity check even though every individual test above would still
    pass against whichever copy it imports."""
    from tldw_chatbook.Utils.fd_protection import (
        protect_file_descriptors as canonical,
    )
    from tldw_chatbook.Embeddings.Embeddings_Lib import (
        protect_file_descriptors as from_embeddings_lib,
    )
    from tldw_chatbook.Local_Ingestion.transcription_service import (
        protect_file_descriptors as from_transcription_service,
    )
    from tldw_chatbook.TTS.backends.higgs import (
        protect_file_descriptors as from_higgs,
    )

    assert from_embeddings_lib is canonical
    assert from_transcription_service is canonical
    assert from_higgs is canonical
