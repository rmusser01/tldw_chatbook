"""Unit tests for the advisory per-profile instance lock (RAG-53 / task-7).

These tests exercise ``tldw_chatbook.Utils.instance_lock`` directly against a
``tmp_path`` -- the module takes a ``Path`` and never touches app config or
any live user directory, so these tests are hermetic.

Empirically verified on this machine (macOS/arm64, portalocker 3.2.0):
portalocker's exclusive lock DOES conflict between two independent
``open()`` file descriptors in the SAME process (flock-style semantics are
keyed on the open file *description*, not the process), so
``test_second_acquire_reports_holder`` below works in-process without
needing a ``multiprocessing`` child.
"""
from __future__ import annotations

from pathlib import Path

import portalocker
import pytest

from tldw_chatbook.Utils import instance_lock as instance_lock_module
from tldw_chatbook.Utils.instance_lock import acquire_profile_instance_lock


def test_first_acquire_succeeds(tmp_path):
    status = acquire_profile_instance_lock(tmp_path)
    assert status.acquired is True
    assert status.handle is not None
    status.handle.close()


def test_second_acquire_reports_holder(tmp_path):
    first = acquire_profile_instance_lock(tmp_path)
    second = acquire_profile_instance_lock(tmp_path)
    assert second.acquired is False
    assert second.holder_pid == first.written_pid  # our own pid
    assert second.handle is None
    first.handle.close()


def test_reacquire_after_release(tmp_path):
    first = acquire_profile_instance_lock(tmp_path)
    first.handle.close()
    second = acquire_profile_instance_lock(tmp_path)
    assert second.acquired is True
    second.handle.close()


def test_unwritable_dir_never_raises(tmp_path):
    target = tmp_path / "nope"
    target.mkdir()
    target.chmod(0o400)
    try:
        status = acquire_profile_instance_lock(target)
        assert status.acquired is True  # unknown -> quiet, never a false warning
    finally:
        target.chmod(0o700)


# --- Additional hardening tests (additive, beyond the brief's four) -------
# The task's cardinal rule is "never blocks, never raises, never prevents
# boot" -- these pin the remaining failure paths the self-review checklist
# calls out that the four required tests above don't reach directly.


def test_unexpected_portalocker_error_never_raises(tmp_path, monkeypatch):
    """A non-lock exception from ``portalocker.lock`` (import weirdness,
    platform quirk, ...) must still degrade to a quiet ``acquired=True``,
    never propagate and never block boot.
    """

    def _boom(handle, flags):
        raise RuntimeError("simulated portalocker weirdness")

    monkeypatch.setattr(instance_lock_module.portalocker, "lock", _boom)
    status = acquire_profile_instance_lock(tmp_path)
    assert status.acquired is True
    assert status.handle is None  # handle was closed before returning


def test_lock_exception_falls_through_to_acquired_true(tmp_path, monkeypatch):
    """``portalocker.exceptions.LockException`` (broken/unsupported locking
    mechanism -- e.g. ENOLCK, EOPNOTSUPP, EBADF, NFS EOFError -- per
    portalocker's POSIX locker) is NOT genuine contention and must NOT be
    mistaken for "someone else holds it".

    This is the exact distinction ``Model_Artifacts/leases.py:250-262``
    draws: only ``AlreadyLocked`` (a ``LockException`` subclass) means
    contention; the broader ``LockException`` means the lock mechanism
    itself failed. Catching the broader ``BaseLockException`` at the
    contention branch would fold this failure into "not acquired" and
    produce a permanent false "Profile already open" warning on every boot
    for anyone on a filesystem where flock is unsupported -- exactly the
    false warning this module exists to avoid.
    """

    def _boom(handle, flags):
        raise portalocker.exceptions.LockException("simulated ENOLCK")

    monkeypatch.setattr(instance_lock_module.portalocker, "lock", _boom)
    status = acquire_profile_instance_lock(tmp_path)
    assert status.acquired is True
    assert status.handle is None  # handle was closed before returning
    assert status.holder_pid is None


def test_body_write_failure_still_acquires(tmp_path, monkeypatch):
    """Writing the informational pid/timestamp body is best-effort -- an
    ``OSError`` there must not turn a real lock acquisition into a failure
    or a raised exception; the lock itself is the signal, not the body.
    """
    real_open = Path.open

    def _flaky_open(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        if mode == "a+b":
            monkeypatch.setattr(
                handle,
                "write",
                lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")),
            )
        return handle

    monkeypatch.setattr(Path, "open", _flaky_open)
    status = acquire_profile_instance_lock(tmp_path)
    assert status.acquired is True
    assert status.handle is not None
    status.handle.close()


def test_held_lock_never_raises(tmp_path):
    """The documented contract for a genuinely held lock: reported as
    ``acquired=False``, no exception escapes.
    """
    first = acquire_profile_instance_lock(tmp_path)
    try:
        second = acquire_profile_instance_lock(tmp_path)
        assert second.acquired is False
        assert second.handle is None
    finally:
        first.handle.close()


def test_read_holder_rejects_digit_like_non_int_pid_line(tmp_path):
    """`str.isdigit()` accepts non-ASCII digit-like characters (e.g. the
    superscript "²") that `int()` rejects with ``ValueError`` --
    ``_read_holder``'s ``except OSError`` didn't cover that, so it escaped
    from inside the ``AlreadyLocked`` handler and leaked the second call's
    fd (``handle.close()`` in ``acquire_profile_instance_lock`` runs only
    *after* ``_read_holder`` returns). A corrupted/hand-edited lock-file
    body must still degrade to an "unknown holder", never raise.
    """
    assert "²".isdigit()
    with pytest.raises(ValueError):
        int("²")

    first = acquire_profile_instance_lock(tmp_path)
    lock_path = tmp_path / ".instance.lock"
    lock_path.write_text("²\n2026-08-04T00:00:00+00:00\n", encoding="utf-8")
    try:
        second = acquire_profile_instance_lock(tmp_path)
        assert second.acquired is False
        assert second.holder_pid is None
        assert second.handle is None
    finally:
        first.handle.close()


def test_lock_file_is_never_unlinked(tmp_path):
    """The lock file must persist on disk after release -- unlinking would
    race a third instance onto a fresh inode and split the lock (see the
    module docstring).
    """
    status = acquire_profile_instance_lock(tmp_path)
    status.handle.close()
    assert (tmp_path / ".instance.lock").exists()
