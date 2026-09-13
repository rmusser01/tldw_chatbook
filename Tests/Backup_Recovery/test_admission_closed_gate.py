"""Closed maintenance gates defer ordinary root scans until admission can proceed."""

import threading

import pytest

from tldw_chatbook.Backup_Recovery.admission import Admission


@pytest.mark.parametrize("remove_root", [False, True])
def test_uncancellable_busy_gate_wait_uses_native_blocking_lock(
    tmp_path, monkeypatch, remove_root
):
    from tldw_chatbook.Backup_Recovery import admission as module

    root = tmp_path / "source"
    root.write_bytes(b"original")
    authority = Admission(tmp_path / "control")
    authority.register("source", (root,))
    waiting = Admission(authority.control_root)
    blocking, entered, done = threading.Event(), threading.Event(), threading.Event()
    errors, attempts = [], []
    original_flock = module.fcntl.flock
    original_open, opened = waiting._open, {}
    gate_name = waiting._key("source", "gate")

    def open_lock(parent, name, flags):
        fd = original_open(parent, name, flags)
        opened[fd] = name
        return fd

    def flock(fd, operation):
        if threading.current_thread() is child and opened.get(fd) == gate_name:
            attempts.append(operation)
            if not operation & module.fcntl.LOCK_NB:
                blocking.set()
        return original_flock(fd, operation)

    def acquire():
        try:
            with waiting.normal(("source",)):
                entered.set()
        except BaseException as error:  # noqa: BLE001 - report worker errors to test thread.
            errors.append(error)
        finally:
            done.set()

    child = threading.Thread(target=acquire, daemon=True)
    monkeypatch.setattr(waiting, "_open", open_lock)
    monkeypatch.setattr(module.fcntl, "flock", flock)
    try:
        with (
            authority._directory() as parent,
            authority._lock(
                parent, authority._key("source", "gate"), module.fcntl.LOCK_EX
            ),
        ):
            child.start()
            assert blocking.wait(5), "unbounded waiter kept polling the native gate"
            assert not entered.is_set()
            observed = len(attempts)
            assert not done.wait(0.2), "waiter entered through a held gate"
            assert len(attempts) == observed, "blocked native wait kept polling"
            if remove_root:
                root.unlink()
        assert done.wait(5), "native wait did not resume after gate release"
    finally:
        child.join(timeout=5)
    assert not child.is_alive()
    if remove_root:
        assert len(errors) == 1 and isinstance(errors[0], FileNotFoundError)
        assert not entered.is_set()
    else:
        assert errors == [] and entered.is_set()


@pytest.mark.parametrize("remove_root", [False, True])
def test_normal_waits_at_closed_requested_gate_before_scanning_roots(
    tmp_path, monkeypatch, remove_root
):
    root = tmp_path / "source"
    root.write_bytes(b"original")
    authority = Admission(tmp_path / "control")
    authority.register("source", (root,))
    waiting = Admission(authority.control_root)
    read, scanned, entered = threading.Event(), threading.Event(), threading.Event()
    cancel = threading.Event()
    completed = threading.Event()
    errors = []
    original_read, original_groups = waiting._read, waiting._groups

    def registry(parent):
        value = original_read(parent)
        read.set()
        return value

    def groups(*args, **kwargs):
        scanned.set()
        return original_groups(*args, **kwargs)

    monkeypatch.setattr(waiting, "_read", registry)
    monkeypatch.setattr(waiting, "_groups", groups)

    def acquire():
        try:
            with waiting._admit(("source",), False, None, cancel):
                entered.set()
        except BaseException as error:  # noqa: BLE001 - assert worker failures on the test thread.
            errors.append(error)
        finally:
            completed.set()

    child = threading.Thread(target=acquire)
    try:
        with authority.maintenance(("source",), 10):
            child.start()
            assert read.wait(5), "normal reader did not reach the registry"
            assert not scanned.wait(0.2), "closed gate triggered a target-root scan"
            assert not entered.is_set()
            if remove_root:
                root.unlink()
        assert completed.wait(5), "normal admission did not resume"
        assert scanned.is_set(), "resumed admission omitted fresh alias validation"
    finally:
        cancel.set()
        child.join(timeout=5)
    assert not child.is_alive()
    if remove_root:
        assert len(errors) == 1 and isinstance(errors[0], FileNotFoundError)
        assert not entered.is_set()
    else:
        assert errors == []
        assert entered.is_set()


@pytest.mark.parametrize("finish", ["release", "cancel", "timeout"])
def test_busy_gate_wait_does_not_reopen_registry_on_each_poll(
    tmp_path, monkeypatch, finish
):
    """A waiter polls one native gate, then revalidates before ordinary entry."""
    from tldw_chatbook.Backup_Recovery import admission as module

    root = tmp_path / "source"
    root.write_bytes(b"original")
    authority = Admission(tmp_path / "control")
    authority.register("source", (root,))
    waiting = Admission(authority.control_root)
    polled, done, entered = threading.Event(), threading.Event(), threading.Event()
    cancel = threading.Event()
    errors, reads, failures = [], [], []
    original_read, original_flock = waiting._read, module.fcntl.flock

    def read(parent):
        reads.append(parent)
        return original_read(parent)

    def flock(fd, operation):
        try:
            return original_flock(fd, operation)
        except BlockingIOError:
            if threading.current_thread() is child:
                failures.append(fd)
                if len(failures) >= 5:
                    polled.set()
            raise

    def acquire():
        try:
            deadline = waiting._deadline(2) if finish == "timeout" else None
            with waiting._admit(("source",), False, deadline, cancel):
                entered.set()
        except BaseException as error:  # noqa: BLE001 - report worker errors to test thread.
            errors.append(error)
        finally:
            done.set()

    monkeypatch.setattr(waiting, "_read", read)
    monkeypatch.setattr(module.fcntl, "flock", flock)
    child = threading.Thread(target=acquire)
    try:
        with (
            authority._directory() as parent,
            authority._lock(
                parent, authority._key("source", "gate"), module.fcntl.LOCK_EX
            ),
        ):
            child.start()
            assert polled.wait(5), "waiter did not poll the busy native gate"
            assert len(reads) == 1, "busy gate repeatedly reopened the registry"
            assert not entered.is_set()
            if finish == "cancel":
                cancel.set()
            if finish != "release":
                assert done.wait(5), "busy wait ignored cancellation or deadline"
        assert done.wait(5), "normal admission did not resume after gate release"
    finally:
        cancel.set()
        child.join(timeout=5)
    assert not child.is_alive()
    if finish == "release":
        assert errors == [] and entered.is_set()
        assert len(reads) == 2, "entry did not re-read the registry after waiting"
    else:
        expected = (
            module.AdmissionCancelled if finish == "cancel" else module.AdmissionTimeout
        )
        assert len(errors) == 1 and isinstance(errors[0], expected)
        assert not entered.is_set()
