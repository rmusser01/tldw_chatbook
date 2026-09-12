"""Closed maintenance gates defer ordinary root scans until admission can proceed."""

import threading

import pytest

from tldw_chatbook.Backup_Recovery.admission import Admission


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
