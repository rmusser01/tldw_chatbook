"""Installed persistence participants must prove actual maintenance coverage."""

import threading
import time
import os
import select

import pytest

from tldw_chatbook.Backup_Recovery.admission import Admission, AdmissionError, fcntl

# These fixtures launch real cooperating processes with native admission locks.
from Tests.Backup_Recovery.test_admission import launch, line, release


@pytest.fixture
def registered(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    authority = Admission(tmp_path / "control")
    authority.register("a", (source,))
    return authority, source


def _eventually(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("expected bounded native state was not observed")


def test_uncovered_persistent_owner_refuses_maintenance(tmp_path):
    import pytest

    from tldw_chatbook.Backup_Recovery.models import StorageItem, Inventory
    from tldw_chatbook.Backup_Recovery.participants import require_participant_coverage

    inventory = Inventory(
        (StorageItem("new.writer", "db", tmp_path / "db", "included", ()),),
        True,
        "scope",
        (),
    )
    with pytest.raises(ValueError, match="participant_missing"):
        require_participant_coverage(inventory, ())


def test_native_gate_contention_requests_pause_while_writer_still_holds_lease(
    registered, launch
):
    authority, _ = registered
    assert authority.pause_requested(("a",)) is False
    with authority.normal(("a",)):
        requester = launch(authority.control_root, "maintenance")
        _eventually(lambda: authority.pause_requested(("a",)))
        # A pause hint never acknowledges retirement or grants capture access.
        assert not select.select([requester.stdout], [], [], 0.05)[0]
    assert line(requester) == "entered"
    release(requester)
    assert authority.pause_requested(("a",)) is False


@pytest.mark.parametrize("ending", ["interrupt", "death"])
def test_requester_disappearance_removes_hint_without_retiring_writer(
    registered, launch, ending
):
    authority, source = registered
    with authority.normal(("a",)):
        requester = launch(authority.control_root, "maintenance")
        _eventually(lambda: authority.pause_requested(("a",)))
        if ending == "death":
            requester.kill()
            requester.wait(timeout=3)
        else:
            # SIGINT unwinds the first request's context and closes its native FDs.
            import signal

            requester.send_signal(signal.SIGINT)
            requester.wait(timeout=3)
        _eventually(lambda: not authority.pause_requested(("a",)))
        source.write_bytes(b"still-owned")
        assert source.read_bytes() == b"still-owned"


def test_normal_readers_are_not_pause_requests(registered, launch):
    authority, _ = registered
    writer = launch(authority.control_root, "normal")
    assert line(writer) == "entered"
    assert authority.pause_requested(("a",)) is False
    release(writer)


def test_probe_checks_shared_alias_group_and_leaves_disjoint_group_available(
    registered, tmp_path, launch
):
    authority, source = registered
    alias = tmp_path / "alias"
    os.link(source, alias)
    other = tmp_path / "other"
    other.write_bytes(b"other")
    authority.register("alias", (alias,))
    authority.register("other", (other,))
    requester = launch(authority.control_root, "maintenance")
    assert line(requester) == "entered"
    assert authority.pause_requested(("alias",)) is True
    assert authority.pause_requested(("other",)) is False
    release(requester)


def test_registry_contention_defers_a_nonblocking_pause_observation(registered):
    authority, _ = registered
    with authority._directory() as parent:
        with authority._lock(parent, "registry.lock", fcntl.LOCK_EX):
            started = time.monotonic()
            assert authority.pause_requested(("a",)) is False
            assert time.monotonic() - started < 0.5
    assert authority.pause_requested(("a",)) is False


@pytest.mark.parametrize("damage", ["missing", "replaced", "unsafe", "pending"])
def test_probe_refuses_changed_or_uncertain_native_evidence(registered, damage):
    authority, _ = registered
    assert authority.pause_requested(("a",)) is False
    gate = authority.control_root / authority._key("a", "gate")
    if damage == "missing":
        gate.unlink()
    elif damage == "replaced":
        gate.rename(gate.with_suffix(".old"))
        gate.write_bytes(b"")
        gate.chmod(0o600)
    elif damage == "unsafe":
        gate.chmod(0o666)
    else:
        pending = authority.control_root / "registry.pending.json"
        pending.write_bytes(b"{invalid")
        pending.chmod(0o600)
    with pytest.raises((AdmissionError, OSError)):
        authority.pause_requested(("a",))


def test_observed_group_change_refuses_reusing_old_pause_scope(registered):
    authority, source = registered
    assert authority.pause_requested(("a",)) is False
    authority.register("new_alias", (source,))
    with pytest.raises(AdmissionError, match="admission_scope_changed"):
        authority.pause_requested(("a",))


def test_competing_maintainers_keep_hint_until_last_native_holder_retires(
    registered, launch
):
    authority, _ = registered
    first = launch(authority.control_root, "maintenance")
    assert line(first) == "entered"
    second = launch(authority.control_root, "maintenance")
    assert authority.pause_requested(("a",)) is True
    assert not select.select([second.stdout], [], [], 0.05)[0]
    release(first)
    assert line(second) == "entered"
    assert authority.pause_requested(("a",)) is True
    release(second)
    assert authority.pause_requested(("a",)) is False


def test_cancellation_event_releases_hint_without_retiring_owner(registered):
    from tldw_chatbook.Backup_Recovery.admission import AdmissionCancelled

    authority, _ = registered
    cancel = threading.Event()
    results = []

    def request():
        try:
            with authority.maintenance(("a",), 3, cancel=cancel):
                results.append("unexpected_capture")
        except AdmissionCancelled:
            results.append("cancelled")

    with authority.normal(("a",)):
        requester = threading.Thread(target=request)
        requester.start()
        _eventually(lambda: authority.pause_requested(("a",)))
        cancel.set()
        requester.join(3)
        assert not requester.is_alive()
        assert results == ["cancelled"]
        assert authority.pause_requested(("a",)) is False


def test_interrupting_pending_acquisition_retires_its_native_thread(
    tmp_path,
):
    import signal
    import subprocess
    import sys

    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    root = tmp_path / "bootstrap"
    authority = admission_authority(root)
    script = """
import sys, threading, time
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
print('attempting', flush=True)
def observe_pending():
    while True:
        with storage_admission._lock:
            holds = tuple(storage_admission._holds.values())
            if holds and not holds[0].ready.is_set():
                print('native-acquisition-pending', flush=True)
                return
        time.sleep(0.005)
threading.Thread(target=observe_pending, daemon=True).start()
try:
    storage_admission.acquire_storage()
except KeyboardInterrupt:
    assert not storage_admission._holds
    assert not storage_admission._retiring_holds
    assert not any(t.name == 'chatbook-storage-admission' for t in threading.enumerate())
    print('cancelled-and-retired', flush=True)
"""
    with authority.maintenance(("bootstrap.unbound",), 3):
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", script, str(root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert line(child) == "attempting"
            assert line(child) == "native-acquisition-pending"
            child.send_signal(signal.SIGINT)
            assert line(child) == "cancelled-and-retired"
            assert child.wait(timeout=3) == 0
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
            child.stdout.close()
            child.stderr.close()


def test_blocked_acquisition_does_not_hold_coordinator_lock(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    first_root = tmp_path / "first"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: first_root)
    first = storage_admission.acquire_storage()
    second_root = tmp_path / "second"
    authority = admission_authority(second_root)
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: second_root)
    acquiring = threading.Event()
    retired = threading.Event()
    result = []

    def acquire():
        acquiring.set()
        result.append(storage_admission.acquire_storage())

    def retire():
        first.close()
        retired.set()

    with authority.maintenance(("bootstrap.unbound",), 2):
        opener = threading.Thread(target=acquire)
        opener.start()
        assert acquiring.wait(1)
        time.sleep(0.1)
        closer = threading.Thread(target=retire)
        closer.start()
        independent_close = retired.wait(0.3)
    opener.join(3)
    closer.join(3)
    assert not opener.is_alive() and not closer.is_alive()
    for lease in result:
        lease.close()
    assert independent_close, "native admission waited while holding coordinator lock"
