"""Behavioral evidence for native maintenance admission."""


def test_publication_never_overwrites_existing_file(tmp_path):
    import pytest
    from tldw_chatbook.Backup_Recovery.native_files import publish_new

    staged, destination = tmp_path / "stage", tmp_path / "good"
    staged.write_bytes(b"new")
    destination.write_bytes(b"previous backup")
    with pytest.raises(FileExistsError):
        publish_new(staged, destination)
    assert destination.read_bytes() == b"previous backup"


import json
import os
import select
import sqlite3
import subprocess
import sys
import threading

import pytest

from tldw_chatbook.Backup_Recovery.admission import (
    Admission,
    AdmissionError,
    AdmissionCancelled,
    AdmissionTimeout,
)

_CHILD = """
import json, sys, sqlite3
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission
root, action, names, extra = sys.argv[1:]
names = tuple(json.loads(names))
print("attempting", flush=True)
try:
    a = Admission(Path(root))
    if action == "remap":
        a.remap(names[0], (Path(extra),), 10)
        print("remapped", flush=True)
    else:
        ctx = a.maintenance(names, 10) if action == "maintenance" else getattr(a, action)(names)
        with ctx:
            connection = None
            if extra:
                connection = sqlite3.connect(extra)
                connection.execute("INSERT INTO records VALUES ('committed')")
            print("entered", flush=True)
            sys.stdin.readline()
            if connection:
                connection.commit()
                connection.close()
            print("retired", flush=True)
except Exception as error:
    print(type(error).__name__ + ":" + str(error), flush=True)
"""


def line(child):
    assert select.select([child.stdout], [], [], 10)[0], "child did not respond"
    data = bytearray()
    while not data.endswith(b"\n"):
        part = os.read(child.stdout.fileno(), 1)
        assert part, "child exited before response"
        data.extend(part)
    return data.decode().strip()


@pytest.fixture
def launch():
    children = []

    def start(control, action, names=("a",), extra=""):
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                _CHILD,
                str(control),
                action,
                json.dumps(names),
                str(extra),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        children.append(child)
        assert line(child) == "attempting"
        return child

    yield start
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()


def release(child):
    child.stdin.write("retire\n")
    child.stdin.flush()
    assert line(child) == "retired"
    assert child.wait(timeout=10) == 0


@pytest.fixture
def registered(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    admission = Admission(tmp_path / "control")
    admission.register("a", (source,))
    return admission, source


def test_maintenance_excludes_child_across_target_inode_replacement(registered, launch):
    admission, source = registered
    lock_inodes = {
        p.name: p.stat().st_ino for p in admission.control_root.glob("*.gate")
    }
    with admission.maintenance(("a",), 2):
        replacement = source.with_name("replacement")
        replacement.write_bytes(b"new")
        os.replace(replacement, source)
        child = launch(admission.control_root, "normal")
        assert not select.select([child.stdout], [], [], 0.1)[0]
    assert line(child) == "entered"
    release(child)
    assert source.read_bytes() == b"new"
    assert lock_inodes == {
        p.name: p.stat().st_ino for p in admission.control_root.glob("*.gate")
    }


@pytest.mark.parametrize("kind", ["hardlink", "symlink", "same_path", "nested"])
def test_aliases_share_maintenance_boundary(tmp_path, launch, kind):
    source = tmp_path / "source"
    if kind == "nested":
        source.mkdir()
        alias = source / "child"
        alias.write_bytes(b"data")
    else:
        source.write_bytes(b"data")
        alias = tmp_path / "alias"
        if kind == "hardlink":
            os.link(source, alias)
        elif kind == "symlink":
            alias.symlink_to(source)
        else:
            alias = source
    admission = Admission(tmp_path / "control")
    admission.register("a", (source,))
    admission.register("b", (alias,))
    with admission.maintenance(("a",), 2):
        child = launch(admission.control_root, "normal", ("b",))
        assert not select.select([child.stdout], [], [], 0.1)[0]
    assert line(child) == "entered"
    release(child)


def test_real_sqlite_transaction_and_connection_retire_before_capture(tmp_path, launch):
    database = tmp_path / "data.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE records(value TEXT)")
    admission = Admission(tmp_path / "control")
    admission.register("a", (database,))
    writer = launch(admission.control_root, "normal", extra=database)
    assert line(writer) == "entered"
    capture = launch(admission.control_root, "maintenance")
    assert not select.select([capture.stdout], [], [], 0.1)[0]
    release(writer)
    assert line(capture) == "entered"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM records").fetchall() == [
            ("committed",)
        ]
    release(capture)


def test_timeout_and_cancel_leave_writer_and_draft_intact(registered, launch):
    admission, source = registered
    writer = launch(admission.control_root, "normal")
    assert line(writer) == "entered"
    with pytest.raises(AdmissionTimeout):
        with admission.maintenance(("a",), 0.1):
            pytest.fail("exclusive admission while writer active")
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    try:
        with pytest.raises(AdmissionCancelled):
            with admission.maintenance(("a",), 5, cancel=cancel):
                pytest.fail("cancelled acquisition entered")
    finally:
        timer.join()
    assert writer.poll() is None
    assert source.read_bytes() == b"original"
    release(writer)
    with admission.maintenance(("a",), 2):
        pass


def test_dead_holder_os_lock_releases_without_deleting_lock_evidence(
    registered, launch
):
    admission, source = registered
    writer = launch(admission.control_root, "normal")
    assert line(writer) == "entered"
    writer.kill()
    writer.wait(timeout=10)
    with admission.maintenance(("a",), 2):
        assert source.read_bytes() == b"original"
    assert len(list(admission.control_root.glob("*.lease"))) == 1


def test_known_incompatible_is_refused_until_os_lifetime_ends(registered, launch):
    admission, _ = registered
    client = launch(admission.control_root, "incompatible")
    assert line(client) == "entered"
    with pytest.raises(AdmissionError, match="known_incompatible_client"):
        with admission.maintenance(("a",), 2):
            pass
    client.kill()
    client.wait(timeout=10)
    with admission.maintenance(("a",), 2):
        pass


def test_remap_reserves_old_and_new_aliases_without_blocking_retirement(
    registered, tmp_path, launch
):
    admission, source = registered
    target = tmp_path / "new"
    target.write_bytes(b"new")
    admission.register("b", (target,))
    old = launch(admission.control_root, "normal")
    new = launch(admission.control_root, "normal", ("b",))
    assert line(old) == line(new) == "entered"
    remap = launch(admission.control_root, "remap", extra=target)
    # A process-local read-only observation synchronizes the persisted reservation.
    for _ in range(200):
        state = json.loads((admission.control_root / "registry.json").read_text())
        if state["entries"]["a"]["pending"]:
            break
        threading.Event().wait(0.01)
    assert state["entries"]["a"]["pending"]
    blocked = launch(admission.control_root, "normal", ("b",))
    assert line(blocked) == "AdmissionError:remap_recovery_required"
    release(old)
    assert not select.select([remap.stdout], [], [], 0.1)[0]
    release(new)
    assert line(remap) == "remapped"
    assert remap.wait(timeout=10) == 0
    with admission.maintenance(("a",), 2):
        child = launch(admission.control_root, "normal", ("b",))
        assert not select.select([child.stdout], [], [], 0.1)[0]
    assert line(child) == "entered"
    release(child)
    assert source.read_bytes() == b"original"


def test_interrupted_remap_preserves_pending_registry_and_both_generations(
    registered, tmp_path, launch
):
    admission, source = registered
    target = tmp_path / "new"
    target.write_bytes(b"candidate")
    writer = launch(admission.control_root, "normal")
    assert line(writer) == "entered"
    with pytest.raises(AdmissionTimeout):
        admission.remap("a", (target,), 0.1)
    state = (admission.control_root / "registry.json").read_bytes()
    assert json.loads(state)["entries"]["a"]["pending"]
    release(writer)
    reopened = Admission(admission.control_root)
    with pytest.raises(AdmissionError, match="remap_recovery_required"):
        with reopened.normal(("a",)):
            pass
    assert (admission.control_root / "registry.json").read_bytes() == state
    assert source.read_bytes() == b"original"
    assert target.read_bytes() == b"candidate"


def test_corrupt_or_missing_control_evidence_never_reinitializes(registered):
    admission, _ = registered
    registry = admission.control_root / "registry.json"
    registry.write_bytes(b"broken")
    with pytest.raises(AdmissionError, match="registry_invalid"):
        Admission(admission.control_root)
    assert registry.read_bytes() == b"broken"
    registry.unlink()
    with pytest.raises(FileNotFoundError):
        Admission(admission.control_root)
    assert not registry.exists()


def test_control_root_overlap_and_nested_admission_are_refused(registered):
    admission, source = registered
    for target in (admission.control_root, admission.control_root.parent):
        with pytest.raises(AdmissionError, match="control_root_overlaps_target"):
            admission.register("b", (target,))
    with admission.normal(("a",)):
        with pytest.raises(AdmissionError, match="nested_admission_forbidden"):
            with admission.maintenance(("a",), 0.1):
                pass


def test_missing_stable_lock_is_not_recreated(registered):
    admission, _ = registered
    next(admission.control_root.glob("*.gate")).unlink()
    with pytest.raises(FileNotFoundError):
        with admission.normal(("a",)):
            pass
    assert not list(admission.control_root.glob("*.gate"))


def test_crashed_remapping_process_keeps_fail_closed_reservation(
    registered, tmp_path, launch
):
    admission, source = registered
    target = tmp_path / "target"
    target.write_bytes(b"new")
    writer = launch(admission.control_root, "normal")
    assert line(writer) == "entered"
    remap = launch(admission.control_root, "remap", extra=target)
    for _ in range(200):
        state = json.loads((admission.control_root / "registry.json").read_text())
        if state["entries"]["a"]["pending"]:
            break
        threading.Event().wait(0.01)
    assert state["entries"]["a"]["pending"]
    remap.kill()
    remap.wait(timeout=10)
    release(writer)
    child = launch(admission.control_root, "normal")
    assert line(child) in {
        "AdmissionError:remap_recovery_required",
        "AdmissionError:registry_publication_recovery_required",
    }  # Crash may precede durable local-intent retirement; both states stay fenced.
    assert source.read_bytes() == b"original"
    assert target.read_bytes() == b"new"


def test_opposing_namespace_orders_do_not_deadlock(registered, tmp_path, launch):
    admission, _ = registered
    target = tmp_path / "target"
    target.write_bytes(b"new")
    admission.register("b", (target,))
    first = launch(admission.control_root, "maintenance", ("b", "a"))
    assert line(first) == "entered"
    second = launch(admission.control_root, "maintenance", ("a", "b"))
    assert not select.select([second.stdout], [], [], 0.1)[0]
    release(first)
    assert line(second) == "entered"
    release(second)


def test_crashed_maintenance_holder_reopens_admission(registered, launch):
    admission, _ = registered
    maintenance = launch(admission.control_root, "maintenance")
    assert line(maintenance) == "entered"
    writer = launch(admission.control_root, "normal")
    assert not select.select([writer.stdout], [], [], 0.1)[0]
    maintenance.kill()
    maintenance.wait(timeout=10)
    assert line(writer) == "entered"
    release(writer)


@pytest.mark.parametrize("phase", ["reservation", "final_mapping"])
def test_failed_remap_metadata_barrier_preserves_authority_across_processes(
    registered, tmp_path, monkeypatch, phase
):
    import errno
    import fcntl
    from tldw_chatbook.Backup_Recovery import native_files

    admission, source = registered
    target = tmp_path / "new"
    target.write_bytes(b"candidate")
    control_inode = admission.control_root.stat().st_ino
    native_fcntl = fcntl.fcntl
    injected = []

    def fail_published_registry_barrier(fd, command, *args):
        result = native_fcntl(fd, command, *args)
        if command != fcntl.F_FULLFSYNC or os.fstat(fd).st_ino != control_inode:
            return result
        state = json.loads((admission.control_root / "registry.json").read_text())
        entry = state["entries"]["a"]
        matches = (
            bool(entry["pending"])
            if phase == "reservation"
            else entry["roots"] == [str(target)] and entry["pending"] is None
        )
        if matches and not injected:
            injected.append(True)
            raise OSError(errno.EIO, "injected_registry_barrier_failure")
        return result

    monkeypatch.setattr(native_files.fcntl, "fcntl", fail_published_registry_barrier)
    with pytest.raises(OSError, match="injected_registry_barrier_failure"):
        admission.remap("a", (target,), 2)
    code = """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission, AdmissionError
try:
    admission = Admission(Path(sys.argv[1]))
    with admission.normal(("a",)):
        print("admitted")
except AdmissionError as error:
    print(str(error))
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(admission.control_root)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() in {
        "remap_recovery_required",
        "registry_publication_recovery_required",
    }
    assert source.read_bytes() == b"original"
    assert target.read_bytes() == b"candidate"
    if phase == "final_mapping":
        intent = json.loads(
            (admission.control_root / "registry.pending.json").read_text()
        )
        assert intent["before"]["entries"]["a"]["pending"]
        assert intent["before"]["entries"]["a"]["roots"] == [str(source)]
        assert intent["after"]["entries"]["a"]["roots"] == [str(target)]


def _child_admission_result(control):
    code = """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission, AdmissionError
try:
    admission = Admission(Path(sys.argv[1]))
    if sys.argv[2] == "normal":
        with admission.normal(("a",)):
            print("admitted")
    else:
        print("initialized")
except AdmissionError as error:
    print(str(error))
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(control),
            "normal" if (control / "registry.json").exists() else "initial",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert child.returncode == 0, child.stderr
    return child.stdout.strip()


def test_initial_registry_barrier_failure_retains_before_after_evidence(
    tmp_path, monkeypatch
):
    import errno
    import fcntl
    from tldw_chatbook.Backup_Recovery import native_files

    control = tmp_path / "control"
    native_fcntl = fcntl.fcntl

    def fail_initial_barrier(fd, command, *args):
        result = native_fcntl(fd, command, *args)
        if (
            command == fcntl.F_FULLFSYNC
            and (control / "registry.json").exists()
            and os.fstat(fd).st_ino == control.stat().st_ino
        ):
            raise OSError(errno.EIO, "injected_initial_registry_barrier_failure")
        return result

    monkeypatch.setattr(native_files.fcntl, "fcntl", fail_initial_barrier)
    with pytest.raises(OSError, match="injected_initial_registry_barrier_failure"):
        Admission(control)
    intent = json.loads((control / "registry.pending.json").read_text())
    assert intent["version"] == 1 and intent["before"] is None
    assert intent["after"] == {"version": 1, "entries": {}}
    assert _child_admission_result(control) == "registry_publication_recovery_required"


@pytest.mark.parametrize("failure", ["unlink", "cleanup_barrier"])
def test_remap_cleanup_failure_exposes_only_durably_committed_mapping(
    registered, tmp_path, monkeypatch, failure
):
    import errno
    import fcntl
    from tldw_chatbook.Backup_Recovery import native_files

    admission, source = registered
    control = admission.control_root
    target = tmp_path / "new"
    target.write_bytes(b"candidate")
    native_fcntl, native_unlink = fcntl.fcntl, os.unlink
    final_mapping_barriers = []

    def final_state():
        entry = json.loads((control / "registry.json").read_text())["entries"]["a"]
        return entry["roots"] == [str(target)] and entry["pending"] is None

    def observe_or_fail(fd, command, *args):
        result = native_fcntl(fd, command, *args)
        if (
            command == fcntl.F_FULLFSYNC
            and os.fstat(fd).st_ino == control.stat().st_ino
            and final_state()
        ):
            if (control / "registry.pending.json").exists():
                final_mapping_barriers.append(True)
            elif failure == "cleanup_barrier":
                assert final_mapping_barriers
                raise OSError(errno.EIO, "injected_intent_cleanup_failure")
        return result

    def fail_unlink(name, *args, **kwargs):
        if name == "registry.pending.json" and final_state() and failure == "unlink":
            assert final_mapping_barriers
            raise OSError(errno.EIO, "injected_intent_cleanup_failure")
        return native_unlink(name, *args, **kwargs)

    monkeypatch.setattr(native_files.fcntl, "fcntl", observe_or_fail)
    monkeypatch.setattr(os, "unlink", fail_unlink)
    with pytest.raises(OSError, match="injected_intent_cleanup_failure"):
        admission.remap("a", (target,), 2)
    assert final_mapping_barriers
    assert source.read_bytes() == b"original"
    assert target.read_bytes() == b"candidate"
    if failure == "unlink":
        assert (control / "registry.pending.json").exists()
        assert (
            _child_admission_result(control) == "registry_publication_recovery_required"
        )
    else:
        assert not (control / "registry.pending.json").exists()
        assert _child_admission_result(control) == "admitted"
        assert final_state()  # Mapping passed its full native barrier before cleanup.


@pytest.mark.parametrize(
    "kind",
    [
        "corrupt",
        "unsupported_version",
        "invalid_before",
        "invalid_after",
        "oversized",
        "mismatched",
    ],
)
def test_existing_write_intent_is_never_overwritten_or_cleared(registered, kind):
    admission, source = registered
    control = admission.control_root
    state = json.loads((control / "registry.json").read_text())
    record = {"version": 1, "write_id": "a" * 32, "before": state, "after": state}
    if kind == "unsupported_version":
        record["version"] = 999
    elif kind == "invalid_before":
        record["before"] = {"version": True, "entries": {}}
    elif kind == "invalid_after":
        record["after"] = {"entries": {}}
    elif kind == "mismatched":
        record["after"]["entries"] = {}
    raw = (
        b"broken"
        if kind == "corrupt"
        else b"x" * 2200000
        if kind == "oversized"
        else json.dumps(record).encode()
    )
    intent = control / "registry.pending.json"
    intent.write_bytes(raw)
    intent.chmod(0o600)
    assert _child_admission_result(control) == "registry_publication_recovery_required"
    with pytest.raises(AdmissionError, match="registry_publication_recovery_required"):
        admission.register("b", (source,))
    assert intent.read_bytes() == raw


@pytest.mark.parametrize("phase", ["intent_file", "intent_directory"])
def test_failed_intent_barrier_never_replaces_registry(
    registered, tmp_path, monkeypatch, phase
):
    import errno
    import fcntl
    from tldw_chatbook.Backup_Recovery import native_files

    admission, source = registered
    control = admission.control_root
    original = (control / "registry.json").read_bytes()
    target = tmp_path / "new"
    target.write_bytes(b"candidate")
    native_fcntl = fcntl.fcntl

    def fail_intent_barrier(fd, command, *args):
        result = native_fcntl(fd, command, *args)
        intent = control / "registry.pending.json"
        if command == fcntl.F_FULLFSYNC and intent.exists():
            expected_inode = (
                intent.stat().st_ino
                if phase == "intent_file"
                else control.stat().st_ino
            )
            if os.fstat(fd).st_ino == expected_inode:
                raise OSError(errno.EIO, "injected_intent_barrier_failure")
        return result

    monkeypatch.setattr(native_files.fcntl, "fcntl", fail_intent_barrier)
    with pytest.raises(OSError, match="injected_intent_barrier_failure"):
        admission.remap("a", (target,), 2)
    assert (control / "registry.json").read_bytes() == original
    assert (control / "registry.pending.json").exists()
    assert _child_admission_result(control) == "registry_publication_recovery_required"
    assert source.read_bytes() == b"original"
    assert target.read_bytes() == b"candidate"


def test_registry_write_cannot_reinitialize_a_disappeared_existing_generation(
    registered,
):
    from tldw_chatbook.Backup_Recovery.admission import _Registry
    from tldw_chatbook.Backup_Recovery.native_files import pinned_directory

    admission, _ = registered
    registry = admission.control_root / "registry.json"
    registry.unlink()
    with pinned_directory(admission.control_root) as parent:
        with pytest.raises(FileNotFoundError):
            admission._write(parent, _Registry(version=1))
    assert not registry.exists()
    assert not (admission.control_root / "registry.pending.json").exists()
