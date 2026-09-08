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
a = Admission(Path(root))
names = tuple(json.loads(names))
print("attempting", flush=True)
try:
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
    assert line(child) == "AdmissionError:remap_recovery_required"
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
