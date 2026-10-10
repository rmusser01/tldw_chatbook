"""Creation intent custody survives native failure and process death."""

import json
import os as system_os
import subprocess
import sys

import pytest

from tldw_chatbook.Backup_Recovery import control_records, native_files
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Utils.platform_files import os


def _intent(root):
    matches = list(root.parent.glob(".bootstrap-create-*.json"))
    assert len(matches) == 1, "creation intent must survive an unsuccessful attempt"
    return matches[0]


def _leave_intent(root, monkeypatch):
    def before_creation(*args, **kwargs):
        raise OSError("fixture_before_creation")

    with monkeypatch.context() as isolated:
        isolated.setattr(os, "mkdir", before_creation)
        with pytest.raises(OSError, match="fixture_before_creation"):
            control_records._ensure(root)
    assert not root.exists()
    return _intent(root)


def _identity(path=None, *, fd=None):
    info = os.fstat(fd) if fd is not None else os.stat(path, follow_symlinks=False)
    return info.st_dev, info.st_ino


def test_interrupted_creation_retains_durable_exact_intent_and_retires_files(
    tmp_path, monkeypatch
):
    root = tmp_path / "bootstrap"
    opened = []
    original = os.open

    def observed_open(path, *args, **kwargs):
        fd = original(path, *args, **kwargs)
        if str(path).startswith(".bootstrap-create-"):
            opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", observed_open)
    marker = _leave_intent(root, monkeypatch)
    record = json.loads(marker.read_bytes())
    assert record == {
        "version": 1,
        "child": os.path.normcase(root.name),
        "parent_device": _identity(tmp_path)[0],
        "parent_inode": _identity(tmp_path)[1],
    }
    assert os.stat(marker).st_uid == os.geteuid()
    assert os.stat(marker).st_mode & 0o777 == 0o600
    assert opened
    for fd in set(opened):
        with pytest.raises(OSError):
            os.fstat(fd)
    control_records._ensure(root)
    assert root.is_dir()
    assert not marker.exists()


@pytest.mark.parametrize("damage", ["invalid-json", "unknown-field", "wrong-child"])
def test_uncertain_creation_intent_cannot_create_lower_directory(
    tmp_path, monkeypatch, damage
):
    root = tmp_path / "bootstrap"
    marker = _leave_intent(root, monkeypatch)
    record = json.loads(marker.read_bytes())
    if damage == "invalid-json":
        data = b"{"
    else:
        if damage == "unknown-field":
            record["imported_authority"] = "another-directory"
        else:
            record["child"] = "another-directory"
        data = json.dumps(record).encode()
    marker.write_bytes(data)
    with pytest.raises((RecoveryRequired, ValueError)):
        control_records._ensure(root)
    assert not root.exists()
    assert marker.read_bytes() == data


def test_moved_creation_intent_refuses_changed_containing_parent(tmp_path, monkeypatch):
    before, after = tmp_path / "before", tmp_path / "after"
    native_files.create_private_directory(before)
    native_files.create_private_directory(after)
    marker = _leave_intent(before / "bootstrap", monkeypatch)
    moved = after / marker.name
    marker.rename(moved)
    data = moved.read_bytes()
    with pytest.raises((RecoveryRequired, ValueError)):
        control_records._ensure(after / "bootstrap")
    assert not (after / "bootstrap").exists()
    assert moved.read_bytes() == data


def test_replaced_containing_parent_cannot_accept_old_creation_intent(
    tmp_path, monkeypatch
):
    parent = tmp_path / "parent"
    native_files.create_private_directory(parent)
    marker = _leave_intent(parent / "bootstrap", monkeypatch)
    parent.rename(tmp_path / "retained-parent")
    native_files.create_private_directory(parent)
    moved = parent / marker.name
    (tmp_path / "retained-parent" / marker.name).rename(moved)
    with pytest.raises((RecoveryRequired, ValueError)):
        control_records._ensure(parent / "bootstrap")
    assert not (parent / "bootstrap").exists()
    assert moved.exists()


@pytest.mark.parametrize("damage", ["removed", "substituted"])
def test_creation_intent_named_binding_is_rechecked_before_mkdir(
    tmp_path, monkeypatch, damage
):
    root = tmp_path / "bootstrap"
    marker = _leave_intent(root, monkeypatch)
    original = control_records._read_creation_intent
    mutated = False

    def retargeted_record(fd):
        nonlocal mutated
        result = original(fd)
        if not mutated:
            mutated = True
            marker.rename(tmp_path / "retained-intent")
            if damage == "substituted":
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                try:
                    os.write(fd, result.model_dump_json().encode())
                finally:
                    os.close(fd)
        return result

    monkeypatch.setattr(control_records, "_read_creation_intent", retargeted_record)
    with pytest.raises((RecoveryRequired, ValueError, FileNotFoundError)):
        control_records._ensure(root)
    assert not root.exists()
    assert (tmp_path / "retained-intent").exists()
    assert marker.exists() == (damage == "substituted")


def test_live_creation_intent_lock_refuses_and_retires_after_creator_exit(
    tmp_path, monkeypatch
):
    root = tmp_path / "bootstrap"
    marker = _leave_intent(root, monkeypatch)
    code = """
import sys
from tldw_chatbook.Backup_Recovery.admission import fcntl
from tldw_chatbook.Utils.platform_files import os
fd = os.open(sys.argv[1], os.O_RDWR | os.O_NOFOLLOW)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('held', flush=True)
    sys.stdin.readline()
finally:
    os.close(fd)
"""
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", code, str(marker)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "held"
        with pytest.raises(RecoveryRequired, match="bootstrap_creation_busy"):
            control_records._ensure(root)
        assert not root.exists()
        assert marker.exists()
        child.stdin.write("retire\n")
        child.stdin.flush()
        _, error = child.communicate(timeout=10)
        assert child.returncode == 0, error
        control_records._ensure(root)
        assert root.is_dir() and not marker.exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


def test_process_death_after_mkdir_retries_exact_entry_barrier_before_lower_creation(
    tmp_path, monkeypatch
):
    root = tmp_path / "first" / "bootstrap"
    code = """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import control_records
from tldw_chatbook.Utils.platform_files import os
mkdir = os.mkdir
def died_after_creation(*args, **kwargs):
    mkdir(*args, **kwargs)
    os._exit(23)
os.mkdir = died_after_creation
control_records._ensure(Path(sys.argv[1]))
"""
    child = subprocess.run([sys.executable, "-c", code, str(root)], timeout=10)
    assert child.returncode == 23
    assert root.parent.is_dir() and not root.exists()
    marker = _intent(root.parent)
    expected = _identity(tmp_path)
    barriers = []
    original_barrier = control_records.flush_directory
    original_mkdir = os.mkdir

    def observed_barrier(fd):
        original_barrier(fd)
        barriers.append(_identity(fd=fd))

    def lower_creation(path, *args, **kwargs):
        assert (
            expected in barriers
        ), "prior created entry barrier must precede lower creation"
        assert not marker.exists()
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(control_records, "flush_directory", observed_barrier)
    monkeypatch.setattr(os, "mkdir", lower_creation)
    control_records._ensure(root)
    assert root.is_dir()
    assert expected in barriers
    assert not marker.exists()


@pytest.mark.skipif(system_os.name != "nt", reason="actual Windows ACL denial")
def test_windows_failed_entry_barrier_retry_synchronizes_exact_prior_entry(
    tmp_path, monkeypatch
):
    from Tests.Utils.test_windows_native_admission import _replace_security
    from tldw_chatbook.Utils import windows_files

    # The elevated runner's pytest parent belongs to TokenOwner. Create
    # our exact barrier target with explicit TokenUser ownership before
    # changing its ACL; production chmod must keep refusing foreign owners.
    parent = tmp_path / "explicit-user-private"
    os.mkdir(parent, 0o700)
    root = parent / "first" / "bootstrap"
    original_mkdir = os.mkdir
    first = True

    def denied_after_creation(path, mode=0o777, *, dir_fd=None):
        nonlocal first
        original_mkdir(path, mode, dir_fd=dir_fd)
        if first:
            first = False
            user = windows_files._native().user_sid
            _replace_security(
                parent,
                f"D:P(D;;0x6;;;{user})(A;;FA;;;{user})(A;;FA;;;SY)(A;;FA;;;BA)",
            )

    with monkeypatch.context() as isolated:
        isolated.setattr(os, "mkdir", denied_after_creation)
        try:
            with pytest.raises(PermissionError) as error:
                control_records._ensure(root)
            assert error.value.winerror == 5
        finally:
            os.chmod(parent, 0o700)
    marker = _intent(root.parent)
    expected = _identity(parent)
    barriers = []
    original = control_records.flush_directory

    def completed_barrier(fd):
        original(fd)
        barriers.append(_identity(fd=fd))

    monkeypatch.setattr(control_records, "flush_directory", completed_barrier)
    control_records._ensure(root)
    assert expected in barriers
    assert not marker.exists()
    assert root.is_dir()


@pytest.mark.skipif(system_os.name != "nt", reason="actual Windows shared ACL")
def test_shared_creation_record_refuses_without_repair(tmp_path, monkeypatch):
    from Tests.Utils.test_windows_native_admission import _replace_security
    from tldw_chatbook.Utils import windows_files

    root = tmp_path / "bootstrap"
    marker = _leave_intent(root, monkeypatch)
    data = marker.read_bytes()
    user = windows_files._native().user_sid
    _replace_security(
        marker,
        f"D:P(A;;FA;;;{user})(A;;FA;;;SY)(A;;FA;;;BA)(A;;FR;;;WD)",
    )
    with pytest.raises((RecoveryRequired, ValueError, PermissionError)):
        control_records._ensure(root)
    assert not root.exists()
    assert marker.read_bytes() == data
    assert os.stat(marker).st_mode & 0o044


def test_failed_initial_intent_barrier_is_reestablished_before_retry_creation(
    tmp_path, monkeypatch
):
    root = tmp_path / "bootstrap"

    def failed_initial_bytes(fd):
        raise OSError("creation_intent_bytes_barrier_failed")

    with monkeypatch.context() as isolated:
        isolated.setattr(control_records, "flush_file", failed_initial_bytes)
        with pytest.raises(OSError, match="creation_intent_bytes_barrier_failed"):
            control_records._ensure(root)
    marker = _intent(root)
    expected = _identity(marker)
    completed = []
    original_barrier = control_records.flush_file
    original_mkdir = os.mkdir

    def completed_bytes(fd):
        original_barrier(fd)
        completed.append(_identity(fd=fd))

    def retried_creation(*args, **kwargs):
        assert (
            expected in completed
        ), "retry requires durable intent before child creation"
        original_mkdir(*args, **kwargs)

    monkeypatch.setattr(control_records, "flush_file", completed_bytes)
    monkeypatch.setattr(os, "mkdir", retried_creation)
    control_records._ensure(root)
    assert root.is_dir()
    assert not marker.exists()


@pytest.mark.skipif(system_os.name != "nt", reason="actual Windows foreign owner")
def test_foreign_creation_record_refuses_without_repair(tmp_path, monkeypatch):
    from Tests.Utils.test_windows_native_admission import (
        _enable_restore_privilege,
        _replace_security,
    )
    from tldw_chatbook.Utils import windows_files

    root = tmp_path / "bootstrap"
    marker = _leave_intent(root, monkeypatch)
    data = marker.read_bytes()
    user = windows_files._native().user_sid
    restore_privilege = _enable_restore_privilege()
    try:
        _replace_security(
            marker,
            f"O:SYD:P(A;;FA;;;{user})(A;;FA;;;SY)(A;;FA;;;BA)",
            replace_owner=True,
        )
        with pytest.raises((RecoveryRequired, ValueError, PermissionError)):
            control_records._ensure(root)
        assert not root.exists()
        assert marker.read_bytes() == data
    finally:
        restore_privilege()
