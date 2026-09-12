"""Windows boundary policy checks and native filesystem regression tests."""

from __future__ import annotations

import os
import stat

import pytest

from tldw_chatbook.Utils.windows_files import (
    WindowsLocks,
    WindowsOS,
    _acl_mode,
    _component,
)


@pytest.mark.parametrize(
    "name",
    ["..", ".", "a/b", "a\\b", "x:y", "x\x00y", "", "file.", "file ", "CON", "nul.txt"],
)
def test_relative_names_cannot_escape_or_alias(name):
    with pytest.raises(ValueError):
        _component(name)


def test_simple_component_is_preserved():
    assert _component("journal-123.json") == "journal-123.json"


def test_other_user_read_grant_is_not_private():
    assert _acl_mode([(0, 0, 1, "other")], "me", is_directory=False) & 0o044


def test_other_user_delete_child_grant_is_not_trusted_parent():
    assert _acl_mode([(0, 0, 0x40, "other")], "me", is_directory=True) & 0o022


def test_inherit_only_public_grant_requires_private_leaf_hardening():
    assert _acl_mode([(0, 8, 0x1F01FF, "other")], "me", is_directory=True) == 0o744


def test_unknown_ace_fails_closed():
    assert _acl_mode([(9, 0, 0, "")], "me", is_directory=False) & 0o077


native = pytest.mark.skipif(
    os.name != "nt", reason="requires native Windows handles and NTFS"
)


@native
def test_private_relative_file_and_exclusive_rename(tmp_path):
    win = WindowsOS()
    win.chmod(tmp_path, 0o700)
    parent = win.open(tmp_path, win.O_RDONLY | win.O_DIRECTORY)
    try:
        descriptor = win.open(
            "source", win.O_RDWR | win.O_CREAT | win.O_EXCL, 0o600, dir_fd=parent
        )
        try:
            win.write(descriptor, b"private payload")
            win.fsync(descriptor)
            info = win.fstat(descriptor)
            assert stat.S_IMODE(info.st_mode) == 0o600
            assert info.st_uid == win.geteuid()
            assert win.stat("source", dir_fd=parent).st_ino == info.st_ino
        finally:
            win.close(descriptor)
        win.rename("source", "published", src_dir_fd=parent, dst_dir_fd=parent)
        another = win.open(
            "source", win.O_RDWR | win.O_CREAT | win.O_EXCL, 0o600, dir_fd=parent
        )
        win.close(another)
        with pytest.raises(FileExistsError):
            win.rename("source", "published", src_dir_fd=parent, dst_dir_fd=parent)
        assert sorted(win.listdir(parent)) == ["published", "source"]
        win.fsync(parent)
    finally:
        win.close(parent)


@native
def test_independent_native_handles_contend_on_shared_and_exclusive_locks(tmp_path):
    win, locks = WindowsOS(), WindowsLocks()
    path = tmp_path / "lock"
    first = win.open(path, win.O_CREAT | win.O_RDWR | win.O_EXCL, 0o600)
    second = win.open(path, win.O_RDWR)
    try:
        locks.flock(first, locks.LOCK_SH | locks.LOCK_NB)
        locks.flock(second, locks.LOCK_SH | locks.LOCK_NB)
        locks.flock(second, locks.LOCK_UN)
        with pytest.raises(BlockingIOError):
            locks.flock(second, locks.LOCK_EX | locks.LOCK_NB)
        locks.flock(first, locks.LOCK_UN)
        locks.flock(second, locks.LOCK_EX | locks.LOCK_NB)
        locks.flock(second, locks.LOCK_UN)
    finally:
        win.close(first)
        win.close(second)


@native
def test_directory_junction_is_refused(tmp_path):
    import subprocess

    win = WindowsOS()
    target, junction = tmp_path / "target", tmp_path / "junction"
    target.mkdir()
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        check=True,
        capture_output=True,
    )  # nosec B603 B607 -- synthetic temp paths, fixed command
    with pytest.raises(OSError):
        win.open(junction, win.O_RDONLY | win.O_DIRECTORY | win.O_NOFOLLOW)


@native
def test_pinned_directory_retains_identity_after_native_publication(tmp_path):
    win = WindowsOS()
    win.mkdir(tmp_path / "old", 0o700)
    parent = win.open(tmp_path, win.O_DIRECTORY | win.O_RDONLY)
    child = win.open("old", win.O_DIRECTORY | win.O_RDONLY, dir_fd=parent)
    try:
        before = win.fstat(child)
        win.rename("old", "new", src_dir_fd=parent, dst_dir_fd=parent)
        assert win.stat("new", dir_fd=parent).st_ino == before.st_ino
        descriptor = win.open(
            "child", win.O_CREAT | win.O_EXCL | win.O_WRONLY, 0o600, dir_fd=child
        )
        win.close(descriptor)
        assert (tmp_path / "new" / "child").is_file()
        win.unlink("child", dir_fd=child)
        win.rmdir("new", dir_fd=parent)
    finally:
        win.close(child)
        win.close(parent)


@native
def test_native_timestamp_and_alternate_stream_detection(tmp_path):
    win = WindowsOS()
    path = tmp_path / "metadata"
    descriptor = win.open(path, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    try:
        timestamp = 1_700_000_000_000_000_000
        win.utime(descriptor, ns=(timestamp, timestamp))
        assert win.fstat(descriptor).st_mtime_ns == timestamp
        assert win.listxattr(descriptor) == []
        with open(str(path) + ":hidden", "wb") as stream:
            stream.write(b"must not be silently omitted")
        assert win.listxattr(descriptor) == [":hidden:$DATA"]
    finally:
        win.close(descriptor)


@native
def test_actual_public_acl_is_detected_then_hardened(tmp_path):
    import subprocess

    win = WindowsOS()
    path = tmp_path / "acl"
    descriptor = win.open(path, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    try:
        subprocess.run(
            ["icacls", str(path), "/grant", "*S-1-1-0:(R)"],
            check=True,
            capture_output=True,
        )  # nosec B603 B607 -- fixed tool and synthetic temp path
        assert win.fstat(descriptor).st_mode & 0o044
        win.fchmod(descriptor, 0o600)
        assert stat.S_IMODE(win.fstat(descriptor).st_mode) == 0o600
    finally:
        win.close(descriptor)


def test_local_facade_operation_identity_is_stable():
    win = WindowsOS()
    assert win.open is win.open
    assert win.stat is win.stat
    assert win.open in win.supports_dir_fd


@native
def test_native_stat_keeps_stdlib_type_and_nanoseconds(tmp_path):
    win = WindowsOS()
    descriptor = win.open(
        tmp_path / "stat", win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600
    )
    try:
        timestamp = 1_700_000_000_123_456_700
        win.utime(descriptor, ns=(timestamp, timestamp))
        info = win.fstat(descriptor)
        assert isinstance(info, os.stat_result)
        assert info.st_mtime_ns == timestamp
        assert info[0] == info.st_mode
    finally:
        win.close(descriptor)


@native
def test_inherit_only_public_acl_hardened_before_ordinary_children(tmp_path):
    import subprocess

    from tldw_chatbook.Utils.private_paths import secure_private_directory

    win = WindowsOS()
    path = tmp_path / "inherited"
    win.mkdir(path, 0o700)
    subprocess.run(
        ["icacls", str(path), "/grant", "*S-1-1-0:(OI)(CI)(IO)(M)"],
        check=True,
        capture_output=True,
    )  # nosec B603 B607 -- synthetic private fixture
    assert win.stat(path).st_mode & 0o044
    secure_private_directory(path, create=False, application_owned=True)
    child = path / "ordinary-child"
    child.write_bytes(b"created without the facade")
    assert stat.S_IMODE(win.stat(child).st_mode) == 0o600


@native
def test_directory_namespace_barriers_after_empty_create_rename_and_remove(tmp_path):
    win = WindowsOS()
    parent = win.open(tmp_path, win.O_RDONLY | win.O_DIRECTORY)
    try:
        win.mkdir("empty", 0o700, dir_fd=parent)
        child = win.open("empty", win.O_RDONLY | win.O_DIRECTORY, dir_fd=parent)
        try:
            win.fsync(child)
            win.fsync(parent)
            win.rename("empty", "renamed", src_dir_fd=parent, dst_dir_fd=parent)
            win.fsync(parent)
        finally:
            win.close(child)
        win.rmdir("renamed", dir_fd=parent)
        win.fsync(parent)
        assert win.listdir(parent) == []
    finally:
        win.close(parent)


@pytest.mark.parametrize("directory", [True, False])
def test_owner_rights_resolves_only_to_measured_current_owner(directory):
    expected = 0o700 if directory else 0o600
    assert (
        _acl_mode(
            [(0, 3, 0x1F01FF, "S-1-3-4")], "me", is_directory=directory, owner_sid="me"
        )
        == expected
    )


@pytest.mark.parametrize("owner", [None, "another-user"])
def test_owner_rights_does_not_trust_an_unknown_or_different_owner(owner):
    assert (
        _acl_mode(
            [(0, 3, 0x1F01FF, "S-1-3-4")], "me", is_directory=True, owner_sid=owner
        )
        & 0o066
    )


def test_owner_rights_does_not_mask_other_public_inheritance():
    assert (
        _acl_mode(
            [(0, 3, 0x1F01FF, "S-1-3-4"), (0, 11, 0x1F01FF, "everyone")],
            "me",
            is_directory=True,
            owner_sid="me",
        )
        & 0o044
    )


@native
def test_stdlib_private_mkdir_and_ordinary_children_are_private(tmp_path):
    import sqlite3

    win = WindowsOS()
    path = tmp_path / "stdlib-private"
    # CPython's Windows mkdir(0700) uses a protected SYSTEM/Admin/OWNER RIGHTS
    # DACL. Exercise that actual descriptor, not one created by our facade.
    path.mkdir(mode=0o700)
    info = win.stat(path)
    assert info.st_uid == win.geteuid()
    assert stat.S_IMODE(info.st_mode) == 0o700
    ordinary = path / "ordinary.txt"
    ordinary.write_text("private content", encoding="utf-8")
    assert stat.S_IMODE(win.stat(ordinary).st_mode) == 0o600
    database = path / "ordinary.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE synthetic (value TEXT)")
        connection.execute("INSERT INTO synthetic VALUES (?)", ("private",))
        connection.commit()
        assert stat.S_IMODE(win.stat(database).st_mode) == 0o600
    finally:
        connection.close()


@native
def test_native_replace_updates_existing_target_without_changing_noreplace(tmp_path):
    win = WindowsOS()
    parent = win.open(tmp_path, win.O_RDONLY | win.O_DIRECTORY)
    try:
        for name, payload in (("target", b"old"), ("incoming", b"new")):
            descriptor = win.open(
                name, win.O_WRONLY | win.O_CREAT | win.O_EXCL, 0o600, dir_fd=parent
            )
            try:
                win.write(descriptor, payload)
                win.fsync(descriptor)
            finally:
                win.close(descriptor)
        incoming_identity = win.stat("incoming", dir_fd=parent).st_ino
        win.replace("incoming", "target", src_dir_fd=parent, dst_dir_fd=parent)
        win.fsync(parent)
        assert (tmp_path / "target").read_bytes() == b"new"
        assert win.stat("target", dir_fd=parent).st_ino == incoming_identity
        assert not (tmp_path / "incoming").exists()
        descriptor = win.open(
            "collision", win.O_WRONLY | win.O_CREAT | win.O_EXCL, 0o600, dir_fd=parent
        )
        win.close(descriptor)
        with pytest.raises(FileExistsError):
            win.rename("collision", "target", src_dir_fd=parent, dst_dir_fd=parent)
        assert (tmp_path / "target").read_bytes() == b"new"
    finally:
        win.close(parent)


@native
def test_atomic_private_write_replaces_existing_config(tmp_path):
    from tldw_chatbook.Utils.private_paths import atomic_private_write_bytes

    target = tmp_path / "config.toml"
    first = atomic_private_write_bytes(
        target, b"generation = 1\n", application_owned_directory=tmp_path
    )
    before = WindowsOS().stat(target).st_ino
    second = atomic_private_write_bytes(
        target, b"generation = 2\n", application_owned_directory=tmp_path
    )
    assert first.verified_private and second.verified_private
    assert target.read_bytes() == b"generation = 2\n"
    assert WindowsOS().stat(target).st_ino != before
    assert list(tmp_path.glob(".config.toml.*.tmp")) == []
