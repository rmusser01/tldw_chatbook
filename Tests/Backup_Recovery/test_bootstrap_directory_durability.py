"""Native bootstrap durability covers changed entries and pending record bytes."""

import os as system_os
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import control_records, native_files, publication
from tldw_chatbook.Backup_Recovery.bootstrap import _key, _records
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Utils.platform_files import os


def _identity(path=None, *, fd=None):
    info = os.fstat(fd) if fd is not None else os.stat(path, follow_symlinks=False)
    return info.st_dev, info.st_ino


def _registered_pending(tmp_path, root=None):
    root = root or tmp_path / "bootstrap"
    control = tmp_path / "control"
    native_files.create_private_directory(control)
    selector = tmp_path / "selected.toml"
    fd = os.open(selector, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, b'[general]\nusers_name="test"\n')
    finally:
        os.close(fd)
    operation_id = "native-barrier-op"
    control_records.register_pending(
        root, operation_id, ("profile",), control, (selector,)
    )
    journal = SimpleNamespace(
        root=control / "journal",
        operation_id=operation_id,
        _flush_record=Journal._flush_record,
    )
    context = SimpleNamespace(
        bootstrap_root=str(root), namespaces=["profile"], selectors=[str(selector)]
    )
    return journal, context, root


def test_pending_revalidation_flushes_only_its_record_directory(tmp_path, monkeypatch):
    journal, context, root = _registered_pending(tmp_path)
    expected = _identity(root)
    barriers = []
    original = publication.flush_directory

    def changed_directory_only(fd):
        observed = _identity(fd=fd)
        assert observed == expected, "unchanged ancestor received a publication barrier"
        original(fd)
        barriers.append(observed)

    monkeypatch.setattr(publication, "flush_directory", changed_directory_only)
    publication._pending(journal, context, durable=True)
    assert barriers == [expected]
    assert _records(root)[0][0]["operation_id"] == journal.operation_id


def test_missing_bootstrap_chain_flushes_each_created_containing_directory(
    tmp_path, monkeypatch
):
    root = tmp_path / "first" / "second" / "bootstrap"
    barriers = []
    original = native_files.flush_directory

    def observed_barrier(fd):
        original(fd)
        barriers.append(_identity(fd=fd))

    monkeypatch.setattr(native_files, "flush_directory", observed_barrier)
    monkeypatch.setattr(control_records, "flush_directory", observed_barrier)
    _registered_pending(tmp_path, root)
    # Control creation is independent; every newly created bootstrap entry has
    # its own actual pinned parent barrier before registration returns.
    expected = [_identity(path) for path in (tmp_path, root.parent.parent, root.parent)]
    # Each entry has a durable intent, entry and intent-removal barrier.
    assert barriers.count(expected[0]) == 4  # Includes independent control creation.
    assert barriers.count(expected[1]) == 3
    assert barriers.count(expected[2]) == 3
    assert _records(root)[0]


def test_missing_bootstrap_parent_barrier_failure_prevents_registration(
    tmp_path, monkeypatch
):
    root = tmp_path / "first" / "bootstrap"
    expected = _identity(tmp_path)
    original = native_files.flush_directory

    def failed_parent_barrier(fd):
        if _identity(fd=fd) == expected and (tmp_path / "first").is_dir():
            raise OSError("bootstrap_creation_barrier_failed")
        original(fd)

    monkeypatch.setattr(control_records, "flush_directory", failed_parent_barrier)
    with pytest.raises(OSError, match="bootstrap_creation_barrier_failed"):
        control_records._ensure(root)
    assert not root.exists()
    assert (tmp_path / "first").is_dir()


def test_pending_record_directory_barrier_failure_retains_fence(tmp_path, monkeypatch):
    journal, context, root = _registered_pending(tmp_path)
    before = _records(root)

    def failed_record_barrier(fd):
        assert _identity(fd=fd) == _identity(root)
        raise OSError("pending_record_barrier_failed")

    monkeypatch.setattr(publication, "flush_directory", failed_record_barrier)
    with pytest.raises(OSError, match="pending_record_barrier_failed"):
        publication._pending(journal, context, durable=True)
    assert _records(root) == before
    assert (root / f"pending-{_key(journal.operation_id)}.json").is_file()


@pytest.mark.parametrize("barrier_fails", [False, True])
def test_concurrent_bootstrap_creation_still_flushes_exact_containing_directory(
    tmp_path, monkeypatch, barrier_fails
):
    root = tmp_path / "bootstrap"
    original_mkdir = os.mkdir
    original_barrier = control_records.flush_directory
    barriers = []

    def competing_creation(path, mode=0o777, *, dir_fd=None):
        # A real private directory appears after _ensure's absent observation,
        # before its exclusive mkdir obtains the name.
        original_mkdir(path, mode, dir_fd=dir_fd)
        raise FileExistsError("another creator obtained the bootstrap name")

    def observed_barrier(fd):
        barriers.append(_identity(fd=fd))
        if barrier_fails and root.is_dir():
            raise OSError("competing_creation_barrier_failed")
        original_barrier(fd)

    monkeypatch.setattr(os, "mkdir", competing_creation)
    monkeypatch.setattr(control_records, "flush_directory", observed_barrier)
    if barrier_fails:
        with pytest.raises(OSError, match="competing_creation_barrier_failed"):
            control_records._ensure(root)
    else:
        control_records._ensure(root)
    assert root.is_dir()
    assert barriers == [_identity(tmp_path)] * (2 if barrier_fails else 3)


def test_pending_record_bytes_barrier_failure_prevents_directory_completion(
    tmp_path, monkeypatch
):
    journal, context, root = _registered_pending(tmp_path)
    before = _records(root)
    directories = []

    def failed_record_bytes(parent, name, expected):
        assert _identity(fd=parent) == _identity(root)
        assert name == f"pending-{_key(journal.operation_id)}.json"
        assert expected == before[0][0]
        raise OSError("pending_record_bytes_barrier_failed")

    monkeypatch.setattr(journal, "_flush_record", failed_record_bytes)
    monkeypatch.setattr(publication, "flush_directory", directories.append)
    with pytest.raises(OSError, match="pending_record_bytes_barrier_failed"):
        publication._pending(journal, context, durable=True)
    assert not directories
    assert _records(root) == before


@pytest.mark.skipif(
    system_os.name != "nt", reason="actual Windows native barrier flags"
)
def test_windows_bootstrap_required_barriers_use_normal_metadata_and_device_sync(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Utils import windows_files

    native = windows_files._native()
    original = native.nt.NtFlushBuffersFileEx
    barriers = []

    def observed_native_barrier(handle, flags, buffer, length, status):
        info = native.info(handle)
        barriers.append(
            ((info.volume, (info.index_high << 32) | info.index_low), flags)
        )
        return original(handle, flags, buffer, length, status)

    monkeypatch.setattr(native.nt, "NtFlushBuffersFileEx", observed_native_barrier)
    root = tmp_path / "first" / "second" / "bootstrap"
    journal, context, root = _registered_pending(tmp_path, root)
    publication._pending(journal, context, durable=True)
    assert barriers
    assert all(flags == 0 for _, flags in barriers)
    assert {
        _identity(path) for path in (tmp_path, root.parent.parent, root.parent, root)
    } <= {identity for identity, _ in barriers}
