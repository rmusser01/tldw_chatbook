"""Native checks for one finite admission tree's parent-bound volume proof."""

import ctypes as C
import copy
import os
import threading

import pytest

from tldw_chatbook.Utils import windows_files

pytestmark = pytest.mark.skipif(os.name != "nt", reason="real Windows volume proof")


def test_metadata_tree_qualifies_only_roots_after_equivalent_fresh_observation(
    tmp_path, monkeypatch
):
    native, win = windows_files._native(), windows_files.WindowsOS()
    left, right, missing = (tmp_path / name for name in ("left", "right", "absent"))
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    expected = {p: win.stat_for_admission(p) for p in (left, right)}
    owner = threading.get_ident()
    original_open, original_close = native.open_handle, native.kernel.CloseHandle
    original_ntfs, original_security = native.ntfs, native.security
    births, current, volumes, security = [], {}, [], []

    def opened(name, **kwargs):
        handle = original_open(name, **kwargs)
        if threading.get_ident() == owner:
            assert handle not in current or not current[handle]["live"]
            row = {"handle": handle, "name": name, "live": True, "closes": 0}
            births.append(row)
            current[handle] = row
        return handle

    def closed(handle):
        row = current.get(getattr(handle, "value", handle))
        result = original_close(handle)
        if threading.get_ident() == owner and row is not None:
            row["closes"] += 1
            if result:
                row["live"] = False
        return result

    def ntfs(handle):
        result = original_ntfs(handle)
        if threading.get_ident() == owner:
            volumes.append(getattr(handle, "value", handle))
        return result

    def secured(handle, *args, **kwargs):
        result = original_security(handle, *args, **kwargs)
        if threading.get_ident() == owner:
            security.append(current[handle]["name"])
        return result

    monkeypatch.setattr(native, "open_handle", opened)
    monkeypatch.setattr(native.kernel, "CloseHandle", closed)
    monkeypatch.setattr(native, "ntfs", ntfs)
    monkeypatch.setattr(native, "security", secured)
    observed = win.stat_many_for_admission((right, missing, left, right))
    assert list(observed) == [right, missing, left] and observed[missing] is None
    for path in (left, right):
        actual, descriptor = observed[path]
        prior, prior_descriptor = expected[path]
        assert tuple(actual) == tuple(prior)
        assert actual.st_mtime_ns == prior.st_mtime_ns
        assert actual.st_ctime_ns == prior.st_ctime_ns
        assert descriptor == prior_descriptor
    assert all(not row["live"] and row["closes"] == 1 for row in births)
    assert len(births) == 2 * (len(left.parents) + 2)
    assert len(security) == len(left.parents) + 2
    assert security.count(left.name) == security.count(right.name) == 1
    for handle in current:
        with pytest.raises(OSError) as invalid:
            native.info(handle)
        assert invalid.value.winerror == 6
    assert len(volumes) == 2, {"volume_queries": len(volumes), "opens": len(births)}


def test_wide_tree_checks_custody_without_comparing_unrelated_births(tmp_path, monkeypatch):
    leaves = []
    for index in range(32):
        directory = tmp_path / str(index)
        directory.mkdir()
        leaf = directory / "leaf"
        leaf.write_bytes(b"ok")
        leaves.append(leaf)

    comparisons = 0

    def equal(left, right):
        nonlocal comparisons
        comparisons += 1
        return left is right

    monkeypatch.setattr(windows_files._AdmissionMetadataHandle, "__eq__", equal)
    observed = windows_files.WindowsOS().stat_many_for_admission(leaves)
    assert all(observed[leaf][0].st_size == 2 for leaf in leaves)
    assert comparisons == 0


@pytest.mark.parametrize(
    "fault",
    [
        "volume",
        "reparse",
        "non_disk",
        "security",
        "foreign",
        "closed",
        "failed",
        "unqualified",
        "non_directory",
        "wrong_handle",
        "other_observation",
        "wrong_path",
        "parent_subclass",
        "child_subclass",
    ],
)
@pytest.mark.parametrize("close_fails", [False, True])
def test_child_proof_refuses_invalid_lineage_and_retires_exact_birth(
    tmp_path, monkeypatch, fault, close_fails
):
    native, win = windows_files._native(), windows_files.WindowsOS()
    selected = tmp_path / "proof-child"
    selected.write_bytes(b"child")
    original_open, original_close = native.open_handle, native.kernel.CloseHandle
    original_create, original_info = native.nt.NtCreateFile, native.info
    original_file_info = native.kernel.GetFileInformationByHandle
    original_type, original_security = native.kernel.GetFileType, native.security
    records, births, current = [], [], {}
    target = None
    changed = False
    child_opens = 0

    def created(*args):
        result = original_create(*args)
        if result >= 0:
            handle = C.cast(args[0], C.POINTER(windows_files._HANDLE)).contents.value
            info = original_info(handle)
            row = {
                "handle": handle,
                "live": True,
                "closes": 0,
                "identity": (info.volume, (info.index_high << 32) | info.index_low),
            }
            assert handle not in current or not current[handle]["live"]
            births.append(row)
            current[handle] = row
        return result

    def opened(name, **kwargs):
        nonlocal target, changed, child_opens
        record = kwargs.get("_metadata_record")
        if record is not None:
            records.append(record)
        if name == selected.name:
            child_opens += 1
        if (
            name == selected.name
            and not changed
            and (fault != "security" or child_opens == 2)
        ):
            target, changed = record, True
            parent = record.parent
            if fault == "foreign":
                record.parent = copy.copy(parent)  # Same numeric handle is not custody.
            elif fault == "closed":
                del parent.owner[id(parent)]
                assert original_close(parent.handle)
                current[parent.handle]["live"] = False
                current[parent.handle]["closes"] += 1
                # Use a separately held fd to open the child after revoking
                # the original parent birth's custody and closing its handle.
                kwargs["parent"] = native.handle(borrowed)
            elif fault == "failed":
                parent.close_error = 5
            elif fault == "unqualified":
                parent.qualified = False
            elif fault == "non_directory":
                parent.directory = False
            elif fault == "wrong_handle":
                kwargs["parent"] = native.handle(borrowed)
            elif fault == "other_observation":
                record.owner = list(parent.owner)
            elif fault == "wrong_path":
                parent.path = parent.path.with_name("different-parent")
            elif fault in {"parent_subclass", "child_subclass"}:

                class ForeignRecord(windows_files._AdmissionMetadataHandle):
                    pass

                if fault == "parent_subclass":
                    parent.__class__ = ForeignRecord
                else:
                    record.__class__ = ForeignRecord
        return original_open(name, **kwargs)

    def is_target(handle):
        return target is not None and target.handle == getattr(handle, "value", handle)

    def file_info(handle, pointer):
        result = original_file_info(handle, pointer)
        if result and is_target(handle):
            value = C.cast(pointer, C.POINTER(windows_files._FileInfo)).contents
            if fault == "volume":
                value.volume ^= 1
            elif fault == "reparse":
                value.attributes |= windows_files._REPARSE
        return result

    def file_type(handle):
        return 2 if fault == "non_disk" and is_target(handle) else original_type(handle)

    def security(handle, *args, **kwargs):
        if fault == "security" and is_target(handle):
            raise OSError("controlled_security_failure")
        return original_security(handle, *args, **kwargs)

    def closed(handle):
        row = current[getattr(handle, "value", handle)]
        row["closes"] += 1
        assert row["closes"] == 1
        if close_fails and is_target(handle):
            C.set_last_error(5)
            return 0
        result = original_close(handle)
        if result:
            row["live"] = False
        return result

    borrowed = win.open(tmp_path, win.O_RDONLY | win.O_DIRECTORY)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(native, "open_handle", opened)
            patch.setattr(native.nt, "NtCreateFile", created)
            patch.setattr(native.kernel, "GetFileInformationByHandle", file_info)
            patch.setattr(native.kernel, "GetFileType", file_type)
            patch.setattr(native, "security", security)
            patch.setattr(native.kernel, "CloseHandle", closed)
            with pytest.raises(OSError) as caught:
                win.stat_many_for_admission((selected,))
            assert changed and target.handle is not None
            if close_fails:
                assert type(caught.value) is windows_files._AdmissionMetadataCloseError
                assert caught.value.failed_handles == (target,)
                assert target.close_error == 5
                assert isinstance(caught.value.__context__, OSError)
            else:
                reason = {
                    "reparse": "windows_reparse_point_refused",
                    "non_disk": "windows_non_disk_object_refused",
                    "security": "controlled_security_failure",
                }.get(fault, "windows_admission_volume_proof_invalid")
                assert reason in str(caught.value)
            assert all(row["closes"] == 1 for row in births)
            assert sum(row["live"] for row in births) == int(close_fails)
    finally:
        # Reclaim only the exact still-current retained birth, after removing
        # injections; never retry a numeric handle from a retired incarnation.
        if target is not None and target.handle in current:
            row = current[target.handle]
            if row["live"]:
                info = original_info(target.handle)
                assert (info.volume, (info.index_high << 32) | info.index_low) == row[
                    "identity"
                ]
                assert original_close(target.handle)
                row["live"] = False
        win.close(borrowed)
    assert births and not any(row["live"] for row in births)


@pytest.mark.parametrize("route", ["ordinary", "borrowed", "custom", "non_issued"])
def test_non_metadata_opens_keep_full_volume_qualification(
    tmp_path, monkeypatch, route
):
    native, win = windows_files._native(), windows_files.WindowsOS()
    selected = tmp_path / "ordinary-child"
    selected.write_bytes(b"ordinary")
    parent = win.open(tmp_path, win.O_RDONLY | win.O_DIRECTORY)
    original, original_open = native.ntfs, native.open_handle
    qualifications, opened = [], []

    def ntfs(handle):
        result = original(handle)
        qualifications.append(getattr(handle, "value", handle))
        return result

    def opening(*args, **kwargs):
        handle = original_open(*args, **kwargs)
        opened.append(handle)
        return handle

    try:
        with monkeypatch.context() as patch:
            patch.setattr(native, "ntfs", ntfs)
            patch.setattr(native, "open_handle", opening)
            if route == "ordinary":
                assert win.stat(selected).st_size == 8
            elif route == "borrowed":
                assert win.stat(selected.name, dir_fd=parent).st_size == 8
            else:
                record = (
                    windows_files._AdmissionMetadataHandle(native, selected)
                    if route == "non_issued"
                    else None
                )
                handle = native.open_handle(
                    selected.name,
                    parent=native.handle(parent),
                    metadata=True,
                    _metadata_record=record,
                )
                try:
                    assert native.info(handle).size_low == 8
                finally:
                    assert native.kernel.CloseHandle(handle)
            assert opened and qualifications == opened
    finally:
        win.close(parent)


def test_independent_drive_roots_each_qualify_on_both_passes(monkeypatch):
    native, win = windows_files._native(), windows_files.WindowsOS()
    roots = {}
    for drive in os.listdrives():
        path = windows_files.Path(drive)
        try:
            info = win.stat(path)
        except (OSError, ValueError):
            continue
        roots.setdefault(info.st_dev, path)
        if len(roots) == 2:
            break
    if len(roots) != 2:
        pytest.skip("two accessible distinct local NTFS volumes required")
    original = native.ntfs
    qualifications = []

    def ntfs(handle):
        result = original(handle)
        qualifications.append(native.info(handle).volume)
        return result

    monkeypatch.setattr(native, "ntfs", ntfs)
    observed = win.stat_many_for_admission(tuple(roots.values()))
    assert {entry[0].st_dev for entry in observed.values()} == set(roots)
    assert len(qualifications) == 4
    assert all(qualifications.count(volume) == 2 for volume in roots)
