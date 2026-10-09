"""Fresh user-directory posture uses one original shared native tree."""

import os
import stat
import sys

import pytest

from Tests.Backup_Recovery import test_raw_native_retirement as retirement_cases
from tldw_chatbook.Backup_Recovery import storage_admission as storage

configured_source = retirement_cases.configured_source
local_root = retirement_cases.local_root


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows metadata schedule")
def test_original_user_directory_stamps_share_one_native_tree(
    configured_source, request
):
    from Tests.Backup_Recovery.test_windows_snapshot_metadata_work import (
        _metadata_counts,
    )
    from tldw_chatbook.Utils import windows_files

    source = configured_source
    user = source.get_user_data_dir()
    left, right = user / "left", user / "right"
    left.mkdir()
    right.mkdir()
    # The real input is an ancestor chain. Siblings and an out-of-order repeated
    # entry also require the helper to preserve tuple order and duplicates.
    paths = (*storage._chain(user), right, left, right)
    expected = tuple(storage._posture(path) for path in paths)
    assert all(stamp is not None and isinstance(stamp[-1], bytes) for stamp in expected)
    native, facade = windows_files._native(), storage.os
    assert type(facade) is windows_files.WindowsOS
    original = source._user_data_dir_stamps
    original_code = original.__code__
    with _metadata_counts(native, facade) as (counts, callers, handles):
        observed = source._user_data_dir_stamps(paths)
    for handle in handles:
        with pytest.raises(OSError) as error:
            native.info(handle)
        assert error.value.winerror == 6
    assert source._user_data_dir_stamps is original
    assert original.__code__ is original_code
    assert observed == expected
    assert len(observed) == len(paths)
    assert observed[-3] == observed[-1]
    assert observed[-2][:2] != observed[-1][:2]
    assert handles and len(handles) == counts["open_handle"]
    nodes = {node for path in paths for node in (*path.parents, path)}
    request.node.user_properties.append(
        (
            "user_directory_metadata_schedule",
            dict(
                requested_entries=len(paths),
                unique_tree_nodes=len(nodes),
                scalar_tree_nodes=sum(len(path.parents) + 1 for path in paths),
                starts=dict(counts),
                info_callers=dict(callers),
                opened_incarnations=len(handles),
                physical_handles_retired=True,
                exact_ordered_stamps=True,
                original_sources_and_monitor_retired=True,
            ),
        )
    )
    # Payload, duplicate ordering and actual native retirement precede the RED.
    assert counts["open_handle"] == 2 * len(nodes)
    assert counts["stat_many_for_admission"] == 1


@pytest.mark.parametrize("shape", ["empty", "missing", "not-directory"])
def test_user_directory_stamp_absence_and_error_parity(configured_source, shape):
    source = configured_source
    user = source.get_user_data_dir()
    missing = user / "absent"
    if shape == "empty":
        if os.name == "nt":
            from Tests.Backup_Recovery.test_windows_snapshot_metadata_work import (
                _metadata_counts,
            )
            from tldw_chatbook.Utils import windows_files

            with _metadata_counts(windows_files._native(), storage.os) as (
                counts,
                _,
                handles,
            ):
                assert source._user_data_dir_stamps(()) == ()
            assert not counts and not handles
        else:
            assert source._user_data_dir_stamps(()) == ()
    elif shape == "missing":
        paths = (user, missing, missing / "child", missing)
        expected = (storage._posture(user), None, None, None)
        assert source._user_data_dir_stamps(paths) == expected
        assert not missing.exists()
    else:
        regular = user / "regular-file"
        regular.write_bytes(b"unchanged")
        # A real native NotADirectory error makes the optional stamp unavailable;
        # it must not escape instead of the resolver's original refusal.
        assert source._user_data_dir_stamps((user, regular / "child")) is None
        assert regular.read_bytes() == b"unchanged"


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows owner/DACL bytes")
@pytest.mark.parametrize("target", ["leaf", "ancestor"])
def test_user_directory_stamps_observe_exact_acl_change(configured_source, target):
    from Tests.Utils.test_windows_native_admission import _replace_security
    from tldw_chatbook.Utils import windows_files

    source = configured_source
    user = source.get_user_data_dir()
    changed = user if target == "leaf" else user.parent
    storage.os.chmod(changed, 0o700)
    descriptor = storage.os.open(changed, storage.os.O_RDONLY | storage.os.O_DIRECTORY)
    paths = (user.parent, user, user.parent)
    try:
        before = source._user_data_dir_stamps(paths)
        native = windows_files._native()
        # A redundant trusted administrator ACE changes the actual descriptor
        # without broadening the projected private mode or changing its owner.
        _replace_security(
            changed,
            f"D:P(A;;FA;;;{native.user_sid})(A;;FA;;;SY)(A;;FA;;;BA)(A;;FR;;;BA)",
        )
        after = source._user_data_dir_stamps(paths)
        index = 1 if target == "leaf" else 0
        assert before[index][:5] == after[index][:5]
        assert before[index][-1] != after[index][-1]
        assert after == tuple(storage._posture(path) for path in paths)
        assert after[0] == after[2]
        assert before[1 - index] == after[1 - index]
    finally:
        # Restore through the exact retained directory even if a test fails.
        try:
            storage.os.fchmod(descriptor, 0o700)
        finally:
            storage.os.close(descriptor)


def test_actual_user_directory_memo_rechecks_changed_private_posture(configured_source):
    source = configured_source
    user = source.get_user_data_dir()
    for _ in range(3):
        assert source.get_user_data_dir() == user
    assert source._USER_DATA_DIR_MEMO is not None
    original = source._resolve_user_data_dir
    code = original.__code__
    calls = []
    previous = sys.getprofile()
    descriptor = storage.os.open(user, storage.os.O_RDONLY | storage.os.O_DIRECTORY)

    def observe(frame, event, value):
        if previous is not None:
            previous(frame, event, value)
        if event == "call" and frame.f_code is code:
            calls.append(True)

    sys.setprofile(observe)
    try:
        assert source.get_user_data_dir() == user
        assert calls == [], "the actual warm memo did not serve its ordinary caller"
        before = storage.os.fstat(descriptor)
        if os.name == "nt":
            from Tests.Utils.test_windows_native_admission import _replace_security
            from tldw_chatbook.Utils import windows_files

            native = windows_files._native()
            _replace_security(
                user,
                f"D:P(A;;FA;;;{native.user_sid})(A;;FA;;;SY)(A;;FA;;;BA)(A;;FR;;;WD)",
            )
        else:
            storage.os.fchmod(descriptor, 0o755)
        changed = storage.os.fstat(descriptor)
        assert (changed.st_dev, changed.st_ino) == (before.st_dev, before.st_ino)
        assert stat.S_IMODE(changed.st_mode) & 0o077
        assert source.get_user_data_dir() == user
        assert calls == [True], "changed posture must reach the original resolver"
        assert stat.S_IMODE(storage.os.fstat(descriptor).st_mode) == 0o700
        assert source._resolve_user_data_dir is original and original.__code__ is code
    finally:
        sys.setprofile(previous)
        try:
            storage.os.fchmod(descriptor, 0o700)
        finally:
            storage.os.close(descriptor)
