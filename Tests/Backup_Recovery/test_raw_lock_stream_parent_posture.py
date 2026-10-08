"""Actual lock creation retains fresh parent and ancestor posture checks."""

import os
import stat
import sys
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_raw_native_retirement as retirement_cases
from Tests.Backup_Recovery.config_test_support import install_config_source
from tldw_chatbook.Agents import hook_permissions
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import private_paths

local_root = retirement_cases.local_root


def _public_posture(path, *, writable):
    if os.name == "nt":
        from Tests.Utils.test_windows_native_admission import _replace_security
        from tldw_chatbook.Utils.windows_files import _native

        private = f"D:P(A;;FA;;;{_native().user_sid})(A;;FA;;;SY)(A;;FA;;;BA)"
        _replace_security(
            path, private + ("(A;;FA;;;WD)" if writable else "(A;;FR;;;WD)")
        )
    else:
        path.chmod(0o777 if writable else 0o755)
    info = private_paths.os.stat(path, follow_symlinks=False)
    assert info.st_mode & 0o044
    assert bool(info.st_mode & 0o022) is writable
    assert not info.st_mode & stat.S_ISVTX


@pytest.fixture
def lock_case(tmp_path, monkeypatch, local_root):
    # Both directories belong solely to this fixture, never pytest's shared root.
    ancestor, parent = tmp_path / "ancestor", tmp_path / "ancestor" / "private"
    ancestor.mkdir(mode=0o700)
    parent.mkdir(mode=0o700)
    private_paths.os.chmod(ancestor, 0o700)
    private_paths.os.chmod(parent, 0o700)
    ancestor_fd = private_paths.os.open(
        ancestor, private_paths.os.O_RDONLY | private_paths.os.O_DIRECTORY
    )
    parent_fd = private_paths.os.open(
        parent, private_paths.os.O_RDONLY | private_paths.os.O_DIRECTORY
    )
    source = None
    try:
        selected = tmp_path / "config.toml"
        selected.write_text(
            '[general]\nusers_name="private"\n[paths]\ndata_dir="'
            + ancestor.as_posix()
            + '"\n',
            encoding="utf-8",
        )
        private_paths.os.chmod(selected, 0o600)
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))
        config = install_config_source(monkeypatch)
        assert config.get_user_data_dir() == parent
        monkeypatch.setattr(hook_permissions, "config", config)
        source = hook_permissions.HookPermissions()
        store_path = parent / "hook_permissions.json"
        lock = store_path.with_name(store_path.name + ".lock")
        assert not lock.exists()
        yield SimpleNamespace(
            source=source,
            config=config,
            selected=selected,
            store_path=store_path,
            lock=lock,
            ancestor=ancestor,
            parent=parent,
            ancestor_fd=ancestor_fd,
            parent_fd=parent_fd,
        )
    finally:
        if source is not None:
            source.close()
        # Restore through retained handles even when pathname trust was revoked.
        try:
            private_paths.os.fchmod(ancestor_fd, 0o700)
            private_paths.os.fchmod(parent_fd, 0o700)
        finally:
            private_paths.os.close(parent_fd)
            private_paths.os.close(ancestor_fd)


@pytest.mark.parametrize("race", ["warm-parent", "cold-ancestor"])
def test_actual_lock_refuses_post_creation_posture_drift(lock_case, monkeypatch, race):
    case = lock_case
    if race == "warm-parent":
        private_paths.create_private_text(
            case.lock, "existing", application_owned_directory=case.parent
        )
    else:
        _public_posture(case.ancestor, writable=False)
    initial_parent = private_paths.os.fstat(case.parent_fd)
    initial_ancestor = private_paths.os.fstat(case.ancestor_fd)
    config_bytes = case.selected.read_bytes()
    original_open = private_paths._native_open
    open_code = original_open.__code__
    original_fsync = private_paths.os.fsync
    previous = sys.getprofile()
    changed, synced, operations = [], [], []
    stream = None
    refusal = None

    def mutate(path, fd, *, writable):
        before = private_paths.os.fstat(fd)
        _public_posture(path, writable=writable)
        after = private_paths.os.fstat(fd)
        assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
        changed.append(stat.S_IMODE(after.st_mode))

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event != "return" or frame.f_code is not open_code or changed:
            return
        args, outcome = frame.f_locals.get("args", ()), frame.f_locals.get("_outcome")
        if (
            len(args) > 1
            and args[0] == case.lock.name
            and args[1] & private_paths.os.O_EXCL
            and outcome is not None
            and outcome.rejected
            and outcome.descriptor is None
        ):
            mutate(case.parent, case.parent_fd, writable=False)

    def sync_then_mutate(fd):
        result = original_fsync(fd)
        actual = private_paths.os.fstat(fd)
        if stat.S_ISREG(actual.st_mode) and not synced:
            named = private_paths.os.stat(case.lock, follow_symlinks=False)
            if (actual.st_dev, actual.st_ino) == (named.st_dev, named.st_ino):
                synced.append((fd, actual.st_dev, actual.st_ino))
                mutate(case.ancestor, case.ancestor_fd, writable=True)
        return result

    with retirement_cases._original_tracked_closes(case.source) as retirement:
        with (
            case.config.locked_hooks_config_snapshot() as config_snapshot,
            raw._scope(
                case.source,
                "hook_permissions",
                writing=True,
                selected_read=case.store_path,
            ) as operation,
        ):
            assert config_snapshot.profile_data_dir == case.parent
            state = raw._states[operation]
            leases = tuple(state.leases)
            operations.append(operation)
            with monkeypatch.context() as injection:
                prior_observer = sys.getprofile()
                # Chain the native retirement observer, never replace its evidence.
                previous = prior_observer
                if race == "warm-parent":
                    sys.setprofile(observe)
                else:
                    injection.setattr(private_paths.os, "fsync", sync_then_mutate)
                try:
                    try:
                        stream = private_paths.open_private_lock_stream(
                            case.lock, application_owned_directory=case.parent
                        )
                    except (OSError, ValueError, RuntimeError) as error:
                        refusal = error
                finally:
                    sys.setprofile(prior_observer)
                    if stream is not None:
                        stream.close()
                        assert stream.closed
    assert private_paths._native_open is original_open
    assert original_open.__code__ is open_code
    assert len(changed) == 1, (
        "the actual native rejection/fsync boundary was not reached",
        repr(refusal),
    )
    if race == "warm-parent":
        assert not synced
        assert case.lock.read_bytes() == b"existing"
    else:
        assert len(synced) == 1, "the actual created lock FD must be synchronized"
        current = private_paths.os.stat(case.lock, follow_symlinks=False)
        assert (current.st_dev, current.st_ino) == synced[0][1:]
        assert case.lock.read_bytes() == b""
    parent, ancestor = (
        private_paths.os.fstat(case.parent_fd),
        private_paths.os.fstat(case.ancestor_fd),
    )
    assert (parent.st_dev, parent.st_ino) == (
        initial_parent.st_dev,
        initial_parent.st_ino,
    )
    assert (ancestor.st_dev, ancestor.st_ino) == (
        initial_ancestor.st_dev,
        initial_ancestor.st_ino,
    )
    assert case.selected.read_bytes() == config_bytes
    assert not state.active and not state.uncertain
    assert not state.pins and not state.files and not state.descriptors
    assert all(lease not in storage._live_leases for lease in leases)
    counts, closed, removed, active, retired = retirement
    assert counts["closes"] > 0 and len(closed) == counts["closes"] and all(closed)
    assert all(removed) and not active
    assert all(item not in raw._states for item in (*operations, *retired))
    assert all(item not in storage._raw_operations for item in (*operations, *retired))
    # Draft RED occurs only after the real race, unchanged identities and cleanup.
    assert refusal is not None, "lock stream accepted changed parent/ancestor posture"
