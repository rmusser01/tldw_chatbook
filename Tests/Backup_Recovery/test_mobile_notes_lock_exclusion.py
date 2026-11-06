"""Exact native device-local Notes lock representation, no sync authority."""

import os

import pytest

from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
from tldw_chatbook.Notes.recovery import _DeviceStateExclusion


def configured(tmp_path):
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    config = {"paths": {"data_dir": str(data)}, "general": {"users_name": "fixture"}}
    parent = user_data_dir(config)
    parent.mkdir(mode=0o700)
    config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(
        tmp_path / "config.toml", "fixture"
    )
    return config, parent / "notes_sync_locks"


def lock_item(config, root):
    return next(
        item for item in _DeviceStateExclusion().discover(config) if item.path == root
    )


def test_native_absent_empty_and_released_lock_are_excluded(tmp_path):
    from tldw_chatbook.Notes.notes_sync_coordinator import NotesSyncRootCoordinator

    config, root = configured(tmp_path)
    assert lock_item(config, root).status == "intentionally_excluded"
    coordinator = NotesSyncRootCoordinator(root)
    assert lock_item(config, root).status == "intentionally_excluded"
    target = tmp_path / "actual-notes-root"
    target.mkdir(mode=0o700)
    admission = coordinator.try_acquire(target)
    lease = admission.lease
    assert lease is not None and lease.authoritative
    handle = lease._handle
    coordinator.release(lease)
    assert handle.closed
    with pytest.raises(ValueError, match="closed"):
        handle.fileno()
    names = tuple(root.iterdir())
    assert len(names) == 1 and names[0].stat().st_size == 0
    assert lock_item(config, root).status == "intentionally_excluded"


@pytest.mark.parametrize(
    "kind",
    [
        "name",
        "uppercase",
        "suffix",
        "nonempty",
        "nested",
        "symlink",
        "hardlink",
        "mode",
        "root_mode",
        "root_link",
        "root_file",
    ],
)
def test_foreign_or_malformed_lock_representation_blocks(tmp_path, kind):
    config, root = configured(tmp_path)
    root.mkdir(mode=0o700)
    leaf = root / ("a" * 64 + ".lock")
    if kind == "root_mode":
        root.chmod(0o755)
    elif kind == "root_link":
        other = root.with_name("other")
        root.rename(other)
        root.symlink_to(other, target_is_directory=True)
    elif kind == "root_file":
        root.rmdir()
        root.write_bytes(b"")
        root.chmod(0o600)
    elif kind == "nested":
        leaf.mkdir(mode=0o700)
    elif kind == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(b"")
        leaf.symlink_to(outside)
    else:
        if kind == "name":
            leaf = root / "unexpected"
        elif kind == "uppercase":
            leaf = root / ("A" * 64 + ".lock")
        elif kind == "suffix":
            leaf = root / ("a" * 64 + ".lock.extra")
        leaf.write_bytes(b"foreign" if kind == "nonempty" else b"")
        leaf.chmod(0o644 if kind == "mode" else 0o600)
        if kind == "hardlink":
            os.link(leaf, tmp_path / "retained-alias")
    assert lock_item(config, root).status in {"unsupported", "unavailable"}


def test_changed_pinned_root_is_unavailable(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import file_inventory

    config, root = configured(tmp_path)
    root.mkdir(mode=0o700)
    original = file_inventory.inventory_tree

    def replace_after_inventory(path, **kwargs):
        result = original(path, **kwargs)
        path.rename(path.with_name("retained-original"))
        path.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(file_inventory, "inventory_tree", replace_after_inventory)
    assert lock_item(config, root).status == "unavailable"
    assert root.with_name("retained-original").is_dir()


def test_native_inventory_limit_is_not_exclusion(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import file_inventory

    config, root = configured(tmp_path)
    root.mkdir(mode=0o700)
    for name in ("a", "b"):
        leaf = root / (name * 64 + ".lock")
        leaf.touch(mode=0o600)
    monkeypatch.setattr(file_inventory, "MAX_ENTRIES", 2)
    assert lock_item(config, root).status == "unavailable"


@pytest.mark.parametrize("exhaust", [False, True])
def test_current_allowance_and_real_census_retirement(tmp_path, monkeypatch, exhaust):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )

    config, root = configured(tmp_path)
    root.mkdir(mode=0o700)
    (root / ("a" * 64 + ".lock")).touch(mode=0o600)
    expected = lock_item(config, root)
    budget = _TraceReadBudget()
    original = bootstrap._bounded_names
    calls = []

    def census(parent, maximum, actual, custody):
        assert actual is budget
        result = original(parent, maximum, actual, custody)
        assert not custody.directory_iterators
        calls.append((actual, custody))
        if exhaust:
            actual.reserve(actual._remaining)
        return result

    monkeypatch.setattr(bootstrap, "_bounded_names", census)
    with storage_admission._preview_reads(_read_budget=budget):
        scope = storage_admission._local.preview_scope
        custody = scope.custody
        if exhaust:
            with pytest.raises(_TraceReadLimit):
                lock_item(config, root)
        else:
            assert lock_item(config, root) == expected
        assert (
            not custody.fds
            and not custody.directory_iterators
            and not custody.close_failed
        )
    assert calls and not scope.active and not scope.resources
    assert budget._remaining < 16 * 1024**2


def test_zero_remaining_refuses_before_directory_open(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import native_files, storage_admission
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )

    config, root = configured(tmp_path)
    budget = _TraceReadBudget()
    with storage_admission._preview_reads(_read_budget=budget):
        scope = storage_admission._local.preview_scope
        budget.reserve(budget._remaining)

        def forbidden(*args, **kwargs):
            raise AssertionError("opened after zero allowance")

        monkeypatch.setattr(native_files, "pinned_directory", forbidden)
        with pytest.raises(_TraceReadLimit):
            lock_item(config, root)
        assert not scope.custody.fds
    assert not scope.resources


def test_unobservable_census_blocks(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import file_inventory

    config, root = configured(tmp_path)
    root.mkdir(mode=0o700)

    def unavailable(*args, **kwargs):
        raise OSError("native census unavailable")

    monkeypatch.setattr(file_inventory, "inventory_tree", unavailable)
    assert lock_item(config, root).status == "unavailable"


def test_changed_leaf_blocks(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import file_inventory

    config, root = configured(tmp_path)
    root.mkdir(mode=0o700)
    leaf = root / ("a" * 64 + ".lock")
    leaf.touch(mode=0o600)
    original = file_inventory.inventory_tree

    def change_after_inventory(path, **kwargs):
        result = original(path, **kwargs)
        leaf.write_bytes(b"foreign")
        return result

    monkeypatch.setattr(file_inventory, "inventory_tree", change_after_inventory)
    assert lock_item(config, root).status == "unsupported"
    assert leaf.read_bytes() == b"foreign"


def test_replaced_ancestor_cannot_hide_current_foreign_payload(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import file_inventory

    config, root = configured(tmp_path)
    root.mkdir(mode=0o700)
    original = file_inventory.inventory_tree
    retained = root.parent.with_name("retained-parent")

    def replace_parent_after_inventory(path, **kwargs):
        result = original(path, **kwargs)
        path.parent.rename(retained)
        path.parent.mkdir(mode=0o700)
        path.mkdir(mode=0o700)
        (path / "foreign-payload").write_bytes(b"retain exact bytes")
        return result

    monkeypatch.setattr(
        file_inventory, "inventory_tree", replace_parent_after_inventory
    )
    assert lock_item(config, root).status == "unavailable"
    assert (root / "foreign-payload").read_bytes() == b"retain exact bytes"
    assert (retained / root.name).is_dir()
