"""Config companion paths share admission without enlarging native scope."""

import pytest

from Tests.Backup_Recovery.test_bootstrap import local_scope  # noqa: F401
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.Backup_Recovery.control_records import bind_profile, register_pending


def test_related_paths_retain_native_exclusion_until_close(local_scope):  # noqa: F811
    root, config, data, authority = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    primary, companion = data / "config.toml", data / "config.toml.lock"
    with storage.acquire_storage(primary, related_paths=(companion,)) as lease:
        assert lease.execution_context(primary)[1]
        with pytest.raises(bootstrap.RecoveryRequired, match="execution_selection_changed"):
            lease.execution_context(companion)
        with pytest.raises(AdmissionTimeout), authority.maintenance(("profile",), 0.02):
            pytest.fail("live config admission allowed maintenance")
    with authority.maintenance(("profile",), 0.1):
        assert not primary.exists() and not companion.exists()


@pytest.mark.parametrize("position", [0, 1])
def test_every_related_path_requires_enrollment(local_scope, position):  # noqa: F811
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    related = [data / "allowed.tmp", data.parent / "outside.tmp"]
    if position == 0:
        related.reverse()
    before = set(storage._live_leases)
    with pytest.raises(bootstrap.RecoveryRequired, match="storage_scope_not_enrolled"):
        storage.acquire_storage(data / "config.toml", related_paths=tuple(related))
    assert storage._live_leases == before
    assert not any(path.exists() for path in related)


def test_file_enrollment_does_not_admit_related_sidecar(local_scope):  # noqa: F811
    root, config, data, authority = local_scope
    selected = data / "only.toml"
    selected.write_text("preserved")
    authority.register("file-only", (selected, config))
    bind_profile(root, config, ("file-only",), root / "admission")
    with pytest.raises(bootstrap.RecoveryRequired, match="storage_scope_not_enrolled"):
        storage.acquire_storage(selected, related_paths=(selected.with_suffix(".tmp"),))
    assert selected.read_text() == "preserved"


def test_related_paths_recheck_pending_recovery_after_native_acquisition(
    local_scope, monkeypatch,  # noqa: F811
):
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    original = bootstrap.startup_permission
    calls = 0

    def permission(selector, selected_root):
        nonlocal calls
        calls += 1
        if calls == 2:
            register_pending(root, "raced", ("profile",), root / "control", (config,))
        return original(selector, selected_root)

    monkeypatch.setattr(bootstrap, "startup_permission", permission)
    before = set(storage._live_leases)
    with pytest.raises(bootstrap.RecoveryRequired, match="recovery_pending"):
        storage.acquire_storage(data / "config.toml", related_paths=(data / "copy.tmp",))
    assert storage._live_leases == before


def test_related_paths_refuse_local_pause_during_acquisition(
    local_scope, monkeypatch,  # noqa: F811
):
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    original = storage._scope
    pauses = []

    def scoped(*args, **kwargs):
        result = original(*args, **kwargs)
        if not pauses:
            pauses.append(storage._begin_local_pause())
        return result

    monkeypatch.setattr(storage, "_scope", scoped)
    before = set(storage._live_leases)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            storage.acquire_storage(data / "config.toml", related_paths=(data / "copy.tmp",))
        assert storage._live_leases == before
    finally:
        for pause in pauses:
            pause.resume()


def test_related_path_moved_outside_enrollment_is_rechecked_after_acquisition(
    local_scope, monkeypatch,  # noqa: F811
):
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    directory = data / "copies"
    directory.mkdir(mode=0o700)
    outside = data.parent / "outside"
    outside.mkdir(mode=0o700)
    original = bootstrap.startup_permission
    calls = 0

    def permission(selector, selected_root):
        nonlocal calls
        calls += 1
        if calls == 2:
            directory.rename(data / "original-copies")
            directory.symlink_to(outside, target_is_directory=True)
        return original(selector, selected_root)

    monkeypatch.setattr(bootstrap, "startup_permission", permission)
    before = set(storage._live_leases)
    with pytest.raises(bootstrap.RecoveryRequired, match="storage_scope_not_enrolled"):
        storage.acquire_storage(data / "config.toml", related_paths=(directory / "copy.tmp",))
    assert storage._live_leases == before
    assert list(outside.iterdir()) == []
