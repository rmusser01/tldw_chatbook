"""Windows folder pins (TASK-40608 D1).

Held chains cannot move, walks below a pin start there, a pinned folder's own
stat is always fresh, and a storage pause releases the pins.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows folder pins")

_OLD_NS = 10**18  # 2001-09-09, well before anything this test writes


@pytest.fixture
def windows_files():
    from tldw_chatbook.Utils import windows_files as module

    module.set_pinned_roots(())
    yield module
    module.set_pinned_roots(())


@pytest.fixture
def profile(tmp_path):
    root = tmp_path / "home" / "profile"
    (root / "data").mkdir(parents=True)
    (root / "data" / "notes.db").write_bytes(b"x")
    return root


def _count_opens(monkeypatch, windows_files):
    names = []
    original = windows_files._Native.open_handle

    def counting(self, name, **kwargs):
        names.append(name)
        return original(self, name, **kwargs)

    monkeypatch.setattr(windows_files._Native, "open_handle", counting)
    return names


def test_a_pinned_folder_and_its_parent_cannot_be_renamed_until_release(windows_files, profile):
    windows_files.set_pinned_roots([profile])
    assert len(windows_files.pinned_roots()) == 1
    with pytest.raises(PermissionError):
        os.rename(profile, profile.with_name("moved"))
    with pytest.raises(PermissionError):
        os.rename(profile.parent, profile.parent.with_name("home-moved"))

    windows_files.release_pinned_roots()
    assert windows_files.pinned_roots() == ()
    os.rename(profile, profile.with_name("moved"))


def test_files_and_child_folders_inside_a_pin_stay_usable(windows_files, profile):
    windows_files.set_pinned_roots([profile])
    assert len(windows_files.pinned_roots()) == 1
    (profile / "data" / "new.txt").write_text("ok", encoding="utf-8")
    os.replace(profile / "data" / "new.txt", profile / "data" / "renamed.txt")
    os.remove(profile / "data" / "renamed.txt")
    os.rename(profile / "data", profile / "data2")
    assert (profile / "data2" / "notes.db").is_file()


def test_facade_writes_directly_inside_a_pinned_folder_work(windows_files, profile):
    facade = windows_files.WindowsOS()
    windows_files.set_pinned_roots([profile])
    assert len(windows_files.pinned_roots()) == 1

    fd = facade.open(profile / "a.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, b"abc")
    finally:
        os.close(fd)
    facade.replace(profile / "a.txt", profile / "b.txt")
    facade.rename(profile / "b.txt", profile / "c.txt")
    facade.link(profile / "c.txt", profile / "d.txt")
    facade.unlink(profile / "d.txt")
    facade.mkdir(profile / "sub", 0o700)
    facade.rmdir(profile / "sub")
    facade.utime(profile / "c.txt", ns=(_OLD_NS, _OLD_NS))
    facade.chmod(profile / "c.txt", 0o600)

    written = facade.stat(profile / "c.txt")
    assert (written.st_size, written.st_mtime_ns) == (3, _OLD_NS)
    assert sorted(os.listdir(profile)) == ["c.txt", "data"]


def test_a_walk_below_a_pin_matches_the_full_walk_with_fewer_opens(
    windows_files, profile, monkeypatch
):
    target = profile / "data" / "notes.db"
    checked = windows_files.WindowsOS()
    unpinned = checked.stat(target)
    full_names = _count_opens(monkeypatch, windows_files)
    checked.stat(target)
    full = list(full_names)
    monkeypatch.undo()

    windows_files.set_pinned_roots([profile])
    pinned_names = _count_opens(monkeypatch, windows_files)
    pinned = checked.stat(target)

    assert (pinned.st_ino, pinned.st_dev, pinned.st_size, pinned.st_mode) == (
        unpinned.st_ino, unpinned.st_dev, unpinned.st_size, unpinned.st_mode,
    )
    assert pinned_names == ["data", "notes.db"]
    assert len(full) > len(pinned_names)


def test_a_released_pin_stays_open_until_the_borrowing_walk_returns_it(windows_files, profile):
    windows_files.set_pinned_roots([profile])
    (pin,) = windows_files._pins.values()
    with windows_files._parent(profile / "data") as (parent, leaf):
        assert parent == pin.handle
        assert pin.borrowers == 1
        assert leaf == "data"
        windows_files.release_pinned_roots()
        assert windows_files.pinned_roots() == ()
        # Release defers the close: the walk's handle stays valid, and the
        # folder still cannot move while the walk holds it.
        assert windows_files._native().info(parent).attributes & windows_files._DIRECTORY
        with pytest.raises(PermissionError):
            os.rename(profile, profile.with_name("moved"))
    assert pin.handle is None
    os.rename(profile, profile.with_name("moved"))


def test_restore_takes_the_configured_pins_again(windows_files, profile):
    windows_files.set_pinned_roots([profile])
    windows_files.release_pinned_roots()
    windows_files.restore_pinned_roots()
    assert len(windows_files.pinned_roots()) == 1
    with pytest.raises(PermissionError):
        os.rename(profile, profile.with_name("moved"))


def test_a_missing_root_is_skipped_and_the_walk_falls_back(windows_files, profile):
    windows_files.set_pinned_roots([profile / "absent", profile])
    assert len(windows_files.pinned_roots()) == 1
    assert windows_files.WindowsOS().stat(profile / "data" / "notes.db").st_size == 1


def test_a_junction_in_the_chain_pins_nothing(windows_files, tmp_path):
    import _winapi

    target = tmp_path / "target"
    (target / "profile").mkdir(parents=True)
    link = tmp_path / "link"
    _winapi.CreateJunction(str(target), str(link))

    windows_files.set_pinned_roots([link / "profile"])
    assert windows_files.pinned_roots() == ()
    with pytest.raises(OSError):
        windows_files.WindowsOS().stat(link / "profile")


def test_a_storage_pause_releases_pins_and_resume_takes_them_again(windows_files, profile):
    from tldw_chatbook.Backup_Recovery import storage_admission

    windows_files.set_pinned_roots([profile])
    pause = storage_admission._begin_local_pause()
    try:
        assert storage_admission._pause is pause
        assert windows_files.pinned_roots() == ()
        os.rename(profile, profile.with_name("moved"))
        os.rename(profile.with_name("moved"), profile)
    finally:
        pause.resume()
    assert storage_admission._pause is None
    assert len(windows_files.pinned_roots()) == 1
    with pytest.raises(PermissionError):
        os.rename(profile, profile.with_name("moved"))


def test_a_refused_resume_keeps_the_pins_released(windows_files, profile):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission

    windows_files.set_pinned_roots([profile])
    pause = storage_admission._begin_local_pause()
    try:
        pause._startup_retired = True
        with pytest.raises(bootstrap.RecoveryRequired, match="startup_reacquisition_required"):
            pause.resume()
        assert storage_admission._pause is pause
        assert windows_files.pinned_roots() == ()
    finally:
        # Leave no module-global pause behind for later tests.
        pause._startup_retired = False
        pause.resume()
    assert len(windows_files.pinned_roots()) == 1


def test_a_pinned_folder_is_stat_from_its_pin_without_a_walk(windows_files, profile, monkeypatch):
    checked = windows_files.WindowsOS()
    plain = checked.stat(profile)
    windows_files.set_pinned_roots([profile])
    names = _count_opens(monkeypatch, windows_files)
    pinned = checked.stat(profile)
    assert names == []
    assert (pinned.st_ino, pinned.st_dev, pinned.st_mode) == (plain.st_ino, plain.st_dev, plain.st_mode)


def test_a_pinned_folder_stat_is_fresh_on_every_call(windows_files, profile, monkeypatch):
    checked = windows_files.WindowsOS()
    os.utime(profile, ns=(_OLD_NS, _OLD_NS))
    windows_files.set_pinned_roots([profile])
    names = _count_opens(monkeypatch, windows_files)

    before = checked.stat(profile)
    assert before.st_mtime_ns == _OLD_NS
    (profile / "added.txt").write_bytes(b"x")
    after = checked.stat(profile)

    assert after is not before
    assert after.st_mtime_ns != _OLD_NS
    assert names == []


def test_backup_preview_sees_a_control_record_written_while_it_reads(
    windows_files, tmp_path, monkeypatch
):
    """The before/after ctime bracket in the fixed-control exclusion must work
    while the recovery bootstrap folder is pinned (the backup preview path)."""
    from tldw_chatbook.Backup_Recovery import bootstrap, inventory
    from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE

    root = tmp_path / "recovery-bootstrap"
    root.mkdir()
    windows_files.set_pinned_roots([root])
    assert len(windows_files.pinned_roots()) == 1

    def records_written_during_the_read(folder):
        time.sleep(0.05)  # past the NTFS timestamp granularity
        (folder / "pending-x.json").write_bytes(b"{}")
        return ((), ())

    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setattr(bootstrap, "_control_records", records_written_during_the_read)
    monkeypatch.setattr(
        bootstrap,
        "_registry",
        lambda folder: {UNBOUND_NAMESPACE: {"roots": [str(folder / "unbound-owner")]}},
    )

    (item,) = inventory._fixed_control_exclusion()
    assert item.status == "unavailable"

    # Negative control: an undisturbed read is excluded as intended.
    monkeypatch.setattr(bootstrap, "_control_records", lambda folder: ((), ()))
    (item,) = inventory._fixed_control_exclusion()
    assert item.status == "intentionally_excluded"


def test_ancestor_stats_are_reused_only_within_the_window(windows_files, profile, monkeypatch):
    checked = windows_files.WindowsOS()
    monkeypatch.setattr(windows_files, "_PINNED_STAT_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    first = checked.stat(profile.parent)
    names = _count_opens(monkeypatch, windows_files)
    assert checked.stat(profile.parent) is first
    assert names == []

    monkeypatch.setattr(windows_files, "_PINNED_STAT_SECONDS", 0.0)
    refreshed = checked.stat(profile.parent)
    assert refreshed is not first
    assert names  # the expired result was observed again with a walk


def test_releasing_pins_drops_reused_ancestor_stats(windows_files, profile, monkeypatch):
    checked = windows_files.WindowsOS()
    monkeypatch.setattr(windows_files, "_PINNED_STAT_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    assert len(windows_files.pinned_roots()) == 1
    first = checked.stat(profile.parent)
    assert checked.stat(profile.parent) is first

    windows_files.release_pinned_roots()
    windows_files.restore_pinned_roots()
    assert len(windows_files.pinned_roots()) == 1
    assert checked.stat(profile.parent) is not first


def test_an_ancestor_stat_computed_across_a_release_is_not_reused(
    windows_files, profile, monkeypatch
):
    checked = windows_files.WindowsOS()
    monkeypatch.setattr(windows_files, "_PINNED_STAT_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    # WindowsOS binds its methods onto the instance at construction, so the
    # walk is replaced on this instance, not on the class.
    walk = checked._walk_stat

    def walk_across_a_pause(path, **kwargs):
        result = walk(path, **kwargs)
        windows_files.release_pinned_roots()
        windows_files.restore_pinned_roots()
        return result

    monkeypatch.setattr(checked, "_walk_stat", walk_across_a_pause)
    first = checked.stat(profile.parent)
    monkeypatch.setattr(checked, "_walk_stat", walk)
    assert len(windows_files.pinned_roots()) == 1
    assert checked.stat(profile.parent) is not first


def test_a_descendant_is_not_reused_and_sees_changes_at_once(windows_files, profile):
    checked = windows_files.WindowsOS()
    windows_files.set_pinned_roots([profile])
    target = profile / "data" / "notes.db"
    before = checked.stat(target)
    target.write_bytes(b"longer")
    assert checked.stat(target).st_size == 6 != before.st_size


def test_a_spelling_that_casefolds_to_a_pin_is_a_different_folder(windows_files, tmp_path):
    pinned = tmp_path / "Straße"
    other = tmp_path / "Strasse"
    pinned.mkdir()
    other.mkdir()
    (pinned / "f.txt").write_bytes(b"pinned")
    (other / "f.txt").write_bytes(b"the other folder")
    checked = windows_files.WindowsOS()

    windows_files.set_pinned_roots([pinned])
    assert len(windows_files.pinned_roots()) == 1
    assert checked.stat(other / "f.txt").st_size == len(b"the other folder")
    assert checked.stat(other).st_ino != checked.stat(pinned).st_ino
    assert checked.stat(pinned / "f.txt").st_size == len(b"pinned")


def test_a_missing_folder_spelled_like_a_pin_is_not_found(windows_files, tmp_path):
    pinned = tmp_path / "Straße"
    pinned.mkdir()
    windows_files.set_pinned_roots([pinned])
    checked = windows_files.WindowsOS()
    with pytest.raises(FileNotFoundError):
        checked.stat(tmp_path / "Strasse")
    with pytest.raises(FileNotFoundError):
        checked.stat(tmp_path / "Strasse" / "f.txt")


def test_a_folder_that_is_both_a_pin_and_an_ancestor_of_one_is_fresh(windows_files, profile):
    # In production the config folder is pinned and also contains the pinned
    # recovery-bootstrap folder: the exact pin must win over ancestor reuse.
    checked = windows_files.WindowsOS()
    os.utime(profile, ns=(_OLD_NS, _OLD_NS))
    windows_files.set_pinned_roots([profile, profile / "data"])
    assert len(windows_files.pinned_roots()) == 2
    assert checked.stat(profile).st_mtime_ns == _OLD_NS
    (profile / "added.txt").write_bytes(b"x")
    assert checked.stat(profile).st_mtime_ns != _OLD_NS


def test_a_repeated_root_is_pinned_once_and_fully_released(windows_files, profile):
    windows_files.set_pinned_roots([profile, profile])
    assert len(windows_files.pinned_roots()) == 1
    windows_files.set_pinned_roots(())
    os.rename(profile, profile.with_name("moved"))


@pytest.fixture(scope="module")
def lifecycle_mixin():
    # Import the app before per-test fixtures redirect the profile, as test
    # modules that import TldwCli at collection do.
    from tldw_chatbook.app_lifecycle import LifecycleMixin

    return LifecycleMixin


@pytest.fixture
def profile_roots(tmp_path, monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission

    data = tmp_path / "data"
    settings = tmp_path / "config"
    recovery = settings / "recovery-bootstrap"
    data.mkdir()
    recovery.mkdir(parents=True)
    (recovery / "registry.json").write_bytes(b"{}")
    monkeypatch.setattr(config, "get_user_data_dir", lambda: data)
    monkeypatch.setattr(storage_admission, "effective_config_path", lambda: settings / "config.toml")
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: recovery)
    return data, settings, recovery


def test_profile_pins_hold_the_profile_roots_until_unpinned(
    windows_files, profile_roots, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import storage_admission

    _, _, recovery = profile_roots
    storage_admission.pin_profile_roots()
    assert len(windows_files.pinned_roots()) == 3
    names = _count_opens(monkeypatch, windows_files)
    windows_files.WindowsOS().stat(recovery / "registry.json")
    assert names == ["registry.json"]

    storage_admission.unpin_profile_roots()
    assert windows_files.pinned_roots() == ()
    assert windows_files._pin_roots == ()


@pytest.mark.asyncio
async def test_the_maintenance_monitor_start_pins_the_profile_first(
    lifecycle_mixin, windows_files, profile_roots, monkeypatch
):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import runtime_maintenance

    pins_when_the_monitor_ran = []

    async def monitor_app(app):
        pins_when_the_monitor_ran.append(windows_files.pinned_roots())

    monkeypatch.setattr(runtime_maintenance, "monitor_app", monitor_app)
    app = SimpleNamespace()
    lifecycle_mixin._start_backup_maintenance_monitor(app)
    await app._backup_maintenance_monitor_task
    assert len(pins_when_the_monitor_ran) == 1
    assert len(pins_when_the_monitor_ran[0]) == 3


@pytest.mark.asyncio
async def test_unmount_releases_the_pins_even_when_shutdown_raises(
    lifecycle_mixin, windows_files, profile_roots
):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import storage_admission

    storage_admission.pin_profile_roots()
    assert len(windows_files.pinned_roots()) == 3

    async def shutdown_fails():
        raise RuntimeError("actor pack recovery shutdown failed")

    app = SimpleNamespace(_shutdown_actor_pack_recovery=shutdown_fails)
    with pytest.raises(RuntimeError, match="actor pack recovery shutdown failed"):
        await lifecycle_mixin.on_unmount(app)
    assert windows_files.pinned_roots() == ()
    assert windows_files._pin_roots == ()
