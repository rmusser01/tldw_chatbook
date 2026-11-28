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


def test_a_walk_beside_or_above_a_pin_starts_at_the_kept_folder(
    windows_files, profile, tmp_path, monkeypatch
):
    # TASK-40608: these walks used to open every folder from the drive root.
    beside = profile.parent / "beside.txt"
    above = tmp_path / "above.txt"
    elsewhere = tmp_path / "other" / "deep" / "file.txt"
    elsewhere.parent.mkdir(parents=True)
    for path in (beside, above, elsewhere):
        path.write_bytes(b"abc")
    checked = windows_files.WindowsOS()
    full = {path: checked.stat(path) for path in (beside, above, elsewhere)}

    windows_files.set_pinned_roots([profile])
    names = _count_opens(monkeypatch, windows_files)
    kept = {path: checked.stat(path) for path in (beside, above, elsewhere)}

    assert names == ["beside.txt", "above.txt", "other", "deep", "file.txt"]
    for path in full:
        assert (kept[path].st_ino, kept[path].st_dev, kept[path].st_size, kept[path].st_mode) == (
            full[path].st_ino, full[path].st_dev, full[path].st_size, full[path].st_mode,
        )


def test_every_folder_above_a_pin_is_kept_once_and_released_with_it(
    windows_files, profile, tmp_path, monkeypatch
):
    windows_files.set_pinned_roots([profile, profile / "data"])
    drive, parts = windows_files._split_local(profile)
    chain = [windows_files._pin_key(drive, parts[:depth]) for depth in range(len(parts) + 1)]
    # The drive root, every folder down to the first pin, and that pin as the
    # folder above the second one.
    assert set(windows_files._ancestors) == set(chain)
    kept = list(windows_files._ancestors.values())
    assert all(pin.handle is not None and pin.borrowers == 0 for pin in kept)

    windows_files.release_pinned_roots()
    assert windows_files._ancestors == {}
    assert all(pin.handle is None for pin in kept)
    names = _count_opens(monkeypatch, windows_files)
    with pytest.raises(FileNotFoundError):
        windows_files.WindowsOS().stat(profile.parent / "absent.txt")
    assert len(names) == len(parts) + 1  # the drive root and every folder again
    monkeypatch.undo()
    os.rename(profile.parent, profile.parent.with_name("home-moved"))


def test_a_change_beside_a_pin_is_seen_at_once_through_the_kept_folder(windows_files, profile):
    checked = windows_files.WindowsOS()
    windows_files.set_pinned_roots([profile])
    beside = profile.parent / "beside.txt"
    with pytest.raises(FileNotFoundError):
        checked.stat(beside)
    beside.write_bytes(b"abc")
    assert checked.stat(beside).st_size == 3
    beside.write_bytes(b"longer")
    assert checked.stat(beside).st_size == 6
    os.remove(beside)
    with pytest.raises(FileNotFoundError):
        checked.stat(beside)


def test_a_junction_beside_a_pin_is_still_refused(windows_files, profile, tmp_path):
    import _winapi

    target = tmp_path / "target"
    target.mkdir()
    (target / "f.txt").write_bytes(b"x")
    link = profile.parent / "link"
    windows_files.set_pinned_roots([profile])
    _winapi.CreateJunction(str(target), str(link))
    checked = windows_files.WindowsOS()
    with pytest.raises(OSError):
        checked.stat(link)
    with pytest.raises(OSError):
        checked.stat(link / "f.txt")


def test_a_kept_folder_stays_open_until_the_borrowing_walk_returns_it(windows_files, profile):
    windows_files.set_pinned_roots([profile])
    drive, parts = windows_files._split_local(profile.parent)
    kept = windows_files._ancestors[windows_files._pin_key(drive, parts)]
    with windows_files._parent(profile.parent / "beside.txt") as (parent, leaf):
        assert (parent, leaf, kept.borrowers) == (kept.handle, "beside.txt", 1)
        windows_files.release_pinned_roots()
        assert windows_files._ancestors == {}
        assert windows_files._native().info(parent).attributes & windows_files._DIRECTORY
    assert kept.handle is None


def test_the_drive_root_is_never_handed_out_through_its_kept_handle(windows_files, profile):
    # WindowsOS.open duplicates the root handle it is given: a duplicate of the
    # kept one would share its directory listing position with every caller.
    windows_files.set_pinned_roots([profile])
    drive, _ = windows_files._split_local(profile)
    kept = windows_files._ancestors[windows_files._pin_key(drive, [])]
    with windows_files._parent(drive + "\\") as (parent, leaf):
        assert leaf is None
        assert parent != kept.handle
        assert kept.borrowers == 0
    checked = windows_files.WindowsOS()
    listings = []
    for _ in range(2):
        fd = checked.open(drive + "\\", os.O_RDONLY | checked.O_DIRECTORY)
        try:
            listings.append(sorted(checked.listdir(fd)))
        finally:
            os.close(fd)
    assert listings[0] == listings[1] != []


def test_a_pin_its_own_path_no_longer_leads_to_keeps_no_folders(
    windows_files, profile, tmp_path, monkeypatch
):
    # A pin opened while a folder of its chain was being moved names a folder
    # that the walk from the drive root does not reach. Nothing above it may
    # be kept: those folders are not the ones the pin holds in place.
    other = tmp_path / "other"
    other.mkdir()
    pin_one = windows_files._pin_one

    def pin_of_another_folder(path):
        key, _ = pin_one(path)
        windows_files._native().kernel.CloseHandle(_.handle)
        return key, pin_one(other)[1]

    monkeypatch.setattr(windows_files, "_pin_one", pin_of_another_folder)
    windows_files.set_pinned_roots([profile])
    assert len(windows_files.pinned_roots()) == 1
    assert windows_files._ancestors == {}


@pytest.fixture
def trusted_walk(windows_files, monkeypatch):
    """``native_files.pinned_directory`` with its memo cleared and its checks recorded."""
    from tldw_chatbook.Backup_Recovery import native_files

    native_files._pinned_ancestors.clear()
    checked, unsafe = [], set()
    fstat = native_files.os.fstat

    def recording_fstat(fd):
        info = fstat(fd)
        checked.append(info.st_ino)
        if info.st_ino in unsafe:
            return os.stat_result((info.st_mode | 0o022, *info[1:]))
        return info

    monkeypatch.setattr(native_files.os, "fstat", recording_fstat)
    yield native_files, checked, unsafe
    native_files._pinned_ancestors.clear()


def _inode(windows_files, path):
    return windows_files.WindowsOS().stat(path).st_ino


def test_the_trusted_folder_walk_starts_at_a_pin_after_one_full_walk(
    windows_files, profile, trusted_walk, monkeypatch
):
    # TASK-40608: the admission folder was walked from the drive root about
    # 200 times in a Console turn, to a folder just below a held pin.
    native_files, checked, _ = trusted_walk
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    chain = [_inode(windows_files, folder) for folder in (*reversed(target.parents), target)]
    names = _count_opens(monkeypatch, windows_files)

    with native_files.pinned_directory(target) as fd:
        assert native_files.os.fstat(fd).st_ino == chain[-1]
    assert checked[: len(chain)] == chain  # every folder from the drive root
    full_opens = len(names)
    del checked[:], names[:]

    with native_files.pinned_directory(target) as fd:
        assert native_files.os.fstat(fd).st_ino == chain[-1]
    # The pin itself and the folder below it are checked again; nothing above.
    assert checked[:2] == chain[-2:]
    assert names == ["data"]
    assert full_opens > len(target.parts) - 2

    del checked[:]
    with native_files.pinned_directory(profile) as fd:  # the pin itself
        assert native_files.os.fstat(fd).st_ino == chain[-2]
    assert checked[:1] == chain[-2:-1]


def test_the_trusted_folder_walk_is_full_after_the_window_and_after_a_release(
    windows_files, profile, trusted_walk, monkeypatch
):
    native_files, checked, _ = trusted_walk
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    depth = len(target.parts)

    def walk():
        del checked[:]
        with native_files.pinned_directory(target):
            return len(checked)

    assert walk() == depth  # the first walk checks the whole chain
    assert walk() == 2
    monkeypatch.setattr(native_files, "_PINNED_ANCESTORS_SECONDS", 0.0)
    assert walk() == depth
    monkeypatch.setattr(native_files, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    assert walk() == 2  # the full walk made after the window ended was recorded

    windows_files.release_pinned_roots()
    assert walk() == depth  # no pin: every folder, and nothing is recorded
    assert walk() == depth
    windows_files.restore_pinned_roots()
    assert walk() == depth  # a new pin generation starts with a full walk
    assert walk() == 2


def test_an_unsafe_folder_at_or_below_the_pin_is_refused_on_every_walk(
    windows_files, profile, trusted_walk, monkeypatch
):
    native_files, _, unsafe = trusted_walk
    monkeypatch.setattr(native_files, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    with native_files.pinned_directory(target):
        pass
    for folder in (target, profile):
        unsafe.add(_inode(windows_files, folder))
        with pytest.raises(OSError, match="unsafe_directory"):
            with native_files.pinned_directory(target):
                pass
        unsafe.clear()
    with native_files.pinned_directory(target):
        pass


def test_an_unsafe_folder_above_the_pin_is_refused_when_the_window_ends(
    windows_files, profile, trusted_walk, monkeypatch
):
    # Owner decision O3: a permission change above a pin is seen within one
    # second, not at once. Its identity cannot change while the pin is held.
    native_files, _, unsafe = trusted_walk
    monkeypatch.setattr(native_files, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    with native_files.pinned_directory(target):
        pass
    unsafe.add(_inode(windows_files, profile.parent))
    with native_files.pinned_directory(target):
        pass
    monkeypatch.setattr(native_files, "_PINNED_ANCESTORS_SECONDS", 0.0)
    with pytest.raises(OSError, match="unsafe_directory"):
        with native_files.pinned_directory(target):
            pass


def test_a_refused_full_walk_records_nothing(windows_files, profile, trusted_walk, monkeypatch):
    native_files, checked, unsafe = trusted_walk
    monkeypatch.setattr(native_files, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    unsafe.add(_inode(windows_files, profile.parent))
    with pytest.raises(OSError, match="unsafe_directory"):
        with native_files.pinned_directory(target):
            pass
    unsafe.clear()
    del checked[:]
    with native_files.pinned_directory(target):
        pass
    assert len(checked) == len(target.parts)  # still a full walk


def test_a_visitor_is_shown_the_whole_chain_on_every_walk(
    windows_files, profile, trusted_walk, monkeypatch
):
    native_files, _, _ = trusted_walk
    monkeypatch.setattr(native_files, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    with native_files.pinned_directory(target):
        pass
    for _ in range(2):
        visited = []
        with native_files.pinned_directory(target, _visit=visited.append):
            pass
        assert len(visited) == len(target.parts)


def test_a_walk_beside_the_pin_is_always_full(windows_files, profile, trusted_walk, monkeypatch):
    native_files, checked, _ = trusted_walk
    monkeypatch.setattr(native_files, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    beside = profile.parent / "beside"
    beside.mkdir()
    with native_files.pinned_directory(profile / "data"):
        pass
    for _ in range(2):
        del checked[:]
        with native_files.pinned_directory(beside):
            pass
        assert len(checked) == len(beside.parts)


@pytest.fixture
def verified_walk(trusted_walk):
    """``private_paths.verify_trusted_directory`` with its memo cleared; checks recorded."""
    from tldw_chatbook.Utils import private_paths

    native_files, checked, unsafe = trusted_walk
    assert private_paths.os is native_files.os  # one recorder serves both walks
    private_paths._pinned_directory_walks.clear()

    def verify(path):
        del checked[:]
        result = private_paths.verify_trusted_directory(path, allow_shared_sticky=False)
        assert result.status is private_paths.PrivatePathStatus.TRUSTED_DIRECTORY
        return list(dict.fromkeys(checked))  # each folder checked, in order

    yield private_paths, verify, unsafe
    private_paths._pinned_directory_walks.clear()


def test_the_directory_verification_starts_beside_a_pin_after_one_full_walk(
    windows_files, profile, verified_walk, monkeypatch
):
    # TASK-40608: each SQLite connect verified its folder with a walk of every
    # folder from the drive root, 23 times in the first Console turn.
    _, verify, _ = verified_walk
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    chain = [_inode(windows_files, folder) for folder in (*reversed(target.parents), target)]

    assert verify(target) == chain
    names = _count_opens(monkeypatch, windows_files)
    # The folder holding the pin, the pin and the folder below it are opened
    # by name and checked again; nothing above them is.
    assert verify(target) == chain[-3:]
    assert names == ["home", "profile", "profile", "data", "data"]
    del names[:]
    assert verify(profile) == chain[-3:-1]  # the pin itself is the folder asked about
    assert names == ["home", "profile", "profile"]


def test_the_directory_verification_is_full_after_the_window_and_after_a_release(
    windows_files, profile, verified_walk, monkeypatch
):
    private_paths, verify, _ = verified_walk
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    depth = len(target.parts)

    assert len(verify(target)) == depth
    assert len(verify(target)) == 3
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 0.0)
    assert len(verify(target)) == depth
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    assert len(verify(target)) == 3

    windows_files.release_pinned_roots()
    assert len(verify(target)) == depth  # no pin: every folder, nothing recorded
    assert len(verify(target)) == depth
    windows_files.restore_pinned_roots()
    assert len(verify(target)) == depth  # a new pin generation starts in full
    assert len(verify(target)) == 3


def test_the_directory_verification_refuses_an_unsafe_folder_from_the_pin_holder_down(
    windows_files, profile, verified_walk, monkeypatch
):
    private_paths, verify, unsafe = verified_walk
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    verify(target)
    for folder in (target, profile, profile.parent):
        unsafe.add(_inode(windows_files, folder))
        with pytest.raises(private_paths.PrivatePathError):
            verify(target)
        unsafe.clear()
    assert len(verify(target)) == 3


def test_the_directory_verification_refuses_an_unsafe_folder_above_when_the_window_ends(
    windows_files, profile, verified_walk, tmp_path, monkeypatch
):
    # Owner decision O3, as for the other checked walks.
    private_paths, verify, unsafe = verified_walk
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    verify(target)
    unsafe.add(_inode(windows_files, tmp_path))
    assert len(verify(target)) == 3
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 0.0)
    with pytest.raises(private_paths.PrivatePathError):
        verify(target)


def test_a_refused_directory_verification_records_nothing(
    windows_files, profile, verified_walk, tmp_path, monkeypatch
):
    private_paths, verify, unsafe = verified_walk
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    unsafe.add(_inode(windows_files, tmp_path))
    with pytest.raises(private_paths.PrivatePathError):
        verify(target)
    unsafe.clear()
    assert len(verify(target)) == len(target.parts)  # still a full walk
    assert len(verify(target)) == 3


def test_a_directory_verification_with_its_own_open_is_always_full(
    windows_files, profile, verified_walk, monkeypatch
):
    private_paths, verify, _ = verified_walk
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data"
    verify(target)
    opened = []

    def recording_open(*args, **kwargs):
        opened.append(args[0])
        return private_paths._native_open(*args, **kwargs)

    for _ in range(2):
        del opened[:]
        private_paths.verify_trusted_directory(
            target, allow_shared_sticky=False, _open=recording_open
        )
        assert len(opened) == len(target.parts)


def test_a_checked_stat_reads_the_handle_information_once(windows_files, profile, monkeypatch):
    # TASK-40608: the open and the stat each read it, 3,500 times a turn.
    target = profile / "data" / "notes.db"
    checked = windows_files.WindowsOS()
    windows_files.set_pinned_roots([profile])
    plain = checked.stat(target)
    reads = []
    info = windows_files._Native.info

    def counting(self, handle):
        reads.append(handle)
        return info(self, handle)

    monkeypatch.setattr(windows_files._Native, "info", counting)
    again = checked.stat(target)
    # One read for the folder walked through and one for the file itself.
    assert len(reads) == 2
    assert again == plain
    del reads[:]
    fd = os.open(target, os.O_RDONLY)
    try:
        assert checked.fstat(fd).st_ino == plain.st_ino
    finally:
        os.close(fd)
    assert len(reads) == 1  # a stat of an open descriptor still reads it


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


@pytest.fixture
def files_walk(trusted_walk, verified_walk):
    """``private_paths._open_verified_parent`` with its memo cleared; the folders it checked."""
    private_paths, verify, unsafe = verified_walk
    _, checked, _ = trusted_walk
    private_paths._pinned_ancestors.clear()

    def walk(target):
        del checked[:]
        folder, leaf = private_paths._open_verified_parent(target, missing_leaf_allowed=False)
        private_paths.os.close(folder)
        assert leaf == target.name
        return list(dict.fromkeys(checked))  # each folder checked, in order

    yield private_paths, walk, verify, unsafe
    private_paths._pinned_ancestors.clear()


def test_the_files_walk_starts_at_a_pin_after_one_full_walk(windows_files, profile, files_walk):
    # TASK-40608: a SQLite connect walks to its database's folder for the
    # files it prepares, from the drive root, to a folder just below a pin.
    _, walk, _, _ = files_walk
    windows_files.set_pinned_roots([profile])
    target = profile / "data" / "notes.db"
    chain = [_inode(windows_files, folder) for folder in reversed(target.parents)]

    assert walk(target) == chain  # every folder from the drive root
    assert walk(target) == chain[-2:]  # the pin and the folder below it


def test_the_files_walk_is_full_after_the_window_and_after_a_release(
    windows_files, profile, files_walk, monkeypatch
):
    private_paths, walk, _, _ = files_walk
    windows_files.set_pinned_roots([profile])
    target = profile / "data" / "notes.db"
    depth = len(target.parent.parts)

    assert len(walk(target)) == depth
    assert len(walk(target)) == 2
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 0.0)
    assert len(walk(target)) == depth
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    assert len(walk(target)) == 2  # the full walk made after the window ended was recorded

    windows_files.release_pinned_roots()
    assert len(walk(target)) == depth  # no pin: every folder, nothing recorded
    assert len(walk(target)) == depth
    windows_files.restore_pinned_roots()
    assert len(walk(target)) == depth  # a new pin generation starts in full
    assert len(walk(target)) == 2


def test_the_files_walk_refuses_an_unsafe_folder_at_or_below_the_pin_on_every_walk(
    windows_files, profile, files_walk, monkeypatch
):
    private_paths, walk, _, unsafe = files_walk
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data" / "notes.db"
    walk(target)
    for folder in (target.parent, profile):
        unsafe.add(_inode(windows_files, folder))
        with pytest.raises(private_paths.PrivatePathError):
            walk(target)
        unsafe.clear()
    assert len(walk(target)) == 2


def test_the_files_walk_refuses_an_unsafe_folder_above_the_pin_when_the_window_ends(
    windows_files, profile, files_walk, tmp_path, monkeypatch
):
    # Owner decision O3, as for the other checked walks.
    private_paths, walk, _, unsafe = files_walk
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data" / "notes.db"
    walk(target)
    unsafe.add(_inode(windows_files, tmp_path))
    assert len(walk(target)) == 2
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 0.0)
    with pytest.raises(private_paths.PrivatePathError):
        walk(target)


def test_a_full_directory_verification_stands_for_the_files_walk(
    windows_files, profile, files_walk, monkeypatch
):
    # A SQLite connect verifies the folder and then walks to it for its files:
    # once the window had run out, both went from the drive root.
    private_paths, walk, verify, _ = files_walk
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data" / "notes.db"

    assert len(verify(target.parent)) == len(target.parent.parts)
    assert len(walk(target)) == 2


def test_a_files_walk_does_not_stand_for_the_directory_verification(
    windows_files, profile, files_walk, monkeypatch
):
    """The verification checks more on each folder, so it walks them itself."""
    private_paths, walk, verify, _ = files_walk
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data" / "notes.db"
    depth = len(target.parent.parts)

    assert len(walk(target)) == depth
    assert len(verify(target.parent)) == depth


def test_a_verification_that_began_beside_the_pin_does_not_stand_for_the_files_walk(
    windows_files, profile, files_walk, monkeypatch
):
    """It checked nothing above the folder holding the pin."""
    private_paths, walk, verify, _ = files_walk
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data" / "notes.db"

    verify(target.parent)
    private_paths._pinned_ancestors.clear()
    assert len(verify(target.parent)) == 3  # began at the folder holding the pin
    assert len(walk(target)) == len(target.parent.parts)


def test_a_refused_directory_verification_does_not_stand_for_the_files_walk(
    windows_files, profile, files_walk, tmp_path, monkeypatch
):
    private_paths, walk, verify, unsafe = files_walk
    monkeypatch.setattr(private_paths, "_PINNED_ANCESTORS_SECONDS", 3600.0)
    windows_files.set_pinned_roots([profile])
    target = profile / "data" / "notes.db"
    unsafe.add(_inode(windows_files, tmp_path))
    with pytest.raises(private_paths.PrivatePathError):
        verify(target.parent)
    unsafe.clear()
    assert len(walk(target)) == len(target.parent.parts)
