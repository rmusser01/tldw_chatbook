"""TASK-33620.15.47: storage admission's per-entry path resolution.

Every ChaChaNotes transaction entry opens a storage-admission operation, and that
resolves the participant path, the effective config selector and the acquired source
path (``_repository_operation``, ``_execution_selection_for``, ``_Operation.check``).
``Path.resolve`` lstat()s one path component per call, and each call releases the
GIL; on the UI loop under host load every release can wait for the scheduler (up to
167 ms per send in TASK-33620.15.21's rounds).

These tests pin:

* the cost: a resolution walks a symlink-free path once in the kernel, so its
  filesystem calls do not grow with the path's depth (fails on origin/dev, where
  every extra directory adds one ``lstat`` per resolution);
* the value: ``_resolved(path)`` is exactly ``path.resolve()`` for every path shape,
  links and errors included, with the one-walk proof on and off;
* the refusals the resolutions exist for, unchanged: a directory swapped for a link
  mid-operation is refused (``operation_path_outside_scope``), and a link swapped
  into the selector's or the source's chain after acquisition is refused
  (``execution_selection_changed``). Only the resolution sees those swaps: the
  parent-identity check follows the link to the same directory. These pass on
  origin/dev too; they fail when ``_resolved`` stops falling back on a link.
"""

from __future__ import annotations

import os
import stat
import sys
from collections import Counter
from pathlib import Path

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
)

# The catalog needs POSIX symlinks, permission bits and /dev/fd.
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX path shapes")

DEEP = 10


def _kernel_refuses_a_link_in_any_component() -> bool:
    """macOS 11+ (Darwin 20+): an ``O_NOFOLLOW_ANY`` open refuses every link.

    Decided here, not read from ``storage_admission``, so that the cost test runs
    (and fails on its measurement) against code that predates the one-walk proof.
    """
    if sys.platform != "darwin" or not getattr(os, "O_NOFOLLOW_ANY", 0):
        return False
    try:
        return int(os.uname().release.split(".", 1)[0]) >= 20
    except ValueError:
        return False


@pytest.fixture
def nested_scope(tmp_path, monkeypatch):
    """``local_scope`` with the selector one directory down, so its chain can swap."""
    root = tmp_path / "bootstrap"
    selector_dir = tmp_path / "cfg"
    selector_dir.mkdir(mode=0o700)
    config = selector_dir / "config.toml"
    config.write_text("scope1")
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    authority = admission_authority(root)
    authority.register("profile", (config, data))
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    return root, config, data, authority


def _swap_for_link(directory: Path) -> Path:
    """Move ``directory`` aside and leave a link to it at the old path."""
    moved = directory.with_name(directory.name + "-moved")
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    return moved


# --------------------------------------------------------------------------- cost
_FS_CALLS = {os.lstat, os.stat, os.open, os.close}
_ADMISSION_FILE = os.path.join("Backup_Recovery", "storage_admission.py")


def _in_admission_resolution(frame) -> bool:
    """Whether ``frame`` runs inside one of admission's path resolutions."""
    while frame is not None:
        code = frame.f_code
        if code.co_filename.endswith(_ADMISSION_FILE) and code.co_name == "_resolved":
            return True
        if code.co_name == "resolve" and code.co_filename.endswith("pathlib.py"):
            caller = frame.f_back
            if caller is not None and caller.f_code.co_filename.endswith(
                _ADMISSION_FILE
            ):
                return True
        frame = frame.f_back
    return False


def _resolution_calls_per_entry(db, entries: int = 6) -> float:
    """Filesystem calls admission's path resolution makes per warm transaction entry."""
    for _ in range(3):
        with db.transaction() as cursor:
            cursor.execute("SELECT 1")
    counts: Counter = Counter()

    def profile(frame, event, arg):
        if event == "c_call" and arg in _FS_CALLS and _in_admission_resolution(frame):
            counts[arg.__name__] += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        for _ in range(entries):
            with db.transaction() as cursor:
                cursor.execute("SELECT 1")
    finally:
        sys.setprofile(previous)
    return sum(counts.values()) / entries


@pytest.mark.skipif(
    not _kernel_refuses_a_link_in_any_component(),
    reason="the one-walk proof needs macOS 11+'s O_NOFOLLOW_ANY; elsewhere Path.resolve runs",
)
@pytest.mark.parametrize("scope", ["unbound", "bound"])
def test_transaction_entry_resolution_cost_does_not_grow_with_path_depth(
    nested_scope, scope
):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    root, config, data, _ = nested_scope
    if scope == "bound":
        bind_profile(root, config, ("profile",), root / "admission")
    deep = data
    for index in range(DEEP):
        deep = deep / f"d{index}"
        deep.mkdir(mode=0o700)
    startup = storage.acquire_storage()
    shallow_db = CharactersRAGDB(data / "shallow.db", "probe")
    deep_db = CharactersRAGDB(deep / "deep.db", "probe")
    try:
        shallow = _resolution_calls_per_entry(shallow_db)
        deeper = _resolution_calls_per_entry(deep_db)
    finally:
        for db in (shallow_db, deep_db):
            db.close_connection()
        startup.close()

    assert shallow > 0, "the probe saw no admission path resolution at all"
    assert deeper == shallow, (
        f"admission's path resolution made {deeper:g} filesystem calls per transaction "
        f"entry at {DEEP} more directories against {shallow:g}: it still walks the "
        "path one component at a time on the entering thread"
    )


# --------------------------------------------------------------------------- value
def _outcome(function, path):
    try:
        return ("value", function(path))
    except Exception as error:  # the oracle compares refusals too
        return ("error", type(error), str(error))


def _catalog(base: Path) -> dict[str, Path]:
    base.mkdir(mode=0o700)
    plain = base / "plain"
    plain.mkdir(mode=0o700)
    (plain / "file.db").write_text("x")
    elsewhere = base / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (elsewhere / "file.db").write_text("y")
    (base / "dir-link").symlink_to(plain, target_is_directory=True)
    (plain / "leaf-link.db").symlink_to(elsewhere / "file.db")
    (plain / "dangling.db").symlink_to(base / "nowhere.db")
    (plain / "loop.db").symlink_to(plain / "loop.db")
    (plain / "a-file").write_text("not a directory")
    locked = base / "locked"
    locked.mkdir(mode=0o700)
    (locked / "inner.db").write_text("z")
    locked.chmod(0o000)
    return {
        "existing file": plain / "file.db",
        "absent leaf": plain / "absent.db",
        "absent parents": plain / "a" / "b" / "absent.db",
        "existing directory": plain,
        "leaf is a link": plain / "leaf-link.db",
        "dangling leaf link": plain / "dangling.db",
        "leaf link loop": plain / "loop.db",
        "parent is a link": base / "dir-link" / "file.db",
        "ancestor is a link": base / "dir-link" / "a" / "absent.db",
        "file used as a directory": plain / "a-file" / "x.db",
        "unreadable parent": locked / "inner.db",
        "embedded NUL in the leaf": plain / "nul\0.db",
        "embedded NUL in a parent": plain / "nul\0dir" / "file.db",
        "relative": Path("relative") / "file.db",
        "dot-dot": plain / ".." / "plain" / "file.db",
        "doubled leading slash": Path("//" + str(plain / "file.db").lstrip("/")),
        "root": Path("/"),
    }


@pytest.mark.parametrize("fast", [True, False], ids=["one-walk", "fallback"])
def test_resolved_is_exactly_path_resolve_for_every_path_shape(
    tmp_path, monkeypatch, fast
):
    if not fast:
        monkeypatch.setattr(storage, "_NOFOLLOW_ANY", 0)
    monkeypatch.chdir(tmp_path)
    catalog = _catalog(tmp_path / "shapes")
    try:
        outcomes = {
            name: (_outcome(storage._resolved, path), _outcome(Path.resolve, path))
            for name, path in catalog.items()
        }
    finally:
        (tmp_path / "shapes" / "locked").chmod(0o700)
    mismatches = {name: pair for name, pair in outcomes.items() if pair[0] != pair[1]}
    assert not mismatches, mismatches


def test_resolved_falls_back_whenever_the_walk_meets_a_link(tmp_path, monkeypatch):
    """A link anywhere in the chain takes Path.resolve itself, never a guess."""
    if not storage._NOFOLLOW_ANY:
        pytest.skip("no O_NOFOLLOW_ANY on this platform: every call is the fallback")
    catalog = _catalog(tmp_path / "shapes")
    calls: list[Path] = []
    real = Path.resolve

    def counted(self, strict=False):
        calls.append(self)
        return real(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", counted)
    try:
        for name in ("existing file", "absent leaf", "existing directory"):
            storage._resolved(catalog[name])
        assert calls == [], "a link-free path still took the per-component walk"
        for name in ("leaf is a link", "parent is a link", "ancestor is a link"):
            storage._resolved(catalog[name])
        assert calls == [
            catalog["leaf is a link"],
            catalog["parent is a link"],
            catalog["ancestor is a link"],
        ]
    finally:
        (tmp_path / "shapes" / "locked").chmod(0o700)


@pytest.mark.parametrize(
    ("platform", "release", "expected"),
    [
        ("darwin", "24.6.0", True),
        ("darwin", "20.1.0", True),
        ("darwin", "19.6.0", False),
        ("linux", "6.8.0", False),
    ],
)
def test_one_walk_is_used_only_where_the_kernel_honours_the_flag(
    monkeypatch, platform, release, expected
):
    """An older kernel may ignore an unknown open flag, which would follow links."""
    monkeypatch.setattr(storage.sys, "platform", platform)
    monkeypatch.setattr(storage.os, "O_NOFOLLOW_ANY", 0x20000000, raising=False)
    monkeypatch.setattr(
        storage.os, "uname", lambda: os.uname_result(("X", "h", release, "v", "m"))
    )
    assert bool(storage._nofollow_any_flag()) is expected


# --------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("swapped", ["directory", "file"])
@pytest.mark.parametrize("scope", ["unbound", "bound"])
def test_a_path_swapped_for_a_link_mid_operation_is_still_refused(
    nested_scope, scope, swapped
):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    root, config, data, _ = nested_scope
    if scope == "bound":
        bind_profile(root, config, ("profile",), root / "admission")
    store = data / "store"
    store.mkdir(mode=0o700)
    startup = storage.acquire_storage()
    db = CharactersRAGDB(store / "ChaChaNotes.db", "probe")
    try:
        with db.transaction() as cursor:
            cursor.execute("SELECT 1")
        participant = _repository_participant(db)
        with participant.operation() as operation:
            parent_before = os.stat(store)
            if swapped == "directory":
                _swap_for_link(store)
            else:
                leaf = store / "ChaChaNotes.db"
                moved = store / "ChaChaNotes.db.moved"
                leaf.rename(moved)
                leaf.symlink_to(moved)
            # The link reaches the same directory: only the resolution can tell.
            after = os.stat(store)
            assert (after.st_dev, after.st_ino) == (
                parent_before.st_dev,
                parent_before.st_ino,
            )
            with pytest.raises(
                bootstrap.RecoveryRequired, match="operation_path_outside_scope"
            ):
                with db.transaction() as cursor:
                    cursor.execute("SELECT 1")
            with pytest.raises(
                bootstrap.RecoveryRequired, match="operation_path_outside_scope"
            ):
                storage._check_operation(operation, store / "ChaChaNotes.db")
    finally:
        db.close_connection()
        startup.close()


@pytest.mark.parametrize(
    "swapped",
    ["selector directory", "source directory", "selector file", "source file"],
)
@pytest.mark.parametrize("scope", ["unbound", "bound"])
def test_a_link_swapped_into_the_selection_after_acquisition_is_still_refused(
    nested_scope, scope, swapped
):
    root, config, data, _ = nested_scope
    if scope == "bound":
        bind_profile(root, config, ("profile",), root / "admission")
    store = data / "store"
    store.mkdir(mode=0o700)
    target = store / "x.db"
    target.write_text("x")
    startup = storage.acquire_storage()
    try:
        for _ in range(2):  # the second acquisition may reuse evidence
            with storage.acquire_storage(target) as lease:
                lease.execution_context(target)
        with storage.acquire_storage(target) as lease:
            assert lease._execution_selection[3:] == (
                config.resolve(),
                target,
                target.resolve(),
            )
            if swapped.endswith("directory"):
                _swap_for_link(
                    config.parent if swapped.startswith("selector") else store
                )
            else:
                leaf = config if swapped.startswith("selector") else target
                moved = leaf.with_name(leaf.name + ".moved")
                leaf.rename(moved)
                leaf.symlink_to(moved)
            with pytest.raises(
                bootstrap.RecoveryRequired, match="execution_selection_changed"
            ):
                lease.execution_context(target)
    finally:
        startup.close()


@pytest.mark.parametrize("scope", ["unbound", "bound"])
def test_execution_selection_values_are_path_resolve_values(nested_scope, scope):
    """The recorded selection is what Path.resolve gives, links in the chain or not."""
    root, config, data, _ = nested_scope
    if scope == "bound":
        bind_profile(root, config, ("profile",), root / "admission")
    linked = data / "linked"
    real_dir = data / "real"
    real_dir.mkdir(mode=0o700)
    linked.symlink_to(real_dir, target_is_directory=True)
    for target in (data / "x.db", data / "absent" / "y.db", linked / "z.db"):
        observed = storage._execution_selection_for(target)
        assert observed == (
            os.getpid(),
            root,
            config,
            config.resolve(),
            target,
            target.resolve(),
        ), target


def test_resolved_never_leaves_a_descriptor_open(tmp_path):
    catalog = _catalog(tmp_path / "shapes")
    try:
        before = len(os.listdir("/dev/fd"))
        for _ in range(3):
            for path in catalog.values():
                _outcome(storage._resolved, path)
        assert len(os.listdir("/dev/fd")) == before
    finally:
        (tmp_path / "shapes" / "locked").chmod(0o700)


def test_unreadable_parent_matches_path_resolve(tmp_path):
    """EACCES on the walk falls back; the value is Path.resolve's lexical answer."""
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o700)
    target = locked / "inner.db"
    target.write_text("x")
    locked.chmod(0o000)
    try:
        assert storage._resolved(target) == target.resolve() == target
        assert stat.S_IMODE(os.stat(locked).st_mode) == 0
    finally:
        locked.chmod(0o700)
