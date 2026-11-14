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
* the descriptors: every one it opens is closed before it returns, and it never opens
  the leaf, so a POSIX lock this process holds on a database survives (checked from
  another process, with a control that drops the lock);
* the refusals the resolutions exist for, unchanged: a directory swapped for a link
  mid-operation is refused (``operation_path_outside_scope``), and a link swapped
  into the selector's or the source's chain after acquisition is refused
  (``execution_selection_changed``). Only the resolution sees those swaps: the
  parent-identity check follows the link to the same directory. These pass on
  origin/dev too; they fail when ``_resolved`` stops falling back on a link.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, profile_paths
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
    monkeypatch.setattr(profile_paths.sys, "platform", platform)
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


def _called_from_resolved(frame) -> bool:
    caller = frame.f_back
    return (
        frame.f_code is profile_paths._verified_resolved_path.__code__
        and caller is not None
        and caller.f_code is storage._resolved.__code__
    )


def test_resolved_closes_every_descriptor_it_opens(tmp_path, monkeypatch):
    """Each descriptor ``_resolved`` opens is closed before it returns, on every shape.

    Only ``_resolved``'s own calls on this thread are recorded, so a descriptor another
    thread opens or closes meanwhile cannot change the result.
    """
    catalog = _catalog(tmp_path / "shapes")
    events: list[tuple[str, int]] = []
    real_open, real_close = os.open, os.close
    thread = threading.get_ident()

    def recorded_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        if threading.get_ident() == thread and _called_from_resolved(sys._getframe(1)):
            events.append(("open", descriptor))
        return descriptor

    def recorded_close(descriptor):
        if threading.get_ident() == thread and _called_from_resolved(sys._getframe(1)):
            events.append(("close", descriptor))
        return real_close(descriptor)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(storage.os, "open", recorded_open)
            patch.setattr(storage.os, "close", recorded_close)
            for _ in range(3):
                for path in catalog.values():
                    _outcome(storage._resolved, path)
    finally:
        (tmp_path / "shapes" / "locked").chmod(0o700)
    opened = [descriptor for kind, descriptor in events if kind == "open"]
    if storage._NOFOLLOW_ANY:
        assert opened, "the one-walk proof opened no parent directory at all"
    assert events == [
        (kind, descriptor) for descriptor in opened for kind in ("open", "close")
    ], "a descriptor _resolved opened was not closed before its next open or return"


# --------------------------------------------------------------------------- locks
_TRY_TO_WRITE = """
import sqlite3, sys
connection = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)
try:
    connection.execute("BEGIN IMMEDIATE")
except sqlite3.OperationalError as error:
    print("locked" if "locked" in str(error) else "error: " + str(error))
else:
    connection.execute("ROLLBACK")
    print("acquired")
"""


def _another_process_tries_to_write(db_path: Path) -> str:
    """POSIX locks are per process, so only another process sees a dropped one."""
    completed = subprocess.run(
        [sys.executable, "-c", _TRY_TO_WRITE, str(db_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return completed.stdout.strip()


def test_resolved_never_drops_a_lock_this_process_holds_on_the_leaf(nested_scope):
    """Closing any descriptor of a file drops every POSIX lock this process holds on it.

    The leaf is often a SQLite database the caller holds a lock on, so the one-walk
    proof may open only the parent directory, never the leaf. Each resolution shape
    that reaches the database runs while this process holds ``BEGIN EXCLUSIVE``, and
    another process must still find it locked. The control at the end closes a
    descriptor of the leaf and shows the probe does see a dropped lock.
    """
    _root, _config, data, _ = nested_scope
    db_path = data / "held.db"
    link = data / "held-link.db"
    holder = sqlite3.connect(db_path, isolation_level=None)
    try:
        holder.execute("CREATE TABLE t (x)")
        link.symlink_to(db_path)
        holder.execute("BEGIN EXCLUSIVE")
        assert _another_process_tries_to_write(db_path) == "locked"
        shapes = {
            "the database": db_path,
            "the database used as a directory": db_path / "x.db",
            "a link to the database": link,
        }
        for name, path in shapes.items():
            assert storage._resolved(path) == path.resolve(), name
            assert _another_process_tries_to_write(db_path) == "locked", name
        assert storage._execution_selection_for(db_path)[5] == db_path
        assert _another_process_tries_to_write(db_path) == "locked", "selection"

        os.close(os.open(db_path, os.O_RDONLY))  # control: the lock is gone now
        assert _another_process_tries_to_write(db_path) == "acquired"
    finally:
        holder.close()


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
