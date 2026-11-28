"""TASK-40608: the files of one SQLite connect share one checked walk to their folder.

On Windows a connect prepares the database file and then its three sidecar
names in this process, and each preparation used to walk the same folder chain
again: four checked walks a few milliseconds apart. ``_one_artifact_parent``
lets the later three take the first walk's descriptor of the folder. Every file
is still opened and checked through that descriptor.

The first group pins the sharing rule with a counted stand-in for the walk and
runs everywhere. The second group runs real connects and is Windows-only,
because POSIX prepares its files in the helper process.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.Utils import private_paths
from tldw_chatbook.Utils.private_paths import PrivatePathError

#: Descriptor numbers no real open returns; the stand-in close knows only these.
_FIRST_FAKE_DESCRIPTOR = 1_000_000_000


class _Walks:
    """A counted stand-in for ``_open_verified_parent`` and for closing its descriptors."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[Path, bool]] = []
        self.closed: list[int] = []
        self.fail_next = False
        self._next = _FIRST_FAKE_DESCRIPTOR
        real_close = private_paths.os.close

        def walk(selected, *, missing_leaf_allowed):
            self.calls.append((selected, missing_leaf_allowed))
            if self.fail_next:
                self.fail_next = False
                raise PermissionError("walk refused")
            self._next += 1
            return self._next, selected.name

        def close(descriptor):
            if descriptor > _FIRST_FAKE_DESCRIPTOR:
                self.closed.append(descriptor)
            else:
                real_close(descriptor)

        monkeypatch.setattr(private_paths, "_open_verified_parent", walk)
        monkeypatch.setattr(private_paths.os, "close", close)


@pytest.fixture
def walks(monkeypatch):
    return _Walks(monkeypatch)


def _files(folder: Path) -> list[Path]:
    database = folder / "store.sqlite"
    return [database, *(Path(f"{database}{suffix}") for suffix in ("-wal", "-shm", "-journal"))]


def test_the_files_of_one_block_share_its_first_walk(walks, tmp_path):
    database, *sidecars = _files(tmp_path)
    with private_paths._one_artifact_parent(database):
        folder, leaf = private_paths._artifact_parent(database, missing_leaf_allowed=True)
        assert leaf == "store.sqlite"
        for sidecar in sidecars:
            assert private_paths._artifact_parent(sidecar, missing_leaf_allowed=False) == (
                folder,
                sidecar.name,
            )
    assert walks.calls == [(database, True)]


def test_a_release_inside_the_block_keeps_the_folder_open_and_leaving_closes_it_once(
    walks, tmp_path
):
    database, wal, *_ = _files(tmp_path)
    with private_paths._one_artifact_parent(database):
        folder, _ = private_paths._artifact_parent(database, missing_leaf_allowed=False)
        private_paths._release_artifact_parent(folder)
        assert walks.closed == [], "the next file needs this descriptor"
        assert private_paths._artifact_parent(wal, missing_leaf_allowed=False)[0] == folder
        private_paths._release_artifact_parent(folder)
        assert walks.closed == []
    assert walks.closed == [folder]


def test_outside_a_block_every_file_walks_and_its_release_closes(walks, tmp_path):
    seen = []
    for selected in _files(tmp_path):
        folder, leaf = private_paths._artifact_parent(selected, missing_leaf_allowed=False)
        assert leaf == selected.name
        private_paths._release_artifact_parent(folder)
        seen.append(folder)
    assert len(walks.calls) == 4 and len(set(seen)) == 4
    assert walks.closed == seen


def test_a_file_in_another_folder_walks_on_its_own(walks, tmp_path):
    database = tmp_path / "one" / "store.sqlite"
    other = tmp_path / "two" / "store.sqlite-wal"
    with private_paths._one_artifact_parent(database):
        shared, _ = private_paths._artifact_parent(database, missing_leaf_allowed=False)
        own, leaf = private_paths._artifact_parent(other, missing_leaf_allowed=False)
        assert own != shared and leaf == "store.sqlite-wal"
        private_paths._release_artifact_parent(own)
        assert walks.closed == [own]
    assert walks.calls == [(database, False), (other, False)]
    assert walks.closed == [own, shared]


def test_a_stricter_question_than_the_blocks_walk_answered_is_walked_again(walks, tmp_path):
    """A walk made for an existing file did not ask what creating a file needs."""
    database, wal, *_ = _files(tmp_path)
    with private_paths._one_artifact_parent(database):
        shared, _ = private_paths._artifact_parent(database, missing_leaf_allowed=False)
        own, _ = private_paths._artifact_parent(wal, missing_leaf_allowed=True)
        assert own != shared
        private_paths._release_artifact_parent(own)
        assert walks.closed == [own]
        # The weaker question is still answered by the first walk.
        assert private_paths._artifact_parent(wal, missing_leaf_allowed=False)[0] == shared
    assert walks.calls == [(database, False), (wal, True)]


def test_a_walk_that_asked_the_stricter_question_serves_the_weaker_one(walks, tmp_path):
    database, wal, *_ = _files(tmp_path)
    with private_paths._one_artifact_parent(database):
        shared, _ = private_paths._artifact_parent(database, missing_leaf_allowed=True)
        assert private_paths._artifact_parent(wal, missing_leaf_allowed=False)[0] == shared
        assert private_paths._artifact_parent(wal, missing_leaf_allowed=True)[0] == shared
    assert walks.calls == [(database, True)]


def test_a_refused_first_walk_shares_nothing(walks, tmp_path):
    database, wal, *_ = _files(tmp_path)
    with private_paths._one_artifact_parent(database):
        walks.fail_next = True
        with pytest.raises(PermissionError):
            private_paths._artifact_parent(database, missing_leaf_allowed=False)
        folder, _ = private_paths._artifact_parent(wal, missing_leaf_allowed=False)
    assert walks.calls == [(database, False), (wal, False)]
    assert walks.closed == [folder]


def test_the_block_closes_its_folder_when_its_body_raises(walks, tmp_path):
    database = _files(tmp_path)[0]
    with pytest.raises(RuntimeError, match="unsafe file"):
        with private_paths._one_artifact_parent(database):
            folder, _ = private_paths._artifact_parent(database, missing_leaf_allowed=False)
            raise RuntimeError("unsafe file")
    assert walks.closed == [folder]
    # Nothing is left behind for the next connect on this thread.
    again, _ = private_paths._artifact_parent(database, missing_leaf_allowed=False)
    assert again != folder and len(walks.calls) == 2


def test_a_block_opened_inside_another_leaves_the_outer_walk_in_place(walks, tmp_path):
    database, wal, *_ = _files(tmp_path)
    with private_paths._one_artifact_parent(database):
        shared, _ = private_paths._artifact_parent(database, missing_leaf_allowed=False)
        with private_paths._one_artifact_parent(tmp_path / "elsewhere" / "other.sqlite"):
            assert private_paths._artifact_parent(wal, missing_leaf_allowed=False)[0] == shared
        assert walks.closed == [], "the inner block does not own the walk"
        assert private_paths._artifact_parent(wal, missing_leaf_allowed=False)[0] == shared
    assert walks.closed == [shared] and len(walks.calls) == 1


def test_another_thread_does_not_see_this_threads_block(walks, tmp_path):
    database, wal, *_ = _files(tmp_path)
    seen = []

    def other_thread():
        folder, _ = private_paths._artifact_parent(wal, missing_leaf_allowed=False)
        private_paths._release_artifact_parent(folder)
        seen.append(folder)

    with private_paths._one_artifact_parent(database):
        shared, _ = private_paths._artifact_parent(database, missing_leaf_allowed=False)
        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join(timeout=20)
        assert seen and seen[0] != shared
        assert walks.closed == seen, "the other thread closed only its own descriptor"
    assert walks.closed == [*seen, shared]


def test_an_operation_that_counts_its_own_descriptors_keeps_one_walk_per_file(
    walks, tmp_path, monkeypatch
):
    monkeypatch.setattr(private_paths, "_ordinary_walk", lambda: False)
    seen = []
    with private_paths._one_artifact_parent(_files(tmp_path)[0]):
        for selected in _files(tmp_path):
            folder, _ = private_paths._artifact_parent(selected, missing_leaf_allowed=False)
            private_paths._release_artifact_parent(folder)
            seen.append(folder)
    assert len(walks.calls) == 4 and len(set(seen)) == 4
    assert walks.closed == seen


@pytest.mark.parametrize("owner", ["none", "raw", "voice", "visual"])
def test_only_a_thread_with_no_owning_operation_shares_a_walk(monkeypatch, owner):
    monkeypatch.setattr(
        private_paths, "_runtime_operation", lambda path=None: object() if owner == "raw" else None
    )
    monkeypatch.setattr(
        private_paths, "_voice_operation", lambda *, native=False: object() if owner == "voice" else None
    )
    monkeypatch.setattr(
        private_paths,
        "_visual_native_scope",
        lambda: (object(), object()) if owner == "visual" else (None, None),
    )
    assert private_paths._ordinary_walk() is (owner == "none")


windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="POSIX prepares the files in its helper process"
)


@pytest.fixture
def counted_real_walks(monkeypatch):
    """Count the real walks and remember the folder descriptors they returned."""
    real = private_paths._open_verified_parent
    seen: list[tuple[str, bool, int]] = []

    def counted(selected, *, missing_leaf_allowed):
        folder, leaf = real(selected, missing_leaf_allowed=missing_leaf_allowed)
        seen.append((selected.name, missing_leaf_allowed, folder))
        return folder, leaf

    monkeypatch.setattr(private_paths, "_open_verified_parent", counted)
    return seen


def _connect(path: Path):
    return private_sqlite.connect_private_sqlite("db.base", path)


@windows_only
@pytest.mark.bootstrap_profile
def test_a_connect_walks_to_its_folder_once_for_the_database_and_its_sidecars(
    tmp_path, counted_real_walks
):
    path = tmp_path / "store.sqlite"
    _connect(path).close()
    assert [(name, create) for name, create, _ in counted_real_walks] == [("store.sqlite", True)]
    del counted_real_walks[:]
    connection = _connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS t (x)")
        connection.execute("INSERT INTO t VALUES (1)")
        connection.commit()
        again = _connect(path)  # the sidecars exist now and are checked through the shared folder
        again.close()
    finally:
        connection.close()
    assert [name for name, _, _ in counted_real_walks] == ["store.sqlite", "store.sqlite"]


@windows_only
@pytest.mark.bootstrap_profile
def test_the_shared_folder_descriptor_is_closed_when_the_connect_returns(
    tmp_path, counted_real_walks
):
    _connect(tmp_path / "store.sqlite").close()
    ((_, _, folder),) = counted_real_walks
    with pytest.raises(OSError):
        private_paths.os.fstat(folder)


@windows_only
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_a_sidecar_that_is_not_a_file_is_still_refused_through_the_shared_folder(
    tmp_path, counted_real_walks, suffix
):
    path = tmp_path / "store.sqlite"
    _connect(path).close()
    del counted_real_walks[:]
    sidecar = Path(f"{path}{suffix}")
    sidecar.mkdir()
    with pytest.raises(PrivatePathError):
        _connect(path)
    # The refusal released the folder: nothing holds it and the next connect works.
    ((_, _, folder),) = counted_real_walks
    with pytest.raises(OSError):
        private_paths.os.fstat(folder)
    sidecar.rmdir()
    _connect(path).close()


@windows_only
@pytest.mark.bootstrap_profile
def test_preparing_a_file_outside_a_connect_walks_for_that_file(tmp_path, counted_real_walks):
    """The sharing is the connect's block; a lone preparation is as it was."""
    path = tmp_path / "store.sqlite"
    _connect(path).close()
    del counted_real_walks[:]
    for suffix in ("", "-wal", "-shm", "-journal"):
        private_sqlite._prepare_artifact(
            Path(f"{path}{suffix}"), writable=True, create_if_missing=False, optional=bool(suffix)
        )
    assert [name for name, _, _ in counted_real_walks] == [
        "store.sqlite",
        "store.sqlite-wal",
        "store.sqlite-shm",
        "store.sqlite-journal",
    ]
    _, _, last = counted_real_walks[-1]
    with pytest.raises(OSError):
        private_paths.os.fstat(last)
