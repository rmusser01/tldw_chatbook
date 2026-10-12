from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Notes import sync_paths
from tldw_chatbook.Notes.sync_paths import (
    PinnedSyncRoot,
    SyncPathError,
    SyncPathPartialError,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="descriptor containment contract is POSIX-specific",
)


def test_selected_root_link_is_rejected(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    selected = tmp_path / "selected"
    selected.symlink_to(canonical, target_is_directory=True)
    with pytest.raises(SyncPathError, match="root_link_or_reparse"):
        PinnedSyncRoot(selected)


def test_unsupported_descriptor_guards_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    monkeypatch.setattr(sync_paths, "_descriptor_guards_available", lambda: False)

    (root_path / "note.md").write_text("existing", encoding="utf-8")

    with PinnedSyncRoot(root_path) as root:
        with pytest.raises(SyncPathError, match="unsupported_platform"):
            root.read_bytes(Path("note.md"))

    assert (root_path / "note.md").read_text(encoding="utf-8") == "existing"


def test_relative_escape_is_rejected(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()

    (tmp_path / "escape.md").write_text("outside", encoding="utf-8")

    with PinnedSyncRoot(root_path) as root:
        with pytest.raises(SyncPathError, match="invalid_relative_path"):
            root.read_bytes(Path("../escape.md"))

    assert (tmp_path / "escape.md").read_text(encoding="utf-8") == "outside"


def test_descriptor_verified_byte_observation_retains_exact_bytes(
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    payload = b"\xef\xbb\xbfline one\r\nline two"
    (root_path / "note.md").write_bytes(payload)

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")

    assert observed.content == payload
    assert observed.identity.link_count == 1
    assert "note.md" not in repr(observed)
    assert str(observed.identity.device) not in repr(observed.identity)
    assert str(observed.identity.inode) not in repr(observed.identity)


def test_byte_observation_rejects_same_inode_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        monkeypatch.setattr(
            root,
            "_after_read",
            lambda _relative_path: target.write_bytes(b"different-size"),
        )
        with pytest.raises(SyncPathError, match="target_changed_during_read"):
            root.read_bytes("note.md")


def test_guarded_byte_replacement_requires_observed_identity(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        target.unlink()
        target.write_bytes(b"racer")
        with pytest.raises(SyncPathError, match="target_identity_changed"):
            root.replace_bytes(
                "note.md",
                b"after",
                expected=observed,
                mode=0o640,
            )

    assert target.read_bytes() == b"racer"


def test_guarded_byte_replacement_does_not_clobber_final_boundary_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")

        def swap(_relative_path: Path) -> None:
            target.unlink()
            target.write_bytes(b"racer")

        monkeypatch.setattr(root, "_before_commit", swap)
        with pytest.raises(SyncPathError, match="target_identity_changed"):
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert target.read_bytes() == b"racer"


def test_exchange_verification_failure_rolls_back_without_losing_old_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        original = root._read_bytes

        def fail_displaced(parent_fd, leaf, relative_path, entry, max_bytes):
            if leaf.startswith(".note.md.tmp-"):
                raise SyncPathError("target_changed_during_read", relative_path)
            return original(parent_fd, leaf, relative_path, entry, max_bytes)

        monkeypatch.setattr(root, "_read_bytes", fail_displaced)
        with pytest.raises(SyncPathError, match="target_changed_during_read"):
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert target.read_bytes() == b"before"
    assert list(root_path.glob(".note.md.tmp-*")) == []


def test_exchange_rollback_failure_preserves_both_byte_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    original_rename = sync_paths._rename_with_flags
    exchanges = 0

    def fail_rollback(source_fd, source, destination_fd, destination, flags):
        nonlocal exchanges
        if flags == sync_paths._RENAME_EXCHANGE:
            exchanges += 1
            if exchanges == 2:
                raise OSError("rollback failed")
        return original_rename(source_fd, source, destination_fd, destination, flags)

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        monkeypatch.setattr(sync_paths, "_rename_with_flags", fail_rollback)
        monkeypatch.setattr(
            root,
            "_read_bytes",
            lambda *_args: (_ for _ in ()).throw(
                SyncPathError("target_changed_during_read", "note.md")
            ),
        )
        with pytest.raises(SyncPathError, match="replacement_rollback_failed"):
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert target.read_bytes() == b"after"
    preserved = list(root_path.glob(".note.md.tmp-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"before"


def test_displaced_cleanup_failure_preserves_old_bytes_for_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    original_unlink = sync_paths.os.unlink

    def fail_displaced_unlink(path, *args, **kwargs):
        if str(path).startswith(".note.md.tmp-"):
            raise OSError("cleanup failed")
        return original_unlink(path, *args, **kwargs)

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        monkeypatch.setattr(sync_paths.os, "unlink", fail_displaced_unlink)
        with pytest.raises(SyncPathPartialError) as raised:
            root.replace_bytes(
                "note.md",
                b"after",
                expected=observed,
                mode=0o600,
            )

    assert raised.value.reason == "replacement_cleanup_pending"
    assert raised.value.cleanup_leaf is not None
    assert "note.md" not in repr(raised.value)
    assert target.read_bytes() == b"after"
    preserved = list(root_path.glob(".note.md.tmp-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"before"


def test_parent_fsync_failure_after_exchange_preserves_old_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    at_commit = False
    original_fsync = sync_paths.os.fsync

    def mark_commit(_relative_path: Path) -> None:
        nonlocal at_commit
        at_commit = True

    def fail_after_commit(descriptor: int) -> None:
        if at_commit and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory fsync failed")
        original_fsync(descriptor)

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        monkeypatch.setattr(root, "_before_commit", mark_commit)
        monkeypatch.setattr(sync_paths.os, "fsync", fail_after_commit)
        with pytest.raises(
            SyncPathPartialError,
            match="replacement_rollback_unverified",
        ):
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert target.read_bytes() == b"before"
    preserved = list(root_path.glob(".note.md.tmp-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"after"


def test_race_after_displaced_verification_restores_old_and_preserves_racer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        monkeypatch.setattr(
            root,
            "_after_displaced_verification",
            lambda _relative_path: target.write_bytes(b"racer"),
        )
        with pytest.raises(SyncPathPartialError) as raised:
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert raised.value.reason == "replacement_raced_after_exchange"
    assert target.read_bytes() == b"before"
    preserved = list(root_path.glob(".note.md.tmp-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"racer"


def test_same_root_move_is_identity_guarded(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    source = root_path / "before.md"
    source.write_bytes(b"body")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        moved = root.move_file(
            "before.md",
            "nested/after.md",
            expected=observed,
        )

    assert moved.relative_path == Path("nested/after.md")
    assert moved.identity == observed.identity
    assert not source.exists()
    assert (root_path / "nested/after.md").read_bytes() == b"body"


def test_same_root_move_rejects_existing_destination(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "before.md").write_bytes(b"before")
    (root_path / "after.md").write_bytes(b"after")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        with pytest.raises(SyncPathError, match="destination_exists"):
            root.move_file(
                "before.md",
                "after.md",
                expected=observed,
            )

    assert (root_path / "before.md").read_bytes() == b"before"
    assert (root_path / "after.md").read_bytes() == b"after"


def test_same_root_move_does_not_clobber_destination_created_at_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    source = root_path / "before.md"
    source.write_bytes(b"before")
    destination = root_path / "after.md"

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        monkeypatch.setattr(
            root,
            "_before_commit",
            lambda _relative_path: destination.write_bytes(b"racer"),
        )
        with pytest.raises(SyncPathError, match="destination_exists"):
            root.move_file("before.md", "after.md", expected=observed)

    assert source.read_bytes() == b"before"
    assert destination.read_bytes() == b"racer"


def test_same_root_move_closes_source_parent_when_destination_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "before.md").write_bytes(b"before")
    captured: list[int] = []

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        original = root._open_parent

        def fail_second(relative_path: Path, *, create: bool) -> int:
            if captured:
                raise SyncPathError("missing_parent", relative_path)
            descriptor = original(relative_path, create=create)
            captured.append(descriptor)
            return descriptor

        monkeypatch.setattr(root, "_open_parent", fail_second)
        with pytest.raises(SyncPathError, match="missing_parent"):
            root.move_file("before.md", "nested/after.md", expected=observed)

    with pytest.raises(OSError):
        os.fstat(captured[0])


def test_move_fsync_failure_surfaces_committed_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    source = root_path / "before.md"
    source.write_bytes(b"before")
    at_commit = False
    original_fsync = sync_paths.os.fsync

    def mark_commit(_relative_path: Path) -> None:
        nonlocal at_commit
        at_commit = True

    def fail_directory_fsync(descriptor: int) -> None:
        if at_commit and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory fsync failed")
        original_fsync(descriptor)

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        monkeypatch.setattr(root, "_before_commit", mark_commit)
        monkeypatch.setattr(sync_paths.os, "fsync", fail_directory_fsync)
        with pytest.raises(SyncPathPartialError) as raised:
            root.move_file("before.md", "after.md", expected=observed)

    assert raised.value.reason == "move_commit_unverified"
    assert raised.value.cleanup_leaf == "after.md"
    assert not source.exists()
    assert (root_path / "after.md").read_bytes() == b"before"


class _Candidates:
    """The candidate descriptors ``_create_temporary`` has open right now.

    Counted by wrapping the two calls that open and close them. ``/dev/fd``
    is the whole process: another thread or a finaliser moves it.
    Install it inside the ``with PinnedSyncRoot(...)`` block: the root checks
    ``os.open`` against ``os.supports_dir_fd`` when it is constructed.
    """

    def __init__(self, patch: pytest.MonkeyPatch) -> None:
        self.open: dict[int, str] = {}
        self.created: list[str] = []
        self.attempts = 0
        self.peak = 0
        #: Raise this from the Nth candidate create (1-based), if set.
        self.fail_create: tuple[int, OSError] | None = None
        #: Report an error from the close of these descriptors (after closing).
        self.fail_close: set[int] = set()
        self.failed_closes: list[int] = []
        #: Candidate descriptors closed again while nothing had the number
        #: (the operating system answered EBADF).
        self.double_closes: list[int] = []
        closed: set[int] = set()
        real_open, real_close = os.open, os.close

        def tracked_open(path, flags, mode=0o777, *, dir_fd=None):
            candidate = (
                isinstance(path, str) and bool(flags & os.O_CREAT) and ".tmp-" in path
            )
            if candidate:
                self.attempts += 1
                if self.fail_create and self.attempts == self.fail_create[0]:
                    raise self.fail_create[1]
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            if candidate:
                self.created.append(os.fspath(path))
                self.open[descriptor] = os.fspath(path)
                self.peak = max(self.peak, len(self.open))
            return descriptor

        def tracked_close(descriptor):
            tracked = self.open.pop(descriptor, None) is not None
            try:
                real_close(descriptor)
            except OSError as error:
                if error.errno == errno.EBADF and descriptor in closed:
                    self.double_closes.append(descriptor)
                raise
            if tracked:
                closed.add(descriptor)
                if descriptor in self.fail_close:
                    self.failed_closes.append(descriptor)
                    raise OSError(errno.EIO, "close failed")

        patch.setattr(sync_paths.os, "open", tracked_open)
        patch.setattr(sync_paths.os, "close", tracked_close)


def _candidate_names(root_path: Path) -> list[str]:
    return sorted(name for name in os.listdir(root_path) if ".tmp-" in name)


@pytest.mark.parametrize("existing", [True, False], ids=["replace", "create"])
def test_a_written_file_never_takes_a_claimed_inode_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    """TASK-34789: ext4 gives a new file the inode number the last replaced
    file gave back, and a sync binding records that number until its own
    commit. A candidate whose number is claimed is not the file that lands
    while an unclaimed one can be had.

    A refused candidate keeps its number by its DESCRIPTOR, not its name: at
    every question the directory holds the one candidate being asked about
    and none of the refused ones, and one more descriptor is open per
    refusal. On ext4 the three numbers differ only because of that."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    if existing:
        target.write_bytes(b"before")
    #: (device, inode, candidate names listed, candidate descriptors open).
    asked: list[tuple[int, int, int, int]] = []

    def claimed(device: int, inode: int) -> bool:
        asked.append((device, inode, len(_candidate_names(root_path)), len(held.open)))
        return len(asked) <= 2  # the first two numbers are someone else's

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        expected = root.read_bytes("note.md") if existing else None
        with monkeypatch.context() as patch:
            held = _Candidates(patch)
            result = root.replace_bytes(
                "note.md", b"after", expected=expected, mode=0o600
            )

    landed = target.stat()
    refused = {(device, inode) for device, inode, _, _ in asked[:2]}
    assert target.read_bytes() == b"after"
    assert (landed.st_dev, landed.st_ino) == asked[2][:2]
    assert (landed.st_dev, landed.st_ino) not in refused
    assert result.identity.inode == landed.st_ino
    assert len({inode for _, inode, _, _ in asked}) == 3
    assert [names for _, _, names, _ in asked] == [1, 1, 1], "a refused name stayed"
    assert [open_ for _, _, _, open_ in asked] == [1, 2, 3], "a number was not held"
    assert held.open == {}
    assert held.double_closes == []
    assert os.listdir(root_path) == ["note.md"]


class _Stop(BaseException):
    """Not an ``Exception``: nothing in the write path may swallow it."""


@pytest.mark.parametrize("error", [RuntimeError, _Stop], ids=["exception", "base"])
def test_a_claim_check_that_raises_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    """Raised at the third question: two refused candidates are held and a
    third is being examined. All three descriptors are closed, no name
    stays, and the exception arrives unchanged."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    asked: list[int] = []

    def claimed(_device: int, inode: int) -> bool:
        asked.append(inode)
        if len(asked) == 3:
            raise error("store unavailable")
        return True

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        observed = root.read_bytes("note.md")
        with monkeypatch.context() as patch:
            held = _Candidates(patch)
            with pytest.raises(error, match="store unavailable"):
                root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert held.peak == 3
    assert held.open == {}
    assert held.double_closes == []
    assert target.read_bytes() == b"before"
    assert os.listdir(root_path) == ["note.md"]


def test_a_close_that_fails_does_not_skip_the_other_cleanups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three refused candidates are held when the search ends; closing the
    first reports an error. The other two are still closed, no name stays,
    and the write lands."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    asked: list[int] = []

    def claimed(_device: int, inode: int) -> bool:
        asked.append(inode)
        if len(asked) == 1:
            held.fail_close.update(held.open)
        return len(asked) <= 3

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        observed = root.read_bytes("note.md")
        with monkeypatch.context() as patch:
            held = _Candidates(patch)
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert len(held.failed_closes) == 1, "the held descriptor's close never failed"
    assert held.peak == 4
    assert held.open == {}
    assert held.double_closes == []
    assert target.read_bytes() == b"after"
    assert os.listdir(root_path) == ["note.md"]


def test_asking_stops_at_the_bound_and_no_more_descriptors_are_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every number is claimed. After ``_MAX_CLAIMED_CANDIDATES`` refusals
    the next candidate is used unasked; a refusal is not a name collision,
    so the write does not end in ``temporary_name_exhausted``.

    Before this PR: the write used its first candidate, also unasked. The
    outcome is the same kind -- a file on a claimed number, for the commit
    to decide -- and not a failed write."""

    bound = sync_paths._MAX_CLAIMED_CANDIDATES
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    asked: list[tuple[int, int]] = []
    names: list[int] = []

    def claimed(device: int, inode: int) -> bool:
        asked.append((device, inode))
        # Fail at once if the bound is gone, instead of looping to the timeout.
        assert len(asked) <= sync_paths._MAX_CLAIMED_CANDIDATES + 2, (
            "asked past the bound"
        )
        names.append(len(_candidate_names(root_path)))
        return True

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        observed = root.read_bytes("note.md")
        with monkeypatch.context() as patch:
            held = _Candidates(patch)
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    landed = target.stat()
    assert len(asked) == bound
    assert len(held.created) == bound + 1
    assert held.peak == bound + 1, "the refused candidates and the one in hand"
    assert set(names) == {1}
    assert (landed.st_dev, landed.st_ino) not in set(asked)
    assert held.open == {}
    assert held.double_closes == []
    assert target.read_bytes() == b"after"
    assert os.listdir(root_path) == ["note.md"]


@pytest.mark.parametrize("spare", [True, False], ids=["a_spare", "no_spare"])
def test_when_the_next_candidate_cannot_be_created_the_spare_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spare: bool
) -> None:
    """Fallback: the disk is full (or a quota, or no descriptors) when the
    candidate after a refusal is created. The refused one still has its name
    -- it is the spare -- and is used unasked.

    Before this PR: the write used that same first candidate. Failing here
    would be a failure the check added. With no spare (the first candidate
    cannot be created) the write fails ``operation_failed``, as it always
    did."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    asked: list[tuple[int, int]] = []

    def claimed(device: int, inode: int) -> bool:
        asked.append((device, inode))
        return True

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        observed = root.read_bytes("note.md")
        with monkeypatch.context() as patch:
            held = _Candidates(patch)
            held.fail_create = (2 if spare else 1, OSError(errno.ENOSPC, "full"))
            if spare:
                root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)
            else:
                with pytest.raises(SyncPathError) as raised:
                    root.replace_bytes(
                        "note.md", b"after", expected=observed, mode=0o600
                    )

    assert held.open == {}
    assert held.double_closes == []
    assert os.listdir(root_path) == ["note.md"]
    if spare:
        landed = target.stat()
        assert asked == [(landed.st_dev, landed.st_ino)], "not the refused spare"
        assert target.read_bytes() == b"after"
    else:
        assert raised.value.reason == "operation_failed"
        assert asked == []
        assert target.read_bytes() == b"before"


def test_a_candidate_that_cannot_be_examined_is_used_unasked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback: ``os.fstat`` on the candidate fails, so there is no number
    to ask about. Before this PR the candidate was never examined at all and
    the write went ahead; it still does."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    asked: list[int] = []
    real_fstat = os.fstat

    def claimed(_device: int, inode: int) -> bool:
        asked.append(inode)
        return True

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        observed = root.read_bytes("note.md")
        with monkeypatch.context() as patch:
            held = _Candidates(patch)

            def unreadable(descriptor: int) -> os.stat_result:
                if descriptor in held.open:
                    raise OSError(errno.EIO, "fstat failed")
                return real_fstat(descriptor)

            patch.setattr(sync_paths.os, "fstat", unreadable)
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert asked == []
    assert len(held.created) == 1
    assert held.open == {}
    assert held.double_closes == []
    assert target.read_bytes() == b"after"
    assert os.listdir(root_path) == ["note.md"]


def test_a_refused_name_that_cannot_be_removed_ends_the_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback: the refused candidate's name cannot be unlinked once its
    replacement exists. Asking on would strand one more name per refusal, so
    the replacement is used unasked and at most one name is ever left. Before
    this PR there was one candidate and no question."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    asked: list[int] = []
    real_unlink = os.unlink
    stuck: list[str] = []

    def claimed(_device: int, inode: int) -> bool:
        asked.append(inode)
        return True

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        observed = root.read_bytes("note.md")
        with monkeypatch.context() as patch:
            held = _Candidates(patch)

            def sticky(path, *, dir_fd=None):
                name = os.fspath(path)
                if name in stuck or (not stuck and name in held.created):
                    if name not in stuck:
                        stuck.append(name)
                    raise OSError(errno.EPERM, "unlink refused")
                return real_unlink(path, dir_fd=dir_fd)

            patch.setattr(sync_paths.os, "unlink", sticky)
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert len(asked) == 1, "the search went on after a name it could not remove"
    assert len(held.created) == 2
    assert held.open == {}
    assert held.double_closes == []
    assert target.read_bytes() == b"after"
    assert _candidate_names(root_path) == stuck


def test_a_name_that_only_goes_once_its_file_is_closed_is_not_stranded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some filesystems will not unlink an open file. The refused candidate
    is then closed and removed, and the write goes ahead on the candidate in
    hand: no name is left behind at all."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    asked: list[int] = []
    real_unlink = os.unlink
    refused_while_open: list[str] = []

    def claimed(_device: int, inode: int) -> bool:
        asked.append(inode)
        return True

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        observed = root.read_bytes("note.md")
        with monkeypatch.context() as patch:
            held = _Candidates(patch)

            def needs_closed(path, *, dir_fd=None):
                name = os.fspath(path)
                if name in held.open.values():
                    refused_while_open.append(name)
                    raise OSError(errno.EBUSY, "file is open")
                return real_unlink(path, dir_fd=dir_fd)

            patch.setattr(sync_paths.os, "unlink", needs_closed)
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert len(refused_while_open) == 1 and len(asked) == 1
    assert held.open == {}
    assert held.double_closes == []
    assert target.read_bytes() == b"after"
    assert _candidate_names(root_path) == []


def test_an_abandoned_candidate_is_closed_before_its_name_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim check raises with a candidate in hand, on a filesystem that
    will not unlink an open file: the candidate must still not be left behind."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    real_unlink = os.unlink

    def claimed(_device: int, _inode: int) -> bool:
        raise RuntimeError("the store went away")

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        observed = root.read_bytes("note.md")
        with monkeypatch.context() as patch:
            held = _Candidates(patch)

            def needs_closed(path, *, dir_fd=None):
                if os.fspath(path) in held.open.values():
                    raise OSError(errno.EBUSY, "file is open")
                return real_unlink(path, dir_fd=dir_fd)

            patch.setattr(sync_paths.os, "unlink", needs_closed)
            with pytest.raises(RuntimeError, match="the store went away"):
                root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert len(held.created) == 1
    assert held.open == {}
    assert held.double_closes == []
    assert target.read_bytes() == b"before"
    assert _candidate_names(root_path) == []


def test_a_refused_name_that_is_already_gone_does_not_end_the_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing is stranded when the name has vanished, and its descriptor
    still holds the number, so the search goes on to an unclaimed candidate."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    asked: list[int] = []
    real_unlink = os.unlink
    vanished: list[str] = []
    open_at_ask: list[int] = []

    def claimed(_device: int, inode: int) -> bool:
        asked.append(inode)
        open_at_ask.append(len(held.open))
        return len(asked) < 3

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        observed = root.read_bytes("note.md")
        with monkeypatch.context() as patch:
            held = _Candidates(patch)

            def gone(path, *, dir_fd=None):
                name = os.fspath(path)
                real_unlink(path, dir_fd=dir_fd)
                if not vanished and name in held.created:
                    vanished.append(name)
                    raise FileNotFoundError(errno.ENOENT, "already gone")

            patch.setattr(sync_paths.os, "unlink", gone)
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert vanished and len(asked) == 3, asked
    assert open_at_ask == [1, 2, 3], "the vanished candidate's descriptor was not held"
    assert len(set(asked)) == 3, "a refused number was handed out again"
    assert held.open == {}
    assert held.double_closes == []
    assert target.read_bytes() == b"after"
    assert _candidate_names(root_path) == []


def test_a_filesystem_that_repeats_a_refused_number_is_not_searched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback: the second candidate reports the number of the first, which
    was refused and is still held open. Holding does not keep numbers apart
    on this filesystem, so there is nothing to search: the candidate is used
    unasked, after two creates instead of the whole bound on every write.

    Before this PR: the write used its first candidate unasked."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    asked: list[int] = []
    real_fstat = os.fstat

    def claimed(_device: int, inode: int) -> bool:
        asked.append(inode)
        # Fail at once if the repeat is not noticed, instead of looping to the timeout.
        assert len(asked) <= 3, "the search went on asking about a repeated number"
        return True

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        observed = root.read_bytes("note.md")
        with monkeypatch.context() as patch:
            held = _Candidates(patch)

            def one_number(descriptor: int):
                entry = real_fstat(descriptor)
                if descriptor not in held.open:
                    return entry
                return SimpleNamespace(st_dev=entry.st_dev, st_ino=7)

            patch.setattr(sync_paths.os, "fstat", one_number)
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert asked == [7]
    assert len(held.created) == 2
    assert held.open == {}
    assert held.double_closes == []
    assert target.read_bytes() == b"after"
    assert os.listdir(root_path) == ["note.md"]


def test_a_recreated_file_never_takes_the_number_its_deleted_predecessor_had(
    tmp_path: Path,
) -> None:
    """TASK-34789 on REAL inode numbers, in one root: a bound file is deleted
    outside the app, its binding still records ``(device, inode)``, and sync
    creates the file again. The claim is the writer's own binding's; nothing
    tells it apart from anyone else's and nothing needs to -- the candidate
    is refused and the file lands on another number.

    Passes where the filesystem really handed the new file the freed number
    (ext4); skips, after the same assertions, where it did not (APFS)."""

    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")
    recorded = target.stat()
    target.unlink()
    offered: list[bool] = []

    def claimed(device: int, inode: int) -> bool:
        offered.append((device, inode) == (recorded.st_dev, recorded.st_ino))
        return offered[-1]

    with PinnedSyncRoot(root_path, identity_claimed=claimed) as root:
        root.replace_bytes("note.md", b"after", expected=None, mode=0o600)

    landed = target.stat()
    assert target.read_bytes() == b"after"
    assert (landed.st_dev, landed.st_ino) != (recorded.st_dev, recorded.st_ino)
    assert offered[-1] is False
    assert os.listdir(root_path) == ["note.md"]
    if not any(offered):
        pytest.skip(
            "this filesystem did not hand the recreated file its deleted "
            "predecessor's inode number in this run"
        )


def test_replace_maps_raw_precommit_oserror_to_bounded_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "PRIVATE-root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_bytes(b"before")

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("note.md")
        monkeypatch.setattr(
            root,
            "_existing_target",
            lambda *_args: (_ for _ in ()).throw(OSError("PRIVATE path")),
        )
        with pytest.raises(SyncPathError) as raised:
            root.replace_bytes("note.md", b"after", expected=observed, mode=0o600)

    assert raised.value.reason == "operation_failed"
    assert raised.value.__cause__ is None
    assert target.read_bytes() == b"before"


def test_replace_raw_postcommit_error_is_distinct_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    calls = 0

    with PinnedSyncRoot(root_path) as root:
        original = root._existing_target

        def fail_after_commit(parent_fd, leaf, relative_path):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("PRIVATE post-commit path")
            return original(parent_fd, leaf, relative_path)

        monkeypatch.setattr(root, "_existing_target", fail_after_commit)
        with pytest.raises(SyncPathPartialError) as raised:
            root.replace_bytes("note.md", b"after", expected=None, mode=0o600)

    assert raised.value.reason == "replacement_commit_unverified"
    assert raised.value.__cause__ is None
    assert (root_path / "note.md").read_bytes() == b"after"


def test_replace_bounded_postcommit_refusal_is_distinct_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    calls = 0

    with PinnedSyncRoot(root_path) as root:
        original = root._existing_target

        def hide_committed_target(parent_fd, leaf, relative_path):
            nonlocal calls
            calls += 1
            if calls == 3:
                return None
            return original(parent_fd, leaf, relative_path)

        monkeypatch.setattr(root, "_existing_target", hide_committed_target)
        with pytest.raises(SyncPathPartialError) as raised:
            root.replace_bytes("note.md", b"after", expected=None, mode=0o600)

    assert raised.value.reason == "replacement_postcondition_failed"
    assert (root_path / "note.md").read_bytes() == b"after"


def test_move_bounded_postcommit_refusal_is_distinct_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    source = root_path / "before.md"
    source.write_bytes(b"before")
    calls = 0

    with PinnedSyncRoot(root_path) as root:
        observed = root.read_bytes("before.md")
        original = root._existing_target

        def fail_committed_observation(parent_fd, leaf, relative_path):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise SyncPathError("target_identity_changed", relative_path)
            return original(parent_fd, leaf, relative_path)

        monkeypatch.setattr(root, "_existing_target", fail_committed_observation)
        with pytest.raises(SyncPathPartialError) as raised:
            root.move_file("before.md", "after.md", expected=observed)

    assert raised.value.reason == "move_postcondition_failed"
    assert not source.exists()
    assert (root_path / "after.md").read_bytes() == b"before"


def test_legacy_pathname_sync_engine_remains_retired() -> None:
    assert not Path(sync_paths.__file__).with_name("sync_engine.py").exists()


# --------------------------------------------------------------------------
# create_new_text -- the never-replace counterpart to write_text (task-19554)
# --------------------------------------------------------------------------
