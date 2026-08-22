from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook.Notes import sync_paths
from tldw_chatbook.Notes.sync_paths import PinnedSyncRoot, SyncPathError


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="descriptor containment contract is POSIX-specific",
)


def _issue_reasons(issues) -> set[str]:
    return {issue.reason for issue in issues}


def test_selected_root_link_is_allowed_but_descendant_links_and_hardlinks_are_skipped(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    selected = tmp_path / "selected"
    selected.symlink_to(canonical, target_is_directory=True)
    outside = tmp_path / "OUTSIDE-TASK493-SENTINEL.md"
    outside.write_text("OUTSIDE-TASK493-SENTINEL", encoding="utf-8")
    (canonical / "safe.md").write_text("safe", encoding="utf-8")
    (canonical / "outside-link.md").symlink_to(outside)
    (canonical / "inside-link.md").symlink_to(canonical / "safe.md")
    linked_dir = canonical / "linked-dir"
    linked_dir.symlink_to(tmp_path, target_is_directory=True)
    os.link(outside, canonical / "outside-hardlink.md")

    with PinnedSyncRoot(selected) as root:
        files, issues = root.scan([".md"])

    assert root.lexical_root == selected
    assert root.canonical_root == canonical.resolve(strict=True)
    assert set(files) == {Path("safe.md")}
    assert files[Path("safe.md")].content == "safe"
    assert "OUTSIDE-TASK493-SENTINEL" not in "".join(
        item.content for item in files.values()
    )
    assert {"link_or_reparse", "multiple_links"} <= _issue_reasons(issues)


def test_scan_skips_cross_device_entry_and_keeps_safe_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "safe.md").write_text("safe", encoding="utf-8")
    (root_path / "mounted.md").write_text("mounted", encoding="utf-8")

    with PinnedSyncRoot(root_path) as root:
        original = root._same_device
        monkeypatch.setattr(
            root,
            "_same_device",
            lambda entry_stat: (
                False
                if entry_stat.st_ino == (root_path / "mounted.md").stat().st_ino
                else original(entry_stat)
            ),
        )
        files, issues = root.scan([".md"])

    assert set(files) == {Path("safe.md")}
    assert "cross_device" in _issue_reasons(issues)


def test_scan_skips_simulated_reparse_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    candidate = root_path / "reparse.md"
    candidate.write_text("must not import", encoding="utf-8")
    candidate_inode = candidate.stat().st_ino
    original = sync_paths._is_reparse
    monkeypatch.setattr(
        sync_paths,
        "_is_reparse",
        lambda entry_stat: (
            entry_stat.st_ino == candidate_inode or original(entry_stat)
        ),
    )

    with PinnedSyncRoot(root_path) as root:
        files, issues = root.scan([".md"])

    assert files == {}
    assert _issue_reasons(issues) == {"link_or_reparse"}


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644])
def test_atomic_replacement_preserves_existing_mode(
    tmp_path: Path,
    mode: int,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_text("before", encoding="utf-8")
    target.chmod(mode)

    with PinnedSyncRoot(root_path) as root:
        result = root.write_text(Path("note.md"), "after")

    assert result.content == "after"
    assert target.read_text(encoding="utf-8") == "after"
    assert stat.S_IMODE(target.stat().st_mode) == mode


def test_new_file_and_parent_are_private(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()

    with PinnedSyncRoot(root_path) as root:
        root.write_text(Path("nested/note.md"), "private")

    assert stat.S_IMODE((root_path / "nested").stat().st_mode) == 0o700
    assert stat.S_IMODE((root_path / "nested/note.md").stat().st_mode) == 0o600


def test_final_target_replacement_race_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md"
    target.write_text("original", encoding="utf-8")

    def replace_target(_relative_path: Path) -> None:
        target.unlink()
        target.write_text("racer", encoding="utf-8")

    with PinnedSyncRoot(root_path) as root:
        monkeypatch.setattr(root, "_before_replace", replace_target)
        with pytest.raises(SyncPathError, match="target_identity_changed"):
            root.write_text(Path("note.md"), "unsafe")

    assert target.read_text(encoding="utf-8") == "racer"


def test_intermediate_parent_replacement_race_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    parent = root_path / "parent"
    parent.mkdir(parents=True)
    target = parent / "note.md"
    target.write_text("original", encoding="utf-8")

    def replace_parent(_relative_path: Path) -> None:
        parent.rename(root_path / "moved-parent")
        parent.mkdir()

    with PinnedSyncRoot(root_path) as root:
        monkeypatch.setattr(root, "_before_replace", replace_parent)
        with pytest.raises(SyncPathError, match="parent_identity_changed"):
            root.write_text(Path("parent/note.md"), "unsafe")

    assert target.exists() is False
    assert (root_path / "moved-parent/note.md").read_text(encoding="utf-8") == (
        "original"
    )


def test_unsupported_descriptor_guards_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    monkeypatch.setattr(sync_paths, "_descriptor_guards_available", lambda: False)

    with PinnedSyncRoot(root_path) as root:
        files, issues = root.scan([".md"])
        with pytest.raises(SyncPathError, match="unsupported_platform"):
            root.write_text(Path("note.md"), "unsafe")

    assert files == {}
    assert _issue_reasons(issues) == {"unsupported_platform"}
    assert not (root_path / "note.md").exists()


def test_relative_escape_is_rejected(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()

    with PinnedSyncRoot(root_path) as root:
        with pytest.raises(SyncPathError, match="invalid_relative_path"):
            root.write_text(Path("../escape.md"), "unsafe")

    assert not (tmp_path / "escape.md").exists()


def test_legacy_engine_source_has_no_pathname_sync_escape_hatches() -> None:
    source = (
        Path(sync_paths.__file__).with_name("sync_engine.py").read_text(
            encoding="utf-8"
        )
    )

    assert ".rglob(" not in source
    assert "file_path.read_text(" not in source
    assert "file_path.parent.mkdir(" not in source
    assert "atomic_write_text(" not in source


# --------------------------------------------------------------------------
# create_new_text -- the never-replace counterpart to write_text (task-19554)
# --------------------------------------------------------------------------
def test_create_new_text_refuses_an_existing_name_without_touching_it(
    tmp_path: Path,
) -> None:
    """The property the preserved-conflict-copy path depends on.

    ``write_text`` renames over its target; for a saved copy of user text
    that is destruction. ``create_new_text`` claims the name with
    ``O_CREAT|O_EXCL`` instead, so a taken name is reported, never replaced.
    """
    root_path = tmp_path / "root"
    root_path.mkdir()
    target = root_path / "note.md.conflict-20260821T203015Z-disk.bak"
    target.write_text("someone else's preserved copy", encoding="utf-8")

    with PinnedSyncRoot(root_path) as root:
        with pytest.raises(FileExistsError):
            root.create_new_text(target.relative_to(root_path), "mine")

    assert target.read_text(encoding="utf-8") == "someone else's preserved copy"


def test_create_new_text_writes_a_private_file_and_reports_it(
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()

    with PinnedSyncRoot(root_path) as root:
        result = root.create_new_text(Path("note.md.conflict-x-db.bak"), "kept")

    created = root_path / "note.md.conflict-x-db.bak"
    assert created.read_text(encoding="utf-8") == "kept"
    assert stat.S_IMODE(created.stat().st_mode) == 0o600
    assert result.absolute_path == created
    assert result.content == "kept"


def test_create_new_text_rejects_a_symlinked_name_rather_than_following_it(
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    outside = tmp_path / "OUTSIDE-19554-SENTINEL.bak"
    outside.write_text("OUTSIDE-19554-SENTINEL", encoding="utf-8")
    (root_path / "note.md.conflict-x-db.bak").symlink_to(outside)

    with PinnedSyncRoot(root_path) as root:
        with pytest.raises(OSError):
            root.create_new_text(Path("note.md.conflict-x-db.bak"), "leaked")

    assert outside.read_text(encoding="utf-8") == "OUTSIDE-19554-SENTINEL"


def test_create_new_text_refuses_a_missing_parent(tmp_path: Path) -> None:
    """A sidecar goes beside an existing note; it never creates directories."""
    root_path = tmp_path / "root"
    root_path.mkdir()

    with PinnedSyncRoot(root_path) as root:
        with pytest.raises(SyncPathError, match="missing_parent"):
            root.create_new_text(Path("nope/note.md.conflict-x-db.bak"), "x")

    assert not (root_path / "nope").exists()


def test_create_new_text_leaves_nothing_behind_when_it_fails_after_creating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller told the write failed must not find a half-written file."""
    root_path = tmp_path / "root"
    root_path.mkdir()

    def explode(_file_fd: int, _content: bytes) -> None:
        raise OSError("short write")

    with PinnedSyncRoot(root_path) as root:
        monkeypatch.setattr(root, "_write_all", explode)
        with pytest.raises(OSError, match="short write"):
            root.create_new_text(Path("note.md.conflict-x-db.bak"), "partial")

    assert list(root_path.iterdir()) == []
