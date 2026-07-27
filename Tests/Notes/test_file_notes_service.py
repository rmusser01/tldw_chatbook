from __future__ import annotations

import hashlib
import os
import stat
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

# Avoid importing the unrelated optional MLX stack during focused Notes tests.
sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

import tldw_chatbook.Notes.file_notes_service as service_module  # noqa: E402
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica  # noqa: E402
from tldw_chatbook.Notes.file_notes_service import (  # noqa: E402
    MAX_FILE_BYTES,
    MAX_FILE_CHARS,
    FileNotesService,
)


def _digest(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


@pytest.fixture
def replica() -> FileNotesReplica:
    value = FileNotesReplica(":memory:")
    yield value
    value.close()


def test_scan_excludes_git_and_symlinks_and_rejects_unsafe_paths(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    outside = tmp_path / "outside"
    (root / ".git").mkdir(parents=True)
    (root / "nested").mkdir()
    outside.mkdir()
    (root / "visible.md").write_text("visible", encoding="utf-8")
    (root / "nested" / "also.TEXT").write_text("also", encoding="utf-8")
    (root / ".git" / "hidden.md").write_text("hidden", encoding="utf-8")
    (root / "ignored.rst").write_text("ignored", encoding="utf-8")
    (outside / "secret.md").write_text("secret", encoding="utf-8")
    (root / "linked.md").symlink_to(outside / "secret.md")
    (root / "linked-dir").symlink_to(outside, target_is_directory=True)
    service = FileNotesService(root, replica)

    result = service.scan()

    assert not result.offline
    assert [entry.relative_path for entry in result.entries] == [
        "nested/also.TEXT",
        "visible.md",
    ]
    with pytest.raises(ValueError, match="unsafe"):
        service.open_file("../outside/secret.md")
    with pytest.raises(ValueError, match="symlink"):
        service.open_file("linked.md")
    assert service.create_file("linked-dir/new.md", "nope").status == "unsafe"
    assert service.move_file("ignored.rst", "moved.md").status == "unsupported"
    assert (root / "ignored.rst").exists()
    assert service.delete_file("ignored.rst").status == "unsupported"
    assert (root / "ignored.rst").exists()


def test_open_and_save_preserve_bom_frontmatter_crlf_final_newline_and_mode(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    path = root / "note.md"
    raw = b"\xef\xbb\xbf---\r\ntitle: Exact\r\n...\r\nold\r\nbody\r\n"
    path.write_bytes(raw)
    path.chmod(0o640)
    service = FileNotesService(root, replica)

    opened = service.open_file("note.md")

    assert opened.preserved_prefix == b"\xef\xbb\xbf---\r\ntitle: Exact\r\n...\r\n"
    assert opened.body == "old\nbody\n"
    assert opened.newline == "\r\n"
    assert opened.has_final_newline
    assert opened.content_hash == _digest(raw)

    result = service.save_file(
        opened,
        "new\nbody",
        session_key="open-1",
    )

    assert result.status == "ok"
    assert path.read_bytes() == (
        b"\xef\xbb\xbf---\r\ntitle: Exact\r\n...\r\nnew\r\nbody\r\n"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert replica.get_bytes(str(root.resolve()), "note.md") == path.read_bytes()
    assert [(change.action, change.relative_path) for change in service.session_changes] == [
        ("modified", "note.md")
    ]


def test_save_copy_preserves_exact_format_and_never_clobbers(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source_bytes = b"\xef\xbb\xbf---\r\ntitle: Exact\r\n---\r\nold\r\nbody\r\n"
    source.write_bytes(source_bytes)
    occupied = root / "occupied.md"
    occupied.write_bytes(b"keep me")
    service = FileNotesService(root, replica)
    opened = service.open_file(source.name)

    result = service.save_copy(opened, "copied\nbody", "copy.md")

    expected = b"\xef\xbb\xbf---\r\ntitle: Exact\r\n---\r\ncopied\r\nbody\r\n"
    assert result.status == "ok"
    assert source.read_bytes() == source_bytes
    assert (root / "copy.md").read_bytes() == expected
    assert replica.get_bytes(str(root.resolve()), "copy.md") == expected
    assert service.session_changes[-1].action == "created"
    changes = service.session_changes

    existing = service.save_copy(opened, "replacement", occupied.name)

    assert existing.status == "exists"
    assert occupied.read_bytes() == b"keep me"
    assert service.session_changes == changes


@pytest.mark.parametrize("operation", ["save", "create", "restore"])
def test_post_publication_stat_failure_still_records_success(
    operation: str,
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    service = FileNotesService(root, replica)
    target = root / f"{operation}.md"
    if operation == "save":
        target.write_bytes(b"before")
        opened = service.open_file(target.name)
        expected = b"after"
        action = "modified"
    elif operation == "restore":
        target.write_bytes(b"restore me")
        opened = service.open_file(target.name)
        assert (
            service.delete_file(
                target.name,
                expected_hash=opened.content_hash,
            ).status
            == "ok"
        )
        expected = b"restore me"
        action = "restored"
    else:
        expected = b"created"
        action = "created"

    real_stat = Path.stat

    def fail_published_stat(
        candidate: Path,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        if candidate == target:
            raise FileNotFoundError("forced post-publication race")
        return real_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_published_stat)
    if operation == "save":
        result = service.save_file(opened, "after", session_key="post-publish")
    elif operation == "restore":
        result = service.restore_file(target.name)
    else:
        result = service.create_file(target.name, "created")
    monkeypatch.setattr(Path, "stat", real_stat)

    assert result.status == "ok"
    assert result.content_hash == _digest(expected)
    assert result.replica_warning
    assert target.read_bytes() == expected
    assert service.session_changes[-1].action == action

    service.reconcile()
    assert replica.get_bytes(str(root.resolve()), target.name) == expected


def test_open_keeps_unclosed_frontmatter_in_body_and_marks_unsafe_text_read_only(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MAX_FILE_BYTES == 8_000_000
    assert MAX_FILE_CHARS == 2_000_000
    root = tmp_path / "notes"
    root.mkdir()
    (root / "unclosed.md").write_bytes(b"\xef\xbb\xbf---\ntitle: Open\nbody")
    (root / "mixed.md").write_bytes(b"one\r\ntwo\n")
    (root / "binary.txt").write_bytes(b"\xef\xbb\xbf---\ntitle: Raw\n---\n\xff\xfe")
    (root / "bytes.text").write_bytes(b"12345")
    (root / "chars.markdown").write_text("abcd", encoding="utf-8")
    service = FileNotesService(root, replica)

    unclosed = service.open_file("unclosed.md")
    assert unclosed.preserved_prefix == b"\xef\xbb\xbf"
    assert unclosed.body == "---\ntitle: Open\nbody"
    assert unclosed.editable
    assert (
        service.save_file(unclosed, "edited\n\n", session_key="no-final").status
        == "ok"
    )
    assert (root / "unclosed.md").read_bytes() == b"\xef\xbb\xbfedited"
    assert service.open_file("mixed.md").read_only_reason == "mixed-newlines"
    binary = service.open_file("binary.txt")
    assert binary.read_only_reason == "undecodable-utf8"
    assert binary.preserved_prefix == b"\xef\xbb\xbf---\ntitle: Raw\n---\n"

    monkeypatch.setattr(service_module, "MAX_FILE_BYTES", 4)

    large = service.open_file("bytes.text")
    assert large.read_only_reason == "too-many-bytes"
    assert large.raw_bytes == b"12345"
    monkeypatch.setattr(service_module, "MAX_FILE_BYTES", 8_000_000)
    monkeypatch.setattr(service_module, "MAX_FILE_CHARS", 3)
    assert service.open_file("chars.markdown").read_only_reason == "too-many-chars"


def test_oversized_file_keeps_exact_replica_and_can_delete_and_restore(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    large_path = root / "large.md"
    grows_path = root / "grows.md"
    large_path.write_bytes(b"12345")
    grows_path.write_bytes(b"old")
    monkeypatch.setattr(service_module, "MAX_FILE_BYTES", 4)
    service = FileNotesService(root, replica)

    scanned = service.scan()

    scanned_by_path = {entry.relative_path: entry for entry in scanned.entries}
    assert scanned_by_path["large.md"].read_only_reason == "too-many-bytes"
    assert replica.get_bytes(str(root.resolve()), "large.md") == b"12345"
    replica_by_path = {
        item.relative_path: item
        for item in replica.list_active_files(str(root.resolve()))
    }
    assert replica_by_path["large.md"].content_hash == _digest(b"12345")

    upsert_calls = 0
    real_upsert = replica.upsert_file

    def record_upsert(*args: object, **kwargs: object) -> None:
        nonlocal upsert_calls
        upsert_calls += 1
        real_upsert(*args, **kwargs)

    monkeypatch.setattr(replica, "upsert_file", record_upsert)
    assert service.reconcile().modified == ()
    assert upsert_calls == 0

    assert replica.search(str(root.resolve()), "old") == ["grows.md"]
    grows_path.write_bytes(b"changed")
    reconciled = service.reconcile()
    assert reconciled.modified == ("grows.md",)
    assert upsert_calls == 1
    assert replica.get_bytes(str(root.resolve()), "grows.md") == b"changed"
    assert replica.search(str(root.resolve()), "old") == []

    opened = service.open_file("large.md")
    assert opened.content_hash == _digest(b"12345")
    deleted = service.delete_file("large.md", expected_hash=opened.content_hash)
    assert deleted.status == "ok"
    assert not large_path.exists()
    assert replica.get_restore_bytes(str(root.resolve()), "large.md") == b"12345"

    restored = service.restore_file("large.md")
    assert restored.status == "ok"
    assert large_path.read_bytes() == b"12345"


def test_save_conflict_and_protected_checkpoint_failure_never_write(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    path = root / "protected.md"
    path.write_text("original\n", encoding="utf-8")
    service = FileNotesService(root, replica)
    opened = service.open_file("protected.md")

    path.write_text("external\n", encoding="utf-8")
    assert service.save_file(opened, "draft\n", session_key="open-1").status == (
        "conflict"
    )
    assert path.read_text(encoding="utf-8") == "external\n"

    opened = service.open_file("protected.md")
    replica.protect(str(root.resolve()), "protected.md")

    def fail_checkpoint(*args: object, **kwargs: object) -> bool:
        raise RuntimeError("replica unavailable")

    monkeypatch.setattr(replica, "checkpoint", fail_checkpoint)
    result = service.save_file(opened, "must not write\n", session_key="open-2")

    assert result.status == "replica-error"
    assert path.read_text(encoding="utf-8") == "external\n"


def test_save_rechecks_hash_immediately_before_replace_and_cleans_temp(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    path = root / "race.md"
    path.write_text("baseline\n", encoding="utf-8")
    service = FileNotesService(root, replica)
    opened = service.open_file("race.md")
    real_mkstemp = service_module.tempfile.mkstemp

    def change_after_temp(*args: object, **kwargs: object) -> tuple[int, str]:
        fd, temp_path = real_mkstemp(*args, **kwargs)
        path.write_text("external race\n", encoding="utf-8")
        return fd, temp_path

    monkeypatch.setattr(service_module.tempfile, "mkstemp", change_after_temp)
    result = service.save_file(opened, "draft\n", session_key="open-race")

    assert result.status == "conflict"
    assert path.read_text(encoding="utf-8") == "external race\n"
    assert list(root.glob("*.tmp")) == []


def test_create_is_exclusive_and_move_is_no_clobber_with_rollback(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    (root / "folder").mkdir(parents=True)
    service = FileNotesService(root, replica)

    assert service.create_file("source.md", "source").status == "ok"
    assert service.create_file("source.md", "replacement").status == "exists"
    (root / "folder" / "target.md").write_text("target", encoding="utf-8")
    assert service.move_file("source.md", "folder/target.md").status == "exists"
    assert (root / "source.md").read_text(encoding="utf-8") == "source"

    real_unlink = service_module.os.unlink

    def fail_source_unlink(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        if Path(path) == root / "source.md":
            raise PermissionError("forced source failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(service_module.os, "unlink", fail_source_unlink)
    result = service.move_file("source.md", "folder/moved.md")

    assert result.status == "error"
    assert (root / "source.md").exists()
    assert not (root / "folder" / "moved.md").exists()


def test_delete_rechecks_after_tombstone_and_restore_requires_absent_destination(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    path = root / "delete.md"
    original = b"exact deletion bytes\r\n"
    path.write_bytes(original)
    service = FileNotesService(root, replica)
    opened = service.open_file("delete.md")
    real_prepare = replica.prepare_deletion

    def change_after_prepare(*args: object, **kwargs: object) -> None:
        real_prepare(*args, **kwargs)
        path.write_bytes(b"external change")

    monkeypatch.setattr(replica, "prepare_deletion", change_after_prepare)
    result = service.delete_file("delete.md", expected_hash=opened.content_hash)

    assert result.status == "conflict"
    assert path.read_bytes() == b"external change"
    assert replica.list_deleted(str(root.resolve())) == []

    monkeypatch.setattr(replica, "prepare_deletion", real_prepare)
    opened = service.open_file("delete.md")
    assert (
        service.delete_file("delete.md", expected_hash=opened.content_hash).status
        == "ok"
    )
    assert not path.exists()
    path.write_bytes(b"occupied")
    assert service.restore_file("delete.md").status == "exists"
    assert path.read_bytes() == b"occupied"
    path.unlink()
    assert service.restore_file("delete.md").status == "ok"
    assert path.read_bytes() == b"external change"


def test_delete_clears_tombstone_when_unlink_fails(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    path = root / "keep.md"
    path.write_text("keep", encoding="utf-8")
    service = FileNotesService(root, replica)
    opened = service.open_file("keep.md")

    def fail_unlink(*args: object, **kwargs: object) -> None:
        raise PermissionError("forced")

    monkeypatch.setattr(service_module.os, "unlink", fail_unlink)
    result = service.delete_file("keep.md", expected_hash=opened.content_hash)

    assert result.status == "error"
    assert path.exists()
    assert replica.list_deleted(str(root.resolve())) == []


def test_delete_clears_tombstone_when_final_reread_becomes_non_regular(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    path = root / "changed-kind.md"
    path.write_text("keep", encoding="utf-8")
    service = FileNotesService(root, replica)
    opened = service.open_file("changed-kind.md")
    real_read = service_module._read_regular_file
    read_count = 0

    def fail_final_read(candidate: Path) -> tuple[bytes, os.stat_result]:
        nonlocal read_count
        read_count += 1
        if read_count == 2:
            raise ValueError("unsafe non-regular file")
        return real_read(candidate)

    monkeypatch.setattr(service_module, "_read_regular_file", fail_final_read)
    result = service.delete_file(
        "changed-kind.md",
        expected_hash=opened.content_hash,
    )

    assert result.status in {"conflict", "error"}
    assert path.exists()
    assert replica.list_deleted(str(root.resolve())) == []


def test_public_operations_and_session_changes_share_one_service_lock(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    path = root / "locked.md"
    path.write_text("locked", encoding="utf-8")
    service = FileNotesService(root, replica)
    opened = service.open_file("locked.md")
    prepared = Event()
    release_delete = Event()
    reconcile_entered = Event()
    real_prepare = replica.prepare_deletion
    real_list_active = replica.list_active_files

    def block_after_prepare(*args: object, **kwargs: object) -> None:
        real_prepare(*args, **kwargs)
        prepared.set()
        assert release_delete.wait(timeout=2)

    def record_reconcile_entry(*args: object, **kwargs: object) -> object:
        reconcile_entered.set()
        return real_list_active(*args, **kwargs)

    monkeypatch.setattr(replica, "prepare_deletion", block_after_prepare)
    monkeypatch.setattr(replica, "list_active_files", record_reconcile_entry)

    with ThreadPoolExecutor(max_workers=3) as executor:
        deletion = executor.submit(
            service.delete_file,
            "locked.md",
            expected_hash=opened.content_hash,
        )
        assert prepared.wait(timeout=1)
        reconciliation = executor.submit(service.reconcile)
        session_read = executor.submit(lambda: service.session_changes)
        try:
            assert not reconcile_entered.wait(timeout=0.1)
            assert not session_read.done()
        finally:
            release_delete.set()

        assert deletion.result(timeout=1).status == "ok"
        assert reconciliation.result(timeout=1).status == "ok"
        assert session_read.result(timeout=1)[-1].action == "deleted"

    assert replica.list_deleted(str(root.resolve())) == ["locked.md"]


def test_double_dot_filename_is_safe_for_file_mutations(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    service = FileNotesService(root, replica)

    created = service.create_file("meeting..draft.md", "draft")
    assert created.status == "ok"
    opened = service.open_file("meeting..draft.md")
    assert service.save_file(opened, "edited", session_key="double-dot").status == "ok"
    assert (
        service.move_file("meeting..draft.md", "meeting..final.md").status
        == "ok"
    )
    moved = service.open_file("meeting..final.md")
    assert (
        service.delete_file(
            "meeting..final.md",
            expected_hash=moved.content_hash,
        ).status
        == "ok"
    )
    assert service.restore_file("meeting..final.md").status == "ok"
    assert (root / "meeting..final.md").read_text(encoding="utf-8") == "edited"


@pytest.mark.skipif(os.name == "nt", reason="Backslash is a separator on Windows")
def test_paths_require_canonical_posix_spelling_but_allow_backslash_filename(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    (root / "folder").mkdir(parents=True)
    (root / "folder" / "note.md").write_text("note", encoding="utf-8")
    service = FileNotesService(root, replica)

    with pytest.raises(ValueError, match="canonical"):
        service.open_file("folder//note.md")
    assert service.create_file("./new.md", "nope").status == "unsafe"

    backslash_name = r"meeting\notes.md"
    assert service.create_file(backslash_name, "legal").status == "ok"
    opened = service.open_file(backslash_name)
    assert opened.body == "legal"


def test_reconcile_projects_external_changes_without_session_changes(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "modified.md").write_text("old", encoding="utf-8")
    (root / "deleted.md").write_text("gone", encoding="utf-8")
    service = FileNotesService(root, replica)
    assert not service.scan().offline
    real_read = service_module._read_regular_file
    reads: list[str] = []

    def count_reads(path: Path) -> tuple[bytes, os.stat_result]:
        reads.append(path.name)
        return real_read(path)

    monkeypatch.setattr(service_module, "_read_regular_file", count_reads)
    unchanged = service.reconcile()
    assert unchanged.created == unchanged.modified == unchanged.deleted == ()
    assert reads == []

    (root / "modified.md").write_text("new content", encoding="utf-8")
    (root / "deleted.md").unlink()
    (root / "created.txt").write_text("created", encoding="utf-8")
    result = service.reconcile()

    assert result.status == "ok"
    assert result.created == ("created.txt",)
    assert result.modified == ("modified.md",)
    assert result.deleted == ("deleted.md",)
    assert sorted(reads) == ["created.txt", "modified.md"]
    assert service.session_changes == ()
    assert replica.search(str(root.resolve()), "created") == ["created.txt"]
    assert replica.list_deleted(str(root.resolve())) == ["deleted.md"]


def test_reconcile_does_not_tombstone_a_present_file_when_its_read_fails(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    keep = root / "keep.md"
    gone = root / "gone.md"
    keep.write_text("old", encoding="utf-8")
    gone.write_text("gone", encoding="utf-8")
    service = FileNotesService(root, replica)
    service.scan()
    gone.unlink()
    keep.write_text("changed", encoding="utf-8")
    real_read = service_module._read_regular_file

    def fail_keep(path: Path) -> tuple[bytes, os.stat_result]:
        if path == keep:
            raise PermissionError("transient")
        return real_read(path)

    monkeypatch.setattr(service_module, "_read_regular_file", fail_keep)
    result = service.reconcile()

    assert result.deleted == ("gone.md",)
    assert replica.list_deleted(str(root.resolve())) == ["gone.md"]
    assert "keep.md" in {
        item.relative_path
        for item in replica.list_active_files(str(root.resolve()))
    }


def test_offline_reconcile_does_not_create_or_touch_replica(tmp_path: Path) -> None:
    class FailingReplica:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"replica must not be touched: {name}")

    root = tmp_path / "missing"
    service = FileNotesService(root, FailingReplica())  # type: ignore[arg-type]

    result = service.reconcile()

    assert result.status == "offline"
    assert result.offline
    assert not root.exists()


def test_replica_failure_warns_but_allows_unprotected_save_and_blocks_delete(
    tmp_path: Path,
) -> None:
    class UnavailableReplica:
        def __getattr__(self, name: str) -> object:
            raise RuntimeError(f"{name} unavailable")

    root = tmp_path / "notes"
    root.mkdir()
    path = root / "note.md"
    path.write_text("old\n", encoding="utf-8")
    service = FileNotesService(root, UnavailableReplica())  # type: ignore[arg-type]
    opened = service.open_file("note.md")

    saved = service.save_file(opened, "new\n", session_key="open-1")

    assert saved.status == "ok"
    assert saved.replica_warning
    assert path.read_text(encoding="utf-8") == "new\n"
    assert service.delete_file("note.md", expected_hash=saved.content_hash).status == (
        "replica-error"
    )
    assert path.exists()
