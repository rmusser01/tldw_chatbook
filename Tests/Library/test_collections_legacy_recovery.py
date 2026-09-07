"""Recovery-only access to the superseded generic Collections tables."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from pathlib import Path

import pytest

from tldw_chatbook.DB.Library_Collections_DB import (
    LibraryCollectionsDB,
    LibraryCollectionsSchemaError,
)
from tldw_chatbook.Library import collections_legacy_recovery as recovery_module
from tldw_chatbook.Library.collections_legacy_recovery import (
    LegacyCollectionsRecovery,
    LegacyCollectionsRecoveryError,
)


def _seed_legacy(db: LibraryCollectionsDB, count: int = 45) -> None:
    collections = []
    memberships = []
    for index in range(count):
        collection_id = f"legacy-{index:03d}"
        collections.append(
            (
                collection_id,
                f"Stored name {index:03d}",
                f"Stored description <{index:03d}>",
                f"2026-08-01T00:{index:02d}:00Z",
                f"2026-08-02T00:{index:02d}:00Z",
                "2026-08-03T00:00:00Z" if index % 7 == 0 else None,
            )
        )
        memberships.append(
            (
                f"membership-{index:03d}",
                collection_id,
                "unknown-source" if index % 2 else "note",
                f"source-{index:03d}",
                f"Stored member title {index:03d}",
                f"2026-08-04T00:{index:02d}:00Z",
            )
        )
    with db.transaction() as connection:
        connection.executemany(
            "INSERT INTO library_collections (collection_id, name, description, "
            "created_at, updated_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?)",
            collections,
        )
        connection.executemany(
            "INSERT INTO library_collection_items (membership_id, collection_id, "
            "source_type, source_id, title, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            memberships,
        )


def _identity(path: Path) -> tuple[int, int]:
    metadata = os.lstat(path)
    return metadata.st_dev, metadata.st_ino


def test_bounded_inspection_reaches_active_and_deleted_rows(tmp_path: Path) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db)
    recovery = LegacyCollectionsRecovery(db)

    pages = [recovery.list_collections(page=page, size=20) for page in (1, 2, 3)]
    memberships = [recovery.list_memberships(page=page, size=20) for page in (1, 2, 3)]

    assert [len(page.items) for page in pages] == [20, 20, 5]
    assert {page.total for page in pages} == {45}
    assert [item.collection_id for page in pages for item in page.items] == [
        f"legacy-{index:03d}" for index in range(45)
    ]
    assert pages[0].items[0].deleted_at == "2026-08-03T00:00:00Z"
    assert pages[0].items[1].deleted_at is None
    assert [len(page.items) for page in memberships] == [20, 20, 5]
    assert {page.total for page in memberships} == {45}
    assert memberships[2].items[-1].membership_id == "membership-044"
    db.close()


def test_inspection_rejects_unbounded_or_invalid_pages(tmp_path: Path) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    recovery = LegacyCollectionsRecovery(db)

    for page, size in (
        (0, 20),
        (1, 0),
        (1, 101),
        (True, 20),
        (1, True),
        (2**63, 20),
    ):
        with pytest.raises(LegacyCollectionsRecoveryError) as caught:
            recovery.list_collections(page=page, size=size)
        assert caught.value.reason == "invalid_legacy_recovery_page"
    db.close()


def test_export_batch_size_is_bounded(tmp_path: Path) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")

    for batch_size in (0, True, 101):
        with pytest.raises(LegacyCollectionsRecoveryError) as caught:
            LegacyCollectionsRecovery(db, export_batch_size=batch_size)
        assert caught.value.reason == "invalid_legacy_export_batch"
    db.close()


def test_export_does_not_require_posix_descriptor_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db, count=1)
    recovery = LegacyCollectionsRecovery(db)
    monkeypatch.delattr(os, "fchmod", raising=False)
    destination = tmp_path / "portable.json"

    recovery.export_json(destination, overwrite_identity=None)

    assert json.loads(destination.read_text(encoding="utf-8"))["version"] == 1
    db.close()


def test_export_explicitly_marks_unverified_platform_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db, count=1)
    recovery = LegacyCollectionsRecovery(db)
    monkeypatch.setattr(
        recovery_module,
        "_secure_dirfd_publication_available",
        lambda: False,
    )
    destination = tmp_path / "unsupported.json"

    assert recovery.export_publication_posture == "unverified_platform"
    recovery.export_json(destination, overwrite_identity=None)

    assert json.loads(destination.read_text(encoding="utf-8"))["version"] == 1
    assert list(tmp_path.glob(".unsupported.json.*.tmp")) == []
    db.close()


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("linux", True), ("linux-aarch64", True), ("darwin", False), ("win32", False)],
)
def test_verified_parent_authority_is_limited_to_linux_mode_semantics(
    platform: str,
    expected: bool,
) -> None:
    assert (
        recovery_module._traditional_mode_write_authority_complete(platform) is expected
    )


def test_unverified_fallback_accepts_platform_without_posix_mode_bits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db, count=1)
    recovery = LegacyCollectionsRecovery(db)
    monkeypatch.setattr(
        recovery_module,
        "_secure_dirfd_publication_available",
        lambda: False,
    )
    monkeypatch.setattr(
        recovery_module,
        "_posix_private_mode_verifiable",
        lambda: False,
    )
    destination = tmp_path / "windows-mode.json"

    def expose_windows_mode() -> None:
        [temporary] = tmp_path.glob(".windows-mode.json.*.tmp")
        temporary.chmod(0o666)

    recovery._before_publish = expose_windows_mode
    recovery.export_json(destination, overwrite_identity=None)

    assert json.loads(destination.read_text(encoding="utf-8"))["version"] == 1
    assert list(tmp_path.glob(".windows-mode.json.*.tmp")) == []
    db.close()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX ownership and modes")
def test_verified_export_rejects_parent_writable_by_other_principals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db, count=1)
    recovery = LegacyCollectionsRecovery(db)
    monkeypatch.setattr(
        recovery_module,
        "_traditional_mode_write_authority_complete",
        lambda: True,
    )
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir(mode=0o700)
    shared_parent.chmod(0o777)
    destination = shared_parent / "legacy.json"

    with pytest.raises(LegacyCollectionsRecoveryError) as caught:
        recovery.export_json(destination, overwrite_identity=None)

    assert caught.value.reason == "invalid_legacy_export_destination"
    assert not destination.exists()
    assert list(shared_parent.glob(".legacy.json.*.tmp")) == []
    db.close()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX ownership and modes")
def test_verified_export_refuses_parent_permission_widening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db, count=1)
    recovery = LegacyCollectionsRecovery(db)
    monkeypatch.setattr(
        recovery_module,
        "_traditional_mode_write_authority_complete",
        lambda: True,
    )
    selected_parent = tmp_path / "selected"
    selected_parent.mkdir(mode=0o700)
    destination = selected_parent / "legacy.json"

    def widen_parent_permissions() -> None:
        selected_parent.chmod(0o777)

    recovery._before_publish = widen_parent_permissions
    try:
        with pytest.raises(LegacyCollectionsRecoveryError) as caught:
            recovery.export_json(destination, overwrite_identity=None)

        assert caught.value.reason == "legacy_export_parent_changed"
        assert not destination.exists()
        assert list(selected_parent.glob(".legacy.json.*.tmp")) == []
    finally:
        selected_parent.chmod(0o700)
        db.close()


def test_export_preserves_exact_v1_records_and_private_atomic_file(
    tmp_path: Path,
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db)
    recovery = LegacyCollectionsRecovery(
        db,
        clock=lambda: "2026-09-01T12:34:56Z",
        export_batch_size=7,
    )
    batches: list[tuple[str, int]] = []
    recovery._on_export_batch = lambda kind, count: batches.append((kind, count))
    destination = tmp_path / "legacy-export.json"

    exported = recovery.export_json(destination, overwrite_identity=None)

    assert exported == destination
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert list(payload) == [
        "format",
        "version",
        "exported_at",
        "collections",
        "memberships",
    ]
    assert payload["format"] == "tldw-chatbook-legacy-collections"
    assert payload["version"] == 1
    assert payload["exported_at"] == "2026-09-01T12:34:56Z"
    assert len(payload["collections"]) == 45
    assert len(payload["memberships"]) == 45
    assert list(payload["collections"][0]) == [
        "collection_id",
        "name",
        "description",
        "created_at",
        "updated_at",
        "deleted_at",
    ]
    assert payload["collections"][0]["description"] == "Stored description <000>"
    assert payload["collections"][-1]["collection_id"] == "legacy-044"
    assert list(payload["memberships"][0]) == [
        "membership_id",
        "collection_id",
        "source_type",
        "source_id",
        "title",
        "created_at",
    ]
    assert payload["memberships"][1]["source_type"] == "unknown-source"
    assert not any("capture" in key for key in payload["collections"][0])
    assert max(count for _kind, count in batches) <= 7
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert list(tmp_path.glob(".legacy-export.json.*.tmp")) == []
    db.close()


def test_export_is_one_snapshot_while_writer_commits(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    db_reader = LibraryCollectionsDB(path)
    db_writer = LibraryCollectionsDB(path)
    _seed_legacy(db_reader, count=2)
    recovery = LegacyCollectionsRecovery(db_reader)
    collections_written = threading.Event()
    writer_done = threading.Event()

    def pause_after_collections() -> None:
        collections_written.set()
        assert writer_done.wait(timeout=10)

    recovery._after_export_collections = pause_after_collections
    destination = tmp_path / "snapshot.json"
    failures: list[BaseException] = []

    def export() -> None:
        try:
            recovery.export_json(destination, overwrite_identity=None)
        except BaseException as exc:  # noqa: BLE001 - preserve thread failure
            failures.append(exc)

    thread = threading.Thread(target=export)
    thread.start()
    assert collections_written.wait(timeout=10)
    with db_writer.transaction() as connection:
        connection.execute(
            "INSERT INTO library_collections (collection_id, name, description, "
            "created_at, updated_at, deleted_at) VALUES "
            "('legacy-new', 'new', '', 'now', 'now', NULL)"
        )
        connection.execute(
            "INSERT INTO library_collection_items (membership_id, collection_id, "
            "source_type, source_id, title, created_at) VALUES "
            "('membership-new', 'legacy-new', 'note', 'note-new', 'new', 'now')"
        )
    writer_done.set()
    thread.join(timeout=10)

    assert failures == []
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert [item["collection_id"] for item in payload["collections"]] == [
        "legacy-000",
        "legacy-001",
    ]
    assert [item["membership_id"] for item in payload["memberships"]] == [
        "membership-000",
        "membership-001",
    ]
    db_reader.close()
    db_writer.close()


def test_export_requires_exact_overwrite_identity(tmp_path: Path) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db, count=1)
    recovery = LegacyCollectionsRecovery(db)
    destination = tmp_path / "legacy.json"
    destination.write_text("original", encoding="utf-8")

    with pytest.raises(LegacyCollectionsRecoveryError) as caught:
        recovery.export_json(destination, overwrite_identity=None)
    assert caught.value.reason == "legacy_export_target_exists"
    with pytest.raises(LegacyCollectionsRecoveryError) as caught:
        recovery.export_json(destination, overwrite_identity=(0, 0))
    assert caught.value.reason == "legacy_export_target_changed"
    assert destination.read_text(encoding="utf-8") == "original"

    identity = _identity(destination)
    recovery.export_json(destination, overwrite_identity=identity)
    assert json.loads(destination.read_text(encoding="utf-8"))["version"] == 1
    assert _identity(destination) != identity
    assert list(tmp_path.glob(".legacy.json.*.tmp")) == []
    db.close()


@pytest.mark.parametrize("overwrite", [False, True])
def test_export_refuses_target_replacement_at_publication(
    tmp_path: Path,
    overwrite: bool,
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db, count=1)
    recovery = LegacyCollectionsRecovery(db)
    destination = tmp_path / "raced.json"
    overwrite_identity = None
    if overwrite:
        destination.write_text("confirmed", encoding="utf-8")
        overwrite_identity = _identity(destination)

    def replace_target() -> None:
        destination.unlink(missing_ok=True)
        destination.write_text("racer", encoding="utf-8")

    recovery._before_publish = replace_target
    with pytest.raises(LegacyCollectionsRecoveryError) as caught:
        recovery.export_json(
            destination,
            overwrite_identity=overwrite_identity,
        )

    assert caught.value.reason == "legacy_export_target_changed"
    assert destination.read_text(encoding="utf-8") == "racer"
    assert list(tmp_path.glob(".raced.json.*.tmp")) == []
    db.close()


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("replacement_kind", ["regular", "symlink"])
def test_export_refuses_temporary_sibling_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
    replacement_kind: str,
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db, count=1)
    recovery = LegacyCollectionsRecovery(db)
    if os.name == "posix":
        monkeypatch.setattr(
            recovery_module,
            "_traditional_mode_write_authority_complete",
            lambda: True,
        )
    destination = tmp_path / "temporary-raced.json"
    overwrite_identity = None
    if overwrite:
        destination.write_text("confirmed", encoding="utf-8")
        overwrite_identity = _identity(destination)
    attacker = tmp_path / "attacker-content"
    attacker.write_text("attacker", encoding="utf-8")

    def replace_temporary() -> None:
        temporary_files = list(tmp_path.glob(".temporary-raced.json.*.tmp"))
        assert len(temporary_files) == 1
        temporary = temporary_files[0]
        temporary.unlink()
        if replacement_kind == "regular":
            temporary.write_text("attacker", encoding="utf-8")
        else:
            temporary.symlink_to(attacker)

    recovery._before_publish = replace_temporary
    with pytest.raises(LegacyCollectionsRecoveryError) as caught:
        recovery.export_json(
            destination,
            overwrite_identity=overwrite_identity,
        )

    assert caught.value.reason == "legacy_export_temporary_changed"
    if overwrite:
        assert destination.read_text(encoding="utf-8") == "confirmed"
    else:
        assert not destination.exists()
    assert list(tmp_path.glob(".temporary-raced.json.*.tmp")) == []
    db.close()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX directory rename")
@pytest.mark.parametrize("swap_moment", ["before_parent_open", "before_publish"])
def test_export_refuses_parent_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_moment: str,
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db, count=1)
    recovery = LegacyCollectionsRecovery(db)
    monkeypatch.setattr(
        recovery_module,
        "_traditional_mode_write_authority_complete",
        lambda: True,
    )
    selected_parent = tmp_path / "selected"
    moved_parent = tmp_path / "selected-moved"
    attacker_parent = tmp_path / "attacker"
    selected_parent.mkdir()
    attacker_parent.mkdir()
    destination = selected_parent / "legacy.json"

    def replace_parent() -> None:
        selected_parent.rename(moved_parent)
        selected_parent.symlink_to(attacker_parent, target_is_directory=True)

    if swap_moment == "before_parent_open":
        recovery._before_parent_open = replace_parent
    else:
        recovery._before_publish = replace_parent

    with pytest.raises(LegacyCollectionsRecoveryError) as caught:
        recovery.export_json(destination, overwrite_identity=None)

    assert caught.value.reason == "legacy_export_parent_changed"
    assert not (attacker_parent / destination.name).exists()
    assert not (moved_parent / destination.name).exists()
    assert list(attacker_parent.glob(".legacy.json.*.tmp")) == []
    assert list(moved_parent.glob(".legacy.json.*.tmp")) == []
    db.close()


def test_export_rejects_unsafe_paths_without_logging_private_values(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(db, count=1)
    recovery = LegacyCollectionsRecovery(db)
    secret_path = tmp_path / "nested" / ".." / ".." / "private-value.json"

    with pytest.raises(LegacyCollectionsRecoveryError) as caught:
        recovery.export_json(secret_path, overwrite_identity=None)

    assert caught.value.reason == "invalid_legacy_export_destination"
    rendered_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "private-value" not in rendered_logs
    assert "Stored name" not in rendered_logs
    assert "Stored description" not in rendered_logs
    assert "legacy-000" not in rendered_logs
    assert "private-value" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    db.close()


def test_recovery_path_opens_legacy_tables_when_capture_schema_is_too_new(
    tmp_path: Path,
) -> None:
    path = tmp_path / "future.db"
    db = LibraryCollectionsDB(path)
    _seed_legacy(db, count=2)
    db.close()
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO schema_version(version) VALUES (99)")

    with pytest.raises(LibraryCollectionsSchemaError) as caught:
        LibraryCollectionsDB(path)
    assert caught.value.reason == "schema_too_new"

    recovery = LegacyCollectionsRecovery(path)
    page = recovery.list_collections(page=1, size=20)
    destination = tmp_path / "future-recovery.json"
    recovery.export_json(destination, overwrite_identity=None)

    assert page.total == 2
    assert (
        json.loads(destination.read_text(encoding="utf-8"))["collections"][1][
            "collection_id"
        ]
        == "legacy-001"
    )


def test_recovery_normalizes_database_disappearance_without_path_disclosure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "private-database-name.db"
    original_is_file = Path.is_file
    original_resolve = Path.resolve

    def raced_is_file(candidate: Path) -> bool:
        if candidate == path:
            return True
        return original_is_file(candidate)

    def raced_resolve(candidate: Path, *args, **kwargs) -> Path:
        if candidate == path:
            raise FileNotFoundError(str(path))
        return original_resolve(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", raced_is_file)
    monkeypatch.setattr(Path, "resolve", raced_resolve)

    with pytest.raises(LegacyCollectionsRecoveryError) as caught:
        LegacyCollectionsRecovery(path)

    assert caught.value.reason == "legacy_database_unavailable"
    assert "private-database-name" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_recovery_refuses_incompatible_legacy_tables(tmp_path: Path) -> None:
    path = tmp_path / "incompatible.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE library_collections(collection_id TEXT)")

    recovery = LegacyCollectionsRecovery(path)
    with pytest.raises(LegacyCollectionsRecoveryError) as caught:
        recovery.list_collections(page=1, size=20)

    assert caught.value.reason == "legacy_schema_unavailable"
