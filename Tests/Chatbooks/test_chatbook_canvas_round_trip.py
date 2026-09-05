"""Whole-graph Canvas archive export/import regression tests."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import zipfile
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import tldw_chatbook.Chatbooks.chatbook_importer as importer_module
from tldw_chatbook.Canvas.archive import export_canvas_archive
from tldw_chatbook.Canvas.repository import (
    CanvasImportBatch,
    CanvasImportDocument,
    CanvasImportRevision,
    CanvasRepository,
)
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _digest(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()


def _seed_canvas_graph(path: Path) -> dict[str, object]:
    db = CharactersRAGDB(path, client_id="canvas-archive-source")
    conversation_id = str(uuid4())
    message_ids = [str(uuid4()) for _ in range(4)]
    db.add_conversation(
        {
            "id": conversation_id,
            "root_id": conversation_id,
            "title": "Canvas archive graph",
        }
    )
    for index, message_id in enumerate(message_ids):
        db.add_message(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "parent_message_id": message_ids[index - 1] if index else None,
                "sender": "assistant" if index else "user",
                "content": f"historical message {index}",
                "timestamp": f"2026-09-04T12:0{index}:00+00:00",
            }
        )
    db.get_connection().execute(
        "UPDATE messages SET deleted = 1 WHERE id = ?",
        (message_ids[3],),
    )
    db.get_connection().execute(
        "UPDATE conversations SET active_leaf_message_id = ? WHERE id = ?",
        (message_ids[2], conversation_id),
    )

    canvas_ids = [str(uuid4()), str(uuid4())]
    revision_ids = [str(uuid4()) for _ in range(4)]
    sources = (
        "<!doctype html><main>root ☃</main>",
        "<!doctype html><main>left λ</main>",
        "<!doctype html><main>right 🌿</main>",
        "<!doctype html><main>future inert</main>",
    )
    created = [f"2026-09-04T12:1{index}:00+00:00" for index in range(4)]
    repository = CanvasRepository(db)
    repository.import_batch(
        CanvasImportBatch(
            conversation_id=conversation_id,
            documents=(
                CanvasImportDocument(
                    canvas_id=canvas_ids[0],
                    conversation_id=conversation_id,
                    created_at=created[0],
                ),
                CanvasImportDocument(
                    canvas_id=canvas_ids[1],
                    conversation_id=conversation_id,
                    created_at=created[3],
                    deleted_at="2026-09-04T12:30:00+00:00",
                ),
            ),
            revisions=(
                CanvasImportRevision(
                    revision_id=revision_ids[0],
                    canvas_id=canvas_ids[0],
                    parent_revision_id=None,
                    sequence=1,
                    title="Planner",
                    runtime_profile="canvas-v1",
                    source=sources[0],
                    content_sha256=_digest(sources[0]),
                    source_bytes=len(sources[0].encode("utf-8")),
                    actor_kind="assistant",
                    origin_message_id=message_ids[0],
                    origin_turn_id="turn-root",
                    created_at=created[0],
                ),
                CanvasImportRevision(
                    revision_id=revision_ids[1],
                    canvas_id=canvas_ids[0],
                    parent_revision_id=revision_ids[0],
                    sequence=2,
                    title="Planner renamed",
                    runtime_profile="canvas-v1",
                    source=sources[1],
                    content_sha256=_digest(sources[1]),
                    source_bytes=len(sources[1].encode("utf-8")),
                    actor_kind="user_rename",
                    origin_message_id=message_ids[3],
                    origin_turn_id="turn-left",
                    created_at=created[1],
                ),
                CanvasImportRevision(
                    revision_id=revision_ids[2],
                    canvas_id=canvas_ids[0],
                    parent_revision_id=revision_ids[0],
                    sequence=3,
                    title="Planner alternate",
                    runtime_profile="canvas-v1",
                    source=sources[2],
                    content_sha256=_digest(sources[2]),
                    source_bytes=len(sources[2].encode("utf-8")),
                    actor_kind="assistant",
                    origin_message_id=message_ids[2],
                    origin_turn_id="turn-right",
                    created_at=created[2],
                ),
                CanvasImportRevision(
                    revision_id=revision_ids[3],
                    canvas_id=canvas_ids[1],
                    parent_revision_id=None,
                    sequence=1,
                    title="Retired future Canvas",
                    runtime_profile="canvas-v9",
                    source=sources[3],
                    content_sha256=_digest(sources[3]),
                    source_bytes=len(sources[3].encode("utf-8")),
                    actor_kind="user_import",
                    origin_message_id=message_ids[1],
                    origin_turn_id="turn-future",
                    created_at=created[3],
                ),
            ),
            reopen_canvas_id=canvas_ids[0],
        )
    )
    db.close_connection()
    return {
        "conversation_id": conversation_id,
        "message_ids": tuple(message_ids),
        "canvas_ids": tuple(canvas_ids),
        "revision_ids": tuple(revision_ids),
        "sources": sources,
    }


def _rewrite_archive(
    source: Path,
    target: Path,
    transform: Callable[[str, bytes], tuple[str, bytes]],
) -> None:
    with zipfile.ZipFile(source) as archive:
        entries = [(item, archive.read(item.filename)) for item in archive.infolist()]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item, payload in entries:
            name, rewritten = transform(item.filename, payload)
            archive.writestr(name, rewritten)


def test_canvas_v3_whole_graph_round_trips_atomically_as_new(tmp_path: Path) -> None:
    """Dropping any branch, historical origin, tombstone, or hint breaks fidelity."""

    source_path = tmp_path / "source.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "canvas-round-trip.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator-temp"
    creator.temp_dir.mkdir()

    success, _, _ = creator.create_chatbook(
        name="Canvas V3",
        description="whole graph",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )

    assert success is True
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert {item.date_time for item in archive.infolist()} == {
            (1980, 1, 1, 0, 0, 0)
        }
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["version"] == "3.0"
        documents = manifest["canvas"]["documents"]
        assert [item["canvas_id"] for item in documents] == sorted(
            expected["canvas_ids"]
        )
        revisions = {
            revision["revision_id"]: revision
            for document in documents
            for revision in document["revisions"]
        }
        assert set(revisions) == set(expected["revision_ids"])
        assert manifest["canvas"]["reopen_hints"] == [
            {
                "conversation_id": expected["conversation_id"],
                "canvas_id": expected["canvas_ids"][0],
            }
        ]
        assert (
            revisions[expected["revision_ids"][2]]["parent_revision_id"]
            == (expected["revision_ids"][0])
        )
        for revision_id, source in zip(expected["revision_ids"], expected["sources"]):
            record = revisions[revision_id]
            assert record["source_bytes"] == len(source.encode("utf-8"))
            assert record["content_sha256"] == _digest(source)
            assert record["source_path"].endswith(".html.txt")
            assert archive.read(record["source_path"]).decode("utf-8") == source
        assert not any(name.endswith(".html") for name in archive.namelist())

    source_path.unlink()
    assert not source_path.exists()

    target_path = tmp_path / "target.sqlite"
    importer = ChatbookImporter({"ChaChaNotes": str(target_path)})
    importer.temp_dir = tmp_path / "import-temp"
    importer.temp_dir.mkdir()
    status = ImportStatus()
    imported, message = importer.import_chatbook(
        archive_path,
        conflict_resolution=ConflictResolution.RENAME,
        import_status=status,
    )
    assert imported is True, message
    assert status.failed_items == 0

    target = CharactersRAGDB(target_path, client_id="canvas-archive-verify")
    matches = target.get_conversation_by_name("Canvas archive graph")
    assert len(matches) == 1
    new_conversation_id = str(matches[0]["id"])
    assert new_conversation_id != expected["conversation_id"]
    repository = CanvasRepository(target)
    identities = repository.list_identities(new_conversation_id, include_deleted=True)
    assert len(identities) == 2
    assert sum(item.deleted_at is not None for item in identities) == 1
    assert {item.deleted_at for item in identities if item.deleted_at is not None} == {
        "2026-09-04T12:30:00+00:00"
    }
    metadata = repository.list_revision_metadata(
        new_conversation_id, include_deleted=True
    )
    assert [item.title for item in metadata] == [
        "Planner",
        "Planner renamed",
        "Planner alternate",
        "Retired future Canvas",
    ]
    assert [item.runtime_profile for item in metadata] == [
        "canvas-v1",
        "canvas-v1",
        "canvas-v1",
        "canvas-v9",
    ]
    assert [item.actor_kind for item in metadata] == [
        "assistant",
        "user_rename",
        "assistant",
        "user_import",
    ]
    imported_message_rows = (
        target.get_connection()
        .execute(
            "SELECT id, content, deleted FROM messages WHERE conversation_id = ?",
            (new_conversation_id,),
        )
        .fetchall()
    )
    message_by_content = {
        str(row["content"]): str(row["id"]) for row in imported_message_rows
    }
    assert {str(row["content"]): int(row["deleted"]) for row in imported_message_rows}[
        "historical message 3"
    ] == 1
    assert [item.origin_message_id for item in metadata] == [
        message_by_content["historical message 0"],
        message_by_content["historical message 3"],
        message_by_content["historical message 2"],
        message_by_content["historical message 1"],
    ]
    assert [item.origin_turn_id for item in metadata] == [
        str(uuid5(NAMESPACE_URL, f"chatbook:{new_conversation_id}:turn:{turn}"))
        for turn in ("turn-root", "turn-left", "turn-right", "turn-future")
    ]
    assert [item.content_sha256 for item in metadata] == [
        _digest(source) for source in expected["sources"]
    ]
    assert [item.source_bytes for item in metadata] == [
        len(source.encode("utf-8")) for source in expected["sources"]
    ]
    imported_sources = [
        repository.read_revision(
            new_conversation_id, item.revision_id, include_deleted=True
        ).source
        for item in metadata
    ]
    assert imported_sources == list(expected["sources"])
    root, left, right, _future = metadata
    assert left.parent_revision_id == root.revision_id
    assert right.parent_revision_id == root.revision_id
    assert repository.get_reopen_hint(new_conversation_id) == identities[0].canvas_id
    assert (
        target.get_connection()
        .execute("SELECT COUNT(*) FROM sync_log WHERE entity LIKE 'canvas%'")
        .fetchone()[0]
        == 0
    )
    for item in (*identities, *metadata):
        UUID(item.canvas_id)
        assert item.canvas_id not in expected["canvas_ids"]

    target.close_connection()


def test_canvas_export_refuses_stored_digest_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "corrupt-source.sqlite"
    expected = _seed_canvas_graph(source_path)
    with sqlite3.connect(source_path) as connection:
        connection.execute("DROP TRIGGER canvas_revisions_no_update")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE canvas_revisions SET content_sha256 = ? WHERE id = ?",
            ("0" * 64, expected["revision_ids"][0]),
        )
        connection.commit()
    archive_path = tmp_path / "must-not-exist.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()

    success, message, _ = creator.create_chatbook(
        name="corrupt",
        description="corrupt",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )

    assert success is False
    assert "stored_source_identity_mismatch" in message
    assert not archive_path.exists()


def test_canvas_export_rejects_origin_missing_from_staged_conversation_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    """Canvas metadata must describe the exact conversation JSON being archived."""

    source_path = tmp_path / "origin-race.sqlite"
    expected = _seed_canvas_graph(source_path)
    original_export = export_canvas_archive

    def export_with_unstaged_origin(*args, **kwargs):
        canvas = original_export(*args, **kwargs)
        assert canvas is not None
        document = canvas.documents[0]
        revision = replace(document.revisions[0], origin_message_id=str(uuid4()))
        return replace(
            canvas,
            documents=(
                replace(
                    document,
                    revisions=(revision, *document.revisions[1:]),
                ),
                *canvas.documents[1:],
            ),
        )

    monkeypatch.setattr(
        "tldw_chatbook.Chatbooks.chatbook_creator.export_canvas_archive",
        export_with_unstaged_origin,
    )
    archive_path = tmp_path / "must-not-exist.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator-race"
    creator.temp_dir.mkdir()

    success, message, _ = creator.create_chatbook(
        name="origin race",
        description="origin race",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )

    assert success is False
    assert "origin_message_not_staged" in message
    assert not archive_path.exists()


def test_canvas_v3_exact_same_identity_restore_is_idempotent(tmp_path: Path) -> None:
    source_path = tmp_path / "same.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "same.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="same",
        description="same",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]

    importer = ChatbookImporter({"ChaChaNotes": str(source_path)})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()
    status = ImportStatus()
    success, message = importer.import_chatbook(archive_path, import_status=status)

    assert success is True, message
    assert status.skipped_items == 1
    database = CharactersRAGDB(source_path, client_id="same-verify")
    assert (
        database.get_connection()
        .execute("SELECT COUNT(*) FROM conversations")
        .fetchone()[0]
        == 1
    )
    assert (
        database.get_connection()
        .execute("SELECT COUNT(*) FROM canvas_revisions")
        .fetchone()[0]
        == 4
    )
    database.close_connection()


def test_canvas_same_identity_restore_ignores_manifest_revision_order(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "same-reordered.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "same-reordered-original.zip"
    reordered_path = tmp_path / "same-reordered.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="same reordered",
        description="same reordered",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]

    def reorder(name: str, payload: bytes) -> tuple[str, bytes]:
        if name != "manifest.json":
            return name, payload
        manifest = json.loads(payload)
        for document in manifest["canvas"]["documents"]:
            document["revisions"].reverse()
        return name, json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive_path, reordered_path, reorder)
    importer = ChatbookImporter({"ChaChaNotes": str(source_path)})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()
    status = ImportStatus()

    success, message = importer.import_chatbook(
        reordered_path,
        import_status=status,
    )

    assert success is True, message
    assert status.skipped_items == 1


def test_canvas_v3_conflicting_same_identity_refuses_before_mutation(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "conflict.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "conflict.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="conflict",
        description="conflict",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]
    database = CharactersRAGDB(source_path, client_id="conflict-mutate")
    database.get_connection().execute(
        "DELETE FROM canvas_conversation_hints WHERE conversation_id = ?",
        (expected["conversation_id"],),
    )
    database.close_connection()

    importer = ChatbookImporter({"ChaChaNotes": str(source_path)})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()
    status = ImportStatus()
    success, message = importer.import_chatbook(archive_path, import_status=status)

    assert success is False
    assert "same_identity_conflict" in message
    verify = CharactersRAGDB(source_path, client_id="conflict-verify")
    assert (
        verify.get_connection()
        .execute("SELECT COUNT(*) FROM conversations")
        .fetchone()[0]
        == 1
    )
    assert (
        verify.get_connection()
        .execute("SELECT COUNT(*) FROM canvas_conversation_hints")
        .fetchone()[0]
        == 0
    )
    verify.close_connection()


def test_canvas_same_revision_id_with_different_digest_is_rejected(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "digest-conflict.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "digest-original.zip"
    conflict_path = tmp_path / "digest-conflict.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="digest conflict",
        description="digest conflict",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]
    revision_id = str(expected["revision_ids"][0])
    replacement_source = b"<!doctype html><main>conflicting source</main>"

    def rewrite(name: str, payload: bytes) -> tuple[str, bytes]:
        if name == f"canvas/{expected['canvas_ids'][0]}/{revision_id}.html.txt":
            return name, replacement_source
        if name != "manifest.json":
            return name, payload
        manifest = json.loads(payload)
        revision = next(
            item
            for document in manifest["canvas"]["documents"]
            for item in document["revisions"]
            if item["revision_id"] == revision_id
        )
        original_bytes = int(revision["source_bytes"])
        revision["source_bytes"] = len(replacement_source)
        revision["content_sha256"] = sha256(replacement_source).hexdigest()
        manifest["canvas"]["total_source_bytes"] += (
            len(replacement_source) - original_bytes
        )
        return name, json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive_path, conflict_path, rewrite)
    importer = ChatbookImporter({"ChaChaNotes": str(source_path)})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(conflict_path)

    assert success is False
    assert "same_identity_conflict" in message
    verify = CharactersRAGDB(source_path, client_id="digest-conflict-verify")
    assert (
        CanvasRepository(verify)
        .read_revision(str(expected["conversation_id"]), revision_id)
        .source
        == expected["sources"][0]
    )
    verify.close_connection()


def test_canvas_same_identity_is_revalidated_under_write_lock(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "same-race.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "same-race.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="same race",
        description="same race",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]
    importer = ChatbookImporter({"ChaChaNotes": str(source_path)})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()
    original_preflight = importer._preflight_canvas_target_conflicts
    calls = 0

    def mutate_after_initial_preflight(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_preflight(*args, **kwargs)
        if calls == 1:
            with sqlite3.connect(source_path) as connection:
                connection.execute(
                    "DELETE FROM canvas_conversation_hints WHERE conversation_id = ?",
                    (expected["conversation_id"],),
                )
        return result

    monkeypatch.setattr(
        importer,
        "_preflight_canvas_target_conflicts",
        mutate_after_initial_preflight,
    )

    status = ImportStatus()
    success, _ = importer.import_chatbook(archive_path, import_status=status)

    assert success is False
    assert any("same_identity_conflict" in error for error in status.errors)
    assert calls == 2


def test_canvas_import_uses_one_open_archive_across_preflight_and_extraction(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "descriptor-source.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "descriptor.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="descriptor",
        description="descriptor",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]
    target_path = tmp_path / "descriptor-target.sqlite"
    importer = ChatbookImporter({"ChaChaNotes": str(target_path)})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()
    original_preflight = importer._preflight_archive
    replacement_path = tmp_path / "replacement.zip"

    def replace_path_after_preflight(source):
        result = original_preflight(source)
        with zipfile.ZipFile(replacement_path, "w") as replacement_archive:
            replacement_archive.writestr("manifest.json", b"{}")
        os.replace(replacement_path, archive_path)
        return result

    monkeypatch.setattr(importer, "_preflight_archive", replace_path_after_preflight)

    success, message = importer.import_chatbook(archive_path)

    assert success is True, message
    verify = CharactersRAGDB(target_path, client_id="descriptor-verify")
    assert (
        verify.get_connection()
        .execute("SELECT COUNT(*) FROM canvas_revisions")
        .fetchone()[0]
        == 4
    )
    verify.close_connection()


def test_archive_container_limit_is_checked_on_open_descriptor(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "container-source.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "container.zip"
    replacement_path = tmp_path / "container-replacement.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="container",
        description="container",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]
    original_size = archive_path.stat().st_size
    with zipfile.ZipFile(archive_path) as source_archive:
        entries = [
            (item, source_archive.read(item.filename))
            for item in source_archive.infolist()
        ]
    with zipfile.ZipFile(
        replacement_path, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for item, payload in entries:
            archive.writestr(item.filename, payload)
        archive.writestr("padding.bin", b"x" * (original_size + 1_024))
    monkeypatch.setattr(
        importer_module,
        "_MAX_ARCHIVE_CONTAINER_BYTES",
        original_size + 512,
    )
    original_open = Path.open
    swapped = False

    def swap_before_open(path: Path, *args, **kwargs):
        nonlocal swapped
        if path == archive_path and not swapped:
            swapped = True
            os.replace(replacement_path, archive_path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_before_open)
    importer = ChatbookImporter({"ChaChaNotes": str(tmp_path / "target.sqlite")})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(archive_path)

    assert success is False
    assert swapped is True
    assert "exceeds safety limits" in message
    assert list(importer.temp_dir.iterdir()) == []


def test_canvas_source_digest_is_validated_before_extraction_or_database(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "good.zip"
    corrupt_path = tmp_path / "corrupt.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="bad",
        description="bad",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]
    wanted = f"{expected['revision_ids'][0]}.html.txt"
    _rewrite_archive(
        archive_path,
        corrupt_path,
        lambda name, data: (
            name,
            data + b"tampered" if name.endswith(wanted) else data,
        ),
    )
    target_path = tmp_path / "target.sqlite"
    importer = ChatbookImporter({"ChaChaNotes": str(target_path)})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(corrupt_path)

    assert success is False
    assert "source_identity_mismatch" in message
    assert list(importer.temp_dir.iterdir()) == []
    assert not target_path.exists()


def test_case_ambiguous_canvas_path_is_rejected_before_extraction(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "good.zip"
    ambiguous_path = tmp_path / "ambiguous.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="paths",
        description="paths",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]
    with zipfile.ZipFile(archive_path) as archive:
        entries = [
            (item.filename, archive.read(item.filename)) for item in archive.infolist()
        ]
    canvas_name, canvas_payload = next(
        (name, data) for name, data in entries if name.startswith("canvas/")
    )
    with zipfile.ZipFile(ambiguous_path, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
        archive.writestr(
            "Canvas/" + canvas_name.removeprefix("canvas/"), canvas_payload
        )
    importer = ChatbookImporter({"ChaChaNotes": str(tmp_path / "target.sqlite")})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(ambiguous_path)

    assert success is False
    assert "duplicate archive member path" in message
    assert list(importer.temp_dir.iterdir()) == []


def test_special_zip_member_is_rejected_before_extraction(tmp_path: Path) -> None:
    archive_path = tmp_path / "special.zip"
    link = zipfile.ZipInfo("redirect")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", b"{}")
        archive.writestr(link, b"manifest.json")
    importer = ChatbookImporter({"ChaChaNotes": str(tmp_path / "target.sqlite")})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(archive_path)

    assert success is False
    assert "Unsupported archive member type" in message
    assert list(importer.temp_dir.iterdir()) == []


def test_zip_compressed_aggregate_limit_is_checked_before_extraction(
    tmp_path: Path, monkeypatch
) -> None:
    archive_path = tmp_path / "compressed-total.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", b"{}")
        archive.writestr("payload.bin", b"abcd")
    monkeypatch.setattr(importer_module, "_MAX_ARCHIVE_TOTAL_COMPRESSED_BYTES", 3)
    importer = ChatbookImporter({"ChaChaNotes": str(tmp_path / "target.sqlite")})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(archive_path)

    assert success is False
    assert "exceeds safety limits" in message
    assert list(importer.temp_dir.iterdir()) == []


def test_file_directory_prefix_collision_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "prefix-collision.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", b"{}")
        archive.writestr("content", b"file")
        archive.writestr("content/conversations/item.json", b"{}")
    importer = ChatbookImporter({"ChaChaNotes": str(tmp_path / "target.sqlite")})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(archive_path)

    assert success is False
    assert "archive member path" in message
    assert list(importer.temp_dir.iterdir()) == []


def test_duplicate_canvas_owner_content_item_is_rejected(tmp_path: Path) -> None:
    source_path = tmp_path / "duplicate-owner-source.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "duplicate-owner-original.zip"
    duplicate_path = tmp_path / "duplicate-owner.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="duplicate owner",
        description="duplicate owner",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]

    def duplicate(name: str, payload: bytes) -> tuple[str, bytes]:
        if name != "manifest.json":
            return name, payload
        manifest = json.loads(payload)
        manifest["content_items"].append(dict(manifest["content_items"][0]))
        return name, json.dumps(manifest).encode("utf-8")

    _rewrite_archive(archive_path, duplicate_path, duplicate)
    importer = ChatbookImporter({"ChaChaNotes": str(tmp_path / "target.sqlite")})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()

    manifest, error = importer.preview_chatbook(duplicate_path)

    assert manifest is None
    assert error is not None
    assert "duplicate_conversation_content_item" in error


def test_canvas_insert_failure_rolls_back_conversation_messages_and_graph(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "failure.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="failure",
        description="failure",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]
    target_path = tmp_path / "target.sqlite"
    database = CharactersRAGDB(target_path, client_id="failure-trigger")
    database.get_connection().execute(
        "CREATE TRIGGER fail_canvas_import BEFORE INSERT ON canvas_revisions "
        "BEGIN SELECT RAISE(ABORT, 'injected'); END"
    )
    database.close_connection()
    importer = ChatbookImporter({"ChaChaNotes": str(target_path)})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()

    success, _ = importer.import_chatbook(archive_path)

    assert success is False
    verify = CharactersRAGDB(target_path, client_id="failure-verify")
    for table in ("conversations", "messages", "canvas_documents", "canvas_revisions"):
        assert (
            verify.get_connection()
            .execute(f"SELECT COUNT(*) FROM {table}")
            .fetchone()[0]
            == 0
        )
    verify.close_connection()


def test_canvas_final_commit_failure_rolls_back_the_complete_graph(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "source.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "commit-failure.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="commit failure",
        description="commit failure",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]
    target_path = tmp_path / "target.sqlite"
    target = CharactersRAGDB(target_path, client_id="initialize-target")
    target.close_connection()
    original_transaction = CharactersRAGDB.transaction

    class _FailBeforeCommit:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            return self.inner.__enter__()

        def __exit__(self, exc_type, exc, traceback):
            injected = sqlite3.OperationalError("injected commit failure")
            self.inner.__exit__(type(injected), injected, traceback)
            raise injected

    def failing_transaction(self, *, immediate=False):
        inner = original_transaction(self, immediate=immediate)
        if immediate and self.client_id == "chatbook_importer":
            return _FailBeforeCommit(inner)
        return inner

    monkeypatch.setattr(CharactersRAGDB, "transaction", failing_transaction)
    importer = ChatbookImporter({"ChaChaNotes": str(target_path)})
    importer.temp_dir = tmp_path / "importer"
    importer.temp_dir.mkdir()

    success, _ = importer.import_chatbook(archive_path)

    assert success is False
    verify = CharactersRAGDB(target_path, client_id="commit-failure-verify")
    for table in ("conversations", "messages", "canvas_documents", "canvas_revisions"):
        assert (
            verify.get_connection()
            .execute(f"SELECT COUNT(*) FROM {table}")
            .fetchone()[0]
            == 0
        )
    verify.close_connection()
