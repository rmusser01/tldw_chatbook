"""Portable archives carry exact note source and Canvas together."""

import json
import zipfile

import pytest

from Tests.Chatbooks.test_chatbook_canvas_round_trip import _seed_canvas_graph
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ContentItem,
    ContentType,
)
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.parametrize("language", [None, "json", "yaml", "jsonl", " future/😀 "])
@pytest.mark.parametrize("with_canvas", [False, True])
def test_portable_notes_preserve_source_language_and_canvas(
    tmp_path, language, with_canvas
):
    path = tmp_path / "source.db"
    graph = _seed_canvas_graph(path) if with_canvas else None
    db = CharactersRAGDB(path, client_id="portable-source")
    source = '\ufeff  {"x":9007199254740993,"n":1e400,"html":"<>&"}\r\n  '
    note_id = db.add_note(title="  Data  ", content=source, content_language=language)
    db.close_connection()
    selections = {ContentType.NOTE: [note_id]}
    if graph:
        selections[ContentType.CONVERSATION] = [graph["conversation_id"]]
    archive_path = tmp_path / "notes.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(path)})
    creator.temp_dir = tmp_path / "export"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="Notes",
        description="Exact",
        content_selections=selections,
        output_path=archive_path,
    )[0]
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["version"] == "4.0"
        assert bool(manifest.get("canvas")) is with_canvas
        item = next(
            item for item in manifest["content_items"] if item["type"] == "note"
        )
        record = json.loads(archive.read(item["file_path"]))
        assert record["content"] == source
        assert record["content_language"] == language
    target = tmp_path / "target.db"
    importer = ChatbookImporter({"ChaChaNotes": str(target)})
    importer.temp_dir = tmp_path / "import"
    importer.temp_dir.mkdir()
    status = ImportStatus()
    success, message = importer.import_chatbook(archive_path, import_status=status)
    assert success, message
    assert status.failed_items == 0
    restored = CharactersRAGDB(target, client_id="portable-verify")
    try:
        notes = (
            restored.get_connection()
            .execute("SELECT title,content,content_language FROM notes WHERE deleted=0")
            .fetchall()
        )
        assert [
            (row["title"], row["content"], row["content_language"]) for row in notes
        ] == [("  Data  ", source, language)]
        assert restored.get_connection().execute(
            "SELECT COUNT(*) FROM canvas_revisions"
        ).fetchone()[0] == (4 if with_canvas else 0)
    finally:
        restored.close_connection()


def test_unknown_archive_version_refuses_before_storage(tmp_path):
    archive_path = tmp_path / "future.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {"version": "99.0", "name": "Future", "description": "Unsupported"}
            ),
        )
    target_path = tmp_path / "target.db"
    importer = ChatbookImporter({"ChaChaNotes": str(target_path)})
    importer.temp_dir = tmp_path / "import"
    importer.temp_dir.mkdir()
    success, message = importer.import_chatbook(archive_path)
    assert not success
    assert "version" in message.lower()
    assert not target_path.exists()
    assert list(importer.temp_dir.iterdir()) == []


@pytest.mark.parametrize(
    "fields", [{"keywords": [3]}, {"content_language": ""}, {"content": {"x": 1}}]
)
def test_invalid_archive_carrier_refuses_before_storage(tmp_path, fields):
    archive_path = tmp_path / "invalid.zip"
    record = {
        "format_version": "tldw.note.v1",
        "id": "note-1",
        "title": "Data",
        "content": "raw",
        "content_language": None,
        **fields,
    }
    manifest = ChatbookManifest(
        ChatbookVersion.V4,
        "Invalid",
        "Invalid",
        content_items=[
            ContentItem(
                "note-1", ContentType.NOTE, "Data", file_path="notes/note-1.json"
            )
        ],
    ).to_dict()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("notes/note-1.json", json.dumps(record))
    target = tmp_path / "target.db"
    importer = ChatbookImporter({"ChaChaNotes": str(target)})
    importer.temp_dir = tmp_path / "import"
    importer.temp_dir.mkdir()
    success, _ = importer.import_chatbook(archive_path)
    assert not success
    assert not target.exists()


@pytest.mark.parametrize("version", [ChatbookVersion.V1, ChatbookVersion.V2])
def test_old_markdown_note_archives_remain_readable(tmp_path, version):
    manifest = ChatbookManifest(
        version,
        "Legacy",
        "Legacy",
        content_items=[
            ContentItem("old", ContentType.NOTE, "Old", file_path="notes/old.md")
        ],
    )
    archive_path = tmp_path / "old.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest.to_dict()))
        archive.writestr("notes/old.md", "# Old\n\nbody\n")
    target = tmp_path / "target.db"
    importer = ChatbookImporter({"ChaChaNotes": str(target)})
    importer.temp_dir = tmp_path / "import"
    importer.temp_dir.mkdir()
    success, message = importer.import_chatbook(archive_path)
    assert success, message
    db = CharactersRAGDB(target, client_id="verify")
    try:
        note = db.list_notes()[0]
        assert (note["title"], note["content"], note["content_language"]) == (
            "Old",
            "# Old\n\nbody\n",
            None,
        )
    finally:
        db.close_connection()


def test_same_title_notes_use_unique_carriers_and_keep_empty_source_and_keywords(
    tmp_path,
):
    path = tmp_path / "source.db"
    db = CharactersRAGDB(path, client_id="source")
    first = db.add_note(title="Same", content="", content_language=None)
    second = db.add_note(title="Same", content="  raw\r\n", content_language="yaml")
    db.link_note_to_keyword(second, db.add_keyword("Tagged"))
    db.close_connection()
    archive_path = tmp_path / "notes.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(path)})
    creator.temp_dir = tmp_path / "export"
    creator.temp_dir.mkdir()
    success, message, _ = creator.create_chatbook(
        name="Notes",
        description="Notes",
        content_selections={ContentType.NOTE: [first, second]},
        output_path=archive_path,
    )
    assert success, message
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        carriers = [item["file_path"] for item in manifest["content_items"]]
        assert len(carriers) == len(set(carriers)) == 2
    target = tmp_path / "target.db"
    importer = ChatbookImporter({"ChaChaNotes": str(target)})
    importer.temp_dir = tmp_path / "import"
    importer.temp_dir.mkdir()
    success, message = importer.import_chatbook(
        archive_path, conflict_resolution=ConflictResolution.RENAME
    )
    assert success, message
    db = CharactersRAGDB(target, client_id="verify")
    try:
        notes = db.list_notes()
        assert {(n["content"], n["content_language"]) for n in notes} == {
            ("", None),
            ("  raw\r\n", "yaml"),
        }
        restored = next(n for n in notes if n["content_language"] == "yaml")
        assert [k["keyword"] for k in db.get_keywords_for_note(restored["id"])] == [
            "Tagged"
        ]
    finally:
        db.close_connection()
