"""Media, note and conversation keywords survive a chatbook export -> import.

TASK-34000.19 (review finding L-04): every keyword was gone from an exported
bundle -- the creator read keyword keys that the source rows never had
(``note.get("keywords")``, ``media_item.get("media_keywords")``) and the
conversation record carried none at all, so ``manifest.json`` said
``"tags": []`` for every item and the importer, which never linked a
keyword, had nothing to restore. Keywords are the Library user's triage and
provenance (j4, j7); a backup that drops them is not a backup.

The precedent is ADR-057: a Prompt record carries "the canonical active
keyword membership" and import "restores semantic membership through the
ordinary Prompt keyword write path". These tests pin the same shape for the
three types the review found empty: the keywords ride in the item's own
record AND in the manifest ``ContentItem.tags``, and come back through
``link_note_to_keyword`` / ``link_conversation_to_keyword`` /
``add_media_with_keywords``.

Real file-backed SQLite throughout (never ``:memory:`` -- the creator and
importer open their own connections from the paths), the real
``ChatbookCreator`` and ``ChatbookImporter``, no mocks: the "it works" claim
is the destination database's keyword rows.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from Tests.Chatbooks.factories import ContentItemFactory, ManifestFactory
from tldw_chatbook.Chatbooks import chatbook_creator as creator_module
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
from tldw_chatbook.DB.Client_Media_DB_v2 import (
    MediaDatabase,
    fetch_keywords_for_media_batch,
)
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

# ``bootstrap_profile``: the creator and importer resolve their private temp
# roots through ``get_user_data_dir()`` at construction, a config read that
# trips ``RecoveryRequired: raw_source_selection_changed`` inside the
# per-test sandbox (lessons-testing-evidence, "Tests/UI ``RecoveryRequired``
# at setup is a profile-selection trip"); the collection-time profile is
# what the sibling Chatbooks round-trip files keep too.
pytestmark = [pytest.mark.integration, pytest.mark.bootstrap_profile]


# --- helpers ------------------------------------------------------------------


def _db_trio(root: Path, client_id: str) -> dict:
    """Three real file-backed databases under ``root`` plus their path map."""
    root.mkdir(parents=True, exist_ok=True)
    db_paths = {
        "ChaChaNotes": str(root / "ChaChaNotes.db"),
        "Media": str(root / "Client_Media_DB.db"),
        "Prompts": str(root / "Prompts_DB.db"),
    }
    return {
        "db_paths": db_paths,
        "chachanotes_db": CharactersRAGDB(db_paths["ChaChaNotes"], client_id),
        "media_db": MediaDatabase(db_paths["Media"], client_id),
        "prompts_db": PromptsDatabase(db_paths["Prompts"], client_id),
    }


def _keyword_id(db: CharactersRAGDB, text: str) -> int:
    """Source-side get-or-create, the shape the harness seed uses."""
    row = db.get_keyword_by_text(text)
    return int(row["id"]) if row else int(db.add_keyword(text))


def _add_conversation(db: CharactersRAGDB, title: str, keywords: list[str]) -> str:
    conv_id = db.add_conversation({"title": title, "root_id": f"root-{title}"})
    db.add_message({"conversation_id": conv_id, "sender": "user", "content": "Hello"})
    db.add_message(
        {"conversation_id": conv_id, "sender": "assistant", "content": "Hi there"}
    )
    for keyword in keywords:
        db.link_conversation_to_keyword(conv_id, _keyword_id(db, keyword))
    return conv_id


def _add_note(db: CharactersRAGDB, title: str, keywords: list[str]) -> str:
    note_id = db.add_note(title=title, content=f"Body of {title}")
    for keyword in keywords:
        db.link_note_to_keyword(note_id, _keyword_id(db, keyword))
    return note_id


def _add_media(db: MediaDatabase, title: str, url: str, keywords: list[str]) -> int:
    media_id, _msg, _status = db.add_media_with_keywords(
        url=url,
        title=title,
        media_type="document",
        content=f"Stored text of {title}",
        keywords=keywords,
    )
    assert media_id is not None
    return int(media_id)


def _export(source: dict, selections: dict, out: Path) -> Path:
    creator = ChatbookCreator(source["db_paths"])
    success, message, _deps = creator.create_chatbook(
        name="Keyword round-trip",
        description="TASK-34000.19",
        content_selections=selections,
        output_path=out,
        include_media=ContentType.MEDIA in selections,
        media_quality="original",
        auto_include_dependencies=False,
    )
    assert success is True, message
    return out


def _import(
    dest: dict,
    bundle: Path,
    *,
    conflict_resolution: ConflictResolution = ConflictResolution.ASK,
) -> ImportStatus:
    status = ImportStatus()
    success, message = ChatbookImporter(dest["db_paths"]).import_chatbook(
        bundle, conflict_resolution=conflict_resolution, import_status=status
    )
    assert success is True, message
    assert status.errors == [], status.errors
    return status


def _manifest_items(bundle: Path, content_type: ContentType) -> list[dict]:
    with zipfile.ZipFile(bundle) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    return [
        item
        for item in manifest["content_items"]
        if item["type"] == content_type.value
    ]


def _read_json(bundle: Path, name: str) -> dict:
    with zipfile.ZipFile(bundle) as zf:
        return json.loads(zf.read(name))


def _read_text(bundle: Path, name: str) -> str:
    with zipfile.ZipFile(bundle) as zf:
        return zf.read(name).decode("utf-8")


def _note_keywords(db: CharactersRAGDB, title: str) -> list[str]:
    rows = db.search_notes(title)
    assert len(rows) == 1, rows
    return db.get_keywords_for_notes_batch([str(rows[0]["id"])]).get(
        str(rows[0]["id"]), []
    )


def _conversation_keywords(db: CharactersRAGDB, title: str) -> list[str]:
    rows = db.search_conversations_by_title(title)
    assert len(rows) == 1, rows
    grouped = db.get_keywords_for_conversations([str(rows[0]["id"])])
    return [row["keyword"] for row in grouped.get(str(rows[0]["id"]), [])]


def _media_keywords(db: MediaDatabase, url: str) -> list[str]:
    row = db.get_media_by_url(url)
    assert row is not None
    return fetch_keywords_for_media_batch(db, [int(row["id"])])[int(row["id"])]


def _keyword_rows(db: CharactersRAGDB, text: str) -> list[dict]:
    cursor = db.execute_query(
        "SELECT id, keyword, deleted FROM keywords WHERE keyword = ? COLLATE NOCASE",
        (text,),
    )
    return [dict(row) for row in cursor.fetchall()]


# --- AC#1, AC#2, AC#5: one round-trip per type ------------------------------


def test_media_keywords_survive_export_and_import(tmp_path):
    """Media: ``metadata.media_keywords`` is the LIST the importer already
    accepts (it used to be ``None``), the manifest item carries the same
    tags, and the destination's MediaKeywords rows match the source.

    Order is the batch reader's canonical read (``k.keyword COLLATE
    NOCASE``), the same rule ADR-057 adopted for Prompts, so it is asserted
    as a list, not a set."""
    source = _db_trio(tmp_path / "source", "kw-source")
    media_id = _add_media(
        source["media_db"],
        "Keyworded Media",
        "https://example.com/keyworded-media",
        ["reference", "markdown"],
    )
    bundle = _export(source, {ContentType.MEDIA: [str(media_id)]}, tmp_path / "m.zip")

    [item] = _manifest_items(bundle, ContentType.MEDIA)
    assert item["tags"] == ["markdown", "reference"]
    metadata = _read_json(bundle, f"content/media/metadata/media_{media_id}.json")
    assert metadata["metadata"]["media_keywords"] == ["markdown", "reference"]

    dest = _db_trio(tmp_path / "dest", "kw-dest")
    _import(dest, bundle)
    assert _media_keywords(dest["media_db"], "https://example.com/keyworded-media") == [
        "markdown",
        "reference",
    ]


def test_note_keywords_survive_export_and_import(tmp_path):
    """Notes: manifest tags, a ``tags:`` frontmatter line in the ``.md``,
    and ``note_keywords`` rows on the imported note."""
    source = _db_trio(tmp_path / "source", "kw-source")
    note_id = _add_note(source["chachanotes_db"], "Meeting Notes", ["meeting", "moc"])
    bundle = _export(source, {ContentType.NOTE: [note_id]}, tmp_path / "n.zip")

    [item] = _manifest_items(bundle, ContentType.NOTE)
    assert item["tags"] == ["meeting", "moc"]
    markdown = _read_text(bundle, item["file_path"])
    frontmatter = markdown.split("---", 2)[1]
    assert "tags: meeting, moc" in frontmatter.splitlines()

    dest = _db_trio(tmp_path / "dest", "kw-dest")
    _import(dest, bundle)
    assert _note_keywords(dest["chachanotes_db"], "Meeting Notes") == ["meeting", "moc"]


def test_conversation_keywords_survive_export_and_import(tmp_path):
    """Conversations: manifest tags, a ``keywords`` list in the conversation
    JSON, and ``conversation_keywords`` rows on the imported conversation."""
    source = _db_trio(tmp_path / "source", "kw-source")
    conv_id = _add_conversation(
        source["chachanotes_db"], "Release notes draft", ["launch"]
    )
    bundle = _export(source, {ContentType.CONVERSATION: [conv_id]}, tmp_path / "c.zip")

    [item] = _manifest_items(bundle, ContentType.CONVERSATION)
    assert item["tags"] == ["launch"]
    conversation = _read_json(bundle, item["file_path"])
    assert conversation["keywords"] == ["launch"]

    dest = _db_trio(tmp_path / "dest", "kw-dest")
    _import(dest, bundle)
    assert _conversation_keywords(dest["chachanotes_db"], "Release notes draft") == [
        "launch"
    ]


# --- AC#4: nothing to carry, and bundles from before this change ------------


def test_items_without_keywords_export_empty_lists_and_import_cleanly(tmp_path):
    """One of each type with no keywords: ``[]`` in the manifest AND in the
    item's own record (the conversation JSON key is present and empty, the
    media metadata is ``[]`` not ``null``), the import succeeds, and the
    destination has no keyword rows for them."""
    source = _db_trio(tmp_path / "source", "kw-source")
    media_id = _add_media(
        source["media_db"], "Plain Media", "https://example.com/plain-media", []
    )
    note_id = _add_note(source["chachanotes_db"], "Plain Note", [])
    conv_id = _add_conversation(source["chachanotes_db"], "Plain Conversation", [])
    bundle = _export(
        source,
        {
            ContentType.MEDIA: [str(media_id)],
            ContentType.NOTE: [note_id],
            ContentType.CONVERSATION: [conv_id],
        },
        tmp_path / "plain.zip",
    )

    for content_type in (ContentType.MEDIA, ContentType.NOTE, ContentType.CONVERSATION):
        [item] = _manifest_items(bundle, content_type)
        assert item["tags"] == [], content_type
    metadata = _read_json(bundle, f"content/media/metadata/media_{media_id}.json")
    assert metadata["metadata"]["media_keywords"] == []
    [conv_item] = _manifest_items(bundle, ContentType.CONVERSATION)
    conversation = _read_json(bundle, conv_item["file_path"])
    assert "keywords" in conversation
    assert conversation["keywords"] == []
    [note_item] = _manifest_items(bundle, ContentType.NOTE)
    frontmatter = _read_text(bundle, note_item["file_path"]).split("---", 2)[1]
    assert not any(line.startswith("tags:") for line in frontmatter.splitlines())

    dest = _db_trio(tmp_path / "dest", "kw-dest")
    status = _import(dest, bundle)
    assert status.successful_items == 3
    assert _media_keywords(dest["media_db"], "https://example.com/plain-media") == []
    assert _note_keywords(dest["chachanotes_db"], "Plain Note") == []
    assert _conversation_keywords(dest["chachanotes_db"], "Plain Conversation") == []
    assert (
        dest["chachanotes_db"].execute_query("SELECT COUNT(*) FROM keywords").fetchone()[0]
        == 0
    )


def _write_legacy_bundle(path: Path) -> None:
    """A bundle as the creator wrote it BEFORE this change: manifest items
    with no ``tags`` key at all, media metadata ``"media_keywords": null``, a
    conversation JSON without ``keywords`` -- plus one manifest item whose
    ``tags`` is an explicit ``null`` (Review Focus 5)."""
    stamp = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    items = [
        ContentItemFactory.create(
            ContentType.NOTE,
            item_id="legacy-note",
            title="Legacy Note",
            tags=[],
            created_at=stamp,
            updated_at=stamp,
            file_path="content/notes/legacy-note.md",
        ),
        ContentItemFactory.create(
            ContentType.MEDIA,
            item_id="7",
            title="Legacy Media",
            tags=[],
            created_at=stamp,
            updated_at=stamp,
            file_path="content/media/metadata/media_7.json",
        ),
        ContentItemFactory.create(
            ContentType.CONVERSATION,
            item_id="legacy-c1",
            title="Legacy Conversation",
            tags=[],
            created_at=stamp,
            updated_at=stamp,
            file_path="content/conversations/conversation_legacy-c1.json",
        ),
    ]
    manifest = ManifestFactory.create(
        name="Legacy bundle", content_items=items, version=ChatbookVersion.V1
    ).to_dict()
    for item in manifest["content_items"]:
        if item["type"] == ContentType.NOTE.value:
            item["tags"] = None  # the hand-edited / foreign-writer shape
        else:
            del item["tags"]  # the key simply absent
    iso = stamp.isoformat()
    note_md = (
        "---\n"
        "id: legacy-note\n"
        "title: Legacy Note\n"
        f"created_at: {iso}\n"
        f"updated_at: {iso}\n"
        "---\n\n"
        "Legacy note body\n"
    )
    media_json = {
        "id": 7,
        "title": "Legacy Media",
        "media_type": "document",
        "url": "https://example.com/legacy-media",
        "author": "Legacy",
        "content": "Legacy media text",
        "created_at": iso,
        "updated_at": iso,
        "metadata": {"ingestion_date": iso, "media_keywords": None},
    }
    conversation_json = {
        "id": "legacy-c1",
        "name": "Legacy Conversation",
        "created_at": iso,
        "updated_at": iso,
        "messages": [{"role": "user", "content": "Hello", "timestamp": iso}],
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("content/notes/legacy-note.md", note_md)
        zf.writestr("content/media/metadata/media_7.json", json.dumps(media_json))
        zf.writestr("content/media/media_7.txt", "Legacy media text")
        zf.writestr(
            "content/conversations/conversation_legacy-c1.json",
            json.dumps(conversation_json),
        )


def test_a_bundle_written_before_this_change_still_imports(tmp_path):
    """The format is additive: a pre-change bundle imports with no error and
    no keyword rows, and a manifest item whose ``tags`` is ``null`` reads as
    ``[]`` (today ``ContentItem.from_dict`` hands back ``None``)."""
    bundle = tmp_path / "legacy.zip"
    _write_legacy_bundle(bundle)

    with zipfile.ZipFile(bundle) as zf:
        manifest = ChatbookManifest.from_dict(json.loads(zf.read("manifest.json")))
    for item in manifest.content_items:
        assert item.tags == [], (item.type, item.tags)

    dest = _db_trio(tmp_path / "dest", "kw-dest")
    status = _import(dest, bundle)
    assert status.successful_items == 3
    assert _media_keywords(dest["media_db"], "https://example.com/legacy-media") == []
    assert _note_keywords(dest["chachanotes_db"], "Legacy Note") == []
    assert _conversation_keywords(dest["chachanotes_db"], "Legacy Conversation") == []
    assert (
        dest["chachanotes_db"].execute_query("SELECT COUNT(*) FROM keywords").fetchone()[0]
        == 0
    )


# --- Review Focus 4: the destination already has the keyword ----------------


def test_a_keyword_that_already_exists_in_the_destination_is_linked_not_duplicated(
    tmp_path,
):
    """An active ``meeting`` in the destination (any case) is linked, never
    duplicated and never a ``ConflictError`` out of ``add_keyword``; a
    soft-deleted ``archive`` is undeleted (``add_keyword``'s documented
    behaviour) and linked. The note's content is restored either way."""
    source = _db_trio(tmp_path / "source", "kw-source")
    note_id = _add_note(
        source["chachanotes_db"], "Standup", ["Meeting", "archive"]
    )
    bundle = _export(source, {ContentType.NOTE: [note_id]}, tmp_path / "n.zip")

    dest = _db_trio(tmp_path / "dest", "kw-dest")
    dest_db = dest["chachanotes_db"]
    existing_meeting_id = int(dest_db.add_keyword("meeting"))
    archive_id = int(dest_db.add_keyword("archive"))
    archive_row = dest_db.get_keyword_by_id(archive_id)
    assert dest_db.soft_delete_keyword(archive_id, int(archive_row["version"])) is True
    assert dest_db.get_keyword_by_text("archive") is None

    _import(dest, bundle)

    meeting_rows = _keyword_rows(dest_db, "meeting")
    assert len(meeting_rows) == 1, meeting_rows
    assert meeting_rows[0]["id"] == existing_meeting_id
    assert meeting_rows[0]["deleted"] == 0
    archive_rows = _keyword_rows(dest_db, "archive")
    assert len(archive_rows) == 1, archive_rows
    assert archive_rows[0]["id"] == archive_id
    assert archive_rows[0]["deleted"] == 0
    assert _note_keywords(dest_db, "Standup") == ["archive", "meeting"]
    [note] = dest_db.search_notes("Standup")
    assert note["content"] == "Body of Standup"


# --- Fix round 1 ---------------------------------------------------------------


def test_a_large_selection_reads_keywords_in_slices_and_still_exports(
    tmp_path, monkeypatch
):
    """I-2: each batch reader binds one SQL variable per id, so a selection
    larger than SQLite's variable limit used to raise ``too many SQL
    variables`` outside the per-item try and fail the whole export. The
    reads now go in slices (``_KEYWORD_READ_BATCH_CAP``, the repo's
    ``_ID_LIST_BATCH_CAP`` shape). The pin lowers the live variable limit on
    every connection the creator opens to 100 and the cap to 40, then
    exports 120 notes and 120 media (unsliced, both reads overflow) and 60
    conversations -- two slices through the same helper, but kept under
    the limit because ``Canvas.archive.export_canvas_archive`` binds one
    variable per conversation id unsliced (pre-existing, outside this
    task; parked as a follow-up)."""
    source = _db_trio(tmp_path / "source", "kw-source")
    note_ids = [
        _add_note(source["chachanotes_db"], f"Note {i:03d}", [f"n{i % 7}"])
        for i in range(120)
    ]
    conv_ids = [
        _add_conversation(source["chachanotes_db"], f"Conv {i:03d}", [f"c{i % 5}"])
        for i in range(60)
    ]
    media_ids = [
        _add_media(
            source["media_db"], f"Media {i:03d}", f"https://example.com/m/{i}", [f"m{i % 3}"]
        )
        for i in range(120)
    ]

    def _limited(original):
        def get_connection(self):
            connection = original(self)
            connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 100)
            return connection

        return get_connection

    monkeypatch.setattr(
        CharactersRAGDB, "get_connection", _limited(CharactersRAGDB.get_connection)
    )
    monkeypatch.setattr(
        MediaDatabase, "get_connection", _limited(MediaDatabase.get_connection)
    )
    # ``raising=False``: on the unsliced base the cap does not exist, and the
    # export must then fail on the variable limit, not on this patch.
    monkeypatch.setattr(creator_module, "_KEYWORD_READ_BATCH_CAP", 40, raising=False)

    bundle = _export(
        source,
        {
            ContentType.NOTE: note_ids,
            ContentType.CONVERSATION: conv_ids,
            ContentType.MEDIA: [str(media_id) for media_id in media_ids],
        },
        tmp_path / "big.zip",
    )
    for content_type, expected in (
        (ContentType.NOTE, 120),
        (ContentType.CONVERSATION, 60),
        (ContentType.MEDIA, 120),
    ):
        items = _manifest_items(bundle, content_type)
        assert len(items) == expected, content_type
        assert all(len(item["tags"]) == 1 for item in items), content_type


def test_keywords_that_differ_only_beyond_ascii_case_all_survive(tmp_path):
    """M-1: the keyword table's ``COLLATE NOCASE`` folds ASCII only, so
    ``Café``/``CAFÉ`` and ``Straße``/``STRASSE`` are four source rows; a
    casefold de-duplication on import collapsed them to two."""
    source = _db_trio(tmp_path / "source", "kw-source")
    note_id = _add_note(
        source["chachanotes_db"], "Unicode", ["Café", "CAFÉ", "Straße", "STRASSE"]
    )
    bundle = _export(source, {ContentType.NOTE: [note_id]}, tmp_path / "u.zip")
    [item] = _manifest_items(bundle, ContentType.NOTE)
    assert sorted(item["tags"]) == ["CAFÉ", "Café", "STRASSE", "Straße"]

    dest = _db_trio(tmp_path / "dest", "kw-dest")
    _import(dest, bundle)
    assert sorted(_note_keywords(dest["chachanotes_db"], "Unicode")) == [
        "CAFÉ",
        "Café",
        "STRASSE",
        "Straße",
    ]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("alpha, beta,,gamma ", ["alpha", "beta", "gamma"]),
        (["a", "b"], ["a", "b"]),
        (("a",), ["a"]),
        (None, []),
        (42, []),
        ({"x": 1}, []),
    ],
)
def test_a_manifest_tags_value_that_is_not_a_list_is_normalized(raw, expected):
    """M-2: ``list("alpha,beta")`` split a string into characters; a string
    splits on commas (as ``media_keywords`` does) and anything that is not
    a list, tuple or string reads as no tags."""
    item = ContentItem.from_dict(
        {"id": "x", "type": "note", "title": "T", "tags": raw}
    )
    assert item.tags == expected


def test_a_keyword_that_cannot_be_linked_leaves_the_item_restored_and_warns(
    tmp_path, monkeypatch
):
    """M-3 / M-5: a refused link never fails the item; the import reports
    one fixed-text warning (no keyword text) so the user learns something
    was lost."""
    source = _db_trio(tmp_path / "source", "kw-source")
    note_id = _add_note(source["chachanotes_db"], "Fragile", ["meeting"])
    bundle = _export(source, {ContentType.NOTE: [note_id]}, tmp_path / "f.zip")

    def refuse(self, note_id, keyword_id, *, cursor=None):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(CharactersRAGDB, "link_note_to_keyword", refuse)
    # Imported here, not at module scope: before the fix the constant does
    # not exist, and that is this test's RED, not the whole file's.
    from tldw_chatbook.Chatbooks.chatbook_importer import KEYWORD_RESTORE_WARNING

    dest = _db_trio(tmp_path / "dest", "kw-dest")
    status = _import(dest, bundle)
    assert status.successful_items == 1
    [note] = dest["chachanotes_db"].search_notes("Fragile")
    assert note["content"] == "Body of Fragile"
    assert _note_keywords(dest["chachanotes_db"], "Fragile") == []
    assert status.warnings.count(KEYWORD_RESTORE_WARNING) == 1
    assert "meeting" not in " ".join(status.warnings)


def test_importing_the_same_bundle_twice_under_skip_adds_no_rows(tmp_path):
    """M-5: a second import of the same bundle with SKIP leaves the keyword
    table and every junction table exactly as the first import left them."""
    source = _db_trio(tmp_path / "source", "kw-source")
    note_id = _add_note(source["chachanotes_db"], "Twice", ["meeting", "moc"])
    conv_id = _add_conversation(source["chachanotes_db"], "Twice conv", ["launch"])
    media_id = _add_media(
        source["media_db"], "Twice media", "https://example.com/twice", ["ref"]
    )
    bundle = _export(
        source,
        {
            ContentType.NOTE: [note_id],
            ContentType.CONVERSATION: [conv_id],
            ContentType.MEDIA: [str(media_id)],
        },
        tmp_path / "twice.zip",
    )
    dest = _db_trio(tmp_path / "dest", "kw-dest")

    def snapshot() -> dict:
        cc = dest["chachanotes_db"]
        md = dest["media_db"]
        return {
            "keywords": cc.execute_query("SELECT COUNT(*) FROM keywords").fetchone()[0],
            "note_keywords": cc.execute_query(
                "SELECT COUNT(*) FROM note_keywords"
            ).fetchone()[0],
            "conversation_keywords": cc.execute_query(
                "SELECT COUNT(*) FROM conversation_keywords"
            ).fetchone()[0],
            "media_keywords": md.execute_query(
                "SELECT COUNT(*) FROM MediaKeywords"
            ).fetchone()[0],
            "Keywords": md.execute_query("SELECT COUNT(*) FROM Keywords").fetchone()[0],
        }

    _import(dest, bundle, conflict_resolution=ConflictResolution.SKIP)
    first = snapshot()
    assert first == {
        "keywords": 3,
        "note_keywords": 2,
        "conversation_keywords": 1,
        "media_keywords": 1,
        "Keywords": 1,
    }
    second_status = _import(dest, bundle, conflict_resolution=ConflictResolution.SKIP)
    assert second_status.skipped_items == 3
    assert snapshot() == first


def test_a_keyword_that_would_break_the_frontmatter_stays_out_of_that_line(tmp_path):
    """M-6: the frontmatter ``tags:`` line is plain text inside a ``---``
    fence, so a keyword holding a newline or ``---`` could corrupt the
    importer's body split. Such keywords stay in the manifest (the record)
    and are left out of the frontmatter line only; the body and every
    keyword still round-trip."""
    source = _db_trio(tmp_path / "source", "kw-source")
    unsafe_fence = "cut---here"
    unsafe_newline = "two\nlines"
    note_id = _add_note(
        source["chachanotes_db"], "Edge", ["safe", unsafe_fence, unsafe_newline]
    )
    bundle = _export(source, {ContentType.NOTE: [note_id]}, tmp_path / "e.zip")

    [item] = _manifest_items(bundle, ContentType.NOTE)
    assert sorted(item["tags"]) == sorted(["safe", unsafe_fence, unsafe_newline])
    markdown = _read_text(bundle, item["file_path"])
    frontmatter = markdown.split("---", 2)[1]
    tag_lines = [line for line in frontmatter.splitlines() if line.startswith("tags:")]
    assert tag_lines == ["tags: safe"]

    dest = _db_trio(tmp_path / "dest", "kw-dest")
    _import(dest, bundle)
    [note] = dest["chachanotes_db"].search_notes("Edge")
    assert note["content"] == "Body of Edge"
    assert sorted(_note_keywords(dest["chachanotes_db"], "Edge")) == sorted(
        ["safe", unsafe_fence, unsafe_newline]
    )
