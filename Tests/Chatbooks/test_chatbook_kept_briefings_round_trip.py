"""Chatbook export/import coverage for kept briefings/scripts (task-1870).

Kept briefings and their kept scripts (`kept_briefings`/`kept_scripts` in
ChaChaNotes, task-1780) previously had no chatbook-export awareness at all --
silently absent from every exported chatbook. This module covers the new
`ContentType.KEPT_BRIEFING` content type end to end: export (JSON + Markdown
per briefing, scripts nested inside), import into a fresh ChaChaNotes
(byte-faithful round trip of every provenance column), re-import idempotency,
cross-device source-id collisions (never silently overwritten, honestly
reported), and backward compatibility with chatbooks that predate this
content type.

Kept scripts are not independently selectable -- they always ride with their
parent kept briefing, mirroring how a conversation's messages are nested
inside the conversation's own exported JSON rather than being their own
content type.
"""

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ContentType
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _db_paths(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "ChaChaNotes": str(root / "chachanotes.db"),
        "Prompts": str(root / "prompts.db"),
        "Media": str(root / "media.db"),
        "Evals": str(root / "evals.db"),
        "RAG": str(root / "rag.db"),
    }


@pytest.fixture
def source_env(tmp_path, chachanotes_template_db):
    """A source ChaChaNotes DB with one kept briefing and two kept scripts.

    One script carries a subscriptions-side `source_script_id`; the other is
    `source_script_id=None` (cast directly from the kept briefing, per the
    kept-briefings design doc) -- covering both idempotency paths import
    must handle.
    """
    import shutil

    db_paths = _db_paths(tmp_path / "source")
    shutil.copyfile(chachanotes_template_db, db_paths["ChaChaNotes"])
    db = CharactersRAGDB(db_paths["ChaChaNotes"], "test-source")

    kept_id = db.create_kept_briefing(
        source_briefing_id=101,
        watchlist_name="Tech Watch",
        body_markdown="# Digest\n\nBody text with **markdown**.",
        covers_through_item_id=555,
        covers_from_ts="2026-07-25T00:00:00Z",
        selection_mode="auto",
        model_used="gpt-test",
        item_count=12,
        featured_count=3,
        overflow_count=2,
        origin="manual",
        original_created_at="2026-07-30T10:00:00Z",
    )
    db.create_kept_script(
        kept_id,
        source_script_id=555,
        preset_name="duo-cast",
        roster_snapshot_json='{"roster": ["A", "B"]}',
        turns_json='[{"speaker": "A", "text": "Hi"}]',
        model_used="gpt-test",
        original_created_at="2026-07-30T11:00:00Z",
    )
    db.create_kept_script(
        kept_id,
        source_script_id=None,
        preset_name="(app default)",
        roster_snapshot_json='{"roster": ["Narrator"]}',
        turns_json='[{"speaker": "Narrator", "text": "Once upon a time"}]',
        model_used=None,
        original_created_at=None,
    )
    db.close_connection()
    return db_paths, kept_id


def _create_chatbook(db_paths, kept_id, tmp_path, name="kept-round-trip.zip") -> Path:
    output = tmp_path / name
    creator = ChatbookCreator(db_paths)
    ok, message, _details = creator.create_chatbook(
        name="Kept Briefing Round Trip",
        description="round trip",
        content_selections={ContentType.KEPT_BRIEFING: [str(kept_id)]},
        output_path=output,
    )
    assert ok, message
    return output


def _dest_db(dest_paths) -> CharactersRAGDB:
    return CharactersRAGDB(dest_paths["ChaChaNotes"], "test-dest")


# --- Export -----------------------------------------------------------


def test_export_writes_kept_briefing_json_and_markdown_and_manifest_entry(
    source_env, tmp_path
):
    db_paths, kept_id = source_env
    output = _create_chatbook(db_paths, kept_id, tmp_path)

    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
        json_name = f"content/kept_briefings/kept_briefing_{kept_id}.json"
        md_name = f"content/kept_briefings/kept_briefing_{kept_id}.md"
        assert json_name in names
        assert md_name in names

        payload = json.loads(zf.read(json_name))
        assert payload["source_briefing_id"] == 101
        assert payload["watchlist_name"] == "Tech Watch"
        assert payload["body_markdown"] == "# Digest\n\nBody text with **markdown**."
        assert payload["covers_through_item_id"] == 555
        assert payload["covers_from_ts"] == "2026-07-25T00:00:00+00:00"
        assert payload["selection_mode"] == "auto"
        assert payload["model_used"] == "gpt-test"
        assert payload["item_count"] == 12
        assert payload["featured_count"] == 3
        assert payload["overflow_count"] == 2
        assert payload["origin"] == "manual"
        assert payload["original_created_at"] == "2026-07-30T10:00:00+00:00"
        assert payload["kept_at"]

        scripts = payload["scripts"]
        assert len(scripts) == 2
        by_source = {s["source_script_id"]: s for s in scripts}
        assert by_source[555]["preset_name"] == "duo-cast"
        assert by_source[555]["roster_snapshot_json"] == '{"roster": ["A", "B"]}'
        assert by_source[555]["turns_json"] == '[{"speaker": "A", "text": "Hi"}]'
        assert by_source[None]["preset_name"] == "(app default)"

        # Human-readable rendition present and non-empty; not used on import.
        md_text = zf.read(md_name).decode("utf-8")
        assert "Tech Watch" in md_text
        assert "Digest" in md_text

        manifest_data = json.loads(zf.read("manifest.json"))
        manifest = ChatbookManifest.from_dict(manifest_data)
        assert manifest.total_kept_briefings == 1
        kept_items = [
            item
            for item in manifest.content_items
            if item.type == ContentType.KEPT_BRIEFING
        ]
        assert len(kept_items) == 1
        assert kept_items[0].id == str(kept_id)
        assert kept_items[0].file_path == json_name


def test_export_with_no_kept_briefings_selected_produces_no_kept_section(
    source_env, tmp_path
):
    """Selecting nothing of this type must not create the kept section at
    all, and must not crash (empty-kept-tables export)."""
    db_paths, _kept_id = source_env
    output = tmp_path / "no-kept.zip"
    creator = ChatbookCreator(db_paths)
    ok, message, _details = creator.create_chatbook(
        name="No Kept Selected",
        description="no kept briefings selected",
        content_selections={},
        output_path=output,
    )
    assert ok, message

    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
        assert not any(name.startswith("content/kept_briefings/") for name in names)
        manifest_data = json.loads(zf.read("manifest.json"))
        manifest = ChatbookManifest.from_dict(manifest_data)
        assert manifest.total_kept_briefings == 0
        assert not any(
            item.type == ContentType.KEPT_BRIEFING for item in manifest.content_items
        )


# --- Import: round trip -------------------------------------------------


def test_import_restores_kept_briefing_and_scripts_byte_faithful(source_env, tmp_path):
    db_paths, kept_id = source_env
    output = _create_chatbook(db_paths, kept_id, tmp_path)

    dest_paths = _db_paths(tmp_path / "dest")
    importer = ChatbookImporter(dest_paths)
    ok, message = importer.import_chatbook(
        output,
        content_selections={ContentType.KEPT_BRIEFING: [str(kept_id)]},
    )
    assert ok, message

    dest_db = _dest_db(dest_paths)
    restored = dest_db.get_kept_briefing_by_source(101)
    assert restored is not None
    assert restored["watchlist_name"] == "Tech Watch"
    assert restored["body_markdown"] == "# Digest\n\nBody text with **markdown**."
    assert restored["covers_through_item_id"] == 555
    assert restored["covers_from_ts"] == datetime(
        2026, 7, 25, 0, 0, tzinfo=timezone.utc
    )
    assert restored["selection_mode"] == "auto"
    assert restored["model_used"] == "gpt-test"
    assert restored["item_count"] == 12
    assert restored["featured_count"] == 3
    assert restored["overflow_count"] == 2
    assert restored["origin"] == "manual"
    assert restored["original_created_at"] == datetime(
        2026, 7, 30, 10, 0, tzinfo=timezone.utc
    )

    scripts = dest_db.list_kept_scripts(restored["id"])
    assert len(scripts) == 2
    by_source = {s["source_script_id"]: s for s in scripts}
    assert by_source[555]["preset_name"] == "duo-cast"
    assert by_source[555]["roster_snapshot_json"] == '{"roster": ["A", "B"]}'
    assert by_source[555]["turns_json"] == '[{"speaker": "A", "text": "Hi"}]'
    assert by_source[555]["model_used"] == "gpt-test"
    assert by_source[None]["preset_name"] == "(app default)"
    assert by_source[None]["roster_snapshot_json"] == '{"roster": ["Narrator"]}'


def test_reimport_is_idempotent_no_duplicates(source_env, tmp_path):
    """Importing the same chatbook twice into the same destination must add
    nothing on the second pass -- neither the briefing nor either script
    (including the NULL-source script, which has no unique-constraint
    backstop of its own)."""
    db_paths, kept_id = source_env
    output = _create_chatbook(db_paths, kept_id, tmp_path)

    dest_paths = _db_paths(tmp_path / "dest")
    importer = ChatbookImporter(dest_paths)
    selections = {ContentType.KEPT_BRIEFING: [str(kept_id)]}

    ok1, message1 = importer.import_chatbook(output, content_selections=selections)
    assert ok1, message1

    ok2, message2 = importer.import_chatbook(output, content_selections=selections)
    assert ok2, message2

    dest_db = _dest_db(dest_paths)
    all_kept = dest_db.list_kept_briefings()
    assert len(all_kept) == 1
    scripts = dest_db.list_kept_scripts(all_kept[0]["id"])
    assert len(scripts) == 2


# --- Import: cross-device collision ------------------------------------


def test_import_reports_conflict_and_never_overwrites_differing_local_row(
    source_env, tmp_path
):
    """A local kept briefing already exists for the same (device-local)
    `source_briefing_id` but with genuinely different content -- the
    existing row must survive unmodified and the conflict must be named in
    the import summary, never silently absorbed as an ordinary skip."""
    db_paths, kept_id = source_env
    output = _create_chatbook(db_paths, kept_id, tmp_path)

    dest_paths = _db_paths(tmp_path / "dest")
    dest_db = _dest_db(dest_paths)
    local_kept_id = dest_db.create_kept_briefing(
        source_briefing_id=101,  # same source id as the incoming briefing
        watchlist_name="Totally Different Local Watchlist",
        body_markdown="Local content that does not match the import at all.",
        origin="scheduled",
    )
    dest_db.close_connection()

    importer = ChatbookImporter(dest_paths)
    status = ImportStatus()
    ok, message = importer.import_chatbook(
        output,
        content_selections={ContentType.KEPT_BRIEFING: [str(kept_id)]},
        import_status=status,
    )
    assert ok, message

    dest_db = _dest_db(dest_paths)
    unchanged = dest_db.get_kept_briefing_by_source(101)
    assert unchanged["id"] == local_kept_id
    assert unchanged["watchlist_name"] == "Totally Different Local Watchlist"
    assert (
        unchanged["body_markdown"]
        == "Local content that does not match the import at all."
    )
    assert len(dest_db.list_kept_briefings()) == 1  # no second row for source 101

    assert any(
        "source_briefing_id=101" in warning and "conflict" in warning.lower()
        for warning in status.warnings
    ), status.warnings


def test_script_conflict_reported_and_local_row_not_overwritten(source_env, tmp_path):
    """`kept_scripts.source_script_id` is a table-wide UNIQUE column, so an
    incoming script can collide with a local script kept under a
    *different* briefing. That local row must not be overwritten, and the
    conflict must be named."""
    db_paths, kept_id = source_env
    output = _create_chatbook(db_paths, kept_id, tmp_path)

    dest_paths = _db_paths(tmp_path / "dest")
    dest_db = _dest_db(dest_paths)
    other_kept_id = dest_db.create_kept_briefing(
        source_briefing_id=999,  # unrelated briefing, no conflict at that level
        watchlist_name="Other",
        body_markdown="Other body",
        origin="manual",
    )
    dest_db.create_kept_script(
        other_kept_id,
        source_script_id=555,  # collides with the incoming script's source id
        preset_name="totally-different-preset",
        roster_snapshot_json="{}",
        turns_json="[]",
    )
    dest_db.close_connection()

    importer = ChatbookImporter(dest_paths)
    status = ImportStatus()
    ok, message = importer.import_chatbook(
        output,
        content_selections={ContentType.KEPT_BRIEFING: [str(kept_id)]},
        import_status=status,
    )
    assert ok, message

    dest_db = _dest_db(dest_paths)
    unchanged = dest_db.get_kept_script_by_source(555)
    assert unchanged["kept_briefing_id"] == other_kept_id
    assert unchanged["preset_name"] == "totally-different-preset"

    assert any("kept script(s)" in warning for warning in status.warnings), (
        status.warnings
    )


# --- Backward compatibility ---------------------------------------------


def test_import_bundle_without_kept_section_is_unaffected(tmp_path, chachanotes_template_db):
    """A chatbook predating this content type has no `kept_briefings`
    manifest entries and no `content/kept_briefings/` folder at all --
    importing it (with the default "import everything" selection) must not
    crash and must leave the destination's kept tables empty."""
    import shutil

    db_paths = _db_paths(tmp_path / "source")
    shutil.copyfile(chachanotes_template_db, db_paths["ChaChaNotes"])
    db = CharactersRAGDB(db_paths["ChaChaNotes"], "test-source")
    note_id = db.add_note(title="Plain Note", content="Nothing kept here.")
    db.close_connection()

    output = tmp_path / "legacy.zip"
    creator = ChatbookCreator(db_paths)
    ok, message, _details = creator.create_chatbook(
        name="Legacy Bundle",
        description="predates kept briefings",
        content_selections={ContentType.NOTE: [str(note_id)]},
        output_path=output,
    )
    assert ok, message

    with zipfile.ZipFile(output) as zf:
        assert not any(
            name.startswith("content/kept_briefings/") for name in zf.namelist()
        )

    dest_paths = _db_paths(tmp_path / "dest")
    importer = ChatbookImporter(dest_paths)
    ok, message = importer.import_chatbook(output, content_selections=None)
    assert ok, message

    dest_db = _dest_db(dest_paths)
    assert dest_db.list_kept_briefings() == []
