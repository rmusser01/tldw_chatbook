"""Notes export writes YAML front matter a parser -- and the app -- read back.

TASK-34000.31 (review finding N-20): both exporters f-stringed each front
matter value, so a title with a colon (``Trip planning: Lisbon``), a leading
``*``/``#``/``-``/``?`` or a quote produced a ``---`` block that
``yaml.safe_load`` rejects or reads wrongly, and the single-note export wrote
its keywords under ``keywords:`` -- a key the app's own importer never read.
Obsidian, Pandoc and **Add from files…** all failed on the file, or kept the
raw block as body text and lost the keywords.

These tests pin the round trip at the two surfaces that matter: a standard
YAML parser (what other tools do) and the app's own Markdown importer
(``split_frontmatter`` + ``_frontmatter_keywords``, what **Add from files…**
does). The bundle arms go through the real ``ChatbookCreator`` and
``ChatbookImporter`` into real file-backed SQLite -- the "it works" claim is
the destination database's row. No app import and no Hypothesis: this file
runs in the fast PR lane; the generated-input properties live in
``test_note_frontmatter_round_trip_properties.py``.
"""

from __future__ import annotations

from datetime import datetime

import pytest
import yaml

from Tests.Chatbooks.test_chatbook_keywords_round_trip import (
    _add_note,
    _db_trio,
    _export,
    _import,
    _manifest_items,
    _note_keywords,
    _read_text,
)
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.Library.library_notes_state import build_note_export_content
from tldw_chatbook.Notes.note_import_parsers import (
    _frontmatter_keywords,
    split_frontmatter,
)
from tldw_chatbook.Notes.note_import_plan_models import ImportBounds

# ``bootstrap_profile``: the creator and importer resolve their private temp
# roots through ``get_user_data_dir()`` at construction (the sibling
# ``test_chatbook_keywords_round_trip.py`` carries the same marker for the
# same reason).
pytestmark = [pytest.mark.integration, pytest.mark.bootstrap_profile]

EXPORT_NOW = datetime(2026, 10, 24, 9, 30, 15)

# The task's AC#1 character set, one title per hazard. Two are negative
# controls that already parsed on the base (``日本語のメモ 📝``, and the
# date-shaped title, whose string form a plain parser also reads as a string
# only because the exporter now quotes it -- on the base it read back as a
# date-looking string too).
AWKWARD_TITLES = (
    "Paper notes: Retrieval practice",
    "Trip planning: Lisbon",
    "* starred",
    "#1 priority",
    "- leading dash",
    "? question",
    '"quoted" start',
    "it's & that's @ 100%",
    "[brackets] {braces}",
    "日本語のメモ 📝",
    "2026-10-24",
    "true",
)


def _bounds() -> ImportBounds:
    """Import once's keyword bounds (the ``test_note_import_obsidian`` shape)."""
    return ImportBounds(
        max_files=200,
        max_file_bytes=100_000,
        max_total_bytes=1_000_000,
        max_depth=6,
    )


def _single_note_block(text: str) -> dict:
    """The YAML block of a single-note export, as any front matter reader finds it."""
    assert text.startswith("---\n")
    return yaml.safe_load(text.split("---\n", 2)[1])


# --- AC#1: a standard parser reads the title back exactly --------------------


@pytest.mark.parametrize("title", AWKWARD_TITLES)
def test_single_note_export_frontmatter_parses_and_keeps_the_title(title):
    """Info > Export Markdown: ``yaml.safe_load`` reads the title back exactly."""
    text = build_note_export_content(
        title, "body", "travel, lisbon", "n-1", "markdown", now=EXPORT_NOW
    )
    data = _single_note_block(text)
    assert data["title"] == title


def test_renderer_escapes_a_nel_so_the_title_reads_back_exactly():
    """PyYAML's emitter writes U+0085 raw inside a quoted scalar when
    ``allow_unicode`` is on and its own loader reads it back as a space (the
    Hypothesis sibling found it); the renderer re-dumps escaped in that case."""
    from tldw_chatbook.Notes.note_frontmatter import render_note_frontmatter

    text = render_note_frontmatter({"title": "a\x85b", "tags": ["日本"]})
    assert yaml.safe_load(text.split("---\n", 2)[1]) == {"title": "a\x85b", "tags": ["日本"]}
    # The readable form stays the default: nothing to escape here.
    assert render_note_frontmatter({"title": "日本語のメモ 📝"}) == "---\ntitle: 日本語のメモ 📝\n---\n"


# --- AC#2: keywords are a list; no keywords is still valid -------------------


def test_single_note_export_tags_are_a_list_the_importer_reads():
    """``tags`` (the key the importer and Obsidian read) holds a list, and
    ``note_id`` stays a string."""
    text = build_note_export_content(
        "Lisbon trip", "body", "travel, lisbon", "n-1", "markdown", now=EXPORT_NOW
    )
    data = _single_note_block(text)
    assert data["tags"] == ["travel", "lisbon"]
    assert data["note_id"] == "n-1"
    assert data["date"] == "2026-10-24 09:30:15"


def test_single_note_export_without_keywords_is_still_valid_frontmatter():
    text = build_note_export_content(
        "Lisbon trip", "body", "", "n-1", "markdown", now=EXPORT_NOW
    )
    data = _single_note_block(text)
    assert data["title"] == "Lisbon trip"
    assert data["tags"] == []
    metadata, _body = split_frontmatter(text)
    assert metadata is not None
    assert _frontmatter_keywords(metadata, _bounds()) == ()


# --- AC#3: Add from files restores title and keywords, body carries no fence --


def test_single_note_export_keywords_reach_the_app_importer():
    """A plain title isolates the second defect: the exporter wrote
    ``keywords:`` and the importer read ``tags:``, so Add from files restored
    the note without its keywords."""
    text = build_note_export_content(
        "Lisbon trip", "Three days.", "travel, lisbon", "n-1", "markdown", now=EXPORT_NOW
    )
    metadata, body = split_frontmatter(text)
    assert metadata is not None
    assert metadata["title"] == "Lisbon trip"
    assert _frontmatter_keywords(metadata, _bounds()) == ("travel", "lisbon")
    assert body.startswith("# Lisbon trip")
    assert "---" not in body


def test_single_note_export_reimports_an_awkward_title_through_the_app_importer():
    """The review's own case: ``Trip planning: Lisbon`` used to leave the raw
    ``---`` block in the imported body because the importer could not parse
    it and, by design, keeps an unparseable block as text."""
    text = build_note_export_content(
        "Trip planning: Lisbon", "Three days.", "travel", "n-1", "markdown", now=EXPORT_NOW
    )
    metadata, body = split_frontmatter(text)
    assert metadata is not None
    assert metadata["title"] == "Trip planning: Lisbon"
    assert _frontmatter_keywords(metadata, _bounds()) == ("travel",)
    assert body.startswith("# Trip planning: Lisbon")
    assert "---" not in body


def test_add_from_files_import_once_reads_the_export_without_obsidian_mode(tmp_path):
    """AC#3 at the surface Add from files uses: a single exported ``.md`` picked
    through Import once is discovered and parsed with ``obsidian_mode=False``
    (the toggle is only offered for a vault). The live walk found the import
    kept the ``---`` block as body text and restored no keywords because
    ``_parse_text`` read front matter only in Obsidian mode."""
    from tldw_chatbook.Notes.note_import_discovery import discover_import_sources
    from tldw_chatbook.Notes.note_import_parsers import parse_import_sources

    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "paper.md").write_text(
        build_note_export_content(
            "Paper notes: Retrieval practice",
            "Attendees: me.",
            "meeting, thesis",
            "n-1",
            "markdown",
            now=EXPORT_NOW,
        ),
        encoding="utf-8",
    )
    bounds = _bounds()
    discovery = discover_import_sources([export_dir], bounds, obsidian_mode=False)
    batch = parse_import_sources(discovery, bounds, obsidian_mode=False)
    [source] = batch.parsed
    [payload] = source.payloads
    assert payload.title == "Paper notes: Retrieval practice"
    assert payload.keywords == ("meeting", "thesis")
    assert payload.content.startswith("# Paper notes: Retrieval practice")
    assert "---" not in payload.content
    assert payload.wikilinks == ()


def test_legacy_single_note_export_with_a_keywords_key_still_imports_its_keywords():
    """Files users exported before this fix wrote ``keywords: a, b``; the
    importer reads that key as a tags alias so those files are not orphaned."""
    legacy = (
        "---\n"
        "title: Lisbon trip\n"
        "date: 2026-07-07 09:30:15\n"
        "keywords: travel, lisbon\n"
        "note_id: n-1\n"
        "---\n\n# Lisbon trip\n\nThree days."
    )
    metadata, body = split_frontmatter(legacy)
    assert metadata is not None
    assert _frontmatter_keywords(metadata, _bounds()) == ("travel", "lisbon")
    assert body.startswith("# Lisbon trip")


# --- Bundles: the same renderer, the real creator and importer ---------------


@pytest.mark.parametrize("title", AWKWARD_TITLES)
def test_bundle_note_frontmatter_parses_and_round_trips_the_title(tmp_path, title):
    """The bundle's note file parses, names the title exactly, and the
    destination note has that title and a body without the fence."""
    source = _db_trio(tmp_path / "source", "fm-source")
    note_id = _add_note(source["chachanotes_db"], title, ["travel"])
    bundle = _export(source, {ContentType.NOTE: [note_id]}, tmp_path / "n.zip")

    [item] = _manifest_items(bundle, ContentType.NOTE)
    markdown = _read_text(bundle, item["file_path"])
    metadata, body = split_frontmatter(markdown)
    assert metadata is not None
    assert metadata["title"] == title
    assert "---" not in body

    dest = _db_trio(tmp_path / "dest", "fm-dest")
    _import(dest, bundle)
    [note] = dest["chachanotes_db"].list_notes(limit=5)
    assert note["title"] == title
    assert note["content"] == f"Body of {title}"


def test_bundle_note_tags_are_a_list_and_survive_the_import(tmp_path):
    source = _db_trio(tmp_path / "source", "fm-source")
    note_id = _add_note(source["chachanotes_db"], "Lisbon trip", ["travel", "lisbon"])
    bundle = _export(source, {ContentType.NOTE: [note_id]}, tmp_path / "n.zip")

    [item] = _manifest_items(bundle, ContentType.NOTE)
    metadata, _body = split_frontmatter(_read_text(bundle, item["file_path"]))
    assert metadata is not None
    assert metadata["tags"] == ["lisbon", "travel"]
    assert metadata["id"] == note_id

    dest = _db_trio(tmp_path / "dest", "fm-dest")
    _import(dest, bundle)
    assert _note_keywords(dest["chachanotes_db"], "Lisbon trip") == ["lisbon", "travel"]


def test_bundle_note_title_holding_a_fence_keeps_the_body_whole(tmp_path):
    """A quoted scalar may hold ``---``; the importer must split on the YAML
    block, not on the first three dashes it finds."""
    title = "before --- after"
    source = _db_trio(tmp_path / "source", "fm-source")
    note_id = _add_note(source["chachanotes_db"], title, ["travel"])
    bundle = _export(source, {ContentType.NOTE: [note_id]}, tmp_path / "n.zip")

    dest = _db_trio(tmp_path / "dest", "fm-dest")
    _import(dest, bundle)
    [note] = dest["chachanotes_db"].list_notes(limit=5)
    assert note["title"] == title
    assert note["content"] == f"Body of {title}"


def test_legacy_bundle_frontmatter_without_manifest_tags_still_yields_keywords(tmp_path):
    """A hand-edited bundle whose manifest carries no tags falls back to the
    note file's ``tags:``, written as a comma string by the old creator or as
    a list by this one -- both read."""
    from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter

    assert ChatbookImporter._tags_from_metadata({"tags": "meeting, moc"}) == [
        "meeting",
        "moc",
    ]
    assert ChatbookImporter._tags_from_metadata({"tags": ["meeting", "moc"]}) == [
        "meeting",
        "moc",
    ]
    assert ChatbookImporter._tags_from_metadata({"tags": None}) == []
    assert ChatbookImporter._tags_from_metadata({}) == []
    assert ChatbookImporter._tags_from_metadata(None) == []
