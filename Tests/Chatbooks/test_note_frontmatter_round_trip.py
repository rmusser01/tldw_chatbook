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

# The task's AC#1 character set, one title per hazard. Two rows are negative
# controls that already parsed on the base (``it's & that's @ 100%`` and
# ``日本語のメモ 📝``); every other row was RED there -- the date-shaped title
# included, which an unquoted block reads back as ``datetime.date``.
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
    "! leading bang",
    'trailing "quote"',
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


def _import_once(root, files: dict, *, obsidian_mode: bool = False):
    """Parse ``files`` (display path -> text) exactly as Add from files does.

    ``discover_import_sources`` + ``parse_import_sources`` with the same
    ``obsidian_mode`` the picker would pass: False for a picked file or plain
    folder, True for a detected vault (``.obsidian/`` present).
    """
    from tldw_chatbook.Notes.note_import_discovery import discover_import_sources
    from tldw_chatbook.Notes.note_import_parsers import parse_import_sources

    root.mkdir(parents=True, exist_ok=True)
    if obsidian_mode:
        (root / ".obsidian").mkdir(exist_ok=True)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    bounds = _bounds()
    discovery = discover_import_sources([root], bounds, obsidian_mode=obsidian_mode)
    batch = parse_import_sources(discovery, bounds, obsidian_mode=obsidian_mode)
    return {
        source.candidate.source.display_path.split("/", 1)[1]: source.payloads[0]
        for source in batch.parsed
    }


def test_add_from_files_import_once_reads_the_export_without_obsidian_mode(tmp_path):
    """AC#3 at the surface Add from files uses: a single exported ``.md`` picked
    through Import once is discovered and parsed with ``obsidian_mode=False``
    (the toggle is only offered for a vault). The live walk found the import
    kept the ``---`` block as body text and restored no keywords because
    ``_parse_text`` read front matter only in Obsidian mode. The app's own
    export keys (``date``, ``note_id``) are not reported as "not imported"."""
    text = build_note_export_content(
        "Paper notes: Retrieval practice",
        "Attendees: me.",
        "meeting, thesis",
        "n-1",
        "markdown",
        now=EXPORT_NOW,
    )
    payload = _import_once(tmp_path / "exports", {"paper.md": text})["paper.md"]
    assert payload.title == "Paper notes: Retrieval practice"
    assert payload.keywords == ("meeting", "thesis")
    assert payload.content.startswith("# Paper notes: Retrieval practice")
    assert "---" not in payload.content
    assert payload.wikilinks == ()
    assert payload.unimported_frontmatter_keys == ()


def test_non_vault_import_keeps_a_block_with_foreign_keys_in_the_body(tmp_path):
    """Fix round 1 (I-1): outside a vault, a block holding any key the import
    does not honour is not the app's own export, so the WHOLE file stays as
    on base -- nothing is dropped and nothing needs reporting."""
    text = "---\nSummary: the meeting ran long\ntags: [meeting]\n---\n\n# Standup\n\nBody."
    payload = _import_once(tmp_path / "plain", {"standup.md": text})["standup.md"]
    assert payload.content == text
    assert payload.title == "Standup"
    assert payload.keywords == ()
    assert payload.unimported_frontmatter_keys == ()


def test_non_vault_import_keeps_a_block_with_non_string_keys_in_the_body(tmp_path):
    """``Yes:`` reads as the bool key True and ``2026:`` as an int key: outside a
    vault the block stays in the body rather than being dropped silently."""
    text = "---\nYes: we shipped it\n2026: a great year\n---\n\n# Year\n\nBody."
    payload = _import_once(tmp_path / "plain", {"year.md": text})["year.md"]
    assert payload.content == text
    assert payload.title == "Year"
    assert payload.unimported_frontmatter_keys == ()


def test_vault_import_reports_non_string_keys_too_many_tags_and_uses_a_numeric_title(
    tmp_path,
):
    """Vault mode strips the block as before, but nothing goes silently: a
    non-string key is reported as ``str(key)``, tags that fail their bounds are
    kept out of the note AND reported, and ``title: 2026`` is used as "2026"."""
    tags = ", ".join(f"t{i}" for i in range(101))
    text = f"---\nYes: shipped\ntitle: 2026\ntags: {tags}\n---\n\n# Heading\n\nBody."
    payload = _import_once(tmp_path / "vault", {"year.md": text}, obsidian_mode=True)[
        "year.md"
    ]
    assert payload.title == "2026"
    assert payload.content == "# Heading\n\nBody."
    assert payload.keywords == ()
    reported = payload.unimported_frontmatter_keys
    assert "True" in reported
    assert any(entry.startswith("tags") for entry in reported), reported


def test_vault_import_does_not_report_the_apps_own_export_keys(tmp_path):
    """M3: ``date``, ``note_id``, ``id``, ``created_at``, ``updated_at`` are the
    exporters' own provenance; dropping them is not a loss worth a receipt line."""
    text = (
        "---\nid: n-1\ntitle: Kept\ndate: '2026-10-24 09:30:15'\nnote_id: n-1\n"
        "created_at: '2026-10-24T09:00:00+00:00'\nupdated_at: '2026-10-24T09:30:00+00:00'\n"
        "tags:\n- a\n---\n\n# Kept\n\nBody."
    )
    payload = _import_once(tmp_path / "vault", {"kept.md": text}, obsidian_mode=True)[
        "kept.md"
    ]
    assert payload.title == "Kept"
    assert payload.keywords == ("a",)
    assert payload.unimported_frontmatter_keys == ()


def test_non_vault_import_keeps_a_capitalised_title_tags_block_whole(tmp_path):
    """R2-1: the readers match keys exactly (``title``, not ``Title``), so a
    ``Title:``/``Tags:`` block is not one the import can read -- outside a vault
    it stays whole, nothing is lost and nothing needs reporting."""
    text = "---\nTitle: My Title\nTags: a, b\n---\nBody"
    payload = _import_once(tmp_path / "plain", {"cap.md": text})["cap.md"]
    assert payload.content == text
    assert payload.title == "cap"
    assert payload.keywords == ()
    assert payload.unimported_frontmatter_keys == ()


def test_vault_import_reports_a_users_own_date_property(tmp_path):
    """R2-3: ``date:`` is one of the most common Obsidian properties. It is
    silenced only when the whole block has the shape of the app's own export;
    beside a ``mood:`` it is the user's and is reported like any other."""
    text = "---\ndate: 2026-01-01\nmood: x\n---\n\n# Day\n\nBody."
    payload = _import_once(tmp_path / "vault", {"day.md": text}, obsidian_mode=True)[
        "day.md"
    ]
    assert payload.unimported_frontmatter_keys == ("date", "mood")


@pytest.mark.parametrize("raw", ["3.10", "12:30", "0755", "2026-10-24"])
def test_vault_import_keeps_a_scalar_title_exactly_as_written(tmp_path, raw):
    """R2-4: YAML 1.1 types ``3.10`` as 3.1, ``12:30`` as 750 (base 60) and
    ``0755`` as 493; the title is the text the author wrote."""
    text = f"---\ntitle: {raw}\n---\n\n# Heading\n\nBody."
    payload = _import_once(tmp_path / "vault", {"t.md": text}, obsidian_mode=True)["t.md"]
    assert payload.title == raw
    assert payload.unimported_frontmatter_keys == ()


def test_vault_import_report_cap_never_hides_the_tags_report(tmp_path):
    """R2-5: twelve other properties fill the report cap; the tags line must
    still be there -- it is the only sign the keywords were lost."""
    others = "".join(f"p{i}: v\n" for i in range(12))
    tags = ", ".join(f"t{i}" for i in range(101))
    text = f"---\n{others}tags: {tags}\n---\n\n# Heading\n\nBody."
    payload = _import_once(tmp_path / "vault", {"many.md": text}, obsidian_mode=True)[
        "many.md"
    ]
    assert payload.keywords == ()
    assert any(entry.startswith("tags (more than") for entry in payload.unimported_frontmatter_keys), (
        payload.unimported_frontmatter_keys
    )


def test_vault_import_combined_cap_counts_after_dedupe(tmp_path):
    """R2-6: 60 tags plus the same 60 as ``keywords`` are 60 keywords, all
    kept -- no "more than 100 in total" report."""
    sixty = ", ".join(f"k{i}" for i in range(60))
    text = f"---\ntags: {sixty}\nkeywords: {sixty}\n---\n\n# Heading\n\nBody."
    payload = _import_once(tmp_path / "vault", {"dup.md": text}, obsidian_mode=True)[
        "dup.md"
    ]
    assert len(payload.keywords) == 60
    assert payload.unimported_frontmatter_keys == ()


FENCE_TITLE = "a\n---\nb"
"""M1: the dumper folds a multi-line title so one continuation line is an
indented ``  ---``; only a fence at column 0 may close the block."""


def test_a_title_holding_a_fence_line_round_trips_through_the_single_note_export():
    text = build_note_export_content(FENCE_TITLE, "body", "travel", "n-1", "markdown", now=EXPORT_NOW)
    metadata, _body = split_frontmatter(text)
    assert metadata is not None
    assert metadata["title"] == FENCE_TITLE


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


BUNDLE_TITLES = (*AWKWARD_TITLES, FENCE_TITLE)
TAGGED_TITLE = "Lisbon trip"


@pytest.fixture(scope="module")
def bundle_round_trip(tmp_path_factory):
    """ONE bundle holding every awkward title, the fence title and a two-keyword
    note, exported by the real creator and imported by the real importer into a
    real destination database (R2-8: fourteen separate round trips cost ~2.3 s
    each; one costs the same and the per-title tests below still name the title)."""
    root = tmp_path_factory.mktemp("fm-bundle")
    source = _db_trio(root / "source", "fm-source")
    note_ids = [_add_note(source["chachanotes_db"], title, ["travel"]) for title in BUNDLE_TITLES]
    note_ids.append(_add_note(source["chachanotes_db"], TAGGED_TITLE, ["travel", "lisbon"]))
    bundle = _export(source, {ContentType.NOTE: note_ids}, root / "all.zip")
    files = {
        item["title"]: _read_text(bundle, item["file_path"])
        for item in _manifest_items(bundle, ContentType.NOTE)
    }
    dest = _db_trio(root / "dest", "fm-dest")
    _import(dest, bundle)
    notes = {note["title"]: note for note in dest["chachanotes_db"].list_notes(limit=100)}
    return {"files": files, "notes": notes, "dest": dest, "manifest_ids": dict(zip(BUNDLE_TITLES, note_ids))}


@pytest.mark.parametrize("title", BUNDLE_TITLES)
def test_bundle_note_frontmatter_parses_and_round_trips_the_title(bundle_round_trip, title):
    """The bundle's note file parses, names the title exactly, and the
    destination note has that title and a body without the fence."""
    metadata, body = split_frontmatter(bundle_round_trip["files"][title])
    assert metadata is not None, title
    assert metadata["title"] == title
    assert metadata["id"] == bundle_round_trip["manifest_ids"][title]
    # The body is exactly the note's text: no fence, no leftover block (the
    # fence title's own body legitimately contains ``---``).
    assert body.strip() == f"Body of {title}", title
    note = bundle_round_trip["notes"][title]
    assert note["title"] == title
    assert note["content"] == f"Body of {title}"


def test_bundle_note_tags_are_a_list_and_survive_the_import(bundle_round_trip):
    metadata, _body = split_frontmatter(bundle_round_trip["files"][TAGGED_TITLE])
    assert metadata is not None
    assert metadata["tags"] == ["lisbon", "travel"]
    assert _note_keywords(bundle_round_trip["dest"]["chachanotes_db"], TAGGED_TITLE) == [
        "lisbon",
        "travel",
    ]


def test_a_pre_fix_bundle_with_an_unparseable_block_still_imports_a_clean_body(tmp_path):
    """M4: a bundle the OLD creator wrote -- unquoted ``title: Trip planning:
    Lisbon`` (no parser reads it) and a comma ``tags:`` line, manifest tags
    ``null`` -- still imports with the body free of the block and the
    keywords taken from that line (the legacy fence-split fallback)."""
    import json
    import zipfile
    from datetime import datetime, timezone

    from Tests.Chatbooks.factories import ContentItemFactory, ManifestFactory
    from tldw_chatbook.Chatbooks.chatbook_models import ChatbookVersion

    stamp = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    item = ContentItemFactory.create(
        ContentType.NOTE,
        item_id="legacy-note",
        title="Trip planning: Lisbon",
        tags=[],
        created_at=stamp,
        updated_at=stamp,
        file_path="content/notes/legacy-note.md",
    )
    manifest = ManifestFactory.create(
        name="Legacy", content_items=[item], version=ChatbookVersion.V1
    ).to_dict()
    manifest["content_items"][0]["tags"] = None
    note_md = (
        "---\n"
        "id: legacy-note\n"
        "title: Trip planning: Lisbon\n"
        f"created_at: {stamp.isoformat()}\n"
        f"updated_at: {stamp.isoformat()}\n"
        "tags: meeting, moc\n"
        "---\n\n"
        "Legacy note body\n"
    )
    bundle = tmp_path / "legacy.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("content/notes/legacy-note.md", note_md)
    assert split_frontmatter(note_md)[0] is None  # the block really is unreadable

    dest = _db_trio(tmp_path / "dest", "fm-dest")
    _import(dest, bundle)
    [note] = dest["chachanotes_db"].list_notes(limit=5)
    assert note["title"] == "Trip planning: Lisbon"
    assert note["content"] == "Legacy note body"
    assert _note_keywords(dest["chachanotes_db"], "Trip planning") == ["meeting", "moc"]


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
