"""Library chatbook export scope resolution contracts.

Covers ``tldw_chatbook.Library.library_export_scope``: the pure module that
decides which media/conversation/note ids a Library bulk export should
include, plus the truncation-proof full-id DB queries it relies on
(``Client_Media_DB_v2.MediaDatabase.get_all_active_media_ids``,
``DB.ChaChaNotes_DB.CharactersRAGDB.get_all_conversation_ids``/
``get_all_note_ids``).

The Library canvases render capped snapshots
(``LIBRARY_SOURCE_PAGE_SIZES`` in ``library_screen.py``: notes 100, media
50, conversations 50). The truncation-lock test below seeds well past
those caps and asserts every id round-trips -- resolving from a rendered
snapshot instead of a fresh query would silently drop rows past the cap.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_export_scope import (
    ExportScope,
    count_export_scope,
    export_scope_label,
    resolve_export_selections,
)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def media_db():
    db = MediaDatabase(":memory:", "export-scope-media-client")
    yield db
    db.close_connection()


@pytest.fixture
def chachanotes_db():
    db = CharactersRAGDB(":memory:", "export-scope-ccn-client")
    yield db
    db.close_connection()


class _PoisonMediaDB:
    """A media source that fails the test if touched -- for scope isolation checks."""

    def get_all_active_media_ids(self, media_type=None):
        raise AssertionError("get_all_active_media_ids must not be called for a media-out-of-scope export")


class _PoisonChaChaNotesDB:
    """A ChaChaNotes source that fails the test if touched -- for scope isolation checks."""

    def get_all_conversation_ids(self):
        raise AssertionError("get_all_conversation_ids must not be called for a conversations-out-of-scope export")

    def get_all_note_ids(self):
        raise AssertionError("get_all_note_ids must not be called for a notes-out-of-scope export")


# --- ExportScope --------------------------------------------------------------


def test_export_scope_rejects_unknown_kind():
    with pytest.raises(ValueError):
        ExportScope(kind="bogus")


# --- THE TRUNCATION LOCK ------------------------------------------------------


def test_truncation_lock_everything_resolves_every_id_beyond_snapshot_caps(media_db, chachanotes_db):
    """Seed well past the Library's 50-row media/conversation snapshot caps.

    ``resolve_export_selections`` must return every seeded id, proving it
    never reads a rendered (capped) Library snapshot.
    """
    seeded_media_ids = []
    for i in range(63):
        media_id, _, _ = media_db.add_media_with_keywords(
            title=f"Media {i}", content=f"content {i}", media_type="article"
        )
        seeded_media_ids.append(str(media_id))

    seeded_conversation_ids = []
    for i in range(63):
        conv_id = chachanotes_db.add_conversation({"title": f"Conversation {i}"})
        seeded_conversation_ids.append(conv_id)

    scope = ExportScope(kind="everything")
    counts = count_export_scope(scope, media_db, chachanotes_db)
    assert counts["media"] == 63
    assert counts["conversations"] == 63

    selections = resolve_export_selections(scope, media_db, chachanotes_db)
    assert set(selections[ContentType.MEDIA]) == set(seeded_media_ids)
    assert set(selections[ContentType.CONVERSATION]) == set(seeded_conversation_ids)
    assert len(selections[ContentType.MEDIA]) == 63
    assert len(selections[ContentType.CONVERSATION]) == 63


# --- Media type filter + soft-delete/trash exclusion -------------------------


def test_media_scope_type_filter_excludes_deleted_and_trashed_rows(media_db):
    video_id_1, _, _ = media_db.add_media_with_keywords(title="V1", content="c1", media_type="video")
    video_id_2, _, _ = media_db.add_media_with_keywords(title="V2", content="c2", media_type="video")
    media_db.add_media_with_keywords(title="A1", content="c3", media_type="article")
    deleted_video_id, _, _ = media_db.add_media_with_keywords(title="V-deleted", content="c4", media_type="video")
    trashed_video_id, _, _ = media_db.add_media_with_keywords(title="V-trashed", content="c5", media_type="video")
    media_db.soft_delete_media(deleted_video_id)
    media_db.mark_as_trash(trashed_video_id)

    scope = ExportScope(kind="media", media_type="video")
    selections = resolve_export_selections(scope, media_db, _PoisonChaChaNotesDB())

    assert set(selections[ContentType.MEDIA]) == {str(video_id_1), str(video_id_2)}
    assert ContentType.CONVERSATION not in selections
    assert ContentType.NOTE not in selections


@pytest.mark.parametrize("unfiltered_value", [None, "All"])
def test_media_scope_type_none_or_all_sentinel_is_unfiltered(media_db, unfiltered_value):
    video_id, _, _ = media_db.add_media_with_keywords(title="V1", content="c1", media_type="video")
    article_id, _, _ = media_db.add_media_with_keywords(title="A1", content="c2", media_type="article")

    scope = ExportScope(kind="media", media_type=unfiltered_value)
    selections = resolve_export_selections(scope, media_db, _PoisonChaChaNotesDB())

    assert set(selections[ContentType.MEDIA]) == {str(video_id), str(article_id)}


# --- Empty scope --------------------------------------------------------------


def test_empty_dbs_everything_scope_counts_zero_and_selections_empty(media_db, chachanotes_db):
    scope = ExportScope(kind="everything")

    counts = count_export_scope(scope, media_db, chachanotes_db)
    assert counts == {"media": 0, "conversations": 0, "notes": 0}

    selections = resolve_export_selections(scope, media_db, chachanotes_db)
    assert selections == {}


# --- Scope isolation: out-of-scope sources are zeroed / omitted, never touched --


def test_count_export_scope_zeroes_out_of_scope_sources(media_db, chachanotes_db):
    media_db.add_media_with_keywords(title="M1", content="c", media_type="video")
    chachanotes_db.add_conversation({"title": "Conv"})
    chachanotes_db.add_note("N1", "content")

    scope = ExportScope(kind="media")
    counts = count_export_scope(scope, media_db, chachanotes_db)
    assert counts == {"media": 1, "conversations": 0, "notes": 0}


def test_resolve_export_selections_conversations_scope_never_touches_media_db(chachanotes_db):
    conv_id = chachanotes_db.add_conversation({"title": "Conv"})

    scope = ExportScope(kind="conversations")
    selections = resolve_export_selections(scope, _PoisonMediaDB(), chachanotes_db)

    assert selections == {ContentType.CONVERSATION: [conv_id]}


def test_resolve_export_selections_notes_scope_never_touches_media_db(chachanotes_db):
    note_id = chachanotes_db.add_note("N1", "content")

    scope = ExportScope(kind="notes")
    selections = resolve_export_selections(scope, _PoisonMediaDB(), chachanotes_db)

    assert selections == {ContentType.NOTE: [note_id]}


# --- Label copy: exact match ---------------------------------------------------


def test_export_scope_label_everything_lists_all_three_counts():
    scope = ExportScope(kind="everything")
    label = export_scope_label(scope, {"media": 128, "conversations": 542, "notes": 87})
    assert label == "Everything: 128 media · 542 conversations · 87 notes"


def test_export_scope_label_everything_includes_zero_count_sources():
    scope = ExportScope(kind="everything")
    label = export_scope_label(scope, {"media": 0, "conversations": 542, "notes": 0})
    assert label == "Everything: 0 media · 542 conversations · 0 notes"


def test_export_scope_label_media_with_type_filter():
    scope = ExportScope(kind="media", media_type="video")
    assert export_scope_label(scope, {"media": 12}) == "Media (type: video) · 12 items"


def test_export_scope_label_media_unfiltered_none():
    scope = ExportScope(kind="media")
    assert export_scope_label(scope, {"media": 12}) == "Media · 12 items"


def test_export_scope_label_media_unfiltered_all_sentinel():
    scope = ExportScope(kind="media", media_type="All")
    assert export_scope_label(scope, {"media": 12}) == "Media · 12 items"


def test_export_scope_label_conversations():
    scope = ExportScope(kind="conversations")
    assert export_scope_label(scope, {"conversations": 542}) == "Conversations · 542 items"


def test_export_scope_label_notes():
    scope = ExportScope(kind="notes")
    assert export_scope_label(scope, {"notes": 87}) == "Notes · 87 items"
