"""Tests for the real ``SourceLister``/``TagLister`` adapters (task-9).

Real file-backed ``MediaDatabase``/``CharactersRAGDB`` (the same fixture
recipe ``Tests/Library/test_library_rag_scope.py`` uses for its real-seam
coverage), driven through the exact app-level seams
(``media_reading_scope_service``, ``notes_scope_service``, ``media_db``,
``chachanotes_db``) ``scope_picker_listers.py`` reads -- never a hand-rolled
fake of the listers themselves.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.scope_picker_listers import (
    build_keyword_tag_lister,
    build_media_source_lister,
    build_notes_source_lister,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

_NOTES_USER = "scope-lister-user"


@pytest.fixture
def lister_stack(tmp_path):
    """Real notes/media DBs wired through the same app-level seams the
    Console screen uses, with NO workspace/visibility-filtering wrapper of
    any kind on the ``app`` namespace -- the listers must still see
    everything (plan-time verification #6 / V6)."""
    chachanotes_db = CharactersRAGDB(tmp_path / "chacha.db", client_id="scope-lister")
    media_db = MediaDatabase(tmp_path / "media.db", client_id="scope-lister")
    notes_interop = NotesInteropService(
        base_db_directory=tmp_path / "notes_base",
        api_client_id="scope-lister",
        global_db_to_use=chachanotes_db,
    )
    app = SimpleNamespace(
        notes_scope_service=NotesScopeService(
            local_notes_service=notes_interop, server_service=None
        ),
        media_reading_scope_service=MediaReadingScopeService(
            LocalMediaReadingService(media_db), None
        ),
        media_db=media_db,
        chachanotes_db=chachanotes_db,
        notes_user_id=_NOTES_USER,
    )
    try:
        yield app, chachanotes_db, media_db, notes_interop
    finally:
        media_db.close_connection()
        chachanotes_db.close_connection()


# -- media ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_lister_list_page_and_list_ids_real_seam(lister_stack):
    app, _chachanotes_db, media_db, _notes_interop = lister_stack
    media_id, _uuid, _msg = media_db.add_media_with_keywords(
        title="Quarterly Roadmap", media_type="document", content="roadmap content"
    )
    other_id, _uuid2, _msg2 = media_db.add_media_with_keywords(
        title="Unrelated report", media_type="document", content="other content"
    )
    lister = build_media_source_lister(app)

    ids = await lister.list_ids(text="roadmap", tags=())
    assert ids == (str(media_id),)

    page = await lister.list_page(
        text="", tags=(), sort="recent", offset=0, limit=20
    )
    returned_ids = {item.source_id for item in page.items}
    assert returned_ids == {str(media_id), str(other_id)}
    assert page.total_matching == 2
    titles = {item.source_id: item.title for item in page.items}
    assert titles[str(media_id)] == "Quarterly Roadmap"


@pytest.mark.asyncio
async def test_media_lister_paginates_and_sorts_by_title(lister_stack):
    app, _chachanotes_db, media_db, _notes_interop = lister_stack
    ids = []
    for title in ("Charlie", "Alpha", "Bravo"):
        media_id, _uuid, _msg = media_db.add_media_with_keywords(
            title=title, media_type="document", content=f"{title} body"
        )
        ids.append(media_id)
    lister = build_media_source_lister(app)

    page1 = await lister.list_page(text="", tags=(), sort="title", offset=0, limit=2)
    page2 = await lister.list_page(text="", tags=(), sort="title", offset=2, limit=2)

    assert [item.title for item in page1.items] == ["Alpha", "Bravo"]
    assert [item.title for item in page2.items] == ["Charlie"]
    assert page1.total_matching == 3
    assert page2.total_matching == 3


@pytest.mark.asyncio
async def test_media_lister_multi_tag_selection_is_or_not_and(lister_stack):
    """task-9 review finding 3: selecting 2+ tags must OR them (any
    selected tag matches), not AND (all selected tags required) --
    ``search_media``'s own ``must_have_keywords`` filter is AND-only at the
    DB layer, so multi-tag selection is computed client-side instead (like
    ``_NotesSourceLister._matching``'s existing client-side-OR pattern)."""
    app, _chachanotes_db, media_db, _notes_interop = lister_stack
    sales_id, _u1, _m1 = media_db.add_media_with_keywords(
        title="Sales report",
        media_type="document",
        content="sales body",
        keywords=["sales"],
    )
    q3_id, _u2, _m2 = media_db.add_media_with_keywords(
        title="Q3 summary",
        media_type="document",
        content="q3 body",
        keywords=["q3"],
    )
    both_id, _u3, _m3 = media_db.add_media_with_keywords(
        title="Sales Q3 combo",
        media_type="document",
        content="combo body",
        keywords=["sales", "q3"],
    )
    neither_id, _u4, _m4 = media_db.add_media_with_keywords(
        title="Unrelated",
        media_type="document",
        content="unrelated body",
    )
    lister = build_media_source_lister(app)

    ids = await lister.list_ids(text="", tags=("sales", "q3"))

    assert set(ids) == {str(sales_id), str(q3_id), str(both_id)}
    assert str(neither_id) not in ids


@pytest.mark.asyncio
async def test_media_lister_single_tag_semantics_unchanged(lister_stack):
    """A single selected tag keeps the existing (unchanged) seam-side AND
    path -- AND of one tag is identical to OR of one tag, so this must
    behave exactly as before the finding-3 fix."""
    app, _chachanotes_db, media_db, _notes_interop = lister_stack
    sales_id, _u1, _m1 = media_db.add_media_with_keywords(
        title="Sales report",
        media_type="document",
        content="sales body",
        keywords=["sales"],
    )
    media_db.add_media_with_keywords(
        title="Q3 summary",
        media_type="document",
        content="q3 body",
        keywords=["q3"],
    )
    lister = build_media_source_lister(app)

    ids = await lister.list_ids(text="", tags=("sales",))

    assert ids == (str(sales_id),)


@pytest.mark.asyncio
async def test_media_lister_multi_tag_still_ands_against_text_query(lister_stack):
    """Multi-tag OR still ANDs against the text query: only items matching
    the text query AND at least one of the selected tags are returned."""
    app, _chachanotes_db, media_db, _notes_interop = lister_stack
    matching_id, _u1, _m1 = media_db.add_media_with_keywords(
        title="Roadmap sales doc",
        media_type="document",
        content="roadmap sales content",
        keywords=["sales"],
    )
    media_db.add_media_with_keywords(
        title="Unrelated q3 doc",
        media_type="document",
        content="unrelated q3 content",
        keywords=["q3"],
    )
    lister = build_media_source_lister(app)

    ids = await lister.list_ids(text="roadmap", tags=("sales", "q3"))

    assert ids == (str(matching_id),)


@pytest.mark.asyncio
async def test_media_lister_missing_seam_degrades_to_empty():
    app = SimpleNamespace(media_reading_scope_service=None)
    lister = build_media_source_lister(app)

    page = await lister.list_page(text="", tags=(), sort="recent", offset=0, limit=20)
    ids = await lister.list_ids(text="", tags=())

    assert page == page.__class__(items=(), total_matching=0)
    assert ids == ()


# -- notes -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notes_lister_list_page_and_list_ids_real_seam(lister_stack):
    app, _chachanotes_db, _media_db, notes_interop = lister_stack
    note_a = notes_interop.add_note(
        user_id=_NOTES_USER, title="Roadmap notes", content="roadmap details"
    )
    note_b = notes_interop.add_note(
        user_id=_NOTES_USER, title="Unrelated notes", content="other details"
    )
    lister = build_notes_source_lister(app, user_id=_NOTES_USER)

    ids = await lister.list_ids(text="roadmap", tags=())
    assert ids == (note_a,)

    page = await lister.list_page(text="", tags=(), sort="recent", offset=0, limit=20)
    returned_ids = {item.source_id for item in page.items}
    assert returned_ids == {note_a, note_b}
    assert page.total_matching == 2


@pytest.mark.asyncio
async def test_notes_lister_paginates_client_side_over_fetch_window(lister_stack):
    app, _chachanotes_db, _media_db, notes_interop = lister_stack
    for title in ("Charlie", "Alpha", "Bravo"):
        notes_interop.add_note(user_id=_NOTES_USER, title=title, content=f"{title} body")
    lister = build_notes_source_lister(app, user_id=_NOTES_USER)

    page1 = await lister.list_page(text="", tags=(), sort="title", offset=0, limit=2)
    page2 = await lister.list_page(text="", tags=(), sort="title", offset=2, limit=2)

    assert [item.title for item in page1.items] == ["Alpha", "Bravo"]
    assert [item.title for item in page2.items] == ["Charlie"]
    assert page1.total_matching == 3


@pytest.mark.asyncio
async def test_notes_lister_tag_filter_restricts_to_tagged_notes(lister_stack):
    app, chachanotes_db, _media_db, notes_interop = lister_stack
    tagged = notes_interop.add_note(
        user_id=_NOTES_USER, title="Tagged note", content="body"
    )
    notes_interop.add_note(user_id=_NOTES_USER, title="Untagged note", content="body")
    keyword_id = chachanotes_db.add_keyword("sales")
    chachanotes_db.link_note_to_keyword(tagged, keyword_id)
    lister = build_notes_source_lister(app, user_id=_NOTES_USER)

    ids = await lister.list_ids(text="", tags=("sales",))

    assert ids == (tagged,)


@pytest.mark.asyncio
async def test_notes_lister_tag_filter_issues_one_batched_keyword_lookup(
    lister_stack, monkeypatch
):
    """PR #747 review: ``_NotesSourceLister._matching`` previously called
    ``get_keywords_for_note`` once PER CANDIDATE inside the tag-filter loop
    -- up to ``_NOTES_FETCH_CAP`` (500) serialized threaded DB calls per
    refresh. Mirrors the media lister's prior batch fix
    (``fetch_keywords_for_media_batch``): tag lookups for the whole
    candidate window must land in exactly ONE DB call
    (``get_keywords_for_notes_batch``), and multi-tag OR + tag+text AND
    matching semantics must stay unchanged."""
    app, chachanotes_db, _media_db, notes_interop = lister_stack
    sales = notes_interop.add_note(
        user_id=_NOTES_USER, title="Sales note", content="body"
    )
    q3 = notes_interop.add_note(user_id=_NOTES_USER, title="Q3 note", content="body")
    both = notes_interop.add_note(
        user_id=_NOTES_USER, title="Sales Q3 note", content="body"
    )
    neither = notes_interop.add_note(
        user_id=_NOTES_USER, title="Unrelated note", content="body"
    )
    sales_kw = chachanotes_db.add_keyword("sales")
    q3_kw = chachanotes_db.add_keyword("q3")
    chachanotes_db.link_note_to_keyword(sales, sales_kw)
    chachanotes_db.link_note_to_keyword(q3, q3_kw)
    chachanotes_db.link_note_to_keyword(both, sales_kw)
    chachanotes_db.link_note_to_keyword(both, q3_kw)

    batch_calls: list[list[str]] = []
    real_batch = chachanotes_db.get_keywords_for_notes_batch

    def _spy_batch(note_ids):
        batch_calls.append(list(note_ids))
        return real_batch(note_ids)

    monkeypatch.setattr(chachanotes_db, "get_keywords_for_notes_batch", _spy_batch)

    per_note_calls: list[str] = []
    real_single = chachanotes_db.get_keywords_for_note

    def _spy_single(note_id):
        per_note_calls.append(note_id)
        return real_single(note_id)

    monkeypatch.setattr(chachanotes_db, "get_keywords_for_note", _spy_single)

    lister = build_notes_source_lister(app, user_id=_NOTES_USER)

    # Multi-tag OR: any note carrying at least one of the selected tags.
    ids = await lister.list_ids(text="", tags=("sales", "q3"))

    assert set(ids) == {sales, q3, both}
    assert neither not in ids
    assert len(batch_calls) == 1, batch_calls
    assert per_note_calls == [], per_note_calls


@pytest.mark.asyncio
async def test_notes_lister_tag_filter_still_ands_against_text_query(
    lister_stack,
):
    """Tag+text semantics unchanged by batching: only notes matching the
    text query AND at least one selected tag are returned."""
    app, chachanotes_db, _media_db, notes_interop = lister_stack
    matching = notes_interop.add_note(
        user_id=_NOTES_USER, title="Roadmap sales note", content="roadmap sales body"
    )
    notes_interop.add_note(
        user_id=_NOTES_USER, title="Unrelated q3 note", content="unrelated q3 body"
    )
    sales_kw = chachanotes_db.add_keyword("sales")
    chachanotes_db.link_note_to_keyword(matching, sales_kw)
    lister = build_notes_source_lister(app, user_id=_NOTES_USER)

    ids = await lister.list_ids(text="roadmap", tags=("sales", "q3"))

    assert ids == (matching,)


@pytest.mark.asyncio
async def test_notes_lister_missing_seam_degrades_to_empty():
    app = SimpleNamespace(chachanotes_db=None)
    lister = build_notes_source_lister(app, user_id=_NOTES_USER)

    page = await lister.list_page(text="", tags=(), sort="recent", offset=0, limit=20)
    ids = await lister.list_ids(text="", tags=())

    assert page.items == ()
    assert page.total_matching == 0
    assert ids == ()


# -- tags -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_tag_lister_unions_media_and_notes_vocabularies(lister_stack):
    app, chachanotes_db, media_db, notes_interop = lister_stack
    # Distinct content per item: `Media.content_hash` is UNIQUE, so two
    # inserts sharing the same content dedup to one row instead of two.
    media_db.add_media_with_keywords(
        title="M1", media_type="document", content="c1", keywords=["alpha"]
    )
    media_db.add_media_with_keywords(
        title="M2", media_type="document", content="c2", keywords=["alpha"]
    )
    note_id = notes_interop.add_note(user_id=_NOTES_USER, title="N1", content="c")
    keyword_id = chachanotes_db.add_keyword("beta")
    chachanotes_db.link_note_to_keyword(note_id, keyword_id)
    tag_lister = build_keyword_tag_lister(app)

    top = await tag_lister("")

    tags_by_name = {tc.tag: tc.count for tc in top}
    assert tags_by_name.get("alpha") == 2
    assert tags_by_name.get("beta") == 1


@pytest.mark.asyncio
async def test_keyword_tag_lister_query_filters_case_insensitive(lister_stack):
    app, _chachanotes_db, media_db, _notes_interop = lister_stack
    # MediaDatabase.add_keyword stores keywords lowercased; the query match
    # itself must still be case-insensitive regardless of casing on either
    # side ("sales" query against the stored lowercase "sales report").
    media_db.add_media_with_keywords(
        title="M1", media_type="document", content="c", keywords=["Sales Report"]
    )
    tag_lister = build_keyword_tag_lister(app)

    suggestions = await tag_lister("SALES")

    assert any(tc.tag == "sales report" for tc in suggestions)


@pytest.mark.asyncio
async def test_keyword_tag_lister_missing_seams_returns_empty():
    app = SimpleNamespace(media_db=None, chachanotes_db=None)
    tag_lister = build_keyword_tag_lister(app)

    result = await tag_lister("")

    assert result == ()


# -- V6: the listers see the full universe, ignoring any Library-side --------
# -- visibility/workspace filtering (plan-time verification #6) --------------


@pytest.mark.asyncio
async def test_v6_listers_return_full_universe_with_no_visibility_wrapper(
    lister_stack,
):
    """The `app` fixture above carries no workspace registry, no
    scope_type-aware wrapper, nothing that could hide an item the way a
    Library-side "current workspace" filter can (the `scope_type="all"`
    hide-bug class referenced in the design spec's plan-time verifications)
    -- every item inserted into the raw DBs is still fully reachable
    through the picker's real listers, proving the adapters query the raw
    seams directly rather than anything Library-screen-filtered."""
    app, chachanotes_db, media_db, notes_interop = lister_stack
    media_ids = {
        media_db.add_media_with_keywords(
            title=f"Media {i}", media_type="document", content=f"body {i}"
        )[0]
        for i in range(5)
    }
    note_ids = {
        notes_interop.add_note(user_id=_NOTES_USER, title=f"Note {i}", content=f"body {i}")
        for i in range(5)
    }
    media_lister = build_media_source_lister(app)
    notes_lister = build_notes_source_lister(app, user_id=_NOTES_USER)

    media_seen = set(await media_lister.list_ids(text="", tags=()))
    notes_seen = set(await notes_lister.list_ids(text="", tags=()))

    assert media_seen == {str(mid) for mid in media_ids}
    assert notes_seen == note_ids
