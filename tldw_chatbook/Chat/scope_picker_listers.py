"""Real ``SourceLister``/``TagLister`` adapters for ``ConsoleScopePickerModal``.

``console_scope_picker_modal.py`` is pure UI/selection bookkeeping and knows
nothing about the app's actual media/notes/keyword seams -- it is driven
entirely by three injected callables (``media_lister``, ``notes_lister``,
``tag_lister``). This module builds the REAL versions of those callables,
wired to the app-level seams the rest of Console already uses:

- Media: ``app.media_reading_scope_service.search_media`` (the same seam
  ``Library.library_local_rag_search_service`` uses for scoped keyword
  search).
- Notes: ``app.notes_scope_service.search_notes`` for text queries,
  ``app.chachanotes_db.list_notes``/``get_keywords_for_note`` for the
  untextfiltered/tag-filtered paths (see ``_NotesSourceLister`` for why).
- Tags: a union of both DBs' Keywords vocabularies -- ``MediaDatabase.
  get_keyword_usage_stats()`` (media) and a capped scan over
  ``CharactersRAGDB.list_keywords``/``get_notes_for_keyword`` (notes; no
  single-query usage-stats seam exists there).

Per the rag-scope-narrowing design's plan-time verification #6, these
listers query the raw seams directly (never anything Library-screen-side),
so they always see the FULL universe regardless of any Library-side
visibility/workspace filtering -- restriction to a workspace's own item set
(the picker's ``universe`` parameter) is applied entirely inside the modal
itself, never here.

All three adapters are plain, dependency-light objects over the app
instance (mirroring ``LibraryLocalRagSearchService``'s own
``getattr(self._app, "...", None)`` seam-lookup convention) so they can be
constructed once per modal-open call.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from tldw_chatbook.DB.Client_Media_DB_v2 import fetch_keywords_for_media_batch
from tldw_chatbook.Library.library_fts_query import build_fts_match_query
from tldw_chatbook.Widgets.Console.console_scope_picker_modal import (
    SORT_TITLE,
    ScopeListItem,
    ScopeListPage,
    SourceLister,
    TagCount,
    TagLister,
)

# -- media -------------------------------------------------------------------

#: Picker sort id -> ``search_media_db`` ``sort_by`` value. "type" has no
#: native single-type equivalent (type is uniform within one lister); it
#: falls back to recency, same as an unrecognized sort id would.
#: DELIBERATELY excludes "title" -- see ``_MEDIA_TITLE_SORT_FETCH_CAP``.
_MEDIA_SORT_MAP = {
    "recent": "last_modified_desc",
    "type": "last_modified_desc",
}
#: Cap for the unpaginated "all matching ids" query (Select-all-matching,
#: live selected/matching count). Mirrors the design spec's ~1-2k Chroma
#: `$in` planning ceiling; a library with more matches than this under one
#: filter is a v1 edge case (documented, not silently wrong -- fewer ids
#: come back than truly match).
_MEDIA_ID_ONLY_LIMIT = 5000
#: Title-sort fetch window, client-side sorted/paginated from here (mirrors
#: the notes lister's ``_NOTES_FETCH_CAP`` approach). ``search_media_db``'s
#: own ``sort_by="title_asc"``/``"title_desc"`` paths build an invalid
#: ``ORDER BY ... ASC COLLATE NOCASE`` clause (COLLATE must precede ASC/DESC)
#: and always raise a SQL syntax error for a non-FTS query -- a pre-existing
#: bug in ``Client_Media_DB_v2.search_media_db``, out of this task's scope
#: to fix (found during task-9 testing; flagged for a follow-up). Sorting
#: client-side over a bounded recency-ordered window sidesteps it entirely.
_MEDIA_TITLE_SORT_FETCH_CAP = 500
#: Bounded, text-filtered candidate window for client-side OR-of-tags
#: matching when 2+ tags are selected (task-9 review finding 3):
#: ``search_media``'s ``must_have_keywords`` filter is AND-only at the DB
#: layer, so multi-tag OR semantics are computed here instead, mirroring
#: ``_NotesSourceLister._matching``'s same client-side-OR pattern. Only the
#: TAG filter moves client-side -- the text query itself still runs
#: server-side, so it stays ANDed against the OR'd tags. Mirrors
#: ``_NOTES_FETCH_CAP``/``_MEDIA_TITLE_SORT_FETCH_CAP``'s bounded-fetch
#: precedent (a filter matching more than this many candidates is a
#: documented v1 edge case, not a silent-wrong-result bug).
_MEDIA_TAG_OR_FETCH_CAP = 500


def _media_scope_item(
    row: Mapping[str, Any], tags: tuple[str, ...] = ()
) -> ScopeListItem:
    """Build a ``ScopeListItem`` from one ``media_reading_scope_service.search_media``
    normalized row.

    ``source_id`` is deliberately the seam's ``source_id`` field (the raw
    media DB row id, coerced to ``str``) -- the SAME field
    ``library_local_rag_search_service._media_row`` reads for scope
    identity -- never the seam's own ``id`` field, which is a
    backend-prefixed canonical string (``"local:media:<id>"``) used for
    cross-backend routing, not the RAG index's ``source_id`` stamp.

    ``tags`` defaults to ``()`` for the untagged/single-tag path (unchanged
    behavior -- those items' tags were never surfaced here and nothing
    downstream reads them); the multi-tag OR path (task-9 review finding 3)
    passes the item's actual keyword tags through instead.
    """
    return ScopeListItem(
        source_id=str(row.get("source_id") or row.get("id") or ""),
        title=str(row.get("title") or ""),
        updated_at=str(row.get("updated_at") or ""),
        tags=tags,
    )


class _MediaSourceLister:
    """``SourceLister`` adapter over ``media_reading_scope_service.search_media``.

    Tag filtering (picker tag chips active) forwards as the seam's
    ``must_have_keywords`` filter, which is an ALL-of-selected-tags (AND)
    match at the DB layer -- the seam exposes no OR-of-keywords filter. For
    a SINGLE selected tag this is identical to the design spec's OR
    semantics (AND of one term == OR of one term), so that path is left
    alone. For TWO OR MORE selected tags, OR is computed client-side
    instead (task-9 review finding 3, see ``_MEDIA_TAG_OR_FETCH_CAP``):
    ``must_have_keywords`` is no longer forwarded, and matching batch-fetches
    each candidate's tags via ``fetch_keywords_for_media_batch`` and keeps
    any item sharing at least one tag with the selection.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    def _service(self) -> Any:
        return getattr(self._app, "media_reading_scope_service", None)

    def _media_db(self) -> Any:
        return getattr(self._app, "media_db", None)

    async def _keywords_for_media(
        self, media_ids: tuple[str, ...]
    ) -> dict[str, tuple[str, ...]]:
        """Batch-fetch keyword tags for a set of media ids, off-loop.

        Single query via ``fetch_keywords_for_media_batch`` (task-9 review
        finding 3) -- mirrors ``_NotesSourceLister._tags_for``'s per-item
        seam, but batched since the media DB exposes a batch query and the
        notes DB does not. Degrades to an empty mapping (no matches) on any
        missing seam or query failure -- callers already only reach here
        under an active tag filter, so a broken lookup must never widen to
        "everything matches".
        """
        db = self._media_db()
        if db is None or not media_ids:
            return {}
        try:
            int_ids = [int(media_id) for media_id in media_ids]
        except (TypeError, ValueError):
            return {}
        try:
            keywords_by_id = await asyncio.to_thread(
                fetch_keywords_for_media_batch, db, int_ids
            )
        except Exception:
            return {}
        return {
            str(media_id): tuple(keywords)
            for media_id, keywords in keywords_by_id.items()
        }

    async def _matching_multi_tag(
        self, *, text: str, tags: tuple[str, ...]
    ) -> list[ScopeListItem]:
        """Client-side OR-of-tags match over a bounded, text-filtered window.

        Only reached for 2+ selected tags -- see the class docstring and
        ``_MEDIA_TAG_OR_FETCH_CAP``.
        """
        service = self._service()
        if service is None:
            return []
        try:
            payload = await service.search_media(
                mode="local",
                query=text or None,
                limit=_MEDIA_TAG_OR_FETCH_CAP,
                offset=0,
                sort_by="last_modified_desc",
            )
        except Exception:
            return []
        items = payload.get("items", []) if isinstance(payload, Mapping) else []
        rows = [row for row in items if isinstance(row, Mapping)]
        media_ids = tuple(
            str(row.get("source_id") or row.get("id") or "")
            for row in rows
            if row.get("source_id") or row.get("id")
        )
        tags_by_id = await self._keywords_for_media(media_ids)
        tag_set = set(tags)
        matched: list[ScopeListItem] = []
        for row in rows:
            source_id = str(row.get("source_id") or row.get("id") or "")
            item_tags = tags_by_id.get(source_id, ())
            if not (tag_set & set(item_tags)):
                continue
            matched.append(_media_scope_item(row, item_tags))
        return matched

    async def list_page(
        self, *, text: str, tags: tuple[str, ...], sort: str, offset: int, limit: int
    ) -> ScopeListPage:
        if len(tags) >= 2:
            matched = await self._matching_multi_tag(text=text, tags=tags)
            if sort == SORT_TITLE:
                matched.sort(key=lambda item: item.title.lower())
            else:
                matched.sort(key=lambda item: item.updated_at or "", reverse=True)
            page = tuple(matched[offset : offset + limit])
            return ScopeListPage(items=page, total_matching=len(matched))
        if sort == SORT_TITLE:
            return await self._list_page_title_sorted(
                text=text, tags=tags, offset=offset, limit=limit
            )
        service = self._service()
        if service is None:
            return ScopeListPage(items=(), total_matching=0)
        kwargs: dict[str, Any] = {
            "sort_by": _MEDIA_SORT_MAP.get(sort, "last_modified_desc")
        }
        if tags:
            kwargs["must_have_keywords"] = list(tags)
        try:
            payload = await service.search_media(
                mode="local", query=text or None, limit=limit, offset=offset, **kwargs
            )
        except Exception:
            return ScopeListPage(items=(), total_matching=0)
        items = payload.get("items", []) if isinstance(payload, Mapping) else []
        rows = [row for row in items if isinstance(row, Mapping)]
        total_raw = payload.get("total", len(rows)) if isinstance(payload, Mapping) else len(rows)
        try:
            total_matching = int(total_raw or 0)
        except (TypeError, ValueError):
            total_matching = len(rows)
        return ScopeListPage(
            items=tuple(_media_scope_item(row) for row in rows),
            total_matching=total_matching,
        )

    async def _list_page_title_sorted(
        self, *, text: str, tags: tuple[str, ...], offset: int, limit: int
    ) -> ScopeListPage:
        """Title-sort path: DB-side recency fetch + client-side sort/page.

        See ``_MEDIA_TITLE_SORT_FETCH_CAP`` for why this avoids
        ``search_media_db``'s own (currently broken) title sort. Only
        reached for 0/1 selected tags (see ``list_page``); 2+ tags route
        through ``_matching_multi_tag`` instead.
        """
        service = self._service()
        if service is None:
            return ScopeListPage(items=(), total_matching=0)
        kwargs: dict[str, Any] = {"sort_by": "last_modified_desc"}
        if tags:
            kwargs["must_have_keywords"] = list(tags)
        try:
            payload = await service.search_media(
                mode="local",
                query=text or None,
                limit=_MEDIA_TITLE_SORT_FETCH_CAP,
                offset=0,
                **kwargs,
            )
        except Exception:
            return ScopeListPage(items=(), total_matching=0)
        items = payload.get("items", []) if isinstance(payload, Mapping) else []
        rows = [_media_scope_item(row) for row in items if isinstance(row, Mapping)]
        rows.sort(key=lambda item: item.title.lower())
        page = tuple(rows[offset : offset + limit])
        return ScopeListPage(items=page, total_matching=len(rows))

    async def list_ids(self, *, text: str, tags: tuple[str, ...]) -> tuple[str, ...]:
        if len(tags) >= 2:
            matched = await self._matching_multi_tag(text=text, tags=tags)
            return tuple(item.source_id for item in matched)
        service = self._service()
        if service is None:
            return ()
        kwargs: dict[str, Any] = {}
        if tags:
            kwargs["must_have_keywords"] = list(tags)
        try:
            payload = await service.search_media(
                mode="local",
                query=text or None,
                limit=_MEDIA_ID_ONLY_LIMIT,
                offset=0,
                **kwargs,
            )
        except Exception:
            return ()
        items = payload.get("items", []) if isinstance(payload, Mapping) else []
        return tuple(
            str(row.get("source_id") or row.get("id") or "")
            for row in items
            if isinstance(row, Mapping) and (row.get("source_id") or row.get("id"))
        )


def build_media_source_lister(app: Any) -> SourceLister:
    """Build the picker's real media ``SourceLister``."""
    return _MediaSourceLister(app)


# -- notes ---------------------------------------------------------------------

#: v1 cap on notes candidates considered per filter (text or unfiltered).
#: Neither ``NotesScopeService.search_notes`` nor the underlying
#: ``CharactersRAGDB.search_notes`` support ``offset`` or a tag filter --
#: this adapter fetches up to this many matches, then paginates/sorts/
#: tag-filters client-side over that fetched window. Exact within the cap
#: (mirrors the modal's own "small enough to fit a handful of pages is
#: exact" v1 posture); a single filtered query matching more than this many
#: notes is a documented v1 edge case, not a silent-wrong-result bug (later
#: matches are simply not offered, never substituted).
_NOTES_FETCH_CAP = 500


class _NotesSourceLister:
    """``SourceLister`` adapter over the notes seams.

    See module and ``_NOTES_FETCH_CAP`` docs for the v1 pagination/tag-
    filter caveats this adapter works around.
    """

    def __init__(self, app: Any, *, user_id: str) -> None:
        self._app = app
        self._user_id = user_id

    def _db(self) -> Any:
        return getattr(self._app, "chachanotes_db", None)

    async def _candidate_rows(self, *, text: str) -> list[dict[str, Any]]:
        db = self._db()
        if db is None:
            return []
        if text:
            service = getattr(self._app, "notes_scope_service", None)
            if service is None:
                return []
            try:
                rows = await service.search_notes(
                    scope="local_note",
                    query=text,
                    limit=_NOTES_FETCH_CAP,
                    user_id=self._user_id,
                    fts_match_query=build_fts_match_query(text),
                )
            except Exception:
                return []
            return [dict(row) for row in rows or () if isinstance(row, Mapping)]
        try:
            rows = await asyncio.to_thread(db.list_notes, limit=_NOTES_FETCH_CAP, offset=0)
        except Exception:
            return []
        return [dict(row) for row in rows or ()]

    async def _tags_for(self, note_id: str) -> tuple[str, ...]:
        db = self._db()
        if db is None:
            return ()
        try:
            rows = await asyncio.to_thread(db.get_keywords_for_note, str(note_id))
        except Exception:
            return ()
        return tuple(
            str(row.get("keyword") or "").strip()
            for row in rows or ()
            if str(row.get("keyword") or "").strip()
        )

    async def _matching(
        self, *, text: str, tags: tuple[str, ...]
    ) -> list[ScopeListItem]:
        candidates = await self._candidate_rows(text=text)
        items: list[ScopeListItem] = []
        for row in candidates:
            note_id = row.get("id")
            if note_id is None:
                continue
            item_tags: tuple[str, ...] = ()
            if tags:
                item_tags = await self._tags_for(str(note_id))
                if not (set(tags) & set(item_tags)):
                    continue
            items.append(
                ScopeListItem(
                    source_id=str(note_id),
                    title=str(row.get("title") or ""),
                    updated_at=str(row.get("last_modified") or ""),
                    tags=item_tags,
                )
            )
        return items

    async def list_page(
        self, *, text: str, tags: tuple[str, ...], sort: str, offset: int, limit: int
    ) -> ScopeListPage:
        matched = await self._matching(text=text, tags=tags)
        if sort == "title":
            matched.sort(key=lambda item: item.title.lower())
        else:
            matched.sort(key=lambda item: item.updated_at or "", reverse=True)
        page = tuple(matched[offset : offset + limit])
        return ScopeListPage(items=page, total_matching=len(matched))

    async def list_ids(self, *, text: str, tags: tuple[str, ...]) -> tuple[str, ...]:
        matched = await self._matching(text=text, tags=tags)
        return tuple(item.source_id for item in matched)


def build_notes_source_lister(app: Any, *, user_id: str) -> SourceLister:
    """Build the picker's real notes ``SourceLister`` for ``user_id``."""
    return _NotesSourceLister(app, user_id=user_id)


# -- tags ------------------------------------------------------------------------

#: Cap on distinct notes keyword rows scanned to compute usage counts (no
#: single-query usage-stats seam exists on the notes side, unlike media's
#: ``get_keyword_usage_stats``).
_NOTES_TAG_VOCAB_CAP = 200
#: Cap per notes-keyword usage count query (a keyword used by more notes
#: than this reports this cap rather than the true count -- affects only
#: sort order among very heavily used tags, never correctness of matching).
_NOTES_TAG_USAGE_SAMPLE_LIMIT = 1000
#: Cap on tags returned to the modal (which itself takes only the first 10
#: for the empty-query chip row; this bounds the non-empty-query
#: autocomplete-suggestion row, which the modal does not truncate itself).
_TAG_RESULT_LIMIT = 25


def _notes_tag_usage(db: Any) -> dict[str, int]:
    """Blocking helper: keyword text -> note usage count, v1-capped."""
    counts: dict[str, int] = {}
    try:
        keyword_rows = db.list_keywords(limit=_NOTES_TAG_VOCAB_CAP)
    except Exception:
        return counts
    for row in keyword_rows or ():
        text = str(row.get("keyword") or "").strip()
        keyword_id = row.get("id")
        if not text or keyword_id is None:
            continue
        try:
            notes = db.get_notes_for_keyword(
                keyword_id, limit=_NOTES_TAG_USAGE_SAMPLE_LIMIT
            )
        except Exception:
            continue
        usage = len(notes or ())
        if usage:
            counts[text] = usage
    return counts


def build_keyword_tag_lister(app: Any) -> TagLister:
    """Build the picker's real ``TagLister``, unioning media + notes vocabularies.

    Per design spec section 4: the chip/suggestion vocabulary itself is a
    union across both DBs' Keywords tables, sorted by usage count
    descending (ties broken alphabetically); an empty ``query`` returns the
    overall top-used tags, a non-empty ``query`` returns a case-insensitive
    substring match over that same unioned vocabulary. All I/O runs off the
    UI loop via ``asyncio.to_thread``.
    """

    async def _tag_lister(query: str) -> tuple[TagCount, ...]:
        counts: dict[str, int] = {}

        media_db = getattr(app, "media_db", None)
        if media_db is not None:
            try:
                stats = await asyncio.to_thread(media_db.get_keyword_usage_stats)
            except Exception:
                stats = []
            for row in stats or ():
                text = str(row.get("keyword") or "").strip()
                if not text:
                    continue
                try:
                    usage = int(row.get("usage_count") or 0)
                except (TypeError, ValueError):
                    usage = 0
                if usage <= 0:
                    continue
                counts[text] = counts.get(text, 0) + usage

        chachanotes_db = getattr(app, "chachanotes_db", None)
        if chachanotes_db is not None:
            try:
                notes_counts = await asyncio.to_thread(
                    _notes_tag_usage, chachanotes_db
                )
            except Exception:
                notes_counts = {}
            for text, usage in notes_counts.items():
                counts[text] = counts.get(text, 0) + usage

        normalized_query = query.strip().casefold()
        filtered = [
            (text, usage)
            for text, usage in counts.items()
            if not normalized_query or normalized_query in text.casefold()
        ]
        filtered.sort(key=lambda pair: (-pair[1], pair[0].casefold()))
        return tuple(
            TagCount(tag=text, count=usage)
            for text, usage in filtered[:_TAG_RESULT_LIMIT]
        )

    return _tag_lister
