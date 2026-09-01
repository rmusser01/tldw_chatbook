"""Authority-scoped SQLite repository for Local Collections captures."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Subscriptions.html_text import readable_body_text

from .collections_capture_models import (
    CAPTURE_PAGE_SIZE,
    CAPTURE_STATUSES,
    CaptureActionResult,
    CaptureConflict,
    CaptureConflictError,
    CaptureDetail,
    CaptureHighlight,
    CaptureHighlightDraft,
    CaptureIdentity,
    CaptureNoteLink,
    CaptureOfflineCopy,
    CapturePage,
    CapturePageRequest,
    CaptureSaveOutcome,
    CaptureSaveRequest,
    CaptureSavedSearchPage,
    CaptureSummary,
    CollectionsCaptureError,
    ExternalMediaReference,
    ExternalNoteReference,
    SavedCaptureSearch,
)


_SORT_SQL = {
    "saved_desc": "item.created_at DESC, item.capture_id DESC",
    "saved_asc": "item.created_at ASC, item.capture_id ASC",
    "updated_desc": "item.updated_at DESC, item.capture_id DESC",
    "updated_asc": "item.updated_at ASC, item.capture_id ASC",
    "title_asc": "COALESCE(item.title, '') COLLATE NOCASE ASC, item.capture_id ASC",
    "title_desc": "COALESCE(item.title, '') COLLATE NOCASE DESC, item.capture_id DESC",
    "relevance": "bm25(collection_capture_search) ASC, item.capture_id ASC",
}
_UPDATE_FIELDS = frozenset(
    {
        "title",
        "summary",
        "freeform_note",
        "text_content",
        "clean_html",
        "byline",
        "published_at",
        "read_at",
        "status",
        "favorite",
        "tags",
    }
)
_STRIPPED_UPDATE_FIELDS = frozenset(
    {"title", "summary", "byline", "published_at", "read_at"}
)
_CONTENT_UPDATE_FIELDS = frozenset(
    {"freeform_note", "text_content", "clean_html"}
)
_EXTRACTION_FAILURE_REASONS = frozenset(
    {
        "dependency_missing",
        "empty_extraction",
        "fetch_failed",
        "invalid_url",
        "network_error",
        "redirect_limit",
        "response_too_large",
        "unsafe_url",
        "unsupported_content",
        "unknown",
    }
)


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _optional_string(value: Any, reason: str, *, strip: bool) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CollectionsCaptureError(reason)
    if not strip:
        return value
    normalized = value.strip()
    return normalized or None


def _normalized_tags(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise CollectionsCaptureError("invalid_tags")
    tags: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise CollectionsCaptureError("invalid_tags")
        display = value.strip()
        tags.setdefault(display.casefold(), display)
    return tuple(tags[key] for key in sorted(tags))


def _canonicalize_url(value: str) -> tuple[str, str]:
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise CollectionsCaptureError("invalid_canonical_url") from exc
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise CollectionsCaptureError("invalid_canonical_url")
    if parsed.username is not None or parsed.password is not None:
        raise CollectionsCaptureError("invalid_canonical_url")
    host = parsed.hostname.casefold()
    rendered_host = f"[{host}]" if ":" in host else host
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        rendered_host = f"{rendered_host}:{port}"
    path = parsed.path or "/"
    canonical = urlunsplit((scheme, rendered_host, path, parsed.query, ""))
    return canonical, host


def _content_hash(text_content: str | None, clean_html: str | None) -> str | None:
    if text_content is None and clean_html is None:
        return None
    body = json.dumps(
        [text_content, clean_html],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


def _fts_query(search: str) -> str | None:
    tokens = tuple(dict.fromkeys(re.findall(r"\w+", search, flags=re.UNICODE)))
    if not tokens:
        return None
    return " AND ".join(f'"{token}"*' for token in tokens)


def _inclusive_date_to(value: str) -> tuple[str, str]:
    """Return a comparison and bound that include a date-only final day."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return "item.created_at <= ?", value
    try:
        following_day = date.fromisoformat(value) + timedelta(days=1)
    except ValueError as exc:
        raise CollectionsCaptureError("invalid_date_to") from exc
    return "item.created_at < ?", following_day.isoformat()


class CollectionsCaptureRepository:
    """Persist one Local capture authority in a shared Collections database."""

    def __init__(
        self,
        db: LibraryCollectionsDB,
        *,
        authority_key: str,
        clock: Callable[[], str] = _now,
    ) -> None:
        if not isinstance(db, LibraryCollectionsDB):
            raise CollectionsCaptureError("invalid_collections_database")
        if not isinstance(authority_key, str) or not authority_key.strip():
            raise CollectionsCaptureError("invalid_authority_key")
        self.db = db
        self.authority_key = authority_key.strip()
        self._clock = clock
        self.db.require_capture_schema()

    def _require_identity(self, identity: CaptureIdentity) -> None:
        if not isinstance(identity, CaptureIdentity):
            raise CollectionsCaptureError("invalid_capture_identity")
        if identity.authority_key != self.authority_key:
            raise CollectionsCaptureError("authority_mismatch")

    def _require_request(self, request: CapturePageRequest | CaptureSaveRequest) -> None:
        if request.authority_key != self.authority_key:
            raise CollectionsCaptureError("authority_mismatch")

    def _after_page_count(self) -> None:
        """Test seam invoked after count has pinned the read snapshot."""

    def list_page(self, request: CapturePageRequest) -> CapturePage:
        """Return one exact page and total from a single read snapshot."""
        if not isinstance(request, CapturePageRequest):
            raise CollectionsCaptureError("invalid_page_request")
        self._require_request(request)

        joins: list[str] = []
        where = ["item.authority_key = ?", "item.purge_state IS NULL"]
        params: list[Any] = [self.authority_key]
        if request.search:
            search_query = _fts_query(request.search)
            if search_query is None:
                where.append("0 = 1")
            else:
                joins.append(
                    "JOIN collection_capture_search "
                    "ON collection_capture_search.authority_key = item.authority_key "
                    "AND collection_capture_search.capture_id = item.capture_id"
                )
                where.append("collection_capture_search MATCH ?")
                params.append(search_query)
        if request.statuses:
            placeholders = ", ".join("?" for _ in request.statuses)
            where.append(f"item.status IN ({placeholders})")
            params.extend(request.statuses)
        if request.favorite is not None:
            where.append("item.favorite = ?")
            params.append(int(request.favorite))
        if request.domain is not None:
            where.append("item.domain = ?")
            params.append(request.domain)
        if request.date_from is not None:
            where.append("item.created_at >= ?")
            params.append(request.date_from)
        if request.date_to is not None:
            date_to_clause, date_to_bound = _inclusive_date_to(request.date_to)
            where.append(date_to_clause)
            params.append(date_to_bound)
        for tag in request.tags:
            where.append(
                "EXISTS ("
                "SELECT 1 FROM collection_capture_item_tags AS item_tag "
                "JOIN collection_capture_tags AS tag_filter "
                "ON tag_filter.authority_key = item_tag.authority_key "
                "AND tag_filter.tag_id = item_tag.tag_id "
                "WHERE item_tag.authority_key = item.authority_key "
                "AND item_tag.capture_id = item.capture_id "
                "AND tag_filter.normalized_name = ?"
                ")"
            )
            params.append(tag)

        from_sql = " ".join(
            ["FROM collection_capture_items AS item", *joins]
        )
        where_sql = " AND ".join(where)
        order_sql = _SORT_SQL[request.sort]
        offset = (request.page - 1) * request.size

        with self.db.read_transaction() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) {from_sql} WHERE {where_sql}",  # noqa: S608
                    tuple(params),
                ).fetchone()[0]
            )
            self._after_page_count()
            rows_sql = (
                "SELECT item.*, "  # nosec B608
                "EXISTS(SELECT 1 FROM collection_capture_offline_files AS offline "
                "WHERE offline.authority_key = item.authority_key "
                "AND offline.capture_id = item.capture_id "
                "AND offline.state = 'ready') AS has_offline_copy "
                f"{from_sql} WHERE {where_sql} ORDER BY {order_sql} LIMIT ? OFFSET ?"
            )
            rows = connection.execute(
                rows_sql,
                tuple([*params, request.size, offset]),
            ).fetchall()
            tags = self._tags_for_ids(
                connection,
                [str(row["capture_id"]) for row in rows],
            )
            items = tuple(
                self._summary_from_row(row, tags.get(str(row["capture_id"]), ()))
                for row in rows
            )
        return CapturePage(applied=request, items=items, total=total)

    def get_detail(self, identity: CaptureIdentity) -> CaptureDetail | None:
        """Return one aggregate or None when it is absent/tombstoned."""
        self._require_identity(identity)
        with self.db.read_transaction() as connection:
            return self._get_detail(connection, identity)

    def save_capture(self, request: CaptureSaveRequest) -> CaptureSaveOutcome:
        """Insert or authority-locally upsert one canonical URL."""
        if not isinstance(request, CaptureSaveRequest):
            raise CollectionsCaptureError("invalid_save_request")
        self._require_request(request)
        canonical_url, domain = _canonicalize_url(
            request.canonical_url or request.submitted_url
        )
        now = self._clock()
        with self.db.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM collection_capture_items "
                "WHERE authority_key = ? AND canonical_url = ?",
                (self.authority_key, canonical_url),
            ).fetchone()
            if existing is not None and existing["purge_state"] is not None:
                raise CollectionsCaptureError("capture_pending_purge")
            if existing is None:
                identity = CaptureIdentity(self.authority_key, _new_id("capture"))
                text_content = request.text_content
                clean_html = request.clean_html
                processing_state = (
                    "ready"
                    if text_content is not None or clean_html is not None
                    else "queued"
                )
                connection.execute(
                    "INSERT INTO collection_capture_items ("
                    "authority_key, capture_id, submitted_url, canonical_url, domain, "
                    "title, summary, freeform_note, text_content, clean_html, byline, "
                    "published_at, read_at, content_hash, word_count, status, favorite, "
                    "processing_state, last_fetch_error, media_authority_key, media_item_id, "
                    "created_at, updated_at, revision, purge_state"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "NULL, NULL, NULL, ?, ?, 1, NULL)",
                    (
                        self.authority_key,
                        identity.capture_id,
                        request.submitted_url,
                        canonical_url,
                        domain,
                        request.title,
                        request.summary,
                        request.freeform_note,
                        text_content,
                        clean_html,
                        request.byline,
                        request.published_at,
                        now if request.status == "read" else None,
                        _content_hash(text_content, clean_html),
                        self._word_count(text_content),
                        request.status or "saved",
                        int(request.favorite or False),
                        processing_state,
                        now,
                        now,
                    ),
                )
                self._add_tags(connection, identity.capture_id, request.tags)
                created = True
            else:
                identity = CaptureIdentity(
                    self.authority_key, str(existing["capture_id"])
                )
                text_content = (
                    request.text_content
                    if request.text_content is not None
                    else existing["text_content"]
                )
                clean_html = (
                    request.clean_html
                    if request.clean_html is not None
                    else existing["clean_html"]
                )
                content_changed = (
                    request.text_content is not None or request.clean_html is not None
                )
                status = request.status or str(existing["status"])
                read_at = existing["read_at"]
                if request.status == "read" and read_at is None:
                    read_at = now
                elif request.status in {"saved", "reading"}:
                    read_at = None
                connection.execute(
                    "UPDATE collection_capture_items SET "
                    "submitted_url = ?, domain = ?, title = ?, summary = ?, "
                    "freeform_note = ?, text_content = ?, clean_html = ?, byline = ?, "
                    "published_at = ?, read_at = ?, content_hash = ?, word_count = ?, "
                    "status = ?, favorite = ?, processing_state = ?, last_fetch_error = ?, "
                    "updated_at = ?, revision = revision + 1 "
                    "WHERE authority_key = ? AND capture_id = ?",
                    (
                        request.submitted_url,
                        domain,
                        request.title if request.title is not None else existing["title"],
                        request.summary if request.summary is not None else existing["summary"],
                        request.freeform_note
                        if request.freeform_note is not None
                        else existing["freeform_note"],
                        text_content,
                        clean_html,
                        request.byline if request.byline is not None else existing["byline"],
                        request.published_at
                        if request.published_at is not None
                        else existing["published_at"],
                        read_at,
                        _content_hash(text_content, clean_html)
                        if content_changed
                        else existing["content_hash"],
                        self._word_count(text_content)
                        if request.text_content is not None
                        else existing["word_count"],
                        status,
                        int(request.favorite)
                        if request.favorite is not None
                        else int(existing["favorite"]),
                        "ready" if content_changed else existing["processing_state"],
                        None if content_changed else existing["last_fetch_error"],
                        now,
                        self.authority_key,
                        identity.capture_id,
                    ),
                )
                self._add_tags(connection, identity.capture_id, request.tags)
                created = False
            detail = self._get_detail(connection, identity)
            if detail is None:
                raise CollectionsCaptureError("capture_write_failed")
        return CaptureSaveOutcome(
            detail,
            created,
            extraction_pending=detail.processing_state in {"queued", "processing"},
        )

    def update_capture(
        self,
        identity: CaptureIdentity,
        *,
        expected_revision: int,
        changes: Mapping[str, Any],
    ) -> CaptureDetail:
        """Apply an allowlisted reading-state update guarded by revision."""
        self._require_identity(identity)
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 1
        ):
            raise CollectionsCaptureError("invalid_revision")
        if not isinstance(changes, Mapping) or set(changes) - _UPDATE_FIELDS:
            raise CollectionsCaptureError("invalid_capture_changes")
        now = self._clock()
        with self.db.transaction() as connection:
            row = self._active_item_row(connection, identity)
            if int(row["revision"]) != expected_revision:
                current = self._get_detail(connection, identity)
                if current is None:
                    raise CollectionsCaptureError("capture_not_found")
                raise CaptureConflictError(
                    CaptureConflict(identity, expected_revision, current)
                )

            values: dict[str, Any] = {}
            for field in _STRIPPED_UPDATE_FIELDS & changes.keys():
                values[field] = _optional_string(
                    changes[field], f"invalid_{field}", strip=True
                )
            for field in _CONTENT_UPDATE_FIELDS & changes.keys():
                values[field] = _optional_string(
                    changes[field], f"invalid_{field}", strip=False
                )
            if "status" in changes:
                status = changes["status"]
                if not isinstance(status, str) or status.casefold() not in CAPTURE_STATUSES:
                    raise CollectionsCaptureError("invalid_status")
                values["status"] = status.casefold()
                if values["status"] == "read" and row["read_at"] is None:
                    values.setdefault("read_at", now)
                elif values["status"] in {"saved", "reading"}:
                    values.setdefault("read_at", None)
            if "favorite" in changes:
                if not isinstance(changes["favorite"], bool):
                    raise CollectionsCaptureError("invalid_favorite")
                values["favorite"] = int(changes["favorite"])
            tags = (
                _normalized_tags(changes["tags"]) if "tags" in changes else None
            )
            text_content = values.get("text_content", row["text_content"])
            clean_html = values.get("clean_html", row["clean_html"])
            if "text_content" in values or "clean_html" in values:
                values["content_hash"] = _content_hash(text_content, clean_html)
                values["word_count"] = self._word_count(text_content)
                values["processing_state"] = "ready"
                values["last_fetch_error"] = None
            values["updated_at"] = now

            assignments = [f"{field} = ?" for field in values]
            assignments.append("revision = revision + 1")
            update_sql = (
                "UPDATE collection_capture_items SET "  # nosec B608
                + ", ".join(assignments)
                + " WHERE authority_key = ? AND capture_id = ? AND revision = ?"
            )
            connection.execute(
                update_sql,
                tuple([*values.values(), self.authority_key, identity.capture_id, expected_revision]),
            )
            if tags is not None:
                self._replace_tags(connection, identity.capture_id, tags)
            detail = self._get_detail(connection, identity)
            if detail is None:
                raise CollectionsCaptureError("capture_update_failed")
            return detail

    def claim_extraction(
        self,
        identity: CaptureIdentity,
        *,
        expected_revision: int,
    ) -> CaptureDetail:
        """Claim one queued capture for extraction."""
        self._require_identity(identity)
        expected_revision = self._expected_revision(expected_revision)
        now = self._clock()
        with self.db.transaction() as connection:
            self._extraction_row(
                connection,
                identity,
                expected_revision=expected_revision,
                states={"queued"},
            )
            connection.execute(
                "UPDATE collection_capture_items SET processing_state = 'processing', "
                "last_fetch_error = NULL, updated_at = ?, revision = revision + 1 "
                "WHERE authority_key = ? AND capture_id = ? AND revision = ? "
                "AND processing_state = 'queued'",
                (now, self.authority_key, identity.capture_id, expected_revision),
            )
            return self._written_detail(connection, identity)

    def complete_extraction(
        self,
        identity: CaptureIdentity,
        *,
        expected_revision: int,
        result: Mapping[str, Any],
    ) -> CaptureDetail:
        """Store extracted content as inert text and finish the active claim."""
        self._require_identity(identity)
        expected_revision = self._expected_revision(expected_revision)
        if not isinstance(result, Mapping) or not isinstance(result.get("content"), str):
            raise CollectionsCaptureError("invalid_extraction_result")
        text_content = readable_body_text(result["content"]).strip()
        if not text_content:
            raise CollectionsCaptureError("empty_extraction_content")
        now = self._clock()
        with self.db.transaction() as connection:
            row = self._extraction_row(
                connection,
                identity,
                expected_revision=expected_revision,
                states={"processing"},
            )
            title = self._inert_optional_extraction_text(
                result.get("title"),
                reason="invalid_extraction_title",
            )
            byline = self._inert_optional_extraction_text(
                result.get("author"),
                reason="invalid_extraction_author",
            )
            connection.execute(
                "UPDATE collection_capture_items SET text_content = ?, clean_html = NULL, "
                "title = ?, byline = ?, content_hash = ?, word_count = ?, "
                "processing_state = 'ready', last_fetch_error = NULL, updated_at = ?, "
                "revision = revision + 1 "
                "WHERE authority_key = ? AND capture_id = ? AND revision = ? "
                "AND processing_state = 'processing'",
                (
                    text_content,
                    title or row["title"],
                    byline or row["byline"],
                    _content_hash(text_content, None),
                    self._word_count(text_content),
                    now,
                    self.authority_key,
                    identity.capture_id,
                    expected_revision,
                ),
            )
            return self._written_detail(connection, identity)

    def fail_extraction(
        self,
        identity: CaptureIdentity,
        *,
        expected_revision: int,
        reason: str,
    ) -> CaptureDetail:
        """Finish an active extraction with a bounded, content-free reason."""
        self._require_identity(identity)
        expected_revision = self._expected_revision(expected_revision)
        if reason not in _EXTRACTION_FAILURE_REASONS:
            raise CollectionsCaptureError("invalid_extraction_failure_reason")
        now = self._clock()
        with self.db.transaction() as connection:
            self._extraction_row(
                connection,
                identity,
                expected_revision=expected_revision,
                states={"processing"},
            )
            connection.execute(
                "UPDATE collection_capture_items SET processing_state = 'failed', "
                "last_fetch_error = ?, updated_at = ?, revision = revision + 1 "
                "WHERE authority_key = ? AND capture_id = ? AND revision = ? "
                "AND processing_state = 'processing'",
                (
                    reason,
                    now,
                    self.authority_key,
                    identity.capture_id,
                    expected_revision,
                ),
            )
            return self._written_detail(connection, identity)

    def retry_extraction(
        self,
        identity: CaptureIdentity,
        *,
        expected_revision: int,
    ) -> CaptureDetail:
        """Requeue one failed or interrupted extraction without changing reading state."""
        self._require_identity(identity)
        expected_revision = self._expected_revision(expected_revision)
        now = self._clock()
        with self.db.transaction() as connection:
            self._extraction_row(
                connection,
                identity,
                expected_revision=expected_revision,
                states={"failed", "interrupted"},
            )
            connection.execute(
                "UPDATE collection_capture_items SET processing_state = 'queued', "
                "last_fetch_error = NULL, updated_at = ?, revision = revision + 1 "
                "WHERE authority_key = ? AND capture_id = ? AND revision = ? "
                "AND processing_state IN ('failed', 'interrupted')",
                (now, self.authority_key, identity.capture_id, expected_revision),
            )
            return self._written_detail(connection, identity)

    def interrupt_stale_extractions(self) -> int:
        """Mark this authority's abandoned processing rows interrupted at startup."""
        now = self._clock()
        with self.db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE collection_capture_items SET processing_state = 'interrupted', "
                "last_fetch_error = 'interrupted', updated_at = ?, revision = revision + 1 "
                "WHERE authority_key = ? AND processing_state = 'processing' "
                "AND purge_state IS NULL",
                (now, self.authority_key),
            )
            return int(cursor.rowcount)

    def list_saved_searches(
        self,
        *,
        page: int,
        size: int = CAPTURE_PAGE_SIZE,
    ) -> CaptureSavedSearchPage:
        request_page = CapturePageRequest(self.authority_key, page=page, size=size).page
        offset = (request_page - 1) * CAPTURE_PAGE_SIZE
        with self.db.read_transaction() as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM collection_capture_saved_searches "
                    "WHERE authority_key = ?",
                    (self.authority_key,),
                ).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT * FROM collection_capture_saved_searches "
                "WHERE authority_key = ? "
                "ORDER BY updated_at DESC, search_id DESC LIMIT ? OFFSET ?",
                (self.authority_key, CAPTURE_PAGE_SIZE, offset),
            ).fetchall()
            items = tuple(self._saved_search_from_row(row) for row in rows)
        return CaptureSavedSearchPage(
            items=items,
            total=total,
            page=request_page,
            size=CAPTURE_PAGE_SIZE,
        )

    def create_saved_search(
        self,
        name: str,
        request: CapturePageRequest,
    ) -> SavedCaptureSearch:
        if not isinstance(request, CapturePageRequest):
            raise CollectionsCaptureError("invalid_saved_search_request")
        self._require_request(request)
        normalized_name = self._saved_search_name(name)
        now = self._clock()
        search_id = _new_id("search")
        try:
            with self.db.transaction() as connection:
                connection.execute(
                    "INSERT INTO collection_capture_saved_searches ("
                    "authority_key, search_id, name, query_json, created_at, updated_at, revision"
                    ") VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (
                        self.authority_key,
                        search_id,
                        normalized_name,
                        self._request_json(request),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise CollectionsCaptureError("saved_search_name_exists") from exc
        return SavedCaptureSearch(
            self.authority_key,
            search_id,
            normalized_name,
            request,
            now,
            now,
            1,
        )

    def update_saved_search(
        self,
        search_id: str,
        *,
        name: str,
        request: CapturePageRequest,
        expected_revision: int,
    ) -> SavedCaptureSearch:
        if not isinstance(request, CapturePageRequest):
            raise CollectionsCaptureError("invalid_saved_search_request")
        self._require_request(request)
        expected_revision = self._expected_revision(expected_revision)
        normalized_id = self._opaque_id(search_id, "invalid_search_id")
        normalized_name = self._saved_search_name(name)
        now = self._clock()
        try:
            with self.db.transaction() as connection:
                row = connection.execute(
                    "SELECT created_at, revision FROM collection_capture_saved_searches "
                    "WHERE authority_key = ? AND search_id = ?",
                    (self.authority_key, normalized_id),
                ).fetchone()
                if row is None:
                    raise CollectionsCaptureError("saved_search_not_found")
                if int(row["revision"]) != expected_revision:
                    raise CollectionsCaptureError("revision_conflict")
                connection.execute(
                    "UPDATE collection_capture_saved_searches SET "
                    "name = ?, query_json = ?, updated_at = ?, revision = revision + 1 "
                    "WHERE authority_key = ? AND search_id = ? AND revision = ?",
                    (
                        normalized_name,
                        self._request_json(request),
                        now,
                        self.authority_key,
                        normalized_id,
                        expected_revision,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise CollectionsCaptureError("saved_search_name_exists") from exc
        return SavedCaptureSearch(
            self.authority_key,
            normalized_id,
            normalized_name,
            request,
            str(row["created_at"]),
            now,
            expected_revision + 1,
        )

    def delete_saved_search(
        self,
        search_id: str,
        *,
        expected_revision: int,
    ) -> CaptureActionResult:
        normalized_id = self._opaque_id(search_id, "invalid_search_id")
        expected_revision = self._expected_revision(expected_revision)
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT revision FROM collection_capture_saved_searches "
                "WHERE authority_key = ? AND search_id = ?",
                (self.authority_key, normalized_id),
            ).fetchone()
            if row is None:
                raise CollectionsCaptureError("saved_search_not_found")
            if int(row["revision"]) != expected_revision:
                raise CollectionsCaptureError("revision_conflict")
            connection.execute(
                "DELETE FROM collection_capture_saved_searches "
                "WHERE authority_key = ? AND search_id = ? AND revision = ?",
                (self.authority_key, normalized_id, expected_revision),
            )
        return CaptureActionResult(
            CaptureIdentity(self.authority_key, normalized_id),
            True,
            revision=expected_revision,
        )

    def list_highlights(
        self, identity: CaptureIdentity
    ) -> tuple[CaptureHighlight, ...]:
        self._require_identity(identity)
        with self.db.read_transaction() as connection:
            self._active_item_row(connection, identity)
            rows = connection.execute(
                "SELECT * FROM collection_capture_highlights "
                "WHERE authority_key = ? AND capture_id = ? "
                "ORDER BY created_at, highlight_id",
                (self.authority_key, identity.capture_id),
            ).fetchall()
        return tuple(self._highlight_from_row(identity, row) for row in rows)

    def save_highlight(
        self,
        identity: CaptureIdentity,
        draft: CaptureHighlightDraft,
    ) -> CaptureHighlight:
        self._require_identity(identity)
        if not isinstance(draft, CaptureHighlightDraft):
            raise CollectionsCaptureError("invalid_highlight_draft")
        highlight_id = _new_id("highlight")
        now = self._clock()
        with self.db.transaction() as connection:
            self._active_item_row(connection, identity)
            connection.execute(
                "INSERT INTO collection_capture_highlights ("
                "authority_key, highlight_id, capture_id, quote, note, anchor_json, "
                "detached, created_at, updated_at, revision"
                ") VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, 1)",
                (
                    self.authority_key,
                    highlight_id,
                    identity.capture_id,
                    draft.quote,
                    draft.note,
                    draft.anchor_json,
                    now,
                    now,
                ),
            )
        return CaptureHighlight(
            identity,
            highlight_id,
            draft.quote,
            draft.note,
            draft.anchor_json,
            False,
            now,
            now,
            1,
        )

    def delete_highlight(
        self,
        identity: CaptureIdentity,
        highlight_id: str,
        *,
        expected_revision: int,
    ) -> CaptureActionResult:
        self._require_identity(identity)
        normalized_id = self._opaque_id(highlight_id, "invalid_highlight_id")
        expected_revision = self._expected_revision(expected_revision)
        with self.db.transaction() as connection:
            self._active_item_row(connection, identity)
            row = connection.execute(
                "SELECT revision FROM collection_capture_highlights "
                "WHERE authority_key = ? AND capture_id = ? AND highlight_id = ?",
                (self.authority_key, identity.capture_id, normalized_id),
            ).fetchone()
            if row is None:
                raise CollectionsCaptureError("highlight_not_found")
            if int(row["revision"]) != expected_revision:
                raise CollectionsCaptureError("revision_conflict")
            connection.execute(
                "DELETE FROM collection_capture_highlights "
                "WHERE authority_key = ? AND capture_id = ? AND highlight_id = ? "
                "AND revision = ?",
                (
                    self.authority_key,
                    identity.capture_id,
                    normalized_id,
                    expected_revision,
                ),
            )
        return CaptureActionResult(identity, True, revision=expected_revision)

    def list_note_links(
        self, identity: CaptureIdentity
    ) -> tuple[CaptureNoteLink, ...]:
        self._require_identity(identity)
        with self.db.read_transaction() as connection:
            self._active_item_row(connection, identity)
            rows = connection.execute(
                "SELECT * FROM collection_capture_note_links "
                "WHERE authority_key = ? AND capture_id = ? "
                "ORDER BY created_at, link_id",
                (self.authority_key, identity.capture_id),
            ).fetchall()
        return tuple(self._note_link_from_row(identity, row) for row in rows)

    def link_note(
        self,
        identity: CaptureIdentity,
        note_reference: ExternalNoteReference,
    ) -> CaptureNoteLink:
        self._require_identity(identity)
        if not isinstance(note_reference, ExternalNoteReference):
            raise CollectionsCaptureError("invalid_note_reference")
        now = self._clock()
        with self.db.transaction() as connection:
            self._active_item_row(connection, identity)
            existing = connection.execute(
                "SELECT * FROM collection_capture_note_links "
                "WHERE authority_key = ? AND capture_id = ? "
                "AND note_authority_key = ? AND note_id = ?",
                (
                    self.authority_key,
                    identity.capture_id,
                    note_reference.authority_key,
                    note_reference.note_id,
                ),
            ).fetchone()
            if existing is not None:
                return self._note_link_from_row(identity, existing)
            link_id = _new_id("link")
            connection.execute(
                "INSERT INTO collection_capture_note_links ("
                "authority_key, link_id, capture_id, note_authority_key, note_id, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.authority_key,
                    link_id,
                    identity.capture_id,
                    note_reference.authority_key,
                    note_reference.note_id,
                    now,
                ),
            )
        return CaptureNoteLink(identity, link_id, note_reference, now)

    def unlink_note(
        self,
        identity: CaptureIdentity,
        link_id: str,
    ) -> CaptureActionResult:
        self._require_identity(identity)
        normalized_id = self._opaque_id(link_id, "invalid_note_link_id")
        with self.db.transaction() as connection:
            capture = self._active_item_row(connection, identity)
            cursor = connection.execute(
                "DELETE FROM collection_capture_note_links "
                "WHERE authority_key = ? AND capture_id = ? AND link_id = ?",
                (self.authority_key, identity.capture_id, normalized_id),
            )
            if cursor.rowcount != 1:
                raise CollectionsCaptureError("note_link_not_found")
        return CaptureActionResult(
            identity,
            True,
            revision=int(capture["revision"]),
        )

    def hard_delete(
        self,
        identity: CaptureIdentity,
        *,
        expected_revision: int,
    ) -> CaptureActionResult:
        """Make a capture inaccessible under a durable purge tombstone."""
        self._require_identity(identity)
        expected_revision = self._expected_revision(expected_revision)
        now = self._clock()
        with self.db.transaction() as connection:
            row = self._active_item_row(connection, identity)
            if int(row["revision"]) != expected_revision:
                current = self._get_detail(connection, identity)
                if current is None:
                    raise CollectionsCaptureError("capture_not_found")
                raise CaptureConflictError(
                    CaptureConflict(identity, expected_revision, current)
                )
            connection.execute(
                "UPDATE collection_capture_items SET purge_state = 'pending', "
                "updated_at = ?, revision = revision + 1 "
                "WHERE authority_key = ? AND capture_id = ? AND revision = ?",
                (now, self.authority_key, identity.capture_id, expected_revision),
            )
        return CaptureActionResult(identity, True, revision=expected_revision + 1)

    def _active_item_row(
        self,
        connection: sqlite3.Connection,
        identity: CaptureIdentity,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM collection_capture_items "
            "WHERE authority_key = ? AND capture_id = ? AND purge_state IS NULL",
            (self.authority_key, identity.capture_id),
        ).fetchone()
        if row is None:
            raise CollectionsCaptureError("capture_not_found")
        return row

    def _extraction_row(
        self,
        connection: sqlite3.Connection,
        identity: CaptureIdentity,
        *,
        expected_revision: int,
        states: set[str],
    ) -> sqlite3.Row:
        row = self._active_item_row(connection, identity)
        if int(row["revision"]) != expected_revision:
            current = self._get_detail(connection, identity)
            if current is None:
                raise CollectionsCaptureError("capture_not_found")
            raise CaptureConflictError(
                CaptureConflict(identity, expected_revision, current)
            )
        if str(row["processing_state"]) not in states:
            raise CollectionsCaptureError("invalid_extraction_state")
        return row

    def _written_detail(
        self,
        connection: sqlite3.Connection,
        identity: CaptureIdentity,
    ) -> CaptureDetail:
        detail = self._get_detail(connection, identity)
        if detail is None:
            raise CollectionsCaptureError("capture_update_failed")
        return detail

    def _get_detail(
        self,
        connection: sqlite3.Connection,
        identity: CaptureIdentity,
    ) -> CaptureDetail | None:
        row = connection.execute(
            "SELECT item.*, "
            "EXISTS(SELECT 1 FROM collection_capture_offline_files AS offline "
            "WHERE offline.authority_key = item.authority_key "
            "AND offline.capture_id = item.capture_id "
            "AND offline.state = 'ready') AS has_offline_copy "
            "FROM collection_capture_items AS item "
            "WHERE item.authority_key = ? AND item.capture_id = ? "
            "AND item.purge_state IS NULL",
            (self.authority_key, identity.capture_id),
        ).fetchone()
        if row is None:
            return None
        tags = self._tags_for_ids(connection, [identity.capture_id]).get(
            identity.capture_id, ()
        )
        offline_row = connection.execute(
            "SELECT * FROM collection_capture_offline_files "
            "WHERE authority_key = ? AND capture_id = ? AND state != 'purging' "
            "ORDER BY CASE state WHEN 'ready' THEN 0 ELSE 1 END, updated_at DESC, file_id DESC "
            "LIMIT 1",
            (self.authority_key, identity.capture_id),
        ).fetchone()
        offline_copy = None
        if offline_row is not None:
            offline_copy = CaptureOfflineCopy(
                identity,
                str(offline_row["file_id"]),
                str(offline_row["state"]),
                content_hash=offline_row["content_hash"],
                size=offline_row["actual_size"],
                media_type=offline_row["media_type"],
                failure_reason=offline_row["failure_reason"],
                revision=int(offline_row["revision"]),
            )
        media_reference = None
        if row["media_authority_key"] is not None and row["media_item_id"] is not None:
            media_reference = ExternalMediaReference(
                str(row["media_authority_key"]), str(row["media_item_id"])
            )
        return CaptureDetail(
            identity=identity,
            canonical_url=str(row["canonical_url"]),
            submitted_url=str(row["submitted_url"]),
            title=row["title"],
            domain=str(row["domain"]),
            summary=row["summary"],
            published_at=row["published_at"],
            status=str(row["status"]),
            favorite=bool(row["favorite"]),
            tags=tags,
            processing_state=str(row["processing_state"]),
            last_fetch_error=row["last_fetch_error"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            read_at=row["read_at"],
            revision=int(row["revision"]),
            has_offline_copy=bool(row["has_offline_copy"]),
            freeform_note=row["freeform_note"],
            text_content=row["text_content"],
            clean_html=row["clean_html"],
            byline=row["byline"],
            content_hash=row["content_hash"],
            word_count=row["word_count"],
            media_reference=media_reference,
            offline_copy=offline_copy,
        )

    def _summary_from_row(
        self,
        row: sqlite3.Row,
        tags: tuple[str, ...],
    ) -> CaptureSummary:
        return CaptureSummary(
            identity=CaptureIdentity(self.authority_key, str(row["capture_id"])),
            canonical_url=str(row["canonical_url"]),
            title=row["title"],
            domain=str(row["domain"]),
            summary=row["summary"],
            published_at=row["published_at"],
            status=str(row["status"]),
            favorite=bool(row["favorite"]),
            tags=tags,
            processing_state=str(row["processing_state"]),
            last_fetch_error=row["last_fetch_error"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            read_at=row["read_at"],
            revision=int(row["revision"]),
            has_offline_copy=bool(row["has_offline_copy"]),
        )

    def _tags_for_ids(
        self,
        connection: sqlite3.Connection,
        capture_ids: Sequence[str],
    ) -> dict[str, tuple[str, ...]]:
        if not capture_ids:
            return {}
        placeholders = ", ".join("?" for _ in capture_ids)
        tags_sql = (
            "SELECT item_tag.capture_id, tag.display_name "  # nosec B608
            "FROM collection_capture_item_tags AS item_tag "
            "JOIN collection_capture_tags AS tag "
            "ON tag.authority_key = item_tag.authority_key "
            "AND tag.tag_id = item_tag.tag_id "
            "WHERE item_tag.authority_key = ? "
            f"AND item_tag.capture_id IN ({placeholders}) "
            "ORDER BY item_tag.capture_id, tag.normalized_name"
        )
        rows = connection.execute(
            tags_sql,
            tuple([self.authority_key, *capture_ids]),
        ).fetchall()
        grouped: dict[str, list[str]] = {capture_id: [] for capture_id in capture_ids}
        for row in rows:
            grouped[str(row["capture_id"])].append(str(row["display_name"]))
        return {key: tuple(values) for key, values in grouped.items()}

    def _add_tags(
        self,
        connection: sqlite3.Connection,
        capture_id: str,
        tags: Sequence[str],
    ) -> None:
        for display in _normalized_tags(tuple(tags)):
            normalized = display.casefold()
            row = connection.execute(
                "SELECT tag_id FROM collection_capture_tags "
                "WHERE authority_key = ? AND normalized_name = ?",
                (self.authority_key, normalized),
            ).fetchone()
            if row is None:
                tag_id = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(tag_id), 0) + 1 "
                        "FROM collection_capture_tags WHERE authority_key = ?",
                        (self.authority_key,),
                    ).fetchone()[0]
                )
                connection.execute(
                    "INSERT INTO collection_capture_tags ("
                    "authority_key, tag_id, normalized_name, display_name"
                    ") VALUES (?, ?, ?, ?)",
                    (self.authority_key, tag_id, normalized, display),
                )
            else:
                tag_id = int(row["tag_id"])
            connection.execute(
                "INSERT OR IGNORE INTO collection_capture_item_tags ("
                "authority_key, capture_id, tag_id) VALUES (?, ?, ?)",
                (self.authority_key, capture_id, tag_id),
            )

    def _replace_tags(
        self,
        connection: sqlite3.Connection,
        capture_id: str,
        tags: Sequence[str],
    ) -> None:
        connection.execute(
            "DELETE FROM collection_capture_item_tags "
            "WHERE authority_key = ? AND capture_id = ?",
            (self.authority_key, capture_id),
        )
        self._add_tags(connection, capture_id, tags)

    def _saved_search_from_row(self, row: sqlite3.Row) -> SavedCaptureSearch:
        try:
            raw_request = json.loads(str(row["query_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CollectionsCaptureError("invalid_saved_search_request") from exc
        request = CapturePageRequest.from_mapping(raw_request)
        if request.authority_key != self.authority_key:
            raise CollectionsCaptureError("saved_search_authority_mismatch")
        return SavedCaptureSearch(
            self.authority_key,
            str(row["search_id"]),
            str(row["name"]),
            request,
            str(row["created_at"]),
            str(row["updated_at"]),
            int(row["revision"]),
        )

    @staticmethod
    def _request_json(request: CapturePageRequest) -> str:
        return json.dumps(asdict(request), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _word_count(text_content: str | None) -> int | None:
        if text_content is None:
            return None
        return len(text_content.split())

    @staticmethod
    def _inert_optional_extraction_text(value: Any, *, reason: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise CollectionsCaptureError(reason)
        return readable_body_text(value).strip() or None

    @staticmethod
    def _saved_search_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise CollectionsCaptureError("invalid_saved_search_name")
        return name.strip()

    @staticmethod
    def _opaque_id(value: str, reason: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise CollectionsCaptureError(reason)
        return value.strip()

    @staticmethod
    def _expected_revision(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CollectionsCaptureError("invalid_revision")
        return value

    @staticmethod
    def _highlight_from_row(
        identity: CaptureIdentity,
        row: sqlite3.Row,
    ) -> CaptureHighlight:
        return CaptureHighlight(
            identity,
            str(row["highlight_id"]),
            str(row["quote"]),
            row["note"],
            row["anchor_json"],
            bool(row["detached"]),
            str(row["created_at"]),
            str(row["updated_at"]),
            int(row["revision"]),
        )

    @staticmethod
    def _note_link_from_row(
        identity: CaptureIdentity,
        row: sqlite3.Row,
    ) -> CaptureNoteLink:
        return CaptureNoteLink(
            identity,
            str(row["link_id"]),
            ExternalNoteReference(
                str(row["note_authority_key"]), str(row["note_id"])
            ),
            str(row["created_at"]),
        )


__all__ = ["CollectionsCaptureRepository"]
