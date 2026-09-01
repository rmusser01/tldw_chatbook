"""Server Reading API adapter for source-neutral Collections captures."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from tldw_chatbook.tldw_api.exceptions import APIConnectionError, APIResponseError
from tldw_chatbook.tldw_api.media_reading_schemas import (
    ReadingHighlightCreateRequest,
    ReadingSaveRequest,
    ReadingSavedSearchCreateRequest,
    ReadingSavedSearchUpdateRequest,
    ReadingUpdateRequest,
)

from .collections_capture_models import (
    CAPTURE_CAPABILITY_NAMES,
    CAPTURE_PROCESSING_STATES,
    CAPTURE_STATUSES,
    CapabilityState,
    CaptureActionResult,
    CaptureAuthority,
    CaptureCapabilities,
    CaptureCapability,
    CaptureConflict,
    CaptureConflictError,
    CaptureContentResult,
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
    CollectionsCaptureError,
    ExternalMediaReference,
    ExternalNoteReference,
    SavedCaptureSearch,
)


SERVER_SORT = {
    "saved_desc": "created_desc",
    "saved_asc": "created_asc",
    "updated_desc": "updated_desc",
    "updated_asc": "updated_asc",
    "title_asc": "title_asc",
    "title_desc": "title_desc",
    "relevance": "relevance",
}

DocsInfoProvider = Callable[[], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    raise CollectionsCaptureError("invalid_server_response")


def _text(value: Any) -> str:
    if value is None:
        return ""
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _positive_id(value: Any, reason: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CollectionsCaptureError(reason) from exc
    if parsed < 1 or str(parsed) != str(value):
        raise CollectionsCaptureError(reason)
    return parsed


def _filter_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    raise CollectionsCaptureError("invalid_server_response")


class ServerCollectionsCaptureService:
    """Map one server principal's Reading API into capture contracts."""

    def __init__(
        self,
        authority: CaptureAuthority,
        client: Any,
        *,
        docs_info_provider: DocsInfoProvider,
        credential_fingerprint: str,
    ) -> None:
        if authority.kind != "server":
            raise CollectionsCaptureError("authority_mismatch")
        if not credential_fingerprint:
            raise CollectionsCaptureError("invalid_credential_fingerprint")
        self.authority = authority
        self.client = client
        self.docs_info_provider = docs_info_provider
        self._credential_fingerprint = credential_fingerprint
        self._capability_context: tuple[str, str, str, str] | None = None
        self._capability_overrides: dict[str, CaptureCapability] = {}
        self._saved_search_revisions: dict[str, int] = {}

    async def _docs_info(self) -> dict[str, Any]:
        result = self.docs_info_provider()
        if inspect.isawaitable(result):
            result = await result
        return _mapping(result)

    async def capabilities(self) -> CaptureCapabilities:
        try:
            docs = await self._docs_info()
        except Exception:
            return CaptureCapabilities(
                {
                    action: CaptureCapability(
                        CapabilityState.UNKNOWN,
                        "server_capability_discovery_failed",
                    )
                    for action in CAPTURE_CAPABILITY_NAMES
                }
            )
        capabilities = docs.get("capabilities")
        api_version = str(docs.get("api_version", "")).strip()
        capability_snapshot = json.dumps(
            capabilities,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        context = (
            self.authority.key,
            self._credential_fingerprint,
            api_version,
            capability_snapshot,
        )
        if context != self._capability_context:
            self._capability_context = context
            self._capability_overrides.clear()
        snapshot_pages = (
            isinstance(capabilities, Mapping)
            and capabilities.get("hasReadingSnapshotPagesV1") is True
        )
        api_v1 = api_version == "1"
        base_supported = {
            "browse",
            "capture",
            "update",
            "highlights",
            "linked_notes",
            "summarize",
            "archive",
            "hard_delete",
        }
        values: dict[str, CaptureCapability] = {}
        for action in CAPTURE_CAPABILITY_NAMES:
            if not api_v1:
                capability = CaptureCapability(
                    CapabilityState.UNSUPPORTED,
                    "server_reading_api_unavailable",
                )
            elif action == "browse" and not snapshot_pages:
                capability = CaptureCapability(
                    CapabilityState.UNSUPPORTED,
                    "server_page_snapshot_unavailable",
                )
            elif action in base_supported:
                capability = CaptureCapability(CapabilityState.SUPPORTED)
            else:
                capability = CaptureCapability(
                    CapabilityState.UNSUPPORTED,
                    {
                        "listen": "server_listener_configuration_unavailable",
                        "offline_copy": "server_offline_copy_unavailable",
                        "retry_extraction": "server_retry_extraction_unavailable",
                        "legacy_recovery": "server_legacy_recovery_unavailable",
                    }.get(action, "server_feature_unavailable"),
                )
            values[action] = self._capability_overrides.get(action, capability)
        return CaptureCapabilities(values)

    async def probe_capability(self, action: str) -> CaptureCapability:
        current = (await self.capabilities()).for_action(action)
        if current.state is not CapabilityState.SUPPORTED:
            return current
        probes = {
            "browse": lambda: self.client.list_reading_items(page=1, size=20),
            "highlights": lambda: self.client.list_reading_highlights(1),
            "linked_notes": lambda: self.client.list_reading_item_note_links(1),
        }
        probe = probes.get(action)
        if probe is None:
            return current
        try:
            await probe()
        except APIResponseError as exc:
            if exc.status_code != 404:
                raise CollectionsCaptureError(
                    "server_capability_probe_failed", retryable=True
                ) from exc
            current = CaptureCapability(
                CapabilityState.UNSUPPORTED,
                "server_feature_unavailable",
            )
            self._capability_overrides[action] = current
        except APIConnectionError as exc:
            raise CollectionsCaptureError(
                "server_capability_probe_failed", retryable=True
            ) from exc
        return current

    async def _require(self, action: str) -> None:
        capability = (await self.capabilities()).for_action(action)
        if capability.state is not CapabilityState.SUPPORTED:
            raise CollectionsCaptureError(capability.reason or "capability_unavailable")

    def _identity(self, capture_id: Any) -> CaptureIdentity:
        parsed = _positive_id(capture_id, "invalid_server_response")
        return CaptureIdentity(self.authority.key, str(parsed))

    def _validate_identity(self, identity: CaptureIdentity) -> int:
        if identity.authority_key != self.authority.key:
            raise CollectionsCaptureError("authority_mismatch")
        return _positive_id(identity.capture_id, "invalid_server_capture_id")

    def _capture(self, value: Any) -> CaptureDetail:
        item = _mapping(value)
        identity = self._identity(item.get("id"))
        canonical_url = item.get("canonical_url") or item.get("url")
        if not canonical_url:
            raise CollectionsCaptureError("invalid_server_response")
        status = str(item.get("status") or "saved").casefold()
        if status not in CAPTURE_STATUSES:
            raise CollectionsCaptureError("invalid_server_response")
        processing_state = str(item.get("processing_status") or "ready").casefold()
        processing_state = {
            "pending": "queued",
            "complete": "ready",
            "completed": "ready",
            "error": "failed",
        }.get(processing_state, processing_state)
        if processing_state not in CAPTURE_PROCESSING_STATES:
            raise CollectionsCaptureError("invalid_server_response")
        favorite = item.get("favorite", False)
        has_archive_copy = item.get("has_archive_copy", False)
        tags = item.get("tags") or ()
        if (
            not isinstance(favorite, bool)
            or not isinstance(has_archive_copy, bool)
            or not isinstance(tags, (tuple, list))
        ):
            raise CollectionsCaptureError("invalid_server_response")
        media_id = item.get("media_id") or item.get("media_uuid")
        media_reference = (
            ExternalMediaReference(self.authority.key, str(media_id))
            if media_id is not None
            else None
        )
        text_content = item.get("text")
        return CaptureDetail(
            identity=identity,
            canonical_url=str(canonical_url),
            title=item.get("title"),
            domain=str(item.get("domain") or ""),
            summary=item.get("summary"),
            published_at=item.get("published_at"),
            status=status,
            favorite=favorite,
            tags=tuple(tags),
            processing_state=processing_state,
            last_fetch_error=item.get("last_fetch_error"),
            created_at=_text(item.get("created_at")),
            updated_at=_text(item.get("updated_at")),
            read_at=item.get("read_at"),
            revision=item.get("revision") or 1,
            has_offline_copy=has_archive_copy,
            submitted_url=str(item.get("url") or canonical_url),
            freeform_note=item.get("notes"),
            text_content=text_content,
            clean_html=item.get("clean_html"),
            word_count=(
                len(text_content.split()) if isinstance(text_content, str) else None
            ),
            media_reference=media_reference,
        )

    async def list_page(self, request: CapturePageRequest) -> CapturePage:
        await self._require("browse")
        if request.authority_key != self.authority.key:
            raise CollectionsCaptureError("authority_mismatch")
        params = {
            key: value
            for key, value in {
                "status": list(request.statuses) or None,
                "tags": list(request.tags) or None,
                "q": request.search or None,
                "domain": request.domain,
                "favorite": request.favorite,
                "date_from": request.date_from,
                "date_to": request.date_to,
                "page": request.page,
                "size": request.size,
                "sort": SERVER_SORT[request.sort],
            }.items()
            if value is not None
        }
        response = _mapping(await self.client.list_reading_items(**params))
        items = tuple(self._capture(item) for item in response.get("items", ()))
        return CapturePage(
            request,
            items,
            response.get("total"),
            source_revision=response.get("source_revision"),
        )

    async def get_detail(self, identity: CaptureIdentity) -> CaptureDetail:
        await self._require("browse")
        return await self._get_detail(identity)

    async def _get_detail(self, identity: CaptureIdentity) -> CaptureDetail:
        item_id = self._validate_identity(identity)
        try:
            return self._capture(await self.client.get_reading_item(item_id))
        except APIResponseError as exc:
            reason = (
                "capture_not_found"
                if exc.status_code == 404
                else "server_detail_failed"
            )
            raise CollectionsCaptureError(
                reason, retryable=exc.status_code >= 500
            ) from exc

    async def save_capture(self, request: CaptureSaveRequest) -> CaptureSaveOutcome:
        await self._require("capture")
        if request.authority_key != self.authority.key:
            raise CollectionsCaptureError("authority_mismatch")
        payload = ReadingSaveRequest(
            url=request.submitted_url,
            title=request.title,
            tags=list(request.tags),
            status=request.status or "saved",
            favorite=request.favorite or False,
            summary=request.summary,
            notes=request.freeform_note,
            content=request.text_content or request.clean_html,
        )
        try:
            capture = self._capture(await self.client.save_reading_item(payload))
        except APIConnectionError:
            return CaptureSaveOutcome(None, None, outcome_unknown=True)
        except APIResponseError as exc:
            raise CollectionsCaptureError("server_save_rejected") from exc
        return CaptureSaveOutcome(
            capture,
            True,
            extraction_pending=capture.processing_state in {"queued", "processing"},
        )

    async def update_capture(
        self,
        identity: CaptureIdentity,
        expected_revision: int,
        changes: Mapping[str, Any],
    ) -> CaptureDetail:
        await self._require("update")
        item_id = self._validate_identity(identity)
        current = await self._get_detail(identity)
        if current.revision != expected_revision:
            raise CaptureConflictError(
                CaptureConflict(identity, expected_revision, current)
            )
        allowed = {"status", "favorite", "tags", "freeform_note", "title"}
        if set(changes) - allowed:
            raise CollectionsCaptureError("unsupported_capture_change")
        payload = ReadingUpdateRequest(
            status=changes.get("status"),
            favorite=changes.get("favorite"),
            tags=(list(changes["tags"]) if "tags" in changes else None),
            notes=changes.get("freeform_note"),
            title=changes.get("title"),
        )
        try:
            return self._capture(
                await self.client.update_reading_item(item_id, payload)
            )
        except APIResponseError as exc:
            raise CollectionsCaptureError("server_update_rejected") from exc

    @staticmethod
    def _server_query(request: CapturePageRequest) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "status": list(request.statuses) or None,
                "tags": list(request.tags) or None,
                "favorite": request.favorite,
                "q": request.search or None,
                "domain": request.domain,
                "date_from": request.date_from,
                "date_to": request.date_to,
            }.items()
            if value is not None
        }

    def _saved_search(self, value: Any) -> SavedCaptureSearch:
        item = _mapping(value)
        query = dict(item.get("query") or {})
        reverse_sort = {server: source for source, server in SERVER_SORT.items()}
        saved_sort = item.get("sort")
        request = CapturePageRequest.from_mapping(
            {
                "authority_key": self.authority.key,
                "search": query.get("q", ""),
                "statuses": _filter_values(query.get("status")),
                "favorite": query.get("favorite"),
                "tags": _filter_values(query.get("tags")),
                "domain": query.get("domain"),
                "date_from": query.get("date_from"),
                "date_to": query.get("date_to"),
                "sort": reverse_sort.get(
                    saved_sort if isinstance(saved_sort, str) else "",
                    "saved_desc",
                ),
            }
        )
        return SavedCaptureSearch(
            self.authority.key,
            str(item.get("id")),
            str(item.get("name")),
            request,
            _text(item.get("created_at")),
            _text(item.get("updated_at")),
            item.get("revision") or 1,
        )

    async def list_saved_searches(
        self,
        page: int,
        size: int = 20,
    ) -> CaptureSavedSearchPage:
        await self._require("browse")
        if size != 20:
            raise CollectionsCaptureError("invalid_page_size")
        response = _mapping(
            await self.client.list_reading_saved_searches(
                limit=size,
                offset=(page - 1) * size,
            )
        )
        items = tuple(self._saved_search(item) for item in response.get("items", ()))
        self._saved_search_revisions.update(
            (item.search_id, item.revision) for item in items
        )
        return CaptureSavedSearchPage(items, response.get("total"), page, size)

    async def save_saved_search(self, search: SavedCaptureSearch) -> SavedCaptureSearch:
        await self._require("browse")
        if search.authority_key != self.authority.key:
            raise CollectionsCaptureError("authority_mismatch")
        query = self._server_query(search.request)
        if search.search_id == "new":
            payload: Any = ReadingSavedSearchCreateRequest(
                name=search.name,
                query=query,
                sort=SERVER_SORT[search.request.sort],
            )
            result = await self.client.create_reading_saved_search(payload)
        else:
            search_id = _positive_id(search.search_id, "invalid_server_search_id")
            payload = ReadingSavedSearchUpdateRequest(
                name=search.name,
                query=query,
                sort=SERVER_SORT[search.request.sort],
            )
            result = await self.client.update_reading_saved_search(search_id, payload)
        saved = self._saved_search(result)
        self._saved_search_revisions[saved.search_id] = saved.revision
        return saved

    async def delete_saved_search(self, search_id: str) -> CaptureActionResult:
        await self._require("browse")
        parsed = _positive_id(search_id, "invalid_server_search_id")
        await self.client.delete_reading_saved_search(parsed)
        self._saved_search_revisions.pop(search_id, None)
        return CaptureActionResult(CaptureIdentity(self.authority.key, search_id), True)

    def _highlight(self, identity: CaptureIdentity, value: Any) -> CaptureHighlight:
        item = _mapping(value)
        anchor = {
            key: item[key]
            for key in ("start_offset", "end_offset", "anchor_strategy")
            if item.get(key) is not None
        }
        created = _text(item.get("created_at"))
        return CaptureHighlight(
            identity,
            str(item.get("id")),
            str(item.get("quote")),
            item.get("note"),
            json.dumps(anchor, sort_keys=True) if anchor else None,
            item.get("state") == "stale",
            created,
            _text(item.get("updated_at")) or created,
            item.get("revision") or 1,
        )

    async def list_highlights(
        self,
        identity: CaptureIdentity,
    ) -> tuple[CaptureHighlight, ...]:
        await self._require("highlights")
        item_id = self._validate_identity(identity)
        return tuple(
            self._highlight(identity, item)
            for item in await self.client.list_reading_highlights(item_id)
        )

    async def save_highlight(
        self,
        identity: CaptureIdentity,
        draft: CaptureHighlightDraft,
    ) -> CaptureHighlight:
        await self._require("highlights")
        item_id = self._validate_identity(identity)
        payload = ReadingHighlightCreateRequest(
            item_id=item_id,
            quote=draft.quote,
            note=draft.note,
        )
        return self._highlight(
            identity,
            await self.client.create_reading_highlight(item_id, payload),
        )

    async def delete_highlight(
        self,
        identity: CaptureIdentity,
        highlight_id: str,
    ) -> CaptureActionResult:
        await self._require("highlights")
        self._validate_identity(identity)
        parsed = _positive_id(highlight_id, "invalid_server_highlight_id")
        await self.client.delete_reading_highlight(parsed)
        return CaptureActionResult(identity, True)

    def _note_link(self, identity: CaptureIdentity, value: Any) -> CaptureNoteLink:
        item = _mapping(value)
        note_id = str(item.get("note_id"))
        return CaptureNoteLink(
            identity,
            note_id,
            ExternalNoteReference(self.authority.key, note_id),
            _text(item.get("created_at")),
        )

    async def list_note_links(
        self,
        identity: CaptureIdentity,
    ) -> tuple[CaptureNoteLink, ...]:
        await self._require("linked_notes")
        item_id = self._validate_identity(identity)
        response = _mapping(await self.client.list_reading_item_note_links(item_id))
        return tuple(
            self._note_link(identity, link) for link in response.get("links", ())
        )

    async def link_note(
        self,
        identity: CaptureIdentity,
        note: ExternalNoteReference,
    ) -> CaptureNoteLink:
        await self._require("linked_notes")
        item_id = self._validate_identity(identity)
        if note.authority_key != self.authority.key:
            raise CollectionsCaptureError("server_note_authority_mismatch")
        return self._note_link(
            identity,
            await self.client.link_note_to_reading_item(item_id, note.note_id),
        )

    async def unlink_note(
        self,
        identity: CaptureIdentity,
        link_id: str,
    ) -> CaptureActionResult:
        await self._require("linked_notes")
        item_id = self._validate_identity(identity)
        await self.client.unlink_note_from_reading_item(item_id, link_id)
        return CaptureActionResult(identity, True)

    async def retry_extraction(self, identity: CaptureIdentity) -> CaptureActionResult:
        await self._require("retry_extraction")
        raise AssertionError("unreachable")

    async def save_offline_copy(self, identity: CaptureIdentity) -> CaptureOfflineCopy:
        await self._require("offline_copy")
        raise AssertionError("unreachable")

    async def delete_offline_copy(
        self, identity: CaptureIdentity
    ) -> CaptureActionResult:
        await self._require("offline_copy")
        raise AssertionError("unreachable")

    async def summarize(self, identity: CaptureIdentity) -> CaptureContentResult:
        await self._require("summarize")
        item_id = self._validate_identity(identity)
        result = _mapping(await self.client.summarize_reading_item(item_id))
        return CaptureContentResult(identity, "summary", text=result.get("summary"))

    async def listen(self, identity: CaptureIdentity) -> CaptureContentResult:
        await self._require("listen")
        raise AssertionError("unreachable")

    async def hard_delete(
        self,
        identity: CaptureIdentity,
        expected_revision: int,
    ) -> CaptureActionResult:
        await self._require("hard_delete")
        item_id = self._validate_identity(identity)
        current = await self._get_detail(identity)
        if current.revision != expected_revision:
            raise CaptureConflictError(
                CaptureConflict(identity, expected_revision, current)
            )
        await self.client.delete_reading_item(item_id, hard=True)
        return CaptureActionResult(identity, True)


__all__ = ["SERVER_SORT", "ServerCollectionsCaptureService"]
