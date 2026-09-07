"""Source-neutral authority seam for Local and Server Collections captures."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import secrets
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, Protocol

from .collections_capture_models import (
    CAPTURE_ANNOTATION_PAGE_SIZE,
    CAPTURE_CAPABILITY_NAMES,
    CapabilityState,
    CaptureActionResult,
    CaptureAuthority,
    CaptureCapabilities,
    CaptureCapability,
    CaptureContentResult,
    CaptureDetail,
    CaptureHighlight,
    CaptureHighlightDraft,
    CaptureHighlightPage,
    CaptureIdentity,
    CaptureNoteLink,
    CaptureNoteLinkPage,
    CaptureOfflineCopy,
    CapturePage,
    CapturePageRequest,
    CaptureSaveOutcome,
    CaptureSaveRequest,
    CaptureSavedSearchPage,
    CollectionsCaptureError,
    ExternalNoteReference,
    ExternalReferenceAvailability,
    ResolvedCaptureDetail,
    SavedCaptureSearch,
)
from .library_content_evidence import LibraryContentEvidence
from .collections_capture_repository import CollectionsCaptureRepository
from .collections_offline_store import CollectionsOfflineStore


ReferenceResolver = Callable[
    [Any],
    ExternalReferenceAvailability | Awaitable[ExternalReferenceAvailability],
]


def _authority(
    kind: Literal["local", "server"],
    *private_parts: object,
) -> CaptureAuthority:
    digest = hashlib.sha256(
        "\0".join(str(part) for part in private_parts).encode("utf-8")
    ).hexdigest()
    return CaptureAuthority(kind, f"{kind}:{digest[:32]}", digest[:16])


def build_local_capture_authority(
    profile_id: str,
    database_identity: object,
) -> CaptureAuthority:
    """Build one opaque Local profile/database authority."""
    return _authority("local", profile_id, database_identity)


def build_server_capture_authority(
    profile_id: str,
    principal_id: str,
) -> CaptureAuthority:
    """Build one opaque Server profile/principal authority without workspace."""
    return _authority("server", profile_id, principal_id)


class CollectionsCaptureBackend(Protocol):
    """Operations required by the Collections reader."""

    async def list_page(self, request: CapturePageRequest) -> CapturePage: ...
    async def get_detail(self, identity: CaptureIdentity) -> CaptureDetail: ...
    async def save_capture(self, request: CaptureSaveRequest) -> CaptureSaveOutcome: ...
    async def update_capture(
        self,
        identity: CaptureIdentity,
        expected_revision: int,
        changes: Mapping[str, Any],
    ) -> CaptureDetail: ...
    async def retry_extraction(
        self, identity: CaptureIdentity
    ) -> CaptureActionResult: ...
    async def list_saved_searches(
        self, page: int, size: int = 20
    ) -> CaptureSavedSearchPage: ...
    async def save_saved_search(
        self, search: SavedCaptureSearch
    ) -> SavedCaptureSearch: ...
    async def delete_saved_search(self, search_id: str) -> CaptureActionResult: ...
    async def list_highlights(
        self,
        identity: CaptureIdentity,
        *,
        page: int = 1,
        size: int = CAPTURE_ANNOTATION_PAGE_SIZE,
    ) -> CaptureHighlightPage: ...
    async def save_highlight(
        self,
        identity: CaptureIdentity,
        draft: CaptureHighlightDraft,
    ) -> CaptureHighlight: ...
    async def delete_highlight(
        self,
        identity: CaptureIdentity,
        highlight_id: str,
    ) -> CaptureActionResult: ...
    async def list_note_links(
        self,
        identity: CaptureIdentity,
        *,
        page: int = 1,
        size: int = CAPTURE_ANNOTATION_PAGE_SIZE,
    ) -> CaptureNoteLinkPage: ...
    async def link_note(
        self,
        identity: CaptureIdentity,
        note: ExternalNoteReference,
    ) -> CaptureNoteLink: ...
    async def unlink_note(
        self,
        identity: CaptureIdentity,
        link_id: str,
    ) -> CaptureActionResult: ...
    async def save_offline_copy(
        self, identity: CaptureIdentity
    ) -> CaptureOfflineCopy: ...
    async def delete_offline_copy(
        self, identity: CaptureIdentity
    ) -> CaptureActionResult: ...
    async def summarize(self, identity: CaptureIdentity) -> CaptureContentResult: ...
    async def listen(self, identity: CaptureIdentity) -> CaptureContentResult: ...
    async def hard_delete(
        self,
        identity: CaptureIdentity,
        expected_revision: int,
    ) -> CaptureActionResult: ...
    async def capabilities(self) -> CaptureCapabilities: ...
    async def probe_capability(self, action: str) -> CaptureCapability: ...


def _capabilities(
    supported: set[str],
    reasons: Mapping[str, str],
) -> CaptureCapabilities:
    return CaptureCapabilities(
        {
            action: CaptureCapability(
                CapabilityState.SUPPORTED
                if action in supported
                else CapabilityState.UNSUPPORTED,
                None if action in supported else reasons[action],
            )
            for action in CAPTURE_CAPABILITY_NAMES
        }
    )


class LocalCollectionsCaptureService:
    """Async adapter over one synchronous Local capture repository."""

    def __init__(
        self,
        authority: CaptureAuthority,
        repository: CollectionsCaptureRepository,
        *,
        offline_store: CollectionsOfflineStore | None = None,
        extractor: Callable[[str], Mapping[str, Any]] | None = None,
        summarizer: Callable[[CaptureDetail], str] | None = None,
        listener: Callable[[CaptureDetail], str] | None = None,
        heartbeat_interval: float = 30.0,
        legacy_recovery_available: bool = False,
    ) -> None:
        if authority.kind != "local" or repository.authority_key != authority.key:
            raise CollectionsCaptureError("authority_mismatch")
        if heartbeat_interval <= 0:
            raise CollectionsCaptureError("invalid_extraction_heartbeat")
        self.authority = authority
        self.repository = repository
        self.offline_store = offline_store
        self.extractor = extractor
        self.summarizer = summarizer
        self.listener = listener
        self.heartbeat_interval = heartbeat_interval
        self.legacy_recovery_available = legacy_recovery_available
        self._extraction_tasks: set[asyncio.Task[None]] = set()
        self._saved_search_revisions: dict[str, int] = {}
        self._highlight_revisions: dict[tuple[str, str], int] = {}

    async def _call(
        self, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if self.repository.db.is_memory_db:
            return function(*args, **kwargs)
        return await asyncio.to_thread(function, *args, **kwargs)

    async def capabilities(self) -> CaptureCapabilities:
        supported = {
            "browse",
            "capture",
            "update",
            "highlights",
            "linked_notes",
            "archive",
            "hard_delete",
        }
        if self.extractor is not None:
            supported.add("retry_extraction")
        if self.offline_store is not None:
            supported.add("offline_copy")
        if self.summarizer is not None:
            supported.add("summarize")
        if self.listener is not None:
            supported.add("listen")
        if self.legacy_recovery_available:
            supported.add("legacy_recovery")
        reasons = {
            action: {
                "offline_copy": "local_offline_store_unavailable",
                "summarize": "local_summarizer_unavailable",
                "listen": "local_listener_unavailable",
                "retry_extraction": "local_extractor_unavailable",
                "legacy_recovery": "legacy_recovery_unavailable",
            }.get(action, "local_feature_unavailable")
            for action in CAPTURE_CAPABILITY_NAMES
        }
        return _capabilities(supported, reasons)

    async def probe_capability(self, action: str) -> CaptureCapability:
        return (await self.capabilities()).for_action(action)

    async def _require(self, action: str) -> None:
        capability = (await self.capabilities()).for_action(action)
        if capability.state is not CapabilityState.SUPPORTED:
            raise CollectionsCaptureError(capability.reason or "capability_unavailable")

    async def list_page(self, request: CapturePageRequest) -> CapturePage:
        await self._require("browse")
        return await self._call(self.repository.list_page, request)

    async def get_detail(self, identity: CaptureIdentity) -> CaptureDetail:
        await self._require("browse")
        detail = await self._call(self.repository.get_detail, identity)
        if detail is None:
            raise CollectionsCaptureError("capture_not_found")
        return detail

    async def save_capture(self, request: CaptureSaveRequest) -> CaptureSaveOutcome:
        await self._require("capture")
        outcome = await self._call(self.repository.save_capture, request)
        if (
            self.extractor is not None
            and outcome.capture is not None
            and outcome.extraction_pending
        ):
            self._start_extraction(outcome.capture)
        return outcome

    async def update_capture(
        self,
        identity: CaptureIdentity,
        expected_revision: int,
        changes: Mapping[str, Any],
    ) -> CaptureDetail:
        await self._require("update")
        return await self._call(
            self.repository.update_capture,
            identity,
            expected_revision=expected_revision,
            changes=changes,
        )

    def _start_extraction(self, detail: CaptureDetail) -> None:
        task = asyncio.create_task(self._run_extraction(detail))
        self._extraction_tasks.add(task)
        task.add_done_callback(self._extraction_tasks.discard)

    async def _run_extraction(self, queued: CaptureDetail) -> None:
        if self.extractor is None:
            return
        owner = secrets.token_hex(16)
        claimed = await self._call(
            self.repository.claim_extraction,
            queued.identity,
            owner_token=owner,
        )
        stopped = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat(claimed.identity, owner, stopped)
        )
        try:
            result = await asyncio.to_thread(self.extractor, claimed.submitted_url)
            stopped.set()
            await heartbeat
            await self._call(
                self.repository.complete_extraction,
                claimed.identity,
                owner_token=owner,
                result=result,
            )
        except asyncio.CancelledError:
            stopped.set()
            if not heartbeat.done():
                await heartbeat
            try:
                await self._call(
                    self.repository.fail_extraction,
                    claimed.identity,
                    owner_token=owner,
                    reason="interrupted",
                )
            except CollectionsCaptureError:
                pass
            raise
        except Exception:
            stopped.set()
            if not heartbeat.done():
                await heartbeat
            try:
                await self._call(
                    self.repository.fail_extraction,
                    claimed.identity,
                    owner_token=owner,
                    reason="unknown",
                )
            except CollectionsCaptureError:
                pass

    async def _heartbeat(
        self,
        identity: CaptureIdentity,
        owner: str,
        stopped: asyncio.Event,
    ) -> None:
        while not stopped.is_set():
            try:
                await asyncio.wait_for(stopped.wait(), timeout=self.heartbeat_interval)
            except TimeoutError:
                await self._call(
                    self.repository.renew_extraction_lease,
                    identity,
                    owner_token=owner,
                )

    async def drain_extractions(self) -> None:
        """Wait for currently scheduled extraction work (test/shutdown seam)."""
        if self._extraction_tasks:
            await asyncio.gather(*tuple(self._extraction_tasks))

    async def cancel_extractions(self) -> None:
        """Cancel and settle every extraction owned by this app lifetime."""
        tasks = tuple(self._extraction_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def retry_extraction(self, identity: CaptureIdentity) -> CaptureActionResult:
        await self._require("retry_extraction")
        if self.extractor is None:
            raise CollectionsCaptureError("local_extractor_unavailable")
        detail = await self.get_detail(identity)
        queued = await self._call(
            self.repository.retry_extraction,
            identity,
            expected_revision=detail.revision,
        )
        self._start_extraction(queued)
        return CaptureActionResult(identity, True, revision=queued.revision)

    async def list_saved_searches(
        self,
        page: int,
        size: int = 20,
    ) -> CaptureSavedSearchPage:
        result = await self._call(
            self.repository.list_saved_searches,
            page=page,
            size=size,
        )
        self._saved_search_revisions.update(
            (item.search_id, item.revision) for item in result.items
        )
        return result

    async def save_saved_search(self, search: SavedCaptureSearch) -> SavedCaptureSearch:
        if search.authority_key != self.authority.key:
            raise CollectionsCaptureError("authority_mismatch")
        if search.search_id == "new":
            result = await self._call(
                self.repository.create_saved_search,
                search.name,
                search.request,
            )
        else:
            result = await self._call(
                self.repository.update_saved_search,
                search.search_id,
                name=search.name,
                request=search.request,
                expected_revision=search.revision,
            )
        self._saved_search_revisions[result.search_id] = result.revision
        return result

    async def delete_saved_search(self, search_id: str) -> CaptureActionResult:
        revision = self._saved_search_revisions.get(search_id)
        if revision is None:
            raise CollectionsCaptureError("stale_saved_search")
        result = await self._call(
            self.repository.delete_saved_search,
            search_id,
            expected_revision=revision,
        )
        self._saved_search_revisions.pop(search_id, None)
        return result

    async def list_highlights(
        self,
        identity: CaptureIdentity,
        *,
        page: int = 1,
        size: int = CAPTURE_ANNOTATION_PAGE_SIZE,
    ) -> CaptureHighlightPage:
        await self._require("highlights")
        result = await self._call(
            self.repository.list_highlights,
            identity,
            page=page,
            size=size,
        )
        self._highlight_revisions.update(
            ((identity.capture_id, item.highlight_id), item.revision)
            for item in result.items
        )
        return result

    async def save_highlight(
        self,
        identity: CaptureIdentity,
        draft: CaptureHighlightDraft,
    ) -> CaptureHighlight:
        await self._require("highlights")
        result = await self._call(self.repository.save_highlight, identity, draft)
        self._highlight_revisions[(identity.capture_id, result.highlight_id)] = (
            result.revision
        )
        return result

    async def delete_highlight(
        self,
        identity: CaptureIdentity,
        highlight_id: str,
    ) -> CaptureActionResult:
        await self._require("highlights")
        revision = self._highlight_revisions.get((identity.capture_id, highlight_id))
        if revision is None:
            raise CollectionsCaptureError("stale_highlight")
        result = await self._call(
            self.repository.delete_highlight,
            identity,
            highlight_id,
            expected_revision=revision,
        )
        self._highlight_revisions.pop((identity.capture_id, highlight_id), None)
        return result

    async def list_note_links(
        self,
        identity: CaptureIdentity,
        *,
        page: int = 1,
        size: int = CAPTURE_ANNOTATION_PAGE_SIZE,
    ) -> CaptureNoteLinkPage:
        await self._require("linked_notes")
        return await self._call(
            self.repository.list_note_links,
            identity,
            page=page,
            size=size,
        )

    async def link_note(
        self,
        identity: CaptureIdentity,
        note: ExternalNoteReference,
    ) -> CaptureNoteLink:
        await self._require("linked_notes")
        return await self._call(self.repository.link_note, identity, note)

    async def unlink_note(
        self,
        identity: CaptureIdentity,
        link_id: str,
    ) -> CaptureActionResult:
        await self._require("linked_notes")
        return await self._call(self.repository.unlink_note, identity, link_id)

    async def save_offline_copy(self, identity: CaptureIdentity) -> CaptureOfflineCopy:
        await self._require("offline_copy")
        detail = await self.get_detail(identity)
        content = detail.clean_html or detail.text_content
        if content is None:
            raise CollectionsCaptureError("offline_content_unavailable")
        assert self.offline_store is not None
        media_type = "text/html" if detail.clean_html else "text/plain"
        return await self._call(
            self.offline_store.save_copy,
            identity,
            content.encode("utf-8"),
            media_type,
        )

    async def delete_offline_copy(
        self, identity: CaptureIdentity
    ) -> CaptureActionResult:
        await self._require("offline_copy")
        assert self.offline_store is not None
        return await self._call(self.offline_store.delete_copy, identity)

    async def summarize(self, identity: CaptureIdentity) -> CaptureContentResult:
        await self._require("summarize")
        assert self.summarizer is not None
        detail = await self.get_detail(identity)
        return CaptureContentResult(
            identity, "summary", text=await self._call(self.summarizer, detail)
        )

    async def listen(self, identity: CaptureIdentity) -> CaptureContentResult:
        await self._require("listen")
        assert self.listener is not None
        detail = await self.get_detail(identity)
        return CaptureContentResult(
            identity,
            "audio",
            artifact_reference=await self._call(self.listener, detail),
        )

    async def hard_delete(
        self,
        identity: CaptureIdentity,
        expected_revision: int,
    ) -> CaptureActionResult:
        await self._require("hard_delete")
        if self.offline_store is not None:
            return await self._call(
                self.offline_store.hard_delete,
                identity,
                expected_revision=expected_revision,
            )
        return await self._call(
            self.repository.hard_delete,
            identity,
            expected_revision=expected_revision,
        )


class CollectionsCaptureScopeService:
    """Fence one active capture authority and retain only its snapshots."""

    def __init__(
        self,
        *,
        resolve_media_reference: ReferenceResolver | None = None,
        resolve_note_reference: ReferenceResolver | None = None,
    ) -> None:
        self._resolve_media_reference = resolve_media_reference
        self._resolve_note_reference = resolve_note_reference
        self.active_authority: CaptureAuthority | None = None
        self._backend: CollectionsCaptureBackend | None = None
        self._generation = 0
        self.page_snapshot: CapturePage | None = None
        self.detail_snapshot: ResolvedCaptureDetail | None = None
        self.saved_search_snapshot: CaptureSavedSearchPage | None = None
        self._archive_status: dict[tuple[str, str], str] = {}

    def activate(
        self,
        authority: CaptureAuthority,
        backend: CollectionsCaptureBackend,
    ) -> None:
        if not isinstance(authority, CaptureAuthority):
            raise CollectionsCaptureError("invalid_authority")
        backend_authority = getattr(backend, "authority", None)
        if backend_authority is not None and backend_authority != authority:
            raise CollectionsCaptureError("authority_mismatch")
        self.active_authority = authority
        self._backend = backend
        self._generation += 1
        self.page_snapshot = None
        self.detail_snapshot = None
        self.saved_search_snapshot = None

    def deactivate(self) -> None:
        """Fence the current authority and discard every owner-bound snapshot."""
        self.active_authority = None
        self._backend = None
        self._generation += 1
        self.page_snapshot = None
        self.detail_snapshot = None
        self.saved_search_snapshot = None
        self._archive_status.clear()

    def _claim(self) -> tuple[int, str, CollectionsCaptureBackend]:
        if self.active_authority is None or self._backend is None:
            raise CollectionsCaptureError("capture_authority_unavailable")
        return self._generation, self.active_authority.key, self._backend

    def _finish(self, generation: int, authority_key: str) -> None:
        if (
            self.active_authority is None
            or generation != self._generation
            or authority_key != self.active_authority.key
        ):
            raise CollectionsCaptureError("stale_authority_result", retryable=True)

    async def _invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        generation, authority_key, backend = self._claim()
        result = await getattr(backend, method)(*args, **kwargs)
        self._finish(generation, authority_key)
        return result

    async def list_page(self, request: CapturePageRequest) -> CapturePage:
        result = await self._invoke("list_page", request)
        self.page_snapshot = result
        return result

    async def get_library_user_content_evidence(self) -> LibraryContentEvidence:
        """Return bounded evidence from the active capture authority."""
        authority = self.active_authority
        if authority is None or self._backend is None:
            return LibraryContentEvidence.UNKNOWN
        try:
            page = await self._invoke(
                "list_page",
                CapturePageRequest(authority.key, page=1),
            )
        except CollectionsCaptureError:
            return LibraryContentEvidence.UNKNOWN
        return (
            LibraryContentEvidence.HAS_USER_CONTENT
            if page.total
            else LibraryContentEvidence.EMPTY
        )

    async def get_detail(self, identity: CaptureIdentity) -> ResolvedCaptureDetail:
        generation, authority_key, backend = self._claim()
        detail = await backend.get_detail(identity)
        media = None
        if detail.media_reference is not None:
            media = await self._resolve(
                self._resolve_media_reference,
                detail.media_reference,
                "media_reference_unavailable",
            )
        links = (await backend.list_note_links(identity)).items
        resolved_links = []
        for link in links:
            availability = await self._resolve(
                self._resolve_note_reference,
                link.note_reference,
                "note_reference_unavailable",
            )
            resolved_links.append((link, availability))
        result = ResolvedCaptureDetail(detail, media, resolved_links)
        self._finish(generation, authority_key)
        self.detail_snapshot = result
        return result

    @staticmethod
    async def _resolve(
        resolver: ReferenceResolver | None,
        reference: Any,
        unavailable_reason: str,
    ) -> ExternalReferenceAvailability:
        if resolver is None:
            return ExternalReferenceAvailability("unavailable", unavailable_reason)
        try:
            result = resolver(reference)
            if inspect.isawaitable(result):
                result = await result
        except CollectionsCaptureError as exc:
            reason = "reference_resolution_retryable" if exc.retryable else exc.reason
            return ExternalReferenceAvailability("unavailable", reason)
        except Exception:
            return ExternalReferenceAvailability(
                "unavailable",
                "reference_resolution_retryable",
            )
        if not isinstance(result, ExternalReferenceAvailability):
            raise CollectionsCaptureError("invalid_reference_availability")
        return result

    async def save_capture(self, request: CaptureSaveRequest) -> CaptureSaveOutcome:
        return await self._invoke("save_capture", request)

    async def update_capture(
        self,
        identity: CaptureIdentity,
        expected_revision: int,
        changes: Mapping[str, Any],
    ) -> CaptureDetail:
        return await self._invoke(
            "update_capture",
            identity,
            expected_revision,
            changes,
        )

    async def archive(
        self,
        identity: CaptureIdentity,
        expected_revision: int,
    ) -> CaptureDetail:
        generation, authority_key, backend = self._claim()
        current = await backend.get_detail(identity)
        changed = await backend.update_capture(
            identity,
            expected_revision,
            {"status": "archived"},
        )
        self._finish(generation, authority_key)
        self._archive_status[(identity.authority_key, identity.capture_id)] = (
            current.status
        )
        return changed

    async def undo_archive(
        self,
        identity: CaptureIdentity,
        expected_revision: int,
    ) -> CaptureDetail:
        key = (identity.authority_key, identity.capture_id)
        previous = self._archive_status.get(key)
        if previous is None:
            raise CollectionsCaptureError("archive_receipt_unavailable")
        changed = await self.update_capture(
            identity,
            expected_revision,
            {"status": previous},
        )
        self._archive_status.pop(key, None)
        return changed

    async def list_saved_searches(
        self, page: int, size: int = 20
    ) -> CaptureSavedSearchPage:
        result = await self._invoke("list_saved_searches", page, size)
        self.saved_search_snapshot = result
        return result

    async def save_saved_search(self, search: SavedCaptureSearch) -> SavedCaptureSearch:
        return await self._invoke("save_saved_search", search)

    async def delete_saved_search(self, search_id: str) -> CaptureActionResult:
        return await self._invoke("delete_saved_search", search_id)

    async def list_highlights(
        self,
        identity: CaptureIdentity,
        *,
        page: int = 1,
        size: int = CAPTURE_ANNOTATION_PAGE_SIZE,
    ) -> CaptureHighlightPage:
        return await self._invoke(
            "list_highlights",
            identity,
            page=page,
            size=size,
        )

    async def save_highlight(
        self,
        identity: CaptureIdentity,
        *,
        quote: str,
        note: str | None = None,
        anchor_json: str | None = None,
    ) -> CaptureHighlight:
        return await self._invoke(
            "save_highlight",
            identity,
            CaptureHighlightDraft(quote, note, anchor_json),
        )

    async def delete_highlight(
        self,
        identity: CaptureIdentity,
        highlight_id: str,
    ) -> CaptureActionResult:
        return await self._invoke("delete_highlight", identity, highlight_id)

    async def list_note_links(
        self,
        identity: CaptureIdentity,
        *,
        page: int = 1,
        size: int = CAPTURE_ANNOTATION_PAGE_SIZE,
    ) -> CaptureNoteLinkPage:
        return await self._invoke(
            "list_note_links",
            identity,
            page=page,
            size=size,
        )

    async def link_note(
        self,
        identity: CaptureIdentity,
        note: ExternalNoteReference,
    ) -> CaptureNoteLink:
        return await self._invoke("link_note", identity, note)

    async def unlink_note(
        self,
        identity: CaptureIdentity,
        link_id: str,
    ) -> CaptureActionResult:
        return await self._invoke("unlink_note", identity, link_id)

    async def capabilities(self) -> CaptureCapabilities:
        return await self._invoke("capabilities")

    async def probe_capability(self, action: str) -> CaptureCapability:
        return await self._invoke("probe_capability", action)

    async def retry_extraction(self, identity: CaptureIdentity) -> CaptureActionResult:
        return await self._invoke("retry_extraction", identity)

    async def save_offline_copy(self, identity: CaptureIdentity) -> CaptureOfflineCopy:
        return await self._invoke("save_offline_copy", identity)

    async def delete_offline_copy(
        self, identity: CaptureIdentity
    ) -> CaptureActionResult:
        return await self._invoke("delete_offline_copy", identity)

    async def summarize(self, identity: CaptureIdentity) -> CaptureContentResult:
        return await self._invoke("summarize", identity)

    async def listen(self, identity: CaptureIdentity) -> CaptureContentResult:
        return await self._invoke("listen", identity)

    async def hard_delete(
        self,
        identity: CaptureIdentity,
        expected_revision: int,
    ) -> CaptureActionResult:
        return await self._invoke("hard_delete", identity, expected_revision)


__all__ = [
    "CollectionsCaptureBackend",
    "CollectionsCaptureScopeService",
    "LocalCollectionsCaptureService",
    "build_local_capture_authority",
    "build_server_capture_authority",
]
