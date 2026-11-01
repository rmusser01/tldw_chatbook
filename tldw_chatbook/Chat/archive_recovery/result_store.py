"""Ephemeral app-owned recovery evidence; handles never grant permission."""

import hashlib
import json
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from threading import RLock

from .contracts import (
    ArchiveCandidate,
    ArchiveRecoveryError,
    ArchiveResultProjection,
    ArchiveResultRow,
    ArchiveSearchPage,
    ArchiveSearchRequest,
)

MAX_RESULT_BYTES = 32 * 1024
MAX_SESSION_PAGES = 10
RESULT_TTL_SECONDS = 30 * 60


def argument_digest(arguments: dict[str, object]) -> str:
    """Hash canonical JSON arguments after boundary validation by the caller."""
    encoded = json.dumps(
        arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _RetainedPage:
    expires_at: float
    request: ArchiveSearchRequest
    rows: tuple[tuple[str, ArchiveCandidate], ...]
    cursor: str | None
    anchor: tuple[int, str, str] | None
    workspace_digest: str
    model_visible: bool


def _row(candidate: ArchiveCandidate, ref: str) -> ArchiveResultRow:
    target = candidate.target
    return ArchiveResultRow(
        result_ref=ref,
        conversation_id=target.conversation_id,
        conversation_version=target.conversation_version,
        title=target.title,
        conversation_archived=target.conversation_archived,
        conversation_archived_at=candidate.conversation_archived_at,
        workspace_id=target.workspace_id,
        workspace=asdict(target.workspace) if target.workspace else None,
        excerpt=candidate.excerpt,
        archive_match_basis=[
            basis for basis in candidate.match_basis if basis in ("chat", "workspace")
        ],
        text_match_basis=[
            basis
            for basis in candidate.match_basis
            if basis in ("title", "keywords", "message")
        ],
        inactive_branch="inactive_branch" in candidate.match_basis,
        match_message_id=candidate.match_message_id,
        chronology_incomplete=candidate.chronology_incomplete,
        allowed_scopes=list(candidate.allowed_scopes),
    )


def _size(projection: ArchiveResultProjection) -> int:
    return len(
        json.dumps(projection, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )


class ArchiveResultStore:
    """Lock-protected, creation-expiring FIFO pages owned by source session.

    Only publish live application search records here. There is deliberately no
    transcript import or deserialization API. Resolved snapshots are evidence for
    the coordinator's revalidation and exact-target confirmation, not approval.
    """

    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic
        self._sessions: dict[str, OrderedDict[str, _RetainedPage]] = {}
        self._lock = RLock()

    def _prune(self, session_id: str) -> OrderedDict[str, _RetainedPage]:
        pages = self._sessions.get(session_id, OrderedDict())
        now = self._monotonic()
        for identity in tuple(pages):
            if pages[identity].expires_at <= now:
                del pages[identity]
        if not pages:
            self._sessions.pop(session_id, None)
        return pages

    def publish(
        self,
        session_id: str,
        request: ArchiveSearchRequest,
        page: ArchiveSearchPage,
        *,
        model_visible: bool,
    ) -> dict[str, object]:
        """Capture live search snapshots and return a separate UTF-8 JSON page."""
        request = ArchiveSearchRequest.model_validate(request.model_dump(mode="json"))
        next_anchor = page.next_anchor
        if next_anchor is not None:
            if (
                not isinstance(next_anchor, (tuple, list))
                or len(next_anchor) != 3
                or type(next_anchor[0]) is not int
                or next_anchor[0] not in (0, 1)
                or type(next_anchor[1]) is not str
                or type(next_anchor[2]) is not str
            ):
                raise ArchiveRecoveryError(
                    "storage_error", "Invalid recovery cursor anchor"
                )
            next_anchor = tuple(next_anchor)
        items = tuple(deepcopy(item) for item in page.items)
        # Normalize tuple-bearing inputs so even supplied mutable containers cannot
        # become retained registry internals.
        items = tuple(
            replace(
                item,
                match_basis=tuple(item.match_basis),
                allowed_scopes=tuple(item.allowed_scopes),
                sort_key=tuple(item.sort_key),
            )
            for item in items
        )
        if len(items) > request.limit or any(
            scope not in ("chat_only", "workspace_only", "chat_and_workspace")
            for item in items
            for scope in item.allowed_scopes
        ):
            raise ArchiveRecoveryError("storage_error", "Invalid recovery search page")
        set_id = secrets.token_urlsafe(24)
        cursor = secrets.token_urlsafe(24)
        refs = tuple(secrets.token_urlsafe(24) for _ in items)
        projection = ArchiveResultProjection(
            result_set_id=set_id,
            items=[_row(item, ref) for item, ref in zip(items, refs)],
            cursor=cursor if next_anchor is not None else None,
            unknown_times_present=page.unknown_times_present,
            truncated=False,
        )
        # Trim display-only strings before omitting rows. Never trim identity,
        # lifecycle, provenance or permission fields.
        while _size(projection) > MAX_RESULT_BYTES:
            projection["truncated"] = True
            display_fields = [
                (row, field)
                for row in projection["items"]
                for field in ("title", "excerpt")
                if len(row[field]) > 1
            ]
            if display_fields:
                row, field = max(
                    display_fields,
                    key=lambda entry: len(entry[0][entry[1]].encode("utf-8")),
                )
                text = row[field]
                row[field] = text[: len(text) // 2] + "…"
                if len(text) == 2:
                    row[field] = "…"
                continue
            if len(projection["items"]) <= 1:
                raise ArchiveRecoveryError(
                    "storage_error", "Recovery identity cannot fit the response limit"
                )
            projection["items"].pop()
            projection["cursor"] = cursor
        emitted = len(projection["items"])
        anchor = items[emitted - 1].sort_key if emitted < len(items) else next_anchor
        # Preserve the original candidate including display text, independently
        # of any public projection fitting.
        retained = _RetainedPage(
            self._monotonic() + RESULT_TTL_SECONDS,
            request,
            tuple(zip(refs[:emitted], items[:emitted])),
            cursor if anchor else None,
            anchor,
            page.workspace_digest,
            model_visible,
        )
        with self._lock:
            pages = self._prune(session_id)
            pages[set_id] = retained
            while len(pages) > MAX_SESSION_PAGES:
                pages.popitem(last=False)
            self._sessions[session_id] = pages
        return dict(projection)

    def model_visible_result_sets(self, session_id: str) -> frozenset[str]:
        """Return live agent-observed set identities, without exposing stored bodies."""
        with self._lock:
            return frozenset(
                identity
                for identity, page in self._prune(session_id).items()
                if page.model_visible
            )

    def resolve(
        self, session_id: str, result_ref: str, *, for_agent: bool = False
    ) -> ArchiveCandidate:
        """Resolve only a live owner reference, respecting manual-only provenance."""
        with self._lock:
            for page in self._prune(session_id).values():
                for ref, item in page.rows:
                    if ref == result_ref:
                        if for_agent and not page.model_visible:
                            raise ArchiveRecoveryError(
                                "unavailable",
                                "A manual result requires a fresh agent search",
                            )
                        return item
        raise ArchiveRecoveryError(
            "expired", "Recovery reference expired or unavailable; refresh search"
        )

    def resolve_cursor(
        self, session_id: str, cursor: str
    ) -> tuple[ArchiveSearchRequest, tuple[int, str, str], str]:
        """Return app-owned original normalized filters, keyset and digest."""
        with self._lock:
            for page in self._prune(session_id).values():
                if page.cursor == cursor and page.anchor is not None:
                    return page.request, page.anchor, page.workspace_digest
        raise ArchiveRecoveryError(
            "expired", "Recovery cursor expired or unavailable; refresh search"
        )

    def close_session(self, session_id: str) -> None:
        """Invalidate source pages; coordinator owns separately pinned targets."""
        with self._lock:
            self._sessions.pop(session_id, None)
