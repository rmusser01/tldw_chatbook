"""Bounded cross-workspace local archive discovery (ADR-166)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, replace

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.DB.conversation_archive_queries import normalized_archive_time
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

from .contracts import (
    ArchiveCandidate,
    ArchiveRecoveryError,
    ArchiveSearchPage,
    ArchiveSearchRequest,
    ArchiveTarget,
    ArchiveWorkspaceSnapshot,
)

SEARCH_BUDGET_SECONDS = 5.0
# Leave room for the adapter's status, reference and cursor envelope.
MAX_PAGE_BYTES = 30 * 1024


def matching_archive_events(
    *,
    chat_at: str | None,
    workspace_at: str | None,
    after: str | None,
    before: str | None,
) -> tuple[str, ...]:
    """Return current events inside the inclusive/exclusive UTC window.

    Invalid legacy observations remain undated; request dates are validated
    before this helper is used. This is also the independent SQL test oracle.
    """
    lower, upper = normalized_archive_time(after), normalized_archive_time(before)
    return tuple(
        kind
        for kind, value in (("chat", chat_at), ("workspace", workspace_at))
        if (event := normalized_archive_time(value)) is not None
        and (lower is None or event >= lower)
        and (upper is None or event < upper)
    )


class ArchiveSearchService:
    """Search saved recovery inventory on the owning storage worker."""

    def __init__(
        self,
        chat_service: ChatConversationService,
        workspace_registry: LocalWorkspaceRegistryService,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.chat_service = chat_service
        self.workspace_registry = workspace_registry
        self.monotonic = monotonic

    def _check_deadline(self, deadline: float) -> None:
        if self.monotonic() >= deadline:
            raise ArchiveRecoveryError(
                "search_incomplete",
                "Archive search exceeded its budget; refine or retry.",
            )

    def _workspace_snapshot(
        self, deadline: float
    ) -> tuple[ArchiveWorkspaceSnapshot, ...]:
        self._check_deadline(deadline)
        # The registry owns this thread-local connection. Install interruption
        # around its existing enumeration, including bounded per-row snapshot projection.
        try:
            with (
                operation_owned_connection(
                    self.workspace_registry.db,
                    operation_deadline=time.monotonic()
                    + max(0.0, deadline - self.monotonic()),
                ),
                self.workspace_registry.db.connection(
                    operation_deadline=time.monotonic()
                    + max(0.0, deadline - self.monotonic())
                ) as conn,
            ):
                self._check_deadline(deadline)
                old_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
                conn.execute(
                    f"PRAGMA busy_timeout = {max(0, int((deadline - self.monotonic()) * 1000))}"
                )
                conn.set_progress_handler(
                    lambda: int(self.monotonic() >= deadline), 100
                )
                try:
                    records = conn.execute(
                        "SELECT workspace_id, name, archived, archived_at, archive_revision "
                        "FROM workspace_records ORDER BY workspace_id"
                    )
                    snapshots = []
                    for record in records:
                        self._check_deadline(deadline)
                        snapshots.append(
                            ArchiveWorkspaceSnapshot(
                                record["workspace_id"],
                                record["name"],
                                bool(record["archived"]),
                                normalized_archive_time(record["archived_at"])
                                if record["archived"]
                                else None,
                                record["archive_revision"],
                            )
                        )
                    self._check_deadline(deadline)
                    return tuple(sorted(snapshots, key=lambda item: item.workspace_id))
                finally:
                    conn.set_progress_handler(None, 0)
                    conn.execute(f"PRAGMA busy_timeout = {old_timeout}")
        except ArchiveRecoveryError:
            raise
        except Exception as exc:
            # The registry sanitizes SQLite errors, so inspect the cause only
            # for classification; never forward storage exception text.
            cause = exc.__cause__ or exc
            if (
                isinstance(exc, TimeoutError)
                or isinstance(cause, TimeoutError)
                or self.monotonic() >= deadline
                or (
                    isinstance(cause, sqlite3.OperationalError)
                    and any(
                        word in str(cause).lower()
                        for word in ("interrupt", "locked", "busy")
                    )
                )
            ):
                raise ArchiveRecoveryError(
                    "search_incomplete",
                    "Workspace inventory search was interrupted; retry.",
                ) from exc
            raise ArchiveRecoveryError(
                "unavailable", "Local workspace inventory is unavailable."
            ) from exc

    def search(
        self,
        request: ArchiveSearchRequest,
        *,
        after: tuple[int, str, str] | None = None,
        expected_workspace_digest: str | None = None,
    ) -> ArchiveSearchPage:
        """Return a bounded live page, rejecting changed workspace inputs.

        Args:
            request: Validated search criteria.
            after: Trusted complete keyset resolved by the runtime cursor owner.
            expected_workspace_digest: Prior page's workspace lifecycle digest.

        Returns:
            Immutable candidates and a continuation anchor if more may remain.

        Raises:
            ArchiveRecoveryError: Stale, unavailable, interrupted or failed storage.
        """
        if not isinstance(request, ArchiveSearchRequest):
            raise ArchiveRecoveryError(
                "invalid_argument", "A validated archive search request is required."
            )
        if after is not None and (
            not isinstance(after, tuple)
            or len(after) != 3
            or type(after[0]) is not int
            or after[0] not in (0, 1)
            or not all(isinstance(v, str) for v in after[1:])
        ):
            raise ArchiveRecoveryError(
                "invalid_argument", "Invalid continuation anchor."
            )
        deadline = self.monotonic() + SEARCH_BUDGET_SECONDS
        snapshots = self._workspace_snapshot(deadline)
        workspace_rows = tuple(asdict(item) for item in snapshots)
        digest = hashlib.sha256(
            json.dumps(
                workspace_rows,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self._check_deadline(deadline)
        if (
            expected_workspace_digest is not None
            and digest != expected_workspace_digest
        ):
            raise ArchiveRecoveryError(
                "stale", "Workspace inventory changed; refresh the search."
            )
        by_id = {item.workspace_id: item for item in snapshots}
        if request.workspace_id is not None and request.workspace_id not in by_id:
            raise ArchiveRecoveryError(
                "unavailable", "The selected workspace is unavailable."
            )
        data = request.model_dump(mode="json")
        try:
            with operation_owned_connection(
                self.chat_service.db,
                operation_deadline=time.monotonic()
                + max(0.0, deadline - self.monotonic()),
            ):
                raw = self.chat_service.query_archive_recovery_page(
                    data,
                    workspace_rows,
                    after=after,
                    deadline=deadline,
                    monotonic=self.monotonic,
                )
        except TimeoutError as exc:
            raise ArchiveRecoveryError(
                "search_incomplete", "Archive search was interrupted; refine or retry."
            ) from exc
        except sqlite3.Error as exc:
            raise ArchiveRecoveryError(
                "storage_error", "Local archive storage is unavailable."
            ) from exc
        items = []
        for row in raw["items"]:
            self._check_deadline(deadline)
            workspace_id = row["workspace_id"]
            workspace = by_id.get(workspace_id) if workspace_id else None
            # Default is global authority even when legacy storage explicitly
            # uses the durable Default ID. Missing named ownership stays named.
            if workspace_id == DEFAULT_WORKSPACE_ID:
                workspace_id, workspace = None, None
            archived = row["conversation_archived"]
            workspace_archived = bool(workspace and workspace.archived)
            scopes = (
                ()
                if workspace_id and workspace is None
                else (
                    ("chat_only", "chat_and_workspace")
                    if archived and workspace_archived
                    else ("chat_only",)
                    if archived
                    else ("workspace_only",)
                )
            )
            target = ArchiveTarget(
                row["conversation_id"],
                row["conversation_version"],
                row["title"],
                archived,
                workspace_id,
                workspace,
            )
            items.append(
                ArchiveCandidate(
                    target,
                    row["conversation_archived_at"],
                    row["excerpt"],
                    row["match_basis"],
                    row["sort_key"],
                    row["chronology_incomplete"],
                    row["match_message_id"],
                    scopes,
                )
            )
        page = ArchiveSearchPage(
            tuple(items), raw["next_anchor"], digest, raw["unknown_times_present"]
        )
        page = self._fit_page(page, deadline)
        self._check_deadline(deadline)
        return page

    def _fit_page(self, page: ArchiveSearchPage, deadline: float) -> ArchiveSearchPage:
        # First shrink optional display fields, never lifecycle names/identities.
        # If a minimal row still cannot fit, stop *before* it and anchor at the
        # last emitted row; a single oversized minimal row fails explicitly.
        items = []
        for index, item in enumerate(page.items):
            self._check_deadline(deadline)
            anchor = (
                item.sort_key
                if index < len(page.items) - 1 or page.next_anchor
                else None
            )
            candidate = replace(page, items=tuple(items + [item]), next_anchor=anchor)
            while self._serialized_size(candidate) > MAX_PAGE_BYTES and (
                len(item.excerpt) > 1 or len(item.target.title) > 1
            ):
                self._check_deadline(deadline)
                if len(item.excerpt) > 1:
                    item = (
                        replace(
                            item, excerpt=item.excerpt[: len(item.excerpt) // 2] + "…"
                        )
                        if len(item.excerpt) > 2
                        else replace(item, excerpt="…")
                    )
                elif len(item.target.title) > 1:
                    title = item.target.title
                    item = replace(
                        item,
                        target=replace(
                            item.target,
                            title=title[: len(title) // 2] + "…"
                            if len(title) > 2
                            else "…",
                        ),
                    )
                candidate = replace(candidate, items=tuple(items + [item]))
            if self._serialized_size(candidate) > MAX_PAGE_BYTES:
                if not items:
                    raise ArchiveRecoveryError(
                        "storage_error",
                        "An archive identity cannot fit in a bounded result.",
                    )
                return replace(page, items=tuple(items), next_anchor=items[-1].sort_key)
            items.append(item)
        return replace(page, items=tuple(items))

    @staticmethod
    def _serialized_size(page: ArchiveSearchPage) -> int:
        return len(
            json.dumps(asdict(page), ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
