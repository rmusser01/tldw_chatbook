"""Bounded, read-only composition of the existing artifact storage owners."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from .library_artifacts_state import (
    ARTIFACT_PAGE_SIZE,
    ArtifactDetail,
    ArtifactKey,
    ArtifactOrderKey,
    ArtifactPage,
    ArtifactScope,
    ArtifactSourceWindow,
    ArtifactSummary,
    ReadDirection,
    chatbook_actions,
    validate_artifact_window,
)


class ArtifactReadError(RuntimeError):
    """An admitted owner failed; its inventory must not look empty."""

    def __init__(
        self, source: str, message: str = "Artifact source unavailable"
    ) -> None:
        self.source = source
        super().__init__(message)


class LibraryArtifactsCatalog:
    """Compose bounded source reads without owning storage or mutations."""

    def __init__(
        self,
        *,
        subscriptions_db: Any,
        chachanotes_db: Any,
        chatbook_service: Any = None,
    ) -> None:
        self.subscriptions_db = subscriptions_db
        self.chachanotes_db = chachanotes_db
        self.chatbook_service = chatbook_service

    @contextmanager
    def _snapshots(self, scope: ArtifactScope) -> Iterator[list[tuple[str, object]]]:
        owners = []
        reports = scope.view in ("reports", "all")
        registry = scope.view in ("chatbooks", "all") and not scope.kept_only
        source = "live_report"
        try:
            # Parse the detached registry before entering DB-specific operation
            # leases; the content owner cannot borrow a different DB's lease.
            if registry and self.chatbook_service is not None:
                source = "chatbook"
                owners.append((source, self.chatbook_service.artifact_read_snapshot()))
            with ExitStack() as stack:
                if (
                    reports
                    and not scope.kept_only
                    and self.subscriptions_db is not None
                ):
                    source = "live_report"
                    stack.enter_context(self.subscriptions_db.artifact_read_snapshot())
                    owners.append((source, self.subscriptions_db))
                if reports and self.chachanotes_db is not None:
                    source = "kept_report"
                    stack.enter_context(self.chachanotes_db.transaction())
                    owners.append((source, self.chachanotes_db))
                yield owners
        except ArtifactReadError:
            raise
        except Exception as exc:
            raise ArtifactReadError(source) from exc

    @staticmethod
    def _call(source: str, owner, method: str, *args, **kwargs):
        try:
            return getattr(owner, method)(*args, **kwargs)
        except ArtifactReadError:
            raise
        except Exception as exc:
            raise ArtifactReadError(source) from exc

    def _windows(
        self,
        owners,
        scope,
        *,
        boundary=None,
        direction="after",
        limit=20,
        inclusive=False,
    ):
        windows = []
        for source, owner in owners:
            window = self._call(
                source,
                owner,
                "read_artifact_window",
                scope,
                boundary=boundary,
                direction=direction,
                limit=limit,
                inclusive=inclusive,
            )
            if not isinstance(window, ArtifactSourceWindow):
                raise ArtifactReadError(source, "Invalid artifact window")
            counts = (window.total, window.before_boundary, window.equal_boundary)
            if any(type(value) is not int or value < 0 for value in counts):
                raise ArtifactReadError(source, "Invalid artifact counts")
            if boundary is None:
                expected_before = window.total if direction == "before" else 0
                if (
                    window.before_boundary != expected_before
                    or window.equal_boundary != 0
                ):
                    raise ArtifactReadError(source, "Inconsistent null-boundary ranks")
            if (
                window.before_boundary + window.equal_boundary > window.total
                or window.equal_boundary > 1
            ):
                raise ArtifactReadError(source, "Inconsistent artifact ranks")
            if not isinstance(window.items, tuple) or len(window.items) > limit:
                raise ArtifactReadError(source, "Unbounded artifact window")
            previous = None
            for row in window.items:
                if not isinstance(row, ArtifactSummary) or row.key.source != source:
                    raise ArtifactReadError(source, "Invalid artifact identity")
                try:
                    validate_artifact_window(scope, row.order_key, direction, limit)
                except ValueError as exc:
                    raise ArtifactReadError(source, "Invalid artifact order") from exc
                if row.order_key[1:] != (row.key.source, row.key.native_id):
                    raise ArtifactReadError(source, "Inconsistent artifact identity")
                if previous is not None and row.order_key <= previous:
                    raise ArtifactReadError(source, "Duplicate or unordered artifacts")
                previous = row.order_key
                if boundary is not None:
                    valid = (
                        row.order_key >= boundary
                        if inclusive
                        else row.order_key > boundary
                    )
                    if direction == "before":
                        valid = (
                            row.order_key <= boundary
                            if inclusive
                            else row.order_key < boundary
                        )
                    if not valid:
                        raise ArtifactReadError(
                            source, "Artifact outside requested boundary"
                        )
            available = (
                window.total - window.before_boundary - window.equal_boundary
                if direction == "after"
                else window.before_boundary
            )
            if inclusive and boundary is not None:
                available += window.equal_boundary
            if len(window.items) != min(limit, available):
                raise ArtifactReadError(source, "Incomplete artifact window")
            windows.append(window)
        return windows

    @staticmethod
    def _merge(windows, direction="after") -> tuple[ArtifactSummary, ...]:
        candidates = sorted(
            (row for window in windows for row in window.items),
            key=lambda row: row.order_key,
        )
        identities = [row.key for row in candidates]
        if len(set(identities)) != len(identities):
            raise ArtifactReadError("all", "Duplicate artifact identities")
        return tuple(
            candidates[:ARTIFACT_PAGE_SIZE]
            if direction == "after"
            else candidates[-ARTIFACT_PAGE_SIZE:]
        )

    def _page(
        self, owners, scope, *, boundary=None, direction="after", inclusive=False
    ):
        windows = self._windows(
            owners, scope, boundary=boundary, direction=direction, inclusive=inclusive
        )
        items = self._merge(windows, direction)
        total = sum(window.total for window in windows)
        if direction == "after":
            start = sum(
                window.before_boundary + (0 if inclusive else window.equal_boundary)
                for window in windows
            )
        else:
            start = max(
                0, sum(window.before_boundary for window in windows) - len(items)
            )
        if start > total or start + len(items) > total:
            raise ArtifactReadError("all", "Inconsistent artifact page range")
        return ArtifactPage(scope, items, total, start)

    def read_page(
        self,
        scope: ArtifactScope,
        *,
        boundary: ArtifactOrderKey | None = None,
        direction: ReadDirection = "after",
    ) -> ArtifactPage:
        """Read at most twenty summaries from coherent, short-lived owner snapshots."""
        validate_artifact_window(scope, boundary, direction, ARTIFACT_PAGE_SIZE)
        with self._snapshots(scope) as owners:
            page = self._page(owners, scope, boundary=boundary, direction=direction)
            if not page.items and page.total:
                # A removed/end cursor gets one recovery at the corresponding end.
                page = self._page(
                    owners,
                    scope,
                    direction="before" if direction == "after" else "after",
                )
                if not page.items:
                    raise ArtifactReadError("all", "Artifact boundary changed; retry")
            return page

    def locate(self, scope: ArtifactScope, key: ArtifactKey) -> ArtifactPage | None:
        """Locate an exact identity in its aligned page without walking prior pages."""
        validate_artifact_window(scope, None, "after", ARTIFACT_PAGE_SIZE)
        with self._snapshots(scope) as owners:
            target_owner = next(
                ((source, owner) for source, owner in owners if source == key.source),
                None,
            )
            if target_owner is None:
                return None
            source, owner = target_owner
            target = self._call(source, owner, "get_artifact_summary", scope, key)
            if target is None:
                return None
            if not isinstance(target, ArtifactSummary) or target.key != key:
                raise ArtifactReadError(source, "Invalid target identity")
            windows = self._windows(
                owners,
                scope,
                boundary=target.order_key,
                direction="before",
                limit=ARTIFACT_PAGE_SIZE - 1,
            )
            rank = sum(window.before_boundary for window in windows)
            offset = rank % ARTIFACT_PAGE_SIZE
            predecessors = self._merge(windows, "before")
            if len(predecessors) < offset:
                raise ArtifactReadError(source, "Inconsistent target rank")
            boundary = predecessors[-offset].order_key if offset else target.order_key
            page = self._page(owners, scope, boundary=boundary, inclusive=True)
            if (
                page.start != rank - offset
                or len(page.items) <= offset
                or page.items[offset].key != key
            ):
                raise ArtifactReadError(source, "Target moved during artifact read")
            return page

    def read_detail(self, key: ArtifactKey) -> ArtifactDetail | None:
        """Read only the selected artifact body through its existing owner."""
        if key.source == "chatbook":
            return self._read_chatbook_detail(key)
        live = key.source == "live_report"
        owner = self.subscriptions_db if live else self.chachanotes_db
        if owner is None:
            return None
        try:
            with owner.artifact_read_snapshot() if live else owner.transaction():
                summary = owner.get_artifact_summary(ArtifactScope(), key)
                if summary is None:
                    return None
                row = (
                    owner.get_briefing(key.native_id)
                    if live
                    else owner.get_kept_briefing(key.native_id)
                )
                if row is None:
                    raise ArtifactReadError(key.source, "Selected artifact disappeared")
                body = row.get("body_markdown") or ""
                complete = summary.status == "complete"
                can_play = False
                if live:
                    from tldw_chatbook.Subscriptions.briefing_audio import (
                        briefing_audio_dir,
                    )
                    from tldw_chatbook.Utils.path_validation import validate_path

                    audio_path = owner.get_artifact_audio_path(key)
                    if audio_path:
                        try:
                            validated_audio = validate_path(
                                audio_path, briefing_audio_dir(), redact_paths=True
                            )
                            can_play = validated_audio.is_file()
                        except (OSError, ValueError, RuntimeError):
                            can_play = False
                details = tuple(
                    (label, str(row[field]))
                    for label, field in (
                        ("Created", "created_at" if live else "original_created_at"),
                        ("Kept", "kept_at"),
                        ("Model", "model_used"),
                        ("Items", "item_count"),
                        ("Retention", "origin"),
                    )
                    if row.get(field) is not None
                )
                return ArtifactDetail(
                    key=key,
                    revision=summary.revision,
                    body=body,
                    truncated=False,
                    can_keep=live and complete and bool(body),
                    can_export=complete and bool(body),
                    can_play=can_play,
                    can_share=False,
                    source_available=live,
                    details=details,
                )
        except ArtifactReadError:
            raise
        except Exception as exc:
            raise ArtifactReadError(key.source) from exc

    def _read_chatbook_detail(self, key: ArtifactKey) -> ArtifactDetail | None:
        if self.chatbook_service is None:
            return None
        try:
            snapshot = self.chatbook_service.artifact_read_snapshot()
            summary = snapshot.get_artifact_summary(
                ArtifactScope(view="chatbooks"), key
            )
            if summary is None:
                return None
            record = snapshot.get_record(key.native_id)
            if record is None:
                raise ArtifactReadError(key.source, "Selected artifact disappeared")
            metadata = record.get("metadata") or {}
            saved = self.chatbook_service._is_console_saved_artifact(record)
            body = str(record.get("description") or "No description available.")
            truncated = False
            if saved:
                from tldw_chatbook.Chat.console_save_targets import (
                    CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS,
                )

                stored = metadata.get("content")
                stored = stored if isinstance(stored, str) else ""
                body = stored[:CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS]
                truncated = (
                    metadata.get("content_truncated") is True
                    or len(stored) > CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS
                )
            from tldw_chatbook.Chatbooks.artifact_registry_snapshot import (
                usable_chatbook_bundle,
            )

            usable_zip, sharing = usable_chatbook_bundle(record.get("file_path"))
            actions = chatbook_actions(is_saved_response=saved, usable_zip=usable_zip)
            conversation_id = metadata.get("conversation_id")
            conversation_id = (
                conversation_id
                if isinstance(conversation_id, str) and conversation_id
                else None
            )
            message_id = metadata.get("message_id")
            message_id = (
                message_id if isinstance(message_id, str) and message_id else None
            )
            source_available = False
            if conversation_id is not None and self.chachanotes_db is not None:
                from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
                from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError

                try:
                    source_available = (
                        self.chachanotes_db.get_conversation_by_id(conversation_id)
                        is not None
                    )
                except (CharactersRAGDBError, RecoveryRequired):
                    # Optional source navigation must not hide a durable saved response.
                    source_available = False
            details = [
                (
                    "Preview",
                    "Saved response excerpt"
                    if truncated
                    else "Saved response"
                    if saved
                    else "Registry metadata",
                ),
                ("Sharing", sharing),
                (
                    "Source",
                    "Conversation available"
                    if source_available
                    else "Source conversation unavailable",
                ),
            ]
            details.extend(
                (label, str(record[field]))
                for label, field in (
                    ("Created", "created_at"),
                    ("Updated", "updated_at"),
                )
                if record.get(field) is not None
            )
            details.extend(
                (label, str(metadata[field]))
                for label, field in (
                    ("Conversation", "conversation_id"),
                    ("Message", "message_id"),
                    ("Role", "message_role"),
                    ("Provider", "provider"),
                    ("Model", "model"),
                )
                if isinstance(metadata.get(field), str) and metadata[field]
            )
            return ArtifactDetail(
                key=key,
                revision=summary.revision,
                body=body,
                truncated=truncated,
                can_keep=False,
                can_export=False,
                can_play=False,
                can_share="share" in actions,
                source_available=source_available,
                details=tuple(details),
                source_conversation_id=conversation_id,
                source_message_id=message_id,
            )
        except ArtifactReadError:
            raise
        except Exception as exc:
            raise ArtifactReadError(key.source) from exc
