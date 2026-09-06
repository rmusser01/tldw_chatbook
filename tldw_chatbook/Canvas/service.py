"""Durable branch-aware Canvas operations scoped to one Console run."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

from .limits import (
    MAX_CANVAS_DURABLE_ACTIVE_PATH_MESSAGES,
    CanvasLimitError,
    CanvasLimits,
    validate_opaque_identifier,
)
from .models import (
    CanvasCompatibilityIssue,
    CanvasConflictResult,
    CanvasCreateResult,
    CanvasListItem,
    CanvasMutationResult,
    CanvasOrigin,
    CanvasQuotaUsage,
    CanvasReadResult,
    CanvasRenderPlan,
    CanvasRevisionInfo,
    CanvasScope,
)
from .repository import (
    CanvasNotFoundError,
    CanvasQuotaError,
    CanvasRepository,
    CanvasRepositoryError,
    CanvasRevision,
    CanvasRevisionMetadata,
    CanvasValidationError,
)


def compile_canvas_document(
    source: str, *, limits: CanvasLimits | None = None
) -> CanvasRenderPlan:
    """Compile on first use while preserving this module's injection seam."""

    from .compiler import compile_canvas_document as compile_source

    return compile_source(source, limits=limits)


class CanvasServiceError(Exception):
    """A bounded service failure that never contains Canvas source."""

    __slots__ = ("code", "issues")

    def __init__(
        self,
        code: str,
        *,
        issues: tuple[CanvasCompatibilityIssue, ...] = (),
    ) -> None:
        self.code = code
        self.issues = issues
        super().__init__(_ERROR_MESSAGES.get(code, _ERROR_MESSAGES["operation_failed"]))


@dataclass(frozen=True, slots=True)
class _VerifiedScope:
    scope: CanvasScope
    path_positions: dict[str, int]


class CanvasService:
    """Resolve immutable Canvas revisions using captured durable chat authority."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        repository: CanvasRepository | None = None,
        compiler: Callable[[str], CanvasRenderPlan] | None = None,
    ) -> None:
        if not isinstance(db, CharactersRAGDB):
            raise TypeError("db must be a CharactersRAGDB")
        self._db = db
        self._repository = repository or CanvasRepository(db)
        self._compiler = compiler or compile_canvas_document

    def list_canvases(self, scope: CanvasScope) -> tuple[CanvasListItem, ...]:
        """Return one source-free reachable head per Canvas."""

        verified = self._validate_scope(scope)
        metadata = self._list_metadata(scope.conversation_id)
        eligible = tuple(
            revision
            for revision in metadata
            if revision.origin_message_id in verified.path_positions
        )
        defaults: dict[str, CanvasRevisionMetadata] = {}
        for revision in eligible:
            current = defaults.get(revision.canvas_id)
            if current is None or self._resolution_key(
                revision, verified.path_positions
            ) > self._resolution_key(current, verified.path_positions):
                defaults[revision.canvas_id] = revision

        selected = self._reachable_selection(scope, eligible)
        repository_error: CanvasServiceError | None = None
        try:
            reopen_hint = self._repository.get_reopen_hint(scope.conversation_id)
        except CanvasRepositoryError as exc:
            repository_error = self._mapped_repository_error(exc)
        except (CharactersRAGDBError, sqlite3.Error):
            repository_error = CanvasServiceError("storage_failure")
        except Exception:  # noqa: BLE001 - sanitize dependency boundary failures
            repository_error = CanvasServiceError("operation_failed")
        if repository_error is not None:
            raise repository_error
        if reopen_hint not in defaults:
            reopen_hint = None

        resolved: list[tuple[CanvasListItem, int]] = []
        for canvas_id, default in defaults.items():
            chosen = (
                selected
                if selected is not None and selected.canvas_id == canvas_id
                else default
            )
            is_selected = selected is not None and selected.canvas_id == canvas_id
            resolved.append(
                (
                    CanvasListItem(
                        canvas_id=canvas_id,
                        revision_id=chosen.revision_id,
                        parent_revision_id=chosen.parent_revision_id,
                        title=chosen.title,
                        runtime_profile=chosen.runtime_profile,
                        content_sha256=chosen.content_sha256,
                        source_bytes=chosen.source_bytes,
                        sequence=chosen.sequence,
                        origin=CanvasOrigin(
                            message_id=chosen.origin_message_id,
                            run_id=chosen.origin_turn_id,
                        ),
                        is_selected=is_selected,
                        is_historical_selection=(
                            is_selected and chosen.revision_id != default.revision_id
                        ),
                    ),
                    verified.path_positions[chosen.origin_message_id],
                )
            )
        resolved.sort(
            key=lambda item: (
                item[0].canvas_id != reopen_hint,
                -item[1],
                -item[0].sequence,
                item[0].canvas_id,
            )
        )
        return tuple(item for item, _position in resolved)

    def quota_usage(self, scope: CanvasScope) -> CanvasQuotaUsage:
        """Return source-free conversation totals for staging admission."""

        self._validate_scope(scope)
        metadata = self._list_metadata(scope.conversation_id, include_deleted=True)
        counts: dict[str, int] = {}
        source_bytes = 0
        for revision in metadata:
            counts[revision.canvas_id] = counts.get(revision.canvas_id, 0) + 1
            source_bytes += revision.source_bytes
        return CanvasQuotaUsage(
            canvas_ids=tuple(sorted(counts)),
            revision_counts=tuple(sorted(counts.items())),
            source_bytes=source_bytes,
        )

    def read_canvas(self, scope: CanvasScope, canvas_id: str) -> CanvasReadResult:
        """Read the exact selected-or-resolved reachable revision and its source."""

        verified = self._validate_scope(scope)
        canvas_id = self._validate_uuid_argument(canvas_id, "invalid_canvas_id")
        metadata = self._list_metadata(scope.conversation_id)
        chosen = self._resolve_canvas(verified, metadata, canvas_id)
        exact = self._read_revision(scope.conversation_id, chosen.revision_id)
        return CanvasReadResult(
            revision=self._revision_info(exact), source=exact.source
        )

    def find_imported_revision(
        self,
        scope: CanvasScope,
        *,
        origin_message_id: str,
        origin_turn_id: str,
        content_sha256: str,
    ) -> CanvasRevisionInfo | None:
        """Resolve a prior user import by durable, branch-bound provenance."""

        verified = self._validate_scope(scope)
        self._validate_scope_id(origin_message_id, "origin message ID")
        self._validate_scope_id(origin_turn_id, "origin turn ID")
        if (
            len(content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in content_sha256)
            or origin_message_id not in verified.path_positions
        ):
            raise CanvasServiceError("invalid_scope")
        matches = tuple(
            revision
            for revision in self._list_metadata(scope.conversation_id)
            if revision.actor_kind == "user_import"
            and revision.origin_message_id == origin_message_id
            and revision.origin_turn_id == origin_turn_id
            and revision.content_sha256 == content_sha256
        )
        if not matches:
            return None
        chosen = min(
            matches,
            key=lambda revision: (
                revision.canvas_created_at,
                revision.revision_created_at,
                revision.sequence,
                revision.revision_id,
            ),
        )
        return self._revision_info(chosen)

    def next_revision_sequence(self, scope: CanvasScope, canvas_id: str) -> int:
        """Return the next owner-global sequence after proving Canvas reachability."""

        verified = self._validate_scope(scope)
        canvas_id = self._validate_uuid_argument(canvas_id, "invalid_canvas_id")
        metadata = self._list_metadata(scope.conversation_id)
        self._resolve_canvas(verified, metadata, canvas_id)
        return (
            max(
                revision.sequence
                for revision in metadata
                if revision.canvas_id == canvas_id
            )
            + 1
        )

    def create_canvas(
        self,
        scope: CanvasScope,
        *,
        title: str,
        source: str,
    ) -> CanvasCreateResult:
        """Compile then durably create a Canvas at the captured active-path leaf."""

        return self._create_canvas(
            scope, title=title, source=source, actor_kind="assistant"
        )

    def import_canvas(
        self,
        scope: CanvasScope,
        *,
        title: str,
        source: str,
        origin_message_id: str | None = None,
        origin_turn_id: str | None = None,
        _prepared_plan: CanvasRenderPlan | None = None,
    ) -> CanvasCreateResult:
        """Durably import a user-selected transcript HTML block."""

        return self._create_canvas(
            scope,
            title=title,
            source=source,
            actor_kind="user_import",
            origin_message_id=origin_message_id,
            origin_turn_id=origin_turn_id,
            _prepared_plan=_prepared_plan,
        )

    def _create_canvas(
        self,
        scope: CanvasScope,
        *,
        title: str,
        source: str,
        actor_kind: str,
        origin_message_id: str | None = None,
        origin_turn_id: str | None = None,
        _prepared_plan: CanvasRenderPlan | None = None,
    ) -> CanvasCreateResult:
        verified = self._validate_scope(scope, require_active_path=True)
        origin_message_id = origin_message_id or scope.active_message_ids[-1]
        origin_turn_id = origin_turn_id or scope.run_id
        self._validate_scope_id(origin_message_id, "origin message ID")
        self._validate_scope_id(origin_turn_id, "origin turn ID")
        if origin_message_id not in verified.path_positions:
            raise CanvasServiceError("invalid_scope")
        origin_path = scope.active_message_ids[
            : verified.path_positions[origin_message_id] + 1
        ]
        plan = self._compile(source, _prepared_plan)
        repository_error: CanvasServiceError | None = None
        try:
            created = self._repository.create_canvas(
                scope.conversation_id,
                title=title,
                source=source,
                runtime_profile="canvas-v1",
                actor_kind=actor_kind,
                origin_message_id=origin_message_id,
                origin_turn_id=origin_turn_id,
                active_message_ids=origin_path,
            )
        except CanvasRepositoryError as exc:
            repository_error = self._mapped_repository_error(exc)
        except (CharactersRAGDBError, sqlite3.Error):
            repository_error = CanvasServiceError("storage_failure")
        except Exception:  # noqa: BLE001 - sanitize dependency boundary failures
            repository_error = CanvasServiceError("operation_failed")
        if repository_error is not None:
            raise repository_error
        return CanvasCreateResult(
            revision=self._revision_info(created.revision),
            source=created.revision.source,
            compatibility_issues=plan.compatibility_issues,
        )

    def update_canvas(
        self,
        scope: CanvasScope,
        canvas_id: str,
        *,
        expected_parent_revision_id: str,
        source: str,
    ) -> CanvasMutationResult | CanvasConflictResult:
        """Append a complete replacement from the exact captured reachable base."""

        return self._update_canvas(
            scope,
            canvas_id,
            expected_parent_revision_id=expected_parent_revision_id,
            source=source,
            actor_kind="assistant",
        )

    def import_update_canvas(
        self,
        scope: CanvasScope,
        canvas_id: str,
        *,
        expected_parent_revision_id: str,
        source: str,
        origin_message_id: str | None = None,
        origin_turn_id: str | None = None,
        _prepared_plan: CanvasRenderPlan | None = None,
    ) -> CanvasMutationResult | CanvasConflictResult:
        """Append a replacement imported explicitly by the user."""

        return self._update_canvas(
            scope,
            canvas_id,
            expected_parent_revision_id=expected_parent_revision_id,
            source=source,
            actor_kind="user_import",
            origin_message_id=origin_message_id,
            origin_turn_id=origin_turn_id,
            _prepared_plan=_prepared_plan,
        )

    def _update_canvas(
        self,
        scope: CanvasScope,
        canvas_id: str,
        *,
        expected_parent_revision_id: str,
        source: str,
        actor_kind: str,
        origin_message_id: str | None = None,
        origin_turn_id: str | None = None,
        _prepared_plan: CanvasRenderPlan | None = None,
    ) -> CanvasMutationResult | CanvasConflictResult:
        verified = self._validate_scope(scope, require_active_path=True)
        origin_message_id = origin_message_id or scope.active_message_ids[-1]
        origin_turn_id = origin_turn_id or scope.run_id
        self._validate_scope_id(origin_message_id, "origin message ID")
        self._validate_scope_id(origin_turn_id, "origin turn ID")
        if origin_message_id not in verified.path_positions:
            raise CanvasServiceError("invalid_scope")
        origin_path = scope.active_message_ids[
            : verified.path_positions[origin_message_id] + 1
        ]
        canvas_id = self._validate_uuid_argument(canvas_id, "invalid_canvas_id")
        expected_parent_revision_id = self._validate_uuid_argument(
            expected_parent_revision_id, "invalid_expected_parent"
        )
        metadata = self._list_metadata(scope.conversation_id)
        base = self._resolve_canvas(verified, metadata, canvas_id)
        if expected_parent_revision_id != base.revision_id:
            return self._conflict(base)

        plan = self._compile(source, _prepared_plan)
        repository_error: CanvasServiceError | None = None
        try:
            revision = self._repository.append_revision(
                scope.conversation_id,
                canvas_id,
                parent_revision_id=base.revision_id,
                title=base.title,
                source=source,
                runtime_profile=base.runtime_profile,
                actor_kind=actor_kind,
                origin_message_id=origin_message_id,
                origin_turn_id=origin_turn_id,
                active_message_ids=origin_path,
            )
        except CanvasRepositoryError as exc:
            repository_error = self._mapped_repository_error(exc)
        except (CharactersRAGDBError, sqlite3.Error):
            repository_error = CanvasServiceError("storage_failure")
        except Exception:  # noqa: BLE001 - sanitize dependency boundary failures
            repository_error = CanvasServiceError("operation_failed")
        if repository_error is not None:
            raise repository_error
        return CanvasMutationResult(
            revision=self._revision_info(revision),
            compatibility_issues=plan.compatibility_issues,
        )

    def rename_canvas(
        self,
        scope: CanvasScope,
        canvas_id: str,
        *,
        expected_parent_revision_id: str,
        title: str,
    ) -> CanvasMutationResult | CanvasConflictResult:
        """Append a title-only child from the exact captured reachable base."""

        verified = self._validate_scope(scope, require_active_path=True)
        canvas_id = self._validate_uuid_argument(canvas_id, "invalid_canvas_id")
        expected_parent_revision_id = self._validate_uuid_argument(
            expected_parent_revision_id, "invalid_expected_parent"
        )
        metadata = self._list_metadata(scope.conversation_id)
        base = self._resolve_canvas(verified, metadata, canvas_id)
        if expected_parent_revision_id != base.revision_id:
            return self._conflict(base)

        exact = self._read_revision(scope.conversation_id, base.revision_id)
        repository_error: CanvasServiceError | None = None
        try:
            revision = self._repository.append_revision(
                scope.conversation_id,
                canvas_id,
                parent_revision_id=base.revision_id,
                title=title,
                source=exact.source,
                runtime_profile=base.runtime_profile,
                actor_kind="user_rename",
                origin_message_id=scope.active_message_ids[-1],
                origin_turn_id=scope.run_id,
                active_message_ids=scope.active_message_ids,
            )
        except CanvasRepositoryError as exc:
            repository_error = self._mapped_repository_error(exc)
        except (CharactersRAGDBError, sqlite3.Error):
            repository_error = CanvasServiceError("storage_failure")
        except Exception:  # noqa: BLE001 - sanitize dependency boundary failures
            repository_error = CanvasServiceError("operation_failed")
        if repository_error is not None:
            raise repository_error
        return CanvasMutationResult(revision=self._revision_info(revision))

    def _validate_scope(
        self,
        scope: CanvasScope,
        *,
        require_active_path: bool = False,
    ) -> _VerifiedScope:
        if not isinstance(scope, CanvasScope):
            raise CanvasServiceError("invalid_scope")
        for value, field_name in (
            (scope.session_id, "session ID"),
            (scope.conversation_id, "conversation ID"),
            (scope.run_id, "run ID"),
        ):
            self._validate_scope_id(value, field_name)
        if type(scope.active_message_ids) is not tuple:
            raise CanvasServiceError("invalid_scope")
        if len(scope.active_message_ids) > MAX_CANVAS_DURABLE_ACTIVE_PATH_MESSAGES:
            raise CanvasServiceError("invalid_scope")
        if (scope.selected_canvas_id is None) != (scope.selected_revision_id is None):
            raise CanvasServiceError("invalid_scope")
        if scope.selected_canvas_id is not None:
            self._validate_scope_id(scope.selected_canvas_id, "selected Canvas ID")
            self._validate_scope_id(scope.selected_revision_id, "selected revision ID")

        seen: set[str] = set()
        for message_id in scope.active_message_ids:
            self._validate_scope_id(message_id, "active message ID")
            if message_id in seen:
                raise CanvasServiceError("invalid_scope")
            seen.add(message_id)
        if require_active_path and not scope.active_message_ids:
            raise CanvasServiceError("invalid_scope")

        storage_failed = False
        try:
            connection = self._db.get_connection()
            owner = connection.execute(
                "SELECT deleted FROM conversations WHERE id = ? LIMIT 1",
                (scope.conversation_id,),
            ).fetchone()
            if owner is None or int(owner[0]) != 0:
                raise CanvasServiceError("invalid_scope")
            if not scope.active_message_ids:
                return _VerifiedScope(scope=scope, path_positions={})
            placeholders = ", ".join("?" for _ in scope.active_message_ids)
            rows = connection.execute(
                "SELECT id, parent_message_id, conversation_id, deleted FROM messages "
                f"WHERE id IN ({placeholders})",
                scope.active_message_ids,
            ).fetchall()
        except CanvasServiceError:
            raise
        except (CharactersRAGDBError, sqlite3.Error):
            storage_failed = True
        except Exception:  # noqa: BLE001 - sanitize database boundary failures
            storage_failed = True
        if storage_failed:
            raise CanvasServiceError("storage_failure")

        by_id = {str(row[0]): row for row in rows}
        if len(by_id) != len(scope.active_message_ids):
            raise CanvasServiceError("invalid_scope")
        previous_id: str | None = None
        for message_id in scope.active_message_ids:
            row = by_id[message_id]
            parent_id = str(row[1]) if row[1] is not None else None
            if (
                str(row[2]) != scope.conversation_id
                or int(row[3]) != 0
                or parent_id != previous_id
            ):
                raise CanvasServiceError("invalid_scope")
            previous_id = message_id
        return _VerifiedScope(
            scope=scope,
            path_positions={
                message_id: position
                for position, message_id in enumerate(scope.active_message_ids)
            },
        )

    def _resolve_canvas(
        self,
        verified: _VerifiedScope,
        metadata: tuple[CanvasRevisionMetadata, ...],
        canvas_id: str,
    ) -> CanvasRevisionMetadata:
        eligible = tuple(
            revision
            for revision in metadata
            if revision.origin_message_id in verified.path_positions
        )
        canvas_revisions = tuple(
            revision for revision in eligible if revision.canvas_id == canvas_id
        )
        if not canvas_revisions:
            raise CanvasServiceError("canvas_not_found")
        default = max(
            canvas_revisions,
            key=lambda revision: self._resolution_key(
                revision, verified.path_positions
            ),
        )
        selected = self._reachable_selection(verified.scope, eligible)
        if selected is not None and selected.canvas_id == canvas_id:
            return selected
        return default

    def _compile(
        self, source: str, prepared_plan: CanvasRenderPlan | None = None
    ) -> CanvasRenderPlan:
        from .compiler import CanvasCompileError

        failure_issues: tuple[CanvasCompatibilityIssue, ...] | None = None
        try:
            plan = self._compiler(source) if prepared_plan is None else prepared_plan
            if not isinstance(plan, CanvasRenderPlan):
                raise CanvasLimitError("compiler returned an invalid render plan")
            if plan.runtime_profile != "canvas-v1":
                raise CanvasLimitError("unsupported Canvas runtime profile")
            plan.source_identity.verify_source(source)
            return plan
        except CanvasCompileError as exc:
            failure_issues = exc.issues
        except (CanvasLimitError, TypeError):
            failure_issues = ()
        except Exception:  # noqa: BLE001 - third-party compiler failures are untrusted
            failure_issues = ()
        if failure_issues is not None:
            raise CanvasServiceError("document_incompatible", issues=failure_issues)
        raise CanvasServiceError("document_incompatible")

    def _read_revision(self, conversation_id: str, revision_id: str) -> CanvasRevision:
        repository_error: CanvasServiceError | None = None
        try:
            revision = self._repository.read_revision(conversation_id, revision_id)
        except CanvasRepositoryError as exc:
            repository_error = self._mapped_repository_error(exc)
        except (CharactersRAGDBError, sqlite3.Error):
            repository_error = CanvasServiceError("storage_failure")
        except Exception:  # noqa: BLE001 - sanitize dependency boundary failures
            repository_error = CanvasServiceError("operation_failed")
        if repository_error is not None:
            raise repository_error
        return revision

    def _list_metadata(
        self, conversation_id: str, *, include_deleted: bool = False
    ) -> tuple[CanvasRevisionMetadata, ...]:
        repository_error: CanvasServiceError | None = None
        try:
            if include_deleted:
                metadata = self._repository.list_revision_metadata(
                    conversation_id, include_deleted=True
                )
            else:
                metadata = self._repository.list_revision_metadata(conversation_id)
        except CanvasRepositoryError as exc:
            repository_error = self._mapped_repository_error(exc)
        except (CharactersRAGDBError, sqlite3.Error):
            repository_error = CanvasServiceError("storage_failure")
        except Exception:  # noqa: BLE001 - sanitize dependency boundary failures
            repository_error = CanvasServiceError("operation_failed")
        if repository_error is not None:
            raise repository_error
        return metadata

    @staticmethod
    def _reachable_selection(
        scope: CanvasScope,
        eligible: tuple[CanvasRevisionMetadata, ...],
    ) -> CanvasRevisionMetadata | None:
        if scope.selected_canvas_id is None or scope.selected_revision_id is None:
            return None
        for revision in eligible:
            if (
                revision.canvas_id == scope.selected_canvas_id
                and revision.revision_id == scope.selected_revision_id
            ):
                return revision
        return None

    @staticmethod
    def _resolution_key(
        revision: CanvasRevisionMetadata,
        path_positions: dict[str, int],
    ) -> tuple[int, int, str]:
        return (
            path_positions[revision.origin_message_id],
            revision.sequence,
            revision.revision_id,
        )

    @staticmethod
    def _revision_info(
        revision: CanvasRevision | CanvasRevisionMetadata,
    ) -> CanvasRevisionInfo:
        return CanvasRevisionInfo(
            canvas_id=revision.canvas_id,
            revision_id=revision.revision_id,
            parent_revision_id=revision.parent_revision_id,
            title=revision.title,
            runtime_profile=revision.runtime_profile,
            content_sha256=revision.content_sha256,
            source_bytes=revision.source_bytes,
            sequence=revision.sequence,
            origin=CanvasOrigin(
                message_id=revision.origin_message_id,
                run_id=revision.origin_turn_id,
            ),
        )

    @staticmethod
    def _conflict(current: CanvasRevisionMetadata) -> CanvasConflictResult:
        return CanvasConflictResult(
            code="stale_parent",
            canvas_id=current.canvas_id,
            current_revision_id=current.revision_id,
            content_sha256=current.content_sha256,
            title=current.title,
            sequence=current.sequence,
            origin=CanvasOrigin(
                message_id=current.origin_message_id,
                run_id=current.origin_turn_id,
            ),
        )

    @staticmethod
    def _validate_uuid_argument(value: object, code: str) -> str:
        if type(value) is not str:
            raise CanvasServiceError(code)
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError):
            raise CanvasServiceError(code) from None
        if str(parsed) != value:
            raise CanvasServiceError(code)
        return value

    @staticmethod
    def _validate_scope_id(value: object, field_name: str) -> None:
        try:
            validate_opaque_identifier(value, field_name=field_name)  # type: ignore[arg-type]
        except (CanvasLimitError, TypeError):
            raise CanvasServiceError("invalid_scope") from None

    @staticmethod
    def _mapped_repository_error(exc: CanvasRepositoryError) -> CanvasServiceError:
        if isinstance(exc, CanvasQuotaError):
            return CanvasServiceError(
                _QUOTA_ERROR_CODES.get(exc.code, "quota_exceeded")
            )
        if isinstance(exc, CanvasNotFoundError):
            return CanvasServiceError("canvas_not_found")
        if isinstance(exc, CanvasValidationError):
            return CanvasServiceError(
                _VALIDATION_ERROR_CODES.get(exc.code, "invalid_canvas_operation")
            )
        if exc.code == "storage_failure":
            return CanvasServiceError("storage_failure")
        return CanvasServiceError("operation_failed")


_ERROR_MESSAGES = {
    "invalid_scope": "Canvas scope is unavailable or no longer valid.",
    "invalid_canvas_id": "Canvas identifier is invalid.",
    "invalid_expected_parent": "Expected Canvas revision identifier is invalid.",
    "invalid_canvas_operation": "Canvas operation is invalid.",
    "invalid_title": "Canvas title is invalid.",
    "invalid_source": "Canvas source is invalid.",
    "canvas_not_found": "Canvas is unavailable on this conversation branch.",
    "document_incompatible": "Canvas document is incompatible with the canvas-v1 runtime.",
    "canvas_count_limit": "Canvas conversation limit reached.",
    "revision_count_limit": "Canvas revision limit reached.",
    "conversation_source_limit": "Canvas conversation storage limit reached.",
    "revision_source_limit": "Canvas revision storage limit reached.",
    "title_limit": "Canvas title limit reached.",
    "quota_exceeded": "Canvas storage limit reached.",
    "storage_failure": "Canvas storage is temporarily unavailable.",
    "operation_failed": "Canvas operation could not be completed.",
}

_QUOTA_ERROR_CODES = {
    "canvas_count": "canvas_count_limit",
    "revision_count": "revision_count_limit",
    "conversation_source_bytes": "conversation_source_limit",
    "revision_source_bytes": "revision_source_limit",
    "title_bytes": "title_limit",
}

_VALIDATION_ERROR_CODES = {
    "invalid_active_path": "invalid_scope",
    "invalid_title": "invalid_title",
    "invalid_source": "invalid_source",
}


__all__ = ["CanvasService", "CanvasServiceError"]
