"""Apply the complete Notes organization domain group to ChaChaNotes."""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

from tldw_chatbook.Sync_Interop.notes_organization import (
    NotesOrganizationValidationError,
    parse_notes_organization_payload,
    validate_organization_object_id,
)

if TYPE_CHECKING:
    from tldw_chatbook.Notes.notes_organization_repository import (
        NotesOrganizationRepository,
    )
    from tldw_chatbook.tldw_api import SyncV2Envelope


class NotesOrganizationSyncAdapter:
    """Validate and transactionally project one organization envelope."""

    def apply(
        self,
        envelope: SyncV2Envelope,
        *,
        repository: NotesOrganizationRepository | None,
        restore_intent: bool,
        record_conflict: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        from tldw_chatbook.Notes.notes_organization_repository import (
            NotesOrganizationRepositoryError,
        )

        if repository is None:
            return self._rejected("notes_organization_repository_unavailable")
        if envelope.schema_version != 1:
            return self._rejected("notes_organization_schema_version_invalid")
        if envelope.encryption_policy != "server_trusted_v1":
            return self._rejected("notes_organization_encryption_policy_invalid")
        object_id = envelope.object_id or envelope.entity_id
        if object_id is None:
            return self._rejected("notes_organization_object_id_missing")

        try:
            payload = parse_notes_organization_payload(
                envelope.domain, envelope.operation, envelope.payload
            )
            validate_organization_object_id(envelope.domain, object_id, payload)
        except NotesOrganizationValidationError as exc:
            return self._rejected(exc.error_code)

        try:
            with repository.db.transaction(immediate=True) as cursor:
                result = repository.apply_envelope(
                    cursor,
                    dataset_id=envelope.dataset_id,
                    domain=envelope.domain,
                    object_id=object_id,
                    operation=envelope.operation,
                    payload=payload,
                    object_revision=envelope.object_revision,
                    object_hash=envelope.payload_hash,
                    server_cursor=(
                        str(envelope.server_cursor)
                        if envelope.server_cursor is not None
                        else ""
                    ),
                    base_server_cursor=(
                        str(envelope.base_server_cursor)
                        if envelope.base_server_cursor is not None
                        else None
                    ),
                    base_object_revision=envelope.base_object_revision,
                    base_object_hash=envelope.base_object_hash,
                    restore_intent=restore_intent,
                )
        except NotesOrganizationRepositoryError as exc:
            return record_conflict(envelope, conflict_type=exc.reason_code)
        except (TypeError, ValueError):
            return self._rejected("notes_organization_envelope_invalid")
        except Exception:
            return self._rejected("notes_organization_apply_failed")

        if result.status == "applied":
            return {"status": "applied"}
        if result.status in {"duplicate", "stale"}:
            return {"status": "noop", "reason": result.status}
        return record_conflict(
            envelope,
            conflict_type=result.reason_code or "notes_organization_apply_blocked",
        )

    @staticmethod
    def _rejected(error_code: str) -> dict[str, Any]:
        return {"status": "rejected", "error_code": error_code}


__all__ = ["NotesOrganizationSyncAdapter"]
