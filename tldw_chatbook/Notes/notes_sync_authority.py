"""Private note-authority models for durable lasting-sync execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections.abc import Mapping
from datetime import datetime

from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_models import (
    validate_notes_sync_digest,
    validate_notes_sync_opaque_id,
    validate_notes_sync_reason_code,
)


class NotesSyncAuthorityError(RuntimeError):
    """A bounded note-authority refusal."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = validate_notes_sync_reason_code(reason_code)
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncNoteSnapshot:
    note_scope_id: str
    note_id: str
    title: str
    content: str
    version: int
    content_digest: str
    updated_at: str | None = None

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(
            self.note_scope_id,
            field_name="note_scope_id",
        )
        validate_notes_sync_opaque_id(self.note_id, field_name="note_id")
        if type(self.title) is not str or type(self.content) is not str:
            raise TypeError("note title and content must be strings.")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("note version must be non-negative.")
        validate_notes_sync_digest(
            self.content_digest,
            field_name="content_digest",
        )
        if self.content_digest != _content_digest(self.content):
            raise ValueError("content_digest must match note content.")
        if self.updated_at is not None:
            if (
                type(self.updated_at) is not str
                or not self.updated_at
                or len(self.updated_at) > 64
                or "\n" in self.updated_at
            ):
                raise ValueError("updated_at must be bounded ISO-8601 text.")
            try:
                datetime.fromisoformat(self.updated_at.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError("updated_at must be bounded ISO-8601 text.") from None

    def __repr__(self) -> str:
        return "NotesSyncNoteSnapshot(<private>)"


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _bounded_reason(error: Exception, fallback: str) -> str:
    reason = str(error)
    if type(error) is RuntimeError and reason in {
        "note_missing",
        "note_verification_failed",
        "server_contract_missing",
        "stale_note",
    }:
        return reason
    return fallback


class NotesScopeSyncAuthority:
    """Lasting-sync note operations routed through NotesScopeService."""

    def __init__(
        self,
        service: NotesScopeService,
        *,
        scope: ScopeType,
        user_id: str | None = None,
        note_scope_id: str | None = None,
    ) -> None:
        if type(service) is not NotesScopeService:
            raise TypeError("service must be a NotesScopeService.")
        if type(scope) is not ScopeType:
            raise TypeError("scope must be a ScopeType.")
        self._service = service
        self._scope = scope
        self._user_id = user_id
        self._note_scope_id = note_scope_id or scope.value
        validate_notes_sync_opaque_id(
            self._note_scope_id,
            field_name="note_scope_id",
        )

    async def observe(self, note_id: str) -> NotesSyncNoteSnapshot:
        try:
            record = await self._service.get_note_for_sync(
                scope=self._scope,
                note_id=note_id,
                user_id=self._user_id,
            )
        except Exception as exc:
            raise NotesSyncAuthorityError(
                _bounded_reason(exc, "note_observation_failed")
            ) from None
        return self._snapshot(record, expected_note_id=note_id)

    async def replace(
        self,
        expected: NotesSyncNoteSnapshot,
        *,
        title: str,
        content: str,
    ) -> NotesSyncNoteSnapshot:
        if type(expected) is not NotesSyncNoteSnapshot:
            raise TypeError("expected must be a NotesSyncNoteSnapshot.")
        if expected.note_scope_id != self._note_scope_id:
            raise NotesSyncAuthorityError("note_scope_changed")
        try:
            record = await self._service.replace_note_for_sync(
                scope=self._scope,
                note_id=expected.note_id,
                title=title,
                content=content,
                expected_version=expected.version,
                user_id=self._user_id,
            )
        except Exception as exc:
            raise NotesSyncAuthorityError(
                _bounded_reason(exc, "note_mutation_failed")
            ) from None
        observed = self._snapshot(record, expected_note_id=expected.note_id)
        if (
            observed.version != expected.version + 1
            or observed.title != title
            or observed.content != content
        ):
            raise NotesSyncAuthorityError("note_postcondition_failed")
        return observed

    async def create(
        self,
        *,
        note_id: str,
        title: str,
        content: str,
    ) -> NotesSyncNoteSnapshot:
        """Create one reviewed caller-identified note through the service."""

        try:
            record = await self._service.create_note_for_sync(
                scope=self._scope,
                note_id=note_id,
                title=title,
                content=content,
                user_id=self._user_id,
            )
        except Exception as exc:
            raise NotesSyncAuthorityError(
                _bounded_reason(exc, "note_mutation_failed")
            ) from None
        observed = self._snapshot(record, expected_note_id=note_id)
        if observed.title != title or observed.content != content:
            raise NotesSyncAuthorityError("note_postcondition_failed")
        return observed

    async def delete(self, expected: NotesSyncNoteSnapshot) -> None:
        """Soft-delete exactly one reviewed note through the scope service."""

        if type(expected) is not NotesSyncNoteSnapshot:
            raise TypeError("expected must be a NotesSyncNoteSnapshot.")
        if expected.note_scope_id != self._note_scope_id:
            raise NotesSyncAuthorityError("note_scope_changed")
        try:
            record = await self._service.delete_note_for_sync(
                scope=self._scope,
                note_id=expected.note_id,
                expected_version=expected.version,
                user_id=self._user_id,
            )
        except Exception as exc:
            raise NotesSyncAuthorityError(
                _bounded_reason(exc, "note_mutation_failed")
            ) from None
        if str(record.get("id")) != expected.note_id or not record.get("deleted"):
            raise NotesSyncAuthorityError("note_postcondition_failed")

    async def reconcile_managed_memberships(
        self,
        *,
        owner_id: str,
        desired: tuple[tuple[str, str], ...],
    ) -> None:
        """Converge one owner's complete placement set through the service."""

        try:
            await self._service.reconcile_note_folder_owner_memberships(
                scope=self._scope,
                owner_id=owner_id,
                desired=desired,
                user_id=self._user_id,
            )
        except Exception as exc:
            raise NotesSyncAuthorityError(
                _bounded_reason(exc, "membership_mutation_failed")
            ) from None

    def _snapshot(
        self,
        record: Mapping[str, object],
        *,
        expected_note_id: str,
    ) -> NotesSyncNoteSnapshot:
        note_id = str(record.get("id") or "")
        if note_id != expected_note_id:
            raise NotesSyncAuthorityError("note_identity_changed")
        if bool(record.get("deleted", False)):
            raise NotesSyncAuthorityError("note_missing")
        try:
            title = record["title"]
            content = record["content"]
            version = record["version"]
        except KeyError:
            raise NotesSyncAuthorityError("note_observation_invalid") from None
        if (
            type(title) is not str
            or type(content) is not str
            or type(version) is not int
        ):
            raise NotesSyncAuthorityError("note_observation_invalid")
        updated_at = (
            record.get("updated_at")
            if "updated_at" in record
            else record.get("last_modified")
        )
        if updated_at is not None and type(updated_at) is not str:
            raise NotesSyncAuthorityError("note_observation_invalid")
        return NotesSyncNoteSnapshot(
            note_scope_id=self._note_scope_id,
            note_id=note_id,
            title=title,
            content=content,
            version=version,
            content_digest=_content_digest(content),
            updated_at=updated_at,
        )


__all__ = [
    "NotesScopeSyncAuthority",
    "NotesSyncAuthorityError",
    "NotesSyncNoteSnapshot",
]
