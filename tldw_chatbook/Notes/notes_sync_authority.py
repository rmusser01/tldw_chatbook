"""Private note-authority models for durable lasting-sync execution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from tldw_chatbook.Notes.note_folder_models import (
    NoteFolder,
    NoteFolderMembership,
    join_normalized_folder_path,
    normalize_folder_name,
)
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
class ManualFolderRequest:
    """Request for deterministic creation or verification of a manual folder.

    Attributes:
        folder_id: Deterministic opaque identifier for the folder.
        parent_id: Optional opaque identifier of the parent folder.
        name: Display name of the requested folder.
        path_segments: Ordered folder path used for winner verification.
    """

    folder_id: str
    parent_id: str | None
    name: str
    path_segments: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.folder_id, field_name="folder_id")
        if self.parent_id is not None:
            validate_notes_sync_opaque_id(self.parent_id, field_name="parent_id")
        if type(self.path_segments) is not tuple or not self.path_segments:
            raise TypeError("path_segments must be a non-empty tuple.")
        normalized = tuple(normalize_folder_name(value) for value in self.path_segments)
        name = normalize_folder_name(self.name)
        if normalized[-1].key != name.key:
            raise ValueError("name must match the final path segment.")

    def __repr__(self) -> str:
        return "ManualFolderRequest(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedFolder:
    """Verified identity and version of one active manual folder.

    Attributes:
        folder_id: Opaque identifier of the verified folder.
        parent_id: Optional opaque identifier of its parent folder.
        normalized_path: Canonical normalized folder path.
        version: Persisted folder version observed during verification.
    """

    folder_id: str
    parent_id: str | None
    normalized_path: str
    version: int


@dataclass(frozen=True, slots=True, repr=False)
class ConflictNoteRequest:
    """Request for deterministic creation or verification of a conflict note.

    Attributes:
        note_id: Deterministic opaque identifier for the conflict copy.
        title: Bounded title of the conflict copy.
        content: Exact content the conflict copy must preserve.
    """

    note_id: str
    title: str
    content: str

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.note_id, field_name="note_id")
        if type(self.title) is not str or type(self.content) is not str:
            raise TypeError("title and content must be strings.")
        if not self.title or len(self.title) > 4096 or "\x00" in self.title:
            raise ValueError("title must be bounded non-empty text.")

    def __repr__(self) -> str:
        return "ConflictNoteRequest(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class ManualPlacementRequest:
    """Request for deterministic creation or verification of a placement.

    Attributes:
        folder_id: Opaque identifier of the target manual folder.
        note_id: Opaque identifier of the note to place.
        expected_note_version: Note version authorized for the placement.
    """

    folder_id: str
    note_id: str
    expected_note_version: int

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.folder_id, field_name="folder_id")
        validate_notes_sync_opaque_id(self.note_id, field_name="note_id")
        if (
            type(self.expected_note_version) is not int
            or self.expected_note_version < 0
        ):
            raise ValueError("expected_note_version must be non-negative.")

    def __repr__(self) -> str:
        return "ManualPlacementRequest(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedPlacement:
    """Verified identity and version of one active manual placement.

    Attributes:
        membership_id: Opaque identifier of the folder membership.
        folder_id: Opaque identifier of the containing folder.
        note_id: Opaque identifier of the placed note.
        version: Persisted membership version observed during verification.
    """

    membership_id: str
    folder_id: str
    note_id: str
    version: int


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncNoteSnapshot:
    """Private note-authority snapshot used for lasting-sync decisions.

    Attributes:
        note_scope_id: Opaque identifier of the note authority scope.
        note_id: Opaque identifier of the note.
        title: Exact observed note title.
        content: Exact observed note content.
        version: Persisted note version.
        content_digest: Digest that must match ``content``.
        updated_at: Optional ISO-8601 update timestamp.
    """

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


def _normalized_folder_path(path_segments: tuple[str, ...]) -> str:
    path = ""
    for segment in path_segments:
        path = join_normalized_folder_path(path, normalize_folder_name(segment).key)
    return path


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

    async def create_or_verify_manual_folder(
        self,
        request: ManualFolderRequest,
    ) -> VerifiedFolder:
        """Create at most one local manual folder, or verify the exact winner."""

        if type(request) is not ManualFolderRequest:
            raise TypeError("request must be a ManualFolderRequest.")
        self._require_local_copy_scope()
        expected_path = _normalized_folder_path(request.path_segments)
        deterministic = await self._read_folder_id(request.folder_id)
        if deterministic is not None and not self._folder_matches(
            deterministic, request, expected_path
        ):
            raise NotesSyncAuthorityError("folder_authority_changed")
        existing = await self._read_folder_path(request.path_segments)
        if existing is not None:
            return await self._verified_folder(existing, request, expected_path)
        if deterministic is not None:
            return await self._verified_folder(deterministic, request, expected_path)
        try:
            await self._service.create_manual_note_folder_for_sync(
                scope=self._scope,
                folder_id=request.folder_id,
                name=normalize_folder_name(request.name).display,
                parent_id=request.parent_id,
                user_id=self._user_id,
            )
        except Exception:
            pass
        winner = await self._read_folder_path(request.path_segments)
        if winner is None:
            winner = await self._read_folder_id(request.folder_id)
        if winner is None:
            raise NotesSyncAuthorityError("folder_mutation_failed")
        return await self._verified_folder(winner, request, expected_path)

    async def create_or_verify_conflict_note(
        self,
        request: ConflictNoteRequest,
    ) -> NotesSyncNoteSnapshot:
        """Create at most one local conflict copy, or verify its exact winner."""

        if type(request) is not ConflictNoteRequest:
            raise TypeError("request must be a ConflictNoteRequest.")
        self._require_local_copy_scope()
        existing = await self._read_conflict_note(request.note_id)
        if existing is not None:
            return self._verify_conflict_note(existing, request)
        try:
            await self._service.create_note_for_sync(
                scope=self._scope,
                note_id=request.note_id,
                title=request.title,
                content=request.content,
                user_id=self._user_id,
            )
        except Exception:
            pass
        winner = await self._read_conflict_note(request.note_id)
        if winner is None:
            raise NotesSyncAuthorityError("note_mutation_failed")
        return self._verify_conflict_note(winner, request)

    async def verify_manual_folder(
        self,
        request: ManualFolderRequest,
        expected: VerifiedFolder,
    ) -> VerifiedFolder:
        """Read and verify one checkpointed active manual folder identity."""

        if type(request) is not ManualFolderRequest:
            raise TypeError("request must be a ManualFolderRequest.")
        if type(expected) is not VerifiedFolder:
            raise TypeError("expected must be a VerifiedFolder.")
        self._require_local_copy_scope()
        deterministic = await self._read_folder_id(request.folder_id)
        by_id = await self._read_folder_id(expected.folder_id)
        by_path = await self._read_folder_path(request.path_segments)
        if (
            by_id is None
            or by_path is None
            or by_path.folder_id != expected.folder_id
            or (
                deterministic is not None
                and deterministic.folder_id != expected.folder_id
            )
        ):
            raise NotesSyncAuthorityError("folder_authority_changed")
        verified = await self._verified_folder(
            by_id,
            request,
            _normalized_folder_path(request.path_segments),
        )
        if (
            verified.folder_id,
            verified.parent_id,
            verified.version,
        ) != (
            expected.folder_id,
            expected.parent_id,
            expected.version,
        ):
            raise NotesSyncAuthorityError("folder_authority_changed")
        return verified

    async def create_or_verify_manual_placement(
        self,
        request: ManualPlacementRequest,
    ) -> VerifiedPlacement:
        """Create at most one local manual placement, or verify its exact winner."""

        if type(request) is not ManualPlacementRequest:
            raise TypeError("request must be a ManualPlacementRequest.")
        self._require_local_copy_scope()
        existing = await self._read_manual_placement(request)
        if existing is not None:
            membership, deleted = existing
            if deleted:
                raise NotesSyncAuthorityError("placement_authority_changed")
            return self._verified_placement(membership, request)
        try:
            await self._service.create_manual_note_placement_for_sync(
                scope=self._scope,
                folder_id=request.folder_id,
                note_id=request.note_id,
                expected_note_version=request.expected_note_version,
                user_id=self._user_id,
            )
        except Exception:
            pass
        winner = await self._read_manual_placement(request)
        if winner is None or winner[1]:
            raise NotesSyncAuthorityError("placement_mutation_failed")
        return self._verified_placement(winner[0], request)

    async def verify_conflict_note(
        self, request: ConflictNoteRequest
    ) -> NotesSyncNoteSnapshot:
        """Read and verify one exact active conflict-copy note."""

        existing = await self._read_conflict_note(request.note_id)
        if existing is None:
            raise NotesSyncAuthorityError("conflict_copy_collision")
        return self._verify_conflict_note(existing, request)

    async def verify_manual_placement(
        self, request: ManualPlacementRequest
    ) -> VerifiedPlacement:
        """Read and verify one exact active manual placement."""

        existing = await self._read_manual_placement(request)
        if existing is None or existing[1]:
            raise NotesSyncAuthorityError("placement_authority_changed")
        return self._verified_placement(existing[0], request)

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

    def _require_local_copy_scope(self) -> None:
        if self._scope is not ScopeType.LOCAL_NOTE:
            raise NotesSyncAuthorityError("server_contract_missing")

    async def _read_folder_id(self, folder_id: str) -> NoteFolder | None:
        try:
            return await self._service.get_note_folder_by_id_for_sync(
                scope=self._scope,
                folder_id=folder_id,
                include_deleted=True,
                user_id=self._user_id,
            )
        except Exception:
            raise NotesSyncAuthorityError("folder_observation_failed") from None

    async def _read_folder_path(
        self, path_segments: tuple[str, ...]
    ) -> NoteFolder | None:
        try:
            return await self._service.get_note_folder_by_path_for_sync(
                scope=self._scope,
                path_segments=path_segments,
                user_id=self._user_id,
            )
        except Exception:
            raise NotesSyncAuthorityError("folder_observation_failed") from None

    @staticmethod
    def _folder_matches(
        folder: NoteFolder,
        request: ManualFolderRequest,
        expected_path: str,
    ) -> bool:
        return (
            not folder.deleted
            and folder.parent_id == request.parent_id
            and folder.normalized_path == expected_path
            and normalize_folder_name(folder.name).key
            == normalize_folder_name(request.name).key
        )

    async def _verified_folder(
        self,
        folder: NoteFolder,
        request: ManualFolderRequest,
        expected_path: str,
    ) -> VerifiedFolder:
        try:
            managed = await self._service.has_managed_note_folder_ownership_for_sync(
                scope=self._scope,
                folder_id=folder.folder_id,
                user_id=self._user_id,
            )
        except Exception:
            raise NotesSyncAuthorityError("folder_observation_failed") from None
        if managed or not self._folder_matches(folder, request, expected_path):
            raise NotesSyncAuthorityError("folder_authority_changed")
        return VerifiedFolder(
            folder_id=folder.folder_id,
            parent_id=folder.parent_id,
            normalized_path=folder.normalized_path,
            version=folder.version,
        )

    async def _read_conflict_note(self, note_id: str) -> NotesSyncNoteSnapshot | None:
        try:
            record = await self._service.get_note_for_sync(
                scope=self._scope,
                note_id=note_id,
                user_id=self._user_id,
            )
        except Exception as exc:
            if type(exc) is RuntimeError and str(exc) == "note_missing":
                return None
            raise NotesSyncAuthorityError("note_observation_failed") from None
        if bool(record.get("deleted", False)):
            raise NotesSyncAuthorityError("conflict_copy_collision")
        return self._snapshot(record, expected_note_id=note_id)

    @staticmethod
    def _verify_conflict_note(
        note: NotesSyncNoteSnapshot,
        request: ConflictNoteRequest,
    ) -> NotesSyncNoteSnapshot:
        if note.title != request.title or note.content != request.content:
            raise NotesSyncAuthorityError("conflict_copy_collision")
        return note

    async def _read_manual_placement(
        self,
        request: ManualPlacementRequest,
    ) -> tuple[NoteFolderMembership, bool] | None:
        try:
            placements = await self._service.list_note_placements_for_sync(
                scope=self._scope,
                note_id=request.note_id,
                user_id=self._user_id,
            )
            exact = await self._service.get_manual_note_placement_for_sync(
                scope=self._scope,
                folder_id=request.folder_id,
                note_id=request.note_id,
                include_deleted=True,
                user_id=self._user_id,
            )
        except Exception:
            raise NotesSyncAuthorityError("placement_observation_failed") from None
        if any(
            placement.folder_id != request.folder_id
            or placement.ownership != "manual"
            or placement.owner_id
            or not placement.owner_active
            for placement in placements
        ):
            raise NotesSyncAuthorityError("placement_authority_changed")
        if (
            exact is not None
            and not exact[1]
            and (
                len(placements) != 1
                or placements[0].membership_id != exact[0].membership_id
            )
        ):
            raise NotesSyncAuthorityError("placement_authority_changed")
        return exact

    @staticmethod
    def _verified_placement(
        membership: NoteFolderMembership,
        request: ManualPlacementRequest,
    ) -> VerifiedPlacement:
        if (
            membership.folder_id != request.folder_id
            or membership.note_id != request.note_id
            or membership.ownership != "manual"
            or membership.owner_id
            or not membership.owner_active
        ):
            raise NotesSyncAuthorityError("placement_authority_changed")
        return VerifiedPlacement(
            membership_id=membership.membership_id,
            folder_id=membership.folder_id,
            note_id=membership.note_id,
            version=membership.version,
        )

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
        if type(updated_at) is datetime:
            updated_at = updated_at.isoformat()
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
    "ConflictNoteRequest",
    "ManualFolderRequest",
    "ManualPlacementRequest",
    "NotesScopeSyncAuthority",
    "NotesSyncAuthorityError",
    "NotesSyncNoteSnapshot",
    "VerifiedFolder",
    "VerifiedPlacement",
]
