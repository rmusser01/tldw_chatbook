"""Durable, recovery-first execution for reviewed lasting-sync actions."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateError,
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncOperationRecord,
    NotesSyncRecoveryRecord,
)
from tldw_chatbook.Notes.notes_sync_authority import (
    NotesSyncAuthorityError,
    NotesSyncNoteSnapshot,
)
from tldw_chatbook.Notes.notes_sync_filesystem import (
    NotesSyncFilesystemError,
    NotesSyncFilesystemPartialError,
    NotesSyncFileSnapshot,
    NotesSyncPrivateCleanupHandle,
    WindowsNotesSyncObservation,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncFileIdentity,
    NotesSyncFileObservation,
    NotesSyncOperationState,
    NotesSyncRootState,
    NotesSyncSerializationProfile,
    normalize_notes_sync_relative_path,
    validate_notes_sync_opaque_id,
    validate_notes_sync_reason_code,
)
from tldw_chatbook.Notes.sync_paths import SafeSyncBytes, SafeSyncFileIdentity


CONFLICT_RECOVERY_RETENTION_NS = 30 * 24 * 60 * 60 * 1_000_000_000
_RESOLUTION_JOURNAL_ACTIONS = {
    "resolve_keep_file": NotesSyncActionKind.UPDATE_NOTE,
    "resolve_keep_note": NotesSyncActionKind.UPDATE_FILE,
}


class NotesSyncRecoveryChoice(StrEnum):
    """Bounded choices exposed for an unresolved operation."""

    RESUME = "resume"
    RESTORE = "restore"
    DISCONNECT = "disconnect"


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncDirectionOverride:
    """Private provenance for one reviewed out-of-direction conflict choice."""

    review_id: str
    action_kind: NotesSyncActionKind
    observation_token: str

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.review_id, field_name="review_id")
        validate_notes_sync_opaque_id(
            self.observation_token,
            field_name="observation_token",
        )
        if type(self.action_kind) is not NotesSyncActionKind:
            raise TypeError("action_kind must be a NotesSyncActionKind.")

    def __repr__(self) -> str:
        return "NotesSyncDirectionOverride(<private>)"


class NotesSyncExecutionPartialError(RuntimeError):
    """A committed cleanup authority could not be made durable."""

    def __init__(
        self,
        reason_code: str,
        cleanup_handle: NotesSyncPrivateCleanupHandle,
    ) -> None:
        self.reason_code = validate_notes_sync_reason_code(reason_code)
        if type(cleanup_handle) is not NotesSyncPrivateCleanupHandle:
            raise TypeError("cleanup_handle must be private cleanup authority.")
        self.cleanup_handle = cleanup_handle
        super().__init__(reason_code)

    def __repr__(self) -> str:
        return "NotesSyncExecutionPartialError(<private>)"


def _encoded_override(
    override: NotesSyncDirectionOverride | None,
) -> dict[str, str] | None:
    if override is None:
        return None
    return {
        "action": override.action_kind.value,
        "observation_token": override.observation_token,
        "review_id": override.review_id,
    }


def _cleanup_padding(metadata: dict[str, object]) -> str:
    """Reserve the exact encoded delta for the worst cleanup intent."""

    relative_path = metadata.get("file_relative_path")
    reviewed_state = metadata.get("file_reviewed_state")
    if type(relative_path) is not str:
        raise RuntimeError("recovery_authority_changed")
    identity: object = None
    if isinstance(reviewed_state, dict):
        identity = reviewed_state.get("identity")
    if identity is None:
        identity = [18446744073709551615] * 3
    baseline = metadata.copy()
    baseline["cleanup_padding"] = ""
    baseline["cleanup_pending"] = False
    for key in (
        "cleanup_identity",
        "cleanup_relative_path",
        "cleanup_reason_code",
    ):
        baseline.pop(key, None)
    target = Path(relative_path)
    displaced = target.parent / f".{target.name}.tmp-{'f' * 32}"
    partial = baseline.copy()
    partial.pop("cleanup_padding", None)
    partial["cleanup_pending"] = True
    partial["cleanup_identity"] = identity
    partial["cleanup_relative_path"] = (
        displaced.parent / f".{displaced.name}.cleanup-{'f' * 32}"
    ).as_posix()
    partial["cleanup_reason_code"] = "replacement_cleanup_pending"
    return " " * max(
        0,
        len(_encode_recovery_intent(partial)) - len(_encode_recovery_intent(baseline)),
    )


def _encode_recovery_intent(metadata: dict[str, object]) -> bytes:
    """Encode the sole private recovery-intent representation."""

    return json.dumps(
        metadata,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


_ATTENTION_CHOICES = (
    NotesSyncRecoveryChoice.RESUME,
    NotesSyncRecoveryChoice.RESTORE,
    NotesSyncRecoveryChoice.DISCONNECT,
)
_INTERNAL_REASONS = frozenset(
    {
        "binding_authority_changed",
        "direction_disallows_action",
        "file_observation_failed",
        "folder_owner_missing",
        "operation_already_completed",
        "operation_needs_attention",
        "postcondition_failed",
        "recovery_authority_changed",
        "restore_postcondition_failed",
        "stale_observation",
        "stale_operation_token",
        "stale_restore_observation",
    }
)
_TYPED_REASON_CODES = frozenset(
    {
        "comparison_root_unavailable",
        "destination_exists",
        "deletion_postcondition_failed",
        "duplicate_stable_identity",
        "expected_path_mismatch",
        "file_observation_failed",
        "guarded_rename_unavailable",
        "invalid_relative_path",
        "link_or_reparse",
        "max_file_bytes_exceeded",
        "membership_mutation_failed",
        "missing_parent",
        "missing_target",
        "mixed_newlines",
        "move_commit_unverified",
        "move_postcondition_failed",
        "multiple_links",
        "non_directory_parent",
        "non_regular",
        "note_identity_changed",
        "note_missing",
        "note_mutation_failed",
        "note_observation_failed",
        "note_observation_invalid",
        "note_postcondition_failed",
        "note_scope_changed",
        "operation_failed",
        "parent_identity_changed",
        "replacement_cleanup_pending",
        "replacement_commit_unverified",
        "replacement_postcondition_failed",
        "replacement_raced_after_exchange",
        "replacement_rollback_failed",
        "replacement_rollback_unverified",
        "root_identity_changed",
        "root_link_or_reparse",
        "root_not_directory",
        "root_unavailable",
        "same_destination",
        "server_contract_missing",
        "stale_note",
        "target_changed_during_read",
        "target_identity_changed",
        "temporary_name_exhausted",
        "unsupported_encoding",
        "unsupported_metadata",
        "unsupported_newline",
        "unsupported_platform",
        "writable_adapter_unavailable",
    }
)
_FileSnapshot = NotesSyncFileSnapshot | WindowsNotesSyncObservation


def _file_relative_path(snapshot: _FileSnapshot) -> str:
    return (
        snapshot.relative_path
        if type(snapshot) is WindowsNotesSyncObservation
        else snapshot.observation.relative_path
    )


def _file_content_digest(snapshot: _FileSnapshot) -> str:
    return (
        snapshot.content_digest
        if type(snapshot) is WindowsNotesSyncObservation
        else snapshot.observation.content_digest
    )


def _file_serialization(snapshot: _FileSnapshot):
    return (
        snapshot.serialization
        if type(snapshot) is WindowsNotesSyncObservation
        else snapshot.observation.serialization
    )


def _file_representation_digest(snapshot: _FileSnapshot) -> str:
    return snapshot.representation_digest


def _encoded_reviewed_state(snapshot: NotesSyncFileSnapshot) -> dict[str, object]:
    state = snapshot.reviewed_state
    return {
        "identity": [
            state.identity.device,
            state.identity.inode,
            state.identity.link_count,
        ],
        "mode": state.mode,
        "size": state.size,
        "mtime_ns": state.mtime_ns,
        "ctime_ns": state.ctime_ns,
        "owner_user": state.owner_user,
        "owner_group": state.owner_group,
        "flags": state.flags,
        "extended_attributes": [
            [
                name,
                base64.b64encode(value).decode("ascii"),
            ]
            for name, value in state.extended_attributes
        ],
        "has_extended_acl": state.has_extended_acl,
    }


def _decoded_reviewed_state(
    relative_path: str,
    value: object,
    content: bytes,
) -> SafeSyncBytes:
    if not isinstance(value, dict):
        raise RuntimeError("recovery_authority_changed")
    try:
        identity = value["identity"]
        integer_fields = (
            "mode",
            "size",
            "mtime_ns",
            "ctime_ns",
            "owner_user",
            "owner_group",
            "flags",
        )
        raw_attributes = value["extended_attributes"]
        if (
            not isinstance(identity, list)
            or len(identity) != 3
            or any(type(item) is not int for item in identity)
            or any(type(value[field]) is not int for field in integer_fields)
            or type(value["has_extended_acl"]) is not bool
            or not isinstance(raw_attributes, list)
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not str
                for pair in raw_attributes
            )
        ):
            raise ValueError
        attributes = tuple(
            (
                pair[0],
                base64.b64decode(pair[1], validate=True),
            )
            for pair in raw_attributes
        )
        raw_identity = SafeSyncFileIdentity(
            device=identity[0],
            inode=identity[1],
            link_count=identity[2],
        )
        return SafeSyncBytes(
            relative_path=Path(relative_path),
            content=content,
            identity=raw_identity,
            mode=value["mode"],
            size=value["size"],
            mtime_ns=value["mtime_ns"],
            ctime_ns=value["ctime_ns"],
            owner_user=value["owner_user"],
            owner_group=value["owner_group"],
            flags=value["flags"],
            extended_attributes=attributes,
            has_extended_acl=value["has_extended_acl"],
        )
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("recovery_authority_changed") from None


def _logical_text(payload: bytes, profile: NotesSyncSerializationProfile) -> str:
    encoded = payload[3:] if profile.utf8_bom else payload
    try:
        return encoded.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError:
        raise RuntimeError("recovery_authority_changed") from None


def _represented_bytes(text: str, profile: NotesSyncSerializationProfile) -> bytes:
    logical = text.replace("\r\n", "\n").replace("\r", "\n")
    if profile.final_newline and not logical.endswith("\n"):
        logical += "\n"
    elif not profile.final_newline:
        logical = logical.rstrip("\n")
    represented = (
        logical.replace("\n", "\r\n") if profile.newline == "crlf" else logical
    )
    encoded = represented.encode("utf-8")
    return (b"\xef\xbb\xbf" + encoded) if profile.utf8_bom else encoded


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncExecutionRequest:
    """Private reviewed authority bundle required to execute or resume."""

    operation_id: str
    root_id: str
    logical_folder_id: str
    direction: NotesSyncDirection
    binding_id: str
    observation_token: str
    action_kind: NotesSyncActionKind
    note: NotesSyncNoteSnapshot | None
    file: _FileSnapshot | None
    desired_title: str
    recovery_id: str
    recovery_expires_at: int
    journal_kind: str | None = None
    direction_override: NotesSyncDirectionOverride | None = None
    candidate_note_scope_id: str | None = None
    candidate_note_id: str | None = None
    candidate_relative_path: str | None = None
    candidate_serialization: NotesSyncSerializationProfile | None = None
    move_destination_relative_path: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("operation_id", self.operation_id),
            ("root_id", self.root_id),
            ("logical_folder_id", self.logical_folder_id),
            ("binding_id", self.binding_id),
            ("observation_token", self.observation_token),
            ("recovery_id", self.recovery_id),
        ):
            validate_notes_sync_opaque_id(value, field_name=name)
        if type(self.action_kind) is not NotesSyncActionKind:
            raise TypeError("action_kind must be a NotesSyncActionKind.")
        if type(self.direction) is not NotesSyncDirection:
            raise TypeError("direction must be a NotesSyncDirection.")
        if self.action_kind not in {
            NotesSyncActionKind.CREATE_NOTE,
            NotesSyncActionKind.UPDATE_NOTE,
            NotesSyncActionKind.CREATE_FILE,
            NotesSyncActionKind.UPDATE_FILE,
            NotesSyncActionKind.MOVE_FILE,
        }:
            raise ValueError("action_kind is not executable by this boundary.")
        if self.action_kind is NotesSyncActionKind.CREATE_NOTE:
            if self.note is not None or type(self.file) not in {
                NotesSyncFileSnapshot,
                WindowsNotesSyncObservation,
            }:
                raise TypeError("create_note requires only file authority.")
            for name, value in (
                ("candidate_note_scope_id", self.candidate_note_scope_id),
                ("candidate_note_id", self.candidate_note_id),
            ):
                validate_notes_sync_opaque_id(value, field_name=name)
        elif self.action_kind is NotesSyncActionKind.CREATE_FILE:
            if type(self.note) is not NotesSyncNoteSnapshot or self.file is not None:
                raise TypeError("create_file requires only note authority.")
            normalize_notes_sync_relative_path(self.candidate_relative_path or "")
            if type(self.candidate_serialization) is not NotesSyncSerializationProfile:
                raise TypeError("candidate_serialization is required for create_file.")
        else:
            if type(self.note) is not NotesSyncNoteSnapshot:
                raise TypeError("note must be a NotesSyncNoteSnapshot.")
            if type(self.file) not in {
                NotesSyncFileSnapshot,
                WindowsNotesSyncObservation,
            }:
                raise TypeError("file must be a supported private file observation.")
        if self.action_kind is NotesSyncActionKind.MOVE_FILE:
            normalize_notes_sync_relative_path(
                self.move_destination_relative_path or ""
            )
        if (
            self.action_kind
            in {NotesSyncActionKind.UPDATE_FILE, NotesSyncActionKind.MOVE_FILE}
            and type(self.file) is WindowsNotesSyncObservation
        ):
            raise ValueError("Windows observations do not grant file write authority.")
        if (
            type(self.desired_title) is not str
            or not self.desired_title
            or len(self.desired_title) > 4096
            or "\x00" in self.desired_title
        ):
            raise ValueError("desired_title must be bounded non-empty text.")
        if type(self.recovery_expires_at) is not int or self.recovery_expires_at <= 0:
            raise ValueError("recovery_expires_at must be positive.")
        if self.journal_kind is not None and (
            _RESOLUTION_JOURNAL_ACTIONS.get(self.journal_kind) is not self.action_kind
        ):
            raise ValueError("journal_kind must match the reviewed conflict action.")
        if self.direction_override is not None:
            if type(self.direction_override) is not NotesSyncDirectionOverride:
                raise TypeError(
                    "direction_override must be a NotesSyncDirectionOverride or None."
                )
            if (
                self.direction_override.action_kind is not self.action_kind
                or self.direction_override.observation_token != self.observation_token
            ):
                raise ValueError(
                    "direction_override must match action_kind and observation_token."
                )

    def __repr__(self) -> str:
        return "NotesSyncExecutionRequest(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncExecutionResult:
    """Path-, content-, and hash-free result of one executor call."""

    operation_id: str
    state: NotesSyncOperationState
    reason_code: str | None = None
    choices: tuple[NotesSyncRecoveryChoice, ...] = ()

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.operation_id, field_name="operation_id")
        if type(self.state) is not NotesSyncOperationState:
            raise TypeError("state must be a NotesSyncOperationState.")
        validate_notes_sync_reason_code(self.reason_code)
        if type(self.choices) is not tuple or any(
            type(choice) is not NotesSyncRecoveryChoice for choice in self.choices
        ):
            raise TypeError("choices must be a tuple of NotesSyncRecoveryChoice.")
        expected = (
            _ATTENTION_CHOICES
            if self.state is NotesSyncOperationState.NEEDS_ATTENTION
            else ()
        )
        if self.choices != expected:
            raise ValueError("choices must match the durable operation state.")
        if self.state is NotesSyncOperationState.NEEDS_ATTENTION:
            if self.reason_code is None:
                raise ValueError("reason_code is required for attention.")
        elif self.reason_code is not None:
            raise ValueError("reason_code is only valid for attention.")

    def __repr__(self) -> str:
        return f"NotesSyncExecutionResult(state={self.state!r})"


class _NoteAuthority(Protocol):
    async def observe(self, note_id: str) -> NotesSyncNoteSnapshot: ...

    async def replace(
        self,
        expected: NotesSyncNoteSnapshot,
        *,
        title: str,
        content: str,
    ) -> NotesSyncNoteSnapshot: ...

    async def create(
        self,
        *,
        note_id: str,
        title: str,
        content: str,
    ) -> NotesSyncNoteSnapshot: ...

    async def delete(self, expected: NotesSyncNoteSnapshot) -> None: ...

    async def reconcile_managed_memberships(
        self,
        *,
        owner_id: str,
        desired: tuple[tuple[str, str], ...],
    ) -> None: ...


class _Filesystem(Protocol):
    def observe(self, *args: str) -> _FileSnapshot | tuple[_FileSnapshot, ...]: ...

    def replace(
        self,
        relative_path: str,
        text: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot: ...

    def create(
        self,
        relative_path: str,
        text: str,
        *,
        profile: NotesSyncSerializationProfile,
    ) -> NotesSyncFileSnapshot: ...

    def delete(self, *, expected: NotesSyncFileSnapshot) -> None: ...

    def move(
        self,
        destination_path: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot: ...

    def resolve_cleanup(self, handle: NotesSyncPrivateCleanupHandle) -> None: ...


class NotesSyncExecutor:
    """Advance one reviewed operation through durable recovery stages.

    Callers must hold the root lease across store instances; executors sharing
    one store coalesce same-operation calls in process.
    """

    def __init__(
        self,
        store: NotesDeviceStateStore,
        note_authority: _NoteAuthority,
        filesystem: _Filesystem,
        *,
        recovery_capacity_bytes: int,
        after_stage: Callable[[NotesSyncOperationState], None] | None = None,
    ) -> None:
        if type(store) is not NotesDeviceStateStore:
            raise TypeError("store must be a NotesDeviceStateStore.")
        if type(recovery_capacity_bytes) is not int or recovery_capacity_bytes <= 0:
            raise ValueError("recovery_capacity_bytes must be positive.")
        self._store = store
        self._notes = note_authority
        self._filesystem = filesystem
        self._capacity = recovery_capacity_bytes
        self._after_stage = after_stage

    @staticmethod
    def stable_identity_digest(snapshot: _FileSnapshot) -> str:
        """Return the device-private binding fingerprint for one stable identity."""

        if type(snapshot) is WindowsNotesSyncObservation:
            return snapshot.stable_identity_digest
        if type(snapshot) is not NotesSyncFileSnapshot:
            raise TypeError("snapshot must be a supported private file observation.")
        identity = snapshot.observation.identity
        value = f"{identity.device}\0{identity.inode}".encode("ascii")
        return hashlib.sha256(value).hexdigest()

    async def execute(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        """Execute a new reviewed operation or return its durable result."""

        return await self._serialized(
            request.operation_id,
            lambda: self._run(request, allow_attention=False),
        )

    async def resume(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        """Resume only when current authorities match reviewed or intended state."""

        return await self._serialized(
            request.operation_id,
            lambda: self._run(request, allow_attention=True),
        )

    async def reconstruct_request(
        self,
        operation_id: str,
    ) -> NotesSyncExecutionRequest:
        """Rebuild one private request from durable intent and fresh authorities."""

        operation = self._store.get_operation(operation_id)
        resolution_action = _RESOLUTION_JOURNAL_ACTIONS.get(operation.kind)
        try:
            operation_action = resolution_action or NotesSyncActionKind(operation.kind)
        except ValueError:
            raise RuntimeError("recovery_authority_changed") from None
        if operation_action in {
            NotesSyncActionKind.CREATE_NOTE,
            NotesSyncActionKind.CREATE_FILE,
            NotesSyncActionKind.MOVE_FILE,
        }:
            return await self._reconstruct_new_request(operation)
        if (
            operation.state is NotesSyncOperationState.COMPLETED
            or operation.binding_id is None
            or operation.expected_note_version is None
            or operation.expected_file_digest is None
        ):
            raise RuntimeError("operation_already_completed")
        recovery = self._store.load_operation_recovery(operation_id)
        metadata = self._recovery_metadata(recovery)
        try:
            action = NotesSyncActionKind(metadata["action"])
            direction = NotesSyncDirection(metadata["direction"])
            logical_folder_id = self._required_metadata_text(
                metadata, "logical_folder_id"
            )
            note_scope_id = self._required_metadata_text(metadata, "note_scope_id")
            note_id = self._required_metadata_text(metadata, "note_id")
            relative_path = self._required_metadata_text(metadata, "file_relative_path")
            desired_title = self._required_metadata_text(metadata, "desired_title")
            recovery_title = self._required_metadata_text(metadata, "recovery_title")
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("recovery_authority_changed") from None
        if action is not operation_action or metadata.get("underlying_action_kind") != (
            action.value if resolution_action is not None else None
        ):
            raise RuntimeError("recovery_authority_changed")
        raw_override = metadata.get("direction_override")
        try:
            direction_override = (
                NotesSyncDirectionOverride(
                    review_id=raw_override["review_id"],
                    action_kind=NotesSyncActionKind(raw_override["action"]),
                    observation_token=raw_override["observation_token"],
                )
                if isinstance(raw_override, dict)
                else None
            )
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("recovery_authority_changed") from None
        current_note = await asyncio.to_thread(
            lambda: asyncio.run(self._notes.observe(note_id))
        )
        current_file = await self._observe_reconstructed_file(
            relative_path,
            windows=metadata.get("windows_observation") is True,
        )
        binding = self._store.get_binding(operation.binding_id)
        reviewed_binding = metadata.get("binding")
        if not isinstance(reviewed_binding, dict):
            raise RuntimeError("recovery_authority_changed")
        if operation.state not in {
            NotesSyncOperationState.BINDING_UPDATED,
            NotesSyncOperationState.VERIFIED,
        } and not self._binding_matches_reviewed(binding, metadata):
            raise RuntimeError("recovery_authority_changed")
        if action is NotesSyncActionKind.UPDATE_NOTE:
            try:
                original_content = recovery.payload.decode("utf-8")
            except UnicodeDecodeError:
                raise RuntimeError("recovery_authority_changed") from None
            payload_digest = hashlib.sha256(recovery.payload).hexdigest()
            if (
                (
                    resolution_action is None
                    and (
                        payload_digest != reviewed_binding.get("content_digest")
                        or operation.expected_note_version
                        != reviewed_binding.get("note_version")
                    )
                )
                or self.stable_identity_digest(current_file)
                != reviewed_binding.get("stable_identity_digest")
                or _file_serialization(current_file) != binding.serialization
                or binding.serialization
                != self._decoded_binding_serialization(reviewed_binding)
            ):
                raise RuntimeError("recovery_authority_changed")
            note = NotesSyncNoteSnapshot(
                note_scope_id=note_scope_id,
                note_id=note_id,
                title=recovery_title,
                content=original_content,
                version=operation.expected_note_version,
                content_digest=payload_digest,
            )
            file = current_file
            if _file_representation_digest(file) != operation.expected_file_digest:
                raise RuntimeError("stale_observation")
        else:
            if type(current_file) is not NotesSyncFileSnapshot:
                raise RuntimeError("recovery_authority_changed")
            if (
                current_note.note_scope_id != note_scope_id
                or current_note.note_id != note_id
                or current_note.version != operation.expected_note_version
                or current_note.content_digest != metadata.get("desired_digest")
            ):
                raise RuntimeError("stale_observation")
            note = current_note
            file = self._reconstructed_original_file(
                relative_path,
                recovery.payload,
                operation.expected_file_digest,
                metadata,
            )
            if (
                hashlib.sha256(recovery.payload).hexdigest()
                != operation.expected_file_digest
                or (
                    resolution_action is None
                    and file.observation.content_digest
                    != reviewed_binding.get("content_digest")
                )
                or self.stable_identity_digest(file)
                != reviewed_binding.get("stable_identity_digest")
                or file.observation.serialization
                != self._decoded_binding_serialization(reviewed_binding)
            ):
                raise RuntimeError("recovery_authority_changed")
        return NotesSyncExecutionRequest(
            operation_id=operation.operation_id,
            root_id=operation.root_id,
            logical_folder_id=logical_folder_id,
            direction=direction,
            binding_id=operation.binding_id,
            observation_token=operation.observation_token,
            action_kind=action,
            note=note,
            file=file,
            desired_title=desired_title,
            recovery_id=recovery.recovery_id,
            recovery_expires_at=recovery.expires_at,
            journal_kind=(operation.kind if resolution_action is not None else None),
            direction_override=direction_override,
        )

    async def _reconstruct_new_request(
        self,
        operation: NotesSyncOperationRecord,
    ) -> NotesSyncExecutionRequest:
        if operation.state is NotesSyncOperationState.COMPLETED:
            raise RuntimeError("operation_already_completed")
        recovery = self._store.load_operation_recovery(operation.operation_id)
        metadata = self._recovery_metadata(recovery)
        try:
            action = NotesSyncActionKind(operation.kind)
            direction = NotesSyncDirection(metadata["direction"])
            logical_folder_id = self._required_metadata_text(
                metadata, "logical_folder_id"
            )
            note_scope_id = self._required_metadata_text(metadata, "note_scope_id")
            note_id = self._required_metadata_text(metadata, "note_id")
            file_path = self._required_metadata_text(metadata, "file_relative_path")
            desired_title = self._required_metadata_text(metadata, "desired_title")
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("recovery_authority_changed") from None
        raw_override = metadata.get("direction_override")
        try:
            direction_override = (
                NotesSyncDirectionOverride(
                    review_id=raw_override["review_id"],
                    action_kind=NotesSyncActionKind(raw_override["action"]),
                    observation_token=raw_override["observation_token"],
                )
                if isinstance(raw_override, dict)
                else None
            )
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("recovery_authority_changed") from None
        if direction_override is not None and (
            direction_override.action_kind is not action
            or direction_override.observation_token != operation.observation_token
        ):
            raise RuntimeError("recovery_authority_changed")
        note: NotesSyncNoteSnapshot | None
        file: _FileSnapshot | None
        candidate_profile: NotesSyncSerializationProfile | None = None
        if action is NotesSyncActionKind.CREATE_NOTE:
            windows_observation = metadata.get("windows_observation")
            windows = isinstance(windows_observation, dict)
            file = await self._observe_reconstructed_file(file_path, windows=windows)
            if (
                operation.expected_file_digest is None
                or _file_representation_digest(file) != operation.expected_file_digest
                or hashlib.sha256(recovery.payload).hexdigest()
                != operation.expected_file_digest
            ):
                raise RuntimeError("recovery_authority_changed")
            if windows and (
                type(file) is not WindowsNotesSyncObservation
                or file.stable_identity_digest
                != windows_observation.get("stable_identity_digest")
                or file.freshness_digest != windows_observation.get("freshness_digest")
            ):
                raise RuntimeError("recovery_authority_changed")
            if not windows and (
                type(file) is not NotesSyncFileSnapshot
                or file.reviewed_state
                != _decoded_reviewed_state(
                    file_path,
                    metadata.get("file_reviewed_state"),
                    recovery.payload,
                )
            ):
                raise RuntimeError("recovery_authority_changed")
            note = None
        elif action is NotesSyncActionKind.CREATE_FILE:
            note = await self._observe_note(note_id)
            if (
                note.note_scope_id != note_scope_id
                or note.note_id != note_id
                or operation.expected_note_version != note.version
                or note.content_digest != metadata.get("note_content_digest")
                or recovery.payload != b""
            ):
                raise RuntimeError("recovery_authority_changed")
            file = None
            candidate_profile = self._decoded_candidate_serialization(metadata)
        else:
            note = await self._observe_note(note_id)
            if (
                note.note_scope_id != note_scope_id
                or note.note_id != note_id
                or operation.expected_note_version != note.version
                or note.content_digest != metadata.get("note_content_digest")
            ):
                raise RuntimeError("recovery_authority_changed")
            file = self._reconstructed_original_file(
                file_path,
                recovery.payload,
                operation.expected_file_digest or "",
                metadata,
            )
            if (
                hashlib.sha256(recovery.payload).hexdigest()
                != operation.expected_file_digest
            ):
                raise RuntimeError("recovery_authority_changed")
        return NotesSyncExecutionRequest(
            operation_id=operation.operation_id,
            root_id=operation.root_id,
            logical_folder_id=logical_folder_id,
            direction=direction,
            binding_id=(
                operation.binding_id
                or self._required_metadata_text(metadata, "binding_id")
            ),
            observation_token=operation.observation_token,
            action_kind=action,
            note=note,
            file=file,
            desired_title=desired_title,
            recovery_id=recovery.recovery_id,
            recovery_expires_at=recovery.expires_at,
            direction_override=direction_override,
            candidate_note_scope_id=(note_scope_id if note is None else None),
            candidate_note_id=(note_id if note is None else None),
            candidate_relative_path=(file_path if file is None else None),
            candidate_serialization=candidate_profile,
            move_destination_relative_path=(
                metadata.get("move_destination_relative_path")
                if action is NotesSyncActionKind.MOVE_FILE
                else None
            ),
        )

    async def resolve_filesystem_cleanup(
        self,
        operation_id: str,
    ) -> NotesSyncExecutionResult:
        """Resolve one persisted private filesystem cleanup authority."""

        return await self._serialized(
            operation_id,
            lambda: self._resolve_filesystem_cleanup(operation_id),
        )

    async def _resolve_filesystem_cleanup(
        self,
        operation_id: str,
    ) -> NotesSyncExecutionResult:
        operation = self._store.get_operation(operation_id)
        if operation.state is not NotesSyncOperationState.NEEDS_ATTENTION:
            raise RuntimeError("operation_needs_attention")
        recovery = self._store.load_operation_recovery(operation_id)
        metadata = self._recovery_metadata(recovery)
        if metadata.get("cleanup_pending") is not True:
            raise RuntimeError("recovery_authority_changed")
        relative_path = metadata.get("cleanup_relative_path")
        cleanup_reason = metadata.get("cleanup_reason_code")
        cleanup_identity = self._decoded_cleanup_identity(
            metadata.get("cleanup_identity")
        )
        resolver = getattr(self._filesystem, "resolve_cleanup", None)
        if (
            type(relative_path) is not str
            or type(cleanup_reason) is not str
            or cleanup_reason != operation.reason_code
            or not callable(resolver)
        ):
            raise RuntimeError("recovery_authority_changed")
        relative_path = normalize_notes_sync_relative_path(relative_path)
        handle = NotesSyncPrivateCleanupHandle(
            relative_path,
            cleanup_reason,
            cleanup_identity,
        )
        try:
            _, cancelled = await self._joined_thread_call(lambda: resolver(handle))
        except NotesSyncFilesystemPartialError as error:
            self._set_cleanup_intent(metadata, error.cleanup_handle, error.reason_code)
            encoded = _encode_recovery_intent(metadata)
            self._store.mark_operation_partial_attention(
                operation_id,
                recovery.recovery_id,
                error.reason_code,
                encoded,
                capacity_bytes=self._capacity,
            )
            return self._result(
                operation_id,
                NotesSyncOperationState.NEEDS_ATTENTION,
                error.reason_code,
            )
        if cancelled:
            self._persist_attention_best_effort(
                operation_id, "cancelled_after_admission"
            )
            raise asyncio.CancelledError
        metadata.pop("cleanup_relative_path", None)
        metadata.pop("cleanup_reason_code", None)
        metadata.pop("cleanup_identity", None)
        metadata["cleanup_pending"] = False
        metadata["cleanup_padding"] = _cleanup_padding(metadata)
        encoded = _encode_recovery_intent(metadata)
        reason = operation.reason_code or "operation_needs_attention"
        self._store.mark_operation_partial_attention(
            operation_id,
            recovery.recovery_id,
            reason,
            encoded,
            capacity_bytes=self._capacity,
        )
        return self._result(
            operation_id,
            NotesSyncOperationState.NEEDS_ATTENTION,
            reason,
        )

    async def restore(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        """Restore the admitted authority, verify it, then disconnect the item."""

        return await self._serialized(
            request.operation_id,
            lambda: self._restore(request),
        )

    async def _restore(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        if type(request) is not NotesSyncExecutionRequest:
            raise TypeError("request must be a NotesSyncExecutionRequest.")
        if request.action_kind in {
            NotesSyncActionKind.CREATE_NOTE,
            NotesSyncActionKind.CREATE_FILE,
        }:
            return await self._restore_create(request)
        if request.action_kind is NotesSyncActionKind.MOVE_FILE:
            return await self._restore_move(request)
        admitted = False
        try:
            operation = self._store.get_operation(request.operation_id)
            admitted = True
            self._validate_operation(request, operation)
            binding = self._store.get_binding(request.binding_id)
            if operation.state is NotesSyncOperationState.COMPLETED:
                if binding.state is NotesSyncBindingState.DISCONNECTED:
                    return self._result(
                        request.operation_id,
                        NotesSyncOperationState.COMPLETED,
                    )
                raise RuntimeError("operation_already_completed")
            self._validate_recovery(request)
            if self._cleanup_pending(request.operation_id):
                raise RuntimeError("operation_needs_attention")
            if operation.state is not NotesSyncOperationState.NEEDS_ATTENTION:
                self._store.mark_operation_attention(
                    request.operation_id,
                    "restore_requested",
                )
            self._transition(
                request.operation_id,
                NotesSyncOperationState.RECOVERY_ADMITTED,
            )
            return await self._advance_restore(request)
        except asyncio.CancelledError:
            if admitted:
                self._persist_attention_best_effort(
                    request.operation_id, "cancelled_after_admission"
                )
            raise
        except Exception as exc:
            reason = self._bounded_reason(exc)
            if admitted:
                self._record_failure_attention(request, exc, reason)
            return self._result(
                request.operation_id,
                NotesSyncOperationState.NEEDS_ATTENTION,
                reason,
            )

    async def disconnect(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        """Relinquish one binding without mutating either content authority."""

        return await self._serialized(
            request.operation_id,
            lambda: self._disconnect(request),
        )

    async def _disconnect(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        if type(request) is not NotesSyncExecutionRequest:
            raise TypeError("request must be a NotesSyncExecutionRequest.")
        if request.action_kind in {
            NotesSyncActionKind.CREATE_NOTE,
            NotesSyncActionKind.CREATE_FILE,
        }:
            return await self._disconnect_create(request)
        admitted = False
        try:
            operation = self._store.get_operation(request.operation_id)
            admitted = True
            self._validate_operation(request, operation)
            binding = self._store.get_binding(request.binding_id)
            if operation.state is NotesSyncOperationState.COMPLETED:
                if binding.state is NotesSyncBindingState.DISCONNECTED:
                    return self._result(
                        request.operation_id,
                        NotesSyncOperationState.COMPLETED,
                    )
                raise RuntimeError("operation_already_completed")
            if self._cleanup_pending(request.operation_id):
                raise RuntimeError("operation_needs_attention")
            if operation.state is not NotesSyncOperationState.NEEDS_ATTENTION:
                self._store.mark_operation_attention(
                    request.operation_id,
                    "disconnect_requested",
                )
            cancelled = await self._reconcile_without_binding(request)
            if cancelled:
                raise asyncio.CancelledError
            self._store.resolve_operation_disconnect(
                request.operation_id,
                binding_id=request.binding_id,
            )
            return self._result(
                request.operation_id,
                NotesSyncOperationState.COMPLETED,
            )
        except asyncio.CancelledError:
            if admitted:
                self._persist_attention_best_effort(
                    request.operation_id, "cancelled_after_admission"
                )
            raise
        except Exception as exc:
            reason = self._bounded_reason(exc)
            if admitted:
                self._record_failure_attention(request, exc, reason)
            return self._result(
                request.operation_id,
                NotesSyncOperationState.NEEDS_ATTENTION,
                reason,
            )

    async def _disconnect_create(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        admitted = False
        try:
            operation = self._store.get_operation(request.operation_id)
            admitted = True
            self._validate_new_operation(request, operation)
            if operation.state is NotesSyncOperationState.COMPLETED:
                return self._result(request.operation_id, operation.state)
            if self._cleanup_pending(request.operation_id):
                raise RuntimeError("operation_needs_attention")
            if operation.state is not NotesSyncOperationState.NEEDS_ATTENTION:
                self._store.mark_operation_attention(
                    request.operation_id,
                    "disconnect_requested",
                )
            self._require_new_root(request)
            self._require_new_candidate_owner(request)
            desired = tuple(
                (request.logical_folder_id, binding.note_id)
                for binding in self._store.list_bindings(request.root_id)
                if binding.state is NotesSyncBindingState.ACTIVE
            )
            _, cancelled = await self._joined_thread_call(
                lambda: asyncio.run(
                    self._notes.reconcile_managed_memberships(
                        owner_id=request.root_id,
                        desired=desired,
                    )
                )
            )
            if cancelled:
                raise asyncio.CancelledError
            self._require_new_root(request)
            self._require_new_candidate_owner(request)
            self._store.resolve_unbound_operation_disconnect(request.operation_id)
            return self._result(
                request.operation_id,
                NotesSyncOperationState.COMPLETED,
            )
        except asyncio.CancelledError:
            if admitted:
                self._persist_attention_best_effort(
                    request.operation_id,
                    "cancelled_after_admission",
                )
            raise
        except Exception as exc:
            reason = self._bounded_reason(exc)
            if admitted:
                self._record_failure_attention(request, exc, reason)
            return self._result(
                request.operation_id,
                NotesSyncOperationState.NEEDS_ATTENTION,
                reason,
            )

    async def _restore_create(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        admitted = False
        try:
            operation = self._store.get_operation(request.operation_id)
            admitted = True
            self._validate_new_operation(request, operation)
            if operation.state is NotesSyncOperationState.COMPLETED:
                return self._result(request.operation_id, operation.state)
            if self._cleanup_pending(request.operation_id):
                raise RuntimeError("operation_needs_attention")
            if operation.state is not NotesSyncOperationState.NEEDS_ATTENTION:
                self._store.mark_operation_attention(
                    request.operation_id,
                    "restore_requested",
                )
            self._store.transition_operation(
                request.operation_id,
                NotesSyncOperationState.RECOVERY_ADMITTED,
            )
            return await self._advance_create_restore(request)
        except asyncio.CancelledError:
            if admitted:
                self._persist_attention_best_effort(
                    request.operation_id,
                    "cancelled_after_admission",
                )
            raise
        except Exception as exc:
            reason = self._bounded_reason(exc)
            if admitted:
                self._record_failure_attention(request, exc, reason)
            return self._result(
                request.operation_id,
                NotesSyncOperationState.NEEDS_ATTENTION,
                reason,
            )

    async def _advance_create_restore(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        self._require_new_root(request)
        self._require_new_candidate_owner(request)
        restored = await self._create_is_restored(request)
        cancelled = False
        if not restored:
            note, file = await self._require_new_desired(request)
            if request.action_kind is NotesSyncActionKind.CREATE_NOTE:
                _, cancelled = await self._joined_thread_call(
                    lambda: asyncio.run(self._notes.delete(note))
                )
            else:
                assert type(file) is NotesSyncFileSnapshot
                _, cancelled = await self._joined_thread_call(
                    lambda: self._filesystem.delete(expected=file)
                )
        self._transition(
            request.operation_id,
            NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
        )
        if cancelled:
            raise asyncio.CancelledError
        self._require_new_root(request)
        self._require_new_candidate_owner(request)
        if not await self._create_is_restored(request):
            raise RuntimeError("restore_postcondition_failed")
        self._transition(
            request.operation_id,
            NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
        )
        desired = tuple(
            (request.logical_folder_id, binding.note_id)
            for binding in self._store.list_bindings(request.root_id)
            if binding.state is NotesSyncBindingState.ACTIVE
        )
        _, cancelled = await self._joined_thread_call(
            lambda: asyncio.run(
                self._notes.reconcile_managed_memberships(
                    owner_id=request.root_id,
                    desired=desired,
                )
            )
        )
        if cancelled:
            raise asyncio.CancelledError
        self._require_new_root(request)
        self._require_new_candidate_owner(request)
        if not await self._create_is_restored(request):
            raise RuntimeError("restore_postcondition_failed")
        self._transition(request.operation_id, NotesSyncOperationState.BINDING_UPDATED)
        self._transition(request.operation_id, NotesSyncOperationState.VERIFIED)
        self._transition(request.operation_id, NotesSyncOperationState.COMPLETED)
        return self._result(request.operation_id, NotesSyncOperationState.COMPLETED)

    async def _create_is_restored(self, request: NotesSyncExecutionRequest) -> bool:
        if request.action_kind is NotesSyncActionKind.CREATE_NOTE:
            assert request.file is not None
            try:
                await self._observe_note(self._request_note_id(request))
            except NotesSyncAuthorityError as error:
                if error.reason_code != "note_missing":
                    raise
            else:
                return False
            current = await self._observe_reconstructed_file(
                _file_relative_path(request.file),
                windows=type(request.file) is WindowsNotesSyncObservation,
            )
            return current == request.file
        assert request.note is not None
        if await self._observe_note(request.note.note_id) != request.note:
            return False
        try:
            await self._observe_file_path(self._request_file_path(request))
        except NotesSyncFilesystemError as error:
            if error.reason_code == "missing_target":
                return True
            raise
        return False

    async def _restore_move(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        admitted = False
        try:
            operation = self._store.get_operation(request.operation_id)
            admitted = True
            self._validate_new_operation(request, operation)
            binding = self._store.get_binding(request.binding_id)
            if operation.state is NotesSyncOperationState.COMPLETED:
                if binding.state is NotesSyncBindingState.DISCONNECTED:
                    return self._result(request.operation_id, operation.state)
                raise RuntimeError("operation_already_completed")
            if self._cleanup_pending(request.operation_id):
                raise RuntimeError("operation_needs_attention")
            if operation.state is not NotesSyncOperationState.NEEDS_ATTENTION:
                self._store.mark_operation_attention(
                    request.operation_id,
                    "restore_requested",
                )
            self._store.transition_operation(
                request.operation_id,
                NotesSyncOperationState.RECOVERY_ADMITTED,
            )
            return await self._advance_move_restore(request)
        except asyncio.CancelledError:
            if admitted:
                self._persist_attention_best_effort(
                    request.operation_id,
                    "cancelled_after_admission",
                )
            raise
        except Exception as exc:
            reason = self._bounded_reason(exc)
            if admitted:
                self._record_failure_attention(request, exc, reason)
            return self._result(
                request.operation_id,
                NotesSyncOperationState.NEEDS_ATTENTION,
                reason,
            )

    async def _advance_move_restore(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        state, note, file = await self._classify_move_restore(request)
        if state == "stale":
            raise RuntimeError("stale_restore_observation")
        self._require_move_restore_owner(request, note, file)
        cancelled = False
        if state == "desired":
            assert type(file) is NotesSyncFileSnapshot
            _, cancelled = await self._joined_thread_call(
                lambda: self._filesystem.move(
                    _file_relative_path(request.file),
                    expected=file,
                )
            )
        self._transition(
            request.operation_id,
            NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
        )
        if cancelled:
            raise asyncio.CancelledError
        state, note, file = await self._classify_move_restore(request)
        if state != "restored":
            raise RuntimeError("restore_postcondition_failed")
        self._require_move_restore_owner(request, note, file)
        self._transition(
            request.operation_id,
            NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
        )
        cancelled = await self._reconcile_without_binding(request)
        if cancelled:
            raise asyncio.CancelledError
        state, note, file = await self._classify_move_restore(request)
        if state != "restored":
            raise RuntimeError("restore_postcondition_failed")
        self._require_move_restore_owner(request, note, file)
        binding = self._store.get_binding(request.binding_id)
        self._store.commit_binding_stage(
            request.operation_id,
            expected=binding,
            replacement=replace(
                binding,
                state=NotesSyncBindingState.DISCONNECTED,
                stable_identity_digest=self.stable_identity_digest(file),
                serialization=_file_serialization(file),
                content_digest=note.content_digest,
                note_version=note.version,
            ),
        )
        self._stage(NotesSyncOperationState.BINDING_UPDATED)
        self._transition(request.operation_id, NotesSyncOperationState.VERIFIED)
        self._transition(request.operation_id, NotesSyncOperationState.COMPLETED)
        return self._result(request.operation_id, NotesSyncOperationState.COMPLETED)

    async def _classify_move_restore(
        self,
        request: NotesSyncExecutionRequest,
    ) -> tuple[str, NotesSyncNoteSnapshot, _FileSnapshot]:
        assert request.note is not None and request.file is not None
        note = await self._observe_note(request.note.note_id)
        if note != request.note:
            return "stale", note, request.file
        try:
            destination = await self._observe_file_path(
                request.move_destination_relative_path or ""
            )
        except NotesSyncFilesystemError as error:
            if error.reason_code != "missing_target":
                raise
        else:
            desired = (
                type(destination) is NotesSyncFileSnapshot
                and destination.raw_bytes == request.file.raw_bytes
                and _file_serialization(destination)
                == _file_serialization(request.file)
                and self.stable_identity_digest(destination)
                == self.stable_identity_digest(request.file)
            )
            return ("desired" if desired else "stale"), note, destination
        source = await self._observe_file_path(_file_relative_path(request.file))
        restored = (
            type(source) is NotesSyncFileSnapshot
            and source.raw_bytes == request.file.raw_bytes
            and _file_serialization(source) == _file_serialization(request.file)
            and self.stable_identity_digest(source)
            == self.stable_identity_digest(request.file)
        )
        return ("restored" if restored else "stale"), note, source

    def _require_move_restore_owner(
        self,
        request: NotesSyncExecutionRequest,
        note: NotesSyncNoteSnapshot,
        file: _FileSnapshot,
    ) -> None:
        root = self._store.get_root(request.root_id)
        binding = self._store.get_binding(request.binding_id)
        if (
            root.state is not NotesSyncRootState.ACTIVE
            or root.logical_folder_id != request.logical_folder_id
            or root.direction is not request.direction
            or binding.state
            not in {NotesSyncBindingState.ACTIVE, NotesSyncBindingState.DISCONNECTED}
            or binding.root_id != request.root_id
            or binding.note_scope_id != note.note_scope_id
            or binding.note_id != note.note_id
            or binding.normalized_relative_path != _file_relative_path(request.file)
        ):
            raise RuntimeError("binding_authority_changed")
        self._require_direction(request)
        if binding.state is NotesSyncBindingState.ACTIVE:
            self._require_reviewed_owner(request)
        elif not (
            binding.stable_identity_digest == self.stable_identity_digest(file)
            and binding.serialization == _file_serialization(file)
            and binding.content_digest == note.content_digest
            and binding.note_version == note.version
        ):
            raise RuntimeError("binding_authority_changed")

    async def _run(
        self,
        request: NotesSyncExecutionRequest,
        *,
        allow_attention: bool,
    ) -> NotesSyncExecutionResult:
        if type(request) is not NotesSyncExecutionRequest:
            raise TypeError("request must be a NotesSyncExecutionRequest.")
        if request.action_kind in {
            NotesSyncActionKind.CREATE_NOTE,
            NotesSyncActionKind.CREATE_FILE,
            NotesSyncActionKind.MOVE_FILE,
        }:
            return await self._run_create_or_move(
                request,
                allow_attention=allow_attention,
            )
        admitted = False
        try:
            operation = self._store.find_operation(request.operation_id)
            if operation is None:
                await self._validate_initial(request)
                admission = self._admit(request)
                if not admission:
                    return self._result(
                        request.operation_id,
                        NotesSyncOperationState.NEEDS_ATTENTION,
                        "recovery_capacity_exceeded",
                    )
                admitted = True
                self._stage(NotesSyncOperationState.RECOVERY_ADMITTED)
            else:
                admitted = True
                self._validate_operation(request, operation)
                if operation.state is NotesSyncOperationState.COMPLETED:
                    return self._result(
                        request.operation_id,
                        NotesSyncOperationState.COMPLETED,
                    )
                self._validate_recovery(request)
                if operation.state is NotesSyncOperationState.NEEDS_ATTENTION:
                    if not allow_attention or self._cleanup_pending(
                        request.operation_id
                    ):
                        return self._result(
                            request.operation_id,
                            operation.state,
                            operation.reason_code or "operation_needs_attention",
                        )
                    self._store.transition_operation(
                        request.operation_id,
                        NotesSyncOperationState.RECOVERY_ADMITTED,
                    )
            return await self._advance(request)
        except asyncio.CancelledError:
            if admitted:
                self._persist_attention_best_effort(
                    request.operation_id, "cancelled_after_admission"
                )
            raise
        except Exception as exc:
            reason = self._bounded_reason(exc)
            if admitted:
                self._record_failure_attention(request, exc, reason)
            return self._result(
                request.operation_id,
                NotesSyncOperationState.NEEDS_ATTENTION,
                reason,
            )

    async def _run_create_or_move(
        self,
        request: NotesSyncExecutionRequest,
        *,
        allow_attention: bool,
    ) -> NotesSyncExecutionResult:
        admitted = False
        try:
            operation = self._store.find_operation(request.operation_id)
            if operation is None:
                await self._validate_new_initial(request)
                if not self._admit_new(request):
                    return self._result(
                        request.operation_id,
                        NotesSyncOperationState.NEEDS_ATTENTION,
                        "recovery_capacity_exceeded",
                    )
                admitted = True
                self._stage(NotesSyncOperationState.RECOVERY_ADMITTED)
            else:
                admitted = True
                self._validate_new_operation(request, operation)
                if operation.state is NotesSyncOperationState.COMPLETED:
                    return self._result(request.operation_id, operation.state)
                if operation.state is NotesSyncOperationState.NEEDS_ATTENTION:
                    if not allow_attention or self._cleanup_pending(
                        request.operation_id
                    ):
                        return self._result(
                            request.operation_id,
                            NotesSyncOperationState.NEEDS_ATTENTION,
                            operation.reason_code or "operation_needs_attention",
                        )
                    self._store.transition_operation(
                        request.operation_id,
                        NotesSyncOperationState.RECOVERY_ADMITTED,
                    )
            return await self._advance_new(request)
        except asyncio.CancelledError:
            if admitted:
                self._persist_attention_best_effort(
                    request.operation_id, "cancelled_after_admission"
                )
            raise
        except Exception as exc:
            reason = self._bounded_reason(exc)
            if admitted:
                self._record_failure_attention(request, exc, reason)
            return self._result(
                request.operation_id,
                NotesSyncOperationState.NEEDS_ATTENTION,
                reason,
            )

    async def _validate_new_initial(self, request: NotesSyncExecutionRequest) -> None:
        self._require_new_root(request)
        if request.action_kind is not NotesSyncActionKind.MOVE_FILE:
            self._require_new_candidate_owner(request)
        if request.action_kind is NotesSyncActionKind.CREATE_NOTE:
            assert request.file is not None
            observed = await self._observe_reconstructed_file(
                _file_relative_path(request.file),
                windows=type(request.file) is WindowsNotesSyncObservation,
            )
            if observed != request.file:
                raise RuntimeError("stale_observation")
            await self._require_note_missing(self._request_note_id(request))
        elif request.action_kind is NotesSyncActionKind.CREATE_FILE:
            assert request.note is not None
            note = await self._observe_note(request.note.note_id)
            if note != request.note:
                raise RuntimeError("stale_observation")
            await self._require_file_missing(self._request_file_path(request))
        else:
            assert request.note is not None and request.file is not None
            await self._validate_initial(request)
            await self._require_file_missing(
                request.move_destination_relative_path or ""
            )

    def _admit_new(self, request: NotesSyncExecutionRequest) -> bool:
        note_scope_id = self._request_note_scope_id(request)
        note_id = self._request_note_id(request)
        file_path = self._request_file_path(request)
        file_snapshot = (
            request.file if type(request.file) is NotesSyncFileSnapshot else None
        )
        existing_binding = (
            self._store.get_binding(request.binding_id)
            if request.action_kind is NotesSyncActionKind.MOVE_FILE
            else None
        )
        intent: dict[str, object] = {
            "action": request.action_kind.value,
            "binding": (
                None
                if existing_binding is None
                else {
                    "content_digest": existing_binding.content_digest,
                    "note_version": existing_binding.note_version,
                    "serialization": {
                        "final_newline": existing_binding.serialization.final_newline,
                        "mode": existing_binding.serialization.mode,
                        "newline": existing_binding.serialization.newline,
                        "utf8_bom": existing_binding.serialization.utf8_bom,
                    },
                    "stable_identity_digest": existing_binding.stable_identity_digest,
                }
            ),
            "binding_id": request.binding_id,
            "cleanup_pending": False,
            "desired_title": request.desired_title,
            "direction": request.direction.value,
            "direction_override": _encoded_override(request.direction_override),
            "file_relative_path": file_path,
            "file_reviewed_state": (
                _encoded_reviewed_state(file_snapshot)
                if file_snapshot is not None
                else None
            ),
            "logical_folder_id": request.logical_folder_id,
            "move_destination_relative_path": request.move_destination_relative_path,
            "note_id": note_id,
            "note_scope_id": note_scope_id,
            "note_content_digest": (
                None if request.note is None else request.note.content_digest
            ),
            "windows_observation": (
                {
                    "freshness_digest": request.file.freshness_digest,
                    "stable_identity_digest": request.file.stable_identity_digest,
                }
                if type(request.file) is WindowsNotesSyncObservation
                else None
            ),
            "candidate_serialization": (
                None
                if request.candidate_serialization is None
                else {
                    "utf8_bom": request.candidate_serialization.utf8_bom,
                    "newline": request.candidate_serialization.newline,
                    "final_newline": request.candidate_serialization.final_newline,
                    "mode": request.candidate_serialization.mode,
                }
            ),
        }
        intent["cleanup_padding"] = _cleanup_padding(intent)
        payload = (
            file_snapshot.raw_bytes
            if file_snapshot is not None
            else (
                _represented_bytes(request.file.text, request.file.serialization)
                if type(request.file) is WindowsNotesSyncObservation
                else b""
            )
        )
        operation_binding = (
            request.binding_id
            if request.action_kind is NotesSyncActionKind.MOVE_FILE
            else None
        )
        decision = self._store.admit_operation_recovery(
            NotesSyncOperationRecord(
                operation_id=request.operation_id,
                root_id=request.root_id,
                binding_id=operation_binding,
                kind=request.action_kind.value,
                state=NotesSyncOperationState.PENDING,
                reason_code=None,
                observation_token=request.observation_token,
                expected_note_version=(
                    None if request.note is None else request.note.version
                ),
                expected_file_digest=(
                    None
                    if request.file is None
                    else _file_representation_digest(request.file)
                ),
            ),
            NotesSyncRecoveryRecord(
                recovery_id=request.recovery_id,
                operation_id=request.operation_id,
                payload=payload,
                metadata=_encode_recovery_intent(intent),
                expires_at=request.recovery_expires_at,
            ),
            capacity_bytes=self._capacity,
        )
        return decision.admitted

    async def _advance_new(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        while True:
            state = self._store.get_operation(request.operation_id).state
            if state is NotesSyncOperationState.RECOVERY_ADMITTED:
                self._require_new_root(request)
                self._require_new_candidate_owner(request)
                cancelled = await self._apply_new_first_authority(request)
                self._transition(
                    request.operation_id,
                    NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
                )
                if cancelled:
                    raise asyncio.CancelledError
                continue
            if state is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
                self._require_new_root(request)
                self._require_new_candidate_owner(request)
                note, file = await self._require_new_desired(request)
                self._transition(
                    request.operation_id,
                    NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
                )
                continue
            if state is NotesSyncOperationState.SECOND_AUTHORITY_APPLIED:
                self._require_new_root(request)
                self._require_new_candidate_owner(request)
                note, file = await self._require_new_desired(request)
                desired = [
                    (request.logical_folder_id, binding.note_id)
                    for binding in self._store.list_bindings(request.root_id)
                    if binding.state is NotesSyncBindingState.ACTIVE
                ]
                if request.action_kind is not NotesSyncActionKind.MOVE_FILE:
                    desired.append((request.logical_folder_id, note.note_id))
                _, cancelled = await self._joined_thread_call(
                    lambda: asyncio.run(
                        self._notes.reconcile_managed_memberships(
                            owner_id=request.root_id,
                            desired=tuple(sorted(set(desired))),
                        )
                    )
                )
                if cancelled:
                    raise asyncio.CancelledError
                self._require_new_root(request)
                self._require_new_candidate_owner(request)
                replacement = NotesSyncBindingRecord(
                    binding_id=request.binding_id,
                    root_id=request.root_id,
                    note_scope_id=note.note_scope_id,
                    note_id=note.note_id,
                    normalized_relative_path=_file_relative_path(file),
                    stable_identity_digest=self.stable_identity_digest(file),
                    state=NotesSyncBindingState.ACTIVE,
                    serialization=_file_serialization(file),
                    content_digest=note.content_digest,
                    note_version=note.version,
                )
                if request.action_kind is NotesSyncActionKind.MOVE_FILE:
                    expected = self._store.get_binding(request.binding_id)
                    self._store.commit_binding_stage(
                        request.operation_id,
                        expected=expected,
                        replacement=replacement,
                    )
                else:
                    self._store.create_binding_stage(
                        request.operation_id,
                        replacement,
                    )
                self._stage(NotesSyncOperationState.BINDING_UPDATED)
                continue
            if state is NotesSyncOperationState.BINDING_UPDATED:
                await self._require_new_desired(request)
                self._transition(request.operation_id, NotesSyncOperationState.VERIFIED)
                continue
            if state is NotesSyncOperationState.VERIFIED:
                await self._require_new_desired(request)
                self._transition(
                    request.operation_id, NotesSyncOperationState.COMPLETED
                )
                return self._result(
                    request.operation_id,
                    NotesSyncOperationState.COMPLETED,
                )
            if state is NotesSyncOperationState.COMPLETED:
                return self._result(request.operation_id, state)
            raise RuntimeError("operation_needs_attention")

    async def _apply_new_first_authority(
        self,
        request: NotesSyncExecutionRequest,
    ) -> bool:
        try:
            await self._require_new_desired(request)
        except NotesSyncAuthorityError as error:
            if error.reason_code != "note_missing":
                raise
        except NotesSyncFilesystemError as error:
            if error.reason_code != "missing_target":
                raise
        else:
            return False
        if request.action_kind is NotesSyncActionKind.CREATE_NOTE:
            assert request.file is not None
            await self._require_note_missing(self._request_note_id(request))
            _, cancelled = await self._joined_thread_call(
                lambda: asyncio.run(
                    self._notes.create(
                        note_id=self._request_note_id(request),
                        title=request.desired_title,
                        content=request.file.text,
                    )
                )
            )
            return cancelled
        if request.action_kind is NotesSyncActionKind.CREATE_FILE:
            assert request.note is not None
            path = self._request_file_path(request)
            await self._require_file_missing(path)
            _, cancelled = await self._joined_thread_call(
                lambda: self._filesystem.create(
                    path,
                    request.note.content,
                    profile=request.candidate_serialization,
                )
            )
            return cancelled
        assert type(request.file) is NotesSyncFileSnapshot
        destination = request.move_destination_relative_path or ""
        await self._require_file_missing(destination)
        _, cancelled = await self._joined_thread_call(
            lambda: self._filesystem.move(destination, expected=request.file)
        )
        return cancelled

    async def _require_new_desired(
        self,
        request: NotesSyncExecutionRequest,
    ) -> tuple[NotesSyncNoteSnapshot, _FileSnapshot]:
        note = await self._observe_note(self._request_note_id(request))
        path = (
            request.move_destination_relative_path
            if request.action_kind is NotesSyncActionKind.MOVE_FILE
            else self._request_file_path(request)
        )
        file = await self._observe_reconstructed_file(
            path or "",
            windows=type(request.file) is WindowsNotesSyncObservation,
        )
        if note.note_scope_id != self._request_note_scope_id(request):
            raise RuntimeError("postcondition_failed")
        if request.action_kind is NotesSyncActionKind.CREATE_NOTE:
            assert request.file is not None
            valid = (
                file == request.file
                and note.title == request.desired_title
                and note.content == request.file.text
            )
        elif request.action_kind is NotesSyncActionKind.CREATE_FILE:
            assert request.note is not None
            assert request.candidate_serialization is not None
            expected_bytes = _represented_bytes(
                request.note.content,
                request.candidate_serialization,
            )
            valid = (
                type(file) is NotesSyncFileSnapshot
                and note == request.note
                and file.text == request.note.content
                and file.raw_bytes == expected_bytes
                and file.representation_digest
                == hashlib.sha256(expected_bytes).hexdigest()
                and file.observation.serialization == request.candidate_serialization
            )
        else:
            assert request.note is not None and request.file is not None
            valid = (
                note == request.note
                and file.raw_bytes == request.file.raw_bytes
                and _file_serialization(file) == _file_serialization(request.file)
                and self.stable_identity_digest(file)
                == self.stable_identity_digest(request.file)
            )
        if not valid:
            raise RuntimeError("postcondition_failed")
        return note, file

    async def _observe_note(self, note_id: str) -> NotesSyncNoteSnapshot:
        return await asyncio.to_thread(
            lambda: asyncio.run(self._notes.observe(note_id))
        )

    async def _observe_file_path(self, relative_path: str) -> _FileSnapshot:
        observed = await asyncio.to_thread(self._filesystem.observe, relative_path)
        if type(observed) not in {NotesSyncFileSnapshot, WindowsNotesSyncObservation}:
            raise RuntimeError("file_observation_failed")
        return observed

    async def _require_note_missing(self, note_id: str) -> None:
        try:
            await self._observe_note(note_id)
        except NotesSyncAuthorityError as error:
            if error.reason_code == "note_missing":
                return
            raise
        raise RuntimeError("stale_observation")

    async def _require_file_missing(self, relative_path: str) -> None:
        try:
            await self._observe_file_path(relative_path)
        except NotesSyncFilesystemError as error:
            if error.reason_code == "missing_target":
                return
            raise
        raise RuntimeError("stale_observation")

    def _require_new_root(self, request: NotesSyncExecutionRequest) -> None:
        root = self._store.get_root(request.root_id)
        if (
            root.state is not NotesSyncRootState.ACTIVE
            or root.logical_folder_id != request.logical_folder_id
            or root.direction is not request.direction
        ):
            raise RuntimeError("binding_authority_changed")
        self._require_direction(request)

    def _require_new_candidate_owner(
        self,
        request: NotesSyncExecutionRequest,
    ) -> None:
        if request.action_kind is NotesSyncActionKind.MOVE_FILE:
            self._require_reviewed_owner(request)
            return
        try:
            self._store.get_binding(request.binding_id)
        except NotesDeviceStateError:
            pass
        else:
            raise RuntimeError("binding_authority_changed")
        note_scope_id = self._request_note_scope_id(request)
        note_id = self._request_note_id(request)
        relative_path = self._request_file_path(request)
        if any(
            (binding.note_scope_id == note_scope_id and binding.note_id == note_id)
            or binding.normalized_relative_path == relative_path
            for binding in self._store.list_bindings(request.root_id)
        ):
            raise RuntimeError("binding_authority_changed")

    @staticmethod
    def _request_note_scope_id(request: NotesSyncExecutionRequest) -> str:
        return (
            request.candidate_note_scope_id
            if request.note is None
            else request.note.note_scope_id
        ) or ""

    @staticmethod
    def _request_note_id(request: NotesSyncExecutionRequest) -> str:
        return (
            request.candidate_note_id if request.note is None else request.note.note_id
        ) or ""

    @staticmethod
    def _request_file_path(request: NotesSyncExecutionRequest) -> str:
        return (
            request.candidate_relative_path
            if request.file is None
            else _file_relative_path(request.file)
        ) or ""

    def _validate_new_operation(
        self,
        request: NotesSyncExecutionRequest,
        operation: NotesSyncOperationRecord,
    ) -> None:
        expected_binding = (
            request.binding_id
            if request.action_kind is NotesSyncActionKind.MOVE_FILE
            else None
        )
        if (
            operation.root_id != request.root_id
            or operation.binding_id not in {expected_binding, request.binding_id}
            or operation.kind != request.action_kind.value
            or operation.observation_token != request.observation_token
        ):
            raise RuntimeError("stale_operation_token")

    def _cleanup_pending(self, operation_id: str) -> bool:
        recovery = self._store.find_operation_recovery(operation_id)
        if recovery is None:
            return False
        pending = self._recovery_metadata(recovery).get("cleanup_pending")
        if type(pending) is not bool:
            raise RuntimeError("recovery_authority_changed")
        return pending

    def _persist_attention_best_effort(
        self,
        operation_id: str,
        reason_code: str,
    ) -> None:
        try:
            self._store.mark_operation_attention(operation_id, reason_code)
        except Exception:
            pass

    def _record_failure_attention(
        self,
        request: NotesSyncExecutionRequest,
        error: Exception,
        reason_code: str,
    ) -> None:
        if type(error) is NotesSyncFilesystemPartialError:
            try:
                self._persist_partial_attention(request, error, reason_code)
            except Exception:
                raise NotesSyncExecutionPartialError(
                    reason_code,
                    error.cleanup_handle,
                ) from None
            return
        try:
            self._store.mark_operation_attention(request.operation_id, reason_code)
        except NotesDeviceStateError:
            pass

    def _persist_partial_attention(
        self,
        request: NotesSyncExecutionRequest,
        error: NotesSyncFilesystemPartialError,
        reason_code: str,
    ) -> None:
        recovery = self._store.load_recovery(request.recovery_id)
        metadata = self._recovery_metadata(recovery)
        handle = error.cleanup_handle
        if type(handle) is not NotesSyncPrivateCleanupHandle:
            raise NotesDeviceStateError("The private cleanup authority is invalid.")
        self._set_cleanup_intent(metadata, handle, reason_code)
        encoded = _encode_recovery_intent(metadata)
        self._store.mark_operation_partial_attention(
            request.operation_id,
            request.recovery_id,
            reason_code,
            encoded,
            capacity_bytes=self._capacity,
        )

    @staticmethod
    def _set_cleanup_intent(
        metadata: dict[str, object],
        handle: NotesSyncPrivateCleanupHandle,
        reason_code: str,
    ) -> None:
        relative_path = (
            normalize_notes_sync_relative_path(handle.private_relative_path)
            if handle.private_relative_path is not None
            else None
        )
        metadata.pop("cleanup_padding", None)
        metadata["cleanup_pending"] = True
        metadata["cleanup_relative_path"] = relative_path
        metadata["cleanup_reason_code"] = reason_code
        metadata["cleanup_identity"] = (
            None
            if handle.private_identity is None
            else [
                handle.private_identity.device,
                handle.private_identity.inode,
                handle.private_identity.link_count,
            ]
        )

    async def _validate_initial(self, request: NotesSyncExecutionRequest) -> None:
        binding = self._require_owner_identity(request)
        if request.journal_kind is not None:
            pass
        elif request.action_kind is NotesSyncActionKind.UPDATE_NOTE:
            if (
                binding.note_version != request.note.version
                or binding.content_digest != request.note.content_digest
            ):
                raise RuntimeError("stale_observation")
        elif (
            binding.content_digest != _file_content_digest(request.file)
            or binding.serialization != _file_serialization(request.file)
            or binding.stable_identity_digest
            != self.stable_identity_digest(request.file)
        ):
            raise RuntimeError("stale_observation")
        note, file = await self._observe(request)
        if note != request.note or file != request.file:
            raise RuntimeError("stale_observation")

    def _admit(self, request: NotesSyncExecutionRequest) -> bool:
        binding = self._store.get_binding(request.binding_id)
        if request.action_kind is NotesSyncActionKind.UPDATE_NOTE:
            payload = request.note.content.encode("utf-8")
            desired_digest = _file_content_digest(request.file)
        else:
            assert type(request.file) is NotesSyncFileSnapshot
            payload = request.file.raw_bytes
            desired_digest = request.note.content_digest
        intent: dict[str, object] = {
            "action": request.action_kind.value,
            "binding": {
                "content_digest": binding.content_digest,
                "note_version": binding.note_version,
                "serialization": {
                    "final_newline": binding.serialization.final_newline,
                    "mode": binding.serialization.mode,
                    "newline": binding.serialization.newline,
                    "utf8_bom": binding.serialization.utf8_bom,
                },
                "stable_identity_digest": binding.stable_identity_digest,
            },
            "cleanup_pending": False,
            "desired_digest": desired_digest,
            "desired_title": request.desired_title,
            "direction": request.direction.value,
            "direction_override": _encoded_override(request.direction_override),
            "file_relative_path": _file_relative_path(request.file),
            "file_reviewed_state": (
                _encoded_reviewed_state(request.file)
                if type(request.file) is NotesSyncFileSnapshot
                and request.action_kind is NotesSyncActionKind.UPDATE_FILE
                else None
            ),
            "logical_folder_id": request.logical_folder_id,
            "note_id": request.note.note_id,
            "note_scope_id": request.note.note_scope_id,
            "recovery_title": request.note.title,
            "windows_observation": type(request.file) is WindowsNotesSyncObservation,
        }
        if request.journal_kind is not None:
            intent["underlying_action_kind"] = request.action_kind.value
        intent["cleanup_padding"] = _cleanup_padding(intent)
        metadata = _encode_recovery_intent(intent)
        decision = self._store.admit_operation_recovery(
            NotesSyncOperationRecord(
                operation_id=request.operation_id,
                root_id=request.root_id,
                binding_id=request.binding_id,
                kind=request.journal_kind or request.action_kind.value,
                state=NotesSyncOperationState.PENDING,
                reason_code=None,
                observation_token=request.observation_token,
                expected_note_version=request.note.version,
                expected_file_digest=_file_representation_digest(request.file),
            ),
            NotesSyncRecoveryRecord(
                recovery_id=request.recovery_id,
                operation_id=request.operation_id,
                payload=payload,
                metadata=metadata,
                expires_at=request.recovery_expires_at,
            ),
            capacity_bytes=self._capacity,
            retention_ns=(
                CONFLICT_RECOVERY_RETENTION_NS
                if request.journal_kind is not None
                else None
            ),
        )
        return decision.admitted

    def _validate_operation(
        self,
        request: NotesSyncExecutionRequest,
        operation: NotesSyncOperationRecord,
    ) -> None:
        if (
            operation.root_id != request.root_id
            or operation.binding_id != request.binding_id
            or operation.kind != (request.journal_kind or request.action_kind.value)
            or operation.observation_token != request.observation_token
            or operation.expected_note_version != request.note.version
            or operation.expected_file_digest
            != _file_representation_digest(request.file)
        ):
            raise RuntimeError("stale_operation_token")

    async def _advance(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        while True:
            state = self._store.get_operation(request.operation_id).state
            if state is NotesSyncOperationState.RECOVERY_ADMITTED:
                self._require_reviewed_owner(request)
                note, file = await self._observe(request)
                target, source = self._classify(request, note, file)
                if not source or target == "stale":
                    raise RuntimeError("stale_observation")
                cancelled = False
                if target == "original":
                    self._require_reviewed_owner(request)
                    if request.action_kind is NotesSyncActionKind.UPDATE_NOTE:
                        _, cancelled = await self._joined_thread_call(
                            lambda: asyncio.run(
                                self._notes.replace(
                                    request.note,
                                    title=request.desired_title,
                                    content=request.file.text,
                                )
                            )
                        )
                    else:
                        _, cancelled = await self._joined_thread_call(
                            lambda: self._filesystem.replace(
                                _file_relative_path(request.file),
                                request.note.content,
                                expected=request.file,
                            )
                        )
                self._transition(
                    request.operation_id,
                    NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
                )
                if cancelled:
                    raise asyncio.CancelledError
                continue
            if state is NotesSyncOperationState.FIRST_AUTHORITY_APPLIED:
                self._require_reviewed_owner(request)
                await self._require_desired(request)
                self._transition(
                    request.operation_id,
                    NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
                )
                continue
            if state is NotesSyncOperationState.SECOND_AUTHORITY_APPLIED:
                self._require_reviewed_owner(request)
                note, file = await self._require_desired(request)
                desired = tuple(
                    (request.logical_folder_id, binding.note_id)
                    for binding in self._store.list_bindings(request.root_id)
                    if binding.state is NotesSyncBindingState.ACTIVE
                )
                self._require_reviewed_owner(request)
                _, cancelled = await self._joined_thread_call(
                    lambda: asyncio.run(
                        self._notes.reconcile_managed_memberships(
                            owner_id=request.root_id,
                            desired=desired,
                        )
                    )
                )
                if cancelled:
                    raise asyncio.CancelledError
                self._require_reviewed_owner(request)
                binding = self._store.get_binding(request.binding_id)
                self._store.commit_binding_stage(
                    request.operation_id,
                    expected=binding,
                    replacement=replace(
                        binding,
                        normalized_relative_path=_file_relative_path(file),
                        stable_identity_digest=self.stable_identity_digest(file),
                        serialization=_file_serialization(file),
                        content_digest=note.content_digest,
                        note_version=note.version,
                    ),
                )
                self._stage(NotesSyncOperationState.BINDING_UPDATED)
                continue
            if state is NotesSyncOperationState.BINDING_UPDATED:
                note, file = await self._require_desired(request)
                self._require_current_owner(request, note=note, file=file)
                self._transition(
                    request.operation_id,
                    NotesSyncOperationState.VERIFIED,
                )
                continue
            if state is NotesSyncOperationState.VERIFIED:
                note, file = await self._require_desired(request)
                self._require_current_owner(request, note=note, file=file)
                self._transition(
                    request.operation_id,
                    NotesSyncOperationState.COMPLETED,
                )
                return self._result(
                    request.operation_id,
                    NotesSyncOperationState.COMPLETED,
                )
            if state is NotesSyncOperationState.COMPLETED:
                return self._result(request.operation_id, state)
            raise RuntimeError("operation_needs_attention")

    def _validate_recovery(self, request: NotesSyncExecutionRequest) -> None:
        recovery = self._store.load_recovery(request.recovery_id)
        if recovery.operation_id != request.operation_id:
            raise RuntimeError("recovery_authority_changed")
        if request.action_kind is NotesSyncActionKind.UPDATE_NOTE:
            expected_payload = request.note.content.encode("utf-8")
        else:
            assert type(request.file) is NotesSyncFileSnapshot
            expected_payload = request.file.raw_bytes
        if recovery.payload != expected_payload:
            raise RuntimeError("recovery_authority_changed")
        metadata = self._recovery_metadata(recovery)
        if (
            not isinstance(metadata, dict)
            or metadata.get("action") != request.action_kind.value
            or metadata.get("desired_digest")
            != (
                _file_content_digest(request.file)
                if request.action_kind is NotesSyncActionKind.UPDATE_NOTE
                else request.note.content_digest
            )
            or metadata.get("desired_title") != request.desired_title
            or metadata.get("direction") != request.direction.value
            or metadata.get("direction_override")
            != _encoded_override(request.direction_override)
            or metadata.get("logical_folder_id") != request.logical_folder_id
            or metadata.get("recovery_title") != request.note.title
            or metadata.get("underlying_action_kind")
            != (request.action_kind.value if request.journal_kind is not None else None)
        ):
            raise RuntimeError("recovery_authority_changed")

    @staticmethod
    def _recovery_metadata(
        recovery: NotesSyncRecoveryRecord,
    ) -> dict[str, object]:
        try:
            metadata = json.loads(recovery.metadata.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("recovery_authority_changed") from None
        if not isinstance(metadata, dict):
            raise RuntimeError("recovery_authority_changed")
        return metadata

    @staticmethod
    def _required_metadata_text(
        metadata: dict[str, object],
        field_name: str,
    ) -> str:
        value = metadata.get(field_name)
        if type(value) is not str:
            raise RuntimeError("recovery_authority_changed")
        return value

    @staticmethod
    def _decoded_binding_serialization(
        reviewed_binding: dict[str, object],
    ) -> NotesSyncSerializationProfile:
        raw = reviewed_binding.get("serialization")
        if not isinstance(raw, dict):
            raise RuntimeError("recovery_authority_changed")
        try:
            return NotesSyncSerializationProfile(
                utf8_bom=raw["utf8_bom"],
                newline=raw["newline"],
                final_newline=raw["final_newline"],
                mode=raw["mode"],
            )
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("recovery_authority_changed") from None

    @staticmethod
    def _decoded_cleanup_identity(value: object) -> SafeSyncFileIdentity:
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(type(item) is not int for item in value)
        ):
            raise RuntimeError("recovery_authority_changed")
        return SafeSyncFileIdentity(
            device=value[0],
            inode=value[1],
            link_count=value[2],
        )

    @staticmethod
    def _decoded_candidate_serialization(
        metadata: dict[str, object],
    ) -> NotesSyncSerializationProfile:
        raw = metadata.get("candidate_serialization")
        if not isinstance(raw, dict):
            raise RuntimeError("recovery_authority_changed")
        try:
            return NotesSyncSerializationProfile(
                utf8_bom=raw["utf8_bom"],
                newline=raw["newline"],
                final_newline=raw["final_newline"],
                mode=raw["mode"],
            )
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("recovery_authority_changed") from None

    async def _serialized(
        self,
        operation_id: str,
        operation: Callable[[], Awaitable[NotesSyncExecutionResult]],
    ) -> NotesSyncExecutionResult:
        """Coalesce callers sharing one executor and durable operation id."""

        lock = self._store.operation_lock(operation_id)
        async with lock:
            return await operation()

    async def _observe_reconstructed_file(
        self,
        relative_path: str,
        *,
        windows: bool,
    ) -> _FileSnapshot:
        if windows:
            candidates = await asyncio.to_thread(self._filesystem.observe)
            if type(candidates) is not tuple:
                raise RuntimeError("file_observation_failed")
            matching = tuple(
                candidate
                for candidate in candidates
                if type(candidate) is WindowsNotesSyncObservation
                and candidate.relative_path == relative_path
            )
            if len(matching) != 1:
                raise RuntimeError("file_observation_failed")
            return matching[0]
        observed = await asyncio.to_thread(self._filesystem.observe, relative_path)
        if type(observed) is not NotesSyncFileSnapshot:
            raise RuntimeError("file_observation_failed")
        return observed

    @staticmethod
    def _reconstructed_original_file(
        relative_path: str,
        payload: bytes,
        representation_digest: str,
        metadata: dict[str, object],
    ) -> NotesSyncFileSnapshot:
        binding = metadata.get("binding")
        if not isinstance(binding, dict):
            raise RuntimeError("recovery_authority_changed")
        raw_profile = binding.get("serialization")
        if not isinstance(raw_profile, dict):
            raise RuntimeError("recovery_authority_changed")
        try:
            profile = NotesSyncSerializationProfile(
                utf8_bom=raw_profile["utf8_bom"],
                newline=raw_profile["newline"],
                final_newline=raw_profile["final_newline"],
                mode=raw_profile["mode"],
            )
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("recovery_authority_changed") from None
        state = _decoded_reviewed_state(
            relative_path,
            metadata.get("file_reviewed_state"),
            payload,
        )
        logical = _logical_text(payload, profile)
        identity = NotesSyncFileIdentity(
            device=state.identity.device,
            inode=state.identity.inode,
            link_count=state.identity.link_count,
        )
        return NotesSyncFileSnapshot(
            observation=NotesSyncFileObservation(
                relative_path=relative_path,
                identity=identity,
                content_digest=hashlib.sha256(logical.encode("utf-8")).hexdigest(),
                size_bytes=len(payload),
                serialization=profile,
            ),
            text=logical,
            raw_bytes=payload,
            reviewed_state=state,
            representation_digest=representation_digest,
        )

    async def _advance_restore(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncExecutionResult:
        note, file = await self._observe(request)
        self._require_restore_owner(request, note, file)
        state = self._classify_restore(request, note, file)
        if state == "stale":
            raise RuntimeError("stale_restore_observation")
        cancelled = False
        if state == "desired":
            self._require_restore_owner(request, note, file)
            if request.action_kind is NotesSyncActionKind.UPDATE_NOTE:
                _, cancelled = await self._joined_thread_call(
                    lambda: asyncio.run(
                        self._notes.replace(
                            note,
                            title=request.note.title,
                            content=request.note.content,
                        )
                    )
                )
            else:
                _, cancelled = await self._joined_thread_call(
                    lambda: self._filesystem.replace(
                        _file_relative_path(request.file),
                        request.file.text,
                        expected=file,
                    )
                )
        self._transition(
            request.operation_id,
            NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
        )
        if cancelled:
            raise asyncio.CancelledError
        note, file = await self._require_restored(request)
        self._require_restore_owner(request, note, file)
        self._transition(
            request.operation_id,
            NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
        )
        self._require_restore_owner(request, note, file)
        cancelled = await self._reconcile_without_binding(request)
        if cancelled:
            raise asyncio.CancelledError
        self._require_restore_owner(request, note, file)
        binding = self._store.get_binding(request.binding_id)
        self._store.commit_binding_stage(
            request.operation_id,
            expected=binding,
            replacement=replace(
                binding,
                state=NotesSyncBindingState.DISCONNECTED,
                normalized_relative_path=_file_relative_path(file),
                stable_identity_digest=self.stable_identity_digest(file),
                serialization=_file_serialization(file),
                content_digest=note.content_digest,
                note_version=note.version,
            ),
        )
        self._stage(NotesSyncOperationState.BINDING_UPDATED)
        note, file = await self._require_restored(request)
        self._require_restore_owner(request, note, file)
        self._transition(
            request.operation_id,
            NotesSyncOperationState.VERIFIED,
        )
        note, file = await self._require_restored(request)
        self._require_restore_owner(request, note, file)
        self._transition(
            request.operation_id,
            NotesSyncOperationState.COMPLETED,
        )
        return self._result(request.operation_id, NotesSyncOperationState.COMPLETED)

    def _require_restore_owner(
        self,
        request: NotesSyncExecutionRequest,
        note: NotesSyncNoteSnapshot,
        file: _FileSnapshot,
    ) -> None:
        root = self._store.get_root(request.root_id)
        binding = self._store.get_binding(request.binding_id)
        if (
            root.state is not NotesSyncRootState.ACTIVE
            or root.logical_folder_id != request.logical_folder_id
            or root.direction is not request.direction
            or binding.state
            not in {NotesSyncBindingState.ACTIVE, NotesSyncBindingState.DISCONNECTED}
            or binding.root_id != request.root_id
            or binding.note_scope_id != request.note.note_scope_id
            or binding.note_id != request.note.note_id
            or binding.normalized_relative_path != _file_relative_path(request.file)
        ):
            raise RuntimeError("binding_authority_changed")
        self._require_direction(request)
        metadata = self._recovery_metadata(
            self._store.load_recovery(request.recovery_id)
        )
        disconnected_match = (
            binding.state is NotesSyncBindingState.DISCONNECTED
            and binding.stable_identity_digest == self.stable_identity_digest(file)
            and binding.serialization == _file_serialization(file)
            and binding.content_digest == note.content_digest
            and binding.note_version == note.version
        )
        if not (
            self._binding_matches_reviewed(binding, metadata)
            or self._binding_matches_current(request, binding, note, file)
            or disconnected_match
        ):
            raise RuntimeError("binding_authority_changed")

    def _classify_restore(
        self,
        request: NotesSyncExecutionRequest,
        note: NotesSyncNoteSnapshot,
        file: NotesSyncFileSnapshot,
    ) -> str:
        if request.action_kind is NotesSyncActionKind.UPDATE_NOTE:
            if file != request.file:
                return "stale"
            restored = (
                note.note_scope_id == request.note.note_scope_id
                and note.note_id == request.note.note_id
                and note.title == request.note.title
                and note.content == request.note.content
                and note.version >= request.note.version
            )
            desired = (
                note.note_scope_id == request.note.note_scope_id
                and note.note_id == request.note.note_id
                and note.title == request.desired_title
                and note.content == request.file.text
                and note.version >= request.note.version + 1
            )
        else:
            if (
                type(file) is not NotesSyncFileSnapshot
                or type(request.file) is not NotesSyncFileSnapshot
            ):
                return "stale"
            if note != request.note:
                return "stale"
            restored = (
                file.observation.relative_path == request.file.observation.relative_path
                and file.raw_bytes == request.file.raw_bytes
                and file.observation.serialization
                == request.file.observation.serialization
            )
            desired = (
                file.observation.relative_path == request.file.observation.relative_path
                and file.text == request.note.content
                and file.observation.serialization
                == request.file.observation.serialization
            )
        return "restored" if restored else "desired" if desired else "stale"

    async def _require_restored(
        self,
        request: NotesSyncExecutionRequest,
    ) -> tuple[NotesSyncNoteSnapshot, _FileSnapshot]:
        note, file = await self._observe(request)
        if self._classify_restore(request, note, file) != "restored":
            raise RuntimeError("restore_postcondition_failed")
        return note, file

    async def _reconcile_without_binding(
        self,
        request: NotesSyncExecutionRequest,
    ) -> bool:
        root = self._store.get_root(request.root_id)
        selected_binding = self._store.get_binding(request.binding_id)
        if (
            root.logical_folder_id != request.logical_folder_id
            or root.direction is not request.direction
            or root.state is not NotesSyncRootState.ACTIVE
            or selected_binding.state
            not in {NotesSyncBindingState.ACTIVE, NotesSyncBindingState.DISCONNECTED}
            or selected_binding.root_id != request.root_id
            or selected_binding.note_scope_id != request.note.note_scope_id
            or selected_binding.note_id != request.note.note_id
        ):
            raise RuntimeError("binding_authority_changed")
        desired = tuple(
            (request.logical_folder_id, binding.note_id)
            for binding in self._store.list_bindings(request.root_id)
            if binding.state is NotesSyncBindingState.ACTIVE
            and binding.binding_id != request.binding_id
        )
        _, cancelled = await self._joined_thread_call(
            lambda: asyncio.run(
                self._notes.reconcile_managed_memberships(
                    owner_id=request.root_id,
                    desired=desired,
                )
            )
        )
        return cancelled

    @staticmethod
    def _require_direction(request: NotesSyncExecutionRequest) -> None:
        allowed = {
            NotesSyncDirection.BIDIRECTIONAL: {
                NotesSyncActionKind.CREATE_NOTE,
                NotesSyncActionKind.UPDATE_NOTE,
                NotesSyncActionKind.CREATE_FILE,
                NotesSyncActionKind.UPDATE_FILE,
                NotesSyncActionKind.MOVE_FILE,
            },
            NotesSyncDirection.FOLDER_TO_NOTES: {
                NotesSyncActionKind.CREATE_NOTE,
                NotesSyncActionKind.UPDATE_NOTE,
                NotesSyncActionKind.MOVE_FILE,
            },
            NotesSyncDirection.NOTES_TO_FOLDER: {
                NotesSyncActionKind.CREATE_FILE,
                NotesSyncActionKind.UPDATE_FILE,
            },
        }
        if (
            request.action_kind not in allowed[request.direction]
            and request.direction_override is None
        ):
            raise RuntimeError("direction_disallows_action")

    def _require_reviewed_owner(self, request: NotesSyncExecutionRequest) -> None:
        binding = self._require_owner_identity(request)
        recovery = self._store.load_recovery(request.recovery_id)
        if not self._binding_matches_reviewed(
            binding,
            self._recovery_metadata(recovery),
        ):
            raise RuntimeError("binding_authority_changed")

    @staticmethod
    def _binding_matches_reviewed(
        binding: NotesSyncBindingRecord,
        metadata: dict[str, object],
    ) -> bool:
        reviewed = metadata.get("binding")
        if not isinstance(reviewed, dict):
            return False
        serialization = reviewed.get("serialization")
        return not (
            not isinstance(serialization, dict)
            or binding.stable_identity_digest != reviewed.get("stable_identity_digest")
            or binding.content_digest != reviewed.get("content_digest")
            or binding.note_version != reviewed.get("note_version")
            or binding.serialization.utf8_bom != serialization.get("utf8_bom")
            or binding.serialization.newline != serialization.get("newline")
            or binding.serialization.final_newline != serialization.get("final_newline")
            or binding.serialization.mode != serialization.get("mode")
        )

    def _require_current_owner(
        self,
        request: NotesSyncExecutionRequest,
        *,
        note: NotesSyncNoteSnapshot,
        file: _FileSnapshot,
    ) -> None:
        binding = self._require_owner_identity(request)
        if not self._binding_matches_current(request, binding, note, file):
            raise RuntimeError("binding_authority_changed")

    def _binding_matches_current(
        self,
        request: NotesSyncExecutionRequest,
        binding: NotesSyncBindingRecord,
        note: NotesSyncNoteSnapshot,
        file: _FileSnapshot,
    ) -> bool:
        expected_digest = (
            note.content_digest
            if request.action_kind is NotesSyncActionKind.UPDATE_NOTE
            else _file_content_digest(file)
        )
        return not (
            binding.stable_identity_digest != self.stable_identity_digest(file)
            or binding.serialization != _file_serialization(file)
            or binding.content_digest != expected_digest
            or binding.note_version != note.version
        )

    def _require_owner_identity(
        self,
        request: NotesSyncExecutionRequest,
    ) -> NotesSyncBindingRecord:
        root = self._store.get_root(request.root_id)
        binding = self._store.get_binding(request.binding_id)
        if (
            root.state is not NotesSyncRootState.ACTIVE
            or root.logical_folder_id != request.logical_folder_id
            or root.direction is not request.direction
            or binding.state is not NotesSyncBindingState.ACTIVE
            or binding.root_id != request.root_id
            or binding.note_scope_id != request.note.note_scope_id
            or binding.note_id != request.note.note_id
            or binding.normalized_relative_path != _file_relative_path(request.file)
        ):
            raise RuntimeError("binding_authority_changed")
        self._require_direction(request)
        return binding

    async def _observe(
        self,
        request: NotesSyncExecutionRequest,
    ) -> tuple[NotesSyncNoteSnapshot, _FileSnapshot]:
        note = await asyncio.to_thread(
            lambda: asyncio.run(self._notes.observe(request.note.note_id))
        )
        if type(request.file) is WindowsNotesSyncObservation:
            candidates = await asyncio.to_thread(self._filesystem.observe)
            if type(candidates) is not tuple:
                raise RuntimeError("file_observation_failed")
            matching = tuple(
                candidate
                for candidate in candidates
                if type(candidate) is WindowsNotesSyncObservation
                and candidate.relative_path == request.file.relative_path
            )
            if len(matching) != 1:
                raise RuntimeError("file_observation_failed")
            return note, matching[0]
        observed = await asyncio.to_thread(
            self._filesystem.observe, _file_relative_path(request.file)
        )
        if type(observed) is not NotesSyncFileSnapshot:
            raise RuntimeError("file_observation_failed")
        return note, observed

    @staticmethod
    async def _joined_thread_call(
        function: Callable[[], object],
    ) -> tuple[object, bool]:
        """Finish one admitted mutation before re-delivering cancellation."""

        task = asyncio.create_task(asyncio.to_thread(function))
        try:
            return await asyncio.shield(task), False
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            return await task, True

    def _classify(
        self,
        request: NotesSyncExecutionRequest,
        note: NotesSyncNoteSnapshot,
        file: _FileSnapshot,
    ) -> tuple[str, bool]:
        if request.action_kind is NotesSyncActionKind.UPDATE_NOTE:
            source_unchanged = file == request.file
            original = note == request.note
            desired = (
                note.note_scope_id == request.note.note_scope_id
                and note.note_id == request.note.note_id
                and note.version == request.note.version + 1
                and note.title == request.desired_title
                and note.content == request.file.text
            )
        else:
            if (
                type(file) is not NotesSyncFileSnapshot
                or type(request.file) is not NotesSyncFileSnapshot
            ):
                return "stale", False
            source_unchanged = note == request.note
            original = file == request.file
            desired = (
                file.observation.relative_path == request.file.observation.relative_path
                and file.text == request.note.content
                and file.observation.content_digest == request.note.content_digest
                and file.observation.serialization
                == request.file.observation.serialization
            )
        return ("original" if original else "desired" if desired else "stale"), (
            source_unchanged
        )

    async def _require_desired(
        self,
        request: NotesSyncExecutionRequest,
    ) -> tuple[NotesSyncNoteSnapshot, _FileSnapshot]:
        note, file = await self._observe(request)
        target, source = self._classify(request, note, file)
        if not source or target != "desired":
            raise RuntimeError("postcondition_failed")
        return note, file

    def _stage(self, state: NotesSyncOperationState) -> None:
        if self._after_stage is not None:
            self._after_stage(state)

    def _transition(
        self,
        operation_id: str,
        state: NotesSyncOperationState,
    ) -> None:
        self._store.transition_operation(operation_id, state)
        self._stage(state)

    @staticmethod
    def _bounded_reason(error: Exception) -> str:
        candidate: object | None = None
        if type(error) in {
            NotesSyncAuthorityError,
            NotesSyncFilesystemError,
            NotesSyncFilesystemPartialError,
        }:
            typed = error.reason_code
            candidate = typed if typed in _TYPED_REASON_CODES else None
        if candidate is None:
            raw = str(error)
            candidate = (
                raw
                if type(error) is RuntimeError and raw in _INTERNAL_REASONS
                else None
            )
        if candidate is None:
            return "executor_failed"
        try:
            selected = validate_notes_sync_reason_code(candidate)
        except (TypeError, ValueError):
            return "executor_failed"
        return selected or "executor_failed"

    @staticmethod
    def _result(
        operation_id: str,
        state: NotesSyncOperationState,
        reason_code: str | None = None,
    ) -> NotesSyncExecutionResult:
        return NotesSyncExecutionResult(
            operation_id=operation_id,
            state=state,
            reason_code=reason_code,
            choices=_ATTENTION_CHOICES
            if state is NotesSyncOperationState.NEEDS_ATTENTION
            else (),
        )


__all__ = [
    "NotesSyncDirectionOverride",
    "NotesSyncExecutionPartialError",
    "NotesSyncExecutionRequest",
    "NotesSyncExecutionResult",
    "NotesSyncExecutor",
    "NotesSyncRecoveryChoice",
]
