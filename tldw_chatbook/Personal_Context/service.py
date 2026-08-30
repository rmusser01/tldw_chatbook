"""Canonical authorized application boundary for local Personal Context."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Lock
from typing import Any

from tldw_profile_core import (
    ProfileControls,
    ProfileManifest,
    ProfilePayload,
    ProfileProvenance,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
    normalize_datetime,
)

from .key_protector import ProfileLockedError
from .repository import (
    ConcurrentProfileUpdateError,
    PersonalContextRepository,
    ProfileIntegrityError,
)
from .runtime_policy import (
    GLOBAL_POLICY_ID,
    AgentAuthority,
    GlobalRuntimePolicy,
    PersonalContextAuthorityError,
    ScopeRuntimePolicy,
    authority_allows,
)


_UNSET_POLICY_VERSION = object()


class ProfileConflictError(RuntimeError):
    """Map repository CAS failures to the application boundary."""


class ProfileKeyCollisionError(ValueError):
    def __init__(self, record_id: str) -> None:
        self.record_id = record_id
        super().__init__("An active record already owns this semantic key.")


class ProfileOperationalState(StrEnum):
    ABSENT = "absent"
    REMOVED = "removed"
    LOCKED = "locked"
    DISABLED = "disabled"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class ProfileOperationalStatus:
    state: ProfileOperationalState
    profile_present: bool
    locked: bool
    runtime_enabled: bool
    reason_code: str | None


@dataclass(frozen=True, slots=True)
class RecordMutation:
    payload: ProfilePayload | None = field(default=None, repr=False)
    semantic_key: SemanticKey | None = field(default=None, repr=False)
    clear_semantic_key: bool = False
    controls: ProfileControls | None = field(default=None, repr=False)
    expires_at: datetime | None = field(default=None, repr=False)
    no_expiry: bool | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class SettingsScopeSnapshot:
    """One readable scope and its peer-local agent authority."""

    scope: ProfileScope = field(repr=False)
    label: str = field(repr=False)
    linked: bool = field(repr=False)
    authority: AgentAuthority = field(repr=False)
    policy_version_id: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class PersonalContextSettingsSnapshot:
    """Immutable Settings-owned view of one local profile generation."""

    status: ProfileOperationalStatus
    scopes: tuple[SettingsScopeSnapshot, ...] = field(default=(), repr=False)
    records: tuple[ProfileRecord, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class AuthorizedProfileContextView:
    """Content-safe immutable input for the read-only context builder."""

    generation: int
    record_set_revision: str
    workspace_scope_id: str | None
    authority_revision: str
    records: tuple[ProfileRecord, ...] = field(default=(), repr=False)
    unsupported_records_present: bool = False
    conflicted_record_ids: tuple[str, ...] = ()


def _default_id(label: str) -> str:
    return f"{label}-{uuid.uuid4()}"


def _default_clock() -> datetime:
    now = datetime.now(UTC)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


class PersonalContextService:
    """Own every authorized local profile lifecycle operation."""

    def __init__(
        self,
        repository: PersonalContextRepository | None,
        *,
        clock: Callable[[], datetime] = _default_clock,
        id_factory: Callable[[str], str] = _default_id,
        locked_reason: str | None = None,
        profile_present_hint: bool = False,
    ) -> None:
        self._repository = repository
        self.clock = clock
        self._ids = id_factory
        self._locked_reason = locked_reason
        self._profile_present_hint = profile_present_hint
        self._destructive_lifecycle_lock = Lock()

    @classmethod
    def locked(
        cls,
        reason_code: str = "profile_locked",
        *,
        profile_present: bool = False,
    ) -> "PersonalContextService":
        return cls(
            None,
            locked_reason=reason_code,
            profile_present_hint=profile_present,
        )

    def _repo(self) -> PersonalContextRepository:
        if self._repository is None or self._locked_reason is not None:
            raise ProfileLockedError("Personal Context profile is locked.")
        return self._repository

    def status(self) -> ProfileOperationalStatus:
        if self._repository is None or self._locked_reason is not None:
            return ProfileOperationalStatus(
                ProfileOperationalState.LOCKED,
                self._profile_present_hint,
                True,
                False,
                self._locked_reason or "profile_locked",
            )
        if self._repository.is_destroyed():
            return ProfileOperationalStatus(
                ProfileOperationalState.REMOVED,
                False,
                False,
                False,
                "local_profile_removed",
            )
        try:
            manifest = self._repository.get_manifest()
        except ProfileLockedError:
            self._locked_reason = "profile_locked"
            self._profile_present_hint = True
            return ProfileOperationalStatus(
                ProfileOperationalState.LOCKED, True, True, False, "profile_locked"
            )
        if manifest is None:
            return ProfileOperationalStatus(
                ProfileOperationalState.ABSENT, False, False, False, "profile_absent"
            )
        try:
            policy = self._global_policy()
        except PersonalContextAuthorityError as exc:
            return ProfileOperationalStatus(
                ProfileOperationalState.DISABLED,
                True,
                False,
                False,
                exc.reason_code,
            )
        if not policy.enabled:
            return ProfileOperationalStatus(
                ProfileOperationalState.DISABLED,
                True,
                False,
                False,
                "personal_context_disabled",
            )
        return ProfileOperationalStatus(
            ProfileOperationalState.READY, True, False, True, None
        )

    def create_profile(self) -> ProfileManifest:
        repository = self._repo()
        if repository.is_destroyed():
            raise ValueError(
                "The local profile was removed; use Start Fresh explicitly."
            )
        now = self.clock()
        profile_id = self._ids("profile-local")
        manifest = ProfileManifest(
            profile_id=profile_id,
            revision=0,
            purge_generation=0,
            created_at=now,
            updated_at=now,
            current_version_id=self._ids("manifest-version"),
        )
        scope = ProfileScope(
            scope_id=self._ids("scope-global"),
            profile_id=profile_id,
            kind=ScopeKind.GLOBAL,
            version_id=self._ids("scope-version"),
            created_at=now,
            updated_at=now,
        )
        repository.create_profile_with_global_scope(manifest, scope)
        return manifest

    def start_fresh_profile(self) -> ProfileManifest:
        """Create a new profile only after an explicit local-removal transition."""

        with self._destructive_lifecycle_lock:
            repository = self._repo()
            if not repository.is_destroyed():
                raise ValueError("Start Fresh is available only after local removal.")
            # A prior removal can have fenced storage before key custody deletion
            # completed. Retry that deletion before provisioning the new generation.
            repository.destroy_profile_content()
            now = self.clock()
            profile_id = self._ids("profile-local")
            manifest = ProfileManifest(
                profile_id=profile_id,
                revision=0,
                purge_generation=0,
                created_at=now,
                updated_at=now,
                current_version_id=self._ids("manifest-version"),
            )
            scope = ProfileScope(
                scope_id=self._ids("scope-global"),
                profile_id=profile_id,
                kind=ScopeKind.GLOBAL,
                version_id=self._ids("scope-version"),
                created_at=now,
                updated_at=now,
            )
            repository.reinitialize_destroyed_profile(manifest, scope)
            self._locked_reason = None
            self._profile_present_hint = True
            return manifest

    def get_manifest(self) -> ProfileManifest:
        manifest = self._repo().get_manifest()
        if manifest is None:
            raise ValueError("Personal Context profile is absent.")
        return manifest

    def list_scopes(self) -> tuple[ProfileScope, ...]:
        return tuple(self._repo().list_scopes())

    def create_workspace_scope(
        self, local_workspace_id: str, label: str
    ) -> ProfileScope:
        local_workspace_id = self._bounded_local_text(
            local_workspace_id, "local workspace id"
        )
        label = self._bounded_local_text(label, "workspace label")
        manifest = self.get_manifest()
        now = self.clock()
        scope = ProfileScope(
            scope_id=self._ids("scope-workspace"),
            profile_id=manifest.profile_id,
            kind=ScopeKind.WORKSPACE,
            version_id=self._ids("scope-version"),
            created_at=now,
            updated_at=now,
        )
        self._repo().commit_scope_with_binding(
            scope,
            {"version": 1, "local_workspace_id": local_workspace_id, "label": label},
        )
        return scope

    def map_workspace_scope(
        self, local_workspace_id: str, scope_id: str
    ) -> ProfileScope:
        local_workspace_id = self._bounded_local_text(
            local_workspace_id, "local workspace id"
        )
        scope = self._require_scope(scope_id)
        if scope.kind is not ScopeKind.WORKSPACE:
            raise ValueError("Only workspace scopes may have local mappings.")
        bindings = self.list_workspace_bindings()
        current = bindings.get(scope_id)
        if (
            current is not None
            and current.get("local_workspace_id") == local_workspace_id
        ):
            return scope
        version = self._repo().get_scope_binding_version(scope_id)
        label = "" if current is None else str(current.get("label", ""))
        self._repo().commit_scope_binding(
            scope_id,
            {"version": 1, "local_workspace_id": local_workspace_id, "label": label},
            expected_version_id=version,
            require_unique_local_workspace_id=True,
        )
        return scope

    def get_workspace_binding(self, scope_id: str) -> dict[str, Any] | None:
        """Return one authenticated exact-v1 peer-local workspace mapping."""

        scope = self._require_scope(scope_id)
        if scope.kind is not ScopeKind.WORKSPACE:
            raise ValueError("Only workspace scopes may have local mappings.")
        return self._repo().get_validated_scope_binding(scope_id)

    def list_workspace_bindings(self) -> dict[str, dict[str, Any]]:
        """Return authenticated exact-v1 peer-local workspace mappings."""

        return self._repo().list_validated_scope_bindings()

    @staticmethod
    def _bounded_local_text(value: str, field: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 16_384:
            raise ValueError(f"{field} must be a bounded non-empty string.")
        return value

    def _require_scope(self, scope_id: str) -> ProfileScope:
        scope = self._repo().get_scope(scope_id)
        if scope is None:
            raise ValueError("Unknown profile scope.")
        return scope

    def _require_record_identity(self, record: ProfileRecord) -> None:
        manifest = self.get_manifest()
        if record.profile_id != manifest.profile_id:
            raise ValueError("Record belongs to another profile.")
        scope = self._require_scope(record.scope_id)
        if scope.profile_id != manifest.profile_id:
            raise ValueError("Record scope belongs to another profile.")
        if (
            scope.kind is ScopeKind.WORKSPACE
            and scope.scope_id not in self.list_workspace_bindings()
        ):
            raise ValueError("Workspace scope must be mapped before mutation.")

    def _active_for_collision(self, record: ProfileRecord) -> bool:
        return record.state is RecordState.ACTIVE and (
            record.expires_at is None or record.expires_at > self.clock()
        )

    def _require_no_collision(
        self, record: ProfileRecord, *, excluding_record_id: str | None = None
    ) -> None:
        if record.semantic_key is None or not self._active_for_collision(record):
            return
        for existing in self._repo().list_records():
            if existing.record_id == excluding_record_id:
                continue
            if (
                existing.scope_id == record.scope_id
                and existing.kind is record.kind
                and existing.semantic_key == record.semantic_key
                and self._active_for_collision(existing)
            ):
                raise ProfileKeyCollisionError(existing.record_id)

    def _next_manifest(self, current: ProfileManifest) -> ProfileManifest:
        return ProfileManifest.model_validate(
            {
                **current.model_dump(mode="python"),
                "revision": current.revision + 1,
                "updated_at": self.clock(),
                "current_version_id": self._ids("manifest-version"),
            }
        )

    def _commit_record(
        self,
        record: ProfileRecord,
        *,
        expected_version_id: str | None,
        before: ProfileRecord | None = None,
        consume_undo_id: str | None = None,
        manifest_fence: ProfileManifest | None = None,
    ) -> ProfileRecord:
        manifest = manifest_fence or self.get_manifest()
        next_manifest = self._next_manifest(manifest)
        undo_id = None
        undo_body = None
        undo_expires = None
        if before is not None:
            undo_id = self._ids("undo")
            undo_expires_at = self.clock() + timedelta(hours=24)
            undo_expires = normalize_datetime(undo_expires_at)
            undo_body = {
                "version": 1,
                "expected_head_version": record.version_id,
                "before_record": before.model_dump(mode="json"),
                "expires_at": undo_expires,
            }
        try:
            self._repo().commit_record_and_manifest(
                record,
                next_manifest,
                expected_record_version=expected_version_id,
                expected_manifest_version=manifest.current_version_id,
                undo_id=undo_id,
                undo_body=undo_body,
                undo_expires_at=undo_expires,
                consume_undo_id=consume_undo_id,
                outbox_body={"version": 1, "record": record.model_dump(mode="json")},
            )
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError(
                "Personal Context changed concurrently."
            ) from exc
        return record

    def create_record(self, record: ProfileRecord) -> ProfileRecord:
        record = ProfileRecord.model_validate(record.model_dump(mode="python"))
        self._require_record_identity(record)
        if (
            record.state is not RecordState.ACTIVE
            or record.parent_version_id is not None
        ):
            raise ValueError("New records must be active and have no parent version.")
        if self._repo().get_record(record.record_id) is not None:
            raise ProfileConflictError("Record identity already exists.")
        manifest_fence = self.get_manifest()
        self._require_no_collision(record)
        return self._commit_record(
            record,
            expected_version_id=None,
            manifest_fence=manifest_fence,
        )

    def create_manual_record(
        self,
        *,
        scope_id: str,
        payload: ProfilePayload,
        semantic_key: SemanticKey | dict[str, Any] | None,
        controls: ProfileControls | dict[str, Any],
        expires_at: datetime | None = None,
        no_expiry: bool = False,
    ) -> ProfileRecord:
        """Create a user-authored record while owning all canonical identity fields."""

        manifest = self.get_manifest()
        scope = self._require_scope(scope_id)
        now = self.clock()
        if payload.kind == "working_context" and expires_at is None and not no_expiry:
            expires_at = now + timedelta(days=30)
        record = ProfileRecord(
            profile_id=manifest.profile_id,
            record_id=self._ids("record"),
            scope_id=scope.scope_id,
            kind=payload.kind,
            payload=payload,
            semantic_key=(
                None
                if semantic_key is None
                else SemanticKey.model_validate(semantic_key)
            ),
            state=RecordState.ACTIVE,
            controls=ProfileControls.model_validate(controls),
            provenance=ProfileProvenance(
                source="manual",
                actor="user",
                reason_code="settings_edit",
            ),
            version_id=self._ids("record-version"),
            parent_version_id=None,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            no_expiry=no_expiry,
        )
        return self.create_record(record)

    def get_record(self, record_id: str) -> ProfileRecord | None:
        return self._repo().get_record(record_id)

    def update_record(
        self,
        record_id: str,
        mutation: RecordMutation,
        *,
        expected_version_id: str,
    ) -> ProfileRecord:
        current = self._require_current(record_id, expected_version_id)
        if current.state is RecordState.DELETED:
            raise ValueError("Deleted records cannot be updated.")
        if mutation.payload is not None and mutation.payload.kind != current.kind.value:
            raise ValueError("Record kind cannot be changed.")
        controls = mutation.controls or current.controls
        semantic_key = (
            None
            if mutation.clear_semantic_key
            else mutation.semantic_key or current.semantic_key
        )
        expires_at = current.expires_at
        no_expiry = current.no_expiry
        if mutation.expires_at is not None:
            expires_at, no_expiry = mutation.expires_at, False
        elif mutation.no_expiry is True:
            expires_at, no_expiry = None, True
        elif mutation.no_expiry is False:
            expires_at, no_expiry = None, False
        if (
            current.controls.sync_mode is SyncMode.SYNCABLE
            and controls.sync_mode is SyncMode.DEVICE_ONLY
        ):
            return self._convert_to_device_only(
                current,
                mutation,
                controls=controls,
                semantic_key=semantic_key,
                expires_at=expires_at,
                no_expiry=no_expiry,
            )
        next_record = ProfileRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "kind": (mutation.payload or current.payload).kind,
                "payload": mutation.payload or current.payload,
                "semantic_key": semantic_key,
                "controls": controls,
                "version_id": self._ids("record-version"),
                "parent_version_id": current.version_id,
                "updated_at": self.clock(),
                "expires_at": expires_at,
                "no_expiry": no_expiry,
            }
        )
        manifest_fence = self.get_manifest()
        self._require_no_collision(next_record, excluding_record_id=record_id)
        return self._commit_record(
            next_record,
            expected_version_id=expected_version_id,
            before=current,
            manifest_fence=manifest_fence,
        )

    def _convert_to_device_only(
        self,
        current: ProfileRecord,
        mutation: RecordMutation,
        *,
        controls: ProfileControls,
        semantic_key: SemanticKey | None,
        expires_at: datetime | None,
        no_expiry: bool,
    ) -> ProfileRecord:
        now = self.clock()
        private_record = ProfileRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "record_id": self._ids("record-device"),
                "kind": (mutation.payload or current.payload).kind,
                "payload": mutation.payload or current.payload,
                "semantic_key": semantic_key,
                "state": RecordState.ACTIVE,
                "controls": controls,
                "version_id": self._ids("record-version"),
                "parent_version_id": None,
                "created_at": now,
                "updated_at": now,
                "expires_at": expires_at,
                "no_expiry": no_expiry,
            }
        )
        tombstone = ProfileRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "payload": None,
                "semantic_key": None,
                "state": RecordState.DELETED,
                "version_id": self._ids("record-version"),
                "parent_version_id": current.version_id,
                "updated_at": now,
                "expires_at": None,
                "no_expiry": False,
            }
        )
        manifest = self.get_manifest()
        self._require_no_collision(
            private_record, excluding_record_id=current.record_id
        )
        next_manifest = self._next_manifest(manifest)
        try:
            self._repo().commit_device_only_split(
                tombstone,
                private_record,
                next_manifest,
                expected_record_version=current.version_id,
                expected_manifest_version=manifest.current_version_id,
            )
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError(
                "Personal Context changed concurrently."
            ) from exc
        return private_record

    def archive_record(
        self, record_id: str, *, expected_version_id: str
    ) -> ProfileRecord:
        return self._transition(
            record_id,
            expected_version_id,
            required=RecordState.ACTIVE,
            target=RecordState.ARCHIVED,
        )

    def restore_record(
        self, record_id: str, *, expected_version_id: str
    ) -> ProfileRecord:
        return self._transition(
            record_id,
            expected_version_id,
            required=RecordState.ARCHIVED,
            target=RecordState.ACTIVE,
        )

    def delete_record(
        self, record_id: str, *, expected_version_id: str
    ) -> ProfileRecord:
        current = self._require_current(record_id, expected_version_id)
        if current.state is RecordState.DELETED:
            raise ValueError("Record is already deleted.")
        tombstone = ProfileRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "payload": None,
                "semantic_key": None,
                "state": RecordState.DELETED,
                "version_id": self._ids("record-version"),
                "parent_version_id": current.version_id,
                "updated_at": self.clock(),
                "expires_at": None,
                "no_expiry": False,
            }
        )
        return self._commit_record(
            tombstone, expected_version_id=expected_version_id, before=current
        )

    def _transition(
        self,
        record_id: str,
        expected_version_id: str,
        *,
        required: RecordState,
        target: RecordState,
    ) -> ProfileRecord:
        current = self._require_current(record_id, expected_version_id)
        if current.state is not required:
            raise ValueError(f"Record must be {required.value}.")
        next_record = ProfileRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "state": target,
                "version_id": self._ids("record-version"),
                "parent_version_id": current.version_id,
                "updated_at": self.clock(),
            }
        )
        manifest_fence = self.get_manifest()
        self._require_no_collision(next_record, excluding_record_id=record_id)
        return self._commit_record(
            next_record,
            expected_version_id=expected_version_id,
            before=current,
            manifest_fence=manifest_fence,
        )

    def _require_current(
        self, record_id: str, expected_version_id: str
    ) -> ProfileRecord:
        current = self._repo().get_record(record_id)
        if current is None:
            raise KeyError(record_id)
        self._require_record_identity(current)
        if current.version_id != expected_version_id:
            raise ProfileConflictError("Record changed concurrently.")
        return current

    def list_records(
        self, *, scope_ids: tuple[str, ...], include_archived: bool = False
    ) -> tuple[ProfileRecord, ...]:
        known = {scope.scope_id for scope in self.list_scopes()}
        if any(scope_id not in known for scope_id in scope_ids):
            raise ValueError("Unknown profile scope.")
        now = self.clock()
        return tuple(
            record
            for record in self._repo().list_records()
            if record.scope_id in scope_ids
            and record.state is not RecordState.DELETED
            and (include_archived or record.state is RecordState.ACTIVE)
            and (record.expires_at is None or record.expires_at > now)
        )

    def list_undo_ids(self) -> tuple[str, ...]:
        return tuple(self._repo().list_undo_ids(now=normalize_datetime(self.clock())))

    def undo(self, undo_id: str) -> ProfileRecord:
        body = self._repo().get_undo(undo_id, now=normalize_datetime(self.clock()))
        if body is None:
            raise ValueError("Undo artifact is unavailable or expired.")
        try:
            before = ProfileRecord.model_validate(body["before_record"])
            expected = body["expected_head_version"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Undo artifact is invalid.") from exc
        current = self._require_current(before.record_id, expected)
        restored = ProfileRecord.model_validate(
            {
                **before.model_dump(mode="python"),
                "version_id": self._ids("record-version"),
                "parent_version_id": current.version_id,
                "updated_at": self.clock(),
            }
        )
        manifest_fence = self.get_manifest()
        self._require_no_collision(restored, excluding_record_id=restored.record_id)
        return self._commit_record(
            restored,
            expected_version_id=current.version_id,
            consume_undo_id=undo_id,
            manifest_fence=manifest_fence,
        )

    def _global_policy(self) -> GlobalRuntimePolicy:
        try:
            body = self._repo().get_runtime_policy(GLOBAL_POLICY_ID)
            return (
                GlobalRuntimePolicy()
                if body is None
                else GlobalRuntimePolicy.model_validate(body)
            )
        except (ProfileIntegrityError, TypeError, ValueError):
            raise PersonalContextAuthorityError("runtime_policy_invalid") from None

    def set_runtime_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        repository = self._repo()
        self.get_manifest()
        repository.commit_runtime_policy(
            GLOBAL_POLICY_ID,
            GlobalRuntimePolicy(enabled=enabled).model_dump(mode="json"),
            expected_version_id=repository.get_runtime_policy_version(GLOBAL_POLICY_ID),
        )

    def set_scope_authority(
        self,
        scope_id: str,
        authority: AgentAuthority,
        *,
        expected_policy_version_id: str | None | object = _UNSET_POLICY_VERSION,
    ) -> None:
        authority = AgentAuthority(authority)
        scope = self._require_scope(scope_id)
        if (
            scope.kind is ScopeKind.WORKSPACE
            and scope_id not in self.list_workspace_bindings()
        ):
            raise PersonalContextAuthorityError("scope_unmapped")
        repository = self._repo()
        expected_version_id = (
            repository.get_runtime_policy_version(scope_id)
            if expected_policy_version_id is _UNSET_POLICY_VERSION
            else expected_policy_version_id
        )
        try:
            repository.commit_runtime_policy(
                scope_id,
                ScopeRuntimePolicy(authority=authority).model_dump(mode="json"),
                expected_version_id=expected_version_id,
            )
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError(
                "Personal Context authority changed concurrently."
            ) from exc

    def get_scope_authority(self, scope_id: str) -> AgentAuthority:
        """Return one authenticated peer-local scope policy for inspection."""

        self._require_scope(scope_id)
        try:
            body = self._repo().get_runtime_policy(scope_id)
            return (
                AgentAuthority.PROPOSE
                if body is None
                else ScopeRuntimePolicy.model_validate(body).authority
            )
        except (ProfileIntegrityError, TypeError, ValueError):
            raise PersonalContextAuthorityError("agent_authority_denied") from None

    def _scope_authority_snapshot(
        self, scope_id: str
    ) -> tuple[AgentAuthority, str | None]:
        repository = self._repo()
        for _attempt in range(3):
            version_before = repository.get_runtime_policy_version(scope_id)
            authority = self.get_scope_authority(scope_id)
            version_after = repository.get_runtime_policy_version(scope_id)
            if version_before == version_after:
                return authority, version_after
        raise ProfileConflictError("Personal Context authority changed concurrently.")

    def settings_snapshot(self) -> PersonalContextSettingsSnapshot:
        """Return the immutable, service-owned Settings presentation snapshot."""

        status = self.status()
        if status.state in {
            ProfileOperationalState.ABSENT,
            ProfileOperationalState.REMOVED,
            ProfileOperationalState.LOCKED,
        }:
            return PersonalContextSettingsSnapshot(status=status)
        scopes = self.list_scopes()
        bindings = self.list_workspace_bindings()
        scope_policies = {
            scope.scope_id: self._scope_authority_snapshot(scope.scope_id)
            for scope in scopes
        }
        scope_rows = tuple(
            SettingsScopeSnapshot(
                scope=scope,
                label=(
                    "Global"
                    if scope.kind is ScopeKind.GLOBAL
                    else (
                        str(bindings[scope.scope_id].get("label") or "")
                        if scope.scope_id in bindings
                        else ""
                    )
                    or "Unlinked workspace"
                ),
                linked=(scope.kind is ScopeKind.GLOBAL or scope.scope_id in bindings),
                authority=scope_policies[scope.scope_id][0],
                policy_version_id=scope_policies[scope.scope_id][1],
            )
            for scope in scopes
        )
        records = tuple(
            record
            for record in self._repo().list_records()
            if record.state is not RecordState.DELETED
        )
        return PersonalContextSettingsSnapshot(status, scope_rows, records)

    def authorized_context_view(
        self,
        *,
        active_workspace_id: str | None = None,
        active_workspace_scope_id: str | None = None,
    ) -> AuthorizedProfileContextView:
        """Return one version-fenced, agent-readable canonical record view.

        ``active_workspace_id`` is the Console's peer-local workspace identity.
        The canonical-id argument exists for non-Console callers and is accepted
        only when that scope has an authenticated local mapping. Concurrent
        canonical or policy changes fail closed instead of returning a mixed
        view.
        """

        if active_workspace_id is not None and active_workspace_scope_id is not None:
            raise ValueError("Specify one active workspace identity, not both.")
        status = self.status()
        if status.state is not ProfileOperationalState.READY:
            raise PersonalContextAuthorityError(
                status.reason_code or "personal_context_unavailable"
            )
        repository = self._repo()
        global_policy_version_before = repository.get_runtime_policy_version(
            GLOBAL_POLICY_ID
        )
        global_policy = self._global_policy()
        global_policy_version = repository.get_runtime_policy_version(GLOBAL_POLICY_ID)
        if global_policy_version_before != global_policy_version:
            raise ProfileConflictError(
                "Personal Context changed concurrently while building context."
            )
        if not global_policy.enabled:
            raise PersonalContextAuthorityError("personal_context_disabled")
        manifest, scopes, records, _proposals = repository.read_export_snapshot()
        workspace_scope_ids = tuple(
            scope.scope_id for scope in scopes if scope.kind is ScopeKind.WORKSPACE
        )
        binding_versions_before = tuple(
            (scope_id, repository.get_scope_binding_version(scope_id))
            for scope_id in workspace_scope_ids
        )
        bindings = self.list_workspace_bindings()
        binding_versions_after = tuple(
            (scope_id, repository.get_scope_binding_version(scope_id))
            for scope_id in workspace_scope_ids
        )
        if binding_versions_before != binding_versions_after:
            raise ProfileConflictError(
                "Personal Context changed concurrently while building context."
            )
        stable_binding_versions = dict(binding_versions_after)
        global_scopes = tuple(
            scope for scope in scopes if scope.kind is ScopeKind.GLOBAL
        )
        if len(global_scopes) != 1:
            raise PersonalContextAuthorityError("profile_scope_invalid")
        global_scope = global_scopes[0]
        workspace_scope = None
        if active_workspace_scope_id is not None:
            workspace_scope = next(
                (
                    scope
                    for scope in scopes
                    if scope.kind is ScopeKind.WORKSPACE
                    and scope.scope_id == active_workspace_scope_id
                    and scope.scope_id in bindings
                ),
                None,
            )
            if workspace_scope is None:
                raise PersonalContextAuthorityError("scope_unmapped")
        elif active_workspace_id is not None:
            matching_scope_ids = tuple(
                scope_id
                for scope_id, body in bindings.items()
                if body.get("local_workspace_id") == active_workspace_id
            )
            if len(matching_scope_ids) > 1:
                raise PersonalContextAuthorityError("workspace_mapping_ambiguous")
            if matching_scope_ids:
                workspace_scope = next(
                    (
                        scope
                        for scope in scopes
                        if scope.kind is ScopeKind.WORKSPACE
                        and scope.scope_id == matching_scope_ids[0]
                    ),
                    None,
                )
            if workspace_scope is None:
                raise PersonalContextAuthorityError("scope_unmapped")

        selected_scopes = (global_scope,) + (
            (workspace_scope,) if workspace_scope is not None else ()
        )
        scope_policy_versions: list[tuple[str, str | None]] = []
        for scope in selected_scopes:
            authority, version = self._scope_authority_snapshot(scope.scope_id)
            if not authority_allows(authority, AgentAuthority.READ_ONLY):
                raise PersonalContextAuthorityError("agent_authority_denied")
            scope_policy_versions.append((scope.scope_id, version))

        selected_binding_version = (
            stable_binding_versions.get(workspace_scope.scope_id)
            if workspace_scope is not None
            else None
        )
        current_manifest = repository.get_manifest()
        current_global_policy_version = repository.get_runtime_policy_version(
            GLOBAL_POLICY_ID
        )
        current_scope_policy_versions = tuple(
            (scope_id, repository.get_runtime_policy_version(scope_id))
            for scope_id, _version in scope_policy_versions
        )
        current_binding_version = (
            repository.get_scope_binding_version(workspace_scope.scope_id)
            if workspace_scope is not None
            else None
        )
        if (
            current_manifest is None
            or current_manifest.current_version_id != manifest.current_version_id
            or current_manifest.purge_generation != manifest.purge_generation
            or current_global_policy_version != global_policy_version
            or current_scope_policy_versions != tuple(scope_policy_versions)
            or current_binding_version != selected_binding_version
        ):
            raise ProfileConflictError(
                "Personal Context changed concurrently while building context."
            )

        quarantine = repository.list_quarantine()
        unsupported = tuple(
            (entry.object_id, entry.version_id)
            for entry in quarantine
            if entry.object_type == "record"
            and entry.reason_code.startswith("unsupported")
        )
        authority_revision = hashlib.sha256(
            json.dumps(
                {
                    "global_policy": global_policy_version,
                    "scope_policies": scope_policy_versions,
                    "workspace_binding": selected_binding_version,
                    "unsupported": unsupported,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        selected_scope_ids = {scope.scope_id for scope in selected_scopes}
        return AuthorizedProfileContextView(
            generation=manifest.purge_generation,
            record_set_revision=manifest.current_version_id,
            workspace_scope_id=(
                workspace_scope.scope_id if workspace_scope is not None else None
            ),
            authority_revision=authority_revision,
            records=tuple(
                record for record in records if record.scope_id in selected_scope_ids
            ),
            unsupported_records_present=bool(unsupported),
        )

    def require_agent_authority(self, scope_id: str, required: AgentAuthority) -> None:
        status = self.status()
        if status.locked:
            raise PersonalContextAuthorityError("profile_locked")
        if not status.runtime_enabled:
            raise PersonalContextAuthorityError(
                status.reason_code or "personal_context_disabled"
            )
        try:
            scope = self._require_scope(scope_id)
        except ValueError:
            raise PersonalContextAuthorityError("scope_unknown") from None
        if (
            scope.kind is ScopeKind.WORKSPACE
            and scope_id not in self.list_workspace_bindings()
        ):
            raise PersonalContextAuthorityError("scope_unmapped")
        try:
            body = self._repo().get_runtime_policy(scope_id)
            if body is None:
                actual = AgentAuthority.PROPOSE
            else:
                actual = ScopeRuntimePolicy.model_validate(body).authority
        except (ProfileIntegrityError, TypeError, ValueError):
            raise PersonalContextAuthorityError("agent_authority_denied") from None
        try:
            required = AgentAuthority(required)
        except (TypeError, ValueError):
            raise PersonalContextAuthorityError("agent_authority_denied") from None
        if not authority_allows(actual, required):
            raise PersonalContextAuthorityError("agent_authority_denied")

    def remove_local_profile(self, *, confirm_only_copy: bool) -> None:
        if confirm_only_copy is not True:
            raise ValueError(
                "Explicit confirmation is required to destroy the only copy."
            )
        with self._destructive_lifecycle_lock:
            self._repo().destroy_profile_content()
            self._locked_reason = None
            self._profile_present_hint = False

    def finish_secure_removal(self) -> None:
        """Retry key-custody deletion without creating a new profile generation."""

        with self._destructive_lifecycle_lock:
            repository = self._repo()
            if not repository.is_destroyed():
                raise ValueError(
                    "Secure-removal repair is available only after removal."
                )
            repository.destroy_profile_content()
            self._locked_reason = None
            self._profile_present_hint = False

    def export_plaintext(self, request: Any):
        from .export_service import export_plaintext

        return export_plaintext(self, request)

    def snapshot_for_export(
        self,
    ) -> tuple[
        ProfileManifest,
        tuple[ProfileScope, ...],
        tuple[ProfileRecord, ...],
        tuple[ProfileProposal, ...],
    ]:
        """Return one transactionally consistent canonical export snapshot."""

        return self._repo().read_export_snapshot()

    def export_recovery(self, request: Any):
        from .export_service import export_recovery

        return export_recovery(self, request)
