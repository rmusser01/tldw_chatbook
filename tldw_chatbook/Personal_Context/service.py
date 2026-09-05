"""Canonical authorized application boundary for local Personal Context."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Lock
from typing import Any

from tldw_profile_core import (
    AgentVisibility,
    InterviewAudience,
    InterviewProposedChange,
    ProfileControls,
    ProfileManifest,
    ProfilePayload,
    ProfileProvenance,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    ProposalOperation,
    ProposalState,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
    normalize_datetime,
)

from .key_protector import ProfileLockedError
from .repository import (
    AgentAuthorityFence,
    ConcurrentProfileUpdateError,
    PersonalContextRepository,
    ProfileIntegrityError,
    RecordCollisionError,
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
    proposals: tuple[ProfileProposal, ...] = field(default=(), repr=False)


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
        self._absent_storage_signature: tuple | None = None

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

    @contextmanager
    def read_operation(self) -> Iterator[None]:
        """Bound synchronous read reuse without caching live agent authority."""

        if self._repository is None or self._locked_reason is not None:
            yield
            return
        with self._repository.operation():
            yield

    def _new_profile_id(self, label: str) -> str:
        """Issue a canonical identity for one service-owned collaborator."""

        return self._ids(label)

    def _commit_profile_proposal(
        self,
        proposal: ProfileProposal,
        *,
        authority_fence: AgentAuthorityFence,
    ) -> None:
        """Persist a proposal through the application-owned mutation boundary."""

        try:
            self._repo().commit_proposal(
                proposal,
                expire_before=self.clock(),
                authority_fence=authority_fence,
            )
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError(
                "Personal Context authority changed concurrently."
            ) from exc

    def _get_profile_proposal(self, proposal_id: str) -> ProfileProposal | None:
        """Read one proposal for the service-owned review collaborator."""

        self._repo().expire_due_proposals(self.clock())
        return self._repo().get_proposal(proposal_id)

    def _list_profile_proposals(self) -> tuple[ProfileProposal, ...]:
        """Read proposal heads for the service-owned review collaborator."""

        self._repo().expire_due_proposals(self.clock())
        return tuple(self._repo().list_proposals())

    def _resolve_profile_proposal(self, proposal_id: str, state) -> ProfileProposal:
        """Write a content-free proposal receipt through the app boundary."""

        self._repo().expire_due_proposals(self.clock())
        return self._repo().resolve_proposal(proposal_id, state)

    def _accept_profile_proposal(
        self,
        proposal_id: str,
        record: ProfileRecord,
        *,
        expected_record_version: str | None,
        allow_user_review_rewrite: bool = False,
    ) -> ProfileRecord:
        """Atomically apply an accepted proposal and its terminal receipt."""

        manifest = self.get_manifest()
        self._require_record_identity(record)
        try:
            receipt = self._repo().accept_proposal_and_record(
                proposal_id,
                record,
                self._next_manifest(manifest),
                expected_record_version=expected_record_version,
                expected_manifest_version=manifest.current_version_id,
                outbox_body={"version": 1, "record": record.model_dump(mode="json")},
                expire_before=self.clock(),
                allow_user_review_rewrite=allow_user_review_rewrite,
            )
            if receipt.state is ProposalState.EXPIRED:
                raise ValueError("proposal_expired")
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError(
                "Personal Context changed concurrently."
            ) from exc
        except RecordCollisionError as exc:
            raise ProfileKeyCollisionError(exc.record_id) from exc
        return record

    def _apply_direct_profile_update(
        self,
        request,
        *,
        scope_id: str,
        evidence_hash: str,
        authority_fence: AgentAuthorityFence,
    ) -> ProfileRecord:
        """Apply a trusted current-user-evidence update with inherited controls."""

        current = self._require_current(request.record_id, request.base_version_id)
        if current.scope_id != scope_id:
            raise PersonalContextAuthorityError("scope_mismatch")
        self._require_agent_eligible_record(current, scope_id)
        if request.proposed_payload.kind != current.kind.value:
            raise ValueError("Record kind cannot be changed.")
        updated = ProfileRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "payload": request.proposed_payload,
                "semantic_key": current.semantic_key,
                "controls": current.controls,
                "provenance": ProfileProvenance(
                    source="agent",
                    actor="agent",
                    reason_code="explicit_user_statement",
                    source_references=(request.current_user_message_id,),
                    source_hashes=(evidence_hash,),
                ),
                "version_id": self._ids("record-version"),
                "parent_version_id": current.version_id,
                "updated_at": self.clock(),
            }
        )
        manifest = self.get_manifest()
        self._require_no_collision(updated, excluding_record_id=updated.record_id)
        return self._commit_record(
            updated,
            expected_version_id=current.version_id,
            before=current,
            manifest_fence=manifest,
            authority_fence=authority_fence,
        )

    def proposal_service(self, *, quota=None):
        """Return the proposal collaborator owned by this application service."""

        from .proposal_service import ProfileProposalQuota, ProfileProposalService

        quota = ProfileProposalQuota() if quota is None else quota
        return ProfileProposalService(self, quota=quota)

    def status(self) -> ProfileOperationalStatus:
        if self._repository is None or self._locked_reason is not None:
            return ProfileOperationalStatus(
                ProfileOperationalState.LOCKED,
                self._profile_present_hint,
                True,
                False,
                self._locked_reason or "profile_locked",
            )
        cached_signature = self._absent_storage_signature
        self._absent_storage_signature = None
        signature = self._repository.storage_signature()
        if cached_signature == signature:
            self._absent_storage_signature = signature
            return ProfileOperationalStatus(
                ProfileOperationalState.ABSENT, False, False, False, "profile_absent"
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
            if signature == self._repository.storage_signature():
                self._absent_storage_signature = signature
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
        self._absent_storage_signature = None
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

    def apply_sync_object(
        self,
        *,
        domain: str,
        value: ProfileManifest
        | ProfileScope
        | ProfileRecord
        | ProfileProposal
        | Mapping[str, Any],
        actor_type: str,
        actor_id: str | None,
        base_object_hash: str | None = None,
    ) -> ProfileManifest | ProfileScope | ProfileRecord | ProfileProposal | Mapping[str, Any]:
        """Apply one adapter-authenticated whole object without outbox echo."""

        del base_object_hash
        if actor_type != "sync" or not isinstance(actor_id, str) or not actor_id:
            raise PermissionError("Personal Context sync actor is invalid.")
        try:
            if domain == "personal_context.manifest":
                manifest = ProfileManifest.model_validate(value)
                current = self.get_manifest()
                if manifest == current:
                    return current
                if manifest.profile_id != current.profile_id:
                    raise ProfileConflictError("Personal Context profile changed.")
                self._repo().commit_manifest_version(
                    manifest,
                    expected_version_id=current.current_version_id,
                )
                return manifest

            manifest = self.get_manifest()
            if domain == "personal_context.scope":
                scope = ProfileScope.model_validate(value)
                if scope.profile_id != manifest.profile_id:
                    raise ProfileConflictError("Personal Context profile changed.")
                current_scope = self._repo().get_scope(scope.scope_id)
                if scope == current_scope:
                    return scope
                if current_scope is not None and (
                    scope.kind is not current_scope.kind
                    or scope.created_at != current_scope.created_at
                    or scope.updated_at < current_scope.updated_at
                    or scope.version_id == current_scope.version_id
                ):
                    raise ProfileConflictError("Personal Context scope changed.")
                if (
                    current_scope is None
                    and scope.kind is ScopeKind.GLOBAL
                    and any(
                        candidate.kind is ScopeKind.GLOBAL
                        for candidate in self._repo().list_scopes()
                    )
                ):
                    raise ProfileConflictError("Personal Context global scope changed.")
                self._repo().commit_scope(
                    scope,
                    expected_version_id=(
                        None if current_scope is None else current_scope.version_id
                    ),
                )
                return scope

            if domain == "personal_context.record":
                record = ProfileRecord.model_validate(value)
                if record.profile_id != manifest.profile_id:
                    raise ProfileConflictError("Personal Context profile changed.")
                scope = self._repo().get_scope(record.scope_id)
                if scope is None or scope.profile_id != manifest.profile_id:
                    raise ProfileConflictError("Personal Context scope changed.")
                if record.controls.sync_mode is SyncMode.DEVICE_ONLY:
                    raise ValueError("Device-only records cannot synchronize.")
                current_record = self._repo().get_record(record.record_id)
                if record == current_record:
                    return record
                expected_version = (
                    None if current_record is None else current_record.version_id
                )
                orphan_tombstone = (
                    current_record is None
                    and record.state is RecordState.DELETED
                    and record.payload is None
                    and record.parent_version_id is not None
                )
                if record.parent_version_id != expected_version and not orphan_tombstone:
                    raise ProfileConflictError("Personal Context record changed.")
                if current_record is not None and (
                    current_record.state is RecordState.DELETED
                    or record.scope_id != current_record.scope_id
                    or record.kind is not current_record.kind
                    or record.created_at != current_record.created_at
                    or record.updated_at < current_record.updated_at
                    or record.version_id == current_record.version_id
                ):
                    raise ProfileConflictError("Personal Context record changed.")
                self._require_no_collision(
                    record,
                    excluding_record_id=(
                        None if current_record is None else current_record.record_id
                    ),
                )
                self._repo().commit_record_version(
                    record,
                    expected_version_id=expected_version,
                    outbox_body=None,
                    allow_orphan_tombstone=orphan_tombstone,
                )
                return record

            if domain == "personal_context.proposal":
                proposal = ProfileProposal.model_validate(value)
                if proposal.profile_id != manifest.profile_id:
                    raise ProfileConflictError("Personal Context profile changed.")
                if (
                    proposal.state is ProposalState.PENDING
                    and proposal.proposed_record is not None
                    and proposal.proposed_record.controls.sync_mode
                    is SyncMode.DEVICE_ONLY
                ):
                    raise ValueError("Device-only proposals cannot synchronize.")
                scope = self._repo().get_scope(proposal.scope_id)
                if scope is None or scope.profile_id != manifest.profile_id:
                    raise ProfileConflictError("Personal Context scope changed.")
                current_proposal = self._repo().get_proposal(proposal.proposal_id)
                if proposal == current_proposal:
                    return proposal
                if current_proposal is None and proposal.state is ProposalState.PENDING:
                    self._repo().commit_proposal(proposal, enqueue_outbox=False)
                else:
                    self._repo().commit_synced_proposal(proposal)
                return proposal

            if domain == "personal_context.purge":
                barrier = dict(value)
                if (
                    set(barrier)
                    != {"schema_version", "profile_id", "purge_generation"}
                    or barrier.get("schema_version") != 1
                    or barrier.get("profile_id") != manifest.profile_id
                ):
                    raise ProfileConflictError("Personal Context purge changed.")
                if barrier.get("purge_generation") == manifest.purge_generation:
                    return barrier
                raise ProfileConflictError("Personal Context purge requires rebootstrap.")
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError("Personal Context changed concurrently.") from exc
        raise ValueError("Unsupported Personal Context Sync domain.")

    def start_fresh_profile(self) -> ProfileManifest:
        """Create a new profile only after an explicit local-removal transition."""

        self._absent_storage_signature = None
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

    def first_link_snapshot(
        self,
    ) -> tuple[
        ProfileManifest,
        tuple[ProfileScope, ...],
        tuple[ProfileRecord, ...],
        tuple[ProfileProposal, ...],
        dict[str, dict[str, Any]],
    ]:
        """Return canonical local heads and peer-local mappings for read-only planning."""

        repository = self._repo()
        return (
            self.get_manifest(),
            tuple(repository.list_scopes()),
            tuple(repository.list_records()),
            tuple(repository.list_proposals()),
            repository.list_validated_scope_bindings(),
        )

    def apply_reviewed_link(self, **kwargs: Any) -> dict[str, Any]:
        """Apply one explicit first-link decision set through the canonical owner."""

        return self._repo().apply_reviewed_link(**kwargs)

    def acquire_first_link_freeze(
        self, *, plan_id: str, snapshot_token: str
    ) -> None:
        """Block normal mutations for one exact first-link review snapshot."""

        self._repo().acquire_first_link_freeze(
            plan_id=plan_id,
            snapshot_token=snapshot_token,
        )

    def release_first_link_freeze(self, *, plan_id: str) -> bool:
        """Release the exact review freeze after cancel or convergence."""

        return self._repo().release_first_link_freeze(plan_id=plan_id)

    def first_link_freeze_plan_id(self) -> str | None:
        """Return the content-free durable review owner during restart repair."""

        return self._repo().first_link_freeze_plan_id()

    def first_link_rebaseline_commit_plan_id(self) -> str | None:
        """Return the content-free durable rebaseline-marker owner, if any."""

        return self._repo().first_link_rebaseline_commit_plan_id()

    def first_link_reconciliation_writes(self, *, plan_id: str):
        """Authorize the private confirming pull to update canonical heads."""

        return self._repo().first_link_reconciliation_writes(plan_id=plan_id)

    def first_link_rebaseline_version(self) -> int:
        """Return the authenticated key generation after interrupted-link recovery."""

        return self._repo().current_key_version()

    def first_link_apply_recovery_state(self, **kwargs: Any) -> tuple[str, int | None]:
        """Read exact content-free interrupted-apply evidence."""

        return self._repo().first_link_apply_recovery_state(**kwargs)

    def clear_first_link_rebaseline_commit(self, **kwargs: Any) -> bool:
        """Clear the exact rebaseline marker after terminal artifact cleanup."""

        return self._repo().clear_first_link_rebaseline_commit(**kwargs)

    def authenticate_legacy_first_link_rebaseline_commit(
        self,
        *,
        plan_id: str,
        target_profile_id: str,
        target_integrity_key_id: str,
        target_key_record_id: str,
        target_purge_generation: int,
        rebaseline_version: int,
        staged_integrity_key: bytes,
    ) -> bool:
        """Bind an exact v7 marker after staged and active key authentication."""

        if not isinstance(target_key_record_id, str) or not target_key_record_id:
            return False
        if not self._repo().active_integrity_key_matches(staged_integrity_key):
            return False
        manifest = self.get_manifest()
        if (
            manifest.profile_id != target_profile_id
            or manifest.purge_generation != target_purge_generation
            or self.first_link_rebaseline_version() != rebaseline_version
        ):
            return False
        return self._repo().bind_legacy_first_link_rebaseline_commit(
            plan_id=plan_id,
            target_profile_id=target_profile_id,
            target_integrity_key_id=target_integrity_key_id,
            target_key_record_id=target_key_record_id,
            target_purge_generation=target_purge_generation,
            rebaseline_version=rebaseline_version,
        )

    def legacy_first_link_rebaseline_commit_matches(self, **kwargs: Any) -> bool:
        """Return whether an exact v7 marker needs authenticated key binding."""

        return self._repo().legacy_first_link_rebaseline_commit_matches(**kwargs)

    def first_link_sync_heads(self) -> dict[str, dict[str, str]]:
        """Return content-free eligible canonical heads for link confirmation."""

        return self._repo().first_link_sync_heads()

    def first_link_reviewed_lineage(self) -> list[list[str]]:
        """Return exact content-free reviewed heads and retained history."""

        return self._repo().first_link_reviewed_lineage()

    def build_personal_context_sync_adapter(self, integrity_key_id: str):
        """Build an adapter from active protected keys without exposing key bytes."""

        from tldw_chatbook.Sync_Interop.personal_context_adapter import (
            PersonalContextSyncAdapter,
        )

        return PersonalContextSyncAdapter(
            integrity_key=self._repo()._require_keys().integrity_key,
            integrity_key_id=integrity_key_id,
        )

    def build_personal_context_outbox_dispatcher(
        self, *, state_repository: Any, integrity_key_id: str
    ):
        """Compose the exact canonical outbox owner with its active adapter."""

        from tldw_chatbook.Sync_Interop.personal_context_dispatcher import (
            PersonalContextOutboxDispatcher,
        )

        from .sync_outbox import ProfileSyncOutbox

        return PersonalContextOutboxDispatcher(
            profile_outbox=ProfileSyncOutbox(self._repo()),
            state_repository=state_repository,
            adapter=self.build_personal_context_sync_adapter(integrity_key_id),
        )

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
            and not self._repo().is_scope_explicitly_unlinked(scope.scope_id)
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

    def _require_agent_eligible_record(
        self, record: ProfileRecord, scope_id: str
    ) -> None:
        scope = self._require_scope(scope_id)
        view = self.authorized_context_view(
            active_workspace_scope_id=(
                scope_id if scope.kind is ScopeKind.WORKSPACE else None
            )
        )
        visible_versions = {
            (candidate.record_id, candidate.version_id) for candidate in view.records
        }
        if (
            record.scope_id != scope_id
            or record.state is not RecordState.ACTIVE
            or record.payload is None
            or record.controls.agent_visibility is not AgentVisibility.AGENT_VISIBLE
            or (record.expires_at is not None and record.expires_at <= self.clock())
            or record.record_id in view.conflicted_record_ids
            or (record.record_id, record.version_id) not in visible_versions
        ):
            raise PersonalContextAuthorityError("record_ineligible")

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
        authority_fence: AgentAuthorityFence | None = None,
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
                authority_fence=authority_fence,
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

    def commit_interview_changes(
        self,
        *,
        scope_id: str,
        audience: InterviewAudience,
        changes: tuple[InterviewProposedChange, ...],
    ) -> tuple[ProfileRecord, ...]:
        """Commit selected reviewed interview changes as one manifest revision."""

        audience = InterviewAudience(audience)
        manifest = self.get_manifest()
        scope = self.validate_interview_target(
            scope_id=scope_id,
            audience=audience,
        )

        allowed_workspace_kinds = {"goal", "working_context", "convention"}
        now = self.clock()
        records: list[ProfileRecord] = []
        expected_versions: dict[str, str | None] = {}
        for change in changes:
            if change.operation is ProposalOperation.PROMOTE:
                raise ValueError("Interview review does not promote records.")
            current: ProfileRecord | None = None
            if change.target_record_id is not None:
                current = self._repo().get_record(change.target_record_id)
                if current is None:
                    raise ProfileConflictError("Interview target is unavailable.")
                self._require_record_identity(current)
                if current.scope_id != scope.scope_id:
                    raise ValueError("Interview changes cannot cross scopes.")
                if current.version_id != change.base_version_id:
                    raise ProfileConflictError("Interview target changed concurrently.")

            if change.operation in {
                ProposalOperation.CREATE,
                ProposalOperation.UPDATE,
            }:
                assert change.proposed_payload is not None
                assert change.controls is not None
                payload = change.proposed_payload
                if (
                    audience is InterviewAudience.WORKSPACE
                    and payload.kind not in allowed_workspace_kinds
                ):
                    raise ValueError(
                        "Workspace interviews allow only workspace-safe kinds."
                    )
                if current is not None and current.kind.value != payload.kind:
                    raise ValueError("Record kind cannot be changed.")
                if current is not None and current.state is not RecordState.ACTIVE:
                    raise ValueError("Interview updates require an active record.")
                if (
                    current is not None
                    and current.expires_at is not None
                    and current.expires_at <= now
                ):
                    raise ProfileConflictError("Interview target is expired.")
                if (
                    current is not None
                    and current.controls.sync_mode is SyncMode.SYNCABLE
                    and change.controls.sync_mode is SyncMode.DEVICE_ONLY
                ):
                    raise ValueError(
                        "Interview updates cannot convert syncable records to device-only."
                    )
                record_id = (
                    self._ids("record") if current is None else current.record_id
                )
                expires_at = None if current is None else current.expires_at
                no_expiry = False if current is None else current.no_expiry
                if payload.kind == "working_context" and current is None:
                    expires_at = now + timedelta(days=30)
                record = ProfileRecord(
                    profile_id=manifest.profile_id,
                    record_id=record_id,
                    scope_id=scope.scope_id,
                    kind=payload.kind,
                    payload=payload,
                    semantic_key=change.semantic_key,
                    state=RecordState.ACTIVE,
                    controls=change.controls,
                    provenance=ProfileProvenance(
                        source="manual",
                        actor="user",
                        reason_code="interview_review",
                    ),
                    version_id=self._ids("record-version"),
                    parent_version_id=None if current is None else current.version_id,
                    created_at=now if current is None else current.created_at,
                    updated_at=now,
                    expires_at=expires_at,
                    no_expiry=no_expiry,
                )
            elif change.operation is ProposalOperation.ARCHIVE:
                if current is None or current.state is not RecordState.ACTIVE:
                    raise ProfileConflictError(
                        "Interview archive target is unavailable."
                    )
                record = ProfileRecord.model_validate(
                    {
                        **current.model_dump(mode="python"),
                        "state": RecordState.ARCHIVED,
                        "version_id": self._ids("record-version"),
                        "parent_version_id": current.version_id,
                        "updated_at": now,
                    }
                )
            else:
                raise ValueError("Unsupported interview change operation.")
            self._require_record_identity(record)
            records.append(record)
            expected_versions[record.record_id] = (
                None if current is None else current.version_id
            )

        if not records:
            return ()
        replaced = {record.record_id for record in records}
        active_keys: dict[tuple[str, str, str, str], str] = {}
        for existing in self._repo().list_records():
            if existing.record_id in replaced or not self._active_for_collision(
                existing
            ):
                continue
            if existing.semantic_key is not None:
                active_keys[
                    (
                        existing.scope_id,
                        existing.kind.value,
                        existing.semantic_key.namespace,
                        existing.semantic_key.subject,
                    )
                ] = existing.record_id
        for record in records:
            if record.semantic_key is None or not self._active_for_collision(record):
                continue
            key = (
                record.scope_id,
                record.kind.value,
                record.semantic_key.namespace,
                record.semantic_key.subject,
            )
            if key in active_keys:
                raise ProfileKeyCollisionError(active_keys[key])
            active_keys[key] = record.record_id
        try:
            self._repo().commit_interview_batch(
                tuple(records),
                self._next_manifest(manifest),
                expected_record_versions=expected_versions,
                expected_manifest_version=manifest.current_version_id,
            )
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError(
                "Personal Context changed concurrently."
            ) from exc
        return tuple(records)

    def validate_interview_target(
        self,
        *,
        scope_id: str,
        audience: InterviewAudience,
    ) -> ProfileScope:
        """Validate that an interview audience may read or mutate one scope."""

        audience = InterviewAudience(audience)
        scope = self._require_scope(scope_id)
        if audience is InterviewAudience.PERSONAL:
            if scope.kind is not ScopeKind.GLOBAL:
                raise ValueError("Personal interviews require the global scope.")
        elif scope.kind is not ScopeKind.WORKSPACE:
            raise ValueError("Workspace interviews require a workspace scope.")
        elif scope.scope_id not in self.list_workspace_bindings():
            raise ValueError("Workspace scope must be mapped before mutation.")
        return scope

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
        proposals = tuple(
            proposal
            for proposal in self._list_profile_proposals()
            if proposal.state is ProposalState.PENDING
        )
        return PersonalContextSettingsSnapshot(
            status=status,
            scopes=scope_rows,
            records=records,
            proposals=proposals,
        )

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

        with self.read_operation():
            return self._authorized_context_view(
                active_workspace_id=active_workspace_id,
                active_workspace_scope_id=active_workspace_scope_id,
            )

    def _authorized_context_view(
        self,
        *,
        active_workspace_id: str | None,
        active_workspace_scope_id: str | None,
    ) -> AuthorizedProfileContextView:
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

    def _capture_agent_authority_fence(
        self, scope_id: str, required: AgentAuthority
    ) -> AgentAuthorityFence:
        repository = self._repo()
        scope = self._require_scope(scope_id)

        def versions() -> tuple[str | None, str | None, str | None]:
            return (
                repository.get_runtime_policy_version(GLOBAL_POLICY_ID),
                repository.get_runtime_policy_version(scope_id),
                repository.get_scope_binding_version(scope_id),
            )

        for _attempt in range(3):
            before = versions()
            self.require_agent_authority(scope_id, required)
            after = versions()
            if before == after:
                return AgentAuthorityFence(
                    scope_id=scope_id,
                    global_policy_version=after[0],
                    scope_policy_version=after[1],
                    binding_version=after[2],
                    binding_required=scope.kind is ScopeKind.WORKSPACE,
                )
        raise ProfileConflictError("Personal Context authority changed concurrently.")

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
