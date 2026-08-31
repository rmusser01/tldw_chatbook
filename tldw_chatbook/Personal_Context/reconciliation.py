"""Content-free planning for reviewed Personal Context first-link reconciliation."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from tldw_profile_core import (
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    RecordState,
    ScopeKind,
    SyncMode,
    canonical_bytes,
)


@dataclass(frozen=True, slots=True)
class CanonicalBootstrapSnapshot:
    """One transient, cursor-bounded canonical server snapshot.

    Canonical bodies and wrapped key material are intentionally omitted from the
    representation so diagnostics and UI state cannot accidentally disclose them.
    """

    dataset_id: str
    authority_id: str
    manifest: ProfileManifest = field(repr=False)
    scopes: tuple[ProfileScope, ...] = field(repr=False)
    records: tuple[ProfileRecord, ...] = field(repr=False)
    proposals: tuple[ProfileProposal, ...] = field(repr=False)
    purge_generation: int
    schema_version: int
    quotas: Mapping[str, int] = field(repr=False)
    cursor: str
    integrity_key_id: str
    key_record_id: str
    wrapped_key_blob: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.manifest.purge_generation != self.purge_generation:
            raise ValueError("bootstrap_purge_generation_mismatch")
        profile_id = self.manifest.profile_id
        if any(item.profile_id != profile_id for item in self.scopes):
            raise ValueError("bootstrap_scope_profile_mismatch")
        if any(item.profile_id != profile_id for item in self.records):
            raise ValueError("bootstrap_record_profile_mismatch")
        if any(item.profile_id != profile_id for item in self.proposals):
            raise ValueError("bootstrap_proposal_profile_mismatch")
        if not self.dataset_id or not self.authority_id or not self.cursor:
            raise ValueError("bootstrap_binding_invalid")

    @classmethod
    def from_response(cls, response: object) -> "CanonicalBootstrapSnapshot":
        """Build a snapshot from the typed API model or an equivalent object."""

        def field_value(name: str) -> object:
            if isinstance(response, Mapping):
                return response[name]
            return getattr(response, name)

        return cls(
            dataset_id=str(field_value("dataset_id")),
            authority_id=str(field_value("authority_id")),
            manifest=ProfileManifest.model_validate(field_value("manifest")),
            scopes=tuple(
                ProfileScope.model_validate(item)
                for item in field_value("scopes")
            ),
            records=tuple(
                ProfileRecord.model_validate(item)
                for item in field_value("records")
            ),
            proposals=tuple(
                ProfileProposal.model_validate(item)
                for item in field_value("proposals")
            ),
            purge_generation=int(field_value("purge_generation")),
            schema_version=int(field_value("schema_version")),
            quotas=dict(field_value("quotas")),
            cursor=str(field_value("cursor")),
            integrity_key_id=str(field_value("integrity_key_id")),
            key_record_id=str(field_value("key_record_id")),
            wrapped_key_blob=str(field_value("wrapped_key_blob")),
        )


@dataclass(frozen=True, slots=True)
class VersionConflict:
    decision_id: str
    record_id: str
    local_version_id: str
    server_version_id: str


@dataclass(frozen=True, slots=True)
class KeyCollision:
    decision_id: str
    scope_id: str
    record_ids: tuple[str, str]


@dataclass(frozen=True, slots=True)
class ReconciliationPlan:
    """User-reviewable content-free merge metadata for one exact cursor."""

    plan_id: str
    dataset_id: str
    authority_id: str
    local_profile_id: str
    server_profile_id: str
    bootstrap_cursor: str
    integrity_key_id: str
    key_record_id: str
    purge_generation: int
    schema_version: int
    global_scope_mapping: tuple[str, str]
    exact_record_ids: tuple[str, ...]
    local_only_record_ids: tuple[str, ...]
    remote_only_record_ids: tuple[str, ...]
    device_only_record_ids: tuple[str, ...]
    version_conflicts: tuple[VersionConflict, ...]
    key_collisions: tuple[KeyCollision, ...]
    unlinked_remote_scope_ids: tuple[str, ...]
    local_workspace_scope_ids: tuple[str, ...]
    required_decision_ids: tuple[str, ...]
    attention_codes: tuple[str, ...]
    local_snapshot_token: str

    @property
    def can_approve(self) -> bool:
        return not self.attention_codes


def _global_scope(scopes: Sequence[ProfileScope], owner: str) -> ProfileScope:
    candidates = [scope for scope in scopes if scope.kind is ScopeKind.GLOBAL]
    if len(candidates) != 1:
        raise ValueError(f"{owner}_global_scope_invalid")
    return candidates[0]


def _mapped_record(
    record: ProfileRecord,
    *,
    profile_id: str,
    scope_mapping: Mapping[str, str],
) -> ProfileRecord:
    return record.model_copy(
        update={
            "profile_id": profile_id,
            "scope_id": scope_mapping.get(record.scope_id, record.scope_id),
        }
    )


def _snapshot_token(
    manifest: ProfileManifest,
    scopes: Sequence[ProfileScope],
    records: Sequence[ProfileRecord],
    proposals: Sequence[ProfileProposal],
) -> str:
    material = "\x00".join(
        (
            manifest.profile_id,
            manifest.current_version_id,
            *(f"s:{item.scope_id}:{item.version_id}" for item in scopes),
            *(f"r:{item.record_id}:{item.version_id}" for item in records),
            *(f"p:{item.proposal_id}:{item.state.value}" for item in proposals),
        )
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(material).hexdigest()


def build_reconciliation_plan(
    *,
    local_manifest: ProfileManifest,
    local_scopes: Sequence[ProfileScope],
    local_records: Sequence[ProfileRecord],
    local_proposals: Sequence[ProfileProposal],
    remote: CanonicalBootstrapSnapshot,
    local_workspace_bindings: Mapping[str, Mapping[str, object]],
    plan_id: str | None = None,
) -> ReconciliationPlan:
    """Compare exact canonical identities without persisting or returning content."""

    local_global = _global_scope(local_scopes, "local")
    remote_global = _global_scope(remote.scopes, "server")
    scope_mapping = {local_global.scope_id: remote_global.scope_id}
    mapped_local = {
        record.record_id: _mapped_record(
            record,
            profile_id=remote.manifest.profile_id,
            scope_mapping=scope_mapping,
        )
        for record in local_records
    }
    remote_by_id = {record.record_id: record for record in remote.records}

    exact: list[str] = []
    local_only: list[str] = []
    remote_only: list[str] = []
    device_only: list[str] = []
    version_conflicts: list[VersionConflict] = []

    for original in local_records:
        if original.controls.sync_mode is SyncMode.DEVICE_ONLY:
            device_only.append(original.record_id)
            continue
        local = mapped_local[original.record_id]
        server = remote_by_id.get(original.record_id)
        if server is None:
            local_only.append(original.record_id)
        elif canonical_bytes(local) == canonical_bytes(server):
            exact.append(original.record_id)
        else:
            version_conflicts.append(
                VersionConflict(
                    decision_id=f"version:{original.record_id}",
                    record_id=original.record_id,
                    local_version_id=original.version_id,
                    server_version_id=server.version_id,
                )
            )

    local_ids = {
        record.record_id
        for record in local_records
        if record.controls.sync_mode is SyncMode.SYNCABLE
    }
    remote_only.extend(
        record.record_id for record in remote.records if record.record_id not in local_ids
    )

    collisions: list[KeyCollision] = []
    local_occupants: dict[tuple[str, str, str], ProfileRecord] = {}
    for original in local_records:
        local = mapped_local[original.record_id]
        if (
            original.controls.sync_mode is SyncMode.DEVICE_ONLY
            or local.state is not RecordState.ACTIVE
            or local.semantic_key is None
        ):
            continue
        key = (
            local.scope_id,
            local.semantic_key.namespace,
            local.semantic_key.subject,
        )
        local_occupants[key] = local
    for server in remote.records:
        if server.state is not RecordState.ACTIVE or server.semantic_key is None:
            continue
        key = (
            server.scope_id,
            server.semantic_key.namespace,
            server.semantic_key.subject,
        )
        local = local_occupants.get(key)
        if local is None or local.record_id == server.record_id:
            continue
        pair = tuple(sorted((local.record_id, server.record_id)))
        collisions.append(
            KeyCollision(
                decision_id=f"key:{server.scope_id}:{pair[0]}:{pair[1]}",
                scope_id=server.scope_id,
                record_ids=pair,
            )
        )

    remote_workspaces = tuple(
        sorted(
            scope.scope_id
            for scope in remote.scopes
            if scope.kind is ScopeKind.WORKSPACE
            and scope.scope_id not in local_workspace_bindings
        )
    )
    local_workspaces = tuple(
        sorted(scope.scope_id for scope in local_scopes if scope.kind is ScopeKind.WORKSPACE)
    )
    attention: list[str] = []
    if local_manifest.purge_generation != remote.purge_generation:
        attention.append("purge_generation_mismatch")

    required = tuple(
        [item.decision_id for item in (*version_conflicts, *collisions)]
        + [f"workspace:{scope_id}" for scope_id in local_workspaces]
    )
    return ReconciliationPlan(
        plan_id=plan_id or f"pc-link-plan-{uuid.uuid4()}",
        dataset_id=remote.dataset_id,
        authority_id=remote.authority_id,
        local_profile_id=local_manifest.profile_id,
        server_profile_id=remote.manifest.profile_id,
        bootstrap_cursor=remote.cursor,
        integrity_key_id=remote.integrity_key_id,
        key_record_id=remote.key_record_id,
        purge_generation=remote.purge_generation,
        schema_version=remote.schema_version,
        global_scope_mapping=(local_global.scope_id, remote_global.scope_id),
        exact_record_ids=tuple(sorted(exact)),
        local_only_record_ids=tuple(sorted(local_only)),
        remote_only_record_ids=tuple(sorted(remote_only)),
        device_only_record_ids=tuple(sorted(device_only)),
        version_conflicts=tuple(version_conflicts),
        key_collisions=tuple(collisions),
        unlinked_remote_scope_ids=remote_workspaces,
        local_workspace_scope_ids=local_workspaces,
        required_decision_ids=required,
        attention_codes=tuple(attention),
        local_snapshot_token=_snapshot_token(
            local_manifest, local_scopes, local_records, local_proposals
        ),
    )


__all__ = [
    "CanonicalBootstrapSnapshot",
    "KeyCollision",
    "ReconciliationPlan",
    "VersionConflict",
    "build_reconciliation_plan",
]
