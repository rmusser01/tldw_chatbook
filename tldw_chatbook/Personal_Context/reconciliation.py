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
from tldw_profile_core.canonical import canonical_json_bytes


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
class WorkspaceMappingConflict:
    """One exact semantic collision that makes a workspace mapping unavailable."""

    local_scope_id: str
    remote_scope_id: str
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
    profile_adoption: tuple[str, str]
    global_scope_mapping: tuple[str, str]
    schema_outcome: str
    quota_outcome: str
    purge_outcome: str
    required_quotas: Mapping[str, int] = field(repr=False)
    server_quotas: Mapping[str, int] = field(repr=False)
    quota_outcomes: tuple[tuple[str, int, int, bool], ...]
    record_outcomes: tuple[tuple[str, str, str | None, str | None], ...]
    proposal_outcomes: tuple[tuple[str, str], ...]
    exact_record_ids: tuple[str, ...]
    local_only_record_ids: tuple[str, ...]
    remote_only_record_ids: tuple[str, ...]
    device_only_record_ids: tuple[str, ...]
    exact_proposal_ids: tuple[str, ...]
    local_only_proposal_ids: tuple[str, ...]
    remote_only_proposal_ids: tuple[str, ...]
    proposal_conflict_ids: tuple[str, ...]
    version_conflicts: tuple[VersionConflict, ...]
    key_collisions: tuple[KeyCollision, ...]
    unlinked_remote_scope_ids: tuple[str, ...]
    local_workspace_scope_ids: tuple[str, ...]
    workspace_new_scope_ids: tuple[tuple[str, str], ...]
    workspace_mapping_conflicts: tuple[WorkspaceMappingConflict, ...]
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
    bindings: Mapping[str, Mapping[str, object]],
) -> str:
    material = "\x00".join(
        (
            manifest.profile_id,
            manifest.current_version_id,
            *(f"s:{item.scope_id}:{item.version_id}" for item in scopes),
            *(f"r:{item.record_id}:{item.version_id}" for item in records),
            *(
                f"p:{item.proposal_id}:"
                + hashlib.sha256(canonical_bytes(item)).hexdigest()
                for item in proposals
            ),
            *(
                "b:"
                + scope_id
                + ":"
                + hashlib.sha256(canonical_json_bytes(dict(binding))).hexdigest()
                for scope_id, binding in sorted(bindings.items())
            ),
        )
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(material).hexdigest()


def canonical_snapshot_heads(
    manifest: ProfileManifest,
    scopes: Sequence[ProfileScope],
    records: Sequence[ProfileRecord],
    proposals: Sequence[ProfileProposal],
) -> dict[str, dict[str, str]]:
    """Return content-free canonical object/version identities for verification."""

    return {
        "personal_context.manifest": {
            manifest.profile_id: manifest.current_version_id
        },
        "personal_context.scope": {
            item.scope_id: item.version_id for item in scopes
        },
        "personal_context.record": {
            item.record_id: item.version_id
            for item in records
            if item.controls.sync_mode is SyncMode.SYNCABLE
        },
        "personal_context.proposal": {
            item.proposal_id: (
                "sync-proposal-sha256:"
                + hashlib.sha256(canonical_bytes(item)).hexdigest()
            )
            for item in proposals
            if item.proposed_record is None
            or item.proposed_record.controls.sync_mode is SyncMode.SYNCABLE
        },
    }


def build_reconciliation_plan(
    *,
    local_manifest: ProfileManifest,
    local_scopes: Sequence[ProfileScope],
    local_records: Sequence[ProfileRecord],
    local_proposals: Sequence[ProfileProposal],
    remote: CanonicalBootstrapSnapshot,
    local_workspace_bindings: Mapping[str, Mapping[str, object]],
    required_schema_version: int | None = None,
    required_quotas: Mapping[str, int] | None = None,
    plan_id: str | None = None,
) -> ReconciliationPlan:
    """Compare exact canonical identities without persisting or returning content."""

    resolved_plan_id = plan_id or f"pc-link-plan-{uuid.uuid4()}"
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

    local_ids = {record.record_id for record in local_records}
    remote_only.extend(
        record.record_id for record in remote.records if record.record_id not in local_ids
    )
    device_only_identity_collisions = tuple(
        sorted(set(device_only).intersection(remote_by_id))
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
    remote_workspace_scopes = tuple(
        scope for scope in remote.scopes if scope.kind is ScopeKind.WORKSPACE
    )
    workspace_mapping_conflicts: list[WorkspaceMappingConflict] = []
    for local_scope_id in local_workspaces:
        local_by_key = {
            (record.semantic_key.namespace, record.semantic_key.subject): record
            for record in local_records
            if record.scope_id == local_scope_id
            and record.controls.sync_mode is SyncMode.SYNCABLE
            and record.state is RecordState.ACTIVE
            and record.semantic_key is not None
        }
        for remote_scope in remote_workspace_scopes:
            for server in remote.records:
                if (
                    server.scope_id != remote_scope.scope_id
                    or server.state is not RecordState.ACTIVE
                    or server.semantic_key is None
                ):
                    continue
                local = local_by_key.get(
                    (server.semantic_key.namespace, server.semantic_key.subject)
                )
                if local is None or local.record_id == server.record_id:
                    continue
                workspace_mapping_conflicts.append(
                    WorkspaceMappingConflict(
                        local_scope_id=local_scope_id,
                        remote_scope_id=remote_scope.scope_id,
                        record_ids=tuple(sorted((local.record_id, server.record_id))),
                    )
                )
    attention: list[str] = []
    if device_only_identity_collisions:
        attention.append("device_only_identity_collision")
    schema_outcome = f"compatible:{remote.schema_version}"
    if (
        required_schema_version is not None
        and remote.schema_version != required_schema_version
    ):
        schema_outcome = (
            f"incompatible:required-{required_schema_version}:server-"
            f"{remote.schema_version}"
        )
        attention.append("schema_incompatible")
    minimum_quotas = dict(required_quotas or {})
    insufficient_quotas = tuple(
        sorted(
            quota
            for quota, minimum in minimum_quotas.items()
            if int(remote.quotas.get(quota, -1)) < int(minimum)
        )
    )
    quota_outcome = (
        "minimums_satisfied"
        if not insufficient_quotas
        else "insufficient:" + ",".join(insufficient_quotas)
    )
    if insufficient_quotas:
        attention.append("quota_minimums_not_met")
    purge_outcome = f"generation_matches:{remote.purge_generation}"
    if local_manifest.purge_generation != remote.purge_generation:
        attention.append("purge_generation_mismatch")
        purge_outcome = (
            f"generation_mismatch:local-{local_manifest.purge_generation}:"
            f"server-{remote.purge_generation}"
        )
    mapped_local_proposals = {
        item.proposal_id: item.model_copy(
            update={
                "profile_id": remote.manifest.profile_id,
                "scope_id": scope_mapping.get(item.scope_id, item.scope_id),
                "proposed_record": (
                    None
                    if item.proposed_record is None
                    else _mapped_record(
                        item.proposed_record,
                        profile_id=remote.manifest.profile_id,
                        scope_mapping=scope_mapping,
                    )
                ),
            }
        )
        for item in local_proposals
    }
    remote_proposals = {item.proposal_id: item for item in remote.proposals}
    exact_proposals = tuple(
        sorted(
            proposal_id
            for proposal_id, item in mapped_local_proposals.items()
            if proposal_id in remote_proposals
            and canonical_bytes(item) == canonical_bytes(remote_proposals[proposal_id])
        )
    )
    proposal_conflicts = tuple(
        sorted(
            proposal_id
            for proposal_id, item in mapped_local_proposals.items()
            if proposal_id in remote_proposals
            and canonical_bytes(item) != canonical_bytes(remote_proposals[proposal_id])
        )
    )
    if proposal_conflicts:
        attention.append("proposal_same_id_diverged")
    local_only_proposals = tuple(
        sorted(set(mapped_local_proposals) - set(remote_proposals))
    )
    remote_only_proposals = tuple(
        sorted(set(remote_proposals) - set(mapped_local_proposals))
    )
    conflict_by_id = {item.record_id: item for item in version_conflicts}
    record_outcomes = tuple(
        sorted(
            [
                (
                    record_id,
                    "exact",
                    mapped_local[record_id].version_id,
                    remote_by_id[record_id].version_id,
                )
                for record_id in exact
            ]
            + [
                (
                    record_id,
                    "local_addition",
                    mapped_local[record_id].version_id,
                    None,
                )
                for record_id in local_only
            ]
            + [
                (
                    record_id,
                    "server_addition",
                    None,
                    remote_by_id[record_id].version_id,
                )
                for record_id in remote_only
            ]
            + [
                (
                    record_id,
                    "lineage_review",
                    conflict.local_version_id,
                    conflict.server_version_id,
                )
                for record_id, conflict in conflict_by_id.items()
            ]
            + [
                (
                    record_id,
                    "device_only_identity_attention",
                    mapped_local[record_id].version_id,
                    remote_by_id[record_id].version_id,
                )
                for record_id in device_only_identity_collisions
            ]
        )
    )
    proposal_outcomes = tuple(
        sorted(
            [(proposal_id, "exact") for proposal_id in exact_proposals]
            + [
                (proposal_id, "local_addition")
                for proposal_id in local_only_proposals
            ]
            + [
                (proposal_id, "server_addition")
                for proposal_id in remote_only_proposals
            ]
            + [
                (proposal_id, "divergence_attention")
                for proposal_id in proposal_conflicts
            ]
        )
    )

    required = tuple(
        [item.decision_id for item in (*version_conflicts, *collisions)]
        + [f"workspace:{scope_id}" for scope_id in local_workspaces]
    )
    return ReconciliationPlan(
        plan_id=resolved_plan_id,
        dataset_id=remote.dataset_id,
        authority_id=remote.authority_id,
        local_profile_id=local_manifest.profile_id,
        server_profile_id=remote.manifest.profile_id,
        bootstrap_cursor=remote.cursor,
        integrity_key_id=remote.integrity_key_id,
        key_record_id=remote.key_record_id,
        purge_generation=remote.purge_generation,
        schema_version=remote.schema_version,
        profile_adoption=(local_manifest.profile_id, remote.manifest.profile_id),
        global_scope_mapping=(local_global.scope_id, remote_global.scope_id),
        schema_outcome=schema_outcome,
        quota_outcome=quota_outcome,
        purge_outcome=purge_outcome,
        required_quotas=minimum_quotas,
        server_quotas=dict(remote.quotas),
        quota_outcomes=tuple(
            (
                quota,
                int(minimum),
                int(remote.quotas.get(quota, -1)),
                int(remote.quotas.get(quota, -1)) >= int(minimum),
            )
            for quota, minimum in sorted(minimum_quotas.items())
        ),
        record_outcomes=record_outcomes,
        proposal_outcomes=proposal_outcomes,
        exact_record_ids=tuple(sorted(exact)),
        local_only_record_ids=tuple(sorted(local_only)),
        remote_only_record_ids=tuple(sorted(remote_only)),
        device_only_record_ids=tuple(sorted(device_only)),
        exact_proposal_ids=exact_proposals,
        local_only_proposal_ids=local_only_proposals,
        remote_only_proposal_ids=remote_only_proposals,
        proposal_conflict_ids=proposal_conflicts,
        version_conflicts=tuple(version_conflicts),
        key_collisions=tuple(collisions),
        unlinked_remote_scope_ids=remote_workspaces,
        local_workspace_scope_ids=local_workspaces,
        workspace_new_scope_ids=tuple(
            (
                scope_id,
                "scope-workspace-"
                + str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{resolved_plan_id}:personal-context:{scope_id}",
                    )
                ),
            )
            for scope_id in local_workspaces
        ),
        workspace_mapping_conflicts=tuple(
            sorted(
                workspace_mapping_conflicts,
                key=lambda item: (
                    item.local_scope_id,
                    item.remote_scope_id,
                    item.record_ids,
                ),
            )
        ),
        required_decision_ids=required,
        attention_codes=tuple(attention),
        local_snapshot_token=_snapshot_token(
            local_manifest,
            local_scopes,
            local_records,
            local_proposals,
            local_workspace_bindings,
        ),
    )


__all__ = [
    "CanonicalBootstrapSnapshot",
    "KeyCollision",
    "ReconciliationPlan",
    "VersionConflict",
    "WorkspaceMappingConflict",
    "build_reconciliation_plan",
    "canonical_snapshot_heads",
]
