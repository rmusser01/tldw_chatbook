"""Compiled native consumers and closed V2 compatibility projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Literal

from tldw_profile_core.canonical import canonical_json_bytes

from .key_protector import ProfileLockedError

if TYPE_CHECKING:
    from .native_codec import DecodedNativeProfileObject


class ProfileCompatibilityError(ProfileLockedError):
    """Report compatibility unavailability without profile metadata."""

    reason_code = "personal_context_compatibility_unavailable"

    def __init__(self) -> None:
        super().__init__(self.reason_code)


@dataclass(frozen=True, slots=True)
class NativeProfileConsumer:
    """A compiled native route declaration; not a qualification receipt."""

    consumer_id: str
    owner_id: str
    operations: tuple[str, ...]


NATIVE_PROFILE_CONSUMER_REVISION = "closed-native-v2-1"
NATIVE_PROFILE_CONSUMERS = (
    NativeProfileConsumer(
        "bootstrap.bootstrap_personal_context_service",
        "bootstrap",
        ("bootstrap_personal_context_service",),
    ),
    NativeProfileConsumer(
        "repository._commit_local_body", "repository", ("_commit_local_body",)
    ),
    NativeProfileConsumer(
        "repository._expire_due_proposals_in_connection",
        "repository",
        ("_expire_due_proposals_in_connection",),
    ),
    NativeProfileConsumer(
        "repository._get_local_body", "repository", ("_get_local_body",)
    ),
    NativeProfileConsumer(
        "repository._get_local_version", "repository", ("_get_local_version",)
    ),
    NativeProfileConsumer("repository._head_row", "repository", ("_head_row",)),
    NativeProfileConsumer(
        "repository._is_quarantined", "repository", ("_is_quarantined",)
    ),
    NativeProfileConsumer(
        "repository._iter_head_rows", "repository", ("_iter_head_rows",)
    ),
    NativeProfileConsumer(
        "repository._rebuild_newly_linked_scope_outbox",
        "repository",
        ("_rebuild_newly_linked_scope_outbox",),
    ),
    NativeProfileConsumer(
        "repository._require_no_record_collision_in_connection",
        "repository",
        ("_require_no_record_collision_in_connection",),
    ),
    NativeProfileConsumer(
        "repository._require_unique_workspace_binding",
        "repository",
        ("_require_unique_workspace_binding",),
    ),
    NativeProfileConsumer(
        "repository._resolve_proposal_in_connection",
        "repository",
        ("_resolve_proposal_in_connection",),
    ),
    NativeProfileConsumer(
        "repository.accept_proposal_and_record",
        "repository",
        ("accept_proposal_and_record",),
    ),
    NativeProfileConsumer(
        "repository.acknowledge_outbox", "repository", ("acknowledge_outbox",)
    ),
    NativeProfileConsumer(
        "repository.acquire_first_link_freeze",
        "repository",
        ("acquire_first_link_freeze",),
    ),
    NativeProfileConsumer(
        "repository.apply_reviewed_link", "repository", ("apply_reviewed_link",)
    ),
    NativeProfileConsumer(
        "repository.bind_legacy_first_link_rebaseline_commit",
        "repository",
        ("bind_legacy_first_link_rebaseline_commit",),
    ),
    NativeProfileConsumer(
        "repository.clear_first_link_rebaseline_commit",
        "repository",
        ("clear_first_link_rebaseline_commit",),
    ),
    NativeProfileConsumer(
        "repository.commit_device_only_split",
        "repository",
        ("commit_device_only_split",),
    ),
    NativeProfileConsumer(
        "repository.commit_interview_batch", "repository", ("commit_interview_batch",)
    ),
    NativeProfileConsumer(
        "repository.commit_manifest_version", "repository", ("commit_manifest_version",)
    ),
    NativeProfileConsumer(
        "repository.commit_outbox_body", "repository", ("commit_outbox_body",)
    ),
    NativeProfileConsumer(
        "repository.commit_proposal", "repository", ("commit_proposal",)
    ),
    NativeProfileConsumer(
        "repository.commit_record_and_manifest",
        "repository",
        ("commit_record_and_manifest",),
    ),
    NativeProfileConsumer(
        "repository.commit_record_version", "repository", ("commit_record_version",)
    ),
    NativeProfileConsumer(
        "repository.commit_runtime_policy", "repository", ("commit_runtime_policy",)
    ),
    NativeProfileConsumer("repository.commit_scope", "repository", ("commit_scope",)),
    NativeProfileConsumer(
        "repository.commit_scope_binding", "repository", ("commit_scope_binding",)
    ),
    NativeProfileConsumer(
        "repository.commit_scope_with_binding",
        "repository",
        ("commit_scope_with_binding",),
    ),
    NativeProfileConsumer(
        "repository.commit_synced_proposal", "repository", ("commit_synced_proposal",)
    ),
    NativeProfileConsumer(
        "repository.create_profile_with_global_scope",
        "repository",
        ("create_profile_with_global_scope",),
    ),
    NativeProfileConsumer(
        "repository.create_provisional_profile",
        "repository",
        ("create_provisional_profile",),
    ),
    NativeProfileConsumer(
        "repository.destroy_profile_content", "repository", ("destroy_profile_content",)
    ),
    NativeProfileConsumer(
        "repository.expire_due_proposals", "repository", ("expire_due_proposals",)
    ),
    NativeProfileConsumer(
        "repository.first_link_apply_recovery_state",
        "repository",
        ("first_link_apply_recovery_state",),
    ),
    NativeProfileConsumer(
        "repository.first_link_freeze_plan_id",
        "repository",
        ("first_link_freeze_plan_id",),
    ),
    NativeProfileConsumer(
        "repository.first_link_head_rows", "repository", ("first_link_head_rows",)
    ),
    NativeProfileConsumer(
        "repository.first_link_rebaseline_commit_plan_id",
        "repository",
        ("first_link_rebaseline_commit_plan_id",),
    ),
    NativeProfileConsumer(
        "repository.first_link_reconciliation_writes",
        "repository",
        ("first_link_reconciliation_writes",),
    ),
    NativeProfileConsumer(
        "repository.first_link_reviewed_lineage",
        "repository",
        ("first_link_reviewed_lineage",),
    ),
    NativeProfileConsumer(
        "repository.first_link_sync_heads", "repository", ("first_link_sync_heads",)
    ),
    NativeProfileConsumer("repository.get_manifest", "repository", ("get_manifest",)),
    NativeProfileConsumer(
        "repository.get_outbox_body", "repository", ("get_outbox_body",)
    ),
    NativeProfileConsumer(
        "repository.get_outbox_quarantine_reason",
        "repository",
        ("get_outbox_quarantine_reason",),
    ),
    NativeProfileConsumer(
        "repository.get_outbox_receipt", "repository", ("get_outbox_receipt",)
    ),
    NativeProfileConsumer("repository.get_proposal", "repository", ("get_proposal",)),
    NativeProfileConsumer("repository.get_record", "repository", ("get_record",)),
    NativeProfileConsumer(
        "repository.get_record_derivation", "repository", ("get_record_derivation",)
    ),
    NativeProfileConsumer(
        "repository.get_runtime_policy", "repository", ("get_runtime_policy",)
    ),
    NativeProfileConsumer(
        "repository.get_runtime_policy_version",
        "repository",
        ("get_runtime_policy_version",),
    ),
    NativeProfileConsumer("repository.get_scope", "repository", ("get_scope",)),
    NativeProfileConsumer(
        "repository.get_scope_binding", "repository", ("get_scope_binding",)
    ),
    NativeProfileConsumer(
        "repository.get_scope_binding_version",
        "repository",
        ("get_scope_binding_version",),
    ),
    NativeProfileConsumer("repository.get_undo", "repository", ("get_undo",)),
    NativeProfileConsumer("repository.is_destroyed", "repository", ("is_destroyed",)),
    NativeProfileConsumer(
        "repository.is_scope_explicitly_unlinked",
        "repository",
        ("is_scope_explicitly_unlinked",),
    ),
    NativeProfileConsumer(
        "repository.legacy_first_link_rebaseline_commit_matches",
        "repository",
        ("legacy_first_link_rebaseline_commit_matches",),
    ),
    NativeProfileConsumer(
        "repository.list_dispatchable_outbox",
        "repository",
        ("list_dispatchable_outbox",),
    ),
    NativeProfileConsumer(
        "repository.list_pending_outbox", "repository", ("list_pending_outbox",)
    ),
    NativeProfileConsumer(
        "repository.list_proposals", "repository", ("list_proposals",)
    ),
    NativeProfileConsumer(
        "repository.list_quarantine", "repository", ("list_quarantine",)
    ),
    NativeProfileConsumer("repository.list_records", "repository", ("list_records",)),
    NativeProfileConsumer(
        "repository.list_scope_bindings", "repository", ("list_scope_bindings",)
    ),
    NativeProfileConsumer("repository.list_scopes", "repository", ("list_scopes",)),
    NativeProfileConsumer("repository.list_undo_ids", "repository", ("list_undo_ids",)),
    NativeProfileConsumer(
        "repository.list_validated_scope_bindings",
        "repository",
        ("list_validated_scope_bindings",),
    ),
    NativeProfileConsumer(
        "repository.quarantine_object", "repository", ("quarantine_object",)
    ),
    NativeProfileConsumer(
        "repository.quarantine_outbox", "repository", ("quarantine_outbox",)
    ),
    NativeProfileConsumer(
        "repository.read_compatibility", "repository", ("read_compatibility",)
    ),
    NativeProfileConsumer(
        "repository.read_export_snapshot", "repository", ("read_export_snapshot",)
    ),
    NativeProfileConsumer(
        "repository.reinitialize_destroyed_profile",
        "repository",
        ("reinitialize_destroyed_profile",),
    ),
    NativeProfileConsumer(
        "repository.release_first_link_freeze",
        "repository",
        ("release_first_link_freeze",),
    ),
    NativeProfileConsumer(
        "repository.resolve_proposal", "repository", ("resolve_proposal",)
    ),
    NativeProfileConsumer("service.status", "service", ("status",)),
)


def _validate_consumers(consumers: tuple[NativeProfileConsumer, ...]) -> None:
    if type(consumers) is not tuple or not consumers:
        raise ProfileCompatibilityError()
    ids = []
    for row in consumers:
        if type(row) is not NativeProfileConsumer or type(row.consumer_id) is not str:
            raise ProfileCompatibilityError()
        if row.owner_id not in ("repository", "service", "bootstrap"):
            raise ProfileCompatibilityError()
        if not row.consumer_id.startswith(row.owner_id + "."):
            raise ProfileCompatibilityError()
        if type(row.operations) is not tuple or not row.operations:
            raise ProfileCompatibilityError()
        if any(type(op) is not str or not op for op in row.operations):
            raise ProfileCompatibilityError()
        if row.operations != tuple(sorted(set(row.operations))):
            raise ProfileCompatibilityError()
        ids.append(row.consumer_id)
    if ids != sorted(set(ids)):
        raise ProfileCompatibilityError()


def validate_native_consumers() -> None:
    """Check the compiled inventory, without registering or granting anything."""
    _validate_consumers(NATIVE_PROFILE_CONSUMERS)


validate_native_consumers()
NATIVE_PROFILE_CONSUMER_DIGEST = sha256(
    canonical_json_bytes(
        [
            {
                "consumer_id": row.consumer_id,
                "owner_id": row.owner_id,
                "operations": list(row.operations),
            }
            for row in NATIVE_PROFILE_CONSUMERS
        ]
    )
).hexdigest()


def require_native_consumer(consumer_id: str) -> NativeProfileConsumer:
    """Require a fixed native route ID; unknown or caller-shaped routes deny."""
    if type(consumer_id) is not str:
        raise ProfileCompatibilityError()
    for row in NATIVE_PROFILE_CONSUMERS:
        if row.consumer_id == consumer_id:
            return row
    raise ProfileCompatibilityError()


@dataclass(frozen=True, slots=True)
class ProfileCompatibilityView:
    """Current validated profile state; no enabled V2 state or permit."""

    state: Literal["legacy_v1", "v2_blocked"]
    schema_version: int
    profile_id: str = field(repr=False)
    manifest_version_id: str = field(repr=False)
    purge_generation: int
    evidence_retirement_epoch: int | None
    registry_revision: str
    registry_digest: str


def profile_compatibility(
    manifest: DecodedNativeProfileObject, *, consumer_id: str
) -> ProfileCompatibilityView:
    """Project exact validated manifest data against a compiled native route.

    Args:
        manifest: Decoded native manifest data; never a qualification token.
        consumer_id: Fixed compiled caller route.

    Returns:
        Immutable legacy/blocked state. All V2 profiles are blocked.

    Raises:
        ProfileCompatibilityError: Unregistered route or invalid decoded state.
    """
    from .native_codec import (
        DecodedNativeProfileObject,
        NativeProfileDecodeError,
        decode_native_profile,
    )

    require_native_consumer(consumer_id)
    try:
        if (
            type(manifest) is not DecodedNativeProfileObject
            or manifest.kind != "manifest"
        ):
            raise ProfileCompatibilityError()
        fresh = decode_native_profile("manifest", manifest.canonical)
        if (
            fresh.schema_version != manifest.schema_version
            or type(fresh.value) is not type(manifest.value)
            or fresh.value != manifest.value
            or fresh.canonical != manifest.canonical
        ):
            raise ProfileCompatibilityError()
        value = fresh.value
        return ProfileCompatibilityView(
            "legacy_v1" if fresh.schema_version == 1 else "v2_blocked",
            fresh.schema_version,
            value.profile_id,
            value.current_version_id,
            value.purge_generation,
            None if fresh.schema_version == 1 else value.evidence_retirement_epoch,
            NATIVE_PROFILE_CONSUMER_REVISION,
            NATIVE_PROFILE_CONSUMER_DIGEST,
        )
    except (NativeProfileDecodeError, TypeError, ValueError):
        raise ProfileCompatibilityError() from None
