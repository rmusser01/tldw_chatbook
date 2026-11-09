"""Private session-grant body; the control-plane service retains all authority."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def apply_session_approval(
    service: Any,
    server_key: str,
    tool_name: str,
    *,
    profile_id: str,
    expected_profile_digest: str | None,
    expected_revision: int | None,
    publish: Any,
    lifecycle: Any,
    policy_digest: Any,
    mutation_error: Any,
) -> None:
    """Grant a session-scoped approval for one profile/server/tool triple.

    Session approvals are held in memory only, for the lifetime of
    this service instance (an app-run singleton) -- they are never
    persisted to disk and do not survive an app restart or a fresh
    instance of this service.

    Args:
        server_key: Prefixed server key the tool belongs to.
        tool_name: Name of the tool being approved.
        profile_id: Exact permission profile receiving the approval.
        expected_profile_digest: Captured policy digest to revalidate
            under the permission-store fence.
        expected_revision: Captured imported-profile revision, when
            applicable.
    """
    if expected_profile_digest is None and expected_revision is None:
        # Compatibility seam for non-UI callers. Captured UI approvals
        # always supply the digest (and imported revision) below.
        with service._session_approvals_lock:
            service._session_approvals.add((profile_id, server_key, tool_name))
        publish("applied", actual_scope="approve_session")
        return
    store = service.permission_store
    if store is None:
        if expected_profile_digest is not None or expected_revision is not None:
            raise mutation_error("stale_profile")
        with service._session_approvals_lock:
            service._session_approvals.add((profile_id, server_key, tool_name))
        publish("applied", actual_scope="approve_session")
        return
    with store.mutation_fence():
        snapshot = store.read_snapshot_strict()
        profile = snapshot.payload["profiles"].get(profile_id)
        if not isinstance(profile, Mapping):
            raise mutation_error("stale_profile")
        disposition = lifecycle(profile)
        if disposition in {"invalid", "tombstone"}:
            raise mutation_error("lifecycle_invalid")
        digest = policy_digest(profile)
        if expected_profile_digest is not None and digest != expected_profile_digest:
            raise mutation_error("stale_profile")
        revision = (
            profile["tool_pack_lifecycle"]["revision"]
            if disposition == "imported"
            else None
        )
        if expected_revision is not None and revision != expected_revision:
            raise mutation_error("stale_revision")
        with service._session_approvals_lock:
            service._session_approvals.add((profile_id, server_key, tool_name))
    publish("applied", actual_scope="approve_session")
