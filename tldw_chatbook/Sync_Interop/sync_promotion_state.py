"""Pure display state for safe write-sync promotion.

This module intentionally does not enqueue, approve, replay, or dispatch sync
mutations. It only turns existing dry-run/readiness signals into user-facing
copy for surfaces that need to explain why writes remain blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tldw_chatbook.runtime_policy.server_parity_models import SourceAuthority, SyncReadinessReport


@dataclass(frozen=True, slots=True)
class SyncPromotionState:
    """User-facing state for one sync promotion domain."""

    domain: str
    surface_label: str
    status: str
    authority_label: str
    sync_label: str
    review_label: str
    conflict_label: str
    rollback_label: str
    mirror_label: str
    primary_recovery: str
    mutation_allowed: bool = False
    workspace_id: str | None = None
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SyncPromotionSurfaceSummary:
    """Compact rollup for multiple sync promotion domains."""

    heading: str
    states: tuple[SyncPromotionState, ...]
    mutation_allowed: bool
    primary_recovery: str


def build_sync_promotion_state(
    *,
    domain: str,
    surface_label: str,
    readiness: SyncReadinessReport | None = None,
    latest_mirror_report: Mapping[str, Any] | None = None,
    conflict_reports: Sequence[Mapping[str, Any]] | None = None,
    profile_state: Mapping[str, Any] | None = None,
    source_authority: SourceAuthority = "local",
    workspace_id: str | None = None,
) -> SyncPromotionState:
    """Build a conservative write-sync promotion state.

    Args:
        domain: Sync domain identifier.
        surface_label: Human-readable surface or object label.
        readiness: Existing dry-run readiness report, when available.
        latest_mirror_report: Existing mirror report row or report payload.
        conflict_reports: Existing conflict rows scoped to this domain.
        profile_state: Existing sync profile state row.
        source_authority: Source authority used for user-facing copy.
        workspace_id: Optional workspace scope.

    Returns:
        A display state. `mutation_allowed` is always False in this tranche.
    """

    reason_codes = tuple(getattr(readiness, "reason_codes", ()) or ())
    conflicts_count = len(tuple(conflict_reports or ()))
    profile = dict(profile_state or {})
    report = _mirror_report_payload(latest_mirror_report)
    readiness_sync_eligible = bool(getattr(readiness, "sync_eligible", False))
    readiness_write_enabled = bool(getattr(readiness, "write_enabled", False))
    rollback_required = _rollback_required(profile)

    authority_label = _authority_label(source_authority)
    mirror_label = _mirror_label(report)

    if rollback_required:
        status = "rollback-required"
        sync_label = "Sync: rollback required"
        review_label = "Review: required before writes"
        conflict_label = _conflict_label(conflicts_count)
        rollback_label = "Rollback: required before writes"
        primary_recovery = "Resolve the failed sync state and verify rollback before writes."
    elif conflicts_count:
        status = "conflict"
        sync_label = "Sync: conflict review required"
        review_label = "Review: required before writes"
        conflict_label = _conflict_label(conflicts_count)
        rollback_label = "Rollback: not required"
        primary_recovery = "Resolve sync conflicts before any write replay is available."
    elif readiness_write_enabled:
        status = "review-gated"
        sync_label = "Sync: review gated"
        review_label = "Review: required before writes"
        conflict_label = "Conflicts: none reported"
        rollback_label = "Rollback: not required"
        primary_recovery = "Writes stay blocked until review, conflict, and rollback gates are ready."
    elif readiness_sync_eligible:
        status = "dry-run"
        sync_label = "Sync: dry-run only"
        review_label = "Review: required before writes"
        conflict_label = "Conflicts: none reported"
        rollback_label = "Rollback: not required"
        primary_recovery = "Review dry-run results before enabling writes."
    else:
        status = "unavailable"
        sync_label = "Sync: unavailable"
        review_label = "Review: unavailable until sync dry-run is configured"
        conflict_label = "Conflicts: unavailable"
        rollback_label = "Rollback: unavailable"
        primary_recovery = "Configure sync dry-run readiness before write promotion."

    return SyncPromotionState(
        domain=domain,
        surface_label=surface_label,
        status=status,
        authority_label=authority_label,
        sync_label=sync_label,
        review_label=review_label,
        conflict_label=conflict_label,
        rollback_label=rollback_label,
        mirror_label=mirror_label,
        primary_recovery=primary_recovery,
        mutation_allowed=False,
        workspace_id=workspace_id,
        reason_codes=reason_codes,
    )


def build_sync_promotion_summary(
    *,
    heading: str,
    states: Sequence[SyncPromotionState],
) -> SyncPromotionSurfaceSummary:
    """Build a compact rollup for a surface with several sync domains."""

    state_tuple = tuple(states)
    blocking = next(
        (
            state
            for state in state_tuple
            if state.status in {"rollback-required", "conflict", "review-gated"}
        ),
        None,
    )
    primary_recovery = (
        blocking.primary_recovery
        if blocking is not None
        else "Writes stay blocked until review, conflict, and rollback gates are ready."
    )
    return SyncPromotionSurfaceSummary(
        heading=heading,
        states=state_tuple,
        mutation_allowed=False,
        primary_recovery=primary_recovery,
    )


def _authority_label(source_authority: SourceAuthority) -> str:
    if source_authority == "server":
        return "Authority: server mirror"
    return "Authority: local-first"


def _mirror_report_payload(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(report, Mapping):
        return {}
    nested = report.get("report")
    if isinstance(nested, Mapping):
        return nested
    return report


def _mirror_label(report: Mapping[str, Any]) -> str:
    if not report:
        return "Mirror: no dry-run report yet"
    try:
        mapped_count = int(report.get("mapped_count", 0))
    except (TypeError, ValueError):
        mapped_count = 0
    noun = "record" if mapped_count == 1 else "records"
    return f"Mirror: {mapped_count} mapped {noun}"


def _conflict_label(count: int) -> str:
    if count <= 0:
        return "Conflicts: none reported"
    noun = "conflict" if count == 1 else "conflicts"
    return f"Conflicts: {count} require review" if count != 1 else f"Conflicts: {count} requires review"


def _rollback_required(profile_state: Mapping[str, Any]) -> bool:
    metadata = profile_state.get("dry_run_metadata")
    if isinstance(metadata, Mapping) and metadata.get("rollback_required"):
        return True
    if profile_state.get("rollback_required"):
        return True
    return bool(profile_state.get("last_error"))
