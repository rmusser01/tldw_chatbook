from __future__ import annotations

from tldw_chatbook.Sync_Interop.sync_promotion_state import build_sync_promotion_state
from tldw_chatbook.Sync_Interop.sync_readiness import (
    SyncDomainEligibility,
    SyncEligibilityRegistry,
    build_sync_readiness_report,
)
from tldw_chatbook.runtime_policy.server_parity_models import SyncReadinessReport


def test_unknown_domain_renders_unavailable_and_blocks_mutation() -> None:
    readiness = build_sync_readiness_report(
        domain="library_collections",
        server_profile_id="server-a",
        workspace_id="workspace-a",
        registry=SyncEligibilityRegistry(),
    )

    state = build_sync_promotion_state(
        domain="library_collections",
        surface_label="Collections",
        readiness=readiness,
        source_authority="local",
        workspace_id="workspace-a",
    )

    assert state.surface_label == "Collections"
    assert state.authority_label == "Authority: local-first"
    assert state.sync_label == "Sync: unavailable"
    assert state.review_label == "Review: unavailable until sync dry-run is configured"
    assert state.mutation_allowed is False
    assert state.status == "unavailable"
    assert "dry-run" in state.primary_recovery


def test_dry_run_eligible_domain_requires_review_before_writes() -> None:
    registry = SyncEligibilityRegistry(
        [
            SyncDomainEligibility(
                domain="workspaces",
                sync_eligible=True,
                write_enabled=False,
                reason_codes=("dry_run_only",),
                details={"mode": "read_only_mirror_report"},
            )
        ]
    )
    readiness = build_sync_readiness_report(
        domain="workspaces",
        server_profile_id="server-a",
        workspace_id="workspace-a",
        registry=registry,
    )

    state = build_sync_promotion_state(
        domain="workspaces",
        surface_label="Workspaces",
        readiness=readiness,
        latest_mirror_report={"dry_run": True, "write_enabled": False, "mapped_count": 3},
        source_authority="local",
        workspace_id="workspace-a",
    )

    assert state.sync_label == "Sync: dry-run only"
    assert state.review_label == "Review: required before writes"
    assert state.mirror_label == "Mirror: 3 mapped records"
    assert state.mutation_allowed is False
    assert state.status == "dry-run"


def test_conflict_reports_override_dry_run_ready_state() -> None:
    registry = SyncEligibilityRegistry(
        [SyncDomainEligibility(domain="library_collections", sync_eligible=True)]
    )
    readiness = build_sync_readiness_report(
        domain="library_collections",
        server_profile_id="server-a",
        workspace_id="workspace-a",
        registry=registry,
    )

    state = build_sync_promotion_state(
        domain="library_collections",
        surface_label="Collections",
        readiness=readiness,
        conflict_reports=(
            {"conflict_type": "duplicate_local_side"},
            {"conflict_type": "duplicate_remote_side"},
        ),
        source_authority="local",
        workspace_id="workspace-a",
    )

    assert state.sync_label == "Sync: conflict review required"
    assert state.conflict_label == "Conflicts: 2 require review"
    assert state.status == "conflict"
    assert state.mutation_allowed is False


def test_rollback_required_profile_blocks_writes_after_error() -> None:
    registry = SyncEligibilityRegistry([SyncDomainEligibility(domain="notes", sync_eligible=True)])
    readiness = build_sync_readiness_report(
        domain="notes",
        server_profile_id="server-a",
        workspace_id="workspace-a",
        registry=registry,
    )

    state = build_sync_promotion_state(
        domain="notes",
        surface_label="Library",
        readiness=readiness,
        profile_state={
            "last_error": "local apply rejected",
            "dry_run_metadata": {"rollback_required": True},
        },
        source_authority="local",
        workspace_id="workspace-a",
    )

    assert state.sync_label == "Sync: rollback required"
    assert state.rollback_label == "Rollback: required before writes"
    assert state.status == "rollback-required"
    assert state.mutation_allowed is False


def test_write_enabled_readiness_is_clamped_until_review_gates_exist() -> None:
    readiness = SyncReadinessReport(
        domain="notes",
        sync_eligible=True,
        write_enabled=True,
        reason_codes=("server_write_supported",),
        server_profile_id="server-a",
        workspace_id="workspace-a",
        details={},
    )

    state = build_sync_promotion_state(
        domain="notes",
        surface_label="Library",
        readiness=readiness,
        source_authority="server",
        workspace_id="workspace-a",
    )

    assert state.authority_label == "Authority: server mirror"
    assert state.sync_label == "Sync: review gated"
    assert state.review_label == "Review: required before writes"
    assert state.mutation_allowed is False
    assert state.primary_recovery == "Writes stay blocked until review, conflict, and rollback gates are ready."
