from __future__ import annotations

from tldw_chatbook.Sync_Interop.sync_readiness import (
    SyncDomainEligibility,
    SyncEligibilityRegistry,
    build_sync_readiness_report,
)
from tldw_chatbook.runtime_policy.server_parity_models import SyncReadinessReport


def test_unknown_domain_defaults_to_not_eligible_and_write_disabled() -> None:
    registry = SyncEligibilityRegistry()

    report = build_sync_readiness_report(
        domain="unknown",
        server_profile_id="server-a",
        workspace_id="workspace-1",
        registry=registry,
    )

    assert isinstance(report, SyncReadinessReport)
    assert report.sync_eligible is False
    assert report.write_enabled is False
    assert report.reason_codes == ("not_registered",)
    assert report.server_profile_id == "server-a"
    assert report.workspace_id == "workspace-1"


def test_registered_read_only_domain_reports_eligible_without_write() -> None:
    registry = SyncEligibilityRegistry()
    registry.register(
        SyncDomainEligibility(
            domain="notes",
            sync_eligible=True,
            write_enabled=False,
            reason_codes=("dry_run_only",),
            details={"collections": ["notes"]},
        )
    )

    report = build_sync_readiness_report(
        domain="notes",
        server_profile_id="server-a",
        workspace_id="workspace-1",
        registry=registry,
    )

    assert report.sync_eligible is True
    assert report.write_enabled is False
    assert report.reason_codes == ("dry_run_only",)
    assert report.details == {"collections": ("notes",)}


def test_readiness_preserves_workspace_boundaries_per_report() -> None:
    registry = SyncEligibilityRegistry()
    registry.register(SyncDomainEligibility(domain="notes", sync_eligible=True))

    workspace_a = build_sync_readiness_report(
        domain="notes",
        server_profile_id="server-a",
        workspace_id="workspace-a",
        registry=registry,
    )
    workspace_b = build_sync_readiness_report(
        domain="notes",
        server_profile_id="server-a",
        workspace_id="workspace-b",
        registry=registry,
    )

    assert workspace_a.workspace_id == "workspace-a"
    assert workspace_b.workspace_id == "workspace-b"

