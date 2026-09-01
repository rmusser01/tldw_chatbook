from __future__ import annotations

from tldw_chatbook.Sync_Interop.sync_readiness import (
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SyncDomainEligibility,
    SyncEligibilityRegistry,
    build_sync_readiness_report,
    personal_context_sync_readiness,
)
from tldw_chatbook.tldw_api.sync_schemas import SyncV2CapabilitiesResponse
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


def test_write_enabled_eligibility_is_clamped_off_for_dry_run_readiness() -> None:
    registry = SyncEligibilityRegistry()
    registry.register(
        SyncDomainEligibility(
            domain="notes",
            sync_eligible=True,
            write_enabled=True,
            reason_codes=("server_write_supported",),
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


def test_personal_context_failure_does_not_change_existing_domain_readiness() -> None:
    capabilities = SyncV2CapabilitiesResponse.model_validate(
        {
            "domains": [*PERSONAL_CONTEXT_SYNC_DOMAINS, "notes.note"],
            "personal_context": None,
        }
    )
    registry = SyncEligibilityRegistry(
        [SyncDomainEligibility(domain="notes", sync_eligible=True)]
    )

    personal_context = personal_context_sync_readiness(capabilities)
    notes = build_sync_readiness_report(
        domain="notes",
        server_profile_id="server-a",
        workspace_id=None,
        registry=registry,
    )

    assert personal_context.write_enabled is False
    assert notes.sync_eligible is True
