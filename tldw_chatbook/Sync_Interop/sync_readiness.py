"""Sync readiness helpers for dry-run parity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from tldw_chatbook.runtime_policy.server_parity_models import SyncReadinessReport


@dataclass(frozen=True, slots=True)
class SyncDomainEligibility:
    """Registered per-domain sync eligibility."""

    domain: str
    sync_eligible: bool = False
    write_enabled: bool = False
    reason_codes: tuple[str, ...] = ("dry_run_only",)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("domain is required")
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "details", dict(self.details))


class SyncEligibilityRegistry:
    """Per-domain registry; unknown domains are not sync eligible."""

    def __init__(self, entries: list[SyncDomainEligibility] | None = None) -> None:
        self._entries: dict[str, SyncDomainEligibility] = {}
        for entry in entries or []:
            self.register(entry)

    def register(self, entry: SyncDomainEligibility) -> None:
        self._entries[entry.domain] = entry

    def get(self, domain: str) -> SyncDomainEligibility:
        return self._entries.get(
            domain,
            SyncDomainEligibility(
                domain=domain,
                sync_eligible=False,
                write_enabled=False,
                reason_codes=("not_registered",),
            ),
        )


def build_sync_readiness_report(
    *,
    domain: str,
    server_profile_id: str | None,
    workspace_id: str | None,
    registry: SyncEligibilityRegistry | None = None,
) -> SyncReadinessReport:
    """Build a workspace-scoped readiness report without enabling writes."""

    eligibility_registry = registry or SyncEligibilityRegistry()
    eligibility = eligibility_registry.get(domain)
    return SyncReadinessReport(
        domain=domain,
        sync_eligible=eligibility.sync_eligible,
        write_enabled=False,
        reason_codes=eligibility.reason_codes,
        server_profile_id=server_profile_id,
        workspace_id=workspace_id,
        details=eligibility.details,
    )


DEFAULT_SYNC_ELIGIBILITY_REGISTRY = SyncEligibilityRegistry(
    [
        SyncDomainEligibility(
            domain="notes",
            sync_eligible=True,
            write_enabled=False,
            reason_codes=("dry_run_only",),
            details={"mode": "read_only_mirror_report"},
        ),
        SyncDomainEligibility(
            domain="workspace_notes",
            sync_eligible=True,
            write_enabled=False,
            reason_codes=("dry_run_only",),
            details={
                "mode": "read_only_mirror_report",
                "workspace_required": True,
            },
        ),
        SyncDomainEligibility(
            domain="media",
            sync_eligible=True,
            write_enabled=False,
            reason_codes=("dry_run_only",),
            details={"mode": "read_only_mirror_report"},
        ),
        SyncDomainEligibility(
            domain="research",
            sync_eligible=True,
            write_enabled=False,
            reason_codes=("dry_run_only",),
            details={"mode": "read_only_mirror_report"},
        ),
        SyncDomainEligibility(
            domain="chat_metadata",
            sync_eligible=True,
            write_enabled=False,
            reason_codes=("dry_run_only", "server_owned"),
            details={
                "mode": "read_only_mirror_report",
                "write_gate": "chat_server_identity_not_ready",
            },
        )
    ]
)
