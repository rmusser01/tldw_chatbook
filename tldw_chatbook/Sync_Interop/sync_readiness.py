"""Sync readiness helpers for dry-run parity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from tldw_profile_core import SERIALIZED_SCHEMA_VERSION

from tldw_chatbook.runtime_policy.server_parity_models import SyncReadinessReport

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api.sync_schemas import SyncV2CapabilitiesResponse

PERSONAL_CONTEXT_SYNC_DOMAINS: tuple[str, ...] = (
    "personal_context.manifest",
    "personal_context.scope",
    "personal_context.record",
    "personal_context.proposal",
    "personal_context.purge",
)
_PERSONAL_CONTEXT_SERVER_BLOCKERS = frozenset(
    {
        "personal_context_profile_key_unavailable",
        "personal_context_schema_unsupported",
        "personal_context_server_trusted_unavailable",
        "personal_context_transport_unavailable",
    }
)
PERSONAL_CONTEXT_MINIMUM_QUOTAS = {
    "max_record_bytes": 16_384,
    "max_search_results": 20,
    "max_proposals_per_turn": 5,
    "max_proposals_per_session": 25,
    "max_unresolved_proposals": 200,
}


@dataclass(frozen=True, slots=True)
class PersonalContextSyncReadiness:
    """Fail-closed result of Personal Context capability negotiation."""

    read_enabled: bool
    write_enabled: bool
    blockers: tuple[str, ...]
    negotiated_schema_version: int | None


def personal_context_sync_readiness(
    capabilities: SyncV2CapabilitiesResponse,
    *,
    require_writable: bool = True,
) -> PersonalContextSyncReadiness:
    """Negotiate the pinned Personal Context contract without affecting other domains."""

    if capabilities.personal_context_validation_error:
        return PersonalContextSyncReadiness(
            read_enabled=False,
            write_enabled=False,
            blockers=("personal_context_capability_malformed",),
            negotiated_schema_version=None,
        )
    contract = capabilities.personal_context
    if contract is None:
        return PersonalContextSyncReadiness(
            read_enabled=False,
            write_enabled=False,
            blockers=("personal_context_capability_missing",),
            negotiated_schema_version=None,
        )
    if not contract.available:
        server_blockers = tuple(
            dict.fromkeys(
                blocker
                if blocker in _PERSONAL_CONTEXT_SERVER_BLOCKERS
                else "personal_context_server_unavailable"
                for blocker in contract.blockers
            )
        )
        return PersonalContextSyncReadiness(
            read_enabled=False,
            write_enabled=False,
            blockers=server_blockers or ("personal_context_server_unavailable",),
            negotiated_schema_version=None,
        )

    blockers: list[str] = []
    advertised_domains = set(capabilities.domains)
    blockers.extend(
        f"personal_context_domain_missing:{domain}"
        for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        if domain not in advertised_domains
    )
    for domain in PERSONAL_CONTEXT_SYNC_DOMAINS:
        operations = set(capabilities.operations.get(domain, ()))
        if not {"upsert", "tombstone"}.issubset(operations):
            blockers.append(f"personal_context_operations_incompatible:{domain}")
        if 1 not in capabilities.supported_adapter_versions.get(domain, ()):
            blockers.append(f"personal_context_adapter_unsupported:{domain}")

    local_schema_version = SERIALIZED_SCHEMA_VERSION
    schema_compatible = (
        contract.min_schema_version
        <= local_schema_version
        <= contract.max_schema_version
    )
    negotiated_schema_version = local_schema_version if schema_compatible else None
    if not schema_compatible:
        blockers.append("personal_context_schema_incompatible")
    if (
        contract.authorization_policy != "server_trusted_v1"
        or "server_trusted_v1" not in capabilities.encryption_policies
    ):
        blockers.append("personal_context_authorization_policy_incompatible")
    if contract.integrity_algorithm != "hmac-sha256-v1":
        blockers.append("personal_context_integrity_incompatible")
    if contract.integrity_key_distribution != "wrapped-bootstrap-v1":
        blockers.append("personal_context_key_distribution_incompatible")
    if contract.privacy_cleanup_ack != "personal-context-cleanup-v1":
        blockers.append("personal_context_cleanup_ack_incompatible")
    if contract.purge_generation != "personal-context-purge-v1":
        blockers.append("personal_context_purge_generation_incompatible")
    for field_name, minimum in PERSONAL_CONTEXT_MINIMUM_QUOTAS.items():
        if getattr(contract, field_name) < minimum:
            blockers.append(f"personal_context_quota_incompatible:{field_name}")

    read_blockers = tuple(dict.fromkeys(blockers))
    read_enabled = not read_blockers
    if require_writable:
        blockers.extend(
            f"personal_context_adapter_not_writable:{domain}"
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
            if 1 not in capabilities.writable_adapter_versions.get(domain, ())
        )
    stable_blockers = tuple(dict.fromkeys(blockers))
    return PersonalContextSyncReadiness(
        read_enabled=read_enabled,
        write_enabled=not stable_blockers,
        blockers=stable_blockers,
        negotiated_schema_version=negotiated_schema_version,
    )


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
        ),
        SyncDomainEligibility(
            domain="library_collections",
            sync_eligible=True,
            write_enabled=False,
            reason_codes=("dry_run_only",),
            details={"mode": "read_only_mirror_report"},
        ),
        SyncDomainEligibility(
            domain="workspaces",
            sync_eligible=True,
            write_enabled=False,
            reason_codes=("dry_run_only",),
            details={"mode": "read_only_mirror_report"},
        ),
    ]
)
