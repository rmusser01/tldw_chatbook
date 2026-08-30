from __future__ import annotations

import pytest

from tldw_chatbook.Sync_Interop.sync_readiness import (
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    personal_context_sync_readiness,
)
from tldw_chatbook.tldw_api.sync_schemas import SyncV2CapabilitiesResponse


def _personal_context_capability(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "available": True,
        "blockers": [],
        "authorization_policy": "server_trusted_v1",
        "min_schema_version": 1,
        "max_schema_version": 1,
        "integrity_algorithm": "hmac-sha256-v1",
        "integrity_key_distribution": "wrapped-bootstrap-v1",
        "privacy_cleanup_ack": "personal-context-cleanup-v1",
        "purge_generation": "personal-context-purge-v1",
        "max_record_bytes": 16_384,
        "max_search_results": 20,
        "max_proposals_per_turn": 5,
        "max_proposals_per_session": 25,
        "max_unresolved_proposals": 200,
    }
    value.update(overrides)
    return value


def _capabilities(**personal_context_overrides: object) -> dict[str, object]:
    domains = list(PERSONAL_CONTEXT_SYNC_DOMAINS)
    return {
        "domains": domains,
        "operations": {
            domain: ["upsert", "tombstone"] for domain in domains
        },
        "supported_adapter_versions": {domain: [1] for domain in domains},
        "writable_adapter_versions": {domain: [1] for domain in domains},
        "encryption_policies": ["server_trusted_v1"],
        "personal_context": _personal_context_capability(**personal_context_overrides),
    }


def test_complete_personal_context_capability_enables_read_and_write() -> None:
    capabilities = SyncV2CapabilitiesResponse.model_validate(_capabilities())

    report = personal_context_sync_readiness(capabilities)

    assert report.read_enabled is True
    assert report.write_enabled is True
    assert report.blockers == ()
    assert report.negotiated_schema_version == 1


def test_personal_context_link_requires_complete_domain_set() -> None:
    payload = _capabilities()
    payload["domains"] = list(PERSONAL_CONTEXT_SYNC_DOMAINS[:-1])

    report = personal_context_sync_readiness(
        SyncV2CapabilitiesResponse.model_validate(payload)
    )

    assert report.write_enabled is False
    assert report.blockers == (
        "personal_context_domain_missing:personal_context.purge",
    )


def test_personal_context_requires_exact_operations_for_every_domain() -> None:
    payload = _capabilities()
    payload["operations"]["personal_context.record"] = ["upsert"]

    report = personal_context_sync_readiness(
        SyncV2CapabilitiesResponse.model_validate(payload)
    )

    assert report.write_enabled is False
    assert report.blockers == (
        "personal_context_operations_incompatible:personal_context.record",
    )


def test_personal_context_requires_supported_adapter_v1_for_reads() -> None:
    payload = _capabilities()
    payload["supported_adapter_versions"]["personal_context.scope"] = []

    report = personal_context_sync_readiness(
        SyncV2CapabilitiesResponse.model_validate(payload)
    )

    assert report.read_enabled is False
    assert report.write_enabled is False
    assert report.blockers == (
        "personal_context_adapter_unsupported:personal_context.scope",
    )


def test_personal_context_requires_writable_adapter_v1_for_writes() -> None:
    payload = _capabilities()
    payload["writable_adapter_versions"]["personal_context.proposal"] = []

    report = personal_context_sync_readiness(
        SyncV2CapabilitiesResponse.model_validate(payload)
    )

    assert report.read_enabled is True
    assert report.write_enabled is False
    assert report.blockers == (
        "personal_context_adapter_not_writable:personal_context.proposal",
    )


def test_personal_context_schema_range_must_overlap_local_core() -> None:
    capabilities = SyncV2CapabilitiesResponse.model_validate(
        _capabilities(min_schema_version=2, max_schema_version=3)
    )

    report = personal_context_sync_readiness(capabilities)

    assert report.blockers == ("personal_context_schema_incompatible",)
    assert report.negotiated_schema_version is None


@pytest.mark.parametrize(
    ("override", "blocker"),
    [
        ({"integrity_algorithm": "sha256"}, "personal_context_integrity_incompatible"),
        (
            {"integrity_key_distribution": "plaintext"},
            "personal_context_key_distribution_incompatible",
        ),
        (
            {"privacy_cleanup_ack": "none"},
            "personal_context_cleanup_ack_incompatible",
        ),
        (
            {"purge_generation": "none"},
            "personal_context_purge_generation_incompatible",
        ),
        (
            {"authorization_policy": "client_private_v1"},
            "personal_context_authorization_policy_incompatible",
        ),
    ],
)
def test_personal_context_contract_mismatches_fail_closed(
    override: dict[str, object], blocker: str
) -> None:
    capabilities = SyncV2CapabilitiesResponse.model_validate(_capabilities(**override))

    assert personal_context_sync_readiness(capabilities).blockers == (blocker,)


def test_server_unavailable_blocker_is_preserved() -> None:
    capabilities = SyncV2CapabilitiesResponse.model_validate(
        _capabilities(
            available=False,
            blockers=["personal_context_profile_key_unavailable"],
        )
    )

    report = personal_context_sync_readiness(capabilities)

    assert report.read_enabled is False
    assert report.write_enabled is False
    assert report.blockers == ("personal_context_profile_key_unavailable",)


def test_missing_personal_context_capability_fails_closed() -> None:
    capabilities = SyncV2CapabilitiesResponse.model_validate(
        {"domains": list(PERSONAL_CONTEXT_SYNC_DOMAINS)}
    )

    assert personal_context_sync_readiness(capabilities).blockers == (
        "personal_context_capability_missing",
    )


def test_malformed_personal_context_capability_is_rejected() -> None:
    capabilities = SyncV2CapabilitiesResponse.model_validate(
        _capabilities(max_record_bytes="unbounded")
    )

    assert personal_context_sync_readiness(capabilities).blockers == (
        "personal_context_capability_malformed",
    )


@pytest.mark.parametrize(
    ("field_name", "malformed_value"),
    [
        ("operations", "upsert"),
        ("supported_adapter_versions", ["v1"]),
        ("writable_adapter_versions", [True]),
    ],
)
def test_malformed_personal_context_transport_entry_is_isolated(
    field_name: str,
    malformed_value: object,
) -> None:
    payload = _capabilities()
    payload[field_name]["personal_context.record"] = malformed_value

    capabilities = SyncV2CapabilitiesResponse.model_validate(payload)

    assert personal_context_sync_readiness(capabilities).blockers == (
        "personal_context_capability_malformed",
    )


def test_malformed_unrelated_future_transport_entries_are_ignored() -> None:
    payload = _capabilities()
    payload["operations"]["future.domain"] = {"future": True}
    payload["supported_adapter_versions"]["future.domain"] = ["v2"]
    payload["writable_adapter_versions"]["future.domain"] = [False]

    capabilities = SyncV2CapabilitiesResponse.model_validate(payload)

    assert "future.domain" not in capabilities.operations
    assert "future.domain" not in capabilities.supported_adapter_versions
    assert "future.domain" not in capabilities.writable_adapter_versions
    assert personal_context_sync_readiness(capabilities).write_enabled is True


def test_partially_implemented_personal_context_capability_fails_closed() -> None:
    payload = _capabilities()
    personal_context = dict(payload["personal_context"])
    personal_context.pop("privacy_cleanup_ack")
    payload["personal_context"] = personal_context

    capabilities = SyncV2CapabilitiesResponse.model_validate(payload)

    assert personal_context_sync_readiness(capabilities).blockers == (
        "personal_context_capability_malformed",
    )


def test_unknown_future_personal_context_fields_are_tolerated() -> None:
    capabilities = SyncV2CapabilitiesResponse.model_validate(
        _capabilities(future_contract="v2")
    )

    assert personal_context_sync_readiness(capabilities).write_enabled is True


def test_wire_payload_cannot_spoof_internal_capability_error() -> None:
    payload = _capabilities()
    payload["personal_context_validation_error"] = "spoofed"

    capabilities = SyncV2CapabilitiesResponse.model_validate(payload)

    assert capabilities.personal_context_validation_error is None
    assert personal_context_sync_readiness(capabilities).write_enabled is True


def test_downgraded_quotas_fail_closed() -> None:
    capabilities = SyncV2CapabilitiesResponse.model_validate(
        _capabilities(max_record_bytes=8_192)
    )

    assert personal_context_sync_readiness(capabilities).blockers == (
        "personal_context_quota_incompatible:max_record_bytes",
    )
