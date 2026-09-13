"""P1: client parses the live M1 server's capability/profile payloads."""

from tldw_chatbook.tldw_api import (
    SyncV2CapabilitiesResponse,
    SyncV2ProfileBootstrapRequest,
    SyncV2ProfileBootstrapResponse,
    SyncV2ProfileResponse,
)
from tldw_chatbook.tldw_api.sync_schemas import (
    SyncV2ProfileDatasetStatus,
    SyncV2PushAcceptedEnvelope,
)

# Captured verbatim from the live codex/sync-v2-m1-next server @ 992e89a03.
LIVE_CAPABILITIES = {
    "protocol_version": "sync-v2-m1",
    "min_supported_protocol_version": "sync-v2-m1",
    "domains": [
        "notes.note",
        "chat.conversation",
        "chat.message",
        "attachment.ref",
        "workspaces.workspace",
        "workspaces.source_ref",
        "source_cache.entry",
        "media.item",
        "media.keyword",
        "media.keyword_link",
    ],
    "operations": {
        "notes.note": ["upsert", "tombstone"],
        "chat.conversation": ["upsert", "tombstone"],
        "chat.message": ["append", "tombstone"],
        "attachment.ref": ["upsert", "tombstone"],
    },
    "encryption": {"policy": "server_trusted_v1", "ready": True},
    "encryption_policies": ["server_trusted_v1"],
    "blob_transfer": {"supported": False},
    "max_batch_size": 100,
    "max_envelope_payload_bytes": 262144,
    "max_attachment_bytes": 1048576,
    "supports_restore_manifest": True,
    "supports_conflicts": True,
    "supports_attachments": False,
    "compatibility_flags": {},
    "quota": {},
    "server_time": "2026-06-14T00:00:00Z",
    "warnings": [],
}


def test_capabilities_parses_live_m1_payload():
    caps = SyncV2CapabilitiesResponse.model_validate(LIVE_CAPABILITIES)
    assert caps.protocol_version == "sync-v2-m1"
    assert caps.min_supported_protocol_version == "sync-v2-m1"
    assert "notes.note" in caps.domains
    assert caps.operations["chat.message"] == ["append", "tombstone"]
    assert caps.encryption_policies == ["server_trusted_v1"]
    assert caps.supports_attachments is False
    assert caps.encryption["policy"] == "server_trusted_v1"


def test_capabilities_preserves_notes_adapter_version_advertisement():
    payload = {
        **LIVE_CAPABILITIES,
        "supported_adapter_versions": {
            "notes.note": [1],
            "notes.keyword": [1],
        },
    }

    caps = SyncV2CapabilitiesResponse.model_validate(payload)

    assert caps.supported_adapter_versions == {
        "notes.note": [1],
        "notes.keyword": [1],
    }


def test_push_accepted_envelope_preserves_materialization_error_details():
    accepted = SyncV2PushAcceptedEnvelope.model_validate(
        {
            "client_envelope_id": "intent-1",
            "server_cursor": 17,
            "apply_status": "failed",
            "apply_error_code": "projection_failed",
            "apply_error_message": "folder parent is missing",
        }
    )

    assert accepted.model_dump(mode="json") == {
        "client_envelope_id": "intent-1",
        "envelope_id": None,
        "server_sequence": 17,
        "domain": None,
        "entity_id": None,
        "object_id": None,
        "object_revision": None,
        "apply_status": "failed",
        "apply_error_code": "projection_failed",
        "apply_error_message": "folder parent is missing",
        "server_cursor": 17,
    }


def test_capabilities_back_compat_properties():
    caps = SyncV2CapabilitiesResponse.model_validate(LIVE_CAPABILITIES)
    # Legacy readers used .supported_domains / .supported_operations.
    assert "notes.note" in caps.supported_domains
    assert "append" in caps.supported_operations


def test_capabilities_parses_legacy_flat_supported_operations():
    caps = SyncV2CapabilitiesResponse.model_validate(
        {
            "protocol_version": 2,
            "supported_domains": ["notes", "chat"],
            "supported_operations": ["upsert", "delete", "resolve_conflict"],
        }
    )

    assert caps.domains == ["notes", "chat"]
    assert caps.operations == {"*": ["upsert", "delete", "resolve_conflict"]}
    assert caps.supported_operations == ["delete", "resolve_conflict", "upsert"]


def test_capabilities_parses_legacy_domain_supported_operations():
    caps = SyncV2CapabilitiesResponse.model_validate(
        {
            "supported_domains": ["notes.note"],
            "supported_operations": {
                "notes.note": ["upsert", "tombstone"],
            },
        }
    )

    assert caps.operations == {"notes.note": ["upsert", "tombstone"]}


def test_capabilities_coerces_legacy_int_protocol_version():
    caps = SyncV2CapabilitiesResponse.model_validate(
        {"protocol_version": 2, "min_supported_protocol_version": 2}
    )
    assert caps.protocol_version == "sync-v2-m1"
    assert caps.min_supported_protocol_version == "sync-v2-m1"


# Shape mirrors tldw_server2 Sync_V2_M1.md POST /profile/bootstrap response.
LIVE_BOOTSTRAP_RESPONSE = {
    "created": True,
    "profile_bootstrapped": True,
    "user_id": "user_123",
    "active_dataset_id": "ds_personal_01HZZ0",
    "device": {
        "device_id": "dev_chatbook_laptop",
        "registered": True,
        "client_profile_id": "chatbook_profile_main",
        "last_seen_at": "2026-06-14T00:00:00Z",
    },
    "dataset": {
        "dataset_id": "ds_personal_01HZZ0",
        "scope": "personal",
        "default_personal": True,
        "client_family": "chatbook",
        "domains": [
            "notes.note",
            "chat.conversation",
            "chat.message",
            "attachment.ref",
        ],
    },
    "server_cursor": 0,
    "capabilities": LIVE_CAPABILITIES,
    "domain_status": [],
    "warnings": [],
}


def test_bootstrap_response_parses_and_exposes_identity():
    resp = SyncV2ProfileBootstrapResponse.model_validate(LIVE_BOOTSTRAP_RESPONSE)
    assert resp.created is True
    assert resp.profile_bootstrapped is True
    assert resp.device.device_id == "dev_chatbook_laptop"
    assert resp.dataset.dataset_id == "ds_personal_01HZZ0"
    assert resp.active_dataset_id == "ds_personal_01HZZ0"
    assert resp.capabilities.protocol_version == "sync-v2-m1"


def test_profile_response_handles_unbootstrapped():
    resp = SyncV2ProfileResponse.model_validate(
        {"profile_bootstrapped": False, "user_id": "user_123", "server_cursor": 0}
    )
    assert resp.profile_bootstrapped is False
    assert resp.dataset is None
    assert resp.device is None


def test_bootstrap_request_defaults_to_m1_domains_and_offline_mode():
    req = SyncV2ProfileBootstrapRequest(device_name="Riley's MacBook")
    dumped = req.model_dump(mode="json")
    assert dumped["mode"] == "offline_sync"
    assert dumped["client_family"] == "chatbook"
    assert dumped["requested_domains"] == [
        "notes.note",
        "chat.conversation",
        "chat.message",
        "attachment.ref",
    ]


def test_profile_dataset_parses_notes_organization_bootstrap_status():
    dataset = SyncV2ProfileDatasetStatus.model_validate(
        {
            "dataset_id": "dataset-1",
            "domains": ["notes.keyword", "notes.folder"],
            "notes_organization": {
                "state": "initializing",
                "captured_count": 2,
                "expected_count": 7,
                "error_code": None,
            },
        }
    )

    assert dataset.notes_organization is not None
    assert dataset.notes_organization.state == "initializing"
    assert dataset.notes_organization.captured_count == 2
    assert dataset.notes_organization.expected_count == 7
    assert dataset.notes_organization.error_code is None


def test_bootstrap_request_carries_complete_versioned_notes_organization_group():
    organization_domains = [
        "notes.keyword",
        "notes.keyword_link",
        "notes.keyword_collection",
        "notes.keyword_collection_link",
        "notes.folder",
        "notes.folder_link",
    ]
    request = SyncV2ProfileBootstrapRequest(
        device_name="Enrollment device",
        requested_domains=[
            "notes.note",
            "chat.conversation",
            *organization_domains,
        ],
        supported_adapter_versions={
            domain: [1]
            for domain in [
                "notes.note",
                "chat.conversation",
                *organization_domains,
            ]
        },
    )

    dumped = request.model_dump(mode="json")
    assert dumped["requested_domains"] == [
        "notes.note",
        "chat.conversation",
        *organization_domains,
    ]
    assert dumped["supported_adapter_versions"] == {
        domain: [1]
        for domain in ["notes.note", "chat.conversation", *organization_domains]
    }
    assert "encryption_policy" not in dumped
