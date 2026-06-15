"""P1: client parses the live M1 server's capability/profile payloads."""

from tldw_chatbook.tldw_api import (
    SyncV2CapabilitiesResponse,
    SyncV2ProfileBootstrapRequest,
    SyncV2ProfileBootstrapResponse,
    SyncV2ProfileResponse,
)

# Captured verbatim from the live codex/sync-v2-m1-next server @ 992e89a03.
LIVE_CAPABILITIES = {
    "protocol_version": "sync-v2-m1",
    "min_supported_protocol_version": "sync-v2-m1",
    "domains": [
        "notes.note", "chat.conversation", "chat.message", "attachment.ref",
        "workspaces.workspace", "workspaces.source_ref", "source_cache.entry",
        "media.item", "media.keyword", "media.keyword_link",
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


def test_capabilities_back_compat_properties():
    caps = SyncV2CapabilitiesResponse.model_validate(LIVE_CAPABILITIES)
    # Legacy readers used .supported_domains / .supported_operations.
    assert "notes.note" in caps.supported_domains
    assert "append" in caps.supported_operations


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
        "domains": ["notes.note", "chat.conversation", "chat.message", "attachment.ref"],
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
        "notes.note", "chat.conversation", "chat.message", "attachment.ref",
    ]
