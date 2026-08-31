from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api import (
    SyncPersonalContextBootstrapRequest,
    SyncPersonalContextBootstrapResponse,
    SyncPersonalContextLinkCompleteRequest,
    TLDWAPIClient,
)


BOOTSTRAP = {
    "dataset_id": "dataset-personal",
    "authority_id": "authority-stable",
    "manifest": {
        "profile_id": "profile-server",
        "revision": 1,
        "purge_generation": 0,
        "created_at": "2026-08-30T12:00:00.000Z",
        "updated_at": "2026-08-30T12:00:00.000Z",
        "current_version_id": "manifest-version-server",
    },
    "scopes": [],
    "records": [],
    "proposals": [],
    "purge_generation": 0,
    "schema_version": 1,
    "quotas": {"max_record_bytes": 16_384},
    "cursor": "sha256:" + "a" * 64,
    "integrity_key_id": "integrity-key-server",
    "key_record_id": "key-record-device",
    "wrapped_key_blob": "rsa-oaep-sha256:ciphertext",
}


def test_personal_context_bootstrap_schemas_are_strict_and_typed() -> None:
    request = SyncPersonalContextBootstrapRequest(
        device_id="device-1",
        required_schema_version=1,
        required_quotas={"max_record_bytes": 16_384},
        expected_purge_generation=0,
    )
    response = SyncPersonalContextBootstrapResponse.model_validate(BOOTSTRAP)

    assert request.model_dump(mode="json") == {
        "device_id": "device-1",
        "required_schema_version": 1,
        "required_quotas": {"max_record_bytes": 16_384},
        "expected_purge_generation": 0,
    }
    assert response.manifest.profile_id == "profile-server"
    assert response.cursor == BOOTSTRAP["cursor"]
    assert response.wrapped_key_blob == BOOTSTRAP["wrapped_key_blob"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SyncPersonalContextBootstrapRequest(device_id="device-1", secret="nope")


@pytest.mark.asyncio
async def test_client_uses_exact_personal_context_bootstrap_and_complete_paths(
    monkeypatch,
) -> None:
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(side_effect=[BOOTSTRAP, None])
    monkeypatch.setattr(client, "_request", mocked)

    snapshot = await client.bootstrap_sync_v2_personal_context(
        SyncPersonalContextBootstrapRequest(
            device_id="device-1", required_schema_version=1
        )
    )
    completed = await client.complete_sync_v2_personal_context_link(
        SyncPersonalContextLinkCompleteRequest(
            device_id="device-1",
            dataset_id=snapshot.dataset_id,
            bootstrap_cursor=snapshot.cursor,
        )
    )

    assert snapshot.manifest.profile_id == "profile-server"
    assert completed is None
    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("POST", "/api/v1/sync/personal-context/bootstrap"),
        ("POST", "/api/v1/sync/personal-context/complete"),
    ]
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "device_id": "device-1",
        "dataset_id": "dataset-personal",
        "bootstrap_cursor": BOOTSTRAP["cursor"],
    }
