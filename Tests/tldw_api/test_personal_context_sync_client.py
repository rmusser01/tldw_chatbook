from __future__ import annotations

import copy
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

import tldw_chatbook.tldw_api as tldw_api
from tldw_chatbook.tldw_api import exceptions, sync_schemas
from tldw_chatbook.tldw_api import (
    SyncPersonalContextBootstrapRequest,
    SyncPersonalContextBootstrapResponse,
    SyncPersonalContextLinkCompleteRequest,
    TLDWAPIClient,
)
from tldw_chatbook.tldw_api.exceptions import APIResponseError


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
    "sync_transport_cursor": "4271",
    "integrity_key_id": "integrity-key-server",
    "key_record_id": "key-record-device",
    "wrapped_key_blob": "rsa-oaep-sha256:ciphertext",
}


ATTENTION_CASES = (
    (
        "schema_incompatible",
        "personal_context_schema_incompatible",
        {
            "kind": "schema_incompatible",
            "required_schema_version": 3,
            "server_min_schema_version": 1,
            "server_max_schema_version": 2,
        },
    ),
    (
        "quota_incompatible",
        "personal_context_quota_incompatible",
        {
            "kind": "quota_incompatible",
            "required_quotas": {"max_record_bytes": 16_384},
            "available_quotas": {"max_record_bytes": 8_192},
            "insufficient_quotas": ["max_record_bytes"],
        },
    ),
    (
        "purge_generation_mismatch",
        "personal_context_purge_generation_stale",
        {
            "kind": "purge_generation_mismatch",
            "expected_purge_generation": 4,
            "current_purge_generation": 5,
        },
    ),
)


def _attention_response(error_code: str, attention: dict) -> dict:
    return {
        "detail": {
            "error_code": error_code,
            "message": "Content-free compatibility attention.",
            "attention": attention,
        }
    }


def _client_with_response(status_code: int, body: dict) -> TLDWAPIClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/sync/personal-context/bootstrap"
        return httpx.Response(status_code, json=body)

    client = TLDWAPIClient("https://server.example")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
    )
    return client


def test_personal_context_bootstrap_attention_types_are_public_api() -> None:
    public_names = (
        "PersonalContextBootstrapAttentionError",
        "SyncPersonalContextBootstrapAttention",
        "SyncPersonalContextBootstrapErrorDetail",
        "SyncPersonalContextBootstrapErrorResponse",
        "SyncPersonalContextPurgeAttention",
        "SyncPersonalContextQuotaAttention",
        "SyncPersonalContextSchemaAttention",
    )

    for name in public_names:
        assert name in tldw_api.__all__
        assert getattr(tldw_api, name) is not None


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
    assert response.sync_transport_cursor == "4271"
    assert response.wrapped_key_blob == BOOTSTRAP["wrapped_key_blob"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SyncPersonalContextBootstrapRequest(device_id="device-1", secret="nope")


@pytest.mark.parametrize(
    "required_quotas",
    (
        {"Bad-Name": 1},
        {1: 1},
        {"a" * 65: 1},
        {"max_record_bytes": True},
        {"max_record_bytes": 1.0},
        {"max_record_bytes": "1"},
        {"max_record_bytes": -1},
        {"max_record_bytes": 2**63},
        {f"quota_{index}": 0 for index in range(33)},
    ),
)
def test_personal_context_bootstrap_request_rejects_malformed_quota_contract(
    required_quotas,
) -> None:
    with pytest.raises(ValidationError):
        SyncPersonalContextBootstrapRequest(
            device_id="device-1",
            required_quotas=required_quotas,
        )


@pytest.mark.parametrize(
    "quotas",
    (
        {"Bad-Name": 1},
        {1: 1},
        {"a" * 65: 1},
        {"max_record_bytes": True},
        {"max_record_bytes": 1.0},
        {"max_record_bytes": "1"},
        {"max_record_bytes": -1},
        {"max_record_bytes": 2**63},
        {f"quota_{index}": 0 for index in range(33)},
    ),
)
def test_personal_context_bootstrap_response_rejects_malformed_quota_contract(
    quotas,
) -> None:
    payload = copy.deepcopy(BOOTSTRAP)
    payload["quotas"] = quotas

    with pytest.raises(ValidationError):
        SyncPersonalContextBootstrapResponse.model_validate(payload)


@pytest.mark.parametrize("quotas", (None, {}), ids=("omitted", "empty"))
def test_personal_context_bootstrap_response_requires_nonempty_quotas(quotas) -> None:
    payload = copy.deepcopy(BOOTSTRAP)
    if quotas is None:
        payload.pop("quotas")
    else:
        payload["quotas"] = quotas

    with pytest.raises(ValidationError):
        SyncPersonalContextBootstrapResponse.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("quotas", (None, {}), ids=("omitted", "empty"))
async def test_client_rejects_bootstrap_success_without_nonempty_quotas(quotas) -> None:
    payload = copy.deepcopy(BOOTSTRAP)
    if quotas is None:
        payload.pop("quotas")
    else:
        payload["quotas"] = quotas
    client = _client_with_response(200, payload)

    try:
        with pytest.raises(ValidationError):
            await client.bootstrap_sync_v2_personal_context(
                SyncPersonalContextBootstrapRequest(device_id="device-1")
            )
    finally:
        await client.close()


@pytest.mark.parametrize(
    "value",
    (None, "", "x" * 32_769),
)
def test_personal_context_bootstrap_requires_bounded_transport_cursor(value) -> None:
    payload = copy.deepcopy(BOOTSTRAP)
    if value is None:
        payload.pop("sync_transport_cursor")
    else:
        payload["sync_transport_cursor"] = value

    with pytest.raises(ValidationError):
        SyncPersonalContextBootstrapResponse.model_validate(payload)


@pytest.mark.parametrize(("kind", "error_code", "attention"), ATTENTION_CASES)
def test_personal_context_bootstrap_attention_schema_is_discriminated_and_strict(
    kind: str,
    error_code: str,
    attention: dict,
) -> None:
    error_response_type = getattr(
        sync_schemas, "SyncPersonalContextBootstrapErrorResponse"
    )

    parsed = error_response_type.model_validate(
        _attention_response(error_code, attention)
    )

    assert parsed.detail.attention.kind == kind
    malformed = _attention_response(error_code, copy.deepcopy(attention))
    malformed["detail"]["attention"]["unexpected"] = "not trusted"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        error_response_type.model_validate(malformed)


@pytest.mark.parametrize(
    ("error_code", "attention"),
    (
        (
            "personal_context_schema_incompatible",
            {
                "kind": "schema_incompatible",
                "required_schema_version": "3",
                "server_min_schema_version": 1,
                "server_max_schema_version": 2,
            },
        ),
        (
            "personal_context_quota_incompatible",
            {
                "kind": "quota_incompatible",
                "required_quotas": {"max_record_bytes": 16_384},
                "available_quotas": {"max_record_bytes": 16_384},
                "insufficient_quotas": ["max_record_bytes"],
            },
        ),
        (
            "personal_context_purge_generation_stale",
            {
                "kind": "purge_generation_mismatch",
                "expected_purge_generation": 4,
                "current_purge_generation": 4,
            },
        ),
        (
            "wrong_error_code",
            {
                "kind": "schema_incompatible",
                "required_schema_version": 3,
                "server_min_schema_version": 1,
                "server_max_schema_version": 2,
            },
        ),
    ),
)
def test_personal_context_bootstrap_attention_rejects_semantically_invalid_bodies(
    error_code: str,
    attention: dict,
) -> None:
    error_response_type = getattr(
        sync_schemas, "SyncPersonalContextBootstrapErrorResponse"
    )

    with pytest.raises(ValidationError):
        error_response_type.model_validate(
            _attention_response(error_code, attention)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "error_code", "attention"), ATTENTION_CASES)
async def test_client_raises_only_typed_content_free_bootstrap_attention(
    kind: str,
    error_code: str,
    attention: dict,
) -> None:
    attention_error_type = getattr(
        exceptions, "PersonalContextBootstrapAttentionError"
    )
    body = _attention_response(error_code, attention)
    body["detail"]["message"] = "private-server-message-must-not-cross"
    client = _client_with_response(409, body)

    try:
        with pytest.raises(attention_error_type) as caught:
            await client.bootstrap_sync_v2_personal_context(
                SyncPersonalContextBootstrapRequest(device_id="device-1")
            )
    finally:
        await client.close()

    assert caught.value.attention.kind == kind
    assert str(caught.value) == "personal_context_bootstrap_attention_required"
    assert "private-server-message" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    (
        {"detail": {"attention": {"kind": "schema_incompatible"}}},
        {
            "detail": {
                "error_code": "personal_context_schema_incompatible",
                "message": "private-malformed-body",
                "attention": {
                    "kind": "schema_incompatible",
                    "required_schema_version": 3,
                    "server_min_schema_version": 1,
                    "server_max_schema_version": 2,
                    "payload": "must-not-be-trusted",
                },
            }
        },
        {"detail": "private-unstructured-error"},
    ),
)
async def test_client_leaves_malformed_bootstrap_attention_as_generic_error(
    body: dict,
) -> None:
    attention_error_type = getattr(
        exceptions, "PersonalContextBootstrapAttentionError"
    )
    client = _client_with_response(409, body)

    try:
        with pytest.raises(APIResponseError) as caught:
            await client.bootstrap_sync_v2_personal_context(
                SyncPersonalContextBootstrapRequest(device_id="device-1")
            )
    finally:
        await client.close()

    assert not isinstance(caught.value, attention_error_type)


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
