from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ConsentPreferencesResponse,
    ConsentRecordResponse,
    PrivilegeSelfResponse,
    TLDWAPIClient,
)


NOW = "2026-04-25T12:00:00Z"


def _consent_record(purpose: str = "analytics", **overrides) -> dict:
    payload = {
        "id": 10,
        "user_id": 42,
        "purpose": purpose,
        "granted_at": NOW,
        "withdrawn_at": None,
        "ip_address": "127.0.0.1",
        "user_agent": "Chatbook",
        "metadata": None,
    }
    payload.update(overrides)
    return payload


def _privilege_self_payload() -> dict:
    return {
        "catalog_version": "2026-04-25",
        "generated_at": NOW,
        "items": [
            {
                "endpoint": "/api/v1/notes",
                "method": "GET",
                "privilege_scope_id": "notes.read",
                "feature_flag_id": None,
                "sensitivity_tier": "user",
                "ownership_predicates": ["owner"],
                "status": "allowed",
                "blocked_reason": None,
                "dependencies": [{"id": "auth", "type": "dependency", "module": "auth"}],
                "dependency_sources": ["auth"],
                "rate_limit_class": "standard",
                "rate_limit_resources": ["notes"],
                "source_module": "notes",
                "summary": "Read own notes",
                "tags": ["notes"],
            }
        ],
        "recommended_actions": [{"privilege_scope_id": "notes.read", "action": "none", "reason": None}],
    }


@pytest.mark.asyncio
async def test_user_governance_client_routes_consent_and_self_privileges(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"user_id": 42, "consents": [_consent_record()]},
            _consent_record("personalization"),
            _consent_record("analytics", withdrawn_at="2026-04-25T12:05:00Z"),
            _privilege_self_payload(),
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    preferences = await client.get_consent_preferences()
    granted = await client.grant_consent("personalization")
    withdrawn = await client.withdraw_consent("analytics")
    privileges = await client.get_self_privilege_map(resource="notes")

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/consent/preferences")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/consent/preferences/personalization")
    assert mocked.await_args_list[2].args[:2] == ("DELETE", "/api/v1/consent/preferences/analytics")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/privileges/self")
    assert mocked.await_args_list[3].kwargs["params"] == {"resource": "notes"}

    assert isinstance(preferences, ConsentPreferencesResponse)
    assert isinstance(granted, ConsentRecordResponse)
    assert isinstance(withdrawn, ConsentRecordResponse)
    assert isinstance(privileges, PrivilegeSelfResponse)
    assert preferences.user_id == 42
    assert granted.purpose == "personalization"
    assert withdrawn.withdrawn_at is not None
    assert privileges.items[0].privilege_scope_id == "notes.read"
