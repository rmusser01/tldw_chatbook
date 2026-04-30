from __future__ import annotations

import base64
import json
from types import SimpleNamespace

from tldw_chatbook.runtime_policy.server_event_scope import event_principal_id_from_active_context


def _unsigned_jwt(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}."


def test_event_principal_id_prefers_jwt_subject_without_storing_token_material():
    context = SimpleNamespace(auth_token=_unsigned_jwt({"sub": "user-123"}), credential_source="stored_token")

    principal_id = event_principal_id_from_active_context(context)

    assert principal_id == "jwt-sub:user-123"


def test_event_principal_id_uses_credential_fingerprint_for_opaque_tokens():
    context = SimpleNamespace(auth_token="opaque-secret-token", credential_source="stored_token")

    principal_id = event_principal_id_from_active_context(context)

    assert principal_id.startswith("credential-fingerprint:stored_token:")
    assert "opaque-secret-token" not in principal_id


def test_event_principal_id_falls_back_to_fingerprint_for_malformed_jwt_like_token():
    context = SimpleNamespace(auth_token="not-a-header.not-base64.", credential_source="stored_token")

    principal_id = event_principal_id_from_active_context(context)

    assert principal_id.startswith("credential-fingerprint:stored_token:")
    assert "not-base64" not in principal_id


def test_event_principal_id_returns_none_without_auth_token():
    context = SimpleNamespace(auth_token=None, credential_source="none")

    assert event_principal_id_from_active_context(context) is None
