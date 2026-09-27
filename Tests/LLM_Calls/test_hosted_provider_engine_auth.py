"""Engine ``bearer_optional`` auth: keyed-or-keyless hosted endpoints (ADR-179).

Phase 2, Task 3: an engine preset may declare
``auth_scheme="bearer_optional"`` so keyless endpoints (the ADR-146 custom
endpoints of Task 6) execute through the engine, while curated ``bearer``
presets keep hard-requiring keys. Resolution contracts pinned here:

- ``bearer`` (default): a missing key is still an actionable configuration
  error -- byte-identical to Phase 1.
- ``bearer_optional``: a record that ships neither an env-var name nor
  candidates still resolves, with ``api_key == ""`` when nothing resolves;
  an explicit ``""`` is "no key", not an invalid key; a resolved key (here
  an explicit one) still wins.

Transport contract (``owned_json_post``): ``bearer_optional`` with an empty
key sends NO ``Authorization`` header at all (never an empty ``Bearer ``);
``bearer_optional`` with a key sends ``Authorization: Bearer <key>``;
``bearer`` with an empty key fails closed as a transport configuration
error before any request fires.

The transport test reuses the proven ``_RecordingSession`` /
``_TransportResponse`` doubles from ``Tests/LLM_Calls/test_qwencloud.py``
(pure in-memory fakes -- no sockets, so no ``allow_network`` opt-in needed).
"""

from __future__ import annotations

import dataclasses

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError, ChatProviderError
import tldw_chatbook.LLM_Calls.hosted_chat as hosted_chat
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedHTTPTransportConfig,
    owned_json_post,
)
from tldw_chatbook.LLM_Calls.hosted_provider_engine import resolve_hosted_request
from tldw_chatbook.provider_registry import CUSTOM_HOSTED, DATABRICKS

from Tests.LLM_Calls.test_qwencloud import _RecordingSession, _TransportResponse

_OPTIONAL = dataclasses.replace(
    DATABRICKS,
    key="optional-auth",
    display_name="Optional Auth",
    api_key_env_var=None,
    api_key_env_candidates=(),
    auth_scheme="bearer_optional",
)

# The global-credential trap (Qodo finding 1, ADR-179): a keyless custom
# entry's URL is user-controlled, so a globally configured
# [api_settings.custom] key (or CUSTOM_API_KEY) must never back-fill the
# gateway's explicit keyless decision.
_GLOBAL_KEY_CONFIG = {
    "api_settings": {
        "custom": {
            "api_key": "GLOBAL-CUSTOM-KEY",
            "api_key_env_var": "CUSTOM_API_KEY",
        }
    }
}


def test_resolved_keyless_decision_ignores_global_custom_key_and_env() -> None:
    resolution = resolve_hosted_request(
        CUSTOM_HOSTED,
        explicit_base_url="https://user-entry.example/v1",
        api_key_resolved=True,
        app_config=_GLOBAL_KEY_CONFIG,
        environ={"CUSTOM_API_KEY": "ENV-CUSTOM-KEY"},
    )
    # The keyless decision survives: no Authorization header (see the
    # transport contract below), no error, and neither fallback value.
    assert resolution.api_key == ""
    assert resolution.base_url == "https://user-entry.example/v1"


def test_resolved_keyed_decision_uses_supplied_key_only() -> None:
    resolution = resolve_hosted_request(
        CUSTOM_HOSTED,
        explicit_api_key="entry-key",
        explicit_base_url="https://user-entry.example/v1",
        api_key_resolved=True,
        app_config=_GLOBAL_KEY_CONFIG,
        environ={"CUSTOM_API_KEY": "ENV-CUSTOM-KEY"},
    )
    assert resolution.api_key == "entry-key"


def test_resolved_unusable_credential_fails_closed_for_bearer_records() -> None:
    with pytest.raises(ChatConfigurationError):
        resolve_hosted_request(
            DATABRICKS,
            explicit_base_url="https://dbc-1.cloud.databricks.com",
            api_key_resolved=True,
            app_config={
                "api_settings": {
                    "databricks": {"api_key": "GLOBAL-DATABRICKS-KEY"}
                }
            },
            environ={"DATABRICKS_TOKEN": "ENV-DATABRICKS-KEY"},
        )


def test_bearer_preset_still_requires_a_key():
    with pytest.raises(ChatConfigurationError):
        resolve_hosted_request(
            DATABRICKS,
            explicit_base_url="https://dbc-1.cloud.databricks.com",
            app_config={"api_settings": {"databricks": {}}},
            environ={},
        )


def test_bearer_optional_resolves_without_a_key_and_without_an_env_name():
    resolution = resolve_hosted_request(
        _OPTIONAL,
        explicit_base_url="https://anywhere.example/v1",
        app_config={"api_settings": {"optional-auth": {}}},
        environ={},
    )
    assert resolution.api_key == ""


# These two resolve without an app_config (the dispatch's binding shape), so
# they read the real runtime config snapshot through the guarded loader --
# a real-app config mount, hence the per-node bootstrap profile (conftest
# TASK-32873 pattern); the other tests here are pure sandbox doubles.
@pytest.mark.bootstrap_profile
def test_bearer_optional_treats_explicit_empty_string_as_no_key():
    resolution = resolve_hosted_request(
        _OPTIONAL,
        explicit_api_key="",
        explicit_base_url="https://anywhere.example/v1",
    )
    assert resolution.api_key == ""


@pytest.mark.bootstrap_profile
def test_bearer_optional_still_uses_a_resolved_key():
    resolution = resolve_hosted_request(
        _OPTIONAL,
        explicit_api_key="stored-key",
        explicit_base_url="https://anywhere.example/v1",
    )
    assert resolution.api_key == "stored-key"


def _transport(auth_scheme: str, api_key: str) -> HostedHTTPTransportConfig:
    return HostedHTTPTransportConfig(
        provider="optional-auth",
        base_url="https://anywhere.example/v1",
        api_key=api_key,
        timeout=0.2,
        retries=0,
        retry_delay=0.0,
        auth_scheme=auth_scheme,
    )


def _record_session(
    monkeypatch: pytest.MonkeyPatch, response: _TransportResponse
) -> _RecordingSession:
    session = _RecordingSession(response)
    monkeypatch.setattr(hosted_chat, "create_default_session", lambda: session)
    return session


def test_owned_json_post_authorization_follows_auth_scheme_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
    request = {"model": "gateway-model", "messages": [], "stream": False}

    # bearer_optional + empty key: no Authorization header at all.
    keyless = _record_session(monkeypatch, _TransportResponse(dict(success)))
    result = owned_json_post(
        config=_transport("bearer_optional", ""),
        route="chat/completions",
        payload=request,
        streaming=False,
    )
    assert result == success
    assert keyless.posts[0]["headers"].get("Authorization") is None

    # bearer_optional + key: exact Bearer header.
    keyed = _record_session(monkeypatch, _TransportResponse(dict(success)))
    owned_json_post(
        config=_transport("bearer_optional", "stored-key"),
        route="chat/completions",
        payload=request,
        streaming=False,
    )
    assert keyed.posts[0]["headers"]["Authorization"] == "Bearer stored-key"

    # bearer + empty key: fails closed before any request fires.
    refused = _record_session(monkeypatch, _TransportResponse(dict(success)))
    with pytest.raises(ChatProviderError):
        owned_json_post(
            config=_transport("bearer", ""),
            route="chat/completions",
            payload=request,
            streaming=False,
        )
    assert refused.posts == []
