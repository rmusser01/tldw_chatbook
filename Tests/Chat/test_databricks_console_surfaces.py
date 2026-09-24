"""Databricks Console provider-surface wiring (ADR-179, Task 10).

Pins the four surfaces the first engine-driven provider must appear on:
readiness requirements (API key AND workspace base URL), the builtin
endpoint documentation map (no shipped default -- the workspace host is
per-account), the display-name table, and provider-setup persistence
ownership.

Readiness tests follow the hand-built-config + injected-environ pattern of
``Tests/Chat/test_provider_readiness.py`` (e.g. its hosted-readiness
parametrizations): ``get_provider_readiness`` reads only the passed
mappings, so no config-loader fixture is needed for a provider with no
legacy ``[API]`` bridge.
"""

import pytest

from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

_DATABRICKS_WORKSPACE_URL = (
    "https://adb-1234567890123456.7.azuredatabricks.net"
)


def test_readiness_blocked_without_api_key_names_the_credential_requirement():
    """No stored key and no DATABRICKS_TOKEN in the environment blocks with
    copy naming the API key -- the same actionable shape every keyed
    provider gets, with Databricks's registry env var (DATABRICKS_TOKEN,
    not the DATABRICKS_API_KEY convention) in the remedy."""
    from_default_table = get_provider_readiness(
        "Databricks",
        {"api_settings": {"databricks": {"api_key_env_var": "DATABRICKS_TOKEN"}}},
        environ={},
    )
    from_empty_settings = get_provider_readiness(
        "Databricks", {"api_settings": {}}, environ={}
    )

    for readiness in (from_default_table, from_empty_settings):
        assert readiness.ready is False
        assert readiness.requires_api_key is True
        assert readiness.reason == "Missing API key"
        assert readiness.env_var == "DATABRICKS_TOKEN"
        assert readiness.user_message == (
            "Databricks is not ready: Missing API key. Set DATABRICKS_TOKEN "
            "or add api_key under [api_settings.databricks]."
        )


def test_readiness_blocked_with_key_but_no_base_url_names_the_workspace_url():
    """A resolved key alone must not mark Databricks ready: the workspace
    host is per-account (provider_registry.DATABRICKS.default_base_url is
    None), so the send path raises without api_base_url. The blocked record
    names the workspace URL, never retains the credential, and reports the
    structured endpoint_missing issue."""
    from tldw_chatbook.Chat.provider_readiness import (
        PROVIDERS_REQUIRING_BASE_URL_KEYS,
    )

    assert "databricks" in PROVIDERS_REQUIRING_BASE_URL_KEYS

    env_sourced = get_provider_readiness(
        "Databricks",
        {"api_settings": {"databricks": {"api_key_env_var": "DATABRICKS_TOKEN"}}},
        environ={"DATABRICKS_TOKEN": "dapi-env-secret-canary"},
    )
    config_sourced = get_provider_readiness(
        "Databricks",
        {"api_settings": {"databricks": {"api_key": "dapi-stored-secret-canary"}}},
        environ={},
    )

    for readiness in (env_sourced, config_sourced):
        assert readiness.ready is False
        assert readiness.reason == "Missing workspace URL"
        assert readiness.configuration_facet == "incomplete"
        assert readiness.configuration_issue == "endpoint_missing"
        assert readiness.api_key is None
        assert readiness.api_key_source is None
        assert "api_base_url" in readiness.recovery
        assert "workspace" in readiness.recovery
        assert "api_settings.databricks" in readiness.user_message
        assert "dapi-env-secret-canary" not in readiness.user_message
        assert "dapi-stored-secret-canary" not in readiness.user_message


@pytest.mark.parametrize(
    "base_url_key", ["api_base_url", "base_url"], ids=["canonical", "alias"]
)
def test_readiness_ready_when_key_and_base_url_are_configured(base_url_key):
    """Both requirements satisfied opens the gate, regardless of which
    credential source and base-URL alias supplied them."""
    env_key = get_provider_readiness(
        "Databricks",
        {
            "api_settings": {
                "databricks": {
                    "api_key_env_var": "DATABRICKS_TOKEN",
                    base_url_key: _DATABRICKS_WORKSPACE_URL,
                }
            }
        },
        environ={"DATABRICKS_TOKEN": "dapi-env-secret-canary"},
    )
    stored_key = get_provider_readiness(
        "Databricks",
        {
            "api_settings": {
                "databricks": {
                    "api_key": "dapi-stored-secret-canary",
                    base_url_key: _DATABRICKS_WORKSPACE_URL,
                }
            }
        },
        environ={},
    )

    assert env_key.ready is True
    assert env_key.api_key == "dapi-env-secret-canary"
    assert env_key.api_key_source == "env:DATABRICKS_TOKEN"
    assert stored_key.ready is True
    assert stored_key.api_key == "dapi-stored-secret-canary"
    assert (
        stored_key.api_key_source == "config:api_settings.databricks.api_key"
    )
    assert "dapi-env-secret-canary" not in env_key.user_message
    assert "dapi-stored-secret-canary" not in stored_key.user_message


def test_display_name_and_endpoint_doc_entries():
    """Display name is "Databricks", and the builtin endpoint map documents
    the per-account shape without shipping a live fallback: every value in
    that map is a real send-path fallback (console_provider_gateway resolves
    effective_base_url through it), and "<workspace-host>/openai/v1" is a
    placeholder, not a URL -- so Databricks must have NO entry, and
    ``builtin_provider_endpoint`` must return None until the user
    configures their workspace."""
    from tldw_chatbook.Chat.console_provider_endpoints import (
        _BUILTIN_PROVIDER_ENDPOINTS,
        builtin_provider_endpoint,
    )
    from tldw_chatbook.Chat.console_provider_support import _PROVIDER_DISPLAY_NAMES

    assert _PROVIDER_DISPLAY_NAMES.get("databricks") == "Databricks"
    # The documented per-account shape: the engine appends the /openai/v1
    # suffix (provider_registry.DATABRICKS.base_url_suffix).
    assert _BUILTIN_PROVIDER_ENDPOINTS.get(
        "databricks", "<workspace-host>/openai/v1"
    ).endswith("/openai/v1")
    assert "databricks" not in _BUILTIN_PROVIDER_ENDPOINTS
    assert builtin_provider_endpoint("databricks", {}) is None


def test_setup_persistence_accepts_databricks():
    """Databricks is a canonical setup-persistence owner: the display-name
    alias resolves to the canonical key, and persistence writes the
    workspace URL under api_base_url (not the api_url alias local
    providers use)."""
    from tldw_chatbook.Chat.provider_setup_persistence import (
        _CANONICAL_PROVIDER_KEYS,
        canonical_provider_key,
        provider_endpoint_key,
    )

    assert "databricks" in _CANONICAL_PROVIDER_KEYS
    assert canonical_provider_key("Databricks") == "databricks"
    assert canonical_provider_key("databricks") == "databricks"
    assert provider_endpoint_key("databricks") == "api_base_url"
