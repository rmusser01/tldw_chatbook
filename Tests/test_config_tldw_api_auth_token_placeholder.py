"""Unit tests for config_module.resolve_tldw_api_auth_token (task-31417).

The end-to-end boot-rewrite + resolve round trip lives in
Tests/RuntimePolicy/test_server_context_provider.py (AC#5 requires that,
not just this predicate in isolation) -- these pin the predicate itself.
"""

from tldw_chatbook import config as config_module


def test_rejects_the_boot_rewrite_placeholder():
    assert (
        config_module.resolve_tldw_api_auth_token(
            config_module.TLDW_API_PLACEHOLDER_AUTH_TOKEN
        )
        is None
    )


def test_rejects_blank_and_provider_key_placeholders():
    assert config_module.resolve_tldw_api_auth_token("") is None
    assert config_module.resolve_tldw_api_auth_token("   ") is None
    assert config_module.resolve_tldw_api_auth_token("<API_KEY_HERE>") is None
    assert config_module.resolve_tldw_api_auth_token(None) is None


def test_accepts_and_strips_a_genuine_token():
    assert (
        config_module.resolve_tldw_api_auth_token("  real-token-value  ")
        == "real-token-value"
    )
