"""Tests for log sanitization utilities."""

import tomllib

import pytest

from tldw_chatbook.config import CONFIG_TOML_CONTENT, DEFAULT_APP_TTS_CONFIG
from tldw_chatbook.Utils.log_sanitizer import (
    create_safe_log_message,
    sanitize_dict,
    sanitize_list,
    sanitize_log_params,
    sanitize_string,
)
from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key


def _iter_leaf_key_names(mapping):
    """Yield leaf mapping keys from the shipped configuration structure."""
    for key, value in mapping.items():
        if isinstance(value, dict):
            yield from _iter_leaf_key_names(value)
        else:
            yield key


def test_real_shipped_sensitive_key_names_are_redacted() -> None:
    """Redact every real secret-bearing default key using the shared policy."""
    default_config = tomllib.loads(CONFIG_TOML_CONTENT)
    key_names = {
        str(key)
        for key in _iter_leaf_key_names(default_config)
        if is_sensitive_config_key(key)
    }
    key_names.update(
        key for key in DEFAULT_APP_TTS_CONFIG if is_sensitive_config_key(key)
    )
    assert {"api_key", "auth_token", "api_token"} <= key_names

    sentinels = {
        key: f"PRIVATE_CONFIG_{index}" for index, key in enumerate(sorted(key_names))
    }
    result = sanitize_dict(sentinels)

    assert set(result) == set(sentinels)
    assert all(value == "***REDACTED***" for value in result.values())


@pytest.mark.parametrize(
    "key",
    [
        "Authorization",
        "Proxy-Authorization",
        "cookie",
        "Set-Cookie",
        "credential",
        "credentials",
        "database_url",
        "connection-string",
        "dsn",
    ],
)
def test_log_protocol_fields_are_redacted_without_expanding_config_policy(key: str) -> None:
    """Redact protocol-only fields while keeping config policy intentionally narrow."""
    assert not is_sensitive_config_key(key)
    assert sanitize_dict({key: "PRIVATE_PROTOCOL_VALUE"})[key] == "***REDACTED***"


def test_non_string_mapping_key_is_safe_and_sensitive_values_are_redacted() -> None:
    """Avoid lower() failures for non-string mapping keys."""
    result = sanitize_dict({1: "safe", "x-api-key": "PRIVATE"})

    assert result == {1: "safe", "x-api-key": "***REDACTED***"}


def test_non_mutating_sanitization_preserves_input_containers() -> None:
    """Return copies without changing supplied nested dictionaries or lists."""
    nested = {"token": "PRIVATE_NESTED"}
    items = [{"api_key": "PRIVATE_LIST"}]
    source = {"nested": nested, "items": items}

    result = sanitize_dict(source)

    assert source == {
        "nested": {"token": "PRIVATE_NESTED"},
        "items": [{"api_key": "PRIVATE_LIST"}],
    }
    assert result == {
        "nested": {"token": "***REDACTED***"},
        "items": [{"api_key": "***REDACTED***"}],
    }


def test_deep_false_returns_new_outer_container_and_sanitizes_direct_strings() -> None:
    """Skip nested traversal without sharing the outer mapping."""
    nested = {"token": "PRIVATE_NESTED"}
    items = ["token=PRIVATE_LIST"]
    source = {"nested": nested, "items": items, "message": "api_key=PRIVATE"}

    result = sanitize_dict(source, deep=False)

    assert result is not source
    assert result["nested"] is nested
    assert result["items"] is items
    assert result["message"] == "api_key=***REDACTED***"


@pytest.mark.parametrize("key", ["api_key_env_var", "max_tokens", "ordinary"])
def test_non_sensitive_keys_remain_unchanged(key: str) -> None:
    """Keep known non-secret configuration names and ordinary data intact."""
    assert sanitize_dict({key: "safe-value"}) == {key: "safe-value"}


@pytest.mark.parametrize("value", [{"nested": "PRIVATE"}, ["PRIVATE"]])
def test_sensitive_container_value_is_redacted_before_recursion(value) -> None:
    """Replace a sensitive container wholly instead of traversing its contents."""
    assert sanitize_dict({"api_key": value}) == {"api_key": "***REDACTED***"}


class TestLogSanitizer:
    """Test the log sanitization utilities."""

    def test_sanitize_string_api_keys(self):
        """Test that API keys are sanitized from strings."""
        test_cases = [
            ("api_key=sk-1234567890abcdef", "api_key=***REDACTED***"),
            (
                "Bearer sk-abcdefghijklmnopqrstuvwxyz123456789012345678",
                "Bearer ***OPENAI_KEY***",
            ),
            ("OPENAI_API_KEY=sk-test123", "OPENAI_API_KEY=***REDACTED***"),
            ('{"api_key": "secret123"}', '{"api_key": "***REDACTED***"}'),
            ("password: mypassword123", "password=***REDACTED***"),
            ("https://user:pass@example.com", "https://***:***@example.com"),
        ]

        for input_str, expected in test_cases:
            result = sanitize_string(input_str)
            assert result == expected

    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        test_dict = {
            "name": "test",
            "api_key": "sk-123456",
            "password": "secret",
            "nested": {"token": "bearer123", "safe": "value"},
            "config": "api_key=embedded_secret",
        }

        result = sanitize_dict(test_dict)

        assert result["name"] == "test"
        assert result["api_key"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"
        assert result["nested"]["token"] == "***REDACTED***"
        assert result["nested"]["safe"] == "value"
        assert "***REDACTED***" in result["config"]

    def test_sanitize_list(self):
        """Test list sanitization."""
        test_list = [
            "safe value",
            "api_key=secret",
            {"password": "hidden"},
            ["nested", "token=abc123"],
        ]

        result = sanitize_list(test_list)

        assert result[0] == "safe value"
        assert "***REDACTED***" in result[1]
        assert result[2]["password"] == "***REDACTED***"
        assert "***REDACTED***" in result[3][1]

    def test_create_safe_log_message(self):
        """Test safe log message creation."""
        # Test with positional args (OpenAI keys need 20+ chars after sk-)
        msg = create_safe_log_message(
            "User {} logged in with key {}",
            "john",
            "sk-abcdefghijklmnopqrstuvwxyz123456",
        )
        assert msg == "User john logged in with key ***OPENAI_KEY***"

        # Test with keyword args
        msg = create_safe_log_message("Config: {config}", config={"api_key": "secret"})
        assert "***REDACTED***" in msg

    def test_sanitize_log_params(self):
        """Test parameter sanitization."""
        args = ("test", {"api_key": "secret"}, "password=123")
        kwargs = {"token": "bearer123", "safe": "value"}

        clean_args, clean_kwargs = sanitize_log_params(*args, **kwargs)

        assert clean_args[0] == "test"
        assert clean_args[1]["api_key"] == "***REDACTED***"
        assert "***REDACTED***" in clean_args[2]
        assert clean_kwargs["token"] == "***REDACTED***"
        assert clean_kwargs["safe"] == "value"
