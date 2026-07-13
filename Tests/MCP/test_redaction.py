from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tldw_chatbook.MCP.redaction import (
    REDACTED,
    is_secret_key,
    redact_args,
    redact_mapping,
    redact_url,
)


def test_is_secret_key_matches_common_forms():
    for key in ("api_key", "API-KEY", "Authorization", "token", "client_secret", "PASSWORD"):
        assert is_secret_key(key), key
    for key in ("command", "name", "url", "working_dir"):
        assert not is_secret_key(key), key


def test_redact_mapping_recurses_and_preserves_safe_values():
    data = {"command": "python", "env": {"API_KEY": "sk-123", "PATH": "/usr/bin"}}
    redacted = redact_mapping(data)
    assert redacted["command"] == "python"
    assert redacted["env"]["API_KEY"] == REDACTED
    assert redacted["env"]["PATH"] == "/usr/bin"
    assert data["env"]["API_KEY"] == "sk-123"  # input not mutated


def test_redact_args_handles_flag_and_inline_forms():
    args = ["--api-key", "sk-123", "--verbose", "token=abc", "plain"]
    assert redact_args(args) == ["--api-key", REDACTED, "--verbose", f"token={REDACTED}", "plain"]


def test_redact_url_strips_secret_query_values():
    url = "https://api.example.com/v1?api_key=sk-123&page=2"
    redacted = redact_url(url)
    assert "sk-123" not in redacted
    assert "page=2" in redacted


@given(st.dictionaries(st.text(min_size=1, max_size=20), st.text(max_size=40), max_size=8))
def test_redact_mapping_never_leaks_values_under_secret_keys(data):
    redacted = redact_mapping(data)
    for key, value in data.items():
        if is_secret_key(key) and value:
            assert redacted[key] == REDACTED
        else:
            assert redacted[key] == value
