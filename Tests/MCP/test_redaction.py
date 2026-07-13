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


def test_redact_mapping_redacts_secret_keyed_mapping_value():
    # A secret-looking key must be redacted even when its value is itself a
    # Mapping; the key check must win over the recursion branch.
    data = {"api_key": {"value": "sk-123"}}
    redacted = redact_mapping(data)
    assert redacted["api_key"] == REDACTED
    assert data["api_key"] == {"value": "sk-123"}  # input not mutated


def test_redact_mapping_redacts_and_deep_copies_nested_sequences():
    leaked = {"api_key": "sk-999", "name": "svc"}
    other = {"name": "other"}
    data = {"servers": [leaked, other]}
    redacted = redact_mapping(data)
    # New list, new inner dicts: no shared mutable references leak out.
    assert redacted["servers"] is not data["servers"]
    assert redacted["servers"][0] is not leaked
    assert redacted["servers"][1] is not other
    assert redacted["servers"][0]["api_key"] == REDACTED
    assert redacted["servers"][0]["name"] == "svc"
    assert redacted["servers"][1]["name"] == "other"
    assert data["servers"][0]["api_key"] == "sk-999"  # input not mutated


def test_redact_mapping_preserves_tuple_type_for_sequences():
    data = {"pair": ({"token": "sk-1"}, {"name": "safe"})}
    redacted = redact_mapping(data)
    assert isinstance(redacted["pair"], tuple)
    assert redacted["pair"][0]["token"] == REDACTED
    assert redacted["pair"][1]["name"] == "safe"


def test_redact_mapping_does_not_iterate_into_strings():
    # Strings/bytes are Sequences too; they must pass through untouched
    # rather than being exploded into a list of characters.
    data = {"note": "credential-free plain text"}
    redacted = redact_mapping(data)
    assert redacted["note"] == data["note"]
    assert isinstance(redacted["note"], str)


def test_redact_args_reevaluates_flag_after_consecutive_secret_flags():
    # The second secret flag must not be swallowed as the first flag's value;
    # it must be re-evaluated as its own flag, so the real secret that
    # follows it is still redacted.
    args = ["--api-key", "--token", "sk-456"]
    result = redact_args(args)
    assert result == ["--api-key", "--token", REDACTED]
    assert "sk-456" not in result
