from __future__ import annotations

import pytest

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
    for key in (
        "api_key",
        "API-KEY",
        "Authorization",
        "token",
        "client_secret",
        "PASSWORD",
    ):
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
    assert redact_args(args) == [
        "--api-key",
        REDACTED,
        "--verbose",
        f"token={REDACTED}",
        "plain",
    ]


def test_redact_url_strips_secret_query_values():
    url = "https://api.example.com/v1?api_key=sk-123&page=2"
    redacted = redact_url(url)
    assert "sk-123" not in redacted
    assert "page=2" in redacted


@given(
    st.dictionaries(st.text(min_size=1, max_size=20), st.text(max_size=40), max_size=8)
)
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


def test_redact_args_redacts_inline_flag_with_leading_dashes():
    # `--api-key=sk-123` is the inline `key=value` form, not the two-token
    # `--api-key sk-123` form -- `key` is "--api-key" (the arg charset
    # includes "-"), which `is_secret_key` still matches via the "api-key"
    # substring, so this is redacted by the inline branch.
    assert redact_args(["--api-key=sk-123"]) == ["--api-key=***"]


def test_redact_args_leaves_non_secret_short_inline_flag_unchanged():
    assert redact_args(["-k=v"]) == ["-k=v"]


def test_redact_args_reevaluates_flag_after_consecutive_secret_flags():
    # The second secret flag must not be swallowed as the first flag's value;
    # it must be re-evaluated as its own flag, so the real secret that
    # follows it is still redacted.
    args = ["--api-key", "--token", "sk-456"]
    result = redact_args(args)
    assert result == ["--api-key", "--token", REDACTED]
    assert "sk-456" not in result


@pytest.mark.parametrize(
    "url",
    [
        "https://example.test/long/path?api_key=synthetic-credential&page=2",
        "postgres://database.test/catalog?password=synthetic-credential&page=2",
        "file:/folder/document?token=synthetic-credential&page=2",
        "//example.test/document?api_key=synthetic-credential&page=2",
        " \t\nhttps://example.test/document?api_key=synthetic-credential&page=2",
        "\x00 //example.test/document?api_key=synthetic-credential&page=2",
    ],
)
def test_mapping_recursively_redacts_url_query_credentials_without_mutation(url):
    data = {"url": url, "nested": {"items": [url, ({"uri": url},)]}}
    redacted = redact_mapping(data)
    values = (
        redacted["url"],
        redacted["nested"]["items"][0],
        redacted["nested"]["items"][1][0]["uri"],
    )
    assert all("synthetic-credential" not in value for value in values)
    assert all("page=2" in value for value in values)
    assert all("%2A%2A%2A" in value for value in values)
    assert data == {"url": url, "nested": {"items": [url, ({"uri": url},)]}}
    assert isinstance(redacted["nested"]["items"][1], tuple)


@pytest.mark.parametrize(
    "value",
    [
        "https://example.test/docs?q=one%20two&path=%2fnotes&flag#section",
        "file:/folder/document?section=one%20two",
        " \t\nhttps://example.test/docs?q=one%20two&flag#section",
        "C:/folder/document.txt",
        "/repo/long-path/document.txt?draft=one%20two",
        "See https://example.test/docs?q=one%20two for details.",
    ],
)
def test_display_redaction_preserves_nonsecret_urls_paths_and_prose(value):
    assert redact_mapping({"value": value}) == {"value": value}
    assert redact_url(value) == value


@pytest.mark.parametrize(
    "url",
    [
        "https://[invalid/document?api_key=synthetic-credential",
        "https://exa\uff0fmple.test/document?api_key=synthetic-credential",
    ],
)
def test_malformed_url_is_safe_to_display_without_query_credentials(url):
    assert redact_url(url) == REDACTED
    assert redact_mapping({"uri": url}) == {"uri": REDACTED}
