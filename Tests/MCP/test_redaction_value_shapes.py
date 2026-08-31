"""Redaction must catch secrets by VALUE SHAPE, not only by key name.

TASK-26009. Before this, `MCP/redaction.py` matched on key names, CLI flag
names and URL query parameter names. A secret sitting under an innocuous key --
`{"note": "sk-live-..."}` -- reached the approval card and any display boundary
untouched. The module also documented its own bypass: a secret value beginning
with "-" was re-read as a new CLI flag and appended unredacted.

The false-positive tests matter as much as the positives: over-redaction makes
an approval card useless for deciding whether to approve.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.MCP.redaction import (
    REDACTED,
    redact_args,
    redact_mapping,
)


SECRET_VALUES = [
    pytest.param("sk-abcdefghijklmnopqrstuvwxyz012345", id="openai-style"),
    pytest.param("sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123", id="anthropic-style"),
    pytest.param("ghp_abcdefghijklmnopqrstuvwxyz0123456789", id="github-pat"),
    pytest.param("github_pat_11ABCDEFG0abcdefghijklmnop", id="github-fine-grained"),
    pytest.param("xoxb-EXAMPLE-not-a-real-token", id="slack-bot"),
    pytest.param("AKIAIOSFODNN7EXAMPLE", id="aws-access-key-id"),
    pytest.param("AIzaSyD-abcdefghijklmnopqrstuvwxyz01234567", id="google-api-key"),
    pytest.param(
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk",
        id="jwt",
    ),
    pytest.param(
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
        id="pem-private-key",
    ),
    pytest.param("postgres://dbuser:hunter2@db.internal:5432/app", id="postgres-uri"),
    pytest.param("mongodb://admin:s3cr3t@10.0.0.5:27017/prod", id="mongodb-uri"),
    pytest.param("Bearer abcdefghijklmnopqrstuvwxyz0123456789", id="bearer-header"),
    # Review M3: AC#2 says "PEM private-key blocks"; the ` BLOCK` suffix used by
    # PGP defeated an anchor that assumed the header ends at `-----`.
    pytest.param(
        "-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYBF...\n-----END PGP PRIVATE KEY BLOCK-----",
        id="pgp-private-key",
    ),
    pytest.param("glpat-abcdefghijklmnopqrst", id="gitlab-pat"),
    pytest.param("xapp-1-A0123456789-abcdefghijk", id="slack-app-token"),
    pytest.param("sk_test_EXAMPLEnotreal01", id="stripe-secret-underscore"),
]

INNOCENT_VALUES = [
    # Review C1: the first version of these patterns destroyed all of these on
    # the model's main MCP input path. `basic ` + any word matched the auth
    # header pattern; any `sk-`/`pk-`/`rk-` kebab path component matched the
    # provider-key pattern. Tool results came back as "***" with no signal.
    pytest.param("The server supports basic authentication and OAuth.", id="prose-basic"),
    pytest.param("basic http-server configuration", id="prose-basic-hyphen"),
    pytest.param("/repo/pkgs/sk-core-runtime-helpers/index.ts", id="path-sk-component"),
    pytest.param("/home/user/sk-notes-archive-2026/readme.md", id="path-sk-dated"),
    pytest.param("pk-config-defaults-2026.json", id="filename-pk"),
    pytest.param("rk-means-clustering-explained", id="slug-rk"),
    pytest.param("the quick brown fox jumps over the lazy dog", id="prose"),
    pytest.param("/Users/someone/Documents/GitHub/project/src/main.py", id="abs-path"),
    pytest.param("./relative/path/to/file.txt", id="rel-path"),
    pytest.param("a1b2c3d4e5f60718293a4b5c6d7e8f9012345678", id="git-sha-40"),
    pytest.param("a1b2c3d", id="git-sha-short"),
    pytest.param("https://example.com/docs/page?section=intro", id="plain-url"),
    pytest.param("2026-08-31T12:00:00Z", id="timestamp"),
    pytest.param("v1.2.3-rc4", id="version-string"),
    pytest.param("SELECT id FROM users WHERE active = true", id="sql"),
    pytest.param("", id="empty"),
]


@pytest.mark.parametrize("secret", SECRET_VALUES)
def test_secret_shaped_value_is_redacted_under_an_innocent_key(secret):
    """AC#1/#2: the key name must not be what saves us."""
    result = redact_mapping({"note": secret})

    assert result["note"] == REDACTED


@pytest.mark.parametrize("secret", SECRET_VALUES)
def test_secret_shaped_value_is_redacted_when_nested(secret):
    result = redact_mapping({"outer": {"inner": {"harmless_name": secret}}})

    assert result["outer"]["inner"]["harmless_name"] == REDACTED


@pytest.mark.parametrize("secret", SECRET_VALUES)
def test_secret_shaped_value_is_redacted_inside_a_list(secret):
    result = redact_mapping({"items": [secret]})

    assert result["items"][0] == REDACTED


@pytest.mark.parametrize("innocent", INNOCENT_VALUES)
def test_ordinary_values_are_left_alone(innocent):
    """AC#5: over-redaction makes an approval card useless."""
    result = redact_mapping({"note": innocent})

    assert result["note"] == innocent


def test_dash_prefixed_secret_after_a_flag_is_redacted():
    """AC#3: the documented bypass at the old redaction.py:64."""
    result = redact_args(["--api-key", "-9f3a2b1c8d7e6f5a4b3c2d1e"])

    assert REDACTED in result
    assert "-9f3a2b1c8d7e6f5a4b3c2d1e" not in result


def test_a_real_flag_after_a_secret_flag_is_not_swallowed():
    """The bypass existed to protect this case -- it must keep working."""
    result = redact_args(["--api-key", "--verbose"])

    assert "--verbose" in result


def test_short_flag_after_a_secret_flag_is_not_swallowed():
    result = redact_args(["--token", "-v"])

    assert "-v" in result


def test_secret_shaped_bare_arg_is_redacted():
    """A secret can arrive as a positional arg with no flag at all."""
    result = redact_args(["--serve", "ghp_abcdefghijklmnopqrstuvwxyz0123456789"])

    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in result


def test_redaction_has_no_io_or_config_dependency():
    """AC#6: pure function. Importing must not touch config or the network."""
    import tldw_chatbook.MCP.redaction as redaction_module

    # Assert on imports, not on the substring "open(" -- that tripped on
    # `.open(`, `reopen(` and the word in a comment, and passed identically
    # against the pre-change module, so it pinned nothing.
    source = open(redaction_module.__file__, encoding="utf-8").read()
    for forbidden in (
        "import requests",
        "import httpx",
        "import socket",
        "get_cli_setting",
    ):
        assert forbidden not in source, f"redaction must stay pure: found {forbidden}"
    # And it holds no module-level state that could vary between calls.
    assert redaction_module.redact_mapping({"a": "b"}) == {"a": "b"}


# --- AC#4: both boundaries, verified separately -----------------------------


def test_approval_card_does_not_render_a_shape_matched_secret():
    """Display path. The card is where a human reads tool arguments."""
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _summarize_arguments,
    )

    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    rendered = _summarize_arguments({"description": secret, "file_path": "/tmp/x"})

    assert secret not in rendered
    assert REDACTED in rendered
    # The non-secret argument must survive, or the card stops being useful.
    assert "/tmp/x" in rendered


def test_execution_log_never_stores_argument_values_at_all():
    """Stored path.

    The execution log needs no redaction because it stores argument NAMES and
    never values -- `build_record` documents itself as metadata-only and routes
    every field through `safe_metadata_token`. That is a stronger guarantee than
    redaction, so this pins the exclusion rather than asserting a redact_* call
    that would be dead code.
    """
    from dataclasses import asdict

    from tldw_chatbook.MCP.execution_log import build_record

    secret = "sk-abcdefghijklmnopqrstuvwxyz012345"
    record = build_record(
        server_key="local:demo",
        tool_name="write_file",
        initiator="chat",
        ok=True,
        duration_ms=5,
        arguments={"api_key": secret, "note": secret, "path": "/tmp/x"},
        registered_argument_names={"api_key", "note", "path"},
        result={"written": True},
    )

    serialized = repr(asdict(record))
    assert secret not in serialized
    assert "/tmp/x" not in serialized, "the log must not carry values of any kind"
    # Names are metadata and are expected to survive.
    assert "path" in serialized


# --- review follow-ups: I5, I6, M1, M3-bytes --------------------------------


def test_url_query_value_is_redacted_by_shape():
    """AC#1 covers argument names too -- redact_url matched key names only."""
    from tldw_chatbook.MCP.redaction import redact_url

    out = redact_url("https://h/x?note=sk-live-abcdefghijklmnopqrstuv")

    assert "sk-live-abcdefghijklmnopqrstuv" not in out


def test_url_userinfo_credentials_are_redacted():
    """The connection-URI pattern existed but was unreachable from redact_url."""
    from tldw_chatbook.MCP.redaction import redact_url

    out = redact_url("https://user:hunter2@host/x?a=1")

    assert "hunter2" not in out
    assert "host" in out, "only the credentials should go"


def test_plain_url_is_untouched():
    from tldw_chatbook.MCP.redaction import redact_url

    url = "https://example.com/docs/page?section=intro"

    assert redact_url(url) == url


@pytest.mark.parametrize(
    "token",
    ["-Xy9_abcdefghijklmnopqrs", "-abc-def-ghi-jkl-mno-pqrs", "-9f3a2b1c8d7e6f5a"],
)
def test_long_dash_prefixed_value_after_a_secret_flag_is_redacted(token):
    """I5: _PLAUSIBLE_FLAG_RE accepted letter-leading tokens, so `--api-key
    -Xy9_...` still leaked. Real flags are short; credentials are not."""
    result = redact_args(["--api-key", token])

    assert token not in result


def test_flag_with_inline_value_after_a_secret_flag_survives():
    """M1: `--out=file.txt` is a flag, not the api-key's value."""
    result = redact_args(["--api-key", "--out=file.txt"])

    assert "--out=file.txt" in result


def test_bytes_secret_in_a_sequence_is_redacted():
    """M3: sequence items skipped the shape check for non-str scalars."""
    result = redact_mapping({"items": [b"ghp_abcdefghijklmnopqrstuvwxyz0123456789"]})

    assert result["items"][0] == REDACTED
