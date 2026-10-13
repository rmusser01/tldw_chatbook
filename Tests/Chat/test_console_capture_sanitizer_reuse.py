"""Exchange capture sanitizes streamed chunks with a reused per-flight sanitizer."""

import pytest

from tldw_chatbook.Chat import console_exchange_capture as capture_module
from tldw_chatbook.Chat.console_exchange_capture import (
    _remove_nested_credentials,
    sanitize_capture_value_with_omission,
    stub_binary_strings,
)
from tldw_chatbook.Chat.console_provider_gateway import _capture_sanitizer_for
from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer

pytestmark = pytest.mark.bootstrap_profile

SHORT_TEXTS = [
    "Hello there",
    'He said "hi" \\ back',
    "héllo — 🚀 mix\nnewline\ttab",
    "",
    "sk-secret987654321",
    "data:text/plain,aGVsbG8=",
]
LONG_TEXT = "x" * 5000

CREDENTIALS = ("sk-secret987654321",)


def _slow_path(value, *, credentials=CREDENTIALS):
    result = CredentialSanitizer(known_credentials=credentials).sanitize(value)
    if not result.available:
        return dict(capture_module._CREDENTIAL_OMISSION), True
    return (
        stub_binary_strings(_remove_nested_credentials(result.value)),
        result.redacted,
    )


@pytest.mark.parametrize(
    "text", SHORT_TEXTS + [LONG_TEXT, {"content": ["a", "sk-secret987654321"]}]
)
def test_fast_path_matches_the_composed_slow_path(text):
    assert sanitize_capture_value_with_omission(
        text, known_credentials=CREDENTIALS
    ) == _slow_path(text)


def test_known_credentials_still_redact_through_the_fast_path():
    value, redacted = sanitize_capture_value_with_omission(
        "token sk-secret987654321 leaked", known_credentials=CREDENTIALS
    )
    assert redacted is True
    assert "sk-secret987654321" not in str(value)


def test_reused_sanitizer_avoids_per_chunk_construction(monkeypatch):
    constructed = []
    real = CredentialSanitizer

    def counting(**kwargs):
        constructed.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(capture_module, "CredentialSanitizer", counting)
    flight_sanitizer = real(known_credentials=CREDENTIALS)

    for chunk in ("one ", "two ", "three"):
        sanitize_capture_value_with_omission(
            chunk, known_credentials=CREDENTIALS, sanitizer=flight_sanitizer
        )
    assert not constructed, "a provided sanitizer must never rebuild"

    sanitize_capture_value_with_omission("fallback", known_credentials=CREDENTIALS)
    assert len(constructed) == 1, "no sanitizer given: construct exactly once"


def test_flight_sanitizer_helper_returns_none_on_invalid_credentials():
    # Over the bounded detector budget: construction must raise, and the
    # helper must swallow it so the per-chunk path keeps its old fallback.
    from tldw_chatbook.Chat.console_trace_redaction import (
        DEFAULT_CREDENTIAL_MAX_TEXT_CODEPOINTS,
    )

    oversized = "x" * (DEFAULT_CREDENTIAL_MAX_TEXT_CODEPOINTS + 1)
    assert _capture_sanitizer_for((oversized,)) is None
