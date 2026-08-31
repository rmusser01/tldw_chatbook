"""Credential-boundary tests for agent-authored lesson content."""

from __future__ import annotations

import logging

import pytest

from tldw_chatbook.Notes.agent_lessons import (
    CREDENTIAL_REFUSAL_CODE,
    classify_lesson_credentials,
)


def _joined(*parts: str) -> str:
    """Construct synthetic credential-shaped fixtures without a literal live key."""

    return "".join(parts)


@pytest.mark.parametrize(
    "content",
    (
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n"
        "-----END PRIVATE KEY-----",
        "-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n"
        "-----END ENCRYPTED PRIVATE KEY-----",
        _joined("sk-", "proj-", "Q7mN2pL8vR4cT9xK6wH3jF5sD1aB0uE2yI7oP9zX"),
        _joined("sk-ant-", "api03-", "Q7mN2pL8vR4cT9xK6wH3jF5sD1aB0uE2yI7oP9zX"),
        _joined("ghp_", "Q7mN2pL8vR4cT9xK6wH3jF5sD1aB0uE2"),
        _joined(
            "sk-", "proj-", "Q7mN2pL8example9xK6wH3jF5sD1aB0uE2yI7oP9zX"
        ),
        'api_key = "Q7mN2pL8vR4cT9xK6wH3jF5sD1aB0uE2"',
        "client_secret: Q7mN2pL8vR4cT9xK6wH3jF5sD1aB0uE2",
        '- api_key = "Q7mN2pL8vR4cT9xK6wH3jF5sD1aB0uE2"',
        "export API_KEY=Q7mN2pL8vR4cT9xK6wH3jF5sD1aB0uE2",
        '"api_key": "Q7mN2pL8vR4cT9xK6wH3jF5sD1aB0uE2",',
    ),
)
def test_high_confidence_private_material_is_rejected_with_one_generic_code(
    content: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    result = classify_lesson_credentials(content)

    assert result.accepted is False
    assert result.reason_code == CREDENTIAL_REFUSAL_CODE
    assert content not in repr(result)
    assert content not in caplog.text


@pytest.mark.parametrize(
    "content",
    (
        "api_key = <redacted>",
        'api_key = "example-not-a-real-key"',
        "client_secret: FAKE_VALUE_FOR_DOCUMENTATION",
        _joined("sk-", "proj-", "example_not_a_real_key_12345678901234567890"),
        "-----BEGIN PRIVATE KEY-----\nREDACTED\n-----END PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----\nEXAMPLE PRIVATE KEY MATERIAL\n"
        "-----END PRIVATE KEY-----",
        "UUID 123e4567-e89b-12d3-a456-426614174000",
        "sha256 " + "a3" * 32,
        "Traceback (most recent call last):\n  File \"worker.py\", line 12\n"
        "ValueError: bad input",
        "Error ID ERR-NOTES-20260830-0123456789abcdef",
        "The credential assignment syntax is api_key=<placeholder>.",
    ),
)
def test_safe_diagnostics_and_explicit_placeholders_remain_accepted(
    content: str,
) -> None:
    result = classify_lesson_credentials(content)

    assert result.accepted is True
    assert result.reason_code is None


def test_validation_has_no_durable_or_logging_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import sqlite3

    sensitive = _joined(
        "authorization_token=", "Q7mN2pL8vR4cT9xK6wH3jF5sD1aB0uE2"
    )

    def fail_connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("credential validation must not open durable storage")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    caplog.set_level(logging.DEBUG)

    result = classify_lesson_credentials(sensitive)

    assert result.accepted is False
    assert result.reason_code == CREDENTIAL_REFUSAL_CODE
    assert sensitive not in repr(result)
    assert sensitive not in caplog.text
