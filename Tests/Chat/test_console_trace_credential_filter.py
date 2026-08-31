"""Mandatory credential filtering before semantic trace persistence."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
import json

from tldw_chatbook.Chat.console_trace_redaction import (
    CREDENTIAL_FILTER_VERSION,
    CREDENTIAL_SANITIZER_UNAVAILABLE,
    CredentialSanitizer,
)


CANARIES = (
    "sk-live-abcdefghijklmnop",
    "Bearer eyJhbGciOiJIUzI1NiJ9.super-secret.signature",
    "nested-password-value",
    "url-password-value",
    "query-token-value",
)


def _serialized(result: object) -> str:
    value = asdict(result) if is_dataclass(result) else result
    return json.dumps(value, sort_keys=True, default=repr)


def test_sanitizer_removes_nested_credential_fields_urls_and_free_text() -> None:
    result = CredentialSanitizer().sanitize(
        {
            "api_key": CANARIES[0],
            "nested": {
                "password": CANARIES[2],
                "safe": "keep",
                "endpoint": (
                    "https://alice:url-password-value@example.invalid/v1"
                    "?api_key=query-token-value#fragment"
                ),
            },
            "diagnostic": f"Authorization: {CANARIES[1]}",
        }
    )

    assert result.available is True
    assert result.redacted is True
    assert result.omission_reason_code is None
    assert result.value == {
        "nested": {
            "safe": "keep",
            "endpoint": "https://example.invalid/v1",
        },
        "diagnostic": "[credential omitted]",
    }
    serialized = _serialized(result)
    assert all(canary not in serialized for canary in CANARIES)
    assert result.detector_version == CREDENTIAL_FILTER_VERSION


def test_sanitizer_drops_compound_credential_field_names() -> None:
    canaries = {
        "token_value": "opaque-token-canary",
        "password_value": "opaque-password-canary",
        "api_token_value": "opaque-api-token-canary",
        "refresh_token_metadata": "opaque-refresh-token-canary",
    }

    result = CredentialSanitizer().sanitize({**canaries, "safe_value": "keep"})

    assert result.available is True
    assert result.redacted is True
    assert result.value == {"safe_value": "keep"}
    assert all(canary not in _serialized(result) for canary in canaries.values())


def test_sanitizer_does_not_treat_credential_substrings_as_components() -> None:
    result = CredentialSanitizer().sanitize(
        {"tokenizer_value": "keep", "secretary_note": "keep too"}
    )

    assert result.available is True
    assert result.redacted is False
    assert result.value == {
        "tokenizer_value": "keep",
        "secretary_note": "keep too",
    }


def test_sanitizer_failure_returns_only_content_free_marker() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    result = CredentialSanitizer().sanitize(recursive)

    assert result.available is False
    assert result.redacted is False
    assert result.value is None
    assert result.omission_reason_code == CREDENTIAL_SANITIZER_UNAVAILABLE
    representation = repr(result)
    assert "recursive" not in representation.lower()
    assert all(canary not in representation for canary in CANARIES)


def test_sanitization_result_hides_value_from_repr() -> None:
    result = CredentialSanitizer().sanitize("ordinary retained content")

    assert result.value == "ordinary retained content"
    assert "ordinary retained content" not in repr(result)
    value_field = next(item for item in fields(result) if item.name == "value")
    assert value_field.repr is False


def test_short_known_credential_does_not_corrupt_unrelated_structure() -> None:
    credential = "abc1234"
    result = CredentialSanitizer(known_credentials=(credential,)).sanitize(
        {
            "kind": "kind",
            "authorization": f"Bearer {credential}",
            "note": f"token {credential}",
            "embedded": f"prefix{credential}suffix",
            f"prefix{credential}suffix": "drop the credential-bearing key",
        }
    )

    assert result.available is True
    assert result.value == {
        "kind": "kind",
        "note": "token [credential omitted]",
        "embedded": "prefix[credential omitted]suffix",
        "[credential omitted]": "drop the credential-bearing key",
    }
    assert result.redacted is True


def test_known_credential_key_placeholder_collision_fails_closed() -> None:
    credential = "abc1234"
    result = CredentialSanitizer(known_credentials=(credential,)).sanitize(
        {
            f"first{credential}": "one",
            f"second{credential}": "two",
        }
    )

    assert result.available is False
    assert result.value is None
    assert result.omission_reason_code == CREDENTIAL_SANITIZER_UNAVAILABLE


def test_sanitizer_rejects_non_json_finite_numbers_as_content_free_failure() -> None:
    result = CredentialSanitizer().sanitize({"temperature": float("nan")})

    assert result.available is False
    assert result.value is None
    assert result.omission_reason_code == CREDENTIAL_SANITIZER_UNAVAILABLE


def test_sanitizer_never_exposes_secret_through_failure_or_diagnostic_text() -> None:
    class Explosive:
        def __repr__(self) -> str:
            raise RuntimeError(CANARIES[0])

    result = CredentialSanitizer().sanitize({"safe": Explosive()})

    assert result.available is False
    assert _serialized(result) == (
        '{"available": false, "detector_version": "credentials-v1", '
        '"omission_reason_code": "credential_sanitizer_unavailable", '
        '"redacted": false, "value": null}'
    )
    assert all(canary not in _serialized(result) for canary in CANARIES)


def test_sanitizer_removes_resolved_credentials_from_arbitrary_nested_text() -> None:
    resolved = "provider-secret-with-no-recognized-shape"

    result = CredentialSanitizer(known_credentials=(resolved,)).sanitize(
        {
            "safe": [
                f"prefix {resolved} suffix",
                {"endpoint": f"https://example.invalid/v1#{resolved}"},
            ]
        }
    )

    assert result.available is True
    assert result.value == {
        "safe": [
            "prefix [credential omitted] suffix",
            {"endpoint": "https://example.invalid/v1"},
        ]
    }
    assert resolved not in _serialized(result)


def test_sanitizer_known_credentials_are_not_exposed_by_repr() -> None:
    resolved = "repr-must-not-retain-this-secret"

    sanitizer = CredentialSanitizer(known_credentials=(resolved,))

    assert resolved not in repr(sanitizer)


def test_sanitizer_marks_unchanged_content_complete() -> None:
    result = CredentialSanitizer().sanitize({"safe": ["unchanged", 1, True]})

    assert result.available is True
    assert result.redacted is False


def test_sanitizer_removes_cookie_private_key_and_credential_aliases() -> None:
    result = CredentialSanitizer().sanitize(
        {
            "headers": {"Set-Cookie": "session=COOKIE-CANARY; Secure"},
            "proxy_authorization": "Basic AUTH-CANARY",
            "database_password": "PASSWORD-CANARY",
            "private": (
                "prefix -----BEGIN PRIVATE KEY-----\nPRIVATE-KEY-CANARY\n"
                "-----END PRIVATE KEY----- suffix"
            ),
            "diagnostic": (
                "request Cookie: session=COOKIE-TEXT-CANARY at "
                "https://user:URL-CANARY@example.invalid/v1?token=QUERY-CANARY#frag"
            ),
        }
    )

    assert result.available is True
    assert result.redacted is True
    serialized = _serialized(result)
    for canary in (
        "COOKIE-CANARY",
        "AUTH-CANARY",
        "PASSWORD-CANARY",
        "PRIVATE-KEY-CANARY",
        "COOKIE-TEXT-CANARY",
        "URL-CANARY",
        "QUERY-CANARY",
    ):
        assert canary not in serialized


def test_sanitizer_reuses_recognized_secret_formats_and_sanitizes_mapping_keys() -> (
    None
):
    recognized = (
        "ghp_" + "A" * 30,
        "hf_" + "B" * 30,
        "AIza" + "C" * 35,
        "AKIA" + "D" * 16,
        "xoxb-" + "E" * 20,
        "eyJheader12.eyJpayload1.eyJsignature1",
    )
    credential_url_key = "socks5://user:password@example.invalid:1080/path?token=x#frag"

    result = CredentialSanitizer().sanitize(
        {
            credential_url_key: "safe",
            "provider_text": " ".join(recognized),
        }
    )

    assert result.available is True
    assert result.value == {
        "socks5://example.invalid:1080/path": "safe",
        "provider_text": " ".join("[credential omitted]" for _ in recognized),
    }
    serialized = _serialized(result)
    assert all(canary not in serialized for canary in recognized)
    assert "user:password" not in serialized


def test_sanitized_mapping_key_collision_fails_content_free() -> None:
    result = CredentialSanitizer().sanitize(
        {
            "https://host/path?first=one": "a",
            "https://host/path?second=two": "b",
        }
    )

    assert result.available is False
    assert result.value is None
    assert result.omission_reason_code == CREDENTIAL_SANITIZER_UNAVAILABLE


def test_sanitizer_bounds_dense_structural_work_without_partial_fallback() -> None:
    sanitizer = CredentialSanitizer(max_nodes=8, max_depth=3, max_text_codepoints=32)

    too_many_nodes = sanitizer.sanitize(list(range(9)))
    too_deep = sanitizer.sanitize([[[["safe"]]]])
    too_much_text = sanitizer.sanitize("x" * 33)

    for result in (too_many_nodes, too_deep, too_much_text):
        assert result.available is False
        assert result.value is None
        assert result.omission_reason_code == CREDENTIAL_SANITIZER_UNAVAILABLE


def test_sanitizer_bounds_are_validated_without_retaining_input() -> None:
    for kwargs in (
        {"max_nodes": 0},
        {"max_depth": 0},
        {"max_text_codepoints": 0},
    ):
        try:
            CredentialSanitizer(**kwargs)
        except ValueError as exc:
            assert "secret-canary" not in str(exc)
        else:  # pragma: no cover - the constructor must reject every case
            raise AssertionError("invalid sanitizer limit accepted")
