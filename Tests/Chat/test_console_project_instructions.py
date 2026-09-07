"""Console project-instruction control-state and identity contracts."""

from __future__ import annotations

import inspect
import json

import pytest

from tldw_chatbook.Chat.console_project_instructions import (
    EPHEMERAL_ORIGIN_KEY,
    LOCATOR_FINGERPRINT_DOMAIN,
    NOTICE_KEY_FINGERPRINT_DOMAIN,
    PROJECT_CONTEXT_VERSION,
    PROVIDER_DESTINATION_FINGERPRINT_DOMAIN,
    ProjectInstructionControlState,
    decode_project_context_json,
    encode_project_context_json,
    fingerprint_canonical_locator,
    fingerprint_provider_destination,
    project_instruction_notice_key,
    sanitized_destination_label,
)


def test_new_session_explicitly_enables_project_instructions() -> None:
    assert EPHEMERAL_ORIGIN_KEY == "_chatbook_ephemeral_origin"
    assert ProjectInstructionControlState.new_session() == (
        ProjectInstructionControlState(project_instructions_enabled=True)
    )


@pytest.mark.parametrize(
    "raw_state",
    [
        None,
        "",
        "not-json",
        "null",
        "[]",
        "{}",
        '{"version": 2}',
        '{"version": true}',
        '{"version": 1}',
        json.dumps(
            {
                "version": 1,
                "project_instructions_enabled": 1,
                "working_folder_binding_id": None,
                "working_folder_locator_fingerprint": None,
                "project_instruction_notice_key": None,
            }
        ),
        json.dumps(
            {
                "version": 1,
                "project_instructions_enabled": True,
                "working_folder_binding_id": 12,
                "working_folder_locator_fingerprint": None,
                "project_instruction_notice_key": None,
            }
        ),
    ],
)
def test_untrusted_or_legacy_state_fails_closed(raw_state: str | None) -> None:
    assert decode_project_context_json(raw_state) == (
        ProjectInstructionControlState.legacy_disabled()
    )


@pytest.mark.parametrize(
    ("duplicate_key", "duplicate_value"),
    [
        ("version", 1),
        ("project_instructions_enabled", True),
        ("working_folder_binding_id", "binding-7"),
        ("working_folder_locator_fingerprint", "locator-fingerprint"),
        ("project_instruction_notice_key", "notice-key"),
    ],
)
def test_duplicate_json_keys_fail_closed(
    duplicate_key: str, duplicate_value: object
) -> None:
    pairs = [
        ("version", 1),
        ("project_instructions_enabled", True),
        ("working_folder_binding_id", "binding-7"),
        ("working_folder_locator_fingerprint", "locator-fingerprint"),
        ("project_instruction_notice_key", "notice-key"),
        (duplicate_key, duplicate_value),
    ]
    raw_state = (
        "{"
        + ",".join(f"{json.dumps(key)}:{json.dumps(value)}" for key, value in pairs)
        + "}"
    )

    assert decode_project_context_json(raw_state) == (
        ProjectInstructionControlState.legacy_disabled()
    )


def test_control_state_round_trips_only_the_version_and_four_control_fields() -> None:
    state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-7",
        working_folder_locator_fingerprint="locator-fingerprint",
        project_instruction_notice_key="notice-key",
    )

    encoded = encode_project_context_json(state)

    assert json.loads(encoded) == {
        "version": PROJECT_CONTEXT_VERSION,
        "project_instructions_enabled": True,
        "working_folder_binding_id": "binding-7",
        "working_folder_locator_fingerprint": "locator-fingerprint",
        "project_instruction_notice_key": "notice-key",
    }
    assert decode_project_context_json(encoded) == state


def test_unknown_or_sensitive_fields_are_not_preserved_on_reencode() -> None:
    raw_values = {
        "version": 1,
        "project_instructions_enabled": True,
        "working_folder_binding_id": "binding-7",
        "working_folder_locator_fingerprint": "opaque-locator-fingerprint",
        "project_instruction_notice_key": "opaque-notice-key",
        "locator": "file:///Users/alice/private/repo",
        "source_path": "secret/AGENTS.md",
        "digest": "raw-instruction-digest",
        "endpoint": "https://user:password@example.test/private/v1",
        "body": "private instruction body",
    }

    decoded = decode_project_context_json(json.dumps(raw_values))
    reencoded = encode_project_context_json(decoded)

    assert decoded == ProjectInstructionControlState.legacy_disabled()
    assert set(json.loads(reencoded)) == {
        "version",
        "project_instructions_enabled",
        "working_folder_binding_id",
        "working_folder_locator_fingerprint",
        "project_instruction_notice_key",
    }
    for sensitive_value in (
        raw_values["locator"],
        raw_values["source_path"],
        raw_values["digest"],
        raw_values["endpoint"],
        raw_values["body"],
    ):
        assert sensitive_value not in reencoded


def test_fingerprint_protocol_domains_and_outputs_are_pinned() -> None:
    assert LOCATOR_FINGERPRINT_DOMAIN == (
        b"tldw_chatbook.console.project-instructions.locator.v1\0"
    )
    assert PROVIDER_DESTINATION_FINGERPRINT_DOMAIN == (
        b"tldw_chatbook.console.project-instructions.provider-destination.v1\0"
    )
    assert NOTICE_KEY_FINGERPRINT_DOMAIN == (
        b"tldw_chatbook.console.project-instructions.notice-key.v1\0"
    )
    locator_fingerprint = fingerprint_canonical_locator("file:///Users/alice/work/repo")
    assert locator_fingerprint == (
        "221fa1a5f342123e6dda9f409b35146dacfaf96978c64832130f0df73ecf7c1b"
    )
    destination_fingerprint = fingerprint_provider_destination(
        "OpenAI",
        "HTTPS://user:secret@API.Example.COM:443/v1/?api_key=secret#fragment",
    )
    assert destination_fingerprint == (
        "a7fd7f4ef42fa29cc07e2e712cce210bc427ea5e293384933c8feb96d13030b4"
    )
    assert (
        fingerprint_provider_destination("openai", "https://api.example.com/v1")
        == destination_fingerprint
    )
    assert (
        project_instruction_notice_key(
            locator_fingerprint,
            "OpenAI",
            "HTTPS://user:secret@API.Example.COM:443/v1/?api_key=secret#fragment",
        )
        == "ef9c37589f7117d0647e6fb350448e68725900920664d9425b9700d156f07e1c"
    )


def test_notice_key_tracks_provider_destination_but_not_model() -> None:
    locator_fingerprint = fingerprint_canonical_locator("file:///repo")
    endpoint = "https://api.example.test/v1"

    baseline = project_instruction_notice_key(locator_fingerprint, "openai", endpoint)
    assert "model" not in inspect.signature(project_instruction_notice_key).parameters
    assert project_instruction_notice_key(
        locator_fingerprint, "openai", endpoint
    ) == project_instruction_notice_key(locator_fingerprint, "openai", endpoint)
    assert (
        project_instruction_notice_key(locator_fingerprint, "anthropic", endpoint)
        != baseline
    )
    assert (
        project_instruction_notice_key(
            locator_fingerprint, "openai", "https://other.example.test/v1"
        )
        != baseline
    )
    assert (
        project_instruction_notice_key(
            locator_fingerprint, "openai", "https://api.example.test/v2"
        )
        != baseline
    )


def test_destination_label_shows_only_provider_and_custom_endpoint_origin() -> None:
    raw_endpoint = (
        "https://user:password@API.Example.test:8443/private/v1?api_key=secret#fragment"
    )

    label = sanitized_destination_label("OpenAI", raw_endpoint)

    assert label == "OpenAI (https://api.example.test:8443)"
    for secret_or_path in ("user", "password", "private", "api_key", "secret"):
        assert secret_or_path not in label
    assert sanitized_destination_label("OpenAI", None) == "OpenAI"


@pytest.mark.parametrize("control_character", ["\0", "\x1f", "\x7f"])
def test_destination_label_rejects_control_characters(
    control_character: str,
) -> None:
    endpoint = f"https://api.example{control_character}.test/private"

    assert sanitized_destination_label("OpenAI", endpoint) == (
        "OpenAI (invalid endpoint)"
    )


@pytest.mark.parametrize(
    "unsafe_endpoint",
    [
        "https://api.example.test\\@evil.test/private",
        "https://api.example\u202e.test/private",
        "https://api.example\u2066.test/private",
        "https://api.example\ue000.test/private",
        "https://api.example.test/private\n",
    ],
)
def test_destination_label_rejects_backslashes_and_bidi_controls(
    unsafe_endpoint: str,
) -> None:
    assert sanitized_destination_label("OpenAI", unsafe_endpoint) == (
        "OpenAI (invalid endpoint)"
    )
    with pytest.raises(ValueError, match="invalid provider endpoint"):
        fingerprint_provider_destination("openai", unsafe_endpoint)


@pytest.mark.parametrize(
    ("unicode_host", "punycode_host"),
    [
        ("münich.example", "xn--mnich-kva.example"),
        ("faß.de", "xn--fa-hia.de"),
    ],
)
def test_unicode_and_punycode_hosts_share_one_destination_identity(
    unicode_host: str,
    punycode_host: str,
) -> None:
    unicode_endpoint = f"https://{unicode_host}/v1"
    punycode_endpoint = f"https://{punycode_host}/v1"

    assert fingerprint_provider_destination(
        "openai", unicode_endpoint
    ) == fingerprint_provider_destination("openai", punycode_endpoint)
    assert sanitized_destination_label("OpenAI", unicode_endpoint) == (
        f"OpenAI (https://{punycode_host})"
    )
    assert sanitized_destination_label("OpenAI", punycode_endpoint) == (
        f"OpenAI (https://{punycode_host})"
    )


@pytest.mark.parametrize(
    ("endpoint", "expected_label"),
    [
        ("https://127.0.0.1:8443/v1", "OpenAI (https://127.0.0.1:8443)"),
        (
            "https://[2001:db8::1]:8443/v1",
            "OpenAI (https://[2001:db8::1]:8443)",
        ),
    ],
)
def test_destination_label_preserves_ip_address_hosts(
    endpoint: str, expected_label: str
) -> None:
    assert sanitized_destination_label("OpenAI", endpoint) == expected_label


def test_invalid_endpoints_cannot_produce_reusable_notice_keys() -> None:
    locator_fingerprint = fingerprint_canonical_locator("file:///repo")

    for malformed_endpoint in ("not a valid endpoint", "https://[broken"):
        with pytest.raises(ValueError, match="invalid provider endpoint"):
            project_instruction_notice_key(
                locator_fingerprint,
                "openai",
                malformed_endpoint,
            )


def test_blank_endpoint_is_the_valid_default_provider_identity() -> None:
    assert fingerprint_provider_destination(
        "openai", None
    ) == fingerprint_provider_destination("openai", "  ")
    assert sanitized_destination_label("OpenAI", "  ") == "OpenAI"
