from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from hypothesis import given, strategies as st

from tldw_chatbook.Prompt_Management.prompt_chatbook_record import (
    CHATBOOK_PROMPT_RECORD_KEYS,
    PromptChatbookRecordError,
    decode_chatbook_prompt_record,
    encode_chatbook_prompt_record,
)


def _detail(**overrides: Any) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "name": "Research summary",
        "author": "Ada",
        "details": "A two-lane Prompt",
        "system_prompt": "[bold]\n研究🙂",
        "user_prompt": "Summarize {{topic}}.\nمرحبا",
        "keywords": ["analysis", "research"],
        "artifact_type": "prompt",
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": '{"kind":"block_prompt","version":2}',
    }
    detail.update(overrides)
    return detail


def _record(**overrides: Any) -> dict[str, Any]:
    record = {
        "record_schema": "tldw-chatbook-prompt",
        "record_version": 1,
        **_detail(),
    }
    record.update(overrides)
    return record


def test_new_record_encode_and_decode_preserve_exact_portable_fields() -> None:
    detail = _detail()

    encoded = encode_chatbook_prompt_record(detail)

    assert tuple(encoded) == CHATBOOK_PROMPT_RECORD_KEYS
    assert encoded == _record()
    assert decode_chatbook_prompt_record(encoded) == detail


@pytest.mark.parametrize("value", [None, "", "line one\nline two", "研究🙂 مرحبا"])
@pytest.mark.parametrize(
    "field",
    ["author", "details", "system_prompt", "user_prompt", "prompt_definition"],
)
def test_new_record_preserves_nullable_text_without_coercion(
    field: str, value: str | None
) -> None:
    detail = _detail(**{field: value})

    encoded = encode_chatbook_prompt_record(detail)

    assert encoded[field] is value
    assert decode_chatbook_prompt_record(encoded)[field] is value


@pytest.mark.parametrize(
    ("artifact_type", "prompt_format", "schema_version"),
    [
        ("prompt", "legacy", None),
        ("prompt", "structured", 2),
        ("recipe", "legacy", None),
        ("recipe", "structured", 1),
        ("recipe", "structured", 999),
    ],
)
def test_new_record_preserves_supported_and_compatibility_only_metadata(
    artifact_type: str, prompt_format: str, schema_version: int | None
) -> None:
    detail = _detail(
        artifact_type=artifact_type,
        prompt_format=prompt_format,
        prompt_schema_version=schema_version,
        prompt_definition='{"foreign":true}' if schema_version else "not-json",
    )

    assert (
        decode_chatbook_prompt_record(encode_chatbook_prompt_record(detail)) == detail
    )


def test_legacy_record_maps_content_to_system_and_ignores_known_metadata() -> None:
    decoded = decode_chatbook_prompt_record(
        {
            "id": 42,
            "name": "Legacy",
            "description": "Old shape",
            "content": "System only",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": None,
        }
    )

    assert decoded == {
        "name": "Legacy",
        "author": None,
        "details": "Old shape",
        "system_prompt": "System only",
        "user_prompt": None,
        "keywords": [],
        "artifact_type": "prompt",
        "prompt_format": "legacy",
        "prompt_schema_version": None,
        "prompt_definition": None,
    }


def test_legacy_record_accepts_only_required_historical_fields() -> None:
    decoded = decode_chatbook_prompt_record(
        {"name": "Legacy", "description": None, "content": ""}
    )

    assert decoded["details"] is None
    assert decoded["system_prompt"] == ""


@pytest.mark.parametrize(
    "mutation",
    [
        {"record_schema": None},
        {"record_schema": "unknown"},
        {"record_version": None},
        {"record_version": True},
        {"record_version": 2},
        {"name": ""},
        {"name": "   "},
        {"name": None},
        {"author": 7},
        {"details": []},
        {"system_prompt": {}},
        {"user_prompt": False},
        {"keywords": "analysis"},
        {"keywords": ["analysis", 7]},
        {"artifact_type": "unknown"},
        {"prompt_format": "markdown"},
        {"prompt_schema_version": True},
        {"prompt_schema_version": "2"},
        {"prompt_definition": {}},
    ],
)
def test_new_record_rejects_invalid_fields(mutation: dict[str, Any]) -> None:
    payload = _record()
    payload.update(mutation)

    with pytest.raises(
        PromptChatbookRecordError, match="Invalid Chatbook Prompt record"
    ):
        decode_chatbook_prompt_record(payload)


@pytest.mark.parametrize("missing", CHATBOOK_PROMPT_RECORD_KEYS)
def test_new_record_requires_every_exact_field(missing: str) -> None:
    payload = _record()
    del payload[missing]

    with pytest.raises(PromptChatbookRecordError):
        decode_chatbook_prompt_record(payload)


def test_new_record_rejects_unknown_extra_key() -> None:
    with pytest.raises(PromptChatbookRecordError):
        decode_chatbook_prompt_record({**_record(), "prompt_defintion": "typo"})


@pytest.mark.parametrize(
    "payload",
    [
        {"record_schema": "tldw-chatbook-prompt", **_detail()},
        {"record_version": 1, **_detail()},
        {**_record(), "content": "legacy"},
        {"name": "Legacy", "description": "Old", "content": "Body", "x": 1},
        {
            "name": "Legacy",
            "description": "Old",
            "content": "Body",
            "id": True,
        },
        {
            "name": "Legacy",
            "description": "Old",
            "content": "Body",
            "created_at": 1,
        },
        {"name": "Legacy", "description": "Old", "content": None},
    ],
)
def test_partial_mixed_or_invalid_legacy_records_fail_closed(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(PromptChatbookRecordError):
        decode_chatbook_prompt_record(payload)


def test_record_error_repr_and_message_never_include_payload_values() -> None:
    sentinel = "TASK197_PROMPT_BODY_MUST_NOT_LEAK"
    payload = _record(system_prompt=sentinel, record_version=999)

    with pytest.raises(PromptChatbookRecordError) as raised:
        decode_chatbook_prompt_record(payload)

    assert sentinel not in str(raised.value)
    assert sentinel not in repr(raised.value)
    assert raised.value.category == "version"


def test_encoder_rejects_missing_source_fields_without_mutating_input() -> None:
    detail = _detail()
    del detail["user_prompt"]
    before = deepcopy(detail)

    with pytest.raises(PromptChatbookRecordError):
        encode_chatbook_prompt_record(detail)

    assert detail == before


@given(st.text(max_size=256))
def test_round_trip_preserves_arbitrary_user_lane_text(value: str) -> None:
    detail = _detail(user_prompt=value)

    assert (
        decode_chatbook_prompt_record(encode_chatbook_prompt_record(detail)) == detail
    )


@given(
    st.one_of(
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.lists(st.text(), max_size=3),
        st.dictionaries(st.text(max_size=4), st.integers(), max_size=3),
    )
)
def test_decoder_rejects_non_string_user_lane_values(value: Any) -> None:
    payload = _record(user_prompt=value)

    with pytest.raises(PromptChatbookRecordError):
        decode_chatbook_prompt_record(payload)
