"""Portable, versioned Chatbook records for local Prompt artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CHATBOOK_PROMPT_RECORD_SCHEMA = "tldw-chatbook-prompt"
CHATBOOK_PROMPT_RECORD_VERSION = 1
CHATBOOK_MIXIN_RECORD_VERSION = 2
MIXIN_MARKDOWN_HEADER = "# Chatbook mix-in"

_PORTABLE_FIELD_KEYS = (
    "name",
    "author",
    "details",
    "system_prompt",
    "user_prompt",
    "keywords",
    "artifact_type",
    "prompt_format",
    "prompt_schema_version",
    "prompt_definition",
)
CHATBOOK_PROMPT_RECORD_KEYS = (
    "record_schema",
    "record_version",
    *_PORTABLE_FIELD_KEYS,
)

_NULLABLE_TEXT_KEYS = (
    "author",
    "details",
    "system_prompt",
    "user_prompt",
    "prompt_definition",
)
_LEGACY_REQUIRED_KEYS = frozenset(("name", "description", "content"))
_LEGACY_OPTIONAL_KEYS = frozenset(("id", "created_at", "updated_at"))


class PromptChatbookRecordError(ValueError):
    """Report a bounded Prompt-record validation category.

    Attributes:
        category: Fixed non-payload classification of the validation failure.
    """

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__("Invalid Chatbook Prompt record.")

    def __repr__(self) -> str:
        return f"PromptChatbookRecordError(category={self.category!r})"


def _invalid(category: str) -> PromptChatbookRecordError:
    return PromptChatbookRecordError(category)


def _validate_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid("field")
    return value


def _validate_nullable_text(value: Any) -> str | None:
    if value is not None and not isinstance(value, str):
        raise _invalid("field")
    return value


def _validate_new_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != set(CHATBOOK_PROMPT_RECORD_KEYS):
        raise _invalid("shape")
    if payload["record_schema"] != CHATBOOK_PROMPT_RECORD_SCHEMA:
        raise _invalid("schema")
    record_version = payload["record_version"]
    if type(record_version) is not int or record_version != (
        CHATBOOK_MIXIN_RECORD_VERSION
        if payload["artifact_type"] == "mixin"
        else CHATBOOK_PROMPT_RECORD_VERSION
    ):
        raise _invalid("version")

    record: dict[str, Any] = {
        "record_schema": CHATBOOK_PROMPT_RECORD_SCHEMA,
        "record_version": record_version,
        "name": _validate_name(payload["name"]),
    }
    for key in _NULLABLE_TEXT_KEYS[:-1]:
        record[key] = _validate_nullable_text(payload[key])

    keywords = payload["keywords"]
    if type(keywords) is not list or any(
        not isinstance(keyword, str) for keyword in keywords
    ):
        raise _invalid("field")
    record["keywords"] = list(keywords)

    artifact_type = payload["artifact_type"]
    if artifact_type not in ("prompt", "recipe", "mixin"):
        raise _invalid("field")
    record["artifact_type"] = artifact_type

    prompt_format = payload["prompt_format"]
    if prompt_format not in ("legacy", "structured"):
        raise _invalid("field")
    record["prompt_format"] = prompt_format

    prompt_schema_version = payload["prompt_schema_version"]
    if prompt_schema_version is not None and type(prompt_schema_version) is not int:
        raise _invalid("field")
    record["prompt_schema_version"] = prompt_schema_version
    record["prompt_definition"] = _validate_nullable_text(payload["prompt_definition"])
    if artifact_type == "mixin":
        from .prompt_source_capabilities import (
            local_prompt_capabilities,
            validate_console_artifact_payload,
            validate_prompt_request_size,
        )

        try:
            capabilities = local_prompt_capabilities()
            definition = record["prompt_definition"]
            if (
                not isinstance(definition, str)
                or len(definition.encode("utf-8")) > capabilities.definition_limit
            ):
                raise ValueError("definition size")
            validate_prompt_request_size(record, capabilities)
            validated = validate_console_artifact_payload(record, capabilities)
            validate_prompt_request_size(validated, capabilities)
            if (record["system_prompt"], record["user_prompt"]) != (
                validated["system_prompt"],
                validated["user_prompt"],
            ):
                raise ValueError("compatibility")
        except (TypeError, ValueError, RecursionError):
            raise _invalid("definition") from None
    return record


def _to_add_prompt_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: record[key] for key in _PORTABLE_FIELD_KEYS}
    result["keywords"] = list(record["keywords"])
    return result


def _decode_legacy(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = set(payload)
    if not _LEGACY_REQUIRED_KEYS.issubset(keys) or not keys.issubset(
        _LEGACY_REQUIRED_KEYS | _LEGACY_OPTIONAL_KEYS
    ):
        raise _invalid("shape")
    name = _validate_name(payload["name"])
    description = _validate_nullable_text(payload["description"])
    content = payload["content"]
    if not isinstance(content, str):
        raise _invalid("field")
    if "id" in payload and type(payload["id"]) is not int:
        raise _invalid("field")
    for key in ("created_at", "updated_at"):
        if key in payload:
            _validate_nullable_text(payload[key])
    return {
        "name": name,
        "author": None,
        "details": description,
        "system_prompt": content,
        "user_prompt": None,
        "keywords": [],
        "artifact_type": "prompt",
        "prompt_format": "legacy",
        "prompt_schema_version": None,
        "prompt_definition": None,
    }


def encode_chatbook_prompt_record(detail: Mapping[str, Any]) -> dict[str, Any]:
    """Encode one Prompt database snapshot as a portable Chatbook record.

    Args:
        detail: Mapping containing every portable Prompt field.

    Returns:
        A detached, strictly validated Prompt/Recipe-v1 or Mix-in-v2 record.

    Raises:
        PromptChatbookRecordError: If a required field is missing or invalid.
    """
    if not isinstance(detail, Mapping) or any(
        key not in detail for key in _PORTABLE_FIELD_KEYS
    ):
        raise _invalid("shape")
    record = {
        "record_schema": CHATBOOK_PROMPT_RECORD_SCHEMA,
        "record_version": (
            CHATBOOK_MIXIN_RECORD_VERSION
            if detail["artifact_type"] == "mixin"
            else CHATBOOK_PROMPT_RECORD_VERSION
        ),
        **{key: detail[key] for key in _PORTABLE_FIELD_KEYS},
    }
    return _validate_new_record(record)


def decode_chatbook_prompt_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a versioned or historical Chatbook Prompt payload.

    Args:
        payload: JSON object loaded from one Chatbook Prompt file.

    Returns:
        A detached mapping of portable ``PromptsDatabase.add_prompt`` fields.

    Raises:
        PromptChatbookRecordError: If the payload is unsupported or invalid.
    """
    if not isinstance(payload, Mapping):
        raise _invalid("shape")
    if "record_schema" in payload or "record_version" in payload:
        return _to_add_prompt_fields(_validate_new_record(payload))
    return _decode_legacy(payload)
