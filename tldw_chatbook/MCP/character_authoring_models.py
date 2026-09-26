"""MCP argument restrictions around the shared character request schemas."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..tldw_api.character_persona_schemas import (
    CharacterCreateRequest,
    CharacterUpdateRequest,
)


def _normalize_fields(
    fields: dict[str, Any],
    request_type: type[CharacterCreateRequest | CharacterUpdateRequest],
) -> dict[str, Any]:
    """Apply MCP restrictions while retaining shared limits and JSON parsing."""
    allowed = request_type.model_fields.keys() - {"image_base64"}
    if fields.keys() - allowed or any(value is None for value in fields.values()):
        raise ValueError("Provide supported, non-null text/JSON card fields.")
    if "name" in fields and (
        not isinstance(fields["name"], str) or not fields["name"].strip()
    ):
        raise ValueError("name must be non-empty.")
    normalized = request_type.model_validate(fields).model_dump(exclude_unset=True)
    for key, expected_type in (
        ("tags", list),
        ("alternate_greetings", list),
        ("extensions", dict),
    ):
        if key in normalized and not isinstance(normalized[key], expected_type):
            raise ValueError(
                "tags and alternate_greetings must be arrays; extensions must be an object."
            )
    return normalized


class MCPCharacterCreateRequest(BaseModel):
    """Validate create arguments and normalize optional fields before dispatch."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    fields: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fields", mode="before")
    @classmethod
    def _optional_fields(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _normalize(self) -> Self:
        if "name" in self.fields:
            raise ValueError("Use the name argument instead of fields.name.")
        normalized = _normalize_fields(
            {"name": self.name, **self.fields}, CharacterCreateRequest
        )
        self.name = normalized.pop("name")
        self.fields = normalized
        return self


class MCPCharacterUpdateRequest(BaseModel):
    """Validate strict optimistic-lock arguments and a nonempty card patch."""

    model_config = ConfigDict(extra="forbid", strict=True)

    character_id: int = Field(gt=0)
    expected_version: int = Field(gt=0)
    fields: dict[str, Any] = Field(min_length=1)

    @field_validator("character_id", "expected_version", mode="before")
    @classmethod
    def _exact_integer(cls, value: Any) -> int:
        # Pydantic strict integers still accept IntEnum and other int subclasses.
        if type(value) is not int:
            raise ValueError("character_id and expected_version must be integers.")
        return value

    @field_validator("fields")
    @classmethod
    def _normalize(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _normalize_fields(value, CharacterUpdateRequest)
