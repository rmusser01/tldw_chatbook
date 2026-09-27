"""Bounded asserted evidence identity; no source access or profile admission."""

from datetime import datetime
from hashlib import sha256
from typing import Annotated, Literal
from unicodedata import category

from pydantic import AfterValidator, BeforeValidator, ConfigDict, Field, model_validator

from .canonical import I_JSON_MAX_INTEGER, PortableDateTime, canonical_bytes
from .payloads import FrozenModel


def _builtin_string(value: object) -> str:
    if type(value) is not str:
        raise ValueError("value must be a built-in string")
    return value


def _builtin_integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("value must be a built-in integer")
    return value


def _opaque_identity(value: str) -> str:
    if not value.strip():
        raise ValueError("identity must not be blank")
    if len(value.encode("utf-8", errors="strict")) > 512:
        raise ValueError("identity must not exceed 512 UTF-8 bytes")
    if any(category(character) in {"Cc", "Cf"} for character in value):
        raise ValueError("identity must not contain control or format characters")
    return value


def _capture_scalar(value: object) -> datetime | str:
    if type(value) not in (datetime, str):
        raise ValueError("capture time must be a built-in datetime or string")
    return value


Identity = Annotated[
    str,
    BeforeValidator(_builtin_string),
    Field(min_length=1, max_length=128),
    AfterValidator(_opaque_identity),
]
Digest = Annotated[
    str,
    BeforeValidator(_builtin_string),
    Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]
Offset = Annotated[
    int, BeforeValidator(_builtin_integer), Field(ge=0, le=I_JSON_MAX_INTEGER)
]
CaptureTime = Annotated[PortableDateTime, BeforeValidator(_capture_scalar)]


class OwnerVersionEvidenceBinding(FrozenModel):
    """Immutable structural assertions about one exact message representation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )

    component_version: Annotated[Literal[1], BeforeValidator(_builtin_integer)] = Field(
        repr=False
    )
    binding_id: Identity = Field(repr=False)
    authority_kind: Annotated[
        Literal["local_profile", "authenticated_tenant"],
        BeforeValidator(_builtin_string),
    ] = Field(repr=False)
    authority_id: Identity = Field(repr=False)
    governance_scope_id: Identity = Field(repr=False)
    source_kind: Annotated[
        Literal["conversation_message"], BeforeValidator(_builtin_string)
    ] = Field(repr=False)
    source_container_id: Identity = Field(repr=False)
    source_object_id: Identity = Field(repr=False)
    version_kind: Annotated[
        Literal["owner_immutable"], BeforeValidator(_builtin_string)
    ] = Field(repr=False)
    source_version_id: Identity = Field(repr=False)
    representation_id: Annotated[
        Literal["message_content_text_v1"], BeforeValidator(_builtin_string)
    ] = Field(repr=False)
    representation_sha256: Digest = Field(repr=False)
    offset_unit: Annotated[
        Literal["unicode_codepoint"], BeforeValidator(_builtin_string)
    ] = Field(repr=False)
    span_start: Offset = Field(repr=False)
    span_end: Offset = Field(repr=False)
    span_sha256: Digest = Field(repr=False)
    source_role: Annotated[
        Literal[
            "direct_user_message",
            "quoted_material",
            "attachment",
            "tool_result",
            "imported_material",
        ],
        BeforeValidator(_builtin_string),
    ] = Field(repr=False)
    captured_at: CaptureTime = Field(repr=False)

    @model_validator(mode="after")
    def _ordered_span(self) -> "OwnerVersionEvidenceBinding":
        if self.span_start > self.span_end:
            raise ValueError("span_start must not exceed span_end")
        return self


def owner_version_evidence_binding_digest(binding: OwnerVersionEvidenceBinding) -> str:
    """Hash a freshly validated snapshot of complete binding field values.

    Args:
        binding: Exact component-class instance; assertions remain untrusted.

    Returns:
        Lowercase SHA-256 of the validated component's canonical UTF-8 bytes.

    Raises:
        TypeError: The argument is not the exact component class.
        pydantic.ValidationError: Stored field values or keys are malformed.
    """
    if type(binding) is not OwnerVersionEvidenceBinding:
        raise TypeError("binding must be an exact OwnerVersionEvidenceBinding instance")
    validated = OwnerVersionEvidenceBinding.model_validate(dict(vars(binding)))
    return sha256(canonical_bytes(validated)).hexdigest()
