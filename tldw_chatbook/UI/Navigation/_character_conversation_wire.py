"""Strict first-use wire boundary; immutable navigation objects stay separate."""

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from .character_conversation_navigation import (
    _RETURN_TARGETS,
    _canonical_text,
)


def _identity_text(value: str) -> str:
    return _canonical_text(value, "identity")


_IdentityText = Annotated[str, AfterValidator(_identity_text)]
_Version = Annotated[int, Field(ge=1, le=1)]


class _StrictWire(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)


class _ReturnTargetWire(_StrictWire):
    screen_id: Annotated[str, Field(min_length=1, max_length=128)]
    focus_id: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$")]

    @model_validator(mode="after")
    def allowed_target(self) -> Self:
        if (self.screen_id, self.focus_id) not in _RETURN_TARGETS:
            raise ValueError("return target is not allowed for character navigation")
        return self


class _ResolvedCharacterWire(_StrictWire):
    version: _Version
    tag: Literal["resolved_local_character"]
    data_authority_id: _IdentityText
    character_id: Annotated[int, Field(ge=1, le=2**63 - 1)]


class _UnresolvedConversationWire(_StrictWire):
    version: _Version
    tag: Literal["unresolved_conversation"]
    data_authority_id: _IdentityText
    conversation_id: _IdentityText


class _RoleplayLinkWire(_StrictWire):
    version: _Version
    source: Literal["local"]
    character: _ResolvedCharacterWire
    conversation_id: _IdentityText | None
    query: Annotated[str, Field(max_length=4096)]
    data_revision: Annotated[int, Field(ge=0)] | None
    return_target: _ReturnTargetWire | None

    @model_validator(mode="after")
    def exact_revision(self) -> Self:
        if (self.conversation_id is None) != (self.data_revision is None):
            raise ValueError(
                "data_revision is required only for an exact conversation_id"
            )
        return self


class _LibraryRepairWire(_StrictWire):
    version: _Version
    source: Literal["local"]
    data_authority_id: _IdentityText
    unresolved: _UnresolvedConversationWire
    expected_conversation_version: Annotated[int, Field(ge=1)]
    historical_display_snapshot: Annotated[str, Field(min_length=1, max_length=1024)]
    return_target: _ReturnTargetWire

    @model_validator(mode="after")
    def repair_evidence(self) -> Self:
        if self.data_authority_id != self.unresolved.data_authority_id:
            raise ValueError("repair authority components do not match")
        _canonical_text(
            self.historical_display_snapshot,
            "historical_display_snapshot",
            max_bytes=4096,
        )
        return self


class _ConsoleContextReturnWire(_ReturnTargetWire):
    screen_id: Literal["chat"]
    focus_id: Literal["console-context-character"]


class _LibraryUnavailableInspectionWire(_StrictWire):
    version: _Version
    source: Literal["local"]
    data_authority_id: _IdentityText
    unresolved: _UnresolvedConversationWire
    return_target: _ConsoleContextReturnWire

    @model_validator(mode="after")
    def same_authority(self) -> Self:
        if self.data_authority_id != self.unresolved.data_authority_id:
            raise ValueError("inspection authority components do not match")
        return self


class _LibraryUnavailableBrowseWire(_StrictWire):
    version: _Version
    source: Literal["local"]
    data_authority_id: _IdentityText
    selected: _UnresolvedConversationWire
    return_target: _ConsoleContextReturnWire

    @model_validator(mode="after")
    def same_authority(self) -> Self:
        if self.data_authority_id != self.selected.data_authority_id:
            raise ValueError("browse authority components do not match")
        return self
