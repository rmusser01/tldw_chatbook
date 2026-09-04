"""Typed, secret-free navigation contracts for Conversation settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ...Utils.input_validation import (
    validate_navigation_context_text,
    validate_navigation_provider_key,
)

_FOCUS_CONTROL_IDS = frozenset(
    {
        "console-settings-provider",
        "console-settings-model-picker",
        "console-settings-model-select",
        "console-settings-model-input",
        "console-settings-model-custom",
        "console-settings-base-url",
        "console-settings-temperature",
        "console-settings-max-tokens",
        "console-settings-streaming",
        "console-settings-view-model",
        "console-settings-view-context",
        "console-context-budget-mode",
        "console-context-custom-budget",
        "console-context-compaction-mode",
        "console-context-compaction-representation",
        "console-context-trigger-percent",
        "console-context-target-percent",
        "console-context-summary-max",
        "console-context-failure-behavior",
        "console-context-carry-forward",
        "console-context-compact-now",
        "console-context-reset-current",
        "console-context-undo-reset",
        "console-context-reset-overrides",
        "console-context-reset-all",
        "console-context-confirm-reset-all",
    }
)


class ConversationSettingsReturnOutcome(StrEnum):
    """Fixed result labels used by the Console return status copy."""

    CREDENTIAL_SAVED = "credential_saved"
    PROVIDER_SETTINGS_SAVED = "provider_settings_saved"
    WITHOUT_SAVING = "without_saving"


class _StrictNavigationContext(BaseModel):
    """Pydantic boundary shared by all external navigation payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _ConversationSettingsReturnIntentContext(_StrictNavigationContext):
    session_id: str = Field(min_length=1, max_length=256)
    settings_revision: int = Field(ge=0)
    active_view: Literal["model", "context"]
    focus_control_id: str | None

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, value: str) -> str:
        return validate_navigation_context_text(
            value,
            name="session_id",
            max_length=256,
        )

    @field_validator("focus_control_id")
    @classmethod
    def _validate_focus_control_id(cls, value: str | None) -> str | None:
        if value is not None and value not in _FOCUS_CONTROL_IDS:
            raise ValueError("focus_control_id is invalid")
        return value


class _ProviderSettingsNavigationContext(_StrictNavigationContext):
    category: Literal["providers-models"]
    provider: str = Field(min_length=1, max_length=128)
    model: str | None = Field(default=None, max_length=512)
    field: Literal["api_key"]
    return_revision: int = Field(ge=1)

    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        return validate_navigation_provider_key(value)

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_navigation_context_text(
            value,
            name="model",
            max_length=512,
        )


class _ConsoleSettingsReturnContext(_ConversationSettingsReturnIntentContext):
    return_revision: int = Field(ge=1)
    outcome: Literal[
        "credential_saved",
        "provider_settings_saved",
        "without_saving",
    ]


@dataclass(frozen=True, slots=True)
class ConversationSettingsReturnIntent:
    """Secret-free request to restore one Console settings draft."""

    session_id: str
    settings_revision: int
    active_view: Literal["model", "context"]
    focus_control_id: str | None

    def __post_init__(self) -> None:
        try:
            validated = _ConversationSettingsReturnIntentContext.model_validate(
                self.to_context()
            )
        except ValidationError as exc:
            raise ValueError("Conversation settings return intent is invalid") from exc
        object.__setattr__(self, "session_id", validated.session_id)
        object.__setattr__(self, "settings_revision", validated.settings_revision)
        object.__setattr__(self, "active_view", validated.active_view)
        object.__setattr__(self, "focus_control_id", validated.focus_control_id)

    def to_context(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "settings_revision": self.settings_revision,
            "active_view": self.active_view,
            "focus_control_id": self.focus_control_id,
        }

    @classmethod
    def from_context(
        cls, context: Mapping[str, object]
    ) -> ConversationSettingsReturnIntent | None:
        if not isinstance(context, Mapping):
            return None
        try:
            validated = _ConversationSettingsReturnIntentContext.model_validate(context)
            return cls(**validated.model_dump())
        except (TypeError, ValueError, ValidationError):
            return None


@dataclass(frozen=True, slots=True)
class ProviderSettingsNavigationTarget:
    """Allowlisted deep-link into the durable Providers & Models settings."""

    category: Literal["providers-models"]
    provider: str
    model: str | None
    field: Literal["api_key"]
    return_revision: int

    def __post_init__(self) -> None:
        try:
            validated = _ProviderSettingsNavigationContext.model_validate(
                self.to_context()
            )
        except ValidationError as exc:
            raise ValueError("Provider settings navigation target is invalid") from exc
        object.__setattr__(self, "category", validated.category)
        object.__setattr__(self, "provider", validated.provider)
        object.__setattr__(self, "model", validated.model)
        object.__setattr__(self, "field", validated.field)
        object.__setattr__(self, "return_revision", validated.return_revision)

    def to_context(self) -> dict[str, object]:
        return {
            "category": self.category,
            "provider": self.provider,
            "model": self.model,
            "field": self.field,
            "return_revision": self.return_revision,
        }

    @classmethod
    def from_context(
        cls, context: Mapping[str, object]
    ) -> ProviderSettingsNavigationTarget | None:
        if not isinstance(context, Mapping):
            return None
        try:
            validated = _ProviderSettingsNavigationContext.model_validate(context)
            return cls(**validated.model_dump())
        except (TypeError, ValueError, ValidationError):
            return None


@dataclass(frozen=True, slots=True)
class ConsoleSettingsReturnTarget:
    """Allowlisted Console destination context returned by Settings."""

    session_id: str
    settings_revision: int
    active_view: Literal["model", "context"]
    focus_control_id: str | None
    return_revision: int
    outcome: ConversationSettingsReturnOutcome

    def __post_init__(self) -> None:
        context = self.to_context()
        try:
            validated = _ConsoleSettingsReturnContext.model_validate(context)
        except ValidationError as exc:
            raise ValueError("Console settings return target is invalid") from exc
        object.__setattr__(self, "session_id", validated.session_id)
        object.__setattr__(self, "settings_revision", validated.settings_revision)
        object.__setattr__(self, "active_view", validated.active_view)
        object.__setattr__(self, "focus_control_id", validated.focus_control_id)
        object.__setattr__(self, "return_revision", validated.return_revision)
        object.__setattr__(
            self,
            "outcome",
            ConversationSettingsReturnOutcome(validated.outcome),
        )

    def to_context(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "settings_revision": self.settings_revision,
            "active_view": self.active_view,
            "focus_control_id": self.focus_control_id,
            "return_revision": self.return_revision,
            "outcome": (
                self.outcome.value
                if isinstance(self.outcome, ConversationSettingsReturnOutcome)
                else self.outcome
            ),
        }

    @classmethod
    def from_context(
        cls, context: Mapping[str, object]
    ) -> ConsoleSettingsReturnTarget | None:
        if not isinstance(context, Mapping):
            return None
        try:
            validated = _ConsoleSettingsReturnContext.model_validate(context)
            return cls(**validated.model_dump())
        except (TypeError, ValueError, ValidationError):
            return None
