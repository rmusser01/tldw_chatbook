"""Typed, secret-free navigation contracts for Conversation settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Literal

from ...Chat.provider_readiness import provider_config_key


_PROVIDER_RE = re.compile(r"[a-z0-9][a-z0-9_]{0,127}\Z")
_SAFE_TEXT_RE = re.compile(r"[^\x00-\x1f\x7f\x80-\x9f]+\Z")
_FOCUS_CONTROL_IDS = frozenset(
    {
        "console-settings-provider",
        "console-settings-model",
        "console-settings-api-key",
        "console-settings-base-url",
        "console-settings-temperature",
        "console-settings-max-tokens",
        "console-settings-streaming",
        "console-settings-context-view",
    }
)


def _text(value: object, *, name: str, limit: int) -> str:
    if type(value) is not str or not value or value != value.strip() or len(value) > limit:
        raise ValueError(f"{name} is invalid")
    if _SAFE_TEXT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _revision(value: object, *, name: str = "revision", positive: bool = True) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


class ConversationSettingsReturnOutcome(StrEnum):
    """Fixed result labels used by the Console return status copy."""

    CREDENTIAL_SAVED = "credential_saved"
    PROVIDER_SETTINGS_SAVED = "provider_settings_saved"
    WITHOUT_SAVING = "without_saving"


@dataclass(frozen=True, slots=True)
class ConversationSettingsReturnIntent:
    """Secret-free request to restore one Console settings draft."""

    session_id: str
    settings_revision: int
    active_view: Literal["model", "context"]
    focus_control_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _text(self.session_id, name="session_id", limit=256))
        object.__setattr__(self, "settings_revision", _revision(self.settings_revision, name="settings_revision", positive=False))
        if self.active_view not in ("model", "context"):
            raise ValueError("active_view is invalid")
        if self.focus_control_id is not None and self.focus_control_id not in _FOCUS_CONTROL_IDS:
            raise ValueError("focus_control_id is invalid")

    def to_context(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "settings_revision": self.settings_revision,
            "active_view": self.active_view,
            "focus_control_id": self.focus_control_id,
        }

    @classmethod
    def from_context(cls, context: Mapping[str, object]) -> ConversationSettingsReturnIntent | None:
        if not isinstance(context, Mapping) or set(context) != set(cls("s", 1, "model", None).to_context()):
            return None
        try:
            return cls(**context)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class ProviderSettingsNavigationTarget:
    """Allowlisted deep-link into the durable Providers & Models settings."""

    category: Literal["providers-models"]
    provider: str
    model: str
    field: Literal["api_key"]
    return_revision: int

    def __post_init__(self) -> None:
        if self.category != "providers-models":
            raise ValueError("category is invalid")
        provider = provider_config_key(self.provider) if type(self.provider) is str else ""
        if not provider or _PROVIDER_RE.fullmatch(provider) is None:
            raise ValueError("provider is invalid")
        object.__setattr__(self, "provider", provider)
        _text(self.model, name="model", limit=512)
        if self.field != "api_key":
            raise ValueError("field is invalid")
        _revision(self.return_revision, name="return_revision")

    def to_context(self) -> dict[str, object]:
        return {
            "category": self.category,
            "provider": self.provider,
            "model": self.model,
            "field": self.field,
            "return_revision": self.return_revision,
        }

    @classmethod
    def from_context(cls, context: Mapping[str, object]) -> ProviderSettingsNavigationTarget | None:
        keys = {"category", "provider", "model", "field", "return_revision"}
        if not isinstance(context, Mapping) or set(context) != keys:
            return None
        try:
            return cls(**context)  # type: ignore[arg-type]
        except (TypeError, ValueError):
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
        intent = ConversationSettingsReturnIntent(
            self.session_id, self.settings_revision, self.active_view, self.focus_control_id
        )
        object.__setattr__(self, "session_id", intent.session_id)
        if not isinstance(self.outcome, ConversationSettingsReturnOutcome):
            try:
                object.__setattr__(self, "outcome", ConversationSettingsReturnOutcome(self.outcome))
            except (TypeError, ValueError):
                raise ValueError("outcome is invalid") from None
        _revision(self.return_revision, name="return_revision")

    def to_context(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "settings_revision": self.settings_revision,
            "active_view": self.active_view,
            "focus_control_id": self.focus_control_id,
            "return_revision": self.return_revision,
            "outcome": self.outcome.value,
        }

    @classmethod
    def from_context(cls, context: Mapping[str, object]) -> ConsoleSettingsReturnTarget | None:
        keys = {"session_id", "settings_revision", "active_view", "focus_control_id", "return_revision", "outcome"}
        if not isinstance(context, Mapping) or set(context) != keys:
            return None
        try:
            return cls(**context)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
