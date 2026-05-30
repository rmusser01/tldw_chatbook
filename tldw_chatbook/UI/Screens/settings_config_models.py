"""Small data models for the destination-native Settings hub."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SettingsCategoryId(StrEnum):
    """Stable category identifiers for the Settings workbench."""

    OVERVIEW = "overview"
    PROVIDERS_MODELS = "providers-models"
    APPEARANCE = "appearance"
    STORAGE = "storage"
    PRIVACY_SECURITY = "privacy-security"
    CONSOLE_BEHAVIOR = "console-behavior"
    DIAGNOSTICS = "diagnostics"
    ADVANCED_CONFIG = "advanced-config"


class SettingsValidationState(StrEnum):
    """Validation states shown by Settings categories."""

    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class SettingsValidationResult:
    """Result from a Settings validation action."""

    valid: bool
    message: str = ""
    state: SettingsValidationState = SettingsValidationState.UNKNOWN

    def __post_init__(self) -> None:
        if self.state is SettingsValidationState.UNKNOWN:
            object.__setattr__(
                self,
                "state",
                SettingsValidationState.VALID if self.valid else SettingsValidationState.INVALID,
            )


@dataclass
class SettingsDraft:
    """Track draft values for a single Settings category."""

    category: SettingsCategoryId
    originals: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)

    def set_value(self, key: str, original_value: Any, draft_value: Any) -> None:
        """Stage a value while retaining its last loaded original."""
        self.originals[key] = original_value
        self.values[key] = draft_value

    @property
    def dirty_keys(self) -> set[str]:
        """Return keys where staged values differ from originals."""
        return {
            key
            for key, value in self.values.items()
            if value != self.originals.get(key)
        }

    @property
    def is_dirty(self) -> bool:
        """Return whether this category has unsaved changes."""
        return bool(self.dirty_keys)


@dataclass(frozen=True)
class SettingsCategorySummary:
    """Display summary for a Settings category."""

    category: SettingsCategoryId
    title: str
    description: str = ""
    status: str = ""


@dataclass(frozen=True)
class SettingsOwnershipRecord:
    """Ownership boundary for a Settings category."""

    category: SettingsCategoryId
    owns_config_sections: tuple[str, ...] = ()
    reads_runtime_state_from: tuple[str, ...] = ()
    writes_allowed: bool = False
    runtime_owner: str = ""
    boundary_copy: str = ""
    recovery_copy: str = ""
    read_only_reason: str = ""


@dataclass(frozen=True)
class SettingsImpactSummary:
    """Display impact metadata for a Settings selection."""

    affects: tuple[str, ...] = ()
    source: str = ""
    restart_required: bool = False
    message: str = ""
