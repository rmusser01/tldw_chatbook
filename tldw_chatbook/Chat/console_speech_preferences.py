"""Versioned conversation metadata for Console reply-speech preferences."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CONSOLE_SPEECH_METADATA_KEY = "console_speech"
CONSOLE_SPEECH_CONSENT_VERSION = 1
_DESTINATION_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OWNED_KEYS = {
    "auto_speak",
    "paused",
    "consent_destination",
    "consent_version",
}


def is_console_speech_destination(value: object) -> bool:
    """Return whether value is one canonical Console TTS destination digest."""
    return type(value) is str and _DESTINATION_PATTERN.fullmatch(value) is not None


class ConsoleSpeechPreferencesVersionError(ValueError):
    """Durable speech preferences belong to a newer application version."""


@dataclass(frozen=True, slots=True)
class ConsoleSpeechPreferences:
    """Opt-in reply-speech state owned by one Console conversation."""

    auto_speak: bool = False
    paused: bool = False
    consent_destination: str | None = None
    consent_version: int = CONSOLE_SPEECH_CONSENT_VERSION

    def __post_init__(self) -> None:
        if type(self.auto_speak) is not bool:
            raise ValueError("auto_speak must be an exact boolean.")
        if type(self.paused) is not bool:
            raise ValueError("paused must be an exact boolean.")
        if self.consent_destination is not None and not is_console_speech_destination(
            self.consent_destination
        ):
            raise ValueError(
                "consent_destination must be a canonical SHA-256 fingerprint."
            )
        if type(self.consent_version) is not int or self.consent_version != 1:
            raise ValueError("consent_version must be exactly 1.")


def parse_console_speech_preferences(metadata: object) -> ConsoleSpeechPreferences:
    """Parse valid version-one speech metadata, failing closed as one unit."""
    outer = _metadata_object(metadata)
    owned = outer.get(CONSOLE_SPEECH_METADATA_KEY)
    if not isinstance(owned, Mapping):
        return ConsoleSpeechPreferences()
    if set(owned) != _OWNED_KEYS:
        return ConsoleSpeechPreferences()
    try:
        return ConsoleSpeechPreferences(
            auto_speak=owned["auto_speak"],
            paused=owned["paused"],
            consent_destination=owned["consent_destination"],
            consent_version=owned["consent_version"],
        )
    except (TypeError, ValueError):
        return ConsoleSpeechPreferences()


def merge_console_speech_preferences(
    metadata: object,
    preferences: ConsoleSpeechPreferences,
) -> dict[str, object]:
    """Replace only the Console speech key while preserving metadata siblings."""
    if not isinstance(preferences, ConsoleSpeechPreferences):
        raise TypeError("preferences must be ConsoleSpeechPreferences.")
    merged = _metadata_object(metadata)
    owned = merged.get(CONSOLE_SPEECH_METADATA_KEY)
    existing_version = (
        owned.get("consent_version") if isinstance(owned, Mapping) else None
    )
    if type(existing_version) is int and existing_version > (
        CONSOLE_SPEECH_CONSENT_VERSION
    ):
        raise ConsoleSpeechPreferencesVersionError(
            "Cannot overwrite Console speech preferences at version "
            f"{existing_version}."
        )
    # Reconstruct to reject objects forged by bypassing the frozen dataclass API.
    validated = ConsoleSpeechPreferences(
        auto_speak=preferences.auto_speak,
        paused=preferences.paused,
        consent_destination=preferences.consent_destination,
        consent_version=preferences.consent_version,
    )
    merged[CONSOLE_SPEECH_METADATA_KEY] = {
        "auto_speak": validated.auto_speak,
        "paused": validated.paused,
        "consent_destination": validated.consent_destination,
        "consent_version": validated.consent_version,
    }
    return merged


def _metadata_object(metadata: object) -> dict[str, Any]:
    if isinstance(metadata, Mapping):
        return dict(metadata)
    if type(metadata) is not str or not metadata:
        return {}
    try:
        decoded = json.loads(metadata)
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}
