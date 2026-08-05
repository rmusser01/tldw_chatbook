"""Immutable provider-neutral defaults for TTS request admission."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal, cast

ModelMode = Literal["exact", "first_available"]
VoiceMode = Literal["exact", "server_default"]

_MISSING = object()
_MODEL_MODES = frozenset({"exact", "first_available"})
_VOICE_MODES = frozenset({"exact", "server_default"})


def _freeze_value(value: Any) -> Any:
    """Return an isolated immutable copy of a configuration value."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                deepcopy(key): _freeze_value(nested_value)
                for key, nested_value in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return deepcopy(value)


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_mode(
    name: str,
    value: object,
    allowed: frozenset[str],
) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")
    return value


def _require_speed(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("default_speed must be a finite positive number")
    return float(value)


def _selected_app_tts(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_config = settings.get("COMPREHENSIVE_CONFIG_RAW")
    if isinstance(raw_config, Mapping) and "app_tts" in raw_config:
        raw_app_tts = raw_config["app_tts"]
        if not isinstance(raw_app_tts, Mapping):
            raise ValueError("app_tts settings must be a mapping")
        return raw_app_tts

    if "app_tts" in settings:
        direct_app_tts = settings["app_tts"]
        if not isinstance(direct_app_tts, Mapping):
            raise ValueError("app_tts settings must be a mapping")
        return direct_app_tts

    normalized_app_tts = settings.get("APP_TTS_CONFIG", {})
    if not isinstance(normalized_app_tts, Mapping):
        raise ValueError("APP_TTS_CONFIG settings must be a mapping")
    return normalized_app_tts


def _resolved_selection(
    values: Mapping[str, Any],
    *,
    provider_id: str,
    mode_key: str,
    id_key: str,
    dynamic_mode: str,
    legacy_default: str,
) -> tuple[str, str | None]:
    exact_id = values.get(
        id_key,
        "" if provider_id == "audio_cpp" else legacy_default,
    )
    explicit_mode = values.get(mode_key, _MISSING)
    if explicit_mode is _MISSING:
        if provider_id == "audio_cpp" and (
            exact_id is None or (isinstance(exact_id, str) and not exact_id.strip())
        ):
            return dynamic_mode, None
        mode = "exact"
    else:
        allowed = _MODEL_MODES if mode_key == "default_model_mode" else _VOICE_MODES
        mode = _require_mode(mode_key, explicit_mode, allowed)

    if mode == dynamic_mode:
        return mode, None
    return mode, _require_identifier(id_key, exact_id)


@dataclass(frozen=True, slots=True)
class TTSConfigMutation:
    """One defensively copied set/delete mutation for TTS configuration."""

    sets: Mapping[str, Mapping[str, object]]
    deletes: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.sets, Mapping) or not isinstance(
            self.deletes,
            Mapping,
        ):
            raise ValueError("TTS configuration mutation must use mappings")

        frozen_sets: dict[str, Mapping[str, object]] = {}
        for section, values in self.sets.items():
            if not isinstance(section, str) or not section:
                raise ValueError("TTS configuration section must be a non-empty string")
            if not isinstance(values, Mapping):
                raise ValueError("TTS configuration set values must be mappings")
            frozen_sets[section] = MappingProxyType(
                {deepcopy(key): _freeze_value(value) for key, value in values.items()}
            )

        frozen_deletes: dict[str, tuple[str, ...]] = {}
        for section, keys in self.deletes.items():
            if not isinstance(section, str) or not section:
                raise ValueError("TTS configuration section must be a non-empty string")
            if isinstance(keys, (str, bytes)):
                raise ValueError("TTS configuration delete keys must be collections")
            copied_keys = tuple(keys)
            if not all(isinstance(key, str) and key for key in copied_keys):
                raise ValueError(
                    "TTS configuration delete keys must be non-empty strings"
                )
            frozen_deletes[section] = copied_keys

        object.__setattr__(self, "sets", MappingProxyType(frozen_sets))
        object.__setattr__(self, "deletes", MappingProxyType(frozen_deletes))


@dataclass(frozen=True, slots=True)
class TTSPreferencesSnapshot:
    """One validated immutable selection for global TTS requests."""

    provider_id: str
    model_mode: ModelMode
    model_id: str | None
    voice_mode: VoiceMode
    voice_id: str | None
    response_format: str
    speed: float

    def __post_init__(self) -> None:
        _require_identifier("provider_id", self.provider_id)
        _require_mode("model_mode", self.model_mode, _MODEL_MODES)
        _require_mode("voice_mode", self.voice_mode, _VOICE_MODES)
        _require_identifier("response_format", self.response_format)

        if self.model_mode == "exact":
            _require_identifier("default_model", self.model_id)
        elif self.model_id is not None:
            raise ValueError("first_available model mode requires model_id to be None")

        if self.voice_mode == "exact":
            _require_identifier("default_voice", self.voice_id)
        elif self.voice_id is not None:
            raise ValueError("server_default voice mode requires voice_id to be None")

        speed = _require_speed(self.speed)
        object.__setattr__(self, "speed", speed)
        if self.provider_id == "audio_cpp":
            if self.response_format != "wav":
                raise ValueError("audio.cpp response format must be wav")
            if speed != 1.0:
                raise ValueError("audio.cpp speed must be exactly 1.0")

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
    ) -> TTSPreferencesSnapshot:
        """Parse supported raw or normalized settings without mutating them.

        Args:
            settings: A raw TOML mapping, normalized application settings, or
                normalized settings carrying ``COMPREHENSIVE_CONFIG_RAW``.

        Returns:
            A validated immutable preference snapshot.

        Raises:
            ValueError: If the settings shape or preference values are invalid.
        """
        if not isinstance(settings, Mapping):
            raise ValueError("TTS settings must be a mapping")
        values = _selected_app_tts(settings)
        provider_id = _require_identifier(
            "default_provider",
            values.get("default_provider", "openai"),
        )

        model_mode, model_id = _resolved_selection(
            values,
            provider_id=provider_id,
            mode_key="default_model_mode",
            id_key="default_model",
            dynamic_mode="first_available",
            legacy_default="tts-1-hd",
        )
        voice_mode, voice_id = _resolved_selection(
            values,
            provider_id=provider_id,
            mode_key="default_voice_mode",
            id_key="default_voice",
            dynamic_mode="server_default",
            legacy_default="shimmer",
        )
        response_format = _require_identifier(
            "default_format",
            values.get(
                "default_format",
                "wav" if provider_id == "audio_cpp" else "mp3",
            ),
        )
        speed = _require_speed(values.get("default_speed", 1.0))

        if provider_id == "audio_cpp":
            for options_key in ("default_options", "options"):
                if values.get(options_key):
                    raise ValueError("audio.cpp default options must be empty")

        return cls(
            provider_id=provider_id,
            model_mode=cast(ModelMode, model_mode),
            model_id=model_id,
            voice_mode=cast(VoiceMode, voice_mode),
            voice_id=voice_id,
            response_format=response_format,
            speed=speed,
        )

    def config_mutation(self) -> TTSConfigMutation:
        """Return the authoritative mode/value mutation for current aliases.

        Returns:
            An immutable set/delete mutation for canonical and legacy aliases.
        """
        app_tts: dict[str, object] = {
            "default_provider": self.provider_id,
            "default_model_mode": self.model_mode,
            "default_voice_mode": self.voice_mode,
            "default_format": self.response_format,
            "default_speed": self.speed,
        }
        legacy: dict[str, object] = {
            "default_tts_provider": self.provider_id,
            "default_openai_tts_output_format": self.response_format,
            "default_openai_tts_speed": self.speed,
        }
        app_tts_deletes: list[str] = []
        legacy_deletes: list[str] = []

        if self.model_mode == "exact":
            app_tts["default_model"] = self.model_id
            legacy["default_openai_tts_model"] = self.model_id
        else:
            app_tts_deletes.append("default_model")
            legacy_deletes.append("default_openai_tts_model")

        if self.voice_mode == "exact":
            app_tts["default_voice"] = self.voice_id
            legacy["default_tts_voice"] = self.voice_id
        else:
            app_tts_deletes.append("default_voice")
            legacy_deletes.append("default_tts_voice")

        deletes: dict[str, tuple[str, ...]] = {}
        if app_tts_deletes:
            deletes["app_tts"] = tuple(app_tts_deletes)
        if legacy_deletes:
            deletes["tts_settings"] = tuple(legacy_deletes)
        return TTSConfigMutation(
            sets={"app_tts": app_tts, "tts_settings": legacy},
            deletes=deletes,
        )
