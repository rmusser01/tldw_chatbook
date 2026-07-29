"""Pure projection helpers for catalog-driven STTS Playground controls."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto

from tldw_chatbook.TTS.adapter_types import (
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderDescriptor,
)
from tldw_chatbook.TTS.profile_service import (
    ProfileAvailabilityState,
    TTSPlaygroundSelectionPreset,
)
from tldw_chatbook.TTS.profile_types import (
    AUDIO_CPP_PROFILE_RESPONSE_FORMAT,
    AUDIO_CPP_PROFILE_SPEED,
)

AUDIO_CPP_PROVIDER_ID = "audio_cpp"
SERVER_DEFAULT_VOICE_LABEL = "Server default"


class SelectSentinel(Enum):
    """Out-of-band values owned by local STTS Select controls."""

    FIRST_AVAILABLE_MODEL = auto()
    SERVER_DEFAULT_VOICE = auto()
    LOADING = auto()
    UNAVAILABLE = auto()


FIRST_AVAILABLE_MODEL_ID = SelectSentinel.FIRST_AVAILABLE_MODEL
SERVER_DEFAULT_VOICE_ID = SelectSentinel.SERVER_DEFAULT_VOICE
LOADING_SELECT_VALUE = SelectSentinel.LOADING
UNAVAILABLE_SELECT_VALUE = SelectSentinel.UNAVAILABLE

SelectValue = str | SelectSentinel
SelectOption = tuple[str, SelectValue]


@dataclass(frozen=True, slots=True)
class CatalogRequestToken:
    """Identity and revision snapshot for one catalog or voice request."""

    provider_id: str
    configuration_revision: int
    catalog_revision: int | None = None
    model_id: str | None = None
    request_generation: int | None = None

    def matches(
        self,
        *,
        provider_id: str,
        configuration_revision: int,
        catalog_revision: int | None,
        model_id: str | None,
        request_generation: int | None = None,
    ) -> bool:
        """Return whether every captured request dimension is still current.

        Args:
            provider_id: Currently selected canonical provider identifier.
            configuration_revision: Current provider configuration revision.
            catalog_revision: Current provider catalog revision, when applicable.
            model_id: Currently selected opaque model identifier, when applicable.
            request_generation: Latest request generation, when applicable.

        Returns:
            ``True`` when every captured dimension matches current state.

        """
        return (
            self.provider_id == provider_id
            and self.configuration_revision == configuration_revision
            and self.catalog_revision == catalog_revision
            and self.model_id == model_id
            and self.request_generation == request_generation
        )


@dataclass(frozen=True, slots=True)
class PlaygroundControls:
    """Provider-neutral state used to populate Playground controls."""

    provider_id: str
    model_options: tuple[SelectOption, ...]
    selected_model_id: str | None
    voice_options: tuple[SelectOption, ...]
    selected_voice_id: SelectValue | None
    format_options: tuple[str, ...]
    selected_format: str | None
    format_locked: bool
    speed: float
    speed_locked: bool
    generation_allowed: bool
    selection_changed: bool


def provider_options(
    descriptors: Iterable[TTSProviderDescriptor],
) -> tuple[SelectOption, ...]:
    """Return display labels paired with exact canonical provider IDs.

    Args:
        descriptors: Sealed registry descriptors in display order.

    Returns:
        Select options containing display labels and canonical provider IDs.

    """
    return tuple(
        (descriptor.display_name, descriptor.provider_id) for descriptor in descriptors
    )


def voice_id_for_request(selected_voice_id: object) -> str | None:
    """Translate the local Server-default sentinel into adapter omission.

    Args:
        selected_voice_id: Selected remote voice ID or local default sentinel.

    Returns:
        The exact remote voice ID, or ``None`` to request the server default.

    """
    if selected_voice_id is SERVER_DEFAULT_VOICE_ID:
        return None
    return selected_voice_id if isinstance(selected_voice_id, str) else None


def controls_from_catalog(
    catalog: TTSProviderCatalog,
    *,
    selected_model_id: str | None,
    selected_voice_id: SelectValue | None,
    discovered_voices: tuple[str, ...] | None,
    selected_format: str | None,
    speed: float,
) -> PlaygroundControls:
    """Resolve one provider catalog into deterministic control state.

    Args:
        catalog: Current provider catalog.
        selected_model_id: Previously selected opaque model ID.
        selected_voice_id: Previously selected opaque voice ID.
        discovered_voices: Lazily discovered voice IDs, when available.
        selected_format: Previously selected response format.
        speed: Previously selected synthesis speed.

    Returns:
        Provider-neutral state for the Playground controls.

    """
    model_options = tuple(
        (model.display_name or model.model_id, model.model_id)
        for model in catalog.models
    )
    model = _selected_model(catalog.models, selected_model_id)
    resolved_model_id = model.model_id if model is not None else None
    selection_changed = (
        selected_model_id is not None and selected_model_id != resolved_model_id
    )

    voice_options: tuple[SelectOption, ...]
    resolved_voice_id: SelectValue | None
    format_options: tuple[str, ...]
    resolved_format: str | None
    if catalog.provider_id == AUDIO_CPP_PROVIDER_ID:
        voice_options, resolved_voice_id, voice_changed = _audio_cpp_voices(
            discovered_voices or (),
            selected_voice_id,
        )
        format_options = ("wav",)
        resolved_format = "wav"
        format_locked = True
        resolved_speed = 1.0
        speed_locked = True
        format_compatible = model is not None and "wav" in model.formats
    else:
        voice_options, resolved_voice_id, voice_changed = _legacy_voices(
            () if model is None else model.voices,
            discovered_voices,
            selected_voice_id,
        )
        format_options = () if model is None else model.formats
        resolved_format = _retain_or_first(selected_format, format_options)
        format_locked = model is None or len(format_options) <= 1
        speed_locked = model is None or not model.supports_speed
        resolved_speed = 1.0 if speed_locked else speed
        format_compatible = resolved_format is not None

    health = catalog.health
    generation_allowed = bool(
        model is not None
        and health.state == "available"
        and health.fresh
        and format_compatible
    )
    return PlaygroundControls(
        provider_id=catalog.provider_id,
        model_options=model_options,
        selected_model_id=resolved_model_id,
        voice_options=voice_options,
        selected_voice_id=resolved_voice_id,
        format_options=format_options,
        selected_format=resolved_format,
        format_locked=format_locked,
        speed=resolved_speed,
        speed_locked=speed_locked,
        generation_allowed=generation_allowed,
        selection_changed=selection_changed or voice_changed,
    )


def controls_from_profile_preset(
    catalog: TTSProviderCatalog | None,
    *,
    preset: TTSPlaygroundSelectionPreset,
    discovered_voices: tuple[str, ...] | None,
) -> PlaygroundControls:
    """Project one exact profile selection without catalog substitution.

    Args:
        catalog: Current provider catalog, or ``None`` when it is unverified.
        preset: Exact persisted profile selection to project.
        discovered_voices: Authoritative voices, or ``None`` when unverified.

    Returns:
        Controls that preserve every exact profile value without fallback.
    """

    model_options = (
        tuple(
            (model.display_name or model.model_id, model.model_id)
            for model in catalog.models
        )
        if catalog is not None
        else ()
    )
    if preset.model_id not in {value for _label, value in model_options}:
        model_options = (*model_options, (preset.model_id, preset.model_id))

    voice_options: tuple[SelectOption, ...] = (
        (SERVER_DEFAULT_VOICE_LABEL, SERVER_DEFAULT_VOICE_ID),
        *((voice, voice) for voice in (discovered_voices or ())),
    )
    selected_voice: SelectValue = (
        SERVER_DEFAULT_VOICE_ID if preset.voice_id is None else preset.voice_id
    )
    if selected_voice not in {value for _label, value in voice_options}:
        assert isinstance(selected_voice, str)
        voice_options = (*voice_options, (selected_voice, selected_voice))

    availability = profile_availability_from_catalog(preset, catalog)
    return PlaygroundControls(
        provider_id=preset.provider_id,
        model_options=model_options,
        selected_model_id=preset.model_id,
        voice_options=voice_options,
        selected_voice_id=selected_voice,
        format_options=(preset.response_format,),
        selected_format=preset.response_format,
        format_locked=True,
        speed=preset.speed,
        speed_locked=True,
        generation_allowed=availability != "unavailable",
        selection_changed=False,
    )


def profile_availability_from_catalog(
    preset: TTSPlaygroundSelectionPreset,
    catalog: TTSProviderCatalog | None,
) -> ProfileAvailabilityState:
    """Conservatively revalidate exact profile fields against one catalog."""
    if preset.availability == "unavailable":
        return "unavailable"
    if (
        preset.provider_id != AUDIO_CPP_PROVIDER_ID
        or preset.response_format != AUDIO_CPP_PROFILE_RESPONSE_FORMAT
        or preset.speed != AUDIO_CPP_PROFILE_SPEED
        or bool(preset.options)
    ):
        return "unavailable"
    if catalog is None:
        return "unverified"
    if catalog.provider_id != preset.provider_id:
        return "unavailable"
    if not catalog.health.fresh or catalog.health.state == "reconfiguring":
        return "unverified"
    if catalog.health.state != "available":
        return "unavailable"
    model = next(
        (item for item in catalog.models if item.model_id == preset.model_id),
        None,
    )
    if model is None or preset.response_format not in model.formats:
        return "unavailable"
    if preset.voice_id is None and not model.omit_voice_uses_server_default:
        return "unavailable"
    return preset.availability


def _selected_model(
    models: tuple[TTSModelInfo, ...],
    selected_model_id: str | None,
) -> TTSModelInfo | None:
    if selected_model_id is not None:
        for model in models:
            if model.model_id == selected_model_id:
                return model
    return models[0] if models else None


def _audio_cpp_voices(
    voices: tuple[str, ...],
    selected_voice_id: SelectValue | None,
) -> tuple[tuple[SelectOption, ...], SelectValue, bool]:
    options: tuple[SelectOption, ...] = (
        (SERVER_DEFAULT_VOICE_LABEL, SERVER_DEFAULT_VOICE_ID),
        *((voice, voice) for voice in voices),
    )
    valid_ids = {value for _, value in options}
    if selected_voice_id is None:
        return options, SERVER_DEFAULT_VOICE_ID, False
    if selected_voice_id in valid_ids:
        return options, selected_voice_id, False
    return options, SERVER_DEFAULT_VOICE_ID, True


def _legacy_voices(
    catalog_voices: tuple[str, ...],
    discovered_voices: tuple[str, ...] | None,
    selected_voice_id: SelectValue | None,
) -> tuple[tuple[SelectOption, ...], str | None, bool]:
    voices = catalog_voices if discovered_voices is None else discovered_voices
    options = tuple((voice, voice) for voice in voices)
    resolved = _retain_or_first(selected_voice_id, voices)
    changed = selected_voice_id is not None and selected_voice_id != resolved
    return options, resolved, changed


def _retain_or_first(
    selected: SelectValue | None,
    values: tuple[str, ...],
) -> str | None:
    if selected in values:
        return selected
    return values[0] if values else None
