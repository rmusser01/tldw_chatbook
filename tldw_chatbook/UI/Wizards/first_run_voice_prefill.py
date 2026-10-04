"""What the first-run Voice step starts from, and when Next may write.

TASK-34100.8 (voice-speech-01, SF3 stage 2 "untouched means unwritten").
The step used to start from a hard-coded PocketTTS draft and post a save on
every Next, so an untouched step overwrote a working voice. It now starts
from the saved voice, read from the RAW ``[app_tts]`` table (the loaded
settings back-fill ``default_provider = "openai"``, which must never read as a
choice), and writes only when the draft differs from what is saved or the
user tested or ticked "Use as default" this run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from tldw_chatbook.TTS.openai_compatible_config import (
    normalize_openai_authentication_mode,
    normalize_openai_compatible_endpoint,
)
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as vs

#: What the runtime uses for the OpenAI slot when nothing is saved
#: (``TTSPreferencesSnapshot.from_settings`` and the OpenAI backend).
_RUNTIME_FALLBACK = ("tts-1-hd", "shimmer", "mp3")

_PRESET_NAMES = {
    vs.VOICE_PRESET_POCKET_TTS: "PocketTTS",
    vs.VOICE_PRESET_OFFICIAL_OPENAI: "OpenAI",
    vs.VOICE_PRESET_OMNIVOICE: "OmniVoice",
}


@dataclass(frozen=True, slots=True)
class SavedVoice:
    """The voice saved before this run, as the step would show it.

    Attributes:
        preset: The Service radio it maps to. ``VOICE_PRESET_NONE`` when the
            saved default belongs to a provider this step does not set up.
        draft: The controls' values, including the "Use as default" box.
        other_provider: That other provider's id, kept as it is.
    """

    preset: str
    draft: vs.VoiceSetupDraft
    other_provider: str = ""


def raw_app_tts(app_config: object) -> Mapping[str, object]:
    """Return the raw ``[app_tts]`` table, never the back-filled loaded view.

    Args:
        app_config: The app's loaded settings (carrying
            ``COMPREHENSIVE_CONFIG_RAW``) or a raw TOML mapping.

    Returns:
        The table as saved in config.toml, or an empty mapping.
    """
    if not isinstance(app_config, Mapping):
        return {}
    raw = app_config.get("COMPREHENSIVE_CONFIG_RAW")
    if not isinstance(raw, Mapping):
        return {}
    table = raw.get("app_tts")
    return table if isinstance(table, Mapping) else {}


def _text(table: Mapping[str, object], key: str) -> str:
    value = table.get(key)
    return value.strip() if isinstance(value, str) else ""


def _speed(table: Mapping[str, object]) -> float:
    value = table.get("default_speed")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0
    speed = float(value)
    return speed if math.isfinite(speed) and 0.25 <= speed <= 4.0 else 1.0


def _openai_slot_preset(speech_url: str) -> str:
    if speech_url == vs.POCKET_TTS_ENDPOINT:
        return vs.VOICE_PRESET_POCKET_TTS
    if speech_url == vs.OFFICIAL_OPENAI_TTS_ENDPOINT:
        return vs.VOICE_PRESET_OFFICIAL_OPENAI
    return vs.VOICE_PRESET_CUSTOM


def saved_voice_from_config(table: object) -> SavedVoice | None:
    """Map a raw ``[app_tts]`` table to the voice the step should prefill.

    Args:
        table: The raw ``[app_tts]`` table (see :func:`raw_app_tts`).

    Returns:
        The saved voice, or None when nothing voice-related is saved.
    """
    if not isinstance(table, Mapping):
        return None
    provider = _text(table, "default_provider")
    base_url = _text(table, "OPENAI_BASE_URL")
    if provider == vs.VOICE_PRESET_OMNIVOICE:
        draft = vs.apply_voice_preset(
            _blank_draft(speed=_speed(table), use_as_default=True),
            vs.VOICE_PRESET_POCKET_TTS,
        )
        return SavedVoice(vs.VOICE_PRESET_OMNIVOICE, draft)
    if not base_url and provider != "openai":
        if not provider:
            return None
        return SavedVoice(vs.VOICE_PRESET_NONE, _blank_draft(), other_provider=provider)
    try:
        endpoint = normalize_openai_compatible_endpoint(
            base_url or vs.OFFICIAL_OPENAI_TTS_ENDPOINT
        )
        auth = normalize_openai_authentication_mode(
            table.get("OPENAI_AUTH_MODE"), endpoint=endpoint
        ).value
    except (TypeError, ValueError):
        return None
    preset = _openai_slot_preset(endpoint.speech_url)
    # The default axes belong to the OpenAI slot only while it is the default
    # provider (or none is saved); another provider's axes are not its own.
    own_axes = provider in {"", "openai"}
    defaults = {
        vs.VOICE_PRESET_POCKET_TTS: (vs.POCKET_TTS_MODEL, vs.POCKET_TTS_VOICE, "wav"),
    }.get(preset, _RUNTIME_FALLBACK)
    model = (_text(table, "default_model") if own_axes else "") or defaults[0]
    voice = (_text(table, "default_voice") if own_axes else "") or defaults[1]
    response_format = (_text(table, "default_format") if own_axes else "").lower()
    if response_format not in vs.RESPONSE_FORMATS:
        response_format = defaults[2]
    draft = vs.VoiceSetupDraft(
        endpoint=endpoint.speech_url,
        authentication_mode=auth,
        model_id=model,
        voice_id=voice,
        response_format=response_format,
        speed=_speed(table) if own_axes else 1.0,
        sample_text=vs.DEFAULT_SAMPLE_TEXT,
        use_as_default=provider == "openai",
    )
    return SavedVoice(preset, draft, other_provider="" if own_axes else provider)


def _blank_draft(*, speed: float = 1.0, use_as_default: bool = False):
    return vs.VoiceSetupDraft(
        endpoint=vs.POCKET_TTS_ENDPOINT,
        authentication_mode="none",
        model_id=vs.POCKET_TTS_MODEL,
        voice_id=vs.POCKET_TTS_VOICE,
        response_format="wav",
        speed=speed,
        sample_text=vs.DEFAULT_SAMPLE_TEXT,
        use_as_default=use_as_default,
    )


def initial_voice_draft() -> vs.VoiceSetupDraft:
    """The controls' values when nothing is saved (PocketTTS's preset)."""
    return _blank_draft()


def reply_voice_uses_openai_slot(table: object) -> bool:
    """Whether replies are read by the OpenAI-compatible slot.

    True when ``default_provider`` is ``openai`` or absent (the runtime's
    fallback). A save that changes that slot's endpoint then changes the
    voice replies use, so its model, voice and format must travel with it.
    """
    if not isinstance(table, Mapping):
        return True
    return _text(table, "default_provider") in {"", "openai"}


def _persisted_identity(draft: vs.VoiceSetupDraft) -> tuple[object, ...]:
    try:
        endpoint = normalize_openai_compatible_endpoint(draft.endpoint).speech_url
    except ValueError:
        endpoint = draft.endpoint
    return (
        endpoint,
        draft.authentication_mode,
        draft.model_id.strip(),
        draft.voice_id.strip(),
        draft.response_format.strip().lower(),
        float(draft.speed),
        draft.use_as_default,
    )


def should_persist_voice_config(
    draft: vs.VoiceSetupDraft,
    baseline: vs.VoiceSetupDraft | None,
    *,
    acted_this_run: bool,
) -> bool:
    """The delta gate: does Next have anything to write?

    Mirrors ``should_persist_speech_config``. The sample text never persists,
    so it is not compared.

    Args:
        draft: The controls' current values.
        baseline: What is saved (None when nothing is).
        acted_this_run: The user tested successfully or set "Use as default".

    Returns:
        True when the save should be posted.
    """
    if acted_this_run or baseline is None:
        return True
    return _persisted_identity(draft) != _persisted_identity(baseline)


def service_name(preset: str, endpoint: str) -> str:
    """Name a service the way the Service radio does."""
    if preset in _PRESET_NAMES:
        return _PRESET_NAMES[preset]
    try:
        url = normalize_openai_compatible_endpoint(endpoint).speech_url
    except ValueError:
        return "Custom endpoint"
    return f"Custom endpoint {vs.endpoint_host(url)}".rstrip()


def voice_label(saved: SavedVoice) -> str:
    """'OpenAI · tts-1-hd · shimmer' -- service, model and voice."""
    if saved.preset == vs.VOICE_PRESET_OMNIVOICE:
        return "OmniVoice"
    if saved.preset == vs.VOICE_PRESET_NONE:
        return saved.other_provider
    name = service_name(saved.preset, saved.draft.endpoint)
    return f"{name} · {saved.draft.model_id} · {saved.draft.voice_id}"


def current_voice_copy(saved: SavedVoice) -> str:
    """The re-run status line naming the voice an untouched Next keeps."""
    if saved.preset == vs.VOICE_PRESET_NONE:
        return (
            f"Current voice: {saved.other_provider} — kept as it is; nothing is "
            "saved here. Change it in Settings ▸ Speech & TTS."
        )
    return f"Current voice: {voice_label(saved)} — unchanged unless you edit it."


__all__ = [
    "SavedVoice",
    "current_voice_copy",
    "initial_voice_draft",
    "raw_app_tts",
    "reply_voice_uses_openai_slot",
    "saved_voice_from_config",
    "service_name",
    "should_persist_voice_config",
    "voice_label",
]
