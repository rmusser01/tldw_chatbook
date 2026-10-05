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
from tldw_chatbook.TTS.pocket_tts_native import (
    POCKET_TTS_VOICES,
    is_pocket_tts_native_url,
)
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as vs

#: What the runtime uses for the OpenAI slot when nothing is saved
#: (``TTSPreferencesSnapshot.from_settings`` and the OpenAI backend).
_RUNTIME_FALLBACK = ("tts-1-hd", "shimmer", "mp3")

#: The presets that save to the OpenAI-compatible slot (one endpoint).
OPENAI_SLOT_PRESETS = frozenset(
    {
        vs.VOICE_PRESET_POCKET_TTS,
        vs.VOICE_PRESET_OFFICIAL_OPENAI,
        vs.VOICE_PRESET_CUSTOM,
    }
)

_PRESET_NAMES = {
    vs.VOICE_PRESET_POCKET_TTS: "PocketTTS",
    vs.VOICE_PRESET_OFFICIAL_OPENAI: "OpenAI",
    vs.VOICE_PRESET_OMNIVOICE: "OmniVoice",
}

#: The PocketTTS address the wizard wrote before TASK-34100.8 (with auth
#: "none"). The real pocket-tts server never serves it -- it speaks POST /tts
#: on :8000 and answers 404 here -- so such a table is the old wizard's
#: untouched write, and it cannot speak (review round 1, F2).
LEGACY_POCKET_TTS_ENDPOINT = "http://127.0.0.1:8765/v1/audio/speech"
_LEGACY_HOST = "127.0.0.1:8765"


@dataclass(frozen=True, slots=True)
class SavedVoice:
    """The voice saved before this run, as the step would show it.

    Attributes:
        preset: The Service radio to preselect. ``VOICE_PRESET_NONE`` when
            the voice replies use belongs to a provider this step does not set
            up, or when the saved OpenAI-slot voice is the old wizard's
            unspeakable write.
        draft: The controls' values, including the "Use as default" box,
            which is True when this is the voice replies use.
        other_provider: The provider that reads replies when it is not one
            this step sets up (e.g. ``kokoro``), kept as it is.
        slot_preset: The Service radio the saved OpenAI-compatible endpoint
            maps to, or "" when none is saved.
        legacy: The saved endpoint is the old wizard's PocketTTS address.
    """

    preset: str
    draft: vs.VoiceSetupDraft
    other_provider: str = ""
    slot_preset: str = ""
    legacy: bool = False


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

    The runtime reads replies with ``default_provider``, falling back to the
    OpenAI-compatible slot when none is saved. So a slot endpoint with no
    ``default_provider`` IS the reply voice (review round 1, F1), while one
    saved beside another provider's default is not the current voice: that
    provider is, and the shared default axes are its own (F3).

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
    own_axes = provider in {"", "openai"}
    other = SavedVoice(vs.VOICE_PRESET_NONE, _blank_draft(), other_provider=provider)
    if not base_url and provider != "openai":
        return other if provider else None
    try:
        endpoint = normalize_openai_compatible_endpoint(
            base_url or vs.OFFICIAL_OPENAI_TTS_ENDPOINT
        )
        auth = normalize_openai_authentication_mode(
            table.get("OPENAI_AUTH_MODE"), endpoint=endpoint
        ).value
    except (TypeError, ValueError):
        return None if own_axes else other
    slot_preset = _openai_slot_preset(endpoint.speech_url)
    # Any pocket-tts /tts address, not only the preset's port: tts-1-hd /
    # shimmer / mp3 there is a draft it can never speak (G8-R2-F2).
    defaults = (
        (vs.POCKET_TTS_MODEL, vs.POCKET_TTS_VOICE, "wav")
        if is_pocket_tts_native_url(endpoint.speech_url)
        else _RUNTIME_FALLBACK
    )
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
        use_as_default=own_axes,
    )
    if not own_axes:
        return SavedVoice(
            vs.VOICE_PRESET_NONE,
            draft,
            other_provider=provider,
            slot_preset=slot_preset,
        )
    if endpoint.speech_url == LEGACY_POCKET_TTS_ENDPOINT and auth == "none":
        return SavedVoice(
            vs.VOICE_PRESET_NONE, draft, slot_preset=slot_preset, legacy=True
        )
    return SavedVoice(slot_preset, draft, slot_preset=slot_preset)


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


def default_box_locked(preset: str, table: object) -> bool:
    """Whether "Use this voice when Chatbook reads replies aloud" is forced on.

    TASK-34100.8 review round 1 (F1 / G8-V1-F1). The OpenAI-compatible slot
    has one endpoint, and while it reads replies a save there IS the reply
    voice. An unticked box used to save it anyway while the box, the Summary
    and the User Guide said it was "not the default voice"; so for those
    services the box is ticked and cannot be unticked.

    Review round 2 (R2-F2): the same holds for a saved OmniVoice reply voice.
    OmniVoice's save only ever makes it the default, so an unticked Next
    posted nothing and OmniVoice kept reading replies while the box said it
    would not. Picking another service is how to change it.

    Args:
        preset: The selected Service radio.
        table: The raw ``[app_tts]`` table.
    """
    if preset == vs.VOICE_PRESET_OMNIVOICE:
        return (
            isinstance(table, Mapping)
            and _text(table, "default_provider") == vs.VOICE_PRESET_OMNIVOICE
        )
    return preset in OPENAI_SLOT_PRESETS and reply_voice_uses_openai_slot(table)


def reply_voice_name(saved: SavedVoice | None) -> str:
    """The provider reading replies when it is not the OpenAI-compatible slot.

    Review round 2 (R2-F1): over a saved OmniVoice voice the help line named
    nobody ("Replies will use this voice instead of .").

    Returns:
        "OmniVoice", another provider's id (e.g. ``kokoro``), or "" when the
        slot itself reads replies or nothing is saved.
    """
    if saved is None:
        return ""
    if saved.preset == vs.VOICE_PRESET_OMNIVOICE:
        return _PRESET_NAMES[vs.VOICE_PRESET_OMNIVOICE]
    return saved.other_provider


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
    if saved.other_provider:
        return saved.other_provider
    if saved.legacy:
        # One name for the old wizard's unspeakable write, wherever it shows.
        return f"PocketTTS at {_LEGACY_HOST} (from an earlier setup)"
    name = service_name(saved.slot_preset or saved.preset, saved.draft.endpoint)
    return f"{name} · {saved.draft.model_id} · {saved.draft.voice_id}"


def current_voice_copy(saved: SavedVoice) -> str:
    """The re-run status line naming the voice an untouched Next keeps."""
    if saved.other_provider:
        return (
            f"Current voice: {saved.other_provider} — kept as it is; nothing is "
            "saved here. Change it in Settings ▸ Speech & TTS."
        )
    if saved.legacy:
        return (
            f"An earlier setup saved PocketTTS at {_LEGACY_HOST}, an address "
            "pocket-tts doesn't serve, so that voice can't speak. Pick a "
            "service to replace it; Next alone leaves it as it is."
        )
    return f"Current voice: {voice_label(saved)} — unchanged unless you edit it."


def legacy_summary_detail() -> str:
    """The Summary's Voice detail for the old wizard's unspeakable write."""
    return (
        f"PocketTTS at {_LEGACY_HOST} (from an earlier setup) can't speak — "
        "set a voice in Settings ▸ Speech & TTS"
    )


#: What each named service offers under Advanced: (models, voices, formats).
#: Picking among these keeps the service; anything else makes it Custom.
_PRESET_CHOICES: Mapping[
    str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
] = {
    vs.VOICE_PRESET_POCKET_TTS: ((vs.POCKET_TTS_MODEL,), POCKET_TTS_VOICES, ("wav",)),
    vs.VOICE_PRESET_OFFICIAL_OPENAI: (
        ("tts-1", "tts-1-hd"),
        vs.OFFICIAL_OPENAI_TTS_VOICES,
        vs.RESPONSE_FORMATS,
    ),
}


def voices_for(preset: str) -> tuple[str, ...]:
    """The Voice picker's list for a service (Custom offers only Other…)."""
    return _PRESET_CHOICES.get(preset, ((), (), ()))[1]


def formats_for(preset: str) -> tuple[str, ...]:
    """The Format picker's list for a service."""
    return _PRESET_CHOICES.get(preset, ((), (), vs.RESPONSE_FORMATS))[2]


def draft_matches_preset(draft: vs.VoiceSetupDraft, preset: str) -> bool:
    """Whether the Advanced fields still describe ``preset``.

    TASK-34100.8 (new-voice-speech-02): the endpoint and authentication must
    be the preset's own, and model, voice and format one it offers.
    """
    if preset not in _PRESET_CHOICES:
        return True
    expected = vs.apply_voice_preset(draft, preset)
    models, voices, formats = _PRESET_CHOICES[preset]
    return (
        draft.endpoint.strip() == expected.endpoint
        and draft.authentication_mode == expected.authentication_mode
        and draft.model_id.strip() in models
        and draft.voice_id.strip() in voices
        and draft.response_format in formats
    )


def draft_from_checkpoint(
    values: Mapping[str, object], fallback: vs.VoiceSetupDraft
) -> vs.VoiceSetupDraft:
    """Rebuild a draft from a resume checkpoint's non-secret Voice values."""

    def text(key: str, default: str) -> str:
        value = values.get(key, default)
        return value if isinstance(value, str) else default

    speed = values.get("speed", fallback.speed)
    return vs.VoiceSetupDraft(
        endpoint=text("endpoint", fallback.endpoint),
        authentication_mode=text("authentication_mode", fallback.authentication_mode),
        model_id=text("model_id", fallback.model_id),
        voice_id=text("voice_id", fallback.voice_id),
        response_format=text("response_format", fallback.response_format),
        speed=float(speed) if isinstance(speed, (int, float)) else fallback.speed,
        sample_text=text("sample_text", fallback.sample_text),
        use_as_default=bool(values.get("use_as_default", fallback.use_as_default)),
    )


__all__ = [
    "LEGACY_POCKET_TTS_ENDPOINT",
    "OPENAI_SLOT_PRESETS",
    "SavedVoice",
    "current_voice_copy",
    "default_box_locked",
    "draft_from_checkpoint",
    "draft_matches_preset",
    "formats_for",
    "initial_voice_draft",
    "legacy_summary_detail",
    "raw_app_tts",
    "reply_voice_name",
    "reply_voice_uses_openai_slot",
    "saved_voice_from_config",
    "service_name",
    "should_persist_voice_config",
    "voice_label",
    "voices_for",
]
