"""Public, per-selection builder for legacy (OpenAI-shaped) TTS speech requests.

Legacy (non-``audio_cpp``) TTS providers are reached through an
OpenAI-compatible request shape plus an "internal model id" that
:mod:`tldw_chatbook.TTS.legacy_bridge` uses to route the request to the right
backend. :func:`build_legacy_speech_request` is that logic promoted out of
:func:`tldw_chatbook.TTS.request_admission._legacy_request`, which derived
everything from the app's single global ``TTSPreferencesSnapshot``. That
coupling only supports one voice for the whole app; briefing scripts need one
request per speaker, so this builder takes the selection as explicit fields
instead, letting a caller build one request per speaker in a roster.

Cross-reference (TASK-1393 pact convention, greppable both ways): a third,
deliberately different copy of this id-derivation logic lives in
``Event_Handlers/STTS_Events/stts_events.py``'s ``_legacy_internal_model_id``,
which derives kokoro's onnx/pytorch suffix from live playground options and
alltalk's suffix from the requested model id. That copy is NOT interchangeable
with this one — see the comment above it.
"""

from __future__ import annotations

from typing import Literal, cast

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest

_AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
_VALID_AUDIO_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})
_LEGACY_MODEL_OVERRIDES = {
    "elevenlabs": "elevenlabs",
    "kokoro": "kokoro",
    "chatterbox": "chatterbox",
    "alltalk": "alltalk",
}
_LEGACY_FORMAT_OVERRIDES = {
    "elevenlabs": "mp3",
    "kokoro": "wav",
    "chatterbox": "wav",
    "alltalk": "wav",
}


def build_legacy_speech_request(
    *,
    provider_id: str,
    model_id: str,
    voice: str,
    text: str,
    response_format: str = "wav",
    speed: float = 1.0,
) -> tuple[OpenAISpeechRequest, str]:
    """Build one legacy speech request for an explicit provider/voice selection.

    Unlike the app-wide preferences snapshot used for the single-voice chat
    experience, a briefing roster needs one request per speaker, each with its
    own provider, model, and voice. This builder takes those selections as
    explicit arguments so a caller can make one call per speaker rather than
    once per app.

    Args:
        provider_id: Exact legacy provider identifier (for example
            ``"openai"``, ``"elevenlabs"``, ``"kokoro"``, ``"chatterbox"``,
            ``"higgs"``, or ``"alltalk"``). Any other non-empty value is
            accepted and passed through: its internal model id becomes the
            (normalized) model id itself.
        model_id: Provider-reported model identifier requested for this
            speaker. Some providers override this value unconditionally (see
            ``_LEGACY_MODEL_OVERRIDES``).
        voice: Exact voice identifier to use. Legacy providers cannot resolve
            a "server default" voice, so this must be a non-empty string.
        text: Text to synthesize.
        response_format: Requested audio container/codec. Falls back to
            ``"wav"`` when not one of the supported formats. Some providers
            override this value unconditionally (see
            ``_LEGACY_FORMAT_OVERRIDES``).
        speed: Playback speed multiplier passed through to the request.

    Returns:
        A tuple of the constructed ``OpenAISpeechRequest`` and the internal
        model id used to route the request through the legacy compatibility
        bridge (:func:`tldw_chatbook.TTS.legacy_bridge.resolve_legacy_route`).

    Raises:
        ValueError: If ``voice`` is empty (legacy providers require an exact
            voice) or ``provider_id`` is empty (no provider was selected).
    """
    if not voice:
        raise ValueError("Legacy TTS providers require an exact voice")
    if not provider_id:
        raise ValueError("Legacy TTS speech requests require a provider id")

    request_model = _LEGACY_MODEL_OVERRIDES.get(provider_id, model_id).lower()
    requested_format = _LEGACY_FORMAT_OVERRIDES.get(provider_id, response_format)
    normalized_format = requested_format.lower().strip()
    if normalized_format not in _VALID_AUDIO_FORMATS:
        normalized_format = "wav"

    request = OpenAISpeechRequest(
        model=request_model,
        input=text,
        voice=voice.lower(),
        response_format=cast(_AudioFormat, normalized_format),
        speed=speed,
    )
    if provider_id == "openai":
        internal_model_id = f"openai_official_{request.model}"
    elif provider_id == "elevenlabs":
        internal_model_id = f"elevenlabs_{request.model}"
    elif provider_id == "kokoro":
        internal_model_id = "local_kokoro_default_onnx"
    elif provider_id == "chatterbox":
        internal_model_id = "local_chatterbox_default"
    elif provider_id == "higgs":
        internal_model_id = "local_higgs_v2"
    elif provider_id == "alltalk":
        internal_model_id = "alltalk_default"
    else:
        internal_model_id = request.model
    return request, internal_model_id
