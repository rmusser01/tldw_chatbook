"""Pure state and validation for the first-run Voice setup step."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal
from urllib.parse import urlsplit

import httpx

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.openai_compatible_config import (
    is_loopback_openai_compatible_endpoint,
    normalize_openai_authentication_mode,
    normalize_openai_compatible_endpoint,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    ProcessProviderTestEvidenceStore,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    ProviderTestFingerprint,
)

VOICE_PRESET_POCKET_TTS = "pocket_tts"
VOICE_PRESET_OFFICIAL_OPENAI = "official_openai"
VOICE_PRESET_CUSTOM = "custom"

POCKET_TTS_ENDPOINT = "http://127.0.0.1:8765/v1/audio/speech"
OFFICIAL_OPENAI_TTS_ENDPOINT = "https://api.openai.com/v1/audio/speech"
POCKET_TTS_MODEL = "pocket-tts"
POCKET_TTS_VOICE = "alba"
OFFICIAL_OPENAI_TTS_MODEL = "tts-1-hd"
OFFICIAL_OPENAI_TTS_VOICE = "shimmer"

_RESPONSE_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav"})
_MAX_IDENTIFIER_CHARACTERS = 512


@dataclass(frozen=True, slots=True)
class VoiceSetupDraft:
    """Non-secret editable Voice configuration owned by onboarding."""

    endpoint: str
    authentication_mode: str
    model_id: str
    voice_id: str
    response_format: str
    speed: float
    sample_text: str
    use_as_default: bool = False

    def __post_init__(self) -> None:
        for value in (
            self.endpoint,
            self.authentication_mode,
            self.model_id,
            self.voice_id,
            self.response_format,
            self.sample_text,
        ):
            if type(value) is not str:
                raise TypeError("Voice setup text fields must be strings")
        if type(self.speed) not in {int, float} or isinstance(self.speed, bool):
            raise TypeError("Voice setup speed must be numeric")
        speed = float(self.speed)
        if not math.isfinite(speed):
            raise ValueError("Voice setup speed must be finite")
        object.__setattr__(self, "speed", speed)
        if type(self.use_as_default) is not bool:
            raise TypeError("Voice default choice must be boolean")


@dataclass(frozen=True, slots=True)
class VoiceSetupValidation:
    """Local validity is independent from process-scoped connection evidence."""

    configuration_valid: bool
    connection_state: Literal["needs_test", "verified"]
    normalized_endpoint: str | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VoiceSampleResult:
    """Bounded, playable process-local sample returned by a Voice test."""

    body: bytes
    content_type: str
    response_format: str
    playable: bool


def validate_voice_sample_text(value: object) -> str:
    """Return a trimmed sample bounded to 1 through 500 characters."""

    if type(value) is not str:
        raise ValueError("Sample text must contain 1 to 500 characters.")
    trimmed = value.strip()
    if not 1 <= len(trimmed) <= 500:
        raise ValueError("Sample text must contain 1 to 500 characters.")
    return trimmed


def _identifier(value: str, label: str) -> str:
    trimmed = value.strip()
    if not trimmed or len(trimmed) > _MAX_IDENTIFIER_CHARACTERS:
        raise ValueError(f"{label} is required.")
    if any(ord(character) < 32 or ord(character) == 127 for character in trimmed):
        raise ValueError(f"{label} is invalid.")
    return trimmed


def validate_voice_setup_draft(draft: VoiceSetupDraft) -> VoiceSetupValidation:
    """Validate configuration without requiring network reachability."""

    if type(draft) is not VoiceSetupDraft:
        raise TypeError("Voice setup draft is invalid")
    errors: list[str] = []
    normalized_endpoint: str | None = None
    try:
        if draft.authentication_mode not in {"none", "api_key"}:
            raise ValueError("Unsupported authentication mode")
        endpoint = normalize_openai_compatible_endpoint(draft.endpoint)
        normalize_openai_authentication_mode(
            draft.authentication_mode,
            endpoint=endpoint,
        )
        if (
            draft.authentication_mode == "api_key"
            and urlsplit(endpoint.origin).scheme == "http"
            and not is_loopback_openai_compatible_endpoint(endpoint)
        ):
            errors.append(
                "API key authentication requires HTTPS or a loopback HTTP endpoint."
            )
        normalized_endpoint = endpoint.speech_url
    except ValueError:
        errors.append(
            "Enter a valid OpenAI-compatible speech endpoint and authentication mode."
        )
    try:
        _identifier(draft.model_id, "Model")
    except ValueError as error:
        errors.append(str(error))
    try:
        _identifier(draft.voice_id, "Voice")
    except ValueError as error:
        errors.append(str(error))
    if draft.response_format not in _RESPONSE_FORMATS:
        errors.append("Choose a supported response format.")
    if not 0.25 <= draft.speed <= 4.0:
        errors.append("Speed must be between 0.25 and 4.0.")
    try:
        validate_voice_sample_text(draft.sample_text)
    except ValueError as error:
        errors.append(str(error))
    return VoiceSetupValidation(
        configuration_valid=not errors,
        connection_state="needs_test",
        normalized_endpoint=normalized_endpoint,
        errors=tuple(errors),
    )


def apply_voice_preset(draft: VoiceSetupDraft, preset: str) -> VoiceSetupDraft:
    """Apply one explicit preset without introducing credential material."""

    if type(draft) is not VoiceSetupDraft:
        raise TypeError("Voice setup draft is invalid")
    if preset == VOICE_PRESET_POCKET_TTS:
        return replace(
            draft,
            endpoint=POCKET_TTS_ENDPOINT,
            authentication_mode="none",
            model_id=POCKET_TTS_MODEL,
            voice_id=POCKET_TTS_VOICE,
            response_format="wav",
        )
    if preset == VOICE_PRESET_OFFICIAL_OPENAI:
        return replace(
            draft,
            endpoint=OFFICIAL_OPENAI_TTS_ENDPOINT,
            authentication_mode="api_key",
            model_id=OFFICIAL_OPENAI_TTS_MODEL,
            voice_id=OFFICIAL_OPENAI_TTS_VOICE,
            response_format="mp3",
        )
    if preset == VOICE_PRESET_CUSTOM:
        return draft
    raise ValueError("Unknown Voice setup preset")


def build_voice_setup_save_event(
    draft: VoiceSetupDraft,
    *,
    request_id: int | None = None,
    reply_to: object | None = None,
) -> STTSSettingsSaveEvent:
    """Build the canonical global settings event for one valid Voice draft."""

    validation = validate_voice_setup_draft(draft)
    if not validation.configuration_valid or validation.normalized_endpoint is None:
        raise ValueError("Voice setup configuration is invalid")
    preferences = (
        TTSPreferencesSnapshot(
            provider_id="openai",
            model_mode="exact",
            model_id=_identifier(draft.model_id, "Model"),
            voice_mode="exact",
            voice_id=_identifier(draft.voice_id, "Voice"),
            response_format=draft.response_format,
            speed=draft.speed,
        )
        if draft.use_as_default
        else None
    )
    return STTSSettingsSaveEvent(
        {
            "OPENAI_BASE_URL": validation.normalized_endpoint,
            "OPENAI_AUTH_MODE": draft.authentication_mode,
        },
        preferences=preferences,
        request_id=request_id,
        reply_to=reply_to,
        commit_defaults_after_handoff=draft.use_as_default,
    )


async def run_voice_sample(
    draft: VoiceSetupDraft,
    *,
    credential: str | None = None,
    max_response_bytes: int = 8 * 1024 * 1024,
    timeout_seconds: float = 20.0,
) -> VoiceSampleResult:
    """Send one exact OpenAI-compatible sample and accept playable audio only."""

    validation = validate_voice_setup_draft(draft)
    if not validation.configuration_valid or validation.normalized_endpoint is None:
        raise ValueError("Voice setup configuration is invalid")
    if type(max_response_bytes) is not int or max_response_bytes <= 0:
        raise ValueError("Voice sample response bound is invalid")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("Voice sample timeout is invalid")
    headers = {"Accept": f"audio/{draft.response_format}"}
    if draft.authentication_mode == "api_key":
        if type(credential) is not str or not credential:
            raise ValueError(
                "An existing OpenAI API key is required to test this voice."
            )
        headers["Authorization"] = f"Bearer {credential}"
    payload = {
        "input": validate_voice_sample_text(draft.sample_text),
        "model": _identifier(draft.model_id, "Model"),
        "voice": _identifier(draft.voice_id, "Voice"),
        "response_format": draft.response_format,
        "speed": draft.speed,
    }
    timeout = httpx.Timeout(float(timeout_seconds))
    async with (
        httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
        ) as client,
        client.stream(
            "POST",
            validation.normalized_endpoint,
            headers=headers,
            json=payload,
        ) as response,
    ):
        if not 200 <= response.status_code < 300:
            raise ValueError("The TTS service did not accept the sample request.")
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as error:
                raise ValueError(
                    "The TTS service returned invalid audio metadata."
                ) from error
            if not 0 < declared_length <= max_response_bytes:
                raise ValueError("The TTS sample exceeded the response limit.")
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_response_bytes:
                raise ValueError("The TTS sample exceeded the response limit.")
            chunks.append(chunk)
        body = b"".join(chunks)
        content_type = response.headers.get("Content-Type", "")

    fingerprint = ProviderTestFingerprint(
        provider_id="openai",
        normalized_fields=(
            ("authentication_mode", draft.authentication_mode),
            ("base_url", validation.normalized_endpoint),
            ("model_id", draft.model_id.strip()),
            ("response_format", draft.response_format),
            ("speed", str(draft.speed)),
            ("voice_id", draft.voice_id.strip()),
        ),
        saved_revision=0,
    )
    playable = ProcessProviderTestEvidenceStore().record_successful_sample(
        fingerprint,
        status_code=200,
        response_format=draft.response_format,
        body=body,
        content_type=content_type,
        max_bytes=max_response_bytes,
    )
    if not playable:
        raise ValueError("The TTS service returned audio that could not be played.")
    return VoiceSampleResult(
        body=body,
        content_type=content_type,
        response_format=draft.response_format,
        playable=True,
    )
