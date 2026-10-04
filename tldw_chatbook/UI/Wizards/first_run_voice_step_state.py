"""Pure state and validation for the first-run Voice setup step."""

from __future__ import annotations

import math
import secrets
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal
from urllib.parse import urlsplit

import httpx

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.omnivoice_artifact_catalog import OMNIVOICE_SEED_LIMIT
from tldw_chatbook.TTS.openai_compatible_config import (
    is_loopback_openai_compatible_endpoint,
    normalize_openai_authentication_mode,
    normalize_openai_compatible_endpoint,
)
from tldw_chatbook.TTS.pocket_tts_native import (
    POCKET_TTS_DEFAULT_ENDPOINT,
    POCKET_TTS_WAV_ONLY_COPY,
    is_pocket_tts_native_url,
    pocket_tts_form,
    repair_streamed_wav,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    ProcessProviderTestEvidenceStore,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    ProviderTestFingerprint,
)

#: TASK-34100.8: 'No voice for now' -- the step writes nothing.
VOICE_PRESET_NONE = "none"
VOICE_PRESET_POCKET_TTS = "pocket_tts"
VOICE_PRESET_OFFICIAL_OPENAI = "official_openai"
VOICE_PRESET_CUSTOM = "custom"
VOICE_PRESET_OMNIVOICE = "omnivoice"

#: TASK-34100.8: pocket-tts's own server speaks POST /tts on port 8000; the
#: OpenAI route this preset used (:8765/v1/audio/speech) answers 404 there.
POCKET_TTS_ENDPOINT = POCKET_TTS_DEFAULT_ENDPOINT
OFFICIAL_OPENAI_TTS_ENDPOINT = "https://api.openai.com/v1/audio/speech"
POCKET_TTS_MODEL = "pocket-tts"
POCKET_TTS_VOICE = "alba"
OFFICIAL_OPENAI_TTS_MODEL = "tts-1-hd"
OFFICIAL_OPENAI_TTS_VOICE = "shimmer"

_RESPONSE_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav"})
RESPONSE_FORMATS: tuple[str, ...] = ("mp3", "opus", "aac", "flac", "wav")
#: The voices the OpenAI backend accepts on the official endpoint (any other
#: name is replaced with "alloy" there, so the picker offers only these).
OFFICIAL_OPENAI_TTS_VOICES: tuple[str, ...] = (
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
)
DEFAULT_SAMPLE_TEXT = "Hello from Chatbook."
_MAX_IDENTIFIER_CHARACTERS = 512
replace_draft = replace


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


def validate_voice_setup_draft(
    draft: VoiceSetupDraft, *, require_sample: bool = True
) -> VoiceSetupValidation:
    """Validate configuration without requiring network reachability.

    Args:
        draft: The Voice draft.
        require_sample: Whether the sample text must be valid. Only the test
            sends it; a save falls back to the default sample (TASK-34100.8).
    """

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
    elif draft.response_format != "wav" and is_pocket_tts_native_url(
        normalized_endpoint
    ):
        errors.append(POCKET_TTS_WAV_ONLY_COPY)
    if not 0.25 <= draft.speed <= 4.0:
        errors.append("Speed must be between 0.25 and 4.0.")
    if require_sample:
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
    include_voice_axes: bool = False,
    credential: str | None = None,
) -> STTSSettingsSaveEvent:
    """Build the canonical global settings event for one valid Voice draft.

    TASK-34100.8 (voice-speech-01). With "Use as default" ticked, the draft's
    own model, voice, format and speed become the default voice. Unticked, no
    default selection is written (``persist_default_preferences=False`` stops
    the handler materializing the current settings as defaults). The
    OpenAI-compatible slot keeps one model/voice/format for its endpoint, in
    the shared default axes, so when that slot is the voice replies use
    (``include_voice_axes``) they travel with the endpoint even unticked: a
    PocketTTS URL is never left paired with tts-1-hd / shimmer / mp3.

    Args:
        draft: A valid Voice draft (a blank sample is allowed).
        request_id: Correlates the save result.
        reply_to: Widget receiving the save result.
        include_voice_axes: Write model/voice/format/speed on an unticked save.
        credential: An OpenAI key the step staged; written where Settings
            writes it (``api_settings.openai.api_key``).
    """

    validation = validate_voice_setup_draft(draft, require_sample=False)
    if not validation.configuration_valid or validation.normalized_endpoint is None:
        raise ValueError("Voice setup configuration is invalid")
    settings: dict[str, object] = {
        "OPENAI_BASE_URL": validation.normalized_endpoint,
        "OPENAI_AUTH_MODE": draft.authentication_mode,
    }
    if include_voice_axes and not draft.use_as_default:
        settings.update(
            default_model=_identifier(draft.model_id, "Model"),
            default_voice=_identifier(draft.voice_id, "Voice"),
            default_format=draft.response_format,
            default_speed=draft.speed,
        )
    if credential:
        settings["openai_api_key"] = credential
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
        settings,
        preferences=preferences,
        request_id=request_id,
        reply_to=reply_to,
        commit_defaults_after_handoff=draft.use_as_default,
        persist_default_preferences=draft.use_as_default,
        # task-32266: the step awaits this save's result and renders every
        # outcome itself, so the app-level toast adds nothing and lands over
        # the docked actions of whichever step the wizard has advanced to.
        notify_outcome=False,
    )


class VoiceSampleError(ValueError):
    """A failed Voice sample, classified so the step can name the cause.

    TASK-34100.8 (voice-speech-03): every failure used to collapse into
    "Not tested yet — the sample failed." ``kind`` is one of ``not_running``
    (nothing listening on a local address), ``unreachable``, ``key_rejected``,
    ``no_endpoint``, ``bad_request``, ``http_status``, ``timeout``,
    ``not_audio`` or ``too_large``.
    """

    def __init__(
        self,
        kind: str,
        *,
        host: str = "",
        status_code: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(f"Voice sample failed ({kind})")
        self.kind = kind
        self.host = host
        self.status_code = status_code
        self.timeout_seconds = timeout_seconds


def endpoint_host(url: str) -> str:
    """Host and port only: a URL's userinfo or path never reaches the screen."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{parts.port}" if parts.port else host


def _status_failure(status_code: int, host: str) -> VoiceSampleError:
    kind = (
        "key_rejected"
        if status_code in {401, 403}
        else "no_endpoint"
        if status_code in {404, 405}
        else "bad_request"
        if status_code in {400, 422}
        else "http_status"
    )
    return VoiceSampleError(kind, host=host, status_code=status_code)


async def run_voice_sample(
    draft: VoiceSetupDraft,
    *,
    credential: str | None = None,
    max_response_bytes: int = 8 * 1024 * 1024,
    timeout_seconds: float = 20.0,
) -> VoiceSampleResult:
    """Send one exact sample and accept playable audio only.

    An endpoint whose path is ``/tts`` speaks pocket-tts's native API (form
    fields, streamed WAV); anything else the OpenAI speech API.

    Raises:
        ValueError: The draft is invalid or a needed key is missing.
        VoiceSampleError: The request failed; ``kind`` names the cause.
    """

    validation = validate_voice_setup_draft(draft)
    if not validation.configuration_valid or validation.normalized_endpoint is None:
        raise ValueError("Voice setup configuration is invalid")
    if type(max_response_bytes) is not int or max_response_bytes <= 0:
        raise ValueError("Voice sample response bound is invalid")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("Voice sample timeout is invalid")
    url = validation.normalized_endpoint
    host = endpoint_host(url)
    native = is_pocket_tts_native_url(url)
    headers = {"Accept": f"audio/{draft.response_format}"}
    if draft.authentication_mode == "api_key":
        if type(credential) is not str or not credential:
            raise ValueError(
                "An existing OpenAI API key is required to test this voice."
            )
        headers["Authorization"] = f"Bearer {credential}"
    text = validate_voice_sample_text(draft.sample_text)
    voice = _identifier(draft.voice_id, "Voice")
    body_kwargs: dict[str, object] = (
        {"data": pocket_tts_form(text, voice)}
        if native
        else {
            "json": {
                "input": text,
                "model": _identifier(draft.model_id, "Model"),
                "voice": voice,
                "response_format": draft.response_format,
                "speed": draft.speed,
            }
        }
    )
    try:
        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(float(timeout_seconds)),
                follow_redirects=False,
            ) as client,
            client.stream("POST", url, headers=headers, **body_kwargs) as response,
        ):
            if not 200 <= response.status_code < 300:
                raise _status_failure(response.status_code, host)
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as error:
                    raise VoiceSampleError("not_audio", host=host) from error
                if not 0 < declared_length <= max_response_bytes:
                    raise VoiceSampleError("too_large", host=host)
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > max_response_bytes:
                    raise VoiceSampleError("too_large", host=host)
                chunks.append(chunk)
            body = b"".join(chunks)
            content_type = response.headers.get("Content-Type", "")
    except httpx.TimeoutException:
        raise VoiceSampleError(
            "timeout", host=host, timeout_seconds=float(timeout_seconds)
        ) from None
    except httpx.ConnectError:
        local = is_loopback_openai_compatible_endpoint(
            normalize_openai_compatible_endpoint(url)
        )
        raise VoiceSampleError(
            "not_running" if local else "unreachable", host=host
        ) from None
    except httpx.TransportError:
        raise VoiceSampleError("unreachable", host=host) from None
    if native:
        body = repair_streamed_wav(body)

    fingerprint = ProviderTestFingerprint(
        provider_id="openai",
        normalized_fields=(
            ("authentication_mode", draft.authentication_mode),
            ("base_url", url),
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
        raise VoiceSampleError("not_audio", host=host)
    return VoiceSampleResult(
        body=body,
        content_type=content_type,
        response_format=draft.response_format,
        playable=True,
    )


OMNIVOICE_MODEL_ID = "omnivoice-int8hq"
OMNIVOICE_VOICE_ID = "default"

OMNIVOICE_PATH_INVALID_COPY = (
    "Your OmniVoice model folder setting points to a missing or incomplete model. "
    "Fix or clear [OmniVoiceSettings] model_root (or OMNIVOICE_MODEL_ROOT), then "
    "run setup again from Settings ▸ Diagnostics ▸ Run Setup Wizard."
)
OMNIVOICE_ENGINE_MISSING_COPY = (
    'OmniVoice needs its local engine: pip install "tldw_chatbook[omnivoice_tts]", '
    "then run setup again from Settings ▸ Diagnostics ▸ Run Setup Wizard."
)
OMNIVOICE_MODEL_MISSING_COPY = "Downloads the OmniVoice model (1.1 GB, one time)."
# task-1 measured SEED_STABLE = False: a fixed seed does not keep the voice
# stable across sentences, so the ready copy points users at Voice Cloning.
OMNIVOICE_READY_COPY = (
    "OmniVoice is installed — runs offline on this computer. Replies may vary "
    "in voice until you create a voice profile in Voice Cloning."
)
OMNIVOICE_CHECKING_COPY = "Checking the OmniVoice model…"
OMNIVOICE_GENERATING_COPY = "Generating locally (first run loads the model)…"
OMNIVOICE_SAMPLE_FAILED_COPY = (
    "Couldn't play a test sample — you can still save and test later in Speech Lab."
)
OMNIVOICE_DEFAULT_WITHOUT_MODEL_COPY = (
    "Install the OmniVoice model first, or uncheck Use as default."
)


def choose_omnivoice_seed(
    existing: object, *, randbelow: Callable[[int], int] = secrets.randbelow
) -> int:
    """Return the seed that fixes OmniVoice's default voice.

    A valid configured seed is kept so a re-run never changes the voice.

    Args:
        existing: ``[OmniVoiceSettings] seed`` as configured (any type).
        randbelow: Random source (injected by tests).

    Returns:
        A non-negative 31-bit integer seed.
    """
    if type(existing) is int and 0 <= existing < OMNIVOICE_SEED_LIMIT:
        return existing
    return randbelow(OMNIVOICE_SEED_LIMIT)


def build_omnivoice_save_event(
    *,
    speed: float,
    seed: int,
    request_id: int | None = None,
    reply_to: object | None = None,
) -> STTSSettingsSaveEvent:
    """Build the save event that makes OmniVoice the default voice.

    Args:
        speed: Default speaking rate.
        seed: The voice seed (from ``choose_omnivoice_seed``).
        request_id: Correlates the save result.
        reply_to: Widget receiving the save result.

    Returns:
        A settings event carrying only ``OMNIVOICE_SEED`` plus defaults.
    """
    return STTSSettingsSaveEvent(
        {"OMNIVOICE_SEED": seed},
        preferences=TTSPreferencesSnapshot(
            provider_id=VOICE_PRESET_OMNIVOICE,
            model_mode="exact",
            model_id=OMNIVOICE_MODEL_ID,
            voice_mode="exact",
            voice_id=OMNIVOICE_VOICE_ID,
            response_format="wav",
            speed=speed,
        ),
        request_id=request_id,
        reply_to=reply_to,
        commit_defaults_after_handoff=True,
        notify_outcome=False,
    )


async def run_omnivoice_sample(
    text: str,
    *,
    speed: float,
    seed: int,
    service: object | None = None,
    max_response_bytes: int = 8 * 1024 * 1024,
) -> VoiceSampleResult:
    """Synthesize one sample through the app's shared TTS service.

    Uses the same legacy request path as briefing audio, so the cached
    backend is reused (no second engine) and the seed matches the save.

    Args:
        text: Sample text (1–500 characters after trimming).
        speed: Speaking rate.
        seed: The voice seed the save will persist.
        service: TTS service (defaults to the app's bound service).
        max_response_bytes: Upper bound on the returned audio.

    Returns:
        A playable WAV sample.

    Raises:
        ValueError: Invalid text, oversize audio, or non-WAV audio.
    """
    from tldw_chatbook.TTS.legacy_request_builder import build_legacy_speech_request

    request, internal_model_id = build_legacy_speech_request(
        provider_id=VOICE_PRESET_OMNIVOICE,
        model_id=OMNIVOICE_MODEL_ID,
        voice=OMNIVOICE_VOICE_ID,
        text=validate_voice_sample_text(text),
        response_format="wav",
        speed=speed,
    )
    request.extra_params = {"seed": seed}
    if service is None:
        from tldw_chatbook.TTS.TTS_Generation import get_tts_service

        service = await get_tts_service()
    chunks: list[bytes] = []
    total = 0
    async for chunk in service.generate_audio_stream(request, internal_model_id):
        total += len(chunk)
        if total > max_response_bytes:
            raise ValueError("The TTS sample exceeded the response limit.")
        chunks.append(bytes(chunk))
    body = b"".join(chunks)
    if not (body.startswith(b"RIFF") and body[8:12] == b"WAVE"):
        raise ValueError("OmniVoice returned audio that could not be played.")
    return VoiceSampleResult(
        body=body, content_type="audio/wav", response_format="wav", playable=True
    )
