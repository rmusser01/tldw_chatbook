"""Headless Console voice-dictation controller.

Deliberately free of Textual imports: the widget layer owns rendering and
threading policy, this module owns availability, provider resolution, and the
dictation state machine. That split is what makes the state machine unit
testable without a running app.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from loguru import logger

from ..config import get_cli_setting

# Capture backends, in preference order. AudioRecordingService picks between
# them itself; we only need to know whether at least one exists.
CAPTURE_MODULES: tuple[str, ...] = ("pyaudio", "sounddevice")

# Provider id -> import name. Only providers that run locally: the dictation
# service's privacy mode rejects everything else (see resolve()).
LOCAL_PROVIDER_MODULES: dict[str, str] = {
    "parakeet-mlx": "parakeet_mlx",
    "faster-whisper": "faster_whisper",
    "lightning-whisper-mlx": "lightning_whisper_mlx",
}

CAPTURE_REASON = "No microphone backend installed."
CAPTURE_REMEDY = (
    "Microphone support isn't installed. "
    "Install with: pip install 'tldw_chatbook[speech_recording]'"
)
PROVIDER_REASON = "No speech-to-text provider installed."
PROVIDER_REMEDY = (
    "No speech-to-text provider installed. "
    "Install with: pip install 'tldw_chatbook[transcription_faster_whisper]'"
)


@dataclass(frozen=True)
class Availability:
    """Whether dictation can run, and what to do about it if not."""

    ok: bool
    kind: str = "ok"  # "ok" | "missing-capture" | "missing-provider"
    reason: str = ""
    remedy: str = ""


def _module_installed(module_name: str) -> bool:
    """Return True when `module_name` is importable, without importing it.

    `find_spec` is required here rather than `optional_deps.check_dependency`,
    which really imports the module and would drag torch/NeMo into app start.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        # A namespace package with a broken parent raises rather than
        # returning None; treat that as "not usable".
        return False


def capture_available() -> bool:
    """Return True when at least one audio capture backend is installed."""
    return any(_module_installed(name) for name in CAPTURE_MODULES)


def installed_local_providers() -> tuple[str, ...]:
    """Return the local transcription providers that are actually installed."""
    return tuple(
        provider
        for provider, module_name in LOCAL_PROVIDER_MODULES.items()
        if _module_installed(module_name)
    )


def probe() -> Availability:
    """Report whether dictation is usable, distinguishing the two failures."""
    if not capture_available():
        logger.debug("Console dictation unavailable: no capture backend")
        return Availability(
            ok=False,
            kind="missing-capture",
            reason=CAPTURE_REASON,
            remedy=CAPTURE_REMEDY,
        )
    if not installed_local_providers():
        logger.debug("Console dictation unavailable: no transcription provider")
        return Availability(
            ok=False,
            kind="missing-provider",
            reason=PROVIDER_REASON,
            remedy=PROVIDER_REMEDY,
        )
    return Availability(ok=True)


DEFAULT_LANGUAGE = "en"


@dataclass(frozen=True)
class EffectiveConfig:
    """The transcription settings dictation will actually run with."""

    provider: str
    model: str | None
    language: str
    configured_provider: str
    was_overridden: bool


def resolve() -> EffectiveConfig | None:
    """Choose the provider before the dictation service gets the chance.

    `LazyLiveDictationService._initialize_streaming_transcriber` rewrites the
    provider to `parakeet-mlx` whenever privacy mode is on and the configured
    provider is not on its allowlist -- silently, and to an Apple-Silicon-only
    provider. Resolving here means the service is always handed a provider that
    is both local and installed, so that branch never fires.

    Returns:
        The settings to run with, or None when no local provider is installed.
    """
    installed = installed_local_providers()
    if not installed:
        return None

    configured = get_cli_setting("transcription", "provider", None) or get_cli_setting(
        "STTSettings", "default_stt_provider", ""
    )
    configured = str(configured or "")

    if configured in installed:
        provider = configured
    else:
        # Preference order is LOCAL_PROVIDER_MODULES' declaration order.
        provider = installed[0]
        if configured:
            logger.info(
                "Console dictation provider '{}' unavailable; using '{}'",
                configured,
                provider,
            )

    model = get_cli_setting("transcription", "model", None)
    language = get_cli_setting("transcription", "language", DEFAULT_LANGUAGE)

    return EffectiveConfig(
        provider=provider,
        model=str(model) if model else None,
        language=str(language or DEFAULT_LANGUAGE),
        configured_provider=configured,
        was_overridden=bool(configured) and provider != configured,
    )
