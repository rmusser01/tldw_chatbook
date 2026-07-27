"""Dependency-free routing policy for batch speech-to-text requests."""

from dataclasses import dataclass


PARAKEET_V2_MODEL = "nemo-parakeet-tdt-0.6b-v2"
PARAKEET_V3_MODEL = "nemo-parakeet-tdt-0.6b-v3"

_DEFAULT_PROVIDER = "default"
_FASTER_WHISPER_PROVIDER = "faster-whisper"
_PARAKEET_PROVIDER = "parakeet-onnx"
_AUTO_LANGUAGE = "auto"
_PARAKEET_V3_LANGUAGES = frozenset(
    {
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "it",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "es",
        "sv",
        "ru",
        "uk",
    }
)


class BatchSTTRoutingError(ValueError):
    """Raised when a requested batch STT provider cannot serve a request."""


@dataclass(frozen=True)
class BatchSTTRoute:
    """Resolved batch STT provider settings."""

    requested_provider: str
    provider: str
    model: str | None
    requested_language: str
    target_language: str | None
    precision: str
    local_files_only: bool
    reason: str


def resolve_batch_stt_route(
    *,
    provider: str | None,
    language: str | None,
    target_language: str | None = None,
    parakeet_defaults_enabled: bool = False,
) -> BatchSTTRoute:
    """Resolve a batch STT request to its permitted local implementation.

    Args:
        provider: Requested provider, or ``None`` for the semantic default.
        language: Requested source language, or ``None`` for English.
        target_language: Translation target, when requested.
        parakeet_defaults_enabled: Whether the Parakeet default promotion gate is open.

    Returns:
        The immutable routing decision.

    Raises:
        BatchSTTRoutingError: If the provider is unknown or an explicit Parakeet
            request cannot be fulfilled.
    """
    requested_provider = _normalize_provider(provider)
    requested_language = _normalize_language(language, default="en")
    normalized_target = _normalize_language(target_language, default=None)

    if requested_provider == _FASTER_WHISPER_PROVIDER:
        return _faster_whisper_route(
            requested_provider,
            requested_language,
            normalized_target,
            "explicit_faster_whisper",
        )

    if requested_provider == _PARAKEET_PROVIDER:
        return _parakeet_route(
            requested_provider,
            requested_language,
            normalized_target,
        )

    if requested_provider != _DEFAULT_PROVIDER:
        raise BatchSTTRoutingError(f"Unsupported batch STT provider: {requested_provider}")

    if not parakeet_defaults_enabled:
        return _faster_whisper_route(
            requested_provider,
            requested_language,
            normalized_target,
            "parakeet_promotion_gate_closed",
        )

    if normalized_target is not None:
        return _faster_whisper_route(
            requested_provider,
            requested_language,
            normalized_target,
            "translation_requires_faster_whisper",
        )
    if requested_language == _AUTO_LANGUAGE or requested_language not in _PARAKEET_V3_LANGUAGES:
        return _faster_whisper_route(
            requested_provider,
            requested_language,
            normalized_target,
            "language_requires_faster_whisper",
        )
    return _parakeet_route(requested_provider, requested_language, normalized_target)


def _normalize_provider(provider: str | None) -> str:
    return _DEFAULT_PROVIDER if provider is None or provider == "" else provider


def _normalize_language(language: str | None, *, default: str | None) -> str | None:
    if language is None or not language.strip():
        return default
    return language.strip().lower()


def _parakeet_route(
    requested_provider: str,
    requested_language: str,
    target_language: str | None,
) -> BatchSTTRoute:
    if target_language is not None:
        raise BatchSTTRoutingError(
            "Parakeet does not support translation. Retry with faster-whisper."
        )
    if requested_language == _AUTO_LANGUAGE or requested_language not in _PARAKEET_V3_LANGUAGES:
        raise BatchSTTRoutingError(
            "Parakeet does not support this language. Retry with faster-whisper."
        )

    model = PARAKEET_V2_MODEL if requested_language == "en" else PARAKEET_V3_MODEL
    return BatchSTTRoute(
        requested_provider=requested_provider,
        provider=_PARAKEET_PROVIDER,
        model=model,
        requested_language=requested_language,
        target_language=target_language,
        precision="int8",
        local_files_only=True,
        reason="parakeet_v2_english" if model == PARAKEET_V2_MODEL else "parakeet_v3_language",
    )


def _faster_whisper_route(
    requested_provider: str,
    requested_language: str,
    target_language: str | None,
    reason: str,
) -> BatchSTTRoute:
    return BatchSTTRoute(
        requested_provider=requested_provider,
        provider=_FASTER_WHISPER_PROVIDER,
        model=None,
        requested_language=requested_language,
        target_language=target_language,
        precision="int8",
        local_files_only=True,
        reason=reason,
    )
