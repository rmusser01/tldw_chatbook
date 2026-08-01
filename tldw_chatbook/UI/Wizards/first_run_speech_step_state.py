"""Pure state contracts for the first-run wizard's Speech transcription step.

No Textual imports, no I/O -- mirrors first_run_setup_state.py's own "pure
transform" contract (the wizard step owns rendering/I/O; this module owns
every decision).

TASK-1301 / ADR-025: language and precision enumeration is sourced from
``tldw_chatbook.STT.routing`` -- the authoritative built-in STT policy and
catalog -- never a hand-rolled list. ``selectable`` is a SEPARATE concern:
it reflects what a curated ``Model_Artifacts`` descriptor can actually
download today (``tldw_chatbook.Model_Artifacts.curated_registry``), which
the wizard step resolves and passes in. Today only Parakeet v2 (English,
INT8) has a curated descriptor, so it is the only selectable combination;
every other catalog-declared language/precision renders present but
disabled, honestly reflecting the STT policy without offering a control
nothing can actually fulfil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tldw_chatbook.STT.routing import RoutingPolicy, build_builtin_registry

TRANSCRIPTION_SECTION = "transcription"

#: ADR-025's validated Parakeet v3 language set, English excluded --
#: RoutingPolicy.__post_init__ rejects "en"/"auto" here on purpose (English
#: is v2's language, not part of the "additional" v3 set).
_V3_LANGUAGES: frozenset[str] = frozenset(
    {
        "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu", "it",
        "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
    }
)

#: Display names for every language code the STT policy currently declares
#: for Parakeet (v2's "en" plus the validated v3 set above).
LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "en": "English",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "ru": "Russian",
    "uk": "Ukrainian",
}

_ROUTING_POLICY = RoutingPolicy(validated_v3_languages=_V3_LANGUAGES)
_REGISTRY = build_builtin_registry(_ROUTING_POLICY)


def routing_policy() -> RoutingPolicy:
    """Return the shared STT routing policy this step's options are built from.

    Returns:
        The immutable ``RoutingPolicy`` sealing the STT catalog this module
        reads from (provider/model identities and the validated v3 language
        set).
    """
    return _ROUTING_POLICY


@dataclass(frozen=True)
class SpeechLanguageOption:
    """One language choice sourced from the canonical STT policy/catalog."""

    code: str
    display_name: str
    model_id: str
    selectable: bool


@dataclass(frozen=True)
class SpeechPrecisionOption:
    """One precision choice sourced from the canonical STT policy/catalog."""

    value: str
    display_name: str
    selectable: bool


def speech_language_options(
    *, curated_model_ids: frozenset[str]
) -> tuple[SpeechLanguageOption, ...]:
    """Every language the STT catalog declares for Parakeet.

    Args:
        curated_model_ids: Model ids with a registered curated descriptor
            (``Model_Artifacts.curated_registry.curated_registry()``); only
            options whose model has a curated descriptor are ``selectable``.

    Returns:
        English (Parakeet v2) first, then the validated Parakeet v3
        languages in sorted code order.
    """
    policy = _ROUTING_POLICY
    v2 = _REGISTRY.model(policy.parakeet_provider_id, policy.parakeet_v2_model_id)
    v3 = _REGISTRY.model(policy.parakeet_provider_id, policy.parakeet_v3_model_id)
    assert v2 is not None and v3 is not None  # sealed by build_builtin_registry

    options = [
        SpeechLanguageOption(
            code="en",
            display_name=LANGUAGE_DISPLAY_NAMES["en"],
            model_id=v2.model_id,
            selectable=v2.model_id in curated_model_ids,
        )
    ]
    options.extend(
        SpeechLanguageOption(
            code=code,
            display_name=LANGUAGE_DISPLAY_NAMES.get(code, code),
            model_id=v3.model_id,
            selectable=v3.model_id in curated_model_ids,
        )
        for code in sorted(v3.capabilities.languages)
    )
    return tuple(options)


def speech_precision_options(
    *, curated_precisions: frozenset[str]
) -> tuple[SpeechPrecisionOption, ...]:
    """Every precision the STT catalog declares for Parakeet v2.

    Args:
        curated_precisions: Precisions with a registered curated Parakeet v2
            descriptor; only these are ``selectable``.

    Returns:
        The declared precisions, the model's own default precision first.
    """
    policy = _ROUTING_POLICY
    v2 = _REGISTRY.model(policy.parakeet_provider_id, policy.parakeet_v2_model_id)
    assert v2 is not None
    return tuple(
        SpeechPrecisionOption(
            value=precision,
            display_name=precision.upper(),
            selectable=precision in curated_precisions,
        )
        for precision in sorted(
            v2.capabilities.precisions, key=lambda p: (p != v2.default_precision, p)
        )
    )


def recommended_speech_selection() -> tuple[str, str, str]:
    """Return ``(provider_id, model_id, language)`` for the recommended default.

    Returns:
        The Parakeet ONNX provider id, Parakeet v2 model id, and "en" --
        the only combination with a curated, downloadable artifact today.
    """
    policy = _ROUTING_POLICY
    return policy.parakeet_provider_id, policy.parakeet_v2_model_id, "en"


def build_speech_transcription_commit(
    *, provider_id: str, model_id: str, language: str
) -> dict[str, dict[str, Any]]:
    """Mutation for the speech transcription step.

    Args:
        provider_id: STT provider id to persist as ``transcription.default_provider``.
        model_id: Model id to persist as ``transcription.default_model``.
        language: Language to persist as ``transcription.default_language``.

    Returns:
        The section/value mapping to persist under ``transcription``.
    """
    return {
        TRANSCRIPTION_SECTION: {
            "default_provider": provider_id,
            "default_model": model_id,
            "default_language": language,
        }
    }


@dataclass(frozen=True)
class SpeechPrefill:
    """Persisted ``[transcription]`` defaults for re-run prefill (no secrets)."""

    provider_id: str = ""
    model_id: str = ""
    language: str = ""


def read_speech_prefill(app_config: Mapping[str, object]) -> SpeechPrefill:
    """Read the persisted transcription defaults straight from config.

    Deliberately does NOT treat the shipped ``[transcription]`` template
    defaults (``default_provider="faster-whisper"`` or a platform MLX
    provider, ``default_model="distil-large-v3"``) as "configured by this
    step" -- callers that need that distinction (e.g. the Summary row)
    compare ``provider_id`` against ``routing_policy().parakeet_provider_id``
    specifically, the one value the shipped template never defaults to.

    Args:
        app_config: Loaded app configuration.

    Returns:
        The persisted provider/model/language, or a blank prefill when the
        section is absent or malformed.
    """
    section = app_config.get(TRANSCRIPTION_SECTION)
    if not isinstance(section, Mapping):
        return SpeechPrefill()
    return SpeechPrefill(
        provider_id=str(section.get("default_provider") or ""),
        model_id=str(section.get("default_model") or ""),
        language=str(section.get("default_language") or ""),
    )


__all__ = [
    "LANGUAGE_DISPLAY_NAMES",
    "TRANSCRIPTION_SECTION",
    "SpeechLanguageOption",
    "SpeechPrecisionOption",
    "SpeechPrefill",
    "build_speech_transcription_commit",
    "read_speech_prefill",
    "recommended_speech_selection",
    "routing_policy",
    "speech_language_options",
    "speech_precision_options",
]
