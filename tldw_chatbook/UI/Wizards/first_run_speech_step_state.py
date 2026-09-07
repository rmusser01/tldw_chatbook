"""Pure state contracts for the first-run wizard's Speech transcription step.

No Textual imports, no I/O -- mirrors first_run_setup_state.py's own "pure
transform" contract (the wizard step owns rendering/I/O; this module owns
every decision).

TASK-1301 / ADR-025: language and precision enumeration is sourced from
``tldw_chatbook.STT.routing`` -- the authoritative built-in STT policy and
catalog -- never a hand-rolled list. ``selectable`` is a separate concern:
it reflects the exact model/precision pairs admitted by the curated artifact
registry, which the wizard resolves and passes in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tldw_chatbook.STT.routing import (
    RoutingPolicy,
    build_builtin_registry,
    default_routing_policy,
)

TRANSCRIPTION_SECTION = "transcription"

#: Display names for every language code the STT policy currently declares
#: for Parakeet (v2's "en" plus ``tldw_chatbook.STT.routing``'s canonical
#: ``VALIDATED_V3_LANGUAGES`` set).
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

_ROUTING_POLICY = default_routing_policy()
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


@dataclass(frozen=True)
class SpeechSelection:
    """One exact first-run Parakeet configuration and managed artifact."""

    provider_id: str
    model_id: str
    language: str
    precision: str


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
    *,
    model_id: str,
    curated_selections: frozenset[tuple[str, str]],
) -> tuple[SpeechPrecisionOption, ...]:
    """Every precision declared for one selected Parakeet model.

    Args:
        model_id: Selected Parakeet model id.
        curated_selections: Exact ``(model_id, precision)`` pairs with a
            registered curated root descriptor.

    Returns:
        The declared precisions, the model's own default precision first.
    """
    policy = _ROUTING_POLICY
    model = _REGISTRY.model(policy.parakeet_provider_id, model_id)
    if model is None:
        return ()
    return tuple(
        SpeechPrecisionOption(
            value=precision,
            display_name=precision.upper(),
            selectable=(model_id, precision) in curated_selections,
        )
        for precision in sorted(
            model.capabilities.precisions,
            key=lambda p: (p != model.default_precision, p),
        )
    )


def recommended_speech_selection() -> SpeechSelection:
    """Return the exact recommended Parakeet v2 English INT8 selection.

    Returns:
        The default selection.
    """
    policy = _ROUTING_POLICY
    return SpeechSelection(
        provider_id=policy.parakeet_provider_id,
        model_id=policy.parakeet_v2_model_id,
        language="en",
        precision="int8",
    )


def resolve_speech_selection(
    *,
    selected_language: str,
    selected_precision: str,
    curated_selections: frozenset[tuple[str, str]],
) -> SpeechSelection | None:
    """Resolve pressed radios into one exact curated artifact selection.

    PR #1184 review (finding 2): ``commit()`` used to persist
    ``recommended_speech_selection()`` unconditionally, even though the step
    renders selectable language/precision ``RadioSet``s. This is the one
    place that maps a live selection to what gets persisted, via the same
    pure catalog helpers
    (``speech_language_options``, ``speech_precision_options``) the step
    already renders from.

    Args:
        selected_language: The code of the pressed language radio (``""``
            when nothing is pressed -- e.g. the step never mounted, or
            ``commit()`` runs before ``on_show()``).
        selected_precision: The value of the pressed precision radio (same
            "empty means nothing pressed" contract).
        curated_selections: Exact model/precision pairs with curated roots.

    Returns:
        The exact selection, the recommended selection when the step has not
        mounted yet, or ``None`` for a stale/unavailable explicit selection.
    """
    if not selected_language and not selected_precision:
        return recommended_speech_selection()
    curated_model_ids = frozenset(model for model, _precision in curated_selections)
    language_option = next(
        (
            option
            for option in speech_language_options(curated_model_ids=curated_model_ids)
            if option.code == selected_language and option.selectable
        ),
        None,
    )
    if language_option is None:
        return None
    precision_option = next(
        (
            option
            for option in speech_precision_options(
                model_id=language_option.model_id,
                curated_selections=curated_selections,
            )
            if option.value == selected_precision and option.selectable
        ),
        None,
    )
    if precision_option is None:
        return None
    return SpeechSelection(
        provider_id=_ROUTING_POLICY.parakeet_provider_id,
        model_id=language_option.model_id,
        language=language_option.code,
        precision=precision_option.value,
    )


def build_speech_transcription_commit(
    *, provider_id: str, model_id: str, language: str, precision: str
) -> dict[str, dict[str, Any]]:
    """Mutation for the speech transcription step.

    Args:
        provider_id: STT provider id to persist as ``transcription.default_provider``.
        model_id: Model id to persist as ``transcription.default_model``.
        language: Language to persist as ``transcription.default_language``.
        precision: Precision to persist as ``transcription.default_precision``.

    Returns:
        The section/value mapping to persist under ``transcription``.
    """
    return {
        TRANSCRIPTION_SECTION: {
            "default_provider": provider_id,
            "default_model": model_id,
            "default_language": language,
            "default_precision": precision,
        }
    }


def speech_config_patch(
    state: SpeechSelection,
    source_commit: Any,
) -> dict[str, object]:
    """Merge speech defaults with a prepared external-source config patch.

    This helper is deliberately write-free. The wizard submits its result to
    its existing single config transaction, then accepts the source commit
    only after that transaction succeeds.

    Args:
        state: Exact provider/model/language/precision selected in First Run.
        source_commit: A source-service prepared commit whose section values
            contain the complete external-source table.

    Returns:
        One complete ``transcription`` section mutation.
    """

    speech_values = build_speech_transcription_commit(
        provider_id=state.provider_id,
        model_id=state.model_id,
        language=state.language,
        precision=state.precision,
    )[TRANSCRIPTION_SECTION]
    source_values = source_commit.section_values[TRANSCRIPTION_SECTION]
    return {
        TRANSCRIPTION_SECTION: {
            **speech_values,
            **source_values,
        }
    }


@dataclass(frozen=True)
class SpeechPrefill:
    """Persisted ``[transcription]`` defaults for re-run prefill (no secrets)."""

    provider_id: str = ""
    model_id: str = ""
    language: str = ""
    precision: str = ""


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
        The persisted provider/model/language/precision, or a blank prefill when the
        section is absent or malformed.
    """
    section = app_config.get(TRANSCRIPTION_SECTION)
    if not isinstance(section, Mapping):
        return SpeechPrefill()
    return SpeechPrefill(
        provider_id=str(section.get("default_provider") or ""),
        model_id=str(section.get("default_model") or ""),
        language=str(section.get("default_language") or ""),
        precision=str(section.get("default_precision") or ""),
    )


def speech_prefill_status(
    prefill: SpeechPrefill,
    *,
    installed_active: bool = False,
    acted_this_run: bool = False,
    runtime_installed: bool = True,
    selected_label: str = "Parakeet v2",
) -> str:
    """Human copy describing what is currently persisted, or "" for nothing.

    TASK-1301 review (Important 3 / AC#5's "re-run prefills" clause): the
    step must SHOW what re-running setup would interact with before the
    user acts, instead of silently overwriting it. This deliberately does
    NOT special-case the shipped template default (``faster-whisper`` /
    ``distil-large-v3``) as "nothing" -- from the user's point of view it
    is a real, currently-effective value that installing/activating here
    would replace, so it is shown exactly like any other persisted
    provider. Contrast ``read_speech_prefill``'s docstring, which is about
    a DIFFERENT question ("did the wizard configure this") that
    legitimately excludes the template default.

    Review NEW-2: the original "installing or activating here will switch
    your default" sentence is FALSE in the installed+active state --
    neither is a real action (Activate is disabled once already active).
    ``installed_active`` and ``acted_this_run`` make this state-aware so
    the copy never promises an outcome no control on screen can deliver.

    Args:
        prefill: The currently persisted transcription defaults.
        installed_active: Whether the managed Parakeet v2 artifact is
            currently installed AND active (so neither installing nor
            activating is an available action -- only the explicit
            "use as default" affordance is, see ``SpeechSetupStep``).
        acted_this_run: Whether the user already opted in this run
            (installed, activated, or used "use as default") -- the
            pending change has not been written to disk yet (that happens
            in ``commit()`` on Next), but is no longer merely "possible".
        runtime_installed: Whether the onnx-asr runtime is importable. When
            it is not, the "use as default" affordance is (correctly) never
            composed, so the sentence must not direct the user to a button
            that is not on screen (final-review residual of NEW-2).
        selected_label: Human label for the exact model/precision selection.

    Returns:
        Empty when nothing is persisted at all; otherwise a sentence
        describing the current default and, when it differs from the
        Parakeet ONNX provider, the real, state-accurate consequence of
        acting on this step.
    """
    if not prefill.provider_id:
        return ""
    if prefill.provider_id == _ROUTING_POLICY.parakeet_provider_id:
        return f"Already your default: {prefill.model_id} ({prefill.language})."
    if acted_this_run:
        return (
            f"{selected_label} will become your default when you continue "
            f"(currently: {prefill.provider_id})."
        )
    if not runtime_installed:
        # No action on this step can switch the default without the
        # runtime, and the "use as default" button is not composed -- state
        # the fact without directing the user to a control that isn't there.
        return f"Currently configured: {prefill.provider_id}."
    if installed_active:
        return (
            f"Currently configured: {prefill.provider_id} — choose "
            f'"Use {selected_label} as my default" below to switch.'
        )
    return (
        f"Currently configured: {prefill.provider_id} — installing or "
        f"activating here will switch your default to {selected_label}."
    )


def should_persist_speech_config(*, active: bool, acted_this_run: bool) -> bool:
    """Whether ``commit()`` should write ``[transcription]``.

    TASK-1301 review (Important 3): AC#5 says persist only after a verified
    active artifact -- but "active" alone is not sufficient, because the
    artifact may have been installed in an earlier session (e.g. from the
    Library screen) while ``[transcription]`` was deliberately configured
    for something else (``remote-whisper``, ``default_language="auto"``,
    ...). Without also requiring that the USER engaged this step during
    THIS run, a re-run that just presses Next through every step would
    silently clobber that existing, unrelated configuration. This is the
    single choke point for that decision so the wizard's byte-identical
    re-run guarantee has one testable, named gate.

    Args:
        active: Whether a verified managed Parakeet v2 artifact is
            currently active (freshly re-checked, never trusted stale
            widget state).
        acted_this_run: Whether the user successfully installed or
            activated the artifact THROUGH THIS STEP during this wizard
            run (never true for a re-run that only presses Next).

    Returns:
        True only when both conditions hold.
    """
    return active and acted_this_run


__all__ = [
    "LANGUAGE_DISPLAY_NAMES",
    "TRANSCRIPTION_SECTION",
    "SpeechLanguageOption",
    "SpeechPrecisionOption",
    "SpeechPrefill",
    "SpeechSelection",
    "build_speech_transcription_commit",
    "read_speech_prefill",
    "speech_config_patch",
    "should_persist_speech_config",
    "speech_prefill_status",
    "recommended_speech_selection",
    "resolve_speech_selection",
    "routing_policy",
    "speech_language_options",
    "speech_precision_options",
]
