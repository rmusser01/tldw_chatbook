"""Pure Speech/TTS panel draft payloads shared with the application shell.

``TldwCli`` needs exactly one thing from the Global Speech & TTS settings
panel: the ``SpeechTTSPanelDraftSnapshot`` payload class, for the
``type(candidate) is SpeechTTSPanelDraftSnapshot`` identity checks in
``_audio_cpp_removal_settings_inputs``. Importing it from
``speech_tts_settings_panel`` put that 5,600-line Textual widget module (and
its ``textual_fspicker``/``lab_speech_status``/``console_voice_input``
subtrees) on the ``import tldw_chatbook.app`` path for a dataclass -- 20
modules and ~13 ms of import self-time exclusive to the panel, measured
2026-08-23, for a payload that touches no widget at all (TASK-21108).

The draft-validation cluster therefore lives here: the bounds, the realtime
sibling draft, the two detach/validate helpers, and the snapshot itself. The
panel re-imports every name from this module, so ``panel.
SpeechTTSPanelDraftSnapshot`` remains the SAME class object -- the identity
checks above and the panel's own tests keep working unchanged.

Nothing here performs widget, filesystem, config, or provider I/O; the only
non-stdlib dependency is ``UI/Screens/settings_speech_tts``, the pure
Speech/TTS settings model, which is already on the app import path through
``Event_Handlers/STTS_Events/stts_events``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import cast

from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    GlobalSpeechTTSCredentialState,
    GlobalSpeechTTSDefaults,
    GlobalSpeechTTSEffectiveSource,
    GlobalSpeechTTSState,
    GlobalSpeechTTSValidationError,
    build_global_speech_tts_save_proposal,
)


_MAX_DRAFT_REVISION = 2**63 - 1
_MAX_DRAFT_TEXT_CHARACTERS = 4096
_MAX_DRAFT_GRAPH_DEPTH = 8
_MAX_DRAFT_GRAPH_NODES = 4096
_MAX_DRAFT_GRAPH_TEXT_CHARACTERS = 262_144
_PRIVATE_DRAFT_KEYS = frozenset(
    {
        "token",
        "api_key",
        "auth_token",
        "client_secret",
        "access_token",
        "refresh_token",
        "password",
        "passphrase",
        "secret",
        "credential",
        "credentials",
        "handoff_token",
    }
)
_PRIVATE_DRAFT_KEY_SUFFIXES = (
    "_token",
    "_secret",
    "_credential",
    "_credentials",
    "_password",
    "_passphrase",
    "_api_key",
)


@dataclass
class _RealtimeSettingsDraft:
    """Local editable copy of the realtime engine's plain config keys.

    `realtime`/`dictation` are plain top-level config sections, not TTS
    provider adapters -- there is no `GlobalSpeechTTSState` provider entry
    for them and no TTS service adapter to reconfigure at runtime. This
    stays a self-contained sibling draft, persisted through the same atomic
    config writer other Settings surfaces use (`save_settings_to_cli_config`,
    which is `apply_settings_mutation_to_cli_config` underneath), never a
    second, bespoke config writer (TASK-2111).
    """

    enabled: bool
    provider: str
    model: str
    voice: str
    idle_timeout_minutes: str
    handsfree_engine: str
    turn_detection: str
    vad_threshold: str
    vad_silence_ms: str

    def snapshot(self) -> tuple[bool, str, str, str, str, str, str, str, str]:
        return (
            self.enabled,
            self.provider,
            self.model,
            self.voice,
            self.idle_timeout_minutes,
            self.handsfree_engine,
            self.turn_detection,
            self.vad_threshold,
            self.vad_silence_ms,
        )


def _validated_realtime_draft_copy(value: object) -> _RealtimeSettingsDraft:
    """Return one detached, structurally bounded Realtime draft."""

    if type(value) is not _RealtimeSettingsDraft:
        raise TypeError("Realtime Settings draft is invalid")
    if type(value.enabled) is not bool:
        raise TypeError("Realtime Settings draft is invalid")
    for field_name in (
        "provider",
        "model",
        "voice",
        "idle_timeout_minutes",
        "handsfree_engine",
        "turn_detection",
        "vad_threshold",
        "vad_silence_ms",
    ):
        text = getattr(value, field_name)
        if type(text) is not str or len(text) > _MAX_DRAFT_TEXT_CHARACTERS:
            raise ValueError("Realtime Settings draft is invalid")
    return replace(value)


def _detached_draft_data(value: object) -> object:
    """Detach one bounded JSON-like provider tree without private payloads."""

    nodes = 0
    text_characters = 0

    def detach(item: object, depth: int) -> object:
        nonlocal nodes, text_characters
        nodes += 1
        if depth > _MAX_DRAFT_GRAPH_DEPTH or nodes > _MAX_DRAFT_GRAPH_NODES:
            raise ValueError("Global Speech & TTS draft is too large")
        if item is None or type(item) is bool or type(item) is int:
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Global Speech & TTS draft is invalid")
            return item
        if type(item) is str:
            text_characters += len(item)
            if (
                len(item) > _MAX_DRAFT_TEXT_CHARACTERS
                or text_characters > _MAX_DRAFT_GRAPH_TEXT_CHARACTERS
            ):
                raise ValueError("Global Speech & TTS draft is too large")
            return item
        if type(item) is list:
            return [detach(child, depth + 1) for child in item]
        if type(item) is tuple:
            return tuple(detach(child, depth + 1) for child in item)
        if type(item) is dict:
            detached: dict[str, object] = {}
            for key, child in item.items():
                if type(key) is not str:
                    raise TypeError("Global Speech & TTS draft key is invalid")
                normalized = key.casefold()
                if normalized in _PRIVATE_DRAFT_KEYS or normalized.endswith(
                    _PRIVATE_DRAFT_KEY_SUFFIXES
                ):
                    raise ValueError("Global Speech & TTS draft is private")
                detached[key] = detach(child, depth + 1)
            return detached
        raise TypeError("Global Speech & TTS draft value is invalid")

    return detach(value, 0)


def _validated_global_speech_tts_state_copy(value: object) -> GlobalSpeechTTSState:
    """Return one detached complete state after existing pure field validation."""

    if type(value) is not GlobalSpeechTTSState:
        raise TypeError("Global Speech & TTS draft is invalid")
    validated = cast(GlobalSpeechTTSState, value)
    defaults = validated.defaults
    if type(defaults) is not GlobalSpeechTTSDefaults:
        raise TypeError("Global Speech & TTS defaults are invalid")
    default_values = (
        defaults.provider_id,
        defaults.model_mode,
        defaults.model_id,
        defaults.voice_mode,
        defaults.voice_id,
        defaults.response_format,
        defaults.speed,
        defaults.default_profile_id,
    )
    detached_defaults = _detached_draft_data(default_values)
    assert isinstance(detached_defaults, tuple)
    copied_defaults = GlobalSpeechTTSDefaults(*detached_defaults)

    if type(validated.providers) is not dict or set(validated.providers) != set(
        BUILT_IN_TTS_PROVIDER_ORDER
    ):
        raise ValueError("Global Speech & TTS draft is invalid")
    copied_providers: dict[str, dict[str, object]] = {}
    for provider_id, provider_values in validated.providers.items():
        if type(provider_values) is not dict or any(
            type(key) is not str for key in provider_values
        ):
            raise ValueError("Global Speech & TTS draft is invalid")
        allowed = set(GLOBAL_TTS_PROVIDER_FIELD_IDS[provider_id]) - {"credential"}
        if not set(provider_values).issubset(allowed):
            raise ValueError("Global Speech & TTS draft is invalid")
        detached = _detached_draft_data(provider_values)
        assert isinstance(detached, dict)
        copied_providers[provider_id] = detached

    provider_ids = set(BUILT_IN_TTS_PROVIDER_ORDER)
    if type(validated.credentials) is not dict or not set(
        validated.credentials
    ).issubset(provider_ids):
        raise ValueError("Global Speech & TTS draft is invalid")
    credential_metadata: list[tuple[object, ...]] = []
    for provider_id, credential in validated.credentials.items():
        if (
            type(provider_id) is not str
            or type(credential) is not GlobalSpeechTTSCredentialState
            or credential.provider_id != provider_id
        ):
            raise ValueError("Global Speech & TTS credential metadata is invalid")
        credential_metadata.append(
            (
                provider_id,
                credential.provider_id,
                credential.setting_key,
                credential.environment_variable,
                credential.source.value,
                credential.local_saved,
                credential.local_shadowed,
            )
        )
    if type(validated.defaults_source) is not GlobalSpeechTTSEffectiveSource:
        raise ValueError("Global Speech & TTS draft is invalid")
    if (
        type(validated.provider_sources) is not dict
        or set(validated.provider_sources) != provider_ids
        or any(
            type(key) is not str or type(source) is not GlobalSpeechTTSEffectiveSource
            for key, source in validated.provider_sources.items()
        )
    ):
        raise ValueError("Global Speech & TTS draft is invalid")
    if (
        type(validated.provider_field_sources) is not dict
        or set(validated.provider_field_sources) != provider_ids
        or any(
            type(provider_id) is not str
            or type(sources) is not dict
            or not set(sources).issubset(GLOBAL_TTS_PROVIDER_FIELD_IDS[provider_id])
            or any(
                type(field_id) is not str
                or type(source) is not GlobalSpeechTTSEffectiveSource
                for field_id, source in sources.items()
            )
            for provider_id, sources in validated.provider_field_sources.items()
        )
    ):
        raise ValueError("Global Speech & TTS draft is invalid")
    _detached_draft_data(
        (
            credential_metadata,
            validated.defaults_source.value,
            tuple(
                (provider_id, source.value)
                for provider_id, source in validated.provider_sources.items()
            ),
            tuple(
                (
                    provider_id,
                    tuple(
                        (field_id, source.value) for field_id, source in sources.items()
                    ),
                )
                for provider_id, sources in validated.provider_field_sources.items()
            ),
        )
    )
    copied = replace(
        validated,
        defaults=copied_defaults,
        providers=copied_providers,
        credentials={},
        defaults_source=GlobalSpeechTTSEffectiveSource.DEFAULT,
        provider_sources={
            provider_id: GlobalSpeechTTSEffectiveSource.DEFAULT
            for provider_id in BUILT_IN_TTS_PROVIDER_ORDER
        },
        provider_field_sources={
            provider_id: {} for provider_id in BUILT_IN_TTS_PROVIDER_ORDER
        },
    )
    # These existing pure validators cover every provider field plus the
    # defaults axes without performing provider, filesystem, or config I/O.
    # A draft snapshot must also preserve an intentionally invalid field so
    # the mounted Save action can focus it and explain the validation error.
    for provider_id in BUILT_IN_TTS_PROVIDER_ORDER:
        try:
            build_global_speech_tts_save_proposal(
                copied,
                copied,
                configure_provider=provider_id,
            )
        except GlobalSpeechTTSValidationError:
            pass
    return copied


@dataclass(frozen=True, slots=True, repr=False)
class SpeechTTSPanelDraftSnapshot:
    """Complete process-local non-secret Speech/TTS panel draft."""

    state: GlobalSpeechTTSState
    original_state: GlobalSpeechTTSState
    realtime_draft: _RealtimeSettingsDraft
    realtime_original: _RealtimeSettingsDraft
    configure_provider: str
    draft_revision: int

    def __post_init__(self) -> None:
        if self.configure_provider not in BUILT_IN_TTS_PROVIDER_ORDER:
            raise ValueError("Speech/TTS draft provider is invalid")
        if (
            type(self.draft_revision) is not int
            or self.draft_revision < 0
            or self.draft_revision > _MAX_DRAFT_REVISION
        ):
            raise ValueError("Speech/TTS draft revision is invalid")
        object.__setattr__(
            self,
            "state",
            _validated_global_speech_tts_state_copy(self.state),
        )
        object.__setattr__(
            self,
            "original_state",
            _validated_global_speech_tts_state_copy(self.original_state),
        )
        object.__setattr__(
            self,
            "realtime_draft",
            _validated_realtime_draft_copy(self.realtime_draft),
        )
        object.__setattr__(
            self,
            "realtime_original",
            _validated_realtime_draft_copy(self.realtime_original),
        )

    def __repr__(self) -> str:
        """Expose only bounded navigation metadata, never draft values."""

        return (
            "SpeechTTSPanelDraftSnapshot("
            f"configure_provider={self.configure_provider!r}, "
            f"draft_revision={self.draft_revision})"
        )
