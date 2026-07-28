"""Which Playground controls are comparison axes and which are tuning knobs.

Pure data plus one lookup, so the classification is testable without
mounting `TTSPlaygroundWidget` -- 2,400 lines that build a TTS playground on
compose.

The split is the redesign's central rule. The Playground exists to test the
available STT/TTS options and identify which works best, so the variables a
user changes to *compare* stay visible, and the ones set once per provider
collapse. Measured on the shipped screen, treating all 57 controls as
equally important made the form 93 rows tall in a 34-row viewport and put
`Generate` 21 rows below the fold.

See `Docs/superpowers/specs/2026-07-27-speech-console-redesign-design.md`.
"""

from __future__ import annotations

#: Changed constantly while comparing options; always visible as chips.
AXIS_CONTROLS: tuple[str, ...] = (
    "tts-provider-select",
    "tts-model-select",
    "tts-voice-select",
    "tts-language-select",
    "tts-format-select",
    "tts-speed-input",
)

#: Post-processing that applies whatever the provider, so it is appended to
#: every provider's group rather than duplicated into each. A provider switch
#: must not make normalisation disappear.
AUDIO_PARAMS: tuple[str, ...] = (
    "tts-preprocess-text-switch",
    "tts-normalize-audio-switch",
    "tts-target-db-input",
)

#: Provider -> its own tuning knobs, excluding :data:`AUDIO_PARAMS`.
#:
#: A provider with an empty tuple is deliberate, not unfinished: audio.cpp,
#: OpenAI and AllTalk expose no per-request tuning in the Playground. Their
#: configuration lives in Settings, which is the larger surface -- 17 of
#: Chatterbox's 18 controls and 16 of Higgs' 21 are there, not here.
PROVIDER_PARAMS: dict[str, tuple[str, ...]] = {
    "elevenlabs": (
        "tts-stability-input",
        "tts-similarity-input",
        "tts-style-input",
        "tts-speaker-boost-switch",
    ),
    "chatterbox": (
        "tts-exaggeration-input",
        "tts-cfg-weight-input",
        "tts-temperature-input",
        "tts-num-candidates-input",
        "tts-validate-whisper-switch",
        "tts-random-seed-input",
    ),
    "higgs": (
        "tts-higgs-temperature-input",
        "tts-higgs-top-p-input",
        "tts-higgs-repetition-penalty-input",
        "tts-higgs-voice-cloning-switch",
        "tts-higgs-multi-speaker-switch",
        "tts-higgs-delimiter-input",
    ),
    "kokoro": ("tts-kokoro-use-onnx",),
    "audio_cpp": (),
    "openai": (),
    "alltalk": (),
}

#: Controls outside the axis/knob split: the text input, actions, status
#: readouts, the audio player, and the per-provider containers.
#:
#: Listed rather than derived so the rebuild can be checked for completeness.
#: `test_the_inventory_matches_the_live_widget` diffs this against the ids
#: `TTSPlaygroundWidget` actually composes, in both directions -- that is the
#: guard against silently dropping a capability while re-siting 57 controls.
_UNSPLIT_CONTROLS: tuple[str, ...] = (
    "tts-text-input",
    # actions
    "tts-generate-btn",
    "tts-random-text-btn",
    "tts-clear-text-btn",
    "tts-refresh-catalog-btn",
    "audio-play-btn",
    "audio-export-btn",
    "pause-audio-btn",
    "stop-audio-btn",
    "reference-audio-btn",
    "clear-reference-audio-btn",
    "higgs-voice-upload-btn",
    "higgs-clear-voice-btn",
    # status readouts
    "tts-provider-status",
    "tts-generation-log",
    "tts-audio-cpp-restrictions",
    "reference-audio-status",
    "higgs-voice-status",
    "audio-player-status",
    "generation-status-container",
    "generation-status-text",
    # player
    "audio-player-container",
    "audio-progress-bar",
    "audio-time-display",
    "generation-progress",
    # per-provider containers the legacy widget shows/hides
    "kokoro-settings",
    "kokoro-language-row",
    "elevenlabs-settings",
    "chatterbox-settings",
    "higgs-settings",
    "higgs-voice-upload-row",
)

#: Every control id the legacy Playground composes. 57 in total.
ALL_PLAYGROUND_CONTROLS: frozenset[str] = frozenset(
    AXIS_CONTROLS
    + AUDIO_PARAMS
    + tuple(control for params in PROVIDER_PARAMS.values() for control in params)
    + _UNSPLIT_CONTROLS
)


def params_for_provider(provider: str) -> tuple[str, ...]:
    """Return the tuning knobs to render for one provider.

    Args:
        provider: The selected provider key, e.g. ``"chatterbox"``.

    Returns:
        That provider's own parameters followed by the shared audio
        post-processing ones. An unknown provider yields only the shared
        set: this is called from ``compose()``, where raising would take the
        screen down rather than show a missing group.
    """
    return PROVIDER_PARAMS.get(provider, ()) + AUDIO_PARAMS
