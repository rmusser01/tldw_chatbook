"""Which settings belong to which provider, and whether it is set up.

Pure data plus two lookups, so the classification is testable without
mounting `TTSSettingsWidget` -- 2,230 lines whose `compose()` alone is 768.

The second question is the one the legacy screen could not answer. It
rendered eight identical collapsed boxes, so "is ElevenLabs configured?"
meant opening each in turn. The spec's rule for this view -- one block per
provider, only the configured ones expanded -- is not implementable without
it.

"Configured" is defined as **holding a non-blank value**, not as differing
from a default. `TTSSettingsWidget.compose()` declares no default values at
all: every control is filled from config at runtime, so there is no literal
to compare against, and inventing one would make the state a guess.

Measured on the shipped screen: every control costs 4 rows (median 4, min 4,
max 4) and `save-settings-btn` sits at y=102 in a 26-row viewport.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PROVIDER_SETTINGS: dict[str, tuple[str, ...]] = {   'defaults': (   'default-format-select',
                    'default-model-select',
                    'default-provider-select',
                    'default-speed-input',
                    'default-voice-select'),
    'audio_cpp': (   'audio-cpp-base-url-input',
                     'audio-cpp-connect-timeout-input',
                     'audio-cpp-max-catalog-models-input',
                     'audio-cpp-max-identifier-characters-input',
                     'audio-cpp-max-input-characters-input',
                     'audio-cpp-max-metadata-bytes-input',
                     'audio-cpp-max-response-bytes-input',
                     'audio-cpp-max-voices-per-model-input',
                     'audio-cpp-mode-value',
                     'audio-cpp-privacy-notice',
                     'audio-cpp-settings',
                     'audio-cpp-synthesis-timeout-input'),
    'openai': (   'openai-api-key-input',
                  'openai-base-url-input',
                  'openai-org-id-input'),
    'elevenlabs': (   'elevenlabs-api-key-input',
                      'elevenlabs-format-select',
                      'elevenlabs-model-select',
                      'elevenlabs-similarity-input',
                      'elevenlabs-speaker-boost-switch',
                      'elevenlabs-stability-input',
                      'elevenlabs-style-input'),
    'kokoro': (   'kokoro-device-select',
                  'kokoro-max-tokens-input',
                  'kokoro-performance-switch',
                  'kokoro-use-onnx-switch',
                  'kokoro-voice-blends-list',
                  'kokoro-voice-mixing-switch'),
    'chatterbox': (   'chatterbox-candidates-input',
                      'chatterbox-cfg-weight-input',
                      'chatterbox-chunk-size-input',
                      'chatterbox-crossfade-ms-input',
                      'chatterbox-crossfade-switch',
                      'chatterbox-device-select',
                      'chatterbox-exaggeration-input',
                      'chatterbox-max-chunk-input',
                      'chatterbox-normalize-switch',
                      'chatterbox-preprocess-switch',
                      'chatterbox-seed-input',
                      'chatterbox-stream-chunk-input',
                      'chatterbox-streaming-switch',
                      'chatterbox-target-db-input',
                      'chatterbox-temperature-input',
                      'chatterbox-whisper-switch'),
    'higgs': (   'higgs-delimiter-input',
                 'higgs-device-select',
                 'higgs-dtype-select',
                 'higgs-flash-attn-switch',
                 'higgs-language-select',
                 'higgs-max-ref-duration-input',
                 'higgs-max-tokens-input',
                 'higgs-model-path-input',
                 'higgs-multi-speaker-switch',
                 'higgs-repetition-penalty-input',
                 'higgs-temperature-input',
                 'higgs-top-p-input',
                 'higgs-track-performance-switch',
                 'higgs-voice-cloning-switch',
                 'higgs-voices-dir-input'),
    'alltalk': (   'alltalk-format-select',
                   'alltalk-language-select',
                   'alltalk-url-input',
                   'alltalk-voice-input')}

#: What a provider cannot work without. Blank while its siblings are set is
#: the `incomplete` state -- the one that fails at generation time with
#: nothing on screen having warned about it.
#:
#: Credentials and endpoints only. A blank temperature is a default; a blank
#: API key is a provider that will refuse the first request.
REQUIRED_SETTINGS: dict[str, tuple[str, ...]] = {
    "audio_cpp": ("audio-cpp-base-url-input",),
    "openai": ("openai-api-key-input",),
    "elevenlabs": ("elevenlabs-api-key-input",),
    "alltalk": ("alltalk-url-input",),
    "kokoro": (),
    "chatterbox": (),
    "higgs": (),
    "defaults": (),
}

#: Display order: the shared defaults first, then providers in the order the
#: legacy screen listed them.
SETTINGS_PROVIDER_ORDER: tuple[str, ...] = tuple(PROVIDER_SETTINGS)

#: Commands, not values to persist. Filing these as settings would mount
#: Save inside a provider group.
SETTINGS_ACTIONS: tuple[str, ...] = (   'add-voice-blend-btn',
    'audio-cpp-refresh-models-btn',
    'audio-cpp-test-connection-btn',
    'chatterbox-browse-voice-dir-btn',
    'export-blends-btn',
    'higgs-voices-browse-btn',
    'import-blends-btn',
    'kokoro-browse-model-btn',
    'kokoro-browse-voices-btn',
    'save-settings-btn')

#: Read-only readouts the pane must still mount.
SETTINGS_STATUS: tuple[str, ...] = ('audio-cpp-discovery-status',)

#: Every id `TTSSettingsWidget` composed, frozen as the yardstick a
#: completeness check measures against.
ALL_SETTINGS_CONTROLS: frozenset[str] = frozenset((   'add-voice-blend-btn',
    'alltalk-format-select',
    'alltalk-language-select',
    'alltalk-url-input',
    'alltalk-voice-input',
    'audio-cpp-base-url-input',
    'audio-cpp-connect-timeout-input',
    'audio-cpp-discovery-status',
    'audio-cpp-max-catalog-models-input',
    'audio-cpp-max-identifier-characters-input',
    'audio-cpp-max-input-characters-input',
    'audio-cpp-max-metadata-bytes-input',
    'audio-cpp-max-response-bytes-input',
    'audio-cpp-max-voices-per-model-input',
    'audio-cpp-mode-value',
    'audio-cpp-privacy-notice',
    'audio-cpp-refresh-models-btn',
    'audio-cpp-settings',
    'audio-cpp-synthesis-timeout-input',
    'audio-cpp-test-connection-btn',
    'chatterbox-browse-voice-dir-btn',
    'chatterbox-candidates-input',
    'chatterbox-cfg-weight-input',
    'chatterbox-chunk-size-input',
    'chatterbox-crossfade-ms-input',
    'chatterbox-crossfade-switch',
    'chatterbox-device-select',
    'chatterbox-exaggeration-input',
    'chatterbox-max-chunk-input',
    'chatterbox-normalize-switch',
    'chatterbox-preprocess-switch',
    'chatterbox-seed-input',
    'chatterbox-stream-chunk-input',
    'chatterbox-streaming-switch',
    'chatterbox-target-db-input',
    'chatterbox-temperature-input',
    'chatterbox-whisper-switch',
    'default-format-select',
    'default-model-select',
    'default-provider-select',
    'default-speed-input',
    'default-voice-select',
    'elevenlabs-api-key-input',
    'elevenlabs-format-select',
    'elevenlabs-model-select',
    'elevenlabs-similarity-input',
    'elevenlabs-speaker-boost-switch',
    'elevenlabs-stability-input',
    'elevenlabs-style-input',
    'export-blends-btn',
    'higgs-delimiter-input',
    'higgs-device-select',
    'higgs-dtype-select',
    'higgs-flash-attn-switch',
    'higgs-language-select',
    'higgs-max-ref-duration-input',
    'higgs-max-tokens-input',
    'higgs-model-path-input',
    'higgs-multi-speaker-switch',
    'higgs-repetition-penalty-input',
    'higgs-temperature-input',
    'higgs-top-p-input',
    'higgs-track-performance-switch',
    'higgs-voice-cloning-switch',
    'higgs-voices-browse-btn',
    'higgs-voices-dir-input',
    'import-blends-btn',
    'kokoro-browse-model-btn',
    'kokoro-browse-voices-btn',
    'kokoro-device-select',
    'kokoro-max-tokens-input',
    'kokoro-performance-switch',
    'kokoro-use-onnx-switch',
    'kokoro-voice-blends-list',
    'kokoro-voice-mixing-switch',
    'openai-api-key-input',
    'openai-base-url-input',
    'openai-org-id-input',
    'save-settings-btn'))


def settings_for_provider(provider: str) -> tuple[str, ...]:
    """Return the settings owned by one provider.

    Args:
        provider: The provider key, e.g. ``"elevenlabs"``.

    Returns:
        Its setting ids, or an empty tuple for a provider this module has
        not been taught. Called from ``compose()``, where raising would take
        the screen down rather than omit a group.
    """
    return PROVIDER_SETTINGS.get(provider, ())


def _is_set(value: Any) -> bool:
    """Return whether a control holds a real value.

    An untouched ``Input`` holds ``""``; treating that as configuration
    would mark every provider configured and expand all eight groups.

    Args:
        value: The control's current value.

    Returns:
        True when the value is neither blank nor whitespace.
    """
    if value is None or isinstance(value, bool):
        return bool(value)
    return bool(str(value).strip())


def configured_state(provider: str, values: Mapping[str, Any]) -> str:
    """Return whether a provider is set up, partly set up, or untouched.

    Args:
        provider: The provider key.
        values: Current value per setting id. Ids belonging to other
            providers are ignored.

    Returns:
        ``"configured"`` when every required setting is filled and at least
        one setting holds a value; ``"incomplete"`` when something is filled
        but a required setting is not; ``"default"`` when nothing is.
    """
    owned = settings_for_provider(provider)
    if not owned:
        return "default"

    if not any(_is_set(values.get(setting)) for setting in owned):
        return "default"

    required = REQUIRED_SETTINGS.get(provider, ())
    if any(not _is_set(values.get(setting)) for setting in required):
        return "incomplete"
    return "configured"
