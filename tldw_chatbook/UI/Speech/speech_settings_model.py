"""Which settings belong to which provider, and whether it is set up.

Pure data plus two lookups, so the classification is testable without
mounting `TTSSettingsWidget` -- 2,230 lines whose `compose()` alone is 768.

The second question is the one the legacy screen could not answer. It
rendered eight identical collapsed boxes, so "is ElevenLabs configured?"
meant opening each in turn. The spec's rule for this view -- one block per
provider, only the configured ones expanded -- is not implementable without
it.

"Configured" means **differing from the shipped default**.

An earlier version defined it as merely holding a non-blank value, on the
grounds that `compose()` declares no literal defaults. That was half right:
the defaults are real, they are just the third argument to each control's
`get_cli_setting(section, key, default)` call -- recorded here as
:data:`SETTING_CONFIG_SOURCES`. Once the pane seeded its controls from config
the way the legacy screen did, the non-blank rule marked every provider
configured and opened all eight groups.

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
#: Read-only readouts, not editable settings. `kokoro-voice-blends-list` is
#: here because `_load_kokoro_voice_blends` does
#: `query_one("#kokoro-voice-blends-list", Static).update(...)` -- rendering
#: it as an Input made that raise, leaving a local unbound and taking the
#: whole of `_set_initial_values` down with it.
SETTINGS_STATUS: tuple[str, ...] = (
    'audio-cpp-discovery-status',
    'kokoro-voice-blends-list',
)

#: Structural leftovers, not settings: a container, a static mode readout and
#: a privacy notice. Filed as settings they rendered as three empty labelled
#: rows -- "External", "Privacy notice", "Settings" -- with nothing in them,
#: which is what driving the screen showed.
NON_SETTING_IDS: frozenset[str] = frozenset(
    {"audio-cpp-settings", "audio-cpp-mode-value", "audio-cpp-privacy-notice"}
)

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


def _differs_from_default(setting: str, values: Mapping[str, Any]) -> bool:
    """Return whether a setting has been changed from its shipped default.

    Args:
        setting: The control id.
        values: Current value per setting id.

    Returns:
        True when the setting holds a value that is both set and different
        from the default recorded in :data:`SETTING_CONFIG_SOURCES`.
    """
    if setting not in values:
        return False
    value = values[setting]
    if not _is_set(value):
        return False
    source = SETTING_CONFIG_SOURCES.get(setting)
    if source is None:
        return True
    return str(value).strip() != str(source[2]).strip()


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

    if not any(_differs_from_default(setting, values) for setting in owned):
        return "default"

    required = REQUIRED_SETTINGS.get(provider, ())
    if any(not _is_set(values.get(setting)) for setting in required):
        return "incomplete"
    return "configured"

#: Control id -> the ``(section, key, default)`` the legacy screen read it
#: from. Every control was seeded this way in `compose()`, not from
#: literals -- which is why a rebuild that skipped it flipped 13 saved
#: values and dropped 4 keys entirely.
SETTING_CONFIG_SOURCES: dict[str, tuple[str, str, object]] = {   'alltalk-url-input': (   'app_tts',
                             'ALLTALK_TTS_URL_DEFAULT',
                             'http://127.0.0.1:7851'),
    'alltalk-voice-input': (   'app_tts',
                               'ALLTALK_TTS_VOICE_DEFAULT',
                               'female_01.wav'),
    'chatterbox-candidates-input': (   'app_tts',
                                       'CHATTERBOX_NUM_CANDIDATES',
                                       '1'),
    'chatterbox-cfg-weight-input': (   'app_tts',
                                       'CHATTERBOX_CFG_WEIGHT',
                                       '0.5'),
    'chatterbox-chunk-size-input': (   'app_tts',
                                       'CHATTERBOX_CHUNK_SIZE',
                                       '1024'),
    'chatterbox-crossfade-ms-input': (   'app_tts',
                                         'CHATTERBOX_CROSSFADE_MS',
                                         '50'),
    'chatterbox-crossfade-switch': (   'app_tts',
                                       'CHATTERBOX_ENABLE_CROSSFADE',
                                       True),
    'chatterbox-exaggeration-input': (   'app_tts',
                                         'CHATTERBOX_EXAGGERATION',
                                         '0.5'),
    'chatterbox-max-chunk-input': (   'app_tts',
                                      'CHATTERBOX_MAX_CHUNK_SIZE',
                                      '500'),
    'chatterbox-normalize-switch': (   'app_tts',
                                       'CHATTERBOX_NORMALIZE_AUDIO',
                                       True),
    'chatterbox-preprocess-switch': (   'app_tts',
                                        'CHATTERBOX_PREPROCESS_TEXT',
                                        True),
    'chatterbox-seed-input': ('app_tts', 'CHATTERBOX_RANDOM_SEED', ''),
    'chatterbox-stream-chunk-input': (   'app_tts',
                                         'CHATTERBOX_STREAM_CHUNK_SIZE',
                                         '4096'),
    'chatterbox-streaming-switch': (   'app_tts',
                                       'CHATTERBOX_STREAMING',
                                       True),
    'chatterbox-target-db-input': (   'app_tts',
                                      'CHATTERBOX_TARGET_DB',
                                      '-20.0'),
    'chatterbox-temperature-input': (   'app_tts',
                                        'CHATTERBOX_TEMPERATURE',
                                        '0.5'),
    'chatterbox-whisper-switch': (   'app_tts',
                                     'CHATTERBOX_VALIDATE_WHISPER',
                                     False),
    'default-speed-input': ('app_tts', 'default_speed', 1.0),
    'elevenlabs-similarity-input': (   'app_tts',
                                       'ELEVENLABS_SIMILARITY_BOOST',
                                       '0.8'),
    'elevenlabs-speaker-boost-switch': (   'app_tts',
                                           'ELEVENLABS_USE_SPEAKER_BOOST',
                                           True),
    'elevenlabs-stability-input': (   'app_tts',
                                      'ELEVENLABS_VOICE_STABILITY',
                                      '0.5'),
    'elevenlabs-style-input': ('app_tts', 'ELEVENLABS_STYLE', '0.0'),
    'higgs-delimiter-input': ('HiggsSettings', 'speaker_delimiter', '|||'),
    'higgs-flash-attn-switch': ('HiggsSettings', 'enable_flash_attn', True),
    'higgs-max-ref-duration-input': (   'HiggsSettings',
                                        'max_reference_duration',
                                        '30'),
    'higgs-max-tokens-input': ('HiggsSettings', 'max_new_tokens', '4096'),
    'higgs-model-path-input': (   'HiggsSettings',
                                  'model_path',
                                  'bosonai/higgs-audio-v2-generation-3B-base'),
    'higgs-multi-speaker-switch': (   'HiggsSettings',
                                      'enable_multi_speaker',
                                      True),
    'higgs-repetition-penalty-input': (   'HiggsSettings',
                                          'repetition_penalty',
                                          '1.1'),
    'higgs-temperature-input': ('HiggsSettings', 'temperature', '0.7'),
    'higgs-top-p-input': ('HiggsSettings', 'top_p', '0.9'),
    'higgs-track-performance-switch': (   'HiggsSettings',
                                          'track_performance',
                                          True),
    'higgs-voice-cloning-switch': (   'HiggsSettings',
                                      'enable_voice_cloning',
                                      True),
    'higgs-voices-dir-input': (   'HiggsSettings',
                                  'voice_samples_dir',
                                  '~/.config/tldw_cli/higgs_voices'),
    'kokoro-max-tokens-input': ('app_tts', 'KOKORO_MAX_TOKENS', '500'),
    'kokoro-performance-switch': (   'app_tts',
                                     'KOKORO_TRACK_PERFORMANCE',
                                     True),
    'kokoro-use-onnx-switch': ('app_tts', 'KOKORO_USE_ONNX', True),
    'kokoro-voice-mixing-switch': (   'app_tts',
                                      'KOKORO_ENABLE_VOICE_MIXING',
                                      False),
    'openai-base-url-input': (   'app_tts',
                                 'OPENAI_BASE_URL',
                                 'https://api.openai.com/v1/audio/speech'),
    'openai-org-id-input': ('app_tts', 'OPENAI_ORG_ID', '')}
