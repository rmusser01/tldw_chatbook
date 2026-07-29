"""One provider's settings, one row each, behind a header that says its state.

Two things the legacy screen did not do.

Every control cost **4 rows** -- a blank row, the input's top border, the
label+value row, the bottom border -- measured median 4, min 4, max 4 across
the whole screen, for values like ``5.0`` and ``10000``. Here a setting is a
label and a control on one line, the same grammar the Playground's axis row
uses.

And a collapsed group said nothing. Eight identical closed boxes meant
answering "is ElevenLabs set up?" by opening each in turn. The header now
carries `configured_state`, so the answer is readable without expanding
anything -- and names the missing field when a provider is half-configured,
because "incomplete" alone just sends the user hunting.

Children are passed to ``Collapsible.__init__``. Subclassing ``Collapsible``
and overriding ``compose()`` replaces its title row and the contents
container it toggles, so the group renders fully expanded while still
reporting ``collapsed is True`` -- a bug a flag-only assertion passed
straight through in phase 1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textual.containers import Horizontal
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Select,
    Static,
    Switch,
)

from .speech_settings_model import (
    REQUIRED_SETTINGS,
    SETTINGS_ACTIONS,
    SETTINGS_STATUS,
    configured_state,
    settings_for_provider,
)

#: Actions belonging to one provider rather than the view. They live inside
#: their group, next to the setting they fill -- and they must be mounted:
#: `_set_initial_values` queries `#kokoro-browse-model-btn` and friends, so
#: omitting them took the whole of initialisation down with a NoMatches.
PROVIDER_ACTION_LABELS: dict[str, str] = {
    "audio-cpp-test-connection-btn": "Test connection",
    "audio-cpp-refresh-models-btn": "Refresh models",
    "kokoro-browse-model-btn": "Browse model…",
    "kokoro-browse-voices-btn": "Browse voices…",
    "chatterbox-browse-voice-dir-btn": "Browse voice dir…",
    "higgs-voices-browse-btn": "Browse voices…",
}

#: Provider key -> the name shown in the header.
PROVIDER_TITLES: dict[str, str] = {
    "defaults": "Defaults",
    "audio_cpp": "audio.cpp",
    "openai": "OpenAI",
    "elevenlabs": "ElevenLabs",
    "kokoro": "Kokoro",
    "chatterbox": "Chatterbox",
    "higgs": "Higgs",
    "alltalk": "AllTalk",
}

#: What a required setting is called when the header has to name it.
REQUIRED_LABELS: dict[str, str] = {
    "audio-cpp-base-url-input": "base URL",
    "openai-api-key-input": "API key",
    "elevenlabs-api-key-input": "API key",
    "alltalk-url-input": "server URL",
}

#: Each select's options, taken from the legacy screen. A Select built
#: with an empty list cannot hold a value: `_set_initial_values` failed
#: with "Illegal select value 'openai'", every select stayed on
#: Select.NULL, and Save then raised "A default TTS provider must be
#: selected" into its own catch-all -- so pressing Save did nothing and
#: said nothing.
SELECT_OPTIONS: dict[str, list[tuple[str, str]]] = {   'alltalk-format-select': [   ('WAV', 'wav'),
                                 ('MP3', 'mp3'),
                                 ('Opus', 'opus'),
                                 ('FLAC', 'flac')],
    'alltalk-language-select': [   ('English', 'en'),
                                   ('Spanish', 'es'),
                                   ('French', 'fr'),
                                   ('German', 'de'),
                                   ('Italian', 'it'),
                                   ('Portuguese', 'pt'),
                                   ('Russian', 'ru'),
                                   ('Chinese', 'zh'),
                                   ('Japanese', 'ja'),
                                   ('Korean', 'ko')],
    'chatterbox-device-select': [('CPU', 'cpu'), ('CUDA (GPU)', 'cuda')],
    'default-format-select': [   ('MP3', 'mp3'),
                                 ('Opus', 'opus'),
                                 ('AAC', 'aac'),
                                 ('FLAC', 'flac'),
                                 ('WAV', 'wav')],
    'default-model-select': [('TTS-1', 'tts-1')],
    'default-provider-select': [   ('OpenAI', 'openai'),
                                   (   'audio.cpp (External Server)',
                                       'audio_cpp'),
                                   ('ElevenLabs', 'elevenlabs'),
                                   ('Kokoro (Local)', 'kokoro'),
                                   ('Chatterbox (Local)', 'chatterbox'),
                                   ('Higgs Audio (Local)', 'higgs'),
                                   ('AllTalk (Local Server)', 'alltalk')],
    'default-voice-select': [('Alloy', 'alloy')],
    'elevenlabs-format-select': [   ('MP3 192kbps', 'mp3_44100_192'),
                                    ('MP3 128kbps', 'mp3_44100_128'),
                                    ('MP3 96kbps', 'mp3_44100_96'),
                                    ('MP3 64kbps', 'mp3_44100_64'),
                                    ('MP3 32kbps', 'mp3_44100_32'),
                                    ('PCM 44.1kHz', 'pcm_44100'),
                                    ('PCM 24kHz', 'pcm_24000'),
                                    ('PCM 16kHz', 'pcm_16000'),
                                    ('μ-law 8kHz', 'ulaw_8000')],
    'elevenlabs-model-select': [   (   'Multilingual v2',
                                       'eleven_multilingual_v2'),
                                   ('Turbo v2', 'eleven_turbo_v2'),
                                   (   'Multilingual v1',
                                       'eleven_multilingual_v1'),
                                   (   'Monolingual v1',
                                       'eleven_monolingual_v1')],
    'higgs-device-select': [   ('Auto-detect', 'auto'),
                               ('CPU', 'cpu'),
                               ('CUDA (GPU)', 'cuda'),
                               ('CUDA Device 0', 'cuda:0'),
                               ('CUDA Device 1', 'cuda:1')],
    'higgs-dtype-select': [   ('Float32 (Full precision)', 'float32'),
                              ('Float16 (Half precision)', 'float16'),
                              ('BFloat16 (Better range)', 'bfloat16')],
    'higgs-language-select': [   ('English', 'en'),
                                 ('Spanish', 'es'),
                                 ('French', 'fr'),
                                 ('German', 'de'),
                                 ('Italian', 'it'),
                                 ('Portuguese', 'pt'),
                                 ('Russian', 'ru'),
                                 ('Chinese', 'zh'),
                                 ('Japanese', 'ja'),
                                 ('Korean', 'ko')],
    'kokoro-device-select': [('CPU', 'cpu'), ('CUDA (GPU)', 'cuda')]}

#: Each control's starting value, placeholder and type, taken from the
#: legacy screen verbatim.
#:
#: Not cosmetic. Switches defaulting to False instead of True and text
#: inputs starting empty changed what Save posted: 13 values flipped and
#: 4 keys vanished entirely, which the save-equivalence baseline caught.
SETTING_DEFAULTS: dict[str, dict[str, object]] = {   'alltalk-url-input': {'placeholder': 'AllTalk server URL'},
    'alltalk-voice-input': {'placeholder': 'Voice file name'},
    'audio-cpp-base-url-input': {'placeholder': 'http://127.0.0.1:8080'},
    'audio-cpp-connect-timeout-input': {'type': 'number'},
    'audio-cpp-max-catalog-models-input': {'type': 'integer'},
    'audio-cpp-max-identifier-characters-input': {'type': 'integer'},
    'audio-cpp-max-input-characters-input': {'type': 'integer'},
    'audio-cpp-max-metadata-bytes-input': {'type': 'integer'},
    'audio-cpp-max-response-bytes-input': {'type': 'integer'},
    'audio-cpp-max-voices-per-model-input': {'type': 'integer'},
    'audio-cpp-synthesis-timeout-input': {'type': 'number'},
    'chatterbox-candidates-input': {'placeholder': '1-5', 'type': 'number'},
    'chatterbox-cfg-weight-input': {   'placeholder': '0.0-1.0',
                                       'type': 'number'},
    'chatterbox-chunk-size-input': {   'placeholder': 'Audio chunk size',
                                       'type': 'number'},
    'chatterbox-crossfade-ms-input': {   'placeholder': 'Duration in ms',
                                         'type': 'number'},
    'chatterbox-exaggeration-input': {   'placeholder': '0.0-1.0',
                                         'type': 'number'},
    'chatterbox-max-chunk-input': {   'placeholder': 'Max characters per '
                                                     'chunk',
                                      'type': 'number'},
    'chatterbox-seed-input': {'placeholder': 'Random seed (optional)'},
    'chatterbox-stream-chunk-input': {   'placeholder': 'Stream chunk size',
                                         'type': 'number'},
    'chatterbox-target-db-input': {   'placeholder': '-40 to 0',
                                      'type': 'number'},
    'chatterbox-temperature-input': {   'placeholder': '0.0-2.0',
                                        'type': 'number'},
    'default-speed-input': {'placeholder': '0.25-4.0', 'type': 'number'},
    'elevenlabs-api-key-input': {'placeholder': 'Your ElevenLabs API key'},
    'elevenlabs-similarity-input': {   'placeholder': '0.0-1.0',
                                       'type': 'number'},
    'elevenlabs-stability-input': {   'placeholder': '0.0-1.0',
                                      'type': 'number'},
    'elevenlabs-style-input': {'placeholder': '0.0-1.0', 'type': 'number'},
    'higgs-delimiter-input': {'placeholder': 'Default: |||'},
    'higgs-max-ref-duration-input': {   'placeholder': 'Seconds (e.g., 30)',
                                        'type': 'number'},
    'higgs-max-tokens-input': {   'placeholder': 'Max tokens to generate',
                                  'type': 'number'},
    'higgs-model-path-input': {   'placeholder': 'Model path or '
                                                 'HuggingFace ID'},
    'higgs-repetition-penalty-input': {   'placeholder': '1.0 = no penalty',
                                          'type': 'number'},
    'higgs-temperature-input': {'placeholder': '0.0-2.0', 'type': 'number'},
    'higgs-top-p-input': {'placeholder': '0.0-1.0', 'type': 'number'},
    'higgs-voices-dir-input': {'placeholder': 'Path to voice samples'},
    'kokoro-max-tokens-input': {   'placeholder': 'Max tokens per chunk',
                                   'type': 'number'},
    'openai-api-key-input': {'placeholder': 'sk-...'},
    'openai-base-url-input': {   'placeholder': 'Custom API endpoint '
                                                '(optional)'},
    'openai-org-id-input': {'placeholder': 'org-... (optional)'}}

#: Setting id -> its label, taken from the legacy screen so the copy is not
#: quietly reinvented while the layout changes.
SETTING_LABELS: dict[str, str] = {
    'audio-cpp-discovery-status': 'Discovery',
    'kokoro-voice-blends-list': 'Voice blends',
    'alltalk-format-select': 'Output Format',
    'alltalk-language-select': 'Language',
    'alltalk-url-input': 'Server URL',
    'alltalk-voice-input': 'Voice',
    'audio-cpp-base-url-input': 'Base URL',
    'audio-cpp-connect-timeout-input': 'Connect timeout',
    'audio-cpp-max-catalog-models-input': 'Max catalog models',
    'audio-cpp-max-identifier-characters-input': 'Max identifier chars',
    'audio-cpp-max-input-characters-input': 'Max input characters',
    'audio-cpp-max-metadata-bytes-input': 'Max metadata bytes',
    'audio-cpp-max-response-bytes-input': 'Max response bytes',
    'audio-cpp-max-voices-per-model-input': 'Max voices per model',
    'audio-cpp-mode-value': 'External',
    'audio-cpp-privacy-notice': 'Privacy notice',
    'audio-cpp-settings': 'Settings',
    'audio-cpp-synthesis-timeout-input': 'Synthesis timeout',
    'chatterbox-candidates-input': 'Number of Candidates',
    'chatterbox-cfg-weight-input': 'CFG Weight',
    'chatterbox-chunk-size-input': 'Chunk Size',
    'chatterbox-crossfade-ms-input': 'Crossfade Duration',
    'chatterbox-crossfade-switch': 'Enable Crossfade',
    'chatterbox-device-select': 'Device',
    'chatterbox-exaggeration-input': 'Emotion Exaggeration',
    'chatterbox-max-chunk-input': 'Max Text Chunk',
    'chatterbox-normalize-switch': 'Audio Normalization',
    'chatterbox-preprocess-switch': 'Text Preprocessing',
    'chatterbox-seed-input': 'Random Seed',
    'chatterbox-stream-chunk-input': 'Stream Chunk Size',
    'chatterbox-streaming-switch': 'Enable Streaming',
    'chatterbox-target-db-input': 'Target dB',
    'chatterbox-temperature-input': 'Temperature',
    'chatterbox-whisper-switch': 'Whisper Validation',
    'default-format-select': 'Default Format',
    'default-model-select': 'Default Model',
    'default-provider-select': 'Default Provider',
    'default-speed-input': 'Default Speed',
    'default-voice-select': 'Default Voice',
    'elevenlabs-api-key-input': 'API Key',
    'elevenlabs-format-select': 'Output Format',
    'elevenlabs-model-select': 'Model',
    'elevenlabs-similarity-input': 'Similarity Boost',
    'elevenlabs-speaker-boost-switch': 'Speaker Boost',
    'elevenlabs-stability-input': 'Voice Stability',
    'elevenlabs-style-input': 'Style',
    'higgs-delimiter-input': 'Speaker Delimiter',
    'higgs-device-select': 'Device',
    'higgs-dtype-select': 'Data Type',
    'higgs-flash-attn-switch': 'Enable Flash Attention',
    'higgs-language-select': 'Default Language',
    'higgs-max-ref-duration-input': 'Max Reference Duration',
    'higgs-max-tokens-input': 'Max New Tokens',
    'higgs-model-path-input': 'Model Path',
    'higgs-multi-speaker-switch': 'Enable Multi-speaker',
    'higgs-repetition-penalty-input': 'Repetition Penalty',
    'higgs-temperature-input': 'Temperature',
    'higgs-top-p-input': 'Top P',
    'higgs-track-performance-switch': 'Performance Tracking',
    'higgs-voice-cloning-switch': 'Enable Voice Cloning',
    'higgs-voices-dir-input': 'Voice Samples Dir',
    'kokoro-device-select': 'Device',
    'kokoro-max-tokens-input': 'Max Tokens',
    'kokoro-performance-switch': 'Performance Tracking',
    'kokoro-use-onnx-switch': 'Use ONNX',
    'kokoro-voice-blends-list': 'Voice Blends',
    'kokoro-voice-mixing-switch': 'Enable Voice Mixing',
    'openai-api-key-input': 'API Key',
    'openai-base-url-input': 'Base URL',
    'openai-org-id-input': 'Organization ID'}


def _state_summary(provider: str, values: Mapping[str, Any]) -> str:
    """Return the state phrase for a provider's header.

    Args:
        provider: The provider key.
        values: Current value per setting id.

    Returns:
        ``"configured"``, ``"defaults"``, or ``"incomplete — API key
        missing"`` naming the first required setting still blank.
    """
    state = configured_state(provider, values)
    if state == "configured":
        return "configured"
    if state == "default":
        return "defaults"

    for setting in REQUIRED_SETTINGS.get(provider, ()):
        value = values.get(setting)
        if not (value is not None and str(value).strip()):
            name = REQUIRED_LABELS.get(setting, setting)
            return f"incomplete — {name} missing"
    return "incomplete"


def _is_switch(setting: str) -> bool:
    return setting.endswith("-switch")


def _is_select(setting: str) -> bool:
    return setting.endswith("-select")


def _rendered_settings(provider: str) -> tuple[str, ...]:
    """Return the ids this group mounts, settings plus its own readouts.

    Args:
        provider: The provider key.

    Returns:
        The provider's settings, followed by any status readouts that belong
        to it.
    """
    prefix = provider.replace("_", "-")
    owned = settings_for_provider(provider)
    readouts = tuple(s for s in SETTINGS_STATUS if s.startswith(prefix))
    actions = tuple(
        a
        for a in SETTINGS_ACTIONS
        if a.startswith(prefix) and a in PROVIDER_ACTION_LABELS
    )
    return owned + readouts + actions


def _setting_rows(provider: str, values: Mapping[str, Any]) -> list[Horizontal]:
    """Build one row per setting.

    Args:
        provider: The provider whose settings to render.
        values: Current value per setting id.

    Returns:
        A list of single-row ``Horizontal`` widgets, each a label and a
        control.
    """
    rows: list[Horizontal] = []
    for setting in _rendered_settings(provider):
        current = values.get(setting)
        spec = SETTING_DEFAULTS.get(setting, {})
        control: Any
        if setting in PROVIDER_ACTION_LABELS:
            rows.append(
                Horizontal(
                    Static("", classes="speech-setting-label"),
                    Button(
                        PROVIDER_ACTION_LABELS[setting],
                        id=setting,
                        classes="workbench-action speech-setting-action",
                        compact=True,
                    ),
                    classes="speech-setting-row",
                )
            )
            continue
        if setting in SETTINGS_STATUS:
            # A readout, not an input. The code that fills these does
            # `query_one(..., Static).update(...)`, which raises against an
            # Input and leaves a local unbound.
            control = Static("", id=setting, classes="speech-setting-readout")
        elif _is_switch(setting):
            control = Switch(
                value=bool(current),
                id=setting,
                classes="speech-setting-control",
            )
        elif _is_select(setting):
            choices = SELECT_OPTIONS.get(setting, [])
            control = Select(
                choices,
                id=setting,
                classes="speech-setting-control",
                allow_blank=True,
                prompt="Not set",
            )
            if current is not None and current in {v for _label, v in choices}:
                control.value = current
        else:
            control = Input(
                value=(
                    str(spec.get("value", ""))
                    if current is None
                    else str(current)
                ),
                placeholder=str(spec.get("placeholder", "")),
                type=spec.get("type", "text"),
                id=setting,
                classes="speech-setting-control",
            )
        rows.append(
            Horizontal(
                Static(
                    SETTING_LABELS.get(setting, setting),
                    classes="speech-setting-label",
                    markup=False,
                ),
                control,
                classes="speech-setting-row",
            )
        )
    return rows


class SpeechSettingsGroup(Collapsible):
    """One provider's persisted settings."""

    def __init__(
        self,
        *,
        provider: str,
        values: Mapping[str, Any] | None = None,
        collapsed: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create the group.

        Args:
            provider: The provider key.
            values: Current value per setting id, used both to fill the
                controls and to derive the header state.
            collapsed: Whether it starts closed. The pane opens the
                configured ones.
            kwargs: Forwarded to ``Collapsible``.
        """
        values = dict(values or {})
        classes = kwargs.pop("classes", "")
        title = (
            f"{PROVIDER_TITLES.get(provider, provider)} · "
            f"{_state_summary(provider, values)}"
        )
        super().__init__(
            *_setting_rows(provider, values),
            title=title,
            collapsed=collapsed,
            classes=f"speech-settings-group {classes}".strip(),
            **kwargs,
        )
        self.provider = provider
        self.values = values
