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
from textual.widgets import Collapsible, Input, Select, Static, Switch

from .speech_settings_model import (
    REQUIRED_SETTINGS,
    configured_state,
    settings_for_provider,
)

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

#: Setting id -> its label, taken from the legacy screen so the copy is not
#: quietly reinvented while the layout changes.
SETTING_LABELS: dict[str, str] = {   'alltalk-format-select': 'Output Format',
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
    for setting in settings_for_provider(provider):
        current = values.get(setting)
        control: Any
        if _is_switch(setting):
            control = Switch(
                value=bool(current),
                id=setting,
                classes="speech-setting-control",
            )
        elif _is_select(setting):
            control = Select(
                [],
                id=setting,
                classes="speech-setting-control",
                allow_blank=True,
                prompt=str(current) if current else "Not set",
            )
        else:
            control = Input(
                value="" if current is None else str(current),
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
