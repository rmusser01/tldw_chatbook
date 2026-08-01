"""Global Speech & TTS Settings panel.

This is a bounded built-in surface.  Provider forms are deliberately explicit
so moving a field between global and Studio ownership is a reviewed code
change rather than a dynamic schema side effect.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Select, Static, Switch

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.Third_Party.textual_fspicker import (
    FileOpen,
    Filters,
    SelectDirectory,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    GLOBAL_TTS_PROVIDER_FIELD_IDS,
    TTS_PROVIDER_LABELS,
    CredentialIntent,
    CredentialSource,
    GlobalSpeechTTSCredentialMutation,
    GlobalSpeechTTSState,
    GlobalSpeechTTSValidationError,
    build_credential_mutation,
    build_global_speech_tts_save_proposal,
    restore_non_secret_defaults,
)

_PROVIDER_OPTIONS = [
    (TTS_PROVIDER_LABELS[provider_id], provider_id)
    for provider_id in BUILT_IN_TTS_PROVIDER_ORDER
]
_LANGUAGE_OPTIONS = [
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
]


class _CredentialEditorModal(ModalScreen[str | None]):
    """Empty masked Set/Replace editor; display placeholders are never input."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, provider_label: str, intent: CredentialIntent) -> None:
        super().__init__()
        self.provider_label = provider_label
        self.intent = intent

    def compose(self) -> ComposeResult:
        action = "Set" if self.intent is CredentialIntent.SET else "Replace"
        with Vertical(classes="settings-speech-credential-modal"):
            yield Static(
                f"{action} {self.provider_label} credential",
                classes="destination-section",
            )
            yield Static(
                "The editor starts empty. This value will be stored as a local "
                "config secret; an environment variable is safer and more portable.",
                classes="settings-detail-row",
                markup=False,
            )
            yield Input(
                id="settings-speech-credential-new-value",
                password=True,
                placeholder="Enter a new credential",
            )
            with Horizontal(classes="settings-action-row"):
                yield Button("Cancel", id="settings-speech-credential-cancel")
                yield Button(
                    action,
                    id="settings-speech-credential-confirm",
                    variant="primary",
                )

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#settings-speech-credential-cancel")
    def handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#settings-speech-credential-confirm")
    def handle_confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(
            self.query_one("#settings-speech-credential-new-value", Input).value
        )


class _CredentialClearModal(ModalScreen[bool]):
    """Explicit confirmation for deleting only a saved local credential."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, provider_label: str) -> None:
        super().__init__()
        self.provider_label = provider_label

    def compose(self) -> ComposeResult:
        with Vertical(classes="settings-speech-credential-modal"):
            yield Static(
                f"Clear saved {self.provider_label} credential?",
                classes="destination-section",
            )
            yield Static(
                "This removes only the local-config value. It cannot change a "
                "process environment variable.",
                classes="settings-detail-row",
                markup=False,
            )
            with Horizontal(classes="settings-action-row"):
                yield Button("Cancel", id="settings-speech-credential-clear-cancel")
                yield Button(
                    "Clear saved credential",
                    id="settings-speech-credential-clear-confirm",
                    variant="warning",
                )

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#settings-speech-credential-clear-cancel")
    def handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    @on(Button.Pressed, "#settings-speech-credential-clear-confirm")
    def handle_confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)


class SpeechTTSSettingsPanel(Vertical):
    """Edit application-wide Speech/TTS defaults and one provider at a time."""

    class DraftModified(Message):
        """Report whether the panel has an unsaved non-secret draft."""

        def __init__(
            self,
            is_modified: bool,
            state: GlobalSpeechTTSState,
            original_state: GlobalSpeechTTSState,
            configure_provider: str,
        ) -> None:
            self.is_modified = is_modified
            self.state = deepcopy(state)
            self.original_state = deepcopy(original_state)
            self.configure_provider = configure_provider
            super().__init__()

    def __init__(
        self,
        *,
        state: GlobalSpeechTTSState,
        original_state: GlobalSpeechTTSState | None = None,
        configure_provider: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.original_state = deepcopy(original_state or state)
        self.state = deepcopy(state)
        self.configure_provider = (
            configure_provider
            if configure_provider in BUILT_IN_TTS_PROVIDER_ORDER
            else state.defaults.provider_id
            if state.defaults.provider_id in BUILT_IN_TTS_PROVIDER_ORDER
            else "audio_cpp"
        )
        self.result_text = "No global Speech & TTS changes saved this session."
        self._syncing = False
        self._field_errors: dict[tuple[str, str], str] = {}
        self._next_request_id = 1
        self._latest_request_id: int | None = None
        self._pending_credential_mutation: GlobalSpeechTTSCredentialMutation | None = (
            None
        )
        self._pending_saved_defaults = None
        self._pending_saved_provider_id: str | None = None
        self._pending_saved_provider_values: dict[str, object] | None = None

    @staticmethod
    def _field_dom_id(provider_id: str, field_id: str) -> str:
        return f"settings-speech-{provider_id}-{field_id.replace('_', '-')}"

    @staticmethod
    def _row(
        label: str,
        control: Input | Select | Switch | Static | Vertical | Horizontal,
        *,
        classes: str = "",
        error: Static | None = None,
    ) -> Horizontal:
        children: list[Any] = [Static(label, classes="settings-input-label")]
        if error is not None:
            children.append(
                Vertical(
                    control,
                    error,
                    classes="settings-speech-control-stack",
                )
            )
        else:
            children.append(control)
        return Horizontal(
            *children,
            classes=(f"settings-input-row settings-speech-input-row {classes}").strip(),
        )

    def _error(self, provider_id: str, field_id: str) -> Static:
        message = self._field_errors.get((provider_id, field_id), "")
        return Static(
            message,
            id=f"{self._field_dom_id(provider_id, field_id)}-error",
            classes=(
                "settings-status-row settings-speech-field-error"
                + (" settings-speech-field-error-visible" if message else "")
            ),
            markup=False,
        )

    def _default_error(self, field_id: str, control_id: str) -> Static:
        message = self._field_errors.get(("defaults", field_id), "")
        return Static(
            message,
            id=f"{control_id}-error",
            classes=(
                "settings-status-row settings-speech-field-error"
                + (" settings-speech-field-error-visible" if message else "")
            ),
            markup=False,
        )

    def _input(
        self,
        provider_id: str,
        field_id: str,
        label: str,
        *,
        placeholder: str = "",
        disabled: bool = False,
    ) -> Horizontal:
        return self._row(
            label,
            Input(
                value=str(self.state.providers[provider_id].get(field_id, "")),
                id=self._field_dom_id(provider_id, field_id),
                placeholder=placeholder,
                disabled=disabled,
                classes=(
                    "settings-compact-input settings-speech-field "
                    "settings-speech-draft-field"
                ),
            ),
            error=self._error(provider_id, field_id),
        )

    def _select(
        self,
        provider_id: str,
        field_id: str,
        label: str,
        options: list[tuple[str, str]],
    ) -> Horizontal:
        return self._row(
            label,
            Select(
                options,
                value=self.state.providers[provider_id].get(field_id),
                id=self._field_dom_id(provider_id, field_id),
                allow_blank=False,
                compact=True,
                classes=(
                    "settings-compact-select settings-speech-field "
                    "settings-speech-draft-field"
                ),
            ),
            classes="settings-select-row",
            error=self._error(provider_id, field_id),
        )

    def _switch(
        self,
        provider_id: str,
        field_id: str,
        label: str,
    ) -> Horizontal:
        return self._row(
            label,
            Switch(
                value=bool(self.state.providers[provider_id].get(field_id)),
                id=self._field_dom_id(provider_id, field_id),
                classes="settings-speech-field settings-speech-draft-field",
            ),
            error=self._error(provider_id, field_id),
        )

    def _path(
        self,
        provider_id: str,
        field_id: str,
        label: str,
        *,
        placeholder: str,
    ) -> Horizontal:
        dom_id = self._field_dom_id(provider_id, field_id)
        return self._row(
            label,
            Horizontal(
                Input(
                    value=str(self.state.providers[provider_id].get(field_id, "")),
                    id=dom_id,
                    placeholder=placeholder,
                    classes=(
                        "settings-compact-input settings-speech-field "
                        "settings-speech-draft-field"
                    ),
                ),
                Button(
                    "Browse…",
                    id=f"{dom_id}-browse",
                    compact=True,
                    classes="settings-speech-path-picker",
                    tooltip=f"Choose {label.lower()} without initializing a provider.",
                ),
                classes="settings-speech-path-control",
            ),
            error=self._error(provider_id, field_id),
        )

    def _credential(self, provider_id: str) -> Vertical:
        state = self.state.credentials[provider_id]
        action_label = "Replace credential" if state.local_saved else "Set credential"
        source_copy = f"Effective source: {state.source.value}"
        if state.source.value == "Environment":
            source_copy += f" ({state.environment_variable}, read-only)"
        if state.local_shadowed:
            source_copy += "; saved local fallback is shadowed by the environment"
        controls: list[Button | Static] = [
            Static(source_copy, classes="settings-status-row", markup=False),
            Static(
                "Local credentials are stored as local config secrets. "
                f"Using {state.environment_variable} is the safer portable option.",
                classes="settings-detail-row",
                markup=False,
            ),
            Button(
                action_label,
                id=f"settings-speech-{provider_id}-credential-edit",
                compact=True,
                disabled=self._latest_request_id is not None,
            ),
        ]
        if state.local_saved:
            controls.append(
                Button(
                    "Clear saved credential",
                    id=f"settings-speech-{provider_id}-credential-clear",
                    compact=True,
                    variant="warning",
                    disabled=self._latest_request_id is not None,
                )
            )
        return Vertical(
            *controls,
            id=f"settings-speech-{provider_id}-credential",
            classes="settings-speech-credential",
        )

    def compose(self) -> ComposeResult:
        yield Static(
            "Speech & TTS", classes="destination-section settings-column-title"
        )
        with Vertical(id="settings-speech-scope-banner", classes="settings-focus-card"):
            yield Static(
                "You are editing application-wide Speech & TTS defaults. "
                "The Speech Studio can keep separate Studio preferences without "
                "changing these values.",
                classes="settings-detail-row",
                markup=False,
            )
            yield Button(
                "Open Speech Lab",
                id="settings-speech-open-lab",
                compact=True,
                tooltip="Open Speech Lab without testing or refreshing a provider.",
            )

        defaults = self.state.defaults
        with Vertical(
            id="settings-speech-global-defaults", classes="settings-focus-card"
        ):
            yield Static("Global defaults", classes="destination-section")
            yield self._row(
                "Default TTS Provider",
                Select(
                    _PROVIDER_OPTIONS,
                    value=defaults.provider_id,
                    id="settings-speech-default-provider",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._default_error(
                    "provider_id",
                    "settings-speech-default-provider",
                ),
            )
            yield self._row(
                "Model policy",
                Select(
                    [("Exact", "exact"), ("First available", "first_available")],
                    value=defaults.model_mode,
                    id="settings-speech-model-policy",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._default_error(
                    "model_mode",
                    "settings-speech-model-policy",
                ),
            )
            yield self._row(
                "Model value",
                Input(
                    value=defaults.model_id or "",
                    id="settings-speech-model-value",
                    disabled=defaults.model_mode != "exact",
                    placeholder="Exact model ID",
                    classes="settings-compact-input settings-speech-draft-field",
                ),
                error=self._default_error(
                    "default_model",
                    "settings-speech-model-value",
                ),
            )
            yield self._row(
                "Voice policy",
                Select(
                    [("Exact", "exact"), ("Server default", "server_default")],
                    value=defaults.voice_mode,
                    id="settings-speech-voice-policy",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._default_error(
                    "voice_mode",
                    "settings-speech-voice-policy",
                ),
            )
            yield self._row(
                "Voice value",
                Input(
                    value=defaults.voice_id or "",
                    id="settings-speech-voice-value",
                    disabled=defaults.voice_mode != "exact",
                    placeholder="Exact voice ID",
                    classes="settings-compact-input settings-speech-draft-field",
                ),
                error=self._default_error(
                    "default_voice",
                    "settings-speech-voice-value",
                ),
            )
            audio_cpp_selected = defaults.provider_id == "audio_cpp"
            yield self._row(
                "Output format",
                Select(
                    [
                        ("MP3", "mp3"),
                        ("Opus", "opus"),
                        ("AAC", "aac"),
                        ("FLAC", "flac"),
                        ("WAV", "wav"),
                    ],
                    value=defaults.response_format,
                    id="settings-speech-output-format",
                    allow_blank=False,
                    compact=True,
                    disabled=audio_cpp_selected,
                    classes="settings-compact-select settings-speech-draft-field",
                ),
                classes="settings-select-row",
                error=self._default_error(
                    "response_format",
                    "settings-speech-output-format",
                ),
            )
            yield self._row(
                "Speed",
                Input(
                    value=str(defaults.speed),
                    id="settings-speech-speed",
                    disabled=audio_cpp_selected,
                    placeholder="0.25 - 4.0",
                    classes="settings-compact-input settings-speech-draft-field",
                ),
                error=self._default_error(
                    "default_speed",
                    "settings-speech-speed",
                ),
            )
            yield Static(
                "audio.cpp requires WAV output and speed 1.0."
                if audio_cpp_selected
                else "Provider capability constraints are validated before Save.",
                id="settings-speech-default-constraints",
                classes="settings-status-row",
                markup=False,
            )

        with Vertical(
            id="settings-speech-provider-setup", classes="settings-focus-card"
        ):
            yield Static("Provider setup", classes="destination-section")
            yield Static(
                "Configure Provider does not change the Default TTS Provider.",
                classes="settings-detail-row",
                markup=False,
            )
            yield self._row(
                "Configure Provider",
                Select(
                    _PROVIDER_OPTIONS,
                    value=self.configure_provider,
                    id="settings-speech-configure-provider",
                    allow_blank=False,
                    compact=True,
                    classes="settings-compact-select",
                ),
                classes="settings-select-row",
            )
            yield from self._compose_provider_form(self.configure_provider)

        with Vertical(id="settings-speech-inspector", classes="settings-focus-card"):
            yield Static("Configuration inspector", classes="destination-section")
            yield Static(
                f"Selected setup: {TTS_PROVIDER_LABELS[self.configure_provider]} | "
                "Configuration: Saved or local draft | Runtime: Not checked",
                id="settings-speech-inspector-summary",
                classes="settings-status-row",
                markup=False,
            )
            yield Static(
                "Ordinary Save validates and persists locally. Use Speech Lab for "
                "connection tests, discovery, generation, and playback.",
                classes="settings-detail-row",
                markup=False,
            )

        with Horizontal(id="settings-speech-actions", classes="settings-action-row"):
            yield Button(
                "Save",
                id="settings-speech-save",
                variant="primary",
                disabled=self._latest_request_id is not None,
            )
            yield Button("Revert", id="settings-speech-revert")
            yield Button(
                "Restore Non-secret Defaults",
                id="settings-speech-restore-defaults",
            )
            yield Button("Open Speech Lab", id="settings-speech-open-lab-bottom")
        yield Static(
            self.result_text,
            id="settings-speech-save-result",
            classes="settings-status-row",
            markup=False,
        )

    def _compose_provider_form(self, provider_id: str) -> ComposeResult:
        with Vertical(
            id=f"settings-speech-provider-{provider_id}",
            classes="settings-speech-provider-form",
        ):
            yield Static(
                TTS_PROVIDER_LABELS[provider_id], classes="destination-section"
            )
            if provider_id == "audio_cpp":
                yield Static(
                    "External server. Chatbook connects to a server that you start and own.",
                    id="settings-speech-audio_cpp-external-mode",
                    classes="settings-detail-row",
                    markup=False,
                )
                yield self._input(
                    provider_id,
                    "base_url",
                    "Server URL",
                    placeholder="http://127.0.0.1:8080",
                )
                yield self._input(
                    provider_id,
                    "connect_timeout_seconds",
                    "Connect timeout (seconds)",
                )
                yield self._input(
                    provider_id,
                    "synthesis_timeout_seconds",
                    "Synthesis timeout (seconds)",
                )
                yield Static(
                    "Generation sends submitted text to this configured server.",
                    id="settings-speech-audio_cpp-privacy-notice",
                    classes="settings-status-row",
                    markup=False,
                )
                with Collapsible(title="Advanced safety limits", collapsed=True):
                    yield self._input(
                        provider_id, "max_input_characters", "Max input characters"
                    )
                    yield self._input(
                        provider_id, "max_response_bytes", "Max response bytes"
                    )
                    yield self._input(
                        provider_id, "max_metadata_bytes", "Max metadata bytes"
                    )
                    yield self._input(
                        provider_id, "max_catalog_models", "Max catalog models"
                    )
                    yield self._input(
                        provider_id,
                        "max_voices_per_model",
                        "Max voices per model",
                    )
                    yield self._input(
                        provider_id,
                        "max_identifier_characters",
                        "Max identifier characters",
                    )
                return

            if provider_id == "openai":
                yield self._credential(provider_id)
                yield self._input(provider_id, "base_url", "Base URL")
                yield self._input(
                    provider_id,
                    "organization_id",
                    "Organization ID",
                    placeholder="Optional",
                )
                return

            if provider_id == "elevenlabs":
                yield self._credential(provider_id)
                yield self._select(
                    provider_id,
                    "output_format",
                    "Output format",
                    [
                        ("MP3 192kbps", "mp3_44100_192"),
                        ("MP3 128kbps", "mp3_44100_128"),
                        ("MP3 96kbps", "mp3_44100_96"),
                        ("MP3 64kbps", "mp3_44100_64"),
                        ("MP3 32kbps", "mp3_44100_32"),
                        ("PCM 44.1kHz", "pcm_44100"),
                        ("PCM 24kHz", "pcm_24000"),
                        ("PCM 16kHz", "pcm_16000"),
                        ("μ-law 8kHz", "ulaw_8000"),
                    ],
                )
                yield self._input(provider_id, "stability", "Voice stability")
                yield self._input(provider_id, "similarity_boost", "Similarity boost")
                yield self._input(provider_id, "style", "Style")
                yield self._switch(provider_id, "speaker_boost", "Speaker boost")
                return

            if provider_id == "kokoro":
                yield self._select(
                    provider_id,
                    "device",
                    "Device",
                    [("CPU", "cpu"), ("CUDA", "cuda"), ("Apple MPS", "mps")],
                )
                yield self._switch(provider_id, "use_onnx", "Use ONNX")
                yield self._path(
                    provider_id,
                    "onnx_model_path",
                    "ONNX model file",
                    placeholder="Path to model file",
                )
                yield self._path(
                    provider_id,
                    "voices_json_path",
                    "Voices JSON file",
                    placeholder="Path to voices.json",
                )
                yield self._input(provider_id, "max_tokens", "Max tokens")
                yield self._switch(provider_id, "voice_mixing", "Voice mixing")
                yield self._switch(
                    provider_id, "track_performance", "Performance tracking"
                )
                return

            if provider_id == "chatterbox":
                yield self._select(
                    provider_id,
                    "device",
                    "Device",
                    [("CPU", "cpu"), ("CUDA", "cuda")],
                )
                yield self._path(
                    provider_id,
                    "voice_resource_directory",
                    "Voice resource directory",
                    placeholder="Path to voice resources",
                )
                yield self._input(provider_id, "temperature", "Temperature")
                yield self._input(provider_id, "chunk_size", "Chunk size")
                yield self._input(
                    provider_id, "random_seed", "Random seed", placeholder="Optional"
                )
                yield self._input(provider_id, "candidates", "Candidates")
                yield self._switch(
                    provider_id, "validate_whisper", "Whisper validation"
                )
                yield self._switch(provider_id, "preprocess_text", "Text preprocessing")
                yield self._switch(
                    provider_id, "normalize_audio", "Audio normalization"
                )
                yield self._input(provider_id, "target_db", "Target dB")
                yield self._input(provider_id, "max_chunk_size", "Max text chunk")
                yield self._switch(provider_id, "streaming", "Streaming")
                yield self._input(provider_id, "stream_chunk_size", "Stream chunk size")
                yield self._switch(provider_id, "crossfade", "Crossfade")
                yield self._input(
                    provider_id, "crossfade_ms", "Crossfade duration (ms)"
                )
                return

            if provider_id == "higgs":
                yield self._path(
                    provider_id,
                    "model_path",
                    "Model path",
                    placeholder="Local path or Hugging Face model ID",
                )
                yield self._path(
                    provider_id,
                    "voice_resource_directory",
                    "Voice resource directory",
                    placeholder="Path to voice samples",
                )
                yield self._select(
                    provider_id,
                    "device",
                    "Device",
                    [
                        ("Auto", "auto"),
                        ("CPU", "cpu"),
                        ("CUDA", "cuda"),
                        ("CUDA 0", "cuda:0"),
                        ("CUDA 1", "cuda:1"),
                        ("Apple MPS", "mps"),
                    ],
                )
                yield self._switch(
                    provider_id,
                    "enable_flash_attention",
                    "Enable flash attention",
                )
                yield self._select(
                    provider_id,
                    "dtype",
                    "Data type",
                    [
                        ("Float32", "float32"),
                        ("Float16", "float16"),
                        ("BFloat16", "bfloat16"),
                    ],
                )
                yield self._input(
                    provider_id,
                    "max_reference_duration",
                    "Max reference duration",
                )
                yield self._select(
                    provider_id, "language", "Default language", _LANGUAGE_OPTIONS
                )
                yield self._switch(provider_id, "voice_cloning", "Voice cloning")
                yield self._switch(provider_id, "multi_speaker", "Multi-speaker")
                yield self._input(provider_id, "speaker_delimiter", "Speaker delimiter")
                yield self._switch(
                    provider_id, "track_performance", "Performance tracking"
                )
                yield self._input(provider_id, "max_new_tokens", "Max new tokens")
                yield self._input(provider_id, "temperature", "Temperature")
                yield self._input(provider_id, "top_p", "Top P")
                yield self._input(
                    provider_id, "repetition_penalty", "Repetition penalty"
                )
                return

            if provider_id == "alltalk":
                yield self._input(
                    provider_id,
                    "server_url",
                    "Server URL",
                    placeholder="http://127.0.0.1:7851",
                )
                yield self._select(
                    provider_id, "language", "Default language", _LANGUAGE_OPTIONS
                )

    def _collect_visible_state(self) -> None:
        """Copy mounted widget values into the in-memory draft."""
        try:
            provider = self.query_one("#settings-speech-default-provider", Select).value
            model_mode = self.query_one("#settings-speech-model-policy", Select).value
            voice_mode = self.query_one("#settings-speech-voice-policy", Select).value
            response_format = self.query_one(
                "#settings-speech-output-format", Select
            ).value
            if isinstance(provider, str):
                self.state.defaults.provider_id = provider
            if isinstance(model_mode, str):
                self.state.defaults.model_mode = model_mode
            model_value = self.query_one("#settings-speech-model-value", Input).value
            self.state.defaults.model_id = (
                model_value if self.state.defaults.model_mode == "exact" else None
            )
            if isinstance(voice_mode, str):
                self.state.defaults.voice_mode = voice_mode
            voice_value = self.query_one("#settings-speech-voice-value", Input).value
            self.state.defaults.voice_id = (
                voice_value if self.state.defaults.voice_mode == "exact" else None
            )
            if isinstance(response_format, str):
                self.state.defaults.response_format = response_format
            speed = self.query_one("#settings-speech-speed", Input).value
            try:
                self.state.defaults.speed = float(speed)
            except ValueError:
                self.state.defaults.speed = speed  # type: ignore[assignment]
        except QueryError:
            pass

        values = self.state.providers[self.configure_provider]
        for field_id in GLOBAL_TTS_PROVIDER_FIELD_IDS[self.configure_provider]:
            if field_id == "credential":
                continue
            selector = f"#{self._field_dom_id(self.configure_provider, field_id)}"
            try:
                widget = self.query_one(selector)
            except QueryError:
                continue
            if isinstance(widget, Input):
                values[field_id] = widget.value
            elif isinstance(widget, Select):
                if isinstance(widget.value, str):
                    values[field_id] = widget.value
            elif isinstance(widget, Switch):
                values[field_id] = widget.value

    def has_unsaved_changes(self) -> bool:
        """Return whether any non-secret global value differs from its baseline."""
        self._collect_visible_state()
        try:
            if (
                self.state.defaults.snapshot()
                != self.original_state.defaults.snapshot()
            ):
                return True
        except GlobalSpeechTTSValidationError:
            if self.state.defaults != self.original_state.defaults:
                return True

        for provider_id in BUILT_IN_TTS_PROVIDER_ORDER:
            try:
                proposal = build_global_speech_tts_save_proposal(
                    self.original_state,
                    self.state,
                    configure_provider=provider_id,
                )
            except GlobalSpeechTTSValidationError:
                if (
                    self.state.providers[provider_id]
                    != self.original_state.providers[provider_id]
                ):
                    return True
                continue
            if proposal.settings or proposal.delete_setting_keys:
                return True
        return False

    def _announce_draft_state(self) -> None:
        """Publish the latest safe draft snapshot to the Settings shell."""
        is_modified = self.has_unsaved_changes()
        self.post_message(
            self.DraftModified(
                is_modified,
                self.state,
                self.original_state,
                self.configure_provider,
            )
        )

    def _set_result(self, text: str, *, severity: str | None = None) -> None:
        self.result_text = text
        try:
            self.query_one("#settings-speech-save-result", Static).update(text)
        except QueryError:
            pass
        if severity is not None:
            self.app.notify(text, severity=severity)

    def _set_save_pending(self, pending: bool) -> None:
        """Prevent overlapping persistence requests while keeping draft edits live."""
        for selector in (
            "#settings-speech-save",
            "#settings-speech-openai-credential-edit",
            "#settings-speech-openai-credential-clear",
            "#settings-speech-elevenlabs-credential-edit",
            "#settings-speech-elevenlabs-credential-clear",
        ):
            try:
                self.query_one(selector, Button).disabled = pending
            except QueryError:
                continue

    def _clear_validation_errors(self) -> None:
        self._field_errors.clear()
        for error in self.query(".settings-speech-field-error"):
            if isinstance(error, Static):
                error.update("")
                error.remove_class("settings-speech-field-error-visible")

    def _show_validation_error(self, error: GlobalSpeechTTSValidationError) -> None:
        key = (error.provider_id, error.field_id)
        self._field_errors[key] = str(error)
        selector = (
            f"#{self._field_dom_id(error.provider_id, error.field_id)}"
            if error.provider_id != "defaults"
            else {
                "provider_id": "#settings-speech-default-provider",
                "model_mode": "#settings-speech-model-policy",
                "default_model": "#settings-speech-model-value",
                "voice_mode": "#settings-speech-voice-policy",
                "default_voice": "#settings-speech-voice-value",
                "response_format": "#settings-speech-output-format",
                "default_speed": "#settings-speech-speed",
            }.get(error.field_id, "#settings-speech-default-provider")
        )
        try:
            error_widget = self.query_one(f"{selector}-error", Static)
            error_widget.update(str(error))
            error_widget.add_class("settings-speech-field-error-visible")
        except QueryError:
            pass
        try:
            self.query_one(selector).focus()
        except QueryError:
            pass
        self._set_result(str(error), severity="error")

    def request_save(self) -> None:
        """Validate locally and post one atomic ordinary-save proposal."""
        self._collect_visible_state()
        if self._latest_request_id is not None:
            self._set_result("A global Speech & TTS save is already in progress.")
            return
        self._clear_validation_errors()
        try:
            proposal = build_global_speech_tts_save_proposal(
                self.original_state,
                self.state,
                configure_provider=self.configure_provider,
            )
            defaults_changed = (
                proposal.preferences != self.original_state.defaults.snapshot()
            )
        except GlobalSpeechTTSValidationError as error:
            self._show_validation_error(error)
            return

        if (
            not defaults_changed
            and not proposal.settings
            and not proposal.delete_setting_keys
        ):
            self._set_result("No global Speech & TTS changes to save.")
            return
        request_id = self._next_request_id
        self._next_request_id += 1
        self._latest_request_id = request_id
        self._set_save_pending(True)
        self._pending_credential_mutation = None
        self._pending_saved_defaults = (
            deepcopy(self.state.defaults) if defaults_changed else None
        )
        self._pending_saved_provider_id = (
            self.configure_provider
            if proposal.settings or proposal.delete_setting_keys
            else None
        )
        self._pending_saved_provider_values = (
            deepcopy(self.state.providers[self.configure_provider])
            if proposal.settings or proposal.delete_setting_keys
            else None
        )
        self._set_result("Saving global Speech & TTS settings locally…")
        self.post_message(
            STTSSettingsSaveEvent(
                proposal.settings,
                delete_setting_keys=proposal.delete_setting_keys,
                preferences=proposal.preferences,
                request_id=request_id,
                reply_to=self,
            )
        )

    def submit_credential_mutation(
        self,
        provider_id: str,
        intent: CredentialIntent,
        value: str | None,
    ) -> None:
        """Post one separately confirmed Set/Replace/Clear mutation."""
        if self._latest_request_id is not None:
            raise ValueError("Wait for the current Settings save to finish")
        if provider_id not in self.state.credentials:
            raise ValueError("This provider has no Settings credential operation")
        mutation = build_credential_mutation(
            self.state.credentials[provider_id],
            intent,
            value,
        )
        request_id = self._next_request_id
        self._next_request_id += 1
        self._latest_request_id = request_id
        self._set_save_pending(True)
        self._pending_credential_mutation = mutation
        self._pending_saved_defaults = None
        self._pending_saved_provider_id = None
        self._pending_saved_provider_values = None
        settings = {} if mutation.delete else {mutation.setting_key: mutation.value}
        delete_keys = (mutation.setting_key,) if mutation.delete else ()
        self._set_result(
            f"Applying explicit {intent.value} credential operation locally…"
        )
        self.post_message(
            STTSSettingsSaveEvent(
                settings,
                delete_setting_keys=delete_keys,
                request_id=request_id,
                reply_to=self,
            )
        )

    def receive_stts_settings_save_result(
        self,
        result: STTSSettingsSaveResult,
    ) -> None:
        """Apply only the latest bounded persistence/reconfiguration result."""
        if result.request_id != self._latest_request_id:
            return
        self._latest_request_id = None
        self._set_save_pending(False)
        mutation = self._pending_credential_mutation
        saved_defaults = self._pending_saved_defaults
        saved_provider_id = self._pending_saved_provider_id
        saved_provider_values = self._pending_saved_provider_values
        self._pending_credential_mutation = None
        self._pending_saved_defaults = None
        self._pending_saved_provider_id = None
        self._pending_saved_provider_values = None
        if not result.persisted:
            failure = (
                " before replacing the config file"
                if result.failure_phase == "before_replace"
                else " while reloading config caches"
                if result.failure_phase == "cache_reload"
                else ""
            )
            self._set_result(
                f"Global Speech & TTS settings were not saved{failure}.",
                severity="error",
            )
            return

        if mutation is not None:
            previous = self.state.credentials[mutation.provider_id]
            local_saved = not mutation.delete
            environment_effective = previous.source is CredentialSource.ENVIRONMENT
            source = (
                CredentialSource.ENVIRONMENT
                if environment_effective
                else CredentialSource.SAVED_LOCAL
                if local_saved
                else CredentialSource.MISSING
            )
            updated = replace(
                previous,
                source=source,
                local_saved=local_saved,
                local_shadowed=environment_effective and local_saved,
            )
            self.state.credentials[mutation.provider_id] = updated
            self.original_state.credentials[mutation.provider_id] = updated
        else:
            if saved_defaults is not None:
                self.original_state.defaults = saved_defaults
            if saved_provider_id is not None and saved_provider_values is not None:
                self.original_state.providers[saved_provider_id] = saved_provider_values

        cache_reload_failed = result.failure_phase == "cache_reload"
        if result.provider_statuses:
            handoff = ", ".join(
                f"{TTS_PROVIDER_LABELS[provider]}: {status}"
                for provider, status in result.provider_statuses.items()
            )
        else:
            handoff = "no provider adapter recreation needed"
        if cache_reload_failed:
            result_copy = (
                "Saved locally, but the runtime configuration cache reload failed; "
                f"restart or retry. Provider handoff: {handoff}."
            )
        else:
            result_copy = f"Saved locally. Runtime reconfiguration: {handoff}."
        self._set_result(
            result_copy,
            severity=(
                "warning"
                if cache_reload_failed
                or "unavailable" in result.provider_statuses.values()
                else "information"
            ),
        )
        self._announce_draft_state()
        if mutation is not None:
            # Credential controls derive their Set/Replace/Clear affordances
            # from safe source metadata, so a successful explicit mutation
            # must repaint the selected form without ever projecting a secret.
            self.call_later(self.recompose)

    def _credential_editor_result(
        self,
        provider_id: str,
        intent: CredentialIntent,
        value: str | None,
    ) -> None:
        if value is None:
            return
        try:
            self.submit_credential_mutation(provider_id, intent, value)
        except (TypeError, ValueError) as error:
            self._set_result(str(error), severity="error")

    def _credential_clear_result(self, provider_id: str, confirmed: bool) -> None:
        if not confirmed:
            return
        try:
            self.submit_credential_mutation(
                provider_id,
                CredentialIntent.CLEAR,
                None,
            )
        except (TypeError, ValueError) as error:
            self._set_result(str(error), severity="error")

    def _path_picker_result(self, target_selector: str, path: Path | None) -> None:
        if path is None:
            return
        try:
            self.query_one(target_selector, Input).value = str(path)
        except QueryError:
            return

    @on(Select.Changed, "#settings-speech-configure-provider")
    async def handle_configure_provider_changed(self, event: Select.Changed) -> None:
        if self._syncing or not isinstance(event.value, str):
            return
        if event.value == self.configure_provider:
            return
        self._collect_visible_state()
        self.configure_provider = event.value
        await self.recompose()
        self._announce_draft_state()

    @on(Select.Changed, "#settings-speech-default-provider")
    async def handle_default_provider_changed(self, event: Select.Changed) -> None:
        if self._syncing or not isinstance(event.value, str):
            return
        if event.value == self.state.defaults.provider_id:
            return
        self._collect_visible_state()
        if event.value == "audio_cpp":
            self.state.defaults.response_format = "wav"
            self.state.defaults.speed = 1.0
        await self.recompose()
        self._announce_draft_state()

    @on(Select.Changed, "#settings-speech-model-policy")
    async def handle_model_policy_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, str):
            return
        if event.value == self.state.defaults.model_mode:
            return
        self._collect_visible_state()
        if event.value != "exact":
            self.state.defaults.model_id = None
        await self.recompose()
        self._announce_draft_state()

    @on(Select.Changed, "#settings-speech-voice-policy")
    async def handle_voice_policy_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, str):
            return
        if event.value == self.state.defaults.voice_mode:
            return
        self._collect_visible_state()
        if event.value != "exact":
            self.state.defaults.voice_id = None
        await self.recompose()
        self._announce_draft_state()

    @on(Input.Changed, ".settings-speech-draft-field")
    @on(Select.Changed, ".settings-speech-draft-field")
    @on(Switch.Changed, ".settings-speech-draft-field")
    def handle_draft_field_changed(
        self,
        _event: Input.Changed | Select.Changed | Switch.Changed,
    ) -> None:
        if not self._syncing:
            self._announce_draft_state()

    async def revert_to_saved(self) -> None:
        """Restore the last successfully loaded or published snapshot."""
        self._clear_validation_errors()
        self.state = deepcopy(self.original_state)
        self.result_text = "Reverted to the last successfully loaded global values."
        await self.recompose()
        self._announce_draft_state()

    @on(Button.Pressed, "#settings-speech-revert")
    async def handle_revert(self, event: Button.Pressed) -> None:
        event.stop()
        await self.revert_to_saved()

    @on(Button.Pressed, "#settings-speech-save")
    def handle_save(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_save()

    @on(Button.Pressed, "#settings-speech-restore-defaults")
    async def handle_restore_defaults(self, event: Button.Pressed) -> None:
        event.stop()
        self._collect_visible_state()
        self._clear_validation_errors()
        self.state = restore_non_secret_defaults(
            self.state,
            configure_provider=self.configure_provider,
        )
        self.result_text = (
            "Non-secret defaults restored in the draft; choose Save to persist them."
        )
        await self.recompose()
        self._announce_draft_state()

    @on(Button.Pressed, "#settings-speech-open-lab")
    @on(Button.Pressed, "#settings-speech-open-lab-bottom")
    def handle_open_lab(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(NavigateToScreen("stts"))

    @on(Button.Pressed, ".settings-speech-credential Button")
    def handle_credential_action(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        provider_id = next(
            (
                provider
                for provider in self.state.credentials
                if button_id.startswith(f"settings-speech-{provider}-credential-")
            ),
            None,
        )
        if provider_id is None:
            return
        credential_state = self.state.credentials[provider_id]
        if button_id.endswith("-edit"):
            intent = (
                CredentialIntent.REPLACE
                if credential_state.local_saved
                else CredentialIntent.SET
            )
            self.app.push_screen(
                _CredentialEditorModal(TTS_PROVIDER_LABELS[provider_id], intent),
                lambda value: self._credential_editor_result(
                    provider_id,
                    intent,
                    value,
                ),
            )
            return
        if button_id.endswith("-clear"):
            self.app.push_screen(
                _CredentialClearModal(TTS_PROVIDER_LABELS[provider_id]),
                lambda confirmed: self._credential_clear_result(
                    provider_id,
                    confirmed,
                ),
            )

    @on(Button.Pressed, ".settings-speech-path-picker")
    def handle_path_picker(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        if not button_id.endswith("-browse"):
            return
        target_selector = f"#{button_id.removesuffix('-browse')}"
        directory_fields = {
            "#settings-speech-chatterbox-voice-resource-directory",
            "#settings-speech-higgs-voice-resource-directory",
        }
        if target_selector in directory_fields:
            picker = SelectDirectory(title="Choose voice resource directory")
        else:
            filters = Filters(("Model and resource files", lambda _path: True))
            picker = FileOpen(
                title="Choose Speech & TTS resource",
                filters=filters,
            )
        self.app.push_screen(
            picker,
            lambda path: self._path_picker_result(target_selector, path),
        )
