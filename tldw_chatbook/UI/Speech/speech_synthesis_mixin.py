"""The synthesis path, shared by the legacy Playground and the rebuild.

Moved here rather than copied. These five methods are the whole generate
flow -- readiness checks, provider/voice resolution, the request build and
the worker launch -- and a second copy living beside the first is how the
two drift until only one of them is correct.

They query their controls by id, so any host that mounts the Playground's
control ids can inherit them unchanged. That is why the rebuild kept the
legacy ids instead of renaming to a cleaner scheme: identity is the seam.

A host must provide, beyond the controls themselves:

- ``reference_audio_path`` / ``higgs_reference_audio_path``: the chosen clip
  for the providers that synthesize from one, or ``None``.
- ``_provider_ids``: the provider values currently offered, used to tell a
  stale selection from an unknown one.
- ``_generation_operation_id``: the in-flight request, or ``None``.

``init_synthesis_state`` sets all four; call it from the host's ``__init__``.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from loguru import logger
from textual.widgets import Button, Input, RichLog, Select, Switch, TextArea

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.TTS import STTSPlaygroundRequest
from tldw_chatbook.UI.stts_playground_catalog import (
    AUDIO_CPP_PROVIDER_ID,
    LOADING_SELECT_VALUE,
    SERVER_DEFAULT_VOICE_ID,
    UNAVAILABLE_SELECT_VALUE,
    SelectValue,
    voice_id_for_request,
)


class SpeechSynthesisMixin:
    """The generate path, independent of which pane hosts the controls."""

    def init_synthesis_state(self) -> None:
        """Initialise the state the synthesis path reads.

        Call from the host's ``__init__`` before the controls mount.
        """
        #: Reference clip for providers that synthesize from one.
        self.reference_audio_path: Any = None
        #: Voice sample for Higgs' own cloning flow.
        self.higgs_reference_audio_path: Any = None
        #: Provider values currently offered, so a selection that is merely
        #: stale can be told apart from one that was never valid.
        self._provider_ids: frozenset[str] = frozenset()
        #: The in-flight generation, or None when idle.
        self._generation_operation_id: Any = None

    def _sync_generate_enabled(self) -> None:
        text_present = bool(self.query_one("#tts-text-input", TextArea).text.strip())
        provider_id = self._selected_provider_id
        revision_matches = False
        if (
            provider_id is not None
            and self._tts_service is not None
            and provider_id in self._catalog_configuration_revisions
        ):
            revision_matches = self._catalog_configuration_revisions[
                provider_id
            ] == self._tts_service.configuration_revision(provider_id)
        self.query_one("#tts-generate-btn", Button).disabled = not (
            text_present
            and self._catalog_generation_allowed
            and revision_matches
            and self._generation_operation_id is None
            and not getattr(self.app, "_is_generating", False)
        )

    def _generation_readiness_error(
        self,
        provider_id: object,
        model_id: object,
    ) -> str | None:
        """Return fixed UI copy when a generation snapshot is not authoritative."""
        if self._generation_operation_id is not None:
            return "TTS generation is already in progress"

        handler = getattr(self.app, "_stts_handler", None)
        state_getter = getattr(handler, "playground_state", None)
        if callable(state_getter):
            try:
                if getattr(state_getter(), "generation_active", False):
                    return "TTS generation is already in progress"
            except Exception:
                return "The TTS service is unavailable"

        if (
            not isinstance(provider_id, str)
            or provider_id != self._selected_provider_id
            or provider_id not in self._provider_ids
        ):
            return "Please select a valid TTS provider"
        if not isinstance(model_id, str):
            return "Please select a valid TTS model"

        service = self._tts_service
        catalog = self._catalogs.get(provider_id)
        if service is None or catalog is None:
            return "The selected provider catalog is not ready; refresh models"
        revision_matches = self._catalog_configuration_revisions.get(
            provider_id
        ) == service.configuration_revision(provider_id)
        if (
            provider_id in self._pending_voice_selections
            and provider_id not in self._stale_providers
            and catalog.health.state == "available"
            and catalog.health.fresh
            and revision_matches
        ):
            return "Voices are still loading; wait before generating"
        if (
            provider_id in self._stale_providers
            or not self._catalog_generation_allowed
            or catalog.health.state != "available"
            or not catalog.health.fresh
            or not revision_matches
        ):
            return "The selected provider catalog is stale; refresh models"
        if not any(model.model_id == model_id for model in catalog.models):
            return "The selected model is no longer available; refresh models"
        return None

    def _get_select_key(self, select_widget: Select) -> SelectValue | None:
        """Return exact canonical values for catalog-driven controls."""
        current = select_widget.value
        if current is LOADING_SELECT_VALUE or current is UNAVAILABLE_SELECT_VALUE:
            return None
        if current is SERVER_DEFAULT_VOICE_ID:
            return current
        if not isinstance(current, str):
            return None
        if select_widget.id == "tts-language-select":
            for language_id, display_name in select_widget._options:
                if display_name == current:
                    return str(language_id)
        return current

    def _is_valid_voice(self, voice: object) -> bool:
        """Check if a voice value is valid (not a separator)."""
        return bool(voice) and not str(voice).startswith("_separator")

    def _generate_tts(self) -> None:
        """Generate TTS audio"""
        if self._generation_operation_id is not None:
            self.app.notify(
                "TTS generation is already in progress",
                severity="warning",
            )
            return

        # Get form values
        text_area = self.query_one("#tts-text-input", TextArea)
        text = text_area.text.strip()

        if not text:
            self.app.notify("Please enter text to synthesize", severity="warning")
            return

        provider_select = self.query_one("#tts-provider-select", Select)
        voice_select = self.query_one("#tts-voice-select", Select)
        model_select = self.query_one("#tts-model-select", Select)

        # Get the actual keys, not display text
        provider = self._get_select_key(provider_select) or provider_select.value
        voice = self._get_select_key(voice_select) or voice_select.value
        model = self._get_select_key(model_select) or model_select.value

        readiness_error = self._generation_readiness_error(provider, model)
        if readiness_error is not None:
            self._sync_generate_enabled()
            self.app.notify(readiness_error, severity="warning")
            return

        # Validate voice selection
        if not self._is_valid_voice(voice):
            self.app.notify("Please select a valid voice", severity="warning")
            return
        speed = float(self.query_one("#tts-speed-input", Input).value or "1.0")
        format_select = self.query_one("#tts-format-select", Select)
        format = format_select.value

        # Ensure format has a valid value
        if not format or format == Select.BLANK or str(format) == "Select.BLANK":
            format = "mp3"
            logger.warning("No format selected, defaulting to mp3")
        elif isinstance(format, tuple):
            # If it's a tuple, take the first element
            format = format[0]

        # Additional validation - also handle uppercase
        valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        format_lower = format.lower() if isinstance(format, str) else format
        if format_lower in valid_formats:
            format = format_lower
        else:
            logger.warning("Invalid Playground audio format; using mp3")
            format = "mp3"

        # Collect provider-specific settings
        extra_params = {}
        if provider == "kokoro":
            language_select = self.query_one("#tts-language-select", Select)
            language = self._get_select_key(language_select) or language_select.value
            extra_params["language"] = language
            # Add ONNX setting
            use_onnx = self.query_one("#tts-kokoro-use-onnx", Switch).value
            extra_params["use_onnx"] = use_onnx
        elif provider == "elevenlabs":
            stability = float(
                self.query_one("#tts-stability-input", Input).value or "0.5"
            )
            similarity = float(
                self.query_one("#tts-similarity-input", Input).value or "0.8"
            )
            style = float(self.query_one("#tts-style-input", Input).value or "0.0")
            speaker_boost = self.query_one("#tts-speaker-boost-switch", Switch).value
            extra_params["stability"] = stability
            extra_params["similarity_boost"] = similarity
            extra_params["style"] = style
            extra_params["use_speaker_boost"] = speaker_boost
        elif provider == "chatterbox":
            exaggeration = float(
                self.query_one("#tts-exaggeration-input", Input).value or "0.5"
            )
            cfg_weight = float(
                self.query_one("#tts-cfg-weight-input", Input).value or "0.5"
            )
            temperature = float(
                self.query_one("#tts-temperature-input", Input).value or "0.5"
            )
            num_candidates = int(
                self.query_one("#tts-num-candidates-input", Input).value or "1"
            )
            validate_whisper = self.query_one(
                "#tts-validate-whisper-switch", Switch
            ).value
            preprocess_text = self.query_one(
                "#tts-preprocess-text-switch", Switch
            ).value
            normalize_audio = self.query_one(
                "#tts-normalize-audio-switch", Switch
            ).value
            target_db = float(
                self.query_one("#tts-target-db-input", Input).value or "-20.0"
            )
            random_seed_input = self.query_one(
                "#tts-random-seed-input", Input
            ).value.strip()

            extra_params["exaggeration"] = exaggeration
            extra_params["cfg_weight"] = cfg_weight
            extra_params["temperature"] = temperature
            extra_params["num_candidates"] = num_candidates
            extra_params["validate_with_whisper"] = validate_whisper
            extra_params["preprocess_text"] = preprocess_text
            extra_params["normalize_audio"] = normalize_audio
            extra_params["target_db"] = target_db
            if random_seed_input:
                extra_params["random_seed"] = int(random_seed_input)

            # Handle voice selection
            if voice == "custom" and self.reference_audio_path:
                # Use custom voice with reference audio
                voice = f"custom:{self.reference_audio_path}"
            elif voice == "custom":
                self.app.notify(
                    "Please select reference audio for custom voice", severity="warning"
                )
                self.query_one("#tts-generate-btn", Button).disabled = False
                return
            elif voice not in [
                "default",
                "custom",
                "_separator",
                "_separator2",
            ] and not voice.startswith(("custom:", "profile:")):
                # This is a saved profile - format it as profile:name
                voice = f"profile:{voice}"
        elif provider == "higgs":
            # Collect Higgs-specific parameters
            temperature = float(
                self.query_one("#tts-higgs-temperature-input", Input).value
            )
            top_p = float(self.query_one("#tts-higgs-top-p-input", Input).value)
            repetition_penalty = float(
                self.query_one("#tts-higgs-repetition-penalty-input", Input).value
            )
            enable_voice_cloning = self.query_one(
                "#tts-higgs-voice-cloning-switch", Switch
            ).value
            enable_multi_speaker = self.query_one(
                "#tts-higgs-multi-speaker-switch", Switch
            ).value
            speaker_delimiter = self.query_one(
                "#tts-higgs-delimiter-input", Input
            ).value

            extra_params["temperature"] = temperature
            extra_params["top_p"] = top_p
            extra_params["repetition_penalty"] = repetition_penalty
            extra_params["enable_voice_cloning"] = enable_voice_cloning
            extra_params["enable_multi_speaker"] = enable_multi_speaker
            extra_params["speaker_delimiter"] = speaker_delimiter

            # Handle voice selection for custom upload
            if (
                voice == "custom"
                and hasattr(self, "higgs_reference_audio_path")
                and self.higgs_reference_audio_path
            ):
                # Use custom voice with reference audio
                voice = f"custom:{self.higgs_reference_audio_path}"
            elif voice == "custom":
                self.app.notify(
                    "Please upload reference audio for custom voice", severity="warning"
                )
                self.query_one("#tts-generate-btn", Button).disabled = False
                return
            elif voice not in [
                "professional_female",
                "warm_female",
                "storyteller_male",
                "deep_male",
                "energetic_female",
                "soft_female",
                "custom",
                "_separator",
                "_separator2",
            ] and not voice.startswith(("custom:", "profile:")):
                # This is a saved profile - format it as profile:name
                voice = f"profile:{voice}"

        # Log the request
        log = self.query_one("#tts-generation-log", RichLog)
        log.write("[bold blue]Generating TTS...[/bold blue]")
        log.write(f"Speed: {speed}")
        log.write(f"Format: {format}")
        log.write(f"Text length: {len(text)} characters")

        if not isinstance(provider, str) or provider not in self._provider_ids:
            self.app.notify("Please select a valid TTS provider", severity="warning")
            return
        if not isinstance(model, str):
            self.app.notify("Please select a valid TTS model", severity="warning")
            return
        if not isinstance(format, str):
            self.app.notify("Please select a valid audio format", severity="warning")
            return
        voice_id = voice_id_for_request(voice)
        if provider == AUDIO_CPP_PROVIDER_ID:
            format = "wav"
            speed = 1.0
            extra_params = {}

        # Disable generate button
        self.query_one("#tts-generate-btn", Button).disabled = True

        request = STTSPlaygroundRequest(
            operation_id=str(uuid4()),
            provider_id=provider,
            model_id=model,
            text=text,
            voice_id=voice_id,
            response_format=format,
            speed=speed,
            options=extra_params,
        )
        self._generation_operation_id = request.operation_id
        self.app.post_message(STTSPlaygroundGenerateEvent(request))
