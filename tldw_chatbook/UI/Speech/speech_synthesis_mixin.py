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

from collections.abc import Mapping
from typing import Any, cast
from uuid import uuid4

from loguru import logger
from textual.css.query import NoMatches
from textual.widgets import Button, Input, RichLog, Select, Switch, TextArea

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.TTS import (
    STTSPlaygroundProfilePreview,
    STTSPlaygroundRequest,
    TTSPlaygroundSelectionPreset,
)
from tldw_chatbook.TTS.adapter_types import TTSRegistryClosedError
from tldw_chatbook.TTS.effective_settings import (
    TTS_REQUEST_OPTION_KEYS,
    TTSSelectionOverrides,
    TTSStudioDraftSelection,
)
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot
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

    def _profile_preview_for_request(self) -> STTSPlaygroundProfilePreview | None:
        """Project only bounded identity for a reference-bearing preview."""

        preset = self._profile_preset
        if type(preset) is not TTSPlaygroundSelectionPreset:
            return None
        if (
            preset.profile_id is None
            or preset.repository_generation is None
            or preset.profile_revision is None
        ):
            return None
        return STTSPlaygroundProfilePreview(
            profile_id=preset.profile_id,
            repository_generation=preset.repository_generation,
            profile_revision=preset.profile_revision,
        )

    def _build_playground_request(
        self,
        *,
        operation_id: str,
        provider: str,
        model: str,
        text: str,
        voice_id: str | None,
        response_format: str,
        speed: float,
        options: Mapping[str, Any],
        studio_draft: TTSStudioDraftSelection | None,
        studio_preferences: StudioTTSPreferencesSnapshot | None,
    ) -> STTSPlaygroundRequest:
        """Build one immutable request without exposing profile reference data."""

        clone_snapshot = None
        clone_snapshot_factory = getattr(self, "_clone_audition_for_request", None)
        if callable(clone_snapshot_factory):
            clone_snapshot = clone_snapshot_factory(provider, model)

        return STTSPlaygroundRequest(
            operation_id=operation_id,
            provider_id=provider,
            model_id=model,
            text=text,
            voice_id=voice_id,
            response_format=response_format,
            speed=speed,
            options=options,
            studio_draft=studio_draft,
            studio_preferences=(
                studio_preferences if studio_draft is not None else None
            ),
            clone_audition=clone_snapshot,
            profile_preview=self._profile_preview_for_request(),
        )

    def _effective_generation_selection(self) -> tuple[object, object]:
        """The (provider_id, model_id) pair the readiness gate should judge.

        An active profile preset's own exact values when one is set --
        mirroring the override `_generate_tts` applies for the same reason
        (a press generates against the preset, not against whatever the
        controls happen to show) -- otherwise the tracked provider
        selection and whatever the model control currently displays.
        """
        preset = self._profile_preset
        if preset is not None:
            return preset.provider_id, preset.model_id
        model_select = self.query_one("#tts-model-select", Select)
        model_id = self._get_select_key(model_select) or model_select.value
        return self._selected_provider_id, model_id

    def _sync_generate_enabled(self) -> None:
        """Keep the button's visual state and `_generate_tts`'s eligibility
        from disagreeing by deriving both from the same gate.

        Used to reimplement its own condition tree in parallel with
        `_generation_readiness_error` -- profile-preset-aware here,
        NOT profile-preset-aware there (that branch never existed in the
        rebuilt mixin; the retired legacy widget had it, this file did
        not). The split let a keyboard press reach `action_generate_tts()`
        and fire a real generation while the button it mirrors sat
        disabled -- TASK-2951's binding-mirror fix is what first made that
        keyboard path reachable at all, which is what turned a latent gap
        into a live bypass. Fixed by asking the one shared gate instead.
        """
        text_present = bool(self.query_one("#tts-text-input", TextArea).text.strip())
        provider_id, model_id = self._effective_generation_selection()
        readiness_error = (
            self._generation_readiness_error(provider_id, model_id)
            if text_present
            else "Enter text before generating speech"
        )
        if (
            provider_id == "audio_cpp"
            and getattr(self, "_audio_cpp_lifecycle_busy", None) is not None
        ):
            readiness_error = "An audio.cpp operation is in progress."
        elif getattr(self.app, "_is_generating", False):
            readiness_error = "TTS generation is already in progress"
        generation_disabled = readiness_error is not None
        generate = self.query_one("#tts-generate-btn", Button)
        generate.disabled = generation_disabled
        generate.tooltip = (
            readiness_error
            if readiness_error is not None
            else "Generate speech with the current Speech Lab controls"
        )
        try:
            repeat = self.query_one("#audio-generate-again-btn", Button)
        except NoMatches:
            pass
        else:
            artifact_missing = getattr(self, "current_audio_artifact", None) is None
            repeat.disabled = bool(generation_disabled or artifact_missing)
            repeat.tooltip = (
                "Generate audio before generating another result"
                if artifact_missing
                else readiness_error
                if readiness_error is not None
                else "Generate another result with the current Speech Lab controls"
            )

    def _generation_readiness_error(
        self,
        provider_id: object,
        model_id: object,
        *,
        clone_action: bool = False,
    ) -> str | None:
        """Return fixed UI copy when a generation snapshot is not authoritative.

        The single authoritative readiness gate: `_sync_generate_enabled`
        (the button's visual state) and `_generate_tts` (what a press or a
        keyboard `action_generate_tts()` actually attempts) both consult
        this and only this, so they cannot disagree.

        An active profile preset takes the branch below over entirely
        rather than falling into the general provider/catalog checks: a
        pending voice validation or a preset marked "unavailable" block
        unconditionally, but a merely "unverified" preset (a naturally
        stale catalog, not a real provider-configuration change) is
        deliberately let through -- callers that care show the "unverified,
        one exact attempt" warning themselves (`_generate_tts` does).
        Ported verbatim from the retired legacy widget's own version of
        this method (`git show HEAD:tldw_chatbook/UI/STTS_Window.py`, prior
        to this branch's widget deletion) -- that branch never existed
        here, which is the root cause TASK-2951's re-review found.
        """
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

        preset = self._profile_preset
        if preset is not None:
            if self._profile_voice_validation_token is not None:
                return (
                    "The exact profile voice is still being checked; "
                    "wait before generating"
                )
            if self._profile_effective_availability == "unavailable":
                return (
                    "The exact profile selection is unavailable; return to Voice "
                    "profiles and choose Edit"
                )
            if provider_id != preset.provider_id or model_id != preset.model_id:
                return "The exact profile selection changed; choose Preview again"
            service = self._tts_service
            if service is None:
                return "The TTS service is unavailable"
            try:
                current_revision = service.configuration_revision(preset.provider_id)
            except (KeyError, TTSRegistryClosedError):
                return "The TTS service is unavailable"
            if (
                self._profile_configuration_revision is None
                or current_revision != self._profile_configuration_revision
            ):
                return "TTS provider settings changed; refresh models"
            if not self._catalog_generation_allowed:
                return (
                    "The exact profile selection is not ready; retry from Voice "
                    "profiles"
                )
            return None

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
        reference_only_clone = self._is_reference_only_clone_action(
            provider_id,
            model_id,
            clone_action=clone_action,
        )
        if (
            provider_id in self._pending_voice_selections
            and not reference_only_clone
            and provider_id not in self._stale_providers
            and catalog.health.state == "available"
            and catalog.health.fresh
            and revision_matches
        ):
            return "Voices are still loading; wait before generating"
        if (
            provider_id in self._stale_providers
            or (
                not reference_only_clone
                and not self._catalog_generation_allowed
            )
            or catalog.health.state != "available"
            or not catalog.health.fresh
            or not revision_matches
        ):
            return "The selected provider catalog is stale; refresh models"
        # An OpenAI model outside the static official catalog is a pinned
        # custom-endpoint id ("no catalog check", TASK-15421) — there is
        # nothing to verify it against, and nothing can "disappear" from a
        # static catalog, so the staleness copy would be false here.
        if provider_id != "openai" and not any(
            model.model_id == model_id for model in catalog.models
        ):
            return "The selected model is no longer available; refresh models"
        clone_readiness = getattr(self, "_clone_setup_generation_error", None)
        if callable(clone_readiness):
            clone_error = clone_readiness(
                provider_id,
                model_id,
                clone_action=clone_action,
            )
            if clone_error is not None:
                return clone_error
        return None

    def _is_reference_only_clone_action(
        self,
        provider_id: object,
        model_id: object,
        *,
        clone_action: bool,
    ) -> bool:
        """Return whether the exact visible action forbids a catalog voice."""

        if not clone_action or provider_id != AUDIO_CPP_PROVIDER_ID:
            return False
        observation = getattr(self, "_audio_cpp_runtime_observation", None)
        projection = None if observation is None else observation.clone_setup
        return bool(
            projection is not None
            and projection.model_id == model_id
            and projection.voice_reference_policy == "reference_only"
        )

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

    def _generate_tts(self, *, clone_action: bool = False) -> None:
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

        voice_select = self.query_one("#tts-voice-select", Select)

        # Get the actual keys, not display text
        voice = self._get_select_key(voice_select) or voice_select.value

        # A profile preset generates against its own exact selection, not
        # whatever the controls currently display (which can transiently
        # show a loading sentinel while a preset's own values are already
        # known) -- the same override `_effective_generation_selection`
        # applies for `_sync_generate_enabled`, so the two cannot disagree
        # about which provider/model are actually in play.
        preset = self._profile_preset
        provider, model = self._effective_generation_selection()
        if preset is not None:
            voice = (
                SERVER_DEFAULT_VOICE_ID if preset.voice_id is None else preset.voice_id
            )
        elif self._is_reference_only_clone_action(
            provider,
            model,
            clone_action=clone_action,
        ):
            # A reference-only recipe rejects any native voice at adapter
            # admission. Ignore a stale/loading catalog selection entirely.
            voice = SERVER_DEFAULT_VOICE_ID

        readiness_error = self._generation_readiness_error(
            provider,
            model,
            clone_action=clone_action,
        )
        if readiness_error is not None:
            self._sync_generate_enabled()
            self.app.notify(readiness_error, severity="warning")
            return
        if preset is not None and self._profile_effective_availability == "unverified":
            self.app.notify(
                "Profile availability is unverified; attempting the exact "
                "selection once without fallback.",
                severity="warning",
            )

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

        studio_preferences = getattr(self, "studio_preferences", None)
        studio_draft = None
        if type(studio_preferences) is StudioTTSPreferencesSnapshot:
            canonical_studio_preferences = cast(
                StudioTTSPreferencesSnapshot,
                studio_preferences,
            )
            allowed_options = TTS_REQUEST_OPTION_KEYS[provider]
            studio_options = {
                key: value
                for key, value in extra_params.items()
                if key in allowed_options
            }
            studio_draft = TTSStudioDraftSelection(
                selection=TTSSelectionOverrides(
                    provider_id=provider,
                    model_mode="exact",
                    model_id=model,
                    voice_mode=("server_default" if voice_id is None else "exact"),
                    voice_id=voice_id,
                    response_format=format,
                    speed=speed,
                    provider_options=studio_options,
                ),
                base_revision=canonical_studio_preferences.revision,
                preview=bool(getattr(self, "_profile_preset", None)),
            )

        # Disable generate button
        self.query_one("#tts-generate-btn", Button).disabled = True
        try:
            self.query_one("#audio-generate-again-btn", Button).disabled = True
        except NoMatches:
            pass

        request = self._build_playground_request(
            operation_id=str(uuid4()),
            provider=provider,
            model=model,
            text=text,
            voice_id=voice_id,
            response_format=format,
            speed=speed,
            options=extra_params,
            studio_draft=studio_draft,
            studio_preferences=studio_preferences,
        )
        self._generation_operation_id = request.operation_id
        self._profile_save_suppressed = True
        self._sync_save_profile_action()
        self.app.post_message(STTSPlaygroundGenerateEvent(request))

    def action_generate_tts(self) -> None:
        """Keyboard shortcut action for generate"""
        self._generate_tts()
