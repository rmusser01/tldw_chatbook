"""The first-run wizard's Voice step.

Moved out of ``FirstRunSetupWizard.py`` (TASK-33921) so that module stays
under its size budget. The class moved whole, so its ``@on`` handlers and
workers stay registered on it; ``FirstRunSetupWizard`` re-exports the name.
Patch this module, not the wizard, to replace what the step calls.
"""

from __future__ import annotations

import asyncio
import math
import os
import tempfile
from pathlib import Path
from typing import (
    Any,
    Dict,
    Mapping,
)

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import (
    Horizontal,
    Vertical,
)
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
)

from tldw_chatbook.TTS.omnivoice_artifact_catalog import (
    omnivoice_setup_state,
    run_omnivoice_preflight,
    run_omnivoice_provision,
)
from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupCheckbox,
    SetupRadioButton,
    SetupRadioSet,
    SetupStep,
)
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker, wizard_work
from tldw_chatbook.Widgets.ModelArtifacts import (
    InstallProgressed,
    ModelInstallModal,
    ModelInstallProgress,
    make_progress_callback,
)


class VoiceSetupStep(SetupStep):
    """Compact OpenAI-compatible TTS setup shared by Quick and Full tracks."""

    _SAVE_TIMEOUT_SECONDS = 30.0

    def __init__(self, wizard=None, config=None, **kwargs: Any) -> None:
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._preset = voice_state.VOICE_PRESET_POCKET_TTS
        self._custom_draft: voice_state.VoiceSetupDraft | None = None
        self._verified_draft: voice_state.VoiceSetupDraft | None = None
        self._next_save_request_id = 1
        self._save_request_id: int | None = None
        self._save_draft: voice_state.VoiceSetupDraft | None = None
        self._save_future: asyncio.Future[tuple[bool, str]] | None = None
        self._test_generation = 0
        self._test_in_progress_generation: int | None = None
        self._sample_audio_path: Path | None = None
        self._omnivoice_state: str | None = None
        # Bumped per state check and on leaving OmniVoice; a check applies
        # only while it is still the latest one (a stale read is dropped).
        self._omnivoice_state_generation = 0
        self._omnivoice_installing = False
        self._omnivoice_report: Any = None
        self._omnivoice_seed: int | None = None
        self._save_provider = "openai"
        self._save_use_as_default = False

    @staticmethod
    def _initial_draft() -> voice_state.VoiceSetupDraft:
        return voice_state.VoiceSetupDraft(
            endpoint=voice_state.POCKET_TTS_ENDPOINT,
            authentication_mode="none",
            model_id=voice_state.POCKET_TTS_MODEL,
            voice_id=voice_state.POCKET_TTS_VOICE,
            response_format="wav",
            speed=1.0,
            sample_text="Hello from Chatbook.",
            use_as_default=False,
        )

    def compose_step(self) -> ComposeResult:
        # TASK-21148 (UAT V-1/V-2): outcome first. The step used to open on
        # raw plumbing (endpoint URL, model ids) and hid its human parts —
        # sample text, "Test and Hear", the default toggle — below the fold
        # at 40-row terminals with no hint of what "voice" was even for.
        # Now: purpose line, service choice, try-it controls; the plumbing
        # lives under an Advanced disclosure with unchanged widget ids.
        draft = self._initial_draft()
        with Vertical(classes="setup-voice"):
            yield Static("Set up a voice", classes="setup-title")
            yield Static(
                "Hear replies read aloud — optional. PocketTTS or OmniVoice "
                "run locally, no account needed; skip with Next if you "
                "don't want voice.",
                classes="setup-subtitle",
            )
            yield Label("Service", classes="setup-field-label")
            with SetupRadioSet(id="setup-voice-preset", classes="setup-voice-segmented"):
                yield SetupRadioButton(
                    "PocketTTS",
                    id="setup-voice-preset-pocket",
                    value=True,
                )
                yield SetupRadioButton(
                    "OpenAI",
                    id="setup-voice-preset-official",
                )
                yield SetupRadioButton(
                    "Custom",
                    id="setup-voice-preset-custom",
                )
                yield SetupRadioButton("OmniVoice", id="setup-voice-preset-omnivoice")
            with Vertical(id="setup-voice-omnivoice-panel") as panel:
                panel.display = False
                yield Static(
                    voice_state.OMNIVOICE_CHECKING_COPY,
                    id="setup-voice-omnivoice-status",
                    classes="setup-subtitle",
                )
                install = Button("Install voice model", id="setup-voice-omnivoice-install")
                install.display = False
                yield install
                progress = ModelInstallProgress(None, id="setup-voice-omnivoice-progress")
                progress.display = False
                yield progress
            yield Label("Sample text", classes="setup-field-label")
            yield Input(
                value=draft.sample_text,
                id="setup-voice-sample",
                max_length=500,
            )
            yield Static(
                f"{len(draft.sample_text)} / 500",
                id="setup-voice-sample-count",
                classes="setup-field-help",
            )
            yield Button(
                "Test and Hear",
                id="setup-voice-test",
                variant="primary",
            )
            yield Static(
                "Not tested yet — that's fine. You can save now and test later.",
                id="setup-voice-status",
                classes="setup-subtitle",
            )
            add_key = Button(
                "Add API key in Settings",
                id="setup-voice-add-key",
            )
            add_key.display = False
            yield add_key
            yield SetupCheckbox(
                "Use as default",
                id="setup-voice-default",
                value=False,
            )
            with Collapsible(
                title="Advanced — endpoint, model & output",
                collapsed=True,
                id="setup-voice-advanced",
            ):
                yield Label("Endpoint", classes="setup-field-label")
                yield Input(
                    value=draft.endpoint,
                    id="setup-voice-endpoint",
                    placeholder="http://127.0.0.1:8765/v1/audio/speech",
                )
                yield Label("Authentication", classes="setup-field-label")
                with SetupRadioSet(
                    id="setup-voice-auth", classes="setup-voice-segmented"
                ):
                    yield SetupRadioButton(
                        "None",
                        id="setup-voice-auth-none",
                        value=True,
                    )
                    yield SetupRadioButton("API key", id="setup-voice-auth-key")
                yield Label("Model", classes="setup-field-label")
                yield Input(value=draft.model_id, id="setup-voice-model")
                yield Label("Voice", classes="setup-field-label")
                yield Input(value=draft.voice_id, id="setup-voice-voice")
                with Horizontal(classes="setup-voice-output-row"):
                    with Vertical():
                        yield Label("Format", classes="setup-field-label")
                        yield Input(
                            value=draft.response_format, id="setup-voice-format"
                        )
                    with Vertical():
                        yield Label("Speed", classes="setup-field-label")
                        yield Input(value=str(draft.speed), id="setup-voice-speed")

    def _selected_authentication(self) -> str:
        pressed = self.query_one("#setup-voice-auth", RadioSet).pressed_button
        return (
            "api_key"
            if pressed is not None and pressed.id == "setup-voice-auth-key"
            else "none"
        )

    def _draft_from_controls(self) -> voice_state.VoiceSetupDraft:
        try:
            speed = float(self.query_one("#setup-voice-speed", Input).value)
        except ValueError as error:
            raise ValueError("Speed must be a number between 0.25 and 4.0.") from error
        return voice_state.VoiceSetupDraft(
            endpoint=self.query_one("#setup-voice-endpoint", Input).value,
            authentication_mode=self._selected_authentication(),
            model_id=self.query_one("#setup-voice-model", Input).value,
            voice_id=self.query_one("#setup-voice-voice", Input).value,
            response_format=self.query_one("#setup-voice-format", Input)
            .value.strip()
            .lower(),
            speed=speed,
            sample_text=self.query_one("#setup-voice-sample", Input).value,
            use_as_default=self.query_one("#setup-voice-default", Checkbox).value,
        )

    def _apply_draft_to_controls(self, draft: voice_state.VoiceSetupDraft) -> None:
        self.query_one("#setup-voice-endpoint", Input).value = draft.endpoint
        self.query_one("#setup-voice-model", Input).value = draft.model_id
        self.query_one("#setup-voice-voice", Input).value = draft.voice_id
        self.query_one("#setup-voice-format", Input).value = draft.response_format
        self.query_one("#setup-voice-speed", Input).value = str(draft.speed)
        self.query_one("#setup-voice-sample", Input).value = draft.sample_text
        self.query_one("#setup-voice-default", Checkbox).value = draft.use_as_default
        auth_id = (
            "setup-voice-auth-none"
            if draft.authentication_mode == "none"
            else "setup-voice-auth-key"
        )
        restore_selection = getattr(self.wizard, "_restore_radio_selection", None)
        if callable(restore_selection):
            restore_selection(
                self.query_one("#setup-voice-auth", RadioSet),
                lambda button: button.id == auth_id,
            )
        else:
            self._set_radio(auth_id)
        self._refresh_sample_state()

    def _set_radio(self, button_id: str) -> None:
        radio_set = self.query_one(f"#{button_id}", RadioButton).parent
        if not isinstance(radio_set, RadioSet):
            return
        buttons = list(radio_set.query(RadioButton))
        selected = next((button for button in buttons if button.id == button_id), None)
        with radio_set.prevent(RadioButton.Changed):
            for button in buttons:
                button.value = button is selected
        radio_set._pressed_button = selected
        radio_set._selected = buttons.index(selected) if selected is not None else None

    def _select_preset_button(self, button_id: str) -> None:
        """Press one service radio exactly as a user would (tests + resume)."""
        self.query_one(f"#{button_id}", RadioButton).value = True

    @on(RadioSet.Changed, "#setup-voice-preset")
    def _on_preset(self, event: RadioSet.Changed) -> None:
        if event.pressed is None:
            return
        preset = {
            "setup-voice-preset-pocket": voice_state.VOICE_PRESET_POCKET_TTS,
            "setup-voice-preset-official": voice_state.VOICE_PRESET_OFFICIAL_OPENAI,
            "setup-voice-preset-custom": voice_state.VOICE_PRESET_CUSTOM,
            "setup-voice-preset-omnivoice": voice_state.VOICE_PRESET_OMNIVOICE,
        }.get(event.pressed.id)
        if preset is None or preset == self._preset:
            return
        try:
            current = self._draft_from_controls()
        except (TypeError, ValueError):
            self.show_step_error(
                "Enter a valid speed before changing the service preset."
            )
            return
        # Fix round 1, Important 1: capture the outgoing Custom draft BEFORE
        # switching, whether the destination is OmniVoice or anything else --
        # the old code captured this only on the non-OmniVoice path, so a
        # Custom edit made just before entering OmniVoice was silently lost
        # (never reached _custom_draft) and a later return to Custom replayed
        # the stale cached draft over it.
        if self._preset == voice_state.VOICE_PRESET_CUSTOM:
            self._custom_draft = current
        leaving_omnivoice = self._preset == voice_state.VOICE_PRESET_OMNIVOICE
        self._preset = preset
        if preset == voice_state.VOICE_PRESET_OMNIVOICE:
            self._set_omnivoice_mode(True)
            return
        if leaving_omnivoice:
            self._set_omnivoice_mode(False)
        base = (
            self._custom_draft
            if preset == voice_state.VOICE_PRESET_CUSTOM
            and self._custom_draft is not None
            else current
        )
        self._apply_draft_to_controls(voice_state.apply_voice_preset(base, preset))

    def _omnivoice_settings(self) -> Mapping[str, object]:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        raw = (
            app_config.get("COMPREHENSIVE_CONFIG_RAW")
            if isinstance(app_config, Mapping)
            else None
        )
        source = raw if isinstance(raw, Mapping) else app_config
        section = (
            source.get("OmniVoiceSettings") if isinstance(source, Mapping) else None
        )
        return section if isinstance(section, Mapping) else {}

    def _omnivoice_seed_value(self) -> int:
        if self._omnivoice_seed is None:
            self._omnivoice_seed = voice_state.choose_omnivoice_seed(
                self._omnivoice_settings().get("seed")
            )
        return self._omnivoice_seed

    def _set_omnivoice_mode(self, enabled: bool) -> None:
        self.query_one("#setup-voice-omnivoice-panel").display = enabled
        self.query_one("#setup-voice-advanced", Collapsible).display = not enabled
        if enabled:
            self.query_one("#setup-voice-add-key", Button).display = False
            # Fix round 1, Minor 2: the PREVIOUS visit's state/status text
            # would otherwise stay live (e.g. a stale "ready") until the
            # fresh read lands, briefly letting Test and Hear enable on
            # data that no longer reflects this visit. Reset to "checking"
            # up front -- unless an install is actively running, in which
            # case that state is still correct and must not flicker.
            if not self._omnivoice_installing:
                self._omnivoice_state = None
                self.query_one("#setup-voice-omnivoice-status", Static).update(
                    voice_state.OMNIVOICE_CHECKING_COPY
                )
                self._refresh_sample_state()
            self._request_omnivoice_state()
        else:
            self._omnivoice_state_generation += 1
        self._invalidate_sample_evidence()

    def _request_omnivoice_state(self) -> None:
        self._omnivoice_state_generation += 1
        self._load_omnivoice_state(self._omnivoice_state_generation)

    @wizard_work(
        thread=True,
        group="setup-voice-omnivoice-state",
        exclusive=True,
    )
    def _load_omnivoice_state(self, generation: int) -> None:
        model_root = self._omnivoice_settings().get("model_root")
        try:
            state = omnivoice_setup_state(
                model_root if isinstance(model_root, str) else None
            )
        except ImportError:
            # The engine's own imports are broken; a model download can't help.
            logger.opt(exception=True).warning("OmniVoice setup state read failed")
            state = "engine_missing"
        except Exception:
            logger.opt(exception=True).warning("OmniVoice setup state read failed")
            state = "model_missing"
        self.app.call_from_thread(self._apply_checked_omnivoice_state, generation, state)

    def _apply_checked_omnivoice_state(self, generation: int, state: str) -> None:
        if (
            generation != self._omnivoice_state_generation
            or self._preset != voice_state.VOICE_PRESET_OMNIVOICE
        ):
            return
        self._apply_omnivoice_state(state)

    def _apply_omnivoice_state(self, state: str, message: str | None = None) -> None:
        self._omnivoice_state = state
        copy = {
            "engine_missing": voice_state.OMNIVOICE_ENGINE_MISSING_COPY,
            "model_missing": voice_state.OMNIVOICE_MODEL_MISSING_COPY,
            "path_invalid": voice_state.OMNIVOICE_PATH_INVALID_COPY,
            "ready": voice_state.OMNIVOICE_READY_COPY,
        }[state]
        try:
            # Fix round 1, Minor 3: a re-read while an install is running
            # (e.g. the step was hidden and re-shown mid-download) must not
            # stomp the status line with e.g. OMNIVOICE_MODEL_MISSING_COPY --
            # the progress bar already shows the install, and this would
            # otherwise be an invented, misleading "not started" message
            # while Install itself correctly stays disabled below. A
            # genuine failure message (`message`) always reaches the user:
            # by the time one is produced, the caller has already reset
            # `_omnivoice_installing` to False.
            if not self._omnivoice_installing:
                self.query_one("#setup-voice-omnivoice-status", Static).update(
                    message or copy
                )
            install = self.query_one("#setup-voice-omnivoice-install", Button)
            install.display = state in {"engine_missing", "model_missing"}
            install.disabled = state != "model_missing" or self._omnivoice_installing
        except NoMatches:
            return
        # Test and Hear is shared with the other services; only OmniVoice's
        # own state may drive it.
        if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
            self._refresh_sample_state()

    @on(Button.Pressed, "#setup-voice-omnivoice-install")
    def _on_omnivoice_install(self, event: Button.Pressed) -> None:
        event.stop()
        if self._omnivoice_installing or self._omnivoice_state != "model_missing":
            return
        self._omnivoice_installing = True
        self.query_one("#setup-voice-omnivoice-install", Button).disabled = True
        self._refresh_sample_state()
        self._omnivoice_preflight()

    @wizard_work(
        thread=True,
        group="setup-voice-omnivoice-install",
        exclusive=True,
    )
    def _omnivoice_preflight(self) -> None:
        try:
            report = asyncio.run(run_omnivoice_preflight())  # policy-exception: worker-thread loop
        except Exception as exc:
            logger.opt(exception=True).error("OmniVoice model preflight failed")
            self.app.call_from_thread(
                self._finish_omnivoice_install,
                install_failure_message(exc, model_label="OmniVoice voice model"),
            )
            return
        self.app.call_from_thread(self._show_omnivoice_consent, report)

    def _show_omnivoice_consent(self, report: Any) -> None:
        self._omnivoice_report = report
        self.app.push_screen(
            ModelInstallModal(
                report,
                model_label="OmniVoice voice model",
                container_id="setup-voice-omnivoice-install-modal",
                confirm_id="setup-voice-omnivoice-install-confirm",
                cancel_id="setup-voice-omnivoice-install-cancel",
            ),
            self._confirm_omnivoice_install,
        )

    def _confirm_omnivoice_install(self, confirmed: bool) -> None:
        if not confirmed:
            self._finish_omnivoice_install(None)
            return
        self._omnivoice_provision()

    @wizard_work(
        thread=True,
        group="setup-voice-omnivoice-install",
        exclusive=True,
    )
    def _omnivoice_provision(self) -> None:
        report = self._omnivoice_report
        try:
            asyncio.run(  # policy-exception: worker-thread loop
                run_omnivoice_provision(
                    report, progress=make_progress_callback(self.post_message)
                )
            )
        except Exception as exc:
            logger.opt(exception=True).error("OmniVoice model installation failed")
            self.app.call_from_thread(
                self._finish_omnivoice_install,
                install_failure_message(exc, model_label="OmniVoice voice model"),
            )
            return
        self.app.call_from_thread(self._finish_omnivoice_install, None)

    def _finish_omnivoice_install(self, error: str | None) -> None:
        self._omnivoice_installing = False
        self._omnivoice_report = None
        try:
            self.query_one(
                "#setup-voice-omnivoice-progress", ModelInstallProgress
            ).display = False
        except NoMatches:
            pass
        if error is not None:
            self._omnivoice_state_generation += 1  # drop any pre-install check
            self._apply_omnivoice_state("model_missing", message=error)
            return
        self._request_omnivoice_state()

    @on(InstallProgressed)
    def _omnivoice_install_progressed(self, event: InstallProgressed) -> None:
        event.stop()
        try:
            progress = self.query_one(
                "#setup-voice-omnivoice-progress", ModelInstallProgress
            )
        except NoMatches:
            return
        progress.display = True
        progress.update_progress(event.progress)

    @on(Input.Changed, "#setup-voice-sample")
    def _on_sample_changed(self) -> None:
        self._invalidate_sample_evidence()
        self._refresh_sample_state()

    @on(Input.Changed)
    def _on_voice_input_changed(self, event: Input.Changed) -> None:
        if (
            event.input.id
            and event.input.id.startswith("setup-voice-")
            and event.input.id != "setup-voice-sample"
        ):
            self._invalidate_sample_evidence()

    @on(RadioSet.Changed, "#setup-voice-auth")
    def _on_authentication_changed(self) -> None:
        self._invalidate_sample_evidence()

    def _invalidate_sample_evidence(self) -> None:
        self._test_generation += 1
        self._test_in_progress_generation = None
        self._verified_draft = None
        try:
            self.workers.cancel_group(self, "setup-voice-sample")
        except Exception:
            pass
        try:
            self.query_one("#setup-voice-status", Static).update(
                "Not tested yet — that's fine. You can save now and test later."
            )
        except Exception:
            pass
        self._refresh_sample_state()

    @staticmethod
    def _sample_identity(draft: voice_state.VoiceSetupDraft) -> tuple[object, ...]:
        """Return only fields that affect the sample request."""

        return (
            draft.endpoint,
            draft.authentication_mode,
            draft.model_id,
            draft.voice_id,
            draft.response_format,
            draft.speed,
            draft.sample_text,
        )

    def _refresh_sample_state(self) -> None:
        try:
            sample = self.query_one("#setup-voice-sample", Input).value
            trimmed_count = len(sample.strip())
            self.query_one("#setup-voice-sample-count", Static).update(
                f"{trimmed_count} / 500"
            )
            if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
                self.query_one("#setup-voice-add-key", Button).display = False
                self.query_one("#setup-voice-test", Button).disabled = (
                    self._test_in_progress_generation is not None
                    or self._omnivoice_installing
                    or self._omnivoice_state != "ready"
                    or not 1 <= trimmed_count <= 500
                )
                return
            try:
                draft = self._draft_from_controls()
                valid = voice_state.validate_voice_setup_draft(
                    draft
                ).configuration_valid
            except (TypeError, ValueError):
                valid = False
                draft = None
            missing_key = (
                draft is not None
                and draft.authentication_mode == "api_key"
                and self._existing_openai_credential() is None
            )
            self.query_one("#setup-voice-add-key", Button).display = missing_key
            self.query_one("#setup-voice-test", Button).disabled = (
                self._test_in_progress_generation is not None
                or not valid
                or missing_key
            )
            status = self.query_one("#setup-voice-status", Static)
            status_text = str(status.renderable)
            if missing_key and self._test_in_progress_generation is None:
                self._verified_draft = None
                status.update(
                    "API key required. Add an API key in Settings to test or save."
                )
            elif (
                not missing_key
                and status_text.startswith("API key required")
                and self._test_in_progress_generation is None
            ):
                status.update(
                    "Not tested yet — that's fine. You can save now and test later."
                )
        except Exception:
            return

    def on_show(self) -> None:
        super().on_show()
        self._refresh_sample_state()
        if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
            self._request_omnivoice_state()

    def _cancel_active_sample(self) -> None:
        if self._test_in_progress_generation is None:
            self._refresh_sample_state()
            return
        self._test_generation += 1
        self._test_in_progress_generation = None
        self._verified_draft = None
        try:
            self.workers.cancel_group(self, "setup-voice-sample")
        except Exception:
            pass
        try:
            self.query_one("#setup-voice-status", Static).update(
                "Not tested yet — the sample was cancelled. Retry when ready."
            )
        except Exception:
            pass
        self._refresh_sample_state()

    def on_hide(self) -> None:
        super().on_hide()
        self._cancel_active_sample()

    def on_unmount(self) -> None:
        self._cancel_active_sample()
        if self._sample_audio_path is not None:
            self._sample_audio_path.unlink(missing_ok=True)
            self._sample_audio_path = None

    @on(Button.Pressed, "#setup-voice-test")
    def _on_test_and_hear(self) -> None:
        if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
            self._start_omnivoice_sample()
            return
        try:
            draft = self._draft_from_controls()
        except (TypeError, ValueError):
            return
        if not voice_state.validate_voice_setup_draft(draft).configuration_valid:
            return
        self._test_generation += 1
        generation = self._test_generation
        self._test_in_progress_generation = generation
        self.query_one("#setup-voice-status", Static).update("Testing voice…")
        self._refresh_sample_state()
        run_wizard_worker(
            self,
            self._run_voice_sample(generation, draft),
            exclusive=True,
            group="setup-voice-sample",
        )

    @on(Button.Pressed, "#setup-voice-add-key")
    def _on_add_api_key(self) -> None:
        callback = getattr(self.wizard, "open_voice_api_key_settings", None)
        if not callable(callback):
            self.query_one("#setup-voice-status", Static).update(
                "Open Settings, then Speech & TTS, to add the OpenAI API key."
            )
            return
        try:
            route = callback(self)
        except Exception:
            self.query_one("#setup-voice-status", Static).update(
                "Could not open Settings. Use Speech & TTS to add the API key."
            )
            return
        if not asyncio.iscoroutine(route):
            self.query_one("#setup-voice-status", Static).update(
                "Could not open Settings. Use Speech & TTS to add the API key."
            )
            return
        run_wizard_worker(
            self,
            route,
            exclusive=True,
            group="setup-voice-api-key-settings",
        )

    def _existing_openai_credential(self) -> str | None:
        if self._selected_authentication() != "api_key":
            return None
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        if isinstance(app_config, Mapping):
            persisted = app_config.get("COMPREHENSIVE_CONFIG_RAW")
            source = persisted if isinstance(persisted, Mapping) else app_config
            locations = (
                ("api_settings", "openai", "api_key"),
                ("openai_api", "api_key"),
                ("API", "openai_api_key"),
            )
            for location in locations:
                current: object = source
                for part in location:
                    if not isinstance(current, Mapping):
                        current = None
                        break
                    current = current.get(part)
                if isinstance(current, str) and current:
                    return current
            api_settings = source.get("api_settings")
            if isinstance(api_settings, Mapping):
                openai = api_settings.get("openai")
                if isinstance(openai, Mapping):
                    environment_name = openai.get("api_key_env_var")
                    if isinstance(environment_name, str) and environment_name:
                        environment_value = os.environ.get(environment_name)
                        if environment_value:
                            return environment_value
            projected = app_config.get("OPENAI_API_KEY")
            if isinstance(projected, str) and projected:
                return projected
        value = os.environ.get("OPENAI_API_KEY")
        return value if value else None

    async def _run_voice_sample(
        self,
        generation: int,
        draft: voice_state.VoiceSetupDraft,
    ) -> None:
        try:
            result = await voice_state.run_voice_sample(
                draft,
                credential=self._existing_openai_credential(),
            )
        except asyncio.CancelledError:
            if generation == self._test_generation:
                self.query_one("#setup-voice-status", Static).update(
                    "Not tested yet — the sample was cancelled. Retry when ready."
                )
            raise
        except Exception:
            if generation == self._test_generation:
                self.query_one("#setup-voice-status", Static).update(
                    "Not tested yet — the sample failed. Check the service, then retry."
                )
            return
        else:
            if generation != self._test_generation:
                return
            try:
                current = self._draft_from_controls()
            except (TypeError, ValueError):
                return
            if self._sample_identity(current) != self._sample_identity(draft):
                return
            self._verified_draft = draft
            try:
                played = await self._play_sample(result)
            except asyncio.CancelledError:
                if generation == self._test_generation:
                    self.query_one("#setup-voice-status", Static).update(
                        "Verified, playback failed. Retry playback/test."
                    )
                raise
            if generation != self._test_generation:
                return
            self.query_one("#setup-voice-status", Static).update(
                "Verified. The sample is ready to hear."
                if played
                else "Verified, playback failed. Retry playback/test."
            )
        finally:
            if self._test_in_progress_generation == generation:
                self._test_in_progress_generation = None
                self._refresh_sample_state()

    async def _play_sample(self, result: voice_state.VoiceSampleResult) -> bool:
        audio_player = getattr(self.app, "audio_player", None)
        if audio_player is None:
            # First run: no Speech screen has created the shared player yet
            # (speech_playback_mixin creates it lazily the same way).
            try:
                from tldw_chatbook.TTS.audio_player import AsyncAudioPlayer

                audio_player = self.app.audio_player = AsyncAudioPlayer()
            except Exception:
                logger.debug("Voice sample player unavailable (category=playback)")
                return False
        play = getattr(audio_player, "play", None)
        if not callable(play):
            return False
        suffix = "." + result.response_format
        sample_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="chatbook-voice-sample-",
                suffix=suffix,
                delete=False,
            ) as handle:
                handle.write(result.body)
                sample_path = Path(handle.name)
            prior_path = self._sample_audio_path
            played = play(sample_path)
            if asyncio.iscoroutine(played):
                played = await played
            if played is False:
                # AsyncAudioPlayer reports "no OS player found" as False.
                sample_path.unlink(missing_ok=True)
                return False
            if prior_path is not None:
                prior_path.unlink(missing_ok=True)
            self._sample_audio_path = sample_path
            return True
        except asyncio.CancelledError:
            if sample_path is not None:
                sample_path.unlink(missing_ok=True)
            raise
        except Exception:
            if sample_path is not None:
                sample_path.unlink(missing_ok=True)
            logger.debug("Voice sample playback failed (category=playback)")
            return False

    def _speed_or_default(self) -> float:
        try:
            speed = float(self.query_one("#setup-voice-speed", Input).value)
        except ValueError:
            return 1.0
        return speed if math.isfinite(speed) and 0.25 <= speed <= 4.0 else 1.0

    def _start_omnivoice_sample(self) -> None:
        if self._omnivoice_state != "ready" or self._omnivoice_installing:
            return
        text = self.query_one("#setup-voice-sample", Input).value
        self._test_generation += 1
        generation = self._test_generation
        self._test_in_progress_generation = generation
        self.query_one("#setup-voice-status", Static).update(
            voice_state.OMNIVOICE_GENERATING_COPY
        )
        self._refresh_sample_state()
        run_wizard_worker(
            self,
            self._run_omnivoice_sample(
                generation, text, self._speed_or_default(), self._omnivoice_seed_value()
            ),
            exclusive=True,
            group="setup-voice-sample",
        )

    def _update_voice_status_text(self, text: str) -> None:
        """Update the shared status line, tolerating an unmounted step.

        Textual worker contract (W002): both call sites in
        ``_run_omnivoice_sample`` resume this lookup after an ``await``, and
        the step can be hidden/unmounted while the sample synthesizes.
        """
        try:
            self.query_one("#setup-voice-status", Static).update(text)
        except NoMatches:
            pass

    async def _run_omnivoice_sample(
        self, generation: int, text: str, speed: float, seed: int
    ) -> None:
        try:
            try:
                result = await voice_state.run_omnivoice_sample(
                    text, speed=speed, seed=seed
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                if generation == self._test_generation:
                    self._test_in_progress_generation = None
                    self._update_voice_status_text(
                        voice_state.OMNIVOICE_SAMPLE_FAILED_COPY
                    )
                    self._refresh_sample_state()
                return
            if generation != self._test_generation:
                return
            played = await self._play_sample(result)
            # A newer test (or a control change) during playback owns the
            # status now; this older result must not overwrite it.
            if generation != self._test_generation:
                return
            self._test_in_progress_generation = None
            self._update_voice_status_text(
                "Verified. The sample is ready to hear."
                if played
                else "Verified, playback failed. Retry playback/test."
            )
            self._refresh_sample_state()
        finally:
            if self._test_in_progress_generation == generation:
                self._test_in_progress_generation = None

    async def commit(self) -> tuple[bool, str]:
        if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
            return await self._commit_omnivoice()
        try:
            draft = self._draft_from_controls()
        except (TypeError, ValueError) as error:
            return False, str(error) or "Review the Voice setup fields."
        validation = voice_state.validate_voice_setup_draft(draft)
        if not validation.configuration_valid:
            return False, validation.errors[
                0
            ] if validation.errors else "Review the Voice setup fields."
        if (
            draft.authentication_mode == "api_key"
            and self._existing_openai_credential() is None
        ):
            return False, "Add an API key in Settings before saving this voice."
        request_id = self._next_save_request_id
        self._next_save_request_id += 1
        self._save_request_id = request_id
        self._save_draft = draft
        self._save_use_as_default = draft.use_as_default
        self._save_future = asyncio.get_running_loop().create_future()
        self.app.post_message(
            voice_state.build_voice_setup_save_event(
                draft,
                request_id=request_id,
                reply_to=self,
            )
        )
        try:
            return await asyncio.wait_for(
                asyncio.shield(self._save_future),
                timeout=self._SAVE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return False, "Voice settings are still applying. Retry to continue."
        finally:
            self._save_request_id = None
            self._save_draft = None
            self._save_future = None

    async def _commit_omnivoice(self) -> tuple[bool, str]:
        if not self.query_one("#setup-voice-default", Checkbox).value:
            return True, ""
        if self._omnivoice_state is None:
            return False, voice_state.OMNIVOICE_CHECKING_COPY
        if self._omnivoice_state == "engine_missing":
            return False, voice_state.OMNIVOICE_ENGINE_MISSING_COPY
        if self._omnivoice_state == "path_invalid":
            return False, voice_state.OMNIVOICE_PATH_INVALID_COPY
        if self._omnivoice_state != "ready":
            return False, voice_state.OMNIVOICE_DEFAULT_WITHOUT_MODEL_COPY
        request_id = self._next_save_request_id
        self._next_save_request_id += 1
        self._save_request_id = request_id
        self._save_provider = "omnivoice"
        self._save_use_as_default = True
        self._save_future = asyncio.get_running_loop().create_future()
        self.app.post_message(
            voice_state.build_omnivoice_save_event(
                speed=self._speed_or_default(),
                seed=self._omnivoice_seed_value(),
                request_id=request_id,
                reply_to=self,
            )
        )
        try:
            return await asyncio.wait_for(
                asyncio.shield(self._save_future), timeout=self._SAVE_TIMEOUT_SECONDS
            )
        except TimeoutError:
            return False, "Voice settings are still applying. Retry to continue."
        finally:
            self._save_request_id = None
            self._save_future = None
            self._save_provider = "openai"

    def _receive_save_result(self, result: object) -> None:
        from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
            STTSSettingsSaveResult,
        )

        provider = self._save_provider
        future = self._save_future
        if (
            type(result) is not STTSSettingsSaveResult
            or result.request_id != self._save_request_id
            or future is None
            or future.done()
        ):
            return
        if not result.persisted:
            future.set_result((False, "Saving the Voice settings failed. Retry."))
            return
        provider_status = result.provider_statuses.get(provider)
        if provider_status == "pending":
            return
        runtime_ready = (
            provider_status in {"applied", "unchanged"}
            and provider in result.provider_configuration_revisions
            and provider in result.provider_runtime_revisions
        )
        if not runtime_ready:
            future.set_result(
                (False, "The Voice settings were saved, but are not active. Retry.")
            )
            return
        if not self._save_use_as_default:
            future.set_result((True, ""))
            return
        if result.defaults_activated is True:
            future.set_result((True, ""))
            return
        future.set_result(
            (
                False,
                "The Voice settings were saved, but the default was not activated. Retry.",
            )
        )

    def receive_stts_settings_save_result(self, result: object) -> None:
        self._receive_save_result(result)

    def receive_stts_settings_runtime_result(self, result: object) -> None:
        self._receive_save_result(result)

    def get_step_data(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {
            "preset": self._preset,
            "endpoint": self.query_one("#setup-voice-endpoint", Input).value,
            "authentication_mode": self._selected_authentication(),
            "model_id": self.query_one("#setup-voice-model", Input).value,
            "voice_id": self.query_one("#setup-voice-voice", Input).value,
            "response_format": self.query_one("#setup-voice-format", Input)
            .value.strip()
            .lower(),
            "sample_text": self.query_one("#setup-voice-sample", Input).value,
            "use_as_default": self.query_one("#setup-voice-default", Checkbox).value,
        }
        try:
            speed = float(self.query_one("#setup-voice-speed", Input).value)
            if not math.isfinite(speed):
                raise ValueError
        except ValueError:
            return values
        values["speed"] = speed
        return values
