"""The first-run wizard's Voice step.

Moved out of ``FirstRunSetupWizard.py`` (TASK-33921) so that module stays
under its size budget. The class moved whole, so its ``@on`` handlers and
workers stay registered on it; ``FirstRunSetupWizard`` re-exports the name.
Patch this module, not the wizard, to replace what the step calls; the
OmniVoice half lives in ``first_run_voice_omnivoice.py`` (TASK-34100.8).

TASK-34100.8 (SF3 "untouched means unwritten"): the step starts from the
saved voice (or "No voice for now"), says on a line under the Service radio
whether the chosen service will work, explains a failed test, and posts a
save only when something changed or the user tested this run.
"""

from __future__ import annotations

import asyncio
import math
from typing import (
    Any,
    Dict,
    Mapping,
)

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
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

from tldw_chatbook.config import resolve_provider_api_key
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards import first_run_voice_prefill as prefill
from tldw_chatbook.UI.Wizards import first_run_voice_status as voice_status
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupCheckbox,
    SetupRadioButton,
    SetupRadioSet,
)
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker, wizard_work
from tldw_chatbook.UI.Wizards.first_run_voice_credentials import find_openai_credential
from tldw_chatbook.UI.Wizards.first_run_voice_omnivoice import OmniVoiceStepBase
from tldw_chatbook.UI.Wizards.first_run_voice_pickers import (
    VoiceOptionPicker,
    compose_voice_advanced,
)
from tldw_chatbook.UI.Wizards.first_run_voice_status import probe_endpoint_reachable
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

#: Service radio: (button id, label, preset). "No voice for now" leads.
_SERVICES = (
    ("setup-voice-preset-none", "No voice for now", voice_state.VOICE_PRESET_NONE),
    ("setup-voice-preset-pocket", "PocketTTS", voice_state.VOICE_PRESET_POCKET_TTS),
    ("setup-voice-preset-official", "OpenAI", voice_state.VOICE_PRESET_OFFICIAL_OPENAI),
    ("setup-voice-preset-custom", "Custom", voice_state.VOICE_PRESET_CUSTOM),
    ("setup-voice-preset-omnivoice", "OmniVoice", voice_state.VOICE_PRESET_OMNIVOICE),
)
_PRESET_BY_BUTTON = {button_id: preset for button_id, _label, preset in _SERVICES}
_BUTTON_BY_PRESET = {preset: button_id for button_id, _label, preset in _SERVICES}
#: Probed with one sub-second connect: the two services that are local servers.
_PROBED_PRESETS = {voice_state.VOICE_PRESET_POCKET_TTS, voice_state.VOICE_PRESET_CUSTOM}


class VoiceSetupStep(OmniVoiceStepBase):
    """Compact OpenAI-compatible TTS setup shared by Quick and Full tracks."""

    #: TASK-34100.8 (voice-speech-05): Enter in Sample text runs the test
    #: instead of advancing, so this step's hint line says so.
    KEY_HINTS = (
        "Enter in Sample text tests it · Ctrl+N next · Ctrl+B back · Esc exit setup"
    )

    def __init__(self, wizard=None, config=None, **kwargs: Any) -> None:
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._custom_draft: voice_state.VoiceSetupDraft | None = None
        self._verified_draft: voice_state.VoiceSetupDraft | None = None
        self._save_draft: voice_state.VoiceSetupDraft | None = None
        self._saved: prefill.SavedVoice | None = None
        self._baseline: voice_state.VoiceSetupDraft | None = None
        self._staged_key: wizard_state.ProviderCredentialDraft | None = None
        self._probe_generation = 0
        self._reachable: bool | None = None
        self._refocus_test = False
        self._probe_timer: Any = None
        self._seen_inputs: dict[str, str] = {}
        # Review round 1 (F1): the box is locked on while the OpenAI slot
        # reads replies; _default_choice is the user's own value under it.
        self._default_locked = False
        self._default_choice = False

    def _raw_table(self) -> Mapping[str, object]:
        app_instance = getattr(self.wizard, "app_instance", None)
        return prefill.raw_app_tts(getattr(app_instance, "app_config", {}) or {})

    def _initial_draft(self) -> voice_state.VoiceSetupDraft:
        """The controls' starting values: the saved voice, else PocketTTS's."""
        if self._saved is not None:
            return self._saved.draft
        return prefill.initial_voice_draft()

    def _refresh_saved(self) -> None:
        """Read what is saved now: at compose, and after this step's save (a
        Back then "No voice for now" must name the voice just saved)."""
        self._saved = prefill.saved_voice_from_config(self._raw_table())
        saved = self._saved
        self._baseline = (
            saved.draft if saved is not None and saved.slot_preset else None
        )
        self._tested_this_run = False

    def compose_step(self) -> ComposeResult:
        # TASK-21148 (UAT V-1/V-2): outcome first -- purpose line, service
        # choice, try-it controls; the plumbing lives under Advanced.
        # TASK-34100.8: the saved voice (raw [app_tts]) is preselected, else
        # "No voice for now", never a server that probably isn't running.
        self._refresh_saved()
        saved = self._saved
        self._preset = saved.preset if saved else voice_state.VOICE_PRESET_NONE
        draft = self._initial_draft()
        self._default_choice = draft.use_as_default
        with Vertical(classes="setup-voice"):
            yield Static("Set up a voice", classes="setup-title")
            yield Static(
                "Hear replies read aloud — optional. Pick a service to try it, "
                'or keep "No voice for now" — nothing is saved unless you '
                "choose one.",
                classes="setup-subtitle",
            )
            yield Label("Service", classes="setup-field-label")
            with SetupRadioSet(
                id="setup-voice-preset", classes="setup-voice-segmented"
            ):
                for button_id, label, preset in _SERVICES:
                    yield SetupRadioButton(
                        label, id=button_id, value=preset == self._preset
                    )
            yield Static(
                self._service_line(),
                id="setup-voice-service-status",
                classes="setup-field-help",
            )
            with Vertical(id="setup-voice-body") as body:
                body.display = self._preset != voice_state.VOICE_PRESET_NONE
                yield from self._compose_omnivoice_panel()
                yield Label("Sample text", classes="setup-field-label")
                yield Input(
                    value=draft.sample_text,
                    id="setup-voice-sample",
                    max_length=500,
                    select_on_focus=False,
                )
                yield Static(
                    f"{len(draft.sample_text)} / 500",
                    id="setup-voice-sample-count",
                    classes="setup-field-help",
                )
                yield Button("Test and Hear", id="setup-voice-test")
                yield Static(
                    prefill.current_voice_copy(saved)
                    if saved is not None
                    and saved.preset != voice_state.VOICE_PRESET_NONE
                    else voice_status.DEFAULT_STATUS_COPY,
                    id="setup-voice-status",
                    classes="setup-subtitle",
                )
                with Vertical(id="setup-voice-key-row") as key_row:
                    key_row.display = False
                    yield Label("OpenAI API key", classes="setup-field-label")
                    yield Input(
                        id="setup-voice-api-key",
                        password=True,
                        placeholder="Paste your OpenAI API key (sk-…)",
                    )
                    yield Button(
                        "Leave setup and add key in Settings…", id="setup-voice-add-key"
                    )
                yield SetupCheckbox(
                    "Use this voice when Chatbook reads replies aloud",
                    id="setup-voice-default",
                    value=draft.use_as_default,
                )
                yield Static(
                    voice_status.DEFAULT_HELP_COPY,
                    id="setup-voice-default-help",
                    classes="setup-field-help",
                )
                with Collapsible(
                    title="Advanced — endpoint, model & output",
                    collapsed=True,
                    id="setup-voice-advanced",
                ) as advanced:
                    advanced.display = (
                        self._preset != voice_state.VOICE_PRESET_OMNIVOICE
                    )
                    yield from compose_voice_advanced(draft, self._preset)

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
        self._sync_pickers()
        self._refresh_sample_state()

    def _sync_pickers(self) -> None:
        for picker in self.query(VoiceOptionPicker):
            options = (
                prefill.voices_for(self._preset)
                if picker.id == "setup-voice-voice-picker"
                else prefill.formats_for(self._preset)
            )
            picker.set_options(options)

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
        preset = _PRESET_BY_BUTTON.get(event.pressed.id or "")
        if preset is None or preset == self._preset:
            return
        self.clear_step_error()
        try:
            current = self._draft_from_controls()
        except (TypeError, ValueError):
            self.show_step_error(
                "Enter a valid speed before changing the service preset."
            )
            return
        # Fix round 1, Important 1: capture the outgoing Custom draft BEFORE
        # switching, whatever the destination.
        if self._preset == voice_state.VOICE_PRESET_CUSTOM:
            self._custom_draft = current
        leaving_omnivoice = self._preset == voice_state.VOICE_PRESET_OMNIVOICE
        self._preset = preset
        self.query_one("#setup-voice-body").display = (
            preset != voice_state.VOICE_PRESET_NONE
        )
        if preset == voice_state.VOICE_PRESET_OMNIVOICE:
            self._set_omnivoice_mode(True)
            self._refresh_service_status()
            return
        if leaving_omnivoice:
            self._set_omnivoice_mode(False)
        if preset == voice_state.VOICE_PRESET_NONE:
            self._invalidate_sample_evidence()
            self._refresh_service_status()
            return
        base = (
            self._custom_draft
            if preset == voice_state.VOICE_PRESET_CUSTOM
            and self._custom_draft is not None
            else current
        )
        self._apply_draft_to_controls(voice_state.apply_voice_preset(base, preset))
        self._start_probe()

    def _maybe_switch_to_custom(self) -> None:
        """An Advanced edit away from the preset makes the service Custom.

        TASK-34100.8 (new-voice-speech-02): the radio kept saying PocketTTS
        while the endpoint pointed elsewhere, and a later service switch
        rebuilt the fields from presets, dropping the typed endpoint.
        """
        if self._preset not in {
            voice_state.VOICE_PRESET_POCKET_TTS,
            voice_state.VOICE_PRESET_OFFICIAL_OPENAI,
        }:
            return
        try:
            current = self._draft_from_controls()
        except (TypeError, ValueError):
            return
        if prefill.draft_matches_preset(current, self._preset):
            return
        self._preset = voice_state.VOICE_PRESET_CUSTOM
        self._custom_draft = current
        self._set_radio("setup-voice-preset-custom")
        self._sync_pickers()
        self._refresh_service_status()

    def on_mount(self) -> None:
        # Seed what each Input holds, so the Changed an Input posts for its
        # own initial value at mount is not taken for an edit.
        self._seen_inputs = {field.id or "": field.value for field in self.query(Input)}

    def _input_really_changed(self, event: Input.Changed) -> bool:
        """Whether an Input's value differs from the last one this step saw.

        The mount-time Changed used to wipe the re-run "Current voice" line.
        """
        input_id = event.input.id or ""
        changed = self._seen_inputs.get(input_id) != event.value
        self._seen_inputs[input_id] = event.value
        return changed

    @on(Input.Changed, "#setup-voice-sample")
    def _on_sample_changed(self, event: Input.Changed) -> None:
        if not self._input_really_changed(event):
            return
        self.clear_step_error()
        self._invalidate_sample_evidence()
        self._refresh_sample_state()

    @on(Input.Changed)
    def _on_voice_input_changed(self, event: Input.Changed) -> None:
        input_id = event.input.id or ""
        if (
            not input_id.startswith("setup-voice-")
            or input_id in {"setup-voice-sample", "setup-voice-api-key"}
            or not self._input_really_changed(event)
        ):
            return
        self.clear_step_error()
        self._invalidate_sample_evidence()
        self._maybe_switch_to_custom()
        if input_id == "setup-voice-endpoint":
            self._schedule_probe()

    @on(RadioSet.Changed, "#setup-voice-auth")
    def _on_authentication_changed(self) -> None:
        self.clear_step_error()
        self._invalidate_sample_evidence()
        self._maybe_switch_to_custom()

    @on(Checkbox.Changed, "#setup-voice-default")
    def _on_default_changed(self, event: Checkbox.Changed) -> None:
        self.clear_step_error()
        if not self._default_locked:
            self._default_choice = event.value
        self._refresh_service_status()

    @on(Input.Changed, "#setup-voice-api-key")
    def _on_api_key_changed(self, event: Input.Changed) -> None:
        """Stage a pasted key in memory only, the way Provider stages its key.

        TASK-34100.8 (voice-speech-04): nothing is written until Next, and
        then only to ``api_settings.openai.api_key`` (where Settings ▸ Speech
        & TTS writes it), so Protect can still encrypt it.
        """
        if not self._input_really_changed(event):
            return
        self.clear_step_error()
        key = resolve_provider_api_key(event.value)
        self._staged_key = (
            wizard_state.ProviderCredentialDraft("draft", key) if key else None
        )
        self._invalidate_sample_evidence()

    @on(Input.Submitted, "#setup-voice-sample")
    def _on_sample_submitted(self, event: Input.Submitted) -> None:
        """Enter in Sample text runs the test (the hint line says so)."""
        event.stop()
        self._on_test_and_hear()

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
                voice_status.DEFAULT_STATUS_COPY
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
            key_row = self.query_one("#setup-voice-key-row")
            if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
                key_row.display = False
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
            uses_key = draft is not None and draft.authentication_mode == "api_key"
            missing_key = uses_key and self._existing_openai_credential() is None
            key_row.display = missing_key or (uses_key and self._staged_key is not None)
            self.query_one("#setup-voice-test", Button).disabled = (
                self._test_in_progress_generation is not None
                or not valid
                or missing_key
            )
            status = self.query_one("#setup-voice-status", Static)
            status_text = str(status.renderable)
            if missing_key and self._test_in_progress_generation is None:
                self._verified_draft = None
                status.update(voice_status.KEY_NEEDED_COPY)
            elif (
                not missing_key
                and status_text == voice_status.KEY_NEEDED_COPY
                and self._test_in_progress_generation is None
            ):
                status.update(voice_status.DEFAULT_STATUS_COPY)
            self._refresh_service_status()
        except Exception:
            return

    # -- service status ----------------------------------------------------
    def _service_line(self) -> str:
        if self._preset == voice_state.VOICE_PRESET_NONE:
            return voice_status.no_voice_copy(self._saved)
        try:
            endpoint = self.query_one("#setup-voice-endpoint", Input).value
        except NoMatches:
            endpoint = self._initial_draft().endpoint
        key_found = None
        if self._preset == voice_state.VOICE_PRESET_OFFICIAL_OPENAI:
            key_found = self._find_openai_credential() is not None
        return voice_status.service_status_copy(
            self._preset,
            endpoint=endpoint,
            reachable=self._reachable,
            key_found=key_found,
        )

    def _refresh_service_status(self) -> None:
        """The line under the radio, and the default box that follows it."""
        try:
            self.query_one("#setup-voice-service-status", Static).update(
                self._service_line()
            )
        except NoMatches:
            return
        self._sync_default_box()

    def _sync_default_box(self) -> None:
        """Lock "Use this voice…" on while the OpenAI slot reads replies (F1)."""
        try:
            box = self.query_one("#setup-voice-default", Checkbox)
            draft = self._draft_from_controls()
        except (NoMatches, TypeError, ValueError):
            return
        locked = prefill.default_box_locked(self._preset, self._raw_table())
        value = True if locked else self._default_choice
        if locked or self._default_locked:
            with box.prevent(Checkbox.Changed):
                box.value = value
        self._default_locked = box.disabled = locked
        saved = self._saved
        replaced = (
            prefill.voice_label(saved)
            if saved is not None and saved.slot_preset and not saved.other_provider
            else ""
        )
        if replaced == prefill.voice_label(prefill.SavedVoice(self._preset, draft)):
            replaced = ""
        self.query_one("#setup-voice-default-help", Static).update(
            voice_status.default_help_copy(
                self._preset,
                locked=locked,
                ticked=box.value,
                reply_voice=saved.other_provider if saved is not None else "",
                replaces=replaced,
            )
        )

    def _schedule_probe(self) -> None:
        """Debounce endpoint typing: probe once the user pauses."""
        if self._probe_timer is not None:
            self._probe_timer.stop()
        self._probe_timer = self.set_timer(0.4, self._start_probe)

    def _start_probe(self) -> None:
        self._probe_timer = None
        self._probe_generation += 1
        self._reachable = None
        if self._preset not in _PROBED_PRESETS or not self.is_attached:
            self._refresh_service_status()
            return
        try:
            url = voice_state.validate_voice_setup_draft(
                self._draft_from_controls(), require_sample=False
            ).normalized_endpoint
        except (TypeError, ValueError):
            url = None
        self._refresh_service_status()
        if url:
            self._probe_service(self._probe_generation, url)

    @wizard_work(thread=True, group="setup-voice-probe", exclusive=True)
    def _probe_service(self, generation: int, url: str) -> None:
        """One sub-second TCP connect, off the event loop (voice-speech-03)."""
        reachable = probe_endpoint_reachable(url)
        self.app.call_from_thread(self._apply_probe, generation, reachable)

    def _apply_probe(self, generation: int, reachable: bool) -> None:
        if generation != self._probe_generation:
            return
        self._reachable = reachable
        self._refresh_service_status()

    def on_show(self) -> None:
        super().on_show()
        self._refresh_sample_state()
        if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
            self._request_omnivoice_state()
        self._start_probe()

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
                voice_status.CANCELLED_COPY
            )
        except Exception:
            pass
        self._refresh_sample_state()

    def on_hide(self) -> None:
        super().on_hide()
        self._cancel_active_sample()

    def on_unmount(self) -> None:
        self._cancel_active_sample()
        self._staged_key = None
        if self._sample_audio_path is not None:
            self._sample_audio_path.unlink(missing_ok=True)
            self._sample_audio_path = None

    @on(Button.Pressed, "#setup-voice-test")
    def _on_test_and_hear(self) -> None:
        test = self.query_one("#setup-voice-test", Button)
        self._refocus_test = self.app.focused is test
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
        self.query_one("#setup-voice-status", Static).update(voice_status.TESTING_COPY)
        self._refresh_sample_state()
        run_wizard_worker(
            self,
            self._run_voice_sample(generation, draft),
            exclusive=True,
            group="setup-voice-sample",
        )

    @on(Button.Pressed, "#setup-voice-add-key")
    def _on_add_api_key(self) -> None:
        """Leaving for Settings ends setup, so ask first (voice-speech-04)."""
        self.app.push_screen(
            ConfirmationDialog(
                title=voice_status.LEAVE_TITLE,
                message=voice_status.LEAVE_MESSAGE,
                confirm_label="Leave setup",
                cancel_label="Stay",
            ),
            self._leave_for_settings,
        )

    def _leave_for_settings(self, confirmed: bool | None) -> None:
        if not confirmed:
            return
        callback = getattr(self.wizard, "open_voice_api_key_settings", None)
        status = self.query_one("#setup-voice-status", Static)
        if not callable(callback):
            status.update(
                "Open Settings, then Speech & TTS, to add the OpenAI API key."
            )
            return
        try:
            route = callback(self)
        except Exception:
            route = None
        if not asyncio.iscoroutine(route):
            status.update(
                "Could not open Settings. Use Speech & TTS to add the API key."
            )
            return
        run_wizard_worker(
            self,
            route,
            exclusive=True,
            group="setup-voice-api-key-settings",
        )

    def _find_openai_credential(self) -> tuple[str, bool] | None:
        """The OpenAI key a test or save would use, and whether Next writes it."""
        return find_openai_credential(
            getattr(self.wizard.app_instance, "app_config", {}) or {},
            staged_key=self._staged_key,
            staged_provider_draft=getattr(self.wizard, "staged_provider_draft", None),
            provider_setup_committed=bool(
                getattr(self.wizard, "provider_setup_committed", False)
            ),
        )

    def _existing_openai_credential(self) -> str | None:
        if self._selected_authentication() != "api_key":
            return None
        found = self._find_openai_credential()
        return found[0] if found else None

    def _mark_sample_verified(self) -> None:
        """A successful test ticks "Use as default" and counts as acting."""
        self._tested_this_run = True
        try:
            self.query_one("#setup-voice-default", Checkbox).value = True
        except NoMatches:
            pass

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
                    voice_status.CANCELLED_COPY
                )
            raise
        except Exception as error:
            if generation == self._test_generation:
                self.query_one("#setup-voice-status", Static).update(
                    voice_status.voice_test_failure_copy(
                        error, preset=self._preset, endpoint=draft.endpoint
                    )
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
            self._mark_sample_verified()
            try:
                played = await self._play_sample(result)
            except asyncio.CancelledError:
                if generation == self._test_generation:
                    self.query_one("#setup-voice-status", Static).update(
                        voice_status.PLAYBACK_FAILED_COPY
                    )
                raise
            if generation != self._test_generation:
                return
            self.query_one("#setup-voice-status", Static).update(
                voice_status.PLAYED_COPY
                if played
                else voice_status.PLAYBACK_FAILED_COPY
            )
        finally:
            if self._test_in_progress_generation == generation:
                self._test_in_progress_generation = None
                self._refresh_sample_state()
                self._restore_test_focus()

    def _restore_test_focus(self) -> None:
        """Disabling Test and Hear dropped focus to the top (voice-speech-05)."""
        if not self._refocus_test or not self.display:
            return
        self._refocus_test = False
        try:
            test = self.query_one("#setup-voice-test", Button)
        except NoMatches:
            return
        if not test.disabled:
            test.focus()

    def _speed_or_default(self) -> float:
        try:
            speed = float(self.query_one("#setup-voice-speed", Input).value)
        except ValueError:
            return 1.0
        return speed if math.isfinite(speed) and 0.25 <= speed <= 4.0 else 1.0

    def _omnivoice_untouched(self) -> bool:
        saved = self._saved
        return (
            saved is not None
            and saved.preset == voice_state.VOICE_PRESET_OMNIVOICE
            and not self._tested_this_run
            and self.query_one("#setup-voice-default", Checkbox).value
            and self._speed_or_default() == saved.draft.speed
        )

    async def commit(self) -> tuple[bool, str]:
        # TASK-34100.8 (voice-speech-01): "No voice for now" and an untouched
        # step post nothing -- Next used to save the PocketTTS draft always.
        if self._preset == voice_state.VOICE_PRESET_NONE:
            return True, ""
        if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
            if self._omnivoice_untouched():
                return True, ""
            outcome = await self._commit_omnivoice()
            if outcome[0]:
                self._refresh_saved()
            return outcome
        self._sync_default_box()
        try:
            draft = self._draft_from_controls()
        except (TypeError, ValueError) as error:
            return False, str(error) or "Review the Voice setup fields."
        if not draft.sample_text.strip():
            # The sample only matters to the test (voice-speech-05).
            draft = voice_state.replace_draft(
                draft, sample_text=voice_state.DEFAULT_SAMPLE_TEXT
            )
        validation = voice_state.validate_voice_setup_draft(draft)
        if not validation.configuration_valid:
            return False, validation.errors[
                0
            ] if validation.errors else "Review the Voice setup fields."
        if not prefill.should_persist_voice_config(
            draft, self._baseline, acted_this_run=self._tested_this_run
        ):
            return True, ""
        found = (
            self._find_openai_credential()
            if draft.authentication_mode == "api_key"
            else None
        )
        if draft.authentication_mode == "api_key" and found is None:
            return False, voice_status.KEY_NEEDED_COPY
        credential = found[0] if found is not None and found[1] else None
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
                credential=credential,
            )
        )
        try:
            outcome = await asyncio.wait_for(
                asyncio.shield(self._save_future),
                timeout=self._SAVE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return False, "Voice settings are still applying. Retry to continue."
        finally:
            self._save_request_id = None
            self._save_draft = None
            self._save_future = None
        if outcome[0]:
            self._refresh_saved()
        if outcome[0] and credential is not None:
            note = getattr(self.wizard, "note_key_entered", None)
            if callable(note):
                note()  # Protect now offers to encrypt the saved key.
        return outcome

    def busy_label(self) -> str:
        """What a slow Next from Voice is doing: the save can take 30 s."""
        return "Saving voice settings…"

    def restore_checkpoint(self, values: Mapping[str, object]) -> None:
        """Put a resumed run's non-secret Voice values back (TASK-1264)."""
        preset = values.get("preset")
        if preset not in _BUTTON_BY_PRESET:
            preset = voice_state.VOICE_PRESET_CUSTOM
        draft = prefill.draft_from_checkpoint(values, self._initial_draft())
        self._preset = str(preset)
        if preset == voice_state.VOICE_PRESET_CUSTOM:
            self._custom_draft = draft
        self._apply_draft_to_controls(draft)
        self._set_radio(_BUTTON_BY_PRESET[self._preset])
        self.query_one("#setup-voice-body").display = (
            self._preset != voice_state.VOICE_PRESET_NONE
        )
        if preset == voice_state.VOICE_PRESET_OMNIVOICE:
            self._set_omnivoice_mode(True)
        self._refresh_service_status()

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
