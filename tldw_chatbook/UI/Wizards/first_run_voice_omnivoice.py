"""The first-run Voice step's OmniVoice half (TASK-34100.8).

Moved out of ``first_run_voice_step.py`` so the Voice step stays under its
size budget while it grows the SF3 fixes. ``OmniVoiceStepBase`` is a real
``SetupStep`` subclass, not a plain mixin: Textual registers ``@on`` handlers
per message-pump class, so the install handlers and ``@wizard_work`` workers
stay dispatched on ``VoiceSetupStep`` through the MRO (pinned by
``Tests/Wizards/test_first_run_step_modules.py``). Patch THIS module to
replace the OmniVoice calls it makes.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Mapping

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Checkbox, Collapsible, Input, Static

from tldw_chatbook.TTS.omnivoice_artifact_catalog import (
    omnivoice_setup_state,
    run_omnivoice_preflight,
    run_omnivoice_provision,
)
from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message
from tldw_chatbook.UI.Wizards import first_run_voice_status as voice_status
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import SetupStep
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker, wizard_work
from tldw_chatbook.Widgets.ModelArtifacts import (
    InstallProgressed,
    ModelInstallModal,
    ModelInstallProgress,
    make_progress_callback,
)


class OmniVoiceStepBase(SetupStep):
    """OmniVoice state check, one-time install, local sample and save.

    The Voice step's shared fields (preset, test generations, save future)
    live here because both halves read them, and so do the plumbing both
    halves call: playing a sample and receiving a settings save's result
    (moved here by review round 1 to keep the step under its size budget).
    """

    _SAVE_TIMEOUT_SECONDS = 30.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._preset = voice_state.VOICE_PRESET_NONE
        self._test_generation = 0
        self._test_in_progress_generation: int | None = None
        self._tested_this_run = False
        self._next_save_request_id = 1
        self._save_request_id: int | None = None
        self._save_future: asyncio.Future[tuple[bool, str]] | None = None
        self._save_provider = "openai"
        self._save_use_as_default = False
        self._omnivoice_state: str | None = None
        # Bumped per state check and on leaving OmniVoice; a check applies
        # only while it is still the latest one (a stale read is dropped).
        self._omnivoice_state_generation = 0
        self._omnivoice_installing = False
        self._omnivoice_report: Any = None
        self._omnivoice_seed: int | None = None
        self._sample_audio_path: Path | None = None

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

    def _compose_omnivoice_panel(self) -> ComposeResult:
        with Vertical(id="setup-voice-omnivoice-panel") as panel:
            panel.display = self._preset == voice_state.VOICE_PRESET_OMNIVOICE
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
            self.query_one("#setup-voice-key-row").display = False
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
        self.app.call_from_thread(
            self._apply_checked_omnivoice_state, generation, state
        )

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
            report = asyncio.run(
                run_omnivoice_preflight()
            )  # policy-exception: worker-thread loop
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
            self._mark_sample_verified()
            self._update_voice_status_text(
                voice_status.PLAYED_COPY
                if played
                else voice_status.PLAYBACK_FAILED_COPY
            )
            self._refresh_sample_state()
        finally:
            if self._test_in_progress_generation == generation:
                self._test_in_progress_generation = None

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
