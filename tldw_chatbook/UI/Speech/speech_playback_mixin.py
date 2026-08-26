"""Playback, export, file pickers and text actions, shared by both playgrounds.

The third and last shared piece: what the buttons do once the catalog has
filled the axes and synthesis has produced audio. `on_button_pressed`
dispatches all twelve playground buttons by id, so a host that mounts those
ids gets every action by inheriting this.

Unlike `SpeechCatalogMixin` this closure carries no `@on` decorators and no
module-level names the tests monkeypatch, so neither of the traps that bit
the catalog move applies. `on_button_pressed` is a naming-convention
handler, which Textual resolves through the MRO by `getattr` rather than
registering in its metaclass -- so unlike `@on`, it does work from a mixin.

A host must NOT also declare its own handler for any of these twelve buttons:
both would run, and pressing Generate would synthesize twice.
"""

from __future__ import annotations

import asyncio
import math
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Optional
import unicodedata

from loguru import logger
from textual.css.query import NoMatches
from textual.worker import WorkerCancelled
from textual.widgets import Button, RichLog, Static, TextArea

from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen as FileOpen,
    EnhancedFileSave as FileSave,
)

from tldw_chatbook.TTS import STTSGeneratedAudio, STTSPlaygroundResultProjection
from tldw_chatbook.TTS.playground_types import PROFILE_SAVE_BLOCK_PROVIDER_OPTIONS


EXAMPLE_TEXTS = [
    "Welcome to the Text-to-Speech playground! This is where you can experiment with different voices, providers, and settings to create natural-sounding speech.",
    "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet.",
    "In a world of artificial intelligence, the ability to convert text into natural speech opens countless possibilities.",
    "Testing, one, two, three. Can you hear the difference between various voice models?",
    "Good morning! Today's weather is sunny with a high of 75 degrees. Perfect for a walk in the park.",
]

_PROVENANCE_FIELD_MAX_CHARACTERS = 80


def _safe_provenance_field(value: object, fallback: str) -> str:
    """Return one bounded control-safe field for literal provenance display.

    Args:
        value: Provider-controlled identifier to render.
        fallback: Copy used when ``value`` is not a usable string.

    Returns:
        A single-line value capped at the provenance field limit.
    """

    if type(value) is not str:
        return fallback
    flattened = "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf", "Cs"} else character
        for character in value
    )
    flattened = " ".join(flattened.split())
    if not flattened:
        return fallback
    if len(flattened) <= _PROVENANCE_FIELD_MAX_CHARACTERS:
        return flattened
    return flattened[: _PROVENANCE_FIELD_MAX_CHARACTERS - 1].rstrip() + "…"


class SpeechPlaybackMixin:
    """Transport, export and picker behaviour, independent of the layout."""

    def _generation_complete(
        self,
        artifact: STTSGeneratedAudio | STTSPlaygroundResultProjection | None,
    ) -> None:
        """Store only sanitized result facts independently of selectors."""
        if type(artifact) is STTSGeneratedAudio:
            artifact = STTSPlaygroundResultProjection.from_artifact(artifact)
        if (
            artifact is not None
            and artifact.operation_id == self._retired_profile_operation_id
        ):
            return
        if (
            artifact is not None
            and self._generation_operation_id is not None
            and artifact.operation_id != self._generation_operation_id
        ):
            return
        if artifact is not None:
            if self._playback_transition_required():
                # Keep the previous result and its reachable Stop control
                # authoritative until its playback has actually retired.
                # Replacing the card first would strand still-audible audio.
                self._generation_operation_id = artifact.operation_id
                self._result_transition_operation_id = artifact.operation_id
                self._sync_generate_enabled()
                play = self.query_one("#audio-play-btn", Button)
                play.disabled = True
                play.tooltip = (
                    "Wait for current playback to stop before playing the new result"
                )
                self.run_worker(
                    self._replace_result_after_playback(artifact),
                    name="replace_tts_result_after_playback",
                    group="replace_tts_result_after_playback",
                    exclusive=True,
                    exit_on_error=False,
                )
                return
            self._generation_operation_id = None
            self._sync_generate_enabled()
            self._publish_delivered_artifact(artifact)
        else:
            self._generation_operation_id = None
            self._sync_generate_enabled()
            self._profile_save_suppressed = True
            self._sync_save_profile_action()
            log = self.query_one("#tts-generation-log", RichLog)
            log.write("[bold red]✗ TTS generation failed![/bold red]")
            result_hook = getattr(self, "_on_generation_result", None)
            if callable(result_hook):
                result_hook(None)

    def _playback_transition_required(self) -> bool:
        """Return whether replacing the result must first retire playback."""

        worker = self._play_worker_task
        worker_active = worker is not None and not worker.is_finished
        timer = self._progress_timer_task
        timer_active = timer is not None and not timer.done()
        return bool(
            worker_active or timer_active or self._active_playback_release is not None
        )

    async def _replace_result_after_playback(
        self,
        artifact: STTSPlaygroundResultProjection,
    ) -> None:
        """Retire older playback before publishing and optionally playing a result."""

        operation_id = artifact.operation_id
        try:
            worker = self._play_worker_task
            if worker is not None and not worker.is_finished:
                worker.cancel()
                try:
                    await worker.wait()
                except WorkerCancelled:
                    pass
                except Exception as error:
                    logger.debug(
                        "Prior playback worker retired with {}",
                        type(error).__name__,
                    )
                finally:
                    if self._play_worker_task is worker:
                        self._play_worker_task = None

            if not await self._stop_audio_async():
                self.app.notify(
                    "Could not replace the current result while audio is playing",
                    severity="warning",
                )
                return
            if (
                self._result_transition_operation_id != operation_id
                or operation_id == self._retired_profile_operation_id
            ):
                return
            # Publish from the same idle lifecycle as an immediate result so
            # native profile actions and Studio auto-play can become eligible.
            if self._generation_operation_id == operation_id:
                self._generation_operation_id = None
                self._sync_generate_enabled()
            self._result_transition_operation_id = None
            self._publish_delivered_artifact(artifact)
        finally:
            if self._result_transition_operation_id == operation_id:
                self._result_transition_operation_id = None
            if self._generation_operation_id == operation_id:
                self._generation_operation_id = None
                self._sync_generate_enabled()

    def _publish_delivered_artifact(
        self,
        artifact: STTSPlaygroundResultProjection,
    ) -> None:
        """Publish one result only after prior playback ownership is settled."""

        self._store_delivered_artifact(artifact, announce=True)
        result_hook = getattr(self, "_on_generation_result", None)
        if callable(result_hook):
            result_hook(artifact)
        preferences = getattr(self, "studio_preferences", None)
        if getattr(preferences, "auto_play", False) is True:
            self._play_audio()
            self.call_after_refresh(
                self._focus_current_result_action,
                "stop-audio-btn",
            )
        else:
            self.call_after_refresh(
                self._focus_current_result_action,
                "audio-play-btn",
            )

    def init_playback_state(self) -> None:
        """Initialise the state playback and export read.

        Call from the host's ``__init__``.
        """
        #: The most recent generated file and its artifact handle.
        self.current_audio_file: Any = None
        self.current_audio_artifact: STTSPlaygroundResultProjection | None = None
        #: In-flight playback and its progress ticker, so both can be
        #: cancelled when a new take starts or the pane unmounts.
        self._progress_timer_task: Any = None
        self._play_worker_task: Any = None
        #: Releases the current artifact's hold; called before replacing it.
        self._active_playback_release: Callable[[], None] | None = None
        #: Result currently waiting for prior playback to stop before delivery.
        self._result_transition_operation_id: str | None = None
        #: A generation retired by an exact profile navigation.
        self._retired_profile_operation_id: str | None = None
        self.example_texts = EXAMPLE_TEXTS

    def _retire_profile_generation_context(self) -> None:
        """Fence prior generation while retaining completed playable audio."""

        operation_id = self._generation_operation_id
        if not isinstance(operation_id, str):
            return
        self._retired_profile_operation_id = operation_id
        self._generation_operation_id = None
        handler = getattr(self.app, "_stts_handler", None)
        retire = getattr(handler, "retire_playground_generation", None)
        if callable(retire):
            try:
                retire(operation_id)
            except Exception as error:
                logger.debug(
                    "Could not retire Playground handler context ({})",
                    type(error).__name__,
                )
        if self.is_mounted:
            self.query_one("#generation-status-container").add_class("hidden")
            if self.current_audio_artifact is None:
                self._sync_idle_transport_actions()
                export = self.query_one("#audio-export-btn", Button)
                export.disabled = True
                export.tooltip = "Generate audio before saving a current result"
            self._sync_generate_enabled()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        logger.debug(f"Playground received button press: {event.button.id}")
        if event.button.id == "tts-generate-btn":
            self._generate_tts()
            event.stop()  # Prevent event from bubbling up
        elif event.button.id == "tts-test-connection-btn":
            lifecycle_handler = getattr(self, "_handle_connection_action", None)
            handled = bool(callable(lifecycle_handler) and lifecycle_handler())
            if not handled and self._selected_provider_id is not None:
                self._load_provider_catalog(
                    self._selected_provider_id,
                    refresh=True,
                )
            event.stop()
        elif event.button.id == "tts-refresh-catalog-btn":
            if self._selected_provider_id is not None:
                self._load_provider_catalog(
                    self._selected_provider_id,
                    refresh=True,
                )
            event.stop()
        elif event.button.id == "tts-random-text-btn":
            self._insert_random_text()
            event.stop()
        elif event.button.id == "tts-clear-text-btn":
            self._clear_text()
            event.stop()
        elif event.button.id == "audio-play-btn":
            self._play_audio()
            event.stop()
        elif event.button.id == "pause-audio-btn":
            logger.debug("Pause button clicked")
            self._pause_audio()
            event.stop()
        elif event.button.id == "stop-audio-btn":
            logger.debug("Stop button clicked")
            self._stop_audio()
            event.stop()
        elif event.button.id == "audio-export-btn":
            self._export_audio()
            event.stop()
        elif event.button.id == "audio-generate-again-btn":
            self._generate_tts()
            event.stop()
        elif event.button.id == "audio-save-profile-btn":
            event.button.disabled = True
            self.run_worker(
                self._save_current_result_as_profile(),
                name="save_tts_result_as_profile",
                group="save_tts_result_as_profile",
                exclusive=True,
                exit_on_error=False,
            )
            event.stop()
        elif event.button.id == "reference-audio-btn":
            self._select_reference_audio()
            event.stop()
        elif event.button.id == "clear-reference-audio-btn":
            self._clear_reference_audio()
            event.stop()
        elif event.button.id == "higgs-voice-upload-btn":
            self._upload_higgs_voice()
            event.stop()
        elif event.button.id == "higgs-clear-voice-btn":
            self._clear_higgs_voice()
            event.stop()

    def _store_delivered_artifact(
        self,
        artifact: STTSGeneratedAudio | STTSPlaygroundResultProjection,
        *,
        announce: bool,
    ) -> None:
        if type(artifact) is STTSGeneratedAudio:
            artifact = STTSPlaygroundResultProjection.from_artifact(artifact)
        self.current_audio_artifact = artifact
        self.current_audio_file = artifact.path
        self._profile_save_suppressed = False
        if announce:
            self.query_one("#tts-generation-log", RichLog).write(
                "[bold green]✓ TTS generation complete![/bold green]"
            )
        self._sync_idle_transport_actions()
        self.query_one("#audio-export-btn", Button).disabled = False
        repeat = self.query_one("#audio-generate-again-btn", Button)
        repeat.disabled = False
        repeat.tooltip = "Generate another result with the current Speech Lab controls"
        save = self.query_one("#audio-export-btn", Button)
        save.label = f"Save {artifact.audio_format.upper()} as…"
        save.tooltip = f"Save the current {artifact.audio_format.upper()} result"
        self._sync_save_profile_action()
        self.query_one("#audio-player-status", Static).update(
            self._current_result_status_copy()
        )
        try:
            self.query_one("#audio-result-lifecycle", Static).update(
                self._result_lifecycle_copy(artifact)
            )
            self.query_one("#audio-result-provenance", Static).update(
                self._current_result_provenance_copy(artifact)
            )
            self.query_one("#audio-player-transport").add_class("hidden")
            self.query_one("#audio-progress-bar").add_class("hidden")
            time_display = self.query_one("#audio-time-display", Static)
            time_display.update("")
            time_display.add_class("hidden")
        except NoMatches:
            pass

    def _sync_idle_transport_actions(self) -> None:
        """Project current-result transport controls in a truthful idle state."""

        has_result = self._current_generated_audio_path() is not None
        play = self.query_one("#audio-play-btn", Button)
        pause = self.query_one("#pause-audio-btn", Button)
        stop = self.query_one("#stop-audio-btn", Button)
        play.disabled = not has_result
        play.tooltip = (
            "Play the current result"
            if has_result
            else "Generate audio before playing the current result"
        )
        pause.label = "Pause"
        pause.disabled = True
        pause.tooltip = (
            "Start playback before pausing"
            if has_result
            else "Generate audio before pausing playback"
        )
        stop.disabled = True
        stop.tooltip = (
            "Start playback before stopping"
            if has_result
            else "Generate audio before stopping playback"
        )

    def _sync_active_transport_actions(self) -> None:
        """Project current-result transport controls while playback is active."""

        play = self.query_one("#audio-play-btn", Button)
        pause = self.query_one("#pause-audio-btn", Button)
        stop = self.query_one("#stop-audio-btn", Button)
        play.disabled = True
        play.tooltip = "Playback is already in progress"
        pause.disabled = False
        pause.tooltip = (
            "Resume playback" if str(pause.label) == "Resume" else "Pause playback"
        )
        stop.disabled = False
        stop.tooltip = "Stop playback"

    @staticmethod
    def _artifact_duration_seconds(
        artifact: STTSPlaygroundResultProjection | None,
    ) -> float | None:
        """Return a positive duration from bounded artifact metadata, if known."""

        if artifact is None:
            return None
        for key, scale in (
            ("audio_duration_ms", 0.001),
            ("duration_seconds", 1.0),
            ("audio_duration", 1.0),
        ):
            value = artifact.metadata.get(key)
            if isinstance(value, bool) or not isinstance(value, Real):
                continue
            duration = float(value) * scale
            if math.isfinite(duration) and duration > 0:
                return duration
        frame_count = artifact.metadata.get("frame_count")
        sample_rate = artifact.metadata.get("sample_rate")
        if (
            not isinstance(frame_count, bool)
            and isinstance(frame_count, Real)
            and not isinstance(sample_rate, bool)
            and isinstance(sample_rate, Real)
            and math.isfinite(float(frame_count))
            and math.isfinite(float(sample_rate))
            and float(frame_count) > 0
            and float(sample_rate) > 0
        ):
            return float(frame_count) / float(sample_rate)
        return None

    @staticmethod
    def _format_result_duration(seconds: float) -> str:
        """Format a known positive duration without rounding it down to zero."""

        whole_seconds = max(1, int(seconds + 0.5))
        minutes, remaining = divmod(whole_seconds, 60)
        return f"{minutes}:{remaining:02d}"

    def _current_result_status_copy(self) -> str:
        """Describe the current artifact using only known immutable facts."""

        artifact = self.current_audio_artifact
        if artifact is None:
            return "No audio generated yet"
        format_copy = artifact.audio_format.upper()
        if (
            artifact.audio_format.casefold() == "wav"
            and artifact.metadata.get("delivery") == "complete_wav"
        ):
            format_copy = "Complete WAV"
        parts = ["Ready", format_copy]
        duration = self._artifact_duration_seconds(artifact)
        if duration is not None:
            parts.append(self._format_result_duration(duration))
        return " · ".join(parts)

    @staticmethod
    def _current_result_provenance_copy(
        artifact: STTSPlaygroundResultProjection,
    ) -> str:
        """Return path-free provenance captured with the delivered artifact."""

        provider = (
            "audio.cpp"
            if artifact.provider_id == "audio_cpp"
            else _safe_provenance_field(
                artifact.provider_id,
                "Unknown provider",
            )
            .replace("_", " ")
            .title()
        )
        model = _safe_provenance_field(artifact.model_id, "Unknown model")
        voice = _safe_provenance_field(artifact.voice_id, "Server default")
        parts = [
            f"Provider: {provider}",
            f"Model: {model}",
            f"Voice: {voice}",
        ]
        selection = artifact.requested_selection
        if selection is not None:
            parts.append(f"configuration revision {selection.configuration_revision}")
        process_generation = artifact.metadata.get("process_generation")
        if type(process_generation) is int and process_generation >= 0:
            parts.append(f"process generation {process_generation}")
        return " · ".join(parts)

    @staticmethod
    def _result_lifecycle_copy(artifact: STTSPlaygroundResultProjection) -> str:
        """State the file's lifecycle, and why Save is missing when it is.

        An unexplained missing Save affordance is the chrome-honesty defect
        this repo forbids: slice-1 profiles fix `options` empty, so a
        generation that used provider-specific options (Higgs/ElevenLabs/
        Chatterbox/Kokoro Inputs) cannot be reproduced by one and is refused
        provenance -- `profile_save_block_code` names that one actionable
        reason. Every other artifact keeps the original copy verbatim.
        """

        if artifact.profile_save_block_code == PROFILE_SAVE_BLOCK_PROVIDER_OPTIONS:
            return (
                "Temporary result — export to keep a copy. Not saved as a "
                "voice profile: this generation used provider-specific "
                "options, which voice profiles don't capture yet."
            )
        return "Temporary result — export to keep a copy."

    def _focus_current_result_action(self, action_id: str) -> None:
        """Focus and reveal the safest next action after artifact delivery."""

        if not self.is_mounted:
            return
        try:
            target = self.query_one(f"#{action_id}", Button)
            if target.disabled:
                target = self.query_one("#audio-play-btn", Button)
        except NoMatches:
            return
        target.focus()
        target.scroll_visible(animate=False)

    def _current_generated_audio_path(self) -> Path | None:
        """Return the delivered artifact path, with legacy path fallback."""
        if self.current_audio_artifact is not None:
            return self.current_audio_artifact.path
        if self.current_audio_file is None:
            return None
        return Path(self.current_audio_file)

    def _capture_audio_for_action(
        self,
    ) -> tuple[Path, Callable[[], None]] | None:
        """Capture and, when handler-owned, lease the current artifact."""
        artifact = self.current_audio_artifact
        audio_path = self._current_generated_audio_path()
        if audio_path is None:
            return None
        if artifact is None or artifact.path != audio_path:
            return audio_path, lambda: None

        handler = getattr(self.app, "_stts_handler", None)
        acquire = getattr(handler, "lease_playground_result", None)
        release = getattr(handler, "release_playground_result", None)
        if not callable(acquire) or not callable(release):
            return audio_path, lambda: None
        try:
            if not acquire(artifact.operation_id, artifact.path):
                return None
        except Exception as error:
            logger.debug(
                "Could not lease Playground artifact ({})",
                type(error).__name__,
            )
            return None

        released = False

        def release_once() -> None:
            nonlocal released
            if released:
                return
            released = True
            try:
                release(artifact.operation_id, artifact.path)
            except Exception as error:
                logger.debug(
                    "Could not release Playground artifact ({})",
                    type(error).__name__,
                )

        return audio_path, release_once

    def _release_playback_artifact(self) -> None:
        """Release the artifact retained by the active playback, if any."""
        release = self._active_playback_release
        self._active_playback_release = None
        if release is not None:
            release()

    def _play_audio(self) -> None:
        """Play the generated audio"""
        logger.debug(
            f"_play_audio called, current_audio_file: {self.current_audio_file}"
        )

        if self._result_transition_operation_id is not None:
            logger.debug("Result replacement is stopping prior playback")
            return

        # Check if we're already playing
        if (
            hasattr(self, "_play_worker_task")
            and self._play_worker_task
            and not self._play_worker_task.is_finished
        ):
            logger.debug("Play already in progress, ignoring request")
            return

        captured = self._capture_audio_for_action()
        if captured is None:
            self.app.notify("No audio file to play", severity="warning")
            return
        audio_path, release_artifact = captured

        logger.debug(
            f"Audio path: {audio_path}, exists: {audio_path.exists() if audio_path else False}"
        )

        if not audio_path.exists():
            release_artifact()
            self.app.notify(
                f"Audio file not found: {audio_path.name}", severity="warning"
            )
            return

        if self._ensure_audio_player():
            # Cancel any existing progress timer first
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                self._progress_timer_task = None
                logger.debug("Cancelled existing progress timer")

            self._sync_active_transport_actions()
            self.query_one("#audio-player-status", Static).update(
                "Playing current result…"
            )

            # Use the new audio player method
            # Store the worker task so we can check if it's running
            playback = self._play_audio_async(audio_path, release_artifact)
            try:
                self._play_worker_task = self.run_worker(
                    playback,
                    group="stts-playback",
                    exclusive=True,
                )
            except Exception:
                playback.close()
                release_artifact()
                raise
        else:
            release_artifact()
            self.app.notify("Audio playback not available", severity="warning")

    async def _play_audio_async(
        self,
        audio_path: Path | None = None,
        release_artifact: Callable[[], None] | None = None,
    ) -> None:
        """Play audio asynchronously using the audio player"""
        try:
            if audio_path is None:
                audio_path = self._current_generated_audio_path()
            if audio_path is not None:
                if audio_path.exists():
                    # Get current player state before stopping
                    current_state = await self.app.audio_player.get_state()
                    logger.debug(f"Current player state before play: {current_state}")

                    # Always force stop any existing playback first
                    stop_result = await self.app.audio_player.stop()
                    logger.debug(f"Stop result: {stop_result}")
                    self._release_playback_artifact()

                    # Small delay to ensure clean state
                    import asyncio

                    await asyncio.sleep(0.2)

                    # Check state after stop
                    state_after_stop = await self.app.audio_player.get_state()
                    logger.debug(f"Player state after stop: {state_after_stop}")

                    # Attempt to play the audio file
                    logger.info(f"Attempting to play audio file: {audio_path}")
                    success = await self.app.audio_player.play(audio_path)
                    logger.debug(f"Play result: {success}")

                    if success:
                        self._active_playback_release = release_artifact
                        release_artifact = None

                        # Cancel any existing progress timer
                        if (
                            self._progress_timer_task
                            and not self._progress_timer_task.done()
                        ):
                            self._progress_timer_task.cancel()
                            await asyncio.sleep(
                                0.05
                            )  # Small delay to ensure cancellation

                        # Start new progress timer
                        self._progress_timer_task = asyncio.create_task(
                            self._update_progress_timer()
                        )
                        logger.debug("Started new progress timer")

                        # Wait a tiny bit to ensure the player has started
                        await asyncio.sleep(0.1)

                        # Double-check the player is actually playing
                        is_playing = await self.app.audio_player.is_playing()
                        logger.debug(
                            f"Player is_playing check after start: {is_playing}"
                        )

                        # Clear the worker task reference as it's now running
                        self._play_worker_task = None
                    else:
                        logger.error("Failed to start playback - play() returned False")
                        self.app.notify("Failed to start playback", severity="error")
                        # Reset button states on failure
                        self._sync_idle_transport_actions()
                        self.query_one("#audio-player-status", Static).update(
                            "Playback failed"
                        )
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
                    self.app.notify(
                        f"Audio file not found: {audio_path.name}", severity="warning"
                    )
            else:
                logger.warning("No audio file to play")
                self.app.notify("No audio file to play", severity="warning")
        except Exception as e:
            logger.opt(exception=True).error(f"Error playing audio: {e}")
            self.app.notify(f"Playback error: {str(e)}", severity="error")
            # Reset button states on error
            self._sync_idle_transport_actions()
            self.query_one("#audio-player-status", Static).update("Playback error")
        finally:
            if release_artifact is not None:
                release_artifact()

    def _ensure_audio_player(self) -> bool:
        """Ensure audio player is initialized (lazy loading)"""
        if not hasattr(self.app, "audio_player"):
            try:
                from tldw_chatbook.TTS.audio_player import AsyncAudioPlayer

                self.app.audio_player = AsyncAudioPlayer()
                logger.info("Audio player initialized on first use")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize audio player: {e}")
                self.app.notify("Failed to initialize audio player", severity="error")
                return False
        return True

    def _pause_audio(self) -> None:
        """Pause audio playback"""
        logger.debug("_pause_audio called")
        if self._ensure_audio_player():
            logger.debug("Audio player available, running pause worker")
            self.run_worker(
                self._pause_audio_async,
                group="stts-playback",
                exclusive=True,
            )
        else:
            logger.debug("Audio player not available")
            self.app.notify("Audio player not available", severity="warning")

    async def _pause_audio_async(self) -> None:
        """Pause audio playback asynchronously"""
        try:
            from tldw_chatbook.TTS.audio_player import PlaybackState
            import asyncio

            logger.debug("_pause_audio_async called")
            # Small delay to ensure UI is ready
            await asyncio.sleep(0.1)

            state = await self.app.audio_player.get_state()
            logger.debug(f"Current playback state: {state}")
            if state == PlaybackState.PLAYING:
                success = await self.app.audio_player.pause()
                if success:
                    # Update button states
                    self.query_one("#pause-audio-btn", Button).label = "Resume"
                    self._sync_active_transport_actions()
                    self.app.notify("Playback paused", severity="information")
                else:
                    self.app.notify("Failed to pause playback", severity="warning")
            elif state == PlaybackState.PAUSED:
                success = await self.app.audio_player.resume()
                if success:
                    # Update button states
                    self.query_one("#pause-audio-btn", Button).label = "Pause"
                    self._sync_active_transport_actions()
                    self.app.notify("Playback resumed", severity="information")
                    # Cancel any existing timer and restart
                    if (
                        self._progress_timer_task
                        and not self._progress_timer_task.done()
                    ):
                        self._progress_timer_task.cancel()
                    import asyncio

                    self._progress_timer_task = asyncio.create_task(
                        self._update_progress_timer()
                    )
                else:
                    self.app.notify("Failed to resume playback", severity="warning")
        except Exception as e:
            logger.error(f"Error toggling pause: {e}")
            from rich.markup import escape

            self.app.notify(f"Error: {escape(str(e))}", severity="error")

    def _stop_audio(self) -> None:
        """Stop audio playback"""
        logger.debug("_stop_audio called")
        if self._ensure_audio_player():
            logger.debug("Audio player available, running stop worker")
            self.run_worker(
                self._stop_audio_async,
                group="stts-playback",
                exclusive=True,
            )
        else:
            logger.debug("Audio player not available")
            self.app.notify("Audio player not available", severity="warning")

    async def _stop_audio_async(self) -> bool:
        """Stop audio playback asynchronously and report a safe idle result."""
        try:
            logger.debug("_stop_audio_async called")
            # Cancel progress timer if running
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                self._progress_timer_task = None

            # Force stop any playback
            success = await self.app.audio_player.stop()
            logger.debug(f"Stop result: {success}")
            self._release_playback_artifact()

            # Also ensure progress timer is cancelled
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                await asyncio.sleep(0.1)  # Give it time to cancel

            # Always reset button states regardless of success
            # (audio may have already finished playing)
            self._sync_idle_transport_actions()
            self.query_one("#audio-player-transport").add_class("hidden")

            if success:
                self.query_one("#audio-player-status", Static).update(
                    self._current_result_status_copy()
                )
                self.app.notify("Playback stopped", severity="information")
            else:
                # Audio already finished or wasn't playing
                self.query_one("#audio-player-status", Static).update(
                    self._current_result_status_copy()
                )
                logger.debug("Audio may have already finished playing")
            return True
        except Exception as e:
            logger.error(f"Error stopping playback: {e}")
            from rich.markup import escape

            self.app.notify(f"Error: {escape(str(e))}", severity="error")
            return False

    def _export_audio(self) -> None:
        """Export the generated audio"""
        captured = self._capture_audio_for_action()
        if captured is None:
            self.app.notify("No audio file to export", severity="warning")
            return
        original_path, release_artifact = captured
        if not original_path.exists():
            release_artifact()
            self.app.notify("No audio file to export", severity="warning")
            return

        # Create file save dialog
        filters = Filters(
            (
                "Audio Files",
                lambda p: (
                    p.suffix.lower() in [".mp3", ".wav", ".aac", ".flac", ".opus"]
                ),
            ),
            ("All Files", lambda p: True),
        )

        # Get original filename and extension
        default_name = f"tts_export_{original_path.stem}{original_path.suffix}"

        file_picker = FileSave(
            title="Export Audio File",
            filters=filters,
            default_filename=default_name,
            context="audio_export",
        )

        def handle_export(path: Optional[str]) -> None:
            try:
                self._handle_audio_export(path, source_path=original_path)
            finally:
                release_artifact()

        try:
            self.app.push_screen(file_picker, handle_export)
        except Exception:
            release_artifact()
            raise

    def _handle_audio_export(
        self,
        path: Optional[str],
        *,
        source_path: Path | None = None,
    ) -> None:
        """Handle audio file export"""
        if source_path is None:
            source_path = self._current_generated_audio_path()
        if not path or source_path is None:
            return

        try:
            import shutil
            from tldw_chatbook.Utils.path_validation import (
                validate_filename,
                validate_path_simple,
            )

            dest_path = Path(path)

            # If different format requested, we need conversion
            if source_path.suffix.lower() != dest_path.suffix.lower():
                # For now, just copy - format conversion would require audio service
                self.app.notify(
                    f"Format conversion not yet implemented. Exporting as {source_path.suffix}",
                    severity="warning",
                )
                dest_path = dest_path.with_suffix(source_path.suffix)

            validate_path_simple(dest_path, require_exists=False)
            validated_parent = validate_path_simple(
                dest_path.parent,
                require_exists=True,
            ).resolve()
            validated_filename = validate_filename(dest_path.name)
            dest_path = validated_parent / validated_filename

            # Copy the file
            shutil.copy2(source_path, dest_path)
            self.app.notify(f"Audio exported to: {dest_path.name}", severity="success")

        except Exception as e:
            logger.error(f"Failed to export audio: {e}")
            self.app.notify(f"Export failed: {str(e)}", severity="error")

    def _select_reference_audio(self) -> None:
        """Select reference audio file for voice cloning"""
        # Create file picker for audio files using pre-imported FileOpen
        filters = Filters(
            (
                "Audio Files",
                lambda p: p.suffix.lower() in [".wav", ".mp3", ".m4a", ".flac", ".aac"],
            ),
            ("All Files", lambda p: True),
        )

        file_picker = FileOpen(
            title="Select Reference Audio", filters=filters, context="reference_audio"
        )

        # Mount the file picker
        self.app.push_screen(file_picker, self._handle_reference_audio_selection)

    def _handle_reference_audio_selection(self, path: Optional[str]) -> None:
        """Handle reference audio file selection"""
        if path:
            self.reference_audio_path = path
            # Update status
            status = self.query_one("#reference-audio-status", Static)
            filename = Path(path).name
            status.update(f"Selected: {filename}")
            # Enable clear button
            self.query_one("#clear-reference-audio-btn", Button).disabled = False
            logger.info(f"Reference audio selected: {path}")
        else:
            logger.info("Reference audio selection cancelled")

    def _clear_reference_audio(self) -> None:
        """Clear the selected reference audio"""
        self.reference_audio_path = None
        # Update status
        status = self.query_one("#reference-audio-status", Static)
        status.update("No reference audio selected")
        # Disable clear button
        self.query_one("#clear-reference-audio-btn", Button).disabled = True
        logger.info("Reference audio cleared")

    def _upload_higgs_voice(self) -> None:
        """Open file dialog to select reference audio for Higgs voice cloning"""

        def handle_selection(path: Optional[Path]) -> None:
            if path:
                self.higgs_reference_audio_path = str(path)
                # Update status
                status = self.query_one("#higgs-voice-status", Static)
                status.update(f"Selected: {path.name}")
                # Enable clear button
                self.query_one("#higgs-clear-voice-btn", Button).disabled = False
                logger.info(f"Higgs reference audio selected: {path}")
            else:
                logger.info("Higgs reference audio selection cancelled")

        file_open = FileOpen(
            title="Select Reference Audio for Voice Cloning",
            filters=Filters(
                (
                    "Audio Files",
                    lambda p: (
                        p.suffix.lower()
                        in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
                    ),
                ),
                ("All Files", lambda p: True),
            ),
            must_exist=True,
        )
        self.app.push_screen(file_open, handle_selection)

    def _clear_higgs_voice(self) -> None:
        """Clear the selected Higgs reference audio"""
        self.higgs_reference_audio_path = None
        # Update status
        status = self.query_one("#higgs-voice-status", Static)
        status.update("No voice reference selected")
        # Disable clear button
        self.query_one("#higgs-clear-voice-btn", Button).disabled = True
        logger.info("Higgs reference audio cleared")

    def _insert_random_text(self) -> None:
        """Insert a random example text"""
        import random

        text_area = self.query_one("#tts-text-input", TextArea)
        text_area.text = random.choice(self.example_texts)
        text_area.focus()
        self.app.notify("Sample text inserted", severity="information")

    def _clear_text(self) -> None:
        """Clear the text input"""
        text_area = self.query_one("#tts-text-input", TextArea)
        text_area.clear()
        text_area.focus()
        self.app.notify("Text cleared", severity="information")

    def action_random_text(self) -> None:
        """Insert sample text from the keyboard shortcut."""
        self._insert_random_text()

    def action_clear_text(self) -> None:
        """Keyboard shortcut action for clear text"""
        self._clear_text()

    def action_play_audio(self) -> None:
        """Keyboard shortcut action for play audio"""
        if not self.query_one("#audio-play-btn", Button).disabled:
            self._play_audio()

    def action_stop_audio(self) -> None:
        """Keyboard shortcut action for stop audio"""
        if not self.query_one("#stop-audio-btn", Button).disabled:
            self._stop_audio()

    async def _update_progress_timer(self) -> None:
        """Update progress bar during playback"""
        import asyncio
        from tldw_chatbook.TTS.audio_player import PlaybackState
        from textual.widgets import ProgressBar

        # Ensure audio player exists
        if not hasattr(self.app, "audio_player"):
            return

        while True:
            try:
                state = await self.app.audio_player.get_state()
                if state == PlaybackState.PLAYING:
                    position = await self.app.audio_player.get_position()
                    duration = await self.app.audio_player.get_duration()

                    if duration and duration > 0:
                        # Update progress bar
                        progress_bar = self.query_one(
                            "#audio-progress-bar", ProgressBar
                        )
                        progress_bar.update(progress=position, total=duration)

                        # Update time display
                        time_display = self.query_one("#audio-time-display")
                        current_time = self._format_time(position)
                        total_time = self._format_time(duration)
                        time_display.update(f"{current_time} / {total_time}")

                        # Show progress elements
                        progress_bar.remove_class("hidden")
                        time_display.remove_class("hidden")
                        self.query_one("#audio-player-transport").remove_class("hidden")
                elif state in [PlaybackState.IDLE, PlaybackState.FINISHED]:
                    self._release_playback_artifact()

                    if self._result_transition_operation_id is None:
                        # Hide progress elements
                        self.query_one("#audio-progress-bar").add_class("hidden")
                        self.query_one("#audio-time-display").add_class("hidden")
                        self.query_one("#audio-player-transport").add_class("hidden")

                        # Reset button states when playback finishes
                        self._sync_idle_transport_actions()
                        self.query_one("#audio-player-status", Static).update(
                            self._current_result_status_copy()
                        )

                    # Notify that playback is complete
                    if state == PlaybackState.FINISHED:
                        self.app.notify("Playback complete", severity="information")

                    break

                await asyncio.sleep(0.1)  # Update every 100ms
            except asyncio.CancelledError:
                logger.debug("Progress timer cancelled")
                break
            except Exception as e:
                logger.error(f"Error updating progress: {e}")
                break

        # Ensure UI is reset on exit
        try:
            if self._result_transition_operation_id is None:
                self._sync_idle_transport_actions()
                self.query_one("#audio-player-transport").add_class("hidden")
                self.query_one("#audio-player-status", Static).update(
                    self._current_result_status_copy()
                )
        except Exception as e:
            logger.debug(f"Could not reset UI on progress timer exit: {e}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    async def on_unmount(self) -> None:
        """Clean up resources when widget is unmounted"""
        try:
            invalidate_profile_mount = getattr(
                self,
                "invalidate_profile_mount_callbacks",
                None,
            )
            if callable(invalidate_profile_mount):
                invalidate_profile_mount()
            retire_profile_test = getattr(
                self,
                "_retire_profile_test_authority",
                None,
            )
            if callable(retire_profile_test):
                retire_profile_test()
            close_clone_setup = getattr(self, "_close_clone_setup", None)
            if callable(close_clone_setup):
                await close_clone_setup()
            self.app.workers.cancel_group(self, "stts-catalog-discovery")
            self.app.workers.cancel_group(self, "stts-voice-discovery")
            self.app.workers.cancel_group(self, "stts-playback")
            self.app.workers.cancel_group(self, "replace_tts_result_after_playback")
            # Cancel any active progress timer
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                await asyncio.sleep(0.05)

            # Cancel any active play worker
            if (
                hasattr(self, "_play_worker_task")
                and self._play_worker_task
                and not self._play_worker_task.is_finished
            ):
                self._play_worker_task.cancel()
                await asyncio.sleep(0.05)

            # Stop audio playback if active
            if hasattr(self.app, "audio_player"):
                await self.app.audio_player.stop()
                self._release_playback_artifact()
            else:
                self._release_playback_artifact()

            logger.debug("Playground cleanup completed")
        except Exception as e:
            logger.error(f"Error during Playground cleanup: {e}")
