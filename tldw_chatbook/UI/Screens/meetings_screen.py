"""Meetings destination: record a call or a room with a live transcript.

The running session is app-owned (`app.meeting_session_owner`, spec §3.4);
this screen attaches on mount and detaches on unmount. Session callbacks
arrive on capture threads and cross to the loop with `call_from_thread`.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Input, ProgressBar, RichLog, Select, Static

from ...Audio.meeting_owner import PrepareResult, meeting_user_display_name, recover_folder
from ...Audio.meeting_session import (
    MeetingResult,
    MeetingSegment,
    format_clock,
    is_widget_safe_cluster_id,
    render_label,
)
from ...config import save_setting_to_cli_config
from ...Constants import LIBRARY_NAV_CONTEXT_INGEST, TAB_LIBRARY
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen

# "you"/"both" are NOT here (task 31746 review): `_coarse_label` resolves
# them through the configured display-name helper instead, so a stale
# literal can't drift out of sync with it again. This is only the fallback
# for "others" and any other coarse label.
LABELS = {"others": "Others"}
STOP_REASON_COPY = {
    "mic_lost": "Microphone stopped delivering audio; the meeting was ended.",
    "disk_error": "Recording stopped: the disk write failed.",
}
#: Rail copy for every `VoiceMatchState.reason` (spec §3.5). Static strings:
#: never a path, a device name or anything derived from the audio.
VOICE_MATCH_OFF_COPY = {
    "disabled": "disabled",
    "plain_call_mode": "plain call mode",
    "live_labels_off": "live speaker labels off",
    "no_voiceprint": "no voiceprint",
    "needs_reenrollment": "needs re-enrollment",
    "cannot_decrypt": "store unreadable",
    "keyring_locked": "keyring locked",
    "store_unavailable": "store unavailable",
}
#: Why an enrollment did not produce a voiceprint (`EnrollResult.reason`).
#: Anything unmapped is a recorder's own first line, already user-facing.
ENROLL_REASON_COPY = {
    "capture_busy": "the microphone is already in use",
    "cancelled": "cancelled",
    "no_audio": "no audio was captured",
    "embed_failed": "the voice model could not use the sample",
    "diarizer_unavailable": "the speaker model is unavailable",
    "capture_failed": "the microphone could not be opened",
    "mic_device_not_found": "the selected microphone was not found",
    "store_unavailable": "the voiceprint store is unavailable",
}
#: Static copy for a failed voiceprint export/import, most specific first.
#: The keys index `_build_store`'s resolved exception classes.
VOICE_TRANSFER_FAILURE_COPY = (
    ("passphrase", "Wrong passphrase"),
    ("destination", "That file is your stored voiceprint — choose another destination"),
    ("model", "Different model — choose Replace"),
    ("locked", "Store locked — try again after unlocking the keyring"),
)
#: The owner's static progress words, as Voice-row copy (final review I3).
#: An unmapped word shows nothing rather than being interpolated.
LEARN_PROGRESS_COPY = {"warming up": "Warming up the voice model…"}
#: Marker on the auto-matched cluster's label, so an automatic name is never
#: mistaken for one the user typed (spec §3.4).
SELF_MARKER = " ·"
ENROLL_SECONDS = 30.0


class MeetingsScreen(BaseAppScreen):
    """Record a call (mic + system audio) or a room (mic only)."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "meetings", **kwargs)
        self._owner = getattr(app_instance, "meeting_session_owner", None)
        self._session: Any | None = None
        # Whether this meeting asked for live speaker labels, and whether a
        # backend was actually built for it (spec §7 footer copy, fix I4).
        self._live_labels_requested = False
        self._live_labels_built = False
        self._level_timer = None
        self._transcribing = False
        # True while a user-initiated Stop is in flight: `owner.stop()`
        # synchronously emits a "state","stopped" event to listeners (from
        # inside `MeetingSession.stop()`) before it even returns to
        # `_stop_worker` -- without this, the "stopped" state event's own
        # finalisation path in `_apply_event` would race `_stop_worker`'s
        # `call_from_thread(self._on_stopped, result)`, finalising twice.
        self._stop_requested = False
        # Starts True, not False: a `Select(..., value=X, allow_blank=False)`
        # only stores X on the private `_value` in `__init__` -- its FIRST
        # real `Select.Changed` fires from `Select._on_mount`'s own
        # `_init_selected_option`, well before the prepare worker's first
        # `call_from_thread` callback ever runs. Starting the guard closed
        # covers that unavoidable self-echo too; `_apply_prepared` clears it
        # once the first prepare cycle has settled.
        self._syncing_pickers = True
        self.rendered_lines: list[str] = []
        # seq -> index into `rendered_lines`, so a re-delivered segment (its
        # near-live speaker-id refinement, or the Stop pass's reconciled id)
        # UPDATES its transcript line in place instead of appending a second
        # one (final whole-branch review I1). Reset on Start.
        self._line_index_by_seq: dict[int, int] = {}
        # Cluster ids (segment.speaker_id) seen so far this meeting, each
        # backing one row in the speaker legend (task 7). Reset on Start.
        self._seen_speakers: set[str] = set()
        # Set once per session the first time the tap reports "lost" (spec
        # §7); reset on Start so a NEW session's tap gets its own chance to
        # show the indicator rather than being permanently suppressed by a
        # previous session's loss.
        self._lost_shown = False
        # ---- self voiceprint (TASK-31826) ----
        # The store is built lazily, OFF the UI thread (the prepare worker),
        # and never imported at module scope: `Audio.voiceprint` pulls the
        # keyring backend, which boot must not (Task 6's import invariant).
        self._store: Any | None = None
        self._store_unavailable = False
        #: The store's named exception classes, resolved WITH the store and
        #: keyed by `VOICE_TRANSFER_FAILURE_COPY`'s keys (never re-imported
        #: while handling a failure, review M9).
        self._store_errors: dict = {}
        self._offer: Any | None = None
        self._enroll_cancel: threading.Event | None = None
        self._enroll_timer = None
        self._enroll_seconds_left = 0
        self._delete_armed = False

    # ---- compose ----------------------------------------------------------
    def compose_content(self) -> ComposeResult:
        with Vertical(id="meetings-shell"):
            yield Static(
                "Meetings | Record a call or a room | Live transcript | Library handoff",
                id="meetings-title",
                classes="ds-destination-header",
            )
            yield Static(
                "Record a call or a room and get a live transcript into the Library.",
                classes="destination-purpose",
            )
            with Horizontal(id="meetings-workbench", classes="ds-panel destination-workbench"):
                # VerticalScroll, not Vertical (review C2): the rail's content
                # is taller than a 30-row terminal's viewport, and a plain
                # Vertical clips the overflow out of the compositor with no
                # scrollbar -- the Voice row was unreachable by any input.
                with VerticalScroll(id="meetings-rail", classes="destination-workbench-pane"):
                    yield Static("Sources", classes="destination-section")
                    yield Select([("System default", "default")], value="default", id="meetings-mic-select", allow_blank=False)
                    yield Select([("Native (auto)", "auto")], value="auto", id="meetings-system-select", allow_blank=False)
                    yield Static("System audio: probing…", id="meetings-system-status")
                    yield Static("Transcriber: probing…", id="meetings-provider-status")
                    yield Static("Speaker labels after the meeting: probing…", id="meetings-diarization-status")
                    yield Static("Live speaker labels: probing…", id="meetings-live-diarization-status")
                    yield Static("Voice match: probing…", id="meetings-voice-match-status")
                    yield Static("Recording other people may require their consent.", id="meetings-consent", classes="destination-note")
                    with Horizontal(id="meetings-controls"):
                        yield Button(
                            "Start", id="meetings-start", variant="success", disabled=True,
                            tooltip="Start recording this meeting and begin live transcription.",
                        )
                        yield Button(
                            "Pause", id="meetings-pause", disabled=True,
                            tooltip="Pause recording; press again to resume.",
                        )
                        yield Button(
                            "Stop", id="meetings-stop", variant="error", disabled=True,
                            tooltip="Stop recording and finalize the transcript.",
                        )
                    yield Static("00:00:00", id="meetings-timer")
                    yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="meetings-level-mic")
                    yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="meetings-level-sys")
                    yield Static("", id="meetings-recovery")
                    yield Button(
                        "Recover", id="meetings-recover", disabled=True,
                        tooltip="Recover the unfinished meeting recording found in this folder.",
                    )
                    # ---- learning offer (post-Stop, never modal) -----------
                    # Every container below carries `meetings-rail-block` /
                    # `meetings-rail-actions` (height: auto) and every Button
                    # `meetings-rail-button` (width: auto; min-width: 0):
                    # Textual's defaults are `height: 1fr` and `min-width: 16`,
                    # which squeezed these rows to zero rows and clipped the
                    # fourth button off a half-width rail (review C1/C2/I4).
                    with Vertical(id="meetings-learn-offer", classes="meetings-rail-block"):
                        yield Static("", id="meetings-learn-offer-copy", markup=False)
                        with Horizontal(id="meetings-learn-offer-actions", classes="meetings-rail-actions"):
                            yield Button(
                                "Accept", id="meetings-learn-accept", variant="success",
                                classes="meetings-rail-button",
                                tooltip="Add this meeting's sample to your stored voiceprint.",
                            )
                            yield Button(
                                "Not now", id="meetings-learn-decline",
                                classes="meetings-rail-button",
                                tooltip="Keep nothing from this meeting; ask again next time.",
                            )
                        # Second row: the three labels total 34 columns, one
                        # more than the rail's 33 at an 80-column terminal.
                        with Horizontal(id="meetings-learn-offer-never-row", classes="meetings-rail-actions"):
                            yield Button(
                                "Don't ask again", id="meetings-learn-never",
                                classes="meetings-rail-button",
                                tooltip="Keep nothing and stop offering to learn your voice.",
                            )
                    # ---- your voice ----------------------------------------
                    yield Static("Your voice", classes="destination-section")
                    yield Static("Voice: checking…", id="meetings-voice-status")
                    # Two rows, not four buttons on one: even sized to their
                    # labels the four need 43 columns, and the rail is half a
                    # workbench (40 at an 80-column terminal).
                    with Horizontal(id="meetings-voice-actions", classes="meetings-rail-actions"):
                        yield Button(
                            "Enroll my voice", id="meetings-enroll",
                            classes="meetings-rail-button",
                            tooltip="Record about 30 seconds of your voice so meetings can label you.",
                        )
                    with Horizontal(id="meetings-voice-store-actions", classes="meetings-rail-actions"):
                        yield Button(
                            "Delete", id="meetings-voice-delete",
                            classes="meetings-rail-button",
                            tooltip="Delete the stored voiceprint from this device.",
                        )
                        yield Button(
                            "Export…", id="meetings-voice-export",
                            classes="meetings-rail-button",
                            tooltip="Save an encrypted copy of your voiceprint to a file.",
                        )
                        yield Button(
                            "Import…", id="meetings-voice-import",
                            classes="meetings-rail-button",
                            tooltip="Load a voiceprint from a file you exported.",
                        )
                    yield Static("", id="meetings-voice-message", markup=False)
                    with Horizontal(id="meetings-enroll-progress-row", classes="meetings-rail-actions"):
                        yield Static("", id="meetings-enroll-progress")
                        yield Button(
                            "Cancel", id="meetings-enroll-cancel",
                            classes="meetings-rail-button",
                            tooltip="Stop this recording and keep nothing.",
                        )
                    with Vertical(id="meetings-voice-form", classes="meetings-rail-block"):
                        yield Input(placeholder="Passphrase", password=True, id="meetings-voice-passphrase")
                        yield Input(placeholder="File path", id="meetings-voice-path")
                        with Horizontal(id="meetings-voice-form-actions", classes="meetings-rail-actions"):
                            yield Button(
                                "Export to file", id="meetings-voice-export-run",
                                classes="meetings-rail-button",
                                tooltip="Write your voiceprint to this file, encrypted with this passphrase.",
                            )
                            yield Button(
                                "Merge", id="meetings-voice-merge",
                                classes="meetings-rail-button",
                                tooltip="Blend the file's voiceprint into the one stored here.",
                            )
                            yield Button(
                                "Replace", id="meetings-voice-replace",
                                classes="meetings-rail-button",
                                tooltip="Discard the stored voiceprint and keep the file's instead.",
                            )
                with Vertical(id="meetings-canvas", classes="destination-workbench-pane"):
                    yield Static("Speakers", id="meetings-speaker-legend-title", classes="destination-section")
                    yield Vertical(id="meetings-speaker-legend")
                    yield RichLog(id="meetings-transcript", wrap=True, highlight=False, markup=False)
                    # markup=False: transcripts carry Whisper's own bracket
                    # tokens ("[BLANK_AUDIO]", "[Music]"), and folder paths
                    # reach the footer -- Rich markup would swallow them or
                    # raise on an unclosed tag.
                    yield Static("", id="meetings-partial", markup=False)
                    yield Static("", id="meetings-footer", markup=False)
                    yield Button(
                        "Open in Library", id="meetings-open-library", disabled=True,
                        tooltip="Open Library's Import rail with this meeting's recording queued.",
                    )

    # ---- lifecycle --------------------------------------------------------
    def on_mount(self) -> None:
        for widget_id in ("#meetings-learn-offer", "#meetings-enroll-progress-row",
                          "#meetings-voice-form"):
            self.query_one(widget_id).display = False
        self._attach_if_running()
        # An offer the owner is still holding (this screen answered nothing
        # before it was remounted) is shown again rather than left orphaned.
        self._show_learning_offer(getattr(self._owner, "pending_offer", None))
        self._run_prepare()
        self._level_timer = self.set_interval(0.2, self._tick)

    def on_unmount(self) -> None:
        self._detach()
        if self._level_timer is not None:
            self._level_timer.stop()
        self._stop_countdown()
        if self._enroll_cancel is not None:
            # A recording that outlives its countdown and Cancel button has
            # no way to be stopped: end it with the screen.
            self._enroll_cancel.set()
        owner = self._owner
        # `owner.pending_offer`, not `self._offer` (review M8): an offer that
        # became pending after this screen was unmounted still holds a warm
        # worker, and nothing else will lapse it until the next meeting.
        if owner is not None and getattr(owner, "pending_offer", None) is not None:
            # The offer lapses with the screen (spec §3.4). A plain daemon
            # thread, not a Textual worker: a worker started on an unmounting
            # node can be cancelled before it runs, and `dismiss_learning`
            # closes a subprocess, which takes seconds (review I1).
            self._offer = None

            def _dismiss() -> None:
                # Runs on the daemon thread: an exception here would otherwise
                # reach `threading.excepthook` and paint a traceback (which can
                # carry a path) over the TUI (re-review N1).
                try:
                    owner.dismiss_learning()
                except Exception as exc:  # noqa: BLE001 - teardown must not raise
                    logger.debug("meetings offer dismiss: {}", type(exc).__name__)

            try:
                threading.Thread(target=_dismiss, name="meetings-dismiss", daemon=True).start()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                logger.debug("meetings offer dismiss: {}", type(exc).__name__)
        # No super().on_unmount(): the dispatcher already invokes
        # BaseAppScreen.on_unmount separately for this Unmount event (TASK-31418).

    def _attach_if_running(self) -> None:
        owner = self._owner
        if owner is None or not owner.is_active or owner.session is None:
            return
        self._session = owner.session
        prepared = getattr(owner, "prepared", None)
        self._live_labels_requested = bool(getattr(prepared, "live_diarization_active", False))
        self._live_labels_built = getattr(self._session, "_diarizer", None) is not None
        self._session.subscribe(self._on_session_event)
        for segment in list(self._session.segments):
            self._render_segment(segment)
        self._set_buttons(self._session.state)

    def _detach(self) -> None:
        if self._session is not None:
            try:
                self._session.unsubscribe(self._on_session_event)
            except Exception as exc:  # noqa: BLE001
                logger.debug("meetings detach: {}", exc)
            self._session = None

    # ---- prepare (worker) -------------------------------------------------
    @work(exclusive=True, group="meetings-prepare", thread=True, exit_on_error=False)
    def _run_prepare(self) -> None:
        # Building the store imports the keyring backend, so it happens here,
        # on the prepare thread, not on the first paint of the Voice row. The
        # result crosses to the UI thread rather than being assigned from
        # here (review M1).
        self.app.call_from_thread(self._adopt_store, *self._build_store())
        if self._owner is None:
            self.app.call_from_thread(self._show_prepare_error, "Meetings are unavailable in this build.")
            return
        try:
            prepared = self._owner.prepare()
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._show_prepare_error, str(exc))
            return
        self.app.call_from_thread(self._apply_prepared, prepared)

    def _show_prepare_error(self, reason: str) -> None:
        if not self.is_mounted:
            return
        self.query_one("#meetings-provider-status", Static).update(f"Transcriber: {reason}")
        self._refresh_voice_row()

    def _live_diarization_copy(self, prepared: PrepareResult) -> str:
        """Rail copy for `PrepareResult.live_diarization_active` (fix I4).

        The flag was computed and read by nobody, so a user who turned live
        diarization on had no way to learn it would not run.

        Args:
            prepared: The owner's probe result.

        Returns:
            "on", or "off (<reason>)" with a static, non-identifying reason.
        """
        if getattr(prepared, "live_diarization_active", False):
            return "on"
        settings = getattr(self._owner, "settings", None)
        if not getattr(settings, "live_diarization", False):
            return "off (not enabled in settings)"
        if prepared.diarization_missing:
            return f"off ({', '.join(prepared.diarization_missing)} missing)"
        backend = getattr(settings, "diarizer_backend", "")
        return f"off (unsupported backend: {backend})" if backend else "off"

    def _apply_prepared(self, prepared: PrepareResult) -> None:
        if not self.is_mounted:
            return
        mode = prepared.tap_mode
        system_copy = mode.reason if mode.kind != "unavailable" else f"Unavailable, mic only ({mode.reason})"
        self.query_one("#meetings-system-status", Static).update(f"System audio: {system_copy}")
        # A missing recorder (no numpy, no audio backend) is reported where
        # the provider goes, and Start stays disabled below: offering a
        # Start that can only fail is worse than saying why (review C1).
        provider_copy = (
            prepared.capture_error
            or f"{prepared.provider} {prepared.model}".rstrip() + " (finalises per segment)"
        )
        self.query_one("#meetings-provider-status", Static).update(f"Transcriber: {provider_copy}")
        if prepared.diarization_available:
            diar = "Speaker labels after the meeting: on"
        else:
            diar = f"Speaker labels after the meeting: off ({', '.join(prepared.diarization_missing)} missing)"
        self.query_one("#meetings-diarization-status", Static).update(diar)
        self.query_one("#meetings-live-diarization-status", Static).update(
            f"Live speaker labels: {self._live_diarization_copy(prepared)}"
        )
        # Provisional: `prepare()` only stats the store, so "on" here means
        # "a voiceprint exists and will be verified at Start" (Task 4).
        self._render_voice_match(getattr(prepared, "voice_match", None))
        self._refresh_voice_row()
        devices = list(prepared.input_devices)
        settings = getattr(self._owner, "settings", None)
        self._syncing_pickers = True
        try:
            mic = self.query_one("#meetings-mic-select", Select)
            mic.set_options([("System default", "default")] + [(d, d) for d in devices])
            mic_value = getattr(settings, "mic_device", "") or "default"
            mic.value = mic_value if mic_value in ({"default"} | set(devices)) else "default"
            system = self.query_one("#meetings-system-select", Select)
            system.set_options([("Native (auto)", "auto")] + [(d, d) for d in devices])
            system_value = getattr(settings, "system_source", "auto") or "auto"
            system.value = system_value if system_value in ({"auto"} | set(devices)) else "auto"
        finally:
            # `Select.Changed` is posted to the widget's own mailbox and
            # bubbles up asynchronously, so it has not been dispatched to
            # the handlers below by the time this synchronous method
            # returns -- resetting the flag here would let our OWN synced
            # value echo straight into `apply_device_choice`. Deferring the
            # reset past the current refresh lets those echoes drain first;
            # a real user selection always arrives well after this (the
            # picker has to render before anyone can click it).
            self.call_after_refresh(self._clear_syncing_pickers)
        recovery = self.query_one("#meetings-recovery", Static)
        recover = self.query_one("#meetings-recover", Button)
        if prepared.recoverable:
            recovery.update("Unfinished meeting found: " + ", ".join(p.name for p in prepared.recoverable))
            recover.disabled = False
        else:
            recovery.update("")
            recover.disabled = True
        if not prepared.capture_error and not (self._owner is not None and self._owner.is_active):
            self.query_one("#meetings-start", Button).disabled = self._enrolling()

    # ---- device pickers ---------------------------------------------------
    def _clear_syncing_pickers(self) -> None:
        self._syncing_pickers = False

    @on(Select.Changed, "#meetings-mic-select")
    def _mic_changed(self, event: Select.Changed) -> None:
        if self._syncing_pickers:
            return
        if self._owner is not None and event.value not in (None, Select.BLANK):
            self._owner.apply_device_choice("mic", str(event.value))

    @on(Select.Changed, "#meetings-system-select")
    def _system_changed(self, event: Select.Changed) -> None:
        if self._syncing_pickers:
            return
        if self._owner is not None and event.value not in (None, Select.BLANK):
            self._owner.apply_device_choice("system", str(event.value))
            self._run_prepare()

    # ---- start / pause / stop ---------------------------------------------
    @on(Button.Pressed, "#meetings-start")
    def _start_pressed(self) -> None:
        self._stop_requested = False
        self._lost_shown = False
        self.query_one("#meetings-start", Button).disabled = True
        self.rendered_lines.clear()
        self._line_index_by_seq.clear()
        self.query_one("#meetings-transcript", RichLog).clear()
        self.query_one("#meetings-footer", Static).update("")
        self.query_one("#meetings-open-library", Button).disabled = True
        self._seen_speakers.clear()
        self.query_one("#meetings-speaker-legend", Vertical).remove_children()
        self._start_worker()

    @work(exclusive=True, group="meetings-start", thread=True)
    def _start_worker(self) -> None:
        try:
            session = self._owner.start()
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._start_failed, str(exc))
            return
        self.app.call_from_thread(self._on_started, session)

    def _start_failed(self, reason: str) -> None:
        if not self.is_mounted:
            return
        self.app_instance.notify(f"Meeting failed to start: {reason}", severity="error")
        self.query_one("#meetings-start", Button).disabled = False

    def _on_started(self, session: Any) -> None:
        # Navigation can unmount Meetings while the threaded start is still
        # running. Subscribing here would leave the dead screen attached to
        # the app-owned session for the rest of the meeting, receiving every
        # event, while the next mount subscribes a second time (Qodo Q14).
        # Nothing is lost by skipping it: the session stays app-owned and
        # `_attach_if_running` replays `session.segments` on the next mount.
        if not self.is_mounted:
            return
        self._session = session
        # Spec §7: a live diarizer that was expected but never got built
        # (spawn failure, missing backend) must be reported at Stop -- the
        # session is detached by then, so record it now (fix I4).
        prepared = getattr(self._owner, "prepared", None)
        self._live_labels_requested = bool(getattr(prepared, "live_diarization_active", False))
        self._live_labels_built = getattr(session, "_diarizer", None) is not None
        session.subscribe(self._on_session_event)
        # The verified verdict: `start()` performed the one decrypt, so the
        # provisional "on" the rail may be showing is now settled either way.
        self._render_voice_match(getattr(self._owner, "voice_match", None))
        self._set_buttons(session.state)

    @on(Button.Pressed, "#meetings-pause")
    def _pause_pressed(self) -> None:
        session = self._session
        if session is None:
            return
        if session.state == "paused":
            self._owner.resume()
        else:
            self._owner.pause()

    @on(Button.Pressed, "#meetings-stop")
    def _stop_pressed(self) -> None:
        self._stop_requested = True
        self.query_one("#meetings-stop", Button).disabled = True
        self.query_one("#meetings-pause", Button).disabled = True
        self._stop_worker()

    @work(exclusive=True, group="meetings-stop", thread=True, exit_on_error=False)
    def _stop_worker(self) -> None:
        try:
            result = self._owner.stop(reason="user")
        except Exception as exc:  # noqa: BLE001 - the screen must not stay wedged
            # Without this, a raising stop() (e.g. write_meeting_json onto a
            # read-only recordings dir) killed the worker: `_on_stopped`
            # never ran, `_stop_requested` stayed True so the "stopped"
            # state event's own finalisation was suppressed too, and all
            # three buttons stayed disabled with no way back (review I2).
            self.app.call_from_thread(self._stop_failed, str(exc))
            return
        # The offer is decided HERE, on the stop thread (review I2): deciding
        # it can read the voiceprint store, which is a bounded but real wait
        # that must not land on the event loop.
        self.app.call_from_thread(self._on_stopped, result, self._offer_for(result))

    def _offer_for(self, result: MeetingResult | None) -> Any | None:
        """The learning offer for a finished meeting. Never called on the UI thread.

        Args:
            result: The finished meeting, or None when there was none.

        Returns:
            The owner's `LearningOffer`, or None (including on failure — an
            offer must never break a Stop).
        """
        owner = self._owner
        if owner is None or result is None:
            return None
        try:
            return owner.learning_offer(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("meetings: learning offer failed ({})", type(exc).__name__)
            return None

    def _stop_failed(self, reason: str) -> None:
        self._stop_requested = False
        if not self.is_mounted:
            return
        self._detach()
        self._set_buttons("stopped")
        self.app_instance.notify(f"Meeting failed to stop cleanly: {reason}", severity="error")

    def _speaker_labels_failure(self, result: MeetingResult) -> str | None:
        """Why live speaker labels did not happen, for the footer (spec §7).

        Args:
            result: The finished meeting's result.

        Returns:
            A static reason ("backend unavailable", "backend crashed"), or
            None when live labels were never requested or ran fine.
        """
        reason = getattr(result, "speaker_labels_reason", None)
        if reason:
            return str(reason)
        if getattr(self, "_live_labels_requested", False) and not getattr(
            self, "_live_labels_built", False
        ):
            return "backend unavailable"
        return None

    def _on_stopped(self, result: MeetingResult | None, offer: Any | None = None) -> None:
        # `on_unmount` has already detached; the widget updates below would
        # raise on a screen that is no longer composed (Qodo Q14).
        if not self.is_mounted:
            self._stop_requested = False
            return
        self._detach()
        self._set_buttons("stopped")
        self._stop_requested = False
        self._render_voice_match(getattr(self._owner, "voice_match", None))
        if result is None:
            return
        sink = getattr(self._owner, "local_sink", None)
        job_id = getattr(sink, "job_id", None)
        error = getattr(sink, "last_submit_error", None)
        parts = [f"Saved {result.segment_count} segments, {format_clock(result.duration_s)}."]
        if not result.transcription_complete:
            parts.append("The last segment was dropped (transcriber did not finish in time).")
        if result.failed_segments:
            parts.append(f"{result.failed_segments} failed segment(s).")
        labels_reason = self._speaker_labels_failure(result)
        if labels_reason:
            parts.append(f"Speaker labels unavailable ({labels_reason}).")
        if result.flagged_speakers:
            # Spec §4: the Stop pass merged two clusters the user named
            # differently; both names were kept and need resolving.
            names = result.meta.speaker_names
            parts.append(
                "Speaker merge to resolve: "
                + ", ".join(names.get(sid, sid) for sid in result.flagged_speakers)
                + "."
            )
        parts.append(f"Folder: {result.meta.folder}.")
        if job_id:
            parts.append(f"Library ingest queued: {job_id}.")
        else:
            parts.append(f"Library: saved locally, not queued ({error or 'no ingest job'}).")
        if result.stop_reason in STOP_REASON_COPY:
            self.app_instance.notify(STOP_REASON_COPY[result.stop_reason], severity="error")
        self.query_one("#meetings-footer", Static).update(" ".join(parts))
        self.query_one("#meetings-open-library", Button).disabled = not bool(job_id)
        self.query_one("#meetings-partial", Static).update("")
        self._show_learning_offer(offer)

    # ---- session events (capture threads -> loop) -------------------------
    def _on_session_event(self, kind: str, payload: Any) -> None:
        if threading.get_ident() == getattr(self.app, "_thread_id", None):
            self._apply_event(kind, payload)
            return
        try:
            self.app.call_from_thread(self._apply_event, kind, payload)
        except Exception as exc:  # noqa: BLE001 - screen may be tearing down
            logger.debug("meetings event dropped: {}", exc)

    def _apply_event(self, kind: str, payload: Any) -> None:
        if not self.is_mounted:
            return
        if kind == "segment":
            self._render_segment(payload)
            self._transcribing = False
            self.query_one("#meetings-partial", Static).update("")
        elif kind == "partial":
            text, label = payload
            prefix = f"{self._coarse_label(label)}: " if label else ""
            self.query_one("#meetings-partial", Static).update(f"{prefix}{text}…")
        elif kind == "transcribing":
            self._transcribing = bool(payload)
            partial = self.query_one("#meetings-partial", Static)
            if self._transcribing and not str(getattr(partial.renderable, "plain", partial.renderable)):
                partial.update("transcribing…")
            elif not self._transcribing and str(getattr(partial.renderable, "plain", partial.renderable)) == "transcribing…":
                partial.update("")
        elif kind == "speakers":
            # A rename, or the self-voiceprint match naming a cluster with the
            # user's display name: both change every row for that speaker.
            self._refresh_speaker_labels()
        elif kind == "state":
            self._set_buttons(str(payload))
            if payload == "stopped" and self._session is not None and not self._stop_requested:
                # Ended by the watchdog or shutdown, not by our Stop button
                # (a user-initiated stop is already being finalised by
                # `_stop_worker`'s own `call_from_thread(self._on_stopped,
                # result)` once `owner.stop()` returns -- `session.stop()`
                # emits this same "stopped" event synchronously from
                # INSIDE that call, so without the `_stop_requested` guard
                # this branch would finalise the same stop a second time).
                # `session.stop()` is idempotent and returns the cached
                # result -- never read `owner.last_result` here, it may not
                # be assigned yet.
                session = self._session
                result = session.stop()
                self._on_stopped(result)
                # We are on the event loop here, so the offer is decided on a
                # thread instead of inline (review I2), the same rule the
                # Stop button's own worker follows.
                if result is not None:
                    self._offer_worker(result)

    @work(exclusive=True, group="meetings-offer", thread=True, exit_on_error=False,
          description="meeting learning offer")
    def _offer_worker(self, result: MeetingResult) -> None:
        # `description=` is not decoration: without it Textual builds the
        # worker's description from `repr()` of its arguments, which for a
        # MeetingResult is the meeting folder's full path (review I3).
        offer = self._offer_for(result)
        if offer is not None:
            self.app.call_from_thread(self._show_learning_offer, offer)

    def _render_segment(self, segment: MeetingSegment) -> None:
        self._note_speaker(segment)
        line = self._line_for_segment(segment)
        idx = self._line_index_by_seq.get(segment.seq)
        if idx is None:
            # First time we've seen this seq: append a new line.
            self._line_index_by_seq[segment.seq] = len(self.rendered_lines)
            self.rendered_lines.append(line)
            self.query_one("#meetings-transcript", RichLog).write(line)
        elif self.rendered_lines[idx] != line:
            # A re-delivery whose label changed (coarse "Others: hi" becoming
            # "Speaker 1: hi"): update in place. RichLog can't rewrite one
            # line, so repaint the whole log -- cheap at meeting sizes.
            self.rendered_lines[idx] = line
            self._repaint_log()

    def _repaint_log(self) -> None:
        """Clear the transcript log and rewrite it from `rendered_lines`."""
        if not self.is_mounted:
            return
        log = self.query_one("#meetings-transcript", RichLog)
        log.clear()
        for line in self.rendered_lines:
            log.write(line)

    # ---- speaker legend + rename (task 7) ----------------------------------
    def _user_display_name(self) -> str:
        """The name that stands in for "you" in the transcript and legend.

        The RUNNING meeting's own stamped `meta.user_display_name` wins
        (Qodo Q4). The owner stamps it once at `start()` and every
        after-the-fact render -- `render_markdown`, the Library item's
        re-render, `meeting.json` -- reads it back from there, so re-reading
        configuration per row instead meant a display-name setting changed
        MID-meeting split the live rows away from the saved transcript: rows
        already on screen kept the old name, new ones took the new one, and
        neither matched what the Library would show.

        Only before a session exists (the idle screen's legend/preview) does
        this fall back to `meeting_owner.meeting_user_display_name`, the ONE
        place the decision is made: `chat_defaults.user_display_name`'s own
        factory default is the literal string ``"User"`` (see `config.py`'s
        `DEFAULT_CONFIG_FROM_TOML`), so a fresh install has no way to tell
        "never touched this setting" apart from "chose User" -- honouring it
        unconditionally would silently turn every untouched install's "You:"
        rows into "User:" rows.
        """
        if self._session is not None:
            return self._session.meta.user_display_name
        return meeting_user_display_name()

    def _coarse_label(self, label: str) -> str:
        """Coarse "you"/"others"/"both" label text for the partial preview.

        The finalized transcript/legend get the configured mic name through
        `render_label` (via `_user_display_name()`); this is the same
        channel-name decision for the still-streaming partial line, so "You:
        hel..." and the finalized "Alice: hello" never disagree (task 31746).
        """
        if label == "you":
            return self._user_display_name()
        if label == "both":
            return f"{self._user_display_name()} + Others"
        return LABELS.get(label, label)

    def _matched_marker(self, cluster_id: str | None) -> str:
        """`SELF_MARKER` for the cluster the voiceprint matched, else "".

        Spec §3.4: the marker says "this name was chosen for you". An
        override (a rename to anything but the user's own display name)
        clears `matched_self_overridden`'s False, and with it the marker.

        Args:
            cluster_id: The cluster being rendered; None/"" never matches.

        Returns:
            The marker suffix, or an empty string.
        """
        session = self._session
        if not cluster_id or session is None:
            return ""
        meta = session.meta
        matched = getattr(meta, "matched_self", None) == cluster_id
        return SELF_MARKER if matched and not getattr(meta, "matched_self_overridden", False) else ""

    def _line_for_segment(self, segment: MeetingSegment) -> str:
        stamp = f"[{format_clock(segment.t_audio_start)}]"
        names = self._session.meta.speaker_names if self._session is not None else {}
        diarize_mic = self._session.meta.diarize_mic_channel if self._session is not None else False
        label = render_label(segment, names, self._user_display_name(), diarize_mic=diarize_mic)
        if not label:
            return f"{stamp} {segment.text}"
        return f"{stamp} {label}{self._matched_marker(segment.speaker_id)}: {segment.text}"

    def _speaker_label(self, cluster_id: str) -> str:
        """The legend row's current display name for `cluster_id`."""
        names = self._session.meta.speaker_names if self._session is not None else {}
        diarize_mic = self._session.meta.diarize_mic_channel if self._session is not None else False
        placeholder = MeetingSegment(0, 0.0, 0.0, 0.0, 0.0, "others", "", speaker_id=cluster_id)
        label = render_label(placeholder, names, self._user_display_name(), diarize_mic=diarize_mic)
        return (label or cluster_id) + self._matched_marker(cluster_id)

    def _refresh_speaker_labels(self) -> None:
        """Repaint every legend label and the transcript from the name map."""
        if not self.is_mounted:
            return
        for cluster_id in self._seen_speakers:
            try:
                self.query_one(f"#speaker-label-{cluster_id}", Static).update(
                    self._speaker_label(cluster_id)
                )
            except NoMatches:
                continue
        self._rerender_transcript()

    def _note_speaker(self, segment: MeetingSegment) -> None:
        """Track a newly-seen `speaker_id`, mounting its legend row once."""
        cluster_id = segment.speaker_id
        if not cluster_id or cluster_id in self._seen_speakers:
            return
        # An id that is not a legal Textual widget id would raise out of the
        # mount below and take the screen down (final review, MINOR).
        if not is_widget_safe_cluster_id(cluster_id):
            return
        self._seen_speakers.add(cluster_id)
        if not self.is_mounted:
            return
        row = Horizontal(
            # markup=False: a typed name is arbitrary text, and Rich would
            # either swallow "[b]" or raise on "[/]" (Qodo Q2). The Library
            # twin's legend already does this.
            Static(self._speaker_label(cluster_id), id=f"speaker-label-{cluster_id}", markup=False),
            Input(placeholder="Rename…", id=f"speaker-input-{cluster_id}"),
            classes="meetings-speaker-row",
        )
        self.query_one("#meetings-speaker-legend", Vertical).mount(row)

    @on(Input.Submitted, "#meetings-speaker-legend Input")
    def _speaker_rename_submitted(self, event: Input.Submitted) -> None:
        prefix = "speaker-input-"
        widget_id = event.input.id or ""
        if not widget_id.startswith(prefix):
            return
        cluster_id = widget_id[len(prefix):]
        self._apply_rename(cluster_id, event.value)
        event.input.value = ""

    def _apply_rename(self, cluster_id: str, name: str) -> None:
        """Rename `cluster_id` to `name` (blank removes it from the map).

        The session owns the sequence (TASK-31826): normalise, override the
        self-voiceprint match when the matched cluster is renamed, pin the
        cluster with the diarizer when it offers `pin`, persist to
        `meeting.json` and re-emit. The state work runs unconditionally so a
        rename racing screen teardown still lands and persists; only the
        widget refresh is `is_mounted`-guarded (phase-1 rule).
        """
        session = self._session
        if session is None:
            return
        session.rename_speaker(cluster_id, name)
        self._rerender_transcript()
        if not self.is_mounted:
            return
        try:
            label_widget = self.query_one(f"#speaker-label-{cluster_id}", Static)
        except NoMatches:
            return
        label_widget.update(self._speaker_label(cluster_id))

    def _rerender_transcript(self) -> None:
        """Recompute `rendered_lines` from `self._session.segments` and, when
        mounted, rewrite the transcript log to match (a rename can change
        every line naming that speaker, not just the newest one)."""
        session = self._session
        segments = list(session.segments) if session else []
        self.rendered_lines = [self._line_for_segment(seg) for seg in segments]
        self._line_index_by_seq = {seg.seq: i for i, seg in enumerate(segments)}
        self._repaint_log()

    def _rendered_transcript_text(self) -> str:
        """Test hook: the transcript as currently rendered, one line each."""
        return "\n".join(self.rendered_lines)

    def _set_buttons(self, state: str) -> None:
        active = state in ("starting", "recording", "paused", "stopping")
        controls_active = state in ("starting", "recording", "paused")
        # Start is refused outright while an enrollment holds the microphone
        # (`owner.start()` raises "enrolling: ..."), so it is not offered.
        self.query_one("#meetings-start", Button).disabled = active or self._enrolling()
        self.query_one("#meetings-stop", Button).disabled = not controls_active
        pause = self.query_one("#meetings-pause", Button)
        pause.disabled = not controls_active
        pause.label = "Resume" if state == "paused" else "Pause"
        # ... and the reverse: enrolling needs the mic a meeting is holding.
        self.query_one("#meetings-enroll", Button).disabled = active or self._enrolling()

    def _sync_controls(self) -> None:
        """Re-apply the button rules for the current session state."""
        if not self.is_mounted:
            return
        self._set_buttons(self._session.state if self._session is not None else "stopped")

    def _tick(self) -> None:
        session = self._session
        if session is None or not self.is_mounted:
            return
        try:
            self.query_one("#meetings-timer", Static).update(format_clock(float(session.capture.audio_position_s)))
            mic, sys_ = session.capture.levels()
            self.query_one("#meetings-level-mic", ProgressBar).progress = int(mic * 100)
            self.query_one("#meetings-level-sys", ProgressBar).progress = int(sys_ * 100)
            if not self._lost_shown and getattr(session.capture, "system_source_state", None) == "lost":
                self._lost_shown = True
                self.query_one("#meetings-system-status", Static).update(
                    "System audio: System source lost — continuing from the microphone"
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("meetings tick: {}", exc)

    # ---- self voiceprint: rail line, offer, enrollment, Voice row ---------
    def _voice_match_copy(self, state: Any | None) -> str:
        """Rail copy for a `VoiceMatchState` (spec §3.5).

        Args:
            state: The owner's verdict, or None when there is none yet.

        Returns:
            "Voice match: on", or "Voice match: off (<static reason>)".
        """
        if getattr(state, "state", "off") == "on":
            return "Voice match: on"
        reason = VOICE_MATCH_OFF_COPY.get(getattr(state, "reason", None) or "")
        return f"Voice match: off ({reason})" if reason else "Voice match: off"

    def _render_voice_match(self, state: Any | None) -> None:
        if not self.is_mounted:
            return
        self.query_one("#meetings-voice-match-status", Static).update(self._voice_match_copy(state))

    def _voice_message(self, copy: str) -> None:
        """The Voice row's one status line. Never carries a passphrase."""
        if not self.is_mounted:
            return
        self.query_one("#meetings-voice-message", Static).update(copy)

    def _build_store(self) -> tuple[Any | None, dict]:
        """Build the store and resolve its exception classes.

        Pure: it returns, it never assigns, so the prepare worker can call it
        off the UI thread and hand the result over (review M1). The classes
        are resolved HERE rather than inside the transfer worker's `except`,
        where a failing import would strand the flow (review M9).

        Imported inside the method on purpose: `Audio.voiceprint` pulls the
        keyring backend, and boot must import neither (Task 6's invariant).

        Returns:
            `(store, {copy key: exception class})`, or `(None, {})`.
        """
        try:
            from ...Audio.voiceprint import (
                ExportRefused, ModelMismatch, StoreUnavailable, WrongPassphrase, default_store,
            )

            return default_store(), {
                "passphrase": WrongPassphrase, "destination": ExportRefused,
                "model": ModelMismatch, "locked": StoreUnavailable,
            }
        except Exception as exc:  # noqa: BLE001 - the screen still works
            logger.warning("meetings: voiceprint store unavailable ({})", type(exc).__name__)
            return None, {}

    def _adopt_store(self, store: Any | None, errors: dict) -> None:
        """Take the store built elsewhere (UI thread only)."""
        if self._store is not None or self._store_unavailable:
            return
        self._store, self._store_errors = store, errors
        self._store_unavailable = store is None
        self._refresh_voice_row()

    def _voice_store(self) -> Any | None:
        """The app's voiceprint store, built once (UI thread).

        Returns:
            The store, or None when one cannot be built at all.
        """
        if self._store is None and not self._store_unavailable:
            self._adopt_store(*self._build_store())
        return self._store

    def _refresh_voice_row(self) -> None:
        """Repaint "Voice: ..." from the store. Never touches the key."""
        if not self.is_mounted:
            return
        store = self._voice_store()
        try:
            if store is None:
                copy = "Voice: unavailable"
            elif not store.exists():
                copy = "Voice: not enrolled"
            else:
                mode = "keyring" if store.mode == "keyring" else "key file"
                copy = f"Voice: enrolled ({mode})"
        except Exception as exc:  # noqa: BLE001 - a rail line, not a feature
            logger.warning("meetings: voiceprint status failed ({})", type(exc).__name__)
            copy = "Voice: unavailable"
        self.query_one("#meetings-voice-status", Static).update(copy)

    # ---- learning offer ----------------------------------------------------
    def _show_learning_offer(self, offer: Any | None) -> None:
        """Show (or hide) the post-Stop offer as a rail prompt, never a modal."""
        self._offer = offer
        if not self.is_mounted:
            return
        block = self.query_one("#meetings-learn-offer")
        if offer is None:
            block.display = False
            return
        copy = (
            "Was it only you on the mic?"
            if getattr(offer, "kind", "") == "mic_channel"
            else "Remember this voice as yours for future meetings?"
        )
        self.query_one("#meetings-learn-offer-copy", Static).update(copy)
        block.display = True

    def _hide_learning_offer(self) -> Any | None:
        """Take the pending offer off screen and return it."""
        offer, self._offer = self._offer, None
        if self.is_mounted:
            self.query_one("#meetings-learn-offer").display = False
        return offer

    @on(Button.Pressed, "#meetings-learn-accept")
    def _learn_accept(self) -> None:
        offer = self._hide_learning_offer()
        if offer is None or self._owner is None:
            return
        self._voice_message("Learning from this meeting…")
        self._accept_learning_worker(offer)

    # `description=` on every worker below: Textual's default description is
    # `repr()` of the arguments, and a LearningOffer carries the meeting
    # folder's path (review I3).
    @work(exclusive=True, group="meetings-learn", thread=True, exit_on_error=False,
          description="voice learning accept")
    def _accept_learning_worker(self, offer: Any) -> None:
        ok = bool(self._owner.accept_learning(offer, progress=self._learn_progress_from_thread))
        self.app.call_from_thread(self._learning_accepted, ok)

    def _learn_progress_from_thread(self, status: str) -> None:
        """Show the owner's warm-up on the Voice row (final review I3).

        Mapped, never interpolated: with no live diarizer, accepting spawns a
        worker and waits out a cold ECAPA download, and the row would
        otherwise read "Learning from this meeting…" for two minutes.
        """
        copy = LEARN_PROGRESS_COPY.get(status)
        if copy is not None:
            self._progress_from_thread(self._voice_message, copy)

    @work(group="meetings-learn-decline", thread=True, exit_on_error=False,
          description="voice learning decline")
    def _decline_learning_worker(self, offer: Any) -> None:
        # Off the UI thread (review I1): `decline_learning` closes the warm
        # diarizer subprocess, which its own docstring bounds at ~12 s.
        self._owner.decline_learning(offer)

    def _learning_accepted(self, ok: bool) -> None:
        if not self.is_mounted:
            return
        self._voice_message(
            "Your voice was updated from this meeting."
            if ok
            else "Could not update your voice from this meeting."
        )
        self._refresh_voice_row()

    @on(Button.Pressed, "#meetings-learn-decline")
    def _learn_decline(self) -> None:
        offer = self._hide_learning_offer()
        if self._owner is not None:
            self._decline_learning_worker(offer)

    @on(Button.Pressed, "#meetings-learn-never")
    def _learn_never(self) -> None:
        offer = self._hide_learning_offer()
        owner = self._owner
        if owner is not None:
            self._decline_learning_worker(offer)
            settings = getattr(owner, "settings", None)
            if settings is not None:
                # The running owner keeps a warm worker for the next offer
                # until this lands, so it is set here and not only on disk.
                settings.voice_learn_offer = False
        try:
            save_setting_to_cli_config("meetings", "voice_learn_offer", False)
        except Exception as exc:  # noqa: BLE001 - the answer still holds
            logger.warning("meetings: saving voice_learn_offer failed ({})", type(exc).__name__)
        self._voice_message("Meetings will not offer to learn your voice again.")

    # ---- enrollment --------------------------------------------------------
    def _enrolling(self) -> bool:
        return self._enroll_cancel is not None or bool(getattr(self._owner, "is_enrolling", False))

    @on(Button.Pressed, "#meetings-enroll")
    def _enroll_pressed(self) -> None:
        owner = self._owner
        if owner is None or self._enroll_cancel is not None:
            return
        if owner.is_active:
            self._voice_message("The microphone is busy — stop the meeting first.")
            return
        self._delete_armed = False
        self._enroll_cancel = threading.Event()
        self._sync_controls()
        self.query_one("#meetings-enroll-progress-row").display = True
        self._set_enroll_progress("Warming up…")
        self._voice_message("")
        self._enroll_worker(self._enroll_cancel)

    @work(exclusive=True, group="meetings-enroll", thread=True, exit_on_error=False)
    def _enroll_worker(self, cancel: threading.Event) -> None:
        try:
            result = self._owner.enroll_from_mic(
                ENROLL_SECONDS, progress=self._enroll_progress_from_thread, cancel=cancel
            )
        except Exception as exc:  # noqa: BLE001 - report, never crash the screen
            self.app.call_from_thread(self._enroll_finished, None, type(exc).__name__)
            return
        self.app.call_from_thread(self._enroll_finished, result, None)

    def _enroll_progress_from_thread(self, status: str) -> None:
        self._progress_from_thread(self._enroll_progress, status)

    def _progress_from_thread(self, render: Any, copy: str) -> None:
        """Marshal one progress line onto the UI thread, best-effort."""
        try:
            self.app.call_from_thread(render, copy)
        except Exception as exc:  # noqa: BLE001 - screen may be tearing down
            logger.debug("meetings enrollment progress dropped: {}", type(exc).__name__)

    def _enroll_progress(self, status: str) -> None:
        """Show the owner's static progress word; count down while recording."""
        if not self.is_mounted:
            return
        if status == "recording":
            self._enroll_seconds_left = int(ENROLL_SECONDS)
            self._tick_countdown()
            if self._enroll_timer is None:
                self._enroll_timer = self.set_interval(1.0, self._tick_countdown)
            return
        self._stop_countdown()
        self._set_enroll_progress(f"{status.capitalize()}…")

    def _tick_countdown(self) -> None:
        if not self.is_mounted:
            return
        self._set_enroll_progress(f"Recording… {max(self._enroll_seconds_left, 0)}s left")
        self._enroll_seconds_left -= 1

    def _stop_countdown(self) -> None:
        if self._enroll_timer is not None:
            self._enroll_timer.stop()
            self._enroll_timer = None

    def _set_enroll_progress(self, copy: str) -> None:
        if not self.is_mounted:
            return
        self.query_one("#meetings-enroll-progress", Static).update(copy)

    @on(Button.Pressed, "#meetings-enroll-cancel")
    def _enroll_cancel_pressed(self) -> None:
        if self._enroll_cancel is None:
            return
        self._enroll_cancel.set()
        self._stop_countdown()
        self._set_enroll_progress("Cancelling…")

    def _enroll_finished(self, result: Any | None, error: str | None) -> None:
        self._enroll_cancel = None
        self._stop_countdown()
        if result is not None and result.ok and self._owner is not None:
            # A new voiceprint invalidates the owner's cached verdict, so the
            # next meeting reads the one just saved (Task 4).
            self._owner.invalidate_voiceprint()
        if not self.is_mounted:
            return
        self.query_one("#meetings-enroll-progress-row").display = False
        self._set_enroll_progress("")
        if error is not None:
            self._voice_message(f"Enrollment failed ({error}).")
        elif result.ok:
            self._voice_message(f"Your voice was enrolled from {result.seconds:.0f}s of audio.")
        else:
            reason = ENROLL_REASON_COPY.get(result.reason or "", result.reason or "unknown")
            self._voice_message(f"Enrollment failed: {reason}.")
        self._refresh_voice_row()
        self._render_voice_match(getattr(self._owner, "voice_match", None))
        self._sync_controls()

    # ---- Voice row: Delete / Export / Import -------------------------------
    @on(Button.Pressed, "#meetings-voice-delete")
    def _voice_delete_pressed(self) -> None:
        if not self._delete_armed:
            # The confirm step: a stored voiceprint cannot be recovered.
            self._delete_armed = True
            self._voice_message("Press Delete again to remove your stored voiceprint.")
            return
        self._delete_armed = False
        store = self._voice_store()
        if store is None:
            self._voice_message("The voiceprint store is unavailable.")
            return
        try:
            removed = store.delete()
        except Exception as exc:  # noqa: BLE001
            self._voice_message(f"Delete failed ({type(exc).__name__}).")
            return
        if self._owner is not None:
            self._owner.invalidate_voiceprint()
        self._voice_message("Your voiceprint was deleted." if removed else "No voiceprint to delete.")
        self._refresh_voice_row()
        self._render_voice_match(getattr(self._owner, "voice_match", None))

    @on(Button.Pressed, "#meetings-voice-export")
    def _voice_export_pressed(self) -> None:
        self._open_voice_form("export")

    @on(Button.Pressed, "#meetings-voice-import")
    def _voice_import_pressed(self) -> None:
        self._open_voice_form("import")

    def _open_voice_form(self, mode: str) -> None:
        self._delete_armed = False
        if not self.is_mounted:
            return
        self._clear_passphrase()
        self.query_one("#meetings-voice-form").display = True
        self.query_one("#meetings-voice-export-run", Button).display = mode == "export"
        for widget_id in ("#meetings-voice-merge", "#meetings-voice-replace"):
            self.query_one(widget_id, Button).display = mode == "import"
        self._voice_message(
            "Type a passphrase and a destination file, then Export to file."
            if mode == "export"
            else "Type the file's passphrase and path, then Merge or Replace."
        )

    def _clear_passphrase(self) -> None:
        """Empty the passphrase Input (review M6: it must not linger)."""
        if self.is_mounted:
            self.query_one("#meetings-voice-passphrase", Input).value = ""

    def _voice_form_values(self, action: str) -> tuple[Path, str] | None:
        """The typed path and passphrase, refusing an empty either way.

        Args:
            action: "Export" or "Import", for the refusal copy.

        Returns:
            `(path, passphrase)`, or None when something was missing (the
            refusal is already on screen and the passphrase field is empty).
        """
        passphrase = self.query_one("#meetings-voice-passphrase", Input).value
        text = self.query_one("#meetings-voice-path", Input).value.strip()
        if not passphrase:
            self._voice_message(f"{action} needs a passphrase.")
            return None
        if not text:
            self._clear_passphrase()
            self._voice_message(f"{action} needs a file path.")
            return None
        return Path(text).expanduser(), passphrase

    @on(Button.Pressed, "#meetings-voice-export-run")
    def _voice_export_run(self) -> None:
        values = self._voice_form_values("Export")
        if values is None:
            return
        if self._voice_store() is None:
            self._voice_message("The voiceprint store is unavailable.")
            return
        self._voice_message("Exporting…")
        self._voice_transfer_worker("export", values[0], values[1], False)

    @on(Button.Pressed, "#meetings-voice-merge")
    def _voice_import_merge(self) -> None:
        self._start_import(replace=False)

    @on(Button.Pressed, "#meetings-voice-replace")
    def _voice_import_replace(self) -> None:
        self._start_import(replace=True)

    def _start_import(self, *, replace: bool) -> None:
        values = self._voice_form_values("Import")
        if values is None:
            return
        if self._voice_store() is None:
            self._voice_message("The voiceprint store is unavailable.")
            return
        self._voice_message("Importing…")
        self._voice_transfer_worker("import", values[0], values[1], replace)

    @work(exclusive=True, group="meetings-voice-transfer", thread=True, exit_on_error=False,
          description="voiceprint transfer")
    def _voice_transfer_worker(self, action: str, path: Path, passphrase: str, replace: bool) -> None:
        # `description=` is required, not cosmetic (review I3): Textual's
        # default builds the description from `repr()` of the arguments, and
        # `Worker.__rich_repr__` yields it to `app.log.worker` on every state
        # change -- which would put the typed passphrase and the full
        # destination path in the Textual log.
        store = self._store
        try:
            if action == "export":
                store.export(path, passphrase)
                copy = "Your voiceprint was exported."
            else:
                store.import_(path, passphrase, replace=replace)
                copy = "The voiceprint was imported."
            ok = True
        except Exception as exc:  # noqa: BLE001 - reported as static copy
            ok, copy = False, self._transfer_failure_copy(action, exc)
        self.app.call_from_thread(self._voice_transfer_done, action, ok, copy)

    def _transfer_failure_copy(self, action: str, exc: Exception) -> str:
        """Static copy for a failed export/import -- never the path or message.

        The named classes were resolved with the store (review M9), so this
        path never imports while handling a failure. Most specific first:
        `WrongPassphrase` and `ExportRefused` are both `ValueError`s, so only
        their order relative to each other would ever matter.
        """
        for key, copy in VOICE_TRANSFER_FAILURE_COPY:
            error = self._store_errors.get(key)
            if error is not None and isinstance(exc, error):
                return copy
        return f"{action.capitalize()} failed ({type(exc).__name__})."

    def _voice_transfer_done(self, action: str, ok: bool, copy: str) -> None:
        if ok and action == "import" and self._owner is not None:
            self._owner.invalidate_voiceprint()
        if not self.is_mounted:
            return
        self._clear_passphrase()
        self._voice_message(copy)
        if not ok:
            return
        self.query_one("#meetings-voice-form").display = False
        self._refresh_voice_row()
        self._render_voice_match(getattr(self._owner, "voice_match", None))

    # ---- recovery + Library -----------------------------------------------
    @on(Button.Pressed, "#meetings-recover")
    def _recover_pressed(self) -> None:
        prepared = getattr(self._owner, "prepared", None)
        folders = tuple(getattr(prepared, "recoverable", ()) or ())
        if not folders:
            return
        self.query_one("#meetings-recover", Button).disabled = True
        self._recover_worker(folders[0])

    @work(exclusive=True, group="meetings-recover", thread=True, description="meetings recover")
    def _recover_worker(self, folder: Path) -> None:
        # `description=` keeps the folder path out of `Worker.description`
        # (which Textual writes to its log on every state change).
        # Guarded separately from the submit below: a truncated or malformed
        # meeting.json used to raise straight out of the worker, which then
        # died silently with the Recover button left disabled and nothing on
        # screen. Recovery failing and ingest failing are different
        # outcomes, so they get different copy (final whole-branch review).
        try:
            payload = recover_folder(folder)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._recovery_failed, f"Recovery failed: {exc}")
            return
        started = str(payload.get("started_at", ""))[:16].replace("T", " ")
        try:
            job_id = self._owner._submit_on_ui_thread(
                source_path=str(Path(folder) / "mixed.wav"), title=f"Meeting {started} (recovered)",
                keywords=("meeting",), detected_type="audio",
                ingest_options={"diarization": bool(getattr(self._owner.settings, "post_diarize", True))},
            )
            copy = f"Recovered {Path(folder).name}: Library ingest queued: {job_id}."
        except Exception as exc:  # noqa: BLE001
            copy = f"Recovered {Path(folder).name}: saved locally, not queued ({exc})."
        self.app.call_from_thread(self._recovered, copy)

    def _recovered(self, copy: str) -> None:
        if not self.is_mounted:
            return
        self.query_one("#meetings-footer", Static).update(copy)
        self.query_one("#meetings-recovery", Static).update("")

    def _recovery_failed(self, copy: str) -> None:
        """Nothing was recovered: keep the recovery line and offer Recover again."""
        if not self.is_mounted:
            return
        self.query_one("#meetings-footer", Static).update(copy)
        self.query_one("#meetings-recover", Button).disabled = False

    @on(Button.Pressed, "#meetings-open-library")
    def _open_library(self) -> None:
        self.app.post_message(NavigateToScreen(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_INGEST: True}))
