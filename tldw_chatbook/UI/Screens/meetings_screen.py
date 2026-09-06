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
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Input, ProgressBar, RichLog, Select, Static

from ...Audio.meeting_owner import PrepareResult, recover_folder
from ...Audio.meeting_session import (
    MeetingResult,
    MeetingSegment,
    format_clock,
    render_label,
    update_meeting_json,
)
from ...Constants import LIBRARY_NAV_CONTEXT_INGEST, TAB_LIBRARY
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen

# Absolute import (not this file's usual relative style): the diagnostic
# inventory checker's safe-path-transform recognizer
# (`scripts/check_persistent_diagnostic_inventory.py`) only matches
# `redact_user_paths` imported as `tldw_chatbook.Utils.log_sanitizer`, not a
# relative `from ...Utils.log_sanitizer import ...` -- matches
# `meeting_session.py`/`library_media_canvas.py`'s own import of it.
from tldw_chatbook.Utils.log_sanitizer import redact_user_paths

LABELS = {"you": "You", "others": "Others", "both": "You + Others"}
STOP_REASON_COPY = {
    "mic_lost": "Microphone stopped delivering audio; the meeting was ended.",
    "disk_error": "Recording stopped: the disk write failed.",
}


class MeetingsScreen(BaseAppScreen):
    """Record a call (mic + system audio) or a room (mic only)."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "meetings", **kwargs)
        self._owner = getattr(app_instance, "meeting_session_owner", None)
        self._session: Any | None = None
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
        # Cluster ids (segment.speaker_id) seen so far this meeting, each
        # backing one row in the speaker legend (task 7). Reset on Start.
        self._seen_speakers: set[str] = set()
        # Set once per session the first time the tap reports "lost" (spec
        # §7); reset on Start so a NEW session's tap gets its own chance to
        # show the indicator rather than being permanently suppressed by a
        # previous session's loss.
        self._lost_shown = False

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
                with Vertical(id="meetings-rail", classes="destination-workbench-pane"):
                    yield Static("Sources", classes="destination-section")
                    yield Select([("System default", "default")], value="default", id="meetings-mic-select", allow_blank=False)
                    yield Select([("Native (auto)", "auto")], value="auto", id="meetings-system-select", allow_blank=False)
                    yield Static("System audio: probing…", id="meetings-system-status")
                    yield Static("Transcriber: probing…", id="meetings-provider-status")
                    yield Static("Speaker labels after the meeting: probing…", id="meetings-diarization-status")
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
        self._attach_if_running()
        self._run_prepare()
        self._level_timer = self.set_interval(0.2, self._tick)

    def on_unmount(self) -> None:
        self._detach()
        if self._level_timer is not None:
            self._level_timer.stop()
        super().on_unmount()

    def _attach_if_running(self) -> None:
        owner = self._owner
        if owner is None or not owner.is_active or owner.session is None:
            return
        self._session = owner.session
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
    @work(exclusive=True, group="meetings-prepare", thread=True)
    def _run_prepare(self) -> None:
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
            self.query_one("#meetings-start", Button).disabled = False

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
        session.subscribe(self._on_session_event)
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

    @work(exclusive=True, group="meetings-stop", thread=True)
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
        self.app.call_from_thread(self._on_stopped, result)

    def _stop_failed(self, reason: str) -> None:
        self._stop_requested = False
        if not self.is_mounted:
            return
        self._detach()
        self._set_buttons("stopped")
        self.app_instance.notify(f"Meeting failed to stop cleanly: {reason}", severity="error")

    def _on_stopped(self, result: MeetingResult | None) -> None:
        # `on_unmount` has already detached; the widget updates below would
        # raise on a screen that is no longer composed (Qodo Q14).
        if not self.is_mounted:
            self._stop_requested = False
            return
        self._detach()
        self._set_buttons("stopped")
        self._stop_requested = False
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
            prefix = f"{LABELS.get(label, label)}: " if label else ""
            self.query_one("#meetings-partial", Static).update(f"{prefix}{text}…")
        elif kind == "transcribing":
            self._transcribing = bool(payload)
            partial = self.query_one("#meetings-partial", Static)
            if self._transcribing and not str(getattr(partial.renderable, "plain", partial.renderable)):
                partial.update("transcribing…")
            elif not self._transcribing and str(getattr(partial.renderable, "plain", partial.renderable)) == "transcribing…":
                partial.update("")
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
                self._on_stopped(session.stop())

    def _render_segment(self, segment: MeetingSegment) -> None:
        self._note_speaker(segment)
        line = self._line_for_segment(segment)
        self.rendered_lines.append(line)
        self.query_one("#meetings-transcript", RichLog).write(line)

    # ---- speaker legend + rename (task 7) ----------------------------------
    def _user_display_name(self) -> str:
        """The name that stands in for "you" in the transcript and legend.

        Deliberately NOT `chat_defaults.user_display_name`: that section's
        own factory default is the literal string ``"User"`` (see
        `config.py`'s `CONFIG_TOML_CONTENT`), so a fresh install has no way
        to tell "never touched this setting" apart from "chose User" --
        wiring it in here would silently turn every untouched install's
        "You:" rows into "User:" rows. `LABELS["you"]` is Meetings' own,
        already-shipped default; a per-meeting override is future scope.
        """
        return LABELS["you"]

    def _line_for_segment(self, segment: MeetingSegment) -> str:
        stamp = f"[{format_clock(segment.t_audio_start)}]"
        names = self._session.meta.speaker_names if self._session is not None else {}
        label = render_label(segment, names, self._user_display_name())
        return f"{stamp} {label}: {segment.text}" if label else f"{stamp} {segment.text}"

    def _speaker_label(self, cluster_id: str) -> str:
        """The legend row's current display name for `cluster_id`."""
        names = self._session.meta.speaker_names if self._session is not None else {}
        placeholder = MeetingSegment(0, 0.0, 0.0, 0.0, 0.0, "others", "", speaker_id=cluster_id)
        return render_label(placeholder, names, self._user_display_name()) or cluster_id

    def _note_speaker(self, segment: MeetingSegment) -> None:
        """Track a newly-seen `speaker_id`, mounting its legend row once."""
        cluster_id = segment.speaker_id
        if not cluster_id or cluster_id in self._seen_speakers:
            return
        self._seen_speakers.add(cluster_id)
        if not self.is_mounted:
            return
        row = Horizontal(
            Static(self._speaker_label(cluster_id), id=f"speaker-label-{cluster_id}"),
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

        Updates the session's live name map, pins the cluster with the
        diarizer when it offers `pin` (most backends won't -- this is
        optional by design), persists to `meeting.json`, and re-renders the
        transcript. The state work below runs unconditionally so a rename
        racing screen teardown still lands and persists; only the widget
        refresh is `is_mounted`-guarded (phase-1 rule).
        """
        session = self._session
        if session is None:
            return
        name = name.strip()
        if name:
            session.meta.speaker_names[cluster_id] = name
        else:
            session.meta.speaker_names.pop(cluster_id, None)
        diarizer = getattr(session, "_diarizer", None)
        if diarizer is not None and hasattr(diarizer, "pin"):
            diarizer.pin(cluster_id)
        try:
            update_meeting_json(session.meta.folder, speaker_names=dict(session.meta.speaker_names))
        except Exception as exc:  # noqa: BLE001 - a rename must not crash the screen
            # `update_meeting_json` failures are usually filesystem errors,
            # whose `str()` embeds the meeting folder path (task-9 diagnostic
            # inventory review) -- same treatment as `meeting_session.py`'s
            # other file-write failure logs.
            logger.debug("meetings rename persist failed: {}", redact_user_paths(str(exc)))
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
        self.rendered_lines = [self._line_for_segment(seg) for seg in (session.segments if session else [])]
        if not self.is_mounted:
            return
        log = self.query_one("#meetings-transcript", RichLog)
        log.clear()
        for line in self.rendered_lines:
            log.write(line)

    def _rendered_transcript_text(self) -> str:
        """Test hook: the transcript as currently rendered, one line each."""
        return "\n".join(self.rendered_lines)

    def _set_buttons(self, state: str) -> None:
        active = state in ("starting", "recording", "paused", "stopping")
        controls_active = state in ("starting", "recording", "paused")
        self.query_one("#meetings-start", Button).disabled = active
        self.query_one("#meetings-stop", Button).disabled = not controls_active
        pause = self.query_one("#meetings-pause", Button)
        pause.disabled = not controls_active
        pause.label = "Resume" if state == "paused" else "Pause"

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

    # ---- recovery + Library -----------------------------------------------
    @on(Button.Pressed, "#meetings-recover")
    def _recover_pressed(self) -> None:
        prepared = getattr(self._owner, "prepared", None)
        folders = tuple(getattr(prepared, "recoverable", ()) or ())
        if not folders:
            return
        self.query_one("#meetings-recover", Button).disabled = True
        self._recover_worker(folders[0])

    @work(exclusive=True, group="meetings-recover", thread=True)
    def _recover_worker(self, folder: Path) -> None:
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
