"""One meeting: dictation callbacks -> labelled segments -> sinks (spec §3.3).

Textual-free. The session owns no devices (the capture does) and no app
objects (the owner does); it knows the capture surface, the dictation
service surface, and a list of sinks.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from loguru import logger

from tldw_chatbook.Utils.log_sanitizer import redact_user_paths

MEETING_JSON = "meeting.json"
TRANSCRIPT_JSONL = "transcript.jsonl"
MEETING_SEGMENT_CAP_S = 10.0
PARTIAL_LABEL_WINDOW_S = 1.0


@dataclass
class MeetingMeta:
    """How one meeting was recorded: folder, mode, sources, transcriber.

    Written to `meeting.json` at start. `mode` is the EFFECTIVE mode
    (`"call"` with system audio, `"room"` mic-only) and is corrected by
    `MeetingSession.start` if the system tap fails to come up.
    """

    folder: Path
    mode: str
    started_at: str
    mic_device: str
    system_source: str
    provider: str
    model: str

    def to_json(self) -> dict:
        """Return the JSON-safe payload (``folder`` stringified)."""
        payload = asdict(self)
        payload["folder"] = str(self.folder)
        return payload


@dataclass
class MeetingSegment:
    """One finalised transcript segment, placed on both clocks.

    Audio time (`t_audio_*`) drives the transcript's timestamps and the
    speaker label window; wall time (`t_wall_*`) is what the user's own
    clock said. `label` is None in room mode.
    """

    seq: int
    t_audio_start: float
    t_audio_end: float
    t_wall_start: float
    t_wall_end: float
    label: str | None
    text: str

    def to_json(self) -> dict:
        """Return the JSONL row for this segment."""
        return asdict(self)


@dataclass
class MeetingResult:
    """The outcome of one meeting: how it ended and what was captured.

    `transcription_complete` is False when the transcriber did not finish
    draining before stop; `failed_segments` counts finals that errored or
    arrived too late to keep.
    """

    meta: MeetingMeta
    ended_at: str
    duration_s: float
    segment_count: int
    transcription_complete: bool
    failed_segments: int
    stop_reason: str
    recovered: bool = False

    def to_json(self) -> dict:
        """Return the full `meeting.json` payload (metadata plus outcome)."""
        payload = self.meta.to_json()
        payload.update(
            ended_at=self.ended_at, duration_s=self.duration_s, segment_count=self.segment_count,
            transcription_complete=self.transcription_complete, failed_segments=self.failed_segments,
            stop_reason=self.stop_reason, recovered=self.recovered,
        )
        return payload


@dataclass
class SpeakerSegment:
    """A diarizer's verdict for one span: who spoke, and optionally what.

    The phase-2 `Diarizer` seam's return type (spec §3.3); nothing produces
    these yet, the energy heuristic in `meeting_capture` labels segments.
    """

    start_s: float
    end_s: float
    speaker: str
    text: str = ""


class MeetingSink(Protocol):
    def on_started(self, meta: MeetingMeta) -> None: ...
    def on_partial(self, text: str, label: str | None) -> None: ...
    def on_segment(self, segment: MeetingSegment) -> None: ...
    def on_stopped(self, result: MeetingResult) -> None: ...


class Diarizer(Protocol):
    """Phase-2 seam: MOSS or the server plugs in here (spec §3.3)."""

    def diarize(self, wav_path: Path, start_s: float, end_s: float) -> list[SpeakerSegment]: ...


def write_meeting_json(folder: Path, payload: dict) -> None:
    """Write `meeting.json` in `folder`, replacing any existing one.

    Args:
        folder: The meeting folder.
        payload: The full payload to persist (JSON-serialisable).

    Raises:
        OSError: The folder is not writable.
    """
    (Path(folder) / MEETING_JSON).write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_meeting_json(folder: Path) -> dict:
    """Read `meeting.json` from `folder`.

    Args:
        folder: The meeting folder.

    Returns:
        The payload, or an empty dict when the file does not exist yet.

    Raises:
        json.JSONDecodeError: The file exists but is truncated or malformed
            (a crash mid-write); recovery reports this to the user.
    """
    path = Path(folder) / MEETING_JSON
    return json.loads(path.read_text()) if path.exists() else {}


def update_meeting_json(folder: Path, **fields: Any) -> dict:
    """Merge `fields` into `folder`'s `meeting.json` and write it back.

    Args:
        folder: The meeting folder.
        **fields: Keys to add or overwrite. Note that `folder` itself is
            positional here, so a payload carrying a "folder" key must have
            it removed before being spread into this call.

    Returns:
        The merged payload.
    """
    payload = read_meeting_json(folder)
    payload.update(fields)
    write_meeting_json(folder, payload)
    return payload


def format_clock(seconds: float) -> str:
    """Format `seconds` as ``HH:MM:SS``; negatives clamp to zero."""
    total = int(max(0.0, seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


class MeetingSession:
    """Turns one dictation service's callbacks into labelled segments."""

    def __init__(
        self,
        *,
        meta: MeetingMeta,
        capture: Any,
        dictation_factory: Callable[[Any], Any],
        sinks: Sequence[MeetingSink],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.meta = meta
        self.capture = capture
        self._dictation_factory = dictation_factory
        self._sinks = list(sinks)
        self._clock = clock
        self.service: Any | None = None
        self.state = "idle"
        self.segments: list[MeetingSegment] = []
        self.failed_segments = 0
        self._listeners: list[Callable[[str, Any], None]] = []
        self._lock = threading.RLock()
        # Separate from `_lock` on purpose (final whole-branch review, C2).
        # `LocalMeetingSink.on_stopped` marshals the Library submit onto the
        # app thread and BLOCKS there; holding `_lock` across that call
        # deadlocked against the screen's own `subscribe`/`unsubscribe`
        # (app thread, same lock) when the user navigated away mid-submit.
        # Sinks are only ever driven from capture/worker threads, never from
        # the app thread, so their own lock still serialises on_segment vs
        # on_stopped without any path back into `_lock`.
        self._sink_lock = threading.Lock()
        self._last_end_s = 0.0
        self._result: MeetingResult | None = None
        self._closing = False
        self._stop_started = False

    # ---- listeners --------------------------------------------------------
    def subscribe(self, listener: Callable[[str, Any], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[str, Any], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _emit(self, kind: str, payload: Any) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(kind, payload)
            except Exception as exc:  # noqa: BLE001
                # `kind` and the listener's identity only: the payload is
                # meeting content (transcript text) and never reaches a log.
                logger.error(
                    "meeting listener error on {} from {}: {}",
                    kind,
                    getattr(listener, "__qualname__", repr(type(listener))),
                    exc,
                )

    def _set_state(self, state: str) -> None:
        self.state = state
        self._emit("state", state)

    def _each_sink(self, method: str, *args: Any) -> None:
        with self._sink_lock:
            for sink in self._sinks:
                try:
                    getattr(sink, method)(*args)
                except Exception as exc:  # noqa: BLE001
                    logger.error("meeting sink {} failed: {}", method, exc)

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> bool:
        self._set_state("starting")
        Path(self.meta.folder).mkdir(parents=True, exist_ok=True)
        payload = self.meta.to_json()
        payload.update(schema=1, ended_at=None, segment_count=0, recovered=False)
        write_meeting_json(self.meta.folder, payload)
        service = self._dictation_factory(self.capture)
        service.privacy_settings["auto_clear_buffer"] = True
        service.MAX_NON_STREAMING_SEGMENT_SECONDS = MEETING_SEGMENT_CAP_S
        self.service = service
        ok = bool(
            service.start_dictation(
                on_partial_transcript=self._on_partial,
                on_final_transcript=self._on_final,
                on_state_change=self._on_service_state,
                on_error=self._on_error,
                on_segment_transcribing=self._on_transcribing,
                on_speech_resumed=self._on_speech_resumed,
                on_segment_no_final=self._on_no_final,
            )
        )
        if not ok:
            self._set_state("error")
            return False
        # The capture settles its effective mode while starting: a system tap
        # that fails to start downgrades "call" to "room" AFTER MeetingMeta
        # was built, and the persisted metadata (and MeetingResult) would
        # otherwise keep claiming system audio was captured (Qodo Q13). The
        # separate "System source lost" indicator reads the tap's own state.
        effective_mode = getattr(self.capture, "mode", self.meta.mode)
        if effective_mode != self.meta.mode:
            self.meta.mode = effective_mode
            payload.update(self.meta.to_json())
            write_meeting_json(self.meta.folder, payload)
        self._set_state("recording")
        self._each_sink("on_started", self.meta)
        return True

    def pause(self) -> None:
        self.capture.pause()
        self._set_state("paused")

    def resume(self) -> None:
        self.capture.resume()
        self._set_state("recording")

    def stop(self, reason: str = "user") -> MeetingResult:
        with self._lock:
            if self._result is not None:
                return self._result
            already_stopping = self._stop_started
            self._stop_started = True
        if already_stopping:
            # ponytail: no threading.Event wait for a genuinely concurrent
            # second caller -- just hand back whatever landed (may still be
            # None if the first caller hasn't finished). Nothing in this
            # codebase calls stop() from more than one thread today; add a
            # threading.Event if that changes.
            with self._lock:
                return self._result
        self._set_state("stopping")
        complete = True
        if self.service is not None:
            try:
                # Finals delivered during this drain must still flow through
                # _on_final normally (_closing isn't set yet).
                outcome = self.service.stop_dictation()
                complete = bool(getattr(outcome, "transcription_complete", True))
            except Exception as exc:  # noqa: BLE001
                logger.error("stop_dictation failed: {}", exc)
                complete = False
        with self._lock:
            self._closing = True
            segment_count = len(self.segments)
        try:
            self.capture.stop_recording()
        except Exception as exc:  # noqa: BLE001
            logger.error("capture stop failed: {}", exc)
        result = MeetingResult(
            meta=self.meta,
            ended_at=datetime.now().isoformat(timespec="seconds"),
            duration_s=float(self.capture.audio_position_s),
            segment_count=segment_count,
            transcription_complete=complete,
            failed_segments=self.failed_segments,
            stop_reason=reason,
        )
        payload = read_meeting_json(self.meta.folder)
        payload.update(result.to_json())
        payload.setdefault("schema", 1)
        write_meeting_json(self.meta.folder, payload)
        with self._lock:
            self._result = result
        self._each_sink("on_stopped", result)
        self._set_state("stopped")
        return result

    # ---- dictation callbacks (capture / processing threads) ---------------
    def _label(self, start_s: float, end_s: float) -> str | None:
        if self.capture.mode != "call":
            return None
        return self.capture.dominant_source(start_s, end_s)

    def _on_partial(self, text: str) -> None:
        if self._closing:
            return
        end = float(self.capture.audio_position_s)
        label = self._label(max(0.0, end - PARTIAL_LABEL_WINDOW_S), end)
        self._emit("partial", (text, label))
        self._each_sink("on_partial", text, label)

    def _on_final(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        late = False
        segment: MeetingSegment | None = None
        with self._lock:
            if self._closing:
                late = True
            else:
                start = self._last_end_s
                closed = self.capture.closed_runs_after(start)
                end = closed[-1].end_s if closed else float(self.capture.last_speech_position_s)
                if end <= start:
                    end = float(self.capture.audio_position_s)
                wall_end = float(self._clock())
                segment = MeetingSegment(
                    seq=len(self.segments),
                    t_audio_start=start,
                    t_audio_end=end,
                    t_wall_start=wall_end - (end - start),
                    t_wall_end=wall_end,
                    label=self._label(start, end),
                    text=text,
                )
                self.segments.append(segment)
                self._last_end_s = end
        if late:
            # Length only: transcript text is meeting content and never goes
            # to the log (final whole-branch review, I1).
            logger.warning("meeting: final transcript arrived after stop and was dropped ({} chars)", len(text))
            self.failed_segments += 1
            return
        self._emit("segment", segment)
        self._each_sink("on_segment", segment)

    def _on_service_state(self, state: str) -> None:
        self._emit("service_state", state)

    def _on_error(self, exc: Exception) -> None:
        self.failed_segments += 1
        self._emit("error", str(exc))

    def _on_transcribing(self, done: bool) -> None:
        self._emit("transcribing", not done)

    def _on_speech_resumed(self) -> None:
        self._emit("speech", True)

    def _on_no_final(self) -> None:
        self._emit("transcribing", False)


def render_markdown(result: MeetingResult, segments: list[MeetingSegment]) -> str:
    """Render a meeting as a Markdown transcript.

    Used when post-meeting re-transcription is off: the Markdown, not the
    audio, is what goes to the Library.

    Args:
        result: The finished meeting.
        segments: Its segments, in order.

    Returns:
        The Markdown document, newline-terminated.
    """
    meta = result.meta
    started = datetime.fromisoformat(meta.started_at)
    lines = [
        f"# Meeting {started:%Y-%m-%d %H:%M}",
        "",
        f"- Audio: `{Path(meta.folder) / 'mixed.wav'}`",
        f"- Mode: {meta.mode}",
        f"- Duration: {format_clock(result.duration_s)}",
        f"- Transcriber: {meta.provider} {meta.model}".rstrip(),
        "",
    ]
    names = {"you": "You", "others": "Others", "both": "You + Others"}
    for segment in segments:
        stamp = f"[{format_clock(segment.t_audio_start)}]"
        if segment.label:
            lines.append(f"{stamp} **{names.get(segment.label, segment.label)}:** {segment.text}")
        else:
            lines.append(f"{stamp} {segment.text}")
    return "\n".join(lines) + "\n"


class LocalMeetingSink:
    """JSONL transcript + Library ingest submit on stop (spec §5).

    Writes one JSON line per finalised segment while the meeting runs, then
    on stop hands the Library either `mixed.wav` (re-transcribed there, with
    diarization) or a rendered `transcript.md`. A submit failure is recorded
    on `last_submit_error` and in `meeting.json` rather than raised: the
    recording is already safe on disk, and the screen's footer reports it.
    """

    def __init__(
        self,
        folder: Path,
        *,
        submit: Callable[..., Optional[str]],
        post_transcribe: bool = True,
        post_diarize: bool = True,
    ) -> None:
        """Build the sink.

        Args:
            folder: The meeting folder; the transcript is written here.
            submit: Library ingest submit, returning a job id. Called on the
                UI thread by the owner's marshalling wrapper.
            post_transcribe: Submit the audio for re-transcription rather
                than the Markdown transcript.
            post_diarize: Ask the Library for speaker labels (audio only).
        """
        self.folder = Path(folder)
        self._submit = submit
        self.post_transcribe = post_transcribe
        self.post_diarize = post_diarize
        self._handle = None
        self._segments: list[MeetingSegment] = []
        self.job_id: str | None = None
        self.last_submit_error: str | None = None

    def __enter__(self) -> "LocalMeetingSink":
        """Return the sink itself; the handle opens on ``on_started``."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the transcript handle however the block is left."""
        self.close()

    def close(self) -> None:
        """Close the JSONL transcript handle. Safe to call more than once."""
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError as exc:
                # A close() error can echo the transcript's absolute path,
                # which sits under the user's recordings dir.
                logger.warning("meeting transcript close failed: {}", redact_user_paths(str(exc)))

    def on_started(self, meta: MeetingMeta) -> None:
        """Open the JSONL transcript for this meeting."""
        self.folder.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.folder / TRANSCRIPT_JSONL, "a", encoding="utf-8")  # noqa: SIM115

    def on_partial(self, text: str, label: str | None) -> None:
        """Ignore partials: only finalised segments are persisted."""
        return None

    def on_segment(self, segment: MeetingSegment) -> None:
        """Append one segment to the JSONL transcript and flush it."""
        self._segments.append(segment)
        if self._handle is not None:
            self._handle.write(json.dumps(segment.to_json()) + "\n")
            self._handle.flush()

    def on_stopped(self, result: MeetingResult) -> None:
        """Close the transcript and hand the meeting to the Library.

        Args:
            result: The finished meeting.

        Raises:
            OSError: The Markdown transcript could not be written (the
                submit's own failures are recorded, not raised).
        """
        # try/finally, not "close first": the markdown render and the Library
        # submit both run here, and either raising used to be a route out of
        # this method that left the transcript handle open (Qodo Q4).
        try:
            started = datetime.fromisoformat(result.meta.started_at)
            title = f"Meeting {started:%Y-%m-%d %H:%M}"
            if self.post_transcribe:
                kwargs = dict(
                    source_path=str(self.folder / "mixed.wav"), title=title, keywords=("meeting",),
                    detected_type="audio", ingest_options={"diarization": bool(self.post_diarize)},
                )
            else:
                md_path = self.folder / "transcript.md"
                md_path.write_text(render_markdown(result, self._segments), encoding="utf-8")
                kwargs = dict(
                    source_path=str(md_path), title=title, keywords=("meeting",),
                    detected_type="document", ingest_options={},
                )
            try:
                self.job_id = self._submit(**kwargs)
                update_meeting_json(self.folder, ingest_job_id=self.job_id)
            except Exception as exc:  # noqa: BLE001 - the footer reports it (spec §7)
                self.last_submit_error = str(exc)
                update_meeting_json(self.folder, ingest_error=str(exc))
                # The submit kwargs carry the meeting folder, so a registry
                # error usually echoes an absolute path back at us.
                logger.error("meeting ingest submit failed: {}", redact_user_paths(str(exc)))
        finally:
            self.close()
