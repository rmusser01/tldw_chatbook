"""One meeting: dictation callbacks -> labelled segments -> sinks (spec §3.3).

Textual-free. The session owns no devices (the capture does) and no app
objects (the owner does); it knows the capture surface, the dictation
service surface, and a list of sinks.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
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
    #: The one shared display name for the mic ("you") channel (task 31746),
    #: stamped by `MeetingSessionOwner.start()` from `meeting_owner.
    #: meeting_user_display_name()` -- so `render_markdown` and every
    #: after-the-fact render agree with what the live session showed.
    #: Defaulted to "You" for direct-construction call sites (tests, and any
    #: `MeetingMeta` built before this field existed).
    user_display_name: str = "You"
    speaker_names: dict = field(default_factory=dict)
    format_version: int = 2

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
    # Trails `text` (not `label`, despite the brief's prose) so the two
    # existing positional `MeetingSegment(...)` call sites in the test suite
    # -- 7 positional args ending at `text` -- keep working: a defaulted
    # field can't sit before `text`, which has no default.
    speaker_id: str | None = None

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
    # Cluster ids whose Stop-pass merge collided two user-assigned names and
    # were kept as "Alice / Bob" for the user to resolve (spec §4). Empty
    # unless the batch pass folded two differently-named live clusters.
    flagged_speakers: list[str] = field(default_factory=list)
    #: Static reason live speaker labels were not produced (spec §7 footer
    #: copy), e.g. "backend unavailable" / "backend crashed". None when live
    #: labelling was never requested or ran fine. Never a path or a name.
    speaker_labels_reason: str | None = None

    def to_json(self) -> dict:
        """Return the full `meeting.json` payload (metadata plus outcome)."""
        payload = self.meta.to_json()
        payload.update(
            ended_at=self.ended_at, duration_s=self.duration_s, segment_count=self.segment_count,
            transcription_complete=self.transcription_complete, failed_segments=self.failed_segments,
            stop_reason=self.stop_reason, recovered=self.recovered,
            flagged_speakers=list(self.flagged_speakers),
            speaker_labels_reason=self.speaker_labels_reason,
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
    """Phase-2 seam: MOSS or the server plugs in here (spec §3.3).

    The backend owns reconciliation: it holds the live online centroids and
    the batch centroids and applies `OnlineClusterer.reconcile` internally.
    The session never calls `reconcile` and never touches centroids -- every
    id `assign`/`diarize` hands back is already the reconciled live cluster
    id.
    """

    def diarize(self, wav_path: Path, start_s: float, end_s: float) -> list[SpeakerSegment]: ...

    def assign(self, pcm: bytes, sample_rate: int, seq: int) -> str | None: ...

    def pin(self, cluster_id: str) -> None: ...

    def centroids(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


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
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    # Back-fill pre-task-2 (format_version 1 or absent) recordings so
    # callers can always read `speaker_names` without a KeyError.
    payload.setdefault("speaker_names", {})
    # Back-fill pre-task-31746 recordings the same way: they were made
    # before the mic channel's display name was persisted, and always
    # showed "You" live.
    payload.setdefault("user_display_name", "You")
    return payload


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


#: Speaker names are typed by the user and land in `meeting.json`, the
#: transcript render, `Media.content` and FTS -- bound them at the one
#: boundary both rename paths share (Qodo Q2).
MAX_SPEAKER_NAME_CHARS = 64
_WIDGET_SAFE_CLUSTER_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def normalize_speaker_name(value: str) -> str:
    """Clean one user-typed speaker name for storage and display.

    Args:
        value: The raw submitted name.

    Returns:
        The name stripped of surrounding whitespace and truncated to
        `MAX_SPEAKER_NAME_CHARS`; `""` means "remove this speaker's name".
    """
    return (value or "").strip()[:MAX_SPEAKER_NAME_CHARS]


def is_widget_safe_cluster_id(cluster_id: str) -> bool:
    """Whether `cluster_id` can be interpolated into a Textual widget id.

    Cluster ids come from `transcript.jsonl`, which a user can hand-edit; a
    value with a space or a "#" would raise out of `compose()` and take the
    screen down, so callers skip the legend row instead.

    Args:
        cluster_id: The candidate id.

    Returns:
        True when it matches Textual's identifier rules.
    """
    return bool(cluster_id) and bool(_WIDGET_SAFE_CLUSTER_ID.match(cluster_id))


def render_label(segment: MeetingSegment, names: dict[str, str], user_display_name: str) -> str | None:
    """Display name for a segment: the user for the mic channel, else the
    named or generic speaker.

    Args:
        segment: The transcript segment to label.
        names: The meeting's `cluster_id -> user name` map; ids absent from
            it fall back to a generic "Speaker N".
        user_display_name: What stands in for the mic channel ("You").

    Returns:
        The display name, or None when the segment carries no label at all
        (room mode before diarization) or is overlap-coarse.
    """
    if segment.label == "you":
        return user_display_name
    if segment.speaker_id:
        if segment.speaker_id in names:
            return names[segment.speaker_id]
        # Strip a leading live-style "S" prefix, and also a stray "F" so a
        # legacy recording whose jsonl still holds an unmatched final-cluster
        # id never renders as "Speaker F0" (final whole-branch review I2; new
        # recordings mint an "S" id in the worker so this only guards old data).
        n = segment.speaker_id[1:] if segment.speaker_id[:1] in ("S", "F") else segment.speaker_id
        return f"Speaker {n}"
    if segment.label in ("others", "both"):
        return "Others"
    return None


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
        diarizer: Diarizer | None = None,
    ) -> None:
        self.meta = meta
        self.capture = capture
        self._dictation_factory = dictation_factory
        self._sinks = list(sinks)
        self._clock = clock
        self._diarizer = diarizer
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
                # meeting content (transcript text) and never reaches a log
                # (Q10). `str(exc)` can still embed a filesystem path though
                # (task-31748) -- redact it, same treatment as this module's
                # other failure logs that keep the exception's message.
                logger.error(
                    "meeting listener error on {} from {}: {}",
                    kind,
                    getattr(listener, "__qualname__", repr(type(listener))),
                    redact_user_paths(str(exc)),
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
                    # Sinks are caller-supplied (`MeetingSession(sinks=...)`),
                    # so an exception's message is as unpredictable as the
                    # payload it may embed -- type name only (task-31748).
                    logger.error("meeting sink {} failed ({})", method, type(exc).__name__)

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
                # task-31748: `str(exc)` can embed a filesystem path (e.g. a
                # missing model file) -- redact it, same treatment as this
                # module's other internal-operation failure logs.
                logger.error("stop_dictation failed: {}", redact_user_paths(str(exc)))
                complete = False
        with self._lock:
            self._closing = True
            segment_count = len(self.segments)
        try:
            self.capture.stop_recording()
        except Exception as exc:  # noqa: BLE001
            # task-31748: `str(exc)` can embed a filesystem path (closing a
            # WAV file, say) -- redact it.
            logger.error("capture stop failed: {}", redact_user_paths(str(exc)))
        flagged_speakers: list[str] = []
        speaker_labels_reason: str | None = None
        if self._diarizer is not None:
            # Best-effort authoritative batch pass (spec §3.3/§4): a failure
            # here must never block the meeting result. `assign` labelled what
            # it could near-live; this overlay fills in (and can correct)
            # speaker ids from the full recording, then PERSISTS them by
            # re-emitting the changed segments so the seq-keyed sink rewrites
            # transcript.jsonl and the screen updates in place (final
            # whole-branch review I2 -- the overlay used to touch in-memory
            # segments only, which nothing persisted read). Never log segment
            # text or speaker names -- lengths/types only.
            try:
                # Diarize the SAME channel the live centroids were built from
                # (`_on_final` used "others" in call mode, "mixed" in room
                # mode) so reconcile compares like-for-like. Absent file ->
                # skip, keep near-live labels, never raise.
                wav_name = "others.wav" if self.capture.mode == "call" else "mixed.wav"
                wav_path = Path(self.meta.folder) / wav_name
                with self._lock:
                    meeting_segments = list(self.segments)
                # 31749: a crashed-and-restarted backend holds NO live
                # centroids, so its batch pass mints ids from scratch. Applied
                # to the whole meeting it would re-label the PRE-crash
                # segments too and silently strand the names the user typed on
                # those ids. Re-label only from the crash on; everything
                # before it keeps the near-live id its name is attached to.
                crash_seq = getattr(self._diarizer, "crashed_at_seq", None)
                start_from: float | None = 0.0
                if crash_seq is not None:
                    meeting_segments = [s for s in meeting_segments if s.seq >= crash_seq]
                    # Nothing after the crash -> nothing this pass may touch.
                    start_from = meeting_segments[0].t_audio_start if meeting_segments else None
                if wav_path.exists() and start_from is not None:
                    duration = float(self.capture.audio_position_s)
                    speaker_segments = self._diarizer.diarize(wav_path, start_from, duration)
                    transitions: list[tuple[str | None, str]] = []
                    changed: list[MeetingSegment] = []
                    for meeting_segment in meeting_segments:
                        midpoint = (meeting_segment.t_audio_start + meeting_segment.t_audio_end) / 2.0
                        old = meeting_segment.speaker_id
                        new = old
                        for speaker_segment in speaker_segments:
                            if speaker_segment.start_s <= midpoint <= speaker_segment.end_s:
                                new = speaker_segment.speaker
                                break
                        if old is not None:
                            transitions.append((old, new))
                        if new != old:
                            with self._lock:
                                meeting_segment.speaker_id = new
                            changed.append(meeting_segment)
                    # Spec §4: a batch merge of two differently user-named live
                    # clusters keeps both names on the survivor and flags it.
                    from tldw_chatbook.Audio.diarizer_cluster import merged_speaker_names
                    merged_names, flagged_speakers = merged_speaker_names(
                        transitions, self.meta.speaker_names
                    )
                    if merged_names:
                        with self._lock:
                            self.meta.speaker_names.update(merged_names)
                    # Persist the authoritative labels (idempotent by seq, I1):
                    # off `_lock` -- the sinks marshal onto the app thread.
                    for seg in changed:
                        self._each_sink("on_segment", seg)
                        self._emit("segment", seg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("meeting: diarizer stop pass failed ({})", type(exc).__name__)
            finally:
                # Read BEFORE close(): the footer needs to say why the meeting
                # ran on coarse labels (spec §7), and close() tears the
                # backend down. Static string only -- never a path or a name.
                speaker_labels_reason = getattr(self._diarizer, "coarse_reason", None)
                try:
                    self._diarizer.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("meeting: diarizer close failed ({})", type(exc).__name__)
        result = MeetingResult(
            meta=self.meta,
            ended_at=datetime.now().isoformat(timespec="seconds"),
            duration_s=float(self.capture.audio_position_s),
            segment_count=segment_count,
            transcription_complete=complete,
            failed_segments=self.failed_segments,
            stop_reason=reason,
            flagged_speakers=flagged_speakers,
            speaker_labels_reason=speaker_labels_reason,
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
        # Near-live speaker labelling (spec §3.3): "others" in call mode, or
        # any segment in room mode (label is None there). `assign` may block
        # on a subprocess in the real backend, so it MUST run off `_lock`,
        # which is already released at this point in the method.
        if self._diarizer is not None and segment is not None and segment.label in ("others", None):
            source = "others" if segment.label == "others" else "mixed"
            pcm = self.capture.pcm_window(source, segment.t_audio_start, segment.t_audio_end)
            sid = None
            if pcm:
                try:
                    sid = self._diarizer.assign(pcm, 16000, segment.seq)  # OFF the lock
                except Exception as exc:  # noqa: BLE001 - best-effort, never breaks the meeting
                    logger.warning("meeting: diarizer assign failed ({})", type(exc).__name__)
            if sid is not None:
                with self._lock:
                    segment.speaker_id = sid
                self._emit("segment", segment)
                self._each_sink("on_segment", segment)

    def _on_final_for_test(self, text: str, *, label: str | None = None) -> None:
        """Test-only: drive `_on_final`, optionally forcing its label.

        A thin wrapper over the real `_on_final` -- when `label` is given it
        temporarily overrides `_label` (the coarse-source lookup) so tests
        don't need a fake capture that reproduces exact dominant-source
        timing math to exercise the "others"/room-mode diarizer paths.
        """
        if label is None:
            self._on_final(text)
            return
        original = self._label
        self._label = lambda *_a, **_kw: label
        try:
            self._on_final(text)
        finally:
            self._label = original

    def _lock_is_held_for_test(self) -> bool:
        """Test-only: True if the CURRENT thread holds `self._lock` right now."""
        return self._lock._is_owned()

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
    # task 31746: the mic channel's name comes from `meta` (stamped by the
    # owner at start, or back-filled by `read_meeting_json`), not a literal.
    names = {"you": meta.user_display_name, "others": "Others", "both": f"{meta.user_display_name} + Others"}
    speaker_names = getattr(meta, "speaker_names", {}) or {}
    for segment in segments:
        stamp = f"[{format_clock(segment.t_audio_start)}]"
        if segment.speaker_id:
            # A diarized segment carries the authoritative (reconciled) speaker
            # id; render its name/"Speaker N" so the markdown reflects the Stop
            # pass, not the coarse You/Others label (final whole-branch review
            # I2). Undiarized segments keep the coarse channel label below.
            who = render_label(segment, speaker_names, names["you"])
            lines.append(f"{stamp} **{who}:** {segment.text}" if who else f"{stamp} {segment.text}")
        elif segment.label:
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
        # Keyed by `seq`, not a list: segment delivery is idempotent by seq
        # (final whole-branch review I1) -- the near-live path emits a segment
        # coarse then again with its speaker id, and the Stop pass re-emits
        # reconciled segments, so a repeat seq must UPDATE its row, not append
        # a duplicate.
        self._segments: dict[int, MeetingSegment] = {}
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
        """Open the JSONL transcript for this meeting (rewritten per segment)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        # "w", not "a": on_segment rewrites the whole file from the seq-keyed
        # map, so a fresh, seekable, truncatable handle is what it needs.
        self._handle = open(self.folder / TRANSCRIPT_JSONL, "w", encoding="utf-8")  # noqa: SIM115

    def on_partial(self, text: str, label: str | None) -> None:
        """Ignore partials: only finalised segments are persisted."""
        return None

    def on_segment(self, segment: MeetingSegment) -> None:
        """Record one segment by ``seq`` and rewrite the JSONL transcript.

        Idempotent by ``seq`` (final whole-branch review I1): a repeat delivery
        for the same segment -- its near-live speaker-id refinement, or the
        Stop pass's reconciled id -- UPDATES that row in place rather than
        appending a duplicate, so transcript.jsonl (and the markdown rendered
        from these segments) carries exactly one row per segment.
        """
        self._segments[segment.seq] = segment
        self._rewrite_transcript()

    def _ordered_segments(self) -> list[MeetingSegment]:
        """Segments in ``seq`` order -- the transcript's natural order."""
        return [self._segments[seq] for seq in sorted(self._segments)]

    def _rewrite_transcript(self) -> None:
        """Rewrite transcript.jsonl from the seq-keyed map (one row per seq).

        ponytail: truncate-and-rewrite through the open handle, not an atomic
        temp+os.replace -- one open handle sidesteps replace-over-open-file
        failing on Windows, and transcript.jsonl only feeds the best-effort
        after-the-fact rename (the audio, not this file, is the record).
        Rewriting the whole file each segment is cheap at meeting sizes.
        """
        if self._handle is None:
            return
        self._handle.seek(0)
        self._handle.truncate()
        for segment in self._ordered_segments():
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
                md_path.write_text(render_markdown(result, self._ordered_segments()), encoding="utf-8")
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
