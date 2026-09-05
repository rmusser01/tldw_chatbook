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

MEETING_JSON = "meeting.json"
TRANSCRIPT_JSONL = "transcript.jsonl"
MEETING_SEGMENT_CAP_S = 10.0
PARTIAL_LABEL_WINDOW_S = 1.0


@dataclass
class MeetingMeta:
    folder: Path
    mode: str
    started_at: str
    mic_device: str
    system_source: str
    provider: str
    model: str

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["folder"] = str(self.folder)
        return payload


@dataclass
class MeetingSegment:
    seq: int
    t_audio_start: float
    t_audio_end: float
    t_wall_start: float
    t_wall_end: float
    label: str | None
    text: str

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class MeetingResult:
    meta: MeetingMeta
    ended_at: str
    duration_s: float
    segment_count: int
    transcription_complete: bool
    failed_segments: int
    stop_reason: str
    recovered: bool = False

    def to_json(self) -> dict:
        payload = self.meta.to_json()
        payload.update(
            ended_at=self.ended_at, duration_s=self.duration_s, segment_count=self.segment_count,
            transcription_complete=self.transcription_complete, failed_segments=self.failed_segments,
            stop_reason=self.stop_reason, recovered=self.recovered,
        )
        return payload


@dataclass
class SpeakerSegment:
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
    (Path(folder) / MEETING_JSON).write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_meeting_json(folder: Path) -> dict:
    path = Path(folder) / MEETING_JSON
    return json.loads(path.read_text()) if path.exists() else {}


def update_meeting_json(folder: Path, **fields: Any) -> dict:
    payload = read_meeting_json(folder)
    payload.update(fields)
    write_meeting_json(folder, payload)
    return payload


def format_clock(seconds: float) -> str:
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
        self._last_end_s = 0.0
        self._result: MeetingResult | None = None

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
                logger.error("meeting listener error: {}", exc)

    def _set_state(self, state: str) -> None:
        self.state = state
        self._emit("state", state)

    def _each_sink(self, method: str, *args: Any) -> None:
        with self._lock:
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
            self._set_state("stopping")
        complete = True
        if self.service is not None:
            try:
                outcome = self.service.stop_dictation()
                complete = bool(getattr(outcome, "transcription_complete", True))
            except Exception as exc:  # noqa: BLE001
                logger.error("stop_dictation failed: {}", exc)
                complete = False
        try:
            self.capture.stop_recording()
        except Exception as exc:  # noqa: BLE001
            logger.error("capture stop failed: {}", exc)
        result = MeetingResult(
            meta=self.meta,
            ended_at=datetime.now().isoformat(timespec="seconds"),
            duration_s=float(self.capture.audio_position_s),
            segment_count=len(self.segments),
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
        end = float(self.capture.audio_position_s)
        label = self._label(max(0.0, end - PARTIAL_LABEL_WINDOW_S), end)
        self._emit("partial", (text, label))
        self._each_sink("on_partial", text, label)

    def _on_final(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
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
    """JSONL transcript + Library ingest submit on stop (spec §5)."""

    def __init__(
        self,
        folder: Path,
        *,
        submit: Callable[..., Optional[str]],
        post_transcribe: bool = True,
        post_diarize: bool = True,
    ) -> None:
        self.folder = Path(folder)
        self._submit = submit
        self.post_transcribe = post_transcribe
        self.post_diarize = post_diarize
        self._handle = None
        self._segments: list[MeetingSegment] = []
        self.job_id: str | None = None
        self.last_submit_error: str | None = None

    def on_started(self, meta: MeetingMeta) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.folder / TRANSCRIPT_JSONL, "a", encoding="utf-8")  # noqa: SIM115

    def on_partial(self, text: str, label: str | None) -> None:
        return None

    def on_segment(self, segment: MeetingSegment) -> None:
        self._segments.append(segment)
        if self._handle is not None:
            self._handle.write(json.dumps(segment.to_json()) + "\n")
            self._handle.flush()

    def on_stopped(self, result: MeetingResult) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
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
            logger.error("meeting ingest submit failed: {}", exc)
