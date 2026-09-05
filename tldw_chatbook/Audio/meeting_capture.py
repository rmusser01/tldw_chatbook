"""Meeting capture: mic + system audio mixed into one dictation stream.

Textual-free. numpy is required (the recorder already requires it) and is
reached through `optional_deps.require_dependency` rather than a bare
import, so an install without the `audio` extra gets the project's standard
missing-dependency message instead of a raw ImportError.

`EnergyRing` answers "who was talking in this window" from per-source RMS
history (spec §4). `MeetingCapture` (Task 5) duck-types the recorder
surface `LazyLiveDictationService` uses.
"""
from __future__ import annotations

import bisect
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Mapping, Optional, Tuple

from loguru import logger

from tldw_chatbook.Utils.optional_deps import require_dependency

from .wav_writer import PlaceholderWavWriter

#: -60 dBFS expressed as int16 RMS. Digital silence must not yield a zero
#: adaptive floor that any dither would exceed.
ABS_MIN_RMS: float = 32768 * 10 ** (-60 / 20)
SHARE_YOU: float = 0.7
SHARE_OTHERS: float = 0.3
FLOOR_MULTIPLIER: float = 3.0

#: numpy feature name: the `audio` extra is the one that ships numpy for
#: capture (`pyproject.toml`), so its name is what the install hint names.
NUMPY_FEATURE = "audio"
_NUMPY: Any = None


def _np() -> Any:
    """Return numpy through the project's optional-dependency guard.

    Cached after the first call: this runs once per 20 ms audio frame.

    Returns:
        The numpy module.

    Raises:
        ImportError: When numpy is unavailable, with the standard
            "install tldw_chatbook[audio]" message.
    """
    global _NUMPY
    if _NUMPY is None:
        _NUMPY = require_dependency("numpy", NUMPY_FEATURE)
    return _NUMPY


def rms_int16(pcm: bytes) -> float:
    """RMS of little-endian int16 PCM; 0.0 for an empty buffer."""
    if len(pcm) < 2:
        return 0.0
    np = _np()
    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


@dataclass
class SpeechRun:
    """A span of VAD-detected speech in audio seconds; ``end_s`` None while open."""

    start_s: float
    end_s: float | None = None


class EnergyRing:
    """Per-source RMS history in fixed buckets, bounded by a time horizon."""

    def __init__(
        self,
        bucket_s: float = 0.1,
        horizon_s: float = 600.0,
        floor_window_s: float = 30.0,
    ) -> None:
        self.bucket_s = bucket_s
        self.horizon_s = horizon_s
        self.floor_window_s = floor_window_s
        self._times: Deque[float] = deque()
        self._mic: Deque[float] = deque()
        self._sys: Deque[float] = deque()

    def add(self, position_s: float, mic_rms: float, sys_rms: float) -> None:
        bucket = math.floor(position_s / self.bucket_s) * self.bucket_s
        if self._times and bucket <= self._times[-1]:
            # Same bucket: keep the louder reading for each source.
            self._mic[-1] = max(self._mic[-1], mic_rms)
            self._sys[-1] = max(self._sys[-1], sys_rms)
        else:
            self._times.append(bucket)
            self._mic.append(mic_rms)
            self._sys.append(sys_rms)
        while self._times and self._times[0] < bucket - self.horizon_s:
            self._times.popleft()
            self._mic.popleft()
            self._sys.popleft()

    def _bucket(self, t: float) -> float:
        return math.floor(t / self.bucket_s) * self.bucket_s

    def _slice(self, start_s: float, end_s: float, *, inclusive_end: bool = True) -> Tuple[list, list]:
        """Buckets with start <= time <= end (or < end when inclusive_end is False)."""
        times = list(self._times)
        lo = bisect.bisect_left(times, self._bucket(start_s))
        hi = (
            bisect.bisect_right(times, end_s)
            if inclusive_end
            else bisect.bisect_left(times, self._bucket(end_s))
        )
        return list(self._mic)[lo:hi], list(self._sys)[lo:hi]

    def floor(self, source: str, before_s: float) -> float:
        """Adaptive noise floor from the 30 s strictly before ``before_s``: 3x its 10th percentile, never below ABS_MIN_RMS."""
        mic, sys_ = self._slice(before_s - self.floor_window_s, before_s, inclusive_end=False)
        values = mic if source == "mic" else sys_
        if not values:
            return ABS_MIN_RMS
        np = _np()
        p10 = float(np.percentile(np.asarray(values, dtype=np.float64), 10))
        return max(FLOOR_MULTIPLIER * p10, ABS_MIN_RMS)

    def dominant_source(self, start_s: float, end_s: float) -> str:
        """Label a window ``you`` / ``others`` / ``both`` (spec §4)."""
        mic, sys_ = self._slice(start_s, end_s)
        mic_floor = self.floor("mic", start_s)
        sys_floor = self.floor("sys", start_s)
        mic_active = sum(v for v in mic if v > mic_floor)
        sys_active = sum(v for v in sys_ if v > sys_floor)
        total = mic_active + sys_active
        if total <= 0.0:
            # ponytail: energy-share heuristic; the Diarizer seam (meeting_session.py)
            # replaces this whole method in phase 2.
            return "you" if sum(mic) > sum(sys_) else "others"
        share_you = mic_active / total
        if share_you >= SHARE_YOU:
            return "you"
        if share_you <= SHARE_OTHERS:
            return "others"
        return "both"


FRAME_BYTES = 640            # 20 ms at 16 kHz mono int16
FRAME_S = 0.02
BACKLOG_BYTES = 10 * FRAME_BYTES   # 200 ms
TAP_BUFFER_MAX = 50 * FRAME_BYTES  # 1 s
BYTES_PER_S = 32000

#: How much recent PCM `pcm_window` can still answer for, per source.
RING_SECONDS = 60
SAMPLE_RATE = 16000


def mix_int16(a: bytes, b: bytes) -> bytes:
    """Saturating sum of two equal-length int16 buffers."""
    np = _np()
    x = np.frombuffer(a, dtype=np.int16).astype(np.int32)
    y = np.frombuffer(b, dtype=np.int16).astype(np.int32)
    return np.clip(x + y, -32768, 32767).astype(np.int16).tobytes()


def _default_vad_factory():
    import webrtcvad

    vad = webrtcvad.Vad()
    vad.set_mode(2)
    return vad


class MeetingCapture:
    """Recorder-shaped source for one meeting: mic + system tap -> one mixed stream.

    The mic callback is the clock (spec §3.2): every mic chunk pulls the
    same number of system bytes from the tap buffer (zero-filled when
    short), writes all tracks, updates the energy ring, gates the mix
    through VAD and hands speech to the dictation callback.
    """

    sample_rate = 16000
    channels = 1

    def __init__(
        self,
        *,
        mic_recorder_factory: Callable[..., Any],
        tap: Any | None,
        writers: Mapping[str, PlaceholderWavWriter],
        vad_factory: Callable[[], Any] | None = None,
        silence_threshold_s: float = 2.0,
        preroll_frames: int = 12,
        mic_device_name: str | None = None,
    ) -> None:
        if "mixed" not in writers:
            raise ValueError("writers must include 'mixed'")
        # Fail here, with the project's optional-dependency message, rather
        # than on the first audio frame inside the recorder callback.
        self._np = _np()
        self._mic_device_name = mic_device_name or None
        self._mic_factory = mic_recorder_factory
        self._tap = tap
        # `start_recording` clears `self._tap` when the tap fails to start,
        # which used to make `system_source_state` read "none" -- the same
        # answer as room mode, where the user never asked for system audio
        # at all. Remembering that a tap WAS configured lets the rail say
        # "lost" (final whole-branch review).
        self._tap_configured = tap is not None
        self._writers = dict(writers)
        self._vad_factory = vad_factory or _default_vad_factory
        self._silence_threshold_s = silence_threshold_s
        self._preroll: deque = deque(maxlen=max(0, preroll_frames))
        self.mode = "call" if tap is not None else "room"
        self._mic: Any | None = None
        self._vad: Any | None = None
        self._callback: Optional[Callable[[bytes], None]] = None
        self._running = False
        self._paused = False
        self._tap_lock = threading.Lock()
        self._tap_buf = bytearray()
        self._ring = EnergyRing()
        self._runs: list[SpeechRun] = []
        self._open_run: SpeechRun | None = None
        self._levels = (0.0, 0.0)
        self.last_speech_position_s = 0.0
        self.fault: Exception | None = None
        self._gate_carry = bytearray()
        self._stopped = False
        self._pcm_rings: dict[str, Deque[Tuple[int, bytes]]] = {
            "you": deque(),
            "others": deque(),
            "mixed": deque(),
        }
        # Cumulative sample count per source, independent of any writer --
        # `_push_pcm` is called directly in tests without a real write path,
        # so the ring's absolute positions can't be read back off a writer.
        self._pcm_next_sample: dict[str, int] = {"you": 0, "others": 0, "mixed": 0}

    # ---- recorder surface -------------------------------------------------
    def start_recording(self, callback=None, save_to_file=None) -> bool:
        """Build the mic recorder, select the device, start mic and tap.

        Args:
            callback: Receives VAD-gated mixed audio (the dictation stream).
            save_to_file: Ignored; this capture owns its own writers.

        Returns:
            True when the microphone is recording. False leaves every writer
            CLOSED (spec §3.2: no half-open files on a failed start) and, for
            an unresolvable device, ``self.fault`` set to a ``LookupError``.
        """
        self._callback = callback
        try:
            self._vad = self._vad_factory()
        except Exception as exc:  # noqa: BLE001 - VAD optional; gate everything through
            logger.warning("Meeting VAD unavailable, passing all audio: {}", exc)
            self._vad = None
        self._mic = self._mic_factory(use_vad=False, retain_audio=False, chunk_size=320)
        self._running = True
        if self._mic_device_name is not None and not self._select_mic_device(self._mic_device_name):
            return self._abandon_start()
        if self._tap is not None and not self._tap.start(self._on_tap_frame):
            logger.warning("System audio tap failed to start; continuing in room mode")
            self._tap = None
            self.mode = "room"
        try:
            started = bool(self._mic.start_recording(callback=self._on_mic_frame))
        except Exception as exc:  # noqa: BLE001 - reported through the return value
            logger.error("Meeting microphone failed to start: {}", exc)
            self.fault = exc
            started = False
        return True if started else self._abandon_start()

    def _select_mic_device(self, name: str) -> bool:
        """Point the recorder at the microphone the user chose, by name.

        Enumeration returns names while ``set_device`` wants an id, so the
        name is resolved against the recorder's own device list. A name that
        no longer resolves is NOT silently downgraded to the system default:
        recording the wrong microphone for a whole meeting is worse than
        refusing to start (same rule as ``DeviceTap``).

        Args:
            name: The configured microphone's device name.

        Returns:
            True when the device was found and selected.
        """
        try:
            devices = list(self._mic.get_audio_devices())
        except Exception as exc:  # noqa: BLE001 - treated as "cannot resolve"
            logger.warning("Meeting microphone enumeration failed: {}", exc)
            devices = []
        for device in devices:
            if str(device.get("name", "")) == name:
                self._mic.set_device(device.get("id", device.get("index")))
                return True
        # The device NAME stays out of the log: audio devices are routinely
        # named after their owner ("<Name>'s AirPods"), and the user picked
        # it -- `self.fault` carries it to the caller instead.
        logger.warning("the configured meeting microphone was not found; not falling back to the default input")
        self.fault = LookupError(f"microphone {name!r} not found")
        return False

    def _abandon_start(self) -> bool:
        """Close everything a partial start opened; always returns False."""
        self.stop_recording()
        return False

    def stop_recording(self) -> None:
        if self._stopped:
            return None
        self._stopped = True
        self._running = False
        if self._mic is not None:
            try:
                self._mic.stop_recording()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Mic stop failed: {}", exc)
        if self._tap is not None:
            self._tap.stop()
        self._close_open_run()
        for writer in self._writers.values():
            writer.close()
        return None

    def get_audio_level(self) -> float:
        return self._levels[0]

    def get_audio_devices(self) -> list:
        return list(self._mic.get_audio_devices()) if self._mic is not None else []

    def set_device(self, device_id) -> bool:
        return bool(self._mic.set_device(device_id)) if self._mic is not None else False

    def is_available(self) -> bool:
        return True

    # ---- meeting surface --------------------------------------------------
    def levels(self) -> tuple[float, float]:
        return self._levels

    @property
    def audio_position_s(self) -> float:
        return self._writers["mixed"].audio_position_s

    @property
    def system_source_state(self) -> str:
        """The system-audio tap's own state, for the rail's "lost" indicator.

        ``"none"`` only in room mode -- the user never asked for system
        audio. A tap that WAS configured but failed to start reports
        ``"lost"``: ``start_recording`` drops ``self._tap`` in that case, so
        the tap object itself is gone and ``_tap_configured`` is the only
        remaining evidence that the degradation is worth telling the user
        about. Otherwise this is the tap's own ``state`` (``"running"``,
        ``"lost"``, etc, per ``system_audio_tap.py``) -- ``self._tap`` stays
        the SAME object through a mid-session restart-once-then-give-up
        cycle, so its own ``state`` is where a mid-session loss is recorded.
        """
        if self._tap is None:
            return "lost" if self._tap_configured else "none"
        return getattr(self._tap, "state", "unknown")

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def closed_runs_after(self, t: float) -> list[SpeechRun]:
        return [r for r in self._runs if r.end_s is not None and r.end_s > t]

    def dominant_source(self, start_s: float, end_s: float) -> str:
        return self._ring.dominant_source(start_s, end_s)

    def _push_pcm(self, source: str, chunk: bytes) -> None:
        """Append ``chunk`` to ``source``'s ring, keyed by absolute sample index.

        Trims to the trailing `RING_SECONDS` so memory stays bounded by time,
        not meeting length -- called from the same place each source's frame
        reaches its WAV writer. Position is tracked per source by a running
        sample count (not read off the writer): all three sources receive
        equal-length chunks per real frame, so the counts stay in lockstep
        with `audio_position_s`, and a direct call (as in tests) still gets
        correct, monotonically increasing positions.
        """
        ring = self._pcm_rings.setdefault(source, deque())
        start_sample = self._pcm_next_sample.get(source, 0)
        n = len(chunk) // 2
        ring.append((start_sample, chunk))
        self._pcm_next_sample[source] = start_sample + n
        cutoff = start_sample + n - RING_SECONDS * SAMPLE_RATE
        while ring and ring[0][0] + len(ring[0][1]) // 2 <= cutoff:
            ring.popleft()

    def pcm_window(self, source: str, start_s: float, end_s: float) -> bytes:
        """PCM16 mono 16 kHz bytes for `source` over `[start_s, end_s)`.

        Clipped to whatever the bounded ring still holds; empty bytes once
        the window has aged out of the ring.
        """
        ring = self._pcm_rings.get(source)
        if not ring:
            return b""
        a = int(start_s * SAMPLE_RATE)
        b = int(end_s * SAMPLE_RATE)
        out = bytearray()
        for start_sample, chunk in ring:
            n = len(chunk) // 2
            lo, hi = start_sample, start_sample + n
            if hi <= a or lo >= b:
                continue
            s = max(a, lo) - lo
            e = min(b, hi) - lo
            out += chunk[s * 2 : e * 2]
        return bytes(out)

    def _tap_backlog_bytes(self) -> int:
        with self._tap_lock:
            return len(self._tap_buf)

    # ---- frame handling ---------------------------------------------------
    def _on_tap_frame(self, frame: bytes) -> None:
        with self._tap_lock:
            self._tap_buf.extend(frame)
            overflow = len(self._tap_buf) - TAP_BUFFER_MAX
            if overflow > 0:
                del self._tap_buf[:overflow]

    def _take_tap_bytes(self, n: int) -> bytes:
        with self._tap_lock:
            part = bytes(self._tap_buf[:n])
            del self._tap_buf[:n]
            if len(self._tap_buf) > BACKLOG_BYTES:
                del self._tap_buf[:FRAME_BYTES]   # one extra frame per tick
        if len(part) < n:
            part += b"\x00" * (n - len(part))
        return part

    def _on_mic_frame(self, chunk: bytes) -> None:
        if not self._running:
            return
        try:
            if self._paused:
                with self._tap_lock:
                    self._tap_buf.clear()
                return
            n = len(chunk)
            sys_part = self._take_tap_bytes(n) if self._tap is not None else b"\x00" * n
            mixed = mix_int16(chunk, sys_part) if self._tap is not None else chunk
            start_pos = self.audio_position_s
            self._writers["mixed"].write(mixed)
            self._push_pcm("mixed", mixed)
            if "you" in self._writers:
                self._writers["you"].write(chunk)
                self._push_pcm("you", chunk)
            if "others" in self._writers:
                self._writers["others"].write(sys_part)
                self._push_pcm("others", sys_part)
            mic_rms, sys_rms = rms_int16(chunk), rms_int16(sys_part)
            self._levels = (min(1.0, mic_rms / 32768.0), min(1.0, sys_rms / 32768.0))
            self._ring.add(start_pos, mic_rms, sys_rms)
            speech = self._gate(mixed, start_pos)
            if speech and self._callback is not None:
                self._callback(speech)
        except Exception as exc:  # noqa: BLE001 - recorder would swallow it (spec §3.2)
            if self.fault is None:
                self.fault = exc
                logger.error("Meeting capture fault: {}", exc)

    def _gate(self, mixed: bytes, start_pos: float) -> bytes:
        # Carry any trailing partial slice from the previous call forward
        # instead of dropping it (spec §3.2 exactness: no audio is discarded
        # from the dictation stream just because a chunk crosses a 640-byte
        # boundary).
        data = bytes(self._gate_carry) + mixed
        data_start_pos = start_pos - len(self._gate_carry) / BYTES_PER_S
        out = bytearray()
        complete_end = (len(data) // FRAME_BYTES) * FRAME_BYTES
        for i in range(0, complete_end, FRAME_BYTES):
            frame = data[i : i + FRAME_BYTES]
            frame_pos = data_start_pos + (i // FRAME_BYTES) * FRAME_S
            is_speech = True if self._vad is None else bool(
                self._vad.is_speech(frame, self.sample_rate)
            )
            if is_speech:
                if self._open_run is None:
                    self._open_run = SpeechRun(max(0.0, frame_pos - len(self._preroll) * FRAME_S))
                    for buffered in self._preroll:
                        out.extend(buffered)
                    self._preroll.clear()
                out.extend(frame)
                self.last_speech_position_s = frame_pos + FRAME_S
            else:
                self._preroll.append(frame)
                if (
                    self._open_run is not None
                    and frame_pos + FRAME_S - self.last_speech_position_s
                    >= self._silence_threshold_s
                ):
                    self._close_open_run()
        self._gate_carry = bytearray(data[complete_end:])
        return bytes(out)

    def _close_open_run(self) -> None:
        if self._open_run is not None:
            self._open_run.end_s = self.last_speech_position_s
            self._runs.append(self._open_run)
            self._open_run = None
