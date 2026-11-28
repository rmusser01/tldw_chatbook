"""Qualify ffplay with synthetic WAVs; loopback samples live in RAM only.

Run without --capture-loopback for numerical checks only. SoundCard is an
optional isolated helper, never an application dependency. Native capture is
owned by a killable process; no microphone fallback is permitted.
"""

import argparse
from bisect import bisect_right
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import warnings

import numpy as np
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt
from pulse_fragments import CLOCK_REASON, Fragment, PulseError, capture_fragments
from wasapi_packets import MAX_POLL_GAP_S, content_span, drain_recorder, quiet_window

RATE = 48000
BLOCK = 480
STOP_BOUND_S = 1.5
TAIL_TOLERANCE_S = 0.020
STATS = re.compile(
    rb"^\s*(\d+(?:\.\d+)?)\s+(?:M-A|A-V):\s*[+-]?(?:\d+(?:\.\d+)?|nan)\s+fd=\s*\d+\s+aq="
)


@dataclass(frozen=True)
class ProgressObservation:
    position_s: float
    observed_at_s: float
    generation: int


def parse_stats_position(record: bytes) -> float | None:
    """Return a finite, nonnegative media clock from a stats record only."""
    match = STATS.match(record)
    if match is None:
        return None
    value = float(match[1])
    return value if math.isfinite(value) and value >= 0 else None


class StatsReader:
    """Retain at most one bounded partial record; discard oversize records."""

    def __init__(self, max_record_bytes: int = 1024):
        self.limit = max_record_bytes
        self.pending = bytearray()
        self.discarding = False
        self.generation = None

    @property
    def retained_bytes(self) -> int:
        return len(self.pending)

    def feed(
        self, data: bytes, *, generation: int, observed_at_s: float
    ) -> list[ProgressObservation]:
        if generation != self.generation:
            self.pending.clear()
            self.discarding = False
            self.generation = generation
        observations = []
        for byte in data:
            if byte in (10, 13):
                if not self.discarding:
                    value = parse_stats_position(bytes(self.pending))
                    if value is not None and math.isfinite(observed_at_s):
                        observations.append(
                            ProgressObservation(value, observed_at_s, generation)
                        )
                self.pending.clear()
                self.discarding = False
            elif not self.discarding:
                if len(self.pending) >= self.limit:
                    self.pending.clear()
                    self.discarding = True
                else:
                    self.pending.append(byte)
        return observations


def select_fresh(
    observations: list[ProgressObservation],
    *,
    generation: int,
    now_s: float,
    max_age_s: float = 0.5,
) -> ProgressObservation | None:
    """Select only current-generation observations within the freshness bound."""
    valid = [
        item
        for item in observations
        if item.generation == generation
        and math.isfinite(item.position_s)
        and item.position_s >= 0
        and 0 <= now_s - item.observed_at_s <= max_age_s
    ]
    return max(valid, key=lambda item: item.observed_at_s, default=None)


def marker_signal(
    duration_s: float, sample_rate: int = RATE, seed: int = 41176
) -> np.ndarray:
    """Return a deterministic, low-volume, unique band-limited noise marker."""
    samples = round(duration_s * sample_rate)
    noise = np.random.default_rng(seed).normal(size=samples)
    signal = sosfiltfilt(
        butter(4, [550, 5500], btype="bandpass", fs=sample_rate, output="sos"),
        noise,
    )
    signal *= 0.018 / np.max(np.abs(signal))
    return signal


def _window_match(window: np.ndarray, source: np.ndarray) -> tuple[int, float]:
    window = window - np.mean(window)
    numer = correlate(source, window, mode="valid", method="fft")
    sums = np.concatenate(([0.0], np.cumsum(source)))
    squares = np.concatenate(([0.0], np.cumsum(source * source)))
    count = len(window)
    energies = squares[count:] - squares[:-count]
    energies -= (sums[count:] - sums[:-count]) ** 2 / count
    denom = np.sqrt(np.maximum(energies, 0) * np.dot(window, window))
    scores = numer / np.maximum(denom, 1e-20)
    index = int(np.argmax(scores))
    return index, float(scores[index])


def measure_match(
    captured: np.ndarray,
    source: np.ndarray,
    *,
    capture_rate: int = RATE,
    source_rate: int = RATE,
    diagnostics: dict | None = None,
) -> dict | None:
    metrics = {} if diagnostics is None else diagnostics
    metrics.clear()
    started = time.perf_counter()
    try:
        return _measure_match(
            captured,
            source,
            capture_rate=capture_rate,
            source_rate=source_rate,
            diagnostics=metrics,
        )
    finally:
        metrics["duration_s"] = time.perf_counter() - started


def _measure_match(captured, source, *, capture_rate, source_rate, diagnostics):
    """Match independent early/tail windows; silence and unrelated audio refuse.

    Positions are measured from synthetic content, never from ffplay telemetry
    or process exit. A missing tail cannot certify completion.
    """
    captured = np.asarray(captured, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    diagnostics["stage"] = "invalid_source"
    if source.ndim != 1 or not np.isfinite(source).all():
        return None
    if captured.ndim == 2:
        captured = captured.mean(axis=1)
    if capture_rate != source_rate:
        divisor = math.gcd(capture_rate, source_rate)
        captured = resample_poly(
            captured, source_rate // divisor, capture_rate // divisor
        )
    diagnostics["stage"] = "invalid_capture"
    if len(captured) < source_rate * 0.03 or not np.isfinite(captured).all():
        return None
    envelope = np.convolve(captured * captured, np.ones(240) / 240, mode="same")
    active = np.flatnonzero(envelope > max(1e-9, float(envelope.max()) * 0.025))
    diagnostics.update(
        stage="insufficient_active_content",
        active_frames=len(active),
        capture_frames=len(captured),
    )
    if len(active) < source_rate * 0.03:
        return None
    start, end = int(active[0]), int(active[-1]) + 1
    # Stay inside the active region so boundary silence cannot inflate scores.
    margin = round(source_rate * 0.008)
    length = min(round(source_rate * 0.045), (end - start - 2 * margin) // 2)
    diagnostics.update(
        stage="window_too_short",
        active_start_frame=start,
        active_end_frame=end,
        window_frames=length,
    )
    if length < source_rate * 0.012 or len(source) < length:
        return None
    early = start + margin
    late = end - margin - length
    first, first_score = _window_match(captured[early : early + length], source)
    last, last_score = _window_match(captured[late : late + length], source)
    drift_s = ((last - first) - (late - early)) / source_rate
    diagnostics.update(
        stage="low_correlation",
        early_score=first_score,
        tail_score=last_score,
        early_source_frame=first,
        tail_source_frame=last,
        early_capture_frame=early,
        tail_capture_frame=late,
        alignment_drift_s=drift_s,
    )
    if min(first_score, last_score) < 0.70:
        return None
    diagnostics["stage"] = "tail_before_early"
    if last < first:
        return None
    # Independent alignments must also agree on how much content elapsed.
    diagnostics["stage"] = "alignment_drift"
    if abs(drift_s) > 0.020:
        return None
    diagnostics["stage"] = "matched"
    return {
        "start_source_s": (first - (early - start)) / source_rate,
        "end_source_s": (last + length + (end - late - length)) / source_rate,
        "capture_onset_s": start / source_rate,
        "capture_end_s": end / source_rate,
        "early_score": first_score,
        "tail_score": last_score,
        "alignment_drift_s": drift_s,
    }


def _count_capture_warning(metrics: dict, message) -> None:
    metrics["warning_count"] += 1
    if "discontinuity" in str(message).lower():
        metrics["discontinuity_count"] += 1


def _capture_worker(
    connection,
    duration_s: float,
    capture_buffer_frames: int = BLOCK,
    capture_clock: str = "legacy",
    finish=None,
    player_pid=None,
) -> None:
    """Capture only the exact render loopback; emit sanitized numerical telemetry."""
    output = open(os.devnull, "w")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = output
    metrics = {
        "warning_count": 0,
        "discontinuity_count": 0,
        "completed_sends": 0,
        "send_wait_max_s": 0.0,
        "send_wait_total_s": 0.0,
        "record_max_s": 0.0,
        "block_gap_max_s": 0.0,
    }
    previous_stamp = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = (
                lambda message, *args, **kwargs: _count_capture_warning(
                    metrics, message
                )
            )
            if capture_clock == "pulse-fragment":
                capture_fragments(
                    connection, duration_s, capture_buffer_frames, finish, player_pid
                )
                return
            import soundcard as sc

            speaker = sc.default_speaker()
            if speaker is None:
                connection.send(("error", "no_default_render"))
                return
            loopback = sc.get_microphone(id=speaker.id, include_loopback=True)
            if not loopback.isloopback or loopback.id != speaker.id:
                connection.send(("error", "no_exact_render_loopback"))
                return
            with loopback.recorder(
                samplerate=RATE, channels=2, blocksize=capture_buffer_frames
            ) as recorder:
                if capture_clock == "wasapi-packet":
                    drain_recorder(
                        connection, recorder, duration_s, capture_buffer_frames
                    )
                    return
                try:
                    actual_frames = int(recorder.buffersize)
                except Exception:
                    actual_frames = None
                connection.send(
                    (
                        "ready",
                        time.perf_counter(),
                        {
                            "actual_buffer_frames": actual_frames,
                            "requested_buffer_frames": capture_buffer_frames,
                            "emitted_block_frames": BLOCK,
                        },
                    )
                )
                deadline = time.perf_counter() + duration_s
                while time.perf_counter() < deadline:
                    record_started = time.perf_counter()
                    data = recorder.record(numframes=BLOCK)
                    stamp = time.perf_counter()
                    metrics["record_max_s"] = max(
                        metrics["record_max_s"], stamp - record_started
                    )
                    if previous_stamp is not None:
                        metrics["block_gap_max_s"] = max(
                            metrics["block_gap_max_s"], stamp - previous_stamp
                        )
                    previous_stamp = stamp
                    # The next block reports this send only after it completes.
                    connection.send(("block", stamp, data, dict(metrics)))
                    elapsed = time.perf_counter() - stamp
                    metrics["completed_sends"] += 1
                    metrics["send_wait_max_s"] = max(
                        metrics["send_wait_max_s"], elapsed
                    )
                    metrics["send_wait_total_s"] += elapsed
    except Exception as error:
        try:
            connection.send(
                (
                    "error",
                    str(error) if isinstance(error, PulseError) else "capture_failed",
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        output.close()


def _stop_process(process: subprocess.Popen) -> bool:
    """Terminate, then kill within separate bounded waits; always reap."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=STOP_BOUND_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=STOP_BOUND_S)
    return process.poll() is not None


def _stats_pump(pipe, observations, generation: int) -> None:
    reader = StatsReader()
    try:
        while data := pipe.read(4096):
            observations.extend(
                reader.feed(
                    data, generation=generation, observed_at_s=time.perf_counter()
                )
            )
    except (OSError, ValueError):
        pass


def resolve_ffplay(explicit: str | None) -> Path:
    """Prefer real Chocolatey payloads to shims; explicit paths must be absolute."""
    default_install = (
        Path(os.environ.get("ProgramData", "C:/ProgramData")) / "chocolatey"
    )
    install = Path(os.environ.get("ChocolateyInstall", str(default_install)))
    if explicit:
        path = Path(explicit)
        if not path.is_absolute() or not path.is_file():
            raise ValueError("ffplay_requires_existing_absolute_path")
    else:
        candidates = (
            sorted((install / "lib").glob("ffmpeg*/tools/**/ffplay.exe"))
            if os.name == "nt"
            else []
        )
        found = candidates[0] if candidates else shutil.which("ffplay")
        if not found:
            raise ValueError("real_ffplay_not_found_supply_absolute_path")
        path = Path(found)
    # Normalize links and '..' before checking every entry path. A shim's
    # launcher PID does not establish ownership of its real playback child.
    resolved = path.resolve()
    if resolved.parent in {
        (default_install / "bin").resolve(),
        (install / "bin").resolve(),
    }:
        raise ValueError("ffplay_shim_refused_supply_real_payload_path")
    return resolved


def _build_receipt(executable: Path) -> dict:
    digest = hashlib.sha256()
    with executable.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    process = None
    try:
        process = subprocess.Popen(
            [str(executable), "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # Reader drains but only retains a bounded prefix; version text stays RAM.
        prefix = bytearray()

        def drain():
            while chunk := process.stdout.read(1024):
                prefix.extend(chunk[: max(0, 4096 - len(prefix))])

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        process.wait(timeout=5)
        reader.join(timeout=1)
        match = re.search(rb"ffplay version ([A-Za-z0-9.+_\-]+)", prefix)
        version = match[1].decode("ascii") if match else "unknown"
        return {"sha256": digest.hexdigest(), "version": version}
    finally:
        try:
            if process is not None:
                _stop_process(process)
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            if "reader" in locals():
                reader.join(timeout=1)


def stop_is_due(
    *,
    now_s: float,
    launched_at_s: float,
    playing_at_s: float | None,
    delay_s: float,
    wait_for_playing: bool,
) -> bool:
    anchor = playing_at_s if wait_for_playing else launched_at_s
    return anchor is not None and now_s - anchor >= delay_s


def observed_playing(
    blocks: list,
    source: np.ndarray,
    *,
    launched_at_s: float,
    now_s: float,
    diagnostics: dict | None = None,
) -> dict | None:
    """Require recent post-launch loopback content, using the unchanged oracle."""
    recent = [
        (stamp, data) for stamp, data in blocks[-20:] if launched_at_s <= stamp <= now_s
    ]
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics["stage"] = "no_recent_postlaunch_capture"
    if not recent or now_s - recent[-1][0] > 0.5:
        return None
    return measure_match(
        np.concatenate([data for _, data in recent]), source, diagnostics=diagnostics
    )


def bookmark_alignment(
    frame: int | None,
    source_end_s: float | None,
    *,
    content_duration_s: float | None = None,
) -> dict:
    """Compare saved progress with independently captured stopped content."""
    if frame is None or source_end_s is None or not math.isfinite(source_end_s):
        return {"bookmark_alignment": "unqualified", "bookmark_end_error_s": None}
    error = frame / RATE - source_end_s
    if (
        content_duration_s is not None
        and source_end_s > content_duration_s - TAIL_TOLERANCE_S
    ):
        return {
            "bookmark_alignment": "unqualified",
            "bookmark_end_error_s": error,
            "bookmark_alignment_reason": "stop_at_logical_eof",
        }
    return {
        "bookmark_alignment": "pass" if abs(error) <= 0.250 else "fail",
        "bookmark_end_error_s": error,
    }


def saved_bookmark_frame(
    observations: list[ProgressObservation],
    *,
    generation: int,
    now_s: float,
    content_frames: int,
) -> int | None:
    """Accept one fresh content clock and convert it once to a PCM frame."""
    fresh = select_fresh(observations, generation=generation, now_s=now_s)
    if fresh is None or fresh.position_s > content_frames / RATE:
        return None
    return min(round(fresh.position_s * RATE), content_frames)


def _write_wav(path: Path, source: np.ndarray, *, tail_padding_ms: int = 0) -> None:
    if tail_padding_ms not in (0, 250):
        raise ValueError("padding_must_be_zero_or_250ms")
    pcm = np.repeat((source * 32767).astype("<i2")[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as output:
        output.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        output.writeframes(pcm.tobytes())
        output.writeframes(b"\x00" * (tail_padding_ms * RATE // 1000) * 4)


def artifact_receipt(wav: Path, content_frames: int) -> dict:
    """Describe physical frames separately from original spoken-content frames."""
    with wave.open(str(wav), "rb") as reader:
        file_frames = reader.getnframes()
    return {
        "artifact_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "content_frames": content_frames,
        "content_duration_s": content_frames / RATE,
        "file_frames": file_frames,
        "file_duration_s": file_frames / RATE,
        "padding_frames": file_frames - content_frames,
        "sample_rate_hz": RATE,
    }


def _match_packets(packets, source):
    metrics = {}
    span = content_span(packets, metrics)
    if span is None:
        return None, metrics
    data, qpc_start = span
    matcher = {}
    match = measure_match(data, source, diagnostics=matcher)
    metrics.update(matcher)
    if match is not None:
        match["capture_onset_qpc_s"] = qpc_start + match["capture_onset_s"]
        match["capture_end_qpc_s"] = qpc_start + match["capture_end_s"]
    return match, metrics


def _match_fragments(fragments, source):
    metrics = {}
    if not fragments:
        return None, {"stage": "no_pulse_fragments"}
    data = np.concatenate([fragment.data for fragment in fragments])
    match = measure_match(data, source, diagnostics=metrics)
    if match is not None:
        match["capture_onset_frame"] = fragments[0].frame + round(
            match["capture_onset_s"] * RATE
        )
        match["capture_end_frame"] = fragments[0].frame + round(
            match["capture_end_s"] * RATE
        )
    return match, metrics


def _recent_fragments(fragments, first_postlaunch):
    if first_postlaunch == len(fragments):
        return []
    last = fragments[-1]
    start = max(
        fragments[first_postlaunch].frame,
        last.frame + len(last.data) - round(0.2 * RATE),
    )
    first = (
        bisect_right(
            fragments, start, lo=first_postlaunch, key=lambda fragment: fragment.frame
        )
        - 1
    )
    recent = fragments[first:]
    recent[0] = Fragment(start, recent[0].data[start - recent[0].frame :])
    return recent


def _fragment_quiet(fragments, tail_frames=None):
    result = {"status": "unqualified", "rms": None, "frames": 0}
    if not fragments:
        return result
    data = np.concatenate([fragment.data for fragment in fragments])
    if tail_frames is not None:
        if len(data) < tail_frames:
            return result
        data = data[-tail_frames:]
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    return {
        "status": "pass" if rms <= 0.0002 else "fail",
        "rms": rms,
        "frames": len(data),
    }


def run_case(
    executable: Path,
    wav: Path,
    source: np.ndarray,
    *,
    offset: float,
    generation: int,
    stop_after: float | None = None,
    padded: bool = False,
    capture_buffer_frames: int = BLOCK,
    capture_clock: str = "legacy",
    stop_clock: str | None = None,
) -> dict:
    """Measure one native run, optionally overriding the default Stop clock."""
    if capture_buffer_frames not in (480, 4800):
        raise ValueError("capture_buffer_frames_must_be_480_or_4800")
    if capture_clock not in ("legacy", "wasapi-packet", "pulse-fragment"):
        raise ValueError("unknown_capture_clock")
    if stop_clock is None:
        stop_clock = "observed_playing" if padded else "launch"
    if stop_clock not in ("launch", "observed_playing"):
        raise ValueError("unknown_stop_clock")
    wait_for_playing = stop_clock == "observed_playing"
    packet_mode = capture_clock == "wasapi-packet"
    pulse_mode = capture_clock == "pulse-fragment"
    native_mode = packet_mode or pulse_mode
    if pulse_mode and platform.system() != "Linux":
        raise ValueError("pulse_requires_linux")
    match_playing = wait_for_playing and (not native_mode or stop_after is not None)
    ready_budget = 30 if padded else 6
    playback_budget = len(source) / RATE + 6 + (0.25 if padded else 0)
    result = {
        "launched": False,
        "generation": generation,
        "duration_s": len(source) / RATE,
        "requested_start_s": offset,
        "stop_after_launch_s": None if wait_for_playing else stop_after,
        "stop_after_playing_s": stop_after if wait_for_playing else None,
        "stop_after_s": stop_after,
        "stop_clock": stop_clock,
        "capture_ready_timeout_s": 30 if padded else 8,
        "playing_ready_timeout_s": ready_budget if wait_for_playing else None,
        "capture_lifetime_s": 45 if padded else 12,
        "max_capture_blocks": 4500 if padded else 1800,
        "capture_buffer_frames": capture_buffer_frames,
        "capture_clock": capture_clock,
        "capture_started": False,
        "diagnostics": {
            "live_matcher_calls": 0,
            "live_matcher_max_s": 0.0,
            "receive_backlog_max_s": 0.0,
            "received_block_gap_max_s": 0.0,
        },
        "endpoint_loopback": "unqualified",
        "seek": "unqualified",
        "tail_coverage": "unqualified",
        "telemetry": "unqualified",
        "process_exit": "unqualified",
        "post_stop_silence": "unqualified",
    }
    context = mp.get_context("spawn")
    finish = context.Event() if pulse_mode else None
    player_pid = context.RawValue("I", 0) if pulse_mode else None
    if pulse_mode:
        result["telemetry_reason"] = CLOCK_REASON
        if stop_after is not None:
            result.update(
                post_stop_latency="unqualified",
                post_stop_latency_reason=CLOCK_REASON,
                post_stop_silence_reason=CLOCK_REASON,
            )
    receiver, sender = context.Pipe(duplex=False)
    capture_args = (sender, result["capture_lifetime_s"], capture_buffer_frames)
    if native_mode:
        capture_args += (capture_clock,)
    if pulse_mode:
        capture_args += (finish, player_pid)
    capture = context.Process(
        target=_capture_worker,
        args=capture_args,
        daemon=True,
    )
    process = None
    reader = None
    blocks = []
    packets = []
    fragments = []
    baseline_count = 0
    exit_frame_count = 0
    pulse_complete = False
    polls = []
    packet_frames = 0
    live_executor = None
    live_future = None
    observations = deque(maxlen=256)
    stop_at = None
    exited_at = None
    launched_at = None
    playing_at = None
    playing_ready_deadline = None
    last_checked_blocks = 0
    capture_started = False
    try:
        capture.start()
        capture_started = True
        result["capture_started"] = True
        sender.close()
        if not receiver.poll(result["capture_ready_timeout_s"]):
            result["reason"] = "capture_ready_timeout"
            return result
        ready = receiver.recv()
        if ready[0] != "ready":
            result["reason"] = ready[1]
            return result
        if len(ready) > 2:
            result["diagnostics"]["capture_configuration"] = ready[2]
        polls.append(ready[1])
        if native_mode and match_playing:
            live_executor = ThreadPoolExecutor(max_workers=1)
        # Independent pre-launch baseline; no output names or samples are saved.
        baseline_duration = 0.35
        baseline_until = time.perf_counter() + baseline_duration
        deadline = (
            time.perf_counter() + playback_budget + (ready_budget if padded else 0)
        )
        while time.perf_counter() < deadline:
            while receiver.poll():
                try:
                    message = receiver.recv()
                except EOFError:
                    if pulse_mode and not pulse_complete:
                        result["reason"] = "capture_monitor_ended_early"
                        return result
                    break
                if message[0] == "error":
                    result["reason"] = message[1]
                    return result
                if message[0] == "pulse_complete" and pulse_mode:
                    if not finish.is_set():
                        result["reason"] = "capture_monitor_ended_early"
                        return result
                    result["player_route"] = message[1]
                    pulse_complete = True
                    break
                if message[0] == "fragment" and pulse_mode:
                    _, stamp, fragment, worker_metrics = message
                    if (
                        fragments
                        and fragments[-1].frame + len(fragments[-1].data)
                        != fragment.frame
                    ):
                        result["reason"] = "pulse_read_index_gap"
                        return result
                    fragments.append(fragment)
                    packet_frames += len(fragment.data)
                    if (
                        len(fragments) > 10000
                        or packet_frames > RATE * result["capture_lifetime_s"]
                    ):
                        result["reason"] = "capture_bound_exceeded"
                        return result
                    result.update(
                        capture_fragments=len(fragments),
                        capture_frames=packet_frames,
                        capture_first_frame=fragments[0].frame,
                        capture_end_frame=fragment.frame + len(fragment.data),
                    )
                    metrics = result["diagnostics"]
                    metrics["worker"] = worker_metrics
                    metrics["receive_backlog_max_s"] = max(
                        metrics["receive_backlog_max_s"], time.perf_counter() - stamp
                    )
                if message[0] == "packets":
                    _, stamp, batch, worker_metrics = message
                    metrics = result["diagnostics"]
                    metrics["worker"] = worker_metrics
                    metrics["receive_backlog_max_s"] = max(
                        metrics["receive_backlog_max_s"], time.perf_counter() - stamp
                    )
                    polls.append(stamp)
                    packets.extend(batch)
                    packet_frames += sum(len(packet.data) for packet in batch)
                    if (
                        len(polls) > 10000
                        or packet_frames > RATE * result["capture_lifetime_s"]
                    ):
                        result["reason"] = "capture_bound_exceeded"
                        return result
                if message[0] == "block":
                    metrics = result["diagnostics"]
                    received_at = time.perf_counter()
                    metrics["receive_backlog_max_s"] = max(
                        metrics["receive_backlog_max_s"], received_at - message[1]
                    )
                    if blocks:
                        metrics["received_block_gap_max_s"] = max(
                            metrics["received_block_gap_max_s"],
                            message[1] - blocks[-1][0],
                        )
                    if len(message) > 3:
                        metrics["worker"] = message[3]
                        metrics["send_metrics_through_previous_block"] = True
                    blocks.append((message[1], message[2]))
                    if len(blocks) > result["max_capture_blocks"]:
                        result["reason"] = "capture_bound_exceeded"
                        return result
            now = time.perf_counter()
            if process is None and now >= baseline_until:
                if (
                    pulse_mode
                    and packet_frames < round(RATE * baseline_duration)
                    and now < baseline_until + 1
                ):
                    time.sleep(0.005)
                    continue
                baseline_count = len(fragments)
                player_options = (
                    {
                        "env": {
                            **os.environ,
                            "PULSE_SINK": ready[3],
                            "SDL_AUDIODRIVER": "pulseaudio",
                        }
                    }
                    if pulse_mode
                    else {}
                )
                launched_at = now
                process = subprocess.Popen(
                    [
                        str(executable),
                        "-nodisp",
                        "-autoexit",
                        "-stats",
                        "-sync",
                        "audio",
                        "-ss",
                        f"{offset:.6f}",
                        str(wav),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    **player_options,
                )
                result["launched"] = True
                if pulse_mode:
                    player_pid.value = process.pid
                if padded or match_playing:
                    ready_by = time.perf_counter() + ready_budget
                    if padded:
                        deadline = ready_by + playback_budget
                    if match_playing:
                        playing_ready_deadline = ready_by
                reader = threading.Thread(
                    target=_stats_pump,
                    args=(process.stderr, observations, generation),
                    daemon=True,
                )
                reader.start()
            if process is not None:
                if live_executor is not None and playing_at is None:
                    if live_future is not None and live_future.done():
                        match, live_metrics = live_future.result()
                        live_future = None
                        metrics = result["diagnostics"]
                        metrics["live_matcher_last"] = live_metrics
                        metrics["live_matcher_max_s"] = max(
                            metrics["live_matcher_max_s"],
                            live_metrics.get("duration_s", 0),
                        )
                        if match is not None:
                            playing_at = time.perf_counter()
                            result.update(
                                playing_observed_at_s=playing_at,
                                playing_startup_wait_s=playing_at - launched_at,
                                playing_source_position_s=match["end_source_s"],
                            )
                            deadline = playing_at + playback_budget
                    if (
                        live_future is None
                        and playing_at is None
                        and len(fragments if pulse_mode else packets)
                        > last_checked_blocks
                    ):
                        if pulse_mode:
                            last_checked_blocks = len(fragments)
                            recent = _recent_fragments(fragments, baseline_count)
                            matcher = _match_fragments
                        else:
                            last_checked_blocks = len(packets)
                            recent = [
                                packet
                                for packet in packets[-32:]
                                if packet.start_s >= max(launched_at, now - 0.20)
                            ]
                            matcher = _match_packets
                        if recent:
                            live_future = live_executor.submit(matcher, recent, source)
                            result["diagnostics"]["live_matcher_calls"] += 1
                if (
                    match_playing
                    and not native_mode
                    and playing_at is None
                    and len(blocks) - last_checked_blocks >= 4
                ):
                    last_checked_blocks = len(blocks)
                    live_metrics = {}
                    match_started = time.perf_counter()
                    match = observed_playing(
                        blocks,
                        source,
                        launched_at_s=launched_at,
                        now_s=time.perf_counter(),
                        diagnostics=live_metrics,
                    )
                    metrics = result["diagnostics"]
                    metrics["live_matcher_calls"] += 1
                    metrics["live_matcher_max_s"] = max(
                        metrics["live_matcher_max_s"],
                        time.perf_counter() - match_started,
                    )
                    metrics["live_matcher_last"] = live_metrics
                    if match is not None:
                        playing_at = time.perf_counter()
                        result["playing_observed_at_s"] = playing_at
                        result["playing_startup_wait_s"] = playing_at - launched_at
                        result["playing_source_position_s"] = match["end_source_s"]
                        deadline = playing_at + playback_budget
                if (
                    playing_ready_deadline is not None
                    and playing_at is None
                    and time.perf_counter() >= playing_ready_deadline
                    and process.poll() is None
                ):
                    result["reason"] = "playing_ready_timeout"
                    break
                if (
                    stop_after is not None
                    and stop_at is None
                    and (not native_mode or process.poll() is None)
                    and stop_is_due(
                        now_s=time.perf_counter(),
                        launched_at_s=launched_at,
                        playing_at_s=playing_at,
                        delay_s=stop_after,
                        wait_for_playing=wait_for_playing,
                    )
                ):
                    stop_at = time.perf_counter()
                    result["stop_requested_after_launch_s"] = stop_at - launched_at
                    frame = saved_bookmark_frame(
                        list(observations),
                        generation=generation,
                        now_s=stop_at,
                        content_frames=len(source),
                    )
                    result["saved_bookmark_frame"] = frame
                    result["saved_bookmark_s"] = (
                        frame / RATE if frame is not None else None
                    )
                    result["bookmark_sample_rate_hz"] = RATE
                    result["bookmark"] = "pass" if frame is not None else "unqualified"
                    _stop_process(process)
                    result["stop_process_latency_s"] = time.perf_counter() - stop_at
                if process.poll() is not None and exited_at is None:
                    exited_at = time.perf_counter()
                    exit_frame_count = packet_frames
                    result["returncode"] = process.returncode
                    result["process_exit"] = (
                        "pass"
                        if stop_at is not None or process.returncode == 0
                        else "fail"
                    )
                if exited_at is not None and now - exited_at >= 0.6:
                    if not pulse_mode or pulse_complete:
                        break
                    if not finish.is_set():
                        finish.set()
                        deadline = now + STOP_BOUND_S
            time.sleep(0.005)
        else:
            if pulse_mode and finish.is_set():
                result["reason"] = "capture_monitor_ended_early"
                return result
            result["reason"] = (
                "playing_ready_timeout"
                if match_playing and playing_at is None
                else "playback_deadline"
            )
            if result["reason"] == "playback_deadline":
                result["process_exit"] = "fail"
        if pulse_mode:
            baseline = _fragment_quiet(fragments[:baseline_count])
            if baseline["frames"] < round(RATE * baseline_duration):
                baseline["status"] = "unqualified"
            result.update(baseline_monitor=baseline, baseline_rms=baseline["rms"])
            if baseline["status"] != "pass":
                result["endpoint_loopback"] = baseline["status"]
                result["reason"] = (
                    "noisy_baseline"
                    if baseline["status"] == "fail"
                    else "baseline_unobserved"
                )
                return result
            result["frames_received_after_exit"] = packet_frames - exit_frame_count
            if (
                not pulse_complete
                or exited_at is None
                or packet_frames - exit_frame_count < round(RATE * 0.4)
            ):
                result.setdefault("reason", "capture_monitor_ended_early")
                return result
            playback_fragments = fragments[baseline_count:]
            measurement, final_metrics = _match_fragments(playback_fragments, source)
            quiet_suffix = _fragment_quiet(
                playback_fragments, tail_frames=round(RATE * 0.2)
            )
            if stop_at is not None:
                result["engine_quiet_samples"] = quiet_suffix
            end_observed = False
            if measurement is not None:
                quiet_frames = (
                    result["capture_end_frame"] - measurement["capture_end_frame"]
                )
                end_observed = (
                    quiet_frames >= quiet_suffix["frames"]
                    and quiet_suffix["status"] == "pass"
                )
            if result["player_route"] != "verified":
                diagnostic = {
                    "attribution": "unattributed",
                    "matcher_stage": final_metrics["stage"],
                    "end_boundary": "observed" if end_observed else "censored",
                }
                if measurement is not None:
                    for key in (
                        "start_source_s",
                        "end_source_s",
                        "early_score",
                        "tail_score",
                        "alignment_drift_s",
                    ):
                        diagnostic[key] = measurement[key]
                result["diagnostics"]["unattributed_source_match"] = diagnostic
                result["reason"] = result["player_route"]
                return result
            if measurement is not None:
                result["source_end_boundary"] = (
                    "pass" if end_observed else "unqualified"
                )
        elif packet_mode:
            result.update(
                capture_packets=len(packets),
                capture_frames=packet_frames,
                capture_poll_observations=len(polls),
            )
            if launched_at is None:
                result.setdefault("reason", "player_not_launched")
                return result
            config = result["diagnostics"]["capture_configuration"]
            max_gap = min(MAX_POLL_GAP_S, config["actual_buffer_frames"] / RATE)
            packet_period = config["device_period_s"]
            baseline = quiet_window(
                packets,
                polls,
                ready[1],
                launched_at,
                max_gap_s=max_gap,
                drain_delay_s=packet_period,
            )
            result["baseline_monitor"] = baseline
            result["baseline_rms"] = baseline["rms"]
            if baseline["status"] != "pass":
                result["reason"] = (
                    "noisy_baseline"
                    if baseline["status"] == "fail"
                    else "baseline_unobserved"
                )
                result["endpoint_loopback"] = baseline["status"]
                return result
            if exited_at is not None and polls[-1] < exited_at + 0.4:
                result["reason"] = "capture_monitor_ended_early"
                return result
            playback_packets = [
                packet for packet in packets if packet.end_s > launched_at
            ]
            measurement, final_metrics = _match_packets(playback_packets, source)
            if stop_at is not None and exited_at is not None:
                post = quiet_window(
                    packets,
                    polls,
                    stop_at + 0.2,
                    exited_at + 0.45,
                    max_gap_s=max_gap,
                    drain_delay_s=packet_period,
                )
                result["post_stop_monitor"] = post
                result["post_stop_silence"] = post["status"]
                result["post_stop_rms"] = post["rms"]
                if post["status"] == "pass" and (
                    measurement is not None
                    or not any(np.any(packet.data) for packet in playback_packets)
                ):
                    last_active = max(
                        (
                            packet.end_s
                            for packet in playback_packets
                            if np.sqrt(np.mean(packet.data**2)) >= 0.0002
                        ),
                        default=stop_at,
                    )
                    result["last_signal_after_stop_s"] = max(0, last_active - stop_at)
                    result["post_stop_latency"] = (
                        "pass"
                        if result["last_signal_after_stop_s"] <= STOP_BOUND_S
                        else "fail"
                    )
        else:
            if not blocks:
                result["reason"] = "no_loopback_samples"
                return result
            baseline = [data for stamp, data in blocks if stamp <= launched_at]
            result["baseline_blocks"] = len(baseline)
            if not baseline:
                result["reason"] = "baseline_unobserved"
                return result
            baseline_rms = float(np.sqrt(np.mean(np.concatenate(baseline) ** 2)))
            result["baseline_rms"] = baseline_rms
            if baseline_rms > 0.0002:
                result["reason"] = "noisy_baseline"
                result["endpoint_loopback"] = "fail"
                return result
            captured = np.concatenate([data for _, data in blocks])
            final_metrics = {}
            measurement = measure_match(captured, source, diagnostics=final_metrics)
        result["diagnostics"]["final_matcher"] = final_metrics
        if not pulse_mode:
            result["capture_blocks"] = len(blocks)
        result["observation_count_retained"] = len(observations)
        if stop_at is not None and not native_mode:
            post = [
                (stamp, data)
                for stamp, data in blocks
                if stamp - BLOCK / RATE >= stop_at + 0.2
            ]
            if len(post) >= 10:
                post_rms = float(
                    np.sqrt(np.mean(np.concatenate([data for _, data in post]) ** 2))
                )
                result["post_stop_rms"] = post_rms
                result["post_stop_silence"] = "pass" if post_rms < 0.0002 else "fail"
                last_active = max(
                    (
                        stamp
                        for stamp, data in blocks
                        if np.sqrt(np.mean(data * data)) >= 0.0002
                    ),
                    default=stop_at,
                )
                result["last_signal_after_stop_s"] = max(0, last_active - stop_at)
                result["post_stop_latency"] = (
                    "pass"
                    if result["last_signal_after_stop_s"] <= STOP_BOUND_S
                    else "fail"
                )
        if measurement is None:
            result.setdefault("reason", "marker_unmatched")
            # Startup Stop may correctly prevent any marker; it still cannot
            # certify seek/tail or establish that capture saw the actual output.
            return result
        result["endpoint_loopback"] = "pass"
        result["measurement"] = measurement
        error = measurement["start_source_s"] - offset
        result["seek_error_s"] = error
        result["seek"] = "pass" if abs(error) <= 0.250 else "fail"
        if pulse_mode and result["source_end_boundary"] != "pass":
            result["reason"] = "pulse_source_end_unobserved"
            if stop_after is not None:
                result["bookmark_alignment"] = "unqualified"
            return result
        if stop_after is not None:
            result.update(
                bookmark_alignment(
                    result.get("saved_bookmark_frame"),
                    measurement["end_source_s"],
                    content_duration_s=len(source) / RATE if native_mode else None,
                )
            )
        remaining = len(source) / RATE - measurement["end_source_s"]
        result["missing_tail_s"] = remaining
        if stop_after is None:
            result["tail_coverage"] = (
                "pass" if abs(remaining) <= TAIL_TOLERANCE_S else "fail"
            )
        if pulse_mode:
            return result
        # Map a matched sample to its acquisition block, independent of launch.
        if packet_mode:
            onset_wall = measurement["capture_onset_qpc_s"]
        else:
            onset_sample = round(measurement["capture_onset_s"] * RATE)
            onset_block = min(onset_sample // BLOCK, len(blocks) - 1)
            onset_wall = blocks[onset_block][0] - (BLOCK - onset_sample % BLOCK) / RATE
        end_wall = (
            onset_wall + measurement["capture_end_s"] - measurement["capture_onset_s"]
        )
        errors = []
        for observation in list(observations):
            fresh = select_fresh(
                [observation], generation=generation, now_s=observation.observed_at_s
            )
            if fresh and onset_wall <= fresh.observed_at_s <= end_wall:
                actual = (
                    measurement["start_source_s"] + fresh.observed_at_s - onset_wall
                )
                errors.append(fresh.position_s - actual)
        if errors:
            result["telemetry_compared_count"] = len(errors)
            result["telemetry_max_absolute_error_s"] = max(
                abs(error) for error in errors
            )
            result["telemetry"] = (
                "pass" if result["telemetry_max_absolute_error_s"] <= 0.250 else "fail"
            )
        return result
    except Exception as error:
        result["reason"] = "case_failed"
        result["error_type"] = type(error).__name__
        return result
    finally:
        if live_executor is not None:
            live_executor.shutdown(wait=False, cancel_futures=True)
        errors = result.setdefault("cleanup_errors", [])

        def cleanup(phase, operation, default=None):
            try:
                return operation()
            except Exception as error:
                errors.append({"phase": phase, "error_type": type(error).__name__})
                return default

        result["player_reaped"] = process is None or cleanup(
            "player_stop", lambda: _stop_process(process), False
        )
        if process is not None and process.stderr is not None:
            cleanup("stderr_close", process.stderr.close)
        if reader is not None:
            cleanup("reader_join", lambda: reader.join(timeout=1))
        cleanup("receiver_close", receiver.close)
        cleanup("sender_close", sender.close)
        if capture_started:
            if cleanup("capture_alive", capture.is_alive, True):
                cleanup("capture_terminate", capture.terminate)
            cleanup("capture_join", lambda: capture.join(timeout=STOP_BOUND_S))
            if cleanup("capture_alive", capture.is_alive, True):
                cleanup("capture_kill", capture.kill)
                cleanup("capture_kill_join", lambda: capture.join(timeout=STOP_BOUND_S))
            result["capture_reaped"] = not cleanup(
                "capture_final_alive", capture.is_alive, True
            )
            cleanup("capture_close", capture.close)
        else:
            result["capture_reaped"] = True
        result["cleanup_status"] = "fail" if errors else "pass"


def _retirement_unconfirmed(run: dict) -> bool:
    return (
        bool(run.get("cleanup_errors"))
        or any(run.get(key) is False for key in ("player_reaped", "capture_reaped"))
        or (run.get("launched") is True and run.get("player_reaped") is not True)
        or (
            run.get("capture_started") is True and run.get("capture_reaped") is not True
        )
    )


def _case_receipt(*args, **kwargs) -> dict:
    try:
        return run_case(*args, **kwargs)
    except Exception as error:
        return {
            "launched": None,
            "reason": "case_exception",
            "cleanup_errors": [
                {"phase": "case_boundary", "error_type": type(error).__name__}
            ],
            "cleanup_status": "fail",
        }


def run_matrix(
    executable: Path,
    scratch: Path,
    *,
    tail_padding_ms: int,
    repeat: int,
    runs: list[dict] | None = None,
    capture_buffer_frames: int = BLOCK,
    capture_clock: str = "legacy",
) -> list[dict]:
    """Reuse three stable artifacts; a resume consumes its confirmed Stop frame."""
    if (
        tail_padding_ms not in (0, 250)
        or repeat not in (1, 2, 3)
        or capture_buffer_frames not in (480, 4800)
        or capture_clock not in ("legacy", "wasapi-packet", "pulse-fragment")
    ):
        raise ValueError("invalid_matrix_settings")
    artifacts = {}
    for duration in (0.120, 0.500, 2.5):
        source = marker_signal(duration, seed=41176 + round(duration * RATE))
        wav = scratch / f"marker-{len(source)}.wav"
        _write_wav(wav, source, tail_padding_ms=tail_padding_ms)
        artifacts[duration] = (source, wav, artifact_receipt(wav, len(source)))
    cases = [
        ("start", 2.5, 0.0, None),
        ("resume_1", 2.5, 0.8, None),
        ("resume_1_repeat", 2.5, 0.8, None),
        ("resume_2", 2.5, 1.4, None),
        ("resume_2_repeat", 2.5, 1.4, None),
        ("short_120ms", 0.120, 0.0, None),
        ("short_500ms", 0.500, 0.0, None),
        ("near_eof", 2.5, 2.36, None),
        ("stop_startup", 2.5, 0.0, 0.020),
        ("stop_playing", 2.5, 0.0, 0.700),
    ]
    if tail_padding_ms:
        cases = [
            ("resume_fixed", 2.5, 0.8, None),
            ("short_120ms", 0.120, 0.0, None),
            ("short_500ms", 0.500, 0.0, None),
            ("near_eof_140ms", 2.5, 2.36, None),
            ("near_eof_60ms", 2.5, 2.44, None),
            ("stop_playing", 2.5, 0.0, 0.700),
        ]
    runs = [] if runs is None else runs
    clock_args = {"capture_clock": capture_clock} if capture_clock != "legacy" else {}
    for round_number in range(1, repeat + 1):
        for name, duration, offset, stop_after in cases:
            source, wav, artifact = artifacts[duration]
            generation = len(runs) + 1
            run = _case_receipt(
                executable,
                wav,
                source,
                offset=offset,
                generation=generation,
                stop_after=stop_after,
                padded=bool(tail_padding_ms),
                capture_buffer_frames=capture_buffer_frames,
                **clock_args,
            )
            run.update(
                artifact,
                case=name,
                round=round_number,
                generation=generation,
                duration_s=duration,
                requested_start_s=offset,
                stop_after_launch_s=stop_after if not tail_padding_ms else None,
                stop_after_playing_s=stop_after if tail_padding_ms else None,
                stop_after_s=stop_after,
            )
            runs.append(run)
            if _retirement_unconfirmed(run):
                run["matrix_aborted"] = True
                return runs
            if not tail_padding_ms or name != "stop_playing":
                continue
            frame = run.get("saved_bookmark_frame")
            safe_resume = (
                isinstance(frame, int)
                and 0 <= frame < len(source)
                and run.get("player_reaped") is True
                and run.get("capture_reaped") is True
                and not _capture_quality_unqualified(run)
                and all(
                    run.get(key) == "pass"
                    for key in (
                        "process_exit",
                        "endpoint_loopback",
                        "post_stop_silence",
                        "post_stop_latency",
                        "bookmark",
                        "bookmark_alignment",
                    )
                )
                and run.get("stop_process_latency_s", math.inf) <= STOP_BOUND_S
            )
            if safe_resume:
                resume = _case_receipt(
                    executable,
                    wav,
                    source,
                    offset=frame / RATE,
                    generation=generation + 1,
                    stop_after=None,
                    padded=True,
                    capture_buffer_frames=capture_buffer_frames,
                    **clock_args,
                )
            else:
                resume = {
                    "launched": False,
                    "reason": "saved_resume_preconditions_unqualified",
                }
            resume.update(
                artifact,
                case="resume_saved",
                round=round_number,
                generation=generation + 1,
                duration_s=duration,
                requested_start_s=frame / RATE if isinstance(frame, int) else None,
                stop_after_launch_s=None,
                stop_after_s=None,
                resume_from_generation=generation,
                resume_from_bookmark_frame=frame,
            )
            runs.append(resume)
            if _retirement_unconfirmed(resume):
                resume["matrix_aborted"] = True
                return runs
    return runs


def _capture_quality_unqualified(run: dict) -> bool:
    worker = run.get("diagnostics", {}).get("worker", {})
    return (
        worker.get("discontinuity_count", 0) > 0
        or worker.get("timestamp_error_count", 0) > 0
    )


def native_result(runs: list[dict]) -> str:
    """Every required case counts; unmatched cases and missing cleanup cannot pass."""
    required = []
    for run in runs:
        telemetry = run.get("telemetry", "unqualified")
        if telemetry == "fail" or run.get("capture_clock") == "pulse-fragment":
            required.append(telemetry)
        if _capture_quality_unqualified(run):
            required.append("unqualified")
        if run.get("cleanup_errors"):
            required.append("fail")
        required.append(run.get("process_exit", "unqualified"))
        for key in ("player_reaped", "capture_reaped"):
            value = run.get(key)
            required.append(
                "pass" if value is True else "fail" if value is False else "unqualified"
            )
        if run["case"] != "stop_startup":
            required.append(run.get("endpoint_loopback", "unqualified"))
        if run.get("stop_after_s", run.get("stop_after_launch_s")) is None:
            required.extend(
                run.get(key, "unqualified") for key in ("seek", "tail_coverage")
            )
            offset = run.get("requested_start_s")
            if offset is not None and run["duration_s"] - offset >= 0.5:
                required.append(telemetry)
        else:
            required.extend(
                run.get(key, "unqualified")
                for key in ("post_stop_silence", "post_stop_latency")
            )
            if run["case"] == "stop_playing":
                required.append(run.get("bookmark", "unqualified"))
                required.append(run.get("bookmark_alignment", "unqualified"))
            latency = run.get("stop_process_latency_s")
            required.append(
                "unqualified"
                if latency is None or not math.isfinite(latency)
                else "fail"
                if latency > STOP_BOUND_S
                else "pass"
            )
    return (
        "fail"
        if "fail" in required
        else "unqualified"
        if not required or "unqualified" in required
        else "pass"
    )


def fixture_check() -> dict:
    """Return numeric known-truth checks without opening a device/player."""
    reader = StatsReader()
    parsed = reader.feed(
        b"  1.25 M-A: 0.000 fd=0 aq=8KB vq=0KB sq=0B f=0/0\r\n",
        generation=2,
        observed_at_s=3.0,
    )
    parser_passed = (
        parsed == [ProgressObservation(1.25, 3.0, 2)]
        and parse_stats_position(b"nan M-A: 0.0 fd=0 aq=8KB") is None
        and select_fresh(parsed, generation=3, now_s=3.0) is None
        and select_fresh(parsed, generation=2, now_s=4.0) is None
    )
    source = marker_signal(2)
    capture = np.concatenate([np.zeros(4800), source[24000:72000], np.zeros(4800)])
    match = measure_match(capture, source)
    passed = (
        match is not None
        and abs(match["start_source_s"] - 0.5) < 0.010
        and abs(match["end_source_s"] - 1.5) < 0.010
        and measure_match(np.zeros(48000), source) is None
    )
    return {
        "result": "pass" if passed and parser_passed else "fail",
        "parser_result": "pass" if parser_passed else "fail",
        "known_start_s": 0.5,
        "known_end_s": 1.5,
        "measurement": match,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-loopback", action="store_true")
    parser.add_argument("--ffplay")
    parser.add_argument("--tail-padding-ms", type=int, choices=(0, 250), default=0)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--capture-buffer-frames", type=int, choices=(480, 4800), default=480
    )
    parser.add_argument(
        "--capture-clock",
        choices=("legacy", "wasapi-packet", "pulse-fragment"),
        default="legacy",
    )
    args = parser.parse_args()
    if (
        args.capture_loopback
        and args.capture_clock == "pulse-fragment"
        and platform.system() != "Linux"
    ):
        parser.error("pulse-fragment capture requires Linux")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clock = time.get_clock_info("perf_counter")
    receipt = {
        "platform": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "clock": {
            "name": "perf_counter",
            "implementation": clock.implementation,
            "resolution_s": clock.resolution,
            "monotonic": clock.monotonic,
        },
        "fixture": fixture_check(),
        "native_platforms": {
            "Windows": "unqualified",
            "Darwin": "unqualified",
            "Linux": "unqualified",
        },
        "physical_output": "unqualified",
        "listener_confirmation": "unqualified",
        "full_gate": "unqualified",
        "runs": [],
        "capture_rate_hz": RATE,
        "capture_channels": 2,
        "stop_wait_bound_s": STOP_BOUND_S,
        "seek_target_s": 0.250,
        "tail_padding_ms": args.tail_padding_ms,
        "repeat": args.repeat,
        "capture_buffer_frames": args.capture_buffer_frames,
        "capture_clock": args.capture_clock,
        "logical_eof": "no_content_or_exact_content_eof_requires_later_service_no_audio_run",
    }
    try:
        if args.capture_loopback:
            executable = resolve_ffplay(args.ffplay)
            receipt["executable"] = _build_receipt(executable)
            # Only synthetic fixture WAVs touch disk, in an owned temporary dir.
            with tempfile.TemporaryDirectory(prefix="ffplay-synthetic-") as scratch:
                run_matrix(
                    executable,
                    Path(scratch),
                    tail_padding_ms=args.tail_padding_ms,
                    repeat=args.repeat,
                    runs=receipt["runs"],
                    capture_buffer_frames=args.capture_buffer_frames,
                    capture_clock=args.capture_clock,
                )
            native = native_result(receipt["runs"])
            if platform.system() in receipt["native_platforms"]:
                receipt["native_platforms"][platform.system()] = native
            if native == "fail":
                receipt["full_gate"] = "fail"
    except Exception:
        # Paths, raw native errors and output may contain private data.
        receipt["runner_error"] = "qualification_failed"
    receipt["run_count"] = sum(run.get("launched") is True for run in receipt["runs"])
    receipt["unknown_launch_count"] = sum(
        run.get("launched") is None for run in receipt["runs"]
    )
    receipt["case_count"] = len(receipt["runs"])
    output = args.output_dir / "qualification.json"
    output.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "receipt": str(output),
                "run_count": receipt["run_count"],
                "full_gate": receipt["full_gate"],
            }
        )
    )
    return (
        1 if "runner_error" in receipt or receipt["fixture"]["result"] != "pass" else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
