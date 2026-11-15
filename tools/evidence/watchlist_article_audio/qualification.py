"""Qualify ffplay with synthetic WAVs; loopback samples live in RAM only.

Run without --capture-loopback for numerical checks only. SoundCard is an
optional isolated helper, never an application dependency. Native capture is
owned by a killable process; no microphone fallback is permitted.
"""

import argparse
from collections import deque
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

RATE = 48000
BLOCK = 480
STOP_BOUND_S = 1.5
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
    connection, duration_s: float, capture_buffer_frames: int = BLOCK
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
    except Exception:
        try:
            connection.send(("error", "capture_failed"))
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


def bookmark_alignment(frame: int | None, source_end_s: float | None) -> dict:
    """Compare saved progress with independently captured stopped content."""
    if frame is None or source_end_s is None or not math.isfinite(source_end_s):
        return {"bookmark_alignment": "unqualified", "bookmark_end_error_s": None}
    error = frame / RATE - source_end_s
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
) -> dict:
    """Measure one bounded native run; all raw samples/output are disposable."""
    if capture_buffer_frames not in (480, 4800):
        raise ValueError("capture_buffer_frames_must_be_480_or_4800")
    result = {
        "launched": False,
        "generation": generation,
        "duration_s": len(source) / RATE,
        "requested_start_s": offset,
        "stop_after_launch_s": stop_after if not padded else None,
        "stop_after_playing_s": stop_after if padded else None,
        "stop_after_s": stop_after,
        "stop_clock": "observed_playing" if padded else "launch",
        "capture_ready_timeout_s": 30 if padded else 8,
        "playing_ready_timeout_s": 30 if padded else None,
        "capture_lifetime_s": 45 if padded else 12,
        "max_capture_blocks": 4500 if padded else 1800,
        "capture_buffer_frames": capture_buffer_frames,
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
    receiver, sender = context.Pipe(duplex=False)
    capture = context.Process(
        target=_capture_worker,
        args=(sender, result["capture_lifetime_s"], capture_buffer_frames),
        daemon=True,
    )
    process = None
    reader = None
    blocks = []
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
        # Independent pre-launch baseline; no output names or samples are saved.
        baseline_until = time.perf_counter() + 0.35
        deadline = (
            time.perf_counter() + len(source) / RATE + 6 + (30.25 if padded else 0)
        )
        while time.perf_counter() < deadline:
            while receiver.poll():
                try:
                    message = receiver.recv()
                except EOFError:
                    break
                if message[0] == "error":
                    result["reason"] = message[1]
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
                )
                result["launched"] = True
                if padded:
                    playing_ready_deadline = time.perf_counter() + 30
                    deadline = playing_ready_deadline + len(source) / RATE + 0.25 + 6
                reader = threading.Thread(
                    target=_stats_pump,
                    args=(process.stderr, observations, generation),
                    daemon=True,
                )
                reader.start()
            if process is not None:
                if (
                    padded
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
                        deadline = playing_at + len(source) / RATE + 0.25 + 6
                if (
                    padded
                    and playing_at is None
                    and time.perf_counter() >= playing_ready_deadline
                    and process.poll() is None
                ):
                    result["reason"] = "playing_ready_timeout"
                    break
                if (
                    stop_after is not None
                    and stop_at is None
                    and stop_is_due(
                        now_s=time.perf_counter(),
                        launched_at_s=launched_at,
                        playing_at_s=playing_at,
                        delay_s=stop_after,
                        wait_for_playing=padded,
                    )
                ):
                    stop_at = time.perf_counter()
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
                    result["returncode"] = process.returncode
                    result["process_exit"] = (
                        "pass"
                        if stop_at is not None or process.returncode == 0
                        else "fail"
                    )
                if exited_at is not None and now - exited_at >= 0.6:
                    break
            time.sleep(0.005)
        else:
            result["reason"] = (
                "playing_ready_timeout"
                if padded and playing_at is None
                else "playback_deadline"
            )
            if result["reason"] == "playback_deadline":
                result["process_exit"] = "fail"
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
        result["capture_blocks"] = len(blocks)
        result["observation_count_retained"] = len(observations)
        if stop_at is not None:
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
        if stop_after is not None:
            result.update(
                bookmark_alignment(
                    result.get("saved_bookmark_frame"), measurement["end_source_s"]
                )
            )
        error = measurement["start_source_s"] - offset
        result["seek_error_s"] = error
        result["seek"] = "pass" if abs(error) <= 0.250 else "fail"
        remaining = len(source) / RATE - measurement["end_source_s"]
        result["missing_tail_s"] = remaining
        if stop_after is None:
            result["tail_coverage"] = "pass" if abs(remaining) <= 0.020 else "fail"
        # Map a matched sample to its acquisition block, independent of launch.
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
) -> list[dict]:
    """Reuse three stable artifacts; a resume consumes its confirmed Stop frame."""
    if tail_padding_ms not in (0, 250) or repeat not in (1, 2, 3):
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


def native_result(runs: list[dict]) -> str:
    """Every required case counts; unmatched cases and missing cleanup cannot pass."""
    required = []
    for run in runs:
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
                required.append(run.get("telemetry", "unqualified"))
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
    args = parser.parse_args()
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
