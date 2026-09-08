"""Bounded live measurements for physical speculative-voice qualification."""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import hashlib
import io
from importlib import import_module
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import time
from typing import Any
import wave

from tldw_chatbook.Audio.aec_backend import AecProcessor, create_aec_processor
from tldw_chatbook.Audio.acoustic_isolation import AcousticIsolationMonitor
from tldw_chatbook.Audio.duplex_contracts import (
    AcousticDemotionReason,
    AcousticSafetyPath,
    AudioFrame,
    RouteKind,
)
from tldw_chatbook.Audio.duplex_transport import (
    FRAME_BYTES,
    FRAME_DURATION_NS,
    FRAME_SAMPLES,
    PROCESSING_SAMPLE_RATE,
    DuplexAudioTransport,
)
from tldw_chatbook.Audio.voice_preprocessor import (
    VoicePreprocessor,
    read_aec_metric_snapshot,
)


_ASSET_PATH = Path(__file__).with_name("assets") / "voice_physical_reference.wav"
_MANIFEST_PATH = Path(__file__).with_name("assets") / "voice_physical_reference.json"
_EXPECTED_ASSET_SHA256 = (
    "d3f1822e269f009d029be6973e89d1f679bd299d9ad6fdec43d80923882cb9c5"
)
_MAX_MANIFEST_BYTES = 16 * 1024
_REFERENCE_FRAMES = 600
_REFERENCE_BYTES = _REFERENCE_FRAMES * FRAME_BYTES
_SAMPLE_INTERVAL_NS = 1_000_000_000
_MAX_STOP_LATENCY_MS = 10_000.0
_MAX_INTERRUPTION_PROMPT_ATTEMPTS = 3
_MAX_OPERATOR_RESPONSE_SECONDS = 10.0
_OBSERVATION_SAMPLE_LIMIT = 3_600
_EXPECTED_MANIFEST_KEYS = {
    "schema_version",
    "audio_format",
    "source",
    "license_spdx",
    "redistribution_grant",
    "sha256",
}
_EXPECTED_FORMAT = {
    "sample_rate_hz": PROCESSING_SAMPLE_RATE,
    "channels": 1,
    "sample_width_bytes": 2,
    "sample_format": "pcm_s16le",
    "frame_duration_ms": 10,
    "frame_count": _REFERENCE_FRAMES,
}
_EXPECTED_SOURCE = {
    "kind": "deterministic_synthesis",
    "generator": "Packaging/voice_aec_corpus.py:iter_voice_aec_case_frames",
    "source_manifest": "Tests/Audio/fixtures/voice_aec/manifest.json",
    "source_case_id": "stationary-echo-30m:first-600-render-frames",
    "source_recipe_sha256": (
        "14952533bb07b5b05c3a30db226532c4a7aa889e728c9c3f1d5fc90b2b17289d"
    ),
    "contains_captured_audio": False,
    "contains_user_audio": False,
}
_REDISTRIBUTION_GRANT = (
    "The deterministic generated audio is dedicated under CC0-1.0 and may be "
    "redistributed in this repository and its distributions without restriction."
)

_TransportFactory = Callable[[], DuplexAudioTransport]
_RouteReader = Callable[[], "RouteIdentity"]
_Prompt = Callable[[str], Awaitable[None]]
_Notify = Callable[[str], None]
_Wait = Callable[[float], Awaitable[None]]
_IsolationFactory = Callable[[], AcousticIsolationMonitor]


class PhysicalVoiceRunnerError(RuntimeError):
    """Raised when a live trial cannot produce trustworthy observations."""


def notify_operator(message: str) -> None:
    """Emit non-blocking progress for a live operator."""

    print(message, flush=True)


@dataclass(frozen=True, slots=True)
class LiveTrialConfig:
    """Bounded durations and trial counts for one live qualification run."""

    soak_seconds: float = 1_800.0
    route_round_trips: int = 3
    interruption_trials: int = 20
    double_talk_trials: int = 20

    def __post_init__(self) -> None:
        if (
            isinstance(self.soak_seconds, bool)
            or not isinstance(self.soak_seconds, (int, float))
            or not math.isfinite(float(self.soak_seconds))
            or not 0.01 <= float(self.soak_seconds) <= 1_800.0
        ):
            raise ValueError("live voice soak must be between 0.01 and 1800 seconds")
        if (
            type(self.route_round_trips) is not int
            or not 1 <= self.route_round_trips <= 3
        ):
            raise ValueError(
                "live voice route round trips must be between one and three"
            )
        for name, value in (
            ("interruption_trials", self.interruption_trials),
            ("double_talk_trials", self.double_talk_trials),
        ):
            if type(value) is not int or not 1 <= value <= 20:
                raise ValueError(f"live voice {name} must be between one and twenty")


@dataclass(frozen=True, slots=True)
class RouteIdentity:
    """In-memory default-route identity and required duplex capabilities."""

    input_index: int
    output_index: int
    input_name: str
    output_name: str
    input_channels: int
    output_channels: int
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if type(self.input_index) is not int or type(self.output_index) is not int:
            raise TypeError("route indices must be integers")
        if not isinstance(self.input_name, str) or not isinstance(
            self.output_name,
            str,
        ):
            raise TypeError("route names must be strings")
        if any(
            type(value) is not int
            for value in (
                self.input_channels,
                self.output_channels,
                self.sample_rate_hz,
            )
        ):
            raise TypeError("route capabilities must be integers")

    @property
    def identity_key(self) -> tuple[str, str]:
        """Return stable endpoint names used only to prove a real route change."""

        return (
            self.input_name,
            self.output_name,
        )

    def require_supported(self) -> None:
        """Reject a missing or non-48-kHz duplex default route."""

        if self.input_index < 0 or self.output_index < 0:
            raise PhysicalVoiceRunnerError(
                "default input and output devices are required"
            )
        if self.input_channels < 1 or self.output_channels < 1:
            raise PhysicalVoiceRunnerError("default route lacks duplex permissions")
        if self.sample_rate_hz != PROCESSING_SAMPLE_RATE:
            raise PhysicalVoiceRunnerError("default route does not support 48 kHz mono")


def read_default_route() -> RouteIdentity:
    """Read and validate the current sounddevice defaults without persisting identity."""

    try:
        sounddevice = import_module("sounddevice")
        return _read_default_route(sounddevice)
    except PhysicalVoiceRunnerError:
        raise
    except Exception as exc:
        raise PhysicalVoiceRunnerError(
            "default duplex audio route is unavailable"
        ) from exc


def refresh_default_route() -> RouteIdentity:
    """Refresh PortAudio inventory after stream teardown and read new defaults."""

    try:
        sounddevice = import_module("sounddevice")
        terminate = getattr(sounddevice, "_terminate", None)
        initialize = getattr(sounddevice, "_initialize", None)
        if not callable(terminate) or not callable(initialize):
            raise PhysicalVoiceRunnerError(
                "audio default device refresh is unsupported"
            )
        terminate()
        initialize()
        return _read_default_route(sounddevice)
    except PhysicalVoiceRunnerError:
        raise
    except Exception as exc:
        raise PhysicalVoiceRunnerError(
            "default duplex audio route refresh failed"
        ) from exc


def _read_default_route(sounddevice: Any) -> RouteIdentity:
    defaults = sounddevice.default.device
    try:
        input_index, output_index = defaults
    except (TypeError, ValueError) as exc:
        raise PhysicalVoiceRunnerError(
            "audio default device API shape is invalid"
        ) from exc
    if type(input_index) is not int or type(output_index) is not int:
        raise PhysicalVoiceRunnerError("audio default devices must use integer indices")
    if input_index < 0 or output_index < 0:
        raise PhysicalVoiceRunnerError(
            "default input and output devices are unavailable"
        )
    input_info = sounddevice.query_devices(input_index, "input")
    output_info = sounddevice.query_devices(output_index, "output")
    sounddevice.check_input_settings(
        device=input_index,
        samplerate=PROCESSING_SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sounddevice.check_output_settings(
        device=output_index,
        samplerate=PROCESSING_SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    route = RouteIdentity(
        input_index=input_index,
        output_index=output_index,
        input_name=str(input_info["name"]),
        output_name=str(output_info["name"]),
        input_channels=int(input_info["max_input_channels"]),
        output_channels=int(output_info["max_output_channels"]),
        sample_rate_hz=PROCESSING_SAMPLE_RATE,
    )
    route.require_supported()
    return route


async def prompt_operator(message: str) -> None:
    """Await an operator acknowledgement without retaining a blocked thread."""

    stream = sys.stdin
    if not bool(getattr(stream, "isatty", lambda: False)()):
        raise PhysicalVoiceRunnerError(
            "operator prompt requires an interactive stdin terminal"
        )
    print(f"{message}\nPress Enter when ready: ", end="", flush=True)
    if os.name == "nt":
        console = import_module("msvcrt")
        while True:
            if console.kbhit():
                character = console.getwch()
                if character in {"\r", "\n"}:
                    print()
                    return
            await asyncio.sleep(0.05)
    if os.name != "posix":
        raise PhysicalVoiceRunnerError(
            "operator prompt is unsupported on this interactive platform"
        )
    try:
        file_descriptor = stream.fileno()
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise PhysicalVoiceRunnerError(
            "operator prompt cannot monitor interactive stdin"
        ) from exc
    loop = asyncio.get_running_loop()
    acknowledged: asyncio.Future[None] = loop.create_future()

    def stdin_ready() -> None:
        if acknowledged.done():
            return
        try:
            line = stream.readline()
        except (OSError, UnicodeError, ValueError) as exc:
            acknowledged.set_exception(exc)
            return
        if line:
            acknowledged.set_result(None)
        else:
            acknowledged.set_exception(EOFError("interactive stdin closed"))

    try:
        loop.add_reader(file_descriptor, stdin_ready)
    except (
        NotImplementedError,
        OSError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise PhysicalVoiceRunnerError(
            "operator prompt cannot monitor interactive stdin"
        ) from exc
    try:
        await acknowledged
    except (EOFError, OSError, UnicodeError, ValueError) as exc:
        raise PhysicalVoiceRunnerError(
            "operator prompt could not read interactive stdin"
        ) from exc
    finally:
        loop.remove_reader(file_descriptor)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PhysicalVoiceRunnerError(
                "physical reference manifest contains duplicate keys"
            )
        value[key] = item
    return value


def _matches_exact_json(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return False
        return all(
            _matches_exact_json(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    return actual == expected


def load_physical_reference(
    asset_path: Path = _ASSET_PATH,
    manifest_path: Path = _MANIFEST_PATH,
) -> tuple[bytes, ...]:
    """Validate and load the checked-in deterministic reference as 10 ms frames."""

    asset = Path(asset_path)
    manifest_file = Path(manifest_path)
    try:
        manifest_raw = _read_regular_file_bounded(
            manifest_file,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
        manifest = json.loads(
            manifest_raw,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except PhysicalVoiceRunnerError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise PhysicalVoiceRunnerError(
            "physical reference manifest is invalid"
        ) from exc
    if type(manifest) is not dict or set(manifest) != _EXPECTED_MANIFEST_KEYS:
        raise PhysicalVoiceRunnerError("physical reference manifest shape is invalid")
    if (
        not _matches_exact_json(manifest.get("schema_version"), 1)
        or not _matches_exact_json(manifest.get("audio_format"), _EXPECTED_FORMAT)
        or not _matches_exact_json(manifest.get("source"), _EXPECTED_SOURCE)
        or not _matches_exact_json(manifest.get("license_spdx"), "CC0-1.0")
        or not _matches_exact_json(
            manifest.get("redistribution_grant"),
            _REDISTRIBUTION_GRANT,
        )
    ):
        raise PhysicalVoiceRunnerError("physical reference provenance is invalid")
    expected_hash = manifest.get("sha256")
    if type(expected_hash) is not str or expected_hash != _EXPECTED_ASSET_SHA256:
        raise PhysicalVoiceRunnerError("physical reference hash is invalid")
    encoded = _read_regular_file_bounded(
        asset,
        maximum_bytes=_REFERENCE_BYTES + 44,
    )
    if len(encoded) != _REFERENCE_BYTES + 44:
        raise PhysicalVoiceRunnerError("physical reference audio size is invalid")
    if hashlib.sha256(encoded).hexdigest() != _EXPECTED_ASSET_SHA256:
        raise PhysicalVoiceRunnerError("physical reference SHA-256 mismatch")
    try:
        with wave.open(io.BytesIO(encoded), "rb") as wav_file:
            params = wav_file.getparams()
            if (
                params.nchannels != 1
                or params.sampwidth != 2
                or params.framerate != PROCESSING_SAMPLE_RATE
                or params.nframes != _REFERENCE_FRAMES * FRAME_SAMPLES
                or params.comptype != "NONE"
            ):
                raise PhysicalVoiceRunnerError(
                    "physical reference WAV format is invalid"
                )
            pcm = wav_file.readframes(params.nframes)
            if wav_file.readframes(1):
                raise PhysicalVoiceRunnerError(
                    "physical reference WAV length is invalid"
                )
    except PhysicalVoiceRunnerError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise PhysicalVoiceRunnerError("physical reference WAV is invalid") from exc
    if len(pcm) != _REFERENCE_BYTES:
        raise PhysicalVoiceRunnerError("physical reference PCM length is invalid")
    return tuple(
        pcm[offset : offset + FRAME_BYTES] for offset in range(0, len(pcm), FRAME_BYTES)
    )


def _read_regular_file_bounded(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one path once through a bounded, regular-file descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PhysicalVoiceRunnerError(
            "physical reference file is missing or unsafe"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        linked = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(linked.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise PhysicalVoiceRunnerError(
                "physical reference file is missing or unsafe"
            )
        if opened.st_size > maximum_bytes:
            raise PhysicalVoiceRunnerError("physical reference file is too large")
        encoded = bytearray()
        while len(encoded) <= maximum_bytes:
            remaining = maximum_bytes + 1 - len(encoded)
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            encoded.extend(chunk)
        if len(encoded) > maximum_bytes:
            raise PhysicalVoiceRunnerError("physical reference file is too large")
        return bytes(encoded)
    except PhysicalVoiceRunnerError:
        raise
    except OSError as exc:
        raise PhysicalVoiceRunnerError("physical reference file is unreadable") from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


class PhysicalVoiceTrialRunner:
    """Measure one bounded trial through the real transport/preprocessor boundary."""

    def __init__(
        self,
        *,
        device_class: str,
        config: LiveTrialConfig = LiveTrialConfig(),
        transport_factory: _TransportFactory = DuplexAudioTransport,
        route_reader: _RouteReader = read_default_route,
        route_refresh_reader: _RouteReader | None = None,
        prompt: _Prompt = prompt_operator,
        notify: _Notify = notify_operator,
        reference_path: Path = _ASSET_PATH,
        manifest_path: Path = _MANIFEST_PATH,
        aec_factory: Callable[[], AecProcessor | None] = create_aec_processor,
        isolation_factory: _IsolationFactory = AcousticIsolationMonitor,
        clock: Callable[[], int] = time.monotonic_ns,
        wait: _Wait = asyncio.sleep,
        identity_salt_factory: Callable[[int], bytes] = secrets.token_bytes,
    ) -> None:
        if device_class not in {"builtin", "usb", "bluetooth"}:
            raise ValueError("physical device class is invalid")
        self.device_class = device_class
        self.config = config
        self.transport_factory = transport_factory
        self.route_reader = route_reader
        if route_refresh_reader is None:
            route_refresh_reader = (
                refresh_default_route
                if route_reader is read_default_route
                else route_reader
            )
        self.route_refresh_reader = route_refresh_reader
        self.prompt = prompt
        self.notify = notify
        self.reference_path = Path(reference_path)
        self.manifest_path = Path(manifest_path)
        self.aec_factory = aec_factory
        self.isolation_factory = isolation_factory
        self.clock = clock
        self.wait = wait
        self.identity_salt_factory = identity_salt_factory
        self._reference_frames: tuple[bytes, ...] = ()
        self._reference_index = 0
        self._pending_references: deque[AudioFrame] = deque()
        self._pending_reference_overflows = 0
        self._mutable_scratch = bytearray(FRAME_BYTES)
        self._transport: DuplexAudioTransport | None = None
        self._preprocessor: VoicePreprocessor | None = None
        self._monitor: AcousticIsolationMonitor | None = None
        self._aec: AecProcessor | None = None
        self._phase = "idle"
        self._phase_detected = False
        self._capture_admitted = False
        self._capture_processing_ok = False
        self._admitted_event_active = False
        self._first_admitted_ns: int | None = None
        self._last_processed_capture: AudioFrame | None = None
        self._last_sample_ns: int | None = None
        self._rendered_frames = 0
        self._silence_frames = 0
        self._rendered_false_barges = 0
        self._render_only_vad_events = 0
        self._double_talk_detected = 0
        self._stop_latency_samples_ms: list[float] = []
        self._manual_interruption_available = False
        self._manual_interruption_latched = False
        self._manual_stop_attempts = 0
        self._manual_stop_successes = 0
        self._unsuppressed_interruption_observed = False
        self._erle_samples: list[float] = []
        self._processor_operational = True
        self._demotions: Counter[str] = Counter()
        self._last_demotion: AcousticDemotionReason | None = None
        self._isolation_warmup_ms = 0.0
        self._route_capture_frames = 0
        self._route_trials: list[dict[str, object]] = []
        self._silence_false_barges = 0
        self._active_trial_ns = 0
        self._prompt_task: asyncio.Task[None] | None = None
        self._runner_task_peak = 0

    @property
    def retained_frame_count(self) -> int:
        """Return runner-owned PCM frame references, used to prove cleanup."""

        return len(self._reference_frames) + len(self._pending_references)

    async def run(self) -> dict[str, object]:
        """Collect content-free observations or raise when evidence is unavailable."""

        try:
            self._reference_frames = load_physical_reference(
                self.reference_path,
                self.manifest_path,
            )
            target_route = self.route_reader()
            target_route.require_supported()
            salt = self.identity_salt_factory(16)
            if not isinstance(salt, bytes) or len(salt) != 16:
                raise PhysicalVoiceRunnerError("device identity salt is invalid")
            device_identifier = _salted_route_identifier(target_route, salt)
            completed = {
                "rendered_speech_completed": False,
                "double_talk_completed": False,
                "interruption_completed": False,
                "silence_completed": False,
                "device_switch_completed": False,
                "soak_completed": False,
            }
            self._transport = self.transport_factory()
            if not isinstance(self._transport, DuplexAudioTransport):
                raise PhysicalVoiceRunnerError(
                    "transport factory returned an invalid port"
                )
            self._aec = self.aec_factory()
            self._monitor = self.isolation_factory()
            self._preprocessor = VoicePreprocessor(
                aec=self._aec,
                on_admitted_frame=self._on_admitted_frame,
                on_processed=self._acknowledge_processed,
                isolation_monitor=self._monitor,
            )
            await self._transport.start(
                device_pair=(target_route.input_index, target_route.output_index),
            )
            if not self._transport.is_open:
                raise PhysicalVoiceRunnerError("duplex transport did not open")

            await self._run_phase(
                self._calibration_seconds(),
                rendering=True,
                phase="render-only",
            )
            completed["rendered_speech_completed"] = True

            await self._require_speech_calibration()
            for _trial in range(self.config.double_talk_trials):
                self._phase_detected = False
                await self._run_phase(
                    self._trial_seconds(),
                    rendering=True,
                    phase="double-talk",
                    abort_rendering=False,
                )
                if self._phase_detected:
                    self._double_talk_detected += 1
                self.notify(
                    f"Double-talk window {_trial + 1}/{self.config.double_talk_trials}: "
                    f"{'detected' if self._phase_detected else 'not detected'}."
                )
            await self._require_transport().abort_output()
            completed["double_talk_completed"] = True
            await self._require_quiet_recovery()

            manual_stop = await self._prompt_for_interruption_trial(trial_number=1)
            for trial in range(self.config.interruption_trials):
                if trial > 0 and (
                    manual_stop or self._uses_manual_interruption_protocol()
                ):
                    manual_stop = await self._prompt_for_interruption_trial(
                        trial_number=trial + 1,
                    )
                if manual_stop:
                    self.notify(
                        f"Manual interruption sample "
                        f"{trial + 1}/{self.config.interruption_trials}: "
                        "stop acknowledged; measuring."
                    )
                else:
                    self.notify(
                        f"Interruption cue "
                        f"{trial + 1}/{self.config.interruption_trials}: speak now."
                    )
                await self._run_interruption_trial(manual_stop=manual_stop)
                latency_ms = self._stop_latency_samples_ms[-1]
                sample_passed = (
                    self._manual_interruption_available
                    if manual_stop
                    else latency_ms <= 150.0
                    and not self._uses_manual_interruption_protocol()
                )
                self.notify(
                    f"Interruption sample {trial + 1}/{self.config.interruption_trials}: "
                    f"{'passed' if sample_passed else 'failed'} "
                    f"({latency_ms:.1f} ms)."
                )
                if not sample_passed:
                    self.notify(
                        "Qualification stopped after the first failed "
                        "interruption sample."
                    )
                    await self._cleanup()
                    return self._build_observations(
                        target_route=target_route,
                        device_identifier=device_identifier,
                        operator_checklist=completed,
                    )
                await self._require_quiet_recovery()
            completed["interruption_completed"] = True

            for trial in range(self.config.route_round_trips):
                start_generation = self._transport.clock_generation
                alternate, alternate_closed = await self._transition_route(
                    prior=target_route,
                    message=(
                        f"Route trial {trial + 1}: switch input and output to an "
                        "alternate route."
                    ),
                )
                alternate_generation = self._transport.clock_generation
                await self._run_phase(
                    self._calibration_seconds(),
                    rendering=True,
                    phase="render-only",
                )
                returned, return_closed = await self._transition_route(
                    prior=alternate,
                    message=(
                        f"Route trial {trial + 1}: switch input and output back to the "
                        "target route."
                    ),
                )
                return_generation = self._transport.clock_generation
                if returned.identity_key != target_route.identity_key:
                    raise PhysicalVoiceRunnerError(
                        "route trial did not return to target"
                    )
                self._route_trials.append(
                    {
                        "start_generation": start_generation,
                        "alternate_generation": alternate_generation,
                        "return_generation": return_generation,
                        "alternate_admission_closed": alternate_closed,
                        "return_admission_closed": return_closed,
                        "returned_to_target": True,
                    }
                )
                await self._run_phase(
                    self._calibration_seconds(),
                    rendering=True,
                    phase="render-only",
                )
            completed["device_switch_completed"] = True

            await self._transport.abort_output()
            await self._run_phase(
                self._minimum_silence_seconds(),
                rendering=False,
                phase="silence",
            )
            completed["silence_completed"] = True

            target_active_ns = round(self.config.soak_seconds * 1_000_000_000)
            rendering = True
            while self._active_trial_ns < target_active_ns:
                await self._run_phase(
                    min(
                        1.0,
                        (target_active_ns - self._active_trial_ns) / 1_000_000_000,
                    ),
                    rendering=rendering,
                    phase="render-only" if rendering else "silence",
                )
                rendering = not rendering
            completed["soak_completed"] = True
            await self._cleanup()
            return self._build_observations(
                target_route=target_route,
                device_identifier=device_identifier,
                operator_checklist=completed,
            )
        finally:
            try:
                await self._cleanup()
            finally:
                self._release_runtime()

    def _calibration_seconds(self) -> float:
        return min(5.0, max(0.05, float(self.config.soak_seconds) / 4.0))

    def _trial_seconds(self) -> float:
        return min(0.5, max(0.05, float(self.config.soak_seconds) / 10.0))

    def _minimum_silence_seconds(self) -> float:
        return min(0.5, max(0.05, float(self.config.soak_seconds) / 10.0))

    def _operator_response_seconds(self) -> float:
        return min(
            _MAX_OPERATOR_RESPONSE_SECONDS,
            max(0.5, float(self.config.soak_seconds) / 10.0),
        )

    def _operator_recovery_seconds(self) -> float:
        return min(1.0, max(0.05, float(self.config.soak_seconds) / 10.0))

    async def _require_speech_calibration(self) -> None:
        await self._prompt_and_service(
            "Double-talk calibration: begin speaking continuously and keep speaking."
        )
        self._enter_phase("speech-calibration")
        self._phase_detected = False
        deadline = self.clock() + round(self._operator_response_seconds() * 1e9)
        while self.clock() < deadline and not self._phase_detected:
            await self._tick(rendering=False, measure=False)
        if not self._phase_detected:
            raise PhysicalVoiceRunnerError(
                "speech calibration did not detect admitted microphone speech"
            )
        self.notify("Speech calibration passed: microphone speech detected.")
        if self._require_preprocessor().safety.path not in {
            AcousticSafetyPath.AEC,
            AcousticSafetyPath.ACOUSTIC_ISOLATION,
        }:
            self._manual_interruption_latched = True
            self.notify(
                "Full-duplex admission is not open after preflight; "
                "using the manual-stop fallback."
            )
            return
        self._phase_detected = False
        await self._run_phase(
            max(0.5, self._trial_seconds()),
            rendering=True,
            phase="double-talk",
            abort_rendering=False,
        )
        if not self._phase_detected:
            raise PhysicalVoiceRunnerError(
                "double-talk calibration did not detect speech during playback"
            )
        self.notify("Double-talk calibration passed: speech detected during playback.")

    async def _require_quiet_recovery(self) -> None:
        """Wait for consecutive captured silence before issuing another cue."""

        self.notify("Stop speaking now; waiting for confirmed microphone silence.")
        self._enter_phase("operator-recovery")
        quiet_frames_required = max(
            1,
            math.ceil(
                self._operator_recovery_seconds()
                * 1_000_000_000
                / FRAME_DURATION_NS
            ),
        )
        quiet_frames = 0
        deadline = self.clock() + round(self._operator_response_seconds() * 1e9)
        while self.clock() < deadline:
            prior_capture = self._last_processed_capture
            await self.wait(0.001)
            await self._process_available(
                rendering=False,
                measure=False,
                capture_limit=1,
            )
            if self._last_processed_capture is prior_capture:
                continue
            if not self._capture_processing_ok:
                raise PhysicalVoiceRunnerError(
                    "operator quiet recovery could not verify capture processing"
                )
            if self._capture_admitted:
                quiet_frames = 0
                continue
            quiet_frames += 1
            if quiet_frames >= quiet_frames_required:
                self.notify("Microphone silence confirmed.")
                return
        raise PhysicalVoiceRunnerError("operator quiet recovery was not detected")

    async def _run_phase(
        self,
        seconds: float,
        *,
        rendering: bool,
        phase: str,
        abort_rendering: bool = True,
    ) -> None:
        if seconds <= 0:
            return
        self._enter_phase(phase)
        phase_started_ns = self.clock()
        deadline = phase_started_ns + max(
            FRAME_DURATION_NS,
            round(seconds * 1e9),
        )
        try:
            while self.clock() < deadline:
                await self._tick(rendering=rendering)
            if rendering and abort_rendering:
                await self._require_transport().abort_output()
        finally:
            self._active_trial_ns += max(0, self.clock() - phase_started_ns)

    async def _prompt_and_service(
        self,
        message: str,
        *,
        rendering: bool = False,
    ) -> None:
        """Keep callback rings drained while the operator reads a prompt."""

        if self._prompt_task is not None:
            raise PhysicalVoiceRunnerError("operator prompt is already active")
        self._enter_phase("prompt")
        task = asyncio.create_task(self.prompt(message))
        self._prompt_task = task
        self._runner_task_peak = max(self._runner_task_peak, 1)
        try:
            while not task.done():
                await self._tick(rendering=rendering, measure=False)
            await task
        finally:
            if task.done():
                self._prompt_task = None

    async def _prompt_for_interruption_trial(self, *, trial_number: int) -> bool:
        for _attempt in range(_MAX_INTERRUPTION_PROMPT_ATTEMPTS):
            manual_stop = self._uses_manual_interruption_protocol()
            if manual_stop:
                self._manual_interruption_latched = True
            message = (
                f"Interruption trial {trial_number}: press Enter to trigger the "
                "manual stop while playback is active."
                if manual_stop
                else f"Interruption trial {trial_number}: speak once over playback."
            )
            await self._prompt_and_service(message, rendering=manual_stop)
            if self._uses_manual_interruption_protocol() == manual_stop:
                return manual_stop
        raise PhysicalVoiceRunnerError(
            "interruption protocol changed repeatedly during operator prompt"
        )

    def _uses_manual_interruption_protocol(self) -> bool:
        return (
            self._manual_interruption_latched
            or self._require_preprocessor().safety.path
            not in {
                AcousticSafetyPath.AEC,
                AcousticSafetyPath.ACOUSTIC_ISOLATION,
            }
        )

    async def _tick(self, *, rendering: bool, measure: bool = True) -> None:
        transport = self._require_transport()
        if rendering:
            occupancy = transport.buffer_occupancy
            capacities = transport.buffer_capacities
            combined_render_frames = (
                occupancy.render_frames + occupancy.render_reference_frames
            )
            shared_capacity = min(
                capacities.render_frames,
                capacities.render_reference_frames,
            )
            if combined_render_frames < shared_capacity:
                pcm = self._reference_frames[self._reference_index]
                queued = transport.queue_render(pcm)
                if queued is not None:
                    self._reference_index = (self._reference_index + 1) % len(
                        self._reference_frames
                    )
        await self.wait(0.001)
        await self._process_available(rendering=rendering, measure=measure)

    async def _process_available(
        self,
        *,
        rendering: bool,
        measure: bool,
        capture_limit: int | None = None,
    ) -> None:
        transport = self._require_transport()
        occupancy = transport.buffer_occupancy
        capacities = transport.buffer_capacities
        reference_budget = occupancy.render_reference_frames
        capture_budget = occupancy.capture_frames
        if capture_limit is not None:
            capture_budget = min(capture_budget, capture_limit)
        pending_overflowed = False
        for _ in range(reference_budget):
            reference = transport.pop_render_reference()
            if reference is None:
                break
            if pending_overflowed:
                continue
            if len(self._pending_references) >= capacities.render_reference_frames:
                self._pending_reference_overflows += 1
                self._pending_references.clear()
                pending_overflowed = True
                continue
            self._pending_references.append(reference)
        for _ in range(capture_budget):
            capture = transport.pop_capture()
            if capture is None:
                break
            references: list[AudioFrame] = []
            boundary = capture.render_reference_sequence
            while self._pending_references and (
                boundary is not None
                and self._pending_references[0].sequence <= boundary
            ):
                references.append(self._pending_references.popleft())
            self._last_processed_capture = capture
            self._route_capture_frames += 1
            self._capture_admitted = False
            self._capture_processing_ok = False
            await self._require_preprocessor().process_capture(
                capture,
                render_frames=tuple(references),
                assistant_rendering=rendering,
            )
            self._observe_safety_state()
            self._record_admission_onset()
            if measure:
                if rendering:
                    self._rendered_frames += 1
                else:
                    self._silence_frames += 1
                self._observe_phase_metrics(capture)

    def _observe_safety_state(self) -> None:
        preprocessor = self._require_preprocessor()
        reason = preprocessor.safety.demotion_reason
        if reason is not None and reason is not self._last_demotion:
            self._demotions[reason.value] += 1
            if reason is AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE:
                self._processor_operational = False
        self._last_demotion = reason

    def _observe_phase_metrics(self, capture: AudioFrame) -> None:
        preprocessor = self._require_preprocessor()
        if (
            preprocessor.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
            and self._isolation_warmup_ms == 0.0
        ):
            self._isolation_warmup_ms = max(
                10.0,
                self._route_capture_frames * 10.0,
            )
        now_ns = self.clock()
        if self._last_sample_ns is not None and (
            now_ns - self._last_sample_ns < _SAMPLE_INTERVAL_NS
        ):
            return
        self._last_sample_ns = now_ns
        if self._aec is None or len(self._erle_samples) >= _OBSERVATION_SAMPLE_LIMIT:
            return
        try:
            metrics, _explicit_health = read_aec_metric_snapshot(self._aec)
            if (
                metrics["delay_estimate_available"] != 1.0
                or metrics["delay_estimate_refined"] != 1.0
            ):
                return
            erle = float(metrics["erle_db"])
            self._erle_samples.append(erle)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            self._processor_operational = False

    def _on_admitted_frame(self, frame: AudioFrame) -> None:
        self._capture_admitted = True
        if self._phase in {"speech-calibration", "double-talk"}:
            self._phase_detected = True
        elif self._phase == "interruption" and self._first_admitted_ns is None:
            self._first_admitted_ns = frame.started_ns

    def _record_admission_onset(self) -> None:
        if self._phase not in {"render-only", "silence"}:
            return
        if not self._capture_admitted:
            self._admitted_event_active = False
            return
        if self._admitted_event_active:
            return
        self._admitted_event_active = True
        if self._phase == "render-only":
            self._rendered_false_barges += 1
            self._render_only_vad_events += 1
        else:
            self._silence_false_barges += 1

    def _enter_phase(self, phase: str) -> None:
        if self._phase != phase:
            self._phase_detected = False
            self._admitted_event_active = False
            self._capture_admitted = False
        self._phase = phase

    def _acknowledge_processed(
        self,
        sequence: int,
        clock_generation: int,
        dsp_ok: bool,
        vad_ok: bool,
    ) -> None:
        self._capture_processing_ok = dsp_ok and vad_ok
        if not dsp_ok or not vad_ok:
            self._processor_operational = False
        self._require_transport().acknowledge_capture(
            sequence,
            clock_generation=clock_generation,
            dsp_ok=dsp_ok,
            vad_ok=vad_ok,
        )

    async def _run_interruption_trial(self, *, manual_stop: bool) -> None:
        trial_started_ns = self.clock()
        try:
            await self._collect_interruption_sample(manual_stop=manual_stop)
        finally:
            self._active_trial_ns += max(0, self.clock() - trial_started_ns)

    async def _collect_interruption_sample(self, *, manual_stop: bool) -> None:
        self._enter_phase("interruption")
        self._first_admitted_ns = None
        full_duplex = not manual_stop
        deadline = self.clock() + round(self._operator_response_seconds() * 1e9)
        while (
            full_duplex and self.clock() < deadline and self._first_admitted_ns is None
        ):
            await self._tick(rendering=True)
        acoustic_trigger_ns = self._first_admitted_ns
        manual_trigger_ns = self.clock()
        trigger_ns = (
            acoustic_trigger_ns
            if acoustic_trigger_ns is not None
            else manual_trigger_ns
        )
        last_reference_sequence = (
            self._last_processed_capture.render_reference_sequence
            if self._last_processed_capture is not None
            else None
        )
        latency_ms = await self._abort_and_measure_silence_boundary(
            trigger_ns=trigger_ns,
            last_reference_sequence=last_reference_sequence,
        )
        if latency_ms is None:
            self._stop_latency_samples_ms.append(_MAX_STOP_LATENCY_MS)
            if full_duplex:
                self._unsuppressed_interruption_observed = True
            return
        if full_duplex and acoustic_trigger_ns is None:
            self._stop_latency_samples_ms.append(_MAX_STOP_LATENCY_MS)
            self._unsuppressed_interruption_observed = True
            return
        self._stop_latency_samples_ms.append(latency_ms)
        if full_duplex and latency_ms > 150.0:
            self._unsuppressed_interruption_observed = True

    async def _abort_and_measure_silence_boundary(
        self,
        *,
        trigger_ns: int,
        last_reference_sequence: int | None,
    ) -> float | None:
        transport = self._require_transport()
        self._manual_stop_attempts += 1
        self._manual_interruption_available = False
        try:
            await transport.abort_output()
        except Exception:
            return None
        abort_completed_ns = self.clock()
        reference_baseline = last_reference_sequence
        silence_deadline = self.clock() + round(self._trial_seconds() * 1e9)
        while self.clock() < silence_deadline:
            prior_capture = self._last_processed_capture
            await self.wait(0.001)
            await self._process_available(
                rendering=False,
                measure=True,
                capture_limit=1,
            )
            boundary = self._last_processed_capture
            if boundary is prior_capture or boundary is None:
                continue
            evidence = boundary.delay_evidence
            if evidence is None:
                continue
            if evidence.observed_ns < abort_completed_ns:
                if boundary.render_reference_sequence is not None:
                    reference_baseline = boundary.render_reference_sequence
                continue
            if (
                boundary.render_reference_sequence == reference_baseline
                and evidence.render_dac_ns >= trigger_ns
                and evidence.occupancy_bounded
                and not evidence.timing_discontinuity
                and not evidence.clock_drift
            ):
                latency_ms = max(
                    0.0,
                    (evidence.render_dac_ns - trigger_ns) / 1_000_000.0,
                )
                self._manual_stop_successes += 1
                self._manual_interruption_available = (
                    self._manual_stop_successes == self._manual_stop_attempts
                )
                return min(_MAX_STOP_LATENCY_MS, latency_ms)
        return None

    async def _transition_route(
        self,
        *,
        prior: RouteIdentity,
        message: str,
    ) -> tuple[RouteIdentity, bool]:
        transport = self._require_transport()
        preprocessor = self._require_preprocessor()
        await self._prompt_and_service(message)
        await transport.notify_route_changed(RouteKind.DUPLEX)
        event = transport.pop_control_event()
        if (
            event is None
            or event.old_clock_generation + 1 != transport.clock_generation
        ):
            raise PhysicalVoiceRunnerError("route fence control event is missing")
        await self.wait(0)
        changed = self.route_refresh_reader()
        changed.require_supported()
        if changed.identity_key == prior.identity_key:
            raise PhysicalVoiceRunnerError("route identity did not change")
        self._pending_references.clear()
        self._last_processed_capture = None
        self._erle_samples.clear()
        preprocessor.reset_for_device_route(transport.clock_generation)
        admission_closed = not preprocessor.safety.admission_open
        if not admission_closed:
            raise PhysicalVoiceRunnerError(
                "route admission reopened before calibration"
            )
        await transport.start(
            device_pair=(changed.input_index, changed.output_index),
        )
        if not transport.is_open:
            raise PhysicalVoiceRunnerError("duplex route did not reopen")
        self._route_capture_frames = 0
        self._isolation_warmup_ms = 0.0
        self._last_sample_ns = None
        self._last_demotion = None
        return changed, admission_closed

    def _build_observations(
        self,
        *,
        target_route: RouteIdentity,
        device_identifier: str,
        operator_checklist: Mapping[str, bool],
    ) -> dict[str, object]:
        transport = self._require_transport()
        preprocessor = self._require_preprocessor()
        metrics: Mapping[str, float] = {}
        if self._aec is not None:
            try:
                metrics, _explicit_health = read_aec_metric_snapshot(self._aec)
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                self._processor_operational = False
                self._erle_samples.clear()
                metrics = {}
        available = metrics.get("delay_estimate_available", 0.0) == 1.0
        refined = metrics.get("delay_estimate_refined", 0.0) == 1.0
        monitor = self._monitor
        correlations = list(monitor.correlation_samples) if monitor is not None else []
        leakages = list(monitor.leakage_db_samples) if monitor is not None else []
        safety_snapshot = preprocessor.safety
        current_path = safety_snapshot.path
        processor_operational = self._aec is not None and self._processor_operational
        safety_path = (
            current_path.value
            if processor_operational
            and current_path
            in {AcousticSafetyPath.AEC, AcousticSafetyPath.ACOUSTIC_ISOLATION}
            else AcousticSafetyPath.HALF_DUPLEX.value
        )
        elapsed_minutes = max(
            0.000001,
            self._active_trial_ns / 60_000_000_000,
        )
        rendered_minutes = max(0.000001, self._rendered_frames / 6_000.0)
        silence_minutes = max(0.000001, self._silence_frames / 6_000.0)
        stop_samples = self._stop_latency_samples_ms or [_MAX_STOP_LATENCY_MS]
        transport_kind = {
            "builtin": "builtin",
            "usb": "usb",
            "bluetooth": "bluetooth",
        }[self.device_class]
        return {
            "aec_delay_estimate_available": available,
            "aec_delay_estimate_refined": refined,
            "aec_health_path": preprocessor.health.value,
            "aec_processor_operational": processor_operational,
            "audio_ring_growth": any(
                (
                    transport.buffer_occupancy.capture_frames,
                    transport.buffer_occupancy.render_frames,
                    transport.buffer_occupancy.render_reference_frames,
                )
            ),
            "capture_overflows": transport.capture_overflows,
            "channels": 1,
            "control_overflows": transport.control_overflows,
            "correlation_samples": correlations,
            "demotion_reason_counts": dict(sorted(self._demotions.items())),
            "device_handle_leaks": (
                int(transport.is_open) + transport.teardown_failures
            ),
            "device_identifier": device_identifier,
            "double_talk_detected": self._double_talk_detected,
            "double_talk_trials": self.config.double_talk_trials,
            "erle_samples_db": list(self._erle_samples),
            "frame_duration_ms": 10,
            "isolation_warmup_duration_ms": self._isolation_warmup_ms,
            "leakage_db_samples": leakages,
            "manual_interruption_available": self._manual_interruption_available,
            "operator_checklist": dict(operator_checklist),
            "playback_speech_admission_closed": not safety_snapshot.admission_open,
            "post_fence_callbacks": transport.stale_callbacks,
            "process_leaks": 0,
            "reference_overflows": transport.render_reference_overflows
            + self._pending_reference_overflows,
            "render_only_vad_events": self._render_only_vad_events,
            "render_overflows": transport.render_overflows,
            "rendered_false_barge_events": self._rendered_false_barges,
            "rendered_speech_minutes": rendered_minutes,
            "route_generation_trials": list(self._route_trials),
            "safety_path": safety_path,
            "sample_rate_hz": target_route.sample_rate_hz,
            "saturation_events": self._demotions.get("saturation", 0),
            "silence_false_barge_events": self._silence_false_barges,
            "silence_minutes": silence_minutes,
            "soak_minutes": elapsed_minutes,
            "stop_latency_samples_ms": list(stop_samples),
            "transport": transport_kind,
            "unbounded_task_growth": self._runner_task_peak > 1
            or self._prompt_task is not None,
            "unsuppressed_interruption_observed": (
                self._unsuppressed_interruption_observed
            ),
        }

    async def _cleanup(self) -> None:
        prompt_task = self._prompt_task
        if prompt_task is not None:
            if not prompt_task.done():
                prompt_task.cancel()
            await asyncio.gather(prompt_task, return_exceptions=True)
            self._prompt_task = None
        transport = self._transport
        if transport is not None:
            try:
                if transport.is_open:
                    await transport.close()
            finally:
                while transport.pop_capture() is not None:
                    pass
                while transport.pop_render_reference() is not None:
                    pass
                while transport.pop_control_event() is not None:
                    pass
        self._release_buffers()
        self._enter_phase("closed")

    def _release_buffers(self) -> None:
        self._pending_references.clear()
        for index in range(len(self._mutable_scratch)):
            self._mutable_scratch[index] = 0
        self._mutable_scratch.clear()
        self._reference_frames = ()
        self._last_processed_capture = None

    def _release_runtime(self) -> None:
        self._preprocessor = None
        self._monitor = None
        self._aec = None
        self._transport = None

    def _require_transport(self) -> DuplexAudioTransport:
        if self._transport is None:
            raise PhysicalVoiceRunnerError("duplex transport is unavailable")
        return self._transport

    def _require_preprocessor(self) -> VoicePreprocessor:
        if self._preprocessor is None:
            raise PhysicalVoiceRunnerError("voice preprocessor is unavailable")
        return self._preprocessor


def _salted_route_identifier(route: RouteIdentity, salt: bytes) -> str:
    material = (
        f"{route.input_index}\0{route.output_index}\0"
        f"{route.input_name}\0{route.output_name}"
    ).encode("utf-8")
    return hashlib.sha256(salt + material).hexdigest()


async def collect_live_observations(
    *,
    device_class: str,
    config: LiveTrialConfig = LiveTrialConfig(),
    transport_factory: _TransportFactory = DuplexAudioTransport,
    route_reader: _RouteReader = read_default_route,
    prompt: _Prompt = prompt_operator,
) -> dict[str, object]:
    """Collect a physical trial through the production defaults."""

    return await PhysicalVoiceTrialRunner(
        device_class=device_class,
        config=config,
        transport_factory=transport_factory,
        route_reader=route_reader,
        prompt=prompt,
    ).run()


__all__ = [
    "LiveTrialConfig",
    "PhysicalVoiceRunnerError",
    "PhysicalVoiceTrialRunner",
    "RouteIdentity",
    "collect_live_observations",
    "load_physical_reference",
    "prompt_operator",
    "read_default_route",
    "refresh_default_route",
]
