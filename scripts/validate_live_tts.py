"""Opt-in local Kokoro playback/cancellation evidence; imports are inert.

See Docs/Development/TTS/Live_Validation.md. Run serially with other audio work.
The public process bounds a private worker; a forced exit is never a cleanup pass.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import wave
from array import array
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, ExitStack, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

DEFAULT_TEXT = (
    "Silver compass opens this reply. The middle sentence describes a quiet orchard. "
    "This complete reply ends with amber sunrise."
)
DEFAULT_ANCHORS = ["silver compass", "quiet orchard", "amber sunrise"]
VOICE_LANGUAGES = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
    "j": "ja",
    "z": "zh",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the inert command-line parser.

    Returns:
        A parser for explicit assets, runtime selection and playback admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("pytorch", "onnx"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--voice-dir", type=Path)
    parser.add_argument("--voices", type=Path, help="ONNX voices binary")
    parser.add_argument("--language", help="Must match the selected voice's language")
    parser.add_argument("--format", choices=("wav", "mp3"), default="wav")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--text-file", type=Path, help="UTF-8 text; keep language segments short"
    )
    parser.add_argument("--cancel-text-file", type=Path)
    parser.add_argument("--scenarios", default="playback,cancel,repeat")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--phase-timeout", type=float, default=300)
    parser.add_argument("--cleanup-timeout", type=float, default=60)
    parser.add_argument("--run-timeout", type=float, default=1800)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New private run directory"
    )
    parser.add_argument(
        "--play-audio",
        action="store_true",
        help="Explicitly enable real device playback",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject unsafe or incomplete invocations before creating a run/profile.

    Args:
        args: Parsed options, updated in place with centrally validated paths.

    Raises:
        ValueError: Admission, assets, paths or worker ownership are invalid.
        OSError: A required local path cannot be inspected."""
    from tldw_chatbook.Utils.input_validation import validate_tts_inference_device

    args.device = validate_tts_inference_device(args.device)
    if not args.play_audio:
        raise ValueError("Real device playback requires --play-audio")
    if not 1 <= args.repeats <= 12:
        raise ValueError("repeats must be between 1 and 12")
    for name in ("phase_timeout", "cleanup_timeout", "run_timeout", "speed"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.run_timeout > 7200 or args.speed > 4:
        raise ValueError(
            "Run timeout cannot exceed 7200 seconds; speed cannot exceed 4"
        )
    scenarios = args.scenarios.split(",")
    if not scenarios or any(
        item not in {"playback", "cancel", "repeat"} for item in scenarios
    ):
        raise ValueError("Unknown scenario; choose playback,cancel,repeat")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", args.voice):
        raise ValueError("voice must be a local Kokoro voice name")
    language = VOICE_LANGUAGES.get(args.voice[0])
    if language is None or (args.language and args.language.lower() != language):
        raise ValueError("language must match the selected voice prefix")
    args.language = language
    from tldw_chatbook.Utils.path_validation import validate_path_simple

    for name in (
        "model",
        "voice_dir",
        "voices",
        "text_file",
        "cancel_text_file",
        "expected_package_root",
    ):
        value = getattr(args, name)
        if value is not None:
            validated = validate_path_simple(value.expanduser(), require_exists=True)
            setattr(args, name, validated.resolve())
    for name in ("voice_dir", "expected_package_root"):
        value = getattr(args, name)
        if value is not None and not value.is_dir():
            raise ValueError(f"{name} must be an existing directory")
    output = validate_path_simple(args.output.expanduser(), probe_existing=False)
    if output.is_symlink() or (os.path.lexists(output) and not args.worker):
        raise ValueError(f"Output already exists or is a symlink: {output}")
    if not output.parent.is_dir() or (args.worker and not output.is_dir()):
        raise ValueError(
            "Output requires an existing parent and a real worker directory"
        )
    args.output = output.parent.resolve() / output.name
    required = [args.model, args.expected_package_root / "__init__.py"]
    if args.engine == "pytorch":
        if args.voice_dir is None:
            raise ValueError("PyTorch requires --voice-dir")
        required += [
            args.model.with_name("config.json"),
            args.voice_dir / f"{args.voice}.pt",
        ]
    else:
        if args.device != "cpu" or args.voices is None:
            raise ValueError("ONNX requires --device cpu and --voices")
        required.append(args.voices)
    required += [value for value in (args.text_file, args.cancel_text_file) if value]
    for path in required:
        validated = validate_path_simple(path)
        if not validated.is_file():
            raise ValueError(f"Required local asset is missing: {validated}")
    if args.worker and os.environ.get("TLDW_LIVE_VALIDATION_WORKER") != str(
        args.output
    ):
        raise ValueError("Private worker must be started by this command")
    for name in ("text_file", "cancel_text_file"):
        path = getattr(args, name)
        if path and (
            path.stat().st_size > 32000
            or not 1 <= len(path.read_text().strip()) <= 8000
        ):
            raise ValueError(
                "Text must contain 1–8000 characters (at most 32000 UTF-8 bytes)"
            )


def check_package_root(actual: Path, expected: Path) -> None:
    """Require the imported package to match the selected installation.

    Args:
        actual: Imported package directory.
        expected: Explicitly selected package directory.

    Raises:
        ValueError: The package locations differ."""
    if actual.resolve() != expected.resolve():
        raise ValueError(
            f"Imported package {actual} differs from requested package {expected}"
        )


def sha256(path: Path) -> str:
    """Hash a caller-validated file using bounded reads.

    Args:
        path: Local file admitted by the runner.

    Returns:
        The complete file's lowercase SHA256 digest.

    Raises:
        OSError: The file cannot be read."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_evidence(path: Path, evidence: dict) -> None:
    """Atomically replace the run's private JSON checkpoint.

    Args:
        path: Evidence destination inside the admitted private run.
        evidence: Complete current observations, including partial failures.

    Raises:
        OSError: Writing, securing or replacing the checkpoint fails."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def observe_native(
    function: Callable[..., Any],
    calls: list[dict[str, Any]],
    details: dict[str, Any],
    *,
    lock: AbstractContextManager[Any] | None = None,
    synchronize: Callable[[], None] | None = None,
) -> Callable[..., Any]:
    """Observe an unchanged synchronous call, including its exceptional exit.

    Args:
        function: Native callable to delegate without changing its arguments.
        calls: Mutable observation ledger receiving monotonic entry/exit rows.
        details: Fixed runtime and request identity attached to each row.
        lock: Optional shared context manager protecting ledger updates.
        synchronize: Optional device barrier before entry and after host return.

    Returns:
        A wrapper that preserves the callable's result and exceptions."""

    def observed(*args, **kwargs):
        if synchronize is not None:
            synchronize()
        row = {
            **details,
            "enter_at": time.monotonic(),
            "thread_id": threading.get_ident(),
        }
        with lock or nullcontext():
            calls.append(row)
        native_error = None
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            native_error = error
            with lock or nullcontext():
                row["exception"] = type(error).__name__
            raise
        finally:
            synchronized = synchronize is None
            if synchronize is not None:
                with lock or nullcontext():
                    row["host_returned_at"] = time.monotonic()
                try:
                    # Never hold the ledger lock while waiting for device work:
                    # the event loop needs it to record Stop during inference.
                    synchronize()
                    synchronized = True
                except BaseException as error:
                    with lock or nullcontext():
                        row["synchronization_exception"] = type(error).__name__
                    if native_error is None:
                        raise
            if synchronized:
                with lock or nullcontext():
                    if synchronize is not None:
                        row["cuda_synchronized"] = True
                    row["exit_at"] = time.monotonic()

    return observed


def load_pytorch_runtime() -> Any:
    """Load PyTorch after worker profile setup, with optional-extra guidance.

    Returns:
        The centrally loaded PyTorch module.

    Raises:
        ImportError: PyTorch is unavailable; includes the supported extra name.
    """
    from tldw_chatbook.Utils.optional_deps import require_dependency

    return require_dependency("torch", "local_tts")


def pytorch_device_provenance(torch: Any, requested: str) -> dict:
    """Reject unavailable accelerators and identify the selected CUDA device.

    Args:
        torch: Already imported runtime; never loaded by inert CLI admission.
        requested: Admitted cpu, mps or cuda device family.

    Returns:
        Runtime version and selected device, with CUDA hardware/build metadata.

    Raises:
        ValueError: The requested accelerator is unavailable.
    """
    result = {"device": requested, "torch_version": str(torch.__version__)}
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS inference is unavailable")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA inference is unavailable")
        device = f"cuda:{torch.cuda.current_device()}"
        properties = torch.cuda.get_device_properties(device)
        result.update(
            device=device,
            cuda_version=torch.version.cuda,
            name=properties.name,
            total_memory_bytes=properties.total_memory,
            compute_capability=[properties.major, properties.minor],
        )
    return result


def observe_pytorch_native(
    function: Callable[..., Any],
    calls: list[dict[str, Any]],
    details: dict[str, Any],
    *,
    expected_device: str,
    torch: Any,
    lock: AbstractContextManager[Any] | None = None,
) -> Callable[..., Any]:
    """Require actual model placement and include CUDA completion in its interval.

    Args:
        function: Original PyTorch forward callable.
        calls: Mutable native observation ledger.
        details: Request metadata including the actual model.device string.
        expected_device: Selected concrete CUDA device or CPU/MPS family.
        torch: Already imported PyTorch runtime.
        lock: Optional shared ledger lock.

    Returns:
        Wrapper preserving native arguments, results and native exceptions.

    Raises:
        ValueError: Model placement differs from the selected device.
    """
    actual = details["device"]
    cuda = expected_device.startswith("cuda:")
    if (actual if cuda else actual.split(":")[0]) != expected_device:
        raise ValueError("Actual PyTorch device differs from requested device")
    return observe_native(
        function,
        calls,
        details,
        lock=lock,
        synchronize=(lambda: torch.cuda.synchronize(expected_device)) if cuda else None,
    )


def validate_cancellation(row: dict) -> None:
    """Require Stop to overlap native work and settlement to follow its exit.

    Args:
        row: Internal phase observations with native intervals and Stop/settlement.

    Raises:
        ValueError: Work did not overlap Stop, outlived settlement or restarted."""
    calls = row["native_calls"]
    stop = row["stop_requested_at"]
    if not any(call["enter_at"] < stop < call.get("exit_at", -1) for call in calls):
        raise ValueError("Cancellation did not overlap an observed native call")
    if any(
        "exit_at" not in call or call["exit_at"] > row["settled_at"] for call in calls
    ):
        raise ValueError("Claimed settlement preceded native completion")
    if any(call["enter_at"] > stop for call in calls):
        raise ValueError("Cancelled request started another native call")


def assert_quiescent(resources: dict) -> None:
    """Require all recorded resource owners to have joined.

    Args:
        resources: Internal resource counts or active-owner flags.

    Raises:
        ValueError: Any owner remains active."""
    active = {key: value for key, value in resources.items() if value}
    if active:
        raise ValueError(f"Resources have not joined: {active}")


def validate_playback(row: dict) -> None:
    """Require full-duration physical output and successful drain or player exit.

    Args:
        row: Internal audio and playback observations for a successful phase.

    Raises:
        ValueError: Playback was incomplete, failed or did not fully drain."""
    audio, playback = row["audio"], row["playback"]
    if playback["elapsed_seconds"] < audio["seconds"] - 0.25:
        raise ValueError("Player finished before the complete decoded audio duration")
    if playback["kind"] == "file":
        if playback.get("exit_code") != 0:
            raise ValueError("File playback did not complete successfully")
    elif (
        not playback.get("drained")
        or playback.get("device_frames", 0) < audio["frames"]
    ):
        raise ValueError("The physical sink did not drain the complete response")


def inspect_wav(path: Path) -> dict:
    """Read every PCM16 frame in bounded blocks, rejecting truncated containers.

    Args:
        path: Admitted local WAV file.

    Returns:
        Full encoded/PCM hashes, frame count, duration and signal statistics.

    Raises:
        ValueError: Audio is unsupported, truncated, empty or silent.
        OSError: The file cannot be read."""
    frames = count = 0
    squares = peak = 0
    pcm_hash = hashlib.sha256()
    with wave.open(str(path), "rb") as source:
        rate, channels = source.getframerate(), source.getnchannels()
        declared = source.getnframes()
        if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError("Evidence decoding requires PCM16 WAV")
        while block := source.readframes(16384):
            if len(block) % (2 * channels):
                raise ValueError("WAV contains a truncated frame")
            frames += len(block) // (2 * channels)
            pcm_hash.update(block)
            samples = array("h", block)
            if sys.byteorder != "little":
                samples.byteswap()
            count += len(samples)
            squares += sum(value * value for value in samples)
            peak = max(peak, max(abs(value) for value in samples))
    if frames != declared:
        raise ValueError("WAV data is truncated relative to its declared frame count")
    if not frames or not peak:
        raise ValueError("Audio is empty or silent")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "pcm_sha256": pcm_hash.hexdigest(),
        "sample_rate": rate,
        "channels": channels,
        "frames": frames,
        "seconds": frames / rate,
        "rms": math.sqrt(squares / count) / 32768,
        "peak": peak / 32768,
    }


def inspect_audio(path: Path) -> dict:
    """Decode an admitted clip and inspect every audio frame.

    Args:
        path: Local WAV or encoded clip inside the private run.

    Returns:
        Complete audio statistics with original and decoded file identities.

    Raises:
        ValueError: Decoded audio fails complete-frame validation.
        subprocess.SubprocessError: The bounded ffmpeg decode fails.
        OSError: Audio files or the decoder cannot be accessed."""
    if path.suffix == ".wav":
        return inspect_wav(path)
    decoded = path.with_suffix(".decoded.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(path),
            "-c:a",
            "pcm_s16le",
            str(decoded),
        ],
        check=True,
        timeout=60,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    result = inspect_wav(decoded)
    result.update(
        decoded_path=str(decoded),
        decoded_sha256=result["sha256"],
        path=str(path),
        sha256=sha256(path),
    )
    return result


def setup_environment(root: Path) -> None:
    """Prepare the admitted worker's private directories and offline environment.

    Args:
        root: Existing private run directory created by the controller.

    Raises:
        OSError: Private directories cannot be created."""
    for name in ("KOKORO_MODEL_PATH", "KOKORO_VOICES_PATH"):
        os.environ.pop(name, None)
    for name in ("profile", "data", "cache", "audio", "blends"):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    os.environ.update(
        {
            "TLDW_CONFIG_PATH": str(root / "profile/config.toml"),
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
            "XDG_CONFIG_HOME": str(root / "profile"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "HF_HOME": str(root / "cache/hf"),
            "TORCH_HOME": str(root / "cache/torch"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_XET": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "6",
            "MKL_NUM_THREADS": "6",
        }
    )


def check_backend_assets(backend: Any, args: argparse.Namespace) -> None:
    """Reject ambient overrides and fallback before loading any native model.

    Args:
        backend: Constructed Kokoro backend before model initialization.
        args: Admitted explicit engine and local asset paths.

    Raises:
        ValueError: The backend engine or asset paths differ from admission."""
    if backend.use_onnx != (args.engine == "onnx"):
        raise ValueError("Backend selected a different inference engine")
    paths = {"model": (backend.model_path, args.model)}
    if args.engine == "onnx":
        paths["voices"] = (backend.voices_json, args.voices)
    else:
        paths["voice directory"] = (backend.voice_dir, args.voice_dir)
    for label, (actual, expected) in paths.items():
        if actual is None or Path(actual).resolve() != expected.resolve():
            raise ValueError(f"Backend selected a different {label}")


def _deny_network(event, arguments):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Live TTS validation forbids network access and downloads")
    if event == "subprocess.Popen":
        argv = arguments[1]
        words = argv if isinstance(argv, (list, tuple)) else [str(argv)]
        if any(str(word) in {"pip", "pip3", "download"} for word in words):
            raise RuntimeError("Live TTS validation forbids runtime installation")


def memory_snapshot(*, cuda_device: str | None = None) -> dict:
    """Observe process, already-loaded MPS and explicitly selected CUDA memory.

    Args:
        cuda_device: Concrete selected CUDA device, or None to leave CUDA untouched.

    Returns:
        Timestamp, threads, RSS/MPS and synchronized CUDA allocator measurements.

    Raises:
        subprocess.SubprocessError: The process memory probe fails."""
    torch = sys.modules.get("torch")
    if cuda_device is not None:
        if torch is None or not torch.cuda.is_available():
            raise ValueError("Requested CUDA memory observation is unavailable")
        torch.cuda.synchronize(cuda_device)
    result = {"at": time.monotonic(), "threads": threading.active_count()}
    if sys.platform in {"darwin", "linux"}:
        import resource

        current = subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "rss="], text=True
        )
        result["rss_bytes"] = int(current.strip()) * 1024
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_bytes"] = int(peak * (1 if sys.platform == "darwin" else 1024))
    if torch is not None and torch.backends.mps.is_available():
        result["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
    if cuda_device is not None:
        result.update(
            cuda_device=cuda_device,
            cuda_synchronized=True,
            cuda_allocated_bytes=torch.cuda.memory_allocated(cuda_device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(cuda_device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(cuda_device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(cuda_device),
        )
    return result


def _source_hashes(package: Path) -> dict:
    return {
        str(path.relative_to(package)): sha256(path)
        for path in sorted(package.rglob("*.py"))
    }


def _versions(names) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


async def run_live(args: argparse.Namespace, evidence: dict) -> None:
    """Run mounted production paths with optional imports isolated to the worker.

    Args:
        args: Validated runtime, assets, scenarios and private output directory.
        evidence: Mutable checkpoint ledger receiving real observations.

    Raises:
        Exception: Runtime, playback or lifecycle validation fails; the controller
            records partial evidence and never treats forced exit as clean join."""
    import toml
    from loguru import logger

    root = args.output
    app_tts = {
        "default_provider": "kokoro",
        "default_model_mode": "exact",
        "default_model": "kokoro",
        "default_voice_mode": "exact",
        "default_voice": args.voice,
        "default_format": args.format,
        "default_speed": args.speed,
        "KOKORO_USE_ONNX": args.engine == "onnx",
        "KOKORO_DEVICE_DEFAULT": args.device,
        "KOKORO_ENABLE_VOICE_MIXING": False,
        "KOKORO_VOICE_BLENDS_DIR": str(root / "blends"),
    }
    assets = [args.model]
    if args.engine == "pytorch":
        app_tts.update(
            KOKORO_PT_MODEL_PATH_DEFAULT=str(args.model),
            KOKORO_VOICE_DIR_PT=str(args.voice_dir),
        )
        assets += [
            args.model.with_name("config.json"),
            args.voice_dir / f"{args.voice}.pt",
        ]
    else:
        app_tts.update(
            KOKORO_ONNX_MODEL_PATH_DEFAULT=str(args.model),
            KOKORO_ONNX_VOICES_JSON_DEFAULT=str(args.voices),
            KOKORO_VOICE_DIR_PT=str(root / "unused-pytorch-voices"),
            KOKORO_PT_MODEL_PATH_DEFAULT=str(root / "unused-pytorch-model.pth"),
        )
        assets.append(args.voices)
    configuration = {
        "general": {"users_name": "tts_live_validation"},
        "paths": {"data_dir": str(root / "data")},
        "first_run": {"setup_completed": True},
        "model_catalog": {"enabled": False},
        "app_tts": app_tts,
    }
    config_path = root / "profile/config.toml"
    config_path.write_text(toml.dumps(configuration))
    config_path.chmod(0o600)
    logger.remove()
    logger.add(root / "application.log", level="DEBUG")

    import tldw_chatbook

    package = Path(tldw_chatbook.__file__).resolve().parent
    check_package_root(package, args.expected_package_root)
    from tldw_chatbook import config

    if not config.get_user_data_dir().resolve().is_relative_to(root):
        raise ValueError("Application data directory escaped the private run profile")
    evidence.update(
        imported_package_root=str(package),
        source_hashes=_source_hashes(package),
        runner_sha256=sha256(Path(__file__)),
        assets={str(path): sha256(path) for path in assets},
    )
    revision = await asyncio.to_thread(
        subprocess.run,
        ["git", "-C", str(package.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    evidence["source_revision_if_git"] = (
        revision.stdout.strip() if revision.returncode == 0 else None
    )
    evidence["versions"] = _versions(
        (
            "tldw_chatbook",
            "textual",
            "numpy",
            "sounddevice",
            "soundfile",
            "kokoro",
            "kokoro-onnx",
            "torch",
            "onnxruntime",
            "misaki",
            "spacy",
            "en-core-web-sm",
        )
    )
    cuda_device = None
    if args.engine == "pytorch":
        torch = load_pytorch_runtime()

        evidence["pytorch_device"] = pytorch_device_provenance(torch, args.device)
        if args.device == "cuda":
            cuda_device = evidence["pytorch_device"]["device"]

    import numpy as np
    import sounddevice as sd
    from textual import on
    from textual.app import App, ComposeResult
    from textual.widgets import Button, Select, TextArea

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_auto_speak import (
        AutoSpeakContext,
        AutoSpeakDisposition,
        decide_auto_speak,
    )
    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleChatStore,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
        STTSEventHandler,
        STTSPlaygroundGenerateEvent,
    )
    from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as events_module
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
        TTSCompleteEvent,
        TTSEventHandler,
        TTSMessageSpeechRequestEvent,
        TTSPlaybackEvent,
        TTSPlaybackLifecycle,
    )
    from tldw_chatbook.TTS.adapter_bootstrap import build_default_tts_service
    from tldw_chatbook.TTS.audio_player import PlaybackState, get_audio_player
    from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
    from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot
    from tldw_chatbook.TTS.TTS_Generation import (
        bind_tts_service,
        reset_tts_service_binding,
    )
    from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
    from tldw_chatbook.UI.STTS_Window import _seed_axis_defaults

    text = args.text_file.read_text().strip() if args.text_file else DEFAULT_TEXT
    cancel_text = (
        args.cancel_text_file.read_text().strip()
        if args.cancel_text_file
        else "\n".join([text] * 4)
    )
    current = [None]
    lock = threading.RLock()
    backends = []
    file_processes = []
    evidence["output_device"] = dict(sd.query_devices(kind="output"))
    evidence["memory_before_model"] = memory_snapshot(cuda_device=cuda_device)

    def checkpoint():
        with lock:
            write_evidence(root / "evidence.json", evidence)

    def phase(name, utterance=text):
        row = {
            "id": name,
            "expected_text": utterance,
            "language": args.language,
            "content_anchors": DEFAULT_ANCHORS if not args.text_file else [],
            "started_at": time.monotonic(),
            "native_calls": [],
            "states": [],
            "outcomes": [],
            "sink_events": [],
            "devices": [],
            "outcome": "running",
        }
        current[0] = row
        evidence["phases"].append(row)
        checkpoint()
        print("PHASE", name, flush=True)
        return row

    async def wait_until(predicate, label, seconds=None):
        deadline = time.monotonic() + (
            args.phase_timeout if seconds is None else seconds
        )
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Deadline exceeded: {label}")
            await asyncio.sleep(0.01)

    async def bounded(operation, label, seconds):
        task = asyncio.ensure_future(operation)
        done, _ = await asyncio.wait({task}, timeout=seconds)
        if not done:
            raise TimeoutError(f"Deadline exceeded: {label}; cleanup remains owned")
        return task.result()

    class Host(App):
        def __init__(self, service):
            super().__init__()
            self.loguru_logger = logger
            self.notices = []
            self.completions = []
            self.requests = []
            studio = StudioTTSPreferencesSnapshot()
            global_preferences = service.preferences_snapshot()
            self.pane = SpeechPlaygroundPane(
                provider="kokoro",
                studio_preferences=studio,
                global_preferences=global_preferences,
                axis_defaults=_seed_axis_defaults(studio, global_preferences),
            )
            self.lab_handler = STTSEventHandler(self)
            self.lab_handler._stts_service = service
            self._stts_handler = self.lab_handler

        def compose(self) -> ComposeResult:
            yield self.pane

        def notify(self, message, **kwargs):
            self.notices.append({"message": str(message), **kwargs})
            return super().notify(message, **kwargs)

        @on(STTSPlaygroundGenerateEvent)
        async def generate(self, event):
            self.requests.append(event.request)
            await self.lab_handler.handle_playground_generate(event)

        @on(TTSCompleteEvent)
        async def completed(self, event):
            self.completions.append(
                {"message_id": event.message_id, "error": event.error}
            )
            await TldwCli.handle_tts_complete_event(self, event)

        @on(TTSPlaybackEvent)
        async def playback(self, event):
            await self._tts_handler.handle_tts_playback(event)

    class Handler(TTSEventHandler):
        def __init__(self, host, service):
            super().__init__(default_profile_id_reader=lambda: None)
            self.app = host
            self._tts_service = service

        async def _stream_response_via_sink(self, plan, source, **kwargs):
            row = current[0]
            raw = root / "audio" / f"{row['id']}.response"
            started = time.monotonic()
            with raw.open("wb") as capture:

                async def observed():
                    async for chunk in source:
                        capture.write(chunk)
                        yield chunk

                outcome = await super()._stream_response_via_sink(
                    plan, observed(), **kwargs
                )
            row["playback"] = {
                "kind": "sink",
                "outcome": outcome,
                "elapsed_seconds": time.monotonic() - started,
            }
            row["source_bytes"] = raw.stat().st_size
            row["source_sha256"] = sha256(raw)
            audio_path = root / "audio" / f"{row['id']}.wav"
            with raw.open("rb") as capture, wave.open(str(audio_path), "wb") as output:
                output.setparams(
                    (plan.channels, 2, plan.sample_rate, 0, "NONE", "not compressed")
                )
                capture.seek(plan.skip_bytes)
                remaining = plan.data_bytes
                while remaining is None or remaining > 0:
                    block = capture.read(
                        65536 if remaining is None else min(65536, remaining)
                    )
                    if not block:
                        break
                    output.writeframesraw(block)
                    if remaining is not None:
                        remaining -= len(block)
                if remaining not in (None, 0):
                    raise ValueError("Sink capture omitted declared response bytes")
            row["audio"] = inspect_wav(audio_path)
            return outcome

    service = build_default_tts_service(
        {"COMPREHENSIVE_CONFIG_RAW": configuration, "APP_TTS_CONFIG": app_tts}
    )
    bind_tts_service(service)
    host = Host(service)
    handler = Handler(host, service)
    host._tts_handler = handler
    player = get_audio_player()

    def resources():
        calls = [call for row in evidence["phases"] for call in row["native_calls"]]
        return {
            "generation_owner": int(handler._console_generation_owner is not None),
            "stream_owner": int(handler._active_stream_playback_owner is not None),
            "file_owner": int(handler._active_file_playback_owner is not None),
            "active_tasks": sum(not task.done() for task in handler._active_tasks),
            "retained_io": sum(
                not task.done() for task in handler._retained_tts_io_tasks
            ),
            "retained_cleanup": sum(
                not task.done() for task in handler._retained_tts_cleanup_tasks
            ),
            "admitted_operations": len(service._admitted_operations),
            "responses": len(service._responses),
            "leases": service.registry._total_leases(),
            "backend_tasks": sum(
                not task.done()
                for backend in backends
                for name in ("_native_tasks", "_pytorch_tasks", "_onnx_tasks")
                for task in getattr(backend, name, ())
            ),
            "native_calls": sum("exit_at" not in call for call in calls),
            "player_processes": sum(
                process.poll() is None for process, _ in file_processes
            ),
        }

    async def quiescent(row):
        deadline = time.monotonic() + args.cleanup_timeout
        while True:
            state = resources()
            if not any(state.values()):
                break
            if state["native_calls"] and not any(
                value for key, value in state.items() if key != "native_calls"
            ):
                raise ValueError(
                    "Native inference outlived all production resource ownership"
                )
            if time.monotonic() >= deadline:
                assert_quiescent(state)
            await asyncio.sleep(0.01)
        memory = memory_snapshot(cuda_device=cuda_device)
        row.update(
            settled_at=time.monotonic(),
            resources_at_settlement=state,
            memory=memory,
        )

    original_initialize = KokoroTTSBackend.initialize

    async def initialize(backend):
        if not any(item is backend for item in backends):
            backends.append(backend)
        check_backend_assets(backend, args)
        await original_initialize(backend)
        check_backend_assets(backend, args)
        if args.engine == "pytorch":
            loaded = Path(backend.kokoro_model_pt.model_path).resolve()
            if loaded != args.model:
                raise ValueError(
                    "Actual PyTorch checkpoint differs from supplied model"
                )
        evidence.setdefault("backend_assets", []).append(
            {"model": str(Path(backend.model_path).resolve()), "engine": args.engine}
        )

    original_pytorch_initialize = KokoroTTSBackend._initialize_pytorch

    async def initialize_pytorch(backend):
        # ONNX can switch to PyTorch inside load_model, before initialize returns.
        # Reject that path before its loader can visit any fallback checkpoint.
        check_backend_assets(backend, args)
        await original_pytorch_initialize(backend)

    original_player = player.play

    def play(audio_file):
        row = current[0]
        saved = root / "audio" / f"{row['id']}{Path(audio_file).suffix}"
        shutil.copyfile(audio_file, saved)
        row["audio"] = inspect_audio(saved)
        started = time.monotonic()
        result = original_player(audio_file)
        process = player._current.process
        if not result or process is None:
            raise ValueError("The real file player did not start")
        row["playback"] = {
            "kind": "file",
            "pid": process.pid,
            "player": player._player_name,
            "started_at": started,
        }
        file_processes.append((process, row))
        return result

    original_sink = events_module.StreamingPcmSink

    def sink(*positional, **keywords):
        row = current[0]
        callback = keywords["on_event"]

        def event_observed(event):
            with lock:
                row["sink_events"].append(
                    {"event": type(event).__name__, "at": time.monotonic()}
                )
            return callback(event)

        keywords["on_event"] = event_observed
        return original_sink(*positional, **keywords)

    original_device = sd.OutputStream

    def output_stream(*positional, **keywords):
        row = current[0]
        callback = keywords["callback"]
        counters = {
            "frames": 0,
            "audible_frames": 0,
            "callbacks": 0,
            "sample_rate": keywords["samplerate"],
            "sha256": None,
        }
        row["devices"].append(counters)
        digest = hashlib.sha256()

        def observed(output, frames, callback_time, status):
            try:
                return callback(output, frames, callback_time, status)
            finally:
                # Only one callback-sized view and counters survive this call.
                with lock:
                    digest.update(memoryview(output).cast("B"))
                    counters["sha256"] = digest.hexdigest()
                    counters["callbacks"] += 1
                    counters["frames"] += frames
                    counters["audible_frames"] += int(
                        np.count_nonzero(np.any(output != 0, axis=1))
                    )
                    if status:
                        counters["last_status"] = str(status)

        keywords["callback"] = observed
        return original_device(*positional, **keywords)

    store = ConsoleChatStore()
    session = store.create_session()

    async def submit(row):
        destination = await handler.resolve_console_speech_destination(None, None)
        if destination is None or destination.charges_may_apply:
            raise ValueError("Trusted local Console destination did not resolve")
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=row["expected_text"]
        )
        preferences = ConsoleSpeechPreferences(
            auto_speak=True, consent_destination=destination.fingerprint
        )
        disposition = decide_auto_speak(
            message,
            session_id=session.id,
            context=AutoSpeakContext(
                preferences=preferences,
                destination_fingerprint=destination.fingerprint,
                active_session_id=session.id,
                hands_free_active=False,
            ),
        )
        if disposition is not AutoSpeakDisposition.SPEAK:
            raise ValueError("Automatic speech did not admit the reply")
        snapshot = store.issue_tts_message_speech_snapshot(message.id)
        lifecycle = TTSPlaybackLifecycle(
            message_id=message.id,
            request_id=len(evidence["phases"]),
            validator=lambda: (
                store.validate_tts_message_speech_snapshot(snapshot)
                == row["expected_text"]
            ),
            callback=row["states"].append,
        )
        row.update(
            message_id=message.id,
            destination_fingerprint=destination.fingerprint,
            disposition=disposition.value,
        )
        await handler.handle_tts_request(
            TTSMessageSpeechRequestEvent(
                snapshot,
                store.validate_tts_message_speech_snapshot,
                outcome_callback=row["outcomes"].append,
                expected_destination_fingerprint=destination.fingerprint,
                playback_lifecycle=lifecycle,
            )
        )
        return lifecycle

    async def complete(row, *, lab=False):
        if lab:
            await wait_until(
                lambda: "playback" in row, "Speech Lab started its real player"
            )
        else:
            await wait_until(
                lambda: row["states"] and row["states"][-1] in {"stopped", "failed"},
                "Console terminal lifecycle",
            )
            if row["states"] != ["playing", "stopped"] or False in row["outcomes"]:
                raise ValueError(
                    f"Console playback failed: {row['states']}; {host.notices}"
                )
        await wait_until(
            lambda: "playback" in row and "audio" in row, "complete audio evidence"
        )
        playback = row["playback"]
        if playback["kind"] == "file":
            process = next(process for process, owner in file_processes if owner is row)
            await wait_until(
                lambda: process.poll() is not None, "exact file-player process exit"
            )
            playback.update(
                exit_code=process.poll(),
                elapsed_seconds=time.monotonic() - playback["started_at"],
            )
            await wait_until(
                lambda: (
                    player.get_state() in {PlaybackState.FINISHED, PlaybackState.ERROR}
                ),
                "player terminal state",
                10,
            )
            if player.get_state() is not PlaybackState.FINISHED:
                raise ValueError("The owning file player reported an error")
        else:
            await wait_until(
                lambda: any(
                    item["event"] == "SinkDrained" for item in row["sink_events"]
                ),
                "actual sink drain",
                10,
            )
            playback.update(
                drained=True,
                device_frames=sum(item["frames"] for item in row["devices"]),
            )
            if playback["outcome"] != "success" or not any(
                item["audible_frames"] for item in row["devices"]
            ):
                raise ValueError("The real sink did not render audible frames")
        validate_playback(row)
        if not row["native_calls"]:
            raise ValueError("No real model inference was observed for the reply")
        await quiescent(row)
        row["outcome"] = "success"
        checkpoint()

    with ExitStack() as observers:
        observers.enter_context(
            patch.object(KokoroTTSBackend, "initialize", initialize)
        )
        observers.enter_context(
            patch.object(KokoroTTSBackend, "_initialize_pytorch", initialize_pytorch)
        )
        observers.enter_context(patch.object(player, "play", play))
        observers.enter_context(patch.object(events_module, "StreamingPcmSink", sink))
        observers.enter_context(patch.object(sd, "OutputStream", output_stream))
        if args.engine == "pytorch":
            import spacy

            if args.language.startswith("en-") and not spacy.util.is_package(
                "en_core_web_sm"
            ):
                raise ValueError(
                    "Install English language assets explicitly before running the harness"
                )
            observers.enter_context(
                patch(
                    "spacy.cli.download",
                    side_effect=RuntimeError("Runtime language downloads are disabled"),
                )
            )
            torch.set_num_threads(6)
            from kokoro import KModel

            original_forward = KModel.forward

            def forward(model, *positional, **keywords):
                row = current[0]
                details = {
                    "request_id": row["id"],
                    "class": type(model).__module__ + "." + type(model).__name__,
                    "device": str(model.device),
                    "speed": positional[2]
                    if len(positional) > 2
                    else keywords.get("speed", 1),
                    "fourier_module": type(model.decoder.generator.stft).__name__,
                }
                return observe_pytorch_native(
                    original_forward,
                    row["native_calls"],
                    details,
                    lock=lock,
                    expected_device=evidence["pytorch_device"]["device"],
                    torch=torch,
                )(model, *positional, **keywords)

            observers.enter_context(patch.object(KModel, "forward", forward))
        else:
            import onnxruntime

            original_run = onnxruntime.InferenceSession.run

            def infer(model, *positional, **keywords):
                row = current[0]
                inputs = (
                    positional[1] if len(positional) > 1 else keywords["input_feed"]
                )
                details = {
                    "request_id": row["id"],
                    "class": type(model).__module__ + "." + type(model).__name__,
                    "execution_providers": model.get_providers(),
                    "model_path": str(model._model_path),
                    "speed_input": np.asarray(inputs.get("speed")).tolist(),
                }
                if Path(model._model_path).resolve() != args.model:
                    raise ValueError("Observed ONNX session loaded a different model")
                return observe_native(
                    original_run, row["native_calls"], details, lock=lock
                )(model, *positional, **keywords)

            observers.enter_context(
                patch.object(onnxruntime.InferenceSession, "run", infer)
            )
        evidence["runtime_files"] = {
            name: {
                "path": str(Path(module.__file__).resolve()),
                "sha256": sha256(Path(module.__file__)),
            }
            for name, module in list(sys.modules.items())
            if name
            in {
                "kokoro.model",
                "torch._C",
                "kokoro_onnx",
                "onnxruntime.capi.onnxruntime_pybind11_state",
            }
            and getattr(module, "__file__", None)
        }
        try:
            async with host.run_test(size=(150, 65), notifications=True) as pilot:
                await wait_until(
                    lambda: (
                        host.pane.is_mounted
                        and not host.pane.query_one(
                            "#tts-generate-btn", Button
                        ).disabled
                    ),
                    "fresh Speech Lab ready",
                )
                if "playback" in args.scenarios:
                    row = phase("speech-lab")
                    if args.language != "en-us":
                        host.pane.query_one(
                            "#tts-language-select", Select
                        ).value = args.language
                    host.pane.query_one("#tts-text-input", TextArea).text = text
                    await pilot.pause()
                    host.pane.query_one("#tts-generate-btn", Button).press()
                    await wait_until(
                        lambda: bool(host.requests),
                        "Speech Lab actual request admission",
                    )
                    request = host.requests[-1]
                    row["request"] = {
                        "model_id": request.model_id,
                        "voice_id": request.voice_id,
                        "speed": request.speed,
                    }
                    if request.voice_id != args.voice or request.speed != args.speed:
                        raise ValueError(
                            "Fresh Speech Lab voice/speed did not follow configured defaults"
                        )
                    await wait_until(
                        lambda: (
                            host.lab_handler._current_playground_artifact is not None
                            or any(
                                notice.get("severity") == "error"
                                for notice in host.notices
                            )
                        ),
                        "Speech Lab artifact",
                    )
                    if host.lab_handler._current_playground_artifact is None:
                        raise ValueError(
                            f"Speech Lab generation failed: {host.notices}"
                        )
                    await wait_until(
                        lambda: (
                            not host.pane.query_one("#audio-play-btn", Button).disabled
                        ),
                        "Speech Lab Play enabled",
                    )
                    host.pane.query_one("#audio-play-btn", Button).press()
                    await complete(row, lab=True)
                row = phase("console-warmup")
                await submit(row)
                await complete(row)
                if "cancel" in args.scenarios:
                    cancelled = phase("console-cancel", cancel_text)
                    lifecycle = await submit(cancelled)
                    await wait_until(
                        lambda: any(
                            "exit_at" not in call for call in cancelled["native_calls"]
                        ),
                        "delegated native inference entry",
                    )
                    await asyncio.sleep(0.01)
                    with lock:
                        if not any(
                            "exit_at" not in call for call in cancelled["native_calls"]
                        ):
                            raise ValueError(
                                "Native call completed before cancellation; use longer --cancel-text-file"
                            )
                        cancelled["stop_requested_at"] = time.monotonic()
                        cancelled["resources_at_stop"] = resources()
                    stop_outcomes = []
                    await bounded(
                        handler.handle_tts_playback(
                            TTSPlaybackEvent(
                                "stop",
                                message_id=lifecycle.message_id,
                                playback_lifecycle=lifecycle,
                                outcome_callback=stop_outcomes.append,
                            )
                        ),
                        "Stop cleanup",
                        args.cleanup_timeout,
                    )
                    cancelled["stop_returned_at"] = time.monotonic()
                    await quiescent(cancelled)
                    validate_cancellation(cancelled)
                    if (
                        stop_outcomes != [True]
                        or cancelled["states"] != ["stopped"]
                        or cancelled["devices"]
                        or "playback" in cancelled
                    ):
                        raise ValueError(
                            "Cancelled inference produced audio or did not settle exactly once"
                        )
                    cancelled["outcome"] = "cancelled"
                    frozen = json.dumps(cancelled, sort_keys=True)
                    checkpoint()
                    row = phase("console-successor")
                    await submit(row)
                    await complete(row)
                    if json.dumps(cancelled, sort_keys=True) != frozen:
                        raise ValueError(
                            "Old cancellation changed after successor playback"
                        )
                if "repeat" in args.scenarios:
                    for index in range(args.repeats):
                        row = phase(f"console-repeat-{index + 1:02d}")
                        await submit(row)
                        await complete(row)
            evidence["status"] = "runtime_passed_content_pending"
        finally:
            cleanup_errors = []
            for label, cleanup in (
                ("console", handler.cleanup_tts_resources),
                ("speech_lab", host.lab_handler.cleanup_tts_resources),
                ("service_close", service.close),
                ("service_wait_closed", service.wait_closed),
            ):
                try:
                    await bounded(cleanup(), label, args.cleanup_timeout)
                except BaseException as error:  # noqa: BLE001 -- report every cleanup failure, then fail the run
                    cleanup_errors.append(
                        {
                            "operation": label,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
            if hasattr(host, "audio_player"):
                await bounded(
                    host.audio_player.cleanup(),
                    "async player cleanup",
                    args.cleanup_timeout,
                )
            player.cleanup()
            reset_tts_service_binding(expected=service)
            evidence.update(
                cleanup_errors=cleanup_errors,
                final_resources=resources(),
                notices=host.notices,
                completions=host.completions,
                memory_after_cleanup=memory_snapshot(cuda_device=cuda_device),
                source_hashes_after=_source_hashes(package),
            )
            evidence["source_unchanged"] = (
                evidence["source_hashes"] == evidence["source_hashes_after"]
            )
            if cleanup_errors or any(evidence["final_resources"].values()):
                evidence["status"] = "failed_cleanup"
            checkpoint()
            assert_quiescent(evidence["final_resources"])
            if cleanup_errors or not evidence["source_unchanged"]:
                raise ValueError("Cleanup failed or source changed during the run")
            evidence["cleanup_joined"] = True


def main(argv: Sequence[str] | None = None) -> int:
    """Admit a live run and supervise its isolated worker to completion.

    Args:
        argv: CLI arguments, or None to use the process arguments.

    Returns:
        Zero for validated completion, otherwise a failing worker/control status.

    Raises:
        SystemExit: Help is requested or admission fails.
        OSError: Private output or worker creation fails."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.worker:
        setup_environment(args.output)
        sys.addaudithook(_deny_network)
        evidence = {
            "schema_version": 1,
            "status": "running",
            "pid": os.getpid(),
            "started_utc": datetime.now(UTC).isoformat(),
            "phases": [],
            "arguments": vars(args),
            "python": sys.version,
            "executable": sys.executable,
            "platform": sys.platform,
            "scope": "Mounted Speech pane and trusted Console handler, real models/codecs/device; no full-shell or acoustic loopback claim.",
        }
        try:
            asyncio.run(run_live(args, evidence))
            return 0
        except BaseException as error:  # noqa: BLE001 -- preserve partial evidence at the worker's exit boundary
            if evidence["status"] != "failed_cleanup":
                evidence["status"] = "failed"
            evidence["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            traceback.print_exc()
            return 1
        finally:
            evidence["ended_utc"] = datetime.now(UTC).isoformat()
            write_evidence(args.output / "evidence.json", evidence)
    args.output.mkdir(mode=0o700, parents=False)
    command = [sys.executable, "-B"]
    if sys.flags.isolated:
        command.append("-I")
    command += [
        str(Path(__file__).resolve()),
        *(sys.argv[1:] if argv is None else argv),
        "--worker",
    ]
    environment = {**os.environ, "TLDW_LIVE_VALIDATION_WORKER": str(args.output)}
    with (args.output / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(
            f"Live worker {process.pid}; log: {args.output / 'worker.log'}", flush=True
        )
        try:
            code = process.wait(timeout=args.run_timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            path = args.output / "evidence.json"
            report = json.loads(path.read_text()) if path.exists() else {}
            report.update(
                status="failed_cleanup",
                forced_worker_exit=True,
                worker_exit=process.returncode,
            )
            write_evidence(path, report)
            code = 1
    print(f"Evidence: {args.output / 'evidence.json'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
