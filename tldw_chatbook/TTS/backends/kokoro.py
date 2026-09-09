# kokoro.py
# Description: Kokoro TTS backend implementation supporting both ONNX and PyTorch
#
from __future__ import annotations

# Imports
import asyncio
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import wave
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from loguru import logger

# Optional requests import for model downloading
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

# Optional numpy import
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    logger.warning("numpy not available. Kokoro TTS backend will not function.")

# Local imports
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.TTS._async_lifecycle import join_retained_task
from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.audio_service import get_audio_service
from tldw_chatbook.TTS.base_backends import LocalTTSBackend
from tldw_chatbook.TTS.text_processing import TextChunker, TextNormalizer
from tldw_chatbook.TTS.voice_blend_paths import (
    default_kokoro_backend_blend_directory,
    write_private_json,
)
from tldw_chatbook.Utils.path_validation import (
    validate_filename,
    validate_path,
    validate_path_simple,
)
from tldw_chatbook.Utils.private_paths import (
    secure_private_directory,
    verify_trusted_directory,
)

#######################################################################################################################
#
# Kokoro TTS Backend Implementation


#: Connect / read timeouts for every Kokoro asset download (task-19560).
#:
#: `requests.get(..., stream=True)` with no timeout waits forever on a
#: half-open connection. These downloads are hundreds of megabytes and ran
#: inline on the event loop, so a stalled CDN froze the whole TUI with no
#: error and no way out. The read timeout applies per-chunk, not to the whole
#: transfer, so a slow-but-progressing download is not killed.
KOKORO_DOWNLOAD_CONNECT_TIMEOUT = 15.0
KOKORO_DOWNLOAD_READ_TIMEOUT = 60.0
KOKORO_DOWNLOAD_TIMEOUT = (
    KOKORO_DOWNLOAD_CONNECT_TIMEOUT,
    KOKORO_DOWNLOAD_READ_TIMEOUT,
)

#: Log a progress line at most this often during a long download.
_KOKORO_PROGRESS_INTERVAL_SECONDS = 2.0

# Encoded requests retain at most five minutes of 24 kHz mono float32 audio
# (28.8 MB). Raw PCM does not require a complete-file buffer.
KOKORO_MAX_ENCODED_AUDIO_SAMPLES = 24000 * 300

# All generation paths infer the same language from official voice prefixes.
KOKORO_VOICE_LANGUAGES = {
    "a": "en-us",
    "b": "en-gb",
    "j": "ja",
    "z": "zh",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}


def _check_audio_sample_limit(total_samples: int, max_samples: int | None) -> None:
    """Reject an encoded utterance before retaining samples beyond its budget."""
    if max_samples is not None and total_samples > max_samples:
        raise TTSOperationError(
            code="request_invalid",
            message=(
                "Kokoro encoded audio is limited to five minutes per request. "
                "Shorten the text or choose PCM for longer speech."
            ),
            retryable=False,
            operation_id=uuid4().hex,
            recovery_action="shorten_text_or_use_pcm",
        )


def _append_audio_samples(
    buffer: bytearray, samples: np.ndarray, max_samples: int | None
) -> None:
    """Copy samples into one growable float32 buffer within its sample budget."""
    sample_count = int(np.size(samples))
    _check_audio_sample_limit(len(buffer) // 4 + sample_count, max_samples)
    if sample_count:
        buffer.extend(np.asarray(samples, dtype=np.float32).tobytes())


def _kokoro_stream_download(
    url: str,
    destination: str,
    *,
    label: str,
    hasher: "hashlib._Hash | None" = None,
) -> str:
    """Stream ``url`` to ``destination`` atomically, with timeouts and progress.

    Blocking by design -- callers run it via ``asyncio.to_thread`` so the event
    loop stays responsive (task-19560).

    Writes to an exclusively created ``.part`` sibling and replaces the target after
    the body is fully read, so an interrupted or failed download can never
    leave a truncated file that the next run's ``os.path.exists`` check treats
    as a complete model.

    Args:
        url: Source URL.
        destination: Final path to place the file at.
        label: Human-readable name used in progress logs.
        hasher: Optional hash object updated with each chunk.

    Returns:
        The destination path.

    Raises:
        requests.RequestException: On any transport failure or timeout.
    """
    target = Path(destination)
    destination = str(validate_path(target.name, target.parent, redact_paths=True))
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    partial = None

    try:
        with requests.get(
            url, stream=True, timeout=KOKORO_DOWNLOAD_TIMEOUT
        ) as response:
            response.raise_for_status()
            total = response.headers.get("content-length")
            total_bytes = int(total) if total and total.isdigit() else 0
            written = 0
            last_log = time.monotonic()

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f"{os.path.basename(destination)}.",
                suffix=".part",
                dir=parent,
                delete=False,
            ) as handle:
                partial = handle.name
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    if hasher is not None:
                        hasher.update(chunk)
                    written += len(chunk)

                    now = time.monotonic()
                    if now - last_log >= _KOKORO_PROGRESS_INTERVAL_SECONDS:
                        if total_bytes:
                            logger.info(
                                f"{label}: {written / 1048576:.1f} MB of "
                                f"{total_bytes / 1048576:.1f} MB "
                                f"({100 * written / total_bytes:.0f}%)"
                            )
                        else:
                            logger.info(
                                f"{label}: {written / 1048576:.1f} MB downloaded"
                            )
                        last_log = now

        os.replace(partial, destination)
        logger.info(f"{label}: complete ({written / 1048576:.1f} MB)")
        return destination
    except BaseException:
        # Includes cancellation: never leave a partial file behind that the
        # next run would mistake for a finished download.
        try:
            if partial is not None and os.path.exists(partial):
                os.remove(partial)
        except OSError as cleanup_exc:
            # The configured directory is user data; keep it out of diagnostics.
            logger.debug(
                f"{label}: could not remove the partial file; "
                f"error_type={type(cleanup_exc).__name__}"
            )
        raise


class KokoroTTSBackend(LocalTTSBackend):
    """
    Kokoro Text-to-Speech backend supporting both ONNX and PyTorch models.

    Features:
    - Voice mixing with weighted combinations
    - Advanced text chunking with token limits
    - Performance metrics tracking
    - Phoneme generation support

    References:
    - https://github.com/thewh1teagle/kokoro-onnx
    - https://huggingface.co/hexgrad/Kokoro-82M
    - https://github.com/remsky/Kokoro-FastAPI
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Check numpy availability
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "numpy is required for Kokoro TTS backend but is not installed. "
                "Install it with: pip install numpy"
            )

        # Check if we're on Windows and pre-validate ONNX dependencies
        if sys.platform == "win32":
            try:
                import onnxruntime  # noqa: F401

                logger.debug("onnxruntime available on Windows")
            except ImportError:
                logger.warning(
                    "onnxruntime not available on Windows - Kokoro ONNX backend may not work. "
                    "Install with: pip install onnxruntime\n"
                    "If you continue to have issues, you may need to install the Microsoft Visual C++ Redistributable."
                )

        # Lazy-loaded heavy dependencies
        self._torch = None
        self._kokoro_onnx = None
        self._kokoro_pt_modules = None
        self._native_tasks: dict[asyncio.Task[Any], Callable[[], None] | None] = {}
        self._onnx_tasks: dict[asyncio.Task[None], Callable[[], None]] = {}
        self._onnx_phonemizers: dict[str, Any] = {}
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

        # Configuration
        self.use_onnx = self.config.get("KOKORO_USE_ONNX", True)
        self.model_path = self.config.get("KOKORO_MODEL_PATH")
        self.voices_json = self.config.get("KOKORO_VOICES_JSON_PATH")
        self.voice_dir = self.config.get("KOKORO_VOICE_DIR_PT")
        self.device = self.config.get("KOKORO_DEVICE", "cpu")

        # Try to get paths from CLI config if not provided
        if not self.model_path:
            self.model_path = get_cli_setting(
                "app_tts",
                "KOKORO_ONNX_MODEL_PATH_DEFAULT"
                if self.use_onnx
                else "KOKORO_PT_MODEL_PATH_DEFAULT",
                "kokoro-v1.0.onnx" if self.use_onnx else None,
            )
        if not self.voices_json:
            self.voices_json = get_cli_setting(
                "app_tts", "KOKORO_ONNX_VOICES_JSON_DEFAULT", "voices-v1.0.bin"
            )

        # Model instances
        self.kokoro_instance = None  # ONNX instance
        self.kokoro_model_pt = None  # PyTorch model

        # Services
        self.audio_service = get_audio_service()
        self._max_tokens = int(self.config.get("KOKORO_MAX_TOKENS", 500))
        self.text_chunker = TextChunker(max_tokens=self._max_tokens)
        self.normalizer = TextNormalizer()

        # Voice mixing configuration
        self.enable_voice_mixing = self.config.get("KOKORO_ENABLE_VOICE_MIXING", False)

        # Voice blend storage
        configured_blends_dir = self.config.get("KOKORO_VOICE_BLENDS_DIR")
        if configured_blends_dir is None:
            configured_blends_dir = get_cli_setting(
                "app_tts", "KOKORO_VOICE_BLENDS_DIR", None
            )
        self._voice_blends_directory_is_application_owned = (
            configured_blends_dir is None
        )
        if self._voice_blends_directory_is_application_owned:
            self.voice_blends_dir = default_kokoro_backend_blend_directory()
            secure_private_directory(
                self.voice_blends_dir,
                create=True,
                application_owned=True,
            )
        else:
            self.voice_blends_dir = validate_path_simple(
                Path(configured_blends_dir).expanduser(),
                probe_existing=False,
            )
            if self.voice_blends_dir.exists():
                verify_trusted_directory(
                    self.voice_blends_dir,
                    allow_shared_sticky=False,
                )
            else:
                secure_private_directory(
                    self.voice_blends_dir,
                    create=True,
                    application_owned=True,
                )
        self.saved_blends = self._load_saved_blends()

        # Initialize default blends if none exist
        if not self.saved_blends:
            self._create_default_blends()

        # Performance tracking
        self.track_performance = self.config.get("KOKORO_TRACK_PERFORMANCE", True)
        self._performance_metrics = {
            "total_tokens": 0,
            "total_time": 0.0,
            "generation_count": 0,
        }

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self._max_tokens = int(value)
        if hasattr(self, "text_chunker"):
            self.text_chunker.max_tokens = self._max_tokens
            self.text_chunker.min_chunk_size = max(10, int(self._max_tokens * 0.1))

    async def initialize(self):
        """Initialize the Kokoro backend"""
        logger.info(
            f"KokoroTTSBackend: Initializing (ONNX: {self.use_onnx}, Device: {self.device})"
        )

        # Ensure paths are initialized regardless of backend
        if not self.model_path:
            from pathlib import Path

            model_dir = Path.home() / ".config" / "tldw_cli" / "models" / "kokoro"
            model_dir.mkdir(parents=True, exist_ok=True)
            # Default model based on backend type
            if self.use_onnx:
                self.model_path = str(model_dir / "kokoro-v1.0.onnx")
            else:
                self.model_path = str(model_dir / "kokoro-v1_0.pth")

        if not self.voice_dir:
            from pathlib import Path

            voice_dir = (
                Path.home() / ".config" / "tldw_cli" / "models" / "kokoro" / "voices"
            )
            voice_dir.mkdir(parents=True, exist_ok=True)
            self.voice_dir = str(voice_dir)

        # Ensure voices.json has proper path
        if self.voices_json and not os.path.isabs(self.voices_json):
            from pathlib import Path

            model_dir = Path.home() / ".config" / "tldw_cli" / "models" / "kokoro"
            self.voices_json = str(model_dir / self.voices_json)

        await self.load_model()

    async def load_model(self):
        """Load the TTS model into memory"""
        if self.use_onnx:
            await self._initialize_onnx()
            # If ONNX initialization failed, try PyTorch
            if not self.use_onnx:
                logger.info("ONNX initialization failed, falling back to PyTorch")
                # Update model path for PyTorch
                from pathlib import Path

                model_dir = Path.home() / ".config" / "tldw_cli" / "models" / "kokoro"
                self.model_path = self.config.get(
                    "KOKORO_PT_MODEL_PATH_DEFAULT"
                ) or str(model_dir / "kokoro-v1_0.pth")
                await self._initialize_pytorch()
        else:
            await self._initialize_pytorch()

        self.model_loaded = True

    async def _initialize_onnx(self):
        """Initialize ONNX backend"""
        try:
            # Try to import kokoro_onnx
            try:
                kokoro_module = self.kokoro_onnx_module
                Kokoro = kokoro_module.Kokoro
                EspeakConfig = kokoro_module.EspeakConfig
            except ImportError as e:
                logger.error(f"Failed to import kokoro_onnx: {e}")
                self.use_onnx = False
                return
            except Exception as e:
                logger.error(f"Failed to load kokoro_onnx classes: {e}")
                self.use_onnx = False
                return

            # Check if model files exist
            if not os.path.exists(self.model_path):
                logger.info(f"Kokoro ONNX model not found at {self.model_path}")
                # Download the model with checksum verification
                tmp_path = None
                try:
                    logger.info("Downloading Kokoro ONNX model...")
                    if not REQUESTS_AVAILABLE:
                        raise ImportError(
                            "requests library required for model download"
                        )

                    url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
                    # Expected SHA256 checksum for kokoro-v0_19.onnx

                    # task-19560: the transfer runs off the event loop with
                    # timeouts; the checksum-verify + move below is unchanged.
                    hasher = hashlib.sha256()
                    # Qodo #1 + #3: the previous form nested a string literal
                    # inside an f-string expression, which is only legal from
                    # Python 3.12 (PEP 701) -- on this project's 3.11 floor the
                    # whole module failed to import. It also used a
                    # pid-deterministic name, so two concurrent initialisations
                    # in one process clobbered each other's in-flight download
                    # and the checksum/move then operated on the wrong file.
                    # mkstemp gives a unique path and creates it 0600.
                    tmp_fd, tmp_path = tempfile.mkstemp(
                        prefix="kokoro-download-", suffix="-model.onnx"
                    )
                    os.close(tmp_fd)
                    await self._run_native_work(
                        _kokoro_stream_download,
                        url,
                        tmp_path,
                        label="Kokoro ONNX model",
                        hasher=hasher,
                    )

                    # Verify checksum
                    actual_checksum = hasher.hexdigest()
                    logger.info(f"Downloaded file checksum: {actual_checksum}")

                    # Note: For now, we'll just log the checksum since we don't have the actual expected value
                    # In production, you should verify against known good checksums
                    logger.warning(
                        "Checksum verification skipped - no known checksum available"
                    )

                    # Move to final location
                    model_dir = (
                        os.path.dirname(self.model_path)
                        if os.path.dirname(self.model_path)
                        else "."
                    )
                    os.makedirs(model_dir, exist_ok=True)
                    shutil.move(tmp_path, self.model_path)

                    logger.info(f"Downloaded ONNX model to {self.model_path}")
                except Exception as e:
                    logger.error(f"Failed to download ONNX model: {e}")
                    self.use_onnx = False
                    return
                finally:
                    if tmp_path is not None:
                        Path(tmp_path).unlink(missing_ok=True)

            if not os.path.exists(self.voices_json):
                logger.info(f"Kokoro voices file not found at {self.voices_json}")
                # Download the voices.json with checksum verification
                tmp_path = None
                try:
                    logger.info("Downloading Kokoro voices file...")
                    if not REQUESTS_AVAILABLE:
                        raise ImportError(
                            "requests library required for voices.json download"
                        )

                    url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

                    # task-19560: the transfer runs off the event loop with
                    # timeouts; the checksum-verify + move below is unchanged.
                    hasher = hashlib.sha256()
                    # Qodo #1 + #3: the previous form nested a string literal
                    # inside an f-string expression, which is only legal from
                    # Python 3.12 (PEP 701) -- on this project's 3.11 floor the
                    # whole module failed to import. It also used a
                    # pid-deterministic name, so two concurrent initialisations
                    # in one process clobbered each other's in-flight download
                    # and the checksum/move then operated on the wrong file.
                    # mkstemp gives a unique path and creates it 0600.
                    tmp_fd, tmp_path = tempfile.mkstemp(
                        prefix="kokoro-download-", suffix="-voices.bin"
                    )
                    os.close(tmp_fd)
                    await self._run_native_work(
                        _kokoro_stream_download,
                        url,
                        tmp_path,
                        label="Kokoro voices file",
                        hasher=hasher,
                    )

                    # Log checksum for future reference
                    actual_checksum = hasher.hexdigest()
                    logger.info(f"Downloaded voices file checksum: {actual_checksum}")

                    # Move to final location
                    os.makedirs(os.path.dirname(self.voices_json), exist_ok=True)
                    shutil.move(tmp_path, self.voices_json)

                    logger.info(f"Downloaded voices file to {self.voices_json}")
                except Exception as e:
                    logger.error(f"Failed to download voices file: {e}")
                    self.use_onnx = False
                    return
                finally:
                    if tmp_path is not None:
                        Path(tmp_path).unlink(missing_ok=True)

            # Check for espeak
            espeak_lib = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
            espeak_config = EspeakConfig(lib_path=espeak_lib) if espeak_lib else None

            # Create Kokoro instance
            self.kokoro_instance = await self._run_native_work(
                Kokoro, self.model_path, self.voices_json, espeak_config=espeak_config
            )

            logger.info("KokoroTTSBackend: ONNX backend initialized successfully")

        except Exception as e:
            logger.opt(exception=True).error(
                f"KokoroTTSBackend: Failed to initialize ONNX backend: {e}"
            )
            self.use_onnx = False

    @property
    def torch(self):
        """Lazy load torch module"""
        if self._torch is None:
            try:
                import torch

                self._torch = torch
            except ImportError:
                raise ImportError(
                    "PyTorch is required but not installed. Install with: pip install torch"
                )
        return self._torch

    @property
    def kokoro_onnx_module(self):
        """Lazy load kokoro_onnx module"""
        if self._kokoro_onnx is None:
            try:
                import kokoro_onnx

                self._kokoro_onnx = kokoro_onnx
                logger.info("Successfully imported kokoro_onnx module")
            except ImportError as e:
                logger.error(f"ImportError when loading kokoro_onnx: {e}")
                raise ImportError(
                    "kokoro_onnx not installed. Please install with: pip install kokoro-onnx"
                )
            except Exception as e:
                # On Windows, sometimes there are DLL or other loading issues
                logger.error(
                    f"Unexpected error loading kokoro_onnx: {type(e).__name__}: {e}"
                )
                import sys

                if sys.platform == "win32":
                    raise ImportError(
                        "Failed to load kokoro_onnx on Windows. This may be due to missing dependencies. "
                        "Please ensure you have installed ALL of the following:\n"
                        "1. pip install kokoro-onnx\n"
                        "2. pip install onnxruntime (or onnxruntime-gpu for GPU support)\n"
                        "3. pip install numpy\n"
                        "4. Microsoft Visual C++ Redistributable (if not already installed)\n"
                        f"Error details: {type(e).__name__}: {e}"
                    )
                else:
                    raise ImportError(f"Failed to load kokoro_onnx: {e}")
        return self._kokoro_onnx

    async def _initialize_pytorch(self):
        """Initialize PyTorch backend"""
        try:
            # Check device
            if self.device == "cuda" and not self.torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"

            # Load model if it exists, otherwise mark for download
            if os.path.exists(self.model_path):
                await self._run_native_work(self._load_pytorch_model)
            else:
                from tldw_chatbook.TTS.kokoro_pytorch import require_runtime

                await self._run_native_work(require_runtime)
                logger.warning(f"Kokoro PyTorch model not found at {self.model_path}")
                # Model download will happen on first use

            logger.info(
                "KokoroTTSBackend: PyTorch backend initialized (model loading deferred)"
            )

        except ImportError as error:
            raise TTSOperationError(
                code="dependency_missing",
                message=(
                    "Kokoro PyTorch needs 'tldw_chatbook[local_tts]' on Python "
                    "3.11 or 3.12; select ONNX on Python 3.13+."
                ),
                retryable=False,
                operation_id="kokoro_pytorch",
                recovery_action="install_kokoro_pytorch",
            ) from error
        except Exception as e:
            logger.opt(exception=True).error(
                f"KokoroTTSBackend: Failed to initialize PyTorch backend: {e}"
            )
            raise

    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech using Kokoro and stream the response.

        Args:
            request: Speech request parameters

        Yields:
            Audio bytes in the requested format
        """
        # Ensure we're initialized
        if not self.model_loaded:
            await self.initialize()

        stream = (
            self._generate_onnx_stream(request)
            if self.use_onnx and self.kokoro_instance
            else self._generate_pytorch_stream(request)
        )
        async with aclosing(stream):
            async for chunk in stream:
                yield chunk

    async def _create_onnx_stream(self, text: str, *args, **kwargs):
        """Join the upstream stream's native executor before releasing its owner.

        kokoro-onnx cancels its producer on generator close, but cancellation
        cannot stop a batch already running in its default executor. A private
        loop owns that executor; its shutdown joins the actual worker. The
        cross-loop handoff waits for consumption without adding native prefetch.
        """
        if self._closing:
            raise RuntimeError("Kokoro backend is closing")
        runtime = self.kokoro_instance
        parent_loop = asyncio.get_running_loop()
        chunks = asyncio.Queue(maxsize=1)
        stopped = threading.Event()
        lock = threading.Lock()
        owner = []

        def stop():
            stopped.set()
            with lock:
                if owner:
                    loop, producer = owner[0]
                    loop.call_soon_threadsafe(producer.cancel)

        async def deliver(chunk):
            await chunks.put(chunk)
            await chunks.join()

        async def produce():
            from tldw_chatbook.TTS.kokoro_languages import prepare_onnx_text

            stream_text, stream_options = text, dict(kwargs)
            if not stream_options.get("is_phonemes"):
                stream_text, language, phonemes = prepare_onnx_text(
                    text, stream_options.get("lang", "en-us"), self._onnx_phonemizers
                )
                stream_options["lang"] = language
                if phonemes:
                    stream_options["is_phonemes"] = True
            # A synchronous frontend can finish after Stop was queued on this
            # worker's loop. Do not let it start an inference call afterward.
            if stopped.is_set():
                return
            async with aclosing(
                runtime.create_stream(stream_text, *args, **stream_options)
            ) as stream:
                async for chunk in stream:
                    if stopped.is_set():
                        break
                    delivery = asyncio.run_coroutine_threadsafe(
                        deliver(chunk), parent_loop
                    )
                    await asyncio.wrap_future(delivery)

        async def run():
            loop = asyncio.get_running_loop()
            producer = asyncio.create_task(produce())
            with lock:
                owner.append((loop, producer))
                if stopped.is_set():
                    producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                if not stopped.is_set():
                    raise
            except TTSOperationError:
                raise
            except Exception as exc:
                raise TTSOperationError(
                    code="generation_failed",
                    message="Kokoro ONNX generation failed.",
                    retryable=True,
                    operation_id="kokoro_onnx",
                    recovery_action="retry",
                ) from exc
            finally:
                with lock:
                    owner.clear()
                # Explicitly join without Runner's finite default timeout.
                # Only the producer is cancelled; repeated Stop cannot cancel
                # this executor shutdown and abandon a live native thread.
                await loop.shutdown_default_executor()

        worker = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(run())))
        self._onnx_tasks[worker] = stop
        receive = None
        try:
            while not worker.done() or not chunks.empty():
                receive = asyncio.create_task(chunks.get())
                done, _ = await asyncio.wait(
                    (receive, worker), return_when=asyncio.FIRST_COMPLETED
                )
                if receive in done:
                    if self._closing:
                        raise asyncio.CancelledError
                    yield receive.result()
                    chunks.task_done()
                else:
                    receive.cancel()
                    break
            await join_retained_task(worker)
            if self._closing:
                raise asyncio.CancelledError
        finally:
            stop()
            if receive is not None:
                receive.cancel()
            try:
                await join_retained_task(worker)
            finally:
                self._onnx_tasks.pop(worker, None)

    async def _generate_onnx_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio using ONNX backend with advanced features"""
        start_time = time.time()

        try:
            # Parse voice for potential mixing
            voice_config = self._parse_voice_config(request.voice)

            # Detect language from voice or use provided language code
            if (
                hasattr(request, "extra_params")
                and request.extra_params
                and "language" in request.extra_params
            ):
                lang = request.extra_params["language"]
            else:
                # Map voice prefix to espeak language codes
                # kokoro voices: 'af_' = American Female, 'am_' = American Male,
                #                'bf_' = British Female, 'bm_' = British Male
                if (
                    voice_config["primary_voice"]
                    and len(voice_config["primary_voice"]) > 0
                ):
                    voice_prefix = voice_config["primary_voice"][0].lower()
                    lang = KOKORO_VOICE_LANGUAGES.get(voice_prefix, "en-us")
                else:
                    lang = "en-us"  # Default to American English

            # Normalize text if requested
            text = request.input
            if request.normalization_options:
                text = self.normalizer.normalize_text(text)

            logger.info(
                f"KokoroTTSBackend: Generating audio for {len(text)} characters, "
                f"voice={voice_config}, lang={lang}, format={request.response_format}"
            )

            # For PCM output, we can stream directly
            if request.response_format == "pcm":
                token_count = 0
                estimated_total_tokens = len(text.split()) * 2  # Rough estimate
                samples_processed = 0

                # Report initial progress
                await self._report_progress(
                    progress=0.0,
                    processed=0,
                    total=estimated_total_tokens,
                    status="Starting PCM audio generation",
                    metrics={"format": "pcm"},
                )

                audio_generator = (
                    self._generate_mixed_voice(
                        text, voice_config, speed=request.speed, lang=lang
                    )
                    if voice_config["is_mixed"]
                    else self._create_onnx_stream(
                        text,
                        voice=voice_config["primary_voice"],
                        speed=request.speed,
                        lang=lang,
                    )
                )
                async with aclosing(audio_generator):
                    async for samples, sample_rate in audio_generator:
                        token_count += len(samples) // 256
                        samples_processed += len(samples)
                        progress = (
                            min(0.95, token_count / estimated_total_tokens)
                            if estimated_total_tokens > 0
                            else 0.5
                        )
                        await self._report_progress(
                            progress=progress,
                            processed=token_count,
                            total=estimated_total_tokens,
                            status=f"Streaming PCM audio: {samples_processed / sample_rate:.1f}s generated",
                            metrics={
                                "sample_rate": sample_rate,
                                "samples_generated": samples_processed,
                                "format": "pcm",
                            },
                        )
                        int16_samples = np.int16(samples * 32767)
                        yield int16_samples.tobytes()

                # Update performance metrics
                if self.track_performance:
                    self._update_performance_metrics(
                        token_count, time.time() - start_time
                    )

                # Report completion
                await self._report_progress(
                    progress=1.0,
                    processed=token_count,
                    total=estimated_total_tokens,
                    status=f"PCM generation complete: {samples_processed / sample_rate:.1f}s of audio",
                    metrics={
                        "sample_rate": sample_rate,
                        "samples_generated": samples_processed,
                        "format": "pcm",
                        "generation_time": time.time() - start_time,
                    },
                )

            else:
                # File formats need one encoder/container for the complete
                # utterance. Concatenating separately encoded chunks leaves
                # headers describing only the first chunk (e.g. MP3 Xing or
                # FLAC STREAMINFO), so decoders can silently truncate playback.
                # Raw PCM above remains incremental.
                sample_rate = 24000
                token_count = 0
                total_samples = 0
                sample_buffer = bytearray()
                if voice_config["is_mixed"]:
                    audio_generator = self._generate_mixed_voice(
                        text,
                        voice_config,
                        speed=request.speed,
                        lang=lang,
                        max_samples=KOKORO_MAX_ENCODED_AUDIO_SAMPLES,
                    )
                else:
                    audio_generator = self._create_onnx_stream(
                        text,
                        voice=voice_config["primary_voice"],
                        speed=request.speed,
                        lang=lang,
                    )

                estimated_total_tokens = len(text.split()) * 2
                async with aclosing(audio_generator):
                    async for samples, sr in audio_generator:
                        sample_rate = sr
                        _append_audio_samples(
                            sample_buffer, samples, KOKORO_MAX_ENCODED_AUDIO_SAMPLES
                        )
                        total_samples = len(sample_buffer) // 4
                        token_count += len(samples) // 256
                        progress = (
                            min(0.95, token_count / estimated_total_tokens)
                            if estimated_total_tokens > 0
                            else 0.5
                        )
                        await self._report_progress(
                            progress=progress,
                            processed=token_count,
                            total=estimated_total_tokens,
                            status=f"Collecting audio: {total_samples / sample_rate:.1f}s",
                            metrics={
                                "sample_rate": sample_rate,
                                "samples_collected": total_samples,
                                "format": request.response_format,
                            },
                        )

                if not total_samples:
                    logger.warning("KokoroTTSBackend: No audio generated")
                    yield b""
                    return
                full_audio = np.frombuffer(sample_buffer, dtype=np.float32)
                try:
                    audio_bytes = await self.audio_service.convert_audio(
                        full_audio,
                        request.response_format,
                        source_format="pcm",
                        sample_rate=sample_rate,
                    )
                except (RuntimeError, ValueError) as error:
                    logger.error(
                        "KokoroTTSBackend: Audio conversion failed ({})",
                        type(error).__name__,
                    )
                    yield b""
                    return
                del full_audio, sample_buffer
                yield audio_bytes

                generation_time = time.time() - start_time
                audio_duration = total_samples / sample_rate
                speed_factor = (
                    audio_duration / generation_time if generation_time > 0 else 0
                )
                if self.track_performance:
                    self._update_performance_metrics(token_count, generation_time)
                logger.info(
                    "KokoroTTSBackend: Generated {:.2f}s of {} audio in {:.2f}s "
                    "({:.1f}x realtime)",
                    audio_duration,
                    request.response_format,
                    generation_time,
                    speed_factor,
                )
                await self._report_progress(
                    progress=1.0,
                    processed=token_count,
                    total=estimated_total_tokens,
                    status=f"Generation complete: {audio_duration:.1f}s of {request.response_format} audio",
                    metrics={
                        "sample_rate": sample_rate,
                        "samples_generated": total_samples,
                        "format": request.response_format,
                        "generation_time": generation_time,
                        "speed_factor": speed_factor,
                    },
                )

        except TTSOperationError:
            raise
        except Exception as e:
            logger.opt(exception=True).error(
                f"KokoroTTSBackend: Error during ONNX generation: {e}"
            )
            raise TTSOperationError(
                code="generation_failed",
                message="Kokoro ONNX generation failed.",
                retryable=True,
                operation_id="kokoro_onnx",
                recovery_action="retry",
            ) from e

    def _parse_voice_config(self, voice_str: str) -> Dict[str, Any]:
        """Parse voice string for potential mixing configuration or preset"""
        # Check if it's a saved blend preset (starts with "blend:")
        if voice_str.startswith("blend:"):
            preset_name = voice_str[6:]  # Remove "blend:" prefix
            blend_str = self.create_blend_from_preset(preset_name)
            if blend_str:
                voice_str = blend_str
                logger.info(f"Using saved blend preset '{preset_name}': {blend_str}")
            else:
                logger.warning(
                    f"Blend preset '{preset_name}' not found, using default voice"
                )
                voice_str = "af_bella"

        if not self.enable_voice_mixing or ":" not in voice_str:
            # Simple voice without mixing
            return {
                "primary_voice": map_voice_to_kokoro(voice_str),
                "is_mixed": False,
                "voices": [(map_voice_to_kokoro(voice_str), 1.0)],
            }

        # Parse mixed voice format: "voice1:weight1,voice2:weight2"
        voices = []
        total_weight = 0

        for voice_part in voice_str.split(","):
            if ":" in voice_part:
                voice_name, weight_str = voice_part.split(":", 1)
                try:
                    weight = float(weight_str)
                except ValueError:
                    weight = 1.0
            else:
                voice_name = voice_part
                weight = 1.0

            voices.append((map_voice_to_kokoro(voice_name.strip()), weight))
            total_weight += weight

        # Normalize weights to sum to 1.0
        if total_weight > 0:
            voices = [(v, w / total_weight) for v, w in voices]

        return {
            "primary_voice": voices[0][0],  # First voice as primary
            "is_mixed": len(voices) > 1,
            "voices": voices,
        }

    async def _generate_mixed_voice(
        self,
        text: str,
        voice_config: dict[str, Any],
        speed: float,
        lang: str,
        *,
        max_samples: int | None = None,
    ) -> AsyncGenerator[tuple[np.ndarray, int], None]:
        """Generate audio with mixed voices"""
        if not voice_config["is_mixed"]:
            # Fallback to single voice
            stream = self._create_onnx_stream(
                text, voice=voice_config["primary_voice"], speed=speed, lang=lang
            )
            total_samples = 0
            async with aclosing(stream):
                async for samples, sr in stream:
                    total_samples += int(np.size(samples))
                    _check_audio_sample_limit(total_samples, max_samples)
                    yield samples, sr
            return

        # Keep one running mix and one voice buffer, independent of voice count.
        mixed_audio = np.zeros(0, dtype=np.float32)
        sample_rate = 24000

        for voice, weight in voice_config["voices"]:
            sample_buffer = bytearray()
            stream = self._create_onnx_stream(text, voice=voice, speed=speed, lang=lang)
            async with aclosing(stream):
                async for samples, sr in stream:
                    sample_rate = sr
                    _append_audio_samples(sample_buffer, samples, max_samples)
            if not sample_buffer:
                continue
            voice_audio = np.frombuffer(sample_buffer, dtype=np.float32)
            if len(voice_audio) > len(mixed_audio):
                mixed_audio = np.pad(
                    mixed_audio, (0, len(voice_audio) - len(mixed_audio))
                )
            mixed_audio[: len(voice_audio)] += voice_audio * weight
            del voice_audio, sample_buffer

        if not len(mixed_audio):
            return

        # Normalize to prevent clipping
        max_val = np.abs(mixed_audio).max()
        if max_val > 1.0:
            mixed_audio /= max_val

        # Yield in chunks for streaming
        chunk_size = 8192
        for i in range(0, len(mixed_audio), chunk_size):
            chunk = mixed_audio[i : i + chunk_size]
            yield chunk, sample_rate

    def _update_performance_metrics(self, token_count: int, generation_time: float):
        """Update performance tracking metrics"""
        self._performance_metrics["total_tokens"] += token_count
        self._performance_metrics["total_time"] += generation_time
        self._performance_metrics["generation_count"] += 1

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self._performance_metrics["generation_count"] == 0:
            return {
                "average_tokens_per_second": 0,
                "total_generations": 0,
                "total_time": 0,
            }

        return {
            "average_tokens_per_second": self._performance_metrics["total_tokens"]
            / self._performance_metrics["total_time"],
            "total_generations": self._performance_metrics["generation_count"],
            "total_time": self._performance_metrics["total_time"],
            "total_tokens": self._performance_metrics["total_tokens"],
        }

    def _load_pytorch_model(self):
        """Load PyTorch model"""
        try:
            # Import Kokoro PyTorch modules dynamically
            if self._kokoro_pt_modules is None:
                try:
                    from tldw_chatbook.TTS import kokoro_pytorch

                    self._kokoro_pt_modules = {
                        "build_model": kokoro_pytorch.build_model,
                        "generate": kokoro_pytorch.generate,
                        "load_voice": kokoro_pytorch.load_voice,
                        "mix_voices": kokoro_pytorch.mix_voices,
                        "parse_voice_mix": kokoro_pytorch.parse_voice_mix,
                        "get_available_voices": kokoro_pytorch.get_available_voices,
                    }
                except ImportError as e:
                    logger.error(f"Failed to import Kokoro PyTorch modules: {e}")
                    raise

            build_model = self._kokoro_pt_modules["build_model"]
            self.kokoro_model_pt = build_model(self.model_path, device=self.device)
            logger.info(f"Loaded Kokoro PyTorch model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise

    async def _run_native_work(self, function, *args, on_cancel=None, **kwargs):
        """Keep native work owned until it really stops, including cancellation."""
        if self._closing:
            raise RuntimeError("Kokoro backend is closing")
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._native_tasks[task] = on_cancel
        try:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as cancelled:
                if on_cancel is not None:
                    on_cancel()
                try:
                    await join_retained_task(task)
                finally:
                    raise cancelled
            if self._closing:
                raise asyncio.CancelledError
            return task.result()
        finally:
            self._native_tasks.pop(task, None)

    async def _run_pytorch_generation(self, function, *args, **kwargs):
        """Stop between upstream segments while joining the active native call."""
        stopped = threading.Event()
        return await self._run_native_work(
            function,
            *args,
            on_cancel=stopped.set,
            is_cancelled=stopped.is_set,
            **kwargs,
        )

    async def _download_model_if_needed(self):
        """Download Kokoro model if not present"""
        if not os.path.exists(self.model_path):
            logger.info("Downloading Kokoro PyTorch model...")
            try:
                if not REQUESTS_AVAILABLE:
                    raise ImportError("requests library required for model download")
                url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth?download=true"
                # task-19560: off the event loop, with timeouts, and atomic --
                # this is a several-hundred-MB transfer.
                await self._run_native_work(
                    _kokoro_stream_download,
                    url,
                    self.model_path,
                    label="Kokoro PyTorch model",
                )
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise ValueError(
                    "Failed to download Kokoro model. Please download manually."
                )
        await self._run_native_work(self._load_pytorch_model)

    def _voice_pack_path(self, voice: str) -> str:
        """Resolve a named pack within its configured directory before any I/O."""
        filename = f"{validate_filename(voice)}.pt"
        return str(
            validate_path(
                filename, Path(self.voice_dir).expanduser(), redact_paths=True
            )
        )

    async def _download_voice_if_needed(self, voice: str):
        """Download voice pack if not present"""
        voice_path = self._voice_pack_path(voice)
        if not os.path.exists(voice_path):
            logger.info(f"Downloading voice pack: {voice}")
            try:
                if not REQUESTS_AVAILABLE:
                    raise ImportError("requests library required for voice download")
                url = f"https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice}.pt?download=true"
                await self._run_native_work(
                    _kokoro_stream_download,
                    url,
                    voice_path,
                    label=f"Kokoro voice '{voice}'",
                )
            except Exception as e:
                logger.error(f"Failed to download voice {voice}: {e}")
                raise ValueError(
                    f"Failed to download voice {voice}. Please download manually."
                )

    def _load_voice_pack(self, voice: str):
        """Load a voice pack for PyTorch"""
        voice_path = self._voice_pack_path(voice)
        if self._kokoro_pt_modules and "load_voice" in self._kokoro_pt_modules:
            load_voice = self._kokoro_pt_modules["load_voice"]
            return load_voice(voice_path, self.device)
        else:
            # Fallback to direct torch load
            if not os.path.exists(voice_path):
                raise FileNotFoundError(f"Voice pack not found: {voice_path}")
            return self.torch.load(voice_path, weights_only=True).to(self.device)

    async def _generate_pytorch_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio using PyTorch backend"""
        start_time = time.time()

        try:
            # Ensure model is loaded
            if not self.kokoro_model_pt:
                await self._download_model_if_needed()

            voice_config = self._parse_voice_config(request.voice)
            voice = voice_config["primary_voice"]
            voice_tensors = []
            for name, weight in voice_config["voices"]:
                await self._download_voice_if_needed(name)
                pack = await self._run_native_work(self._load_voice_pack, name)
                voice_tensors.append((pack, weight))
            if voice_config["is_mixed"]:
                voice_pack = self._kokoro_pt_modules["mix_voices"](voice_tensors)
            else:
                voice_pack = voice_tensors[0][0]

            # Detect language from voice or use provided language code
            if (
                hasattr(request, "extra_params")
                and request.extra_params
                and "language" in request.extra_params
            ):
                lang = request.extra_params["language"]
            else:
                # Map voice prefix to espeak language codes
                if voice and len(voice) > 0:
                    voice_prefix = voice[0].lower()
                    lang = KOKORO_VOICE_LANGUAGES.get(voice_prefix, "en-us")
                else:
                    lang = "en-us"

            # Split text into chunks
            text_chunks = self._split_text_for_pytorch(request.input)

            # Get generation function from cached modules
            if self._kokoro_pt_modules is None:
                # Load modules if not already loaded
                await self._run_native_work(self._load_pytorch_model)
            generate = self._kokoro_pt_modules["generate"]

            # Generate and stream audio for each chunk
            token_count = 0
            first_chunk_time = None
            total_audio_duration = 0.0
            sample_buffer = bytearray()
            estimated_total_tokens = (
                sum(len(chunk.split()) for chunk in text_chunks) * 2
            )

            # Report initial progress
            await self._report_progress(
                progress=0.0,
                processed=0,
                total=estimated_total_tokens,
                status=f"Starting PyTorch generation with {len(text_chunks)} chunks",
                total_chunks=len(text_chunks),
                metrics={"backend": "pytorch", "device": self.device},
            )

            for i, chunk in enumerate(text_chunks):
                # Report chunk start
                await self._report_progress(
                    progress=i
                    / len(text_chunks)
                    * 0.9,  # Reserve 10% for final processing
                    processed=token_count,
                    total=estimated_total_tokens,
                    status=f"Processing chunk {i + 1}/{len(text_chunks)}",
                    current_chunk=i + 1,
                    total_chunks=len(text_chunks),
                    metrics={"backend": "pytorch", "device": self.device},
                )

                # Generate audio
                audio_tensor, phonemes = await self._run_pytorch_generation(
                    generate,
                    self.kokoro_model_pt,
                    chunk,
                    voice_pack,
                    lang=lang,
                    speed=request.speed,
                    voice_dir=self.voice_dir,
                )

                # Convert to numpy
                if isinstance(audio_tensor, self.torch.Tensor):
                    audio_data = audio_tensor.cpu().numpy()
                else:
                    audio_data = audio_tensor

                # Track metrics
                token_count += len(chunk.split())  # Approximate
                chunk_duration = len(audio_data) / 24000  # Assuming 24kHz
                total_audio_duration += chunk_duration

                # Convert and yield based on format
                if request.response_format == "pcm":
                    # For PCM, convert to int16 and yield directly
                    int16_samples = np.int16(audio_data * 32767)
                    yield int16_samples.tobytes()
                else:
                    # File containers must describe the complete utterance.
                    _append_audio_samples(
                        sample_buffer, audio_data, KOKORO_MAX_ENCODED_AUDIO_SAMPLES
                    )

                # Track first chunk latency
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    logger.debug(
                        f"KokoroTTSBackend PyTorch: First chunk latency: {first_chunk_time:.3f}s"
                    )

                # Log progress
                logger.debug(
                    f"KokoroTTSBackend PyTorch: Processed chunk {i + 1}/{len(text_chunks)} "
                    f"({chunk_duration:.2f}s of audio)"
                )

            if request.response_format != "pcm":
                if not sample_buffer:
                    yield b""
                    return
                audio_array = np.frombuffer(sample_buffer, dtype=np.float32)
                try:
                    audio_bytes = await self.audio_service.convert_audio(
                        audio_array,
                        request.response_format,
                        source_format="pcm",
                        sample_rate=24000,
                    )
                except (RuntimeError, ValueError) as exc:
                    logger.error(
                        "KokoroTTSBackend: PyTorch audio conversion failed ({})",
                        type(exc).__name__,
                    )
                    yield b""
                    return
                del audio_array, sample_buffer
                yield audio_bytes

            # Update metrics
            if self.track_performance:
                self._update_performance_metrics(token_count, time.time() - start_time)

            # Log performance summary
            if token_count > 0:
                generation_time = time.time() - start_time
                speed_factor = (
                    total_audio_duration / generation_time if generation_time > 0 else 0
                )
                logger.info(
                    f"KokoroTTSBackend PyTorch: Generated {total_audio_duration:.2f}s of audio "
                    f"in {generation_time:.2f}s ({speed_factor:.1f}x realtime, "
                    f"first chunk: {first_chunk_time:.3f}s)"
                )

                # Report completion
                await self._report_progress(
                    progress=1.0,
                    processed=token_count,
                    total=estimated_total_tokens,
                    status=f"PyTorch generation complete: {total_audio_duration:.1f}s of audio",
                    total_chunks=len(text_chunks),
                    metrics={
                        "backend": "pytorch",
                        "device": self.device,
                        "generation_time": generation_time,
                        "speed_factor": speed_factor,
                        "audio_duration": total_audio_duration,
                        "first_chunk_latency": first_chunk_time,
                    },
                )

        except TTSOperationError:
            raise
        except Exception as e:
            logger.opt(exception=True).error(
                f"KokoroTTSBackend: PyTorch generation failed: {e}"
            )
            raise TTSOperationError(
                code="generation_failed",
                message="Kokoro PyTorch generation failed.",
                retryable=True,
                operation_id="kokoro_pytorch",
                recovery_action="retry",
            ) from e

    def _split_text_for_pytorch(self, text: str, max_tokens: int = 150) -> list[str]:
        """Bound work per call while preserving explicit language boundaries.

        The official pipeline owns phoneme tokenization. This outer word budget
        allows cancellation and encoded-audio limits between bounded requests.
        """
        chunks = []
        chunk_size = max(1, max_tokens)
        for paragraph in text.splitlines():
            words = paragraph.split()
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i : i + chunk_size]))
        return chunks

    async def generate_with_timestamps(
        self, text: str, voice: str = "af_bella", speed: float = 1.0
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        """
        Generate speech with word-level timestamps.

        Args:
            text: Text to synthesize
            voice: Voice to use
            speed: Speed factor

        Returns:
            Tuple of (audio_bytes, word_timestamps)
            where word_timestamps is a list of dicts with 'word', 'start', 'end' keys
        """
        try:
            if self.use_onnx and self.kokoro_instance:
                # ONNX implementation
                return await self._generate_onnx_with_timestamps(text, voice, speed)
            else:
                # PyTorch implementation
                return await self._generate_pytorch_with_timestamps(text, voice, speed)
        except Exception as e:
            logger.error(f"Failed to generate with timestamps: {e}")
            raise

    async def _generate_onnx_with_timestamps(
        self, text: str, voice: str, speed: float
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        """Generate audio with timestamps using ONNX backend"""

        # Map voice
        kokoro_voice = map_voice_to_kokoro(voice)
        # Map voice prefix to espeak language codes
        if kokoro_voice and len(kokoro_voice) > 0:
            voice_prefix = kokoro_voice[0].lower()
            lang = KOKORO_VOICE_LANGUAGES.get(voice_prefix, "en-us")
        else:
            lang = "en-us"

        # Split text into words for timing estimation
        words = text.split()
        word_timestamps = []

        # Generate audio
        sample_rate = 24000
        current_time = 0.0

        # Generate full audio first
        samples_list = []
        stream = self._create_onnx_stream(
            text, voice=kokoro_voice, speed=speed, lang=lang
        )
        async with aclosing(stream):
            async for samples, sr in stream:
                samples_list.append(samples)
                sample_rate = sr

        if not samples_list:
            return b"", []

        # Combine all samples
        full_audio = np.concatenate(samples_list)

        # Estimate word timings based on audio length and word count
        # This is approximate - for accurate timestamps we'd need phoneme alignment
        total_duration = len(full_audio) / sample_rate
        avg_word_duration = total_duration / len(words) if words else 0

        for i, word in enumerate(words):
            start_time = current_time
            # Adjust duration based on word length (rough approximation)
            word_factor = len(word) / (sum(len(w) for w in words) / len(words))
            word_duration = avg_word_duration * word_factor
            end_time = start_time + word_duration

            word_timestamps.append(
                {
                    "word": word,
                    "start": start_time,
                    "end": end_time,
                    "confidence": 0.7,  # Estimated timing confidence
                }
            )

            current_time = end_time

        # Convert audio to bytes
        int16_samples = np.int16(full_audio * 32767)
        audio_bytes = int16_samples.tobytes()

        # Wrap in WAV format
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)

        return buffer.getvalue(), word_timestamps

    async def _generate_pytorch_with_timestamps(
        self, text: str, voice: str, speed: float
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        """Generate audio with timestamps using PyTorch backend"""
        # Ensure model is loaded
        if not self.kokoro_model_pt:
            await self._download_model_if_needed()

        # Map voice and load voice pack
        kokoro_voice = map_voice_to_kokoro(voice)
        await self._download_voice_if_needed(kokoro_voice)
        voice_pack = await self._run_native_work(self._load_voice_pack, kokoro_voice)

        # Detect language
        if kokoro_voice and len(kokoro_voice) > 0:
            voice_prefix = kokoro_voice[0].lower()
            lang = KOKORO_VOICE_LANGUAGES.get(voice_prefix, "en-us")
        else:
            lang = "en-us"

        # Get generation function from cached modules
        if self._kokoro_pt_modules is None:
            # Load modules if not already loaded
            await self._run_native_work(self._load_pytorch_model)
        generate = self._kokoro_pt_modules["generate"]

        # Generate audio with phonemes
        audio_tensor, phonemes = await self._run_pytorch_generation(
            generate, self.kokoro_model_pt, text, voice_pack, lang=lang, speed=speed
        )

        # Convert to numpy
        if isinstance(audio_tensor, self.torch.Tensor):
            audio_data = audio_tensor.cpu().numpy()
        else:
            audio_data = audio_tensor

        # Parse phonemes to create word timestamps
        word_timestamps = self._phonemes_to_word_timestamps(
            text, phonemes, len(audio_data) / 24000
        )

        # Convert audio to WAV bytes
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)
            int16_samples = np.int16(audio_data * 32767)
            wav_file.writeframes(int16_samples.tobytes())

        return buffer.getvalue(), word_timestamps

    def _phonemes_to_word_timestamps(
        self, text: str, phonemes: Any, total_duration: float
    ) -> List[Dict[str, Any]]:
        """Convert phoneme information to word-level timestamps"""
        words = text.split()

        # If we don't have detailed phoneme timing, estimate
        if not phonemes or not hasattr(phonemes, "__iter__"):
            return self._estimate_word_timestamps(words, total_duration)

        # TODO: Implement actual phoneme-to-word alignment
        # For now, use estimation
        return self._estimate_word_timestamps(words, total_duration)

    def _estimate_word_timestamps(
        self, words: List[str], total_duration: float
    ) -> List[Dict[str, Any]]:
        """Estimate word timestamps based on word length"""
        if not words:
            return []

        word_timestamps = []
        total_chars = sum(len(w) for w in words)
        current_time = 0.0

        for word in words:
            # Estimate duration based on character count
            word_duration = (
                (len(word) / total_chars) * total_duration if total_chars > 0 else 0
            )

            word_timestamps.append(
                {
                    "word": word,
                    "start": current_time,
                    "end": current_time + word_duration,
                    "confidence": 0.5,  # Low confidence for estimation
                }
            )

            current_time += word_duration

        return word_timestamps

    def _load_saved_blends(self) -> Dict[str, Dict[str, Any]]:
        """Load saved voice blends from disk"""
        blends = {}
        blend_file = self.voice_blends_dir / "voice_blends.json"

        if blend_file.exists():
            try:
                with open(blend_file, "r") as f:
                    blends = json.load(f)
                logger.info(f"Loaded {len(blends)} saved voice blends")
            except Exception as e:
                logger.error(f"Failed to load voice blends: {e}")

        return blends

    def _save_blends(self) -> bool:
        """Atomically save voice blends to disk."""
        blend_file = self.voice_blends_dir / "voice_blends.json"
        try:
            write_private_json(
                blend_file,
                self.saved_blends,
                application_owned_directory=(
                    self.voice_blends_dir
                    if self._voice_blends_directory_is_application_owned
                    else None
                ),
            )
            logger.info(f"Saved {len(self.saved_blends)} voice blends")
            return True
        except Exception as e:
            logger.error(f"Failed to save voice blends: {e}")
            return False

    def _create_default_blends(self):
        """Create default voice blend presets"""
        default_blends = [
            # Professional blends
            (
                "professional_female",
                [("af_bella", 0.6), ("af_sarah", 0.4)],
                "Professional female voice blend",
            ),
            (
                "professional_male",
                [("am_adam", 0.7), ("am_michael", 0.3)],
                "Professional male voice blend",
            ),
            # Character blends
            (
                "warm_storyteller",
                [("af_nicole", 0.5), ("bf_emma", 0.5)],
                "Warm storytelling voice",
            ),
            (
                "dynamic_narrator",
                [("am_michael", 0.4), ("bm_george", 0.3), ("am_adam", 0.3)],
                "Dynamic narrator with varied tones",
            ),
            # Language-specific blends
            (
                "english_blend",
                [("af_bella", 0.3), ("af_sarah", 0.3), ("bf_emma", 0.4)],
                "Balanced English female voices",
            ),
            (
                "male_chorus",
                [
                    ("am_adam", 0.25),
                    ("am_michael", 0.25),
                    ("bm_george", 0.25),
                    ("bm_lewis", 0.25),
                ],
                "All male voices blended equally",
            ),
            # Creative blends
            (
                "soft_whisper",
                [("af_sky", 0.7), ("bf_isabella", 0.3)],
                "Soft, gentle voice blend",
            ),
            (
                "energetic",
                [("bf_emma", 0.6), ("af_nicole", 0.4)],
                "Energetic and upbeat voice",
            ),
        ]

        for name, voices, description in default_blends:
            self.save_voice_blend(name, voices, description, {"is_default": True})

        logger.info(f"Created {len(default_blends)} default voice blends")

    def save_voice_blend(
        self,
        name: str,
        voices: List[Tuple[str, float]],
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save a custom voice blend for later use.

        Args:
            name: Unique name for the blend
            voices: List of (voice_name, weight) tuples
            description: Optional description
            metadata: Optional additional metadata

        Returns:
            Success status
        """
        updated = False
        had_existing_blend = name in self.saved_blends
        previous_blend = self.saved_blends.get(name)
        try:
            # Validate and normalize weights
            total_weight = sum(w for _, w in voices)
            if total_weight <= 0:
                raise ValueError("Total weight must be positive")

            normalized_voices = [(v, w / total_weight) for v, w in voices]

            # Create blend entry
            blend_data = {
                "voices": normalized_voices,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            # Save to memory and disk
            self.saved_blends[name] = blend_data
            updated = True
            if not self._save_blends():
                if had_existing_blend:
                    self.saved_blends[name] = previous_blend
                else:
                    self.saved_blends.pop(name, None)
                return False

            logger.info(f"Saved voice blend '{name}' with {len(voices)} voices")
            return True

        except Exception as e:
            if updated:
                if had_existing_blend:
                    self.saved_blends[name] = previous_blend
                else:
                    self.saved_blends.pop(name, None)
            logger.error(f"Failed to save voice blend: {e}")
            return False

    def get_voice_blend(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a saved voice blend by name"""
        return self.saved_blends.get(name)

    def list_voice_blends(self) -> List[Dict[str, Any]]:
        """List all saved voice blends"""
        blends = []
        for name, data in self.saved_blends.items():
            blend_info = {
                "name": name,
                "voices": data["voices"],
                "description": data.get("description", ""),
                "created_at": data.get("created_at", ""),
                "voice_count": len(data["voices"]),
            }
            blends.append(blend_info)
        return blends

    def delete_voice_blend(self, name: str) -> bool:
        """Delete a saved voice blend"""
        if name not in self.saved_blends:
            return False

        removed_blend = self.saved_blends.pop(name)
        try:
            if not self._save_blends():
                self.saved_blends[name] = removed_blend
                return False
            logger.info(f"Deleted voice blend '{name}'")
            return True
        except Exception as e:
            self.saved_blends[name] = removed_blend
            logger.error(f"Failed to delete voice blend: {e}")
            return False

    def create_blend_from_preset(self, preset_name: str) -> Optional[str]:
        """
        Create a voice blend string from a saved preset.

        Args:
            preset_name: Name of the saved blend

        Returns:
            Voice string in format "voice1:weight1,voice2:weight2" or None
        """
        blend = self.get_voice_blend(preset_name)
        if not blend:
            return None

        voice_parts = []
        for voice, weight in blend["voices"]:
            voice_parts.append(f"{voice}:{weight:.2f}")

        return ",".join(voice_parts)

    async def generate_from_phonemes(
        self, phonemes: str, voice: str = "af_bella", speed: float = 1.0
    ) -> bytes:
        """
        Generate speech from phoneme input.

        Args:
            phonemes: Phoneme string (e.g., "HH AH0 L OW1")
            voice: Voice to use
            speed: Speed factor

        Returns:
            Audio bytes in PCM format
        """
        if not self.use_onnx:
            raise NotImplementedError(
                "Phoneme generation only supported with ONNX backend"
            )

        try:
            kokoro_voice = map_voice_to_kokoro(voice)

            # Check if kokoro_instance has phoneme support
            if hasattr(self.kokoro_instance, "generate_from_phonemes"):
                samples = await self._run_native_work(
                    self.kokoro_instance.generate_from_phonemes,
                    phonemes,
                    voice=kokoro_voice,
                    speed=speed,
                )

                # Convert to PCM bytes
                int16_samples = np.int16(samples * 32767)
                return int16_samples.tobytes()
            else:
                logger.warning("Kokoro instance doesn't support phoneme generation")
                raise NotImplementedError(
                    "This version of kokoro_onnx doesn't support phoneme generation"
                )

        except (TTSOperationError, NotImplementedError):
            raise
        except Exception as exc:
            raise TTSOperationError(
                code="generation_failed",
                message="Kokoro ONNX generation failed.",
                retryable=True,
                operation_id="kokoro_onnx",
                recovery_action="retry",
            ) from exc

    async def close(self):
        """Clean up resources"""
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_resources())
        await join_retained_task(self._close_task)

    async def _close_resources(self) -> None:
        """Join native work before releasing its models."""
        for stop in tuple(self._native_tasks.values()):
            if stop is not None:
                stop()
        for stop in tuple(self._onnx_tasks.values()):
            stop()
        pending = tuple(self._native_tasks) + tuple(self._onnx_tasks)
        if pending:
            await asyncio.wait(pending)

        self._onnx_phonemizers.clear()
        await super().close()
        # Clean up model instances if needed
        self.kokoro_instance = None
        self.kokoro_model_pt = None

        # Log final performance stats if tracking
        if self.track_performance and self._performance_metrics["generation_count"] > 0:
            stats = self.get_performance_stats()
            logger.info(
                f"KokoroTTSBackend: Final performance stats - "
                f"Avg: {stats['average_tokens_per_second']:.1f} tokens/s, "
                f"Total: {stats['total_generations']} generations in {stats['total_time']:.1f}s"
            )


# Voice mapping for Kokoro
KOKORO_VOICE_MAP = {
    # OpenAI-style names to Kokoro voices
    "alloy": "af_bella",
    "echo": "af_sarah",
    "fable": "am_adam",
    "onyx": "am_michael",
    "nova": "bf_emma",
    "shimmer": "bf_isabella",
    # Direct Kokoro voice names (already supported)
    # Female voices
    "af_bella": "af_bella",
    "af_nicole": "af_nicole",
    "af_sarah": "af_sarah",
    "af_sky": "af_sky",
    "bf_emma": "bf_emma",
    "bf_isabella": "bf_isabella",
    # Male voices
    "am_adam": "am_adam",
    "am_michael": "am_michael",
    "bm_george": "bm_george",
    "bm_lewis": "bm_lewis",
    # Aliases for convenience
    "bella": "af_bella",
    "nicole": "af_nicole",
    "sarah": "af_sarah",
    "sky": "af_sky",
    "emma": "bf_emma",
    "isabella": "bf_isabella",
    "adam": "am_adam",
    "michael": "am_michael",
    "george": "bm_george",
    "lewis": "bm_lewis",
}


def map_voice_to_kokoro(voice: str) -> str:
    """Map OpenAI or other voice names to Kokoro voice names"""
    return KOKORO_VOICE_MAP.get(voice.lower(), voice)


#
# End of kokoro.py
#######################################################################################################################
