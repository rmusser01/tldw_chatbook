from tldw_chatbook.TTS import loose_voice_lifetime as voice_files
# chatterbox.py
# Description: Chatterbox TTS backend implementation with streaming support
#
# Imports
import os
import sys
import subprocess
from typing import AsyncGenerator, Optional, Dict, Any, List
from pathlib import Path
import asyncio
import re
import json
import base64
from datetime import datetime
from uuid import uuid4
from difflib import SequenceMatcher
from loguru import logger
import tempfile  # Still needed for audio file handling
import contextlib

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS._async_lifecycle import join_retained_task
from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.audio_limits import check_buffered_audio_size
from tldw_chatbook.TTS.TTS_Backends import TTSBackendBase
from tldw_chatbook.TTS.audio_service import get_audio_service
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Utils.optional_deps import check_dependency

#######################################################################################################################
#
# Utility Functions
#


@contextlib.contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr at the file descriptor level.

    This is necessary to prevent libraries from writing directly to the terminal,
    which can corrupt TUI applications.
    """
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


#######################################################################################################################
#
# Chatterbox TTS Backend Implementation
#


class ChatterboxTTSBackend(TTSBackendBase):
    """
    Enhanced Chatterbox Text-to-Speech backend by Resemble AI.

    Features:
    - Zero-shot voice cloning with 7-20 seconds of reference audio
    - Emotion exaggeration control
    - Ultra-low latency streaming (< 200ms)
    - Watermarked audio outputs
    - MIT licensed open-source model

    Extended Features:
    - Advanced text preprocessing (dot-letter correction, reference removal)
    - Multi-candidate generation with Whisper validation
    - Text chunking for long content
    - Audio normalization and post-processing
    - Voice management with metadata
    - Fallback strategies for robust generation
    """

    @voice_files.call
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Core Configuration
        self.device = self.config.get(
            "CHATTERBOX_DEVICE", get_cli_setting("app_tts", "CHATTERBOX_DEVICE", "cuda")
        )
        self.exaggeration = float(
            self.config.get(
                "CHATTERBOX_EXAGGERATION",
                get_cli_setting("app_tts", "CHATTERBOX_EXAGGERATION", 0.5),
            )
        )
        self.cfg_weight = float(
            self.config.get(
                "CHATTERBOX_CFG_WEIGHT",
                get_cli_setting("app_tts", "CHATTERBOX_CFG_WEIGHT", 0.5),
            )
        )
        self.chunk_size = int(
            self.config.get(
                "CHATTERBOX_CHUNK_SIZE",
                get_cli_setting("app_tts", "CHATTERBOX_CHUNK_SIZE", 1024),
            )
        )

        # Extended Configuration
        self.temperature = float(
            self.config.get(
                "CHATTERBOX_TEMPERATURE",
                get_cli_setting("app_tts", "CHATTERBOX_TEMPERATURE", 0.5),
            )
        )
        self.random_seed = self.config.get(
            "CHATTERBOX_RANDOM_SEED",
            get_cli_setting("app_tts", "CHATTERBOX_RANDOM_SEED", None),
        )
        self.num_candidates = int(
            self.config.get(
                "CHATTERBOX_NUM_CANDIDATES",
                get_cli_setting("app_tts", "CHATTERBOX_NUM_CANDIDATES", 1),
            )
        )
        self.validate_with_whisper = self.config.get(
            "CHATTERBOX_VALIDATE_WHISPER",
            get_cli_setting("app_tts", "CHATTERBOX_VALIDATE_WHISPER", False),
        )
        self.preprocess_text_enabled = self.config.get(
            "CHATTERBOX_PREPROCESS_TEXT",
            get_cli_setting("app_tts", "CHATTERBOX_PREPROCESS_TEXT", True),
        )
        self.normalize_audio_enabled = self.config.get(
            "CHATTERBOX_NORMALIZE_AUDIO",
            get_cli_setting("app_tts", "CHATTERBOX_NORMALIZE_AUDIO", True),
        )
        self.target_db = float(
            self.config.get(
                "CHATTERBOX_TARGET_DB",
                get_cli_setting("app_tts", "CHATTERBOX_TARGET_DB", -20.0),
            )
        )
        self.max_chunk_size = int(
            self.config.get(
                "CHATTERBOX_MAX_CHUNK_SIZE",
                get_cli_setting("app_tts", "CHATTERBOX_MAX_CHUNK_SIZE", 500),
            )
        )

        # Voice settings
        self.voice_dir = Path(
            self.config.get(
                "CHATTERBOX_VOICE_DIR",
                get_cli_setting(
                    "app_tts",
                    "CHATTERBOX_VOICE_DIR",
                    "~/.config/tldw_cli/chatterbox_voices",
                ),
            )
        ).expanduser()
        voice_files.mkdir(self, self.voice_dir, parents=True, exist_ok=True)

        # Streaming configuration
        self.streaming_enabled = self.config.get(
            "CHATTERBOX_STREAMING",
            get_cli_setting("app_tts", "CHATTERBOX_STREAMING", True),
        )
        self.stream_chunk_size = int(
            self.config.get(
                "CHATTERBOX_STREAM_CHUNK_SIZE",
                get_cli_setting("app_tts", "CHATTERBOX_STREAM_CHUNK_SIZE", 4096),
            )
        )
        self.enable_crossfade = self.config.get(
            "CHATTERBOX_ENABLE_CROSSFADE",
            get_cli_setting("app_tts", "CHATTERBOX_ENABLE_CROSSFADE", True),
        )
        self.crossfade_duration_ms = int(
            self.config.get(
                "CHATTERBOX_CROSSFADE_MS",
                get_cli_setting("app_tts", "CHATTERBOX_CROSSFADE_MS", 50),
            )
        )

        # Model instances
        self.model = None
        self.audio_service = get_audio_service()
        self.transcription_service = (
            None  # Will be initialized if validation is enabled
        )

        # Check dependencies
        self.deps_available = check_dependency("chatterbox", "chatterbox")

        # Process management for isolated execution
        self.process: Optional[subprocess.Popen] = None
        self._process_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()
        self._initialized = False
        self._initializing = False
        self._closing = False
        self._initialization_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None

    async def initialize(self):
        """Initialize the Chatterbox backend using isolated process"""
        if self._closing:
            return
        if not self.deps_available:
            logger.warning(
                "ChatterboxTTSBackend: Dependencies not available. Please install with: pip install chatterbox-tts torchaudio"
            )
            return

        if self._initialized or (
            self._initialization_task is not None
            and not self._initialization_task.done()
        ):
            return

        self._initializing = True
        # Run initialization in background to avoid blocking UI
        self._initialization_task = asyncio.create_task(
            self._initialize_isolated_process()
        )

    async def _initialize_isolated_process(self):
        """Initialize Chatterbox in an isolated subprocess"""
        async with self._process_lock:
            if self._initialized or self._closing:
                self._initializing = False
                return

            try:
                # Find the process wrapper script
                wrapper_path = Path(__file__).parent / "chatterbox_process.py"
                if not wrapper_path.exists():
                    logger.error(
                        f"Chatterbox process wrapper not found at {wrapper_path}"
                    )
                    # Fall back to the old method in background
                    await join_retained_task(
                        asyncio.create_task(asyncio.to_thread(self._initialize_sync))
                    )
                    return

                logger.info(
                    f"Starting Chatterbox in isolated process on {self.device}..."
                )

                # Retain process acquisition so cancellation cannot lose a late child.
                await join_retained_task(
                    asyncio.create_task(self._spawn_isolated_process(wrapper_path))
                )
                if self._closing:
                    return

                # Send initialization command
                await self._send_command(
                    {"command": "initialize", "device": self.device}
                )

                await self._wait_for_initialization()

            except Exception as e:
                logger.error(f"Failed to initialize Chatterbox process: {e}")
                await self._discard_process()
                if not self._closing:
                    logger.info("Falling back to in-process initialization")
                    await join_retained_task(
                        asyncio.create_task(asyncio.to_thread(self._initialize_sync))
                    )
            finally:
                self._initializing = False

    async def _spawn_isolated_process(self, wrapper_path: Path) -> None:
        """Publish the acquired child inside the retained acquisition task."""
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(wrapper_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=10 * 1024 * 1024,  # Allow base64-encoded audio responses.
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    async def _send_command(self, command: Dict[str, Any]):
        """Send command to subprocess"""
        if not self.process or self.process.returncode is not None:
            raise Exception("Process not running")

        json_data = json.dumps(command) + "\n"
        self.process.stdin.write(json_data.encode())
        await self.process.stdin.drain()

    async def _wait_for_initialization(self):
        """Wait for initialization response in background"""
        try:
            # Wait for multiple responses during initialization
            while True:
                try:
                    response = await self._read_response(
                        timeout=60
                    )  # 60 second timeout for model loading

                    if response.get("type") == "status":
                        logger.info(
                            f"Chatterbox init status: {response.get('message')}"
                        )
                    elif response.get("type") == "warning":
                        logger.warning(
                            f"Chatterbox init warning: {response.get('message')}"
                        )
                    elif response.get("type") == "success":
                        if self._closing:
                            return
                        logger.info("Chatterbox process initialized successfully")
                        self._initialized = True
                        self._initializing = False
                        self.model = (
                            "process"  # Placeholder to indicate model is loaded
                        )
                        break
                    elif response.get("type") == "error":
                        error_msg = response.get("message", "Unknown error")
                        if "traceback" in response:
                            logger.error(
                                f"Chatterbox init traceback:\n{response['traceback']}"
                            )
                        raise Exception(f"Initialization failed: {error_msg}")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    # Check if process is still alive
                    if self.process and self.process.returncode is not None:
                        raise Exception(f"Process died during initialization: {e}")
                    raise
        except Exception as e:
            logger.error(f"Background initialization failed: {e}")
            self._initialized = False
            self._initializing = False
            # Clean up the process
            await self._discard_process()

    async def _read_response(self, timeout: float = 10) -> Dict[str, Any]:
        """Read response from subprocess via stdout pipe"""
        if not self.process or self.process.returncode is not None:
            raise Exception("Process not running")

        try:
            # Read line from stdout with timeout
            line = await asyncio.wait_for(
                self.process.stdout.readline(), timeout=timeout
            )

            if not line:
                raise Exception("Process closed stdout")

            # Decode and parse JSON
            response = json.loads(line.decode().strip())
            return response

        except TimeoutError:
            raise TimeoutError(f"No response from subprocess within {timeout} seconds")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {e}")

    async def _read_chunked_audio(self, timeout: float = 60) -> bytes:
        """Read chunked audio data from subprocess"""
        chunks = {}
        total_chunks = None
        retained_bytes = 0

        while True:
            response = await self._read_response(timeout)
            msg_type = response.get("type")

            if msg_type == "audio":
                # Single message with complete audio
                check_buffered_audio_size(len(response["data"]))
                return base64.b64decode(response["data"], validate=True)

            elif msg_type == "audio_chunk":
                # Part of a chunked transfer
                chunk_id = response.get("chunk_id")
                chunk_data = response.get("data")

                if chunk_id is not None and chunk_data:
                    retained_bytes += len(chunk_data) - len(chunks.get(chunk_id, ""))
                    check_buffered_audio_size(retained_bytes)
                    chunks[chunk_id] = chunk_data

                    # Track total chunks from first message
                    if total_chunks is None:
                        total_chunks = response.get("total_chunks")

                    logger.debug(f"Received audio chunk {chunk_id + 1}/{total_chunks}")

            elif msg_type == "audio_complete":
                # All chunks received
                expected_total = response.get("total_chunks", total_chunks)

                # Verify we have all chunks
                if len(chunks) != expected_total:
                    raise Exception(
                        f"Missing audio chunks: got {len(chunks)}, expected {expected_total}"
                    )

                # Reassemble chunks in order
                for i in range(expected_total):
                    if i not in chunks:
                        raise Exception(f"Missing chunk {i}")
                audio_data = "".join(chunks[i] for i in range(expected_total))

                # Decode base64 data
                return base64.b64decode(audio_data, validate=True)

            elif msg_type == "error":
                error_msg = response.get("message", "Unknown error")
                raise Exception(f"Audio generation error: {error_msg}")

            else:
                # Unexpected message type during chunked transfer
                logger.warning(
                    f"Unexpected message type during chunked transfer: {msg_type}"
                )

    def _initialize_sync(self):
        """Synchronous initialization to run in a thread"""
        try:
            # Import Chatterbox
            from chatterbox.tts import ChatterboxTTS
            import torch

            # Check CUDA availability
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"

            # Import protect_file_descriptors if available
            try:
                from tldw_chatbook.Utils.fd_protection import (
                    protect_file_descriptors,
                )

                # Load model with file descriptor protection AND output capture
                logger.info(f"Loading Chatterbox model on {self.device}...")

                # Load model with both protections
                with suppress_output():
                    with protect_file_descriptors():
                        self.model = ChatterboxTTS.from_pretrained(device=self.device)

                logger.info("Chatterbox model loaded successfully")
            except ImportError:
                # Fallback without protection if not available
                logger.info(
                    f"Loading Chatterbox model on {self.device} (without FD protection)..."
                )

                # Still capture output even without protect_file_descriptors
                with suppress_output():
                    self.model = ChatterboxTTS.from_pretrained(device=self.device)

                logger.info("Chatterbox model loaded successfully")

            self._initialized = True  # Mark as initialized

            # Initialize transcription service if validation is enabled
            if self.validate_with_whisper:
                try:
                    from tldw_chatbook.Local_Ingestion.transcription_service import (
                        TranscriptionService,
                    )

                    self.transcription_service = TranscriptionService()
                    logger.info("Transcription service initialized for validation")
                except Exception as e:
                    logger.warning(f"Failed to initialize transcription service: {e}")
                    logger.warning("Whisper validation will be disabled")
                    self.validate_with_whisper = False

        except ImportError as e:
            logger.error(f"Failed to import Chatterbox: {e}")
            logger.info("Please install Chatterbox with: pip install chatterbox-tts")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize Chatterbox: {e}")
            self.model = None

    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing for better pronunciation.

        Features:
        - Normalize whitespace
        - Convert "J.R.R." to "J R R"
        - Remove inline reference numbers [1], [2], etc.
        - Optional lowercase conversion
        """
        if not self.preprocess_text_enabled:
            return text

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Convert initial sequences like "J.R.R. Tolkien" to "J R R Tolkien"
        # for better pronunciation, including the trailing initial before a name.
        text = re.sub(r"\b([A-Z])\.(?=(?:[A-Z]\.)|(?:\s+[A-Z]))", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)

        # Remove inline reference numbers like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)

        # Remove URLs if present
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Convert multiple punctuation to single
        text = re.sub(r"([.!?])\1+", r"\1", text)

        # Add space after punctuation if missing
        text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

        logger.debug(
            f"Text preprocessing complete. Original length: {len(text)}, Processed length: {len(text)}"
        )

        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks at sentence boundaries.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                word_chunk = ""
                for word in sentence.split():
                    separator = " " if word_chunk else ""
                    if (
                        len(word_chunk) + len(separator) + len(word)
                        > self.max_chunk_size
                    ):
                        if word_chunk:
                            chunks.append(word_chunk.strip())
                        word_chunk = word
                    else:
                        word_chunk = f"{word_chunk}{separator}{word}"
                if word_chunk:
                    chunks.append(word_chunk.strip())
                continue

            # If adding this sentence would exceed max chunk size
            if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " "
                current_chunk += sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"Text chunked into {len(chunks)} chunks")
        return chunks

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using SequenceMatcher.

        Returns:
            Similarity score between 0 and 1
        """
        # Normalize texts for comparison
        text1_normalized = text1.lower().strip()
        text2_normalized = text2.lower().strip()

        # Use SequenceMatcher to calculate similarity
        matcher = SequenceMatcher(None, text1_normalized, text2_normalized)
        return matcher.ratio()

    def normalize_audio(self, audio_tensor, target_db: Optional[float] = None):
        """
        Normalize audio to target dB level.

        Args:
            audio_tensor: PyTorch tensor of audio
            target_db: Target dB level (uses self.target_db if not specified)

        Returns:
            Normalized audio tensor
        """
        if not self.normalize_audio_enabled:
            return audio_tensor

        try:
            import torch

            if target_db is None:
                target_db = self.target_db

            # Calculate current RMS
            rms = torch.sqrt(torch.mean(audio_tensor**2))
            current_db = 20 * torch.log10(rms + 1e-10)

            # Calculate gain needed
            gain_db = target_db - current_db
            gain = 10 ** (gain_db / 20)

            # Apply gain
            normalized = audio_tensor * gain

            # Prevent clipping
            normalized = torch.clamp(normalized, -1.0, 1.0)

            logger.debug(f"Audio normalized from {current_db:.1f} dB to {target_db} dB")
            return normalized

        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}")
            return audio_tensor

    def apply_fade_in_out(self, audio_tensor, fade_duration_ms: int = 50):
        """
        Apply fade in/out to audio tensor.

        Args:
            audio_tensor: PyTorch tensor of audio
            fade_duration_ms: Duration of fade in milliseconds

        Returns:
            Audio tensor with fades applied
        """
        try:
            import torch

            # Assuming 24kHz sample rate (typical for Chatterbox)
            sample_rate = getattr(self.model, "sr", 24000)
            fade_samples = int(sample_rate * fade_duration_ms / 1000)

            if fade_samples >= len(audio_tensor):
                return audio_tensor

            # Create fade curves
            fade_in = torch.linspace(0, 1, fade_samples)
            fade_out = torch.linspace(1, 0, fade_samples)

            # Apply fades
            audio_tensor[:fade_samples] *= fade_in
            audio_tensor[-fade_samples:] *= fade_out

            return audio_tensor

        except Exception as e:
            logger.warning(f"Fade application failed: {e}")
            return audio_tensor

    def crossfade_audio_chunks(
        self, chunk1_tensor, chunk2_tensor, crossfade_duration_ms: int = 50
    ):
        """
        Apply crossfade between two audio chunks for smooth transitions.

        Args:
            chunk1_tensor: First audio chunk (PyTorch tensor)
            chunk2_tensor: Second audio chunk (PyTorch tensor)
            crossfade_duration_ms: Duration of crossfade in milliseconds

        Returns:
            Combined audio tensor with crossfade applied
        """
        try:
            import torch

            # Get sample rate
            sample_rate = getattr(self.model, "sr", 24000)
            crossfade_samples = int(sample_rate * crossfade_duration_ms / 1000)

            # Ensure we have enough samples for crossfade
            if crossfade_samples >= len(chunk1_tensor) or crossfade_samples >= len(
                chunk2_tensor
            ):
                logger.warning("Chunks too short for crossfade, concatenating directly")
                return torch.cat([chunk1_tensor, chunk2_tensor])

            # Create fade curves
            fade_out = torch.linspace(1, 0, crossfade_samples)
            fade_in = torch.linspace(0, 1, crossfade_samples)

            # Apply fades to overlap region
            chunk1_fade = chunk1_tensor.clone()
            chunk1_fade[-crossfade_samples:] *= fade_out

            chunk2_fade = chunk2_tensor.clone()
            chunk2_fade[:crossfade_samples] *= fade_in

            # Create result tensor
            total_length = len(chunk1_tensor) + len(chunk2_tensor) - crossfade_samples
            result = torch.zeros(total_length)

            # Copy non-overlapping parts
            result[: len(chunk1_tensor) - crossfade_samples] = chunk1_tensor[
                :-crossfade_samples
            ]
            result[len(chunk1_tensor) :] = chunk2_tensor[crossfade_samples:]

            # Add crossfaded overlap
            result[len(chunk1_tensor) - crossfade_samples : len(chunk1_tensor)] = (
                chunk1_fade[-crossfade_samples:] + chunk2_fade[:crossfade_samples]
            )

            return result

        except Exception as e:
            logger.error(f"Crossfade failed: {e}")
            # Fallback to simple concatenation
            import torch

            return torch.cat([chunk1_tensor, chunk2_tensor])

    async def _transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio using Whisper for validation.

        Args:
            audio_bytes: Audio data in bytes

        Returns:
            Transcribed text
        """
        if not self.transcription_service:
            return ""

        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            # Transcribe using the service
            result = await self.transcription_service.transcribe_audio(
                audio_path=tmp_path,
                provider="faster-whisper",
                model="base",
                language=None,  # Auto-detect
            )

            # Clean up
            os.unlink(tmp_path)

            # Extract text from segments
            if isinstance(result, dict) and "segments" in result:
                text = " ".join(
                    segment.get("text", "") for segment in result["segments"]
                )
                return text.strip()
            elif isinstance(result, str):
                return result.strip()
            else:
                return ""

        except Exception as e:
            logger.warning(f"Audio transcription failed: {e}")
            return ""

    async def _generate_single_isolated(
        self,
        text: str,
        reference_audio_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
        temperature: Optional[float] = None,
    ) -> bytes:
        """Generate audio using the isolated subprocess"""
        try:
            # Apply temperature variation if specified
            if temperature is not None:
                exaggeration = exaggeration * (1 + temperature * 0.1)
                cfg_weight = cfg_weight * (1 - temperature * 0.05)

            # Prepare command
            command = {
                "command": "generate",
                "text": text,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            }

            if reference_audio_path:
                command["audio_prompt_path"] = reference_audio_path

            # Send generation command
            await self._send_command(command)

            # Read response - now handles both single and chunked messages
            audio_bytes = await self._read_chunked_audio(timeout=60)
            return audio_bytes

        except BaseException:
            cleanup = asyncio.create_task(self._discard_process())
            await join_retained_task(cleanup)
            raise

    async def _discard_process(self) -> None:
        """Reap the current worker before permitting another IPC exchange."""
        process = self.process
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except TimeoutError:
            process.kill()
            await process.wait()
        if self.process is process:
            self.process = None
            self.model = None
            self._initialized = False
            self._initializing = False

    async def _generate_single(
        self,
        text: str,
        reference_audio_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
        temperature: Optional[float] = None,
    ) -> bytes:
        """
        Generate a single audio candidate.

        Returns:
            Audio bytes in WAV format
        """
        # Check if we're using isolated process
        if self._initialized and self.process and self.process.returncode is None:
            return await self._generate_single_isolated(
                text, reference_audio_path, exaggeration, cfg_weight, temperature
            )

        import torch

        # Set random seed if specified
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        # Apply temperature variation if specified
        if temperature is not None:
            # Slightly vary parameters based on temperature
            exaggeration = exaggeration * (1 + temperature * 0.1)
            cfg_weight = cfg_weight * (1 - temperature * 0.05)

        # Generate audio with proper isolation to avoid file descriptor issues
        def generate_with_isolation():
            """Generate audio with complete isolation"""
            # Import here to avoid issues
            try:
                from tldw_chatbook.Utils.fd_protection import (
                    protect_file_descriptors,
                )

                has_protect_fd = True
            except ImportError:
                has_protect_fd = False

            # Generate with file descriptor protection if available
            with suppress_output():
                if has_protect_fd:
                    with protect_file_descriptors():
                        if reference_audio_path:
                            return self.model.generate(
                                text,
                                audio_prompt_path=reference_audio_path,
                                exaggeration=exaggeration,
                                cfg_weight=cfg_weight,
                            )
                        else:
                            return self.model.generate(
                                text, exaggeration=exaggeration, cfg_weight=cfg_weight
                            )
                else:
                    # Generate without FD protection
                    if reference_audio_path:
                        return self.model.generate(
                            text,
                            audio_prompt_path=reference_audio_path,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                        )
                    else:
                        return self.model.generate(
                            text, exaggeration=exaggeration, cfg_weight=cfg_weight
                        )

        # Run in thread with isolation
        generation = asyncio.create_task(asyncio.to_thread(generate_with_isolation))
        await join_retained_task(generation)
        wav = generation.result()

        # Apply post-processing
        if self.normalize_audio_enabled:
            wav = self.normalize_audio(wav)

        wav = self.apply_fade_in_out(wav)

        # Convert to WAV bytes
        wav_bytes = self._tensor_to_wav_bytes(wav, self.model.sr)

        return wav_bytes

    async def generate_with_validation(
        self,
        text: str,
        reference_audio_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
    ) -> bytes:
        """
        Generate multiple candidates and select the best one using Whisper validation.

        Returns:
            Best audio candidate in WAV format
        """
        candidates = []

        for i in range(self.num_candidates):
            try:
                logger.info(f"Generating candidate {i + 1}/{self.num_candidates}")

                # Generate with slight variation for each candidate
                temperature = i * 0.2 if self.num_candidates > 1 else 0
                audio_bytes = await self._generate_single(
                    text, reference_audio_path, exaggeration, cfg_weight, temperature
                )

                if self.validate_with_whisper and self.transcription_service:
                    # Transcribe and calculate similarity
                    transcript = await self._transcribe_audio(audio_bytes)
                    similarity = self._calculate_similarity(text, transcript)
                    logger.info(f"Candidate {i + 1} similarity score: {similarity:.2f}")
                    candidates.append((audio_bytes, similarity, transcript))
                else:
                    # No validation, just add the candidate
                    candidates.append((audio_bytes, 1.0, ""))

            except TTSOperationError:
                raise
            except Exception as e:
                logger.warning(f"Candidate {i + 1} generation failed: {e}")

        if not candidates:
            raise ValueError("Failed to generate any valid candidates")

        # Select best candidate
        best_audio, best_score, best_transcript = max(candidates, key=lambda x: x[1])

        if self.validate_with_whisper:
            logger.info(f"Selected best candidate with score {best_score:.2f}")
            if best_score < 0.7:
                logger.warning(
                    f"Low similarity score ({best_score:.2f}). Transcript: '{best_transcript}'"
                )

        return best_audio

    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """Deliver one utterance while retaining exclusive inference ownership.

        Args:
            request: Text, voice, speed, output format, and streaming preference.
                Setting ``stream=False`` selects batch inference.

        Yields:
            One complete encoded file for container formats, or signed 16-bit
            mono PCM chunks at 24 kHz when raw PCM streaming is requested.
        """
        async with (
            self._generation_lock,
            contextlib.aclosing(self._generate_speech_stream(request)) as stream,
        ):
            if self._closing:
                raise RuntimeError("Chatterbox backend is closing")
            async for chunk in stream:
                yield chunk

    async def _generate_speech_stream(
        self, request: OpenAISpeechRequest, *, allow_fallback: bool = True
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech using Chatterbox and stream the response.

        Args:
            request: Speech request parameters

        Yields:
            Audio bytes in the requested format
        """
        if not self._initialized and self.model is not None and self.model != "process":
            self._initialized = True

        # If not initialized, start initialization and wait
        if not self._initialized:
            logger.info("Chatterbox not initialized, starting initialization...")
            await self.initialize()

            # Now wait for initialization to complete
            start_time = asyncio.get_event_loop().time()
            while (
                not self._closing
                and (self._initializing or not self._initialized)
                and (asyncio.get_event_loop().time() - start_time) < 60
            ):
                await asyncio.sleep(0.1)
                # Check if process died
                if (
                    hasattr(self, "process")
                    and self.process
                    and self.process.returncode is not None
                ):
                    logger.error("Chatterbox process died during initialization")
                    break

        if self._closing:
            raise RuntimeError("Chatterbox backend is closing")

        if not self._initialized:
            logger.error("ChatterboxTTSBackend: Model not initialized after waiting")
            raise ValueError(
                "Chatterbox model not initialized. Please check installation."
            )

        # Validate input
        if not request.input:
            raise ValueError("Text input is required.")

        # Preprocess text if enabled
        text = self.preprocess_text(request.input)

        # Get voice settings from request
        voice = request.voice
        reference_audio_path = None

        # Check if using custom voice
        if voice.startswith("custom:"):
            # Extract reference audio path
            reference_audio_path = voice.replace("custom:", "")
            if not os.path.exists(reference_audio_path):
                logger.warning(f"Reference audio not found: {reference_audio_path}")
                reference_audio_path = None
        elif voice.startswith("profile:"):
            # Handle voice profile
            profile_name = voice[8:]  # Remove "profile:" prefix
            # Try to get profile from voice manager
            try:
                from .chatterbox_voice_manager import ChatterboxVoiceManager

                manager = ChatterboxVoiceManager(self.voice_dir)
                reference_audio_path = manager.get_reference_audio_path(profile_name)
                if not reference_audio_path:
                    logger.warning(f"Voice profile '{profile_name}' not found")
            except Exception as e:
                logger.error(f"Failed to load voice profile: {e}")
                reference_audio_path = None
        elif voice != "default":
            # Check for predefined voice
            predefined_path = self.voice_dir / f"{voice}.wav"
            if predefined_path.exists():
                reference_audio_path = str(predefined_path)

        emitted = False
        try:
            # Get extra parameters if available
            exaggeration = self.exaggeration
            cfg_weight = self.cfg_weight
            temperature = self.temperature
            num_candidates = self.num_candidates
            validate_with_whisper = self.validate_with_whisper

            # Check for custom parameters in the request
            if hasattr(request, "extra_params") and request.extra_params:
                exaggeration = request.extra_params.get("exaggeration", exaggeration)
                cfg_weight = request.extra_params.get("cfg_weight", cfg_weight)
                temperature = request.extra_params.get("temperature", temperature)
                num_candidates = request.extra_params.get(
                    "num_candidates", num_candidates
                )
                validate_with_whisper = request.extra_params.get(
                    "validate_with_whisper", validate_with_whisper
                )

            logger.info(
                f"Generating speech with Chatterbox (exaggeration={exaggeration}, cfg_weight={cfg_weight}, candidates={num_candidates})"
            )

            # Check if text needs chunking
            if len(text) > self.max_chunk_size:
                logger.info(
                    f"Text length ({len(text)}) exceeds max chunk size ({self.max_chunk_size}), chunking..."
                )
                chunks = self.chunk_text(text)

                # Generate audio for each chunk
                all_audio_bytes = []
                retained_bytes = 0
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i + 1}/{len(chunks)}")

                    # Generate audio for chunk
                    if num_candidates > 1 or validate_with_whisper:
                        audio_bytes = await self.generate_with_validation(
                            chunk, reference_audio_path, exaggeration, cfg_weight
                        )
                    else:
                        audio_bytes = await self._generate_single(
                            chunk,
                            reference_audio_path,
                            exaggeration,
                            cfg_weight,
                            temperature,
                        )

                    retained_bytes += len(audio_bytes)
                    check_buffered_audio_size(retained_bytes)
                    all_audio_bytes.append(audio_bytes)

                # Combine all chunks with crossfade if enabled
                if self.enable_crossfade and len(all_audio_bytes) > 1:
                    logger.info("Applying crossfade between audio chunks")
                    combined_audio = await self._combine_audio_with_crossfade(
                        all_audio_bytes
                    )
                else:
                    combined_audio = await self._combine_audio_with_crossfade(
                        all_audio_bytes, crossfade=False
                    )

                # Convert format if needed
                output_bytes = await self._encode_audio(
                    combined_audio, request.response_format
                )
                emitted = True
                yield output_bytes

            else:
                # Single chunk generation
                if (
                    request.stream
                    and self.streaming_enabled
                    and num_candidates <= 1
                    and not validate_with_whisper
                    and hasattr(self.model, "generate_stream")
                ):
                    audio_chunks = []
                    retained_bytes = 0
                    async with contextlib.aclosing(
                        self._generate_stream_async(
                            text, reference_audio_path, exaggeration, cfg_weight
                        )
                    ) as stream:
                        async for audio_chunk, _metrics in stream:
                            chunk_bytes = self._tensor_to_wav_bytes(
                                audio_chunk, self.model.sr
                            )
                            if request.response_format == "pcm":
                                # Raw samples concatenate safely; file containers do not.
                                output = await self._encode_audio(chunk_bytes, "pcm")
                                emitted = True
                                yield output
                            else:
                                retained_bytes += len(chunk_bytes)
                                check_buffered_audio_size(retained_bytes)
                                audio_chunks.append(chunk_bytes)
                    if not emitted and not audio_chunks:
                        raise ValueError("Chatterbox returned no audio.")
                    if request.response_format != "pcm":
                        complete = await self._combine_audio_with_crossfade(
                            audio_chunks, crossfade=False
                        )
                        output = await self._encode_audio(
                            complete, request.response_format
                        )
                        emitted = True
                        yield output

                    await self._report_progress(
                        progress=1.0,
                        processed=1,
                        total=1,
                        status="Generation complete",
                        metrics={"format": request.response_format, "streaming": True},
                    )
                    return

                if num_candidates > 1 or validate_with_whisper:
                    # Use multi-candidate generation with validation
                    audio_bytes = await self.generate_with_validation(
                        text, reference_audio_path, exaggeration, cfg_weight
                    )
                else:
                    # Use single generation
                    audio_bytes = await self._generate_single(
                        text,
                        reference_audio_path,
                        exaggeration,
                        cfg_weight,
                        temperature,
                    )

                # Report near completion
                await self._report_progress(
                    progress=0.9, processed=1, total=1, status="Finalizing audio"
                )

                # Convert format if needed
                output_bytes = await self._encode_audio(
                    audio_bytes, request.response_format
                )

                # Report completion
                await self._report_progress(
                    progress=1.0,
                    processed=1,
                    total=1,
                    status="Generation complete",
                    metrics={"format": request.response_format},
                )

                emitted = True
                yield output_bytes

        except TTSOperationError:
            raise
        except Exception as e:
            logger.error(f"Chatterbox generation failed: {e}")
            if emitted or not allow_fallback:
                raise
            async with contextlib.aclosing(
                self.generate_speech_stream_with_fallback(request)
            ) as stream:
                async for chunk in stream:
                    yield chunk

    async def _encode_audio(self, wav_bytes: bytes, audio_format: str) -> bytes:
        """Encode one complete utterance without retrying deterministic codec errors."""
        check_buffered_audio_size(len(wav_bytes))
        if audio_format == "wav":
            return wav_bytes
        try:
            return await self.audio_service.convert_audio(
                wav_bytes,
                audio_format,
                source_format="wav",
                sample_rate=24000 if audio_format == "pcm" else None,
            )
        except TTSOperationError:
            raise
        except Exception:
            raise TTSOperationError(
                code="audio_response_invalid",
                message="Unable to encode Chatterbox audio. Try WAV output.",
                retryable=False,
                operation_id=uuid4().hex,
                recovery_action="use_wav",
            ) from None

    async def _generate_stream_async(
        self,
        text: str,
        audio_prompt_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
    ) -> AsyncGenerator:
        """Advance inference only on demand and join a running step on cancellation."""
        generator = self.model.generate_stream(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            chunk_size=self.chunk_size,
        )
        sentinel = object()
        try:
            while True:
                step = asyncio.create_task(asyncio.to_thread(next, generator, sentinel))
                await join_retained_task(step)
                result = step.result()
                if result is sentinel:
                    break
                yield result
        finally:
            close = getattr(generator, "close", None)
            if close is not None:
                cleanup = asyncio.create_task(asyncio.to_thread(close))
                await join_retained_task(cleanup)

    async def _combine_audio_with_crossfade(
        self, audio_chunks: List[bytes], *, crossfade: bool = True
    ) -> bytes:
        """Decode WAV chunks and encode one file; never concatenate containers."""
        import io
        import torch
        import wave
        import numpy as np

        if not audio_chunks:
            raise ValueError("Chatterbox returned no audio.")
        if len(audio_chunks) == 1:
            check_buffered_audio_size(len(audio_chunks[0]))
            return audio_chunks[0]
        tensors = []
        sample_rate = None
        retained_bytes = 0
        for chunk in audio_chunks:
            check_buffered_audio_size(len(chunk) * 2)
            with wave.open(io.BytesIO(chunk), "rb") as wav:
                rate = wav.getframerate()
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                if width != 2:
                    raise ValueError("Chatterbox returned unsupported WAV samples.")
                raw = wav.readframes(wav.getnframes())
                if len(raw) != wav.getnframes() * channels * width:
                    raise ValueError("Chatterbox returned truncated WAV audio.")
            if sample_rate is not None and rate != sample_rate:
                raise ValueError("Chatterbox returned inconsistent sample rates.")
            sample_rate = rate
            retained_bytes += len(raw) * 2
            check_buffered_audio_size(retained_bytes)
            samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
            tensor = torch.from_numpy(samples)
            tensors.append(tensor)
        if crossfade:
            result = tensors[0]
            for tensor in tensors[1:]:
                result = self.crossfade_audio_chunks(
                    result, tensor, self.crossfade_duration_ms
                )
        else:
            result = torch.cat(tensors)
        return self._tensor_to_wav_bytes(result, sample_rate)

    def _tensor_to_wav_bytes(self, tensor, sample_rate: int) -> bytes:
        """Encode finite mono PCM16 WAV without torchaudio's optional codecs."""
        import io
        import wave
        import torch

        check_buffered_audio_size(tensor.numel() * tensor.element_size())
        if (
            tensor.dim() not in (1, 2)
            or tensor.numel() == 0
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Chatterbox returned invalid audio samples.")
        audio = tensor.detach().cpu()
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
        pcm = (audio.clamp(-1.0, 1.0).numpy() * 32767.0).astype("<i2").tobytes()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setparams((1, 2, int(sample_rate), 0, "NONE", "not compressed"))
            output.writeframes(pcm)
        return buffer.getvalue()

    @voice_files.call
    async def list_voices(self) -> list[str]:
        """List available voices (predefined and custom)"""
        voices = ["default"]

        # Add predefined voices from voice directory
        if self.voice_dir.exists():
            for voice_file in self.voice_dir.glob("*.wav"):
                voices.append(voice_file.stem)

        return voices

    @voice_files.call
    async def save_reference_voice(self, name: str, audio_path: str) -> bool:
        """Save a reference audio file as a predefined voice"""
        try:

            # Validate audio file
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return False

            # Copy to voice directory
            dest_path = self.voice_dir / f"{name}.wav"
            voice_files.copy(self, audio_path, dest_path)

            logger.info(f"Saved voice '{name}' to {dest_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save reference voice: {e}")
            return False

    @voice_files.call
    async def save_reference_voice_with_metadata(
        self, name: str, audio_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a reference audio file as a predefined voice with metadata.

        Args:
            name: Voice name
            audio_path: Path to audio file
            metadata: Optional metadata dictionary

        Returns:
            Success status
        """
        try:

            # Validate audio file
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return False

            # Save audio file
            voice_path = self.voice_dir / f"{name}.wav"
            voice_files.copy(self, audio_path, voice_path)

            # Prepare metadata
            if metadata is None:
                metadata = {}

            metadata.update(
                {
                    "created_at": datetime.now().isoformat(),
                    "audio_file": f"{name}.wav",
                    "file_size": os.path.getsize(audio_path),
                    "original_path": audio_path,
                }
            )

            # Try to get audio duration
            try:
                import torchaudio

                info = torchaudio.info(audio_path)
                metadata["duration_seconds"] = info.num_frames / info.sample_rate
                metadata["sample_rate"] = info.sample_rate
            except Exception as e:
                logger.debug(f"Could not extract audio info: {e}")

            # Save metadata
            metadata_path = self.voice_dir / f"{name}_metadata.json"
            with voice_files.open_text(self, metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved voice '{name}' with metadata to {voice_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save voice with metadata: {e}")
            return False

    @voice_files.call
    async def list_voices_with_metadata(self) -> List[Dict[str, Any]]:
        """
        List available voices with their metadata.

        Returns:
            List of voice dictionaries with metadata
        """
        voices = []

        # Add default voice
        voices.append({"name": "default", "has_metadata": False, "metadata": {}})

        # Add predefined voices from voice directory
        if self.voice_dir.exists():
            for voice_file in self.voice_dir.glob("*.wav"):
                voice_name = voice_file.stem
                voice_info = {
                    "name": voice_name,
                    "file_path": str(voice_file),
                    "has_metadata": False,
                    "metadata": {},
                }

                # Check for metadata file
                metadata_path = self.voice_dir / f"{voice_name}_metadata.json"
                if metadata_path.exists():
                    try:
                        with voice_files.open_text(self, metadata_path, "r") as f:
                            voice_info["metadata"] = json.load(f)
                            voice_info["has_metadata"] = True
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {voice_name}: {e}")

                voices.append(voice_info)

        return voices

    async def generate_speech_stream_with_fallback(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech with multiple fallback strategies.

        Strategies:
        1. High quality: Default parameters
        2. Balanced: Reduced exaggeration/CFG
        3. Safe: Minimal exaggeration/CFG
        """
        strategies = [
            (
                "high_quality",
                {"exaggeration": 0.5, "cfg_weight": 0.5, "num_candidates": 3},
            ),
            ("balanced", {"exaggeration": 0.3, "cfg_weight": 0.7, "num_candidates": 2}),
            ("safe", {"exaggeration": 0.1, "cfg_weight": 0.9, "num_candidates": 1}),
        ]

        original_params = {
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
            "num_candidates": self.num_candidates,
        }

        try:
            for strategy_name, params in strategies:
                emitted = False
                try:
                    logger.info(f"Trying {strategy_name} generation strategy")
                    for key, value in params.items():
                        setattr(self, key, value)
                    # Per-request overrides otherwise mask the fallback parameters.
                    attempt = request.model_copy(
                        update={
                            "extra_params": {**(request.extra_params or {}), **params}
                        }
                    )
                    async with contextlib.aclosing(
                        self._generate_speech_stream(attempt, allow_fallback=False)
                    ) as stream:
                        async for chunk in stream:
                            emitted = True
                            yield chunk
                    return
                except TTSOperationError:
                    raise
                except Exception as e:
                    if emitted:
                        raise
                    logger.warning(f"{strategy_name} strategy failed: {e}")
        finally:
            for key, value in original_params.items():
                setattr(self, key, value)
        raise ValueError("All generation strategies failed")

    async def close(self):
        """Join owned initialization and inference before releasing resources."""
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned_resources())
        await join_retained_task(self._close_task)

    async def _close_owned_resources(self) -> None:
        """Cancel readiness, retaining native loading and child acquisition."""
        if self._initialization_task is not None:
            self._initialization_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await join_retained_task(self._initialization_task)
        async with self._generation_lock:
            await self._close_resources()

    async def _close_resources(self):
        """Clean up resources after the active operation has stopped."""
        # Clean up subprocess if running
        if hasattr(self, "process") and self.process:
            try:
                # Send shutdown command
                await self._send_command({"command": "shutdown"})
                # Give it time to shut down gracefully
                await asyncio.sleep(0.5)
            except Exception:
                pass

            await self._discard_process()

        # Clean up model if needed (only if not using process)
        if self.model is not None and self.model != "process":
            del self.model
        self.model = None
        self._initialized = False
        self._initializing = False

        # Clean up transcription service
        if self.transcription_service is not None:
            self.transcription_service = None

        logger.info("ChatterboxTTSBackend: Resources cleaned up")


#
# End of chatterbox.py
#######################################################################################################################
