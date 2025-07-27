# higgs.py
# Description: Higgs Audio V2 TTS backend implementation
#
# Imports
import asyncio
import os
import sys
import time
import io
import json
import wave
import tempfile
import shutil
import hashlib
import subprocess
import base64
from contextlib import contextmanager
from typing import AsyncGenerator, Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
from pathlib import Path
from loguru import logger

# Optional imports for audio processing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    logger.warning("numpy not available. Higgs Audio backend requires numpy.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    logger.warning("PyTorch not available. Higgs Audio backend requires PyTorch.")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None
    logger.warning("torchaudio not available. Voice cloning features will be limited.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None
    logger.warning("librosa not available. Advanced audio analysis features will be limited.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None
    logger.warning("soundfile not available. Some audio format support will be limited.")

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.base_backends import LocalTTSBackend
from tldw_chatbook.TTS.audio_service import AudioService, get_audio_service
from tldw_chatbook.TTS.text_processing import TextChunker, TextNormalizer, detect_language
from tldw_chatbook.config import get_cli_setting


@contextmanager
def protect_file_descriptors():
    """Context manager to protect file descriptors during subprocess operations.
    
    This fixes the "bad value(s) in fds_to_keep" error on macOS when the 
    transformers library spawns subprocesses for model downloads.
    """
    # Save original file descriptors
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_stdin = sys.stdin
    
    # Save original environment
    env_backup = os.environ.copy()
    
    # Save original subprocess.Popen to restore later
    original_popen = subprocess.Popen
    
    try:
        # Ensure we have real file descriptors, not wrapped objects
        # This is crucial for subprocess operations
        try:
            # Test if stdout/stderr are real files with valid file descriptors
            stdout_fd = sys.stdout.fileno()
            stderr_fd = sys.stderr.fileno()
            # Verify they're valid by attempting to use them
            os.fstat(stdout_fd)
            os.fstat(stderr_fd)
        except (AttributeError, ValueError, OSError):
            # stdout/stderr are wrapped/captured or invalid, create new ones
            # Use the original file descriptors 1 and 2 directly
            try:
                sys.stdout = os.fdopen(1, 'w')
                sys.stderr = os.fdopen(2, 'w')
            except OSError:
                # If that fails, use devnull as a fallback
                devnull = open(os.devnull, 'w')
                sys.stdout = devnull
                sys.stderr = devnull
        
        # Set environment to prevent subprocess issues
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        
        # For macOS specifically
        if sys.platform == 'darwin':
            os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
            # Ensure subprocess doesn't inherit bad file descriptors
            os.environ['PYTHONNOUSERSITE'] = '1'
            # Force subprocess to close all file descriptors except 0,1,2
            os.environ['PYTHON_SUBPROCESS_CLOSE_FDS'] = '1'
        
        yield
        
    finally:
        # Restore original file descriptors
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        sys.stdin = original_stdin
        
        # Close any temporary files we created
        if sys.stdout != original_stdout and hasattr(sys.stdout, 'close'):
            try:
                sys.stdout.close()
            except:
                pass
        if sys.stderr != original_stderr and hasattr(sys.stderr, 'close'):
            try:
                sys.stderr.close()
            except:
                pass
        
        # Restore environment
        os.environ.clear()
        os.environ.update(env_backup)
        
        # Restore subprocess.Popen
        subprocess.Popen = original_popen


#######################################################################################################################
#
# Higgs Audio TTS Backend Implementation
#
class HiggsAudioTTSBackend(LocalTTSBackend):
    """
    Higgs Audio V2 Text-to-Speech backend.
    
    Features:
    - Zero-shot voice cloning from reference audio
    - Multi-speaker dialog generation
    - Multilingual audio generation
    - Automatic prosody adaptation
    - Background music support (optional)
    - Streaming audio generation
    
    Based on: https://github.com/boson-ai/higgs-audio
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Validate required dependencies
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for Higgs Audio backend but is not installed. "
                            "Install it with: pip install numpy")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Higgs Audio backend but is not installed. "
                            "Install it with: pip install torch")
        
        # Lazy-loaded Higgs modules
        self._higgs_serve_engine = None
        self._boson_multimodal = None
        
        # Shutdown and task tracking
        self._active_tasks = set()
        self._shutdown_event = asyncio.Event()
        self._generation_lock = asyncio.Lock()
        
        # Model configuration
        self.model_path = self.config.get("HIGGS_MODEL_PATH", 
                                         get_cli_setting("HiggsSettings", "model_path",
                                                       "bosonai/higgs-audio-v2-generation-3B-base"))
        self.audio_tokenizer_path = self.config.get("HIGGS_AUDIO_TOKENIZER_PATH",
                                                   get_cli_setting("HiggsSettings", "audio_tokenizer_path",
                                                                 "bosonai/higgs-audio-v2-tokenizer"))
        self.device = self.config.get("HIGGS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.enable_flash_attn = self.config.get("HIGGS_ENABLE_FLASH_ATTN", True)
        self.dtype = self.config.get("HIGGS_DTYPE", "bfloat16")  # Options: "float32", "float16", "bfloat16"
        
        # Voice configuration
        self.voice_samples_dir = Path(self.config.get("HIGGS_VOICE_SAMPLES_DIR",
                                                     get_cli_setting("HiggsSettings", "voice_samples_dir",
                                                                   "~/.config/tldw_cli/higgs_voices"))).expanduser()
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_voice_cloning = self.config.get("HIGGS_ENABLE_VOICE_CLONING", True)
        self.max_reference_duration = self.config.get("HIGGS_MAX_REFERENCE_DURATION", 30)  # seconds
        self.default_language = self.config.get("HIGGS_DEFAULT_LANGUAGE", "en")
        self.enable_background_music = self.config.get("HIGGS_ENABLE_BACKGROUND_MUSIC", False)
        
        # Voice profiles storage
        self.voice_profiles_file = self.voice_samples_dir / "voice_profiles.json"
        self.voice_profiles = self._load_voice_profiles()
        
        # Create profiles directory for saved profiles
        self.profiles_dir = self.voice_samples_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default voice profiles if none exist
        if not self.voice_profiles:
            self._create_default_profiles()
        
        logger.info(f"Loaded {len(self.voice_profiles)} voice profiles from {self.voice_profiles_file}")
        
        # Services
        self.audio_service = get_audio_service()
        self.text_chunker = TextChunker(max_tokens=self.config.get("HIGGS_MAX_TOKENS", 500))
        self.normalizer = TextNormalizer()
        
        # Multi-speaker configuration
        self.enable_multi_speaker = self.config.get("HIGGS_ENABLE_MULTI_SPEAKER", True)
        self.speaker_delimiter = self.config.get("HIGGS_SPEAKER_DELIMITER", "|||")
        
        # Performance tracking
        self.track_performance = self.config.get("HIGGS_TRACK_PERFORMANCE", True)
        self._performance_metrics = {
            "total_tokens": 0,
            "total_time": 0.0,
            "generation_count": 0,
            "voice_cloning_count": 0
        }
        
        # Model instance
        self.serve_engine = None
        
        logger.info(f"HiggsAudioTTSBackend initialized with device: {self.device}, model: {self.model_path}")
    
    async def initialize(self):
        """Initialize the Higgs Audio backend"""
        logger.info("═" * 80)
        logger.info("🚀 HIGGS AUDIO INITIALIZATION STARTED")
        logger.info(f"📦 Model: {self.model_path}")
        logger.info(f"🖥️  Device: {self.device}")
        logger.info("═" * 80)
        
        start_time = time.time()
        await self.load_model()
        
        elapsed = time.time() - start_time
        logger.info("═" * 80)
        logger.info(f"✅ HIGGS AUDIO INITIALIZATION COMPLETE (took {elapsed:.1f}s)")
        logger.info("═" * 80)
    
    async def load_model(self):
        """Load the Higgs Audio model"""
        try:
            # Import Higgs modules
            if self._boson_multimodal is None:
                logger.info("📚 Step 1/4: Importing Higgs Audio modules...")
                try:
                    import boson_multimodal
                    self._boson_multimodal = boson_multimodal
                    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
                    self._higgs_serve_engine = HiggsAudioServeEngine
                    # Import the required data types
                    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
                    self._ChatMLSample = ChatMLSample
                    self._Message = Message
                    self._AudioContent = AudioContent
                    logger.info("✓ Step 1/4: Higgs Audio modules imported successfully")
                except ImportError as e:
                    raise ImportError(
                        "\n\n❌ Higgs Audio backend requires manual installation!\n\n"
                        "The 'boson-multimodal' package cannot be installed via pip.\n"
                        "Please follow these steps:\n\n"
                        "1. Clone and install Higgs Audio:\n"
                        "   git clone https://github.com/boson-ai/higgs-audio.git\n"
                        "   cd higgs-audio\n"
                        "   pip install -r requirements.txt\n"
                        "   pip install -e .\n"
                        "   cd ..\n\n"
                        "2. Then reinstall tldw_chatbook with Higgs support:\n"
                        "   pip install -e \".[higgs_tts]\"\n\n"
                        "For automated installation, run: ./scripts/install_higgs.sh\n"
                        "For detailed instructions, see: Docs/Higgs-Audio-TTS-Guide.md\n"
                    ) from e
            
            # Convert dtype string to torch dtype
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)
            
            # Create serve engine with configuration
            logger.info("🔍 Step 2/4: Checking HiggsAudioServeEngine parameters...")
            
            # Check which parameters the HiggsAudioServeEngine accepts
            import inspect
            init_params = inspect.signature(self._higgs_serve_engine.__init__).parameters
            logger.info(f"✓ Step 2/4: Found {len(init_params)} supported parameters")
            
            # Build kwargs based on what the engine accepts
            engine_kwargs = {
                "model_name_or_path": self.model_path,
                "device": self.device
            }
            
            # Add audio_tokenizer_name_or_path if the engine supports it
            if "audio_tokenizer_name_or_path" in init_params:
                engine_kwargs["audio_tokenizer_name_or_path"] = self.audio_tokenizer_path
                logger.debug("HiggsAudioServeEngine supports audio_tokenizer_name_or_path parameter")
            else:
                logger.debug("HiggsAudioServeEngine does not support audio_tokenizer_name_or_path parameter, skipping")
            
            # Only add dtype if the engine supports it
            if "dtype" in init_params:
                engine_kwargs["dtype"] = torch_dtype
                logger.debug("HiggsAudioServeEngine supports dtype parameter")
            else:
                logger.debug("HiggsAudioServeEngine does not support dtype parameter, skipping")
            
            # Only add enable_flash_attn if the engine supports it
            if "enable_flash_attn" in init_params:
                engine_kwargs["enable_flash_attn"] = self.enable_flash_attn
                logger.debug("HiggsAudioServeEngine supports enable_flash_attn parameter")
            else:
                logger.debug("HiggsAudioServeEngine does not support enable_flash_attn parameter, skipping")
            
            # Use protect_file_descriptors to prevent "bad value(s) in fds_to_keep" error
            # Run model loading in a thread to avoid blocking the UI
            logger.info("🔄 Step 3/4: Loading model (this may take a while on first run)...")
            logger.info(f"📥 Model path: {engine_kwargs.get('model_name_or_path')}")
            logger.info(f"🎵 Audio tokenizer: {engine_kwargs.get('audio_tokenizer_name_or_path', 'default')}")
            logger.info("⏳ Running in background thread to keep UI responsive...")
            
            load_start = time.time()
            
            # Check if shutdown requested before loading
            if self._shutdown_event.is_set():
                logger.info("Shutdown requested, aborting model load")
                return
            
            # Create the loading function (not async since it runs in thread)
            def load_model_thread():
                with protect_file_descriptors():
                    return self._higgs_serve_engine(**engine_kwargs)
            
            # Run in thread with cancellation support
            try:
                load_task = asyncio.create_task(asyncio.to_thread(load_model_thread))
                self._active_tasks.add(load_task)
                
                # Wait for either completion or shutdown
                done, pending = await asyncio.wait(
                    [load_task, asyncio.create_task(self._shutdown_event.wait())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if self._shutdown_event.is_set():
                    logger.info("Shutdown during model load, cancelling...")
                    load_task.cancel()
                    try:
                        await load_task
                    except asyncio.CancelledError:
                        pass
                    return
                
                self.serve_engine = await load_task
                
            finally:
                self._active_tasks.discard(load_task)
            
            load_elapsed = time.time() - load_start
            logger.info(f"✓ Step 3/4: Model loaded successfully in {load_elapsed:.1f}s")
            
            logger.info("🏁 Step 4/4: Finalizing initialization...")
            self.model_loaded = True
            logger.info(f"✓ Step 4/4: Higgs Audio ready on {self.device}")
            
        except Exception as e:
            logger.error("═" * 80)
            logger.error("❌ HIGGS AUDIO INITIALIZATION FAILED")
            logger.error(f"🚨 Error: {str(e)}")
            logger.error("═" * 80)
            logger.error(f"Failed to load Higgs Audio model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Higgs Audio model: {str(e)}")
    
    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech using Higgs Audio and stream the response.
        
        Args:
            request: Speech request parameters
            
        Yields:
            Audio bytes in the requested format
        """
        # Check for shutdown
        if self._shutdown_event.is_set():
            logger.info("Shutdown requested, aborting generation")
            return
        
        # Ensure model is loaded
        if not self.model_loaded:
            logger.info("⚠️  Model not loaded, initializing now...")
            await self.initialize()
            
            # Check again after initialization
            if self._shutdown_event.is_set():
                logger.info("Shutdown requested after init, aborting generation")
                return
        
        start_time = time.time()
        
        # Acquire generation lock to track active generation
        async with self._generation_lock:
            try:
                # Parse voice configuration
                voice_config = await self._prepare_voice_config(request.voice)
            
                # Detect or use specified language
                language = self._get_language(request)
                
                # Normalize text if requested
                text = request.input
                if request.normalization_options:
                    text = self.normalizer.normalize_text(text)
                
                logger.info("─" * 60)
                logger.info("🎤 STARTING TTS GENERATION")
                logger.info(f"📝 Text length: {len(text)} characters")
                logger.info(f"🗣️  Voice: {voice_config.get('display_name', 'custom')}")
                logger.info(f"🌐 Language: {language}")
                logger.info(f"📀 Format: {request.response_format}")
                logger.info("─" * 60)
                
                # Check for multi-speaker dialog
                if self.enable_multi_speaker and self.speaker_delimiter in text:
                    # Generate multi-speaker dialog
                    async for chunk in self._generate_multi_speaker_stream(
                        text, voice_config, request, language
                    ):
                        yield chunk
                else:
                    # Single speaker generation
                    async for chunk in self._generate_single_speaker_stream(
                        text, voice_config, request, language
                    ):
                        yield chunk
                
                # Update performance metrics
                if self.track_performance:
                    generation_time = time.time() - start_time
                    self._update_performance_metrics(len(text.split()), generation_time)
                    
            except Exception as e:
                logger.error("═" * 60)
                logger.error("❌ GENERATION FAILED")
                logger.error(f"🚨 Error: {str(e)}")
                logger.error("═" * 60)
                logger.error(f"HiggsAudioTTSBackend: Error during generation: {e}", exc_info=True)
                yield f"ERROR: Higgs Audio generation failed - {str(e)}".encode('utf-8')
    
    async def _generate_single_speaker_stream(
        self, text: str, voice_config: Dict[str, Any], 
        request: OpenAISpeechRequest, language: str
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio for a single speaker"""
        try:
            start_time = time.time()
            
            # Prepare messages for Higgs
            messages = self._prepare_messages(text, voice_config, language)
            
            # Track generation progress
            total_samples = 0
            sample_rate = 24000  # Higgs default sample rate
            first_chunk_time = None
            
            # Report initial progress
            await self._report_progress(
                progress=0.0,
                processed=0,
                total=len(text.split()),
                status="Starting Higgs Audio generation",
                metrics={"voice": voice_config.get("display_name", "custom"), "language": language}
            )
            
            # Create ChatMLSample from messages
            chat_ml_sample = self._ChatMLSample(messages=messages)
            
            # Check for shutdown before generation
            if self._shutdown_event.is_set():
                logger.info("Shutdown requested, aborting single speaker generation")
                return
            
            # Generate audio using Higgs
            logger.info("🎵 Generating audio with Higgs Audio model...")
            gen_start = time.time()
            
            # Create generation task with cancellation support
            gen_task = asyncio.create_task(
                asyncio.to_thread(
                    self.serve_engine.generate,
                    chat_ml_sample=chat_ml_sample,  # Use ChatMLSample instead of messages
                    max_new_tokens=self.config.get("HIGGS_MAX_NEW_TOKENS", 4096),
                    temperature=self.config.get("HIGGS_TEMPERATURE", 0.7),
                    top_p=self.config.get("HIGGS_TOP_P", 0.95),
                    top_k=self.config.get("HIGGS_TOP_K", 50),
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                    force_audio_gen=True,  # Force audio generation
                    ras_win_len=7,
                    ras_win_max_num_repeat=2
                )
            )
            
            self._active_tasks.add(gen_task)
            
            try:
                # Wait for either generation or shutdown
                done, pending = await asyncio.wait(
                    [gen_task, asyncio.create_task(self._shutdown_event.wait())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if self._shutdown_event.is_set():
                    logger.info("Shutdown during generation, cancelling...")
                    gen_task.cancel()
                    try:
                        await gen_task
                    except asyncio.CancelledError:
                        pass
                    return
                
                output = await gen_task
                
            finally:
                self._active_tasks.discard(gen_task)
            
            gen_elapsed = time.time() - gen_start
            logger.info(f"✓ Audio generated in {gen_elapsed:.1f}s")
            
            # Extract audio from output
            if hasattr(output, 'audio') and output.audio is not None:
                audio_data = output.audio
                
                # Convert to numpy array if needed
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()
                
                # Ensure audio is in the right format (mono, float32)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=0)  # Convert to mono
                
                audio_data = audio_data.astype(np.float32)
                
                # Normalize audio to prevent clipping
                max_val = np.abs(audio_data).max()
                if max_val > 1.0:
                    audio_data = audio_data / max_val
                
                total_samples = len(audio_data)
                
                # Convert to requested format and stream
                if request.response_format == "pcm":
                    # Direct PCM streaming
                    chunk_size = 8192
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        int16_samples = np.int16(chunk * 32767)
                        
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - start_time
                        
                        yield int16_samples.tobytes()
                        
                        # Report progress
                        progress = min(0.95, (i + len(chunk)) / len(audio_data))
                        await self._report_progress(
                            progress=progress,
                            processed=i + len(chunk),
                            total=len(audio_data),
                            status=f"Streaming PCM audio: {(i + len(chunk)) / sample_rate:.1f}s",
                            metrics={"format": "pcm", "sample_rate": sample_rate}
                        )
                else:
                    # Convert to other formats
                    audio_bytes = await self.audio_service.convert_audio(
                        audio_data,
                        request.response_format,
                        source_format="pcm",
                        sample_rate=sample_rate
                    )
                    
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                    
                    yield audio_bytes
                
                # Log performance
                generation_time = time.time() - start_time
                audio_duration = total_samples / sample_rate
                speed_factor = audio_duration / generation_time if generation_time > 0 else 0
                
                logger.info("─" * 60)
                logger.info("✅ GENERATION COMPLETE")
                logger.info(f"⏱️  Total time: {generation_time:.2f}s")
                logger.info(f"🎵 Audio duration: {audio_duration:.2f}s")
                logger.info(f"⚡ Speed: {speed_factor:.1f}x realtime")
                logger.info(f"📊 First chunk latency: {first_chunk_time:.3f}s")
                logger.info("─" * 60)
                
                # Report completion
                await self._report_progress(
                    progress=1.0,
                    processed=total_samples,
                    total=total_samples,
                    status=f"Generation complete: {audio_duration:.1f}s of {request.response_format} audio",
                    metrics={
                        "sample_rate": sample_rate,
                        "audio_duration": audio_duration,
                        "generation_time": generation_time,
                        "speed_factor": speed_factor
                    }
                )
            else:
                logger.error("⚠️  No audio generated from model - output.audio is None")
                yield b""
                
        except Exception as e:
            logger.error(f"HiggsAudioTTSBackend: Single speaker generation failed: {e}", exc_info=True)
            raise
    
    async def _generate_multi_speaker_stream(
        self, text: str, default_voice_config: Dict[str, Any],
        request: OpenAISpeechRequest, language: str
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio for multi-speaker dialog"""
        try:
            # Parse speaker sections
            sections = self._parse_multi_speaker_text(text)
            
            if not sections:
                # Fallback to single speaker if parsing fails
                async for chunk in self._generate_single_speaker_stream(
                    text, default_voice_config, request, language
                ):
                    yield chunk
                return
            
            logger.info(f"HiggsAudioTTSBackend: Generating multi-speaker dialog with {len(sections)} sections")
            
            # Generate audio for each section
            all_audio = []
            sample_rate = 24000
            
            for i, (speaker, section_text) in enumerate(sections):
                # Get voice config for this speaker
                if speaker and speaker != "narrator":
                    speaker_voice_config = await self._prepare_voice_config(speaker)
                else:
                    speaker_voice_config = default_voice_config
                
                logger.debug(f"Section {i+1}: Speaker='{speaker}', Voice={speaker_voice_config.get('display_name', 'custom')}")
                
                # Generate audio for this section
                messages = self._prepare_messages(section_text, speaker_voice_config, language)
                chat_ml_sample = self._ChatMLSample(messages=messages)
                
                # Check for shutdown before each section
                if self._shutdown_event.is_set():
                    logger.info("Shutdown requested during multi-speaker generation")
                    return
                
                # Create generation task for this section
                section_task = asyncio.create_task(
                    asyncio.to_thread(
                        self.serve_engine.generate,
                        chat_ml_sample=chat_ml_sample,  # Use ChatMLSample instead of messages
                        max_new_tokens=self.config.get("HIGGS_MAX_NEW_TOKENS", 4096),
                        temperature=self.config.get("HIGGS_TEMPERATURE", 0.7),
                        top_p=self.config.get("HIGGS_TOP_P", 0.95),
                        top_k=self.config.get("HIGGS_TOP_K", 50),
                        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                        force_audio_gen=True
                    )
                )
                
                self._active_tasks.add(section_task)
                
                try:
                    output = await section_task
                finally:
                    self._active_tasks.discard(section_task)
                
                if hasattr(output, 'audio') and output.audio is not None:
                    audio_data = output.audio
                    
                    # Convert to numpy
                    if isinstance(audio_data, torch.Tensor):
                        audio_data = audio_data.cpu().numpy()
                    
                    # Ensure mono
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=0)
                    
                    all_audio.append(audio_data)
                
                # Report progress
                progress = (i + 1) / len(sections) * 0.9
                await self._report_progress(
                    progress=progress,
                    processed=i + 1,
                    total=len(sections),
                    status=f"Generated section {i+1}/{len(sections)}",
                    current_chunk=i + 1,
                    total_chunks=len(sections)
                )
            
            # Concatenate all audio
            if all_audio:
                full_audio = np.concatenate(all_audio)
                
                # Normalize
                max_val = np.abs(full_audio).max()
                if max_val > 1.0:
                    full_audio = full_audio / max_val
                
                # Convert to requested format
                if request.response_format == "pcm":
                    int16_samples = np.int16(full_audio * 32767)
                    yield int16_samples.tobytes()
                else:
                    audio_bytes = await self.audio_service.convert_audio(
                        full_audio,
                        request.response_format,
                        source_format="pcm",
                        sample_rate=sample_rate
                    )
                    yield audio_bytes
                
                # Report completion
                audio_duration = len(full_audio) / sample_rate
                await self._report_progress(
                    progress=1.0,
                    processed=len(sections),
                    total=len(sections),
                    status=f"Multi-speaker generation complete: {audio_duration:.1f}s",
                    total_chunks=len(sections),
                    metrics={"audio_duration": audio_duration, "sections": len(sections)}
                )
            
        except Exception as e:
            logger.error(f"HiggsAudioTTSBackend: Multi-speaker generation failed: {e}", exc_info=True)
            raise
    
    def _parse_multi_speaker_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse text with speaker markers.
        
        Format: "Speaker1|||Hello there! Speaker2|||Hi, how are you?"
        Returns: [("Speaker1", "Hello there!"), ("Speaker2", "Hi, how are you?")]
        """
        sections = []
        
        # If no delimiter present, return as narrator
        if self.speaker_delimiter not in text:
            return [("narrator", text.strip())]
        
        # Split by delimiter to get parts
        parts = text.split(self.speaker_delimiter)
        
        # Handle edge case of empty parts
        parts = [p for p in parts if p]  # Remove empty strings
        
        if not parts:
            return [("narrator", "")]
        
        # Process parts
        i = 0
        
        # Check if first part is narrator text (doesn't look like a speaker name)
        if i < len(parts) and not self._is_speaker_name(parts[i]):
            # First part contains spaces or punctuation, likely narrator text
            sections.append(("narrator", parts[i].strip()))
            i += 1
        
        # Process remaining parts as speaker-text pairs
        while i < len(parts):
            if i + 1 < len(parts):
                # We have a speaker and their text
                speaker_part = parts[i].strip()
                text_part = parts[i + 1].strip()
                
                # The speaker might be at the end of the previous text
                # E.g., "Hello there! John" where "John" is the speaker
                words = speaker_part.split()
                if len(words) > 1:
                    # Multiple words - last word is likely the speaker
                    speaker = words[-1]
                    # Add remaining text to previous speaker if any
                    remaining = " ".join(words[:-1]).strip()
                    if remaining and sections:
                        prev_speaker, prev_text = sections[-1]
                        sections[-1] = (prev_speaker, prev_text + " " + remaining)
                else:
                    # Single word - it's the speaker
                    speaker = speaker_part
                
                # Add this speaker and their text
                if text_part:  # Only add if there's actual text
                    sections.append((speaker, text_part))
                
                i += 2
            else:
                # Odd number of parts - last part is either a speaker with no text
                # or continuation of previous speaker's text
                last_part = parts[i].strip()
                if self._is_speaker_name(last_part) and not sections:
                    # It's a speaker with no text - ignore
                    pass
                elif sections:
                    # Add to previous speaker's text
                    prev_speaker, prev_text = sections[-1]
                    sections[-1] = (prev_speaker, prev_text + " " + last_part)
                else:
                    # No previous speaker, treat as narrator
                    sections.append(("narrator", last_part))
                i += 1
        
        return sections if sections else [("narrator", text.strip())]
    
    def _is_speaker_name(self, text: str) -> bool:
        """Check if text looks like a speaker name (single word, no spaces or special chars except dash/underscore)"""
        if not text:
            return False
        # Speaker names are typically single words, possibly with dash or underscore
        # They shouldn't contain spaces, punctuation (except - and _), or be too long
        import re
        return bool(re.match(r'^[\w-]+$', text)) and len(text) < 50 and ' ' not in text
    
    async def _prepare_voice_config(self, voice_name: str) -> Dict[str, Any]:
        """Prepare voice configuration from voice name or profile"""
        original_voice_name = voice_name
        
        # Handle custom: prefix for direct audio files
        if voice_name.startswith("custom:"):
            audio_path = voice_name[7:]  # Remove "custom:" prefix
            reference_path = Path(audio_path)
            if reference_path.exists():
                logger.info(f"Using custom voice from: {reference_path}")
                return {
                    "type": "zero_shot",
                    "reference_audio": str(reference_path),
                    "display_name": f"Cloned from {reference_path.name}"
                }
            else:
                logger.warning(f"Custom voice audio not found: {audio_path}")
        
        # Handle profile: prefix for saved profiles
        if voice_name.startswith("profile:"):
            profile_name = voice_name[8:]  # Remove "profile:" prefix
            profile_path = self.profiles_dir / f"{profile_name}.json"
            
            # Try to load profile from file
            if profile_path.exists():
                try:
                    profile_data = json.loads(profile_path.read_text())
                    if "reference_audio" in profile_data and Path(profile_data["reference_audio"]).exists():
                        logger.info(f"Loaded voice profile: {profile_name}")
                        return {
                            "type": "profile",
                            "profile_name": profile_name,
                            "display_name": profile_data.get("display_name", profile_name),
                            "reference_audio": profile_data["reference_audio"],
                            "language": profile_data.get("language", self.default_language),
                            "metadata": profile_data.get("metadata", {})
                        }
                except Exception as e:
                    logger.warning(f"Failed to load profile {profile_name}: {e}")
            
            # Also check in-memory profiles
            if profile_name in self.voice_profiles:
                profile = self.voice_profiles[profile_name]
                return {
                    "type": "profile",
                    "profile_name": profile_name,
                    "display_name": profile.get("display_name", profile_name),
                    "reference_audio": profile.get("reference_audio"),
                    "language": profile.get("language", self.default_language),
                    "metadata": profile.get("metadata", {})
                }
            
            logger.warning(f"Profile not found: {profile_name}")
        
        # Check if it's a voice profile (without prefix)
        if voice_name in self.voice_profiles:
            profile = self.voice_profiles[voice_name]
            return {
                "type": "profile",
                "profile_name": voice_name,
                "display_name": profile.get("display_name", voice_name),
                "reference_audio": profile.get("reference_audio"),
                "language": profile.get("language", self.default_language),
                "metadata": profile.get("metadata", {})
            }
        
        # Check if it's a path to an audio file (for zero-shot cloning)
        if self.enable_voice_cloning and (
            voice_name.endswith(('.wav', '.mp3', '.flac', '.ogg')) or
            os.path.exists(voice_name)
        ):
            # This is a reference audio file
            reference_path = Path(voice_name)
            if reference_path.exists():
                return {
                    "type": "zero_shot",
                    "reference_audio": str(reference_path),
                    "display_name": f"Cloned from {reference_path.name}"
                }
        
        # Check for OpenAI-style voice mapping
        voice_map = {
            "alloy": "professional_female",
            "echo": "warm_female",
            "fable": "storyteller_male",
            "onyx": "deep_male",
            "nova": "energetic_female",
            "shimmer": "soft_female"
        }
        
        mapped_voice = voice_map.get(voice_name.lower())
        if mapped_voice and mapped_voice in self.voice_profiles:
            return await self._prepare_voice_config(mapped_voice)
        
        # Default voice
        logger.info(f"Using default voice for: {original_voice_name}")
        return {
            "type": "default",
            "display_name": "Default Higgs Voice",
            "language": self.default_language
        }
    
    def _prepare_messages(self, text: str, voice_config: Dict[str, Any], language: str) -> List[Any]:
        """Prepare messages for Higgs Audio generation"""
        messages = []
        
        # Handle voice cloning with reference audio
        if voice_config.get("reference_audio"):
            ref_audio_path = voice_config["reference_audio"]
            if os.path.exists(ref_audio_path):
                # Voice cloning mode
                try:
                    import base64
                    with open(ref_audio_path, "rb") as audio_file:
                        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
                    
                    # Add reference text and audio for voice cloning
                    reference_text = "This is the reference voice for cloning."
                    messages.append(self._Message(
                        role="user",
                        content=reference_text
                    ))
                    messages.append(self._Message(
                        role="assistant",
                        content=self._AudioContent(raw_audio=audio_base64, audio_url="placeholder")
                    ))
                    logger.info(f"Added reference audio for voice cloning: {ref_audio_path}")
                except Exception as e:
                    logger.error(f"Failed to load reference audio: {e}")
        else:
            # Zero-shot mode - add system prompt
            system_content = f"Generate audio following instruction.\n\n<|scene_desc_start|>\nSPEAKER0: "
            
            # Add voice characteristics from profile if available
            if voice_config.get("type") == "profile" and voice_config.get("metadata"):
                metadata = voice_config["metadata"]
                descriptors = []
                
                if "style" in metadata:
                    descriptors.append(metadata["style"])
                if "pitch" in metadata:
                    descriptors.append(f"{metadata['pitch']} pitch")
                if "speed" in metadata:
                    descriptors.append(metadata["speed"])
                if "description" in metadata:
                    # Extract key characteristics from description
                    desc = metadata["description"].lower()
                    if "male" in desc:
                        descriptors.append("masculine")
                    elif "female" in desc:
                        descriptors.append("feminine")
                
                if descriptors:
                    system_content += ";".join(descriptors)
                else:
                    system_content += "natural voice"
            else:
                system_content += "natural voice"
            
            system_content += "\n<|scene_desc_end|>"
            
            messages.append(self._Message(
                role="system",
                content=system_content
            ))
        
        # Add the text to generate
        messages.append(self._Message(
            role="user", 
            content=text
        ))
        
        return messages
    
    def _get_language(self, request: OpenAISpeechRequest) -> str:
        """Get language for generation"""
        # Check if language is specified in request
        if hasattr(request, 'extra_params') and request.extra_params and 'language' in request.extra_params:
            return request.extra_params['language']
        
        # Try to detect from text
        try:
            detected_lang = detect_language(request.input)
            if detected_lang:
                return detected_lang
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
        
        # Use default
        return self.default_language
    
    def _load_voice_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load saved voice profiles"""
        if self.voice_profiles_file.exists():
            try:
                with open(self.voice_profiles_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load voice profiles: {e}")
        return {}
    
    def _save_voice_profiles(self):
        """Save voice profiles to disk"""
        try:
            with open(self.voice_profiles_file, 'w') as f:
                json.dump(self.voice_profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save voice profiles: {e}")
    
    def _create_default_profiles(self):
        """Create default voice profiles"""
        default_profiles = {
            "professional_female": {
                "display_name": "Professional Female",
                "language": "en",
                "metadata": {
                    "description": "Clear, professional female voice suitable for presentations",
                    "style": "formal, articulate",
                    "pitch": "medium",
                    "speed": "moderate"
                }
            },
            "warm_female": {
                "display_name": "Warm Female",
                "language": "en",
                "metadata": {
                    "description": "Friendly, warm female voice for conversational content",
                    "style": "casual, friendly",
                    "pitch": "medium-high",
                    "speed": "moderate"
                }
            },
            "storyteller_male": {
                "display_name": "Storyteller Male",
                "language": "en",
                "metadata": {
                    "description": "Engaging male voice perfect for narration and storytelling",
                    "style": "expressive, dynamic",
                    "pitch": "medium",
                    "speed": "variable"
                }
            },
            "deep_male": {
                "display_name": "Deep Male",
                "language": "en",
                "metadata": {
                    "description": "Deep, authoritative male voice for serious content",
                    "style": "formal, commanding",
                    "pitch": "low",
                    "speed": "slow-moderate"
                }
            },
            "energetic_female": {
                "display_name": "Energetic Female",
                "language": "en",
                "metadata": {
                    "description": "Upbeat, energetic female voice for dynamic content",
                    "style": "enthusiastic, lively",
                    "pitch": "high",
                    "speed": "fast-moderate"
                }
            },
            "soft_female": {
                "display_name": "Soft Female",
                "language": "en",
                "metadata": {
                    "description": "Gentle, soothing female voice for relaxing content",
                    "style": "calm, gentle",
                    "pitch": "medium",
                    "speed": "slow"
                }
            }
        }
        
        self.voice_profiles.update(default_profiles)
        self._save_voice_profiles()
        logger.info(f"Created {len(default_profiles)} default voice profiles")
    
    async def create_voice_profile(
        self, profile_name: str, reference_audio_path: str,
        display_name: Optional[str] = None, language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new voice profile from reference audio.
        
        Args:
            profile_name: Unique identifier for the profile
            reference_audio_path: Path to reference audio file
            display_name: Human-readable name
            language: Language code (e.g., 'en', 'es')
            metadata: Additional metadata (description, style, etc.)
            
        Returns:
            Success status
        """
        try:
            # Validate reference audio
            ref_path = Path(reference_audio_path)
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
            
            # Copy reference audio to voice samples directory
            profile_audio_dir = self.voice_samples_dir / profile_name
            profile_audio_dir.mkdir(exist_ok=True)
            
            dest_path = profile_audio_dir / f"reference{ref_path.suffix}"
            shutil.copy2(ref_path, dest_path)
            
            # Create profile
            profile = {
                "display_name": display_name or profile_name,
                "reference_audio": str(dest_path),
                "language": language or self.default_language,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Analyze reference audio for characteristics
            if LIBROSA_AVAILABLE:
                try:
                    y, sr = librosa.load(reference_audio_path)
                    
                    # Extract basic features
                    pitch = librosa.yin(y, fmin=50, fmax=500).mean()
                    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
                    
                    profile["metadata"].update({
                        "analyzed_pitch": float(pitch),
                        "analyzed_tempo": float(tempo),
                        "duration": float(len(y) / sr)
                    })
                except Exception as e:
                    logger.warning(f"Audio analysis failed: {e}")
            
            # Save profile
            self.voice_profiles[profile_name] = profile
            self._save_voice_profiles()
            
            logger.info(f"Created voice profile '{profile_name}' from {reference_audio_path}")
            
            # Update performance metrics
            if self.track_performance:
                self._performance_metrics["voice_cloning_count"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create voice profile: {e}")
            return False
    
    def list_voice_profiles(self) -> List[Dict[str, Any]]:
        """List all available voice profiles"""
        profiles = []
        for name, data in self.voice_profiles.items():
            profile_info = {
                "name": name,
                "display_name": data.get("display_name", name),
                "language": data.get("language", "unknown"),
                "has_reference": bool(data.get("reference_audio")),
                "created_at": data.get("created_at", "unknown"),
                "metadata": data.get("metadata", {})
            }
            profiles.append(profile_info)
        return profiles
    
    def delete_voice_profile(self, profile_name: str) -> bool:
        """Delete a voice profile"""
        if profile_name in self.voice_profiles:
            # Remove reference audio if exists
            profile = self.voice_profiles[profile_name]
            if "reference_audio" in profile:
                try:
                    ref_path = Path(profile["reference_audio"])
                    if ref_path.exists():
                        # Remove entire profile directory
                        profile_dir = ref_path.parent
                        if profile_dir.name == profile_name:
                            shutil.rmtree(profile_dir)
                except Exception as e:
                    logger.warning(f"⚠️  Failed to remove reference audio: {e}")
            
            # Remove from profiles
            del self.voice_profiles[profile_name]
            self._save_voice_profiles()
            
            logger.info(f"Deleted voice profile '{profile_name}'")
            return True
        
        return False
    
    def _update_performance_metrics(self, token_count: int, generation_time: float):
        """Update performance tracking metrics"""
        self._performance_metrics['total_tokens'] += token_count
        self._performance_metrics['total_time'] += generation_time
        self._performance_metrics['generation_count'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'total_generations': self._performance_metrics['generation_count'],
            'total_time': self._performance_metrics['total_time'],
            'total_tokens': self._performance_metrics['total_tokens'],
            'voice_profiles_created': self._performance_metrics['voice_cloning_count']
        }
        
        if stats['total_generations'] > 0:
            stats['average_generation_time'] = stats['total_time'] / stats['total_generations']
            stats['average_tokens_per_generation'] = stats['total_tokens'] / stats['total_generations']
        
        return stats
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get backend capabilities"""
        return {
            "streaming": True,
            "formats": ["mp3", "wav", "opus", "aac", "flac", "pcm"],
            "voices": list(self.voice_profiles.keys()),
            "features": {
                "voice_cloning": self.enable_voice_cloning,
                "multi_speaker": self.enable_multi_speaker,
                "background_music": self.enable_background_music,
                "multilingual": True,
                "custom_voices": True
            },
            "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],  # Add more as needed
            "max_reference_duration": self.max_reference_duration,
            "device": self.device,
            "model": self.model_path
        }
    
    async def close(self):
        """Clean up resources with proper task cancellation"""
        logger.info("HiggsAudioTTSBackend: Starting cleanup...")
        
        # Signal shutdown to stop new operations
        self._shutdown_event.set()
        
        # Cancel all active tasks with timeout
        if self._active_tasks:
            logger.info(f"Cancelling {len(self._active_tasks)} active tasks...")
            
            # Create list to avoid set modification during iteration
            tasks_to_cancel = list(self._active_tasks)
            
            # Cancel all tasks
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
            
            # Wait for cancellation with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=5.0
                )
                logger.info("All tasks cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not cancel within timeout")
                # Force clear the tasks
                self._active_tasks.clear()
        
        # Wait a bit for any ongoing generation to notice shutdown
        await asyncio.sleep(0.1)
        
        # Clean up model
        if self.serve_engine is not None:
            try:
                logger.info("Cleaning up Higgs serve engine...")
                # Force cleanup by clearing reference
                self.serve_engine = None
                
                # Clear GPU memory if using CUDA
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure all CUDA operations complete
                    
                logger.info("Higgs serve engine cleaned up")
            except Exception as e:
                logger.error(f"Error during Higgs engine cleanup: {e}")
        
        # Log final performance stats
        if self.track_performance and self._performance_metrics['generation_count'] > 0:
            stats = self.get_performance_stats()
            logger.info(f"HiggsAudioTTSBackend: Final stats - "
                       f"{stats['total_generations']} generations in {stats['total_time']:.1f}s, "
                       f"{stats['voice_profiles_created']} voice profiles created")
        
        # Clear all references
        self.model_loaded = False
        self._higgs_serve_engine = None
        self._boson_multimodal = None
        
        # Call parent cleanup
        await super().close()
        
        logger.info("HiggsAudioTTSBackend: Cleanup complete")

#
# End of higgs.py
#######################################################################################################################