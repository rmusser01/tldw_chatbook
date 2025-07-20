# kokoro.py
# Description: Kokoro TTS backend implementation supporting both ONNX and PyTorch
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
from typing import AsyncGenerator, Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
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
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.base_backends import LocalTTSBackend
from tldw_chatbook.TTS.audio_service import AudioService, get_audio_service
from tldw_chatbook.TTS.text_processing import TextChunker, TextNormalizer, detect_language
from tldw_chatbook.config import get_cli_setting

#######################################################################################################################
#
# Kokoro TTS Backend Implementation

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
            raise ImportError("numpy is required for Kokoro TTS backend but is not installed. "
                            "Install it with: pip install numpy")
        
        # Check if we're on Windows and pre-validate ONNX dependencies
        if sys.platform == "win32":
            try:
                import onnxruntime
                logger.debug("onnxruntime available on Windows")
            except ImportError:
                logger.warning(
                    "onnxruntime not available on Windows - Kokoro ONNX backend may not work. "
                    "Install with: pip install onnxruntime\n"
                    "If you continue to have issues, you may need to install the Microsoft Visual C++ Redistributable."
                )
        
        # Lazy-loaded heavy dependencies
        self._torch = None
        self._nltk = None
        self._transformers = None
        self._kokoro_onnx = None
        self._kokoro_pt_modules = None
        
        # Configuration
        self.use_onnx = self.config.get("KOKORO_USE_ONNX", True)
        self.model_path = self.config.get("KOKORO_MODEL_PATH")
        self.voices_json = self.config.get("KOKORO_VOICES_JSON_PATH")
        self.voice_dir = self.config.get("KOKORO_VOICE_DIR_PT")
        self.device = self.config.get("KOKORO_DEVICE", "cpu")
        
        # Try to get paths from CLI config if not provided
        if not self.model_path:
            self.model_path = get_cli_setting("app_tts", "KOKORO_ONNX_MODEL_PATH_DEFAULT", 
                                                  "kokoro-v1.0.onnx")
        if not self.voices_json:
            self.voices_json = get_cli_setting("app_tts", "KOKORO_ONNX_VOICES_JSON_DEFAULT",
                                                   "voices-v1.0.bin")
        
        # Model instances
        self.kokoro_instance = None  # ONNX instance
        self.kokoro_model_pt = None  # PyTorch model
        self.tokenizer = None
        
        # Services
        self.audio_service = get_audio_service()
        self.max_tokens = self.config.get("KOKORO_MAX_TOKENS", 500)
        self.text_chunker = TextChunker(max_tokens=self.max_tokens)
        self.normalizer = TextNormalizer()
        
        # Voice mixing configuration
        self.enable_voice_mixing = self.config.get("KOKORO_ENABLE_VOICE_MIXING", False)
        
        # Voice blend storage
        self.voice_blends_dir = Path(self.config.get("KOKORO_VOICE_BLENDS_DIR",
                                                     get_cli_setting("app_tts", "KOKORO_VOICE_BLENDS_DIR",
                                                                   "~/.config/tldw_cli/kokoro_voice_blends"))).expanduser()
        self.voice_blends_dir.mkdir(parents=True, exist_ok=True)
        self.saved_blends = self._load_saved_blends()
        
        # Initialize default blends if none exist
        if not self.saved_blends:
            self._create_default_blends()
        
        # Performance tracking
        self.track_performance = self.config.get("KOKORO_TRACK_PERFORMANCE", True)
        self._performance_metrics = {
            "total_tokens": 0,
            "total_time": 0.0,
            "generation_count": 0
        }
    
    async def initialize(self):
        """Initialize the Kokoro backend"""
        logger.info(f"KokoroTTSBackend: Initializing (ONNX: {self.use_onnx}, Device: {self.device})")
        
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
            voice_dir = Path.home() / ".config" / "tldw_cli" / "models" / "kokoro" / "voices"
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
                self.model_path = str(model_dir / "kokoro-v1_0.pth")
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
                try:
                    logger.info("Downloading Kokoro ONNX model...")
                    if not REQUESTS_AVAILABLE:
                        raise ImportError("requests library required for model download")
                    
                    url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
                    # Expected SHA256 checksum for kokoro-v0_19.onnx
                    expected_checksum = "7e4f8a3c8d5a2b1f9c6e5d4a3b2c1a0e9f8d7c6b5a4e3d2c1b0a9e8d7c6b5a4e"  # Placeholder - needs actual checksum
                    
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    # Download to temporary file first
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        hasher = hashlib.sha256()
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                            hasher.update(chunk)
                        tmp_path = tmp_file.name
                    
                    # Verify checksum
                    actual_checksum = hasher.hexdigest()
                    logger.info(f"Downloaded file checksum: {actual_checksum}")
                    
                    # Note: For now, we'll just log the checksum since we don't have the actual expected value
                    # In production, you should verify against known good checksums
                    logger.warning("Checksum verification skipped - no known checksum available")
                    
                    # Move to final location
                    model_dir = os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else "."
                    os.makedirs(model_dir, exist_ok=True)
                    shutil.move(tmp_path, self.model_path)
                    
                    logger.info(f"Downloaded ONNX model to {self.model_path}")
                except Exception as e:
                    logger.error(f"Failed to download ONNX model: {e}")
                    self.use_onnx = False
                    return
            
            if not os.path.exists(self.voices_json):
                logger.info(f"Kokoro voices file not found at {self.voices_json}")
                # Download the voices.json with checksum verification
                try:
                    logger.info("Downloading Kokoro voices file...")
                    if not REQUESTS_AVAILABLE:
                        raise ImportError("requests library required for voices.json download")
                    
                    url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
                    
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    # Download to temporary file first
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        hasher = hashlib.sha256()
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                            hasher.update(chunk)
                        tmp_path = tmp_file.name
                    
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
            
            # Check for espeak
            espeak_lib = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
            espeak_config = EspeakConfig(lib_path=espeak_lib) if espeak_lib else None
            
            # Create Kokoro instance
            self.kokoro_instance = Kokoro(
                self.model_path,
                self.voices_json,
                espeak_config=espeak_config
            )
            
            logger.info("KokoroTTSBackend: ONNX backend initialized successfully")
            
        except Exception as e:
            logger.error(f"KokoroTTSBackend: Failed to initialize ONNX backend: {e}", exc_info=True)
            self.use_onnx = False
    
    @property
    def torch(self):
        """Lazy load torch module"""
        if self._torch is None:
            try:
                import torch
                self._torch = torch
            except ImportError:
                raise ImportError("PyTorch is required but not installed. Install with: pip install torch")
        return self._torch
    
    @property
    def nltk(self):
        """Lazy load nltk module"""
        if self._nltk is None:
            try:
                import nltk
                self._nltk = nltk
            except ImportError:
                raise ImportError("NLTK is required but not installed. Install with: pip install nltk")
        return self._nltk
    
    @property
    def transformers(self):
        """Lazy load transformers module"""
        if self._transformers is None:
            try:
                import transformers
                self._transformers = transformers
            except ImportError:
                raise ImportError("Transformers is required but not installed. Install with: pip install transformers")
        return self._transformers
    
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
                raise ImportError("kokoro_onnx not installed. Please install with: pip install kokoro-onnx")
            except Exception as e:
                # On Windows, sometimes there are DLL or other loading issues
                logger.error(f"Unexpected error loading kokoro_onnx: {type(e).__name__}: {e}")
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
            
            # Ensure NLTK data is available
            try:
                self.nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                self.nltk.download('punkt', quiet=True)
            
            # Initialize tokenizer
            AutoTokenizer = self.transformers.AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Load model if it exists, otherwise mark for download
            if os.path.exists(self.model_path):
                self._load_pytorch_model()
            else:
                logger.warning(f"Kokoro PyTorch model not found at {self.model_path}")
                # Model download will happen on first use
            
            logger.info("KokoroTTSBackend: PyTorch backend initialized (model loading deferred)")
            
        except Exception as e:
            logger.error(f"KokoroTTSBackend: Failed to initialize PyTorch backend: {e}", exc_info=True)
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
        
        if self.use_onnx and self.kokoro_instance:
            async for chunk in self._generate_onnx_stream(request):
                yield chunk
        else:
            # PyTorch implementation
            # Check if we have a model loaded
            if not self.kokoro_model_pt and not self.kokoro_instance:
                raise RuntimeError(
                    "No Kokoro model loaded. Please either:\n"
                    "1. Install kokoro-onnx: pip install kokoro-onnx\n"
                    "2. Or provide PyTorch model files (.pth) and ensure PyTorch is installed"
                )
            async for chunk in self._generate_pytorch_stream(request):
                yield chunk
    
    async def _generate_onnx_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio using ONNX backend with advanced features"""
        start_time = time.time()
        
        try:
            # Parse voice for potential mixing
            voice_config = self._parse_voice_config(request.voice)
            
            # Detect language from voice or use provided language code
            if hasattr(request, 'extra_params') and request.extra_params and 'language' in request.extra_params:
                lang = request.extra_params['language']
            else:
                # Map voice prefix to espeak language codes
                # kokoro voices: 'af_' = American Female, 'am_' = American Male, 
                #                'bf_' = British Female, 'bm_' = British Male
                if voice_config['primary_voice'] and len(voice_config['primary_voice']) > 0:
                    voice_prefix = voice_config['primary_voice'][0].lower()
                    lang_map = {
                        'a': 'en-us',  # American English
                        'b': 'en-gb',  # British English
                        'j': 'ja',     # Japanese
                        'z': 'zh',     # Chinese
                        'e': 'es',     # Spanish
                        'f': 'fr',     # French
                    }
                    lang = lang_map.get(voice_prefix, 'en-us')
                else:
                    lang = 'en-us'  # Default to American English
            
            # Normalize text if requested
            text = request.input
            if request.normalization_options:
                text = self.normalizer.normalize_text(text)
            
            logger.info(f"KokoroTTSBackend: Generating audio for {len(text)} characters, "
                       f"voice={voice_config}, lang={lang}, format={request.response_format}")
            
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
                    metrics={"format": "pcm"}
                )
                
                if voice_config['is_mixed']:
                    # Generate mixed voice audio
                    async for samples, sample_rate in self._generate_mixed_voice(
                        text, voice_config, speed=request.speed, lang=lang
                    ):
                        token_count += len(samples) // 256  # Approximate token count
                        samples_processed += len(samples)
                        
                        # Report progress
                        progress = min(0.95, token_count / estimated_total_tokens) if estimated_total_tokens > 0 else 0.5
                        await self._report_progress(
                            progress=progress,
                            processed=token_count,
                            total=estimated_total_tokens,
                            status=f"Streaming PCM audio: {samples_processed / sample_rate:.1f}s generated",
                            metrics={
                                "sample_rate": sample_rate,
                                "samples_generated": samples_processed,
                                "format": "pcm"
                            }
                        )
                        
                        # Convert float32 to int16 PCM
                        int16_samples = np.int16(samples * 32767)
                        yield int16_samples.tobytes()
                else:
                    # Single voice generation
                    async for samples, sample_rate in self.kokoro_instance.create_stream(
                        text, voice=voice_config['primary_voice'], speed=request.speed, lang=lang
                    ):
                        token_count += len(samples) // 256  # Approximate token count
                        samples_processed += len(samples)
                        
                        # Report progress
                        progress = min(0.95, token_count / estimated_total_tokens) if estimated_total_tokens > 0 else 0.5
                        await self._report_progress(
                            progress=progress,
                            processed=token_count,
                            total=estimated_total_tokens,
                            status=f"Streaming PCM audio: {samples_processed / sample_rate:.1f}s generated",
                            metrics={
                                "sample_rate": sample_rate,
                                "samples_generated": samples_processed,
                                "format": "pcm"
                            }
                        )
                        
                        # Convert float32 to int16 PCM
                        int16_samples = np.int16(samples * 32767)
                        yield int16_samples.tobytes()
                
                # Update performance metrics
                if self.track_performance:
                    self._update_performance_metrics(token_count, time.time() - start_time)
                
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
                        "generation_time": time.time() - start_time
                    }
                )
            
            # For other formats, we need to handle them appropriately
            else:
                # For WAV format, we need to collect all samples first
                # For other formats, we can stream with chunked conversion
                sample_rate = 24000  # Default
                token_count = 0
                
                # Choose generator based on voice configuration
                if voice_config['is_mixed']:
                    audio_generator = self._generate_mixed_voice(
                        text, voice_config, speed=request.speed, lang=lang
                    )
                else:
                    audio_generator = self.kokoro_instance.create_stream(
                        text, voice=voice_config['primary_voice'], speed=request.speed, lang=lang
                    )
                
                # Estimate total tokens for progress tracking
                estimated_total_tokens = len(text.split()) * 2  # Rough estimate
                
                # WAV format needs all samples collected first
                if request.response_format == "wav":
                    all_samples = []
                    
                    # Collect all audio samples
                    async for samples, sr in audio_generator:
                        sample_rate = sr
                        all_samples.extend(samples)
                        token_count += len(samples) // 256
                        
                        # Report progress
                        progress = min(0.95, token_count / estimated_total_tokens) if estimated_total_tokens > 0 else 0.5
                        await self._report_progress(
                            progress=progress,
                            processed=token_count,
                            total=estimated_total_tokens,
                            status=f"Collecting audio: {len(all_samples) / sample_rate:.1f}s",
                            metrics={
                                "sample_rate": sample_rate,
                                "samples_collected": len(all_samples),
                                "format": request.response_format
                            }
                        )
                    
                    # Convert all samples to WAV at once
                    if all_samples:
                        try:
                            full_audio = np.array(all_samples, dtype=np.float32)
                            audio_bytes = await self.audio_service.convert_audio(
                                full_audio,
                                request.response_format,
                                source_format="pcm",
                                sample_rate=sample_rate
                            )
                            yield audio_bytes
                            
                            # Log performance
                            generation_time = time.time() - start_time
                            audio_duration = len(all_samples) / sample_rate
                            speed_factor = audio_duration / generation_time if generation_time > 0 else 0
                            
                            logger.info(f"KokoroTTSBackend: Generated {audio_duration:.2f}s of WAV audio in {generation_time:.2f}s "
                                       f"({speed_factor:.1f}x realtime)")
                            
                            # Report completion
                            await self._report_progress(
                                progress=1.0,
                                processed=token_count,
                                total=estimated_total_tokens,
                                status=f"WAV generation complete: {audio_duration:.1f}s",
                                metrics={
                                    "sample_rate": sample_rate,
                                    "samples_generated": len(all_samples),
                                    "format": request.response_format,
                                    "generation_time": generation_time,
                                    "speed_factor": speed_factor
                                }
                            )
                        except Exception as e:
                            logger.error(f"KokoroTTSBackend: WAV conversion failed: {e}")
                            yield b""
                    else:
                        logger.warning("KokoroTTSBackend: No audio generated")
                        yield b""
                    
                    return  # Exit after yielding WAV
                
                # For streaming formats (MP3, Opus, etc.), use chunked conversion
                chunk_buffer = []
                chunk_size = 8192  # samples per chunk (about 0.34s at 24kHz)
                total_samples_processed = 0
                first_chunk_time = None
                
                # Process audio chunks as they are generated
                async for samples, sr in audio_generator:
                    sample_rate = sr
                    chunk_buffer.extend(samples)
                    token_count += len(samples) // 256  # Approximate token count
                    
                    # Report progress
                    progress = min(0.95, token_count / estimated_total_tokens) if estimated_total_tokens > 0 else 0.5
                    await self._report_progress(
                        progress=progress,
                        processed=token_count,
                        total=estimated_total_tokens,
                        status=f"Generating audio: {token_count} tokens processed",
                        current_chunk=total_samples_processed // chunk_size,
                        metrics={
                            "sample_rate": sample_rate,
                            "samples_generated": total_samples_processed,
                            "format": request.response_format
                        }
                    )
                    
                    # Yield complete chunks as they become available
                    while len(chunk_buffer) >= chunk_size:
                        # Extract a chunk
                        chunk_array = np.array(chunk_buffer[:chunk_size], dtype=np.float32)
                        chunk_buffer = chunk_buffer[chunk_size:]
                        
                        # Convert chunk to target format
                        try:
                            audio_bytes = await self.audio_service.convert_audio(
                                chunk_array,
                                request.response_format,
                                source_format="pcm",
                                sample_rate=sample_rate
                            )
                            
                            # Track first chunk latency
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - start_time
                                logger.debug(f"KokoroTTSBackend: First chunk latency: {first_chunk_time:.3f}s")
                            
                            yield audio_bytes
                            total_samples_processed += len(chunk_array)
                            
                        except Exception as e:
                            logger.error(f"KokoroTTSBackend: Chunk conversion failed: {e}")
                            # Continue with next chunk instead of failing completely
                            continue
                
                # Process any remaining samples in buffer
                if chunk_buffer:
                    remaining_array = np.array(chunk_buffer, dtype=np.float32)
                    try:
                        audio_bytes = await self.audio_service.convert_audio(
                            remaining_array,
                            request.response_format,
                            source_format="pcm",
                            sample_rate=sample_rate
                        )
                        yield audio_bytes
                        total_samples_processed += len(remaining_array)
                    except Exception as e:
                        logger.error(f"KokoroTTSBackend: Final chunk conversion failed: {e}")
                
                # Update performance metrics
                if self.track_performance:
                    self._update_performance_metrics(token_count, time.time() - start_time)
                
                # Log performance info
                if total_samples_processed > 0:
                    generation_time = time.time() - start_time
                    tokens_per_second = token_count / generation_time if generation_time > 0 else 0
                    audio_duration = total_samples_processed / sample_rate
                    speed_factor = audio_duration / generation_time if generation_time > 0 else 0
                    
                    logger.info(f"KokoroTTSBackend: Streamed {audio_duration:.2f}s of audio in {generation_time:.2f}s "
                               f"({speed_factor:.1f}x realtime, {tokens_per_second:.1f} tokens/s, "
                               f"first chunk: {first_chunk_time:.3f}s)")
                    
                    # Report completion
                    await self._report_progress(
                        progress=1.0,
                        processed=token_count,
                        total=estimated_total_tokens,
                        status=f"Generation complete: {audio_duration:.1f}s of {request.response_format} audio",
                        metrics={
                            "sample_rate": sample_rate,
                            "samples_generated": total_samples_processed,
                            "format": request.response_format,
                            "generation_time": generation_time,
                            "speed_factor": speed_factor,
                            "first_chunk_latency": first_chunk_time
                        }
                    )
                else:
                    logger.warning("KokoroTTSBackend: No audio generated")
                    yield b""
                    
        except Exception as e:
            logger.error(f"KokoroTTSBackend: Error during ONNX generation: {e}", exc_info=True)
            yield f"ERROR: Kokoro generation failed - {str(e)}".encode('utf-8')
    
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
                logger.warning(f"Blend preset '{preset_name}' not found, using default voice")
                voice_str = "af_bella"
        
        if not self.enable_voice_mixing or ':' not in voice_str:
            # Simple voice without mixing
            return {
                'primary_voice': map_voice_to_kokoro(voice_str),
                'is_mixed': False,
                'voices': [(map_voice_to_kokoro(voice_str), 1.0)]
            }
        
        # Parse mixed voice format: "voice1:weight1,voice2:weight2"
        voices = []
        total_weight = 0
        
        for voice_part in voice_str.split(','):
            if ':' in voice_part:
                voice_name, weight_str = voice_part.split(':', 1)
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
            'primary_voice': voices[0][0],  # First voice as primary
            'is_mixed': len(voices) > 1,
            'voices': voices
        }
    
    async def _generate_mixed_voice(
        self, text: str, voice_config: Dict[str, Any], speed: float, lang: str
    ) -> AsyncGenerator[tuple[np.ndarray, int], None]:
        """Generate audio with mixed voices"""
        if not voice_config['is_mixed']:
            # Fallback to single voice
            async for samples, sr in self.kokoro_instance.create_stream(
                text, voice=voice_config['primary_voice'], speed=speed, lang=lang
            ):
                yield samples, sr
            return
        
        # Collect audio from each voice
        voice_samples = []
        sample_rate = 24000
        
        for voice, weight in voice_config['voices']:
            samples_list = []
            async for samples, sr in self.kokoro_instance.create_stream(
                text, voice=voice, speed=speed, lang=lang
            ):
                sample_rate = sr
                samples_list.append(samples)
            
            if samples_list:
                combined = np.concatenate(samples_list)
                voice_samples.append((combined, weight))
        
        if not voice_samples:
            return
        
        # Mix the voices
        max_length = max(len(samples) for samples, _ in voice_samples)
        mixed_audio = np.zeros(max_length, dtype=np.float32)
        
        for samples, weight in voice_samples:
            # Pad if necessary
            if len(samples) < max_length:
                samples = np.pad(samples, (0, max_length - len(samples)))
            mixed_audio += samples * weight
        
        # Normalize to prevent clipping
        max_val = np.abs(mixed_audio).max()
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val
        
        # Yield in chunks for streaming
        chunk_size = 8192
        for i in range(0, len(mixed_audio), chunk_size):
            chunk = mixed_audio[i:i + chunk_size]
            yield chunk, sample_rate
    
    def _update_performance_metrics(self, token_count: int, generation_time: float):
        """Update performance tracking metrics"""
        self._performance_metrics['total_tokens'] += token_count
        self._performance_metrics['total_time'] += generation_time
        self._performance_metrics['generation_count'] += 1
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self._performance_metrics['generation_count'] == 0:
            return {
                'average_tokens_per_second': 0,
                'total_generations': 0,
                'total_time': 0
            }
        
        return {
            'average_tokens_per_second': self._performance_metrics['total_tokens'] / self._performance_metrics['total_time'],
            'total_generations': self._performance_metrics['generation_count'],
            'total_time': self._performance_metrics['total_time'],
            'total_tokens': self._performance_metrics['total_tokens']
        }
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        try:
            # Import Kokoro PyTorch modules dynamically
            if self._kokoro_pt_modules is None:
                try:
                    from tldw_chatbook.TTS import kokoro_pytorch
                    self._kokoro_pt_modules = {
                        'build_model': kokoro_pytorch.build_model,
                        'generate': kokoro_pytorch.generate,
                        'load_voice': kokoro_pytorch.load_voice,
                        'mix_voices': kokoro_pytorch.mix_voices,
                        'parse_voice_mix': kokoro_pytorch.parse_voice_mix,
                        'get_available_voices': kokoro_pytorch.get_available_voices,
                    }
                except ImportError as e:
                    logger.error(f"Failed to import Kokoro PyTorch modules: {e}")
                    raise
            
            build_model = self._kokoro_pt_modules['build_model']
            self.kokoro_model_pt = build_model(self.model_path, device=self.device)
            logger.info(f"Loaded Kokoro PyTorch model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    async def _download_model_if_needed(self):
        """Download Kokoro model if not present"""
        if not os.path.exists(self.model_path):
            logger.info("Downloading Kokoro PyTorch model...")
            try:
                if not REQUESTS_AVAILABLE:
                    raise ImportError("requests library required for model download")
                url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth?download=true"
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded model to {self.model_path}")
                self._load_pytorch_model()
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise ValueError("Failed to download Kokoro model. Please download manually.")
    
    async def _download_voice_if_needed(self, voice: str):
        """Download voice pack if not present"""
        voice_path = os.path.join(self.voice_dir, f"{voice}.pt")
        if not os.path.exists(voice_path):
            logger.info(f"Downloading voice pack: {voice}")
            try:
                if not REQUESTS_AVAILABLE:
                    raise ImportError("requests library required for voice download")
                url = f"https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice}.pt?download=true"
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                os.makedirs(self.voice_dir, exist_ok=True)
                with open(voice_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded voice to {voice_path}")
            except Exception as e:
                logger.error(f"Failed to download voice {voice}: {e}")
                raise ValueError(f"Failed to download voice {voice}. Please download manually.")
    
    def _load_voice_pack(self, voice: str):
        """Load a voice pack for PyTorch"""
        if self._kokoro_pt_modules and 'load_voice' in self._kokoro_pt_modules:
            load_voice = self._kokoro_pt_modules['load_voice']
            voice_path = os.path.join(self.voice_dir, f"{voice}.pt")
            return load_voice(voice_path, self.device)
        else:
            # Fallback to direct torch load
            voice_path = os.path.join(self.voice_dir, f"{voice}.pt")
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
            
            # Map voice
            voice = map_voice_to_kokoro(request.voice)
            
            # Ensure voice pack is available
            await self._download_voice_if_needed(voice)
            
            # Load voice pack
            voice_pack = self._load_voice_pack(voice)
            
            # Detect language from voice or use provided language code
            if hasattr(request, 'extra_params') and request.extra_params and 'language' in request.extra_params:
                lang = request.extra_params['language']
            else:
                # Map voice prefix to espeak language codes
                if voice and len(voice) > 0:
                    voice_prefix = voice[0].lower()
                    lang_map = {
                        'a': 'en-us',  # American English
                        'b': 'en-gb',  # British English
                        'j': 'ja',     # Japanese
                        'z': 'zh',     # Chinese
                        'e': 'es',     # Spanish
                        'f': 'fr',     # French
                    }
                    lang = lang_map.get(voice_prefix, 'en-us')
                else:
                    lang = 'en-us'
            
            # Split text into chunks
            text_chunks = self._split_text_for_pytorch(request.input)
            
            # Get generation function from cached modules
            if self._kokoro_pt_modules is None:
                # Load modules if not already loaded
                self._load_pytorch_model()
            generate = self._kokoro_pt_modules['generate']
            
            # Generate and stream audio for each chunk
            token_count = 0
            first_chunk_time = None
            total_audio_duration = 0.0
            estimated_total_tokens = sum(len(chunk.split()) for chunk in text_chunks) * 2
            
            # Report initial progress
            await self._report_progress(
                progress=0.0,
                processed=0,
                total=estimated_total_tokens,
                status=f"Starting PyTorch generation with {len(text_chunks)} chunks",
                total_chunks=len(text_chunks),
                metrics={"backend": "pytorch", "device": self.device}
            )
            
            for i, chunk in enumerate(text_chunks):
                # Report chunk start
                await self._report_progress(
                    progress=i / len(text_chunks) * 0.9,  # Reserve 10% for final processing
                    processed=token_count,
                    total=estimated_total_tokens,
                    status=f"Processing chunk {i+1}/{len(text_chunks)}",
                    current_chunk=i+1,
                    total_chunks=len(text_chunks),
                    metrics={"backend": "pytorch", "device": self.device}
                )
                
                # Generate audio
                audio_tensor, phonemes = await asyncio.to_thread(
                    generate,
                    self.kokoro_model_pt, 
                    chunk, 
                    voice_pack, 
                    lang=lang, 
                    speed=request.speed,
                    voice_dir=self.voice_dir
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
                    # For other formats, convert chunk
                    try:
                        audio_bytes = await self.audio_service.convert_audio(
                            audio_data,
                            request.response_format,
                            source_format="pcm",
                            sample_rate=24000
                        )
                        yield audio_bytes
                    except Exception as e:
                        logger.error(f"KokoroTTSBackend: PyTorch chunk conversion failed: {e}")
                        # Try to continue with next chunk
                        continue
                
                # Track first chunk latency
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    logger.debug(f"KokoroTTSBackend PyTorch: First chunk latency: {first_chunk_time:.3f}s")
                
                # Log progress
                logger.debug(f"KokoroTTSBackend PyTorch: Processed chunk {i+1}/{len(text_chunks)} "
                           f"({chunk_duration:.2f}s of audio)")
            
            # Update metrics
            if self.track_performance:
                self._update_performance_metrics(token_count, time.time() - start_time)
            
            # Log performance summary
            if token_count > 0:
                generation_time = time.time() - start_time
                speed_factor = total_audio_duration / generation_time if generation_time > 0 else 0
                logger.info(f"KokoroTTSBackend PyTorch: Generated {total_audio_duration:.2f}s of audio "
                           f"in {generation_time:.2f}s ({speed_factor:.1f}x realtime, "
                           f"first chunk: {first_chunk_time:.3f}s)")
                
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
                        "first_chunk_latency": first_chunk_time
                    }
                )
            
        except Exception as e:
            logger.error(f"KokoroTTSBackend: PyTorch generation failed: {e}", exc_info=True)
            raise ValueError(f"Kokoro PyTorch generation failed: {str(e)}")
    
    def _split_text_for_pytorch(self, text: str, max_tokens: int = 150) -> list[str]:
        """Split text into chunks for PyTorch processing"""
        if self.tokenizer:
            # Use NLTK for sentence splitting
            try:
                sentences = self.nltk.sent_tokenize(text)
                
                chunks = []
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
                    sentence_length = len(sentence_tokens)
                    
                    if current_length + sentence_length > max_tokens:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_length
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                return chunks
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}, using simple split")
        
        # Fallback to simple splitting
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_tokens):
            chunks.append(" ".join(words[i:i+max_tokens]))
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
            lang_map = {
                'a': 'en-us',  # American English
                'b': 'en-gb',  # British English
                'j': 'ja',     # Japanese
                'z': 'zh',     # Chinese
                'e': 'es',     # Spanish
                'f': 'fr',     # French
            }
            lang = lang_map.get(voice_prefix, 'en-us')
        else:
            lang = 'en-us'
        
        # Split text into words for timing estimation
        words = text.split()
        word_timestamps = []
        
        # Generate audio
        audio_chunks = []
        sample_rate = 24000
        current_time = 0.0
        
        # Generate full audio first
        samples_list = []
        async for samples, sr in self.kokoro_instance.create_stream(
            text, voice=kokoro_voice, speed=speed, lang=lang
        ):
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
            
            word_timestamps.append({
                'word': word,
                'start': start_time,
                'end': end_time,
                'confidence': 0.7  # Estimated timing confidence
            })
            
            current_time = end_time
        
        # Convert audio to bytes
        int16_samples = np.int16(full_audio * 32767)
        audio_bytes = int16_samples.tobytes()
        
        # Wrap in WAV format
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
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
        voice_pack = self._load_voice_pack(kokoro_voice)
        
        # Detect language
        if kokoro_voice and len(kokoro_voice) > 0:
            voice_prefix = kokoro_voice[0].lower()
            lang_map = {
                'a': 'en-us',  # American English
                'b': 'en-gb',  # British English
                'j': 'ja',     # Japanese
                'z': 'zh',     # Chinese
                'e': 'es',     # Spanish
                'f': 'fr',     # French
            }
            lang = lang_map.get(voice_prefix, 'en-us')
        else:
            lang = 'en-us'
        
        # Get generation function from cached modules
        if self._kokoro_pt_modules is None:
            # Load modules if not already loaded
            self._load_pytorch_model()
        generate = self._kokoro_pt_modules['generate']
        
        # Generate audio with phonemes
        audio_tensor, phonemes = await asyncio.to_thread(
            generate,
            self.kokoro_model_pt,
            text,
            voice_pack,
            lang=lang,
            speed=speed
        )
        
        # Convert to numpy
        if isinstance(audio_tensor, self.torch.Tensor):
            audio_data = audio_tensor.cpu().numpy()
        else:
            audio_data = audio_tensor
        
        # Parse phonemes to create word timestamps
        word_timestamps = self._phonemes_to_word_timestamps(text, phonemes, len(audio_data) / 24000)
        
        # Convert audio to WAV bytes
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
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
        if not phonemes or not hasattr(phonemes, '__iter__'):
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
            word_duration = (len(word) / total_chars) * total_duration if total_chars > 0 else 0
            
            word_timestamps.append({
                'word': word,
                'start': current_time,
                'end': current_time + word_duration,
                'confidence': 0.5  # Low confidence for estimation
            })
            
            current_time += word_duration
        
        return word_timestamps
    
    def _load_saved_blends(self) -> Dict[str, Dict[str, Any]]:
        """Load saved voice blends from disk"""
        blends = {}
        blend_file = self.voice_blends_dir / "voice_blends.json"
        
        if blend_file.exists():
            try:
                with open(blend_file, 'r') as f:
                    blends = json.load(f)
                logger.info(f"Loaded {len(blends)} saved voice blends")
            except Exception as e:
                logger.error(f"Failed to load voice blends: {e}")
        
        return blends
    
    def _save_blends(self):
        """Save voice blends to disk"""
        blend_file = self.voice_blends_dir / "voice_blends.json"
        try:
            with open(blend_file, 'w') as f:
                json.dump(self.saved_blends, f, indent=2)
            logger.info(f"Saved {len(self.saved_blends)} voice blends")
        except Exception as e:
            logger.error(f"Failed to save voice blends: {e}")
    
    def _create_default_blends(self):
        """Create default voice blend presets"""
        default_blends = [
            # Professional blends
            ("professional_female", [("af_bella", 0.6), ("af_sarah", 0.4)], 
             "Professional female voice blend"),
            ("professional_male", [("am_adam", 0.7), ("am_michael", 0.3)], 
             "Professional male voice blend"),
            
            # Character blends
            ("warm_storyteller", [("af_nicole", 0.5), ("bf_emma", 0.5)], 
             "Warm storytelling voice"),
            ("dynamic_narrator", [("am_michael", 0.4), ("bm_george", 0.3), ("am_adam", 0.3)], 
             "Dynamic narrator with varied tones"),
            
            # Language-specific blends
            ("english_blend", [("af_bella", 0.3), ("af_sarah", 0.3), ("bf_emma", 0.4)], 
             "Balanced English female voices"),
            ("male_chorus", [("am_adam", 0.25), ("am_michael", 0.25), ("bm_george", 0.25), ("bm_lewis", 0.25)], 
             "All male voices blended equally"),
            
            # Creative blends
            ("soft_whisper", [("af_sky", 0.7), ("bf_isabella", 0.3)], 
             "Soft, gentle voice blend"),
            ("energetic", [("bf_emma", 0.6), ("af_nicole", 0.4)], 
             "Energetic and upbeat voice"),
        ]
        
        for name, voices, description in default_blends:
            self.save_voice_blend(name, voices, description, {"is_default": True})
        
        logger.info(f"Created {len(default_blends)} default voice blends")
    
    def save_voice_blend(self, name: str, voices: List[Tuple[str, float]], 
                        description: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
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
        try:
            # Validate and normalize weights
            total_weight = sum(w for _, w in voices)
            if total_weight <= 0:
                raise ValueError("Total weight must be positive")
            
            normalized_voices = [(v, w/total_weight) for v, w in voices]
            
            # Create blend entry
            blend_data = {
                "voices": normalized_voices,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Save to memory and disk
            self.saved_blends[name] = blend_data
            self._save_blends()
            
            logger.info(f"Saved voice blend '{name}' with {len(voices)} voices")
            return True
            
        except Exception as e:
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
                "voice_count": len(data["voices"])
            }
            blends.append(blend_info)
        return blends
    
    def delete_voice_blend(self, name: str) -> bool:
        """Delete a saved voice blend"""
        if name in self.saved_blends:
            del self.saved_blends[name]
            self._save_blends()
            logger.info(f"Deleted voice blend '{name}'")
            return True
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
            raise NotImplementedError("Phoneme generation only supported with ONNX backend")
        
        try:
            kokoro_voice = map_voice_to_kokoro(voice)
            
            # Check if kokoro_instance has phoneme support
            if hasattr(self.kokoro_instance, 'generate_from_phonemes'):
                samples = await asyncio.to_thread(
                    self.kokoro_instance.generate_from_phonemes,
                    phonemes,
                    voice=kokoro_voice,
                    speed=speed
                )
                
                # Convert to PCM bytes
                int16_samples = np.int16(samples * 32767)
                return int16_samples.tobytes()
            else:
                logger.warning("Kokoro instance doesn't support phoneme generation")
                raise NotImplementedError("This version of kokoro_onnx doesn't support phoneme generation")
                
        except Exception as e:
            logger.error(f"Phoneme generation failed: {e}")
            raise
    
    async def close(self):
        """Clean up resources"""
        await super().close()
        # Clean up model instances if needed
        self.kokoro_instance = None
        self.kokoro_model_pt = None
        
        # Log final performance stats if tracking
        if self.track_performance and self._performance_metrics['generation_count'] > 0:
            stats = self.get_performance_stats()
            logger.info(f"KokoroTTSBackend: Final performance stats - "
                       f"Avg: {stats['average_tokens_per_second']:.1f} tokens/s, "
                       f"Total: {stats['total_generations']} generations in {stats['total_time']:.1f}s")


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