# transcription_service.py
"""
Unified transcription service for tldw_chatbook.
Supports multiple transcription backends including faster-whisper, Qwen2Audio, etc.
"""

import os
import subprocess
import tempfile
import time
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
import json
from loguru import logger

# Fix for multiprocessing issues in some environments
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TQDM_DISABLE'] = '1'

import numpy as np

# Local imports  
from ..config import get_cli_setting
from ..Utils.text import sanitize_filename
from contextlib import contextmanager

# Optional imports with graceful degradation
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not available. Install with: pip install faster-whisper")

try:
    if sys.platform == 'darwin':
        from lightning_whisper_mlx import LightningWhisperMLX
        LIGHTNING_WHISPER_AVAILABLE = True
    else:
        LIGHTNING_WHISPER_AVAILABLE = False
except ImportError:
    LIGHTNING_WHISPER_AVAILABLE = False
    if sys.platform == 'darwin':
        logger.warning("lightning-whisper-mlx not available. Install with: pip install lightning-whisper-mlx")

try:
    if sys.platform == 'darwin':
        from parakeet_mlx import from_pretrained as parakeet_from_pretrained
        PARAKEET_MLX_AVAILABLE = True
        logger.info("parakeet-mlx is available for real-time ASR on Apple Silicon")
    else:
        PARAKEET_MLX_AVAILABLE = False
except ImportError:
    PARAKEET_MLX_AVAILABLE = False
    if sys.platform == 'darwin':
        logger.warning("parakeet-mlx not available. Install with: pip install parakeet-mlx")

try:
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    QWEN2AUDIO_AVAILABLE = True
except ImportError:
    QWEN2AUDIO_AVAILABLE = False
    logger.warning("Qwen2Audio not available. Install transformers and torch for Qwen2Audio support.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available. Install with: pip install soundfile")

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some audio processing features may be limited.")

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logger.warning("NeMo toolkit not available. Install with: pip install nemo-toolkit[asr]")

# Using loguru logger imported above


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


class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass


class ConversionError(TranscriptionError):
    """Error during audio format conversion."""
    pass


class TranscriptionService:
    """Unified service for audio transcription with multiple backend support."""
    
    def __init__(self):
        """Initialize the transcription service."""
        logger.info("Initializing TranscriptionService...")
        
        # Determine default provider based on platform and availability
        default_provider_fallback = 'faster-whisper'
        if sys.platform == 'darwin':
            if PARAKEET_MLX_AVAILABLE:
                default_provider_fallback = 'parakeet-mlx'
            elif LIGHTNING_WHISPER_AVAILABLE:
                default_provider_fallback = 'lightning-whisper-mlx'
        
        # Set appropriate default model based on provider
        default_model = 'base'
        if default_provider_fallback == 'parakeet-mlx':
            default_model = 'mlx-community/parakeet-tdt-0.6b-v2'
        
        self.config = {
            'default_provider': get_cli_setting('transcription.default_provider', default_provider_fallback) or default_provider_fallback,
            'default_model': get_cli_setting('transcription.default_model', default_model) or default_model,
            'default_language': get_cli_setting('transcription.default_language', 'en') or 'en',
            'default_source_language': get_cli_setting('transcription.default_source_language', '') or '',
            'default_target_language': get_cli_setting('transcription.default_target_language', '') or '',
            'device': get_cli_setting('transcription.device', 'cpu') or 'cpu',
            'compute_type': get_cli_setting('transcription.compute_type', 'int8') or 'int8',
            'chunk_length_seconds': get_cli_setting('transcription.chunk_length_seconds', 40.0) or 40.0,
        }
        
        logger.debug(f"Transcription service configuration: {self.config}")
        
        # Model cache and thread safety
        self._model_cache = {}
        self._model_cache_lock = threading.RLock()
        logger.debug("Initialized empty model cache for faster-whisper models with thread safety")
        
        # Qwen2Audio models (lazy loaded)
        self._qwen_processor = None
        self._qwen_model = None
        logger.debug("Qwen2Audio models will be lazy-loaded on first use")
        
        # Parakeet/NeMo models (lazy loaded)
        self._parakeet_model = None
        logger.debug("Parakeet model will be lazy-loaded on first use")
        
        # Canary model (lazy loaded)
        self._canary_model = None
        self._canary_decoding = None
        logger.debug("Canary model will be lazy-loaded on first use")
        
        # Lightning Whisper MLX model (lazy loaded, macOS only)
        self._lightning_whisper_model = None
        self._lightning_whisper_config = {
            'batch_size': get_cli_setting('transcription.lightning_batch_size', 12) or 12,
            'quant': get_cli_setting('transcription.lightning_quant', None),  # None, '4bit', '8bit'
        }
        if LIGHTNING_WHISPER_AVAILABLE:
            logger.debug("Lightning Whisper MLX model will be lazy-loaded on first use")
        
        # Parakeet MLX model (lazy loaded, macOS only)
        self._parakeet_mlx_model = None
        self._parakeet_mlx_model_lock = threading.RLock()
        self._parakeet_mlx_config = {
            'model': get_cli_setting('transcription.parakeet_model', 'mlx-community/parakeet-tdt-0.6b-v2') or 'mlx-community/parakeet-tdt-0.6b-v2',
            'precision': get_cli_setting('transcription.parakeet_precision', 'bf16') or 'bf16',
            'attention_type': get_cli_setting('transcription.parakeet_attention', 'local') or 'local',
            'chunk_duration': get_cli_setting('transcription.parakeet_chunk_duration', 120.0) or 120.0,
            'overlap_duration': get_cli_setting('transcription.parakeet_overlap_duration', 0.5) or 0.5,  # 0.5 seconds to avoid cutting words mid-sentence
            'auto_chunk_threshold': get_cli_setting('transcription.parakeet_auto_chunk_threshold', 600.0) or 600.0,
        }
        if PARAKEET_MLX_AVAILABLE:
            logger.debug("Parakeet MLX model will be lazy-loaded on first use")
        
        # Log available providers on initialization
        available_providers = self.get_available_providers()
        logger.info(f"TranscriptionService initialized with {len(available_providers)} available providers")
        
    def transcribe(
        self,
        audio_path: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        source_lang: Optional[str] = None,  # Explicit source language
        target_lang: Optional[str] = None,  # Translation target language
        vad_filter: bool = False,
        diarize: bool = False,
        progress_callback: Optional[Callable[[float, str, Optional[Dict]], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using specified provider.
        
        Args:
            audio_path: Path to audio file
            provider: Transcription provider ('faster-whisper', 'qwen2audio', 'parakeet', 'canary')
            model: Model name/size
            language: Language code (for backward compatibility)
            source_lang: Explicit source language for transcription
            target_lang: Target language for translation (if supported)
            vad_filter: Apply voice activity detection
            diarize: Perform speaker diarization (placeholder)
            progress_callback: Optional callback for progress updates (progress: 0-100, status: str, data: dict)
            
        Returns:
            Dict with 'text' and 'segments' keys
        """
        provider = provider or self.config['default_provider']
        
        # Handle provider-specific default models
        if not model:
            if provider == 'parakeet-mlx':
                model = 'mlx-community/parakeet-tdt-0.6b-v2'
            elif provider == 'qwen2audio':
                model = 'Qwen2-Audio-7B-Instruct'
            else:
                model = self.config['default_model']
        # Handle source language - prefer explicit source_lang over language param
        source_lang = source_lang or self.config['default_source_language'] or language or self.config['default_language']
        # Handle target language
        target_lang = target_lang or self.config['default_target_language'] or None
        # For backward compatibility, set language to source_lang if not specified
        language = language or source_lang
        
        logger.info(f"Starting transcription with provider={provider}, model={model}, source_lang={source_lang}, target_lang={target_lang}")
        logger.debug(f"Audio path: {audio_path}, VAD filter: {vad_filter}, Diarize: {diarize}")
        logger.debug(f"Additional kwargs: {kwargs}")
        
        # Convert to WAV if needed
        logger.debug(f"Checking if audio format conversion needed for: {audio_path}")
        wav_path = self._ensure_wav_format(audio_path)
        if wav_path != audio_path:
            logger.info(f"Converted audio to WAV format: {audio_path} -> {wav_path}")
        
        try:
            logger.info(f"Starting transcription with {provider} provider")
            transcription_start_time = time.time()
            
            if provider == 'parakeet-mlx':
                if not PARAKEET_MLX_AVAILABLE:
                    if sys.platform != 'darwin':
                        raise ValueError(f"parakeet-mlx is only available on macOS (Apple Silicon)")
                    else:
                        raise ValueError(f"parakeet-mlx is not installed. Install with: pip install parakeet-mlx")
                
                try:
                    result = self._transcribe_with_parakeet_mlx(
                        wav_path, model, source_lang,
                        progress_callback=progress_callback, **kwargs
                    )
                except TranscriptionError as e:
                    # Check if this is a memory error and fallback is enabled
                    if ("memory allocation error" in str(e) and 
                        kwargs.get('fallback_on_memory_error', True)):
                        
                        logger.warning(f"Parakeet MLX failed with memory error, attempting fallback")
                        
                        # Try fallback providers in order
                        fallback_providers = ['faster-whisper', 'lightning-whisper-mlx']
                        available_providers = self.get_available_providers()
                        
                        for fallback_provider in fallback_providers:
                            if fallback_provider in available_providers:
                                logger.info(f"Falling back to {fallback_provider} provider")
                                
                                # Recursive call with different provider
                                return self.transcribe(
                                    audio_path=audio_path,
                                    provider=fallback_provider,
                                    model=None,  # Use default model for fallback provider
                                    language=language,
                                    source_lang=source_lang,
                                    target_lang=target_lang,
                                    vad_filter=vad_filter,
                                    diarize=diarize,
                                    progress_callback=progress_callback,
                                    **{k: v for k, v in kwargs.items() if k != 'fallback_on_memory_error'}
                                )
                        
                        # No fallback available
                        logger.error("No fallback transcription provider available")
                        raise
                    else:
                        # Not a memory error or fallback disabled
                        raise
            elif provider == 'lightning-whisper-mlx':
                if not LIGHTNING_WHISPER_AVAILABLE:
                    if sys.platform != 'darwin':
                        raise ValueError(f"lightning-whisper-mlx is only available on macOS (Apple Silicon)")
                    else:
                        raise ValueError(f"lightning-whisper-mlx is not installed. Install with: pip install lightning-whisper-mlx")
                result = self._transcribe_with_lightning_whisper_mlx(
                    wav_path, model, source_lang, target_lang,
                    progress_callback=progress_callback, **kwargs
                )
            elif provider == 'faster-whisper':
                if not FASTER_WHISPER_AVAILABLE:
                    raise ValueError(f"faster-whisper is not installed. Install with: pip install faster-whisper")
                result = self._transcribe_with_faster_whisper(
                    wav_path, model, language, vad_filter, source_lang, target_lang, 
                    progress_callback=progress_callback, **kwargs
                )
            elif provider == 'qwen2audio':
                if not QWEN2AUDIO_AVAILABLE:
                    raise ValueError(f"Qwen2Audio is not installed. Install transformers and torch for Qwen2Audio support")
                result = self._transcribe_with_qwen2audio(wav_path, progress_callback=progress_callback, **kwargs)
            elif provider == 'parakeet':
                if not NEMO_AVAILABLE:
                    raise ValueError(f"NeMo toolkit is not installed. Install with: pip install nemo-toolkit[asr]")
                result = self._transcribe_with_parakeet(wav_path, model, source_lang, progress_callback=progress_callback, **kwargs)
            elif provider == 'canary':
                if not NEMO_AVAILABLE:
                    raise ValueError(f"NeMo toolkit is not installed. Install with: pip install nemo-toolkit[asr]")
                result = self._transcribe_with_canary(
                    wav_path, model, source_lang, target_lang, progress_callback=progress_callback, **kwargs
                )
            else:
                available = self.get_available_providers()
                raise ValueError(f"Unknown or unavailable transcription provider: {provider}. Available providers: {', '.join(available)}")
            
            transcription_time = time.time() - transcription_start_time
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
            logger.info(f"Result: {len(result.get('text', ''))} characters, {len(result.get('segments', []))} segments")
            logger.debug(f"First 200 chars of transcription: {result.get('text', '')[:200]}...")
            
            return result
                
        finally:
            # Clean up temporary WAV if created
            if wav_path != audio_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                    logger.debug(f"Cleaned up temporary WAV file: {wav_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary WAV file {wav_path}: {e}")
    
    def _ensure_wav_format(self, audio_path: str) -> str:
        """
        Convert audio to WAV format if needed.
        
        Returns:
            Path to WAV file (may be same as input if already WAV)
        """
        audio_path = Path(audio_path)
        logger.debug(f"Checking audio format for: {audio_path}, suffix: {audio_path.suffix}")
        
        # Check if already WAV
        if audio_path.suffix.lower() == '.wav':
            logger.debug(f"Audio is already in WAV format: {audio_path}")
            return str(audio_path)
        
        # Convert to WAV
        wav_path = audio_path.with_suffix('.wav')
        
        # If file exists with same name, use temp file
        if wav_path.exists():
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            wav_path = Path(temp_file.name)
            temp_file.close()
        
        try:
            logger.info(f"Converting audio to WAV: {audio_path} -> {wav_path}")
            self._convert_to_wav(str(audio_path), str(wav_path))
            logger.info(f"Successfully converted to WAV: {wav_path}")
            return str(wav_path)
        except Exception as e:
            # Clean up on error
            if wav_path.exists() and wav_path != audio_path:
                wav_path.unlink()
                logger.debug(f"Cleaned up failed conversion output: {wav_path}")
            raise ConversionError(f"Failed to convert to WAV: {str(e)}") from e
    
    def _convert_to_wav(self, input_path: str, output_path: str):
        """Convert audio file to WAV using ffmpeg."""
        
        # Find ffmpeg
        ffmpeg_cmd = self._find_ffmpeg()
        
        # Conversion parameters optimized for speech recognition
        # 16kHz mono is standard for most speech models
        command = [
            ffmpeg_cmd,
            '-i', input_path,
            '-ar', '16000',      # 16kHz sample rate
            '-ac', '1',          # Mono
            '-c:a', 'pcm_s16le', # 16-bit PCM
            '-y',                # Overwrite output
            output_path
        ]
        
        logger.debug(f"Running FFmpeg command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"FFmpeg conversion successful: {output_path}")
            logger.debug(f"FFmpeg stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"FFmpeg stderr: {result.stderr}")
            
        except subprocess.CalledProcessError as e:
            raise ConversionError(
                f"FFmpeg conversion failed: {e.stderr}"
            ) from e
    
    def _find_ffmpeg(self) -> str:
        """Find ffmpeg executable."""
        # Check config first
        ffmpeg_path = get_cli_setting('media_processing.ffmpeg_path')
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path
        
        # Check common locations
        import shutil
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            return ffmpeg
        
        # Platform-specific paths
        if os.name == 'nt':  # Windows
            common_paths = [
                'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
                'C:\\ffmpeg\\bin\\ffmpeg.exe',
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return path
        
        raise FileNotFoundError(
            "ffmpeg not found. Please install ffmpeg or set media_processing.ffmpeg_path in config"
        )
    
    def _transcribe_with_faster_whisper(
        self,
        audio_path: str,
        model: str,
        language: str,
        vad_filter: bool,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str, Optional[Dict]], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using faster-whisper."""
        
        if not FASTER_WHISPER_AVAILABLE:
            logger.error("faster-whisper is not installed")
            raise TranscriptionError("faster-whisper is not installed")
        
        logger.info(f"Starting faster-whisper transcription: model={model}, language={language}, source_lang={source_lang}, target_lang={target_lang}")
        logger.debug(f"VAD filter: {vad_filter}, additional kwargs: {kwargs}")
        
        # Get or create model instance with thread safety
        cache_key = (model, self.config['device'], self.config['compute_type'])
        
        with self._model_cache_lock:
            if cache_key not in self._model_cache:
                logger.info(f"Loading Whisper model: {model} (device: {self.config['device']}, compute_type: {self.config['compute_type']})")
                model_load_start = time.time()
                try:
                    # Use protect_file_descriptors to handle the file descriptor issue
                    with protect_file_descriptors():
                        self._model_cache[cache_key] = WhisperModel(
                            model,
                            device=self.config['device'],
                            compute_type=self.config['compute_type'],
                            download_root=None,  # Use default cache directory
                            local_files_only=False  # Allow downloading if needed
                        )
                    model_load_time = time.time() - model_load_start
                    logger.info(f"Whisper model loaded successfully in {model_load_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Failed to load Whisper model: {str(e)}", exc_info=True)
                    # Provide more helpful error message
                    error_msg = f"Failed to load model {model}: {str(e)}"
                    if "bad value(s) in fds_to_keep" in str(e):
                        error_msg += "\n\nThis error typically occurs when file descriptors are not properly handled."
                        error_msg += "\nPossible solutions:"
                        error_msg += "\n1. Restart the application"
                        error_msg += "\n2. Pre-download the model manually:"
                        error_msg += f"\n   huggingface-cli download openai/whisper-{model}"
                        error_msg += "\n3. Set environment variable before starting:"
                        error_msg += "\n   export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES"
                    raise TranscriptionError(error_msg) from e
            
            whisper_model = self._model_cache[cache_key]
        logger.debug(f"Using cached Whisper model for {cache_key}")
        
        # Use source_lang if provided, otherwise fall back to language
        transcribe_language = source_lang or language
        
        # Determine task - translate if target language is English and source is non-English
        # For auto-detection cases (None or 'auto'), we need to detect language first
        if target_lang and target_lang == 'en':
            if transcribe_language and transcribe_language not in ['en', 'auto', None]:
                task = 'translate'
            else:
                # For auto-detection, we'll decide after language detection
                task = 'transcribe'
        else:
            task = 'transcribe'
        
        logger.info(f"Transcription task: {task}, language: {transcribe_language}")
        
        # Transcription options
        options = {
            'beam_size': 5,
            'best_of': 5,
            'vad_filter': vad_filter,
            'language': transcribe_language if transcribe_language != 'auto' else None,
            'task': task,
        }
        
        try:
            # Perform transcription
            if progress_callback:
                try:
                    progress_callback(0, "Starting transcription...", None)
                except Exception as e:
                    logger.warning(f"Progress callback error (ignored): {e}")
            
            logger.info(f"Starting Whisper transcription with options: {options}")
            transcribe_start = time.time()
            
            segments_generator, info = whisper_model.transcribe(audio_path, **options)
            
            logger.debug(f"Whisper transcription started, processing segments...")
            
            # Get total duration for progress calculation
            total_duration = info.duration if info.duration else 0
            logger.info(f"Audio duration: {total_duration:.2f} seconds")
            
            # Log detection info
            if info.language:
                logger.info(
                    f"Detected language: {info.language} "
                    f"(confidence: {info.language_probability:.2f})"
                )
                if progress_callback:
                    try:
                        progress_callback(
                            5, 
                            f"Language detected: {info.language} (confidence: {info.language_probability:.2f})",
                            {"language": info.language, "confidence": info.language_probability}
                        )
                    except Exception as e:
                        logger.warning(f"Progress callback error (ignored): {e}")
            
            # Collect segments
            segments = []
            full_text = []
            segment_count = 0
            last_progress = 5  # Start after language detection
            segment_start_time = time.time()
            
            for segment in segments_generator:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "Time_Start": segment.start,  # Legacy format support
                    "Time_End": segment.end,
                    "Text": segment.text.strip()
                }
                segments.append(segment_dict)
                # Only add non-empty text to full_text to avoid extra spaces when joining
                stripped_text = segment.text.strip()
                if stripped_text:
                    full_text.append(stripped_text)
                segment_count += 1
                
                if segment_count % 10 == 0:
                    logger.debug(f"Processed {segment_count} segments, last segment: {segment.start:.1f}s - {segment.end:.1f}s")
                
                # Calculate and report progress
                if progress_callback and total_duration > 0:
                    # Calculate progress based on time (5-95% range, leaving room for finalization)
                    time_progress = 5 + (segment.end / total_duration) * 90
                    
                    # Only update if progress increased by at least 1%
                    if time_progress - last_progress >= 1:
                        try:
                            progress_callback(
                                time_progress,
                                f"Transcribing: {segment.end:.1f}s / {total_duration:.1f}s",
                                {
                                    "segment_num": segment_count,
                                    "segment_text": segment.text.strip(),
                                    "current_time": segment.end,
                                    "total_time": total_duration
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Progress callback error (ignored): {e}")
                        last_progress = time_progress
            
            segment_time = time.time() - segment_start_time
            total_time = time.time() - transcribe_start
            
            result = {
                "text": " ".join(full_text),
                "segments": segments,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "provider": "faster-whisper",
                "model": model,
            }
            
            logger.info(f"Faster-whisper transcription complete:")
            logger.info(f"  - Total segments: {len(segments)}")
            logger.info(f"  - Total characters: {len(result['text'])}")
            logger.info(f"  - Processing time: {total_time:.2f}s (segment collection: {segment_time:.2f}s)")
            logger.info(f"  - Speed: {total_duration / total_time:.2f}x realtime" if total_duration > 0 else "  - Speed: N/A")
            
            # Add translation info if applicable
            if task == 'translate':
                result["task"] = "translation"
                result["source_language"] = transcribe_language or info.language
                result["target_language"] = "en"
                result["translation"] = result["text"]
            
            # Final progress update
            if progress_callback:
                try:
                    progress_callback(
                        100,
                        f"Transcription complete: {len(segments)} segments, {len(result['text'])} characters",
                        {
                            "total_segments": len(segments),
                            "total_chars": len(result["text"]),
                            "duration": info.duration
                        }
                    )
                except Exception as e:
                    logger.warning(f"Progress callback error (ignored): {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Faster-whisper transcription failed: {str(e)}", exc_info=True)
            raise TranscriptionError(
                f"Transcription failed: {str(e)}"
            ) from e
    
    def _transcribe_with_qwen2audio(
        self,
        audio_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using Qwen2Audio model."""
        
        if not QWEN2AUDIO_AVAILABLE:
            logger.error("Qwen2Audio dependencies not installed")
            raise TranscriptionError(
                "Qwen2Audio dependencies not installed. "
                "Install with: pip install transformers torch"
            )
        
        logger.info("Starting Qwen2Audio transcription")
        transcribe_start = time.time()
        
        if not SOUNDFILE_AVAILABLE:
            raise TranscriptionError("soundfile required for Qwen2Audio")
        
        # Lazy load Qwen2Audio models
        if self._qwen_processor is None:
            logger.info("Loading Qwen2Audio model for first time...")
            model_load_start = time.time()
            try:
                self._qwen_processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2-Audio-7B-Instruct"
                )
                self._qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-Audio-7B-Instruct",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
                model_load_time = time.time() - model_load_start
                logger.info(f"Qwen2Audio model loaded successfully in {model_load_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to load Qwen2Audio model: {str(e)}", exc_info=True)
                raise TranscriptionError(
                    f"Failed to load Qwen2Audio: {str(e)}"
                ) from e
        
        try:
            # Load audio
            logger.debug(f"Loading audio file: {audio_path}")
            audio_data, sample_rate = sf.read(audio_path)
            logger.info(f"Loaded audio: sample_rate={sample_rate}, duration={len(audio_data)/sample_rate:.2f}s")
            
            # Prepare prompt for transcription
            prompt_text = (
                "System: You are a transcription model.\n"
                "User: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                "Assistant:"
            )
            
            # Process audio
            logger.debug("Processing audio with Qwen2Audio processor")
            process_start = time.time()
            
            inputs = self._qwen_processor(
                text=prompt_text,
                audios=audio_data,
                return_tensors="pt",
                sampling_rate=sample_rate
            )
            
            process_time = time.time() - process_start
            logger.debug(f"Audio processing completed in {process_time:.2f} seconds")
            
            # Move to device
            device = self._qwen_model.device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            
            # Generate transcription
            logger.info("Generating transcription with Qwen2Audio")
            generate_start = time.time()
            
            with torch.no_grad():
                generated_ids = self._qwen_model.generate(
                    **inputs,
                    max_new_tokens=512
                )
            
            generate_time = time.time() - generate_start
            logger.info(f"Generation completed in {generate_time:.2f} seconds")
            
            # Decode output
            transcription = self._qwen_processor.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            
            # Extract transcription after "Assistant:"
            if "Assistant:" in transcription:
                transcription = transcription.split("Assistant:")[-1].strip()
            
            # Create segment (Qwen2Audio doesn't provide timestamps)
            total_time = time.time() - transcribe_start
            
            logger.info(f"Qwen2Audio transcription complete:")
            logger.info(f"  - Total characters: {len(transcription)}")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.debug(f"First 200 chars: {transcription[:200]}...")
            
            return {
                "text": transcription,
                "segments": [{
                    "start": 0.0,
                    "end": 0.0,  # Duration unknown without processing
                    "text": transcription,
                    "Time_Start": 0.0,
                    "Time_End": 0.0,
                    "Text": transcription
                }],
                "language": "unknown",  # Qwen2Audio doesn't detect language
            }
            
        except Exception as e:
            logger.error(f"Qwen2Audio transcription failed: {str(e)}", exc_info=True)
            raise TranscriptionError(
                f"Qwen2Audio transcription failed: {str(e)}"
            ) from e
    
    def _transcribe_with_parakeet(
        self,
        audio_path: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using NVIDIA Parakeet models via NeMo."""
        
        if not NEMO_AVAILABLE:
            logger.error("NeMo toolkit not installed")
            raise TranscriptionError(
                "NeMo toolkit not installed. "
                "Install with: pip install nemo-toolkit[asr]"
            )
        
        logger.info(f"Starting Parakeet transcription with model: {model or 'nvidia/parakeet-tdt-1.1b'}")
        transcribe_start = time.time()
        
        # Lazy load Parakeet model
        if self._parakeet_model is None:
            model_name = model or "nvidia/parakeet-tdt-1.1b"
            logger.info(f"Loading Parakeet model: {model_name} for first time...")
            model_load_start = time.time()
            try:
                model_name = model or "nvidia/parakeet-tdt-1.1b"
                self._parakeet_model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_name
                )
                
                # Enable optimizations for streaming/long audio
                self._parakeet_model.change_attention_model(
                    "rel_pos_local_attn", 
                    context_sizes=[128, 128]
                )
                self._parakeet_model.change_subsampling_conv_chunking_factor(1)
                
                # Move to appropriate device
                if self.config['device'] == 'cuda' and torch.cuda.is_available():
                    self._parakeet_model = self._parakeet_model.cuda()
                
                model_load_time = time.time() - model_load_start
                logger.info(f"Parakeet model loaded successfully in {model_load_time:.2f} seconds")
                logger.debug(f"Model device: {self._parakeet_model.device}")
                
            except Exception as e:
                logger.error(f"Failed to load Parakeet model: {str(e)}", exc_info=True)
                raise TranscriptionError(
                    f"Failed to load Parakeet model: {str(e)}"
                ) from e
        
        try:
            # Transcribe the audio
            logger.info(f"Transcribing audio file: {audio_path}")
            transcribe_audio_start = time.time()
            
            transcripts = self._parakeet_model.transcribe(
                paths2audio_files=[audio_path],
                batch_size=1,
                return_hypotheses=True,
                verbose=False
            )
            
            transcribe_audio_time = time.time() - transcribe_audio_start
            logger.info(f"Parakeet transcription completed in {transcribe_audio_time:.2f} seconds")
            
            # Process results
            if transcripts and len(transcripts) > 0:
                # Get the first (and only) transcript
                transcript = transcripts[0]
                
                # Handle different return formats from NeMo
                if isinstance(transcript, str):
                    text = transcript
                elif hasattr(transcript, 'text'):
                    text = transcript.text
                elif isinstance(transcript, list) and len(transcript) > 0:
                    text = transcript[0]
                else:
                    text = str(transcript)
                
                # Create segments (Parakeet doesn't provide timestamps by default)
                total_time = time.time() - transcribe_start
                
                logger.info(f"Parakeet transcription complete:")
                logger.info(f"  - Total characters: {len(text)}")
                logger.info(f"  - Total time: {total_time:.2f}s")
                logger.debug(f"First 200 chars: {text[:200]}...")
                
                return {
                    "text": text,
                    "segments": [{
                        "start": 0.0,
                        "end": 0.0,
                        "text": text,
                        "Time_Start": 0.0,
                        "Time_End": 0.0,
                        "Text": text
                    }],
                    "language": language or "unknown",
                    "provider": "parakeet",
                    "model": model or "nvidia/parakeet-tdt-1.1b"
                }
            else:
                logger.error("Parakeet produced no transcription output")
                raise TranscriptionError("No transcription produced")
                
        except Exception as e:
            logger.error(f"Parakeet transcription failed: {str(e)}", exc_info=True)
            raise TranscriptionError(
                f"Parakeet transcription failed: {str(e)}"
            ) from e
    
    def _transcribe_with_canary(
        self,
        audio_path: str,
        model: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        chunk_len_in_secs: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using NVIDIA Canary model with chunked inference for long audio."""
        
        # Use configured chunk length if not specified
        chunk_len_in_secs = chunk_len_in_secs or self.config['chunk_length_seconds']
        
        logger.info(f"Starting Canary transcription with model: {model or 'nvidia/canary-1b-flash'}")
        logger.info(f"Source language: {source_lang}, Target language: {target_lang}, Chunk length: {chunk_len_in_secs}s")
        transcribe_start = time.time()
        
        if not NEMO_AVAILABLE:
            logger.error("NeMo toolkit not installed")
            raise TranscriptionError(
                "NeMo toolkit not installed. "
                "Install with: pip install nemo-toolkit[asr]"
            )
        
        # Import additional NeMo modules needed for Canary
        try:
            from nemo.collections.asr.models import EncDecMultiTaskModel
            from nemo.collections.asr.parts.utils.transcribe_utils import (
                transcribe_partial_audio,
                get_buffered_pred_feat_multitaskAED
            )
        except ImportError as e:
            raise TranscriptionError(
                f"Failed to import required NeMo modules: {str(e)}"
            ) from e
        
        # Lazy load Canary model
        if self._canary_model is None:
            model_name = model or "nvidia/canary-1b-flash"
            logger.info(f"Loading Canary model: {model_name} for first time...")
            model_load_start = time.time()
            try:
                model_name = model or "nvidia/canary-1b-flash"
                self._canary_model = EncDecMultiTaskModel.from_pretrained(
                    model_name=model_name
                )
                
                # Initialize decoding strategy
                self._canary_decoding = self._canary_model.decoding
                
                # Move to appropriate device
                if self.config['device'] == 'cuda' and torch.cuda.is_available():
                    self._canary_model = self._canary_model.cuda()
                
                # Set to evaluation mode
                self._canary_model.eval()
                
                model_load_time = time.time() - model_load_start
                logger.info(f"Canary model loaded successfully in {model_load_time:.2f} seconds")
                logger.debug(f"Model device: {self._canary_model.device}")
                
            except Exception as e:
                logger.error(f"Failed to load Canary model: {str(e)}", exc_info=True)
                raise TranscriptionError(
                    f"Failed to load Canary model: {str(e)}"
                ) from e
        
        try:
            # Determine task type based on target language
            taskname = "s2t_translation" if target_lang and target_lang != source_lang else "asr"
            logger.info(f"Canary task type: {taskname}")
            
            # Map language codes to Canary format
            lang_map = {
                "en": "en", "de": "de", "es": "es", "fr": "fr",
                "english": "en", "german": "de", "spanish": "es", "french": "fr"
            }
            
            source_lang = lang_map.get((source_lang or "en").lower(), "en")
            if target_lang:
                target_lang = lang_map.get(target_lang.lower(), target_lang)
            
            # For ASR, source and target should be the same
            if taskname == "asr":
                target_lang = source_lang
            
            # Create manifest entry for Canary
            manifest_entry = {
                "audio_filepath": audio_path,
                "taskname": taskname,
                "source_lang": source_lang,
                "target_lang": target_lang or source_lang,
                "pnc": "yes",  # Punctuation and capitalization
                "answer": "na",  # Not used for inference
                "duration": None  # Will be computed
            }
            
            # Get audio duration
            logger.debug(f"Getting audio info for: {audio_path}")
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
            manifest_entry["duration"] = duration
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Use chunked inference for long audio
            if duration > chunk_len_in_secs:
                logger.info(f"Using chunked inference for {duration:.1f}s audio (chunk size: {chunk_len_in_secs}s)")
                chunk_start = time.time()
                
                # Prepare for chunked processing
                model_stride_in_secs = 0.04  # 40ms model stride
                tokens_per_chunk = int(chunk_len_in_secs / model_stride_in_secs)
                
                # Perform chunked transcription
                hyps = get_buffered_pred_feat_multitaskAED(
                    self._canary_model.cfg.preprocessor,
                    [manifest_entry],
                    self._canary_model,
                    chunk_len_in_secs,
                    tokens_per_chunk,
                    self._canary_model.device,
                )
                
                # Extract transcription text
                text = hyps[0] if hyps else ""
                
                chunk_time = time.time() - chunk_start
                logger.info(f"Chunked transcription completed in {chunk_time:.2f} seconds")
                logger.debug(f"Number of chunks processed: {int(duration / chunk_len_in_secs) + 1}")
                
            else:
                # For short audio, use regular inference
                logger.info(f"Using regular inference for {duration:.1f}s audio")
                regular_start = time.time()
                
                transcripts = self._canary_model.transcribe(
                    audio=[audio_path],
                    batch_size=1,
                    task=taskname,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    pnc=True,
                    verbose=False
                )
                
                text = transcripts[0] if transcripts else ""
                
                regular_time = time.time() - regular_start
                logger.info(f"Regular transcription completed in {regular_time:.2f} seconds")
            
            # Create response
            total_time = time.time() - transcribe_start
            
            logger.info(f"Canary transcription complete:")
            logger.info(f"  - Task: {taskname}")
            logger.info(f"  - Total characters: {len(text)}")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.info(f"  - Speed: {duration / total_time:.2f}x realtime" if duration > 0 else "  - Speed: N/A")
            logger.debug(f"First 200 chars: {text[:200]}...")
            
            result = {
                "text": text,
                "segments": [{
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                    "Time_Start": 0.0,
                    "Time_End": duration,
                    "Text": text
                }],
                "language": source_lang,
                "provider": "canary",
                "model": model or "nvidia/canary-1b-flash",
                "task": taskname
            }
            
            # Add translation info if applicable
            if taskname == "s2t_translation":
                result["source_language"] = source_lang
                result["target_language"] = target_lang
                result["translation"] = text
            
            return result
            
        except Exception as e:
            logger.error(f"Canary transcription failed: {str(e)}", exc_info=True)
            raise TranscriptionError(
                f"Canary transcription failed: {str(e)}"
            ) from e
    
    def _transcribe_with_lightning_whisper_mlx(
        self,
        audio_path: str,
        model: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        batch_size: Optional[int] = None,
        quant: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str, Optional[Dict]], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using Lightning Whisper MLX (Apple Silicon optimized)."""
        
        if not LIGHTNING_WHISPER_AVAILABLE:
            logger.error("lightning-whisper-mlx is not installed")
            raise TranscriptionError("lightning-whisper-mlx is not installed")
        
        # Use configured settings or provided parameters
        batch_size = batch_size or self._lightning_whisper_config['batch_size']
        quant = quant or self._lightning_whisper_config['quant']
        
        # Map model names to lightning-whisper-mlx format
        model = model or self.config['default_model']
        
        # Lightning Whisper MLX supports same model names as Whisper
        # but may need mapping for some variants
        model_mapping = {
            'large': 'large-v3',
            'distil-large-v2': 'distil-large-v2',
            'distil-large-v3': 'distil-large-v3',
            'distil-medium.en': 'distil-medium.en',
            'distil-small.en': 'distil-small.en',
        }
        
        model = model_mapping.get(model, model)
        
        logger.info(f"Starting Lightning Whisper MLX transcription: model={model}, batch_size={batch_size}, quant={quant}")
        logger.info(f"Source language: {source_lang}, Target language: {target_lang}")
        transcribe_start = time.time()
        
        # Lazy load Lightning Whisper MLX model
        if self._lightning_whisper_model is None or \
           getattr(self._lightning_whisper_model, '_model_name', None) != model or \
           getattr(self._lightning_whisper_model, '_batch_size', None) != batch_size or \
           getattr(self._lightning_whisper_model, '_quant', None) != quant:
            
            logger.info(f"Loading Lightning Whisper MLX model: {model}")
            model_load_start = time.time()
            
            try:
                self._lightning_whisper_model = LightningWhisperMLX(
                    model=model,
                    batch_size=batch_size,
                    quant=quant
                )
                # Store config for cache comparison
                self._lightning_whisper_model._model_name = model
                self._lightning_whisper_model._batch_size = batch_size
                self._lightning_whisper_model._quant = quant
                
                model_load_time = time.time() - model_load_start
                logger.info(f"Lightning Whisper MLX model loaded successfully in {model_load_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Failed to load Lightning Whisper MLX model: {str(e)}", exc_info=True)
                raise TranscriptionError(
                    f"Failed to load Lightning Whisper MLX model: {str(e)}"
                ) from e
        
        try:
            # Report progress
            if progress_callback:
                try:
                    progress_callback(0, "Starting transcription with Lightning Whisper MLX...", None)
                except Exception as e:
                    logger.warning(f"Progress callback error (ignored): {e}")
            
            logger.info(f"Transcribing audio file: {audio_path}")
            transcribe_audio_start = time.time()
            
            # Transcribe the audio
            # Lightning Whisper MLX returns a dict with 'text' and possibly other fields
            result_dict = self._lightning_whisper_model.transcribe(audio_path)
            
            transcribe_audio_time = time.time() - transcribe_audio_start
            logger.info(f"Lightning Whisper MLX transcription completed in {transcribe_audio_time:.2f} seconds")
            
            # Handle different return formats from Lightning Whisper MLX
            if isinstance(result_dict, dict):
                # Expected format
                text = result_dict.get('text', '')
                segments = result_dict.get('segments', [])
            elif isinstance(result_dict, list):
                # Some versions return a list
                logger.warning("Lightning Whisper MLX returned a list instead of dict, converting...")
                text = ' '.join(str(item) for item in result_dict if item)
                segments = []
            else:
                # Fallback
                logger.warning(f"Unexpected result type from Lightning Whisper MLX: {type(result_dict)}")
                text = str(result_dict)
                segments = []
            
            # If no segments provided, create a single segment
            if not segments and text:
                segments = [{
                    "start": 0.0,
                    "end": 0.0,  # Duration unknown
                    "text": text,
                    "Time_Start": 0.0,
                    "Time_End": 0.0,
                    "Text": text
                }]
            else:
                # Ensure segments have the expected format
                formatted_segments = []
                for seg in segments:
                    if isinstance(seg, dict):
                        formatted_segments.append({
                            "start": seg.get('start', 0.0),
                            "end": seg.get('end', 0.0),
                            "text": seg.get('text', '').strip(),
                            "Time_Start": seg.get('start', 0.0),
                            "Time_End": seg.get('end', 0.0),
                            "Text": seg.get('text', '').strip()
                        })
                    else:
                        # Handle non-dict segments (e.g., strings)
                        formatted_segments.append({
                            "start": 0.0,
                            "end": 0.0,
                            "text": str(seg).strip(),
                            "Time_Start": 0.0,
                            "Time_End": 0.0,
                            "Text": str(seg).strip()
                        })
                segments = formatted_segments
            
            # Create response
            total_time = time.time() - transcribe_start
            
            logger.info(f"Lightning Whisper MLX transcription complete:")
            logger.info(f"  - Model: {model}")
            logger.info(f"  - Batch size: {batch_size}")
            logger.info(f"  - Quantization: {quant or 'None'}")
            logger.info(f"  - Total characters: {len(text)}")
            logger.info(f"  - Total segments: {len(segments)}")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.debug(f"First 200 chars: {text[:200]}...")
            
            result = {
                "text": text,
                "segments": segments,
                "language": result_dict.get('language', source_lang or 'unknown') if isinstance(result_dict, dict) else (source_lang or 'unknown'),
                "provider": "lightning-whisper-mlx",
                "model": model,
                "batch_size": batch_size,
                "quantization": quant
            }
            
            # Check if translation was performed
            if isinstance(result_dict, dict) and 'translation' in result_dict:
                result["translation"] = result_dict['translation']
                result["source_language"] = source_lang or result_dict.get('language', 'unknown')
                result["target_language"] = target_lang or 'en'
            
            # Final progress update
            if progress_callback:
                try:
                    progress_callback(
                        100,
                        f"Transcription complete: {len(segments)} segments, {len(text) if text else 0} characters",
                        {
                            "total_segments": len(segments),
                            "total_chars": len(text),
                            "model": model,
                            "batch_size": batch_size
                        }
                    )
                except Exception as e:
                    logger.warning(f"Progress callback error (ignored): {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Lightning Whisper MLX transcription failed: {str(e)}", exc_info=True)
            raise TranscriptionError(
                f"Lightning Whisper MLX transcription failed: {str(e)}"
            ) from e
    
    def _transcribe_with_parakeet_mlx(
        self,
        audio_path: str,
        model: Optional[str] = None,
        source_lang: Optional[str] = None,
        precision: Optional[str] = None,
        attention_type: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str, Optional[Dict]], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using Parakeet MLX (Apple Silicon optimized real-time ASR)."""
        
        if not PARAKEET_MLX_AVAILABLE:
            logger.error("parakeet-mlx is not installed")
            raise TranscriptionError("parakeet-mlx is not installed")
        
        # Use configured settings or provided parameters
        model = model or self._parakeet_mlx_config['model']
        precision = precision or self._parakeet_mlx_config['precision']
        attention_type = attention_type or self._parakeet_mlx_config['attention_type']
        
        logger.debug(f"Parakeet MLX config: {self._parakeet_mlx_config}")
        logger.debug(f"Using model: {model}, precision: {precision}")
        logger.info(f"Starting Parakeet MLX transcription: model={model}, precision={precision}")
        transcribe_start = time.time()
        
        # Lazy load Parakeet MLX model with thread safety
        with self._parakeet_mlx_model_lock:
            if self._parakeet_mlx_model is None or \
               getattr(self._parakeet_mlx_model, '_model_name', None) != model:
                
                logger.info(f"Loading Parakeet MLX model: {model}")
                model_load_start = time.time()
                
                try:
                    # Ensure model is a string and not None
                    if not model or not isinstance(model, str):
                        logger.error(f"Invalid model value: {model} (type: {type(model)})")
                        model = 'mlx-community/parakeet-tdt-0.6b-v2'
                        logger.warning(f"Using default model: {model}")
                    
                    # Map precision to MLX dtype
                    import mlx.core as mx
                    dtype_map = {
                        'fp32': mx.float32,
                        'fp16': mx.float16,
                        'bf16': mx.bfloat16,
                        'bfloat16': mx.bfloat16
                    }
                    dtype = dtype_map.get(precision, mx.bfloat16)
                    
                    logger.debug(f"Calling parakeet_from_pretrained with model='{model}', dtype={dtype}")
                    self._parakeet_mlx_model = parakeet_from_pretrained(
                        model,
                        dtype=dtype
                    )
                    # Store model name for cache comparison
                    self._parakeet_mlx_model._model_name = model
                    
                    model_load_time = time.time() - model_load_start
                    logger.info(f"Parakeet MLX model loaded successfully in {model_load_time:.2f} seconds")
                    
                except Exception as e:
                    logger.error(f"Failed to load Parakeet MLX model: {str(e)}", exc_info=True)
                    raise TranscriptionError(
                        f"Failed to load Parakeet MLX model: {str(e)}"
                    ) from e
        
        try:
            # Report progress
            if progress_callback:
                progress_callback(0, "Starting transcription with Parakeet MLX...", None)
            
            logger.info(f"Starting Parakeet MLX transcription")
            logger.info(f"  Audio file: {audio_path}")
            logger.info(f"  Model: {model}")
            logger.info(f"  Precision: {precision}")
            
            # Check audio duration if soundfile is available
            audio_duration = None
            chunk_duration = kwargs.get('chunk_duration', self._parakeet_mlx_config['chunk_duration'])
            overlap_duration = kwargs.get('overlap_duration', self._parakeet_mlx_config['overlap_duration'])
            auto_chunk_threshold = kwargs.get('auto_chunk_threshold', self._parakeet_mlx_config['auto_chunk_threshold'])
            
            if SOUNDFILE_AVAILABLE:
                try:
                    import soundfile as sf
                    audio_info = sf.info(audio_path)
                    audio_duration = audio_info.duration
                    try:
                        logger.info(f"  Audio duration: {audio_duration:.2f} seconds")
                    except (TypeError, ValueError):
                        logger.info(f"  Audio duration: {audio_duration} seconds")
                except Exception as e:
                    logger.warning(f"Could not get audio duration: {e}")
            
            transcribe_audio_start = time.time()
            
            # Determine if we should use chunking
            use_chunking = False
            if audio_duration and audio_duration > auto_chunk_threshold:
                use_chunking = True
                # Use safe formatting for potentially mocked values
                try:
                    logger.info(f"Audio duration ({audio_duration:.1f}s) exceeds threshold ({auto_chunk_threshold:.1f}s), will use chunking")
                except (TypeError, ValueError):
                    logger.info(f"Audio duration ({audio_duration}s) exceeds threshold ({auto_chunk_threshold}s), will use chunking")
            
            # Transcribe the audio
            if use_chunking and hasattr(self._parakeet_mlx_model, 'transcribe'):
                # Check if the model supports chunking parameters
                try:
                    # Try with chunking parameters first
                    result = self._parakeet_mlx_model.transcribe(
                        audio_path,
                        chunk_duration=chunk_duration,
                        overlap_duration=overlap_duration
                    )
                    try:
                        logger.info(f"Using chunked transcription with chunk_duration={chunk_duration}s, overlap={overlap_duration}s")
                    except (TypeError, ValueError):
                        logger.info(f"Using chunked transcription with chunk_duration={chunk_duration}, overlap={overlap_duration}")
                except TypeError as e:
                    # If chunking not supported, fall back to regular transcription
                    logger.warning(f"Chunking parameters not supported by model, using regular transcription: {e}")
                    result = self._parakeet_mlx_model.transcribe(audio_path)
            else:
                # Regular transcription
                result = self._parakeet_mlx_model.transcribe(audio_path)
            
            transcribe_audio_time = time.time() - transcribe_audio_start
            try:
                logger.info(f"Parakeet MLX transcription completed in {transcribe_audio_time:.2f} seconds")
            except (TypeError, ValueError):
                logger.info(f"Parakeet MLX transcription completed in {transcribe_audio_time} seconds")
            
            # Extract text from result
            text = result.text if hasattr(result, 'text') else str(result)
            
            # Handle None result
            if text is None:
                logger.warning("Parakeet MLX returned None text result")
                text = ""
            
            # Parakeet MLX provides sentence-level timestamps
            segments = []
            if hasattr(result, 'sentences') and result.sentences:
                for sentence in result.sentences:
                    segment_dict = {
                        "start": sentence.start if hasattr(sentence, 'start') else 0.0,
                        "end": sentence.end if hasattr(sentence, 'end') else 0.0,
                        "text": sentence.text.strip() if hasattr(sentence, 'text') else str(sentence),
                        "Time_Start": sentence.start if hasattr(sentence, 'start') else 0.0,
                        "Time_End": sentence.end if hasattr(sentence, 'end') else 0.0,
                        "Text": sentence.text.strip() if hasattr(sentence, 'text') else str(sentence)
                    }
                    segments.append(segment_dict)
            else:
                # No sentence-level timing, create single segment
                segments = [{
                    "start": 0.0,
                    "end": 0.0,
                    "text": text,
                    "Time_Start": 0.0,
                    "Time_End": 0.0,
                    "Text": text
                }]
            
            # Create response
            total_time = time.time() - transcribe_start
            
            logger.info(f"Parakeet MLX transcription complete:")
            logger.info(f"  - Model: {model}")
            logger.info(f"  - Precision: {precision}")
            logger.info(f"  - Total characters: {len(text) if text else 0}")
            logger.info(f"  - Total segments: {len(segments)}")
            try:
                logger.info(f"  - Total time: {total_time:.2f}s")
            except (TypeError, ValueError):
                logger.info(f"  - Total time: {total_time}s")
            
            # Calculate speed if we have duration info
            if hasattr(result, 'duration'):
                try:
                    speed = result.duration / total_time
                    logger.info(f"  - Speed: {speed:.2f}x realtime")
                except (TypeError, ValueError, ZeroDivisionError):
                    logger.info(f"  - Speed: Unable to calculate")
            
            if text:
                logger.debug(f"First 200 chars: {text[:200]}...")
            else:
                logger.debug("No text content in transcription result")
            
            result_dict = {
                "text": text,
                "segments": segments,
                "language": source_lang or 'en',  # Parakeet is English-only
                "provider": "parakeet-mlx",
                "model": model,
                "precision": precision,
                "attention_type": attention_type
            }
            
            # Add duration if available
            if hasattr(result, 'duration'):
                result_dict["duration"] = result.duration
            
            # Final progress update
            if progress_callback:
                progress_callback(
                    100,
                    f"Transcription complete: {len(segments)} segments, {len(text) if text else 0} characters",
                    {
                        "total_segments": len(segments),
                        "total_chars": len(text) if text else 0,
                        "model": model
                    }
                )
            
            return result_dict
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Parakeet MLX transcription failed: {error_msg}", exc_info=True)
            
            # Check if this is a memory allocation error
            if "metal::malloc" in error_msg.lower() or "buffer size" in error_msg.lower():
                logger.warning("Memory allocation error detected, suggesting fallback to another provider")
                
                # Add information about the error type
                raise TranscriptionError(
                    f"Parakeet MLX transcription failed due to memory allocation error: {error_msg}\n"
                    f"Consider using a different provider (e.g., 'faster-whisper' or 'lightning-whisper-mlx') "
                    f"or processing smaller audio segments."
                ) from e
            else:
                raise TranscriptionError(
                    f"Parakeet MLX transcription failed: {error_msg}"
                ) from e
    
    def get_available_providers(self) -> List[str]:
        """Get list of available transcription providers based on installed dependencies."""
        logger.debug("Checking available transcription providers...")
        providers = []
        
        if PARAKEET_MLX_AVAILABLE:
            providers.append('parakeet-mlx')
            logger.debug("parakeet-mlx is available (Real-time ASR for Apple Silicon)")
        
        if LIGHTNING_WHISPER_AVAILABLE:
            providers.append('lightning-whisper-mlx')
            logger.debug("lightning-whisper-mlx is available (Apple Silicon optimized)")
        
        if FASTER_WHISPER_AVAILABLE:
            providers.append('faster-whisper')
            logger.debug("faster-whisper is available")
        
        if QWEN2AUDIO_AVAILABLE:
            providers.append('qwen2audio')
            logger.debug("qwen2audio is available")
            
        if NEMO_AVAILABLE:
            providers.extend(['parakeet', 'canary'])
            logger.debug("parakeet and canary are available (NeMo installed)")
            
        logger.info(f"Available transcription providers: {providers}")
        return providers
    
    def list_available_models(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """List available models for each provider."""
        logger.debug(f"Listing available models for provider: {provider or 'all'}")
        
        models = {}
        
        if PARAKEET_MLX_AVAILABLE:
            models['parakeet-mlx'] = [
                'mlx-community/parakeet-tdt-0.6b-v2',  # Default model
                # Additional models can be added here as they become available
            ]
        
        if LIGHTNING_WHISPER_AVAILABLE:
            models['lightning-whisper-mlx'] = [
                'tiny', 'tiny.en', 'base', 'base.en', 
                'small', 'small.en', 'medium', 'medium.en',
                'large-v1', 'large-v2', 'large-v3', 'large',
                'distil-large-v2', 'distil-medium.en', 'distil-small.en', 'distil-large-v3'
            ]
        
        if FASTER_WHISPER_AVAILABLE:
            models['faster-whisper'] = [
                'tiny', 'tiny.en', 'base', 'base.en',
                'small', 'small.en', 'medium', 'medium.en',
                'large-v1', 'large-v2', 'large-v3', 'large',
                'distil-large-v2', 'distil-medium.en', 'distil-small.en', 'distil-large-v3',
                'deepdml/faster-distil-whisper-large-v3.5',
                'deepdml/faster-whisper-large-v3-turbo-ct2',
                'nyrahealth/faster_CrisperWhisper'
            ]
        
        if QWEN2AUDIO_AVAILABLE:
            models['qwen2audio'] = ['Qwen2-Audio-7B-Instruct']
        
        if NEMO_AVAILABLE:
            models['parakeet'] = [
                'nvidia/parakeet-tdt-1.1b',
                'nvidia/parakeet-rnnt-1.1b',
                'nvidia/parakeet-ctc-1.1b',
                'nvidia/parakeet-tdt-0.6b',
                'nvidia/parakeet-rnnt-0.6b',
                'nvidia/parakeet-ctc-0.6b',
                'nvidia/parakeet-tdt-0.6b-v2'
            ]
            models['canary'] = [
                'nvidia/canary-1b-flash',
                'nvidia/canary-1b'
            ]
        
        if provider:
            result = {provider: models.get(provider, [])}
            logger.info(f"Available models for {provider}: {len(result[provider])} models")
            return result
        
        total_models = sum(len(m) for m in models.values())
        logger.info(f"Total available models across all providers: {total_models}")
        return models
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available compute devices."""
        logger.debug("Getting compute device information...")
        
        info = {
            'cpu': True,
            'cuda': False,
            'mps': False,  # Apple Silicon
        }
        
        if torch and hasattr(torch, 'cuda'):
            info['cuda'] = torch.cuda.is_available()
            if info['cuda']:
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                logger.info(f"CUDA available: {info['cuda_device_count']} device(s), primary: {info['cuda_device_name']}")
            else:
                logger.debug("CUDA not available")
        
        if torch and hasattr(torch.backends, 'mps'):
            info['mps'] = torch.backends.mps.is_available()
            if info['mps']:
                logger.info("Apple Silicon MPS available")
            else:
                logger.debug("Apple Silicon MPS not available")
        
        logger.debug(f"Device info: {info}")
        return info
    
    def format_segments_with_timestamps(
        self,
        segments: List[Dict[str, Any]],
        include_timestamps: bool = True
    ) -> str:
        """Format segments with optional timestamps."""
        
        if not segments:
            return ""
        
        lines = []
        
        for segment in segments:
            if include_timestamps:
                start = segment.get('start', segment.get('Time_Start', 0))
                end = segment.get('end', segment.get('Time_End', 0))
                text = segment.get('text', segment.get('Text', ''))
                
                # Format timestamps as HH:MM:SS
                start_str = time.strftime('%H:%M:%S', time.gmtime(start))
                end_str = time.strftime('%H:%M:%S', time.gmtime(end))
                
                lines.append(f"[{start_str} - {end_str}] {text}")
            else:
                text = segment.get('text', segment.get('Text', ''))
                lines.append(text)
        
        return '\n'.join(lines)
    
    def create_streaming_transcriber(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        source_lang: Optional[str] = None,
        **kwargs
    ):
        """
        Create a streaming transcriber for real-time STT.
        Currently only supported with parakeet-mlx on macOS.
        
        Returns:
            StreamingTranscriber object or None if not supported
        """
        provider = provider or self.config['default_provider']
        
        if provider == 'parakeet-mlx' and PARAKEET_MLX_AVAILABLE:
            logger.info("Creating Parakeet MLX streaming transcriber")
            
            # Use configured settings or provided parameters
            model = model or self._parakeet_mlx_config['model']
            precision = kwargs.get('precision', self._parakeet_mlx_config['precision'])
            attention_type = kwargs.get('attention_type', self._parakeet_mlx_config['attention_type'])
            
            # Load model if not already loaded
            if self._parakeet_mlx_model is None or \
               getattr(self._parakeet_mlx_model, '_model_name', None) != model:
                
                logger.info(f"Loading Parakeet MLX model for streaming: {model}")
                try:
                    # Map precision to MLX dtype
                    import mlx.core as mx
                    dtype_map = {
                        'fp32': mx.float32,
                        'fp16': mx.float16,
                        'bf16': mx.bfloat16,
                        'bfloat16': mx.bfloat16
                    }
                    dtype = dtype_map.get(precision, mx.bfloat16)
                    
                    self._parakeet_mlx_model = parakeet_from_pretrained(
                        model,
                        dtype=dtype
                    )
                    self._parakeet_mlx_model._model_name = model
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load model: {str(e)}")
                    return None
            
            # Return a streaming transcriber wrapper
            return ParakeetMLXStreamingTranscriber(
                self._parakeet_mlx_model,
                source_lang=source_lang or 'en'
            )
        else:
            logger.warning(f"Streaming transcription not supported for provider: {provider}")
            return None


class ParakeetMLXStreamingTranscriber:
    """Wrapper for Parakeet MLX streaming transcription."""
    
    def __init__(self, model, source_lang='en'):
        self.model = model
        self.source_lang = source_lang
        self.audio_buffer = []
        self.sample_rate = 16000  # Parakeet expects 16kHz
        self.context_window = 3.0  # seconds of context to keep
        
    def add_audio(self, audio_chunk: np.ndarray, sample_rate: int = 16000):
        """
        Add audio chunk to the buffer and transcribe.
        
        Args:
            audio_chunk: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dict with transcription results or None
        """
        # Resample if needed
        if sample_rate != self.sample_rate:
            # Simple resampling - for production use a proper resampling library
            factor = self.sample_rate / sample_rate
            indices = np.arange(0, len(audio_chunk), 1/factor).astype(int)
            indices = indices[indices < len(audio_chunk)]
            audio_chunk = audio_chunk[indices]
        
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        
        # Keep only context_window seconds
        max_samples = int(self.context_window * self.sample_rate)
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]
        
        # Transcribe if we have enough audio (at least 0.5 seconds)
        min_samples = int(0.5 * self.sample_rate)
        if len(self.audio_buffer) >= min_samples:
            try:
                # Convert buffer to numpy array
                audio_array = np.array(self.audio_buffer, dtype=np.float32)
                
                # Transcribe using the model's streaming capability
                # According to the documentation, parakeet-mlx uses transcribe_stream context manager
                # For now, we'll use regular transcribe on the buffer
                result = self.model.transcribe(audio_array)
                
                # Return formatted result
                return {
                    'text': result.text if hasattr(result, 'text') else str(result),
                    'partial': True,
                    'timestamp': time.time()
                }
            except Exception as e:
                logger.error(f"Streaming transcription error: {str(e)}")
                return None
        
        return None
    
    def finalize(self):
        """
        Finalize transcription and return any remaining text.
        
        Returns:
            Dict with final transcription or None
        """
        if len(self.audio_buffer) > 0:
            try:
                audio_array = np.array(self.audio_buffer, dtype=np.float32)
                result = self.model.transcribe(audio_array)
                
                text = result.text if hasattr(result, 'text') else str(result)
                if text is None:
                    text = ""
                
                return {
                    'text': text,
                    'partial': False,
                    'timestamp': time.time()
                }
            except Exception as e:
                logger.error(f"Final transcription error: {str(e)}")
                return None
        
        return None
    
    def reset(self):
        """Reset the audio buffer for a new session."""
        self.audio_buffer = []