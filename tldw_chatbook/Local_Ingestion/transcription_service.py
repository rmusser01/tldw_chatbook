# transcription_service.py
"""
Unified transcription service for tldw_chatbook.
Supports multiple transcription backends including faster-whisper, Qwen2Audio, etc.
"""

import os
import subprocess
import tempfile
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json

import numpy as np

# Local imports  
from ..config import get_cli_setting
from ..Utils.text import sanitize_filename

# Optional imports with graceful degradation
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not available. Install with: pip install faster-whisper")

try:
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    QWEN2AUDIO_AVAILABLE = True
except ImportError:
    QWEN2AUDIO_AVAILABLE = False
    logging.warning("Qwen2Audio not available. Install transformers and torch for Qwen2Audio support.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logging.warning("soundfile not available. Install with: pip install soundfile")

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available. Some audio processing features may be limited.")

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logging.warning("NeMo toolkit not available. Install with: pip install nemo-toolkit[asr]")

logger = logging.getLogger(__name__)


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
        self.config = {
            'default_provider': get_cli_setting('transcription.default_provider', 'faster-whisper'),
            'default_model': get_cli_setting('transcription.default_model', 'base'),
            'default_language': get_cli_setting('transcription.default_language', 'en'),
            'default_source_language': get_cli_setting('transcription.default_source_language', ''),
            'default_target_language': get_cli_setting('transcription.default_target_language', ''),
            'device': get_cli_setting('transcription.device', 'cpu'),
            'compute_type': get_cli_setting('transcription.compute_type', 'int8'),
            'chunk_length_seconds': get_cli_setting('transcription.chunk_length_seconds', 40.0),
        }
        
        # Model cache
        self._model_cache = {}
        
        # Qwen2Audio models (lazy loaded)
        self._qwen_processor = None
        self._qwen_model = None
        
        # Parakeet/NeMo models (lazy loaded)
        self._parakeet_model = None
        
        # Canary model (lazy loaded)
        self._canary_model = None
        self._canary_decoding = None
        
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
            
        Returns:
            Dict with 'text' and 'segments' keys
        """
        provider = provider or self.config['default_provider']
        model = model or self.config['default_model']
        # Handle source language - prefer explicit source_lang over language param
        source_lang = source_lang or self.config['default_source_language'] or language or self.config['default_language']
        # Handle target language
        target_lang = target_lang or self.config['default_target_language'] or None
        # For backward compatibility, set language to source_lang if not specified
        language = language or source_lang
        
        # Convert to WAV if needed
        wav_path = self._ensure_wav_format(audio_path)
        
        try:
            if provider == 'faster-whisper':
                return self._transcribe_with_faster_whisper(
                    wav_path, model, language, vad_filter, source_lang, target_lang, **kwargs
                )
            elif provider == 'qwen2audio':
                return self._transcribe_with_qwen2audio(wav_path, **kwargs)
            elif provider == 'parakeet':
                return self._transcribe_with_parakeet(wav_path, model, source_lang, **kwargs)
            elif provider == 'canary':
                return self._transcribe_with_canary(
                    wav_path, model, source_lang, target_lang, **kwargs
                )
            else:
                raise ValueError(f"Unknown transcription provider: {provider}")
                
        finally:
            # Clean up temporary WAV if created
            if wav_path != audio_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass
    
    def _ensure_wav_format(self, audio_path: str) -> str:
        """
        Convert audio to WAV format if needed.
        
        Returns:
            Path to WAV file (may be same as input if already WAV)
        """
        audio_path = Path(audio_path)
        
        # Check if already WAV
        if audio_path.suffix.lower() == '.wav':
            return str(audio_path)
        
        # Convert to WAV
        wav_path = audio_path.with_suffix('.wav')
        
        # If file exists with same name, use temp file
        if wav_path.exists():
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            wav_path = Path(temp_file.name)
            temp_file.close()
        
        try:
            self._convert_to_wav(str(audio_path), str(wav_path))
            return str(wav_path)
        except Exception as e:
            # Clean up on error
            if wav_path.exists() and wav_path != audio_path:
                wav_path.unlink()
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
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Converted audio to WAV: {output_path}")
            
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
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using faster-whisper."""
        
        if not FASTER_WHISPER_AVAILABLE:
            raise TranscriptionError("faster-whisper is not installed")
        
        # Get or create model instance
        cache_key = (model, self.config['device'], self.config['compute_type'])
        
        if cache_key not in self._model_cache:
            logger.info(f"Loading Whisper model: {model}")
            try:
                self._model_cache[cache_key] = WhisperModel(
                    model,
                    device=self.config['device'],
                    compute_type=self.config['compute_type']
                )
            except Exception as e:
                raise TranscriptionError(f"Failed to load model {model}: {str(e)}") from e
        
        whisper_model = self._model_cache[cache_key]
        
        # Use source_lang if provided, otherwise fall back to language
        transcribe_language = source_lang or language
        
        # Determine task - translate if target language is specified and different from source
        task = 'translate' if target_lang and target_lang == 'en' and transcribe_language != 'en' else 'transcribe'
        
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
            segments_generator, info = whisper_model.transcribe(audio_path, **options)
            
            # Collect segments
            segments = []
            full_text = []
            
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
                full_text.append(segment.text.strip())
            
            # Log detection info
            if info.language:
                logger.info(
                    f"Detected language: {info.language} "
                    f"(confidence: {info.language_probability:.2f})"
                )
            
            result = {
                "text": " ".join(full_text),
                "segments": segments,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "provider": "faster-whisper",
                "model": model,
            }
            
            # Add translation info if applicable
            if task == 'translate':
                result["task"] = "translation"
                result["source_language"] = transcribe_language or info.language
                result["target_language"] = "en"
                result["translation"] = result["text"]
            
            return result
            
        except Exception as e:
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
            raise TranscriptionError(
                "Qwen2Audio dependencies not installed. "
                "Install with: pip install transformers torch"
            )
        
        if not SOUNDFILE_AVAILABLE:
            raise TranscriptionError("soundfile required for Qwen2Audio")
        
        # Lazy load Qwen2Audio models
        if self._qwen_processor is None:
            logger.info("Loading Qwen2Audio model...")
            try:
                self._qwen_processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2-Audio-7B-Instruct"
                )
                self._qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-Audio-7B-Instruct",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
            except Exception as e:
                raise TranscriptionError(
                    f"Failed to load Qwen2Audio: {str(e)}"
                ) from e
        
        try:
            # Load audio
            audio_data, sample_rate = sf.read(audio_path)
            
            # Prepare prompt for transcription
            prompt_text = (
                "System: You are a transcription model.\n"
                "User: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                "Assistant:"
            )
            
            # Process audio
            inputs = self._qwen_processor(
                text=prompt_text,
                audios=audio_data,
                return_tensors="pt",
                sampling_rate=sample_rate
            )
            
            # Move to device
            device = self._qwen_model.device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self._qwen_model.generate(
                    **inputs,
                    max_new_tokens=512
                )
            
            # Decode output
            transcription = self._qwen_processor.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            
            # Extract transcription after "Assistant:"
            if "Assistant:" in transcription:
                transcription = transcription.split("Assistant:")[-1].strip()
            
            # Create segment (Qwen2Audio doesn't provide timestamps)
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
            raise TranscriptionError(
                "NeMo toolkit not installed. "
                "Install with: pip install nemo-toolkit[asr]"
            )
        
        # Lazy load Parakeet model
        if self._parakeet_model is None:
            logger.info(f"Loading Parakeet model: {model or 'nvidia/parakeet-tdt-1.1b'}")
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
                
            except Exception as e:
                raise TranscriptionError(
                    f"Failed to load Parakeet model: {str(e)}"
                ) from e
        
        try:
            # Transcribe the audio
            transcripts = self._parakeet_model.transcribe(
                paths2audio_files=[audio_path],
                batch_size=1,
                return_hypotheses=True,
                verbose=False
            )
            
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
                raise TranscriptionError("No transcription produced")
                
        except Exception as e:
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
        
        if not NEMO_AVAILABLE:
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
            logger.info(f"Loading Canary model: {model or 'nvidia/canary-1b-flash'}")
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
                
            except Exception as e:
                raise TranscriptionError(
                    f"Failed to load Canary model: {str(e)}"
                ) from e
        
        try:
            # Determine task type based on target language
            taskname = "s2t_translation" if target_lang and target_lang != source_lang else "asr"
            
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
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
            manifest_entry["duration"] = duration
            
            # Use chunked inference for long audio
            if duration > chunk_len_in_secs:
                logger.info(f"Using chunked inference for {duration:.1f}s audio")
                
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
                
            else:
                # For short audio, use regular inference
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
            
            # Create response
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
            raise TranscriptionError(
                f"Canary transcription failed: {str(e)}"
            ) from e
    
    def get_available_providers(self) -> List[str]:
        """Get list of available transcription providers based on installed dependencies."""
        providers = []
        
        if FASTER_WHISPER_AVAILABLE:
            providers.append('faster-whisper')
        
        if QWEN2AUDIO_AVAILABLE:
            providers.append('qwen2audio')
            
        if NEMO_AVAILABLE:
            providers.extend(['parakeet', 'canary'])
            
        return providers
    
    def list_available_models(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """List available models for each provider."""
        
        models = {}
        
        if FASTER_WHISPER_AVAILABLE:
            models['faster-whisper'] = [
                'tiny', 'tiny.en',
                'base', 'base.en',
                'small', 'small.en',
                'medium', 'medium.en',
                'large-v1', 'large-v2', 'large-v3',
                'distil-large-v2', 'distil-large-v3',
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
            return {provider: models.get(provider, [])}
        
        return models
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available compute devices."""
        
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
        
        if torch and hasattr(torch.backends, 'mps'):
            info['mps'] = torch.backends.mps.is_available()
        
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