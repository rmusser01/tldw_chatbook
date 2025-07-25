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
        
        # Model configuration
        self.model_path = self.config.get("HIGGS_MODEL_PATH", 
                                         get_cli_setting("HiggsSettings", "model_path",
                                                       "bosonai/higgs-audio-v2-generation-3B-base"))
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
        
        # Create default voice profiles if none exist
        if not self.voice_profiles:
            self._create_default_profiles()
        
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
        logger.info(f"HiggsAudioTTSBackend: Initializing with model {self.model_path}")
        await self.load_model()
    
    async def load_model(self):
        """Load the Higgs Audio model"""
        try:
            # Import Higgs modules
            if self._boson_multimodal is None:
                try:
                    import boson_multimodal
                    self._boson_multimodal = boson_multimodal
                    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
                    self._higgs_serve_engine = HiggsAudioServeEngine
                except ImportError as e:
                    raise ImportError(
                        "boson-multimodal is required for Higgs Audio backend but is not installed. "
                        "Install it with: pip install boson-multimodal"
                    ) from e
            
            # Convert dtype string to torch dtype
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)
            
            # Create serve engine with configuration
            logger.info(f"Loading Higgs Audio model from {self.model_path}")
            self.serve_engine = self._higgs_serve_engine(
                model_name_or_path=self.model_path,
                device=self.device,
                enable_flash_attn=self.enable_flash_attn,
                dtype=torch_dtype
            )
            
            self.model_loaded = True
            logger.info(f"HiggsAudioTTSBackend: Model loaded successfully on {self.device}")
            
        except Exception as e:
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
        # Ensure model is loaded
        if not self.model_loaded:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Parse voice configuration
            voice_config = await self._prepare_voice_config(request.voice)
            
            # Detect or use specified language
            language = self._get_language(request)
            
            # Normalize text if requested
            text = request.input
            if request.normalization_options:
                text = self.normalizer.normalize_text(text)
            
            logger.info(f"HiggsAudioTTSBackend: Generating audio for {len(text)} characters, "
                       f"voice={voice_config.get('display_name', 'custom')}, "
                       f"language={language}, format={request.response_format}")
            
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
            
            # Generate audio using Higgs
            output = await asyncio.to_thread(
                self.serve_engine.generate,
                messages=messages,
                max_new_tokens=self.config.get("HIGGS_MAX_NEW_TOKENS", 4096),
                temperature=self.config.get("HIGGS_TEMPERATURE", 0.7),
                top_p=self.config.get("HIGGS_TOP_P", 0.9),
                repetition_penalty=self.config.get("HIGGS_REPETITION_PENALTY", 1.1)
            )
            
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
                
                logger.info(f"HiggsAudioTTSBackend: Generated {audio_duration:.2f}s of audio in {generation_time:.2f}s "
                           f"({speed_factor:.1f}x realtime, first chunk: {first_chunk_time:.3f}s)")
                
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
                logger.error("HiggsAudioTTSBackend: No audio generated from model")
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
                
                output = await asyncio.to_thread(
                    self.serve_engine.generate,
                    messages=messages,
                    max_new_tokens=self.config.get("HIGGS_MAX_NEW_TOKENS", 4096)
                )
                
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
        parts = text.split(self.speaker_delimiter)
        
        if len(parts) < 2:
            # No speaker markers found
            return [("narrator", text)]
        
        # First part might be narrator text before any speaker
        if parts[0].strip() and not any(parts[0].endswith(f"{speaker}") for speaker in self.voice_profiles):
            sections.append(("narrator", parts[0].strip()))
            start_idx = 1
        else:
            start_idx = 0
        
        # Parse speaker sections
        i = start_idx
        while i < len(parts) - 1:
            # Extract speaker name from end of previous part
            speaker_text = parts[i].strip()
            words = speaker_text.split()
            
            if words:
                # Last word is the speaker name
                speaker = words[-1]
                # Remove speaker name from text
                if i > 0:
                    remaining_text = " ".join(words[:-1])
                    if remaining_text and i > start_idx:
                        # Add remaining text to previous section
                        if sections:
                            prev_speaker, prev_text = sections[-1]
                            sections[-1] = (prev_speaker, prev_text + " " + remaining_text)
                
                # Add new section
                section_text = parts[i + 1].strip()
                if section_text:
                    sections.append((speaker, section_text))
            
            i += 1
        
        return sections
    
    async def _prepare_voice_config(self, voice_name: str) -> Dict[str, Any]:
        """Prepare voice configuration from voice name or profile"""
        # Check if it's a voice profile
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
        return {
            "type": "default",
            "display_name": "Default Higgs Voice",
            "language": self.default_language
        }
    
    def _prepare_messages(self, text: str, voice_config: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
        """Prepare messages for Higgs Audio generation"""
        messages = []
        
        # Add system message with voice and language configuration
        system_content = f"Generate speech in {language} language."
        
        if voice_config.get("type") == "profile" and voice_config.get("metadata"):
            # Add voice characteristics from profile
            metadata = voice_config["metadata"]
            if "description" in metadata:
                system_content += f" Voice description: {metadata['description']}"
            if "style" in metadata:
                system_content += f" Speaking style: {metadata['style']}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add reference audio if available
        if voice_config.get("reference_audio"):
            ref_audio_path = voice_config["reference_audio"]
            if os.path.exists(ref_audio_path):
                # Load reference audio
                try:
                    if TORCHAUDIO_AVAILABLE:
                        waveform, sample_rate = torchaudio.load(ref_audio_path)
                        # Resample to 24kHz if needed
                        if sample_rate != 24000:
                            resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                            waveform = resampler(waveform)
                        
                        # Convert to mono if stereo
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                        
                        # Limit duration
                        max_samples = int(self.max_reference_duration * 24000)
                        if waveform.shape[1] > max_samples:
                            waveform = waveform[:, :max_samples]
                        
                        messages.append({
                            "role": "user",
                            "content": "Use this voice as reference:",
                            "audio": waveform.numpy()
                        })
                    else:
                        logger.warning(f"Cannot load reference audio without torchaudio: {ref_audio_path}")
                except Exception as e:
                    logger.error(f"Failed to load reference audio: {e}")
        
        # Add the text to generate
        messages.append({
            "role": "user", 
            "content": text
        })
        
        # Add background music if enabled
        if self.enable_background_music and self.config.get("HIGGS_BACKGROUND_MUSIC_PATH"):
            music_path = self.config["HIGGS_BACKGROUND_MUSIC_PATH"]
            if os.path.exists(music_path):
                try:
                    # Load background music
                    music_waveform, music_sr = torchaudio.load(music_path)
                    if music_sr != 24000:
                        resampler = torchaudio.transforms.Resample(music_sr, 24000)
                        music_waveform = resampler(music_waveform)
                    
                    messages.append({
                        "role": "assistant",
                        "content": "Adding background music",
                        "background_audio": music_waveform.numpy()
                    })
                except Exception as e:
                    logger.error(f"Failed to load background music: {e}")
        
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
                    logger.warning(f"Failed to remove reference audio: {e}")
            
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
        """Clean up resources"""
        await super().close()
        
        # Clean up model
        if self.serve_engine is not None:
            try:
                # Higgs serve engine cleanup if needed
                self.serve_engine = None
                torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
            except Exception as e:
                logger.warning(f"Error during Higgs cleanup: {e}")
        
        # Log final performance stats
        if self.track_performance and self._performance_metrics['generation_count'] > 0:
            stats = self.get_performance_stats()
            logger.info(f"HiggsAudioTTSBackend: Final stats - "
                       f"{stats['total_generations']} generations in {stats['total_time']:.1f}s, "
                       f"{stats['voice_profiles_created']} voice profiles created")
        
        self.model_loaded = False

#
# End of higgs.py
#######################################################################################################################