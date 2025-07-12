# chatterbox.py
# Description: Chatterbox TTS backend implementation with streaming support
#
# Imports
import os
import tempfile
from typing import AsyncGenerator, Optional, Dict, Any, List, Tuple
from pathlib import Path
import asyncio
import re
import json
from datetime import datetime
from difflib import SequenceMatcher
from loguru import logger

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.TTS_Backends import TTSBackendBase
from tldw_chatbook.TTS.audio_service import AudioService, get_audio_service
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Utils.optional_deps import check_dependency

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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Core Configuration
        self.device = self.config.get("CHATTERBOX_DEVICE", 
                                     get_cli_setting("app_tts", "CHATTERBOX_DEVICE", "cuda"))
        self.exaggeration = float(self.config.get("CHATTERBOX_EXAGGERATION", 
                                                  get_cli_setting("app_tts", "CHATTERBOX_EXAGGERATION", 0.5)))
        self.cfg_weight = float(self.config.get("CHATTERBOX_CFG_WEIGHT", 
                                               get_cli_setting("app_tts", "CHATTERBOX_CFG_WEIGHT", 0.5)))
        self.chunk_size = int(self.config.get("CHATTERBOX_CHUNK_SIZE", 
                                             get_cli_setting("app_tts", "CHATTERBOX_CHUNK_SIZE", 1024)))
        
        # Extended Configuration
        self.temperature = float(self.config.get("CHATTERBOX_TEMPERATURE", 
                                                get_cli_setting("app_tts", "CHATTERBOX_TEMPERATURE", 0.5)))
        self.random_seed = self.config.get("CHATTERBOX_RANDOM_SEED", 
                                          get_cli_setting("app_tts", "CHATTERBOX_RANDOM_SEED", None))
        self.num_candidates = int(self.config.get("CHATTERBOX_NUM_CANDIDATES", 
                                                 get_cli_setting("app_tts", "CHATTERBOX_NUM_CANDIDATES", 1)))
        self.validate_with_whisper = self.config.get("CHATTERBOX_VALIDATE_WHISPER", 
                                                     get_cli_setting("app_tts", "CHATTERBOX_VALIDATE_WHISPER", False))
        self.preprocess_text_enabled = self.config.get("CHATTERBOX_PREPROCESS_TEXT", 
                                                       get_cli_setting("app_tts", "CHATTERBOX_PREPROCESS_TEXT", True))
        self.normalize_audio_enabled = self.config.get("CHATTERBOX_NORMALIZE_AUDIO", 
                                                       get_cli_setting("app_tts", "CHATTERBOX_NORMALIZE_AUDIO", True))
        self.target_db = float(self.config.get("CHATTERBOX_TARGET_DB", 
                                              get_cli_setting("app_tts", "CHATTERBOX_TARGET_DB", -20.0)))
        self.max_chunk_size = int(self.config.get("CHATTERBOX_MAX_CHUNK_SIZE", 
                                                 get_cli_setting("app_tts", "CHATTERBOX_MAX_CHUNK_SIZE", 500)))
        
        # Voice settings
        self.voice_dir = Path(self.config.get("CHATTERBOX_VOICE_DIR",
                                             get_cli_setting("app_tts", "CHATTERBOX_VOICE_DIR",
                                                           "~/.config/tldw_cli/chatterbox_voices"))).expanduser()
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        
        # Model instances
        self.model = None
        self.audio_service = get_audio_service()
        self.transcription_service = None  # Will be initialized if validation is enabled
        
        # Check dependencies
        self.deps_available = check_dependency("chatterbox", "chatterbox")
        
    async def initialize(self):
        """Initialize the Chatterbox backend"""
        if not self.deps_available:
            logger.warning("ChatterboxTTSBackend: Dependencies not available. Please install with: pip install chatterbox-tts torchaudio")
            return
            
        try:
            # Import Chatterbox
            from chatterbox.tts import ChatterboxTTS
            import torch
            
            # Check CUDA availability
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"
            
            # Load model
            logger.info(f"Loading Chatterbox model on {self.device}...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            logger.info("Chatterbox model loaded successfully")
            
            # Initialize transcription service if validation is enabled
            if self.validate_with_whisper:
                try:
                    from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService
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
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert "J.R.R." to "J R R" for better pronunciation
        # This handles abbreviations with periods between capital letters
        text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', ' ', text)
        
        # Remove inline reference numbers like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove URLs if present
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Convert multiple punctuation to single
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Add space after punctuation if missing
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        logger.debug(f"Text preprocessing complete. Original length: {len(text)}, Processed length: {len(text)}")
        
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
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
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
            rms = torch.sqrt(torch.mean(audio_tensor ** 2))
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
            sample_rate = getattr(self.model, 'sr', 24000)
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
                language=None  # Auto-detect
            )
            
            # Clean up
            os.unlink(tmp_path)
            
            # Extract text from segments
            if isinstance(result, dict) and 'segments' in result:
                text = " ".join(segment.get('text', '') for segment in result['segments'])
                return text.strip()
            elif isinstance(result, str):
                return result.strip()
            else:
                return ""
                
        except Exception as e:
            logger.warning(f"Audio transcription failed: {e}")
            return ""
    
    async def _generate_single(
        self, 
        text: str, 
        reference_audio_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
        temperature: Optional[float] = None
    ) -> bytes:
        """
        Generate a single audio candidate.
        
        Returns:
            Audio bytes in WAV format
        """
        import torch
        
        # Set random seed if specified
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        
        # Apply temperature variation if specified
        if temperature is not None:
            # Slightly vary parameters based on temperature
            exaggeration = exaggeration * (1 + temperature * 0.1)
            cfg_weight = cfg_weight * (1 - temperature * 0.05)
        
        # Generate audio
        if reference_audio_path:
            wav = await asyncio.to_thread(
                self.model.generate,
                text,
                audio_prompt_path=reference_audio_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        else:
            wav = await asyncio.to_thread(
                self.model.generate,
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        
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
        cfg_weight: float
    ) -> bytes:
        """
        Generate multiple candidates and select the best one using Whisper validation.
        
        Returns:
            Best audio candidate in WAV format
        """
        candidates = []
        
        for i in range(self.num_candidates):
            try:
                logger.info(f"Generating candidate {i+1}/{self.num_candidates}")
                
                # Generate with slight variation for each candidate
                temperature = i * 0.2 if self.num_candidates > 1 else 0
                audio_bytes = await self._generate_single(
                    text, reference_audio_path, exaggeration, cfg_weight, temperature
                )
                
                if self.validate_with_whisper and self.transcription_service:
                    # Transcribe and calculate similarity
                    transcript = await self._transcribe_audio(audio_bytes)
                    similarity = self._calculate_similarity(text, transcript)
                    logger.info(f"Candidate {i+1} similarity score: {similarity:.2f}")
                    candidates.append((audio_bytes, similarity, transcript))
                else:
                    # No validation, just add the candidate
                    candidates.append((audio_bytes, 1.0, ""))
                    
            except Exception as e:
                logger.warning(f"Candidate {i+1} generation failed: {e}")
        
        if not candidates:
            raise ValueError("Failed to generate any valid candidates")
        
        # Select best candidate
        best_audio, best_score, best_transcript = max(candidates, key=lambda x: x[1])
        
        if self.validate_with_whisper:
            logger.info(f"Selected best candidate with score {best_score:.2f}")
            if best_score < 0.7:
                logger.warning(f"Low similarity score ({best_score:.2f}). Transcript: '{best_transcript}'")
        
        return best_audio
    
    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech using Chatterbox and stream the response.
        
        Args:
            request: Speech request parameters
            
        Yields:
            Audio bytes in the requested format
        """
        if not self.model:
            logger.error("ChatterboxTTSBackend: Model not initialized")
            raise ValueError("Chatterbox model not initialized. Please check installation.")
        
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
        elif voice != "default":
            # Check for predefined voice
            predefined_path = self.voice_dir / f"{voice}.wav"
            if predefined_path.exists():
                reference_audio_path = str(predefined_path)
        
        try:
            # Get extra parameters if available
            exaggeration = self.exaggeration
            cfg_weight = self.cfg_weight
            temperature = self.temperature
            num_candidates = self.num_candidates
            validate_with_whisper = self.validate_with_whisper
            
            # Check for custom parameters in the request
            if hasattr(request, 'extra_params') and request.extra_params:
                exaggeration = request.extra_params.get('exaggeration', exaggeration)
                cfg_weight = request.extra_params.get('cfg_weight', cfg_weight)
                temperature = request.extra_params.get('temperature', temperature)
                num_candidates = request.extra_params.get('num_candidates', num_candidates)
                validate_with_whisper = request.extra_params.get('validate_with_whisper', validate_with_whisper)
            
            logger.info(f"Generating speech with Chatterbox (exaggeration={exaggeration}, cfg_weight={cfg_weight}, candidates={num_candidates})")
            
            # Check if text needs chunking
            if len(text) > self.max_chunk_size:
                logger.info(f"Text length ({len(text)}) exceeds max chunk size ({self.max_chunk_size}), chunking...")
                chunks = self.chunk_text(text)
                
                # Generate audio for each chunk
                all_audio_bytes = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    # Generate audio for chunk
                    if num_candidates > 1 or validate_with_whisper:
                        audio_bytes = await self.generate_with_validation(
                            chunk, reference_audio_path, exaggeration, cfg_weight
                        )
                    else:
                        audio_bytes = await self._generate_single(
                            chunk, reference_audio_path, exaggeration, cfg_weight, temperature
                        )
                    
                    all_audio_bytes.append(audio_bytes)
                
                # Combine all chunks
                # TODO: Implement crossfade for smoother transitions
                combined_audio = b''.join(all_audio_bytes)
                
                # Convert format if needed
                if request.response_format != "wav":
                    output_bytes = await self.audio_service.convert_audio_format(
                        combined_audio, "wav", request.response_format
                    )
                else:
                    output_bytes = combined_audio
                
                yield output_bytes
                
            else:
                # Single chunk generation
                if num_candidates > 1 or validate_with_whisper:
                    # Use multi-candidate generation with validation
                    audio_bytes = await self.generate_with_validation(
                        text, reference_audio_path, exaggeration, cfg_weight
                    )
                else:
                    # Use single generation
                    audio_bytes = await self._generate_single(
                        text, reference_audio_path, exaggeration, cfg_weight, temperature
                    )
                
                # Convert format if needed
                if request.response_format != "wav":
                    output_bytes = await self.audio_service.convert_audio_format(
                        audio_bytes, "wav", request.response_format
                    )
                else:
                    output_bytes = audio_bytes
                
                yield output_bytes
                
        except Exception as e:
            logger.error(f"Chatterbox generation failed: {e}")
            # Try fallback strategy if available
            if hasattr(self, 'generate_speech_stream_with_fallback'):
                logger.info("Attempting fallback generation strategy...")
                async for chunk in self.generate_speech_stream_with_fallback(request):
                    yield chunk
            else:
                raise ValueError(f"Failed to generate speech: {str(e)}")
    
    async def _generate_stream_async(self, text: str, audio_prompt_path: Optional[str], 
                                   exaggeration: float, cfg_weight: float) -> AsyncGenerator:
        """Async wrapper for streaming generation"""
        # Run the synchronous generator in a thread
        loop = asyncio.get_event_loop()
        
        # Define a generator function
        def sync_generator():
            if audio_prompt_path:
                return self.model.generate_stream(
                    text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    chunk_size=self.chunk_size
                )
            else:
                return self.model.generate_stream(
                    text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    chunk_size=self.chunk_size
                )
        
        # Execute in thread and yield results
        generator = await loop.run_in_executor(None, sync_generator)
        
        for chunk, metrics in generator:
            yield chunk, metrics
    
    def _tensor_to_wav_bytes(self, tensor, sample_rate: int) -> bytes:
        """Convert PyTorch tensor to WAV bytes"""
        import io
        import torchaudio
        
        # Create a bytes buffer
        buffer = io.BytesIO()
        
        # Ensure tensor is on CPU and has correct shape
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Add batch dimension if needed
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        # Save to buffer
        torchaudio.save(buffer, tensor, sample_rate, format="wav")
        
        # Get bytes
        buffer.seek(0)
        return buffer.read()
    
    async def list_voices(self) -> list[str]:
        """List available voices (predefined and custom)"""
        voices = ["default"]
        
        # Add predefined voices from voice directory
        if self.voice_dir.exists():
            for voice_file in self.voice_dir.glob("*.wav"):
                voices.append(voice_file.stem)
        
        return voices
    
    async def save_reference_voice(self, name: str, audio_path: str) -> bool:
        """Save a reference audio file as a predefined voice"""
        try:
            import shutil
            
            # Validate audio file
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            # Copy to voice directory
            dest_path = self.voice_dir / f"{name}.wav"
            shutil.copy2(audio_path, dest_path)
            
            logger.info(f"Saved voice '{name}' to {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save reference voice: {e}")
            return False
    
    async def save_reference_voice_with_metadata(
        self, 
        name: str, 
        audio_path: str, 
        metadata: Optional[Dict[str, Any]] = None
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
            import shutil
            
            # Validate audio file
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            # Save audio file
            voice_path = self.voice_dir / f"{name}.wav"
            shutil.copy2(audio_path, voice_path)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "created_at": datetime.now().isoformat(),
                "audio_file": f"{name}.wav",
                "file_size": os.path.getsize(audio_path),
                "original_path": audio_path
            })
            
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
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved voice '{name}' with metadata to {voice_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save voice with metadata: {e}")
            return False
    
    async def list_voices_with_metadata(self) -> List[Dict[str, Any]]:
        """
        List available voices with their metadata.
        
        Returns:
            List of voice dictionaries with metadata
        """
        voices = []
        
        # Add default voice
        voices.append({
            "name": "default",
            "has_metadata": False,
            "metadata": {}
        })
        
        # Add predefined voices from voice directory
        if self.voice_dir.exists():
            for voice_file in self.voice_dir.glob("*.wav"):
                voice_name = voice_file.stem
                voice_info = {
                    "name": voice_name,
                    "file_path": str(voice_file),
                    "has_metadata": False,
                    "metadata": {}
                }
                
                # Check for metadata file
                metadata_path = self.voice_dir / f"{voice_name}_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
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
            ("high_quality", {"exaggeration": 0.5, "cfg_weight": 0.5, "num_candidates": 3}),
            ("balanced", {"exaggeration": 0.3, "cfg_weight": 0.7, "num_candidates": 2}),
            ("safe", {"exaggeration": 0.1, "cfg_weight": 0.9, "num_candidates": 1}),
        ]
        
        original_params = {
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
            "num_candidates": self.num_candidates
        }
        
        for strategy_name, params in strategies:
            try:
                logger.info(f"Trying {strategy_name} generation strategy")
                
                # Update parameters
                self.exaggeration = params["exaggeration"]
                self.cfg_weight = params["cfg_weight"]
                self.num_candidates = params["num_candidates"]
                
                # Attempt generation
                async for chunk in self.generate_speech_stream(request):
                    yield chunk
                
                # Success - restore original params and exit
                for key, value in original_params.items():
                    setattr(self, key, value)
                return
                
            except Exception as e:
                logger.warning(f"{strategy_name} strategy failed: {e}")
                continue
        
        # Restore original parameters
        for key, value in original_params.items():
            setattr(self, key, value)
        
        # All strategies failed
        raise ValueError("All generation strategies failed")

#
# End of chatterbox.py
#######################################################################################################################