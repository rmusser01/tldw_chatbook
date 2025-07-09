# kokoro.py
# Description: Kokoro TTS backend implementation supporting both ONNX and PyTorch
#
# Imports
import os
from typing import AsyncGenerator, Optional, Dict, Any
import numpy as np
from loguru import logger

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.TTS_Backends import TTSBackendBase
from tldw_chatbook.TTS.audio_service import AudioService, get_audio_service
from tldw_chatbook.TTS.text_processing import TextChunker, TextNormalizer, detect_language
from tldw_chatbook.config import get_cli_setting

#######################################################################################################################
#
# Kokoro TTS Backend Implementation

class KokoroTTSBackend(TTSBackendBase):
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
        
        # Configuration
        self.use_onnx = self.config.get("KOKORO_USE_ONNX", True)
        self.model_path = self.config.get("KOKORO_MODEL_PATH")
        self.voices_json = self.config.get("KOKORO_VOICES_JSON_PATH")
        self.voice_dir = self.config.get("KOKORO_VOICE_DIR_PT")
        self.device = self.config.get("KOKORO_DEVICE", "cpu")
        
        # Try to get paths from CLI config if not provided
        if not self.model_path:
            self.model_path = get_cli_setting("app_tts", "KOKORO_ONNX_MODEL_PATH_DEFAULT", 
                                                  "kokoro-v0_19.onnx")
        if not self.voices_json:
            self.voices_json = get_cli_setting("app_tts", "KOKORO_ONNX_VOICES_JSON_DEFAULT",
                                                   "voices.json")
        
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
        
        if self.use_onnx:
            await self._initialize_onnx()
        else:
            await self._initialize_pytorch()
    
    async def _initialize_onnx(self):
        """Initialize ONNX backend"""
        try:
            # Try to import kokoro_onnx
            try:
                from kokoro_onnx import Kokoro, EspeakConfig
            except ImportError:
                logger.error("kokoro_onnx not installed. Please install with: pip install kokoro-onnx")
                self.use_onnx = False
                return
            
            # Check if model files exist
            if not os.path.exists(self.model_path):
                logger.error(f"Kokoro ONNX model not found at {self.model_path}")
                # TODO: Add auto-download logic here
                self.use_onnx = False
                return
            
            if not os.path.exists(self.voices_json):
                logger.error(f"Kokoro voices.json not found at {self.voices_json}")
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
    
    async def _initialize_pytorch(self):
        """Initialize PyTorch backend"""
        logger.warning("KokoroTTSBackend: PyTorch backend not yet implemented")
        # TODO: Implement PyTorch initialization
        pass
    
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
        if self.use_onnx and self.kokoro_instance:
            async for chunk in self._generate_onnx_stream(request):
                yield chunk
        else:
            # Fallback or PyTorch implementation
            logger.error("KokoroTTSBackend: No working backend available")
            yield b"ERROR: Kokoro backend not properly initialized"
    
    async def _generate_onnx_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio using ONNX backend with advanced features"""
        import time
        start_time = time.time()
        
        try:
            # Parse voice for potential mixing
            voice_config = self._parse_voice_config(request.voice)
            
            # Detect language from voice
            lang = detect_language(request.input, voice_config['primary_voice'])
            
            # Normalize text if requested
            text = request.input
            if request.normalization_options:
                text = self.normalizer.normalize_text(text)
            
            logger.info(f"KokoroTTSBackend: Generating audio for {len(text)} characters, "
                       f"voice={voice_config}, lang={lang}, format={request.response_format}")
            
            # For PCM output, we can stream directly
            if request.response_format == "pcm":
                token_count = 0
                if voice_config['is_mixed']:
                    # Generate mixed voice audio
                    async for samples, sample_rate in self._generate_mixed_voice(
                        text, voice_config, speed=request.speed, lang=lang
                    ):
                        token_count += len(samples) // 256  # Approximate token count
                        # Convert float32 to int16 PCM
                        int16_samples = np.int16(samples * 32767)
                        yield int16_samples.tobytes()
                else:
                    # Single voice generation
                    async for samples, sample_rate in self.kokoro_instance.create_stream(
                        text, voice=voice_config['primary_voice'], speed=request.speed, lang=lang
                    ):
                        token_count += len(samples) // 256  # Approximate token count
                        # Convert float32 to int16 PCM
                        int16_samples = np.int16(samples * 32767)
                        yield int16_samples.tobytes()
                
                # Update performance metrics
                if self.track_performance:
                    self._update_performance_metrics(token_count, time.time() - start_time)
            
            # For other formats, we need to collect and convert
            else:
                # Collect all audio samples
                all_samples = []
                sample_rate = 24000  # Default
                token_count = 0
                
                if voice_config['is_mixed']:
                    # Generate mixed voice audio
                    async for samples, sr in self._generate_mixed_voice(
                        text, voice_config, speed=request.speed, lang=lang
                    ):
                        sample_rate = sr
                        all_samples.append(samples)
                        token_count += len(samples) // 256
                else:
                    # Single voice generation
                    async for samples, sr in self.kokoro_instance.create_stream(
                        text, voice=voice_config['primary_voice'], speed=request.speed, lang=lang
                    ):
                        sample_rate = sr
                        all_samples.append(samples)
                        token_count += len(samples) // 256
                
                if all_samples:
                    # Concatenate all samples
                    combined_samples = np.concatenate(all_samples)
                    
                    # Convert to requested format
                    audio_bytes = self.audio_service.convert_audio_sync(
                        combined_samples,
                        request.response_format,
                        sample_rate=sample_rate
                    )
                    
                    yield audio_bytes
                    
                    # Update performance metrics
                    if self.track_performance:
                        self._update_performance_metrics(token_count, time.time() - start_time)
                    
                    # Log performance info
                    generation_time = time.time() - start_time
                    tokens_per_second = token_count / generation_time if generation_time > 0 else 0
                    audio_duration = len(combined_samples) / sample_rate
                    speed_factor = audio_duration / generation_time if generation_time > 0 else 0
                    
                    logger.info(f"KokoroTTSBackend: Generated {audio_duration:.2f}s of audio in {generation_time:.2f}s "
                               f"({speed_factor:.1f}x realtime, {tokens_per_second:.1f} tokens/s)")
                else:
                    logger.warning("KokoroTTSBackend: No audio generated")
                    yield b""
                    
        except Exception as e:
            logger.error(f"KokoroTTSBackend: Error during ONNX generation: {e}", exc_info=True)
            yield f"ERROR: Kokoro generation failed - {str(e)}".encode('utf-8')
    
    def _parse_voice_config(self, voice_str: str) -> Dict[str, Any]:
        """Parse voice string for potential mixing configuration"""
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