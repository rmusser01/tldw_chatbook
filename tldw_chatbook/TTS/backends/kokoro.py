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
from tldw_chatbook.config import get_cli_config_value

#######################################################################################################################
#
# Kokoro TTS Backend Implementation

class KokoroTTSBackend(TTSBackendBase):
    """
    Kokoro Text-to-Speech backend supporting both ONNX and PyTorch models.
    
    References:
    - https://github.com/thewh1teagle/kokoro-onnx
    - https://huggingface.co/hexgrad/Kokoro-82M
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
            self.model_path = get_cli_config_value("app_tts", "KOKORO_ONNX_MODEL_PATH_DEFAULT", 
                                                  "kokoro-v0_19.onnx")
        if not self.voices_json:
            self.voices_json = get_cli_config_value("app_tts", "KOKORO_ONNX_VOICES_JSON_DEFAULT",
                                                   "voices.json")
        
        # Model instances
        self.kokoro_instance = None  # ONNX instance
        self.kokoro_model_pt = None  # PyTorch model
        self.tokenizer = None
        
        # Services
        self.audio_service = get_audio_service()
        self.text_chunker = TextChunker(max_tokens=500)
        self.normalizer = TextNormalizer()
    
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
        """Generate audio using ONNX backend"""
        try:
            # Detect language from voice
            lang = detect_language(request.input, request.voice)
            
            # Normalize text if requested
            text = request.input
            if request.normalization_options:
                text = self.normalizer.normalize_text(text)
            
            logger.info(f"KokoroTTSBackend: Generating audio for {len(text)} characters, "
                       f"voice={request.voice}, lang={lang}, format={request.response_format}")
            
            # For PCM output, we can stream directly
            if request.response_format == "pcm":
                async for samples, sample_rate in self.kokoro_instance.create_stream(
                    text, voice=request.voice, speed=request.speed, lang=lang
                ):
                    # Convert float32 to int16 PCM
                    int16_samples = np.int16(samples * 32767)
                    yield int16_samples.tobytes()
            
            # For other formats, we need to collect and convert
            else:
                # Collect all audio samples
                all_samples = []
                sample_rate = 24000  # Default
                
                async for samples, sr in self.kokoro_instance.create_stream(
                    text, voice=request.voice, speed=request.speed, lang=lang
                ):
                    sample_rate = sr
                    all_samples.append(samples)
                
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
                else:
                    logger.warning("KokoroTTSBackend: No audio generated")
                    yield b""
                    
        except Exception as e:
            logger.error(f"KokoroTTSBackend: Error during ONNX generation: {e}", exc_info=True)
            yield f"ERROR: Kokoro generation failed - {str(e)}".encode('utf-8')
    
    async def close(self):
        """Clean up resources"""
        await super().close()
        # Clean up model instances if needed
        self.kokoro_instance = None
        self.kokoro_model_pt = None


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
}

def map_voice_to_kokoro(voice: str) -> str:
    """Map OpenAI or other voice names to Kokoro voice names"""
    return KOKORO_VOICE_MAP.get(voice, voice)

#
# End of kokoro.py
#######################################################################################################################