# elevenlabs.py
# Description: ElevenLabs TTS API backend implementation
#
# Imports
import tempfile
from typing import AsyncGenerator, Optional, Dict, Any
import httpx
from loguru import logger

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.TTS_Backends import TTSBackendBase
from tldw_chatbook.config import get_cli_setting

#######################################################################################################################
#
# ElevenLabs TTS Backend Implementation

class ElevenLabsTTSBackend(TTSBackendBase):
    """ElevenLabs Text-to-Speech API backend"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Try to get API key from config
        self.api_key = self.config.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            # Try from CLI config
            self.api_key = get_cli_setting("API", "elevenlabs_api_key")
        if not self.api_key:
            # Try from app_tts config
            self.api_key = get_cli_setting("app_tts", "ELEVENLABS_API_KEY_fallback")
        
        # Get voice settings from config
        self.default_voice = self.config.get("ELEVENLABS_DEFAULT_VOICE",
                                            get_cli_setting("app_tts", "ELEVENLABS_DEFAULT_VOICE", "21m00Tcm4TlvDq8ikWAM"))
        self.default_model = self.config.get("ELEVENLABS_DEFAULT_MODEL",
                                           get_cli_setting("app_tts", "ELEVENLABS_DEFAULT_MODEL", "eleven_multilingual_v2"))
        self.output_format = self.config.get("ELEVENLABS_OUTPUT_FORMAT",
                                           get_cli_setting("app_tts", "ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_192"))
        
        # Voice settings with proper type conversion
        self.voice_stability = float(self.config.get("ELEVENLABS_VOICE_STABILITY",
                                                    get_cli_setting("app_tts", "ELEVENLABS_VOICE_STABILITY", "0.5")))
        self.similarity_boost = float(self.config.get("ELEVENLABS_SIMILARITY_BOOST",
                                                     get_cli_setting("app_tts", "ELEVENLABS_SIMILARITY_BOOST", "0.8")))
        self.style = float(self.config.get("ELEVENLABS_STYLE",
                                          get_cli_setting("app_tts", "ELEVENLABS_STYLE", "0.0")))
        
        # Boolean conversion
        speaker_boost_str = self.config.get("ELEVENLABS_USE_SPEAKER_BOOST",
                                          get_cli_setting("app_tts", "ELEVENLABS_USE_SPEAKER_BOOST", "true"))
        self.use_speaker_boost = str(speaker_boost_str).lower() == "true"
        
        self.base_url = "https://api.elevenlabs.io/v1"
        
        if not self.api_key:
            logger.warning("ElevenLabsTTSBackend: No API key configured")
    
    async def initialize(self):
        """Initialize the backend"""
        logger.info("ElevenLabsTTSBackend initialized")
        if not self.api_key:
            logger.warning("ElevenLabsTTSBackend: No API key available. Requests will fail.")
    
    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech using ElevenLabs API and stream the response.
        
        Args:
            request: Speech request parameters
            
        Yields:
            Audio bytes in the requested format
        """
        if not self.api_key:
            logger.error("ElevenLabsTTSBackend: Cannot generate speech without API key")
            yield b"ERROR: ElevenLabs API key not configured"
            return
        
        # Use request voice or default
        voice_id = request.voice if request.voice != "alloy" else self.default_voice
        
        # Map OpenAI format to ElevenLabs format if needed
        output_format = self._map_output_format(request.response_format)
        
        # Construct URL
        url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
        
        headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        # Map model names if needed
        model_id = request.model
        if model_id in ["tts-1", "tts-1-hd", "kokoro"]:
            # Use default ElevenLabs model for non-ElevenLabs models
            model_id = self.default_model
        
        payload = {
            "text": request.input,
            "model_id": model_id,
            "output_format": output_format,
            "voice_settings": {
                "stability": self.voice_stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style,
                "use_speaker_boost": self.use_speaker_boost
            }
        }
        
        # Add optional parameters
        if hasattr(request, 'seed') and request.seed is not None:
            payload["seed"] = request.seed
        
        logger.info(f"ElevenLabsTTSBackend: Requesting TTS for {len(request.input)} characters")
        logger.debug(f"ElevenLabsTTSBackend: Request params: model={model_id}, voice={voice_id}, "
                    f"format={output_format}, stability={self.voice_stability}, "
                    f"similarity={self.similarity_boost}")
        
        try:
            async with self.client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                
                # Stream the audio data
                chunk_size = 1024
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    yield chunk
                    
            logger.info("ElevenLabsTTSBackend: Successfully completed TTS generation")
            
        except httpx.HTTPStatusError as e:
            error_content = await e.response.aread()
            error_msg = error_content.decode('utf-8', errors='ignore')
            logger.error(f"ElevenLabs API HTTP error {e.response.status_code}: {error_msg}")
            
            # Try to extract meaningful error message
            try:
                import json
                error_data = json.loads(error_msg)
                if 'detail' in error_data:
                    if isinstance(error_data['detail'], dict) and 'message' in error_data['detail']:
                        error_msg = error_data['detail']['message']
                    else:
                        error_msg = str(error_data['detail'])
            except:
                pass
            
            yield f"ERROR: ElevenLabs API error - {error_msg}".encode('utf-8')
            
        except httpx.RequestError as e:
            logger.error(f"ElevenLabsTTSBackend: Request error: {e}")
            yield f"ERROR: Failed to connect to ElevenLabs API - {str(e)}".encode('utf-8')
            
        except Exception as e:
            logger.error(f"ElevenLabsTTSBackend: Unexpected error: {e}", exc_info=True)
            yield f"ERROR: Unexpected error - {str(e)}".encode('utf-8')
    
    def _map_output_format(self, format: str) -> str:
        """Map common format names to ElevenLabs format strings"""
        format_map = {
            "mp3": self.output_format,  # Use configured default
            "opus": "opus",
            "aac": "aac",
            "flac": "flac",
            "wav": "pcm_44100",  # WAV is PCM in ElevenLabs
            "pcm": "pcm_44100",
        }
        
        # If format is already an ElevenLabs format string, use it directly
        if "_" in format:  # e.g., "mp3_44100_192"
            return format
        
        return format_map.get(format.lower(), self.output_format)

# Voice ID mapping for common names
ELEVENLABS_VOICE_MAP = {
    # Map OpenAI-style names to ElevenLabs voice IDs (these are examples)
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "domi": "AZnzlk1XvdvUeBnXmlld",
    "bella": "EXAVITQu4vr4xnSDxMaL",
    "antoni": "ErXwobaYiN019PkySvjV",
    "elli": "MF3mGyEYCl7XYWbV9V6O",
    "josh": "TxGEqnHWrfWFTfGW9XjX",
    "arnold": "VR6AewLTigWG4xSOukaG",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "sam": "yoZ06aMxZJJ28mfd3POQ",
}

def get_elevenlabs_voice_id(name: str) -> str:
    """Get ElevenLabs voice ID from a friendly name or return as-is if already an ID"""
    # If it looks like a voice ID (long string), return as-is
    if len(name) > 15:
        return name
    
    # Otherwise try to map from friendly name
    return ELEVENLABS_VOICE_MAP.get(name.lower(), name)

#
# End of elevenlabs.py
#######################################################################################################################