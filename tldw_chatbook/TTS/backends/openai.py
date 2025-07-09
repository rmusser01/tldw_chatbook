# openai.py
# Description: OpenAI TTS API backend implementation
#
# Imports
from typing import AsyncGenerator, Optional, Dict, Any
import httpx
from loguru import logger

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.TTS_Backends import TTSBackendBase
from tldw_chatbook.config import get_cli_config_value

#######################################################################################################################
#
# OpenAI TTS Backend Implementation

class OpenAITTSBackend(TTSBackendBase):
    """OpenAI Text-to-Speech API backend"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Try to get API key from config
        self.api_key = self.config.get("OPENAI_API_KEY")
        if not self.api_key:
            # Try from CLI config
            self.api_key = get_cli_config_value("API", "openai_api_key")
        if not self.api_key:
            # Try from app_tts config
            self.api_key = get_cli_config_value("app_tts", "OPENAI_API_KEY_fallback")
        
        self.base_url = "https://api.openai.com/v1/audio/speech"
        
        if not self.api_key:
            logger.warning("OpenAITTSBackend: No API key configured")
    
    async def initialize(self):
        """Initialize the backend"""
        logger.info("OpenAITTSBackend initialized")
        if not self.api_key:
            logger.warning("OpenAITTSBackend: No API key available. Requests will fail.")
    
    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech using OpenAI's API and stream the response.
        
        Args:
            request: OpenAI speech request parameters
            
        Yields:
            Audio bytes in the requested format
        """
        if not self.api_key:
            logger.error("OpenAITTSBackend: Cannot generate speech without API key")
            yield b"ERROR: OpenAI API key not configured"
            return
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Map internal model names to OpenAI model names if needed
        model = request.model
        if model in ["tts-1", "tts-1-hd"]:
            # These are already OpenAI model names
            pass
        else:
            # Default to tts-1 for unknown models
            logger.warning(f"Unknown model '{model}', defaulting to 'tts-1'")
            model = "tts-1"
        
        payload = {
            "model": model,
            "input": request.input,
            "voice": request.voice,
            "response_format": request.response_format,
            "speed": request.speed,
        }
        
        logger.info(f"OpenAITTSBackend: Requesting TTS for {len(request.input)} characters")
        logger.debug(f"OpenAITTSBackend: Request params: model={model}, voice={request.voice}, "
                    f"format={request.response_format}, speed={request.speed}")
        
        try:
            async with self.client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                
                # Stream the audio data
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    yield chunk
                    
            logger.info("OpenAITTSBackend: Successfully completed TTS generation")
            
        except httpx.HTTPStatusError as e:
            error_content = await e.response.aread()
            error_msg = error_content.decode('utf-8', errors='ignore')
            logger.error(f"OpenAI API HTTP error {e.response.status_code}: {error_msg}")
            
            # Try to extract meaningful error message
            try:
                import json
                error_data = json.loads(error_msg)
                if 'error' in error_data and 'message' in error_data['error']:
                    error_msg = error_data['error']['message']
            except:
                pass
            
            yield f"ERROR: OpenAI API error - {error_msg}".encode('utf-8')
            
        except httpx.RequestError as e:
            logger.error(f"OpenAITTSBackend: Request error: {e}")
            yield f"ERROR: Failed to connect to OpenAI API - {str(e)}".encode('utf-8')
            
        except Exception as e:
            logger.error(f"OpenAITTSBackend: Unexpected error: {e}", exc_info=True)
            yield f"ERROR: Unexpected error - {str(e)}".encode('utf-8')

#
# End of openai.py
#######################################################################################################################