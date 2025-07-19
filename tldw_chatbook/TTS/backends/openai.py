# openai.py
# Description: OpenAI TTS API backend implementation
#
# Imports
from typing import AsyncGenerator, Optional, Dict, Any
import json
import httpx
import os
from loguru import logger

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.base_backends import APITTSBackend
from tldw_chatbook.config import get_cli_setting

#######################################################################################################################
#
# OpenAI TTS Backend Implementation

class OpenAITTSBackend(APITTSBackend):
    """OpenAI Text-to-Speech API backend"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Try environment variable first
        self.api_key = os.getenv("OPENAI_API_KEY")
        logger.debug(f"OpenAITTSBackend: Checking env var OPENAI_API_KEY: {'found' if self.api_key else 'not found'}")
        
        if not self.api_key:
            # Try to get API key from config dict
            self.api_key = self.config.get("OPENAI_API_KEY")
            logger.debug(f"OpenAITTSBackend: Checking config dict for OPENAI_API_KEY: {'found' if self.api_key else 'not found'}")
        
        if not self.api_key:
            # Try from api_settings.openai config (new standard location)
            # We need to access the full config and navigate to api_settings.openai.api_key
            from tldw_chatbook.config import load_cli_config_and_ensure_existence
            full_config = load_cli_config_and_ensure_existence()
            api_settings = full_config.get("api_settings", {})
            if isinstance(api_settings, dict):
                openai_settings = api_settings.get("openai", {})
                self.api_key = openai_settings.get("api_key")
            logger.debug(f"OpenAITTSBackend: Checking api_settings.openai/api_key: {'found' if self.api_key else 'not found'}")
            if self.api_key:
                logger.debug(f"OpenAITTSBackend: API key length: {len(self.api_key)}, starts with: {self.api_key[:10] if len(self.api_key) > 10 else self.api_key}")
            
        if not self.api_key:
            # Try from openai_api config (legacy location)
            openai_api_settings = get_cli_setting("openai_api")
            if openai_api_settings and isinstance(openai_api_settings, dict):
                self.api_key = openai_api_settings.get("api_key")
            logger.debug(f"OpenAITTSBackend: Checking openai_api/api_key: {'found' if self.api_key else 'not found'}")
            
        if not self.api_key:
            # Try from CLI config
            api_settings = get_cli_setting("API")
            if api_settings and isinstance(api_settings, dict):
                self.api_key = api_settings.get("openai_api_key")
            logger.debug(f"OpenAITTSBackend: Checking API/openai_api_key: {'found' if self.api_key else 'not found'}")
            
        if not self.api_key:
            # Try from app_tts config
            app_tts_settings = get_cli_setting("app_tts")
            if app_tts_settings and isinstance(app_tts_settings, dict):
                self.api_key = app_tts_settings.get("OPENAI_API_KEY_fallback")
            logger.debug(f"OpenAITTSBackend: Checking app_tts/OPENAI_API_KEY_fallback: {'found' if self.api_key else 'not found'}")
        
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
        # Use base class method to validate API key
        self._validate_api_key()
        
        # Validate input text
        if not request.input:
            raise ValueError("Text input is required.")
        
        # Input length validation (OpenAI has a 4096 character limit)
        if len(request.input) > 4096:
            raise ValueError("Text input exceeds maximum length of 4096 characters.")
        
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
        
        # Validate voice selection
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if request.voice not in valid_voices:
            logger.warning(f"Invalid voice '{request.voice}', defaulting to 'alloy'")
            voice = "alloy"
        else:
            voice = request.voice
        
        # Validate response format
        valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        if request.response_format not in valid_formats:
            logger.warning(f"Invalid format '{request.response_format}', defaulting to 'mp3'")
            response_format = "mp3"
        else:
            response_format = request.response_format
        
        # Validate speed (0.25 to 4.0)
        speed = max(0.25, min(4.0, request.speed))
        if speed != request.speed:
            logger.warning(f"Speed {request.speed} clamped to {speed}")
        
        payload = {
            "model": model,
            "input": request.input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        
        logger.info(f"OpenAITTSBackend: Requesting TTS for {len(request.input)} characters")
        logger.debug(f"OpenAITTSBackend: Request params: model={model}, voice={voice}, "
                    f"format={response_format}, speed={speed}")
        
        try:
            async with self.client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                
                # Stream the audio data
                chunk_size = 1024 if response_format == "pcm" else 8192
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    yield chunk
                    
            logger.info("OpenAITTSBackend: Successfully completed TTS generation")
            
        except httpx.HTTPStatusError as e:
            # Try to read error content safely
            error_msg = f"HTTP {e.response.status_code}"
            error_details = None
            
            # Check if response can still be read
            if hasattr(e.response, 'is_closed') and not e.response.is_closed:
                try:
                    error_content = await e.response.aread()
                    error_details = error_content.decode('utf-8', errors='ignore')
                    
                    # Try to extract meaningful error message
                    try:
                        error_data = json.loads(error_details)
                        if 'error' in error_data and 'message' in error_data['error']:
                            error_msg = error_data['error']['message']
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # Keep the status code message if JSON parsing fails
                        pass
                except Exception as read_error:
                    logger.debug(f"Could not read error response body: {read_error}")
            
            # Log error without exposing sensitive information
            logger.error(f"OpenAI API error {e.response.status_code}: {error_msg}")
            
            # Provide user-friendly error messages
            if e.response.status_code == 401:
                raise ValueError("Authentication failed. Please check your API configuration.")
            elif e.response.status_code == 429:
                raise ValueError("Rate limit exceeded. Please try again later.")
            elif e.response.status_code >= 500:
                raise ValueError("TTS service temporarily unavailable. Please try again later.")
            else:
                raise ValueError(f"TTS request failed: {error_msg}")
            
        except httpx.RequestError as e:
            # Log without exposing connection details
            logger.error(f"OpenAITTSBackend: Network request failed")
            raise ValueError("Unable to connect to TTS service. Please check your internet connection.")
            
        except Exception as e:
            # Log error without stack trace that might contain sensitive data
            logger.error(f"OpenAITTSBackend: Unexpected error during TTS generation")
            raise ValueError("An unexpected error occurred during TTS generation.")

#
# End of openai.py
#######################################################################################################################