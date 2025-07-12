# base_backends.py
# Description: Separate base classes for API and Local TTS backends
#
# Imports
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncGenerator
import httpx
from loguru import logger

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest

#######################################################################################################################
#
# Base Backend Classes

class TTSBackendBase(ABC):
    """Abstract base class for all TTS backends"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    async def initialize(self):
        """Async initialization for the backend (e.g., load models)."""
        pass
    
    @abstractmethod
    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Generates audio for the given text and streams it.
        Should yield bytes of the audio in the request.response_format.
        """
        pass
    
    @abstractmethod
    async def close(self):
        """Clean up resources."""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get backend capabilities"""
        return {
            "streaming": True,
            "formats": ["mp3", "wav", "opus", "aac", "flac", "pcm"],
            "voices": [],
            "models": [],
        }


class APITTSBackend(TTSBackendBase):
    """Base class for API-based TTS backends (OpenAI, ElevenLabs, etc.)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Only API backends get HTTP client
        self.client = httpx.AsyncClient(timeout=60.0)
        self.api_key: Optional[str] = None
        self.base_url: Optional[str] = None
    
    async def close(self):
        """Clean up HTTP client"""
        await self.client.aclose()
    
    def _validate_api_key(self):
        """Validate API key is configured"""
        if not self.api_key:
            logger.error(f"{self.__class__.__name__}: No API key configured")
            raise ValueError("TTS service not configured. Please check your settings.")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get common HTTP headers"""
        return {
            "Content-Type": "application/json",
        }


class LocalTTSBackend(TTSBackendBase):
    """Base class for local TTS backends (Kokoro, Chatterbox, etc.)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Local backends don't need HTTP client
        self.model_path: Optional[str] = None
        self.device: str = self.config.get("device", "cpu")
        self.model_loaded: bool = False
    
    async def close(self):
        """Clean up model resources"""
        # Override in subclasses to unload models
        self.model_loaded = False
    
    def _validate_model_loaded(self):
        """Validate model is loaded"""
        if not self.model_loaded:
            logger.error(f"{self.__class__.__name__}: Model not loaded")
            raise RuntimeError("TTS model not loaded. Please check your configuration.")
    
    @abstractmethod
    async def load_model(self):
        """Load the TTS model into memory"""
        pass


class StreamingTTSMixin:
    """Mixin for streaming audio generation"""
    
    async def stream_with_chunks(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        chunk_size: int = 8192
    ) -> AsyncGenerator[bytes, None]:
        """
        Helper to ensure consistent chunk sizes for streaming.
        
        Args:
            audio_generator: Source audio generator
            chunk_size: Target chunk size in bytes
            
        Yields:
            Audio chunks of consistent size (except last chunk)
        """
        buffer = bytearray()
        
        async for chunk in audio_generator:
            buffer.extend(chunk)
            
            # Yield complete chunks
            while len(buffer) >= chunk_size:
                yield bytes(buffer[:chunk_size])
                buffer = buffer[chunk_size:]
        
        # Yield remaining data
        if buffer:
            yield bytes(buffer)

#
# End of base_backends.py
#######################################################################################################################