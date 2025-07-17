# TTS_Generation.py
# Description: This module handles the text-to-speech (TTS) generation process.
#
# Imports
from typing import AsyncGenerator, Optional, Dict, Any
#
# Third-party Imports
import asyncio # For semaphore
#
# Local Imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager, TTSBackendBase
#
#######################################################################################################################
#
# Functions:



# For logging
import logging
logger = logging.getLogger(__name__)

class TTSService:
    """
    TTS Service orchestrator that manages backend selection and audio generation.
    
    Rate Limiting Strategy:
    - Global semaphore limits concurrent TTS generations to 4 across all backends
    - This prevents overwhelming system resources and API rate limits
    - Individual backends may have their own rate limiting as well
    """
    
    # Global rate limiting: Maximum 4 concurrent TTS generations
    # This prevents resource exhaustion and helps respect API rate limits
    _backend_semaphore = asyncio.Semaphore(4)

    def __init__(self, backend_manager: TTSBackendManager):
        self.backend_manager = backend_manager

    async def generate_audio_stream(
        self, request: OpenAISpeechRequest, internal_model_id: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Orchestrates fetching the backend and calling its stream generation.
        Handles text processing (chunking, normalization) if the backend expects it.
        Handles audio format conversion if the backend produces raw audio.
        """
        backend: Optional[TTSBackendBase] = await self.backend_manager.get_backend(internal_model_id)
        if not backend:
            logger.error(f"TTSService: No backend found for internal_model_id: {internal_model_id}")
            # Don't expose internal model IDs to users
            raise ValueError(f"TTS model '{request.model}' is not available. Please choose a different model.")

        logger.info(f"TTSService: Using backend {type(backend).__name__} for model '{request.model}' (internal: {internal_model_id})")
        try:
            async with self._backend_semaphore:
                 async for audio_bytes_chunk in backend.generate_speech_stream(request):
                    yield audio_bytes_chunk
        except Exception as e:
            logger.error(f"TTSService: Error streaming from backend {type(backend).__name__}: {e}", exc_info=True)
            # Decide how to propagate: re-raise, or yield an error marker if the protocol supports it.
            # Raising here will likely lead to the StreamingResponse stopping and client getting an error.
            raise # Re-raise to be caught by the main endpoint handler

# --- Singleton pattern for TTSService (simplified for single-user app) ---
_tts_service_instance: Optional[TTSService] = None
_tts_backend_manager_instance: Optional[TTSBackendManager] = None

async def get_tts_service(app_config: Optional[Dict[str, Any]] = None) -> TTSService:
    """Get the singleton TTSService instance.
    
    Args:
        app_config: Configuration dict required on first initialization
        
    Returns:
        TTSService singleton instance
        
    Raises:
        ValueError: If app_config is not provided on first initialization
    """
    global _tts_service_instance, _tts_backend_manager_instance
    
    # Simple check for single-user app
    if _tts_service_instance is not None:
        return _tts_service_instance
            
    if app_config is None:
        raise ValueError("TTSService requires app_config on first initialization.")
    
    # Create instances
    try:
        _tts_backend_manager_instance = TTSBackendManager(app_config=app_config)
        _tts_service_instance = TTSService(backend_manager=_tts_backend_manager_instance)
        logger.info("TTSService initialized successfully.")
    except Exception as e:
        # Clean up on initialization failure
        _tts_backend_manager_instance = None
        _tts_service_instance = None
        logger.error(f"Failed to initialize TTSService: {e}")
        raise
            
    return _tts_service_instance

async def close_tts_resources():
    """Call this during application shutdown.
    
    Properly closes all TTS resources including backends and cleans up singletons.
    """
    global _tts_backend_manager_instance, _tts_service_instance
    
    if _tts_backend_manager_instance:
        try:
            logger.info("Closing TTS backend resources...")
            await _tts_backend_manager_instance.close_all_backends()
            logger.info("TTS backend resources closed successfully.")
        except Exception as e:
            logger.error(f"Error closing TTS backend resources: {e}")
        finally:
            # Always clear the instances to allow re-initialization
            _tts_backend_manager_instance = None
            _tts_service_instance = None

#
# End of tts_generation.py
#######################################################################################################################
