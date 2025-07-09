# tts_backends.py
# Description: File contains
#
# Imports
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncGenerator, List
#
# Third Party Libraries
import httpx
import numpy as np
from loguru import logger
#
# Local Libraries
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
#
#######################################################################################################################
#
# Functions

# FIXME - placheolder for TTS backend

# --- Load your existing config for API keys etc. ---
from tldw_chatbook.config import load_cli_config_and_ensure_existence, get_cli_setting

# --- Abstract Base Class for TTS Backends ---
class TTSBackendBase(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.client = httpx.AsyncClient(timeout=60.0) # Shared client for API backends

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

    async def close(self):
        """Clean up resources, like closing the httpx client."""
        await self.client.aclose()





# --- Add more backends: ElevenLabsBackend, AllTalkBackend etc. ---
# FIXME
# class ElevenLabsBackend(TTSBackendBase):
#// ... implementation using your generate_audio_elevenlabs logic ...
# Remember to adapt it to be async and yield bytes. Httpx can stream responses.


# --- Backend Registry ---
class BackendRegistry:
    """Registry for TTS backend classes"""
    _registry: Dict[str, type[TTSBackendBase]] = {}
    
    @classmethod
    def register(cls, backend_id: str, backend_class: type[TTSBackendBase]):
        """Register a backend class"""
        cls._registry[backend_id] = backend_class
        logger.info(f"Registered TTS backend: {backend_id} -> {backend_class.__name__}")
    
    @classmethod
    def get(cls, backend_id: str) -> Optional[type[TTSBackendBase]]:
        """Get a backend class by ID"""
        # Try exact match first
        if backend_id in cls._registry:
            return cls._registry[backend_id]
        
        # Try prefix matching (e.g., "elevenlabs_*" matches any elevenlabs backend)
        for registered_id, backend_class in cls._registry.items():
            if backend_id.startswith(registered_id.rstrip('*')):
                return backend_class
        
        return None
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backend IDs"""
        return list(cls._registry.keys())


# --- Backend Manager ---
class TTSBackendManager:
    def __init__(self, app_config: Dict[str, Any]):
        self.app_config = app_config # Your global app config
        self._backends: Dict[str, TTSBackendBase] = {}
        self._initialized_backends: set[str] = set()
        
        # Register built-in backends
        self._register_builtin_backends()

    def _register_builtin_backends(self):
        """Register built-in backend implementations"""
        # Lazy imports to avoid circular dependencies
        try:
            from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend
            BackendRegistry.register("openai_official_*", OpenAITTSBackend)
        except ImportError:
            logger.warning("OpenAI TTS backend not available")
        
        try:
            from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
            BackendRegistry.register("local_kokoro_*", KokoroTTSBackend)
        except ImportError:
            logger.warning("Kokoro TTS backend not available")
        
        try:
            from tldw_chatbook.TTS.backends.elevenlabs import ElevenLabsTTSBackend
            BackendRegistry.register("elevenlabs_*", ElevenLabsTTSBackend)
        except ImportError:
            logger.warning("ElevenLabs TTS backend not available")
    
    async def get_backend(self, backend_id: str) -> Optional[TTSBackendBase]:
        if backend_id not in self._backends:
            logger.info(f"TTSBackendManager: Creating backend for ID: {backend_id}")
            
            # Get backend class from registry
            backend_class = BackendRegistry.get(backend_id)
            if not backend_class:
                logger.error(f"TTSBackendManager: No backend registered for ID: {backend_id}")
                return None
            
            # Prepare configuration
            specific_config = self._prepare_backend_config(backend_id)
            
            # Create backend instance
            try:
                self._backends[backend_id] = backend_class(config=specific_config)
            except Exception as e:
                logger.error(f"TTSBackendManager: Failed to create backend {backend_id}: {e}")
                return None

        backend = self._backends[backend_id]
        if backend_id not in self._initialized_backends:
            logger.info(f"TTSBackendManager: Initializing backend: {backend_id}")
            await backend.initialize()
            self._initialized_backends.add(backend_id)
        return backend
    
    def _prepare_backend_config(self, backend_id: str) -> Dict[str, Any]:
        """Prepare configuration for a specific backend"""
        # Get backend-specific config
        specific_config = self.app_config.get(backend_id, {})
        
        # Merge with global TTS settings
        specific_config.update(self.app_config.get("global_tts_settings", {}))
        
        # Add app_tts config section
        if "app_tts" in self.app_config:
            specific_config.update(self.app_config["app_tts"])
        
        # Special handling for specific backends
        if backend_id.startswith("local_kokoro"):
            kokoro_defaults = {
                "KOKORO_USE_ONNX": True,
                "KOKORO_MODEL_PATH": self.app_config.get("KOKORO_ONNX_MODEL_PATH_DEFAULT", "models/kokoro-v0_19.onnx"),
                "KOKORO_VOICES_JSON_PATH": self.app_config.get("KOKORO_ONNX_VOICES_JSON_DEFAULT", "models/voices.json"),
                "KOKORO_DEVICE": self.app_config.get("KOKORO_DEVICE_DEFAULT", "cpu"),
                "KOKORO_MAX_TOKENS": self.app_config.get("KOKORO_MAX_TOKENS", 500),
                "KOKORO_ENABLE_VOICE_MIXING": self.app_config.get("KOKORO_ENABLE_VOICE_MIXING", False),
            }
            # Let specific config override defaults
            kokoro_defaults.update(specific_config)
            specific_config = kokoro_defaults
        
        return specific_config
    
    def list_available_backends(self) -> List[str]:
        """List all available backend IDs"""
        return BackendRegistry.list_backends()

    async def close_all_backends(self):
        logger.info("TTSBackendManager: Closing all backends.")
        for backend_id, backend in self._backends.items():
            try:
                logger.info(f"Closing backend: {backend_id}")
                await backend.close()
            except Exception as e:
                logger.error(f"Error closing backend {backend_id}: {e}")
        self._backends.clear()
        self._initialized_backends.clear()
    
    def get_backend_info(self, backend_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a backend"""
        if backend_id in self._backends:
            backend = self._backends[backend_id]
            return {
                "id": backend_id,
                "class": backend.__class__.__name__,
                "initialized": backend_id in self._initialized_backends,
                "capabilities": getattr(backend, "capabilities", {})
            }
        return None


#
# End of tts_backends.py
#######################################################################################################################
