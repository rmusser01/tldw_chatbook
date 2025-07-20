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
from loguru import logger
try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("numpy not available - some TTS features may be limited")

#
# Local Libraries
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
#
#######################################################################################################################
#
# Functions

# --- Load your existing config for API keys etc. ---
from tldw_chatbook.config import load_cli_config_and_ensure_existence, get_cli_setting

# Import new base classes
from tldw_chatbook.TTS.base_backends import TTSBackendBase, APITTSBackend, LocalTTSBackend







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
        
        try:
            from tldw_chatbook.TTS.backends.chatterbox import ChatterboxTTSBackend
            BackendRegistry.register("local_chatterbox_*", ChatterboxTTSBackend)
        except ImportError:
            logger.warning("Chatterbox TTS backend not available")
        
        try:
            from tldw_chatbook.TTS.backends.alltalk import AllTalkTTSBackend
            BackendRegistry.register("alltalk_*", AllTalkTTSBackend)
        except ImportError:
            logger.warning("AllTalk TTS backend not available")
    
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
        # Start with global TTS settings as base
        config = self.app_config.get("global_tts_settings", {}).copy()
        
        # Add app_tts config section (general TTS settings)
        if "app_tts" in self.app_config:
            config.update(self.app_config["app_tts"])
        
        # Special handling for specific backends - set defaults first
        if backend_id.startswith("openai_official"):
            # Add OpenAI API key from various sources
            import os
            openai_key = None
            
            # Check environment variable
            openai_key = os.getenv("OPENAI_API_KEY")
            
            # Check api_settings.openai section
            if not openai_key and "api_settings.openai" in self.app_config:
                openai_key = self.app_config["api_settings.openai"].get("api_key")
            
            # Check openai_api section (legacy)
            if not openai_key and "openai_api" in self.app_config:
                openai_key = self.app_config["openai_api"].get("api_key")
            
            # Check API section
            if not openai_key and "API" in self.app_config:
                openai_key = self.app_config["API"].get("openai_api_key")
                
            if openai_key:
                config["OPENAI_API_KEY"] = openai_key
                
        elif backend_id.startswith("local_kokoro"):
            # Get Kokoro-specific paths from environment or config
            import os
            
            # Determine if this is PyTorch or ONNX variant
            use_onnx = not backend_id.endswith("_pytorch")
            
            kokoro_defaults = {
                "KOKORO_USE_ONNX": use_onnx,
                "KOKORO_MODEL_PATH": os.getenv("KOKORO_MODEL_PATH", 
                    self.app_config.get("KOKORO_ONNX_MODEL_PATH_DEFAULT", "models/kokoro-v0_19.onnx") if use_onnx
                    else self.app_config.get("KOKORO_PT_MODEL_PATH_DEFAULT", "models/kokoro-v0_19.pth")),
                "KOKORO_VOICES_JSON_PATH": os.getenv("KOKORO_VOICES_PATH",
                    self.app_config.get("KOKORO_ONNX_VOICES_JSON_DEFAULT", "models/voices.json")),
                "KOKORO_DEVICE": self.app_config.get("KOKORO_DEVICE_DEFAULT", "cpu"),
                "KOKORO_MAX_TOKENS": self.app_config.get("KOKORO_MAX_TOKENS", 500),
                "KOKORO_ENABLE_VOICE_MIXING": self.app_config.get("KOKORO_ENABLE_VOICE_MIXING", False),
            }
            config.update(kokoro_defaults)
        
        # Finally, apply backend-specific config overrides (highest priority)
        backend_specific = self.app_config.get(backend_id, {})
        config.update(backend_specific)
        
        return config
    
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
