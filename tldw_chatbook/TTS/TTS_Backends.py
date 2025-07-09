# tts_backends.py
# Description: File contains
#
# Imports
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncGenerator
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
from tldw_chatbook.config import load_cli_config_and_ensure_existence, get_cli_config_value

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


# --- Backend Manager (Simplified from target app's ModelManager) ---
class TTSBackendManager:
    def __init__(self, app_config: Dict[str, Any]):
        self.app_config = app_config # Your global app config
        self._backends: Dict[str, TTSBackendBase] = {}
        self._initialized_backends: set[str] = set()

    async def get_backend(self, backend_id: str) -> Optional[TTSBackendBase]:
        if backend_id not in self._backends:
            logger.info(f"TTSBackendManager: Creating backend for ID: {backend_id}")
            # --- Logic to create the correct backend based on backend_id ---
            # This is where you map your internal backend_id (from openai_mappings.json)
            # to the actual backend class.
            specific_config = self.app_config.get(backend_id, {}) # Get backend-specific config
            specific_config.update(self.app_config.get("global_tts_settings", {})) # Merge global settings

            if backend_id == "openai_official_tts-1" or backend_id == "openai_official_tts-1-hd":
                from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend
                self._backends[backend_id] = OpenAITTSBackend(config=specific_config)
            elif backend_id == "local_kokoro_default_onnx":
                from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
                # Example specific config for this Kokoro instance
                kokoro_cfg = {
                    "KOKORO_USE_ONNX": True,
                    "KOKORO_MODEL_PATH": self.app_config.get("KOKORO_ONNX_MODEL_PATH_DEFAULT", "path/to/kokoro.onnx"),
                    "KOKORO_VOICES_JSON_PATH": self.app_config.get("KOKORO_ONNX_VOICES_JSON_DEFAULT", "path/to/voices.json"),
                    "KOKORO_DEVICE": self.app_config.get("KOKORO_DEVICE_DEFAULT", "cpu"),
                }
                kokoro_cfg.update(specific_config) # Allow overrides
                self._backends[backend_id] = KokoroTTSBackend(config=kokoro_cfg)
            # Add elif for ElevenLabs, AllTalk, your PyTorch Kokoro, etc.
            # elif backend_id == "elevenlabs_english_v1":
            #     self._backends[backend_id] = ElevenLabsBackend(config=specific_config)
            else:
                logger.error(f"TTSBackendManager: Unknown backend ID: {backend_id}")
                return None

        backend = self._backends[backend_id]
        if backend_id not in self._initialized_backends:
            logger.info(f"TTSBackendManager: Initializing backend: {backend_id}")
            await backend.initialize()
            self._initialized_backends.add(backend_id)
        return backend

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


#
# End of tts_backends.py
#######################################################################################################################
