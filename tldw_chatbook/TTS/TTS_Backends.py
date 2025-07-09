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




# --- Concrete Backend for Your Local ro ---
class LocalKokoroBackend(TTSBackendBase):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Config for your local Kokoro: model path, voice dir, ONNX/PyTorch, device
        self.use_onnx = self.config.get("KOKORO_USE_ONNX", True)
        self.kokoro_model_path = self.config.get("KOKORO_MODEL_PATH", "kokoro-v0_19.onnx") # or .pth
        self.kokoro_voices_json = self.config.get("KOKORO_VOICES_JSON_PATH", "voices.json") # For ONNX
        self.kokoro_voice_dir = self.config.get("KOKORO_VOICE_DIR_PT", "App_Function_Libraries/TTS/Kokoro/voices") # For PyTorch
        self.device = self.config.get("KOKORO_DEVICE", "cpu")
        self.kokoro_instance = None # For ONNX Kokoro library
        self.kokoro_model_pt = None # For PyTorch model
        self.tokenizer = None

        # Borrow text processing from the target app (if you want its advanced features)
        # from target_app.services.text_processing import smart_split as target_smart_split
        # from target_app.services.audio import AudioService as TargetAudioService
        # from target_app.services.streaming_audio_writer import StreamingAudioWriter as TargetSAW
        # self.smart_split = target_smart_split
        # self.audio_service = TargetAudioService
        # self.streaming_audio_writer_class = TargetSAW


    async def initialize(self):
        logger.info(f"LocalKokoroBackend: Initializing (ONNX: {self.use_onnx}, Device: {self.device})")
        if self.use_onnx:
            try:
                # Import kokoro_onnx if available
                try:
                    from kokoro_onnx import Kokoro, EspeakConfig
                except ImportError:
                    logger.error("kokoro_onnx not installed. Please install with: pip install kokoro-onnx")
                    raise
                # Ensure model files exist, download if not (like in your TTS_Providers_Local.py)
                if not os.path.exists(self.kokoro_model_path):
                    logger.error(f"Kokoro ONNX model not found at {self.kokoro_model_path}")
                    # Add download logic here if needed
                    return
                if not os.path.exists(self.kokoro_voices_json):
                    logger.error(f"Kokoro voices.json not found at {self.kokoro_voices_json}")
                    # Add download logic here if needed
                    return

                espeak_lib = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
                self.kokoro_instance = Kokoro(
                    self.kokoro_model_path,
                    self.kokoro_voices_json,
                    espeak_config=EspeakConfig(lib_path=espeak_lib) if espeak_lib else None
                )
                logger.info("LocalKokoroBackend: ONNX instance created.")
            except ImportError:
                logger.error("LocalKokoroBackend: kokoro_onnx library not found. ONNX mode unavailable.")
                self.use_onnx = False # Fallback or error
            except Exception as e:
                logger.error(f"LocalKokoroBackend: Error initializing ONNX Kokoro: {e}", exc_info=True)
                self.use_onnx = False
        else: # PyTorch mode (adapt your existing logic)
            try:
                # from PoC_Version.App_Function_Libraries.TTS.Kokoro.models import build_model # Your model loader
                # from transformers import AutoTokenizer # For chunking
                # import torch

                # self.kokoro_model_pt = get_kokoro_model(device=self.device) # Your existing function
                # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                logger.info("LocalKokoroBackend: PyTorch model/tokenizer placeholder setup.")
                # Actual loading of PyTorch model would happen here using your existing functions
                pass
            except ImportError:
                logger.error("LocalKokoroBackend: PyTorch dependencies (e.g., torch, transformers) not found.")
            except Exception as e:
                logger.error(f"LocalKokoroBackend: Error initializing PyTorch Kokoro: {e}", exc_info=True)
        logger.info("LocalKokoroBackend initialized.")


    async def _generate_with_kokoro_onnx(self, request: OpenAISpeechRequest) -> AsyncGenerator[bytes, None]:
        if not self.kokoro_instance:
            logger.error("LocalKokoroBackend (ONNX): Not initialized.")
            yield b"ERROR: Kokoro ONNX backend not initialized"
            return

        # Your TTS_Providers_Local.py uses `kokoro.create_stream` which is an async generator.
        # It yields (samples, sr). We need to convert these samples to the target `response_format`.
        # This is where the target app's StreamingAudioWriter and AudioService would be very handy.

        # For simplicity now, let's assume WAV output and directly yield, then you can adapt.
        # A more robust solution would use something like target app's StreamingAudioWriter.
        # Example:
        # saw = self.streaming_audio_writer_class(request.response_format, sample_rate=24000)

        lang = 'en-us' # Determine from voice, e.g., request.voice == "af_bella" -> 'en-us'
        if request.voice.startswith('a'): # Simple heuristic
            lang = 'en-us'
        elif request.voice.startswith('b'):
            lang = 'en-gb'
        else: # Fallback or derive from a lang_code field if you add it
            lang = 'en-us'

        # This part needs the most adaptation to fit how you want to handle audio format conversion
        # The `kokoro_onnx.create_stream` yields (samples_np_array, sample_rate)
        # You need to convert `samples_np_array` to the `request.response_format`
        logger.info(f"Kokoro ONNX: Streaming for voice '{request.voice}', lang '{lang}'")
        try:
            # Using the target app's StreamingAudioWriter pattern:
            # saw = TargetSAW(format=request.response_format, sample_rate=24000) # Assuming Kokoro outputs 24kHz

            # This is a simplified placeholder.
            # For real audio conversion, integrate the target app's StreamingAudioWriter
            # or a similar library that can encode raw PCM (from Kokoro) to MP3/Opus etc. in chunks.
            if request.response_format == "wav": # Simplest case, can write WAV header and then PCM
                 # Yield WAV header first (simplified)
                # This is a very basic header and might not be fully correct for all players.
                # A proper WAV library should be used.
                # For streaming, you'd ideally use a library that handles the format correctly.
                # For now, we'll simulate by collecting all and then encoding.
                # This negates true streaming benefits for formats other than PCM.
                all_samples = []
                sample_rate = 24000 # Assume this from Kokoro

                async for samples_chunk, sr_chunk in self.kokoro_instance.create_stream(
                    request.input, voice=request.voice, speed=request.speed, lang=lang
                ):
                    sample_rate = sr_chunk # Update sample rate from first chunk
                    all_samples.append(samples_chunk)

                if all_samples:
                    combined_samples = np.concatenate(all_samples)
                    # Now convert combined_samples to WAV bytes
                    # This is NOT true streaming for WAV but demonstrates the flow.
                    # You would use `scipy.io.wavfile.write` to a BytesIO buffer.
                    import io
                    import scipy.io.wavfile
                    byte_io = io.BytesIO()
                    # Kokoro ONNX typically outputs float32, scale to int16 for WAV
                    scaled_samples = np.int16(combined_samples * 32767)
                    scipy.io.wavfile.write(byte_io, sample_rate, scaled_samples)
                    yield byte_io.getvalue()
                else:
                    yield b"" # No audio generated
            elif request.response_format == "pcm":
                 async for samples_chunk, sr_chunk in self.kokoro_instance.create_stream(
                    request.input, voice=request.voice, speed=request.speed, lang=lang
                ):
                    # Kokoro samples are float32. PCM for OpenAI often means S16LE.
                    # Convert to int16 bytes
                    int16_samples = np.int16(samples_chunk * 32767)
                    yield int16_samples.tobytes()
            else:
                # For MP3, Opus, etc., you NEED a streaming encoder.
                # The target app's StreamingAudioWriter using pyav is ideal here.
                # Placeholder:
                logger.warning(f"Streaming {request.response_format} from LocalKokoroBackend is non-trivial without a streaming encoder. Yielding placeholder.")
                yield f"Placeholder for {request.response_format} stream. Implement proper encoding.".encode()

        except Exception as e:
            logger.error(f"LocalKokoroBackend (ONNX) error during generation: {e}", exc_info=True)
            yield b"ERROR: Kokoro ONNX generation failed"


    async def _generate_with_kokoro_pytorch(self, request: OpenAISpeechRequest) -> AsyncGenerator[bytes, None]:
        # This would adapt your existing PyTorch Kokoro logic.
        # It involves:
        # 1. Text chunking (e.g., your `split_text_into_sentence_chunks` or target app's `smart_split`)
        # 2. For each chunk:
        #    - Call your `generate(MODEL, chunk_text, VOICEPACK, ...)`
        #    - Convert the output tensor to numpy array
        #    - Convert this raw audio to `request.response_format` bytes (again, StreamingAudioWriter is good)
        #    - Yield the bytes
        logger.warning("LocalKokoroBackend (PyTorch): generate_speech_stream not fully implemented.")
        # Example of yielding chunks (you'd replace this with actual generation and encoding)
        # text_chunks = self.smart_split(request.input, lang_code=...) # if using target app's
        # for chunk_text in text_chunks:
        #     # raw_audio_np = your_pytorch_kokoro_generate(chunk_text, ...)
        #     # encoded_bytes = your_streaming_encoder_instance.write_chunk(raw_audio_np)
        #     # yield encoded_bytes
        # # encoded_bytes = your_streaming_encoder_instance.write_chunk(finalize=True)
        # # yield encoded_bytes
        yield b"PyTorch Kokoro stream placeholder"


    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        logger.info(f"LocalKokoroBackend: Generating speech. ONNX: {self.use_onnx}")
        if self.use_onnx:
            async for chunk in self._generate_with_kokoro_onnx(request):
                yield chunk
        else:
            async for chunk in self._generate_with_kokoro_pytorch(request):
                yield chunk

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
                # Example specific config for this Kokoro instance
                kokoro_cfg = {
                    "KOKORO_USE_ONNX": True,
                    "KOKORO_MODEL_PATH": self.app_config.get("KOKORO_ONNX_MODEL_PATH_DEFAULT", "path/to/kokoro.onnx"),
                    "KOKORO_VOICES_JSON_PATH": self.app_config.get("KOKORO_ONNX_VOICES_JSON_DEFAULT", "path/to/voices.json"),
                    "KOKORO_DEVICE": self.app_config.get("KOKORO_DEVICE_DEFAULT", "cpu"),
                }
                kokoro_cfg.update(specific_config) # Allow overrides
                self._backends[backend_id] = LocalKokoroBackend(config=kokoro_cfg)
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
