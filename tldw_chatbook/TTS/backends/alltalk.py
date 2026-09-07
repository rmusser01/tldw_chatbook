# alltalk.py
# Description: AllTalk TTS API backend implementation (OpenAI-compatible)
#
# Imports
import logging
from contextlib import contextmanager
from typing import AsyncGenerator, Optional, Dict, Any
import httpx
from loguru import logger

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.base_backends import APITTSBackend

_DEFAULT_BASE_URL = "http://127.0.0.1:7851"
_HTTP_LOGGER_NAMES = ("httpx", "httpcore")


class _AllTalkHTTPPrivacyFilter(logging.Filter):
    def filter(self, _record: logging.LogRecord) -> bool:
        return False

#######################################################################################################################
#
# AllTalk TTS Backend Implementation
#


class AllTalkTTSBackend(APITTSBackend):
    """
    AllTalk Text-to-Speech API backend (OpenAI-compatible).

    AllTalk is a local TTS server that provides an OpenAI-compatible API endpoint.
    It supports multiple voices and languages with customizable output formats.

    Features:
    - OpenAI-compatible API at /v1/audio/speech
    - Multiple voice files support (e.g., female_01.wav, male_01.wav)
    - Language selection
    - Various output formats (wav, mp3, opus, etc.)
    - No API key required (local server)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.base_url = (
            self.config.get("ALLTALK_TTS_URL")
            or self.config.get("ALLTALK_TTS_URL_DEFAULT")
            or _DEFAULT_BASE_URL
        ).rstrip("/")

        self.default_voice = self.config.get(
            "ALLTALK_TTS_VOICE",
            self.config.get("ALLTALK_TTS_VOICE_DEFAULT", "female_01.wav"),
        )

        self.default_language = self.config.get(
            "ALLTALK_TTS_LANGUAGE",
            self.config.get("ALLTALK_TTS_LANGUAGE_DEFAULT", "en"),
        )

        self.output_format = self.config.get(
            "ALLTALK_TTS_OUTPUT_FORMAT",
            self.config.get("ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT", "wav"),
        )

        # AllTalk uses OpenAI-compatible endpoint
        self.endpoint = f"{self.base_url}/v1/audio/speech"

        # AllTalk doesn't require API key, but we keep the field for compatibility
        self.api_key = None
        self._http_privacy_filter = _AllTalkHTTPPrivacyFilter()
        self._http_log_suppression_users = 0

        logger.info("AllTalk backend configured (provider=alltalk)")

    @contextmanager
    def _suppress_http_logs(self):
        self._http_log_suppression_users += 1
        if self._http_log_suppression_users == 1:
            for logger_name in _HTTP_LOGGER_NAMES:
                logging.getLogger(logger_name).addFilter(
                    self._http_privacy_filter
                )
        try:
            yield
        finally:
            self._http_log_suppression_users -= 1
            if self._http_log_suppression_users == 0:
                for logger_name in _HTTP_LOGGER_NAMES:
                    logging.getLogger(logger_name).removeFilter(
                        self._http_privacy_filter
                    )

    async def initialize(self):
        """Initialize the backend and test connection"""
        logger.info("AllTalk backend initialization started (provider=alltalk)")

        # Test connection to AllTalk server
        try:
            # Try to fetch available voices
            voices_url = f"{self.base_url}/api/voices"
            with self._suppress_http_logs():
                async with self.client.get(voices_url, timeout=5.0) as response:
                    if response.status_code == 200:
                        try:
                            voices_data = response.json()
                            if isinstance(voices_data, list):
                                logger.info("AllTalk voices loaded (provider=alltalk)")
                            else:
                                logger.info("AllTalk server responded with voices data")
                        except Exception:
                            logger.debug("AllTalk voices response was invalid")
                    else:
                        logger.warning(
                            "AllTalk voices request rejected "
                            f"(status={response.status_code})"
                        )
        except httpx.ConnectError:
            logger.warning("AllTalk connection unavailable (provider=alltalk)")
        except Exception:
            logger.warning("AllTalk initialization failed (provider=alltalk)")

    def _validate_api_key(self):
        """Override parent method - AllTalk doesn't need API key"""
        # AllTalk is a local server and doesn't require authentication
        pass

    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech using AllTalk API and stream the response.

        Args:
            request: Speech request parameters

        Yields:
            Audio bytes in the requested format
        """

        # Validate input text
        if not request.input:
            raise ValueError("Text input is required.")

        # AllTalk may have different length limits than OpenAI
        # Using a reasonable default limit
        if len(request.input) > 10000:
            raise ValueError("Text input exceeds maximum length of 10000 characters.")

        # Map voice to AllTalk format
        voice = request.voice
        if not voice.endswith(".wav"):
            # If voice doesn't have .wav extension, try to map it
            voice = self._map_voice_to_alltalk(voice)

        # Map response format
        response_format = request.response_format
        if response_format == "pcm":
            # AllTalk might not support raw PCM, use WAV instead
            response_format = "wav"
            logger.debug("AllTalk: Converting PCM request to WAV format")

        # AllTalk OpenAI-compatible payload
        # Note: AllTalk ignores the model parameter but it's required by OpenAI spec
        payload = {
            "model": "tts-1",  # Required by OpenAI spec, ignored by AllTalk
            "input": request.input,
            "voice": voice,
            "response_format": response_format,
            "speed": request.speed,
        }

        # Add AllTalk-specific parameters if available
        if hasattr(request, "language") and request.language:
            payload["language"] = request.language
        else:
            payload["language"] = self.default_language

        headers = {
            "Content-Type": "application/json",
        }

        logger.info("AllTalk generation started (provider=alltalk)")

        try:
            with self._suppress_http_logs():
                async with self.client.stream(
                    "POST", self.endpoint, headers=headers, json=payload, timeout=60.0
                ) as response:
                    response.raise_for_status()

                    chunk_size = 8192
                    async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                        yield chunk

            logger.info("AllTalk generation completed (provider=alltalk)")

        except httpx.ConnectError:
            logger.error("AllTalk connection failed (provider=alltalk)")
            raise ValueError(
                "Could not connect to AllTalk server. Please ensure it's running."
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                "AllTalk request rejected (provider=alltalk, status={})",
                e.response.status_code,
            )
            if e.response.status_code == 404:
                raise ValueError(
                    "AllTalk API endpoint not found. Please check AllTalk version and configuration."
                ) from None
            elif e.response.status_code == 500:
                raise ValueError("AllTalk server error.") from None
            else:
                raise ValueError(
                    f"AllTalk TTS request failed (HTTP {e.response.status_code})."
                ) from None
        except httpx.TimeoutException:
            logger.error("AllTalk request timed out")
            raise ValueError(
                "AllTalk request timed out. The server may be overloaded or the text may be too long."
            )
        except ValueError:
            raise
        except Exception:
            logger.error("AllTalk generation failed (provider=alltalk)")
            raise ValueError("AllTalk TTS generation failed.") from None

    def _map_voice_to_alltalk(self, voice: str) -> str:
        """
        Map common voice names to AllTalk voice file names.

        Args:
            voice: Input voice name

        Returns:
            AllTalk voice file name
        """
        # Common voice mappings
        voice_map = {
            # OpenAI-style voices to AllTalk voices
            "alloy": "female_01.wav",
            "echo": "male_01.wav",
            "fable": "female_02.wav",
            "onyx": "male_02.wav",
            "nova": "female_03.wav",
            "shimmer": "female_04.wav",
            # Gender-based mappings
            "female": "female_01.wav",
            "male": "male_01.wav",
            "woman": "female_01.wav",
            "man": "male_01.wav",
            # Numbered mappings
            "voice1": "female_01.wav",
            "voice2": "male_01.wav",
            "voice3": "female_02.wav",
            "voice4": "male_02.wav",
            # Default
            "default": self.default_voice,
        }

        mapped_voice = voice_map.get(voice.lower(), voice)

        # Ensure .wav extension if not present
        if not mapped_voice.endswith(".wav"):
            mapped_voice = f"{mapped_voice}.wav"

        return mapped_voice

    async def list_voices(self) -> list[str]:
        """
        Get list of available voices from AllTalk server.

        Returns:
            List of available voice names
        """
        try:
            voices_url = f"{self.base_url}/api/voices"
            with self._suppress_http_logs():
                async with self.client.get(voices_url, timeout=5.0) as response:
                    if response.status_code == 200:
                        voices = response.json()
                        if isinstance(voices, list):
                            return voices
                        else:
                            logger.warning(
                                f"Unexpected voices response format: {type(voices)}"
                            )
                            return [self.default_voice]
                    else:
                        logger.warning(
                            "AllTalk voices request rejected "
                            f"(status={response.status_code})"
                        )
                        return [self.default_voice]
        except Exception:
            logger.error("AllTalk voice discovery failed (provider=alltalk)")
            return [self.default_voice]

    def get_capabilities(self) -> Dict[str, Any]:
        """Get backend capabilities"""
        return {
            "streaming": True,
            "formats": [
                "wav",
                "mp3",
                "opus",
                "aac",
                "flac",
            ],  # AllTalk supported formats
            "voices": [],  # Will be populated dynamically
            "models": ["alltalk"],  # AllTalk doesn't use models
            "languages": [
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cs",
                "ar",
                "zh",
                "ja",
                "ko",
            ],  # Common AllTalk languages
            "max_length": 10000,
            "requires_api_key": False,
            "local": True,  # This is a local service
        }


#
# End of alltalk.py
#######################################################################################################################
