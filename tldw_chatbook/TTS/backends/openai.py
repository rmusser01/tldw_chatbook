# openai.py
# Description: OpenAI TTS API backend implementation
#
# Imports
from typing import AsyncGenerator, Optional, Dict, Any
import httpx
import os
from loguru import logger

# Local imports
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.base_backends import (
    APITTSBackend,
    TTSBackendConnectionError,
)
from tldw_chatbook.TTS.openai_compatible_config import (
    OpenAIAuthenticationMode,
    normalize_openai_authentication_mode,
    normalize_openai_compatible_endpoint,
)
from tldw_chatbook.config import get_cli_setting

#######################################################################################################################
#
# OpenAI TTS Backend Implementation


_DEFAULT_OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"


def _http_status_failure(status_code: int) -> tuple[str, ValueError]:
    if status_code == 401:
        return (
            "authentication_failed",
            ValueError(
                "Authentication failed. Please check your API configuration (HTTP 401)."
            ),
        )
    if status_code == 429:
        return (
            "rate_limited",
            ValueError("Rate limit exceeded. Please try again later (HTTP 429)."),
        )
    if status_code >= 500:
        return (
            "service_unavailable",
            ValueError(
                "TTS service temporarily unavailable. Please try again later "
                f"(HTTP {status_code})."
            ),
        )
    return (
        "request_rejected",
        ValueError(f"TTS request was rejected by the service (HTTP {status_code})."),
    )


class OpenAITTSBackend(APITTSBackend):
    """OpenAI Text-to-Speech API backend"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        supplied_config = config or {}
        endpoint = normalize_openai_compatible_endpoint(
            supplied_config.get("OPENAI_BASE_URL", _DEFAULT_OPENAI_TTS_URL)
        )
        authentication_mode = normalize_openai_authentication_mode(
            supplied_config.get("OPENAI_AUTH_MODE"),
            endpoint=endpoint,
        )
        organization_id = str(supplied_config.get("OPENAI_ORG_ID") or "").strip()
        if "\r" in organization_id or "\n" in organization_id:
            raise ValueError("OpenAI organization ID cannot contain line breaks")

        super().__init__(config)

        self.authentication_mode = authentication_mode
        self.api_key = None
        if authentication_mode is OpenAIAuthenticationMode.API_KEY:
            self.api_key = os.getenv("OPENAI_API_KEY")
            logger.debug(
                f"OpenAITTSBackend: Checking env var OPENAI_API_KEY: {'found' if self.api_key else 'not found'}"
            )
            if not self.api_key:
                self.api_key = self.config.get("OPENAI_API_KEY")
                logger.debug(
                    f"OpenAITTSBackend: Checking config dict for OPENAI_API_KEY: {'found' if self.api_key else 'not found'}"
                )
            if not self.api_key:
                from tldw_chatbook.config import load_cli_config_and_ensure_existence

                full_config = load_cli_config_and_ensure_existence()
                api_settings = full_config.get("api_settings", {})
                if isinstance(api_settings, dict):
                    openai_settings = api_settings.get("openai", {})
                    if isinstance(openai_settings, dict):
                        self.api_key = openai_settings.get("api_key")
                logger.debug(
                    f"OpenAITTSBackend: Checking api_settings.openai/api_key: {'found' if self.api_key else 'not found'}"
                )
            if not self.api_key:
                openai_api_settings = get_cli_setting("openai_api")
                if openai_api_settings and isinstance(openai_api_settings, dict):
                    self.api_key = openai_api_settings.get("api_key")
                logger.debug(
                    f"OpenAITTSBackend: Checking openai_api/api_key: {'found' if self.api_key else 'not found'}"
                )
            if not self.api_key:
                api_settings = get_cli_setting("API")
                if api_settings and isinstance(api_settings, dict):
                    self.api_key = api_settings.get("openai_api_key")
                logger.debug(
                    f"OpenAITTSBackend: Checking API/openai_api_key: {'found' if self.api_key else 'not found'}"
                )
            if not self.api_key:
                app_tts_settings = get_cli_setting("app_tts")
                if app_tts_settings and isinstance(app_tts_settings, dict):
                    self.api_key = app_tts_settings.get("OPENAI_API_KEY_fallback")
                logger.debug(
                    f"OpenAITTSBackend: Checking app_tts/OPENAI_API_KEY_fallback: {'found' if self.api_key else 'not found'}"
                )

        self.endpoint = endpoint
        self.base_url = endpoint.speech_url
        self.organization_id = organization_id or None
        # OpenAI-compatible servers (e.g. pocket-tts) define their own models and
        # voices and are typically keyless, so OpenAI-specific constraints only
        # apply when talking to the official endpoint.
        self.is_custom_endpoint = not endpoint.official

        if not self.api_key and authentication_mode is OpenAIAuthenticationMode.API_KEY:
            logger.warning("OpenAITTSBackend: No API key configured")

    async def initialize(self):
        """Initialize the backend"""
        logger.info("OpenAITTSBackend initialized")
        if (
            not self.api_key
            and self.authentication_mode is OpenAIAuthenticationMode.API_KEY
        ):
            logger.warning(
                "OpenAITTSBackend: No API key available. Requests will fail."
            )

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
        if self.authentication_mode is OpenAIAuthenticationMode.API_KEY:
            self._validate_api_key()

        # Validate input text
        if not request.input:
            raise ValueError("Text input is required.")

        # Input length validation (OpenAI has a 4096 character limit)
        if len(request.input) > 4096:
            raise ValueError("Text input exceeds maximum length of 4096 characters.")

        headers = {"Content-Type": "application/json"}
        if (
            self.authentication_mode is OpenAIAuthenticationMode.API_KEY
            and self.api_key
        ):
            headers["Authorization"] = f"Bearer {self.api_key}"
        # The org ID is OpenAI account metadata — never forward it to
        # third-party OpenAI-compatible servers.
        if self.organization_id and not self.is_custom_endpoint:
            headers["OpenAI-Organization"] = self.organization_id

        # Map internal model names to OpenAI model names if needed; custom
        # endpoints define their own model names, so pass them through as-is.
        model = request.model
        if self.is_custom_endpoint or model in ["tts-1", "tts-1-hd"]:
            pass
        else:
            # Default to tts-1 for unknown models
            logger.warning(f"Unknown model '{model}', defaulting to 'tts-1'")
            model = "tts-1"

        # Validate voice selection; custom endpoints define their own voices.
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if self.is_custom_endpoint or request.voice in valid_voices:
            voice = request.voice
        else:
            logger.warning(f"Invalid voice '{request.voice}', defaulting to 'alloy'")
            voice = "alloy"

        # Validate response format
        valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        if request.response_format not in valid_formats:
            logger.warning(
                f"Invalid format '{request.response_format}', defaulting to 'mp3'"
            )
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

        logger.info(
            f"OpenAITTSBackend: Requesting TTS for {len(request.input)} characters"
        )
        logger.debug(
            f"OpenAITTSBackend: Request params: model={model}, voice={voice}, "
            f"format={response_format}, speed={speed}"
        )

        safe_failure: ValueError | None = None
        try:
            async with self.client.stream(
                "POST", self.base_url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()

                # Report that we're receiving data
                await self._report_progress(
                    progress=0.1,
                    processed=0,
                    total=len(request.input),
                    status="Receiving audio from OpenAI",
                )

                # Stream the audio data
                chunk_size = 1024 if response_format == "pcm" else 8192
                total_bytes = 0
                chunk_count = 0

                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    yield chunk
                    total_bytes += len(chunk)
                    chunk_count += 1

                    # Update progress periodically
                    if chunk_count % 10 == 0:
                        # Estimate progress based on typical audio size
                        estimated_progress = min(
                            0.9, total_bytes / (len(request.input) * 100)
                        )
                        await self._report_progress(
                            progress=estimated_progress,
                            processed=total_bytes,
                            total=total_bytes,
                            status=f"Streaming audio: {total_bytes / 1024:.1f} KB",
                            metrics={"chunks": chunk_count},
                        )

                # Report completion
                await self._report_progress(
                    progress=1.0,
                    processed=total_bytes,
                    total=total_bytes,
                    status=f"Completed: {total_bytes / 1024:.1f} KB of {response_format} audio",
                    metrics={"total_bytes": total_bytes, "chunks": chunk_count},
                )

            logger.info("OpenAITTSBackend: Successfully completed TTS generation")

        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            category, safe_failure = _http_status_failure(status_code)
            logger.error(
                "OpenAITTSBackend: Request failed "
                f"(http_status={status_code}, category={category})"
            )

        except httpx.RequestError:
            # Log without exposing connection details
            logger.error("OpenAITTSBackend: Network request failed")
            safe_failure = TTSBackendConnectionError(
                "Unable to connect to TTS service. Please check your internet connection."
            )

        except Exception:
            # Log error without stack trace that might contain sensitive data
            logger.error("OpenAITTSBackend: Unexpected error during TTS generation")
            safe_failure = ValueError(
                "An unexpected error occurred during TTS generation."
            )

        if safe_failure is not None:
            raise safe_failure


#
# End of openai.py
#######################################################################################################################
