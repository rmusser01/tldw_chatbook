"""Bounded discovery adapter for an external audio.cpp HTTP server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

import httpx

from tldw_chatbook.TTS.adapter_types import (
    ProgressSink,
    ProviderHealth,
    TTSAudioResponse,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSRequest,
)
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_contract import (
    AudioCppContractError,
    parse_health_response,
    parse_models_response,
    parse_voices_response,
)

_PROVIDER_ID = "audio_cpp"
_TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_GET_ATTEMPTS = 2
_HTTP_LOGGER_NAMES = ("httpx", "httpcore")

_INITIAL_HEALTH = ProviderHealth(
    state="unavailable",
    fresh=False,
    diagnostic="The audio.cpp server is unavailable",
    retryable=True,
    recovery_action="retry",
)
_AVAILABLE_HEALTH = ProviderHealth(state="available", fresh=True)
_NOT_CONFIGURED_HEALTH = ProviderHealth(
    state="not_configured",
    fresh=True,
    diagnostic="No audio.cpp TTS models are configured",
    recovery_action="check_server",
)
_TRANSIENT_FAILURE_HEALTH = _INITIAL_HEALTH
_CONTRACT_FAILURE_HEALTH = ProviderHealth(
    state="unavailable",
    fresh=False,
    diagnostic="The audio.cpp server response is incompatible",
    recovery_action="check_server",
)
_CLOSED_HEALTH = ProviderHealth(
    state="closed",
    fresh=False,
    diagnostic="The audio.cpp adapter is closed",
)


class _TransientHttpFailure(Exception):
    """Internal value-free marker for a retryable safe-GET failure."""


class _HttpContractFailure(Exception):
    """Internal value-free marker for a non-retryable HTTP contract failure."""


class _HttpxPrivacyFilter(logging.Filter):
    """Suppress httpx's value-bearing request summary for this origin."""

    def __init__(self, origin: str) -> None:
        super().__init__()
        self._origin = httpx.URL(origin)
        self._origin_prefix = origin.rstrip("/")

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and any(
            isinstance(value, httpx.URL)
            and value.scheme == self._origin.scheme
            and value.host == self._origin.host
            and value.port == self._origin.port
            for value in args
        ):
            return False
        message = record.getMessage()
        return self._origin_prefix not in message and self._origin.host not in message


class AudioCppAdapter:
    """Discover bounded model and voice metadata from one external server.

    Construction creates the owned HTTP client but performs no network I/O.
    Speech synthesis is intentionally deferred to the next implementation
    slice.

    Args:
        config: Validated immutable external audio.cpp configuration.
        transport: Optional fake HTTP transport for deterministic tests.
    """

    def __init__(
        self,
        config: AudioCppConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            transport=transport,
            follow_redirects=False,
            trust_env=False,
            headers={"Accept-Encoding": "identity"},
            timeout=httpx.Timeout(
                connect=config.connect_timeout_seconds,
                read=None,
                write=None,
                pool=None,
            ),
        )
        self._catalog = TTSProviderCatalog(
            provider_id=_PROVIDER_ID,
            revision=0,
            health=_INITIAL_HEALTH,
            models=(),
        )
        self._refresh_lock = asyncio.Lock()
        self._refresh_generation = 0
        self._voice_cache: dict[tuple[int, str], tuple[str, ...]] = {}
        self._voice_generation: dict[tuple[int, str], int] = {}
        self._voice_locks: dict[tuple[int, str], asyncio.Lock] = {}
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._httpx_privacy_filter = _HttpxPrivacyFilter(config.base_url)
        for logger_name in _HTTP_LOGGER_NAMES:
            logging.getLogger(logger_name).addFilter(self._httpx_privacy_filter)

    async def ensure_ready(self) -> None:
        """Perform the first authoritative refresh and cache fresh readiness."""
        await self._refresh_catalog(force=False)

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        """Return the immutable catalog, optionally forcing one refresh."""
        await self._refresh_catalog(force=refresh)
        return self._catalog

    async def get_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> tuple[str, ...]:
        """Return optional voices for one exact model in the current catalog."""
        await self.ensure_ready()
        force = refresh

        while True:
            catalog = self._catalog
            if self._closed or not self._catalog_contains(catalog, model_id):
                return ()

            key = (catalog.revision, model_id)
            started_generation = self._voice_generation.get(key, 0)
            lock = self._voice_locks.setdefault(key, asyncio.Lock())
            async with lock:
                current = self._catalog
                if current.revision != catalog.revision or not self._catalog_contains(
                    current, model_id
                ):
                    continue
                if self._closed:
                    return ()

                current_generation = self._voice_generation.get(key, 0)
                if force and current_generation != started_generation:
                    return self._voice_cache.get(key, ())
                if not force and key in self._voice_cache:
                    return self._voice_cache[key]

                voices = await self._fetch_voices(model_id)
                if self._closed:
                    return ()
                if self._catalog.revision != catalog.revision:
                    continue

                self._voice_cache[key] = voices
                self._voice_generation[key] = current_generation + 1
                return voices

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        """Temporarily reject synthesis until the speech-contract slice."""
        del request, progress_sink
        raise NotImplementedError(
            "audio.cpp synthesis is deferred to the speech-contract slice"
        )

    async def close(self) -> None:
        """Close the owned client once and expose a closed health snapshot."""
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            async with self._refresh_lock:
                current = self._catalog
                self._catalog = TTSProviderCatalog(
                    provider_id=_PROVIDER_ID,
                    revision=current.revision,
                    health=_CLOSED_HEALTH,
                    models=current.models,
                )
            try:
                await self._client.aclose()
            finally:
                for logger_name in _HTTP_LOGGER_NAMES:
                    logging.getLogger(logger_name).removeFilter(
                        self._httpx_privacy_filter
                    )

    async def _refresh_catalog(self, *, force: bool) -> None:
        started_generation = self._refresh_generation
        async with self._refresh_lock:
            if self._closed:
                return
            if self._refresh_generation != started_generation:
                return
            if not force and self._catalog.health.fresh:
                return

            previous = self._catalog
            try:
                async with asyncio.timeout(self._config.connect_timeout_seconds):
                    health_body = await self._safe_get("/health")
                    parse_health_response(
                        health_body,
                        max_metadata_bytes=self._config.max_metadata_bytes,
                        max_identifier_characters=(
                            self._config.max_identifier_characters
                        ),
                        max_models=self._config.max_catalog_models,
                    )
                    models_body = await self._safe_get("/v1/models")
                    upstream_models = parse_models_response(
                        models_body,
                        max_metadata_bytes=self._config.max_metadata_bytes,
                        max_identifier_characters=(
                            self._config.max_identifier_characters
                        ),
                        max_models=self._config.max_catalog_models,
                    )
            except asyncio.CancelledError:
                raise
            except (TimeoutError, _TransientHttpFailure):
                if not self._closed:
                    self._catalog = self._failed_catalog(
                        previous,
                        _TRANSIENT_FAILURE_HEALTH,
                    )
                    self._refresh_generation += 1
                return
            except (_HttpContractFailure, AudioCppContractError):
                if not self._closed:
                    self._catalog = self._failed_catalog(
                        previous,
                        _CONTRACT_FAILURE_HEALTH,
                    )
                    self._refresh_generation += 1
                return

            if self._closed:
                return
            models = tuple(
                TTSModelInfo(
                    model_id=model.model_id,
                    display_name=model.model_id,
                    family=model.family,
                    upstream_mode=model.mode,
                    formats=("wav",),
                    voices=(),
                    supports_speed=False,
                    supports_options=(),
                    omit_voice_uses_server_default=True,
                )
                for model in upstream_models
            )
            health = _AVAILABLE_HEALTH if models else _NOT_CONFIGURED_HEALTH
            self._catalog = TTSProviderCatalog(
                provider_id=_PROVIDER_ID,
                revision=previous.revision + 1,
                health=health,
                models=models,
            )
            self._refresh_generation += 1
            self._voice_cache.clear()
            self._voice_generation.clear()
            self._voice_locks.clear()

    async def _fetch_voices(self, model_id: str) -> tuple[str, ...]:
        try:
            async with asyncio.timeout(self._config.connect_timeout_seconds):
                body = await self._safe_get(
                    "/v1/audio/voices",
                    params={"model": model_id},
                )
                return parse_voices_response(
                    body,
                    max_metadata_bytes=self._config.max_metadata_bytes,
                    max_identifier_characters=(self._config.max_identifier_characters),
                    max_voices=self._config.max_voices_per_model,
                )
        except asyncio.CancelledError:
            raise
        except (
            TimeoutError,
            _TransientHttpFailure,
            _HttpContractFailure,
            AudioCppContractError,
        ):
            return ()

    async def _safe_get(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
    ) -> bytes:
        for attempt in range(_MAX_GET_ATTEMPTS):
            try:
                self._client.cookies.clear()
                async with self._client.stream(
                    "GET",
                    path,
                    params=params,
                ) as response:
                    self._client.cookies.clear()
                    if response.status_code != 200:
                        if response.status_code in _TRANSIENT_STATUSES:
                            if attempt + 1 < _MAX_GET_ATTEMPTS:
                                continue
                            raise _TransientHttpFailure
                        raise _HttpContractFailure
                    return await self._read_bounded_metadata(response)
            except asyncio.CancelledError:
                raise
            except _HttpContractFailure:
                raise
            except _TransientHttpFailure:
                raise
            except httpx.StreamError:
                raise _HttpContractFailure from None
            except (TimeoutError, httpx.TransportError):
                if attempt + 1 >= _MAX_GET_ATTEMPTS:
                    raise _TransientHttpFailure from None
        raise _TransientHttpFailure

    async def _read_bounded_metadata(
        self,
        response: httpx.Response,
    ) -> bytes:
        content_encodings = response.headers.get_list("content-encoding")
        if len(content_encodings) > 1:
            raise _HttpContractFailure
        if content_encodings and content_encodings[0].strip().casefold() != "identity":
            raise _HttpContractFailure

        declared_lengths = response.headers.get_list("content-length")
        if len(declared_lengths) > 1:
            raise _HttpContractFailure
        declared_length: int | None = None
        if declared_lengths:
            raw_length = declared_lengths[0]
            if not raw_length or not raw_length.isascii() or not raw_length.isdecimal():
                raise _HttpContractFailure
            declared_length = int(raw_length)
            if declared_length > self._config.max_metadata_bytes:
                raise _HttpContractFailure

        body = bytearray()
        async for chunk in response.aiter_raw():
            body.extend(chunk)
            if len(body) > self._config.max_metadata_bytes:
                raise _HttpContractFailure
        if declared_length is not None and len(body) != declared_length:
            raise _HttpContractFailure
        return bytes(body)

    @staticmethod
    def _failed_catalog(
        previous: TTSProviderCatalog,
        health: ProviderHealth,
    ) -> TTSProviderCatalog:
        return TTSProviderCatalog(
            provider_id=_PROVIDER_ID,
            revision=previous.revision,
            health=health,
            models=previous.models,
        )

    @staticmethod
    def _catalog_contains(
        catalog: TTSProviderCatalog,
        model_id: str,
    ) -> bool:
        return any(model.model_id == model_id for model in catalog.models)
