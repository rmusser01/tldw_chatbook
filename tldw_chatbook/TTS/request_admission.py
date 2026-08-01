"""Coherent default-request resolution and TTS resource admission."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from tldw_chatbook.TTS._async_lifecycle import join_retained_task
from tldw_chatbook.TTS.adapter_types import (
    ProgressSink,
    TTSAudioResponse,
    TTSConfigurationRevisionError,
    TTSProviderUnavailableError,
    TTSRequest,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import resolve_legacy_route
from tldw_chatbook.TTS.legacy_request_builder import build_legacy_speech_request
from tldw_chatbook.TTS.playground_types import TTSRequestedSelectionSnapshot
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot

if TYPE_CHECKING:
    from tldw_chatbook.TTS.adapter_registry import TTSAdapterLease
    from tldw_chatbook.TTS.TTS_Generation import (
        TTSService,
        _AdmittedTTSOperation,
        _OperationCapacityReservation,
    )


class _WriterPreferredGate:
    """A cancellation-safe shared/exclusive gate with writer preference."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._reader_count = 0
        self._active_writer = False
        self._waiting_writer_count = 0

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        """Enter the shared side without holding the condition during work."""
        admitted = False
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._active_writer and self._waiting_writer_count == 0
            )
            self._reader_count += 1
            admitted = True

        try:
            yield
        finally:
            if admitted:

                async def release_reader() -> None:
                    async with self._condition:
                        self._reader_count -= 1
                        if self._reader_count == 0:
                            self._condition.notify_all()

                release_task = asyncio.create_task(release_reader())
                await join_retained_task(release_task)

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        """Enter the exclusive side and prevent later readers from bypassing."""
        admitted = False
        async with self._condition:
            self._waiting_writer_count += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._active_writer and self._reader_count == 0
                )
            finally:
                self._waiting_writer_count -= 1
                self._condition.notify_all()
            self._active_writer = True
            admitted = True

        try:
            yield
        finally:
            if admitted:

                async def release_writer() -> None:
                    async with self._condition:
                        self._active_writer = False
                        self._condition.notify_all()

                release_task = asyncio.create_task(release_writer())
                await join_retained_task(release_task)


class TTSRequestAdmissionCoordinator:
    """Freeze preferences and acquire matching service resources atomically."""

    def __init__(
        self,
        service: TTSService,
        preferences: TTSPreferencesSnapshot | None,
    ) -> None:
        self._service = service
        self._preferences = preferences
        self._preferences_generation = 0
        self._gate = _WriterPreferredGate()
        self._publication_lock = asyncio.Lock()

    def preferences_snapshot(self) -> TTSPreferencesSnapshot | None:
        """Return canonical preferences, or None while unconfigured."""
        return self._preferences

    def preferences_generation(self) -> int:
        """Return the latest saved settings generation published in memory."""
        return self._preferences_generation

    def _publish_preferences(
        self,
        preferences: TTSPreferencesSnapshot,
        generation: int,
    ) -> None:
        """Publish one already-persisted immutable snapshot exactly once."""
        if generation <= self._preferences_generation:
            return
        self._preferences = preferences
        self._preferences_generation = generation

    async def synthesize_default(
        self,
        *,
        text: str,
        voice_override: str | None = None,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        """Resolve and admit one coherent default request, then execute it.

        Args:
            text: Text to synthesize.
            voice_override: Optional request-scoped voice identifier.
            progress_sink: Optional callback for bounded synthesis progress.

        Returns:
            The admitted provider response. The caller owns and must close it.

        Raises:
            TTSProviderUnavailableError: If no default provider is configured
                or its dynamic model catalog is empty.
            TTSRegistryClosedError: If service shutdown has begun.
            TTSProviderReconfiguringError: If the selected provider is in an
                exclusive settings handoff.
            TTSConfigurationRevisionError: If the selected provider revision
                changes before admission completes.
            TTSOperationError: If admission or synthesis fails.
            ValueError: If the published default preferences are invalid.
        """
        reservation: _OperationCapacityReservation | None = None
        operation: _AdmittedTTSOperation | None = None
        try:
            reservation = await self._service._reserve_operation_capacity()
            async with self._gate.read():
                preferences = self._preferences
                if preferences is None:
                    raise TTSProviderUnavailableError(
                        "TTS default provider is not configured"
                    )
                model_id = await self._resolve_model(preferences)
                revision = self._service.configuration_revision(preferences.provider_id)
                request = self._build_request(
                    preferences,
                    model_id=model_id,
                    text=text,
                    voice_override=voice_override,
                )
                operation = await self._service._admit_reserved(
                    request,
                    reservation,
                    expected_configuration_revision=revision,
                )
                operation.claim()
        except BaseException as error:
            if operation is None:
                if reservation is not None:
                    reservation.release_if_untransferred()
            else:
                await self._service._close_admitted_operation_preserving_primary(
                    operation,
                    error,
                )
            raise

        assert operation is not None
        return await operation.synthesize(progress_sink)

    async def acquire_native_capability_lease(
        self,
        provider_id: str,
    ) -> tuple[int, TTSAdapterLease]:
        """Acquire one revision-matched lease without retaining the read gate."""
        lease: TTSAdapterLease | None = None
        try:
            async with self._gate.read():
                revision = self._service.configuration_revision(provider_id)
                lease = await self._service.registry.acquire(
                    provider_id,
                    expected_revision=revision,
                )
        except BaseException as error:
            if lease is not None:
                await self._service._release_lease_preserving_primary(
                    lease,
                    error,
                )
            raise

        assert lease is not None
        return revision, lease

    async def synthesize_exact(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> tuple[TTSAudioResponse, TTSRequestedSelectionSnapshot]:
        """Freeze and admit one exact request under the shared read side."""
        reservation: _OperationCapacityReservation | None = None
        operation: _AdmittedTTSOperation | None = None
        try:
            reservation = await self._service._reserve_operation_capacity()
            async with self._gate.read():
                revision = self._service.configuration_revision(request.provider_id)
                selection = TTSRequestedSelectionSnapshot(
                    provider_id=request.provider_id,
                    model_id=request.model_id,
                    voice_id=request.voice,
                    response_format=request.response_format,
                    speed=request.speed,
                    options=request.options,
                    configuration_revision=revision,
                )
                admitted_request = TTSRequest(
                    provider_id=selection.provider_id,
                    model_id=selection.model_id,
                    text=request.text,
                    voice=selection.voice_id,
                    response_format=selection.response_format,
                    speed=selection.speed,
                    options=selection.options,
                )
                operation = await self._service._admit_reserved(
                    admitted_request,
                    reservation,
                    expected_configuration_revision=revision,
                )
                operation.claim()
        except BaseException as error:
            if operation is None:
                if reservation is not None:
                    reservation.release_if_untransferred()
            else:
                await self._service._close_admitted_operation_preserving_primary(
                    operation,
                    error,
                )
            raise

        assert operation is not None
        response = await operation.synthesize(progress_sink)
        return response, selection

    async def require_current_configuration_revision(
        self,
        provider_id: str,
        expected_revision: int,
    ) -> None:
        """Make one writer-ordered provider revision decision."""
        async with self._gate.read():
            current_revision = self._service.configuration_revision(provider_id)
            if current_revision != expected_revision:
                raise TTSConfigurationRevisionError(
                    f"TTS provider configuration changed: {provider_id}"
                )

    async def _resolve_model(
        self,
        preferences: TTSPreferencesSnapshot,
    ) -> str:
        if preferences.model_mode == "exact":
            assert preferences.model_id is not None
            return preferences.model_id

        catalog = await self._service.get_catalog(preferences.provider_id)
        if not catalog.models:
            raise TTSProviderUnavailableError(
                f"TTS provider is unavailable: {preferences.provider_id}"
            )
        return catalog.models[0].model_id

    @staticmethod
    def _build_request(
        preferences: TTSPreferencesSnapshot,
        *,
        model_id: str,
        text: str,
        voice_override: str | None,
    ) -> TTSRequest:
        selected_voice = voice_override or preferences.voice_id
        if preferences.provider_id == "audio_cpp":
            return TTSRequest(
                provider_id="audio_cpp",
                model_id=model_id,
                text=text,
                voice=selected_voice,
                response_format="wav",
                speed=1.0,
                options={},
            )

        legacy_request, internal_model_id = _legacy_request(
            preferences,
            model_id=model_id,
            text=text,
            voice=selected_voice,
        )
        route = resolve_legacy_route(internal_model_id)
        if route.provider_id != preferences.provider_id:
            raise ValueError("Legacy TTS route does not match provider")
        return TTSRequest(
            provider_id=preferences.provider_id,
            model_id=legacy_request.model,
            text=text,
            voice=legacy_request.voice,
            response_format=legacy_request.response_format,
            speed=legacy_request.speed,
            options={
                "_legacy_openai_request": legacy_request,
                "_legacy_internal_model_id": route.internal_model_id,
            },
        )


def _legacy_request(
    preferences: TTSPreferencesSnapshot,
    *,
    model_id: str,
    text: str,
    voice: str | None,
) -> tuple[OpenAISpeechRequest, str]:
    """Adapt the app-wide preferences snapshot into an explicit builder call.

    This is a thin adapter over
    :func:`tldw_chatbook.TTS.legacy_request_builder.build_legacy_speech_request`,
    which holds the actual id/override derivation logic. All behaviour lives
    there; this function only reads the relevant fields off ``preferences``.
    """
    return build_legacy_speech_request(
        provider_id=preferences.provider_id,
        model_id=model_id,
        voice=voice or "",
        text=text,
        response_format=preferences.response_format,
        speed=preferences.speed,
    )
