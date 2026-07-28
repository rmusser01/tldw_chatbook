"""Temporary injected adapter for retained transcription providers.

This module deliberately does not import the legacy transcription service.
The compatibility facade supplies its backend factory at runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from threading import RLock
from typing import Any, Protocol, cast

from .contracts import (
    BufferAudioSource,
    ExecutionDevice,
    FileAudioSource,
    ProducedCapabilities,
    ResolvedTranscriptionRequest,
    TimestampGranularity,
    TranscriptionPhase,
    TranscriptionProgress,
    TranscriptionSegment,
    TranscriptionTask,
    TranscriptionTimings,
)
from .registry import (
    ModelMetadata,
    ProviderMetadata,
    ProviderTranscriptionOutput,
    RuntimeObservation,
)

_SANITIZED_FAILURE_MESSAGE = "The retained speech-to-text provider failed."


class _LegacyBackend(Protocol):
    """Structural subset consumed by the bridge."""

    config: dict[str, Any]

    def transcribe(self, *args: object, **kwargs: object) -> dict[str, Any]: ...

    def transcribe_buffer(
        self,
        *args: object,
        **kwargs: object,
    ) -> dict[str, Any]: ...

    def cleanup(self) -> object: ...

    def get_available_providers(self) -> list[str]: ...

    def list_available_models(
        self,
        provider: str | None = None,
    ) -> dict[str, list[str]]: ...

    def get_device_info(self) -> dict[str, Any]: ...

    def is_diarization_available(self) -> bool: ...

    def get_diarization_requirements(self) -> dict[str, bool]: ...

    def format_segments_with_timestamps(
        self,
        *args: object,
        **kwargs: object,
    ) -> str: ...

    def create_streaming_transcriber(
        self,
        *args: object,
        **kwargs: object,
    ) -> object: ...


class LegacyTranscriptionBridgeError(Exception):
    """Sanitized retained-provider adapter failure."""

    def __init__(self) -> None:
        super().__init__(_SANITIZED_FAILURE_MESSAGE)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class LegacyTranscriptionBridge:
    """Lazily bridge one exact provider or the public compatibility surface."""

    def __init__(
        self,
        backend_factory: Callable[[], _LegacyBackend],
        *,
        provider_metadata: ProviderMetadata | None = None,
        models: tuple[ModelMetadata, ...] = (),
        legacy_provider_id: str | None = None,
    ) -> None:
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        if (
            provider_metadata is not None
            and type(provider_metadata) is not ProviderMetadata
        ):
            raise TypeError("provider_metadata must be a ProviderMetadata")
        if type(models) is not tuple or not all(
            type(model) is ModelMetadata for model in models
        ):
            raise TypeError("models must be a tuple of ModelMetadata values")
        if provider_metadata is None:
            if models or legacy_provider_id is not None:
                raise ValueError("provider metadata is required for an adapter binding")
        else:
            if not models:
                raise ValueError("an adapter binding requires at least one model")
            if any(
                model.provider_id != provider_metadata.provider_id for model in models
            ):
                raise ValueError("all models must belong to the bridge provider")
            if type(legacy_provider_id) is not str or not legacy_provider_id:
                raise ValueError(
                    "an adapter binding requires a legacy provider identifier"
                )

        self._backend_factory = backend_factory
        self._provider_metadata = provider_metadata
        self._models = models
        self._legacy_provider_id = legacy_provider_id
        self._backend: _LegacyBackend | None = None
        self._backend_lock = RLock()

    def _get_backend(self) -> _LegacyBackend:
        with self._backend_lock:
            if self._backend is None:
                self._backend = self._backend_factory()
            return self._backend

    @property
    def config(self) -> dict[str, Any]:
        """Expose the retained mutable config for compatibility callers."""

        return self._get_backend().config

    def provider(self) -> ProviderMetadata:
        """Return the exact provider declaration for adapter use."""

        if self._provider_metadata is None:
            raise LegacyTranscriptionBridgeError()
        return self._provider_metadata

    def describe(self) -> tuple[ModelMetadata, ...]:
        """Return the exact model declarations served by this binding."""

        if self._provider_metadata is None:
            raise LegacyTranscriptionBridgeError()
        return self._models

    def probe(self, model_id: str) -> RuntimeObservation:
        """Map legacy discovery to one exact provider/model observation."""

        model = next((item for item in self._models if item.model_id == model_id), None)
        provider = self._provider_metadata
        legacy_provider_id = self._legacy_provider_id
        if model is None or provider is None or legacy_provider_id is None:
            provider_id = provider.provider_id if provider is not None else "legacy"
            return RuntimeObservation(
                provider_id=provider_id,
                model_id=model_id,
                available=False,
                capabilities=None,
                detail_code="legacy_model_unavailable",
            )
        try:
            backend = self._get_backend()
            providers = backend.get_available_providers()
            models = backend.list_available_models(legacy_provider_id).get(
                legacy_provider_id,
                [],
            )
            effective_device, effective_precision = self._execution_config(
                backend,
                model,
            )
        except Exception:
            return RuntimeObservation(
                provider_id=provider.provider_id,
                model_id=model.model_id,
                available=False,
                capabilities=None,
                detail_code="legacy_probe_failed",
            )
        if legacy_provider_id not in providers or model.model_id not in models:
            return RuntimeObservation(
                provider_id=provider.provider_id,
                model_id=model.model_id,
                available=False,
                capabilities=None,
                detail_code="legacy_model_unavailable",
            )
        return RuntimeObservation(
            provider_id=provider.provider_id,
            model_id=model.model_id,
            available=True,
            capabilities=replace(
                model.capabilities,
                execution_devices=frozenset({effective_device}),
                precisions=frozenset({effective_precision}),
            ),
        )

    def transcribe(
        self,
        request: ResolvedTranscriptionRequest,
    ) -> ProviderTranscriptionOutput:
        """Translate one canonical request into the retained backend call."""

        if type(request) is not ResolvedTranscriptionRequest:
            raise TypeError("request must be a ResolvedTranscriptionRequest")
        model = next(
            (item for item in self._models if item.model_id == request.model_id),
            None,
        )
        if (
            model is None
            or self._provider_metadata is None
            or request.provider_id != self._provider_metadata.provider_id
            or self._legacy_provider_id is None
        ):
            raise LegacyTranscriptionBridgeError()

        legacy_progress = self._legacy_progress_callback(request)
        try:
            backend = self._get_backend()
            execution_config = self._execution_config(backend, model)
            effective_device, effective_precision = execution_config
            if effective_precision != request.precision or (
                request.request.device is not ExecutionDevice.AUTO
                and effective_device is not request.request.device
            ):
                raise LegacyTranscriptionBridgeError()
            source = request.request.source
            if type(source) is FileAudioSource:
                result = backend.transcribe(
                    audio_path=str(source.path),
                    provider=self._legacy_provider_id,
                    model=request.model_id,
                    language=request.effective_language,
                    source_lang=request.effective_language,
                    target_lang=(
                        "en"
                        if request.request.task is TranscriptionTask.TRANSLATE
                        else None
                    ),
                    vad_filter=request.request.vad,
                    diarize=request.request.diarization,
                    progress_callback=legacy_progress,
                    batch_route_resolved=True,
                )
            elif type(source) is BufferAudioSource:
                result = backend.transcribe_buffer(
                    audio_data=source.audio,
                    sample_rate=source.sample_rate,
                    channels=source.channels,
                    sample_width=source.sample_width,
                    provider=self._legacy_provider_id,
                    model=request.model_id,
                    language=request.effective_language,
                    vad_filter=request.request.vad,
                    diarize=request.request.diarization,
                    task=request.request.task.value,
                    **(
                        {"progress_callback": legacy_progress}
                        if legacy_progress is not None
                        else {}
                    ),
                )
            else:
                raise TypeError("unsupported audio source")
            if self._execution_config(backend, model) != execution_config:
                raise LegacyTranscriptionBridgeError()
            return self._normalize_result(
                request,
                model,
                result,
                effective_device=effective_device,
            )
        except LegacyTranscriptionBridgeError:
            raise
        except Exception:
            raise LegacyTranscriptionBridgeError() from None

    def _legacy_progress_callback(
        self,
        request: ResolvedTranscriptionRequest,
    ) -> Callable[[float, str, dict[str, object] | None], None] | None:
        sink = request.request.progress
        if sink is None:
            return None

        def report(
            progress: float,
            _status: str,
            _data: dict[str, object] | None,
        ) -> None:
            if type(progress) not in (int, float) or not math.isfinite(progress):
                return
            sink(
                TranscriptionProgress(
                    attempt_id=request.request.attempt_id,
                    batch_id=request.request.batch_id,
                    job_id=request.request.job_id,
                    phase=TranscriptionPhase.TRANSCRIBING,
                    fraction=max(0.0, min(1.0, float(progress) / 100.0)),
                )
            )

        return report

    def _normalize_result(
        self,
        request: ResolvedTranscriptionRequest,
        model: ModelMetadata,
        result: object,
        *,
        effective_device: ExecutionDevice,
    ) -> ProviderTranscriptionOutput:
        if type(result) is not dict:
            raise LegacyTranscriptionBridgeError()
        legacy_result = cast(dict[str, object], result)
        text = legacy_result.get("text", "")
        if type(text) is not str:
            raise LegacyTranscriptionBridgeError()

        raw_segments = legacy_result.get("segments", [])
        if type(raw_segments) not in (list, tuple):
            raise LegacyTranscriptionBridgeError()
        segment_values = cast("list[object] | tuple[object, ...]", raw_segments)
        segments = tuple(self._normalize_segment(segment) for segment in segment_values)
        duration = legacy_result.get("duration")
        if type(duration) not in (int, float):
            duration = max((segment.end_seconds for segment in segments), default=0.0)
        numeric_duration = cast("int | float", duration)

        effective_language = request.effective_language
        detected_language = None
        if request.requested_language == "auto":
            for candidate in (
                legacy_result.get("detected_language"),
                legacy_result.get("language"),
            ):
                if type(candidate) is not str:
                    continue
                candidate = candidate.strip().lower()
                if candidate in model.capabilities.languages:
                    detected_language = candidate
                    break

        reported_device = legacy_result.get("device")
        if (
            reported_device is not None
            and self._parse_device(reported_device) is not effective_device
        ):
            raise LegacyTranscriptionBridgeError()
        has_speakers = any(segment.speaker is not None for segment in segments)
        diarization = legacy_result.get("diarization_performed")
        produced_diarization = diarization is True or has_speakers
        timestamps = (
            request.request.timestamps if segments else TimestampGranularity.NONE
        )
        if segments and timestamps is TimestampGranularity.NONE:
            timestamps = TimestampGranularity.SEGMENT

        return ProviderTranscriptionOutput(
            text=text,
            segments=segments,
            effective_language=effective_language,
            detected_language=detected_language,
            effective_device=effective_device,
            produced_capabilities=ProducedCapabilities(
                timestamps=timestamps,
                punctuation=model.capabilities.punctuation,
                capitalization=model.capabilities.capitalization,
                vad=request.request.vad,
                diarization=produced_diarization,
            ),
            duration_seconds=float(numeric_duration),
            timings=TranscriptionTimings(),
            warnings=request.warning_codes,
        )

    @staticmethod
    def _normalize_segment(segment: object) -> TranscriptionSegment:
        if type(segment) is not dict:
            raise LegacyTranscriptionBridgeError()
        value = cast(dict[str, object], segment)
        start = value.get("start", value.get("Time_Start", 0.0))
        end = value.get("end", value.get("Time_End", start))
        text = value.get("text", value.get("Text", ""))
        speaker = value.get("speaker_label")
        if speaker is None and value.get("speaker_id") is not None:
            speaker = f"SPEAKER_{value['speaker_id']}"
        if type(start) not in (int, float) or type(end) not in (int, float):
            raise LegacyTranscriptionBridgeError()
        numeric_start = cast("int | float", start)
        numeric_end = cast("int | float", end)
        if type(text) is not str or (speaker is not None and type(speaker) is not str):
            raise LegacyTranscriptionBridgeError()
        return TranscriptionSegment(
            start_seconds=float(numeric_start),
            end_seconds=float(numeric_end),
            text=text,
            speaker=speaker,
        )

    @staticmethod
    def _parse_device(value: object) -> ExecutionDevice:
        aliases = {
            "cpu": ExecutionDevice.CPU,
            "cuda": ExecutionDevice.CUDA,
            "metal": ExecutionDevice.METAL,
            "mps": ExecutionDevice.METAL,
        }
        if type(value) is not str:
            raise LegacyTranscriptionBridgeError()
        normalized = value.strip().lower()
        if normalized not in aliases:
            raise LegacyTranscriptionBridgeError()
        return aliases[normalized]

    @classmethod
    def _execution_config(
        cls,
        backend: _LegacyBackend,
        model: ModelMetadata,
    ) -> tuple[ExecutionDevice, str]:
        device = cls._parse_device(backend.config.get("device"))
        precision = backend.config.get("compute_type")
        if (
            device not in model.capabilities.execution_devices
            or type(precision) is not str
        ):
            raise LegacyTranscriptionBridgeError()
        precision = precision.strip().lower()
        if precision not in model.capabilities.precisions:
            raise LegacyTranscriptionBridgeError()
        return device, precision

    def close(self) -> None:
        """Release an already-created retained backend without constructing it."""

        with self._backend_lock:
            backend = self._backend
        if backend is not None:
            try:
                backend.cleanup()
            except Exception:
                return

    # The methods below preserve the existing facade without broad attribute
    # forwarding. Exceptions intentionally retain their historical behavior.
    def transcribe_legacy(self, *args: object, **kwargs: object) -> object:
        return self._get_backend().transcribe(*args, **kwargs)

    def transcribe_buffer_legacy(self, *args: object, **kwargs: object) -> object:
        return self._get_backend().transcribe_buffer(*args, **kwargs)

    def cleanup_legacy(self) -> object:
        return self._get_backend().cleanup()

    def get_available_providers_legacy(self) -> object:
        return self._get_backend().get_available_providers()

    def list_available_models_legacy(self, provider: str | None = None) -> object:
        return self._get_backend().list_available_models(provider)

    def get_device_info_legacy(self) -> object:
        return self._get_backend().get_device_info()

    def is_diarization_available_legacy(self) -> object:
        return self._get_backend().is_diarization_available()

    def get_diarization_requirements_legacy(self) -> object:
        return self._get_backend().get_diarization_requirements()

    def format_segments_with_timestamps_legacy(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        return self._get_backend().format_segments_with_timestamps(
            *args,
            **kwargs,
        )

    def create_streaming_transcriber_legacy(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        return self._get_backend().create_streaming_transcriber(
            *args,
            **kwargs,
        )


__all__ = [
    "LegacyTranscriptionBridge",
    "LegacyTranscriptionBridgeError",
]
