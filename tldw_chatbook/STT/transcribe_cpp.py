"""Direct-local adapter for the optional pinned ``transcribe.cpp`` runtime.

The native package is intentionally imported only by :func:`transcribe_file`,
which the Library ingestion pipeline calls inside its spawn worker. Importing
this module is safe during normal application startup.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from array import array
from pathlib import Path
from typing import Any

from tldw_chatbook.Model_Artifacts.gguf_admission import validate_local_gguf

from .contracts import (
    CancellationGranularity,
    DeviceFailureOrigin,
    ExecutionDevice,
    FileAudioSource,
    InputKind,
    LanguageInputMode,
    PipelineCapabilities,
    PrivacyRequirements,
    ProducedCapabilities,
    ResolvedTranscriptionRequest,
    TimestampGranularity,
    TranscriptionFailureCode,
    TranscriptionRequest,
    TranscriptionSegment,
    TranscriptionTask,
    TranscriptionTimings,
)
from .coordinator import TranscriptionCoordinator, TranscriptionCoordinatorError
from .registry import (
    CapabilitySet,
    CatalogDeclarations,
    ModelMetadata,
    ProviderMetadata,
    ProviderTranscriptionOutput,
    RuntimeObservation,
)
from .routing import (
    TranscriptionRouter,
    build_builtin_registry,
    default_routing_policy,
)
from .persistence import (
    FailedTranscriptionAttempt,
    dump_failed_transcription_attempt,
    load_failed_transcription_attempt,
)


PROVIDER_ID = "transcribe-cpp"
PRECISION = "native"
_CHOOSE_ANOTHER_GGUF = "choose_another_gguf"
_RETRY_FASTER_WHISPER = "retry_faster_whisper"


class TranscribeCppFailure(Exception):
    """Path-safe direct-local failure with bounded recovery actions."""

    __slots__ = (
        "actions",
        "code",
        "device_failure_origin",
        "error_detail",
        "failed_device",
        "model_id",
        "stt_failure_provenance",
    )

    def __init__(
        self,
        code: TranscriptionFailureCode,
        *,
        model_id: str = "local-gguf:unavailable",
        actions: tuple[str, ...] = (),
        failed_attempt: dict[str, Any] | None = None,
        device_failure_origin: DeviceFailureOrigin | None = None,
        failed_device: ExecutionDevice | None = None,
    ) -> None:
        if (device_failure_origin is None) != (failed_device is None):
            raise ValueError(
                "device_failure_origin and failed_device must be provided together"
            )
        if (
            device_failure_origin is not None
            and type(device_failure_origin) is not DeviceFailureOrigin
        ):
            raise TypeError("device_failure_origin must be a DeviceFailureOrigin")
        if failed_device is not None and type(failed_device) is not ExecutionDevice:
            raise TypeError("failed_device must be an ExecutionDevice")
        self.code = code
        self.model_id = model_id
        self.actions = actions
        self.device_failure_origin = device_failure_origin
        self.failed_device = failed_device
        message = _failure_message(code)
        self.error_detail = {
            "category": "stt_failure",
            "code": code.value,
            "message": message,
            "actions": list(actions),
        }
        self.stt_failure_provenance = failed_attempt
        super().__init__(message)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code.value!r})"


def _failure_message(code: TranscriptionFailureCode) -> str:
    if code is TranscriptionFailureCode.PROVIDER_UNAVAILABLE:
        return "The transcribe.cpp runtime is unavailable."
    if code in {
        TranscriptionFailureCode.ARTIFACT_CORRUPT,
        TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
        TranscriptionFailureCode.MODEL_NOT_INSTALLED,
    }:
        return "The selected GGUF cannot be used by transcribe.cpp."
    return "transcribe.cpp could not complete this transcription."


def _device_from_native_kind(value: object) -> ExecutionDevice:
    normalized = value.casefold() if isinstance(value, str) else ""
    mapping = {
        "cpu": ExecutionDevice.CPU,
        "accel": ExecutionDevice.CPU,
        "cpu_accel": ExecutionDevice.CPU,
        "cuda": ExecutionDevice.CUDA,
        "metal": ExecutionDevice.METAL,
        "mps": ExecutionDevice.METAL,
        "vulkan": ExecutionDevice.VULKAN,
    }
    if normalized not in mapping:
        raise ValueError("unsupported transcribe.cpp execution device")
    return mapping[normalized]


def _device_from_model(model: object) -> ExecutionDevice:
    device = getattr(getattr(model, "device", None), "kind", None)
    if not isinstance(device, str):
        device = getattr(model, "backend", None)
    return _device_from_native_kind(device)


def _failed_accelerator_candidate(
    runtime: object,
    requested: ExecutionDevice,
) -> ExecutionDevice | None:
    """Return a concrete requested or unambiguous auto-selected accelerator."""

    accelerators = {
        ExecutionDevice.CUDA,
        ExecutionDevice.METAL,
        ExecutionDevice.VULKAN,
    }
    if requested in accelerators:
        return requested
    if requested is not ExecutionDevice.AUTO:
        return None
    backend_override = os.environ.get("TRANSCRIBE_BACKEND")
    if backend_override and backend_override.casefold() != "auto":
        try:
            override_device = _device_from_native_kind(backend_override)
        except ValueError:
            return None
        return override_device if override_device in accelerators else None
    backends = getattr(runtime, "backends", None)
    if not callable(backends):
        return None
    try:
        devices = backends()
    except Exception:
        return None
    if not isinstance(devices, (list, tuple)):
        return None
    candidates: list[ExecutionDevice] = []
    for descriptor in devices:
        try:
            candidate = _device_from_native_kind(getattr(descriptor, "kind", None))
        except ValueError:
            continue
        if candidate in accelerators and candidate not in candidates:
            candidates.append(candidate)
    return candidates[0] if len(candidates) == 1 else None


def _timestamp_capabilities(maximum: object) -> frozenset[TimestampGranularity]:
    normalized = maximum.casefold() if isinstance(maximum, str) else "none"
    values = {TimestampGranularity.NONE}
    if normalized in {"segment", "word", "token"}:
        values.add(TimestampGranularity.SEGMENT)
    if normalized in {"word", "token"}:
        values.add(TimestampGranularity.WORD)
    return frozenset(values)


def _runtime_capabilities(model: object) -> CapabilitySet:
    native = getattr(model, "capabilities")
    languages = frozenset(
        language.strip().lower()
        for language in getattr(native, "languages", ())
        if isinstance(language, str) and language.strip() and language != "auto"
    )
    automatic = bool(getattr(native, "supports_language_detect", False))
    language_mode = (
        LanguageInputMode.AUTOMATIC if automatic else LanguageInputMode.ENFORCED
    )
    tasks = {TranscriptionTask.TRANSCRIBE}
    if bool(getattr(native, "supports_translate", False)):
        tasks.add(TranscriptionTask.TRANSLATE)
    return CapabilitySet(
        languages=languages,
        automatic_language=automatic,
        tasks=frozenset(tasks),
        inputs=frozenset({InputKind.FILE}),
        timestamps=_timestamp_capabilities(
            getattr(native, "max_timestamp_kind", "none")
        ),
        true_streaming=False,
        batch=False,
        cancellation=CancellationGranularity.BEFORE_EXECUTION,
        vad=False,
        diarization=False,
        punctuation=False,
        capitalization=False,
        language_input_mode=language_mode,
        execution_devices=frozenset({_device_from_model(model)}),
        precisions=frozenset({PRECISION}),
    )


def _read_normalized_wav(path: Path) -> tuple[array[float], float]:
    with wave.open(str(path), "rb") as wav_file:
        if (
            wav_file.getnchannels() != 1
            or wav_file.getsampwidth() != 2
            or wav_file.getframerate() != 16_000
            or wav_file.getcomptype() != "NONE"
        ):
            raise ValueError("audio is not normalized")
        frame_count = wav_file.getnframes()
        samples = array("h")
        samples.frombytes(wav_file.readframes(frame_count))
    if sys.byteorder != "little":
        samples.byteswap()
    return array("f", (sample / 32768.0 for sample in samples)), frame_count / 16_000


def _pcm_16k_mono(
    audio_path: Path,
    *,
    ffmpeg_path: str | None,
) -> tuple[array[float], float]:
    try:
        return _read_normalized_wav(audio_path)
    except (OSError, EOFError, ValueError, wave.Error):
        pass

    executable = ffmpeg_path or shutil.which("ffmpeg")
    if not executable:
        raise ValueError("ffmpeg is unavailable")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="tldw_stt_", suffix=".wav", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                os.chmod(temporary_path, 0o600)
            except OSError:
                pass
        subprocess.run(
            [
                executable,
                "-i",
                str(audio_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(temporary_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return _read_normalized_wav(temporary_path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


class TranscribeCppAdapter:
    """One already-loaded direct-local model, owned by one worker job."""

    def __init__(
        self,
        *,
        model: object,
        architecture: str,
        model_load_seconds: float,
        ffmpeg_path: str | None = None,
    ) -> None:
        self._model = model
        self._model_load_seconds = model_load_seconds
        self._ffmpeg_path = ffmpeg_path
        self._closed = False
        self._provider = ProviderMetadata(PROVIDER_ID, "transcribe.cpp", True)
        capabilities = _runtime_capabilities(model)
        mode = capabilities.language_input_mode
        self._metadata = ModelMetadata(
            provider_id=PROVIDER_ID,
            model_id=f"local-gguf:{architecture}",
            display_name=f"Local {architecture} GGUF",
            capabilities=capabilities,
            default_precision=PRECISION,
            semantic_default_eligible=False,
            enforces_language_hint=mode
            in {LanguageInputMode.ENFORCED, LanguageInputMode.AUTOMATIC},
        )

    def provider(self) -> ProviderMetadata:
        return self._provider

    def describe(self) -> tuple[ModelMetadata, ...]:
        return (self._metadata,)

    def probe(self, model_id: str) -> RuntimeObservation:
        if model_id != self._metadata.model_id or self._closed:
            return RuntimeObservation(
                provider_id=PROVIDER_ID,
                model_id=model_id,
                available=False,
                capabilities=None,
                detail_code="model_unavailable",
            )
        return RuntimeObservation(
            provider_id=PROVIDER_ID,
            model_id=model_id,
            available=True,
            capabilities=self._metadata.capabilities,
        )

    def transcribe(
        self,
        request: ResolvedTranscriptionRequest,
    ) -> ProviderTranscriptionOutput:
        source = request.request.source
        if type(source) is not FileAudioSource:
            raise TypeError("transcribe.cpp requires a file source")
        pcm, duration = _pcm_16k_mono(
            source.path,
            ffmpeg_path=self._ffmpeg_path,
        )
        timestamp_kind = request.request.timestamps.value
        language = (
            None if request.effective_language == "auto" else request.effective_language
        )
        started = time.perf_counter()
        with self._model.session() as session:
            native_result = session.run(
                pcm,
                task=request.request.task.value,
                language=language,
                timestamps=timestamp_kind,
            )
        inference_seconds = time.perf_counter() - started
        segments = ()
        if request.request.timestamps is not TimestampGranularity.NONE:
            segments = tuple(
                TranscriptionSegment(
                    start_seconds=float(segment.t0_ms) / 1000,
                    end_seconds=float(segment.t1_ms) / 1000,
                    text=str(segment.text),
                )
                for segment in getattr(native_result, "segments", ())
            )
        detected = (
            str(getattr(native_result, "language", "")).strip().lower() or None
            if request.request.language == "auto"
            else None
        )
        return ProviderTranscriptionOutput(
            text=str(getattr(native_result, "text", "")),
            segments=segments,
            effective_language=request.effective_language,
            detected_language=detected,
            effective_device=_device_from_model(self._model),
            produced_capabilities=ProducedCapabilities(
                timestamps=request.request.timestamps,
                punctuation=False,
                capitalization=False,
                vad=False,
                diarization=False,
            ),
            duration_seconds=duration,
            timings=TranscriptionTimings(
                model_load_seconds=self._model_load_seconds,
                inference_seconds=inference_seconds,
                total_seconds=self._model_load_seconds + inference_seconds,
            ),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._model, "close", None)
        if callable(close):
            close()
            return
        exit_method = getattr(self._model, "__exit__", None)
        if callable(exit_method):
            exit_method(None, None, None)

    def mark_model_reused(self) -> None:
        """Exclude one-time model load cost from later resident jobs."""

        self._model_load_seconds = 0.0


def _failure_actions(code: TranscriptionFailureCode) -> tuple[str, ...]:
    if code in {
        TranscriptionFailureCode.MODEL_NOT_INSTALLED,
        TranscriptionFailureCode.ARTIFACT_CORRUPT,
        TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
    }:
        return (_CHOOSE_ANOTHER_GGUF, _RETRY_FASTER_WHISPER)
    if code is TranscriptionFailureCode.CANCELLED:
        return ()
    return (_RETRY_FASTER_WHISPER,)


def _failed_attempt_document(
    *,
    code: TranscriptionFailureCode,
    attempt_id: str,
    batch_id: str | None,
    job_id: str | None,
    model_id: str,
    language: str,
    requested_device: ExecutionDevice = ExecutionDevice.AUTO,
) -> dict[str, Any]:
    attempt = FailedTranscriptionAttempt(
        attempt_id=attempt_id,
        batch_id=batch_id,
        job_id=job_id,
        provider_id=PROVIDER_ID,
        model_id=model_id,
        artifact_root=None,
        artifact_dependencies=(),
        precision=PRECISION,
        requested_device=requested_device,
        effective_device=None,
        requested_language=language,
        effective_language=language,
        detected_language=None,
        task=TranscriptionTask.TRANSCRIBE,
        error_code=code,
    )
    return load_failed_transcription_attempt(dump_failed_transcription_attempt(attempt))


def _runtime_failure(
    *,
    code: TranscriptionFailureCode,
    attempt_id: str,
    model_id: str,
    language: str,
    batch_id: str | None = None,
    job_id: str | None = None,
    actions: tuple[str, ...] | None = None,
    requested_device: ExecutionDevice = ExecutionDevice.AUTO,
    device_failure_origin: DeviceFailureOrigin | None = None,
    failed_device: ExecutionDevice | None = None,
) -> TranscribeCppFailure:
    selected_actions = _failure_actions(code) if actions is None else actions
    normalized_language = (language or "en").strip().lower()
    return TranscribeCppFailure(
        code,
        model_id=model_id,
        actions=selected_actions,
        device_failure_origin=device_failure_origin,
        failed_device=failed_device,
        failed_attempt=_failed_attempt_document(
            code=code,
            attempt_id=attempt_id,
            batch_id=batch_id,
            job_id=job_id,
            model_id=model_id,
            language=normalized_language,
            requested_device=requested_device,
        ),
    )


class TranscribeCppRuntime:
    """One admitted native model reusable across matching worker jobs."""

    def __init__(
        self,
        *,
        adapter: TranscribeCppAdapter,
        coordinator: TranscriptionCoordinator,
        model_id: str,
        maximum_timestamps: frozenset[TimestampGranularity],
        requested_device: ExecutionDevice,
    ) -> None:
        self._adapter = adapter
        self._coordinator = coordinator
        self._model_id = model_id
        self._maximum_timestamps = maximum_timestamps
        self._requested_device = requested_device
        self._closed = False

    @classmethod
    def load(
        cls,
        *,
        model_path: Path | None,
        attempt_id: str,
        batch_id: str | None = None,
        job_id: str | None = None,
        language: str = "en",
        ffmpeg_path: str | None = None,
        device: ExecutionDevice = ExecutionDevice.AUTO,
    ) -> TranscribeCppRuntime:
        """Admit and load one direct-local GGUF without leaking native detail."""

        unavailable_model = "local-gguf:unavailable"
        if model_path is None:
            raise _runtime_failure(
                code=TranscriptionFailureCode.MODEL_NOT_INSTALLED,
                attempt_id=attempt_id,
                batch_id=batch_id,
                job_id=job_id,
                model_id=unavailable_model,
                language=language,
                actions=(_CHOOSE_ANOTHER_GGUF, _RETRY_FASTER_WHISPER),
                requested_device=device,
            )
        try:
            admission = validate_local_gguf(model_path)
        except Exception:
            raise _runtime_failure(
                code=TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                attempt_id=attempt_id,
                batch_id=batch_id,
                job_id=job_id,
                model_id=unavailable_model,
                language=language,
                actions=(_CHOOSE_ANOTHER_GGUF, _RETRY_FASTER_WHISPER),
                requested_device=device,
            ) from None
        try:
            runtime = importlib.import_module("transcribe_cpp")
        except Exception:
            raise _runtime_failure(
                code=TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
                attempt_id=attempt_id,
                batch_id=batch_id,
                job_id=job_id,
                model_id=unavailable_model,
                language=language,
                actions=(_RETRY_FASTER_WHISPER,),
                requested_device=device,
            ) from None

        model: object | None = None
        adapter: TranscribeCppAdapter | None = None
        model_id = f"local-gguf:{admission.metadata.architecture}"
        failed_accelerator = _failed_accelerator_candidate(runtime, device)
        try:
            set_log_callback = getattr(runtime, "set_log_callback", None)
            if callable(set_log_callback):
                set_log_callback(lambda *_args: None)
            load_started = time.perf_counter()
            model = runtime.Model(str(admission.path), backend=device.value)
            load_seconds = time.perf_counter() - load_started
            if getattr(model, "arch", None) != admission.metadata.architecture:
                raise ValueError("native model architecture mismatch")
            adapter = TranscribeCppAdapter(
                model=model,
                architecture=admission.metadata.architecture,
                model_load_seconds=load_seconds,
                ffmpeg_path=ffmpeg_path,
            )
            policy = default_routing_policy()
            declarations = CatalogDeclarations(
                providers=(adapter.provider(),),
                models=adapter.describe(),
            )
            registry = build_builtin_registry(
                policy,
                adapters=(adapter,),
                extra_declarations=declarations,
            )
            coordinator = TranscriptionCoordinator(
                registry=registry,
                router=TranscriptionRouter(policy),
                pipeline=PipelineCapabilities(),
            )
            maximum_timestamps = adapter.describe()[0].capabilities.timestamps
            return cls(
                adapter=adapter,
                coordinator=coordinator,
                model_id=model_id,
                maximum_timestamps=maximum_timestamps,
                requested_device=device,
            )
        except Exception as error:
            if adapter is not None:
                adapter.close()
            elif model is not None:
                close = getattr(model, "close", None)
                if callable(close):
                    close()
            backend_error = getattr(runtime, "BackendError", None)
            typed_backend_failure = bool(
                failed_accelerator is not None
                and isinstance(backend_error, type)
                and isinstance(error, backend_error)
            )
            raise _runtime_failure(
                code=(
                    TranscriptionFailureCode.PROVIDER_UNAVAILABLE
                    if typed_backend_failure
                    else TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE
                ),
                attempt_id=attempt_id,
                batch_id=batch_id,
                job_id=job_id,
                model_id=model_id,
                language=language,
                actions=(
                    (_RETRY_FASTER_WHISPER,)
                    if typed_backend_failure
                    else (_CHOOSE_ANOTHER_GGUF, _RETRY_FASTER_WHISPER)
                ),
                requested_device=device,
                device_failure_origin=(
                    DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION
                    if typed_backend_failure
                    else None
                ),
                failed_device=failed_accelerator if typed_backend_failure else None,
            ) from None

    def transcribe(
        self,
        *,
        audio_path: Path,
        attempt_id: str,
        batch_id: str | None = None,
        job_id: str | None = None,
        retry_of_attempt_id: str | None = None,
        retry_of_job_id: str | None = None,
        language: str = "en",
        timestamps: bool = False,
    ) -> Any:
        """Transcribe one file with this already-loaded exact model."""

        if self._closed:
            raise _runtime_failure(
                code=TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
                attempt_id=attempt_id,
                batch_id=batch_id,
                job_id=job_id,
                model_id=self._model_id,
                language=language,
                requested_device=self._requested_device,
            )
        normalized_language = (language or "en").strip().lower()
        timestamp_request = (
            TimestampGranularity.SEGMENT
            if timestamps and TimestampGranularity.SEGMENT in self._maximum_timestamps
            else TimestampGranularity.NONE
        )
        request = TranscriptionRequest(
            attempt_id=attempt_id,
            batch_id=batch_id,
            job_id=job_id,
            retry_of_attempt_id=retry_of_attempt_id,
            retry_of_job_id=retry_of_job_id,
            source=FileAudioSource(audio_path),
            provider_id=PROVIDER_ID,
            model_id=self._model_id,
            language=normalized_language,
            task=TranscriptionTask.TRANSCRIBE,
            precision=PRECISION,
            device=self._requested_device,
            timestamps=timestamp_request,
            privacy=PrivacyRequirements(allow_remote_processing=False),
        )
        try:
            return self._coordinator.transcribe(request)
        except TranscriptionCoordinatorError as error:
            raise _runtime_failure(
                code=error.failure.code,
                attempt_id=attempt_id,
                batch_id=batch_id,
                job_id=job_id,
                model_id=self._model_id,
                language=normalized_language,
                actions=_failure_actions(error.failure.code),
                requested_device=self._requested_device,
            ) from None
        except TranscribeCppFailure:
            raise
        except Exception:
            raise _runtime_failure(
                code=TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                attempt_id=attempt_id,
                batch_id=batch_id,
                job_id=job_id,
                model_id=self._model_id,
                language=normalized_language,
                actions=(_CHOOSE_ANOTHER_GGUF, _RETRY_FASTER_WHISPER),
                requested_device=self._requested_device,
            ) from None
        finally:
            self._adapter.mark_model_reused()

    def close(self) -> None:
        """Close the resident native model once."""

        if self._closed:
            return
        self._closed = True
        self._adapter.close()


def transcribe_file(
    *,
    audio_path: Path,
    model_path: Path | None,
    attempt_id: str,
    batch_id: str | None = None,
    job_id: str | None = None,
    retry_of_attempt_id: str | None = None,
    retry_of_job_id: str | None = None,
    language: str = "en",
    timestamps: bool = False,
    ffmpeg_path: str | None = None,
    device: ExecutionDevice = ExecutionDevice.AUTO,
) -> Any:
    """Load, use, and close one direct-local model for legacy callers."""

    runtime = TranscribeCppRuntime.load(
        model_path=model_path,
        attempt_id=attempt_id,
        batch_id=batch_id,
        job_id=job_id,
        language=language,
        ffmpeg_path=ffmpeg_path,
        device=device,
    )
    try:
        return runtime.transcribe(
            audio_path=audio_path,
            attempt_id=attempt_id,
            batch_id=batch_id,
            job_id=job_id,
            retry_of_attempt_id=retry_of_attempt_id,
            retry_of_job_id=retry_of_job_id,
            language=language,
            timestamps=timestamps,
        )
    finally:
        runtime.close()


__all__ = [
    "PROVIDER_ID",
    "TranscribeCppAdapter",
    "TranscribeCppFailure",
    "TranscribeCppRuntime",
    "transcribe_file",
]
