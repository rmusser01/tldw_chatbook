"""Spawn entry point for the app-owned resident local STT worker."""

from __future__ import annotations

import inspect
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .contracts import (
    BufferAudioSource,
    DeviceFailureOrigin,
    ExecutionDevice,
    FileAudioSource,
    TranscriptionFailureCode,
    TranscriptionWarningCode,
)
from .executor import (
    _CPU_FALLBACK_REQUESTED_DEVICE_OPTION,
    ExecutorEvent,
    ExecutorFailure,
    ExecutorRequest,
    ExecutorResident,
    ExecutorResult,
    LocalSourceChangedError,
    WorkerPhase,
    validate_local_source_snapshot,
)
from .executor_process_tree import enter_worker_containment

TranscriptionRunner = Callable[..., dict[str, Any]]
ProviderBuilder = Callable[
    [ExecutorRequest, Path | None, Any | None, Callable[[], bool]],
    "ProviderRuntime",
]
ParseJob = Callable[..., dict[str, Any]]


@dataclass(slots=True)
class ProviderRuntime:
    """The two operations needed from a resident provider implementation."""

    runner: TranscriptionRunner
    close: Callable[[], None]
    buffer_runner: Callable[..., dict[str, Any]] | None = None


@dataclass(slots=True)
class _ResidentRuntime:
    identity: object
    provider: ProviderRuntime
    local_snapshot_token: str | None
    managed_store_root: Path | None
    managed_artifact_ref: tuple[str, str, str] | None
    managed_dependency_refs: tuple[tuple[str, str, str], ...]
    managed_lease_refs: tuple[tuple[str, str, str], ...]
    lease: Any | None = None
    reported: bool = False

    def close(self) -> None:
        """Close native state before releasing its protecting artifact lease."""

        try:
            self.provider.close()
        finally:
            if self.lease is not None:
                self.lease.close()
                self.lease = None


class _ProviderLoadFailure(RuntimeError):
    def __init__(
        self,
        code: TranscriptionFailureCode,
        actions: tuple[str, ...] = (),
    ) -> None:
        self.code = code
        self.actions = actions
        super().__init__(code.value)


def _cancelled_failure(request: ExecutorRequest) -> ExecutorFailure:
    return ExecutorFailure(
        request.generation,
        request.attempt_id,
        TranscriptionFailureCode.CANCELLED,
    )


def _default_recovery_actions(
    request: ExecutorRequest,
    code: TranscriptionFailureCode,
) -> tuple[str, ...]:
    if code is TranscriptionFailureCode.CANCELLED:
        return ()
    if request.identity.provider_id == "transcribe-cpp" and code in {
        TranscriptionFailureCode.MODEL_NOT_INSTALLED,
        TranscriptionFailureCode.ARTIFACT_CORRUPT,
        TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
    }:
        return ("choose_another_gguf", "retry_faster_whisper")
    return ("retry_faster_whisper",)


def _typed_device_failure(
    error: BaseException,
) -> tuple[DeviceFailureOrigin | None, ExecutionDevice | None]:
    origin = getattr(error, "device_failure_origin", None)
    failed_device = getattr(error, "failed_device", None)
    if type(origin) is DeviceFailureOrigin and type(failed_device) is ExecutionDevice:
        return origin, failed_device
    return None, None


def _executor_failure(
    request: ExecutorRequest,
    error: BaseException,
    code: TranscriptionFailureCode,
    *,
    actions: tuple[str, ...] = (),
    failed_attempt: dict[str, Any] | None = None,
) -> ExecutorFailure:
    origin, failed_device = _typed_device_failure(error)
    return ExecutorFailure(
        request.generation,
        request.attempt_id,
        code,
        recovery_actions=actions or _default_recovery_actions(request, code),
        failed_attempt=failed_attempt,
        device_failure_origin=origin,
        failed_device=failed_device,
    )


def _failure_from_exception(
    request: ExecutorRequest,
    error: BaseException,
) -> ExecutorFailure:
    if isinstance(error, LocalSourceChangedError):
        return _executor_failure(
            request,
            error,
            TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
        )
    if isinstance(error, _ProviderLoadFailure):
        return _executor_failure(
            request,
            error,
            error.code,
            actions=error.actions,
        )

    from .transcribe_cpp import TranscribeCppFailure

    if isinstance(error, TranscribeCppFailure):
        return _executor_failure(
            request,
            error,
            error.code,
            actions=tuple(error.actions),
            failed_attempt=error.stt_failure_provenance,
        )

    error_detail = getattr(error, "error_detail", None)
    if isinstance(error_detail, dict):
        try:
            code = TranscriptionFailureCode(error_detail.get("code"))
        except (TypeError, ValueError):
            code = TranscriptionFailureCode.INFERENCE_FAILED
        actions = error_detail.get("actions", ())
        if not isinstance(actions, (list, tuple)):
            actions = ()
        safe_actions = tuple(
            action[:80]
            for action in actions
            if isinstance(action, str) and action.strip()
        )[:8]
        failed_attempt = getattr(error, "stt_failure_provenance", None)
        if not isinstance(failed_attempt, dict):
            failed_attempt = None
        return _executor_failure(
            request,
            error,
            code,
            actions=safe_actions,
            failed_attempt=failed_attempt,
        )
    return _executor_failure(
        request,
        error,
        TranscriptionFailureCode.INFERENCE_FAILED,
    )


def _failure_from_worker_exception(
    request: ExecutorRequest,
    error: BaseException,
    *,
    cancelled: bool,
) -> ExecutorFailure:
    """Prefer a cancellation terminal when the shared event is already set."""
    if cancelled:
        return _cancelled_failure(request)
    return _failure_from_exception(request, error)


def _caused_by_missing_path(error: BaseException) -> bool:
    cause = error.__cause__
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, (FileNotFoundError, NotADirectoryError)):
            return True
        cause = cause.__cause__
    return False


def _dependency_failure_code(error: BaseException) -> TranscriptionFailureCode:
    from tldw_chatbook.Model_Artifacts import (
        ArtifactDependencyError,
        ArtifactIntegrityError,
        ArtifactLeaseError,
        ArtifactStateError,
    )

    if isinstance(error, ArtifactDependencyError):
        if _caused_by_missing_path(error):
            return TranscriptionFailureCode.MODEL_NOT_INSTALLED
        if error.__cause__ is not None:
            return TranscriptionFailureCode.ARTIFACT_CORRUPT
        return TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE
    if isinstance(error, ArtifactIntegrityError):
        if _caused_by_missing_path(error):
            return TranscriptionFailureCode.MODEL_NOT_INSTALLED
        return TranscriptionFailureCode.ARTIFACT_CORRUPT
    if isinstance(error, (ArtifactLeaseError, ArtifactStateError)):
        return TranscriptionFailureCode.PROVIDER_UNAVAILABLE
    return TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE


def _acquire_managed_model(request: ExecutorRequest) -> tuple[Any, Any] | None:
    if request.managed_artifact_ref is None and not request.managed_dependency_refs:
        return None
    assert request.managed_store_root is not None
    from tldw_chatbook.Model_Artifacts import ArtifactRef, ModelArtifactService

    if request.managed_artifact_ref is None:
        try:
            references = tuple(
                ArtifactRef(*reference) for reference in request.managed_dependency_refs
            )
            leased = ModelArtifactService(
                request.managed_store_root
            ).acquire_dependencies(references)
            return leased, leased.handle
        except Exception as error:
            raise _ProviderLoadFailure(_dependency_failure_code(error)) from None

    try:
        reference = ArtifactRef(*request.managed_artifact_ref)
        leased = ModelArtifactService(request.managed_store_root).acquire(reference)
        handle = leased.handle
        if (
            handle.root != reference
            or handle.root.revision != request.identity.root_revision
            or handle.closure_fingerprint != request.identity.closure_fingerprint
        ):
            leased.close()
            raise _ProviderLoadFailure(TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE)
        return leased, handle
    except _ProviderLoadFailure:
        raise
    except Exception:
        raise _ProviderLoadFailure(
            TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE
        ) from None


def _direct_local_model_root(request: ExecutorRequest) -> Path | None:
    snapshot = request.local_source
    if snapshot is None:
        return None
    validate_local_source_snapshot(snapshot)
    if request.identity.local_snapshot_token != snapshot.token:
        raise LocalSourceChangedError("Local STT model files changed")
    if request.identity.provider_id == "transcribe-cpp":
        if len(snapshot.paths) != 1:
            raise LocalSourceChangedError("Local STT model files changed")
        return snapshot.paths[0]
    parents = {path.parent for path in snapshot.paths}
    if len(parents) != 1:
        raise LocalSourceChangedError("Local STT model files changed")
    return parents.pop()


def _load_resident(
    request: ExecutorRequest,
    provider_builder: ProviderBuilder,
    is_cancelled: Callable[[], bool],
) -> _ResidentRuntime:
    model_root = _direct_local_model_root(request)
    acquired = _acquire_managed_model(request)
    lease = None
    handle = None
    if acquired is not None:
        lease, handle = acquired
        if request.managed_artifact_ref is not None:
            model_root = dict(handle.paths)[handle.root]
    try:
        if request.local_source is not None and handle is not None:
            model_root = _direct_local_model_root(request)
        provider = provider_builder(request, model_root, handle, is_cancelled)
    except Exception:
        if lease is not None:
            lease.close()
        raise
    if handle is None:
        managed_lease_refs = ()
    elif request.managed_artifact_ref is not None:
        managed_lease_refs = tuple(
            (reference.artifact_id, reference.revision, reference.variant)
            for reference in handle.closure
        )
    else:
        managed_lease_refs = tuple(
            (reference.artifact_id, reference.revision, reference.variant)
            for reference in handle.references
        )
    return _ResidentRuntime(
        identity=request.identity,
        provider=provider,
        local_snapshot_token=(
            request.local_source.token if request.local_source is not None else None
        ),
        managed_store_root=(
            request.managed_store_root.absolute()
            if request.managed_store_root is not None
            else None
        ),
        managed_artifact_ref=request.managed_artifact_ref,
        managed_dependency_refs=request.managed_dependency_refs,
        managed_lease_refs=managed_lease_refs,
        lease=lease,
    )


def _validate_reuse(request: ExecutorRequest, resident: _ResidentRuntime) -> None:
    if request.identity != resident.identity:
        raise _ProviderLoadFailure(TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE)
    if request.local_source is not None:
        validate_local_source_snapshot(request.local_source)
        if request.local_source.token != resident.local_snapshot_token:
            raise LocalSourceChangedError("Local STT model files changed")
    request_store = (
        request.managed_store_root.absolute()
        if request.managed_store_root is not None
        else None
    )
    if request.managed_dependency_refs != resident.managed_dependency_refs:
        raise LocalSourceChangedError("Local STT model dependencies changed")
    if (
        request_store != resident.managed_store_root
        or request.managed_artifact_ref != resident.managed_artifact_ref
    ):
        raise _ProviderLoadFailure(TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE)
    if request.local_source is not None and request.managed_dependency_refs:
        acquired = _acquire_managed_model(request)
        if acquired is None:
            raise _ProviderLoadFailure(TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE)
        acquired[0].close()


def _transcribe_cpp_provider(
    request: ExecutorRequest,
    model_root: Path | None,
    _managed_handle: Any | None,
    _is_cancelled: Callable[[], bool],
) -> ProviderRuntime:
    from .persistence import build_transcription_provenance_document
    from .transcribe_cpp import TranscribeCppRuntime

    model_path = model_root
    if model_path is not None and model_path.is_dir():
        relative_value = request.options.get("managed_model_relative_path")
        if not isinstance(relative_value, str) or not relative_value.strip():
            raise _ProviderLoadFailure(
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                ("choose_another_gguf", "retry_faster_whisper"),
            )
        relative_value = relative_value.strip()
        relative = PurePosixPath(relative_value)
        windows_relative = PureWindowsPath(relative_value)
        if (
            not relative.parts
            or relative.is_absolute()
            or windows_relative.is_absolute()
            or bool(windows_relative.drive)
            or bool(windows_relative.root)
            or "\\" in relative_value
            or ".." in relative.parts
            or ".." in windows_relative.parts
        ):
            raise _ProviderLoadFailure(
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                ("choose_another_gguf", "retry_faster_whisper"),
            )
        root = model_path.resolve()
        candidate = root.joinpath(*relative.parts).resolve()
        if not candidate.is_relative_to(root):
            raise _ProviderLoadFailure(
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                ("choose_another_gguf", "retry_faster_whisper"),
            )
        model_path = candidate
    context = request.options.get("transcription_context") or {}
    if not isinstance(context, dict):
        context = {}
    runtime = TranscribeCppRuntime.load(
        model_path=model_path,
        attempt_id=request.attempt_id,
        batch_id=context.get("batch_id"),
        job_id=request.job_id,
        language=request.options.get("language") or "en",
        ffmpeg_path=request.options.get("ffmpeg_path"),
        device=request.identity.device,
    )

    def runner(audio_path: str, **kwargs: Any) -> dict[str, Any]:
        normalized = runtime.transcribe(
            audio_path=Path(audio_path),
            attempt_id=kwargs.get("attempt_id") or request.attempt_id,
            batch_id=kwargs.get("batch_id") or context.get("batch_id"),
            job_id=kwargs.get("job_id") or request.job_id,
            retry_of_attempt_id=kwargs.get("retry_of_attempt_id"),
            retry_of_job_id=kwargs.get("retry_of_job_id"),
            language=kwargs.get("language") or "en",
            timestamps=bool(kwargs.get("timestamps", True)),
        )
        fallback_value = request.options.get(_CPU_FALLBACK_REQUESTED_DEVICE_OPTION)
        if request.identity.device is ExecutionDevice.CPU and isinstance(
            fallback_value, str
        ):
            try:
                requested_device = ExecutionDevice(fallback_value)
            except ValueError:
                requested_device = None
            if (
                requested_device is not None
                and requested_device is not ExecutionDevice.CPU
            ):
                warning = TranscriptionWarningCode.DEVICE_FALLBACK_TO_CPU
                warnings = (
                    normalized.warnings
                    if warning in normalized.warnings
                    else (*normalized.warnings, warning)
                )
                normalized = replace(
                    normalized,
                    provenance=replace(
                        normalized.provenance,
                        requested_device=requested_device,
                    ),
                    warnings=warnings,
                )
        provenance = build_transcription_provenance_document(
            normalized,
            failed_attempt=kwargs.get("retry_source_failure_provenance"),
        )
        return {
            "text": normalized.text,
            "segments": [
                {
                    "start": segment.start_seconds,
                    "end": segment.end_seconds,
                    "text": segment.text,
                }
                for segment in normalized.segments
            ],
            "transcription_model": normalized.provenance.model_id,
            "transcription_provenance": provenance,
        }

    return ProviderRuntime(runner=runner, close=runtime.close)


def _parakeet_provider(
    request: ExecutorRequest,
    model_root: Path | None,
    managed_handle: Any | None,
    is_cancelled: Callable[[], bool],
) -> ProviderRuntime:
    from .parakeet_onnx import (
        ParakeetOnnxCancelled,
        ParakeetOnnxFailure,
        ParakeetOnnxRuntime,
    )

    context = request.options.get("transcription_context") or {}
    if not isinstance(context, dict):
        context = {}
    model_id = request.identity.model_id
    precision = request.identity.precision
    requested_language = request.options.get("language") or "en"
    effective_language = (
        "auto"
        if model_id == "nemo-parakeet-tdt-0.6b-v3"
        else "en"
    )
    artifact_root = None
    artifact_dependencies: tuple[Any, ...] = ()

    missing = object()

    def failure(
        code: TranscriptionFailureCode,
        message: str,
        *,
        effective_device: ExecutionDevice | None = None,
        attempt_id: str | None = None,
        batch_id: str | None = None,
        job_id: str | None | object = missing,
        language: str | None = None,
    ) -> ParakeetOnnxFailure:
        return ParakeetOnnxFailure(
            code,
            message,
            attempt_id=attempt_id or request.attempt_id,
            batch_id=batch_id if batch_id is not None else context.get("batch_id"),
            job_id=request.job_id if job_id is missing else job_id,
            model_id=model_id,
            artifact_root=artifact_root,
            artifact_dependencies=artifact_dependencies,
            precision=precision,
            requested_language=language or requested_language,
            effective_language=effective_language,
            effective_device=effective_device,
        )

    if request.options.get("_verify_legacy_parakeet_v2"):
        from tldw_chatbook.Local_Ingestion.parakeet_v2_installer import (
            verify_parakeet_v2_bundle,
        )

        if model_root is None or not verify_parakeet_v2_bundle(model_root):
            raise failure(
                TranscriptionFailureCode.ARTIFACT_CORRUPT,
                "The selected Parakeet ONNX model is corrupt.",
            )

    if model_root is None:
        raise failure(
            TranscriptionFailureCode.MODEL_NOT_INSTALLED,
            "The selected Parakeet ONNX model is not installed.",
        )

    vad_root = None
    if managed_handle is not None:
        paths = dict(managed_handle.paths)
        if hasattr(managed_handle, "root"):
            artifact_root = managed_handle.root.lease_key()
            dependency_refs = tuple(
                reference
                for reference in managed_handle.closure
                if reference != managed_handle.root
            )
        else:
            dependency_refs = managed_handle.references
        artifact_dependencies = tuple(
            reference.lease_key() for reference in dependency_refs
        )
        vad_ref = next(
            (
                reference
                for reference in dependency_refs
                if reference.artifact_id == "silero-vad"
            ),
            None,
        )
        if vad_ref is None or vad_ref not in paths:
            raise failure(
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                "The managed Parakeet artifact closure is incomplete.",
            )
        vad_root = paths[vad_ref]

    from .persistence import build_transcription_provenance_document

    try:
        runtime = ParakeetOnnxRuntime.load(
            model_root=model_root,
            vad_root=vad_root,
            model_id=model_id,
            precision=precision,
            artifact_root=artifact_root,
            artifact_dependencies=artifact_dependencies,
        )
    except ModuleNotFoundError:
        raise failure(
            TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
            "The Parakeet ONNX runtime is unavailable.",
        ) from None
    except ParakeetOnnxFailure:
        raise
    except Exception:
        raise failure(
            TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
            "The selected Parakeet ONNX model cannot be loaded.",
        ) from None

    def runner(audio_path: str, **kwargs: Any) -> dict[str, Any]:
        attempt_id = kwargs.get("attempt_id") or request.attempt_id
        batch_id = kwargs.get("batch_id") or context.get("batch_id")
        job_id = kwargs.get("job_id") or request.job_id
        language = (
            kwargs.get("language") or request.options.get("language") or "en"
        )
        try:
            normalized = runtime.transcribe(
                audio_path=Path(audio_path),
                attempt_id=attempt_id,
                batch_id=batch_id,
                job_id=job_id,
                retry_of_attempt_id=kwargs.get("retry_of_attempt_id")
                or context.get("retry_of_attempt_id"),
                retry_of_job_id=kwargs.get("retry_of_job_id")
                or context.get("retry_of_job_id"),
                language=language,
                timestamps=bool(
                    kwargs.get("timestamps", request.options.get("timestamps", True))
                ),
                vad=bool(
                    kwargs.get("vad_filter", request.options.get("vad_use", False))
                ),
                is_cancelled=is_cancelled,
                ffmpeg_path=request.options.get("ffmpeg_path"),
            )
        except (ParakeetOnnxCancelled, ParakeetOnnxFailure):
            raise
        except Exception:
            raise failure(
                TranscriptionFailureCode.INFERENCE_FAILED,
                "Parakeet ONNX could not complete this transcription.",
                effective_device=ExecutionDevice.CPU,
                attempt_id=attempt_id,
                batch_id=batch_id,
                job_id=job_id,
                language=language,
            ) from None
        provenance = build_transcription_provenance_document(
            normalized,
            failed_attempt=kwargs.get("retry_source_failure_provenance")
            or context.get("retry_source_failure_provenance"),
        )
        return {
            "text": normalized.text,
            "segments": [
                {
                    "start": segment.start_seconds,
                    "end": segment.end_seconds,
                    "text": segment.text,
                }
                for segment in normalized.segments
            ],
            "transcription_model": normalized.provenance.model_id,
            "transcription_provenance": provenance,
        }

    def buffer_runner(
        source: BufferAudioSource,
        *,
        segment_end_frames: tuple[int, ...],
        attempt_id: str,
        job_id: str | None,
        language: str,
        transcription_context: dict[str, Any],
    ) -> dict[str, Any]:
        if source.sample_width != 2:
            raise _ProviderLoadFailure(
                TranscriptionFailureCode.UNSUPPORTED_CAPABILITY
            )
        current_context = (
            transcription_context
            if isinstance(transcription_context, dict)
            else {}
        )

        def buffer_failure() -> ParakeetOnnxFailure:
            return ParakeetOnnxFailure(
                TranscriptionFailureCode.INFERENCE_FAILED,
                "Parakeet ONNX could not complete this transcription.",
                attempt_id=attempt_id,
                batch_id=current_context.get("batch_id"),
                job_id=job_id,
                model_id=model_id,
                artifact_root=artifact_root,
                artifact_dependencies=artifact_dependencies,
                precision=precision,
                requested_language=language,
                effective_language=effective_language,
                effective_device=ExecutionDevice.CPU,
            )

        try:
            result = runtime.transcribe_buffer(
                source=source,
                segment_end_frames=segment_end_frames,
                attempt_id=attempt_id,
                language=language,
                job_id=job_id,
                is_cancelled=is_cancelled,
            )
        except (ParakeetOnnxCancelled, ParakeetOnnxFailure):
            raise
        except Exception:
            raise buffer_failure() from None
        normalized = replace(
            result.normalized,
            provenance=replace(
                result.normalized.provenance,
                batch_id=current_context.get("batch_id"),
            ),
        )
        provenance = build_transcription_provenance_document(
            normalized,
            failed_attempt=current_context.get(
                "retry_source_failure_provenance"
            ),
        )
        return {
            "text": normalized.text,
            "logical_segments": result.logical_segments,
            "duration": normalized.duration_seconds,
            "transcription_model": normalized.provenance.model_id,
            "transcription_provenance": provenance,
        }

    return ProviderRuntime(
        runner=runner,
        close=runtime.close,
        buffer_runner=buffer_runner,
    )


def _buffer_runner_kwargs(
    runner: Callable[..., dict[str, Any]],
    request: ExecutorRequest,
) -> dict[str, Any]:
    """Build current-request metadata accepted by one buffer runner."""

    context = request.options.get("transcription_context") or {}
    if not isinstance(context, dict):
        context = {}
    values = {
        "segment_end_frames": request.segment_end_frames,
        "attempt_id": request.attempt_id,
        "job_id": request.job_id,
        "language": request.options.get("language") or "en",
        "transcription_context": dict(context),
    }
    try:
        parameters = inspect.signature(runner).parameters.values()
    except (TypeError, ValueError):
        return values
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        return values
    accepted = {parameter.name for parameter in parameters}
    return {name: value for name, value in values.items() if name in accepted}


def _default_provider_builder(
    request: ExecutorRequest,
    model_root: Path | None,
    managed_handle: Any | None,
    is_cancelled: Callable[[], bool],
) -> ProviderRuntime:
    if request.identity.provider_id == "transcribe-cpp":
        return _transcribe_cpp_provider(
            request,
            model_root,
            managed_handle,
            is_cancelled,
        )
    if request.identity.provider_id == "parakeet-onnx":
        return _parakeet_provider(
            request,
            model_root,
            managed_handle,
            is_cancelled,
        )
    raise _ProviderLoadFailure(TranscriptionFailureCode.PROVIDER_UNAVAILABLE)


def _default_parse_job(
    file_path: str | Path,
    options: dict[str, Any],
    *,
    transcription_runner: TranscriptionRunner,
) -> dict[str, Any]:
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
        parse_local_file_for_ingest,
    )

    return parse_local_file_for_ingest(
        file_path,
        options,
        transcription_runner=transcription_runner,
    )


def _run_executor_worker(
    connection: Connection,
    admission_event: Any,
    cancellation_event: Any,
    generation: int,
    scratch_path: str,
    *,
    provider_builder: ProviderBuilder = _default_provider_builder,
    parse_job: ParseJob = _default_parse_job,
) -> None:
    identity = enter_worker_containment()
    connection.send(("bootstrap", identity))
    if not admission_event.wait(30.0):
        return
    from tldw_chatbook.Local_Ingestion.ingest_parse_worker import (
        silence_ingest_worker_import_noise,
    )

    silence_ingest_worker_import_noise()
    scratch = Path(scratch_path)
    if not scratch.is_dir():
        return
    tempfile.tempdir = str(scratch)
    connection.send(("ready", generation))
    resident: _ResidentRuntime | None = None
    try:
        while True:
            command = connection.recv()
            if command == ("close", generation):
                return
            if type(command) is not ExecutorRequest or command.generation != generation:
                continue
            request = command
            connection.send(
                ExecutorEvent(generation, request.attempt_id, WorkerPhase.PREPARING)
            )
            if cancellation_event.is_set():
                connection.send(_cancelled_failure(request))
                continue
            try:
                if resident is None:
                    connection.send(
                        ExecutorEvent(
                            generation,
                            request.attempt_id,
                            WorkerPhase.LOADING,
                        )
                    )
                    resident = _load_resident(
                        request,
                        provider_builder,
                        cancellation_event.is_set,
                    )
                else:
                    _validate_reuse(request, resident)
                if not resident.reported:
                    connection.send(
                        ExecutorResident(
                            generation,
                            request.attempt_id,
                            request.identity,
                            resident.managed_lease_refs,
                        )
                    )
                    resident.reported = True
                if cancellation_event.is_set():
                    connection.send(_cancelled_failure(request))
                    continue
                connection.send(
                    ExecutorEvent(
                        generation,
                        request.attempt_id,
                        WorkerPhase.TRANSCRIBING,
                    )
                )
                if type(request.source) is FileAudioSource:
                    payload = parse_job(
                        request.source.path,
                        dict(request.options),
                        transcription_runner=resident.provider.runner,
                    )
                elif (
                    request.identity.provider_id == "parakeet-onnx"
                    and resident.provider.buffer_runner is not None
                ):
                    payload = resident.provider.buffer_runner(
                        request.source,
                        **_buffer_runner_kwargs(
                            resident.provider.buffer_runner,
                            request,
                        ),
                    )
                else:
                    raise _ProviderLoadFailure(
                        TranscriptionFailureCode.UNSUPPORTED_CAPABILITY
                    )
                if cancellation_event.is_set():
                    connection.send(_cancelled_failure(request))
                    continue
                connection.send(
                    ExecutorEvent(
                        generation,
                        request.attempt_id,
                        WorkerPhase.POST_PROCESSING,
                    )
                )
                connection.send(ExecutorResult(generation, request.attempt_id, payload))
            except Exception as error:
                connection.send(
                    _failure_from_worker_exception(
                        request,
                        error,
                        cancelled=cancellation_event.is_set(),
                    )
                )
                if isinstance(error, LocalSourceChangedError) or (
                    isinstance(error, _ProviderLoadFailure)
                    and error.code
                    in {
                        TranscriptionFailureCode.MODEL_NOT_INSTALLED,
                        TranscriptionFailureCode.ARTIFACT_CORRUPT,
                    }
                ):
                    return
    except (EOFError, OSError):
        return
    finally:
        try:
            if resident is not None:
                resident.close()
        finally:
            connection.close()


def run_executor_worker(
    connection: Connection,
    admission_event: Any,
    cancellation_event: Any,
    generation: int,
    scratch_path: str,
) -> None:
    """Run one admitted resident worker until its parent closes it."""

    _run_executor_worker(
        connection,
        admission_event,
        cancellation_event,
        generation,
        scratch_path,
    )
