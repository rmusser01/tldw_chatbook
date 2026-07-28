"""Dependency-free coordination for one exact speech-to-text provider."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import NoReturn

from .contracts import (
    BufferAudioSource,
    DeviceFailureOrigin,
    DeviceRetryPolicy,
    ExecutionDevice,
    FileAudioSource,
    InputKind,
    LanguageInputMode,
    PipelineCapabilities,
    ResolvedTranscriptionRequest,
    TimestampGranularity,
    TranscriptionAction,
    TranscriptionFailure,
    TranscriptionFailureCode,
    TranscriptionPhase,
    TranscriptionProgress,
    TranscriptionProvenance,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionWarningCode,
)
from .registry import (
    ProviderRegistry,
    ProviderTranscriptionOutput,
    RuntimeCapabilityError,
    RuntimeObservation,
)
from .routing import RoutingResolutionError, TranscriptionRouter


class _AdapterProgressGate:
    """Forward only monotonic in-phase adapter progress with request identity."""

    __slots__ = (
        "_active",
        "_coordinator",
        "_delivery_lock",
        "_failure",
        "_last_fraction",
        "_request",
        "_resolved",
    )

    def __init__(
        self,
        coordinator: TranscriptionCoordinator,
        request: TranscriptionRequest,
        resolved: ResolvedTranscriptionRequest,
    ) -> None:
        self._coordinator = coordinator
        self._request = request
        self._resolved = resolved
        self._active = True
        self._delivery_lock = RLock()
        self._last_fraction: float | None = None
        self._failure: TranscriptionCoordinatorError | None = None

    def __call__(self, event: TranscriptionProgress) -> None:
        with self._delivery_lock:
            if (
                not self._active
                or type(event) is not TranscriptionProgress
                or event.phase is not TranscriptionPhase.TRANSCRIBING
                or (
                    event.fraction is not None
                    and self._last_fraction is not None
                    and event.fraction < self._last_fraction
                )
            ):
                return
            if event.fraction is not None:
                self._last_fraction = event.fraction
            forwarded = TranscriptionProgress(
                attempt_id=self._request.attempt_id,
                batch_id=self._request.batch_id,
                job_id=self._request.job_id,
                phase=TranscriptionPhase.TRANSCRIBING,
                fraction=event.fraction,
                detail_code=event.detail_code,
            )
            try:
                self._coordinator._deliver_progress(
                    self._request,
                    forwarded,
                    resolved=self._resolved,
                )
            except TranscriptionCoordinatorError as error:
                self._failure = error
                self._active = False
                raise

    def close(self) -> None:
        """Stop retained adapter callbacks from reaching the caller."""

        with self._delivery_lock:
            self._active = False

    def raise_if_failed(self) -> None:
        """Preserve callback failure even if an adapter swallowed it."""

        if self._failure is not None:
            raise self._failure


def device_retry_policy_for_failure(
    *,
    requested_device: ExecutionDevice,
    failed_device: ExecutionDevice,
    origin: DeviceFailureOrigin,
    retry_device: ExecutionDevice,
    worker_will_recycle: bool,
) -> DeviceRetryPolicy:
    """Return the coordinator's one safe same-model device retry policy."""

    return DeviceRetryPolicy.for_failure(
        requested_device=requested_device,
        failed_device=failed_device,
        origin=origin,
        retry_device=retry_device,
        worker_will_recycle=worker_will_recycle,
    )


@dataclass(frozen=True, slots=True)
class TranscriptionFailureDecision:
    """Immutable failure, contextual actions, and same-model device policy."""

    failure: TranscriptionFailure
    actions: tuple[TranscriptionAction, ...]
    device_retry_policy: DeviceRetryPolicy

    def __post_init__(self) -> None:
        if type(self.failure) is not TranscriptionFailure:
            raise TypeError("failure must be a TranscriptionFailure")
        if type(self.actions) is not tuple or not all(
            type(action) is TranscriptionAction for action in self.actions
        ):
            raise TypeError("actions must be a tuple of TranscriptionAction values")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must not contain duplicates")
        if type(self.device_retry_policy) is not DeviceRetryPolicy:
            raise TypeError("device_retry_policy must be a DeviceRetryPolicy")


class TranscriptionCoordinatorError(Exception):
    """A sanitized exception carrying one immutable failure decision."""

    __slots__ = ("decision",)

    def __init__(self, decision: TranscriptionFailureDecision) -> None:
        if type(decision) is not TranscriptionFailureDecision:
            raise TypeError("decision must be a TranscriptionFailureDecision")
        self.decision = decision
        super().__init__(decision.failure.message)

    @property
    def failure(self) -> TranscriptionFailure:
        """Return the immutable failure envelope."""

        return self.decision.failure

    @property
    def actions(self) -> tuple[TranscriptionAction, ...]:
        """Return immutable coordinator-selected future actions."""

        return self.decision.actions

    @property
    def device_retry_policy(self) -> DeviceRetryPolicy:
        """Return the explicit same-model device policy."""

        return self.decision.device_retry_policy

    def __repr__(self) -> str:
        """Render stable classifications without request identities."""

        return (
            f"{type(self).__name__}(code={self.failure.code.value!r}, "
            f"phase={self.failure.phase.value!r})"
        )

    def __str__(self) -> str:
        """Return only the fixed failure-contract message."""

        return self.failure.message


@dataclass(frozen=True, slots=True)
class TranscriptionCoordinator:
    """Resolve, preflight, and execute one selected provider adapter."""

    registry: ProviderRegistry
    router: TranscriptionRouter
    pipeline: PipelineCapabilities

    def __post_init__(self) -> None:
        if type(self.registry) is not ProviderRegistry:
            raise TypeError("registry must be a ProviderRegistry")
        if type(self.router) is not TranscriptionRouter:
            raise TypeError("router must be a TranscriptionRouter")
        if type(self.pipeline) is not PipelineCapabilities:
            raise TypeError("pipeline must be a PipelineCapabilities")

    def resolve(
        self,
        request: TranscriptionRequest,
    ) -> ResolvedTranscriptionRequest:
        """Route and validate declared/composed capabilities without probing."""

        if type(request) is not TranscriptionRequest:
            raise TypeError("request must be a TranscriptionRequest")
        try:
            resolved = self.router.resolve(request, self.registry)
        except RoutingResolutionError as error:
            self._raise_failure(
                request,
                error.code,
                phase=TranscriptionPhase.QUEUED,
                provider_id=error.provider_id,
                model_id=error.model_id,
            )
        self._validate_resolved_declaration(resolved, verify_routing=False)
        return resolved

    def preflight(
        self,
        resolved: ResolvedTranscriptionRequest,
    ) -> RuntimeObservation:
        """Probe and validate only the exact selected model adapter."""

        if type(resolved) is not ResolvedTranscriptionRequest:
            raise TypeError("resolved must be a ResolvedTranscriptionRequest")
        self._validate_resolved_declaration(resolved)
        request = resolved.request
        self._check_cancelled(
            request,
            phase=TranscriptionPhase.LOADING,
            resolved=resolved,
        )

        model = self.registry.model(resolved.provider_id, resolved.model_id)
        if model is None:
            self._raise_failure(
                request,
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                phase=TranscriptionPhase.LOADING,
                resolved=resolved,
            )
        adapter = self.registry.adapter_for_model(
            resolved.provider_id,
            resolved.model_id,
        )
        if adapter is None:
            self._raise_failure(
                request,
                TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
                phase=TranscriptionPhase.LOADING,
                resolved=resolved,
            )

        try:
            observation = adapter.probe(resolved.model_id)
        except Exception:
            self._check_cancelled(
                request,
                phase=TranscriptionPhase.LOADING,
                resolved=resolved,
            )
            self._raise_failure(
                request,
                TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
                phase=TranscriptionPhase.LOADING,
                resolved=resolved,
            )
        self._check_cancelled(
            request,
            phase=TranscriptionPhase.LOADING,
            resolved=resolved,
        )
        try:
            validated = self.registry.validate_observation(model, observation)
        except (RuntimeCapabilityError, TypeError, ValueError):
            self._raise_failure(
                request,
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                phase=TranscriptionPhase.LOADING,
                resolved=resolved,
            )
        if not validated.available:
            self._raise_failure(
                request,
                TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
                phase=TranscriptionPhase.LOADING,
                resolved=resolved,
            )

        runtime = validated.capabilities
        if runtime is None:
            self._raise_failure(
                request,
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                phase=TranscriptionPhase.LOADING,
                resolved=resolved,
            )
        if resolved.precision not in runtime.precisions:
            self._raise_failure(
                request,
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                phase=TranscriptionPhase.LOADING,
                resolved=resolved,
            )
        if (
            request.device is not ExecutionDevice.AUTO
            and request.device not in runtime.execution_devices
        ):
            self._raise_failure(
                request,
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                phase=TranscriptionPhase.LOADING,
                resolved=resolved,
            )
        self._check_cancelled(
            request,
            phase=TranscriptionPhase.LOADING,
            resolved=resolved,
        )
        return validated

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Execute exactly one selected adapter and normalize its output."""

        if type(request) is not TranscriptionRequest:
            raise TypeError("request must be a TranscriptionRequest")
        self._emit_progress(request, TranscriptionPhase.QUEUED)
        self._check_cancelled(request, phase=TranscriptionPhase.QUEUED)
        resolved = self.resolve(request)

        self._emit_progress(request, TranscriptionPhase.LOADING, resolved=resolved)
        observation = self.preflight(resolved)
        self._check_cancelled(
            request,
            phase=TranscriptionPhase.TRANSCRIBING,
            resolved=resolved,
        )

        adapter = self.registry.adapter_for_model(
            resolved.provider_id,
            resolved.model_id,
        )
        if adapter is None:
            self._raise_failure(
                request,
                TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
                phase=TranscriptionPhase.LOADING,
                resolved=resolved,
            )

        self._emit_progress(
            request,
            TranscriptionPhase.TRANSCRIBING,
            resolved=resolved,
        )
        self._check_cancelled(
            request,
            phase=TranscriptionPhase.TRANSCRIBING,
            resolved=resolved,
        )
        progress_gate: _AdapterProgressGate | None = None
        adapter_request = resolved
        if request.progress is not None:
            progress_gate = _AdapterProgressGate(self, request, resolved)
            adapter_request = replace(
                resolved,
                request=replace(request, progress=progress_gate),
            )
        try:
            output = adapter.transcribe(adapter_request)
        except Exception:
            if progress_gate is not None:
                progress_gate.close()
                progress_gate.raise_if_failed()
            self._check_cancelled(
                request,
                phase=TranscriptionPhase.TRANSCRIBING,
                resolved=resolved,
            )
            self._raise_failure(
                request,
                TranscriptionFailureCode.INFERENCE_FAILED,
                phase=TranscriptionPhase.TRANSCRIBING,
                resolved=resolved,
            )
        if progress_gate is not None:
            progress_gate.close()
            progress_gate.raise_if_failed()
        self._check_cancelled(
            request,
            phase=TranscriptionPhase.TRANSCRIBING,
            resolved=resolved,
        )

        self._emit_progress(
            request,
            TranscriptionPhase.POST_PROCESSING,
            resolved=resolved,
        )
        self._check_cancelled(
            request,
            phase=TranscriptionPhase.POST_PROCESSING,
            resolved=resolved,
        )
        try:
            result = self._normalize_output(
                resolved,
                observation,
                output,
            )
        except TranscriptionCoordinatorError:
            raise
        except (TypeError, ValueError):
            self._raise_failure(
                request,
                TranscriptionFailureCode.INFERENCE_FAILED,
                phase=TranscriptionPhase.POST_PROCESSING,
                resolved=resolved,
                effective_device=self._effective_device(output),
            )
        self._emit_progress(
            request,
            TranscriptionPhase.COMPLETE,
            resolved=resolved,
            effective_device=result.provenance.effective_device,
        )
        return result

    def failure_decision(
        self,
        request: TranscriptionRequest,
        failure: TranscriptionFailure,
        *,
        device_retry_policy: DeviceRetryPolicy | None = None,
    ) -> TranscriptionFailureDecision:
        """Select contextual actions without probing, retrying, or mutating."""

        if type(request) is not TranscriptionRequest:
            raise TypeError("request must be a TranscriptionRequest")
        if type(failure) is not TranscriptionFailure:
            raise TypeError("failure must be a TranscriptionFailure")
        policy = device_retry_policy or DeviceRetryPolicy.no_retry()
        if type(policy) is not DeviceRetryPolicy:
            raise TypeError("device_retry_policy must be a DeviceRetryPolicy")

        actions = list(self._base_actions(failure.code))
        retry_action = TranscriptionAction.RETRY_WITH_FASTER_WHISPER
        if retry_action in actions and (
            failure.provider_id == self.router.policy.faster_whisper_provider_id
            or not self._faster_whisper_satisfies(request)
        ):
            actions.remove(retry_action)

        selected = self.registry.model(failure.provider_id, failure.model_id)
        if (
            failure.code is TranscriptionFailureCode.UNSUPPORTED_LANGUAGE
            and selected is not None
            and selected.capabilities.language_input_mode
            is LanguageInputMode.AUTOMATIC_ONLY
        ):
            actions.append(TranscriptionAction.CHANGE_LANGUAGE_TO_AUTO)

        return TranscriptionFailureDecision(
            failure=failure,
            actions=tuple(actions),
            device_retry_policy=policy,
        )

    @staticmethod
    def _base_actions(
        code: TranscriptionFailureCode,
    ) -> tuple[TranscriptionAction, ...]:
        retry_faster = TranscriptionAction.RETRY_WITH_FASTER_WHISPER
        retry_same = TranscriptionAction.RETRY_SAME_CONFIGURATION
        if code in {
            TranscriptionFailureCode.MODEL_NOT_INSTALLED,
            TranscriptionFailureCode.ARTIFACT_CORRUPT,
        }:
            return (
                TranscriptionAction.INSTALL_MODEL,
                TranscriptionAction.CHOOSE_INSTALLED_MODEL,
                retry_faster,
            )
        if code is TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE:
            return (TranscriptionAction.CHOOSE_INSTALLED_MODEL, retry_faster)
        if code in {
            TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
            TranscriptionFailureCode.ENGINE_CRASHED,
        }:
            return (retry_same, retry_faster)
        if code is TranscriptionFailureCode.CANCELLED:
            return (retry_same,)
        return (retry_faster,)

    def _faster_whisper_satisfies(
        self,
        request: TranscriptionRequest,
    ) -> bool:
        provider = self.registry.provider(self.router.policy.faster_whisper_provider_id)
        model = self.registry.model(
            self.router.policy.faster_whisper_provider_id,
            self.router.policy.faster_whisper_model_id,
        )
        if provider is None or model is None:
            return False
        if (
            not provider.local_processing
            and not request.privacy.allow_remote_processing
        ):
            return False
        if (
            type(request.source) is BufferAudioSource
            and self.pipeline.requires_disk_staging_for_buffer
            and not request.privacy.allow_disk_staging
        ):
            return False

        capabilities = model.capabilities
        input_kind = (
            InputKind.FILE
            if type(request.source) is FileAudioSource
            else InputKind.BUFFER
        )
        language = request.language or "en"
        if language == "auto":
            language_supported = capabilities.automatic_language
        else:
            language_supported = (
                capabilities.language_input_mode is not LanguageInputMode.AUTOMATIC_ONLY
                and language in capabilities.languages
            )
        precision = request.precision or model.default_precision
        timestamp_supported = (
            request.timestamps is TimestampGranularity.NONE
            or request.timestamps in capabilities.timestamps
            or request.timestamps in self.pipeline.timestamps
        )
        return (
            language_supported
            and request.task in capabilities.tasks
            and input_kind in capabilities.inputs
            and precision in capabilities.precisions
            and (
                request.device is ExecutionDevice.AUTO
                or request.device in capabilities.execution_devices
            )
            and timestamp_supported
            and (not request.vad or capabilities.vad or self.pipeline.vad)
            and (
                not request.diarization
                or capabilities.diarization
                or self.pipeline.diarization
            )
        )

    def _validate_resolved_declaration(
        self,
        resolved: ResolvedTranscriptionRequest,
        *,
        verify_routing: bool = True,
    ) -> None:
        request = resolved.request
        if verify_routing:
            try:
                expected = self.router.resolve(request, self.registry)
            except RoutingResolutionError as error:
                self._raise_failure(
                    request,
                    error.code,
                    phase=TranscriptionPhase.QUEUED,
                    provider_id=error.provider_id,
                    model_id=error.model_id,
                )
            if expected != resolved:
                self._raise_failure(
                    request,
                    TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
                    phase=TranscriptionPhase.QUEUED,
                    resolved=resolved,
                )

        provider = self.registry.provider(resolved.provider_id)
        model = self.registry.model(resolved.provider_id, resolved.model_id)
        if provider is None or model is None:
            self._raise_failure(
                request,
                TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
                phase=TranscriptionPhase.QUEUED,
                resolved=resolved,
            )
        if (
            not provider.local_processing
            and not request.privacy.allow_remote_processing
        ):
            self._raise_failure(
                request,
                TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
                phase=TranscriptionPhase.QUEUED,
                resolved=resolved,
            )
        if (
            type(request.source) is BufferAudioSource
            and self.pipeline.requires_disk_staging_for_buffer
            and not request.privacy.allow_disk_staging
        ):
            self._raise_failure(
                request,
                TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
                phase=TranscriptionPhase.QUEUED,
                resolved=resolved,
            )

        input_kind = (
            InputKind.FILE
            if type(request.source) is FileAudioSource
            else InputKind.BUFFER
        )
        capabilities = model.capabilities
        requested_timestamp_supported = (
            request.timestamps is TimestampGranularity.NONE
            or request.timestamps in capabilities.timestamps
            or request.timestamps in self.pipeline.timestamps
        )
        if (
            input_kind not in capabilities.inputs
            or request.task not in capabilities.tasks
            or resolved.precision not in capabilities.precisions
            or (
                request.device is not ExecutionDevice.AUTO
                and request.device not in capabilities.execution_devices
            )
            or not requested_timestamp_supported
            or request.vad
            and not (capabilities.vad or self.pipeline.vad)
            or request.diarization
            and not (capabilities.diarization or self.pipeline.diarization)
        ):
            self._raise_failure(
                request,
                TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
                phase=TranscriptionPhase.QUEUED,
                resolved=resolved,
            )

    def _normalize_output(
        self,
        resolved: ResolvedTranscriptionRequest,
        observation: RuntimeObservation,
        output: ProviderTranscriptionOutput,
    ) -> TranscriptionResult:
        if type(output) is not ProviderTranscriptionOutput:
            raise TypeError("adapter output must be ProviderTranscriptionOutput")
        runtime = observation.capabilities
        if runtime is None:
            raise ValueError("runtime capabilities are required")
        request = resolved.request
        model = self.registry.model(resolved.provider_id, resolved.model_id)
        if model is None:
            raise ValueError("selected model declaration is required")

        produced = output.produced_capabilities
        if output.effective_device not in runtime.execution_devices:
            raise ValueError("effective device was not observed at runtime")
        if (
            request.device is not ExecutionDevice.AUTO
            and output.effective_device is not request.device
        ):
            raise ValueError("effective device contradicts the request")
        if output.effective_language != resolved.effective_language:
            raise ValueError("effective language contradicts routing")
        if output.detected_language == "auto":
            raise ValueError("detected language must identify a concrete language")
        if (
            output.detected_language is not None
            and resolved.requested_language != "auto"
        ):
            raise ValueError("detected language was not requested")
        if (
            TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED
            in resolved.warning_codes
            and output.detected_language is not None
        ):
            raise ValueError("routing-only language has no trusted detection")
        if (
            TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED in output.warnings
            and TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED
            not in resolved.warning_codes
        ):
            raise ValueError("provider warning contradicts routed language semantics")
        if produced.timestamps is not request.timestamps:
            raise ValueError("produced timestamps contradict the request")
        if produced.vad is not request.vad:
            raise ValueError("produced VAD contradicts the request")
        if produced.diarization is not request.diarization:
            raise ValueError("produced diarization contradicts the request")
        if produced.punctuation and not model.capabilities.punctuation:
            raise ValueError("punctuation exceeds composed capabilities")
        if produced.capitalization and not model.capabilities.capitalization:
            raise ValueError("capitalization exceeds composed capabilities")

        warnings = tuple(dict.fromkeys((*resolved.warning_codes, *output.warnings)))
        provenance = TranscriptionProvenance(
            schema_version=1,
            attempt_id=request.attempt_id,
            batch_id=request.batch_id,
            job_id=request.job_id,
            retry_of_attempt_id=request.retry_of_attempt_id,
            retry_of_job_id=request.retry_of_job_id,
            provider_id=resolved.provider_id,
            model_id=resolved.model_id,
            artifact_root=None,
            artifact_dependencies=(),
            precision=resolved.precision,
            requested_device=request.device,
            effective_device=output.effective_device,
            requested_language=resolved.requested_language,
            effective_language=resolved.effective_language,
            detected_language=output.detected_language,
            task=request.task,
        )
        return TranscriptionResult(
            text=output.text,
            segments=output.segments,
            provenance=provenance,
            produced_capabilities=produced,
            duration_seconds=output.duration_seconds,
            timings=output.timings,
            warnings=warnings,
        )

    @staticmethod
    def _effective_device(
        output: object,
    ) -> ExecutionDevice | None:
        if type(output) is ProviderTranscriptionOutput:
            return output.effective_device
        return None

    def _check_cancelled(
        self,
        request: TranscriptionRequest,
        *,
        phase: TranscriptionPhase,
        resolved: ResolvedTranscriptionRequest | None = None,
    ) -> None:
        if request.cancellation is None:
            return
        try:
            cancelled = request.cancellation.is_cancelled()
        except Exception:
            self._raise_failure(
                request,
                TranscriptionFailureCode.INFERENCE_FAILED,
                phase=phase,
                resolved=resolved,
            )
        if type(cancelled) is not bool:
            self._raise_failure(
                request,
                TranscriptionFailureCode.INFERENCE_FAILED,
                phase=phase,
                resolved=resolved,
            )
        if cancelled:
            self._raise_failure(
                request,
                TranscriptionFailureCode.CANCELLED,
                phase=phase,
                resolved=resolved,
            )

    def _emit_progress(
        self,
        request: TranscriptionRequest,
        phase: TranscriptionPhase,
        *,
        resolved: ResolvedTranscriptionRequest | None = None,
        effective_device: ExecutionDevice | None = None,
    ) -> None:
        if request.progress is None:
            return
        event = TranscriptionProgress(
            attempt_id=request.attempt_id,
            batch_id=request.batch_id,
            job_id=request.job_id,
            phase=phase,
        )
        self._deliver_progress(
            request,
            event,
            resolved=resolved,
            effective_device=effective_device,
        )

    def _deliver_progress(
        self,
        request: TranscriptionRequest,
        event: TranscriptionProgress,
        *,
        resolved: ResolvedTranscriptionRequest | None,
        effective_device: ExecutionDevice | None = None,
    ) -> None:
        if request.progress is None:
            return
        try:
            request.progress(event)
        except Exception:
            self._raise_failure(
                request,
                TranscriptionFailureCode.INFERENCE_FAILED,
                phase=event.phase,
                resolved=resolved,
                effective_device=effective_device,
            )

    def _raise_failure(
        self,
        request: TranscriptionRequest,
        code: TranscriptionFailureCode,
        *,
        phase: TranscriptionPhase,
        resolved: ResolvedTranscriptionRequest | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
        effective_device: ExecutionDevice | None = None,
    ) -> NoReturn:
        selected_provider_id = (
            resolved.provider_id
            if resolved is not None
            else provider_id or request.provider_id
        )
        selected_model_id = (
            resolved.model_id
            if resolved is not None
            else model_id or request.model_id or "unresolved"
        )
        model = self.registry.model(selected_provider_id, selected_model_id)
        precision = (
            resolved.precision
            if resolved is not None
            else request.precision
            or (model.default_precision if model is not None else "unresolved")
        )
        failure = TranscriptionFailure(
            code=code,
            attempt_id=request.attempt_id,
            batch_id=request.batch_id,
            job_id=request.job_id,
            phase=phase,
            provider_id=selected_provider_id,
            model_id=selected_model_id,
            artifact_root=None,
            precision=precision,
            requested_device=request.device,
            effective_device=effective_device,
        )
        decision = self.failure_decision(request, failure)
        raise TranscriptionCoordinatorError(decision) from None


__all__ = [
    "TranscriptionCoordinator",
    "TranscriptionCoordinatorError",
    "TranscriptionFailureDecision",
    "device_retry_policy_for_failure",
]
