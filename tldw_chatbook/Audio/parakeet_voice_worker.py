"""Session-owned local STT process; model work never enters the audio interpreter."""

from __future__ import annotations

import contextlib
import multiprocessing
import threading
import time
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any, Literal
from uuid import uuid4

from tldw_chatbook.Utils.fd_protection import protect_file_descriptors
from tldw_chatbook.Utils.local_stt_providers import LOCAL_STT_PROVIDERS
from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

_MAX_PCM_BYTES = 960_000  # Ten seconds of mono 48 kHz PCM16.
_MAX_TEXT_CHARS = 65_536


@dataclass(frozen=True)
class LocalSttOptions:
    """Resolved nonsecret options; no app configuration crosses this boundary."""

    device: str | None = None
    compute_type: str | None = None
    precision: str | None = None

    def __post_init__(self) -> None:
        for value, choices in (
            (self.device, {"auto", "cpu", "cuda", "mps"}),
            (
                self.compute_type,
                {
                    "default",
                    "auto",
                    "int8",
                    "int8_float16",
                    "int8_float32",
                    "int8_bfloat16",
                    "int16",
                    "float16",
                    "float32",
                    "bfloat16",
                },
            ),
            (
                self.precision,
                {
                    "fp16",
                    "fp32",
                    "bf16",
                    "float16",
                    "float32",
                    "bfloat16",
                    "int8",
                    "f32",
                },
            ),
        ):
            if value is not None and (type(value) is not str or value not in choices):
                raise ValueError("local_stt_options_invalid")


@dataclass(frozen=True)
class _TranscriptResult:
    """Only bounded text and its existing interpretation may leave the model."""

    text: str | None = None
    partial: str | bool | None = None
    final: str | None = None
    cumulative: bool | None = None

    @classmethod
    def from_result(cls, result: Any) -> _TranscriptResult:
        if result is None:
            result = {}
        if isinstance(result, str):
            result = {"text": result}
        if not isinstance(result, dict):
            raise ValueError("result_invalid")
        values = {
            name: result.get(name)
            for name in ("text", "partial", "final", "cumulative")
        }
        for name, value in values.items():
            permitted = (
                (bool,)
                if name == "cumulative"
                else (str, bool)
                if name == "partial"
                else (str,)
            )
            if value is not None and type(value) not in permitted:
                raise ValueError("result_invalid")
        if (
            sum(len(value) for value in values.values() if isinstance(value, str))
            > _MAX_TEXT_CHARS
        ):
            raise ValueError("result_limit")
        return cls(**values)

    def as_dict(self) -> dict[str, str | bool]:
        return {
            name: value
            for name in ("text", "partial", "final", "cumulative")
            if (value := getattr(self, name)) is not None
        }


ParakeetFailureReason = Literal[
    "context_busy",
    "identity_mismatch",
    "closed",
    "disconnected",
    "timeout",
    "ownership_timeout",
    "unknown_native",
]
_PARAKEET_FAILURE_REASONS = frozenset(
    {
        "context_busy",
        "identity_mismatch",
        "closed",
        "disconnected",
        "timeout",
        "ownership_timeout",
        "unknown_native",
    }
)
_PARAKEET_FAILURE_MESSAGES = {
    "context_busy": "parakeet_worker_stream_active",
    "identity_mismatch": "parakeet_worker_stream_identity",
    "closed": "parakeet_worker_closed",
    "disconnected": "parakeet_worker_disconnected",
    "timeout": "parakeet_worker_timeout",
    "ownership_timeout": "parakeet_worker_ownership_timeout",
    "unknown_native": "parakeet_worker_context_unavailable",
}


class _ChildOwnedFailure(RuntimeError):
    """Tag child control-flow failures without inspecting provider messages."""

    def __init__(self, reason: ParakeetFailureReason) -> None:
        self.reason = reason
        super().__init__(reason)


class ParakeetFailure(RuntimeError):
    """An app-owned Parakeet failure with content-free diagnostic identity."""

    def __init__(
        self,
        reason: ParakeetFailureReason,
        *,
        native_exception_type: str | None = None,
    ) -> None:
        if reason not in _PARAKEET_FAILURE_REASONS:
            raise ValueError("unknown Parakeet failure reason")
        self.reason = reason
        self.native_exception_type = (
            native_exception_type
            if isinstance(native_exception_type, str)
            and native_exception_type.isidentifier()
            and len(native_exception_type) <= 128
            else None
        )
        message = _PARAKEET_FAILURE_MESSAGES[reason]
        if reason == "unknown_native" and self.native_exception_type is not None:
            message = f"parakeet_worker_{self.native_exception_type}"
        super().__init__(message)


class ParakeetServiceUnavailable(ParakeetFailure):
    """The model is fenced; replay through this service is not safe."""


class ParakeetRequestTimeout(ParakeetServiceUnavailable, TimeoutError):
    """An RPC timed out and fenced the model service."""

    def __init__(self, *, startup: bool = False) -> None:
        super().__init__("timeout")
        if startup:
            self.args = ("fd_protection_busy",)


class ParakeetOwnershipTimeout(TimeoutError, ParakeetFailure):
    """The lease waiter expired without changing the active model owner."""

    def __init__(self) -> None:
        ParakeetFailure.__init__(self, "ownership_timeout")


def _validate_pcm(pcm: bytes) -> None:
    if (
        not isinstance(pcm, bytes)
        or not pcm
        or len(pcm) % 2
        or len(pcm) > _MAX_PCM_BYTES
    ):
        raise ValueError("parakeet_worker_pcm_invalid")


class _ParakeetRuntime:
    def __init__(self, model: Any) -> None:
        self.model = model

    @staticmethod
    def _audio(pcm: bytes) -> Any:
        import mlx.core as mx
        import numpy as np

        samples = np.frombuffer(pcm, dtype="<i2")[::3].astype(np.float32) / 32_768
        return mx.array(samples, dtype=mx.float32)

    def open(self, context_size: tuple[int, int]) -> Any:
        return self.model.transcribe_stream(context_size=context_size)

    def push(self, stream: Any, pcm: bytes) -> str:
        stream.add_audio(self._audio(pcm))
        return stream.result.text

    def rolling(self, pcm: bytes) -> str:
        from parakeet_mlx.audio import get_logmel

        mel = get_logmel(self._audio(pcm), self.model.preprocessor_config)
        return self.model.generate(mel)[0].text


def _close_candidate(candidate: Any) -> None:
    for name in ("aclose", "close", "finalize"):
        close = getattr(candidate, name, None)
        if callable(close):
            _finish_cleanup(close)
            return


def _finish_cleanup(close: Any) -> None:
    # Existing candidates may expose async aclose. Its completion, including
    # failure, is part of this model owner's receipt, never a discarded coroutine.
    import asyncio
    import inspect

    result = close()
    if inspect.isawaitable(result):

        async def finish():
            await result

        asyncio.run(finish())


class _CandidateContext:
    """Retain the actual candidate and its native context in the model PID."""

    def __init__(self, candidate: Any) -> None:
        self.candidate = candidate
        self.closed = False

    def __enter__(self) -> Any:
        enter = getattr(self.candidate, "__enter__", None)
        return enter() if callable(enter) else self.candidate

    def __exit__(self, *args: Any) -> None:
        try:
            exit_context = getattr(self.candidate, "__exit__", None)
            if callable(exit_context):
                exit_context(*args)
        finally:
            self.close()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            _close_candidate(self.candidate)


class _LocalSttRuntime:
    def __init__(
        self,
        service: Any,
        *,
        provider: str,
        model: str | None,
        language: str,
        options: LocalSttOptions,
    ) -> None:
        self.service = service
        self.provider, self.model, self.language = provider, model, language
        self.options = {
            name: value
            for name in ("precision", "compute_type")
            if (value := getattr(options, name)) is not None
        }
        for name in ("device", "compute_type"):
            value = getattr(options, name)
            if value is not None:
                service.config[name] = value
        self.candidate = self._candidate()
        self.active = None
        self.parakeet = None
        if provider == "parakeet-mlx":
            if self.candidate is None:
                raise RuntimeError("parakeet_model_unavailable")
            self.parakeet = _ParakeetRuntime(self.candidate.model)
            self.kind = "parakeet"
        else:
            self.kind = (
                "process_audio"
                if callable(getattr(self.candidate, "process_audio", None))
                else "rolling"
            )

    def _candidate(self) -> Any:
        return self.service.create_streaming_transcriber(
            provider=self.provider,
            model=self.model,
            source_lang=self.language,
            **self.options,
        )

    def open(self, context_size: tuple[int, int]) -> Any:
        if self.parakeet is not None:
            return self.parakeet.open(context_size)
        candidate = self.candidate if self.candidate is not None else self._candidate()
        self.candidate = None
        self.active = _CandidateContext(candidate)
        if not callable(getattr(candidate, "process_audio", None)):
            raise RuntimeError("native_streaming_stt_unavailable")
        return self.active

    def push(self, stream: Any, pcm: bytes) -> Any:
        return (
            self.parakeet.push(stream, pcm)
            if self.parakeet is not None
            else stream.process_audio(pcm)
        )

    def rolling(self, pcm: bytes) -> Any:
        if self.parakeet is not None:
            return self.parakeet.rolling(pcm)
        return self.service.transcribe_buffer(
            audio_data=pcm,
            sample_rate=48_000,
            channels=1,
            sample_width=2,
            provider=self.provider,
            model=self.model,
            language=self.language,
            **self.options,
        )

    def close(self) -> None:
        try:
            if self.active is not None:
                self.active.close()
            if self.candidate is not None:
                _close_candidate(self.candidate)
        finally:
            _finish_cleanup(self.service.cleanup)


def _load_runtime(
    model: str | None,
    language: str,
    *,
    provider: str = "parakeet-mlx",
    options: LocalSttOptions | None = None,
) -> _LocalSttRuntime:
    # Service/config/model imports and native objects belong only to this PID.
    # Parakeet retains its in-memory path and configured precision resolution.
    from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService

    if provider == "parakeet-onnx":
        from tldw_chatbook.STT.executor_worker import ResidentBufferRuntime

        service = TranscriptionService(local_buffer_owner=ResidentBufferRuntime())
    else:
        service = TranscriptionService()
    try:
        return _LocalSttRuntime(
            service,
            provider=provider,
            model=model,
            language=language,
            options=options or LocalSttOptions(),
        )
    except BaseException:
        _finish_cleanup(service.cleanup)
        raise


def _serve(
    connection: Any, runtime_factory: Any, model: str | None, language: str
) -> None:
    # Libraries can log provider errors containing content. The child returns
    # only categorical errors, never traceback text or captured audio on stdout.
    import os
    import sys

    with open(os.devnull, "w") as quiet:
        os.dup2(quiet.fileno(), 1)
        os.dup2(quiet.fileno(), 2)
        sys.stdout = quiet
        sys.stderr = quiet
        runtime = context = stream = stream_id = None
        graceful = False
        abnormal = False
        try:
            while True:
                operation, identity, payload = connection.recv()
                if operation == "close":
                    graceful = True
                    break
                try:
                    result = ""
                    if runtime is None:
                        try:
                            runtime = runtime_factory(model, language)
                        except BaseException:
                            abnormal = True
                            connection.send((False, ("terminal", "unknown_native")))
                            break
                    if operation == "ready":
                        result = getattr(runtime, "kind", "parakeet")
                    elif operation == "open":
                        if context is not None:
                            raise _ChildOwnedFailure("context_busy")
                        try:
                            context = runtime.open(payload)
                            stream = context.__enter__()
                        except BaseException:
                            # Entry can mutate model mode before raising. Retire
                            # this runtime instead of advertising an empty slot.
                            context = None
                            abnormal = True
                            connection.send((False, ("terminal", "unknown_native")))
                            break
                        stream_id = identity
                    elif operation in ("push", "exit"):
                        if context is None or identity != stream_id:
                            raise _ChildOwnedFailure("identity_mismatch")
                        if operation == "push":
                            _validate_pcm(payload)
                            result = _TranscriptResult.from_result(
                                runtime.push(stream, payload)
                            )
                        else:
                            try:
                                context.__exit__(None, None, None)
                            except BaseException:
                                context = None
                                abnormal = True
                                connection.send((False, ("terminal", "unknown_native")))
                                break
                            context = stream = stream_id = None
                    elif operation == "rolling":
                        if context is not None:
                            raise _ChildOwnedFailure("context_busy")
                        _validate_pcm(payload)
                        result = _TranscriptResult.from_result(runtime.rolling(payload))
                    else:
                        raise ValueError("unknown_operation")
                    connection.send((True, result))
                except Exception as exc:
                    result = (
                        ("reason", exc.reason)
                        if isinstance(exc, _ChildOwnedFailure)
                        else ("exception", type(exc).__name__)
                    )
                    connection.send((False, result))
        except (EOFError, OSError):
            pass
        finally:
            if context is not None:
                try:
                    context.__exit__(None, None, None)
                except BaseException:
                    abnormal = True
            close = getattr(runtime, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    abnormal = True
            outcome = (
                AttemptCleanupOutcome.CLEAN
                if graceful and not abnormal
                else AttemptCleanupOutcome.FORCE_CLOSED
            )
            try:
                with contextlib.suppress(OSError):
                    connection.send(("cleanup", outcome))
            finally:
                try:
                    connection.close()
                finally:
                    if outcome is AttemptCleanupOutcome.FORCE_CLOSED:
                        # Keep failed native owners and protecting leases alive
                        # through OS exit, not just until _serve returns before
                        # multiprocessing finalizers or thread shutdown run.
                        os._exit(1)


class _RemoteStream:
    def __init__(
        self, owner: ParakeetVoiceProcess, context_size: tuple[int, int]
    ) -> None:
        self._owner = owner
        self._context_size = context_size
        self._identity = uuid4().hex
        self.result = SimpleNamespace(text="")

    def __enter__(self) -> _RemoteStream:
        self._owner._request("open", self._identity, self._context_size)
        return self

    def add_pcm16(self, pcm: bytes) -> None:
        _validate_pcm(pcm)
        result = self._owner._request("push", self._identity, pcm)
        self.result.text = result.text or ""

    def process_audio(self, pcm: bytes) -> dict[str, str | bool]:
        _validate_pcm(pcm)
        return self._owner._request("push", self._identity, pcm).as_dict()

    def __exit__(self, *_args: Any) -> None:
        # A fence is not evidence that this context restored model mode.
        # Confirmed process death is the other valid cleanup boundary.
        if self._owner.reaped:
            return
        self._owner._request("exit", self._identity)


class _RemoteAudioCandidate:
    def __init__(self, owner: ParakeetVoiceProcess) -> None:
        self._owner = owner

    def stream_context(self, *, context_size: tuple[int, int]) -> _RemoteStream:
        return _RemoteStream(self._owner, context_size)

    def process_audio(self, pcm: bytes) -> dict[str, str | bool]:
        # The transcription wrapper holds a context for the entire native turn.
        # Direct candidate use still gets an identified, checked single burst.
        with self.stream_context(context_size=(64, 64)) as stream:
            return stream.process_audio(pcm)


class ParakeetVoiceProcess:
    """Blocking STT service proxy, called from the session's existing workers.

    One serialized pipe bounds in-flight work to one request. Closing does not
    wait on the request lock: it reaps a stuck decoder and wakes its caller.
    """

    def __init__(
        self,
        *,
        model: str | None,
        language: str,
        provider: str = "parakeet-mlx",
        options: LocalSttOptions | None = None,
    ) -> None:
        if (
            type(provider) is not str
            or provider not in LOCAL_STT_PROVIDERS
            or (
                model is not None
                and (type(model) is not str or not model or len(model) > 512)
            )
            or type(language) is not str
            or not language
            or len(language) > 32
            or (options is not None and type(options) is not LocalSttOptions)
        ):
            raise ValueError("local_stt_settings_invalid")
        self._model = model
        self._language = language
        self._runtime_factory = partial(
            _load_runtime, provider=provider, options=options
        )
        self._request_timeout = 30.0
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._process = None
        self._connection = None
        self._cleanup_outcome = AttemptCleanupOutcome.CLEAN
        self._abnormal = False

    def _start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise ParakeetServiceUnavailable("closed")
            if self._process is not None:
                return
            ctx = multiprocessing.get_context("spawn")
            with protect_file_descriptors(timeout=0.5):
                self._connection, child = ctx.Pipe()
                self._process = ctx.Process(
                    target=_serve,
                    args=(child, self._runtime_factory, self._model, self._language),
                    name="speculative-voice-parakeet",
                    daemon=True,
                )
                try:
                    self._process.start()
                    self._cleanup_outcome = AttemptCleanupOutcome.DETACHED
                except BaseException:
                    self._connection.close()
                    self._process = None
                    raise
                finally:
                    child.close()

    def _request(self, operation: str, identity: str = "", payload: Any = None) -> Any:
        with self._lock:
            try:
                self._start()
            except TimeoutError:
                self.close()
                raise ParakeetRequestTimeout(startup=True) from None
            try:
                self._connection.send((operation, identity, payload))
                timeout = 120.0 if operation == "ready" else self._request_timeout
                deadline = time.monotonic() + timeout
                while not self._connection.poll(
                    min(0.05, max(0.0, deadline - time.monotonic()))
                ):
                    if self._closed:
                        raise ParakeetServiceUnavailable("closed")
                    if time.monotonic() >= deadline:
                        raise ParakeetRequestTimeout()
                ok, result = self._connection.recv()
                if type(ok) is not bool:
                    # A terminal cleanup receipt can arrive after BaseException
                    # interrupted inference. It is never an RPC success record.
                    raise ValueError("parakeet_worker_response_invalid")
                if ok:
                    if operation in ("push", "rolling"):
                        valid = type(result) is _TranscriptResult
                    elif operation == "ready":
                        valid = type(result) is str and result in {
                            "parakeet",
                            "process_audio",
                            "rolling",
                        }
                    else:
                        valid = type(result) is str and result == ""
                    if not valid:
                        raise ValueError("parakeet_worker_response_invalid")
            except (EOFError, OSError, TimeoutError, ValueError) as exc:
                was_closed = self._closed
                self._abnormal = True
                self.close()
                if isinstance(exc, ParakeetRequestTimeout):
                    raise
                if isinstance(exc, TimeoutError):
                    raise ParakeetRequestTimeout() from None
                if was_closed:
                    raise ParakeetServiceUnavailable("closed") from None
                raise ParakeetServiceUnavailable("disconnected") from None
            if not ok:
                if result == ("terminal", "unknown_native"):
                    self._abnormal = True
                    self.close()
                    raise ParakeetServiceUnavailable("unknown_native")
                if result == ("reason", "context_busy"):
                    raise ParakeetFailure("context_busy")
                if result == ("reason", "identity_mismatch"):
                    raise ParakeetFailure("identity_mismatch")
                native_exception_type = (
                    result[1]
                    if isinstance(result, tuple)
                    and len(result) == 2
                    and result[0] == "exception"
                    else None
                )
                raise ParakeetFailure(
                    "unknown_native",
                    native_exception_type=native_exception_type,
                )
            return result

    def create_streaming_transcriber(self, **_kwargs: Any) -> Any:
        """Return a disposable candidate; the session retains process ownership."""
        kind = self._request("ready")
        if kind == "parakeet":
            return SimpleNamespace(model=self)
        if kind == "process_audio":
            return _RemoteAudioCandidate(self)
        if kind == "rolling":
            return None
        self._abnormal = True
        self.close()
        raise ParakeetServiceUnavailable("unknown_native")

    @property
    def closed(self) -> bool:
        """Whether shutdown or a terminal RPC failure has fenced this service."""
        return self._closed

    @property
    def reaped(self) -> bool:
        """Whether a fenced runtime is confirmed absent, not merely closing."""
        return self._closed and (self._process is None or not self._process.is_alive())

    @property
    def cleanup_outcome(self) -> AttemptCleanupOutcome:
        """Actual model-owner disposition; process absence alone is not CLEAN."""
        return self._cleanup_outcome

    def transcribe_stream(self, *, context_size: tuple[int, int]) -> _RemoteStream:
        return _RemoteStream(self, context_size)

    def transcribe_buffer(
        self,
        *,
        audio_data: bytes,
        sample_rate: int,
        channels: int,
        sample_width: int,
        **_kwargs: Any,
    ) -> dict[str, str | bool]:
        """Recognize bounded pipeline PCM directly, without microphone files."""
        if (sample_rate, channels, sample_width) != (48_000, 1, 2):
            raise ValueError("parakeet_worker_format_invalid")
        _validate_pcm(audio_data)
        return self._request("rolling", payload=audio_data).as_dict()

    def close(self) -> None:
        """Reap a started process with bounded joins, even during inference."""
        with self._lifecycle_lock:
            if self.reaped:
                return
            self._closed = True
            process = self._process
            if process is None:
                return
            owns_pipe = self._lock.acquire(blocking=False)
            if owns_pipe:
                try:
                    with contextlib.suppress(OSError):
                        self._connection.send(("close", "", None))
                finally:
                    self._lock.release()
            process.join(1)
            if process.is_alive():
                self._abnormal = True
                process.terminate()
                process.join(1)
            if process.is_alive():
                self._abnormal = True
                process.kill()
                process.join(1)
            receipt = None
            if owns_pipe:
                with contextlib.suppress(EOFError, OSError):
                    if self._connection.poll():
                        receipt = self._connection.recv()
            self._connection.close()
            if process.is_alive():
                self._abnormal = True
                self._cleanup_outcome = AttemptCleanupOutcome.DETACHED
                raise RuntimeError("parakeet_worker_not_reaped")
            self._cleanup_outcome = (
                AttemptCleanupOutcome.CLEAN
                if not self._abnormal
                and process.exitcode == 0
                and receipt == ("cleanup", AttemptCleanupOutcome.CLEAN)
                else AttemptCleanupOutcome.FORCE_CLOSED
            )


class LocalVoiceSttProcess(ParakeetVoiceProcess):
    """The same bounded owner for any resolved allowlisted local STT provider."""

    def __init__(
        self,
        *,
        provider: str,
        model: str | None,
        language: str,
        options: LocalSttOptions | None = None,
    ) -> None:
        super().__init__(
            provider=provider, model=model, language=language, options=options
        )
