"""Spawn-importable helpers for local STT executor process tests."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


_resident_runtime_loads = 0


class _FakeResidentRuntime:
    """Small provider runtime used only inside spawned executor tests."""

    def __init__(self, load_number: int) -> None:
        self.load_number = load_number

    def transcribe(self, audio_path: str, **_kwargs: Any) -> dict[str, Any]:
        return {
            "text": f"transcript:{Path(audio_path).name}",
            "segments": [],
            "runtime_load_number": self.load_number,
        }

    def close(self) -> None:
        return


def _protocol_provider_builder(
    request: Any,
    _model_root: Path | None,
    _managed_handle: Any | None,
    _is_cancelled: Any,
) -> Any:
    """Build a deterministic runtime that exposes both executor source paths."""

    from tldw_chatbook.STT.executor_worker import ProviderRuntime

    def file_runner(audio_path: str, **kwargs: Any) -> dict[str, Any]:
        return {"audio_path": audio_path, "kwargs": kwargs}

    def buffer_runner(
        source: Any, *, segment_end_frames: tuple[int, ...]
    ) -> dict[str, Any]:
        return {
            "audio_bytes": len(source.audio),
            "sample_rate": source.sample_rate,
            "segment_end_frames": segment_end_frames,
        }

    return ProviderRuntime(
        runner=file_runner,
        buffer_runner=(
            None if request.options.get("test_no_buffer_runner") else buffer_runner
        ),
        close=lambda: None,
    )


def _protocol_parse_job(
    file_path: str | Path,
    options: dict[str, Any],
    *,
    transcription_runner: Any,
) -> dict[str, Any]:
    """Expose the exact file-parser inputs in a worker result payload."""

    return {
        "file_path": str(file_path),
        "options": dict(options),
        "runner_payload": transcription_runner(
            str(file_path),
            provider=options.get("transcription_provider"),
        ),
    }


def protocol_executor_worker(
    connection: Connection,
    admission_event: Any,
    cancellation_event: Any,
    generation: int,
    scratch_path: str,
) -> None:
    """Run the real worker loop with observable file and buffer seams."""

    from tldw_chatbook.STT.executor_worker import _run_executor_worker

    _run_executor_worker(
        connection,
        admission_event,
        cancellation_event,
        generation,
        scratch_path,
        provider_builder=_protocol_provider_builder,
        parse_job=_protocol_parse_job,
    )


def _fake_provider_builder(
    _request: Any,
    _model_root: Path | None,
    managed_handle: Any | None,
    _is_cancelled: Any,
) -> Any:
    global _resident_runtime_loads
    from tldw_chatbook.STT.executor_worker import ProviderRuntime

    _resident_runtime_loads += 1
    runtime = _FakeResidentRuntime(_resident_runtime_loads)
    if managed_handle is not None:
        assert all(path.is_dir() for _reference, path in managed_handle.paths)
    return ProviderRuntime(runner=runtime.transcribe, close=runtime.close)


def _fake_parse_job(
    file_path: str | Path,
    options: dict[str, Any],
    *,
    transcription_runner: Any,
) -> dict[str, Any]:
    if options.get("test_worker_fail_parse"):
        raise RuntimeError("synthetic parse failure")
    if options.get("test_worker_crash"):
        os._exit(71)
    if options.get("test_worker_hold"):
        while True:
            time.sleep(1.0)
    transcription = transcription_runner(
        str(file_path),
        provider=options.get("transcription_provider"),
    )
    payload = {
        "content": transcription["text"],
        "runtime_load_number": transcription["runtime_load_number"],
    }
    fallback_device = options.get("_local_stt_cpu_fallback_requested_device")
    if fallback_device is not None:
        payload["cpu_fallback_requested_device"] = fallback_device
    return payload


def resident_executor_worker(
    connection: Connection,
    admission_event: Any,
    cancellation_event: Any,
    generation: int,
    scratch_path: str,
) -> None:
    """Run the real resident loop with dependency-free fake provider work."""

    from tldw_chatbook.STT.executor_worker import _run_executor_worker

    _run_executor_worker(
        connection,
        admission_event,
        cancellation_event,
        generation,
        scratch_path,
        provider_builder=_fake_provider_builder,
        parse_job=_fake_parse_job,
    )


def _device_retry_provider_builder(
    request: Any,
    _model_root: Path | None,
    _managed_handle: Any | None,
    _is_cancelled: Any,
) -> Any:
    from tldw_chatbook.STT.contracts import (
        DeviceFailureOrigin,
        ExecutionDevice,
    )
    from tldw_chatbook.STT.executor_worker import ProviderRuntime

    if request.identity.device is not ExecutionDevice.CPU:
        error = RuntimeError("typed execution provider initialization failure")
        error.device_failure_origin = (  # type: ignore[attr-defined]
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION
        )
        error.failed_device = request.identity.device  # type: ignore[attr-defined]
        raise error
    runtime = _FakeResidentRuntime(1)
    return ProviderRuntime(runner=runtime.transcribe, close=runtime.close)


def device_retry_executor_worker(
    connection: Connection,
    admission_event: Any,
    cancellation_event: Any,
    generation: int,
    scratch_path: str,
) -> None:
    """Exercise CPU retry through the production resident worker loop."""

    from tldw_chatbook.STT.executor_worker import _run_executor_worker

    _run_executor_worker(
        connection,
        admission_event,
        cancellation_event,
        generation,
        scratch_path,
        provider_builder=_device_retry_provider_builder,
        parse_job=_fake_parse_job,
    )


def _private_log_provider_builder(
    _request: Any,
    _model_root: Path | None,
    _managed_handle: Any | None,
    _is_cancelled: Any,
) -> Any:
    from loguru import logger

    logger.error("private worker path: /private/models/secret.onnx")
    raise RuntimeError("private worker path: /private/models/secret.onnx")


def private_log_executor_worker(
    connection: Connection,
    admission_event: Any,
    cancellation_event: Any,
    generation: int,
    scratch_path: str,
) -> None:
    """Attempt a path-bearing legacy log inside the production worker loop."""

    from tldw_chatbook.STT.executor_worker import _run_executor_worker

    _run_executor_worker(
        connection,
        admission_event,
        cancellation_event,
        generation,
        scratch_path,
        provider_builder=_private_log_provider_builder,
        parse_job=_fake_parse_job,
    )


def containment_probe(connection: Connection, admission_event: Any) -> None:
    """Report containment identity, then prove admission gates worker progress."""

    from tldw_chatbook.STT.executor_process_tree import enter_worker_containment

    identity = enter_worker_containment()
    connection.send(("identity", identity))
    admitted = admission_event.wait(10.0)
    connection.send(("admitted", admitted))


def containment_descendant(
    connection: Connection,
    admission_event: Any,
    scratch_path: str,
) -> None:
    """Launch one descendant only after containment admission."""

    from tldw_chatbook.STT.executor_process_tree import enter_worker_containment

    identity = enter_worker_containment()
    connection.send(("identity", identity))
    if not admission_event.wait(10.0):
        return
    marker = Path(scratch_path) / "worker-admitted"
    marker.write_text("ready", encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    connection.send(("child", child.pid))
    while True:
        time.sleep(1.0)


def containment_crashed_leader_with_term_ignoring_descendant(
    connection: Connection,
    admission_event: Any,
    scratch_path: str,
) -> None:
    """Crash after starting a descendant that ignores graceful termination."""

    from tldw_chatbook.STT.executor_process_tree import enter_worker_containment

    identity = enter_worker_containment()
    connection.send(("identity", identity))
    if not admission_event.wait(10.0):
        return
    ready = Path(scratch_path) / "descendant-ready"
    ignore_sigterm = (
        "import signal;signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        if os.name == "posix"
        else ""
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib,time;"
                f"{ignore_sigterm}"
                f"pathlib.Path({str(ready)!r}).write_text('ready');"
                "time.sleep(120)"
            ),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10.0
    while not ready.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    connection.send(("child", child.pid))
    connection.close()
    os._exit(73)


def fake_executor_worker(
    connection: Connection,
    admission_event: Any,
    cancellation_event: Any,
    generation: int,
    scratch_path: str,
) -> None:
    """Deterministic resident worker used by controller spawn tests."""

    from tldw_chatbook.STT.contracts import (
        DeviceFailureOrigin,
        ExecutionDevice,
        TranscriptionFailureCode,
    )
    from tldw_chatbook.STT.executor import (
        ExecutorEvent,
        ExecutorFailure,
        ExecutorRequest,
        ExecutorResident,
        ExecutorResult,
        WorkerPhase,
    )
    from tldw_chatbook.STT.executor_process_tree import enter_worker_containment

    identity = enter_worker_containment()
    connection.send(("bootstrap", identity))
    if not admission_event.wait(10.0):
        return
    connection.send(("ready", generation))
    resident = None
    while True:
        command = connection.recv()
        if command == ("close", generation):
            return
        if type(command) is not ExecutorRequest:
            continue
        request = command
        connection.send(
            ExecutorEvent(generation, request.attempt_id, WorkerPhase.PREPARING)
        )
        if resident != request.identity:
            connection.send(
                ExecutorEvent(generation, request.attempt_id, WorkerPhase.LOADING)
            )
            resident = request.identity
            connection.send(ExecutorResident(generation, request.attempt_id, resident))
        mode = request.options.get("test_mode", "succeed")
        if mode == "crash_loading":
            os._exit(70)
        connection.send(
            ExecutorEvent(generation, request.attempt_id, WorkerPhase.TRANSCRIBING)
        )
        if mode == "ignore_cancel":
            while True:
                time.sleep(1.0)
        if mode == "hold":
            while not cancellation_event.is_set():
                time.sleep(0.01)
            connection.send(
                ExecutorFailure(
                    generation,
                    request.attempt_id,
                    TranscriptionFailureCode.CANCELLED,
                )
            )
            continue
        if (
            mode == "device_failure"
            and request.identity.device is not ExecutionDevice.CPU
        ):
            connection.send(
                ExecutorFailure(
                    generation,
                    request.attempt_id,
                    TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
                    device_failure_origin=DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
                    failed_device=request.identity.device,
                )
            )
            continue
        payload = {
            "content": "transcript",
            "worker_pid": os.getpid(),
            "device": request.identity.device.value,
        }
        fallback_device = request.options.get(
            "_local_stt_cpu_fallback_requested_device"
        )
        if fallback_device is not None:
            payload["cpu_fallback_requested_device"] = fallback_device
        result = ExecutorResult(generation, request.attempt_id, payload)
        if mode == "stale_then_succeed":
            connection.send(
                ExecutorResult(generation + 1, request.attempt_id, {"content": "stale"})
            )
        connection.send(result)
        if mode == "duplicate":
            connection.send(result)
