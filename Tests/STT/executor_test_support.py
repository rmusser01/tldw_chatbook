"""Spawn-importable helpers for local STT executor process tests."""

from __future__ import annotations

import subprocess
import sys
import time
import os
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


def _fake_provider_builder(_request: Any, _model_root: Path | None) -> Any:
    global _resident_runtime_loads
    from tldw_chatbook.STT.executor_worker import ProviderRuntime

    _resident_runtime_loads += 1
    runtime = _FakeResidentRuntime(_resident_runtime_loads)
    return ProviderRuntime(runner=runtime.transcribe, close=runtime.close)


def _fake_parse_job(
    file_path: str | Path,
    options: dict[str, Any],
    *,
    transcription_runner: Any,
) -> dict[str, Any]:
    if options.get("test_worker_crash"):
        os._exit(71)
    if options.get("test_worker_hold"):
        while True:
            time.sleep(1.0)
    transcription = transcription_runner(
        str(file_path),
        provider=options.get("transcription_provider"),
    )
    return {
        "content": transcription["text"],
        "runtime_load_number": transcription["runtime_load_number"],
    }


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
        result = ExecutorResult(
            generation,
            request.attempt_id,
            {
                "content": "transcript",
                "worker_pid": os.getpid(),
                "device": request.identity.device.value,
            },
        )
        if mode == "stale_then_succeed":
            connection.send(
                ExecutorResult(generation + 1, request.attempt_id, {"content": "stale"})
            )
        connection.send(result)
        if mode == "duplicate":
            connection.send(result)
