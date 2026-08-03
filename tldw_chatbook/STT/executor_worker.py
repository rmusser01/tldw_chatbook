"""Spawn entry point for the app-owned local STT worker."""

from __future__ import annotations

import tempfile
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from .contracts import TranscriptionFailureCode
from .executor import ExecutorEvent, ExecutorFailure, ExecutorRequest, WorkerPhase
from .executor_process_tree import enter_worker_containment


def run_executor_worker(
    connection: Connection,
    admission_event: Any,
    cancellation_event: Any,
    generation: int,
    scratch_path: str,
) -> None:
    """Run the admitted worker loop until provider wiring is installed."""

    del cancellation_event
    identity = enter_worker_containment()
    connection.send(("bootstrap", identity))
    if not admission_event.wait(30.0):
        return
    scratch = Path(scratch_path)
    if not scratch.is_dir():
        return
    tempfile.tempdir = str(scratch)
    connection.send(("ready", generation))
    try:
        while True:
            command = connection.recv()
            if command == ("close", generation):
                return
            if type(command) is not ExecutorRequest or command.generation != generation:
                continue
            connection.send(
                ExecutorEvent(generation, command.attempt_id, WorkerPhase.PREPARING)
            )
            connection.send(
                ExecutorFailure(
                    generation,
                    command.attempt_id,
                    TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
                )
            )
    except (EOFError, OSError):
        return
    finally:
        connection.close()
