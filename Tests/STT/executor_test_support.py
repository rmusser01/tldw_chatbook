"""Spawn-importable helpers for local STT executor process tests."""

from __future__ import annotations

import subprocess
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


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
