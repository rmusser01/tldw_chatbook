"""Diagnostic-only durations around original initialization; no guard replacement."""

from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time


def install(output: Path, stage):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    original = storage._Acquisition.initializing
    rows = []

    @contextmanager
    def observed(self, root, path):
        started = time.perf_counter()
        entered = None
        body_done = None
        row = {"stage": stage(), "thread": threading.current_thread().name}
        try:
            with original(self, root, path):
                entered = time.perf_counter()
                try:
                    yield
                finally:
                    body_done = time.perf_counter()
        finally:
            finished = time.perf_counter()
            row.update(
                wait_ms=((entered or finished) - started) * 1000,
                body_ms=(body_done - entered) * 1000 if body_done else None,
                exit_ms=(finished - body_done) * 1000 if body_done else None,
            )
            rows.append(row)

    storage._Acquisition.initializing = observed

    def finish():
        storage._Acquisition.initializing = original
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    return finish
