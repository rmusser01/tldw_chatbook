"""Compare one unchanged full-app test with bounded main-thread timing output.

This optional CI diagnostic uses cProfile, so its timings include profiling
overhead. It records code coordinates and aggregate times, never call values.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import sys
import threading
import time
from pathlib import Path


def snapshot(profile: cProfile.Profile, source: Path, label: str) -> dict:
    """Return the largest observed product costs without source or local values."""
    rows = []
    package = source / "tldw_chatbook"
    for entry in profile.getstats():
        code = entry.code
        if isinstance(code, str):
            continue
        try:
            relative = Path(code.co_filename).relative_to(package)
        except ValueError:
            continue
        rows.append(
            {
                "file": relative.as_posix(),
                "function": code.co_name,
                "line": code.co_firstlineno,
                "calls": entry.callcount,
                "total_seconds": round(entry.totaltime, 6),
                "self_seconds": round(entry.inlinetime, 6),
            }
        )
    module = sys.modules.get("tldw_chatbook")
    loaded = None if module is None else Path(module.__file__).resolve()
    return {
        "diagnostic": "backup_startup_profile",
        "label": label,
        "source_matches": None if loaded is None else loaded.is_relative_to(package),
        "largest_total": sorted(
            rows, key=lambda row: row["total_seconds"], reverse=True
        )[:20],
        "largest_self": sorted(rows, key=lambda row: row["self_seconds"], reverse=True)[
            :20
        ],
    }


def main() -> int:
    """Run one actual case with its existing 60-second ceiling in a fresh process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--label", choices=("dev", "candidate"), required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    os.chdir(source)
    sys.path[0] = str(source)
    os.environ["TLDW_TEST_PRIVATE_PROFILE_NODE"] = args.node
    import pytest

    profile = cProfile.Profile()
    stopping = threading.Event()
    started = time.monotonic()
    failures = []

    def emit():
        try:
            record = snapshot(profile, source, args.label)
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["observer_errors"] = list(failures)
            print(json.dumps(record), file=sys.stderr, flush=True)
        except Exception as error:  # noqa: BLE001 - optional output cannot replace the observed test outcome.
            failures.append(type(error).__name__)
            del failures[:-4]

    def sample():
        while not stopping.wait(10):
            emit()

    observer = threading.Thread(
        target=sample, name="backup-startup-profile", daemon=True
    )
    profile.enable()
    try:
        try:
            observer.start()
        except RuntimeError as error:
            failures.append(type(error).__name__)
        return pytest.main([args.node, "--timeout=60", "-q", "--capture=no"])
    finally:
        profile.disable()
        stopping.set()
        if observer.ident is not None:
            observer.join(timeout=5)
        emit()


if __name__ == "__main__":
    raise SystemExit(main())
