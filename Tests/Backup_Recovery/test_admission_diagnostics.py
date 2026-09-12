"""Timing observations retain real outcomes and only aggregate numeric data."""

import json
import threading
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.admission_diagnostics import _observe


def test_admission_timing_preserves_results_errors_and_method_identity(tmp_path):
    secret = "synthetic-private-value"
    failure = ValueError(secret)

    class Subject:
        def _groups(self, registry, names):
            return names

        def security(self, value):
            if value == secret:
                raise failure
            return value

    originals = {name: Subject.__dict__[name] for name in ("_groups", "security")}
    path = tmp_path / "timing.log"
    stop = _observe(path, [(Subject, name) for name in originals], interval=0.01)
    try:
        subject = Subject()
        registry = SimpleNamespace(
            entries={
                "hidden": SimpleNamespace(
                    roots=("/private/one", "/private/one"), proposed=()
                )
            }
        )
        for _ in range(20):
            assert subject._groups(registry, (secret,)) == (secret,)
        assert subject.security(42) == 42
        with pytest.raises(ValueError) as caught:
            subject.security(secret)
        assert caught.value is failure
    finally:
        stop()
    assert all(
        Subject.__dict__[name] is original for name, original in originals.items()
    )
    assert not any(
        thread.name == "test-admission-timing" for thread in threading.enumerate()
    )
    text = path.read_text()
    assert secret not in text and "/private" not in text and "hidden" not in text
    snapshot = json.loads(text)
    assert len(snapshot["groups"]) == 16
    assert snapshot["groups"][0]["total_roots"] == 2
    assert snapshot["groups"][0]["unique_roots"] == 1
    assert snapshot["calls"]["security"]["completed"] == 2
    assert snapshot["calls"]["security"]["errors"] == 1
    assert snapshot["calls"]["_groups"]["wall_ns"] > 0


def test_timing_is_written_while_a_real_call_is_still_running(tmp_path):
    import time

    entered, release = threading.Event(), threading.Event()

    class Subject:
        def security(self):
            entered.set()
            assert release.wait(5)

    path = tmp_path / "timing.log"
    stop = _observe(path, [(Subject, "security")], interval=0.01)
    caller = threading.Thread(target=Subject().security)
    caller.start()
    try:
        assert entered.wait(5)
        deadline = time.monotonic() + 5
        observed = False
        while time.monotonic() < deadline:
            try:
                row = json.loads(path.read_text())["calls"]["security"]
            except json.JSONDecodeError:
                continue
            if row["started"] == 1 and row["completed"] == 0:
                observed = True
                break
            time.sleep(0.01)
        assert observed
    finally:
        release.set()
        caller.join(timeout=5)
        stop()
    assert json.loads(path.read_text())["calls"]["security"]["completed"] == 1
