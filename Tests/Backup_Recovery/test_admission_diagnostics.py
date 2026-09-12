"""Timing observations retain real outcomes and only aggregate numeric data."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.admission_diagnostics import _observe


def test_admission_timing_preserves_results_errors_and_method_identity(tmp_path):
    private_value = "synthetic-private-value"
    failure = ValueError(private_value)

    class Subject:
        def _groups(self, registry, names):
            return names

        def security(self, value):
            if value == private_value:
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
            assert subject._groups(registry, (private_value,)) == (private_value,)
        assert subject.security(42) == 42
        with pytest.raises(ValueError) as caught:
            subject.security(private_value)
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
    assert private_value not in text and "/private" not in text and "hidden" not in text
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


@pytest.mark.parametrize("failed", (False, True))
def test_group_timing_separates_threads_and_preserves_failure(tmp_path, failed):
    entered, release = threading.Event(), threading.Event()
    private_value = "synthetic-path-or-sid-must-not-appear"
    failure = OSError(private_value)
    results, errors = [], []
    registry = SimpleNamespace(
        entries={
            private_value: SimpleNamespace(
                roots=("/private/" + private_value,), proposed=()
            )
        }
    )

    class Subject:
        def _groups(self, registry, names):
            self.security(True)
            if failed:
                raise failure
            return names

        def security(self, block):
            if block:
                entered.set()
                assert release.wait(5)
            return private_value

    def final_scan():
        try:
            results.append(Subject()._groups(registry, (private_value,)))
        except BaseException as error:  # noqa: BLE001 - assert original identity below.
            errors.append(error)
        finally:
            Subject().security(False)

    originals = {name: Subject.__dict__[name] for name in ("_groups", "security")}
    path = tmp_path / "timing.log"
    stop = _observe(path, [(Subject, name) for name in originals], interval=0.01)
    caller = threading.Thread(target=final_scan)
    caller.start()
    try:
        assert entered.wait(5)
        for _ in range(3):
            assert Subject().security(False) == private_value
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                pending = json.loads(path.read_text())["groups"]
            except json.JSONDecodeError:
                pending = []
            if pending and pending[0]["calls"].get("security", {}).get("started"):
                break
            time.sleep(0.01)
        assert pending[0]["completed"] == 0
        assert pending[0]["wall_ns"] > 0
        assert pending[0]["calls"]["security"]["completed"] == 0
    finally:
        release.set()
        caller.join(timeout=5)
        stop()
    assert not caller.is_alive()
    assert errors == ([failure] if failed else [])
    assert results == ([] if failed else [(private_value,)])
    text = path.read_text()
    assert private_value not in text and "/private" not in text
    snapshot = json.loads(text)
    group = snapshot["groups"][0]
    assert group["caller"] == {
        "file": "test_admission_diagnostics.py",
        "function": "final_scan",
        "line": final_scan.__code__.co_firstlineno + 2,
    }
    assert group["completed"] == 1 and group["errors"] == int(failed)
    assert group["wall_ns"] > 0
    assert group["thread_cpu_ns"] >= 0 and group["process_cpu_ns"] >= 0
    assert group["calls"]["security"]["started"] == 1
    assert group["calls"]["security"]["completed"] == 1
    assert group["calls"]["security"]["wall_ns"] > 0
    assert snapshot["calls"]["security"]["completed"] == 5
    assert all(
        Subject.__dict__[name] is original for name, original in originals.items()
    )
