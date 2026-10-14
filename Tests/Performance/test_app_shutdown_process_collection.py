"""Finite original shutdown cleanup controls, without importing the App."""

import ast
import copy
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest


_LIFECYCLE_PATH = Path(__file__).resolve().parents[2] / "tldw_chatbook/app_lifecycle.py"


def _cleanup_function(subprocess_namespace):
    original = _LIFECYCLE_PATH.read_bytes()
    tree = ast.parse(original)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LifecycleMixin"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "on_unmount"
    )
    candidates = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.For)
            and "subprocess._active" in ast.unparse(child.iter)
            for child in node.body
        )
    ]
    assert len(candidates) == 1
    original_try = candidates[0]
    loops = [
        node
        for node in original_try.body
        if isinstance(node, ast.For)
        and (
            "subprocess._active" in ast.unparse(node.iter)
            or ast.unparse(node.iter) == "threading.enumerate()"
        )
    ]
    assert len(loops) == 2
    finite = ast.FunctionDef(
        name="finite_original_cleanup",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Try(
                body=copy.deepcopy(loops),
                handlers=copy.deepcopy(original_try.handlers),
                orelse=[],
                finalbody=[],
            )
        ],
        decorator_list=[],
    )
    namespace = {"subprocess": subprocess_namespace, "threading": threading}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[finite], type_ignores=[])),
            str(_LIFECYCLE_PATH),
            "exec",
        ),
        namespace,
    )
    assert _LIFECYCLE_PATH.read_bytes() == original
    return namespace["finite_original_cleanup"]


class _CooperativeThread(threading.Thread):
    def __init__(self):
        super().__init__(name="shutdown-pure-control")
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stop_calls = 0

    def run(self):
        self.entered.set()
        self.release.wait()

    def stop(self):
        self.stop_calls += 1
        self.release.set()


def _exercise(subprocess_namespace):
    source = _LIFECYCLE_PATH.read_bytes()
    errors, warnings, infos = [], [], []
    logger = SimpleNamespace(
        error=errors.append, warning=warnings.append, info=infos.append
    )
    worker = _CooperativeThread()
    worker.start()
    try:
        assert worker.entered.wait(1.0)
        _cleanup_function(subprocess_namespace)(SimpleNamespace(loguru_logger=logger))
        return {
            "stop_calls": worker.stop_calls,
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
        }
    finally:
        # This control owns this Thread only, including the original RED exit.
        worker.release.set()
        worker.join(1.0)
        assert not worker.is_alive()
        assert _LIFECYCLE_PATH.read_bytes() == source


@pytest.mark.unit
@pytest.mark.skipif(
    subprocess._active is not None,
    reason="CPython Windows has no active-process registry",
)
def test_actual_windows_registry_none_does_not_skip_cooperative_stop():
    result = _exercise(subprocess)
    assert result["stop_calls"] == 1, result
    assert not result["errors"], result


@pytest.mark.unit
@pytest.mark.parametrize("active", [None, []], ids=["windows_none", "posix_empty"])
def test_empty_platform_registry_does_not_skip_cooperative_stop(active):
    result = _exercise(
        SimpleNamespace(_active=active, TimeoutExpired=subprocess.TimeoutExpired)
    )
    assert result["stop_calls"] == 1, result
    assert not result["errors"], result


class _Process:
    """Method-shape unit control; it launches or targets no actual process."""

    pid = "pure-control"

    def __init__(self, active, *, timeout=False, error=False):
        self.active = active
        self.timeout = timeout
        self.error = error
        self.calls = []

    def poll(self):
        self.calls.append("poll")
        if self.error:
            raise RuntimeError("pure poll error")
        return None

    def terminate(self):
        self.calls.append("terminate")
        self.active.remove(self)

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self.timeout and timeout is not None:
            raise subprocess.TimeoutExpired("pure-control", timeout)

    def kill(self):
        self.calls.append("kill")


@pytest.mark.unit
def test_posix_process_list_snapshot_preserves_both_original_targets():
    active = []
    first, second = _Process(active), _Process(active)
    active.extend((first, second))
    result = _exercise(
        SimpleNamespace(_active=active, TimeoutExpired=subprocess.TimeoutExpired)
    )
    assert first.calls == second.calls == ["poll", "terminate", ("wait", 1.0)]
    assert not active
    assert result["stop_calls"] == 1 and not result["errors"], result


@pytest.mark.unit
def test_original_timeout_kill_and_wait_shape_is_preserved():
    active = []
    process = _Process(active, timeout=True)
    active.append(process)
    result = _exercise(
        SimpleNamespace(_active=active, TimeoutExpired=subprocess.TimeoutExpired)
    )
    assert process.calls == ["poll", "terminate", ("wait", 1.0), "kill", ("wait", None)]
    assert result["stop_calls"] == 1 and not result["errors"], result


@pytest.mark.unit
def test_original_process_error_still_reaches_thread_cleanup():
    active = []
    process = _Process(active, error=True)
    active.append(process)
    result = _exercise(
        SimpleNamespace(_active=active, TimeoutExpired=subprocess.TimeoutExpired)
    )
    assert process.calls == ["poll"]
    assert result["stop_calls"] == 1
    assert result["errors"] == ["Error terminating subprocess: pure poll error"]
