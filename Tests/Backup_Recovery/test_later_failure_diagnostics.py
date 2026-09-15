"""A failed later child retains bounded metadata before the suite can time out."""

import ast
import asyncio
import json

# Synthetic subprocess results/errors only; the launch is monkeypatched.
import subprocess  # nosec B404

import pytest


@pytest.mark.parametrize("kind", ["timeout", "launch", "exit"])
def test_parent_records_immediate_later_child_failure(tmp_path, monkeypatch, kind):
    from Tests.Backup_Recovery import test_later_rollback_credential_ui as workflow

    monkeypatch.setattr(workflow, "_earn_replacement", lambda *a, **k: None)
    error = (
        subprocess.TimeoutExpired("private-command", 900, output=b"private-output")
        if kind == "timeout"
        else OSError(5, "private-launch-message")
    )

    def run(*args, **kwargs):
        assert kwargs["timeout"] == (900 if workflow.sys.platform == "win32" else 300)
        if kind == "exit":
            return subprocess.CompletedProcess("private-command", 17)
        raise error

    monkeypatch.setattr(workflow.subprocess, "run", run)
    with pytest.raises(AssertionError if kind == "exit" else type(error)) as caught:
        workflow.test_f9_later_rollback_requires_explicit_credential_review(
            tmp_path, tmp_path / "installed", default_profile=True
        )
    if kind != "exit":
        assert caught.value is error
    data = json.loads((tmp_path / "later-parent-failure.json.log").read_text())
    if kind == "exit":
        assert data == {"event": "child_exit", "returncode": 17}
    else:
        assert data["error"]["error_class"] == type(error).__name__
    assert "private-" not in json.dumps(data)


@pytest.mark.parametrize("write_fails", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_actual_child_entry_guard_retains_original_exception(
    tmp_path, monkeypatch, write_fails, cleanup_fails
):
    from Tests.Backup_Recovery import admission_diagnostics, thread_diagnostics
    from Tests.Backup_Recovery.test_later_rollback_credential_ui import _LATER

    error = RuntimeError("private-child-message")

    async def main():
        raise error

    stopped = []

    def observe(kind):
        def start(*args, **kwargs):
            def stop():
                stopped.append(kind)
                if cleanup_fails:
                    raise OSError("private-cleanup-message")
            return stop
        return start

    monkeypatch.setattr(thread_diagnostics, "observe_threads", observe("threads"))
    monkeypatch.setattr(admission_diagnostics, "observe_admission", observe("admission"))
    if write_fails:

        def broken(*args):
            raise OSError("private-write-message")

        monkeypatch.setattr(thread_diagnostics, "_write", broken)
    blocks = [node for node in ast.parse(_LATER).body if isinstance(node, ast.Try)]
    assert len(blocks) == 1, "the actual child entry needs an immediate exception guard"
    namespace = {"asyncio": asyncio, "main": main, "home": tmp_path,
                 "retain": lambda *args, **kwargs: None}
    with pytest.raises(RuntimeError) as caught:
        # Fixed test-driver AST; no external code.
        exec(  # noqa: S102  # nosec B102
            compile(
                ast.Module(body=blocks, type_ignores=[]), "later-child-entry", "exec"
            ),
            namespace,
        )
    assert caught.value is error
    assert stopped == ["admission", "threads"]

    path = tmp_path / "later-child-failure.json.log"
    if not write_fails:
        data = json.loads(path.read_text())
        assert data["error"]["error_class"] == "RuntimeError"
        assert "private-" not in json.dumps(data)


def test_later_phase_timing_is_bounded_and_excludes_checkpoint_values(tmp_path):
    import time

    from Tests.Backup_Recovery.test_later_rollback_credential_ui import _LATER
    from Tests.Backup_Recovery.thread_diagnostics import _write

    retain = next(node for node in ast.parse(_LATER).body
                  if isinstance(node, ast.FunctionDef) and node.name == "retain")
    namespace = {"home": tmp_path, "time": time, "json": json,
                 "phase_times": [], "_write": _write}
    exec(  # noqa: S102  # nosec B102 - exact test-driver function, no external code.
        compile(ast.Module(body=[retain], type_ignores=[]), "later-phase", "exec"),
        namespace,
    )
    for index in range(40):
        namespace["retain"]("verified_copy_listed", value="private-value", index=index)
    text = (tmp_path / "later-phase-timing.log").read_text()
    rows = json.loads(text)
    assert len(rows) == 32 and "private-value" not in text
    assert all(set(row) == {"checkpoint", "monotonic_seconds"} for row in rows)
    assert all(row["checkpoint"] == "verified_copy_listed" for row in rows)
    assert rows[0]["monotonic_seconds"] <= rows[-1]["monotonic_seconds"]


@pytest.mark.parametrize("filename", [
    "later-child-failure.json.log", "mounted-child-failure.json.log",
    "mounted-body-failure.json.log", "mounted-wait-failure.json.log",
])
def test_failure_record_is_collected_before_pytest_finalization(tmp_path, filename):
    from Tests.Backup_Recovery.later_failure_diagnostics import record_failure
    from Tests.Backup_Recovery.run_platform_product import _collect_safe_logs

    private = tmp_path / "private"
    child = private / "product-pytest" / "later" / "home"
    child.mkdir(parents=True)
    record_failure(
        child / filename,
        error=RuntimeError("private-child-message"),
    )
    artifacts = tmp_path / "artifacts"
    assert _collect_safe_logs(private, artifacts) == 1
    recorded = next(artifacts.rglob(filename)).read_text()
    assert json.loads(recorded)["error"]["error_class"] == "RuntimeError"
    assert "private-child-message" not in recorded


@pytest.mark.asyncio
@pytest.mark.parametrize("primary,cleanup_fails,write_fails", [
    ("error", False, False), ("cancel", False, False),
    ("success", False, False), ("error", True, False),
    ("error", False, True),
])
async def test_mounted_body_failure_is_recorded_before_blocking_service_close(
    tmp_path, monkeypatch, primary, cleanup_fails, write_fails
):
    import time
    from pathlib import Path
    from threading import Event

    from Tests.Backup_Recovery import thread_diagnostics
    from Tests.Backup_Recovery.later_failure_diagnostics import record_failure
    from Tests.Backup_Recovery.test_mounted_console_backup import _SCRIPT

    original = RuntimeError("private-body") if primary == "error" else asyncio.CancelledError()
    cleanup = OSError("private-close")
    entered, release = Event(), Event()

    async def synthetic_body():
        if primary != "success":
            raise original

    class Service:
        def close(self):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release cleanup")
            if cleanup_fails:
                raise cleanup

    def broken_write(*args):
        raise OSError("private-write")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    if write_fails:
        monkeypatch.setattr(thread_diagnostics, "_write", broken_write)
    tree = ast.parse(_SCRIPT)
    main = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                and node.name == "main")
    boundary = next(node for node in main.body if isinstance(node, ast.Try))
    # Replace only the expensive app body; execute its actual handlers/cleanup.
    boundary.body = ast.parse("await synthetic_body()").body
    main.body = [boundary]
    phase = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "phase")
    namespace = {
        "asyncio": asyncio, "Path": Path, "time": time, "phases": [],
        "phase_start": time.monotonic(), "_write": thread_diagnostics._write,
        "synthetic_body": synthetic_body, "service": Service(),
        "record_failure": record_failure,
    }
    exec(  # noqa: S102  # nosec B102 - fixed test-driver AST, no external code.
        compile(ast.fix_missing_locations(ast.Module(
            body=[phase, main], type_ignores=[])), "mounted-body-boundary", "exec"),
        namespace,
    )
    task = asyncio.create_task(namespace["main"]())
    raised = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        path = tmp_path / "mounted-body-failure.json.log"
        if primary != "success" and not write_fails:
            assert path.is_file(), "body failure must survive an unfinished service close"
            encoded = path.read_text()
            assert json.loads(encoded)["error"]["error_class"] == type(original).__name__
            assert "private-" not in encoded
        else:
            assert not path.exists()
        assert [row["phase"] for row in namespace["phases"]] == ["service_close_begin"]
    finally:
        release.set()
        try:
            await task
        except BaseException as error:  # noqa: BLE001 - assert exact failure/cancellation below.
            raised = error
    assert raised is (cleanup if cleanup_fails else original if primary != "success" else None)
    assert [row["phase"] for row in namespace["phases"]] == (
        ["service_close_begin"] if cleanup_fails
        else ["service_close_begin", "service_close_complete"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel,status_fails", [(False, False), (True, False), (False, True)])
async def test_mounted_wait_failure_is_retained_before_observer_and_app_cleanup(
    tmp_path, monkeypatch, cancel, status_fails
):
    from pathlib import Path

    from Tests.Backup_Recovery.later_failure_diagnostics import record_failure
    from Tests.Backup_Recovery.test_mounted_console_backup import _SCRIPT
    from Tests.Backup_Recovery.thread_diagnostics import stop_observer

    original = asyncio.CancelledError() if cancel else TimeoutError("private-wait")
    cleanup = RuntimeError("private-status")
    observed = []
    recorded_errors = []
    path = tmp_path / "mounted-wait-failure.json.log"

    def retain(path, *, error):
        # asyncio's Future boundary can reconstruct a worker TimeoutError.
        recorded_errors.append(error)
        record_failure(path, error=error)

    class Service:
        def wait(self, operation, *, timeout):
            raise original

        def status(self, operation):
            if status_fails:
                raise cleanup
            return {"state": "running", "phase": "capturing", "issues": ()}

    def stop_loop():
        observed.append(json.loads(path.read_text()) if path.exists() else None)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    tree = ast.parse(_SCRIPT)
    # Execute the actual service.wait try, including its observer/status finally.
    boundary = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
                    and any(isinstance(child, ast.Call)
                            and isinstance(child.func, ast.Attribute)
                            and child.func.attr == "to_thread"
                            and child.args and isinstance(child.args[0], ast.Attribute)
                            and child.args[0].attr == "wait"
                            for child in ast.walk(node.body[0])))
    entry = ast.parse("async def entry(): pass").body[0]
    entry.body = [boundary]
    namespace = {"asyncio": asyncio, "Path": Path, "record_failure": retain,
                 "service": Service(), "operation": "test-operation",
                 "stop_observer": stop_observer, "stop_loop": stop_loop}
    exec(  # noqa: S102  # nosec B102 - fixed test-driver AST, no external code.
        compile(ast.fix_missing_locations(ast.Module(
            body=[entry], type_ignores=[])), "mounted-wait-boundary", "exec"),
        namespace,
    )
    with pytest.raises(type(cleanup if status_fails else original)) as caught:
        await namespace["entry"]()
    assert observed and observed[0] is not None, "record wait failure before cleanup begins"
    assert observed[0]["error"]["error_class"] == type(original).__name__
    assert "private-" not in json.dumps(observed)
    assert caught.value is (cleanup if status_fails else recorded_errors[0])


@pytest.mark.parametrize("metadata_error", [RuntimeError, asyncio.CancelledError])
def test_diagnostic_metadata_failure_does_not_replace_parent_error(
    tmp_path, monkeypatch, metadata_error
):
    from Tests.Backup_Recovery import test_later_rollback_credential_ui as workflow
    from Tests.Backup_Recovery import thread_diagnostics

    monkeypatch.setattr(workflow, "_earn_replacement", lambda *a, **k: None)
    error = subprocess.TimeoutExpired("private-command", 900)

    def fail_run(*args, **kwargs):
        raise error

    def fail_metadata(*args):
        raise metadata_error("private-metadata-message")

    monkeypatch.setattr(workflow.subprocess, "run", fail_run)
    monkeypatch.setattr(thread_diagnostics, "_error_metadata", fail_metadata)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        workflow.test_f9_later_rollback_requires_explicit_credential_review(
            tmp_path, tmp_path / "installed"
        )
    assert caught.value is error


@pytest.mark.asyncio
async def test_actual_textual_worker_preserves_original_callback_metadata(tmp_path):
    from textual.app import App
    from textual.worker import WorkerFailed

    from Tests.Backup_Recovery.later_failure_diagnostics import record_failure

    original = RuntimeError("private-worker-message")

    async def failing_callback():
        raise original

    class FailingApp(App):
        def on_mount(self):
            self.run_worker(failing_callback())

    with pytest.raises(WorkerFailed) as caught:
        async with FailingApp().run_test() as pilot:
            await pilot.pause()
    assert caught.value.error is original
    path = tmp_path / "worker.log"
    record_failure(path, error=caught.value)
    encoded = path.read_text()
    record = json.loads(encoded)
    assert record["worker_error"]["frames"][-1]["function"] == "failing_callback"
    assert record["worker_error"]["error_class"] == "RuntimeError"
    assert "private-worker-message" not in encoded


@pytest.mark.parametrize("value", ["missing", "none", "text", "self", "nested", "subclass"])
def test_worker_metadata_is_exact_and_one_level(tmp_path, value):
    from textual.worker import WorkerFailed

    from Tests.Backup_Recovery.later_failure_diagnostics import record_failure

    class HostileWorker(WorkerFailed):
        @property
        def error(self):
            raise AssertionError("private-property-was-called")

    error = WorkerFailed(RuntimeError("private-inner"))
    if value == "missing":
        del error.error
    elif value == "none":
        error.error = None
    elif value == "text":
        error.error = "private-text"
    elif value == "self":
        error.error = error
    elif value == "nested":
        error.error = WorkerFailed(RuntimeError("private-deep"))
    else:
        error = Exception.__new__(HostileWorker)
    path = tmp_path / "worker.log"
    record_failure(path, error=error)
    encoded = path.read_text()
    record = json.loads(encoded)
    assert "private-" not in encoded
    if value == "nested":
        assert record["worker_error"]["error_class"] == "WorkerFailed"
        assert "worker_error" not in record["worker_error"]
    elif value == "subclass":
        assert "worker_error" not in record
    else:
        assert record["worker_error"] is None


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_worker_metadata_failure_keeps_root_record(tmp_path, monkeypatch, failure):
    from textual.worker import WorkerFailed

    from Tests.Backup_Recovery import thread_diagnostics
    from Tests.Backup_Recovery.later_failure_diagnostics import record_failure

    original = RuntimeError("private-inner")
    wrapper = WorkerFailed(original)
    metadata = thread_diagnostics._error_metadata

    def observed(error):
        if error is original:
            raise failure("private-metadata-error")
        return metadata(error)

    monkeypatch.setattr(thread_diagnostics, "_error_metadata", observed)
    path = tmp_path / "worker.log"
    record_failure(path, error=wrapper)
    record = json.loads(path.read_text())
    assert record["error"]["error_class"] == "WorkerFailed"
    assert record["worker_error"] is None


@pytest.mark.parametrize("primary", ["error", "cancel", "success"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("write_fails", [False, True])
def test_actual_mounted_entry_retains_error_and_cleanup_contract(
    tmp_path, monkeypatch, primary, cleanup_fails, write_fails
):
    from pathlib import Path

    from Tests.Backup_Recovery.later_failure_diagnostics import record_failure
    from Tests.Backup_Recovery.test_mounted_console_backup import _SCRIPT
    from Tests.Backup_Recovery.thread_diagnostics import stop_observer

    original = RuntimeError("private-mounted") if primary == "error" else asyncio.CancelledError()
    cleanup = OSError("private-cleanup")
    trace = None
    closed = []

    async def main():
        nonlocal trace
        if primary != "success":
            try:
                raise original
            except BaseException as error:
                trace = error.__traceback__
                raise

    class Diagnostics:
        def close(self):
            closed.append(True)
            if cleanup_fails:
                raise cleanup

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    if write_fails:
        from Tests.Backup_Recovery import thread_diagnostics

        def fail_write(*args):
            raise OSError("private-output-error")

        monkeypatch.setattr(thread_diagnostics, "_write", fail_write)
    blocks = [node for node in ast.parse(_SCRIPT).body if isinstance(node, ast.Try)]
    assert len(blocks) == 1
    code = compile(ast.Module(body=blocks, type_ignores=[]), "mounted-child-entry", "exec")
    namespace = {"asyncio": asyncio, "main": main, "Path": Path,
                 "diagnostics": Diagnostics(), "stop_observer": stop_observer,
                 "record_failure": record_failure}
    if primary != "success" or cleanup_fails:
        expected = original if primary != "success" else cleanup
        with pytest.raises(type(expected)) as caught:
            exec(code, namespace)  # noqa: S102  # nosec B102 - execute the actual fixed test entry guard.
        assert caught.value is expected
        if primary != "success":
            tb = caught.value.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next
            assert tb is trace
    else:
        exec(code, namespace)  # noqa: S102  # nosec B102 - same fixed entry on success.
    assert closed == [True]
    path = tmp_path / "mounted-child-failure.json.log"
    if primary == "success" or write_fails:
        assert not path.exists()
    else:
        assert json.loads(path.read_text())["error"]["error_class"] == type(original).__name__
        assert "private-" not in path.read_text()
