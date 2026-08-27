"""Process, streaming, and cleanup evidence for the raw one-shot CLI executor."""

from __future__ import annotations

import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_chatbook.Tools.raw_cli_executor as raw_cli
from tldw_chatbook.STT.executor_process_tree import (
    ExecutorProcessTree,
    WorkerContainmentIdentity,
)


def _request(
    directory: Path,
    command: str,
    *,
    timeout_seconds: float = 10.0,
) -> Any:
    return raw_cli.RawCliRequest(
        invocation_id="inv-process-1",
        caller="user",
        command=command,
        shell="cmd" if os.name == "nt" else "bash",
        initial_directory=directory,
        timeout_seconds=timeout_seconds,
        console_session_id="console-1",
        transcript_anchor_id=None,
    )


def _executor() -> Any:
    executor_type = getattr(raw_cli, "RawShellExecutor", None)
    assert executor_type is not None, "RawShellExecutor is missing"
    return executor_type()


def _admit(tree: ExecutorProcessTree, commit_launch: Any) -> bool:
    tree.admit()
    commit_launch()
    return True


def _python_command(source: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([sys.executable, "-c", source])
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


def _pid_has_exited(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _wait_for_pid_exit(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not _pid_has_exited(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    return _pid_has_exited(pid)


@pytest.fixture
def spool_owners(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    owners: list[Any] = []
    real_open_spool = raw_cli._open_spool

    def recording_open_spool(max_record_bytes: int) -> Any:
        owner = real_open_spool(max_record_bytes)
        owners.append(owner)
        return owner

    monkeypatch.setattr(raw_cli, "_open_spool", recording_open_spool)
    return owners


def test_request_validation_happens_before_worker_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_start(_process: Any) -> None:
        raise AssertionError("invalid request reached process start")

    monkeypatch.setattr(
        multiprocessing.context.SpawnProcess,
        "start",
        unexpected_start,
    )
    request = _request(tmp_path, "   ")

    with pytest.raises(ValueError, match="empty|whitespace"):
        _executor().execute(
            request,
            cancel_event=threading.Event(),
            on_event=lambda _event: None,
            admit_worker=_admit,
        )


def test_spawned_worker_reports_identity_and_waits_for_admission_before_shell(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "shell-started"
    identities: list[WorkerContainmentIdentity] = []

    def admit(tree: ExecutorProcessTree, commit_launch: Any) -> bool:
        identity = tree._identity
        assert type(identity) is WorkerContainmentIdentity
        assert identity.pid == tree._process.pid
        if os.name == "posix":
            assert identity.process_group_id == identity.pid
        assert marker.exists() is False
        identities.append(identity)
        tree.admit()
        commit_launch()
        return True

    command = _python_command(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('started')"
    )
    result = _executor().execute(
        _request(tmp_path, command),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=admit,
    )

    assert len(identities) == 1
    assert marker.read_text(encoding="utf-8") == "started"
    assert result.terminal_state == "exited"
    assert result.exit_code == 0


@pytest.mark.parametrize("admission_mode", ["false", "raise"])
def test_failed_or_exceptional_admission_kills_waiting_worker_without_shell(
    tmp_path: Path,
    spool_owners: list[Any],
    admission_mode: str,
) -> None:
    marker = tmp_path / "must-not-launch"
    captured_processes: list[Any] = []

    def refuse(tree: ExecutorProcessTree, _commit_launch: Any) -> bool:
        captured_processes.append(tree._process)
        assert marker.exists() is False
        tree.admit()
        if admission_mode == "raise":
            raise RuntimeError("admission callback failed")
        return False

    command = _python_command(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('unsafe')"
    )
    result = _executor().execute(
        _request(tmp_path, command),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=refuse,
    )

    assert result.terminal_state == "containment_unavailable"
    assert result.exit_code is None
    assert marker.exists() is False
    assert captured_processes and captured_processes[0].is_alive() is False
    assert len(spool_owners) == 1
    assert spool_owners[0].closed is True


def test_runtime_recheck_refusal_keeps_launch_gate_closed(
    tmp_path: Path,
) -> None:
    runtime_lock = threading.Lock()
    authority = {"armed": True}
    callback_waiting = threading.Event()
    marker = tmp_path / "revocation-won"
    results: list[Any] = []

    def admit(tree: ExecutorProcessTree, commit_launch: Any) -> bool:
        callback_waiting.set()
        with runtime_lock:
            if not authority["armed"]:
                return False
            tree.admit()
            commit_launch()
            return True

    runtime_lock.acquire()
    thread = threading.Thread(
        target=lambda: results.append(
            _executor().execute(
                _request(
                    tmp_path,
                    _python_command(
                        f"from pathlib import Path; Path({str(marker)!r}).touch()"
                    ),
                ),
                cancel_event=threading.Event(),
                on_event=lambda _event: None,
                admit_worker=admit,
            )
        )
    )
    thread.start()
    try:
        assert callback_waiting.wait(5.0)
        authority["armed"] = False
    finally:
        runtime_lock.release()
    thread.join(10.0)

    assert thread.is_alive() is False
    assert results[0].terminal_state == "containment_unavailable"
    assert marker.exists() is False


def test_runtime_admission_commits_under_lock_then_disarm_cancels(
    tmp_path: Path,
) -> None:
    runtime_lock = threading.Lock()
    cancel_event = threading.Event()
    launched = threading.Event()
    actions: list[tuple[str, bool]] = []
    workers: list[Any] = []
    results: list[Any] = []

    def admit(tree: ExecutorProcessTree, commit_launch: Any) -> bool:
        with runtime_lock:
            workers.append(tree._process)
            tree.admit()
            actions.append(("admit", runtime_lock.locked()))
            commit_launch()
            commit_launch()
            actions.append(("commit", runtime_lock.locked()))
            return True

    command = _python_command(
        "import threading; print('launched', flush=True); threading.Event().wait(120)"
    )
    thread = threading.Thread(
        target=lambda: results.append(
            _executor().execute(
                _request(tmp_path, command),
                cancel_event=cancel_event,
                on_event=lambda event: launched.set() if event.text else None,
                admit_worker=admit,
            )
        )
    )
    thread.start()
    assert launched.wait(5.0)
    with runtime_lock:
        cancel_event.set()
    thread.join(10.0)

    assert thread.is_alive() is False
    assert actions == [("admit", True), ("commit", True)]
    assert results[0].terminal_state == "cancelled"
    assert workers and workers[0].is_alive() is False


def test_worker_needs_parent_commit_after_containment_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_sent = threading.Event()
    launch_called = threading.Event()
    containment_admission = threading.Event()
    containment_admission.set()
    launch_commit = threading.Event()
    abort = threading.Event()

    class Connection:
        def send(self, _identity: WorkerContainmentIdentity) -> None:
            identity_sent.set()

        def close(self) -> None:
            pass

    class OutputQueue:
        def put(self, _message: tuple[Any, ...]) -> None:
            pass

        def close(self) -> None:
            pass

        def join_thread(self) -> None:
            pass

    monkeypatch.setattr(
        raw_cli,
        "enter_worker_containment",
        lambda: WorkerContainmentIdentity(pid=os.getpid(), process_group_id=None),
    )
    monkeypatch.setattr(raw_cli, "_launch_shell", lambda *_args: launch_called.set())
    thread = threading.Thread(
        target=raw_cli._raw_cli_worker_entry,
        args=(
            _request(tmp_path, "must not launch"),
            Connection(),
            containment_admission,
            launch_commit,
            abort,
            OutputQueue(),
        ),
    )
    thread.start()

    assert identity_sent.wait(5.0)
    assert launch_called.is_set() is False
    abort.set()
    thread.join(5.0)
    assert thread.is_alive() is False


def test_outer_shell_launch_is_noninteractive_and_never_uses_shell_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    sentinel = object()

    def popen(argv: tuple[str, ...], **kwargs: Any) -> object:
        calls.append((argv, kwargs))
        return sentinel

    monkeypatch.setattr(raw_cli.subprocess, "Popen", popen)
    request = _request(tmp_path, "ignored")
    argv = ("/fixed/bash", "--noprofile", "--norc", "-c", "ignored")

    process = raw_cli._launch_shell(argv, request)

    assert process is sentinel
    assert calls == [
        (
            argv,
            {
                "shell": False,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "cwd": str(tmp_path),
                "env": raw_cli.build_scrubbed_environment(),
            },
        )
    ]


def test_streams_stay_distinct_invalid_utf8_is_replaced_and_nonzero_is_exited(
    tmp_path: Path,
) -> None:
    command = _python_command(
        "import os;"
        "os.write(1, b'out:\\xff\\n');"
        "os.write(2, b'err:\\xfe\\n');"
        "raise SystemExit(7)"
    )
    events: list[Any] = []

    result = _executor().execute(
        _request(tmp_path, command),
        cancel_event=threading.Event(),
        on_event=events.append,
        admit_worker=_admit,
    )

    assert result.terminal_state == "exited"
    assert result.exit_code == 7
    assert result.stdout_preview == "out:\ufffd\n"
    assert result.stderr_preview == "err:\ufffd\n"
    assert "[stdout] out:\ufffd" in result.record_output
    assert "[stderr] err:\ufffd" in result.record_output
    assert {event.stream for event in events} == {"stdout", "stderr"}


def test_timeout_and_cancellation_share_idempotent_tree_cleanup(
    tmp_path: Path,
    spool_owners: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = raw_cli.ExecutorProcessTree.terminate_tree

    def recording_terminate(tree: ExecutorProcessTree, **kwargs: Any) -> bool:
        calls.append(tree._identity.pid)
        return original(tree, **kwargs)

    monkeypatch.setattr(
        raw_cli.ExecutorProcessTree,
        "terminate_tree",
        recording_terminate,
    )
    hold = _python_command(
        "import sys,threading; print('ready', flush=True); threading.Event().wait(120)"
    )

    timed_out = _executor().execute(
        _request(tmp_path, hold, timeout_seconds=0.1),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=_admit,
    )

    cancel_event = threading.Event()
    ready = threading.Event()
    result_box: list[Any] = []

    def execute_cancelled() -> None:
        result_box.append(
            _executor().execute(
                _request(tmp_path, hold),
                cancel_event=cancel_event,
                on_event=lambda event: ready.set() if "ready" in event.text else None,
                admit_worker=_admit,
            )
        )

    thread = threading.Thread(target=execute_cancelled)
    thread.start()
    assert ready.wait(5.0)
    cancel_event.set()
    thread.join(10.0)

    assert thread.is_alive() is False
    assert timed_out.terminal_state == "timed_out"
    assert result_box[0].terminal_state == "cancelled"
    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert len(spool_owners) == 2
    assert all(owner.closed for owner in spool_owners)


def test_execution_timeout_clock_starts_only_after_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(
        raw_cli,
        "time",
        SimpleNamespace(monotonic=lambda: clock.value),
    )

    def admit_after_clock_jump(
        tree: ExecutorProcessTree,
        commit_launch: Any,
    ) -> bool:
        tree.admit()
        commit_launch()
        clock.value = 1000.0
        return True

    result = _executor().execute(
        _request(
            tmp_path,
            _python_command("print('after admission')"),
            timeout_seconds=0.01,
        ),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=admit_after_clock_jump,
    )

    assert result.terminal_state == "exited"
    assert result.exit_code == 0


@pytest.mark.parametrize("trigger", ["cancelled", "timed_out"])
def test_specific_trigger_survives_unproven_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
) -> None:
    original = raw_cli.ExecutorProcessTree.terminate_tree

    def cleanup_then_report_unproven(
        tree: ExecutorProcessTree,
        **kwargs: Any,
    ) -> bool:
        original(tree, **kwargs)
        return False

    monkeypatch.setattr(
        raw_cli.ExecutorProcessTree,
        "terminate_tree",
        cleanup_then_report_unproven,
    )
    command = _python_command(
        "import sys,threading; print('ready', flush=True); threading.Event().wait(120)"
    )
    cancel_event = threading.Event()
    ready = threading.Event()
    results: list[Any] = []

    if trigger == "timed_out":
        result = _executor().execute(
            _request(tmp_path, command, timeout_seconds=0.1),
            cancel_event=cancel_event,
            on_event=lambda _event: None,
            admit_worker=_admit,
        )
    else:
        thread = threading.Thread(
            target=lambda: results.append(
                _executor().execute(
                    _request(tmp_path, command),
                    cancel_event=cancel_event,
                    on_event=(
                        lambda event: ready.set() if "ready" in event.text else None
                    ),
                    admit_worker=_admit,
                )
            )
        )
        thread.start()
        assert ready.wait(5.0)
        cancel_event.set()
        thread.join(10.0)
        assert thread.is_alive() is False
        result = results[0]

    assert result.terminal_state == trigger
    assert result.cleanup_proven is False


def test_cleanup_exception_preserves_timeout_and_marks_death_unproven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = raw_cli.ExecutorProcessTree.terminate_tree

    def cleanup_then_raise(tree: ExecutorProcessTree, **kwargs: Any) -> bool:
        original(tree, **kwargs)
        raise OSError("synthetic death-proof failure")

    monkeypatch.setattr(
        raw_cli.ExecutorProcessTree,
        "terminate_tree",
        cleanup_then_raise,
    )
    command = _python_command("import threading; threading.Event().wait(120)")

    result = _executor().execute(
        _request(tmp_path, command, timeout_seconds=0.1),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=_admit,
    )

    assert result.terminal_state == "timed_out"
    assert result.cleanup_proven is False


def test_output_flood_is_drained_bounded_and_truncated_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(raw_cli, "configured_max_record_bytes", lambda: 4096)
    command = _python_command(
        "import os;"
        "chunk_out=b'x'*8192; chunk_err=b'y'*8192;"
        "[(os.write(1, chunk_out), os.write(2, chunk_err)) for _ in range(64)]"
    )

    result = _executor().execute(
        _request(tmp_path, command),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=_admit,
    )

    preview_bytes = len(result.stdout_preview.encode()) + len(
        result.stderr_preview.encode()
    )
    assert result.terminal_state == "exited"
    assert result.exit_code == 0
    assert preview_bytes <= raw_cli.MAX_RAW_PREVIEW_BYTES
    assert len(result.record_output.encode()) <= 4096
    assert result.truncated is True
    assert result.cleanup_proven is True


def test_worker_coalesces_small_writes_before_bounded_ipc() -> None:
    class BytePipe:
        def __init__(self) -> None:
            self._chunks = iter((b"a", b"b", b"c", b""))
            self.closed = False

        def read(self, _size: int) -> bytes:
            return next(self._chunks)

        read1 = read

        def close(self) -> None:
            self.closed = True

    class RecordingQueue:
        def __init__(self) -> None:
            self.messages: list[tuple[Any, ...]] = []

        def put(self, message: tuple[Any, ...]) -> None:
            self.messages.append(message)

    pipe = BytePipe()
    output_queue = RecordingQueue()

    raw_cli._drain_pipe("stdout", pipe, output_queue)

    assert output_queue.messages == [
        ("output", "stdout", b"abc"),
        ("stream_end", "stdout", 3),
    ]
    assert pipe.closed is True


def test_late_cancel_after_worker_exit_cannot_replace_exited_result(
    tmp_path: Path,
) -> None:
    cancel_event = threading.Event()
    worker: list[Any] = []

    def admit(tree: ExecutorProcessTree, commit_launch: Any) -> bool:
        worker.append(tree._process)
        tree.admit()
        commit_launch()
        return True

    def cancel_after_worker_exit(event: Any) -> None:
        if event.text:
            assert wait([worker[0].sentinel], timeout=5.0)
            cancel_event.set()

    result = _executor().execute(
        _request(tmp_path, _python_command("print('finished', flush=True)")),
        cancel_event=cancel_event,
        on_event=cancel_after_worker_exit,
        admit_worker=admit,
    )

    assert cancel_event.is_set()
    assert result.terminal_state == "exited"
    assert result.exit_code == 0


def test_streaming_sanitizer_handles_control_sequences_split_across_chunks() -> None:
    sanitizer = raw_cli._StreamSanitizer()

    text = sanitizer.feed("left\x1b]0;ti")
    text += sanitizer.feed("tle\x07right\x1b[3")
    text += sanitizer.feed("1mred\x1b[0m\r")
    text += sanitizer.feed("\nnext\u0084\x1bQ", final=True)

    assert text == "leftrightred\nnextQ"


def test_ansi_osc_controls_are_removed_and_rich_markup_remains_literal(
    tmp_path: Path,
) -> None:
    payload = (
        b"plain\x1b[31mred\x1b[0m"
        b"\x1b]0;title\x07"
        b"\x1b]8;;https://example.invalid\x1b\\link\x1b]8;;\x1b\\"
        b"\x00\x08\xc2\x84\xc2\x9b31m"
        b"\r\nnext\rrow\t[bold]literal[/bold]\n"
    )
    command = _python_command(f"import os; os.write(1, {payload!r})")

    result = _executor().execute(
        _request(tmp_path, command),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=_admit,
    )

    assert result.stdout_preview == ("plainredlink\nnext\nrow\t[bold]literal[/bold]\n")
    assert "\x1b" not in result.record_output
    assert "title" not in result.record_output


def test_disk_spool_owner_closes_before_success_and_preserves_record_cap(
    tmp_path: Path,
    spool_owners: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_limit = 128
    monkeypatch.setattr(raw_cli, "configured_max_record_bytes", lambda: record_limit)

    result = _executor().execute(
        _request(tmp_path, _python_command("print('x' * 4096)")),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=_admit,
    )

    assert result.terminal_state == "exited"
    assert len(result.record_output.encode("utf-8")) <= record_limit
    assert len(spool_owners) == 1
    assert spool_owners[0].closed is True


@pytest.mark.skipif(os.name != "posix", reason="native POSIX disk-spool evidence")
def test_posix_spool_is_anonymous_disk_backed_and_mode_0600() -> None:
    owner = raw_cli._open_spool(128)
    assert hasattr(owner, "file"), "disk spool owner is missing"
    spool = owner.file
    descriptor = spool.fileno()

    assert stat.S_ISREG(os.fstat(descriptor).st_mode)
    assert stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o600
    assert not isinstance(spool.name, (str, bytes, os.PathLike))

    owner.close()
    assert owner.closed is True


@pytest.mark.skipif(os.name != "nt", reason="native Windows disk-spool evidence")
def test_windows_spool_uses_temporary_delete_on_close_storage() -> None:
    owner = raw_cli._open_spool(128)
    assert hasattr(owner, "file"), "disk spool owner is missing"
    spool = owner.file
    name = spool.name
    assert spool.fileno() >= 0
    assert isinstance(name, (str, bytes, os.PathLike))
    assert Path(name).exists()

    owner.close()
    assert owner.closed is True
    assert Path(name).exists() is False


def test_spool_is_removed_after_spawn_failure(
    tmp_path: Path,
    spool_owners: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_start(_process: Any) -> None:
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", fail_start)

    result = _executor().execute(
        _request(tmp_path, _python_command("print('never')")),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=_admit,
    )

    assert result.terminal_state == "spawn_failed"
    assert len(spool_owners) == 1
    assert spool_owners[0].closed is True


def test_spool_setup_failure_exposes_no_private_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path = "/private/raw-cli-secret"

    def fail_spool_setup(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError(f"synthetic setup failure at {private_path}")

    monkeypatch.setattr(
        tempfile,
        "mkstemp" if os.name == "posix" else "TemporaryFile",
        fail_spool_setup,
    )

    with pytest.raises(OSError, match="raw CLI output spool unavailable") as error:
        _executor().execute(
            _request(tmp_path, _python_command("print('never')")),
            cancel_event=threading.Event(),
            on_event=lambda _event: None,
            admit_worker=_admit,
        )

    assert private_path not in str(error.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX unlink-before-write evidence")
def test_transient_posix_setup_unlink_failure_is_retried_and_pathless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_mkstemp = tempfile.mkstemp
    real_unlink = os.unlink
    attempts: list[Path] = []

    def local_mkstemp(*_args: Any, **_kwargs: Any) -> tuple[int, str]:
        return real_mkstemp(dir=tmp_path)

    def transient_unlink(path: Any) -> None:
        attempts.append(Path(path))
        if len(attempts) == 1:
            raise OSError(f"transient unlink failure at {path}")
        real_unlink(path)

    monkeypatch.setattr(tempfile, "mkstemp", local_mkstemp)
    monkeypatch.setattr(raw_cli.os, "unlink", transient_unlink)

    with pytest.raises(OSError, match="raw CLI output spool unavailable") as error:
        raw_cli._open_spool(128)

    assert len(attempts) >= 2
    assert attempts[0].exists() is False
    assert str(tmp_path) not in str(error.value)


def test_spool_and_process_are_cleaned_when_event_callback_raises(
    tmp_path: Path,
    spool_owners: list[Any],
) -> None:
    worker: list[Any] = []

    def admit(tree: ExecutorProcessTree, commit_launch: Any) -> bool:
        worker.append(tree._process)
        tree.admit()
        commit_launch()
        return True

    def fail_callback(_event: Any) -> None:
        raise RuntimeError("display callback failed")

    command = _python_command(
        "import threading; print('event', flush=True); threading.Event().wait(120)"
    )
    with pytest.raises(RuntimeError, match="display callback failed"):
        _executor().execute(
            _request(tmp_path, command),
            cancel_event=threading.Event(),
            on_event=fail_callback,
            admit_worker=admit,
        )

    assert worker and worker[0].is_alive() is False
    assert len(spool_owners) == 1
    assert spool_owners[0].closed is True


def test_disk_spool_close_retries_when_first_close_fails_before_closing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers: list[Any] = []
    underlying = tempfile.TemporaryFile(mode="w+b")
    close_attempts = 0

    class CloseFailingSpool:
        def __getattr__(self, name: str) -> Any:
            return getattr(underlying, name)

        def close(self) -> None:
            nonlocal close_attempts
            close_attempts += 1
            if close_attempts == 1:
                raise OSError("synthetic close failure at /private/raw-cli-secret")
            underlying.close()

    def admit(tree: ExecutorProcessTree, commit_launch: Any) -> bool:
        workers.append(tree._process)
        tree.admit()
        commit_launch()
        return True

    owner_type = getattr(raw_cli, "_DiskSpoolOwner", None)
    assert owner_type is not None, "disk spool owner is missing"
    owner = owner_type(CloseFailingSpool())
    monkeypatch.setattr(raw_cli, "_open_spool", lambda *_args: owner)
    result = _executor().execute(
        _request(tmp_path, _python_command("print('close failure')")),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=admit,
    )

    assert result.terminal_state == "exited"
    assert close_attempts == 2
    assert underlying.closed is True
    assert owner.closed is True
    assert workers and workers[0].is_alive() is False


def test_spool_is_gone_before_a_downstream_run_log_failure(
    tmp_path: Path,
    spool_owners: list[Any],
) -> None:
    result = _executor().execute(
        _request(tmp_path, _python_command("print('persist me')")),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=_admit,
    )
    assert len(spool_owners) == 1
    assert spool_owners[0].closed is True

    def fail_run_log(_result: Any) -> None:
        raise OSError("simulated run-log failure")

    with pytest.raises(OSError, match="run-log failure"):
        fail_run_log(result)


@pytest.mark.skipif(os.name != "posix", reason="native POSIX process-group evidence")
def test_posix_cancellation_removes_shell_child_and_grandchild_group(
    tmp_path: Path,
) -> None:
    source = (
        "import os,subprocess,sys,threading;"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import threading;threading.Event().wait(120)']);"
        "print(f'{os.getpid()} {child.pid}',flush=True);"
        "threading.Event().wait(120)"
    )
    cancel_event = threading.Event()
    pids_ready = threading.Event()
    captured_pids: list[int] = []
    result_box: list[Any] = []

    def on_event(event: Any) -> None:
        if event.stream == "stdout" and event.text.strip():
            captured_pids.extend(int(value) for value in event.text.split())
            pids_ready.set()

    def execute() -> None:
        result_box.append(
            _executor().execute(
                _request(tmp_path, _python_command(source)),
                cancel_event=cancel_event,
                on_event=on_event,
                admit_worker=_admit,
            )
        )

    thread = threading.Thread(target=execute)
    thread.start()
    assert pids_ready.wait(5.0)
    cancel_event.set()
    thread.join(10.0)

    assert thread.is_alive() is False
    assert result_box[0].terminal_state == "cancelled"
    assert result_box[0].cleanup_proven is True
    assert len(captured_pids) == 2
    assert all(_wait_for_pid_exit(pid) for pid in captured_pids)


@pytest.mark.skipif(os.name != "posix", reason="native POSIX process-group evidence")
def test_posix_ordinary_finalization_removes_redirected_background_child(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "background.pid"
    child_source = (
        "import os,pathlib,threading;"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
        "threading.Event().wait(120)"
    )
    command = (
        f"{_python_command(child_source)} >/dev/null 2>&1 & "
        f"while [ ! -s {shlex.quote(str(pid_file))} ]; do :; done"
    )

    result = _executor().execute(
        _request(tmp_path, command),
        cancel_event=threading.Event(),
        on_event=lambda _event: None,
        admit_worker=_admit,
    )
    child_pid = int(pid_file.read_text(encoding="utf-8"))

    assert result.terminal_state == "exited"
    assert result.exit_code == 0
    assert result.cleanup_proven is True
    assert _wait_for_pid_exit(child_pid)
