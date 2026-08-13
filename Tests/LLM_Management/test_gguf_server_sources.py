from __future__ import annotations

import asyncio
import builtins
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, Callable

import pytest
from textual.widgets import Button

from Tests.Model_Artifacts.gguf_test_helpers import make_gguf
from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    llm_management_events as events,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    llm_management_events_mlx_lm as mlx_events,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    llm_management_events_vllm as vllm_events,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events.gguf_source_modes import (
    GGUFSourceMode,
    GGUFSourceSelection,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events import server_lifecycle
from tldw_chatbook.Model_Artifacts import (
    ArtifactInUseError,
    ArtifactRef,
    ModelArtifactService,
)
from tldw_chatbook.Model_Artifacts.gguf_admission import OpenedLocalGGUF


REF = ArtifactRef("local-gguf-example", "sha256-abc", "q4-k-m")
INACTIVE_REF = ArtifactRef("inactive", "sha256-def", "q8-0")


class _Logger:
    def __init__(self) -> None:
        self.records: list[tuple[str, tuple[object, ...]]] = []

    def _record(self, message: str, args: tuple[object, ...]) -> None:
        self.records.append((message, args))

    def info(self, message: str, *args: object) -> None:
        self._record(message, args)

    def error(self, message: str, *args: object) -> None:
        self._record(message, args)

    def debug(self, message: str, *args: object) -> None:
        self._record(message, args)


class _Destination:
    is_mounted = True

    def __init__(self) -> None:
        self.state_changes: list[tuple[str, str | None]] = []

    def _handle_server_process_state_change(
        self,
        provider: str,
        status: str | None = None,
    ) -> None:
        self.state_changes.append((provider, status))


class _App:
    def __init__(self) -> None:
        self._llm_server_lifecycle_lock = threading.RLock()
        self._llm_server_launch_claims: dict[str, object] = {}
        self.llamacpp_server_process = None
        self.llamafile_server_process = None
        self.vllm_server_process = None
        self.mlx_server_process = None
        self.destination = _Destination()
        self.screen_stack = [type("Screen", (), {"llm_window": self.destination})()]
        self.loguru_logger = _Logger()
        self.notifications: list[tuple[str, str | None]] = []
        self.workers: list[tuple[Callable[[], str], dict[str, object]]] = []
        self.after_callback: Callable[[Any, tuple[Any, ...], Any], None] | None = None

    def call_from_thread(self, callback: Callable[..., Any], *args: Any) -> Any:
        result = callback(*args)
        if self.after_callback is not None:
            self.after_callback(callback, args, result)
        return result

    def notify(self, message: str, severity: str | None = None) -> None:
        self.notifications.append((message, severity))

    def run_worker(self, work: Callable[[], str], **kwargs: object) -> None:
        self.workers.append((work, kwargs))


class _Lease:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _Process:
    pid = 4242

    def __init__(self, *, stubborn: bool = False, returncode: int = 0) -> None:
        self.stubborn = stubborn
        self.returncode = returncode
        self.running = True

    def poll(self) -> int | None:
        return None if self.running else self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None and self.stubborn:
            raise subprocess.TimeoutExpired("PRIVATE_COMMAND", timeout)
        self.running = False
        return self.returncode

    def terminate(self) -> None:
        if not self.stubborn:
            self.running = False

    def kill(self) -> None:
        if not self.stubborn:
            self.running = False


class _BlockingProcess(_Process):
    def __init__(self) -> None:
        super().__init__()
        self.wait_started = threading.Event()
        self.finish_wait = threading.Event()

    def wait(self, timeout: float | None = None) -> int:
        if timeout is None:
            self.wait_started.set()
            assert self.finish_wait.wait(timeout=5)
            return self.returncode
        if self.running:
            raise subprocess.TimeoutExpired("PRIVATE_COMMAND", timeout)
        self.finish_wait.set()
        return self.returncode

    def terminate(self) -> None:
        self.running = False
        self.finish_wait.set()


class _Subprocess:
    DEVNULL = subprocess.DEVNULL

    def __init__(
        self,
        process: _Process | None = None,
        *,
        error: Exception | None = None,
        before_popen: Callable[[list[str]], None] | None = None,
    ) -> None:
        self.process = process
        self.error = error
        self.before_popen = before_popen
        self.commands: list[list[str]] = []

    def Popen(self, command: list[str], **kwargs: object) -> _Process:
        captured = list(command)
        self.commands.append(captured)
        if self.before_popen is not None:
            self.before_popen(captured)
        if self.error is not None:
            raise self.error
        assert self.process is not None
        return self.process


@dataclass
class _InputWidget:
    value: str = ""
    text: str = ""
    focused: bool = False

    def focus(self) -> None:
        self.focused = True


class _LogWidget:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def clear(self) -> None:
        self.lines.clear()

    def write(self, message: str) -> None:
        self.lines.append(message)


class _Window:
    def __init__(self, widgets: dict[str, object]) -> None:
        self.widgets = widgets
        self.synced: list[str] = []

    def query_one(self, selector: str, _widget_type: type[object]) -> object:
        return self.widgets[selector]

    def _sync_process_controls(self, provider: str) -> None:
        self.synced.append(provider)


def _write_sparse_gguf(path: Path) -> None:
    data = make_gguf(architecture="llama", name="External")
    with path.open("wb") as handle:
        handle.write(data)
        handle.seek(2 * 1024 * 1024)
        handle.write(b"\0")


def _selection(
    mode: GGUFSourceMode,
    *,
    external_path: Path | None = None,
    managed_ref: ArtifactRef | None = None,
) -> GGUFSourceSelection:
    return GGUFSourceSelection(
        mode=mode,
        managed_ref=managed_ref,
        external_path=external_path,
    )


def _reserve(
    app: _App,
    provider: str,
    selection: GGUFSourceSelection,
) -> server_lifecycle.ServerLaunchClaim:
    claim = server_lifecycle.reserve_server_launch(
        app,
        provider,
        authority=selection.authority,
    )
    assert claim is not None
    return claim


@pytest.mark.parametrize(
    ("provider", "expected_message"),
    [
        ("llamacpp", "Llama.cpp executable was not found."),
        ("llamafile", "Llamafile executable was not found."),
    ],
)
@pytest.mark.asyncio
async def test_missing_executable_is_rejected_before_claim_or_worker(
    tmp_path: Path,
    provider: str,
    expected_message: str,
) -> None:
    missing_executable = tmp_path / "missing-runtime"
    model_id = f"#{provider}-model-path"
    executable = _InputWidget(value=str(missing_executable))
    widgets: dict[str, object] = {
        f"#{provider}-exec-path": executable,
        model_id: _InputWidget(value="" if provider == "llamafile" else "/model.gguf"),
        f"#{provider}-host": _InputWidget(value="127.0.0.1"),
        f"#{provider}-port": _InputWidget(value="8000"),
        f"#{provider}-additional-args": _InputWidget(),
        f"#{provider}-log-output": _LogWidget(),
    }
    window = _Window(widgets)
    app = _App()
    app.screen_stack = [type("Screen", (), {"llm_window": window})()]
    handler = (
        events.handle_start_llamacpp_server_button_pressed
        if provider == "llamacpp"
        else events.handle_start_llamafile_server_button_pressed
    )

    await handler(window, app, Button.Pressed(Button("Start")))

    assert app.notifications == [(expected_message, "error")]
    assert executable.focused is True
    assert app.workers == []
    assert server_lifecycle.current_server_claim(app, provider) is None
    assert window.synced == []


@pytest.mark.asyncio
async def test_vllm_command_snapshot_is_unchanged() -> None:
    widgets: dict[str, object] = {
        "#vllm-python-path": _InputWidget(value="python3"),
        "#vllm-model-path": _InputWidget(value="org/model"),
        "#vllm-host": _InputWidget(value="127.0.0.1"),
        "#vllm-port": _InputWidget(value="8124"),
        "#vllm-additional-args": _InputWidget(text="--dtype float16"),
        "#vllm-log-output": _LogWidget(),
    }
    window = _Window(widgets)
    app = _App()
    app.screen_stack = [type("Screen", (), {"llm_window": window})()]

    await vllm_events.handle_start_vllm_server_button_pressed(
        window,
        app,
        Button.Pressed(Button("Start")),
    )

    assert len(app.workers) == 1
    work, _options = app.workers[0]
    assert getattr(work, "args")[1] == [
        "python3",
        "-m",
        "vllm.entrypoints.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        "8124",
        "--model",
        "org/model",
        "--dtype",
        "float16",
    ]
    claim = server_lifecycle.current_server_claim(app, "vllm")
    assert claim is not None
    server_lifecycle.release_server_claim(app, "vllm", claim)


@pytest.mark.asyncio
async def test_mlx_command_snapshot_is_unchanged() -> None:
    widgets: dict[str, object] = {
        "#mlx-model-path": _InputWidget(value="org/mlx-model"),
        "#mlx-host": _InputWidget(value="0.0.0.0"),
        "#mlx-port": _InputWidget(value="8125"),
        "#mlx-additional-args": _InputWidget(text="--trust-remote-code"),
        "#mlx-log-output": _LogWidget(),
    }
    window = _Window(widgets)
    app = _App()
    app.screen_stack = [type("Screen", (), {"llm_window": window})()]

    await mlx_events.handle_start_mlx_server_button_pressed(
        window,
        app,
        Button.Pressed(Button("Start")),
    )

    assert len(app.workers) == 1
    work, _options = app.workers[0]
    assert getattr(work, "args")[1] == [
        "python",
        "-m",
        "mlx_lm.server",
        "--model",
        "org/mlx-model",
        "--host",
        "0.0.0.0",
        "--port",
        "8125",
        "--trust-remote-code",
    ]
    claim = server_lifecycle.current_server_claim(app, "mlx")
    assert claim is not None
    server_lifecycle.release_server_claim(app, "mlx", claim)


@pytest.mark.parametrize(
    ("provider", "selection_factory", "model_flag"),
    [
        (
            "llamacpp",
            lambda outside: _selection(
                GGUFSourceMode.MANAGED,
                managed_ref=REF,
                external_path=outside / "PRIVATE_INACTIVE_EXTERNAL.gguf",
            ),
            "--model",
        ),
        (
            "llamacpp",
            lambda outside: _selection(
                GGUFSourceMode.EXTERNAL,
                managed_ref=INACTIVE_REF,
                external_path=outside / "external.gguf",
            ),
            "--model",
        ),
        (
            "llamafile",
            lambda outside: _selection(
                GGUFSourceMode.EMBEDDED,
                managed_ref=INACTIVE_REF,
                external_path=outside / "PRIVATE_INACTIVE_EXTERNAL.gguf",
            ),
            None,
        ),
        (
            "llamafile",
            lambda outside: _selection(
                GGUFSourceMode.MANAGED,
                managed_ref=REF,
                external_path=outside / "PRIVATE_INACTIVE_EXTERNAL.gguf",
            ),
            "-m",
        ),
        (
            "llamafile",
            lambda outside: _selection(
                GGUFSourceMode.EXTERNAL,
                managed_ref=INACTIVE_REF,
                external_path=outside / "external.gguf",
            ),
            "-m",
        ),
    ],
)
def test_source_command_matrix_uses_only_active_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    selection_factory: Callable[[Path], GGUFSourceSelection],
    model_flag: str | None,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "external.gguf"
    _write_sparse_gguf(external)
    managed_payload = tmp_path / "managed" / "model.gguf"
    lease = _Lease()
    selection = selection_factory(outside)
    app = _App()
    claim = _reserve(app, provider, selection)
    commands: list[list[str]] = []
    acquired: list[ArtifactRef] = []

    def acquire(_service: object, reference: ArtifactRef) -> tuple[Path, _Lease]:
        acquired.append(reference)
        return managed_payload, lease

    def run(
        _app: object,
        actual_provider: str,
        command: list[str],
        actual_claim: object,
        _subprocess_module: object,
        **_kwargs: object,
    ) -> str:
        assert actual_provider == provider
        assert actual_claim is claim
        if selection.mode is GGUFSourceMode.MANAGED:
            assert claim._resource is lease
        commands.append(command)
        return "captured"

    monkeypatch.setattr(events, "managed_service", lambda: object())
    monkeypatch.setattr(events, "acquire_managed_gguf", acquire)
    monkeypatch.setattr(events, "run_server_subprocess", run)

    worker = (
        events.run_llamacpp_server_worker
        if provider == "llamacpp"
        else events.run_llamafile_server_worker
    )
    result = worker(
        app,
        "/private/runtime",
        "127.0.0.1",
        "8123",
        ("--threads", "2"),
        selection,
        claim,
    )

    assert result == "captured"
    assert len(commands) == 1
    command = commands[0]
    expected = ["/private/runtime"]
    if model_flag is None:
        assert "-m" not in command
        assert "--model" not in command
        assert str(external) not in command
        assert str(managed_payload) not in command
    else:
        expected_model = (
            managed_payload if selection.mode is GGUFSourceMode.MANAGED else external
        )
        expected.extend((model_flag, str(expected_model)))
    expected.extend(("--host", "127.0.0.1", "--port", "8123", "--threads", "2"))
    assert command == expected
    assert str(outside / "PRIVATE_INACTIVE_EXTERNAL.gguf") not in command
    assert acquired == ([REF] if selection.mode is GGUFSourceMode.MANAGED else [])
    server_lifecycle.release_server_claim(app, provider, claim)


@pytest.mark.asyncio
async def test_legacy_blank_llamafile_handler_schedules_embedded_static_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "llamafile"
    executable.touch()
    widgets: dict[str, object] = {
        "#llamafile-exec-path": _InputWidget(value=str(executable)),
        "#llamafile-model-path": _InputWidget(value=""),
        "#llamafile-host": _InputWidget(value="127.0.0.1"),
        "#llamafile-port": _InputWidget(value="8000"),
        "#llamafile-additional-args": _InputWidget(text="--threads 2"),
        "#llamafile-log-output": _LogWidget(),
    }
    window = _Window(widgets)
    app = _App()
    app.screen_stack = [type("Screen", (), {"llm_window": window})()]

    await events.handle_start_llamafile_server_button_pressed(
        window,
        app,
        Button.Pressed(Button("Start")),
    )

    assert len(app.workers) == 1
    work, options = app.workers[0]
    assert options["thread"] is True
    assert options["description"] == "Running Llamafile server process"
    assert "/private" not in str(options)
    assert window.synced[0] == "llamafile"
    claim = server_lifecycle.current_server_claim(app, "llamafile")
    assert claim is not None
    assert claim.authority == "Embedded"

    captured: list[list[str]] = []
    monkeypatch.setattr(
        events,
        "run_server_subprocess",
        lambda _app, _provider, command, *_args, **_kwargs: (
            captured.append(command) or "captured"
        ),
    )
    assert await asyncio.to_thread(work) == "captured"
    assert "-m" not in captured[0]
    assert "/private" not in " ".join(widgets["#llamafile-log-output"].lines)
    server_lifecycle.release_server_claim(app, "llamafile", claim)


@pytest.mark.asyncio
async def test_handler_prefers_explicit_source_snapshot_without_model_input(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "llama-server"
    executable.touch()
    widgets: dict[str, object] = {
        "#llamacpp-exec-path": _InputWidget(value=str(executable)),
        "#llamacpp-host": _InputWidget(value="127.0.0.1"),
        "#llamacpp-port": _InputWidget(value="8001"),
        "#llamacpp-additional-args": _InputWidget(value=""),
        "#llamacpp-log-output": _LogWidget(),
    }
    window = _Window(widgets)
    selection = _selection(GGUFSourceMode.MANAGED, managed_ref=REF)
    window.gguf_source_snapshot = lambda provider: selection  # type: ignore[attr-defined]
    app = _App()
    app.screen_stack = [type("Screen", (), {"llm_window": window})()]

    await events.handle_start_llamacpp_server_button_pressed(
        window,
        app,
        Button.Pressed(Button("Start")),
    )

    assert len(app.workers) == 1
    claim = server_lifecycle.current_server_claim(app, "llamacpp")
    assert claim is not None
    assert claim.authority == "Managed GGUF"
    server_lifecycle.release_server_claim(app, "llamacpp", claim)


@pytest.mark.asyncio
async def test_external_handler_keeps_event_loop_alive_while_worker_validates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "outside.gguf"
    _write_sparse_gguf(source)
    executable = tmp_path / "llama-server"
    executable.touch()
    widgets: dict[str, object] = {
        "#llamacpp-exec-path": _InputWidget(value=str(executable)),
        "#llamacpp-model-path": _InputWidget(value=str(source)),
        "#llamacpp-host": _InputWidget(value="127.0.0.1"),
        "#llamacpp-port": _InputWidget(value="8001"),
        "#llamacpp-additional-args": _InputWidget(value=""),
        "#llamacpp-log-output": _LogWidget(),
    }
    window = _Window(widgets)
    app = _App()
    app.screen_stack = [type("Screen", (), {"llm_window": window})()]
    entered = threading.Event()
    release = threading.Event()
    worker_threads: list[int] = []
    original_inspect = events.inspect_gguf_structure

    def blocking_inspect(handle: Any, *, file_size: int) -> object:
        worker_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(timeout=5)
        return original_inspect(handle, file_size=file_size)

    monkeypatch.setattr(events, "inspect_gguf_structure", blocking_inspect)
    monkeypatch.setattr(
        events,
        "run_server_subprocess",
        lambda *_args, **_kwargs: "captured",
    )

    await events.handle_start_llamacpp_server_button_pressed(
        window,
        app,
        Button.Pressed(Button("Start")),
    )
    assert len(app.workers) == 1
    work, _options = app.workers[0]
    heartbeat = 0
    pulsing = True

    async def pulse() -> None:
        nonlocal heartbeat
        while pulsing:
            heartbeat += 1
            await asyncio.sleep(0)

    pulse_task = asyncio.create_task(pulse())
    worker_task = asyncio.create_task(asyncio.to_thread(work))
    assert await asyncio.to_thread(entered.wait, 5)
    for _ in range(10):
        await asyncio.sleep(0)

    assert heartbeat > 0
    assert worker_threads and worker_threads[0] != threading.get_ident()
    release.set()
    assert await worker_task == "captured"
    pulsing = False
    await pulse_task
    claim = server_lifecycle.current_server_claim(app, "llamacpp")
    assert claim is not None
    server_lifecycle.release_server_claim(app, "llamacpp", claim)


def test_external_source_validation_is_worker_thread_store_free_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    source = outside / "private-external.gguf"
    _write_sparse_gguf(source)
    before_bytes = source.read_bytes()
    before_stat = source.stat()
    before_tree = tuple(
        sorted(
            (path.relative_to(tmp_path), path.stat().st_size)
            for path in tmp_path.rglob("*")
            if path.is_file()
        )
    )
    selection = _selection(GGUFSourceMode.EXTERNAL, external_path=source)
    app = _App()
    claim = _reserve(app, "llamacpp", selection)
    main_thread = threading.get_ident()
    inspection_threads: list[int] = []
    recheck_threads: list[int] = []
    commands: list[list[str]] = []
    source_os_opens: list[Path] = []
    original_inspect = events.inspect_gguf_structure
    original_open_local = events.open_local_gguf
    original_recheck = OpenedLocalGGUF.recheck
    original_os_open = os.open
    original_builtin_open = builtins.open
    original_path_open = Path.open
    meters: list[Any] = []

    class ReadMeter:
        def __init__(self, handle: Any) -> None:
            self.handle = handle
            self.bytes_read = 0
            self.requests: list[int] = []

        def read(self, size: int = -1) -> bytes:
            assert size >= 0, "external validation attempted an unbounded read"
            self.requests.append(size)
            data = self.handle.read(size)
            self.bytes_read += len(data)
            return data

        def __getattr__(self, name: str) -> Any:
            return getattr(self.handle, name)

    @contextmanager
    def metered_open(path: Path):
        with original_open_local(path) as opened:
            meter = ReadMeter(opened.handle)
            meters.append(meter)
            opened.handle = meter
            yield opened

    def fail_managed_service() -> object:
        raise AssertionError("external launch touched managed service")

    def fail_store_operation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external launch touched managed artifact state")

    def guarded_os_open(path: object, flags: int, *args: object, **kwargs: object):
        selected = Path(path).absolute()
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError("external launch attempted a filesystem write")
        if selected == source.absolute():
            source_os_opens.append(selected)
        return original_os_open(path, flags, *args, **kwargs)

    def guarded_builtin_open(
        file: object,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> Any:
        if any(flag in mode for flag in "wax+"):
            raise AssertionError("external launch attempted a filesystem write")
        if (
            isinstance(file, (str, os.PathLike))
            and Path(file).absolute() == source.absolute()
        ):
            raise AssertionError("external launch reopened the selected source")
        return original_builtin_open(file, mode, *args, **kwargs)

    def guarded_path_open(
        path: Path,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> Any:
        if any(flag in mode for flag in "wax+"):
            raise AssertionError("external launch attempted a filesystem write")
        if path.absolute() == source.absolute():
            raise AssertionError("external launch reopened the selected source")
        return original_path_open(path, mode, *args, **kwargs)

    def inspect(handle: Any, *, file_size: int) -> object:
        inspection_threads.append(threading.get_ident())
        return original_inspect(handle, file_size=file_size)

    def recheck(opened: OpenedLocalGGUF) -> None:
        recheck_threads.append(threading.get_ident())
        original_recheck(opened)

    monkeypatch.setattr(events, "managed_service", fail_managed_service)
    monkeypatch.setattr(events, "initial_gguf_selection", fail_store_operation)
    monkeypatch.setattr(
        events,
        "ModelArtifactService",
        fail_store_operation,
        raising=False,
    )
    monkeypatch.setattr(events, "open_local_gguf", metered_open)
    monkeypatch.setattr(events, "inspect_gguf_structure", inspect)
    monkeypatch.setattr(OpenedLocalGGUF, "recheck", recheck)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(Path, "write_bytes", fail_store_operation)
    monkeypatch.setattr(Path, "write_text", fail_store_operation)
    for method in (
        "list_installed",
        "acquire",
        "activate",
        "delete",
        "import_local_gguf",
    ):
        monkeypatch.setattr(ModelArtifactService, method, fail_store_operation)
    for operation in ("copy", "copy2", "copyfile"):
        monkeypatch.setattr(shutil, operation, fail_store_operation)
    monkeypatch.setattr(
        events,
        "run_server_subprocess",
        lambda _app, _provider, command, *_args, **_kwargs: (
            commands.append(command) or "captured"
        ),
    )

    results: list[str | None] = []
    thread = threading.Thread(
        target=lambda: results.append(
            events.run_llamacpp_server_worker(
                app,
                "/runtime",
                "127.0.0.1",
                "8001",
                (),
                selection,
                claim,
            )
        )
    )
    thread.start()
    thread.join(timeout=5)

    assert thread.is_alive() is False
    assert results == ["captured"]
    assert inspection_threads and set(inspection_threads) == {thread.ident}
    assert recheck_threads and set(recheck_threads) == {thread.ident}
    assert main_thread not in inspection_threads
    assert commands[0][commands[0].index("--model") + 1] == str(source.absolute())
    assert source_os_opens == [source.absolute()]
    assert len(meters) == 1
    assert 0 < meters[0].bytes_read <= 4096
    assert max(meters[0].requests) <= 4096
    with original_path_open(source, "rb") as handle:
        assert handle.read() == before_bytes
    after_stat = source.stat()
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert list(managed_root.iterdir()) == []
    assert (
        tuple(
            sorted(
                (path.relative_to(tmp_path), path.stat().st_size)
                for path in tmp_path.rglob("*")
                if path.is_file()
            )
        )
        == before_tree
    )
    server_lifecycle.release_server_claim(app, "llamacpp", claim)


@pytest.mark.parametrize("case", ["malformed", "missing", "symlink", "special"])
def test_external_source_rejections_happen_before_popen_without_private_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    private_marker = "PRIVATE_EXTERNAL_SOURCE"
    source = tmp_path / f"{private_marker}.gguf"
    if case == "malformed":
        source.write_bytes(b"not a gguf")
    elif case == "symlink":
        target = tmp_path / "target.gguf"
        _write_sparse_gguf(target)
        try:
            source.symlink_to(target)
        except OSError:
            pytest.skip("symlinks unavailable")
    elif case == "special":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable")
        os.mkfifo(source)

    selection = _selection(GGUFSourceMode.EXTERNAL, external_path=source)
    app = _App()
    claim = _reserve(app, "llamacpp", selection)
    monkeypatch.setattr(
        events,
        "managed_service",
        lambda: pytest.fail("external source touched managed service"),
    )
    monkeypatch.setattr(
        events,
        "run_server_subprocess",
        lambda *_args, **_kwargs: pytest.fail("invalid source reached Popen"),
    )

    result = events.run_llamacpp_server_worker(
        app,
        "/runtime",
        "127.0.0.1",
        "8001",
        (),
        selection,
        claim,
    )

    assert result == "llamacpp source preparation failed"
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None
    captured = repr(
        (
            result,
            app.destination.state_changes,
            app.notifications,
            app.loguru_logger.records,
        )
    )
    assert private_marker not in captured
    assert "not a gguf" not in captured


def test_external_replacement_after_inspection_fails_final_recheck_before_popen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "PRIVATE_SELECTED.gguf"
    replacement = tmp_path / "PRIVATE_REPLACEMENT.gguf"
    _write_sparse_gguf(source)
    replacement.write_bytes(make_gguf(architecture="llama", name="Replacement"))
    selection = _selection(GGUFSourceMode.EXTERNAL, external_path=source)
    app = _App()
    claim = _reserve(app, "llamacpp", selection)
    original_inspect = events.inspect_gguf_structure

    def inspect_then_replace(handle: Any, *, file_size: int) -> object:
        inspected = original_inspect(handle, file_size=file_size)
        replacement.replace(source)
        return inspected

    monkeypatch.setattr(events, "inspect_gguf_structure", inspect_then_replace)
    monkeypatch.setattr(
        events,
        "run_server_subprocess",
        lambda *_args, **_kwargs: pytest.fail("replaced source reached Popen"),
    )

    result = events.run_llamacpp_server_worker(
        app,
        "/runtime",
        "127.0.0.1",
        "8001",
        (),
        selection,
        claim,
    )

    assert result == "llamacpp source preparation failed"
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None
    assert "PRIVATE" not in repr(
        (result, app.destination.state_changes, app.loguru_logger.records)
    )


def test_managed_transfer_precedes_popen_and_spawn_failure_closes_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = tmp_path / "managed" / "model.gguf"
    lease = _Lease()
    app = _App()
    selection = _selection(GGUFSourceMode.MANAGED, managed_ref=REF)
    claim = _reserve(app, "llamacpp", selection)
    acquired: list[ArtifactRef] = []

    def acquire(_service: object, reference: ArtifactRef) -> tuple[Path, _Lease]:
        acquired.append(reference)
        return payload, lease

    scenario = _Subprocess(
        error=OSError("PRIVATE_SPAWN_FAILURE"),
        before_popen=lambda _command: (
            pytest.fail("lease not transferred before Popen")
            if claim._resource is not lease
            else None
        ),
    )
    monkeypatch.setattr(events, "managed_service", lambda: object())
    monkeypatch.setattr(events, "acquire_managed_gguf", acquire)
    monkeypatch.setattr(events, "subprocess", scenario)

    result = events.run_llamacpp_server_worker(
        app,
        "/runtime",
        "127.0.0.1",
        "8001",
        (),
        selection,
        claim,
    )

    assert acquired == [REF]
    assert scenario.commands[0][2] == str(payload)
    assert result == "llamacpp server failed (category=OSError)"
    assert "PRIVATE" not in result
    assert lease.close_count == 1
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None


def test_rejected_managed_transfer_closes_worker_lease_and_preserves_new_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lease = _Lease()
    app = _App()
    selection = _selection(GGUFSourceMode.MANAGED, managed_ref=REF)
    stale = _reserve(app, "llamacpp", selection)
    assert server_lifecycle.release_server_claim(app, "llamacpp", stale)
    current = _reserve(app, "llamacpp", selection)
    monkeypatch.setattr(events, "managed_service", lambda: object())
    monkeypatch.setattr(
        events,
        "acquire_managed_gguf",
        lambda _service, reference: (tmp_path / "model.gguf", lease),
    )
    monkeypatch.setattr(
        events,
        "run_server_subprocess",
        lambda *_args, **_kwargs: pytest.fail("stale transfer reached Popen"),
    )

    result = events.run_llamacpp_server_worker(
        app,
        "/runtime",
        "127.0.0.1",
        "8001",
        (),
        selection,
        stale,
    )

    assert result == "llamacpp launch cancelled"
    assert lease.close_count == 1
    assert server_lifecycle.current_server_claim(app, "llamacpp") is current
    assert app.destination.state_changes == []
    server_lifecycle.release_server_claim(app, "llamacpp", current)


@pytest.mark.asyncio
async def test_successful_stop_of_managed_worker_closes_lease_after_process_death(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lease = _Lease()
    app = _App()
    selection = _selection(GGUFSourceMode.MANAGED, managed_ref=REF)
    claim = _reserve(app, "llamacpp", selection)
    process = _BlockingProcess()
    monkeypatch.setattr(events, "managed_service", lambda: object())
    monkeypatch.setattr(
        events,
        "acquire_managed_gguf",
        lambda _service, _reference: (tmp_path / "managed" / "model.gguf", lease),
    )
    monkeypatch.setattr(events, "subprocess", _Subprocess(process))
    results: list[str | None] = []
    worker = threading.Thread(
        target=lambda: results.append(
            events.run_llamacpp_server_worker(
                app,
                "/runtime",
                "127.0.0.1",
                "8001",
                (),
                selection,
                claim,
            )
        )
    )
    worker.start()
    assert await asyncio.to_thread(process.wait_started.wait, 5)

    assert await server_lifecycle.stop_server_process(
        app,
        "llamacpp",
        "Llama.cpp server",
    )
    await asyncio.to_thread(worker.join, 5)

    assert worker.is_alive() is False
    assert results == ["llamacpp server exited (code=0)"]
    assert process.poll() is not None
    assert lease.close_count == 1
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None


def test_cancel_before_managed_preparation_releases_claim_without_acquire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _App()
    selection = _selection(GGUFSourceMode.MANAGED, managed_ref=REF)
    claim = _reserve(app, "llamacpp", selection)
    claim.cancel_event.set()
    monkeypatch.setattr(
        events,
        "managed_service",
        lambda: pytest.fail("cancelled launch reached managed service"),
    )

    result = events.run_llamacpp_server_worker(
        app,
        "/runtime",
        "127.0.0.1",
        "8001",
        (),
        selection,
        claim,
    )

    assert result == "llamacpp launch cancelled"
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None


def test_real_managed_lease_blocks_delete_until_exact_claim_and_process_death(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.gguf"
    source.write_bytes(make_gguf(architecture="llama", name="Managed", file_type=7))
    service = ModelArtifactService(
        tmp_path / "store",
        lease_timeout_seconds=0.01,
    )
    reference = service.import_local_gguf(source).reference
    service.activate(reference)
    app = _App()
    selection = _selection(GGUFSourceMode.MANAGED, managed_ref=reference)
    claim = _reserve(app, "llamacpp", selection)
    process = _Process(stubborn=True)
    acquired: list[ArtifactRef] = []
    real_acquire = service.acquire

    def acquire_exact(exact_reference: ArtifactRef) -> object:
        acquired.append(exact_reference)
        return real_acquire(exact_reference)

    def cancel_after_publication(
        callback: Any,
        _args: tuple[Any, ...],
        published: Any,
    ) -> None:
        if callback is server_lifecycle.publish_server_process and published:
            claim.cancel_event.set()

    app.after_callback = cancel_after_publication
    monkeypatch.setattr(service, "acquire", acquire_exact)
    monkeypatch.setattr(events, "managed_service", lambda: service)
    monkeypatch.setattr(events, "subprocess", _Subprocess(process))

    result = events.run_llamacpp_server_worker(
        app,
        "/runtime",
        "127.0.0.1",
        "8001",
        (),
        selection,
        claim,
    )

    assert result == "llamacpp launch cancelled"
    assert acquired == [reference]
    assert server_lifecycle.current_server_claim(app, "llamacpp") is claim
    assert server_lifecycle.server_process(app, "llamacpp") is process
    assert claim._resource is not None
    with pytest.raises(ArtifactInUseError):
        service.delete(reference)

    process.running = False
    assert server_lifecycle.clear_server_process(app, "llamacpp", claim, process)
    service.delete(reference)
    assert service.artifact_path(reference).exists() is False


def test_managed_source_failure_is_path_private_and_does_not_overwrite_newer_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_marker = "PRIVATE_MANAGED_EXCEPTION"
    app = _App()
    selection = _selection(GGUFSourceMode.MANAGED, managed_ref=REF)
    stale = _reserve(app, "llamacpp", selection)
    assert server_lifecycle.release_server_claim(app, "llamacpp", stale)
    current = _reserve(app, "llamacpp", selection)
    monkeypatch.setattr(events, "managed_service", lambda: object())
    monkeypatch.setattr(
        events,
        "acquire_managed_gguf",
        lambda _service, _reference: (_ for _ in ()).throw(
            RuntimeError(private_marker)
        ),
    )

    result = events.run_llamacpp_server_worker(
        app,
        "/runtime",
        "127.0.0.1",
        "8001",
        (),
        selection,
        stale,
    )

    assert result == "llamacpp source preparation failed"
    assert server_lifecycle.current_server_claim(app, "llamacpp") is current
    captured = repr((result, app.destination.state_changes, app.loguru_logger.records))
    assert private_marker not in captured
    assert app.destination.state_changes == []
    server_lifecycle.release_server_claim(app, "llamacpp", current)
