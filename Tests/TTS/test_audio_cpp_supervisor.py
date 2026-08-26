from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from loguru import logger

from Tests.TTS.fixtures.fake_audiocpp_server import write_executable_wrapper
from tldw_chatbook.TTS import audio_cpp_supervisor as supervisor_module
from tldw_chatbook.TTS._async_lifecycle import shutdown_deadline_scope
from tldw_chatbook.TTS.adapter_types import (
    TTSOperationError,
    TTSProviderReconfiguringError,
)
from tldw_chatbook.TTS.audio_cpp_managed_config import AudioCppManagedLaunchConfig
from tldw_chatbook.TTS.audio_cpp_supervisor import (
    AudioCppGenerationHooks,
    AudioCppSupervisor,
    _AudioCppDiagnosticRing,
    _AudioCppGenerationChanged,
    _OwnedAudioCppProcess,
)

# Network opt-in (task-15111): this module talks to `fake_audiocpp_server`,
# an in-process HTTP server on an ephemeral loopback port.
# The autouse guard in Tests/conftest.py denies egress by default; every address
# these tests reach is a port this process itself is listening on.
pytestmark = pytest.mark.allow_network


class _FakeReader:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
        self.read_calls = 0

    async def read(self, _size: int = -1) -> bytes:
        self.read_calls += 1
        item = await self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return b"" if item is None else item

    def feed(self, value: bytes) -> None:
        self._queue.put_nowait(value)

    def fail(self, error: BaseException) -> None:
        self._queue.put_nowait(error)

    def finish(self) -> None:
        self._queue.put_nowait(None)


def _exception_graph(error: BaseException) -> list[BaseException]:
    pending = [error]
    seen: set[int] = set()
    graph: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        graph.append(current)
        for linked in (current.__context__, current.__cause__):
            if linked is not None:
                pending.append(linked)
    return graph


class _FakeProcess:
    def __init__(
        self,
        *,
        exit_on_terminate: bool = True,
        finish_pipes_on_exit: bool = True,
    ) -> None:
        self.returncode: int | None = None
        self.stdout = _FakeReader()
        self.stderr = _FakeReader()
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0
        self.close_parent_calls = 0
        self._exit_on_terminate = exit_on_terminate
        self._finish_pipes_on_exit = finish_pipes_on_exit
        self._exited = asyncio.Event()

    async def wait(self) -> int:
        self.wait_calls += 1
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._exit_on_terminate:
            self.exit(-15)

    def kill(self) -> None:
        self.kill_calls += 1
        self.exit(-9)

    def exit(self, returncode: int = 0) -> None:
        self.publish_returncode(returncode)
        self.complete_exit()

    def publish_returncode(self, returncode: int = 0) -> None:
        if self.returncode is not None:
            return
        self.returncode = returncode

    def complete_exit(self) -> None:
        if self.returncode is None or self._exited.is_set():
            return
        if self._finish_pipes_on_exit:
            self.stdout.finish()
            self.stderr.finish()
        self._exited.set()

    def close_parent_pipes(self) -> None:
        self.close_parent_calls += 1
        self.stdout.finish()
        self.stderr.finish()


class _GeneratedArtifactSpy:
    def __init__(
        self,
        *,
        validate_failure: BaseException | None = None,
        cleanup_failure: BaseException | None = None,
    ) -> None:
        self.validate_calls = 0
        self.cleanup_calls = 0
        self.validate_failure = validate_failure
        self.cleanup_failure = cleanup_failure

    def validate(self) -> None:
        self.validate_calls += 1
        if self.validate_failure is not None:
            raise self.validate_failure

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.cleanup_failure is not None:
            raise self.cleanup_failure


class _FakeLauncher:
    def __init__(
        self,
        processes: list[_FakeProcess] | None = None,
        *,
        gate: asyncio.Event | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self.processes = processes or []
        self.gate = gate
        self.failure = failure
        self.calls: list[tuple[AudioCppManagedLaunchConfig, dict[str, str]]] = []

    async def __call__(
        self,
        launch: AudioCppManagedLaunchConfig,
        child_environment: dict[str, str],
    ) -> _OwnedAudioCppProcess:
        self.calls.append((launch, dict(child_environment)))
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        process = self.processes.pop(0) if self.processes else _FakeProcess()
        return _OwnedAudioCppProcess(
            process=process,
            close_parent_pipes=process.close_parent_pipes,
        )


class _HooksFactory:
    def __init__(
        self,
        *,
        contract: str = "available",
        health: bool = True,
        gate: asyncio.Event | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self.contract = contract
        self.health = health
        self.gate = gate
        self.failure = failure
        self.calls: list[int] = []
        self.contract_calls = 0
        self.health_calls = 0
        self.cleanup_calls = 0

    async def __call__(self, generation: int) -> AudioCppGenerationHooks:
        self.calls.append(generation)
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure

        async def contract_probe() -> str:
            self.contract_calls += 1
            return self.contract

        async def health_probe() -> bool:
            self.health_calls += 1
            return self.health

        async def cleanup() -> None:
            self.cleanup_calls += 1

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=health_probe,
            cleanup=cleanup,
        )


class _ManualSleep:
    def __init__(self) -> None:
        self.waiters: list[asyncio.Future[None]] = []
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        waiter = asyncio.get_running_loop().create_future()
        self.waiters.append(waiter)
        await waiter

    def release_next(self) -> None:
        for waiter in self.waiters:
            if not waiter.done():
                waiter.set_result(None)
                return
        raise AssertionError("no sleeping task is waiting")


class _ControlledHooksFactory:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.health_calls = 0
        self.active_health_calls = 0
        self.max_active_health_calls = 0
        self.cleanup_calls = 0
        self.health_results: asyncio.Queue[bool | asyncio.Future[bool]] = (
            asyncio.Queue()
        )
        self.health_results.put_nowait(True)

    async def __call__(self, generation: int) -> AudioCppGenerationHooks:
        self.calls.append(generation)

        async def contract_probe() -> str:
            return "available"

        async def health_probe() -> bool:
            self.health_calls += 1
            self.active_health_calls += 1
            self.max_active_health_calls = max(
                self.max_active_health_calls, self.active_health_calls
            )
            try:
                result = await self.health_results.get()
                if isinstance(result, asyncio.Future):
                    return await result
                return result
            finally:
                self.active_health_calls -= 1

        async def cleanup() -> None:
            self.cleanup_calls += 1

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=health_probe,
            cleanup=cleanup,
        )

    def queue_health(self, result: bool | asyncio.Future[bool]) -> None:
        self.health_results.put_nowait(result)


def _make_launch(
    tmp_path: Path,
    *,
    port: int = 19_876,
    startup_timeout_seconds: float = 30.0,
    health_check_interval_seconds: float = 60.0,
    termination_grace_seconds: float = 5.0,
) -> AudioCppManagedLaunchConfig:
    binary = tmp_path / "audiocpp_server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    server_json = tmp_path / "server.json"
    server_json.write_text(
        json.dumps({"host": "127.0.0.1", "port": port}),
        encoding="utf-8",
    )
    return AudioCppManagedLaunchConfig(
        binary_path=binary,
        server_json_path=server_json,
        working_directory=tmp_path,
        base_url=f"http://127.0.0.1:{port}",
        startup_timeout_seconds=startup_timeout_seconds,
        health_check_interval_seconds=health_check_interval_seconds,
        termination_grace_seconds=termination_grace_seconds,
    )


async def _available_preflight(_port: int, _timeout: float) -> str:
    return "available"


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


async def _run_periodic_probe(
    sleep: _ManualSleep,
    hooks: _ControlledHooksFactory,
    result: bool | asyncio.Future[bool],
) -> None:
    prior_calls = hooks.health_calls
    hooks.queue_health(result)
    await _wait_until(lambda: any(not waiter.done() for waiter in sleep.waiters))
    sleep.release_next()
    await _wait_until(lambda: hooks.health_calls == prior_calls + 1)
    if isinstance(result, bool):
        await _wait_until(lambda: hooks.active_health_calls == 0)


def _require_real_child_support() -> None:
    if os.name != "posix":
        pytest.skip("direct executable shebang wrappers require a POSIX host")
    if any(character.isspace() for character in sys.executable):
        pytest.skip("the current Python path cannot be represented in a shebang")


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _make_real_launch(
    tmp_path: Path,
    *,
    behavior: dict[str, Any] | None = None,
    startup_timeout_seconds: float = 30.0,
    termination_grace_seconds: float = 0.1,
) -> AudioCppManagedLaunchConfig:
    _require_real_child_support()
    wrapper = write_executable_wrapper(tmp_path / "fake_audiocpp_server")
    server_json = tmp_path / "server.json"
    port = _unused_loopback_port()
    server_json.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": port,
                "test_behavior": behavior or {},
            }
        ),
        encoding="utf-8",
    )
    return AudioCppManagedLaunchConfig(
        binary_path=wrapper,
        server_json_path=server_json,
        working_directory=tmp_path,
        base_url=f"http://127.0.0.1:{port}",
        startup_timeout_seconds=startup_timeout_seconds,
        health_check_interval_seconds=2.0,
        termination_grace_seconds=termination_grace_seconds,
    )


class _CountingProcess:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self.stdout = process.stdout
        self.stderr = process.stderr
        self.wait_calls = 0

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def wait(self) -> int:
        self.wait_calls += 1
        return await self._process.wait()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()


class _RealLauncher:
    def __init__(self) -> None:
        self.processes: list[_CountingProcess] = []

    async def __call__(
        self,
        launch: AudioCppManagedLaunchConfig,
        child_environment: dict[str, str],
    ) -> _OwnedAudioCppProcess:
        owned = await supervisor_module._default_process_launcher(
            launch,
            child_environment,
        )
        process = _CountingProcess(owned.process)
        self.processes.append(process)
        return _OwnedAudioCppProcess(
            process=process,
            close_parent_pipes=owned.close_parent_pipes,
            close_native_transport=owned.close_native_transport,
        )


class _RealHttpHooksFactory:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self.clients: list[httpx.AsyncClient] = []

    async def __call__(self, _generation: int) -> AudioCppGenerationHooks:
        client = httpx.AsyncClient(
            base_url=self._base_url,
            trust_env=False,
            follow_redirects=False,
            timeout=1.0,
        )
        self.clients.append(client)

        async def health_probe() -> bool:
            try:
                response = await client.get("/health")
                return response.status_code == 200
            except httpx.HTTPError:
                return False

        async def contract_probe() -> str:
            response = await client.get("/v1/models")
            response.raise_for_status()
            return "available"

        async def cleanup() -> None:
            await client.aclose()

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=health_probe,
            cleanup=cleanup,
        )


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _wait_for_pid_exit(pid: int, *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while _pid_exists(pid):
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"fixture process {pid} did not exit")
        await asyncio.sleep(0.01)


async def _wait_for_real_condition(
    predicate: Callable[[], bool],
    *,
    timeout: float = 3.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("real-child condition did not become true")
        await asyncio.sleep(0.01)


async def _terminate_fixture_pid(pid: int) -> None:
    if not _pid_exists(pid):
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await _wait_for_pid_exit(pid)


def _snapshot_texts(ring: _AudioCppDiagnosticRing) -> tuple[str, ...]:
    lines, _dropped = ring.snapshot()
    return tuple(line.text for line in lines)


def test_diagnostics_bound_lines_total_utf8_bytes_and_each_line() -> None:
    line_bounded = _AudioCppDiagnosticRing()
    line_bounded.feed("stdout", (("😀" * 2_000) + "\n").encode())
    line_bounded.finish("stdout")
    line_lines, _ = line_bounded.snapshot()
    assert line_lines
    assert all(len(line.text.encode("utf-8")) <= 4_096 for line in line_lines)

    count_bounded = _AudioCppDiagnosticRing()
    count_bounded.feed("stdout", b"line\n" * 205)
    count_bounded.finish("stdout")
    count_lines, count_dropped = count_bounded.snapshot()
    assert len(count_lines) == 200
    assert count_dropped == 5

    byte_bounded = _AudioCppDiagnosticRing()
    byte_bounded.feed("stderr", (("x" * 2_048) + "\n").encode() * 40)
    byte_bounded.finish("stderr")
    byte_lines, byte_dropped = byte_bounded.snapshot()
    assert sum(len(line.text.encode("utf-8")) for line in byte_lines) <= 65_536
    assert byte_dropped > 0


def test_diagnostics_flush_an_overlong_stream_without_waiting_for_newline() -> None:
    ring = _AudioCppDiagnosticRing()

    ring.feed("stdout", b"x" * 5_000)

    assert _snapshot_texts(ring) == ("x" * 4_096,)
    ring.finish("stdout")
    assert _snapshot_texts(ring) == ("x" * 4_096, "x" * 904)


def test_diagnostics_replacement_decode_invalid_utf8() -> None:
    ring = _AudioCppDiagnosticRing()

    ring.feed("stderr", b"before\xffafter\n")

    assert _snapshot_texts(ring) == ("before\ufffdafter",)


def test_diagnostics_remove_ansi_controls_and_escape_rich_markup() -> None:
    ring = _AudioCppDiagnosticRing()

    ring.feed(
        "stdout",
        b"\x1b[31m[bold]danger[/bold]\x1b[0m\x00\x08\x7f\n",
    )

    assert _snapshot_texts(ring) == (r"\[bold]danger\[/bold]",)


def test_diagnostics_redact_credentials_and_normalize_home_prefix(
    tmp_path: Path,
) -> None:
    home = tmp_path / "synthetic-home"
    ring = _AudioCppDiagnosticRing(home_directory=home)
    secret = "SYNTHETIC_ASSIGNMENT_SECRET"
    bearer = "SYNTHETIC_BEARER_SECRET"
    token = "SYNTHETIC_QUOTED_SECRET"

    ring.feed(
        "stderr",
        (
            f"model={home}/models/model.gguf api_key={secret} "
            f"Authorization: Bearer {bearer} token='{token}'\n"
        ).encode(),
    )
    rendered = _snapshot_texts(ring)[0]

    assert "~/models/model.gguf" in rendered
    assert str(home) not in rendered
    assert secret not in rendered
    assert bearer not in rendered
    assert token not in rendered
    assert rendered.count("<redacted>") == 3


def test_diagnostics_report_eviction_count_and_clear_per_generation() -> None:
    ring = _AudioCppDiagnosticRing()
    ring.feed("stdout", b"line\n" * 201)

    lines, dropped = ring.snapshot()
    assert len(lines) == 200
    assert dropped == 1

    ring.clear()

    assert ring.snapshot() == ((), 0)
    ring.feed("stderr", b"new generation\n")
    assert _snapshot_texts(ring) == ("new generation",)


def test_diagnostics_never_emit_to_python_or_loguru_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_output = "SYNTHETIC_PRIVATE_CHILD_OUTPUT"
    loguru_messages: list[str] = []
    caplog.set_level(logging.DEBUG)
    sink_id = logger.add(loguru_messages.append, level="DEBUG", format="{message}")
    try:
        ring = _AudioCppDiagnosticRing()
        ring.feed("stderr", f"detail={private_output}\n".encode())
        ring.finish("stderr")
    finally:
        logger.remove(sink_id)

    assert private_output in _snapshot_texts(ring)[0]
    assert private_output not in caplog.text
    assert private_output not in "".join(loguru_messages)


def test_construction_is_stopped_and_performs_no_io() -> None:
    def forbidden_clock() -> float:
        raise AssertionError("construction called the clock")

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=lambda *_args: (_ for _ in ()).throw(
            AssertionError("construction launched a process")
        ),
        port_preflight=lambda *_args: (_ for _ in ()).throw(
            AssertionError("construction probed a port")
        ),
        monotonic=forbidden_clock,
    )

    snapshot = supervisor.snapshot()
    admission = supervisor.admission_snapshot()

    assert snapshot.state == "stopped"
    assert snapshot.process_generation == 0
    assert snapshot.endpoint is None
    assert snapshot.last_failure is None
    assert admission.stage_application_eligible is True


def test_construction_survives_an_unavailable_home_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_home() -> Path:
        raise RuntimeError("synthetic home lookup failure")

    monkeypatch.setattr(supervisor_module.Path, "home", unavailable_home)

    supervisor = AudioCppSupervisor(source_environment={})

    assert supervisor.snapshot().state == "stopped"


@pytest.mark.asyncio
async def test_first_deliberate_use_starts_one_generation(tmp_path: Path) -> None:
    process = _FakeProcess()
    launcher = _FakeLauncher([process])
    hooks = _HooksFactory()
    supervisor = AudioCppSupervisor(
        source_environment={"LANG": "C"},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )

    endpoint = await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=hooks
    )

    assert endpoint.base_url == "http://127.0.0.1:19876"
    assert endpoint.process_generation == 1
    assert len(launcher.calls) == 1
    assert hooks.calls == [1]
    assert supervisor.snapshot().state == "running"
    await supervisor.stop()


@pytest.mark.asyncio
async def test_concurrent_first_use_shares_one_start_task(tmp_path: Path) -> None:
    gate = asyncio.Event()
    launcher = _FakeLauncher([_FakeProcess()], gate=gate)
    hooks = _HooksFactory()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)

    first = asyncio.create_task(
        supervisor.ensure_running(launch, generation_hooks_factory=hooks)
    )
    second = asyncio.create_task(
        supervisor.ensure_running(launch, generation_hooks_factory=hooks)
    )
    await _wait_until(lambda: len(launcher.calls) == 1)
    gate.set()

    first_result, second_result = await asyncio.gather(first, second)
    assert first_result == second_result
    assert len(launcher.calls) == 1
    assert hooks.calls == [1]
    await supervisor.stop()


@pytest.mark.asyncio
async def test_one_waiter_cancellation_does_not_cancel_shared_start(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    launcher = _FakeLauncher([_FakeProcess()], gate=gate)
    hooks = _HooksFactory()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)
    cancelled_waiter = asyncio.create_task(
        supervisor.ensure_running(launch, generation_hooks_factory=hooks)
    )
    surviving_waiter = asyncio.create_task(
        supervisor.ensure_running(launch, generation_hooks_factory=hooks)
    )
    await _wait_until(lambda: len(launcher.calls) == 1)

    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    gate.set()

    assert (await surviving_waiter).process_generation == 1
    assert len(launcher.calls) == 1
    await supervisor.stop()


@pytest.mark.asyncio
async def test_generation_hooks_factory_runs_once_only_for_new_generation(
    tmp_path: Path,
) -> None:
    launcher = _FakeLauncher([_FakeProcess()])
    first_hooks = _HooksFactory()
    replacement_hooks = _HooksFactory()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)

    first = await supervisor.ensure_running(
        launch, generation_hooks_factory=first_hooks
    )
    second = await supervisor.ensure_running(
        launch, generation_hooks_factory=replacement_hooks
    )

    assert first == second
    assert first_hooks.calls == [1]
    assert replacement_hooks.calls == []
    await supervisor.stop()


@pytest.mark.asyncio
async def test_generation_hooks_factory_failure_rolls_back_child_and_uses_safe_code(
    tmp_path: Path,
) -> None:
    process = _FakeProcess()
    launcher = _FakeLauncher([process])
    private_detail = "SYNTHETIC_PRIVATE_FACTORY_DETAIL"
    hooks = _HooksFactory(failure=RuntimeError(private_detail))
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )

    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=hooks
        )

    assert raised.value.code == "process_spawn_failed"
    assert private_detail not in str(raised.value)
    assert _exception_graph(raised.value) == [raised.value]
    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    snapshot = supervisor.snapshot()
    assert snapshot.state == "unavailable"
    assert snapshot.last_failure is not None
    assert private_detail not in snapshot.last_failure.message


@pytest.mark.asyncio
async def test_generated_artifact_is_revalidated_and_cleaned_after_exact_stop(
    tmp_path: Path,
) -> None:
    artifact = _GeneratedArtifactSpy()
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    launch = replace(_make_launch(tmp_path), generated_artifact=artifact)

    await supervisor.ensure_running(
        launch,
        generation_hooks_factory=_HooksFactory(),
    )
    await supervisor.stop()

    assert artifact.validate_calls == 2
    assert artifact.cleanup_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_snapshot_projects_only_bounded_generated_artifact_privacy_posture(
    tmp_path: Path,
) -> None:
    artifact = _GeneratedArtifactSpy()
    artifact.privacy_posture = "windows_account_protected"
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )

    await supervisor.ensure_running(
        replace(_make_launch(tmp_path), generated_artifact=artifact),
        generation_hooks_factory=_HooksFactory(),
    )

    assert (
        supervisor.snapshot().generated_artifact_privacy_posture
        == "windows_account_protected"
    )
    artifact.privacy_posture = "PRIVATE_UNKNOWN_POSTURE"
    assert supervisor.snapshot().generated_artifact_privacy_posture == "unverified"
    await supervisor.stop()
    assert supervisor.snapshot().generated_artifact_privacy_posture == "not_applicable"


@pytest.mark.asyncio
async def test_generated_artifact_runtime_handle_outlives_owned_child(
    tmp_path: Path,
) -> None:
    """Runtime leases must remain held until the owned child has exited."""
    process = _FakeProcess()

    class _OrderingArtifact(_GeneratedArtifactSpy):
        cleanup_attempts = 0

        def cleanup(self) -> None:
            self.cleanup_attempts += 1
            assert process.returncode is not None
            super().cleanup()

    artifact = _OrderingArtifact()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    launch = replace(_make_launch(tmp_path), generated_artifact=artifact)

    await supervisor.ensure_running(
        launch,
        generation_hooks_factory=_HooksFactory(),
    )
    await supervisor.stop()

    assert artifact.cleanup_attempts == artifact.cleanup_calls == 1


@pytest.mark.asyncio
async def test_generated_artifact_is_cleaned_after_pre_spawn_failure(
    tmp_path: Path,
) -> None:
    artifact = _GeneratedArtifactSpy()

    async def unavailable_port(_port: int, _timeout: float) -> str:
        return "occupied"

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher(),
        port_preflight=unavailable_port,
    )
    launch = replace(_make_launch(tmp_path), generated_artifact=artifact)

    with pytest.raises(TTSOperationError) as caught:
        await supervisor.ensure_running(
            launch,
            generation_hooks_factory=_HooksFactory(),
        )

    assert caught.value.code == "port_in_use"
    assert artifact.validate_calls == 1
    assert artifact.cleanup_calls == 1


@pytest.mark.asyncio
async def test_pre_spawn_cleanup_failure_blocks_later_launch_with_safe_phase(
    tmp_path: Path,
) -> None:
    private_detail = "SYNTHETIC_PRIVATE_PRESPAWN_CLEANUP_DETAIL"
    artifact = _GeneratedArtifactSpy(cleanup_failure=RuntimeError(private_detail))

    port_available = False

    async def unavailable_port(_port: int, _timeout: float) -> str:
        return "available" if port_available else "occupied"

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([_FakeProcess()]),
        port_preflight=unavailable_port,
    )
    launch = replace(_make_launch(tmp_path), generated_artifact=artifact)

    try:
        with pytest.raises(TTSOperationError) as initial:
            await supervisor.ensure_running(
                launch,
                generation_hooks_factory=_HooksFactory(),
            )

        assert initial.value.code == "port_in_use"
        assert _exception_graph(initial.value) == [initial.value]
        snapshot = supervisor.snapshot()
        assert snapshot.state == "unavailable"
        assert snapshot.last_failure is not None
        assert snapshot.last_failure.code == "cleanup_failed"
        assert private_detail not in snapshot.last_failure.message
        assert supervisor.admission_snapshot().stage_application_eligible is False
        assert artifact.cleanup_calls == 1

        with pytest.raises(TTSOperationError) as retried:
            await supervisor.ensure_running(
                launch,
                generation_hooks_factory=_HooksFactory(),
            )
        assert retried.value.code == "cleanup_failed"
        assert _exception_graph(retried.value) == [retried.value]

        artifact.cleanup_failure = None
        await supervisor.stop()
        assert artifact.cleanup_calls == 2
        assert supervisor.admission_snapshot().stage_application_eligible is True

        port_available = True
        replacement = replace(_make_launch(tmp_path), generated_artifact=None)
        endpoint = await supervisor.ensure_running(
            replacement,
            generation_hooks_factory=_HooksFactory(),
        )
        assert endpoint.process_generation == 1
        await supervisor.stop()
    finally:
        await asyncio.gather(supervisor.close(), return_exceptions=True)
        await asyncio.gather(supervisor.wait_closed(), return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_pre_spawn_cleanup_failure_is_retained_for_second_stop(
    tmp_path: Path,
) -> None:
    """Cancelled startup leaves one retryable pre-spawn artifact authority."""

    artifact = _GeneratedArtifactSpy(
        cleanup_failure=RuntimeError("PRIVATE_CANCELLED_PRESPAWN_CLEANUP")
    )
    spawn_started = asyncio.Event()

    async def blocked_launcher(
        _launch: AudioCppManagedLaunchConfig,
        _environment: dict[str, str],
    ) -> _OwnedAudioCppProcess:
        spawn_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=blocked_launcher,
        port_preflight=_available_preflight,
    )
    launch = replace(_make_launch(tmp_path), generated_artifact=artifact)
    start = asyncio.create_task(
        supervisor.ensure_running(
            launch,
            generation_hooks_factory=_HooksFactory(),
        )
    )
    await spawn_started.wait()

    try:
        with pytest.raises(TTSOperationError) as first_stop:
            await supervisor.stop()
        assert first_stop.value.code == "cleanup_failed"
        with pytest.raises(asyncio.CancelledError):
            await start
        assert supervisor.admission_snapshot().stage_application_eligible is False

        artifact.cleanup_failure = None
        await supervisor.stop()
        assert supervisor.admission_snapshot().stage_application_eligible is True
        assert artifact.cleanup_calls >= 2
    finally:
        await asyncio.gather(supervisor.close(), return_exceptions=True)
        await asyncio.gather(supervisor.wait_closed(), return_exceptions=True)


def test_generated_artifact_validation_does_not_swallow_control_flow(
    tmp_path: Path,
) -> None:
    class _ControlFlowSignal(BaseException):
        pass

    signal = _ControlFlowSignal()
    artifact = _GeneratedArtifactSpy(validate_failure=signal)
    supervisor = AudioCppSupervisor(source_environment={})
    launch = replace(_make_launch(tmp_path), generated_artifact=artifact)

    with pytest.raises(_ControlFlowSignal) as caught:
        supervisor._revalidate_launch(launch)

    assert caught.value is signal


def test_unexpected_artifact_validation_failure_records_only_safe_phase(
    tmp_path: Path,
) -> None:
    private_detail = "SYNTHETIC_PRIVATE_REVALIDATION_DETAIL"
    private_error_type = type(
        "SYNTHETIC_PRIVATE_REVALIDATION_TYPE",
        (Exception,),
        {},
    )
    artifact = _GeneratedArtifactSpy(
        validate_failure=private_error_type(private_detail)
    )
    supervisor = AudioCppSupervisor(source_environment={})
    launch = replace(_make_launch(tmp_path), generated_artifact=artifact)

    with pytest.raises(TTSOperationError) as caught:
        supervisor._revalidate_launch(launch)

    assert caught.value.code == "configuration_invalid"
    assert _exception_graph(caught.value) == [caught.value]
    diagnostics = supervisor.snapshot().diagnostics
    assert tuple(line.text for line in diagnostics) == (
        "Chatbook internal supervisor failure "
        "(phase=launch_revalidation, category=unexpected_exception).",
    )
    assert private_detail not in repr(diagnostics)
    assert private_error_type.__name__ not in repr(diagnostics)


@pytest.mark.asyncio
async def test_generated_artifact_cleanup_failure_is_safe_and_blocks_relaunch(
    tmp_path: Path,
) -> None:
    private_detail = "SYNTHETIC_PRIVATE_ARTIFACT_CLEANUP_DETAIL"
    artifact = _GeneratedArtifactSpy(cleanup_failure=RuntimeError(private_detail))
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    launch = replace(_make_launch(tmp_path), generated_artifact=artifact)
    hooks = _HooksFactory()
    await supervisor.ensure_running(
        launch,
        generation_hooks_factory=hooks,
    )

    try:
        with pytest.raises(TTSOperationError) as stopped:
            await supervisor.stop()

        assert stopped.value.code == "cleanup_failed"
        assert private_detail not in str(stopped.value)
        assert _exception_graph(stopped.value) == [stopped.value]
        snapshot = supervisor.snapshot()
        assert snapshot.state == "unavailable"
        assert snapshot.last_failure is not None
        assert snapshot.last_failure.code == "cleanup_failed"
        assert tuple(line.text for line in snapshot.diagnostics) == (
            "Chatbook internal supervisor failure "
            "(phase=artifact_cleanup, category=runtime_error).",
        )
        assert private_detail not in repr(snapshot.diagnostics)
        assert supervisor.admission_snapshot().stage_application_eligible is False
        assert artifact.cleanup_calls == 1
        assert process.wait_calls == 1
        retained = supervisor._generation
        assert retained is not None

        with pytest.raises(TTSOperationError) as retried:
            await supervisor.ensure_running(
                launch,
                generation_hooks_factory=_HooksFactory(),
            )
        assert retried.value.code == "cleanup_failed"
        assert _exception_graph(retried.value) == [retried.value]

        artifact.cleanup_failure = None
        await supervisor.stop()
        assert artifact.cleanup_calls == 2
        assert hooks.cleanup_calls == 1
        assert supervisor._generation is None
        assert process.wait_calls == 1
    finally:
        await asyncio.gather(supervisor.close(), return_exceptions=True)
        await asyncio.gather(supervisor.wait_closed(), return_exceptions=True)


@pytest.mark.asyncio
async def test_unexpected_generation_cleanup_failure_records_only_safe_phase(
    tmp_path: Path,
) -> None:
    private_detail = "SYNTHETIC_PRIVATE_GENERATION_CLEANUP_DETAIL"
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )

    async def hooks_factory(_generation: int) -> AudioCppGenerationHooks:
        async def contract_probe() -> str:
            return "available"

        async def health_probe() -> bool:
            return True

        async def cleanup() -> None:
            raise RuntimeError(private_detail)

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=health_probe,
            cleanup=cleanup,
        )

    await supervisor.ensure_running(
        _make_launch(tmp_path),
        generation_hooks_factory=hooks_factory,
    )

    with pytest.raises(TTSOperationError) as stopped:
        await supervisor.stop()

    assert stopped.value.code == "cleanup_failed"
    assert _exception_graph(stopped.value) == [stopped.value]
    diagnostics = supervisor.snapshot().diagnostics
    assert tuple(line.text for line in diagnostics) == (
        "Chatbook internal supervisor failure "
        "(phase=generation_cleanup, category=runtime_error).",
    )
    assert private_detail not in repr(diagnostics)


@pytest.mark.asyncio
async def test_output_failure_cleanup_task_reports_artifact_failure_via_snapshot(
    tmp_path: Path,
) -> None:
    artifact = _GeneratedArtifactSpy(
        cleanup_failure=RuntimeError("SYNTHETIC_PRIVATE_OUTPUT_CLEANUP_DETAIL")
    )
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    launch = replace(_make_launch(tmp_path), generated_artifact=artifact)
    await supervisor.ensure_running(
        launch,
        generation_hooks_factory=_HooksFactory(),
    )
    record = supervisor._generation
    assert record is not None

    try:
        process.stderr.fail(RuntimeError("synthetic output failure"))
        await _wait_until(lambda: record.output_failure_cleanup is not None)
        cleanup_task = record.output_failure_cleanup
        assert cleanup_task is not None

        await cleanup_task

        snapshot = supervisor.snapshot()
        assert snapshot.state == "unavailable"
        assert snapshot.last_failure is not None
        assert snapshot.last_failure.code == "cleanup_failed"
        assert supervisor.admission_snapshot().stage_application_eligible is False
    finally:
        await asyncio.gather(supervisor.close(), return_exceptions=True)
        await asyncio.gather(supervisor.wait_closed(), return_exceptions=True)


@pytest.mark.asyncio
async def test_new_generation_clears_prior_diagnostics(tmp_path: Path) -> None:
    first_process = _FakeProcess()
    second_process = _FakeProcess()
    launcher = _FakeLauncher([first_process, second_process])
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    first_process.stderr.feed(b"synthetic prior detail\n")
    await _wait_until(lambda: bool(supervisor.snapshot().diagnostics))
    first_process.exit(7)
    await _wait_until(lambda: supervisor.snapshot().state == "unavailable")

    await supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())

    assert supervisor.snapshot().diagnostics == ()
    await supervisor.stop()


@pytest.mark.asyncio
async def test_starting_and_stopping_are_never_stage_application_eligible(
    tmp_path: Path,
) -> None:
    launch_gate = asyncio.Event()
    process = _FakeProcess(exit_on_terminate=False)
    launcher = _FakeLauncher([process], gate=launch_gate)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    start = asyncio.create_task(
        supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
        )
    )
    await _wait_until(lambda: supervisor.snapshot().state == "starting")
    assert supervisor.admission_snapshot().stage_application_eligible is False
    launch_gate.set()
    await start

    stop = asyncio.create_task(supervisor.stop())
    await _wait_until(lambda: supervisor.snapshot().state == "stopping")
    assert supervisor.admission_snapshot().stage_application_eligible is False
    process.exit(0)
    await stop


@pytest.mark.asyncio
async def test_process_snapshot_retains_only_fixed_safe_last_failure(
    tmp_path: Path,
) -> None:
    async def ambiguous_preflight(_port: int, _timeout: float) -> str:
        return "ambiguous"

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher(),
        port_preflight=ambiguous_preflight,
    )
    private_path = str(tmp_path)

    with pytest.raises(TTSOperationError):
        await supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
        )

    failure = supervisor.snapshot().last_failure
    assert failure is not None
    assert failure.code == "port_in_use"
    assert failure.retryable is True
    assert failure.recovery_action == "open_settings"
    assert private_path not in failure.message


@pytest.mark.asyncio
async def test_successful_new_generation_clears_prior_failure(tmp_path: Path) -> None:
    private_detail = "SYNTHETIC_SPAWN_PRIVATE_DETAIL"
    launcher = _FakeLauncher([_FakeProcess()], failure=OSError(private_detail))
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)
    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            launch, generation_hooks_factory=_HooksFactory()
        )
    assert _exception_graph(raised.value) == [raised.value]
    assert supervisor.snapshot().last_failure is not None

    await supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())

    assert supervisor.snapshot().last_failure is None
    await supervisor.stop()


@pytest.mark.asyncio
async def test_occupied_port_fails_closed_without_spawn_or_adoption(
    tmp_path: Path,
) -> None:
    async def occupied_preflight(_port: int, _timeout: float) -> str:
        return "occupied"

    launcher = _FakeLauncher()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=occupied_preflight,
    )

    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
        )

    assert raised.value.code == "port_in_use"
    assert launcher.calls == []
    assert supervisor.snapshot().process_generation == 0


@pytest.mark.asyncio
async def test_spawn_uses_exact_argv_cwd_stdin_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    captured: dict[str, Any] = {}
    transport_closes = 0

    class Transport:
        def close(self) -> None:
            nonlocal transport_closes
            transport_closes += 1

    process._transport = Transport()

    async def create_subprocess_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        captured["argv"] = argv
        captured.update(kwargs)
        return process

    monkeypatch.setattr(
        supervisor_module.asyncio,
        "create_subprocess_exec",
        create_subprocess_exec,
    )
    launch = _make_launch(tmp_path)
    supervisor = AudioCppSupervisor(
        source_environment={
            "LANG": "C",
            "OPENAI_API_KEY": "SYNTHETIC_SECRET",
            "NOT_ALLOWED": "private",
        },
        provider_credential_names=frozenset({"OPENAI_API_KEY"}),
        port_preflight=_available_preflight,
    )

    await supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())

    assert captured == {
        "argv": (str(launch.binary_path), "--config", str(launch.server_json_path)),
        "cwd": str(launch.working_directory),
        "env": {"LANG": "C"},
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
    }
    await supervisor.stop()
    assert transport_closes == 1


@pytest.mark.asyncio
async def test_cancelled_spawn_settles_late_process_and_native_transport(
    tmp_path: Path,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    process = _FakeProcess()
    transport_closes = 0

    def close_transport() -> None:
        nonlocal transport_closes
        transport_closes += 1

    async def cancellation_resistant_launcher(
        _launch: AudioCppManagedLaunchConfig,
        _environment: dict[str, str],
    ) -> _OwnedAudioCppProcess:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return _OwnedAudioCppProcess(
            process=process,
            close_parent_pipes=process.close_parent_pipes,
            close_native_transport=close_transport,
        )

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=cancellation_resistant_launcher,
        port_preflight=_available_preflight,
    )
    start = asyncio.create_task(
        supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
        )
    )
    await entered.wait()
    stopping = asyncio.create_task(supervisor.stop())
    await asyncio.sleep(0)
    assert not stopping.done()
    release.set()

    await stopping
    await asyncio.gather(start, return_exceptions=True)

    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    assert transport_closes == 1
    assert supervisor._generation is None
    await supervisor.close()


@pytest.mark.asyncio
async def test_process_launcher_settlement_retains_late_owner_after_cancellation() -> (
    None
):
    entered = asyncio.Event()
    release = asyncio.Event()
    owned = _OwnedAudioCppProcess(
        process=_FakeProcess(),
        close_parent_pipes=lambda: None,
    )

    async def cancellation_resistant_launcher() -> _OwnedAudioCppProcess:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return owned

    settlement = asyncio.create_task(
        supervisor_module._settle_process_launcher(
            cancellation_resistant_launcher(),
            timeout=10.0,
        )
    )
    await entered.wait()
    settlement.cancel()
    await asyncio.sleep(0)
    completed_before_owner = settlement.done()
    release.set()

    cancellation, timed_out, settled_owner, error = await settlement

    assert completed_before_owner is False
    assert cancellation is not None
    assert timed_out is False
    assert settled_owner is owned
    assert error is None


@pytest.mark.asyncio
async def test_native_transport_closes_after_wait_hooks_and_artifact_then_retries(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    process = _FakeProcess()
    real_wait = process.wait

    async def ordered_wait() -> int:
        result = await real_wait()
        events.append("wait")
        return result

    process.wait = ordered_wait  # type: ignore[method-assign]
    artifact = _GeneratedArtifactSpy()
    real_artifact_cleanup = artifact.cleanup

    def ordered_artifact_cleanup() -> None:
        events.append("artifact")
        real_artifact_cleanup()

    artifact.cleanup = ordered_artifact_cleanup  # type: ignore[method-assign]
    transport_attempts = 0

    def close_transport() -> None:
        nonlocal transport_attempts
        transport_attempts += 1
        events.append("transport")
        if transport_attempts == 1:
            raise RuntimeError("PRIVATE TRANSPORT DETAIL")

    async def launcher(
        _launch: AudioCppManagedLaunchConfig,
        _environment: dict[str, str],
    ) -> _OwnedAudioCppProcess:
        return _OwnedAudioCppProcess(
            process=process,
            close_parent_pipes=process.close_parent_pipes,
            close_native_transport=close_transport,
        )

    class OrderedHooks(_HooksFactory):
        async def __call__(self, generation: int) -> AudioCppGenerationHooks:
            hooks = await super().__call__(generation)

            async def cleanup() -> None:
                events.append("hooks")

            return AudioCppGenerationHooks(
                contract_probe=hooks.contract_probe,
                health_probe=hooks.health_probe,
                cleanup=cleanup,
            )

    launch = replace(_make_launch(tmp_path), generated_artifact=artifact)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    await supervisor.ensure_running(launch, generation_hooks_factory=OrderedHooks())

    with pytest.raises(TTSOperationError) as first:
        await supervisor.stop()

    assert first.value.code == "cleanup_failed"
    assert events.index("wait") < events.index("hooks")
    assert events.index("hooks") < events.index("artifact")
    assert events.index("artifact") < events.index("transport")
    assert supervisor._generation is not None
    await supervisor.stop()
    assert transport_attempts == 2
    assert supervisor._generation is None


@pytest.mark.asyncio
async def test_default_launcher_suppresses_paths_in_asyncio_debug_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    launch = _make_launch(tmp_path)
    loop = asyncio.get_running_loop()
    prior_debug = loop.get_debug()
    caplog.set_level(logging.DEBUG, logger="asyncio")
    asyncio_logger = logging.getLogger("asyncio")

    try:
        loop.set_debug(True)
        asyncio_logger.debug("UNRELATED_ASYNCIO_BEFORE")
        owned = await supervisor_module._default_process_launcher(launch, {})
        await asyncio.wait_for(owned.process.wait(), timeout=1)
        owned.close_parent_pipes()
        owned.close_native_transport()
        asyncio_logger.debug("UNRELATED_ASYNCIO_AFTER")
    finally:
        loop.set_debug(prior_debug)

    assert str(launch.binary_path) not in caplog.text
    assert str(launch.server_json_path) not in caplog.text
    assert "UNRELATED_ASYNCIO_BEFORE" in caplog.text
    assert "UNRELATED_ASYNCIO_AFTER" in caplog.text


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows subprocesses")
@pytest.mark.asyncio
async def test_native_windows_transport_close_invalidates_only_exact_child_handle() -> (
    None
):
    import _winapi

    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    sibling = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(10)",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    transport = child._transport
    native_process = transport.get_extra_info("subprocess")
    native_handle = native_process._handle
    close_transport = supervisor_module._process_native_transport_closer(child)
    try:
        assert await child.wait() == 0
        _winapi.GetExitCodeProcess(native_handle)
        close_transport()
        close_transport()
        with pytest.raises(OSError):
            _winapi.GetExitCodeProcess(native_handle)
        assert sibling.returncode is None
    finally:
        if sibling.returncode is None:
            sibling.terminate()
        await sibling.wait()


@pytest.mark.asyncio
async def test_early_exit_rolls_back_monitor_and_both_drains(tmp_path: Path) -> None:
    process = _FakeProcess()
    process.exit(9)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )

    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
        )

    assert raised.value.code == "process_exited"
    assert process.wait_calls == 1
    assert process.stdout.read_calls > 0
    assert process.stderr.read_calls > 0
    assert process.close_parent_calls == 1
    assert supervisor.snapshot().state == "unavailable"


@pytest.mark.asyncio
async def test_startup_timeout_kills_exact_child_and_joins_generation_tasks(
    tmp_path: Path,
) -> None:
    now = 0.0

    def monotonic() -> float:
        return now

    async def advancing_sleep(delay: float) -> None:
        nonlocal now
        now += delay
        await asyncio.sleep(0)

    process = _FakeProcess(exit_on_terminate=False)
    hooks = _HooksFactory(health=False)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
        monotonic=monotonic,
        sleep=advancing_sleep,
    )

    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            _make_launch(tmp_path, startup_timeout_seconds=1.0),
            generation_hooks_factory=hooks,
        )

    assert raised.value.code == "process_startup_timeout"
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 1
    assert hooks.cleanup_calls == 1
    assert supervisor.snapshot().state == "unavailable"


@pytest.mark.asyncio
async def test_contract_failure_rolls_back_before_running(tmp_path: Path) -> None:
    process = _FakeProcess()
    contract_failure = TTSOperationError(
        code="contract_incompatible",
        message="The audio.cpp server contract is incompatible",
        retryable=False,
        operation_id="synthetic-contract",
        recovery_action="open_settings",
    )
    hooks = _HooksFactory()

    async def factory(generation: int) -> AudioCppGenerationHooks:
        base = await hooks(generation)

        async def contract_probe() -> str:
            raise contract_failure

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=base.health_probe,
            cleanup=base.cleanup,
        )

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )

    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=factory
        )

    assert raised.value.code == "contract_incompatible"
    assert process.terminate_calls == 1
    assert hooks.cleanup_calls == 1
    assert supervisor.snapshot().state == "unavailable"


@pytest.mark.asyncio
async def test_contract_deadline_failure_uses_startup_timeout_code(
    tmp_path: Path,
) -> None:
    process = _FakeProcess()

    async def factory(_generation: int) -> AudioCppGenerationHooks:
        async def contract_probe() -> str:
            raise asyncio.TimeoutError

        async def health_probe() -> bool:
            return True

        async def cleanup() -> None:
            return None

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=health_probe,
            cleanup=cleanup,
        )

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )

    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=factory
        )

    assert raised.value.code == "process_startup_timeout"
    assert process.terminate_calls == 1


@pytest.mark.asyncio
async def test_launch_files_changing_after_preflight_fail_before_spawn(
    tmp_path: Path,
) -> None:
    launch = _make_launch(tmp_path)

    async def mutating_preflight(_port: int, _timeout: float) -> str:
        launch.server_json_path.write_text(
            json.dumps({"host": "127.0.0.1", "port": 19_877}),
            encoding="utf-8",
        )
        return "available"

    launcher = _FakeLauncher()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=mutating_preflight,
    )

    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            launch, generation_hooks_factory=_HooksFactory()
        )

    assert raised.value.code == "configuration_invalid"
    assert launcher.calls == []


@pytest.mark.asyncio
async def test_zero_tts_models_reaches_running_not_configured(tmp_path: Path) -> None:
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([_FakeProcess()]),
        port_preflight=_available_preflight,
    )

    endpoint = await supervisor.ensure_running(
        _make_launch(tmp_path),
        generation_hooks_factory=_HooksFactory(contract="not_configured"),
    )

    assert endpoint.process_generation == 1
    snapshot = supervisor.snapshot()
    assert snapshot.state == "running"
    assert snapshot.tts_capability == "not_configured"
    await supervisor.stop()


@pytest.mark.asyncio
async def test_stale_generation_cannot_publish_running(tmp_path: Path) -> None:
    contract_gate = asyncio.Event()
    process = _FakeProcess()
    hooks = _HooksFactory()

    async def factory(generation: int) -> AudioCppGenerationHooks:
        base = await hooks(generation)

        async def contract_probe() -> str:
            await contract_gate.wait()
            return "available"

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=base.health_probe,
            cleanup=base.cleanup,
        )

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    start = asyncio.create_task(
        supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=factory
        )
    )
    await _wait_until(lambda: hooks.health_calls == 1)

    await supervisor.stop()
    contract_gate.set()

    with pytest.raises(asyncio.CancelledError):
        await start
    assert supervisor.snapshot().state == "stopped"


@pytest.mark.asyncio
async def test_exit_during_contract_probe_cancels_startup_promptly(
    tmp_path: Path,
) -> None:
    process = _FakeProcess()
    contract_started = asyncio.Event()

    async def factory(_generation: int) -> AudioCppGenerationHooks:
        async def contract_probe() -> str:
            contract_started.set()
            await asyncio.Event().wait()
            return "available"

        async def health_probe() -> bool:
            return True

        async def cleanup() -> None:
            return None

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=health_probe,
            cleanup=cleanup,
        )

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    start = asyncio.create_task(
        supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=factory
        )
    )
    await contract_started.wait()

    process.exit(8)
    for _ in range(20):
        if start.done():
            break
        await asyncio.sleep(0)

    assert start.done(), "process exit left startup waiting on the contract probe"
    with pytest.raises(TTSOperationError) as raised:
        await start
    assert raised.value.code == "process_exited"
    assert supervisor.snapshot().state == "unavailable"


@pytest.mark.asyncio
async def test_exit_at_successful_contract_completion_is_process_exited(
    tmp_path: Path,
) -> None:
    process = _FakeProcess()

    async def factory(_generation: int) -> AudioCppGenerationHooks:
        async def contract_probe() -> str:
            process.publish_returncode(8)
            asyncio.get_running_loop().call_soon(process.complete_exit)
            return "available"

        async def health_probe() -> bool:
            return True

        async def cleanup() -> None:
            return None

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=health_probe,
            cleanup=cleanup,
        )

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )

    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=factory
        )

    assert raised.value.code == "process_exited"
    assert supervisor.snapshot().state == "unavailable"


@pytest.mark.asyncio
async def test_output_failure_at_contract_completion_is_process_exited(
    tmp_path: Path,
) -> None:
    process = _FakeProcess(exit_on_terminate=False)
    supervisor: AudioCppSupervisor

    async def factory(_generation: int) -> AudioCppGenerationHooks:
        async def contract_probe() -> str:
            process.stderr.fail(RuntimeError("private output failure"))
            await _wait_until(lambda: supervisor.snapshot().state == "unavailable")
            return "available"

        async def health_probe() -> bool:
            return True

        async def cleanup() -> None:
            return None

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=health_probe,
            cleanup=cleanup,
        )

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )

    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            _make_launch(tmp_path, termination_grace_seconds=0.1),
            generation_hooks_factory=factory,
        )

    assert raised.value.code == "process_exited"
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert supervisor.snapshot().state == "unavailable"


@pytest.mark.asyncio
async def test_draining_rejects_new_deliberate_use(tmp_path: Path) -> None:
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([_FakeProcess()]),
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    await supervisor.begin_draining()

    with pytest.raises(TTSProviderReconfiguringError):
        await supervisor.ensure_running(
            launch, generation_hooks_factory=_HooksFactory()
        )

    await supervisor.stop()


@pytest.mark.asyncio
async def test_require_existing_generation_refuses_to_spawn_after_concurrent_exit(
    tmp_path: Path,
) -> None:
    process = _FakeProcess()
    launcher = _FakeLauncher([process, _FakeProcess()])
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    admission = supervisor.admission_snapshot()
    process.exit(5)
    await _wait_until(lambda: supervisor.snapshot().state == "unavailable")

    with pytest.raises(_AudioCppGenerationChanged):
        await supervisor.ensure_running(
            launch,
            generation_hooks_factory=_HooksFactory(),
            require_existing=admission,
        )

    assert len(launcher.calls) == 1


@pytest.mark.asyncio
async def test_stop_during_pre_spawn_startup_invalidates_and_joins_it(
    tmp_path: Path,
) -> None:
    process = _FakeProcess()
    spawn_started = asyncio.Event()
    cancellation_received = asyncio.Event()
    release_after_cancellation = asyncio.Event()

    async def cancellation_resistant_launcher(
        _launch: AudioCppManagedLaunchConfig,
        _environment: dict[str, str],
    ) -> _OwnedAudioCppProcess:
        spawn_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_received.set()
            await release_after_cancellation.wait()
        return _OwnedAudioCppProcess(
            process=process,
            close_parent_pipes=process.close_parent_pipes,
        )

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=cancellation_resistant_launcher,
        port_preflight=_available_preflight,
    )
    start = asyncio.create_task(
        supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
        )
    )
    await spawn_started.wait()

    stop = asyncio.create_task(supervisor.stop())
    await cancellation_received.wait()
    release_after_cancellation.set()
    await stop

    with pytest.raises(asyncio.CancelledError):
        await start
    assert supervisor.snapshot().state == "stopped"
    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    assert process.close_parent_calls == 1


@pytest.mark.asyncio
async def test_stop_during_post_spawn_startup_rolls_back_exact_child(
    tmp_path: Path,
) -> None:
    hooks_gate = asyncio.Event()
    process = _FakeProcess()
    hooks = _HooksFactory(gate=hooks_gate)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    start = asyncio.create_task(
        supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=hooks
        )
    )
    await _wait_until(lambda: hooks.calls == [1])

    await supervisor.stop()

    with pytest.raises(asyncio.CancelledError):
        await start
    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    assert supervisor.snapshot().state == "stopped"


@pytest.mark.asyncio
async def test_periodic_health_probes_never_overlap(tmp_path: Path) -> None:
    sleep = _ManualSleep()
    hooks = _ControlledHooksFactory()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([_FakeProcess()]),
        port_preflight=_available_preflight,
        sleep=sleep,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=hooks
    )
    gate = asyncio.get_running_loop().create_future()

    await _run_periodic_probe(sleep, hooks, gate)
    await asyncio.sleep(0)

    assert hooks.health_calls == 2
    assert hooks.max_active_health_calls == 1
    gate.set_result(True)
    await _wait_until(lambda: hooks.active_health_calls == 0)
    await supervisor.stop()


@pytest.mark.asyncio
async def test_periodic_and_immediate_health_probes_share_one_inflight_probe(
    tmp_path: Path,
) -> None:
    sleep = _ManualSleep()
    hooks = _ControlledHooksFactory()
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
        sleep=sleep,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=hooks)
    await _run_periodic_probe(sleep, hooks, False)
    await _run_periodic_probe(sleep, hooks, False)
    assert supervisor.snapshot().state == "unhealthy"
    gate = asyncio.get_running_loop().create_future()
    await _run_periodic_probe(sleep, hooks, gate)

    request = asyncio.create_task(
        supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    )
    await asyncio.sleep(0)

    assert hooks.health_calls == 4
    gate.set_result(True)
    assert (await request).process_generation == 1
    assert hooks.health_calls == 4
    await supervisor.stop()


@pytest.mark.asyncio
async def test_concurrent_unhealthy_requests_share_one_immediate_probe(
    tmp_path: Path,
) -> None:
    sleep = _ManualSleep()
    hooks = _ControlledHooksFactory()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([_FakeProcess()]),
        port_preflight=_available_preflight,
        sleep=sleep,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=hooks)
    await _run_periodic_probe(sleep, hooks, False)
    await _run_periodic_probe(sleep, hooks, False)
    gate = asyncio.get_running_loop().create_future()
    hooks.queue_health(gate)

    first = asyncio.create_task(
        supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    )
    second = asyncio.create_task(
        supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    )
    await _wait_until(lambda: hooks.health_calls == 4)

    assert hooks.active_health_calls == 1
    gate.set_result(True)
    first_result, second_result = await asyncio.gather(first, second)
    assert first_result == second_result
    assert hooks.health_calls == 4
    await supervisor.stop()


@pytest.mark.asyncio
async def test_two_failures_mark_unhealthy_and_one_success_recovers(
    tmp_path: Path,
) -> None:
    sleep = _ManualSleep()
    hooks = _ControlledHooksFactory()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([_FakeProcess()]),
        port_preflight=_available_preflight,
        sleep=sleep,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=hooks)
    running_version = supervisor.snapshot().observation_version

    await _run_periodic_probe(sleep, hooks, False)
    first_failure = supervisor.snapshot()
    assert first_failure.state == "running"
    assert first_failure.consecutive_health_failures == 1
    assert first_failure.observation_version > running_version
    await _run_periodic_probe(sleep, hooks, False)
    unhealthy = supervisor.snapshot()
    assert unhealthy.state == "unhealthy"
    assert unhealthy.observation_version > first_failure.observation_version
    hooks.queue_health(True)

    await supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())

    snapshot = supervisor.snapshot()
    assert snapshot.state == "running"
    assert snapshot.consecutive_health_failures == 0
    assert snapshot.last_failure is None
    assert snapshot.observation_version > unhealthy.observation_version
    await supervisor.stop()


@pytest.mark.asyncio
async def test_successful_recovery_probe_cannot_publish_a_dead_process(
    tmp_path: Path,
) -> None:
    sleep = _ManualSleep()
    hooks = _ControlledHooksFactory()
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
        sleep=sleep,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=hooks)
    await _run_periodic_probe(sleep, hooks, False)
    await _run_periodic_probe(sleep, hooks, False)
    assert supervisor.snapshot().state == "unhealthy"
    recovery = asyncio.get_running_loop().create_future()
    hooks.queue_health(recovery)

    request = asyncio.create_task(
        supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    )
    await _wait_until(lambda: hooks.health_calls == 4)
    process.publish_returncode(8)
    recovery.set_result(True)

    try:
        with pytest.raises(TTSOperationError) as raised:
            await request
        assert raised.value.code == "process_exited"
        snapshot = supervisor.snapshot()
        assert snapshot.state == "unavailable"
        assert snapshot.endpoint is None
    finally:
        process.complete_exit()
        await supervisor.close()


@pytest.mark.asyncio
async def test_running_fast_path_cannot_return_a_dead_process(
    tmp_path: Path,
) -> None:
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    process.publish_returncode(8)

    try:
        with pytest.raises(TTSOperationError) as raised:
            await supervisor.ensure_running(
                launch, generation_hooks_factory=_HooksFactory()
            )
        assert raised.value.code == "process_exited"
        snapshot = supervisor.snapshot()
        assert snapshot.state == "unavailable"
        assert snapshot.endpoint is None
    finally:
        process.complete_exit()
        await supervisor.close()


@pytest.mark.asyncio
async def test_request_probe_failure_does_not_kill_unhealthy_child(
    tmp_path: Path,
) -> None:
    sleep = _ManualSleep()
    hooks = _ControlledHooksFactory()
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
        sleep=sleep,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=hooks)
    await _run_periodic_probe(sleep, hooks, False)
    await _run_periodic_probe(sleep, hooks, False)
    hooks.queue_health(False)

    with pytest.raises(TTSOperationError) as raised:
        await supervisor.ensure_running(
            launch, generation_hooks_factory=_HooksFactory()
        )

    assert raised.value.code == "runtime_unhealthy"
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert supervisor.snapshot().state == "unhealthy"
    await supervisor.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_result", [False, True])
async def test_inflight_health_result_cannot_overwrite_draining(
    tmp_path: Path,
    probe_result: bool,
) -> None:
    sleep = _ManualSleep()
    hooks = _ControlledHooksFactory()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([_FakeProcess()]),
        port_preflight=_available_preflight,
        sleep=sleep,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path),
        generation_hooks_factory=hooks,
    )
    await _run_periodic_probe(sleep, hooks, False)
    blocked_result = asyncio.get_running_loop().create_future()
    await _run_periodic_probe(sleep, hooks, blocked_result)

    await supervisor.begin_draining()
    blocked_result.set_result(probe_result)
    await _wait_until(lambda: hooks.active_health_calls == 0)

    snapshot = supervisor.snapshot()
    assert snapshot.state == "draining"
    assert snapshot.consecutive_health_failures == 1
    await supervisor.stop()


@pytest.mark.asyncio
async def test_unexpected_exit_invalidates_generation_without_restart(
    tmp_path: Path,
) -> None:
    process = _FakeProcess()
    launcher = _FakeLauncher([process, _FakeProcess()])
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
    )

    process.exit(12)
    await _wait_until(lambda: supervisor.snapshot().state == "unavailable")

    snapshot = supervisor.snapshot()
    assert snapshot.endpoint is None
    assert snapshot.last_failure is not None
    assert snapshot.last_failure.code == "process_exited"
    assert len(launcher.calls) == 1


@pytest.mark.asyncio
async def test_unexpected_exit_is_public_before_generation_cleanup_finishes(
    tmp_path: Path,
) -> None:
    process = _FakeProcess()
    replacement = _FakeProcess()
    launcher = _FakeLauncher([process, replacement])
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    hooks = _HooksFactory()

    async def blocking_cleanup_factory(generation: int) -> AudioCppGenerationHooks:
        base = await hooks(generation)

        async def cleanup() -> None:
            cleanup_started.set()
            await release_cleanup.wait()
            await base.cleanup()

        return AudioCppGenerationHooks(
            contract_probe=base.contract_probe,
            health_probe=base.health_probe,
            cleanup=cleanup,
        )

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(
        launch,
        generation_hooks_factory=blocking_cleanup_factory,
    )

    process.exit(12)
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    snapshot = supervisor.snapshot()
    admission = supervisor.admission_snapshot()
    replacement_start = asyncio.create_task(
        supervisor.ensure_running(
            launch,
            generation_hooks_factory=_HooksFactory(),
        )
    )

    try:
        assert snapshot.state == "unavailable"
        assert snapshot.endpoint is None
        assert snapshot.tts_capability == "unknown"
        assert snapshot.last_failure is not None
        assert snapshot.last_failure.code == "process_exited"
        assert admission.stage_application_eligible is False
        await asyncio.sleep(0)
        assert replacement_start.done() is False

        release_cleanup.set()
        endpoint = await asyncio.wait_for(replacement_start, timeout=1)
        assert endpoint.process_generation == 2
        assert len(launcher.calls) == 2
    finally:
        release_cleanup.set()
        await asyncio.gather(replacement_start, return_exceptions=True)
        await supervisor.stop()


@pytest.mark.asyncio
async def test_later_deliberate_use_starts_one_replacement(tmp_path: Path) -> None:
    first = _FakeProcess()
    replacement = _FakeProcess()
    launcher = _FakeLauncher([first, replacement])
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)
    first_endpoint = await supervisor.ensure_running(
        launch, generation_hooks_factory=_HooksFactory()
    )
    first.exit(3)
    await _wait_until(lambda: supervisor.snapshot().state == "unavailable")

    replacement_endpoint = await supervisor.ensure_running(
        launch, generation_hooks_factory=_HooksFactory()
    )

    assert first_endpoint.process_generation == 1
    assert replacement_endpoint.process_generation == 2
    assert len(launcher.calls) == 2
    await supervisor.stop()


@pytest.mark.asyncio
async def test_expected_stop_is_not_reported_as_a_crash(tmp_path: Path) -> None:
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
    )

    await supervisor.stop()

    snapshot = supervisor.snapshot()
    assert snapshot.state == "stopped"
    assert snapshot.last_failure is None


@pytest.mark.asyncio
async def test_health_probe_waiter_cancellation_does_not_cancel_shared_probe(
    tmp_path: Path,
) -> None:
    sleep = _ManualSleep()
    hooks = _ControlledHooksFactory()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([_FakeProcess()]),
        port_preflight=_available_preflight,
        sleep=sleep,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=hooks)
    await _run_periodic_probe(sleep, hooks, False)
    await _run_periodic_probe(sleep, hooks, False)
    gate = asyncio.get_running_loop().create_future()
    hooks.queue_health(gate)
    cancelled = asyncio.create_task(
        supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    )
    survivor = asyncio.create_task(
        supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    )
    await _wait_until(lambda: hooks.health_calls == 4)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert not gate.cancelled()
    gate.set_result(True)

    assert (await survivor).process_generation == 1
    assert hooks.health_calls == 4
    await supervisor.stop()


@pytest.mark.asyncio
async def test_stop_terminates_then_kills_only_the_owned_child(
    tmp_path: Path,
) -> None:
    now = 100.0
    process = _FakeProcess(exit_on_terminate=False)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
        monotonic=lambda: now,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
    )

    with shutdown_deadline_scope(now):
        await supervisor.stop(application_shutdown=True)

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_stop_waiter_cancellation_does_not_abandon_cleanup(
    tmp_path: Path,
) -> None:
    process = _FakeProcess(exit_on_terminate=False)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
    )
    waiter = asyncio.create_task(supervisor.stop())
    await _wait_until(lambda: process.terminate_calls == 1)

    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()
    process.exit(0)

    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert supervisor.snapshot().state == "stopped"
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_expected_generation_stop_is_noop_for_replacement(
    tmp_path: Path,
) -> None:
    first = _FakeProcess()
    replacement = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([first, replacement]),
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)
    first_endpoint = await supervisor.ensure_running(
        launch, generation_hooks_factory=_HooksFactory()
    )
    original_stop_impl = supervisor._stop_impl
    stop_impl_entered = asyncio.Event()
    release_stop_impl = asyncio.Event()

    async def delayed_stop_impl(**kwargs: Any) -> None:
        stop_impl_entered.set()
        await release_stop_impl.wait()
        await original_stop_impl(**kwargs)

    supervisor._stop_impl = delayed_stop_impl  # type: ignore[method-assign]
    retiring_stop = asyncio.create_task(
        supervisor.stop(expected_process_generation=first_endpoint.process_generation)
    )
    await stop_impl_entered.wait()
    first.exit(0)
    await _wait_until(lambda: supervisor.snapshot().state == "unavailable")
    replacement_endpoint = await supervisor.ensure_running(
        launch, generation_hooks_factory=_HooksFactory()
    )

    release_stop_impl.set()
    await retiring_stop

    assert replacement_endpoint.process_generation == 2
    assert supervisor.snapshot().state == "running"
    assert replacement.terminate_calls == 0
    await supervisor.stop()


@pytest.mark.asyncio
async def test_stop_joins_inflight_probe_before_generation_cleanup(
    tmp_path: Path,
) -> None:
    sleep = _ManualSleep()
    hooks = _ControlledHooksFactory()
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
        sleep=sleep,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=hooks
    )
    gate = asyncio.get_running_loop().create_future()
    await _run_periodic_probe(sleep, hooks, gate)

    await supervisor.stop()

    assert gate.cancelled()
    assert hooks.active_health_calls == 0
    assert hooks.cleanup_calls == 1


@pytest.mark.asyncio
async def test_application_deadline_caps_termination_grace(tmp_path: Path) -> None:
    now = 200.0
    process = _FakeProcess(exit_on_terminate=False)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
        monotonic=lambda: now,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path, termination_grace_seconds=60.0),
        generation_hooks_factory=_HooksFactory(),
    )

    with shutdown_deadline_scope(now):
        await supervisor.stop(application_shutdown=True)

    assert process.kill_calls == 1
    assert supervisor.snapshot().state == "stopped"


@pytest.mark.asyncio
async def test_application_deadline_caps_post_spawn_startup_rollback(
    tmp_path: Path,
) -> None:
    now = 300.0
    process = _FakeProcess(exit_on_terminate=False)
    hooks_gate = asyncio.Event()
    hooks = _HooksFactory(gate=hooks_gate)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
        monotonic=lambda: now,
    )
    start = asyncio.create_task(
        supervisor.ensure_running(
            _make_launch(tmp_path, termination_grace_seconds=60.0),
            generation_hooks_factory=hooks,
        )
    )
    await _wait_until(lambda: hooks.calls == [1])

    with shutdown_deadline_scope(now):
        stop = asyncio.create_task(supervisor.stop(application_shutdown=True))
    for _ in range(20):
        if process.kill_calls:
            break
        await asyncio.sleep(0)
    deadline_was_applied = process.kill_calls == 1
    if not deadline_was_applied:
        process.exit(0)
    await stop
    with pytest.raises(asyncio.CancelledError):
        await start

    assert deadline_was_applied
    assert supervisor.snapshot().state == "stopped"


@pytest.mark.asyncio
async def test_close_and_wait_closed_are_idempotent(tmp_path: Path) -> None:
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
    )

    await asyncio.gather(supervisor.close(), supervisor.close())
    await asyncio.gather(supervisor.wait_closed(), supervisor.wait_closed())

    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    assert supervisor.snapshot().state == "stopped"


@pytest.mark.asyncio
async def test_terminal_close_leaves_no_child_or_task_reference(
    tmp_path: Path,
) -> None:
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([_FakeProcess()]),
        port_preflight=_available_preflight,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
    )

    await supervisor.close()
    await supervisor.wait_closed()

    assert supervisor._generation is None
    assert supervisor._startup_task is None
    assert supervisor._stop_task is None
    assert supervisor._close_task is None
    assert supervisor.snapshot().diagnostics == ()
    assert supervisor.snapshot().last_failure is None


@pytest.mark.asyncio
async def test_exit_monitor_is_the_only_process_wait_owner(tmp_path: Path) -> None:
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
    )

    await supervisor.stop()

    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_stale_exit_monitor_cannot_reap_or_mutate_replacement(
    tmp_path: Path,
) -> None:
    first = _FakeProcess()
    replacement = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([first, replacement]),
        port_preflight=_available_preflight,
    )
    launch = _make_launch(tmp_path)
    await supervisor.ensure_running(launch, generation_hooks_factory=_HooksFactory())
    first.exit(4)
    await _wait_until(lambda: supervisor.snapshot().state == "unavailable")
    endpoint = await supervisor.ensure_running(
        launch, generation_hooks_factory=_HooksFactory()
    )

    for _ in range(10):
        await asyncio.sleep(0)

    assert endpoint.process_generation == 2
    assert supervisor.snapshot().process_generation == 2
    assert supervisor.snapshot().state == "running"
    assert replacement.wait_calls == 1
    await supervisor.stop()


@pytest.mark.asyncio
async def test_output_drain_failure_stops_child_and_records_safe_failure(
    tmp_path: Path,
) -> None:
    private_detail = "SYNTHETIC_PRIVATE_DRAIN_DETAIL"
    process = _FakeProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path), generation_hooks_factory=_HooksFactory()
    )

    process.stderr.fail(RuntimeError(private_detail))
    await _wait_until(
        lambda: (
            supervisor.snapshot().state == "unavailable"
            and process.terminate_calls == 1
        )
    )

    failure = supervisor.snapshot().last_failure
    assert failure is not None
    assert failure.code == "process_exited"
    assert private_detail not in failure.message
    assert process.terminate_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_output_drain_failure_seals_admission_before_stubborn_child_exits(
    tmp_path: Path,
) -> None:
    process = _FakeProcess(exit_on_terminate=False)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
    )
    await supervisor.ensure_running(
        _make_launch(tmp_path),
        generation_hooks_factory=_HooksFactory(),
    )
    running_admission = supervisor.admission_snapshot()

    process.stderr.fail(RuntimeError("private drain failure"))
    await _wait_until(lambda: process.terminate_calls == 1)

    snapshot = supervisor.snapshot()
    failed_admission = supervisor.admission_snapshot()
    assert snapshot.state == "unavailable"
    assert snapshot.endpoint is None
    assert snapshot.tts_capability == "unknown"
    assert snapshot.last_failure is not None
    assert snapshot.last_failure.code == "process_exited"
    assert failed_admission.lifecycle_epoch > running_admission.lifecycle_epoch
    assert failed_admission.stage_application_eligible is False

    process.exit(7)
    await _wait_until(
        lambda: supervisor.admission_snapshot().stage_application_eligible
    )


@pytest.mark.asyncio
async def test_inherited_pipe_descriptor_cannot_block_generation_cleanup(
    tmp_path: Path,
) -> None:
    process = _FakeProcess(finish_pipes_on_exit=False)
    cleanup_calls = 0

    async def immediate_sleep(_delay: float) -> None:
        await asyncio.sleep(0)

    async def factory(_generation: int) -> AudioCppGenerationHooks:
        async def contract_probe() -> str:
            return "available"

        async def health_probe() -> bool:
            process.exit(6)
            return False

        async def cleanup() -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

        return AudioCppGenerationHooks(
            contract_probe=contract_probe,
            health_probe=health_probe,
            cleanup=cleanup,
        )

    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_FakeLauncher([process]),
        port_preflight=_available_preflight,
        sleep=immediate_sleep,
    )
    start = asyncio.create_task(
        supervisor.ensure_running(
            _make_launch(tmp_path), generation_hooks_factory=factory
        )
    )

    await _wait_until(start.done)
    with pytest.raises(TTSOperationError):
        await start
    assert cleanup_calls == 1
    assert process.close_parent_calls == 1
    assert supervisor.snapshot().state == "unavailable"


@pytest.mark.asyncio
async def test_real_child_early_exit_drains_output_and_leaves_no_process(
    tmp_path: Path,
) -> None:
    launch = _make_real_launch(
        tmp_path,
        behavior={
            "early_exit": True,
            "exit_code": 9,
            "stdout_chunks": ["fixture early stdout\n"],
            "stderr_chunks": ["fixture early stderr\n"],
        },
    )
    launcher = _RealLauncher()
    supervisor = AudioCppSupervisor(
        source_environment={"PATH": os.environ.get("PATH", "")},
        process_launcher=launcher,
    )

    try:
        with pytest.raises(TTSOperationError) as raised:
            await supervisor.ensure_running(
                launch,
                generation_hooks_factory=_RealHttpHooksFactory(launch.base_url),
            )
        snapshot = supervisor.snapshot()

        assert raised.value.code == "process_exited"
        assert {line.text for line in snapshot.diagnostics} == {
            "fixture early stdout",
            "fixture early stderr",
        }
        assert snapshot.state == "unavailable"
        assert supervisor._generation is None
        assert len(launcher.processes) == 1
        process = launcher.processes[0]
        await _wait_for_pid_exit(process.pid)
        assert process.wait_calls == 1
    finally:
        await supervisor.close()
        await supervisor.wait_closed()


@pytest.mark.asyncio
async def test_real_child_force_kill_and_monitor_cleanup(tmp_path: Path) -> None:
    launch = _make_real_launch(
        tmp_path,
        behavior={"ignore_terminate": True},
        termination_grace_seconds=0.1,
    )
    launcher = _RealLauncher()
    supervisor = AudioCppSupervisor(
        source_environment={"PATH": os.environ.get("PATH", "")},
        process_launcher=launcher,
    )

    try:
        await supervisor.ensure_running(
            launch,
            generation_hooks_factory=_RealHttpHooksFactory(launch.base_url),
        )
        process = launcher.processes[0]

        await supervisor.stop()

        await _wait_for_pid_exit(process.pid)
        assert process.returncode == -signal.SIGKILL
        assert process.wait_calls == 1
        assert supervisor.snapshot().state == "stopped"
        assert supervisor._generation is None
        assert supervisor._startup_task is None
        assert supervisor._stop_task is None
    finally:
        await supervisor.close()
        await supervisor.wait_closed()


@pytest.mark.asyncio
async def test_real_child_inherited_pipes_finish_cleanup_without_killing_descendant(
    tmp_path: Path,
) -> None:
    descendant_pid_file = tmp_path / "descendant.pid"
    launch = _make_real_launch(
        tmp_path,
        behavior={
            "inherit_pipes_descendant": True,
            "descendant_pid_file": str(descendant_pid_file),
            "descendant_hold_seconds": 30.0,
            "exit_after_models": True,
            "stdout_chunks": ["parent output before inherited pipe\n"],
        },
    )
    launcher = _RealLauncher()
    supervisor = AudioCppSupervisor(
        source_environment={"PATH": os.environ.get("PATH", "")},
        process_launcher=launcher,
    )
    descendant_pid: int | None = None

    try:
        try:
            await supervisor.ensure_running(
                launch,
                generation_hooks_factory=_RealHttpHooksFactory(launch.base_url),
            )
        except TTSOperationError as error:
            assert error.code == "process_exited"
        await _wait_for_real_condition(descendant_pid_file.exists)
        descendant_pid = int(descendant_pid_file.read_text(encoding="ascii"))
        await _wait_for_real_condition(lambda: supervisor._generation is None)

        process = launcher.processes[0]
        await _wait_for_pid_exit(process.pid)
        assert process.wait_calls == 1
        assert supervisor.snapshot().state == "unavailable"
        assert _pid_exists(descendant_pid)

        await supervisor.close()
        await supervisor.wait_closed()

        assert _pid_exists(descendant_pid)
    finally:
        if descendant_pid is not None:
            await _terminate_fixture_pid(descendant_pid)
        await supervisor.close()
        await supervisor.wait_closed()


@pytest.mark.asyncio
async def test_repeated_real_spawn_stop_has_one_reaper_and_no_retained_generation(
    tmp_path: Path,
) -> None:
    launch = _make_real_launch(tmp_path)
    launcher = _RealLauncher()
    supervisor = AudioCppSupervisor(
        source_environment={"PATH": os.environ.get("PATH", "")},
        process_launcher=launcher,
    )

    try:
        for expected_generation in range(1, 6):
            endpoint = await supervisor.ensure_running(
                launch,
                generation_hooks_factory=_RealHttpHooksFactory(launch.base_url),
            )
            process = launcher.processes[-1]

            await supervisor.stop()

            await _wait_for_pid_exit(process.pid)
            assert endpoint.process_generation == expected_generation
            assert process.wait_calls == 1
            assert supervisor._generation is None
            assert supervisor._startup_task is None
            assert supervisor._stop_task is None
        assert len(launcher.processes) == 5
    finally:
        await supervisor.close()
        await supervisor.wait_closed()
