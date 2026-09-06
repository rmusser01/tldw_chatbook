from __future__ import annotations

import logging
import subprocess
import threading
from typing import Any, Callable

import pytest

from tldw_chatbook.Event_Handlers.LLM_Management_Events import server_lifecycle


class _App:
    def __init__(self) -> None:
        self._llm_server_lifecycle_lock = threading.RLock()
        self._llm_server_launch_claims: dict[str, object] = {}
        self.llamacpp_server_process = None
        self.screen_stack: list[object] = []
        self.before_callback: Callable[[Any, tuple[Any, ...]], None] | None = None
        self.after_callback: Callable[[Any, tuple[Any, ...], Any], None] | None = None
        self.notifications: list[tuple[str, str | None]] = []

    def call_from_thread(self, callback: Callable[..., Any], *args: Any) -> Any:
        if self.before_callback is not None:
            self.before_callback(callback, args)
        result = callback(*args)
        if self.after_callback is not None:
            self.after_callback(callback, args, result)
        return result

    def notify(self, message: str, severity: str | None = None) -> None:
        self.notifications.append((message, severity))


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


class _Resource:
    def __init__(self, private_marker: str = "PRIVATE_RESOURCE_PATH") -> None:
        self.private_marker = private_marker
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def __repr__(self) -> str:
        return f"_Resource({self.private_marker!r})"

    def delete_if_unleased(self) -> bool:
        return self.close_count > 0


class _FailingResource(_Resource):
    def close(self) -> None:
        self.close_count += 1
        raise RuntimeError(self.private_marker)


class _TrackingLock:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.depth = 0
        self.entry_attempted = threading.Event()

    def __enter__(self) -> None:
        self.entry_attempted.set()
        self._lock.acquire()
        self.depth += 1

    def __exit__(self, *args: object) -> None:
        self.depth -= 1
        self._lock.release()


class _LockCheckingResource(_Resource):
    def __init__(self, lock: _TrackingLock) -> None:
        super().__init__()
        self.lock = lock
        self.close_lock_depths: list[int] = []

    def close(self) -> None:
        self.close_lock_depths.append(self.lock.depth)
        super().close()


class _Process:
    pid = 4242

    def __init__(
        self,
        *,
        returncode: int = 0,
        stubborn: bool = False,
        resource: _Resource | None = None,
    ) -> None:
        self.returncode = returncode
        self.stubborn = stubborn
        self.resource = resource
        self.running = True
        self.close_counts_during_wait: list[int] = []

    def poll(self) -> int | None:
        return None if self.running else self.returncode

    def terminate(self) -> None:
        if not self.stubborn:
            self.running = False

    def kill(self) -> None:
        if not self.stubborn:
            self.running = False

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None and self.stubborn:
            raise subprocess.TimeoutExpired("PRIVATE_COMMAND", timeout)
        if self.resource is not None:
            self.close_counts_during_wait.append(self.resource.close_count)
        self.running = False
        return self.returncode


class _SubprocessModule:
    DEVNULL = subprocess.DEVNULL

    def __init__(
        self,
        process: _Process | None = None,
        error: Exception | None = None,
    ) -> None:
        self.process = process
        self.error = error

    def Popen(self, command: list[str], **kwargs: Any) -> _Process:
        if self.error is not None:
            raise self.error
        assert self.process is not None
        return self.process


@pytest.mark.parametrize("configured", [False, True])
def test_snapshot_subprocess_options_are_opt_in(configured):
    import os

    app = _App()
    claim = server_lifecycle.reserve_server_launch(app, "llamacpp")
    captured = []

    class Subprocess:
        def Popen(self, command, **kwargs):
            captured.append(kwargs)
            return _Process()

    options = {"env": {"FROZEN": "yes"}, "private_umask": 0o077} if configured else {}
    server_lifecycle.run_server_subprocess(
        app, "llamacpp", ["server"], claim, Subprocess(), **options
    )
    assert len(captured) == 1
    if configured:
        assert captured[0]["env"] == {"FROZEN": "yes"}
        assert captured[0].get("umask") == (0o077 if os.name == "posix" else None)
    else:
        assert "env" not in captured[0]
        assert "umask" not in captured[0]


def test_snapshot_live_predicate_requires_exact_child_not_listener():
    app = _App()
    claim = server_lifecycle.reserve_server_launch(app, "llamacpp")
    assert not server_lifecycle.snapshot_claim_is_live(app, claim)
    process = _Process()
    server_lifecycle.publish_server_process(app, "llamacpp", claim, process)
    assert server_lifecycle.snapshot_claim_is_live(app, claim)
    process.running = False
    assert not server_lifecycle.snapshot_claim_is_live(app, claim)


def test_snapshot_readiness_rejects_unverifiable_child_liveness():
    app = _App()
    claim = server_lifecycle.reserve_server_launch(app, "llamacpp")

    class UnverifiableProcess:
        def poll(self):
            raise OSError("private-process-canary")

    app.llamacpp_server_process = UnverifiableProcess()
    assert server_lifecycle.snapshot_claim_is_live(app, claim) is False


def _reserve_with_resource(
    app: _App,
    resource: _Resource,
) -> server_lifecycle.ServerLaunchClaim:
    claim = server_lifecycle.reserve_server_launch(
        app,
        "llamacpp",
        authority="Managed GGUF",
    )
    assert claim is not None
    assert server_lifecycle.attach_server_claim_resource(
        app,
        "llamacpp",
        claim,
        resource,
    )
    return claim


def test_attach_current_claim_records_authority_without_private_resource_repr() -> None:
    app = _App()
    resource = _Resource()

    claim = server_lifecycle.reserve_server_launch(
        app,
        "llamacpp",
        authority="Managed GGUF",
    )

    assert claim is not None
    assert (
        server_lifecycle.attach_server_claim_resource(
            app,
            "llamacpp",
            claim,
            resource,
        )
        is True
    )
    assert claim.authority == "Managed GGUF"
    assert "PRIVATE_RESOURCE_PATH" not in repr(claim)
    assert resource.close_count == 0


def test_stale_claim_attach_rejects_without_closing_either_resource() -> None:
    app = _App()
    stale_claim = server_lifecycle.reserve_server_launch(app, "llamacpp")
    assert stale_claim is not None
    assert server_lifecycle.release_server_claim(app, "llamacpp", stale_claim) is True
    current_claim = server_lifecycle.reserve_server_launch(app, "llamacpp")
    assert current_claim is not None
    current_resource = _Resource("PRIVATE_CURRENT_PATH")
    stale_resource = _Resource("PRIVATE_STALE_PATH")
    assert (
        server_lifecycle.attach_server_claim_resource(
            app,
            "llamacpp",
            current_claim,
            current_resource,
        )
        is True
    )

    assert (
        server_lifecycle.attach_server_claim_resource(
            app,
            "llamacpp",
            stale_claim,
            stale_resource,
        )
        is False
    )
    assert stale_resource.close_count == 0
    assert current_resource.close_count == 0


def test_second_attachment_is_rejected_and_caller_retains_ownership() -> None:
    app = _App()
    claim = server_lifecycle.reserve_server_launch(app, "llamacpp")
    assert claim is not None
    current_resource = _Resource("PRIVATE_CURRENT_PATH")
    rejected_resource = _Resource("PRIVATE_REJECTED_PATH")
    assert (
        server_lifecycle.attach_server_claim_resource(
            app,
            "llamacpp",
            claim,
            current_resource,
        )
        is True
    )

    assert (
        server_lifecycle.attach_server_claim_resource(
            app,
            "llamacpp",
            claim,
            rejected_resource,
        )
        is False
    )
    assert current_resource.close_count == 0
    assert rejected_resource.close_count == 0
    rejected_resource.close()
    assert rejected_resource.close_count == 1


def test_attach_cancelled_claim_rejects_and_caller_retains_ownership() -> None:
    app = _App()
    claim = server_lifecycle.reserve_server_launch(app, "llamacpp")
    assert claim is not None
    claim.cancel_event.set()
    resource = _Resource("PRIVATE_CANCELLED_PATH")

    assert (
        server_lifecycle.attach_server_claim_resource(
            app,
            "llamacpp",
            claim,
            resource,
        )
        is False
    )
    assert resource.close_count == 0
    resource.close()
    assert resource.close_count == 1


def test_attach_waits_for_claim_settlement_and_rejects_stale_transfer() -> None:
    app = _App()
    lock = _TrackingLock()
    app._llm_server_lifecycle_lock = lock
    claim = server_lifecycle.reserve_server_launch(app, "llamacpp")
    assert claim is not None
    resource = _Resource("PRIVATE_CONTENDED_PATH")
    results: list[bool] = []

    def attach() -> None:
        results.append(
            server_lifecycle.attach_server_claim_resource(
                app,
                "llamacpp",
                claim,
                resource,
            )
        )

    with lock:
        lock.entry_attempted.clear()
        thread = threading.Thread(target=attach)
        thread.start()
        assert lock.entry_attempted.wait(timeout=5)
        assert server_lifecycle.release_server_claim(app, "llamacpp", claim) is True
    thread.join(timeout=5)

    assert thread.is_alive() is False
    assert results == [False]
    assert resource.close_count == 0
    resource.close()
    assert resource.close_count == 1


def test_attach_rejects_resource_without_callable_close() -> None:
    app = _App()
    claim = server_lifecycle.reserve_server_launch(app, "llamacpp")
    assert claim is not None

    assert (
        server_lifecycle.attach_server_claim_resource(
            app,
            "llamacpp",
            claim,
            object(),
        )
        is False
    )
    resource = _Resource()
    assert server_lifecycle.attach_server_claim_resource(
        app,
        "llamacpp",
        claim,
        resource,
    )


def test_release_refuses_between_popen_and_publication_until_process_death() -> None:
    app = _App()
    resource = _Resource()
    claim = _reserve_with_resource(app, resource)
    process = _Process(resource=resource)
    publication_entered = threading.Event()
    continue_publication = threading.Event()
    results: list[str] = []

    def block_before_publication(callback: Any, _args: tuple[Any, ...]) -> None:
        if callback is server_lifecycle.publish_server_process:
            publication_entered.set()
            assert continue_publication.wait(timeout=5)

    app.before_callback = block_before_publication
    worker = threading.Thread(
        target=lambda: results.append(
            server_lifecycle.run_server_subprocess(
                app,
                "llamacpp",
                ["PRIVATE_COMMAND"],
                claim,
                _SubprocessModule(process),
            )
        )
    )
    worker.start()
    try:
        assert publication_entered.wait(timeout=5)
        assert process.poll() is None
        assert server_lifecycle.release_server_claim(app, "llamacpp", claim) is False
        assert resource.close_count == 0
    finally:
        continue_publication.set()
        worker.join(timeout=5)

    assert worker.is_alive() is False
    assert results == ["llamacpp server exited (code=0)"]
    assert process.poll() is not None
    assert resource.close_count == 1


def test_cancelled_before_spawn_closes_resource_once_when_claim_releases() -> None:
    app = _App()
    resource = _Resource()
    claim = _reserve_with_resource(app, resource)
    claim.cancel_event.set()

    result = server_lifecycle.run_server_subprocess(
        app,
        "llamacpp",
        ["PRIVATE_COMMAND"],
        claim,
        _SubprocessModule(),
    )

    assert result == "llamacpp launch cancelled"
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None
    assert resource.close_count == 1
    assert server_lifecycle.release_server_claim(app, "llamacpp", claim) is False
    assert resource.close_count == 1


def test_popen_failure_closes_resource_once_without_private_output() -> None:
    app = _App()
    resource = _Resource("PRIVATE_LEASE_PATH")
    claim = _reserve_with_resource(app, resource)

    result = server_lifecycle.run_server_subprocess(
        app,
        "llamacpp",
        ["PRIVATE_COMMAND"],
        claim,
        _SubprocessModule(error=RuntimeError("PRIVATE_SPAWN_DETAIL")),
    )

    assert result == "llamacpp server failed (category=RuntimeError)"
    assert "PRIVATE" not in result
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None
    assert resource.close_count == 1


def test_publication_marshalling_failure_proves_death_before_release() -> None:
    app = _App()
    resource = _Resource("PRIVATE_LEASE_PATH")
    claim = _reserve_with_resource(app, resource)
    process = _Process(resource=resource)

    def fail_publication_marshalling(
        callback: Any,
        _args: tuple[Any, ...],
    ) -> None:
        if callback is server_lifecycle.publish_server_process:
            raise RuntimeError("PRIVATE_MARSHALLING_DETAIL")

    app.before_callback = fail_publication_marshalling
    result = server_lifecycle.run_server_subprocess(
        app,
        "llamacpp",
        ["PRIVATE_COMMAND"],
        claim,
        _SubprocessModule(process),
    )

    assert result == "llamacpp server failed (category=RuntimeError)"
    assert "PRIVATE" not in result
    assert process.close_counts_during_wait == [0]
    assert process.poll() is not None
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None
    assert resource.close_count == 1


def test_failed_publication_with_proven_death_closes_resource_once() -> None:
    app = _App()
    resource = _Resource()
    claim = _reserve_with_resource(app, resource)
    process = _Process(resource=resource)
    publication_results: list[bool] = []

    def cancel_before_publication(callback: Any, _args: tuple[Any, ...]) -> None:
        if callback is server_lifecycle.publish_server_process:
            claim.cancel_event.set()

    def record_publication(
        callback: Any,
        _args: tuple[Any, ...],
        result: Any,
    ) -> None:
        if callback is server_lifecycle.publish_server_process:
            publication_results.append(bool(result))

    app.before_callback = cancel_before_publication
    app.after_callback = record_publication
    result = server_lifecycle.run_server_subprocess(
        app,
        "llamacpp",
        ["PRIVATE_COMMAND"],
        claim,
        _SubprocessModule(process),
    )

    assert result == "llamacpp launch cancelled"
    assert publication_results == [False]
    assert process.poll() is not None
    assert process.close_counts_during_wait == [0]
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None
    assert resource.close_count == 1


@pytest.mark.parametrize("returncode", [0, 7])
def test_process_exit_closes_resource_only_after_exact_death(returncode: int) -> None:
    app = _App()
    resource = _Resource()
    claim = _reserve_with_resource(app, resource)
    process = _Process(returncode=returncode, resource=resource)

    result = server_lifecycle.run_server_subprocess(
        app,
        "llamacpp",
        ["PRIVATE_COMMAND"],
        claim,
        _SubprocessModule(process),
    )

    assert result == f"llamacpp server exited (code={returncode})"
    assert process.close_counts_during_wait == [0]
    assert process.poll() is not None
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None
    assert resource.close_count == 1


@pytest.mark.parametrize("provider", ("vllm", "mlx"))
def test_default_nonzero_status_contract_remains_raw_for_non_gguf_callers(
    provider: str,
) -> None:
    app = _App()
    destination = _Destination()
    app.screen_stack = [type("Screen", (), {"llm_window": destination})()]
    claim = server_lifecycle.reserve_server_launch(app, provider)
    assert claim is not None
    process = _Process(returncode=19)

    result = server_lifecycle.run_server_subprocess(
        app,
        provider,
        ["PRIVATE_COMMAND"],
        claim,
        _SubprocessModule(process),
    )

    assert result == f"{provider} server exited (code=19)"
    assert destination.state_changes[-1] == (
        provider,
        f"{provider} server exited (code=19)",
    )


@pytest.mark.asyncio
async def test_successful_stop_closes_resource_once_after_process_death() -> None:
    app = _App()
    resource = _Resource()
    claim = _reserve_with_resource(app, resource)
    process = _Process(resource=resource)
    assert server_lifecycle.publish_server_process(
        app,
        "llamacpp",
        claim,
        process,
    )

    assert (
        await server_lifecycle.stop_server_process(app, "llamacpp", "llama.cpp") is True
    )
    assert process.poll() is not None
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None
    assert resource.close_count == 1


@pytest.mark.asyncio
async def test_stop_failure_notification_is_actionable_without_process_id() -> None:
    app = _App()
    claim = server_lifecycle.reserve_server_launch(app, "llamacpp")
    assert claim is not None
    process = _Process(stubborn=True)
    assert server_lifecycle.publish_server_process(
        app,
        "llamacpp",
        claim,
        process,
    )

    assert (
        await server_lifecycle.stop_server_process(app, "llamacpp", "llama.cpp")
        is False
    )

    assert app.notifications == [
        ("llama.cpp did not stop; retry Stop.", "error"),
    ]
    assert str(process.pid) not in app.notifications[0][0]


def test_stubborn_cancelled_process_retains_resource_until_proven_dead() -> None:
    app = _App()
    resource = _Resource()
    claim = _reserve_with_resource(app, resource)
    process = _Process(stubborn=True, resource=resource)

    def cancel_after_publication(
        callback: Any,
        _args: tuple[Any, ...],
        published: Any,
    ) -> None:
        if callback is server_lifecycle.publish_server_process and published:
            claim.cancel_event.set()

    app.after_callback = cancel_after_publication
    result = server_lifecycle.run_server_subprocess(
        app,
        "llamacpp",
        ["PRIVATE_COMMAND"],
        claim,
        _SubprocessModule(process),
    )

    assert result == "llamacpp launch cancelled"
    assert process.poll() is None
    assert server_lifecycle.current_server_claim(app, "llamacpp") is claim
    assert server_lifecycle.server_process(app, "llamacpp") is process
    assert resource.close_count == 0
    assert resource.delete_if_unleased() is False

    process.running = False
    assert server_lifecycle.clear_server_process(
        app,
        "llamacpp",
        claim,
        process,
    )
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None
    assert resource.close_count == 1
    assert resource.delete_if_unleased() is True


def test_stale_release_and_clear_cannot_close_any_generation_resource() -> None:
    app = _App()
    current_resource = _Resource("PRIVATE_CURRENT_PATH")
    current_claim = _reserve_with_resource(app, current_resource)
    current_process = _Process(resource=current_resource)
    assert server_lifecycle.publish_server_process(
        app,
        "llamacpp",
        current_claim,
        current_process,
    )
    stale_resource = _Resource("PRIVATE_STALE_PATH")
    stale_claim = server_lifecycle.ServerLaunchClaim(
        provider="llamacpp",
        _resource=stale_resource,
    )
    current_process.running = False

    assert server_lifecycle.release_server_claim(app, "llamacpp", stale_claim) is False
    assert (
        server_lifecycle.clear_server_process(
            app,
            "llamacpp",
            stale_claim,
            current_process,
        )
        is False
    )
    assert current_resource.close_count == 0
    assert stale_resource.close_count == 0
    assert server_lifecycle.current_server_claim(app, "llamacpp") is current_claim

    assert server_lifecycle.clear_server_process(
        app,
        "llamacpp",
        current_claim,
        current_process,
    )
    assert current_resource.close_count == 1
    assert stale_resource.close_count == 0


def test_stale_generation_cannot_clear_current_spawn_protection() -> None:
    app = _App()
    resource = _Resource("PRIVATE_CURRENT_PATH")
    current_claim = _reserve_with_resource(app, resource)
    stale_claim = server_lifecycle.ServerLaunchClaim(
        provider="llamacpp",
        _spawning=True,
    )
    assert server_lifecycle._begin_server_process_spawn(
        app,
        "llamacpp",
        current_claim,
    )

    assert (
        server_lifecycle._finish_server_process_spawn(
            app,
            "llamacpp",
            stale_claim,
        )
        is False
    )
    assert (
        server_lifecycle.release_server_claim(app, "llamacpp", current_claim) is False
    )
    assert resource.close_count == 0

    assert server_lifecycle._finish_server_process_spawn(
        app,
        "llamacpp",
        current_claim,
    )
    assert server_lifecycle.release_server_claim(app, "llamacpp", current_claim)
    assert resource.close_count == 1


def test_close_failure_settles_state_and_reports_only_stable_category(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _App()
    resource = _FailingResource("PRIVATE_CLOSE_DETAIL")
    claim = _reserve_with_resource(app, resource)

    with caplog.at_level(logging.ERROR, logger=server_lifecycle.__name__):
        assert server_lifecycle.release_server_claim(app, "llamacpp", claim) is True

    assert resource.close_count == 1
    assert server_lifecycle.current_server_claim(app, "llamacpp") is None
    replacement = server_lifecycle.reserve_server_launch(app, "llamacpp")
    assert replacement is not None
    report = "\n".join(record.getMessage() for record in caplog.records)
    assert "category=resource_close_failed" in report
    assert "PRIVATE_CLOSE_DETAIL" not in report
    assert "PRIVATE_CLOSE_DETAIL" not in repr(claim)


def test_resource_is_detached_under_lock_and_closed_after_unlock() -> None:
    app = _App()
    lock = _TrackingLock()
    app._llm_server_lifecycle_lock = lock
    resource = _LockCheckingResource(lock)
    claim = _reserve_with_resource(app, resource)

    assert server_lifecycle.release_server_claim(app, "llamacpp", claim) is True

    assert resource.close_count == 1
    assert resource.close_lock_depths == [0]
