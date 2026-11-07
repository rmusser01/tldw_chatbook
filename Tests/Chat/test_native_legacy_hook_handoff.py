"""Legacy prompt outcomes retain the actual SQLite postcommit consumer."""

import asyncio
import sys

import pytest
import toml

from Tests.Agents.test_hook_permissions import _approve
from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.Agents.test_native_legacy_hooks import opaque_busy
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified

hook_file = _hook_file

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.skipif(
        not native_workflow_qualified(), reason="native POSIX group custody required"
    ),
]


@pytest.fixture
def saved_prompt_hook(hook_file):
    raw = toml.loads(hook_file.read_text())
    raw["hooks"] = {
        "hook": [
            {
                "id": "prompt",
                "event": "UserPromptSubmit",
                "command": [sys.executable, "-c", "print('owned prompt context')"],
            }
        ]
    }
    hook_file.write_text(toml.dumps(raw))


@pytest.mark.parametrize("finish", ["retry", "discard", "close"])
async def test_original_failed_publication_retains_hook_until_retry_or_shutdown(
    tmp_path, monkeypatch, owned_console_databases, saved_prompt_hook, finish
):
    db, store, controller, _gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    runtime = ConsoleRuntime(None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    controller._ensure_run_hooks = runtime.ensure_run_hooks
    controller._hook_permissions_accessor = runtime.ensure_hook_permissions
    _approve(runtime.ensure_hook_permissions())
    original = store.persist_message_if_needed
    calls = []
    fail = True

    def held_publication(message_id):
        if str(message_id).startswith("hook:"):
            calls.append(message_id)
            assert runtime.execution_capacity.snapshot().tool_workers == 1
            assert opaque_busy(runtime)
            if fail:
                raise RuntimeError("actual optional audit publication failed")
        return original(message_id)

    monkeypatch.setattr(store, "persist_message_if_needed", held_publication)
    try:
        result = await controller.submit_draft("hello", session_id="session-1")
        assert result.accepted and not result.provider_started
        assert len(calls) == 1
        assert opaque_busy(runtime), "failed host publication retired the event"
        assert runtime.execution_capacity.snapshot().tool_workers == 1
        assert len(controller._durable_postcommit_continuations) == 1
        if finish == "retry":
            fail = False
            again = await controller.resume_durable_postcommit(result.preparation_id)
            assert again.accepted and again.provider_started
            assert len(calls) == 2
        elif finish == "discard":
            fail = False
            discarded = await controller.discard_dispatch_recovery("session-1")
            assert discarded.accepted and not discarded.provider_started
            assert not controller._durable_postcommit_continuations
        else:
            await asyncio.wait_for(runtime.dispose(timeout_seconds=0), 5)
            assert not controller._durable_postcommit_continuations
        assert not opaque_busy(runtime)
        assert not runtime.execution_capacity.snapshot().executions
    finally:
        fail = False
        await runtime.dispose()


async def test_failed_required_native_commit_never_settles_hook_as_empty_obligation(
    tmp_path, monkeypatch, owned_console_databases, saved_prompt_hook
):
    import os
    import subprocess
    import traceback

    marker = "RETAINED_REQUIRED_HOOK_NOT_DRAINED"
    child_flag = "TLDW_TEST_RETAINED_REQUIRED_HOOK"
    if os.environ.get(child_flag) != "1":
        log = tmp_path / "retained-required-hook-child.log"
        with log.open("w") as output:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable, "-m", "pytest", "-q", "-s",
                    __file__ + "::test_failed_required_native_commit_never_settles_hook_as_empty_obligation",
                    "--basetemp=" + str(tmp_path / "retained-required-hook-child"),
                ],
                env=dict(os.environ, **{child_flag: "1"}),
                stdout=output, stderr=subprocess.STDOUT, timeout=40,
            )
        evidence = log.read_text()
        assert result.returncode == 0, evidence
        assert marker in evidence, evidence
        # The child exits with original custody retained, never after drain.
        return

    try:
        db, store, controller, _gateway = _controller(tmp_path)
        owned_console_databases(db, controller)
        runtime = ConsoleRuntime(None)
        runtime.set_chat_store(store)
        runtime.set_chat_controller(controller)
        controller._ensure_run_hooks = runtime.ensure_run_hooks
        controller._hook_permissions_accessor = runtime.ensure_hook_permissions
        _approve(runtime.ensure_hook_permissions())

        def failed_save(_acceptance):
            raise OSError("original required save unavailable")

        monkeypatch.setattr(store, "commit_durable_turn", failed_save)
        result = await controller.submit_draft("hello", session_id="session-1")
        assert not result.accepted
        assert opaque_busy(runtime)
        source = runtime._legacy_hooks_source
        assert source.cleanup_pending and len(source._events) == 1
        event = next(iter(source._events))
        assert event.failed_write and event.host_returned and event.targets
        assert all(target.returned and target.proven for target in event.targets)
        assert not event.retired and not any(target.released for target in event.targets)
        assert runtime.execution_capacity.snapshot().tool_workers == 1
        assert len(runtime.execution_capacity.snapshot().executions) == 1
        shutdown = asyncio.create_task(runtime.dispose(timeout_seconds=0))
        deadline = asyncio.get_running_loop().time() + 5
        while not (runtime._disposed and controller._shutdown_requested.is_set()):
            assert not shutdown.done()
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        await asyncio.sleep(0)
        assert not shutdown.done()
        assert event in source._events and source.cleanup_pending
        assert opaque_busy(runtime)
        assert runtime.execution_capacity.snapshot().tool_workers == 1
        assert len(runtime.execution_capacity.snapshot().executions) == 1
        print(marker, flush=True)
        # No successful shutdown or retirement is claimed by this process exit.
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)


async def test_unpublished_continuation_does_not_orphan_original_hook_handoff(
    tmp_path, monkeypatch, owned_console_databases, saved_prompt_hook
):
    from tldw_chatbook.Chat import console_chat_controller as module

    db, store, controller, _gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    runtime = ConsoleRuntime(None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    controller._ensure_run_hooks = runtime.ensure_run_hooks
    controller._hook_permissions_accessor = runtime.ensure_hook_permissions
    _approve(runtime.ensure_hook_permissions())
    original = module._DurablePostcommitContinuation
    unpublished = []

    def interrupted_construction(**kwargs):
        continuation = original(**kwargs)
        unpublished.append(continuation)
        raise RuntimeError("continuation construction interrupted")

    monkeypatch.setattr(
        module, "_DurablePostcommitContinuation", interrupted_construction
    )
    try:
        with pytest.raises(RuntimeError, match="continuation construction interrupted"):
            await controller.submit_draft("hello", session_id="session-1")
        assert unpublished and not controller._durable_postcommit_continuations
        assert not opaque_busy(runtime), (
            "unpublished continuation orphaned original host"
        )
        assert not runtime.execution_capacity.snapshot().executions
    finally:
        # A failing regression must still release its original test-owned event.
        for continuation in unpublished:
            if continuation.hook_handoff is not None:
                continuation.hook_handoff.complete()
        await runtime.dispose()


async def test_empty_original_prompt_outcome_retires_before_host_scope_exits(
    tmp_path, owned_console_databases, hook_file
):
    from tldw_chatbook.Agents.legacy_hook_ownership import (
        pending_prompt_handoff,
        prompt_handoff,
    )

    raw = toml.loads(hook_file.read_text())
    raw["hooks"] = {
        "hook": [{
            "id": "empty-prompt",
            "event": "UserPromptSubmit",
            "command": [sys.executable, "-c", "pass"],
        }]
    }
    hook_file.write_text(toml.dumps(raw))
    db, store, controller, _gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    runtime = ConsoleRuntime(None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    controller._ensure_run_hooks = runtime.ensure_run_hooks
    controller._hook_permissions_accessor = runtime.ensure_hook_permissions
    _approve(runtime.ensure_hook_permissions())
    try:
        with prompt_handoff(controller):
            outcome = await controller._legacy_submission_hook_input(
                "hello", "session-1"
            )
            assert outcome is not None and not outcome.blocked and not outcome.context
            assert pending_prompt_handoff() is None
            assert not opaque_busy(runtime)
            assert not runtime.execution_capacity.snapshot().executions
    finally:
        await runtime.dispose()
