"""Native finite outcomes and failed resource exits preserve their actual owners."""

import asyncio
import threading
from contextvars import ContextVar

import pytest

from Tests.Chat.test_console_native_preparation import (
    native_host as native_host,  # noqa: PLC0414
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.mark.parametrize(
    "mode", ["value", "error", "async-error", "cancel", "async-cancel", "stop"]
)
async def test_finite_callback_outcome_returns_all_native_resources(native_host, mode):
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner
    from tldw_chatbook.Chat.console_workflow_preparation import (
        current_workflow_preparation,
    )

    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    before = h.db.automatic_work.snapshot(prepared.authorization.claim.chain_id)
    value = object()
    error = (
        asyncio.CancelledError("callback cancelled")
        if "cancel" in mode
        else StopIteration("callback stopped")
        if mode == "stop"
        else RuntimeError("callback failed")
    )
    marker = ContextVar("native_finite_caller", default=None)
    token = marker.set(value)
    caller = asyncio.current_task()
    observed, drains = {}, []

    def enter_body():
        record = prepared._finite_reads[-1]
        owner, claim = prepared._finite
        assert owner is current_execution_owner()
        assert current_workflow_preparation() is prepared
        assert current_automatic_work() is h.phase.context
        assert marker.get() is value
        observed.update(record=record, owner=owner, claim=claim)
        owner.on_drained(
            lambda proven: drains.append((proven, threading.current_thread()))
        )

    async def pending():
        enter_body()
        raise error

    def callback():
        if mode.startswith("async-"):
            return pending()
        enter_body()
        if mode != "value":
            raise error
        return value

    try:
        if mode == "value":
            assert await prepared._read(callback) is value
        elif mode == "stop":
            with pytest.raises(
                RuntimeError, match="coroutine raised StopIteration"
            ) as caught:
                await prepared._read(callback)
            assert caught.value.__cause__ is error
        else:
            with pytest.raises(type(error)) as caught:
                await prepared._read(callback)
            assert caught.value is error
        assert asyncio.current_task() is caller and marker.get() is value
        record, owner, claim = (observed[key] for key in ("record", "owner", "claim"))
        assert (
            record.positive and record.scopes_returned and record.wrapper_error is None
        )
        assert record.body_error is (None if mode == "value" else error)
        assert record.proof[0].callback_returned
        assert record.proof[1].done() and record.proof[1].exception() is None
        assert record.proof[2].done() and record.proof[2].result() is None
        assert len(drains) == 1 and drains[0][0] is True
        assert not owner.is_admitted and prepared._finite is None
        assert claim.id not in h.runtime.worktree_activity._claims
        assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
        after = h.db.automatic_work.snapshot(before.chain_id)
        assert after.used == before.used and after.reserved == before.reserved
    finally:
        marker.reset(token)
        await prepared.close()
    assert prepared._closed and prepared._source_scope is None
    assert prepared not in h.runtime._workflow_preparations


_SCOPE_SCRIPT = """
import asyncio, os, sys
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install(); real_profile_guard.install()
from Tests.Chat.test_workflow_native_return import scope_failure_journey
with user_fixture_default_owner():
    asyncio.run(scope_failure_journey(Path(os.environ["XDG_DATA_HOME"]).parent, sys.argv[2]))
"""


@pytest.mark.parametrize("mode", ["source-exit", "storage-exit", "operation-finish"])
async def test_failed_native_scope_cannot_certify_callback_return(tmp_path, mode):
    from Tests.Chat.test_console_workflow_scheduling import (
        _expect_retained_planning_child,
    )

    await asyncio.to_thread(
        _expect_retained_planning_child,
        (tmp_path, "native-scope", mode),
        {"script": _SCOPE_SCRIPT, "timeout": 40},
        "FINITE_NATIVE_SCOPE_RETAINED_" + mode,
    )


async def scope_failure_journey(tmp_path, mode):
    """A private process may end with intentionally unproven native resources."""
    import os
    from contextlib import contextmanager
    from types import SimpleNamespace

    from Tests.Chat import test_console_native_preparation as native_tests
    from tldw_chatbook.Agents.execution_capacity import OwnedOperation
    from tldw_chatbook.Chat import console_workflow_preparation as preparation

    with pytest.MonkeyPatch.context() as monkeypatch:
        host = native_tests.native_host.__wrapped__(
            tmp_path, monkeypatch, SimpleNamespace(param=None)
        )
        h = await anext(host)
        prepared = await h.bridge._prepare_workflow_reply(
            phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
        )
        failure = RuntimeError("native resource exit failed")
        observed = {}

        def callback():
            record = prepared._finite_reads[-1]
            observed.update(
                record=record,
                owner=record.owner,
                claim=record.claim,
                operation=record.operation,
            )
            return "body returned"

        if mode == "operation-finish":
            original = OwnedOperation.finish

            def finish(operation):
                if operation is observed.get("operation"):
                    raise failure
                return original(operation)

            monkeypatch.setattr(OwnedOperation, "finish", finish)
        else:
            original = (
                prepared.activate_source
                if mode == "source-exit"
                else preparation.owned_goal_storage
            )

            @contextmanager
            def scope(*args, **kwargs):
                with original(*args, **kwargs):
                    yield
                if observed:
                    raise failure

            if mode == "source-exit":
                monkeypatch.setattr(prepared, "activate_source", scope)
            else:
                monkeypatch.setattr(preparation, "owned_goal_storage", scope)
        with pytest.raises(RuntimeError) as caught:
            await prepared._read(callback)
        assert caught.value is failure
        record = observed["record"]
        assert record.body_error is None and record.wrapper_error is failure
        assert not record.positive and not record.scopes_returned
        assert observed["owner"].is_admitted
        assert observed["claim"].id in h.runtime.worktree_activity._claims
        assert prepared in h.runtime._workflow_preparations
        with pytest.raises(RuntimeError):
            await prepared.close()
        assert not h.calls
        print("FINITE_NATIVE_SCOPE_RETAINED_" + mode, flush=True)
        # The failed scope deliberately leaves no positive drain receipt.
        os._exit(0)
