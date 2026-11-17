"""Original trace writes and executor custody across selected source fences."""

from __future__ import annotations

import asyncio
import sys
import threading
from contextlib import contextmanager

import pytest

from Tests.Chat.test_console_domain_boundary import (
    _selection,
    domain_case as _domain_case,
)
from Tests.Chat.test_console_first_send_atomicity import _until
from Tests.Chat.test_console_refused_stream_retry import (
    _MODEL,
    _gateway_config,
    _server,
)

domain_case = _domain_case
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.loopback_network,
]


@contextmanager
def _observe(code, event, callback, *, values=False):
    monitoring = sys.monitoring
    slot = next(
        value
        for value in range(6)
        if value != monitoring.DEBUGGER_ID and monitoring.get_tool(value) is None
    )
    monitoring.use_tool_id(slot, "task60-trace-custody")

    def observe(_code, *_args):
        if _code is not code:
            return
        if values:
            callback(sys._getframe(1), _args)
        else:
            callback(sys._getframe(1))

    monitoring.register_callback(slot, event, observe)
    global_event = event == monitoring.events.RAISE
    if global_event:
        monitoring.set_events(slot, event)
    else:
        monitoring.set_local_events(slot, code, event)
    try:
        yield
    finally:
        if global_event:
            monitoring.set_events(slot, 0)
        else:
            monitoring.set_local_events(slot, code, 0)
        monitoring.register_callback(slot, event, None)
        monitoring.free_tool_id(slot)


class _Journey:
    def __init__(self, case, controls):
        self.case, self.controls = case, controls
        self.sequence = 0

    def identity(self):
        from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity

        self.sequence += 1
        return ControlIdentity(
            "profile",
            "service",
            self.sequence,
            f"trace-{self.sequence}",
            self.case.session.id,
            self.case.session.incarnation_id,
            self.controls.source_revision,
        )

    async def start(self):
        from tldw_chatbook.Chat.console_control_dispatch import (
            CommitPlanAck,
            ReceiveTurn,
            UpdateInputs,
        )

        updated = self.controls.dispatch(
            UpdateInputs(self.identity(), "Trace original", (1, self.sequence))
        )
        receipt = self.controls.dispatch(
            ReceiveTurn(
                self.identity(),
                f"turn-{self.sequence}",
                updated.applied.result(),
                _selection(self.case),
            )
        )
        assert await _until(lambda: self.controls.commit_plans or receipt.work.done())
        if receipt.work.done():
            receipt.work.result()
        plan = self.controls.commit_plans[0]
        self.controls.dispatch(
            CommitPlanAck(
                self.identity(),
                plan.received_request_id,
                plan.turn_id,
                plan.ordinal,
                plan.fingerprint,
            )
        )
        return receipt


async def _prepare(case, server, *, kill_switch=True):
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case.controller.prompt_history = PromptHistory(
        case.database.db_path.parent / "trace-history.jsonl"
    )
    case.sources.configuration.update(_gateway_config("openai", server.base_url))
    case.store.replace_session_settings(
        case.session.id,
        ConsoleSessionSettings(
            provider="openai",
            model=_MODEL,
            base_url=server.base_url,
            streaming=False,
        ),
    )
    case.store.replace_session_trace_privacy_override(
        case.session.id,
        capture_enabled=True,
        pii_redaction_enabled=None,
        expected_policy_revision=case.store.capture_policy_state(
            case.session.id
        ).policy_revision,
    )
    await (
        case.runtime._execution_service.ensure_permission_operations().set_kill_switch(
            kill_switch
        )
    )
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    return _Journey(case, controls)


@pytest.mark.parametrize("domain_case", ["control_trace"], indirect=True)
async def test_trace_stage_finishes_before_exact_executor_future_and_drain(domain_case):
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_control_dispatch import Drain, FenceSource

    case = domain_case
    entered, release = threading.Event(), threading.Event()

    def hold_return(frame):
        if frame.f_locals.get("self") is case.store:
            entered.set()
            assert release.wait(10), "Original trace executor was not released"

    with _server() as server:
        journey = await _prepare(case, server)
        with _observe(
            ConsoleChatStore._drain_provider_trace_settlement_work.__code__,
            sys.monitoring.events.PY_RETURN,
            hold_return,
        ):
            receipt = await journey.start()
            try:
                assert await _until(entered.is_set)
                assert (
                    await asyncio.wait_for(asyncio.shield(receipt.work), 20)
                ).accepted
                assert not journey.controls.admitted_effects
                with case.store._provider_trace_settlement_lock:
                    future = case.store._provider_trace_settlement_worker_future
                    assert not case.store._provider_trace_settlement_worker_active
                assert future is not None and not future.done()
                fence = journey.controls.dispatch(
                    FenceSource(journey.identity(), "configuration")
                )
                await fence.applied
                drain = journey.controls.dispatch(Drain(journey.identity()))
                await asyncio.sleep(0)
                assert not drain.applied.done() and not future.done()
                release.set()
                assert await drain.applied is True
                assert future.done()
            finally:
                release.set()
                await asyncio.gather(receipt.work, return_exceptions=True)


@pytest.mark.parametrize("domain_case", ["control_trace"], indirect=True)
async def test_old_trace_retry_keeps_each_call_source_and_shutdown_retains_failure(
    domain_case,
):
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_control_dispatch import FenceSource, Shutdown
    from tldw_chatbook.Chat.console_control_trace import drain_trace_worker

    case = domain_case
    calls = {}

    def retain_original(frame):
        handoff = frame.f_locals["handoff"]
        if handoff._prepared._control_owner.dispatch.controls._store is case.store:
            calls.setdefault(
                handoff.call_id, (handoff, frame.f_locals["canonical_message_id"])
            )

    with _server() as server:
        journey = await _prepare(case, server)
        with _observe(
            ConsoleChatStore._run_provider_trace_settlement.__code__,
            sys.monitoring.events.PY_START,
            retain_original,
        ):
            for _ in range(2):
                receipt = await journey.start()
                assert (
                    await asyncio.wait_for(asyncio.shield(receipt.work), 20)
                ).accepted
                await drain_trace_worker(case.store)
            assert len(calls) == 2
            owners = {
                call: handoff._prepared._control_owner
                for call, (handoff, _message) in calls.items()
            }
            assert (
                len(
                    {
                        owner.dispatch.source.identity.request_id
                        for owner in owners.values()
                    }
                )
                == 2
            )
            await journey.controls.dispatch(
                FenceSource(journey.identity(), "configuration")
            ).applied
            for handoff, message in calls.values():
                case.store._submit_provider_trace_settlement(handoff, message)
                await drain_trace_worker(case.store)
            coordinator = next(iter(journey.controls._trace_coordinators))
            with coordinator._queue_lock:
                assert set(coordinator._pending) == set(calls)
                for call, (_database, prepared) in coordinator._pending.items():
                    assert prepared._control_owner is owners[call]
            with case.store._provider_trace_settlement_lock:
                assert set(case.store._provider_trace_settlement_failed_work) == set(
                    calls
                )
            shutdown = journey.controls.dispatch(Shutdown(journey.identity()))
            with pytest.raises(RuntimeError, match="unresolved"):
                await shutdown.applied
            assert case.runtime._disposed
            assert case.store._provider_trace_settlement_executor_closed
            assert (
                case.store._provider_trace_settlement_executor_close_complete.is_set()
            )
            with case.store._provider_trace_settlement_lock:
                assert set(case.store._provider_trace_settlement_failed_work) == set(
                    calls
                )
