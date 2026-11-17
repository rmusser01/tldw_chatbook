"""Primary selected decisions on the actual received turn and original host."""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import replace

import pytest

from Tests.Chat.test_console_control_mcp import (
    _await_advertised_tool,
    _character_server,
)
from Tests.Chat.test_console_control_trace import _observe, _prepare
from Tests.Chat.test_console_domain_boundary import domain_case as _domain_case
from Tests.Chat.test_console_first_send_atomicity import _until
from Tests.Chat.test_console_runtime_lifetime import _View

domain_case = _domain_case
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.loopback_network,
]


def _rounds(case):
    host = case.controller._interrupt_host
    with host.lock:
        return {
            key: state
            for key, state in host.registries["approval"].items()
            if state["session_id"] == case.session.id
        }


def _mounted(cards):
    return cards[-1].get("round_id") if cards and cards[-1] else None


async def _primary_case(case, cancel):
    from tldw_chatbook.Agents.run_context import current_run_id, use_run_id
    from tldw_chatbook.Chat.console_control_decisions import ApprovalRoundDecision
    from tldw_chatbook.Chat.console_control_dispatch import (
        CancelTurn,
        Drain,
        ResolveDecision,
        _command_source,
    )
    from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost

    cards = []
    view = _View({"set_pending_approval": cards.append})
    view.attachment_generation = case.runtime.attach_view(view)
    case.runtime.finish_view_reconciliation(view, view.attachment_generation)
    host = case.controller._interrupt_host
    assert host._resolve_mcp_approval_timeout_seconds() == 0
    operations = case.runtime._execution_service.ensure_permission_operations()
    await operations.set_tool_state("builtin:tldw_chatbook", "list_characters", "ask")
    observed = []

    def observe_round(frame):
        if frame.f_locals["self"] is host:
            observed.append((_command_source.get(), current_run_id()))

    with _character_server() as server:
        server.terminal_response.clear()
        journey = await _prepare(case, server, kill_switch=False)
        receipt = await journey.start()
        queued = None
        try:
            await _await_advertised_tool(case, receipt, server, "list_characters")
            with _observe(
                InterruptRoundHost.request_mcp_approvals.__code__,
                sys.monitoring.events.PY_START,
                observe_round,
            ):
                server.tool_response.set()
                assert await _until(lambda: bool(_rounds(case)) or receipt.work.done())
                assert len(_rounds(case)) == 1
                head, state = next(iter(_rounds(case).items()))
                signal = case.controller._active_cancel_events[case.session.id]
                assert state["cancel_event"] is signal
                assert len(observed) == 1
                source, run_id = observed[0]
                assert (
                    source.controls is journey.controls
                    and source.sources is case.sources
                )
                assert run_id == state["run_id"]
                assert await _until(lambda: _mounted(cards) == head)
            assert case.controller.set_answerable_decision(case.session.id, head)
            await asyncio.sleep(
                1.1
            )  # Original timeout0 survives a complete polling cycle.
            assert head in _rounds(case) and not receipt.work.done()

            def decide(round_id, original, verdict="deny"):
                return journey.controls.dispatch(
                    ResolveDecision(
                        journey.identity(),
                        ApprovalRoundDecision(
                            round_id,
                            tuple(original["names"]),
                            {key: verdict for key in original["names"]},
                        ),
                    )
                ).applied.result()

            if cancel:
                turn_id = next(
                    key
                    for key, row in journey.controls._turns.items()
                    if row[2] is receipt.work
                )
                assert journey.controls.dispatch(
                    CancelTurn(journey.identity(), turn_id)
                ).applied.result()
                assert signal.is_set()
                server.terminal_response.set()
                await asyncio.gather(receipt.work, return_exceptions=True)
                assert await _until(lambda: not _rounds(case))
                assert decide(head, state) is False
            else:
                background = case.store.create_session(
                    title="Parked target", workspace_id="global", activate=False
                )
                case.controller.switch_session(background.id)
                assert await _until(lambda: _mounted(cards) is None)
                assert head in _rounds(case) and not signal.is_set()
                case.controller.switch_session(case.session.id)
                assert await _until(lambda: _mounted(cards) == head)
                assert case.controller.set_answerable_decision(case.session.id, head)
                probe = replace(state["calls"][0], call_id="task60-queued-round-probe")
                returned = threading.Event()

                def ask_queued():
                    # Round-mechanism probe, not another dispatched tool. Only
                    # actual observed command/run identities cross this worker.
                    token = _command_source.set(source)
                    try:
                        with use_run_id(run_id):
                            return host.request_mcp_approvals(
                                [probe], session_id=case.session.id
                            )
                    finally:
                        _command_source.reset(token)
                        returned.set()

                queued = asyncio.create_task(asyncio.to_thread(ask_queued))
                assert await _until(lambda: len(_rounds(case)) == 2)
                queued_id = next(key for key in _rounds(case) if key != head)
                queued_state = _rounds(case)[queued_id]
                assert queued_state["cancel_event"] is signal
                assert queued_state["run_id"] == run_id
                assert _mounted(cards) == head and not queued.done()
                assert decide(head, state) is True
                assert await _until(lambda: _mounted(cards) == queued_id)
                assert case.controller.set_answerable_decision(
                    case.session.id, queued_id
                )
                assert not signal.is_set() and not receipt.work.done()
                assert decide(queued_id, queued_state) is True
                assert await queued == {probe.call_id: "deny"}
                assert returned.is_set() and not _rounds(case)
                assert decide(queued_id, queued_state) is False
                server.terminal_response.set()
                assert (
                    await asyncio.wait_for(asyncio.shield(receipt.work), 20)
                ).accepted
            assert (
                await journey.controls.dispatch(Drain(journey.identity())).applied
                is True
            )
            assert not _rounds(case) and not journey.controls._tool_threads
        finally:
            server.tool_response.set()
            server.terminal_response.set()
            case.controller._signal_stop(session_id=case.session.id)
            await asyncio.gather(receipt.work, return_exceptions=True)
            if queued is not None:
                await asyncio.shield(queued)


@pytest.mark.parametrize("domain_case", ["control_mcp"], indirect=True)
async def test_primary_approval_fifo_parking_and_zero_timeout(domain_case):
    await _primary_case(domain_case, False)


@pytest.mark.parametrize("domain_case", ["control_mcp"], indirect=True)
async def test_cancel_received_turn_reaches_original_primary_round(domain_case):
    await _primary_case(domain_case, True)
