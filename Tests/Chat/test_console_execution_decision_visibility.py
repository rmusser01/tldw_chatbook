"""The exact queued decision-read task also observes modal restoration."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_execution_process import _ready_client, _reply, _retrieve
from Tests.Chat.test_console_execution_workspace import _until
from tldw_chatbook.Chat.console_execution_decision_head import ConsoleDecisionHead, encode_decision_head
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.UI.Console_Modules.execution_decisions import _ProjectModal, _begin_project_visibility_restore


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["current", "source", "failed"])
async def test_queued_head_read_joins_the_actual_scope_transition_before_submission(outcome):
    # Transport ordering only: the UI test proves the real restoration future
    # follows the original visible FULL; these doubles grant no native proof.
    client, screen = _ready_client(), object()
    client._view, client._visit = screen, 1
    row = SimpleNamespace(id="session", incarnation_id="incarnation")
    client.projection_view = SimpleNamespace(
        source_revision=0, active_session_id=row.id, session=lambda _: row,
    )
    client.require_view = lambda view, visit: (
        view is client._view and visit == client._visit
    ) or (_ for _ in ()).throw(ConsoleChannelLost("changed view"))
    task = asyncio.create_task(client.read_decision_head(screen, 1, row.id, row.incarnation_id))
    await asyncio.sleep(0)  # Outer read has queued its retained inner owner.
    assert len(client._projection_reads) == 1 and not client._channel.messages
    state = _ProjectModal(
        client, screen, 1, 0,
        SimpleNamespace(session_id=row.id, incarnation_id=row.incarnation_id), object(),
        attachment_id=client._attachment_id,
    )
    _begin_project_visibility_restore(state)
    receipt = None
    try:
        await asyncio.sleep(0)
        assert not client._channel.messages and not task.done()
        if outcome == "failed":
            failure = RuntimeError("actual visible checkpoint failed")
            state.restoration.set_exception(failure)
            with pytest.raises(RuntimeError) as raised:
                await task
            assert raised.value is failure and not client._channel.messages
            return
        if outcome == "source":
            client._source_revision += 1
        state.restoration.set_result(True)
        client._project_modal_transition = None
        if outcome == "source":
            with pytest.raises(ConsoleChannelLost, match="scope"):
                await task
            assert not client._channel.messages
            return
        await _until(lambda: bool(client._channel.messages))
        receipt = next(iter(client._requests.values()))
        head = ConsoleDecisionHead(0, row.id, row.incarnation_id, client._attachment_id, 1, None, None)
        client._receive(_reply(client, receipt, 1, "received", {"sequence": receipt.identity.sequence}))
        client._receive(_reply(client, receipt, 2, "applied", encode_decision_head(head)))
        await asyncio.sleep(0)
        assert not task.done() and client.decision_head is None
        client._receive(_reply(client, receipt, 3, "retired", {
            "operation_id": receipt.identity.request_id, "retired": True,
        }))
        assert await task == head
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if receipt is not None:
            await _retrieve(receipt)
