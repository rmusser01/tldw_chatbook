"""Same-visit transport proof; real backend token retirement is tested separately."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_execution_process import _ready_client, _reply, _retrieve
from Tests.Chat.test_console_execution_workspace import _until
from tldw_chatbook.Chat.console_execution_process import ConsoleDisplaySuperseded, ConsoleRemoteFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["hidden-false", "settled-first", "mounted", "wrong-attachment", "source", "negative", "failed", "unrelated"])
async def test_same_visit_supersession_requires_the_actual_new_checkpoint_receipt(case):
    client = _ready_client()
    screen = object()
    client._view, client._visit = screen, 7
    app = SimpleNamespace(console_execution_client=client, console_runtime=None)
    client._app = app
    row = SimpleNamespace(incarnation_id="incarnation")
    client.projection_view = SimpleNamespace(active_session_id="session", session=lambda _: row)
    client.require_view = lambda view, visit: (
        view is screen and visit == 7 and client._view is view and client._visit == visit
    ) or pytest.fail("unexpected view")
    task = asyncio.create_task(client.read_projection("session", "incarnation"))
    original = checkpoint = None
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        original = next(iter(client._requests.values()))
        client._receive(_reply(client, original, 1, "received", {"sequence": original.identity.sequence}))
        checkpoint = client.submit(
            "ProjectDecisionPresentation" if case != "unrelated" else "ResolveDecision",
            {
                "attachment_id": "wrong" if case == "wrong-attachment" else client._attachment_id,
                "visit": 7, "decision_id": "notice", "action": "mounted" if case == "mounted" else "hidden",
            } if case != "unrelated" else {
                "decision": {"kind": "ProjectDispatchDecision", "decision_id": "notice", "action": "cancel"},
            },
            session_id="session", incarnation_id="incarnation",
        )
        sequence = 2
        if case == "settled-first":
            client._receive(_reply(client, checkpoint, sequence, "received", {"sequence": checkpoint.identity.sequence}))
            client._receive(_reply(client, checkpoint, sequence + 1, "applied", {"applied": False}))
            client._receive(_reply(client, checkpoint, sequence + 2, "retired", {
                "operation_id": checkpoint.identity.request_id, "retired": True,
            }))
            assert checkpoint.identity.request_id not in client._requests
            sequence += 3
        client._receive(_reply(client, original, sequence, "failed", {
            "code": "view_checkpoint_superseded", "stage": "applied",
        }))
        client._receive(_reply(client, original, sequence + 1, "retired", {
            "operation_id": original.identity.request_id, "retired": True,
        }))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if case not in {"wrong-attachment", "unrelated", "settled-first"}:
            assert not task.done()  # Submission alone is not visibility proof.
        if case != "settled-first":
            client._receive(_reply(client, checkpoint, 4, "received", {"sequence": checkpoint.identity.sequence}))
            client._receive(_reply(client, checkpoint, 5,
                "failed" if case == "failed" else "applied",
                {"code": "command_failed", "stage": "applied"} if case == "failed" else {"applied": case == "mounted"},
            ))
        if case == "source":
            client._source_revision += 1
        if case != "settled-first":
            client._receive(_reply(client, checkpoint, 6, "retired", {
                "operation_id": checkpoint.identity.request_id, "retired": case != "negative",
            }))
        if case in {"hidden-false", "settled-first", "mounted"}:
            with pytest.raises(ConsoleDisplaySuperseded):
                await task
        else:
            with pytest.raises(ConsoleRemoteFailure) as error:
                await task
            assert error.value is original.applied.exception()
        assert client._projection is None and client._channel_error is None
    finally:
        client._fail(RuntimeError("test cleanup"))
        await asyncio.gather(*client._projection_reads, task, return_exceptions=True)
        for receipt in (original, checkpoint):
            if receipt is not None:
                await _retrieve(receipt)
