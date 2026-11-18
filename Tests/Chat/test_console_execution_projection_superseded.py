"""Only a retired, authentically scoped obsolete display may be deferred.

These are transport/classification tests, not evidence of native retirement.
The backend's physical retirement classification is tested at its own owner.
"""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_execution_process import _ready_client, _reply, _retrieve
from Tests.Chat.test_console_execution_workspace import _until
from tldw_chatbook.Chat.console_execution_process import (
    ConsoleChannelLost,
    ConsoleRemoteFailure,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "plain",
        "mount",
        "current-visit",
        "source",
        "attachment",
        "session",
        "app",
        "code",
        "stage",
        "no-received",
        "negative-retirement",
    ],
)
async def test_superseded_projection_requires_positive_retirement_and_exact_scope(case):
    client = _ready_client()
    screen = object()
    client._view, client._visit = screen, 7
    app = SimpleNamespace(console_execution_client=client, console_runtime=None)
    client._app = app
    row = SimpleNamespace(incarnation_id="incarnation")
    publication = SimpleNamespace(active_session_id="session", session=lambda _: row)
    client.projection_view = publication
    client.require_view = lambda view, visit: (
        view is screen and visit == 7
    ) or pytest.fail("unexpected initial visit")
    task = asyncio.create_task(
        client.read_projection(
            "session", "incarnation", _checkpoint="mount" if case == "mount" else None
        )
    )
    receipt = None
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        if case != "no-received":
            client._receive(
                _reply(
                    client,
                    receipt,
                    1,
                    "received",
                    {"sequence": receipt.identity.sequence},
                )
            )
        if case != "current-visit":
            client._view, client._visit = None, 8
        if case == "source":
            client._source_revision += 1
        elif case == "attachment":
            client._attachment_id = "replacement"
        elif case == "session":
            row.incarnation_id = "replacement"
        elif case == "app":
            app.console_execution_client = object()
        client._receive(
            _reply(
                client,
                receipt,
                1 if case == "no-received" else 2,
                "failed",
                {
                    "code": "command_failed"
                    if case == "code"
                    else "view_checkpoint_superseded",
                    "stage": "admission" if case == "stage" else "applied",
                },
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done() and client._projection_lock.locked()
        assert not receipt.retired.done()
        client._receive(
            _reply(
                client,
                receipt,
                2 if case == "no-received" else 3,
                "retired",
                {
                    "operation_id": receipt.identity.request_id,
                    "retired": case != "negative-retirement",
                },
            )
        )
        await _until(task.done)
        if case in {"plain", "mount"}:
            with pytest.raises(
                ConsoleChannelLost, match="visit was superseded"
            ) as caught:
                await task
            assert caught.value.__cause__ is receipt.applied.exception()
        elif case == "negative-retirement":
            with pytest.raises(ConsoleChannelLost, match="retirement is unproven"):
                await task
            assert client._channel_error is not None
        else:
            with pytest.raises(ConsoleRemoteFailure) as caught:
                await task
            assert caught.value is receipt.applied.exception()
        assert client.projection_view is publication and client._projection is None
        assert not client._projection_lock.locked()
        if case != "negative-retirement":
            assert client._channel_error is None
    finally:
        client._fail(RuntimeError("test cleanup"))
        await asyncio.gather(*client._projection_reads, task, return_exceptions=True)
        if receipt is not None:
            await _retrieve(receipt)
