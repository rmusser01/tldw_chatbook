"""Transport controls for a retired readiness-input change, not native proof."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_execution_process import _ready_client, _reply, _retrieve
from Tests.Chat.test_console_execution_workspace import _until
from tldw_chatbook.Chat.console_execution_process import (
    ConsoleChannelLost,
    ConsoleReadinessExpired,
    ConsoleRemoteFailure,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "current",
        "preparing",
        "source",
        "view",
        "visit",
        "attachment",
        "incarnation",
        "preparation",
        "app",
        "ordinary-failure",
        "admission",
        "negative-retirement",
    ],
)
async def test_readiness_change_requires_retirement_and_unchanged_original_scope(case):
    client = _ready_client()
    screen = object()
    client._view, client._visit = screen, 7
    app = SimpleNamespace(console_execution_client=client, console_runtime=None)
    client._app = app
    row = SimpleNamespace(incarnation_id="incarnation")
    publication = SimpleNamespace(active_session_id="session", session=lambda _: row)
    if case != "preparing":
        client.projection_view = publication
    preparation = object() if case == "preparing" else None
    client._view_preparation = preparation
    client.require_view = lambda view, visit: (
        view is screen and visit == 7
    ) or pytest.fail("unexpected active view")
    task = asyncio.create_task(
        client.read_projection(
            "session",
            "incarnation",
            _preparation=preparation,
            _checkpoint="mount" if case == "preparing" else None,
        )
    )
    receipt = None
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        sequence = 1
        if case != "admission":
            client._receive(
                _reply(
                    client,
                    receipt,
                    sequence,
                    "received",
                    {
                        "sequence": receipt.identity.sequence,
                    },
                )
            )
            sequence += 1
        if case == "source":
            client._source_revision += 1
        elif case == "view":
            client._view = None
        elif case == "visit":
            client._visit += 1
        elif case == "attachment":
            client._attachment_id = "replacement"
        elif case == "incarnation":
            row.incarnation_id = "replacement"
        elif case == "preparation":
            client._view_preparation = object()
        elif case == "app":
            app.console_execution_client = object()
        client._receive(
            _reply(
                client,
                receipt,
                sequence,
                "failed",
                {
                    "code": "command_failed"
                    if case == "ordinary-failure"
                    else "readiness_context_changed",
                    "stage": "admission" if case == "admission" else "applied",
                },
            )
        )
        if case != "admission":
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done() and client._projection_lock.locked()
            assert not receipt.retired.done()
            client._receive(
                _reply(
                    client,
                    receipt,
                    sequence + 1,
                    "retired",
                    {
                        "operation_id": receipt.identity.request_id,
                        "retired": case != "negative-retirement",
                    },
                )
            )
        await _until(task.done)
        if case in {"current", "preparing"}:
            with pytest.raises(ConsoleReadinessExpired) as caught:
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
        assert client._projection is None
        assert client.projection_view is (None if case == "preparing" else publication)
        assert not client._projection_lock.locked()
        if case != "negative-retirement":
            assert client._channel_error is None
        if case == "admission":
            assert not receipt.retired.done()
    finally:
        client._fail(RuntimeError("test cleanup"))
        await asyncio.gather(*client._projection_reads, task, return_exceptions=True)
        if receipt is not None:
            await _retrieve(receipt)


@pytest.mark.asyncio
async def test_retired_stale_display_keeps_the_original_pressed_composer(monkeypatch):
    """A transport refusal must not submit ReceiveTurn or consume the real draft."""
    from textual.app import App, ComposeResult

    from Tests.Chat.test_console_execution_protocol import receive_payload
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.UI.Console_Modules import execution_send, execution_state
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    class ComposerApp(App):
        def compose(self) -> ComposeResult:
            yield ConsoleComposerBar(id="console-native-composer")

    app = ComposerApp()
    client = _ready_client()
    notices = []
    monkeypatch.setattr(app, "notify", lambda message, **_: notices.append(message))
    app.console_execution_client, app.console_runtime = client, None
    client._app = app
    receipts = []
    task = None
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.load_draft("original pressed text")
        await pilot.pause()
        stash = composer.capture_draft_for_send()
        row = SimpleNamespace(
            id="session",
            incarnation_id="incarnation",
            settings_revision=7,
            settings=ConsoleSessionSettings(provider="openai"),
        )
        client.projection_view = SimpleNamespace(
            active_session_id=row.id,
            session=lambda _: row,
        )
        client.pressed_inputs = receive_payload()["pressed_inputs"]
        screen = SimpleNamespace(
            app_instance=app,
            _console_visible_draft_session_id=row.id,
            _console_composer_or_none=lambda: composer,
            _session=SimpleNamespace(),
        )
        client._view, client._visit = screen, 7
        client.require_view = lambda view, visit: (
            view is screen and visit == 7
        ) or pytest.fail("unexpected pressed view")
        # This test isolates dispatch/transport, not attachment qualification.
        monkeypatch.setattr(execution_send, "selected_client", lambda _: client)
        monkeypatch.setattr(execution_state, "selected_client", lambda _: client)
        captured = execution_send._SelectedPress(
            client,
            screen,
            7,
            client._source_revision,
            row.id,
            row.incarnation_id,
            stash,
            (stash.text, stash.generation, stash.edit_serial),
            client.pressed_inputs,
            (),
        )
        captured.registration = (captured, client, screen)
        token = object()
        screen._console_pending_send = SimpleNamespace(
            token=token,
            received_inputs=captured,
            session_id=row.id,
            stash=stash,
        )
        try:
            task = asyncio.create_task(
                execution_send.dispatch_send(
                    screen,
                    session_id=row.id,
                    pending_send_token=token,
                )
            )
            await _until(lambda: len(client._channel.messages) == 2)
            receipts = list(client._requests.values())
            update, read = receipts
            assert (update.operation, read.operation) == (
                "UpdateInputs",
                "ReadProjection",
            )
            for receipt, kind, payload in (
                (update, "received", {"sequence": update.identity.sequence}),
                (update, "applied", {"draft_revision": 1}),
                (
                    update,
                    "retired",
                    {"operation_id": update.identity.request_id, "retired": True},
                ),
                (read, "received", {"sequence": read.identity.sequence}),
                (
                    read,
                    "failed",
                    {"code": "readiness_context_changed", "stage": "applied"},
                ),
            ):
                client._receive(
                    _reply(
                        client, receipt, client._received_sequence + 1, kind, payload
                    )
                )
            await asyncio.sleep(0)
            assert client._input_press is not None and not task.done()
            client._receive(
                _reply(
                    client,
                    read,
                    client._received_sequence + 1,
                    "retired",
                    {
                        "operation_id": read.identity.request_id,
                        "retired": True,
                    },
                )
            )
            assert await task is False
            await asyncio.gather(*client._input_tasks)
            assert composer.draft_text() == "original pressed text"
            assert client._input_press is None and client._channel_error is None
            assert [message.operation for message in client._channel.messages] == [
                "UpdateInputs",
                "ReadProjection",
            ]
            assert notices == ["Console input changed; your draft was kept."]
        finally:
            client._fail(RuntimeError("test cleanup"))
            await asyncio.gather(
                *client._input_tasks,
                *client._projection_reads,
                *((task,) if task is not None else ()),
                return_exceptions=True,
            )
            for receipt in receipts:
                await _retrieve(receipt)
