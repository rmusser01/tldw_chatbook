"""Exact transport retirement isolates only a periodic display observer."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_execution_process import _ready_client, _reply, _retrieve
from Tests.Chat.test_console_execution_workspace import _until
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost, ConsoleRemoteFailure


def _current_client():
    client = _ready_client()
    view = object()
    client._view, client._visit = view, 3
    app = SimpleNamespace(console_execution_client=client, console_runtime=None)
    client._app = app
    row = SimpleNamespace(incarnation_id="incarnation")
    client.projection_view = SimpleNamespace(active_session_id="session", session=lambda _: row)

    def require_view(expected, visit):
        if (client._view is not expected or client._visit != visit
            or client._channel_error is not None):
            raise ConsoleChannelLost("Original view check refused")

    client.require_view = require_view
    return client, view, app, row


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "current", "source", "view", "visit", "attachment", "incarnation", "session",
    "app", "preparation", "publication", "channel", "negative-retirement",
    "admission", "no-received", "mount", "visible", "history", "inspector", "wrong-code",
])
async def test_passive_failure_requires_exact_completed_plain_read_and_current_scope(case):
    client, view, app, row = _current_client()
    options = (
        {"_checkpoint": case} if case in {"mount", "visible"} else
        {"_run_history": {"conversation_id": "saved", "include_superseded": False,
                          "limit": 1, "agent_kind": None}} if case == "history" else
        {"_inspector_refresh": "dictionary"} if case == "inspector" else {}
    )
    task = asyncio.create_task(client.read_projection("session", "incarnation", **options))
    receipt = None
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        sequence = 1
        if case not in {"admission", "no-received"}:
            client._receive(_reply(client, receipt, sequence, "received", {
                "sequence": receipt.identity.sequence,
            }))
            sequence += 1
        client._receive(_reply(client, receipt, sequence, "failed", {
            "code": "view_checkpoint_superseded" if case == "wrong-code" else "command_failed",
            "stage": "admission" if case == "admission" else "applied",
        }))
        if case != "admission":
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done() and client._projection_lock.locked()
            client._receive(_reply(client, receipt, sequence + 1, "retired", {
                "operation_id": receipt.identity.request_id,
                "retired": case != "negative-retirement",
            }))
        if case == "negative-retirement":
            with pytest.raises(ConsoleChannelLost, match="retirement is unproven"):
                await task
            assert client._channel_error is not None
            return
        with pytest.raises(ConsoleRemoteFailure) as caught:
            await task  # Authored/direct callers still receive this exact refusal.
        error = caught.value
        assert error is receipt.applied.exception()
        assert client._projection is None and client._readiness_deadline == 0
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
        elif case == "session":
            client.projection_view.active_session_id = "replacement"
        elif case == "app":
            app.console_execution_client = object()
        elif case == "preparation":
            client._view_preparation = object()
        elif case == "publication":
            client.projection_view = SimpleNamespace(active_session_id="session", session=lambda _: row)
        elif case == "channel":
            client._channel_error = RuntimeError("Actual channel loss")
        if case == "current":
            assert client.require_passive_projection_failure(error, view, 3) is None
        else:
            with pytest.raises((ConsoleRemoteFailure, ConsoleChannelLost)):
                client.require_passive_projection_failure(error, view, 3)
    finally:
        client._fail(RuntimeError("test cleanup"))
        await asyncio.gather(*client._projection_reads, task, return_exceptions=True)
        if receipt is not None:
            await _retrieve(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["passive", "explicit", "mount", "source", "definitions", "unproven"])
async def test_reconciliation_reports_only_qualified_passive_failure(monkeypatch, case):
    from Tests.UI.test_console_selected_refresh_coverage import _visit
    from tldw_chatbook.UI.Console_Modules import execution_state

    screen, client, _app = _visit(monkeypatch)
    error = ConsoleRemoteFailure("command_failed", "applied")
    verified = []
    client.require_view = lambda *_: None
    client.require_passive_projection_failure = lambda *args: verified.append(args)

    async def refused(*_, **__):
        if case == "source":
            client._source_revision += 1
        if case == "definitions":
            def changed_sources():
                raise RuntimeError("Original source check refused")

            monkeypatch.setattr(execution_state, "_require_sources", changed_sources)
        if case == "unproven":
            raise ConsoleChannelLost("Console projection retirement is unproven")
        raise error

    if case == "mount":
        screen._console_process_mounted_visit = None
    monkeypatch.setattr(execution_state, "refresh_selected_view", refused)
    if case == "passive":
        await execution_state.reconcile_selected_view(screen, passive=True)
        assert len(verified) == 1 and verified[0] == (error, screen, client._visit)
        assert screen._console_passive_projection_failure[-2:] == ("command_failed", "applied")
        assert not screen._console_process_surface_reconciled
        assert client._readiness_deadline == 0
    else:
        expected = (ConsoleChannelLost if case == "unproven" else
                    RuntimeError if case == "definitions" else ConsoleRemoteFailure)
        with pytest.raises(expected):
            await execution_state.reconcile_selected_view(screen, passive=case != "explicit")
        assert not verified
        assert getattr(screen, "_console_passive_projection_failure", None) is None
    assert not screen._console_attach_reconcile_running


@pytest.mark.asyncio
@pytest.mark.parametrize("refreshed", [False, True])
async def test_passive_failure_state_clears_only_after_successful_checked_refresh(monkeypatch, refreshed):
    from Tests.UI.test_console_selected_refresh_coverage import _visit
    from tldw_chatbook.UI.Console_Modules import execution_state

    screen, _client, _app = _visit(monkeypatch)
    failure = (object(),)
    screen._console_passive_projection_failure = failure
    screen.call_later = lambda *_: None

    async def completed(*_, **__):
        return refreshed

    monkeypatch.setattr(execution_state, "_refresh_current_selected_view", completed)
    assert await execution_state.refresh_selected_view(screen) is refreshed
    assert screen._console_passive_projection_failure is (None if refreshed else failure)
