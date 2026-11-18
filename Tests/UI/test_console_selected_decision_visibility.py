"""A dismissed project modal owes actual visibility before another head read."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_selected_decision_wake import _deliver, _hint, _scope
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.UI.Console_Modules import execution_decisions, execution_state


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["current", "negative", "source"])
async def test_project_answer_holds_hints_through_resolution_and_visible_full(
    monkeypatch, outcome,
):
    # UI orchestration doubles; native proof belongs to the actual backend tests.
    screen, client, _, scheduled, workers = _scope(monkeypatch)
    row = client.projection_view.session(client.projection_view.active_session_id)
    head = SimpleNamespace(
        decision_id="notice", kind="dispatch", session_id=row.id,
        incarnation_id=row.incarnation_id,
    )
    modal = SimpleNamespace(_console_process_modal_owner=None)
    screen._session = SimpleNamespace(_project_instruction_decision_modals={head.decision_id: modal})
    state = execution_decisions._ProjectModal(
        client, screen, client._visit, client._source_revision, head, modal,
        attachment_id=client._attachment_id,
    )
    client._project_modal = state
    modal._console_process_modal_owner = state
    monkeypatch.setattr(execution_decisions, "selected_client", lambda _: client)
    monkeypatch.setattr(execution_decisions, "_require_methods", lambda *_: None)
    commands, tasks = [], []
    original_worker = screen.run_worker

    def submit(operation, payload, **scope):
        assert scope == {"session_id": row.id, "incarnation_id": row.incarnation_id}
        receipt = SimpleNamespace(
            applied=asyncio.get_running_loop().create_future(),
            retired=asyncio.get_running_loop().create_future(),
        )
        commands.append((operation, payload, receipt))
        return receipt

    def start(work, **kwargs):
        if kwargs["group"] != "console-process-decision":
            return original_worker(work, **kwargs)
        task = asyncio.create_task(work)
        tasks.append(task)
        return task

    client.submit, screen.run_worker = submit, start
    entered, release = asyncio.Event(), asyncio.Event()

    async def visible(*_, checkpoint):
        assert checkpoint == "visible"
        entered.set()
        await release.wait()
        return True

    monkeypatch.setattr(execution_state, "refresh_selected_view", visible)
    execution_decisions._finish_project_modal(state, "proceed")
    assert [item[0] for item in commands] == ["ProjectDecisionPresentation", "ResolveDecision"]
    assert client._project_modal is None
    hint = _hint(client)
    assert execution_state.request_selected_decision_refresh(client, hint)
    _deliver(scheduled)
    assert not workers and client._decision_rendered_hint is None
    hidden, answer = commands[0][2], commands[1][2]
    hidden.applied.set_result({"applied": False})
    hidden.retired.set_result(True)
    answer.applied.set_result({"applied": True})
    try:
        await asyncio.sleep(0)
        assert not entered.is_set()
        answer.retired.set_result(outcome != "negative")
        if outcome == "negative":
            with pytest.raises(ConsoleChannelLost, match="retirement"):
                await tasks[-1]
            assert not entered.is_set()
            assert execution_decisions.pending_project_visibility(client, screen, client._visit) is not None
            return
        await asyncio.wait_for(entered.wait(), 5)
        later = _hint(client, 2)
        assert execution_state.request_selected_decision_refresh(client, later)
        _deliver(scheduled)
        assert not workers and client._decision_rendered_hint is None
        if outcome == "source":
            client._source_revision += 1
        release.set()
        if outcome == "source":
            with pytest.raises(ConsoleChannelLost, match="visibility"):
                await tasks[-1]
            assert not workers and client._decision_rendered_hint is None
        else:
            await tasks[-1]
            assert client._project_modal_transition is None
            assert len(workers) == 1
            await workers[0].work()
            assert client._decision_rendered_hint == later
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
