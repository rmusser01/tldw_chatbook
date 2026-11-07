"""A refused UI worker leaves workflow setup retryable without starting a plan."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


async def test_refused_planning_worker_is_callable_and_keeps_setup_retryable(
    tmp_path, monkeypatch
):
    from textual.widgets import Button

    from Tests.Agents.workflow_support import make_workflow_request
    from tldw_chatbook.Widgets.Console.console_workflow_setup import (
        ConsoleWorkflowSetup,
    )

    request = make_workflow_request(tmp_path)
    screen = ConsoleWorkflowSetup(
        SimpleNamespace(),
        request.conversation_id,
        ("openai", "test-model"),
        request.policy,
        (((request.binding, "selected"),), (), False),
    )
    button = Button("Prepare plan", id="workflow-plan")
    notices, admitted = [], []
    fields = {
        "#workflow-binding": SimpleNamespace(value=request.binding.binding_id),
        "#workflow-target": SimpleNamespace(value="@chat"),
        "#workflow-objective": SimpleNamespace(text="Plan [literal] changes"),
        "#workflow-notice": SimpleNamespace(update=notices.append),
    }
    monkeypatch.setattr(screen, "query_one", lambda selector, *_: fields[selector])
    monkeypatch.setattr(screen, "_limits", lambda: (request.policy, None))

    def refuse(work):
        assert callable(work), "coroutine created before worker admission"
        admitted.append(work)
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(screen, "run_worker", refuse)
    await screen.on_button_pressed(Button.Pressed(button))
    assert len(admitted) == 1 and not button.disabled and not screen._sending
    assert notices[-1] != "Preparing the plan…"
    assert screen._submitted.objective == "Plan [literal] changes"
