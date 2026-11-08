"""A refused UI worker leaves workflow setup retryable without starting a plan."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.mark.parametrize("remove_notice", [False, True])
async def test_refused_planning_worker_is_callable_and_keeps_setup_retryable(
    tmp_path, monkeypatch, remove_notice
):
    from textual.css.query import NoMatches
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

    def query_one(selector, *_):
        if selector not in fields:
            raise NoMatches(selector)
        return fields[selector]

    monkeypatch.setattr(screen, "query_one", query_one)
    monkeypatch.setattr(
        screen,
        "query",
        lambda selector: [fields[selector]] if selector in fields else [],
    )
    monkeypatch.setattr(screen, "_limits", lambda: (request.policy, None))

    def refuse(work):
        assert callable(work), "coroutine created before worker admission"
        admitted.append(work)
        if remove_notice:
            del fields["#workflow-notice"]
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(screen, "run_worker", refuse)
    await screen.on_button_pressed(Button.Pressed(button))
    assert len(admitted) == 1 and not button.disabled and not screen._sending
    if not remove_notice:
        assert notices[-1] != "Preparing the plan…"
    assert screen._submitted.objective == "Plan [literal] changes"
