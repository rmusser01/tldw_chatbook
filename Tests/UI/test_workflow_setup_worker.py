"""A refused UI worker leaves workflow setup retryable without starting a plan."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Button, Select

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


@pytest.fixture(autouse=True)
def manual_workflow_reads(monkeypatch):
    from tldw_chatbook.Widgets.Console.console_workflow_setup import (
        ConsoleWorkflowSetup,
    )
    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowHistory,
        ConsoleWorkflowStatus,
    )

    for cls in (ConsoleWorkflowSetup, ConsoleWorkflowHistory, ConsoleWorkflowStatus):
        monkeypatch.setattr(cls, "on_mount", lambda self: None)


@asynccontextmanager
async def mounted(screen):
    class Host(App):
        def on_mount(self):
            self.push_screen(screen)

    async with Host().run_test() as pilot:
        await pilot.pause()
        yield pilot


def passive(cls, *args, **kwargs):
    # Keep presentation reads explicit so the test owns their completion barrier.
    return type(
        "Passive" + cls.__name__,
        (cls,),
        {"on_select_changed": lambda self, event: event.prevent_default()},
    )(*args, **kwargs)


def status_screen(coordinator):
    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowStatus,
    )

    screen = passive(
        ConsoleWorkflowStatus,
        coordinator,
        "chat",
        "workflow",
        open_changes=lambda *args: None,
        open_fleet=lambda *args: None,
        pending_approvals=lambda *args: (),
        open_approval=lambda *args: None,
    )
    screen._unsubscribe = lambda: None
    screen.request_refresh = lambda: None
    return screen


@pytest.mark.parametrize("failed", [False, True])
async def test_history_read_survives_partial_dom_removal(failed):
    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowHistory,
    )

    entered, release = asyncio.Event(), asyncio.Event()

    async def call(*args, **kwargs):
        entered.set()
        await release.wait()
        if failed:
            raise ValueError("history unavailable")
        return ()

    service = SimpleNamespace(call=call, list_history=object())
    screen = passive(
        ConsoleWorkflowHistory, SimpleNamespace(service=service), "chat", None, None
    )
    async with mounted(screen):
        task = asyncio.create_task(screen._load())
        await entered.wait()
        await screen.query_one("#workflow-open").remove()
        release.set()
        await task
        assert screen.reads.tasks == set()
        assert screen.query("#workflow-page")


@pytest.mark.parametrize("failed", [False, True])
async def test_checkpoint_read_survives_partial_dom_removal(failed):
    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowStatus,
    )

    entered, release = asyncio.Event(), asyncio.Event()

    async def call(*args):
        entered.set()
        await release.wait()
        if failed:
            raise ValueError("checkpoint unavailable")
        return SimpleNamespace(
            verifiers=(), change_review_disclosure="retained", change_window_refs=()
        )

    service = SimpleNamespace(call=call, read_checkpoint=object())
    screen = status_screen(SimpleNamespace(service=service))
    async with mounted(screen) as pilot:
        result = SimpleNamespace(
            attempt_id="result",
            status="completed",
            summary="saved",
            handoff="done",
            checks=(),
        )
        screen._visible_results = (result,)
        selector = screen.query_one("#workflow-results", Select)
        selector.set_options([("Saved", "result")])
        selector.value = "result"
        await pilot.pause()
        task = asyncio.create_task(
            ConsoleWorkflowStatus._select_result(
                screen, SimpleNamespace(value="result")
            )
        )
        await entered.wait()
        await screen.query_one("#workflow-evidence").remove()
        release.set()
        await task
        assert screen.reads.tasks == set()
        assert screen.query_one("#workflow-changes", Button).disabled
        assert screen.checkpoint is None


async def test_refresh_error_survives_partial_dom_removal():
    entered, release = asyncio.Event(), asyncio.Event()

    async def get(*args):
        entered.set()
        await release.wait()
        raise ValueError("review unavailable")

    screen = status_screen(SimpleNamespace(service=SimpleNamespace(), get=get))
    async with mounted(screen):
        screen._reading = screen._read_again = True
        task = asyncio.create_task(screen._refresh())
        await entered.wait()
        await screen.query_one("#workflow-approve").remove()
        release.set()
        await task
        assert not screen._reading and screen.decision is None
        assert screen.reads.tasks == set()


async def test_setup_completion_survives_partial_dom_removal(tmp_path):
    from Tests.Agents.workflow_support import make_workflow_request
    from tldw_chatbook.Widgets.Console.console_workflow_setup import (
        ConsoleWorkflowSetup,
    )

    entered, release = asyncio.Event(), asyncio.Event()

    async def start(*args, **kwargs):
        entered.set()
        await release.wait()
        raise ValueError("planning unavailable")

    request = make_workflow_request(tmp_path)
    screen = passive(
        ConsoleWorkflowSetup,
        SimpleNamespace(start=start),
        "chat",
        ("openai", "test-model"),
        request.policy,
        (((request.binding, 0),), (), None),
    )
    async with mounted(screen):
        screen._sending = True
        task = asyncio.create_task(
            screen._submit(request, screen._safe_mount_generation)
        )
        await entered.wait()
        await screen.query_one("#workflow-notice").remove()
        release.set()
        await task
        assert not screen._sending


@pytest.mark.parametrize("kind", ["history", "status", "setup"])
async def test_queued_button_after_its_controls_disappear(tmp_path, kind):
    from Tests.Agents.workflow_support import make_workflow_request
    from tldw_chatbook.Widgets.Console.console_workflow_setup import (
        ConsoleWorkflowSetup,
    )
    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowHistory,
    )

    if kind == "history":
        screen = passive(ConsoleWorkflowHistory, None, "chat", None, None)
        missing, action = "#workflow-saved", "workflow-open"
    elif kind == "status":
        screen = status_screen(None)
        missing, action = "#workflow-windows", "workflow-changes"
    else:
        request = make_workflow_request(tmp_path)
        screen = passive(
            ConsoleWorkflowSetup,
            None,
            "chat",
            ("openai", "test-model"),
            request.policy,
            (((request.binding, 0),), (), None),
        )
        missing, action = "#workflow-objective", "workflow-plan"
    async with mounted(screen):
        button = screen.query_one("#" + action, Button)
        await screen.query_one(missing).remove()
        await screen.on_button_pressed(Button.Pressed(button))
        assert not getattr(screen, "_sending", False)
