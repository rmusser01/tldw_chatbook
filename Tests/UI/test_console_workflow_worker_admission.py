"""Workflow UI allocates coroutines only after Textual admits its workers."""

import asyncio
import gc
import warnings
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Button, Input, TextArea
from textual.worker import WorkerState

from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
from tldw_chatbook.Agents.goal_models import GoalBindingRef
from tldw_chatbook.UI.Console_Modules import workflows
from tldw_chatbook.Widgets.Console.console_workflow_setup import ConsoleWorkflowSetup
from tldw_chatbook.Widgets.Console.console_workflow_status import (
    ConsoleWorkflowHistory,
    ConsoleWorkflowStatus,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize(
    "admission,operation",
    [
        (admission, operation)
        for admission in ("cancel", "refuse", "enter")
        for operation in (
            "select",
            "published",
            "setup",
            "submit",
            "history",
            "refresh",
            "act",
        )
    ]
    + [("replace", operation) for operation in ("select", "published", "setup")],
)
async def test_workflow_ui_defers_coroutines_until_native_worker_entry(
    monkeypatch,
    operation,
    admission,
):
    binding = GoalBindingRef(
        workspace_id="workspace", binding_id="folder", locator="/tmp", access="rw"
    )
    session = SimpleNamespace(
        id="session",
        persisted_conversation_id="chat",
        workspace_id="workspace",
        ephemeral=False,
        runtime_backend="local",
        assistant_kind="assistant",
    )
    choices = (((binding, 0),), (), None)
    calls = []
    full = SimpleNamespace(id="workflow", status="paused", control_revision=7)

    class Coordinator:
        action_ready = False

        def __init__(self):
            self.service = self
            self.controller = SimpleNamespace(
                store=SimpleNamespace(
                    sessions=lambda: [session], active_session_id="session"
                ),
                _provider_selection_for_session=lambda _: SimpleNamespace(
                    provider="openai", explicit_model="test", configured_model="test"
                ),
            )

        def subscribe(self, callback):
            return lambda: None

        async def call(self, method, *args, **kwargs):
            return method(*args, **kwargs)

        def list_history(self, chat, **kwargs):
            calls.append(("history", chat))
            return [{"id": "workflow", "status": "paused"}]

        def setup_choices(self, workspace):
            calls.append(("choices", workspace))
            return choices

        async def get(self, workflow):
            if operation in {"refresh", "act"} and not self.action_ready:
                raise ValueError("revision_conflict")
            return full

        async def start(self, setup, *, launch_id):
            calls.append(("start", setup.conversation_id, setup.objective, launch_id))
            return full

        async def revise(
            self, workflow, *, expected_control_revision, instruction, command_id
        ):
            self.action_ready = False
            calls.append(
                ("revise", workflow, expected_control_revision, instruction, command_id)
            )
            return SimpleNamespace(control_revision=8)

    coordinator = Coordinator()
    host = App()
    monkeypatch.setattr(workflows, "workflows_enabled", lambda: True)
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_workflow_runs.native_workflow_qualified",
        lambda: True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        async with host.run_test(size=(100, 40)) as pilot:
            screen = host.screen
            await screen.mount(workflows.ConsoleWorkflowLine())
            screen._console_runtime = lambda: SimpleNamespace(
                chat_controller=coordinator.controller,
                ensure_workflow_coordinator=lambda: coordinator,
            )
            bridge = workflows.ConsoleWorkflowsController(screen)
            modal = None
            if operation == "submit":
                modal = ConsoleWorkflowSetup(
                    coordinator,
                    "chat",
                    ("openai", "test"),
                    AutomaticWorkLimits(),
                    choices,
                )
            elif operation == "history":
                modal = ConsoleWorkflowHistory(
                    coordinator, "chat", lambda *_: None, lambda *_: None
                )
            elif operation in {"refresh", "act"}:
                modal = ConsoleWorkflowStatus(
                    coordinator,
                    "chat",
                    "workflow",
                    open_changes=lambda *_: None,
                    open_fleet=lambda *_: None,
                    pending_approvals=lambda *_: False,
                    open_approval=lambda *_: None,
                )
            if modal is not None:
                await host.push_screen(modal)
                await pilot.pause()
                # Initial mount reads finish before testing a fresh submission.
                async with asyncio.timeout(2):
                    while any(w.node is modal for w in host.workers):
                        await asyncio.sleep(0)
            owner = modal or screen
            if operation == "submit":
                modal.query_one("#workflow-objective", TextArea).load_text(
                    "Captured goal"
                )
                await pilot.pause()
            if operation == "act":
                modal.snapshot = full
                coordinator.action_ready = True
                modal.query_one(
                    "#workflow-instruction", Input
                ).value = "Captured revision"
                modal.query_one("#workflow-revise", Button).disabled = False
            calls.clear()
            submitted = []
            original = owner.run_worker

            def schedule(work, *args, **kwargs):
                if admission == "refuse":
                    raise RuntimeError("worker submission refused")
                worker = original(work, *args, **kwargs)
                submitted.append(worker)
                if admission == "cancel":
                    worker.cancel()
                return worker

            monkeypatch.setattr(owner, "run_worker", schedule)

            async def request():
                if operation == "select":
                    bridge.select_chat(session)
                elif operation == "published":
                    bridge._active_id = "chat"
                    bridge._background = SimpleNamespace(id="background")
                    screen.query_one("#console-workflow-line").target = (
                        "other",
                        "background",
                    )
                    bridge._published(
                        SimpleNamespace(
                            id="background", conversation_id="other", status="completed"
                        )
                    )
                elif operation == "setup":
                    bridge.open_setup()
                elif operation == "submit":
                    await modal.on_button_pressed(
                        Button.Pressed(modal.query_one("#workflow-plan", Button))
                    )
                elif operation == "history":
                    modal.on_mount()
                elif operation == "refresh":
                    modal.request_refresh()
                else:
                    await modal.on_button_pressed(
                        Button.Pressed(modal.query_one("#workflow-revise", Button))
                    )

            if admission == "refuse":
                with pytest.raises(RuntimeError, match="worker submission refused"):
                    await request()
            else:
                await request()
                if admission == "replace":
                    bridge._active_id = None
                    await request()
                    assert submitted[0].is_cancelled
                    assert submitted[0].group == submitted[1].group
                worker = submitted[-1]
                async with asyncio.timeout(2):
                    await asyncio.gather(worker._task, return_exceptions=True)
                await pilot.pause()
                if admission == "cancel":
                    assert worker.is_cancelled
                    assert not calls
                else:
                    assert worker.state == WorkerState.SUCCESS
                    if operation in {"select", "published"}:
                        assert calls == [("history", "chat")]
                        assert screen.query_one("#console-workflow-line").target == (
                            "chat",
                            "workflow",
                        )
                    elif operation == "setup":
                        assert calls == [("choices", "workspace")]
                        assert isinstance(host.screen, ConsoleWorkflowSetup)
                    elif operation == "submit":
                        assert calls == [
                            ("start", "chat", "Captured goal", modal.launch_id)
                        ]
                    elif operation == "history":
                        assert calls == [("history", "chat")]
                        assert not modal.query_one("#workflow-open", Button).disabled
                    elif operation == "refresh":
                        assert "revision_conflict" in str(
                            modal.query_one("#workflow-notice").render()
                        )
                    else:
                        assert calls[0][:4] == (
                            "revise",
                            "workflow",
                            7,
                            "Captured revision",
                        )
                        assert "Revision received" in modal._action_notice
                submitted.clear()
                del worker
            await bridge.detach()
            gc.collect()
        gc.collect()
    leaked = [
        str(item.message)
        for item in caught
        if issubclass(item.category, RuntimeWarning)
    ]
    assert not leaked, leaked
