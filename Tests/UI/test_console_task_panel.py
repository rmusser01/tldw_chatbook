"""PRD Feature B (milestone M1): the pinned Console task panel.

Covers the pure renderer, the widget's show/hide/collapse behaviour under
the real consolidated CSS, and the controller/runtime wiring that feeds it
(the ``todo_*`` change callback on the worker thread, the session
activation re-derives on the UI thread, and the re-derive at view attach).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from Tests.Chat.test_console_runtime_lifetime import _runtime_with, _View

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Widgets.Console.console_task_panel import (
    ConsoleTaskPanel,
    render_task_lines,
)


def _tasks() -> list[dict[str, object]]:
    return [
        {"id": "1", "content": "Read the schema", "status": "completed"},
        {
            "id": "2",
            "content": "Write the migration",
            "status": "in_progress",
            "activeForm": "Writing the migration",
        },
        {"id": "3", "content": "Run the DB tests", "status": "pending"},
    ]


# --- pure renderer ---------------------------------------------------------


def test_render_header_counts_done_and_names_the_active_form():
    header, rows = render_task_lines(_tasks())
    assert header == "▾ Tasks · 1 of 3 done · Writing the migration"
    assert rows == [
        ("completed", "[x] Read the schema"),
        ("in_progress", "[~] Writing the migration"),
        ("pending", "[ ] Run the DB tests"),
    ]


def test_render_collapsed_flips_the_chevron_and_omits_active_when_none():
    header, _ = render_task_lines(
        [{"content": "a", "status": "pending"}], collapsed=True
    )
    assert header == "▸ Tasks · 0 of 1 done"


def test_render_sanitises_labels_like_the_transcript_marker():
    _, rows = render_task_lines(
        [{"content": "line one\nline two\x07", "status": "pending"}]
    )
    (row,) = rows
    assert "\n" not in row[1] and "\x07" not in row[1]
    assert row[1].startswith("[ ] line one")


# --- widget under the real CSS ---------------------------------------------


class _Harness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleTaskPanel(id="console-task-panel")


def _rows(panel: ConsoleTaskPanel) -> list[str]:
    return str(panel.query_one("#console-task-panel-rows", Static).render()).splitlines()


@pytest.mark.asyncio
async def test_panel_hidden_until_tasks_arrive_then_hidden_again_when_cleared():
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(ConsoleTaskPanel)
        assert panel.display is False, "AC-B1: no panel without tasks"

        panel.set_tasks("s1", _tasks())
        await pilot.pause()
        assert panel.display is True
        header = panel.query_one("#console-task-panel-header", Static)
        assert "1 of 3 done" in str(header.render())
        assert "Writing the migration" in str(header.render())
        assert _rows(panel) == [
            "[x] Read the schema",
            "[~] Writing the migration",
            "[ ] Run the DB tests",
        ]

        panel.set_tasks("s1", [])
        await pilot.pause()
        assert panel.display is False


@pytest.mark.asyncio
async def test_back_to_back_snapshots_render_only_the_newest():
    """Repaint is a synchronous ``update()``: no remove/mount cycle to race."""
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(ConsoleTaskPanel)
        for n in range(1, 6):
            panel.set_tasks(
                "s1",
                [{"content": f"task {i}", "status": "pending"} for i in range(n)],
            )
        await pilot.pause()
        assert _rows(panel) == [f"[ ] task {i}" for i in range(5)]


@pytest.mark.asyncio
async def test_collapse_is_remembered_per_session_and_survives_updates():
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(ConsoleTaskPanel)
        panel.set_tasks("s1", _tasks())
        await pilot.pause()
        body = panel.query_one("#console-task-panel-body")
        assert body.display is True

        await pilot.click("#console-task-panel-header")
        await pilot.pause()
        assert body.display is False, "click on the header collapses"
        assert str(
            panel.query_one("#console-task-panel-header", Static).render()
        ).startswith("▸")

        # A live update keeps the collapsed state (AC-B6).
        panel.set_tasks("s1", _tasks()[:2])
        await pilot.pause()
        assert body.display is False

        # Another session starts expanded; coming back restores collapsed.
        panel.set_tasks("s2", _tasks()[:1])
        await pilot.pause()
        assert body.display is True
        panel.set_tasks("s1", _tasks())
        await pilot.pause()
        assert body.display is False


# --- controller and runtime wiring -------------------------------------------


def _wired_controller() -> tuple[ConsoleChatController, list[tuple[str | None, list]]]:
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=None)
    calls: list[tuple[str | None, list]] = []
    controller.app = SimpleNamespace(call_from_thread=lambda fn, *args: fn(*args))
    controller.set_task_panel = lambda session_id, tasks: calls.append(
        (session_id, list(tasks))
    )
    return controller, calls


def test_todo_change_feeds_the_panel_alongside_the_transcript_marker():
    controller, calls = _wired_controller()
    session = controller.new_session(title="t")
    calls.clear()
    markers: list[tuple[str, list]] = []
    controller._agent_bridge = SimpleNamespace(
        append_todo_marker=lambda sid, tasks: markers.append((sid, tasks))
    )

    wiring = controller._todo_wiring(session.id)
    snapshot = [{"id": "1", "content": "x", "status": "pending"}]
    wiring["on_todo_change"](snapshot)

    assert markers == [(session.id, snapshot)], "transcript marker still fires"
    assert calls == [(session.id, snapshot)], "AC-B4: panel sink fires too"


def test_session_activation_re_derives_the_panel_from_the_todo_store():
    controller, calls = _wired_controller()
    first = controller.new_session(title="first")
    assert calls[-1] == (first.id, []), "a new session clears the panel"
    first.todo_store.create(content="only in first")

    second = controller.new_session(title="second")
    assert calls[-1] == (second.id, [])

    controller.switch_session(first.id)
    session_id, tasks = calls[-1]
    assert session_id == first.id
    assert [t["content"] for t in tasks] == ["only in first"], "AC-B5"

    controller.set_task_panel = None
    controller._remount_task_panel(first.id)  # viewless: no-op, no raise


def test_closing_the_last_session_clears_the_panel():
    """No neighbour to activate still has to take the departed tasks down."""
    controller, calls = _wired_controller()
    only = controller.new_session(title="only")
    only.todo_store.create(content="doomed")
    controller._remount_task_panel(only.id)
    assert calls[-1][1], "precondition: the panel shows the task"

    controller.close_session(only.id)
    assert controller.store.active_session_id is None
    assert calls[-1] == (None, [])


def test_attaching_a_new_view_pushes_the_active_sessions_tasks():
    """The runtime outlives the screen; a fresh panel must not start empty."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=None)
    session = controller.new_session(title="kept")
    session.todo_store.create(content="survives navigation")

    calls: list[tuple[str | None, list]] = []
    view = _View(
        {"set_task_panel": lambda sid, tasks: calls.append((sid, list(tasks)))}
    )
    _runtime_with(controller, view)

    assert calls, "attach must re-derive the panel"
    session_id, tasks = calls[-1]
    assert session_id == session.id
    assert [t["content"] for t in tasks] == ["survives navigation"]
