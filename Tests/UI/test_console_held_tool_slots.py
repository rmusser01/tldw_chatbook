"""The status line's held tool slots and their list (TASK-41307)."""

import threading
from types import SimpleNamespace

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_console_rate_chip import _state
from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController
from tldw_chatbook.Widgets.Console.console_cleanup_release import RELEASED_COPY
from tldw_chatbook.Widgets.Console.console_held_tool_slots import (
    HELD_SLOTS_COPY,
    ConsoleHeldToolSlotsModal,
)
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleHeldSlotsChip,
    ConsoleStatusChips,
    ConsoleStatusText,
)

pytestmark = pytest.mark.bootstrap_profile


class _ChipsApp(App):
    CSS_PATH = str(BUNDLED_STYLESHEET)
    CSS = ".console-status-text { width: auto; }"

    def __init__(self) -> None:
        super().__init__()
        self.opened = 0

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(_state(), id="console-status-chips")

    @on(ConsoleHeldSlotsChip.OpenRequested)
    def _opened(self) -> None:
        self.opened += 1


async def test_the_held_slots_segment_shows_only_while_a_slot_is_held():
    async with _ChipsApp().run_test(size=(160, 6)) as pilot:
        strip = pilot.app.query_one(ConsoleStatusChips)
        assert not strip.segment_visible("held-slots")
        strip.sync_held_slots(2)
        await pilot.pause()
        assert strip.segment_text("held-slots") == "Held tool slots: 2"
        line = strip.query_one("#console-status-text", ConsoleStatusText)
        line.action_open_segment("held-slots")
        await pilot.pause()
        assert pilot.app.opened == 1
        strip.sync_held_slots(0)
        await pilot.pause()
        assert not strip.segment_visible("held-slots")


async def test_the_list_names_each_holder_and_releases_off_the_ui_thread():
    released = []

    def release(key):
        assert threading.current_thread() is not threading.main_thread()
        released.append(key)
        return True

    modal = ConsoleHeldToolSlotsModal(
        {"undo-owner": ("Undo", False), "run-1": ("fs_read", True)}, release
    )

    class Harness(App):
        def on_mount(self):
            self.push_screen(modal)

    async with Harness().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        texts = [str(w.render()) for w in modal.query(Static) if not isinstance(w, Button)]
        assert HELD_SLOTS_COPY in texts
        assert "The cleanup of Undo could not be proven." in texts
        assert "The cleanup of fs_read could not be proven." in texts
        # An already released slot says what stays held; no second Release.
        assert str(modal.query_one("#console-cleanup-release-run-1").render()) == (
            RELEASED_COPY
        )
        button = modal.query_one("#console-cleanup-release-undo-owner", Button)
        await pilot.click(f"#{button.id}")
        for _ in range(50):
            if str(button.label) == RELEASED_COPY:
                break
            await pilot.pause(0.02)
        assert released == ["undo-owner"]
        assert str(button.label) == RELEASED_COPY and button.disabled


def test_opening_the_list_reads_every_held_slot():
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    owner.bind_run("run-1")
    owner.reserve_tool().retain_unproven("fs_read")
    controller = SimpleNamespace(
        run_terminal_rows={},
        _hooks_v2_runtime=SimpleNamespace(_execution_capacity=capacity),
    )
    pushed = []
    agent = object.__new__(ConsoleAgentController)
    agent._chat_controller_accessor = lambda: controller
    agent._screen = SimpleNamespace(
        app=SimpleNamespace(
            push_screen=lambda screen, callback=None: pushed.append(screen)
        )
    )
    agent.open_held_tool_slots()
    (modal,) = pushed
    assert modal.slots == {"run-1": ("fs_read", False)}
    assert modal.release("run-1")
    assert capacity.snapshot().tool_workers == 0
