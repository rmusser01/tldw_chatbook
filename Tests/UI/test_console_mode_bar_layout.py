"""Unchanged hidden mode copy must not invalidate native layout."""

import pytest
from textual.widgets import Static

from Tests.UI.test_console_mcp_approval import (
    _build_ready_production_console_app,
    _wait_for_production_console_ready,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusChips

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
async def test_equal_hidden_mode_copy_keeps_visual_and_layout(size, monkeypatch):
    app = _build_ready_production_console_app()
    async with app.run_test(size=size) as pilot:
        screen = await _wait_for_production_console_ready(app, pilot)
        screen._sync_console_mode_bar()
        mode = screen.query_one("#console-mode-bar", Static)
        chips = screen.query_one("#console-status-chips", ConsoleStatusChips)
        copy, visual, focused = mode.content, mode.visual, screen.focused
        native_update, native_chip = mode.update, chips.sync_run_chip
        updates, projections = [], []

        def update(content="", *, layout=True):
            updates.append((content, layout))
            return native_update(content, layout=layout)

        def sync_chip(visible, text):
            projections.append((visible, text))
            return native_chip(visible, text)

        monkeypatch.setattr(mode, "update", update)
        monkeypatch.setattr(chips, "sync_run_chip", sync_chip)
        for _ in range(3):
            screen._sync_console_mode_bar()
        assert updates == []
        assert len(projections) == 3
        assert mode.content == copy and mode.visual is visual
        assert mode.display is False and screen.focused is focused

        controller = screen._ensure_console_chat_controller()
        for state, text, visible in (
            (ConsoleRunStatus.STREAMING, "Native run in progress.", True),
            (ConsoleRunStatus.COMPLETED, "Native run complete.", False),
        ):
            controller._set_run_state(ConsoleRunState(state, text))
            count = len(updates)
            screen._sync_console_mode_bar()
            assert len(updates) == count + 1
            assert f"Run: {text}" in mode.content
            assert screen.query_one("#console-run-chip").display is visible
            if visible:
                assert (
                    str(screen.query_one("#console-run-chip").render())
                    == f"Run: {text}"
                )
            screen._sync_console_mode_bar()
            assert len(updates) == count + 1
        assert len(projections) == 7
        assert mode.display is False and screen.focused is focused
