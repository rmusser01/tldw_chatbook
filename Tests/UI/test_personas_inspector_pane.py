"""Mounted tests for the Personas inspector pane."""

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
    PersonasInspectorPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    ConversationRowSelected,
)

pytestmark = pytest.mark.asyncio


class InspectorApp(App):
    def compose(self):
        yield PersonasInspectorPane(id="personas-inspector-pane")


async def test_default_state_shows_no_selection_and_disabled_actions():
    app = InspectorApp()
    async with app.run_test() as pilot:
        assert "Selected: none" in str(
            pilot.app.query_one("#personas-selected-name", Static).renderable
        )
        for button_id in (
            "#personas-attach-to-console",
            "#personas-start-chat",
            "#personas-export-json",
            "#personas-export-png",
            "#personas-delete",
        ):
            assert pilot.app.query_one(button_id, Button).disabled is True
        assert "Console: Blocked" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )


async def test_show_selection_enables_actions_and_shows_authority():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character", authority="Local")
        await pilot.pause()
        assert "Selected: Detective Sam" in str(
            pilot.app.query_one("#personas-selected-name", Static).renderable
        )
        assert "Authority: Local" in str(
            pilot.app.query_one("#personas-selected-authority", Static).renderable
        )
        assert pilot.app.query_one("#personas-attach-to-console", Button).disabled is False
        assert pilot.app.query_one("#personas-export-png", Button).disabled is False
        assert "Console: Ready" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )


async def test_persona_selection_disables_png_export():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Archivist", kind="persona_profile", authority="Local")
        await pilot.pause()
        assert pilot.app.query_one("#personas-export-json", Button).disabled is False
        assert pilot.app.query_one("#personas-export-png", Button).disabled is True


async def test_unsaved_disables_attach_and_export_with_reason():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Tutor", kind="character", authority="Local")
        pane.set_unsaved(True)
        await pilot.pause()
        attach = pilot.app.query_one("#personas-attach-to-console", Button)
        assert attach.disabled is True
        assert "unsaved" in str(attach.tooltip).lower()
        assert pilot.app.query_one("#personas-export-json", Button).disabled is True
        assert "Console: Blocked - unsaved edits" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )
        pane.set_unsaved(False)
        await pilot.pause()
        assert pilot.app.query_one("#personas-attach-to-console", Button).disabled is False


async def test_show_validation_errors_renders_messages_and_clears():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_validation(("name: required", "first_message: required"))
        await pilot.pause()
        summary = str(pilot.app.query_one("#personas-validation-summary", Static).renderable)
        assert "name: required" in summary
        assert "first_message: required" in summary
        pane.show_validation(())
        await pilot.pause()
        assert "Validation: OK" in str(
            pilot.app.query_one("#personas-validation-summary", Static).renderable
        )


async def test_conversations_panel_rows_post_selection():
    received = []

    class CaptureApp(InspectorApp):
        def on_conversation_row_selected(self, message: ConversationRowSelected) -> None:
            received.append(message.conversation_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        await pane.show_conversations((("conv-1", "First case"), ("conv-2", "Cold trail")))
        await pilot.pause()
        assert len(pilot.app.query(".personas-conversation-row")) == 2
        await pilot.click("#personas-conversation-row-conv-1")
        await pilot.pause()
    assert received == ["conv-1"]


async def test_conversation_click_after_rerender_posts_new_id():
    received = []

    class CaptureApp(InspectorApp):
        def on_conversation_row_selected(self, message: ConversationRowSelected) -> None:
            received.append(message.conversation_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        await pane.show_conversations((("conv-1", "First case"),))
        await pane.show_conversations((("conv-9", "New case"),))
        await pilot.pause()
        await pilot.click("#personas-conversation-row-conv-9")
        await pilot.pause()
    assert received == ["conv-9"]


async def test_clear_selection_resets_everything():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character", authority="Server")
        await pane.show_conversations((("conv-1", "First case"),))
        pane.set_unsaved(True)
        await pilot.pause()
        await pane.clear_selection()
        await pilot.pause()
        assert "Selected: none" in str(
            pilot.app.query_one("#personas-selected-name", Static).renderable
        )
        assert (
            str(pilot.app.query_one("#personas-selected-authority", Static).renderable)
            == "Authority: Local"
        )
        assert pilot.app.query_one("#personas-attach-to-console", Button).disabled is True
        assert len(pilot.app.query(".personas-conversation-row")) == 0
        assert "Validation: OK" in str(
            pilot.app.query_one("#personas-validation-summary", Static).renderable
        )


async def test_show_conversations_twice_in_same_tick_does_not_crash():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        await pane.show_conversations((("conv-1", "First case"),))
        await pane.show_conversations((("conv-2", "Cold trail"),))
        await pilot.pause()
        rows = pilot.app.query(".personas-conversation-row")
        assert len(rows) == 1
        assert str(rows.first().label) == "Cold trail"
