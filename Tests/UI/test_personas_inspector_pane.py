"""Mounted tests for the Personas inspector pane."""

import pytest
from textual.app import App
from textual.widgets import Button, ListItem, ListView, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
    PersonasInspectorPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    ConversationRowSelected,
)

pytestmark = pytest.mark.asyncio


def _row_text(item: ListItem) -> str:
    """Visible text of a conversation row (the ListItem's inner Static)."""
    return str(item.query_one(Static).renderable)


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
        assert "Console blocked" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )


async def test_readiness_copy_is_compact_for_narrow_inspector():
    app = InspectorApp()
    async with app.run_test(size=(24, 20)) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        readiness = pilot.app.query_one("#personas-readiness-console", Static)

        default_copy = str(readiness.renderable)
        assert default_copy == "Console blocked: select an item"
        assert " - " not in default_copy

        pane.set_console_actions_enabled(False, reason="prompts are not attachable")
        await pilot.pause()

        blocked_copy = str(readiness.renderable)
        assert blocked_copy == "Console blocked: prompts are not attachable"
        assert " - " not in blocked_copy


async def test_action_buttons_carry_shared_flat_button_classes():
    app = InspectorApp()
    async with app.run_test() as pilot:
        for button_id in ("#personas-attach-to-console", "#personas-start-chat"):
            assert pilot.app.query_one(button_id, Button).has_class(
                "console-action-secondary"
            )
        for button_id in ("#personas-export-json", "#personas-export-png"):
            assert pilot.app.query_one(button_id, Button).has_class(
                "console-action-subdued"
            )
        delete = pilot.app.query_one("#personas-delete", Button)
        assert delete.has_class("console-action-subdued")
        assert delete.has_class("personas-destructive")


async def test_conversations_list_is_height_capped():
    """The conversations list must never push readiness/actions off-pane."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        styles = pilot.app.query_one("#personas-conversations-list").styles
        assert styles.max_height is not None
        assert styles.max_height.value <= 10


async def test_conversation_rows_carry_subdued_class():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        await pane.show_conversations((("conv-1", "First case"),))
        await pilot.pause()
        row = pilot.app.query_one("#personas-conversation-row-conv-1", ListItem)
        assert row.has_class("personas-conversation-row")
        assert row.has_class("console-action-subdued")


async def test_show_selection_enables_export_and_shows_authority():
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
        assert pilot.app.query_one("#personas-attach-to-console", Button).disabled is True
        assert pilot.app.query_one("#personas-start-chat", Button).disabled is True
        assert pilot.app.query_one("#personas-export-json", Button).disabled is False
        assert pilot.app.query_one("#personas-export-png", Button).disabled is False
        assert "Console blocked: select an item" in str(
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


async def test_console_action_enablement_is_explicitly_screen_owned():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Tutor", kind="character", authority="Local")
        await pilot.pause()

        # Selection/export state is inspector-local, but Console attach/start
        # availability is pushed by PersonasScreen from _console_action_allowed().
        assert pilot.app.query_one("#personas-export-json", Button).disabled is False
        assert pilot.app.query_one("#personas-delete", Button).disabled is False
        assert pilot.app.query_one("#personas-attach-to-console", Button).disabled is True
        assert pilot.app.query_one("#personas-start-chat", Button).disabled is True
        assert "Console blocked: select an item" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )

        pane.set_console_actions_enabled(True)
        await pilot.pause()

        assert pilot.app.query_one("#personas-attach-to-console", Button).disabled is False
        assert pilot.app.query_one("#personas-start-chat", Button).disabled is False
        assert "Console ready" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )

        pane.set_console_actions_enabled(False, reason="prompts are not attachable")
        await pilot.pause()

        assert pilot.app.query_one("#personas-export-json", Button).disabled is False
        assert pilot.app.query_one("#personas-delete", Button).disabled is False
        assert pilot.app.query_one("#personas-attach-to-console", Button).disabled is True
        assert pilot.app.query_one("#personas-start-chat", Button).disabled is True
        assert "Console blocked: prompts are not attachable" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )


async def test_unsaved_disables_attach_and_export_with_reason():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Tutor", kind="character", authority="Local")
        pane.set_console_actions_enabled(True)
        pane.set_unsaved(True)
        pane.set_console_actions_enabled(False, reason="unsaved edits")
        await pilot.pause()
        attach = pilot.app.query_one("#personas-attach-to-console", Button)
        assert attach.disabled is True
        assert "unsaved" in str(attach.tooltip).lower()
        assert pilot.app.query_one("#personas-export-json", Button).disabled is True
        assert "Console blocked: unsaved edits" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )
        pane.set_unsaved(False)
        pane.set_console_actions_enabled(True)
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


async def test_conversation_list_arrow_enter():
    """Down/Down highlights without selecting; Enter posts the row's id."""
    received = []

    class CaptureApp(InspectorApp):
        def on_conversation_row_selected(self, message: ConversationRowSelected) -> None:
            received.append(message.conversation_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        await pane.show_conversations(
            (("conv-1", "First case"), ("conv-2", "Cold trail"), ("conv-3", "Closed file"))
        )
        await pilot.pause()
        list_view = pilot.app.query_one("#personas-conversations-list", ListView)
        list_view.focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()
        # Arrow browsing must not open a conversation.
        assert received == []
        await pilot.press("enter")
        await pilot.pause()
    assert received == ["conv-2"]


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
        assert _row_text(rows.first(ListItem)) == "Cold trail"
