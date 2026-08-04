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
        assert str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        ) == "Pick a character or persona to start chatting."


async def test_no_selection_shows_single_guidance_line():
    """F-031: pre-selection the inspector is one plain guidance line - no
    wall of disabled buttons, no dangling section headers, no false
    "Validation: OK"."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        assert (
            pilot.app.query_one("#personas-inspector-actions").display is False
        )
        assert (
            pilot.app.query_one("#personas-conversations-header", Static).display
            is False
        )
        assert (
            pilot.app.query_one("#personas-conversations-list", ListView).display
            is False
        )
        assert (
            pilot.app.query_one("#personas-readiness-header", Static).display
            is False
        )
        assert (
            pilot.app.query_one("#personas-validation-summary", Static).display
            is False
        )
        guidance = pilot.app.query_one("#personas-readiness-console", Static)
        assert guidance.display is True
        assert (
            str(guidance.renderable)
            == "Pick a character or persona to start chatting."
        )


async def test_selection_reveals_inspector_sections():
    """F-031: selecting an item wakes the whole inspector back up."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pilot.pause()
        assert (
            pilot.app.query_one("#personas-inspector-actions").display is True
        )
        assert (
            pilot.app.query_one("#personas-conversations-header", Static).display
            is True
        )
        assert (
            pilot.app.query_one("#personas-conversations-list", ListView).display
            is True
        )
        assert (
            pilot.app.query_one("#personas-readiness-header", Static).display
            is True
        )
        assert (
            pilot.app.query_one("#personas-validation-summary", Static).display
            is True
        )


async def test_validation_line_stays_hidden_until_first_selection():
    """F-031: a fresh inspector must not claim "Validation: OK" for nothing."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        summary = pilot.app.query_one("#personas-validation-summary", Static)
        assert summary.display is False
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Tutor", kind="character")
        await pilot.pause()
        assert summary.display is True
        await pane.clear_selection()
        await pilot.pause()
        assert summary.display is False


async def test_character_with_no_conversations_shows_empty_copy():
    """F-036: selected-but-no-conversations renders the empty-state copy
    instead of a bare header over nothing."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations((), empty_copy="No saved conversations.")
        await pilot.pause()
        assert (
            pilot.app.query_one("#personas-conversations-header", Static).display
            is True
        )
        assert (
            pilot.app.query_one("#personas-conversations-list", ListView).display
            is True
        )
        texts = [
            str(s.renderable)
            for s in pilot.app.query_one("#personas-conversations-list").query(Static)
        ]
        assert any("No saved conversations." in text for text in texts)


async def test_non_character_selections_hide_conversations_section():
    """F-036: personas/dictionaries/lore have no saved conversations - the
    section hides for those kinds (the task-443 inspector idiom) rather
    than dangling a header over an empty list."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        for kind in ("persona", "dictionary", "lore"):
            pane.show_selection(name="Item", kind=kind)
            await pilot.pause()
            assert (
                pilot.app.query_one(
                    "#personas-conversations-header", Static
                ).display
                is False
            ), kind
            assert (
                pilot.app.query_one(
                    "#personas-conversations-list", ListView
                ).display
                is False
            ), kind
        # ...and a character selection reveals it again.
        pane.show_selection(name="Detective Sam", kind="character")
        await pilot.pause()
        assert (
            pilot.app.query_one("#personas-conversations-header", Static).display
            is True
        )


async def test_disabled_actions_carry_reason_tooltips_without_selection():
    """F-037: every disabled inspector action explains why - even in the
    pre-selection state (hidden on the screen, but the pane contract holds)."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        expectations = {
            "#personas-start-chat": "Pick a character or persona to start chatting.",
            "#personas-attach-to-console": (
                "Pick a character or persona to start chatting."
            ),
            "#personas-export-json": "Select an item to export.",
            "#personas-export-png": "Select an item to export.",
            "#personas-delete": "Select an item to delete.",
        }
        for button_id, expected_tooltip in expectations.items():
            button = pilot.app.query_one(button_id, Button)
            assert button.disabled is True, button_id
            assert button.tooltip == expected_tooltip, button_id


async def test_blocked_console_tooltip_uses_intent_copy():
    """F-037: a screen-blocked Console action explains itself in intent copy."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Tutor", kind="character")
        pane.set_console_actions_enabled(False, reason="prompts are not attachable")
        await pilot.pause()
        attach = pilot.app.query_one("#personas-attach-to-console", Button)
        assert attach.disabled is True
        assert attach.tooltip == "Chat blocked: prompts are not attachable"


async def test_readiness_copy_is_compact_for_narrow_inspector():
    app = InspectorApp()
    async with app.run_test(size=(24, 20)) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        readiness = pilot.app.query_one("#personas-readiness-console", Static)

        default_copy = str(readiness.renderable)
        assert default_copy == "Pick a character or persona to start chatting."
        assert " - " not in default_copy

        # A screen-supplied reason renders once there is a selection for the
        # copy to be about (pre-selection the guidance line owns the copy).
        pane.show_selection(name="Tutor", kind="character")
        pane.set_console_actions_enabled(False, reason="prompts are not attachable")
        await pilot.pause()

        blocked_copy = str(readiness.renderable)
        assert blocked_copy == "Chat blocked: prompts are not attachable"
        assert " - " not in blocked_copy


async def test_action_buttons_carry_shared_flat_button_classes():
    app = InspectorApp()
    async with app.run_test() as pilot:
        # F-032: one primary Console CTA (Chat now), one secondary (Send to
        # Console draft).
        assert pilot.app.query_one("#personas-start-chat", Button).has_class(
            "console-action-primary"
        )
        assert pilot.app.query_one("#personas-attach-to-console", Button).has_class(
            "console-action-secondary"
        )
        for button_id in ("#personas-export-json", "#personas-export-png"):
            assert pilot.app.query_one(button_id, Button).has_class(
                "console-action-subdued"
            )
        delete = pilot.app.query_one("#personas-delete", Button)
        assert delete.has_class("console-action-subdued")
        assert delete.has_class("personas-destructive")


async def test_console_ctas_speak_in_intent():
    """F-032: the Console CTAs are one primary + one secondary, named by
    intent, primary first."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        actions = pilot.app.query_one("#personas-inspector-actions")
        labels = [str(button.label) for button in actions.query(Button)]
        assert labels[:2] == ["Chat now", "Send to Console draft"]


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


async def test_show_selection_enables_export_actions():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pilot.pause()
        assert "Selected: Detective Sam" in str(
            pilot.app.query_one("#personas-selected-name", Static).renderable
        )
        assert (
            pilot.app.query_one("#personas-attach-to-console", Button).disabled is True
        )
        assert pilot.app.query_one("#personas-start-chat", Button).disabled is True
        assert pilot.app.query_one("#personas-export-json", Button).disabled is False
        assert pilot.app.query_one("#personas-export-png", Button).disabled is False
        assert "Chat blocked: select an item" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )


async def test_persona_selection_disables_png_export():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Archivist", kind="persona")
        await pilot.pause()
        assert pilot.app.query_one("#personas-export-json", Button).disabled is False
        assert pilot.app.query_one("#personas-export-png", Button).disabled is True


async def test_character_selection_renders_all_actions():
    """Task-443: character is the applicable kind for every action button."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pilot.pause()
        for button_id in (
            "#personas-attach-to-console",
            "#personas-start-chat",
            "#personas-export-json",
            "#personas-export-png",
            "#personas-delete",
        ):
            assert (
                pilot.app.query_one(button_id, Button).display is True
            ), button_id


async def test_persona_selection_hides_only_export_png():
    """Task-443 AC1: personas have no PNG card, so Export PNG does not
    render for a persona selection - Attach/Start Chat/Export JSON
    still apply (the readiness gate, not kind, controls their disabled
    state)."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Archivist", kind="persona")
        await pilot.pause()
        for button_id in (
            "#personas-attach-to-console",
            "#personas-start-chat",
            "#personas-export-json",
            "#personas-delete",
        ):
            assert (
                pilot.app.query_one(button_id, Button).display is True
            ), button_id
        assert pilot.app.query_one("#personas-export-png", Button).display is False


async def test_dictionary_selection_hides_console_and_export_actions():
    """Task-443 AC1: dictionaries can never Attach/Start Chat/export a card -
    those buttons must not render at all (not merely stay disabled), while
    Delete (which does apply) stays visible."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Combat Slang", kind="dictionary")
        await pilot.pause()
        for button_id in (
            "#personas-attach-to-console",
            "#personas-start-chat",
            "#personas-export-json",
            "#personas-export-png",
        ):
            assert (
                pilot.app.query_one(button_id, Button).display is False
            ), button_id
        assert pilot.app.query_one("#personas-delete", Button).display is True
        # F-032: the readiness line says what DOES apply, in intent language.
        assert "Console chat is for characters and personas." in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )


async def test_lore_selection_hides_console_and_export_actions():
    """Task-443 AC1: same as dictionaries - lore books never Attach/Start
    Chat/export a card."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Frontier World", kind="lore")
        await pilot.pause()
        for button_id in (
            "#personas-attach-to-console",
            "#personas-start-chat",
            "#personas-export-json",
            "#personas-export-png",
        ):
            assert (
                pilot.app.query_one(button_id, Button).display is False
            ), button_id
        assert pilot.app.query_one("#personas-delete", Button).display is True


async def test_clear_selection_restores_action_visibility():
    """Task-443: leaving a dictionary/lore selection (kind -> None) must not
    leave the never-applies buttons permanently hidden - their per-button
    display flags reset to the pre-selection baseline. F-031 layers on top:
    with no selection the whole action STACK is hidden behind the guidance
    line, so the restored flags only take effect on the next selection."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Combat Slang", kind="dictionary")
        await pilot.pause()
        assert pilot.app.query_one("#personas-start-chat", Button).display is False
        await pane.clear_selection()
        await pilot.pause()
        for button_id in (
            "#personas-attach-to-console",
            "#personas-start-chat",
            "#personas-export-json",
            "#personas-export-png",
            "#personas-delete",
        ):
            assert (
                pilot.app.query_one(button_id, Button).display is True
            ), button_id
        # ...but the stack itself is hidden again until a selection returns.
        assert pilot.app.query_one("#personas-inspector-actions").display is False
        pane.show_selection(name="Detective Sam", kind="character")
        await pilot.pause()
        assert pilot.app.query_one("#personas-inspector-actions").display is True


async def test_console_action_enablement_is_explicitly_screen_owned():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Tutor", kind="character")
        await pilot.pause()

        # Selection/export state is inspector-local, but Console attach/start
        # availability is pushed by PersonasScreen from _console_action_allowed().
        assert pilot.app.query_one("#personas-export-json", Button).disabled is False
        assert pilot.app.query_one("#personas-delete", Button).disabled is False
        assert (
            pilot.app.query_one("#personas-attach-to-console", Button).disabled is True
        )
        assert pilot.app.query_one("#personas-start-chat", Button).disabled is True
        assert "Chat blocked: select an item" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )

        pane.set_console_actions_enabled(True)
        await pilot.pause()

        assert (
            pilot.app.query_one("#personas-attach-to-console", Button).disabled is False
        )
        assert pilot.app.query_one("#personas-start-chat", Button).disabled is False
        assert "Ready to chat in Console." in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )

        pane.set_console_actions_enabled(False, reason="prompts are not attachable")
        await pilot.pause()

        assert pilot.app.query_one("#personas-export-json", Button).disabled is False
        assert pilot.app.query_one("#personas-delete", Button).disabled is False
        assert (
            pilot.app.query_one("#personas-attach-to-console", Button).disabled is True
        )
        assert pilot.app.query_one("#personas-start-chat", Button).disabled is True
        assert "Chat blocked: prompts are not attachable" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )


async def test_unsaved_disables_attach_and_export_with_reason():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Tutor", kind="character")
        pane.set_console_actions_enabled(True)
        pane.set_unsaved(True)
        pane.set_console_actions_enabled(False, reason="unsaved edits")
        await pilot.pause()
        attach = pilot.app.query_one("#personas-attach-to-console", Button)
        assert attach.disabled is True
        assert "unsaved" in str(attach.tooltip).lower()
        assert pilot.app.query_one("#personas-export-json", Button).disabled is True
        assert "Save or discard your edits to chat in Console." in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )
        pane.set_unsaved(False)
        pane.set_console_actions_enabled(True)
        await pilot.pause()
        assert (
            pilot.app.query_one("#personas-attach-to-console", Button).disabled is False
        )


async def test_show_validation_errors_renders_messages_and_clears():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_validation(("name: required", "first_message: required"))
        await pilot.pause()
        summary = str(
            pilot.app.query_one("#personas-validation-summary", Static).renderable
        )
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
        def on_conversation_row_selected(
            self, message: ConversationRowSelected
        ) -> None:
            received.append(message.conversation_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        # The conversations panel only renders for a selection (F-031).
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations(
            (("conv-1", "First case"), ("conv-2", "Cold trail"))
        )
        await pilot.pause()
        assert len(pilot.app.query(".personas-conversation-row")) == 2
        await pilot.click("#personas-conversation-row-conv-1")
        await pilot.pause()
    assert received == ["conv-1"]


async def test_conversation_click_after_rerender_posts_new_id():
    received = []

    class CaptureApp(InspectorApp):
        def on_conversation_row_selected(
            self, message: ConversationRowSelected
        ) -> None:
            received.append(message.conversation_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
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
        def on_conversation_row_selected(
            self, message: ConversationRowSelected
        ) -> None:
            received.append(message.conversation_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations(
            (
                ("conv-1", "First case"),
                ("conv-2", "Cold trail"),
                ("conv-3", "Closed file"),
            )
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
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations((("conv-1", "First case"),))
        pane.set_unsaved(True)
        await pilot.pause()
        await pane.clear_selection()
        await pilot.pause()
        assert "Selected: none" in str(
            pilot.app.query_one("#personas-selected-name", Static).renderable
        )
        assert (
            pilot.app.query_one("#personas-attach-to-console", Button).disabled is True
        )
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
