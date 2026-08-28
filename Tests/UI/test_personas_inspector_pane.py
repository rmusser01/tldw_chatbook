"""Mounted tests for the Personas inspector pane."""

import asyncio
from pathlib import Path

import pytest
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Checkbox, ListItem, ListView, Static

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook import Constants as constants
from tldw_chatbook.Widgets.Persona_Widgets import (
    personas_messages,
    personas_pane_messages,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
    PersonasInspectorPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    ConversationRowSelected,
    OlderConversationsRequested,
)

pytestmark = pytest.mark.asyncio


def _row_text(item: ListItem) -> str:
    """Visible text of a conversation row (the ListItem's inner Static)."""
    return str(item.query_one(Static).renderable)


def _painted_region_text(app, widget) -> tuple[str, ...]:
    """Return only compositor-painted cells inside a contained widget region."""
    screen_region = app.screen.region
    assert screen_region.contains_region(widget.region), (
        f"{widget!r} must fit the viewport: "
        f"widget_region={widget.region!r}, screen_region={screen_region!r}"
    )
    strips = list(app.screen._compositor.render_strips())
    assert 0 <= widget.region.y and widget.region.bottom <= len(strips)
    return tuple(
        "".join(
            segment.text
            for segment in strips[y].crop(widget.region.x, widget.region.right)
        )
        for y in range(widget.region.y, widget.region.bottom)
    )


class InspectorApp(ConsolidatedCSSApp):
    def __init__(self):
        super().__init__()
        self.buddy_messages = []
        self.actor_pack_export_messages = []
        self.conversation_messages = []
        self.older_conversation_messages = []

    def compose(self):
        yield PersonasInspectorPane(id="personas-inspector-pane")

    def on_persona_buddy_action_requested(self, message) -> None:
        self.buddy_messages.append(message)

    def on_actor_pack_export_requested(self, message) -> None:
        self.actor_pack_export_messages.append(message)

    def on_conversation_row_selected(self, message: ConversationRowSelected) -> None:
        self.conversation_messages.append(message)

    def on_older_conversations_requested(
        self, message: OlderConversationsRequested
    ) -> None:
        self.older_conversation_messages.append(message)


class StyledInspectorApp(InspectorApp):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def compose(self):
        with Horizontal(
            id="personas-workbench",
            classes="destination-workbench personas-workbench-compact",
        ):
            yield PersonasInspectorPane(
                id="personas-inspector-pane",
                classes=(
                    "destination-workbench-pane ds-inspector "
                    "personas-workbench-compact-pane"
                ),
            )


async def test_older_conversations_request_is_a_parameterless_typed_message():
    message_type = getattr(personas_pane_messages, "OlderConversationsRequested", None)

    assert message_type is not None
    message = message_type()
    assert isinstance(message, Message)
    assert not hasattr(message, "conversation_id")


async def test_actor_pack_export_message_is_typed_frozen_and_slotted():
    message_type = getattr(personas_messages, "ActorPackExportRequested", None)
    assert message_type is not None
    assert message_type.__slots__ == ("_payload",)
    assert "__dict__" not in message_type.__slots__
    assert message_type.set_sender is Message.set_sender
    assert message_type.stop is Message.stop

    message = message_type(
        actor_kind="persona",
        source="local",
        local_actor_id="persona-7",
        actor_revision=4,
    )

    assert (
        message.actor_kind,
        message.source,
        message.local_actor_id,
        message.actor_revision,
    ) == ("persona", "local", "persona-7", 4)
    with pytest.raises(AttributeError):
        message.actor_revision = 5


@pytest.mark.parametrize("size", ((170, 50), (80, 24)))
@pytest.mark.parametrize("actor_kind", ("character", "persona"))
async def test_eligible_local_actor_pack_action_is_labelled_focusable_and_typed(
    size, actor_kind
):
    app = InspectorApp()
    async with app.run_test(size=size) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        actor_id = "7" if actor_kind == "character" else "persona-7"
        pane.show_selection(name="Portable", kind=actor_kind)
        pane.set_actor_pack_export_state(
            source="local",
            actor_kind=actor_kind,
            local_actor_id=actor_id,
            actor_revision=4,
            eligible=True,
        )
        await pilot.pause()

        button = pane.query_one("#personas-export-actor-pack", Button)
        assert str(button.label) == "Export Actor Pack"
        assert button.display is True
        assert button.disabled is False
        button.focus(scroll_visible=True)
        await pilot.pause()
        assert button.has_focus is True
        await pilot.click("#personas-export-actor-pack")
        await pilot.pause()

        message = app.actor_pack_export_messages[-1]
        assert (
            message.actor_kind,
            message.source,
            message.local_actor_id,
            message.actor_revision,
        ) == (actor_kind, "local", actor_id, 4)


async def test_server_actor_pack_action_names_save_local_copy_recovery():
    app = InspectorApp()
    async with app.run_test(size=(80, 24)) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Remote", kind="persona")
        pane.set_actor_pack_export_state(
            source="server",
            actor_kind="persona",
            local_actor_id="persona-7",
            actor_revision=4,
            eligible=True,
            reason="Save a local copy first",
        )
        await pilot.pause()

        button = pane.query_one("#personas-export-actor-pack", Button)
        assert button.display is True
        assert button.disabled is True
        assert button.tooltip == "Save a local copy first"


async def test_corrupt_or_missing_actor_pack_selection_stays_disabled():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Broken", kind="character")
        pane.set_actor_pack_export_state(
            source="local",
            actor_kind="character",
            local_actor_id="7",
            actor_revision=4,
            eligible=False,
            reason="Actor or portrait is unavailable. Refresh and try again.",
        )
        await pilot.pause()

        button = pane.query_one("#personas-export-actor-pack", Button)
        assert button.disabled is True
        assert (
            button.tooltip == "Actor or portrait is unavailable. Refresh and try again."
        )


async def test_persona_buddy_action_message_is_typed_frozen_and_slotted():
    message_type = getattr(personas_messages, "PersonaBuddyActionRequested", None)
    assert message_type is not None
    assert message_type.__slots__ == ("_payload",)
    assert "__dict__" not in message_type.__slots__
    assert message_type.set_sender is Message.set_sender
    assert message_type.prevent_default is Message.prevent_default
    assert message_type.stop is Message.stop

    message = message_type(
        action="use",
        source="local",
        persona_id="persona-7",
        revision=4,
    )

    assert (message.action, message.source, message.persona_id, message.revision) == (
        "use",
        "local",
        "persona-7",
        4,
    )
    with pytest.raises(AttributeError):
        message.action = "show"


async def test_persona_buddy_action_message_uses_normal_textual_delivery():
    app = InspectorApp()
    async with app.run_test(size=(80, 24)) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        message = personas_messages.PersonaBuddyActionRequested(
            action="use",
            source="local",
            persona_id="persona-7",
            revision=4,
        )

        pane.post_message(message)
        await pilot.pause()

        assert app.buddy_messages == [message]


@pytest.mark.parametrize(
    ("button_id", "label", "action"),
    (
        ("#personas-buddy-use", "Use for Buddy", "use"),
        ("#personas-buddy-show", "Show Buddy", "show"),
        ("#personas-buddy-close", "Close Buddy", "close"),
        ("#personas-buddy-disable", "Disable Buddy", "disable"),
    ),
)
async def test_active_local_persona_buddy_actions_are_explicit_and_typed(
    button_id: str,
    label: str,
    action: str,
):
    app = InspectorApp()
    async with app.run_test(size=(170, 50)) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(
            name="Archivist",
            kind="persona",
            source="local",
            entity_id="persona-7",
            revision=4,
            active=True,
            profile_current=True,
        )
        pane.set_buddy_status(
            source="local",
            persona_id="persona-7",
            enabled=True,
            open=action != "show",
        )
        await pilot.pause()

        button = pilot.app.query_one(button_id, Button)
        assert str(button.label) == label
        assert button.disabled is False
        await pilot.click(button_id)
        await pilot.pause()

        message = app.buddy_messages[-1]
        assert (
            message.action,
            message.source,
            message.persona_id,
            message.revision,
        ) == (
            action,
            "local",
            "persona-7",
            4,
        )


async def test_server_persona_buddy_actions_are_disabled_with_exact_recovery_copy():
    app = InspectorApp()
    async with app.run_test(size=(170, 50)) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(
            name="Remote Archivist",
            kind="persona",
            source="server",
            entity_id="persona-7",
            revision=4,
            active=True,
            profile_current=True,
        )
        await pilot.pause()

        for button_id in (
            "#personas-buddy-use",
            "#personas-buddy-show",
            "#personas-buddy-close",
            "#personas-buddy-disable",
        ):
            button = pilot.app.query_one(button_id, Button)
            assert button.disabled is True
            assert button.tooltip == "Save a local copy first"


async def test_non_owner_highlight_only_enables_use_with_truthful_tooltip():
    app = InspectorApp()
    async with app.run_test(size=(170, 50)) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(
            name="Navigator",
            kind="persona",
            source="local",
            entity_id="persona-2",
            revision=5,
            active=True,
            profile_current=True,
        )
        pane.set_buddy_status(
            source="local",
            persona_id="persona-1",
            enabled=True,
            open=True,
        )
        await pilot.pause()

        assert pane.query_one("#personas-buddy-use", Button).disabled is False
        for button_id in (
            "#personas-buddy-show",
            "#personas-buddy-close",
            "#personas-buddy-disable",
        ):
            button = pane.query_one(button_id, Button)
            assert button.disabled is True
            assert button.tooltip == "Select the Persona currently used by Buddy"


async def test_persona_highlight_alone_emits_no_buddy_action():
    app = InspectorApp()
    async with app.run_test(size=(80, 24)) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(
            name="Archivist",
            kind="persona",
            source="local",
            entity_id="persona-7",
            revision=4,
            active=True,
            profile_current=True,
        )
        await pilot.pause()

        assert app.buddy_messages == []


@pytest.mark.parametrize("size", ((170, 50), (80, 24)))
async def test_buddy_actions_keep_exact_labels_and_focus_without_keybindings(size):
    app = InspectorApp()
    async with app.run_test(size=size) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(
            name="Archivist",
            kind="persona",
            source="local",
            entity_id="persona-7",
            revision=4,
            active=True,
            profile_current=True,
        )
        await pilot.pause()

        expected = (
            ("#personas-buddy-use", "Use for Buddy"),
            ("#personas-buddy-show", "Show Buddy"),
            ("#personas-buddy-close", "Close Buddy"),
            ("#personas-buddy-disable", "Disable Buddy"),
        )
        for button_id, label in expected:
            pane.set_buddy_status(
                source="local",
                persona_id="persona-7",
                enabled=True,
                open=button_id != "#personas-buddy-show",
            )
            button = pilot.app.query_one(button_id, Button)
            assert str(button.label) == label
            assert button.can_focus is True
            button.focus(scroll_visible=True)
            await pilot.pause()
            assert button.has_focus is True

        await pilot.press("u", "s", "c", "d")
        assert app.buddy_messages == []


@pytest.mark.parametrize(
    ("entity_id", "revision", "active"),
    ((None, 4, True), ("persona-7", None, True), ("persona-7", 4, False)),
)
async def test_buddy_rejects_incomplete_or_inactive_local_persona(
    entity_id, revision, active
):
    app = InspectorApp()
    async with app.run_test(size=(170, 50)) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(
            name="Archivist",
            kind="persona",
            source="local",
            entity_id=entity_id,
            revision=revision,
            active=active,
            profile_current=True,
        )
        await pilot.pause()

        for button in pilot.app.query(".persona-buddy-action").results(Button):
            assert button.disabled is True


async def test_buddy_requires_current_complete_profile_not_cached_eligibility():
    app = InspectorApp()
    async with app.run_test(size=(170, 50)) as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(
            name="Cached Persona",
            kind="persona",
            source="local",
            entity_id="persona-7",
            revision=4,
            active=True,
            profile_current=False,
        )
        await pilot.pause()

        for button in pilot.app.query(".persona-buddy-action").results(Button):
            assert button.disabled is True
            assert (
                button.tooltip
                == "Persona details are unavailable. Refresh and try again."
            )


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
        assert (
            str(pilot.app.query_one("#personas-readiness-console", Static).renderable)
            == "Pick a character or persona to start chatting."
        )


async def test_no_selection_shows_single_guidance_line():
    """F-031: pre-selection the inspector is one plain guidance line - no
    wall of disabled buttons, no dangling section headers, no false
    "Validation: OK"."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#personas-inspector-actions").display is False
        assert (
            pilot.app.query_one("#personas-conversations-header", Static).display
            is False
        )
        assert (
            pilot.app.query_one("#personas-conversations-list", ListView).display
            is False
        )
        assert (
            pilot.app.query_one("#personas-readiness-header", Static).display is False
        )
        assert (
            pilot.app.query_one("#personas-validation-summary", Static).display is False
        )
        guidance = pilot.app.query_one("#personas-readiness-console", Static)
        assert guidance.display is True
        assert (
            str(guidance.renderable) == "Pick a character or persona to start chatting."
        )


async def test_selection_reveals_inspector_sections():
    """F-031: selecting an item wakes the whole inspector back up."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pilot.pause()
        assert pilot.app.query_one("#personas-inspector-actions").display is True
        assert (
            pilot.app.query_one("#personas-conversations-header", Static).display
            is True
        )
        assert (
            pilot.app.query_one("#personas-conversations-list", ListView).display
            is True
        )
        assert pilot.app.query_one("#personas-readiness-header", Static).display is True
        assert (
            pilot.app.query_one("#personas-validation-summary", Static).display is True
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
                pilot.app.query_one("#personas-conversations-header", Static).display
                is False
            ), kind
            assert (
                pilot.app.query_one("#personas-conversations-list", ListView).display
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
        assert attach.tooltip == (
            "Chat now and Send to Console draft blocked: prompts are not attachable"
        )


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
        assert blocked_copy == (
            "Chat now and Send to Console draft blocked: prompts are not attachable"
        )
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


async def test_disabled_tts_checkbox_stays_legible():
    """F-041: the disabled voice-profile checkbox reads as a disabled
    control - dimmed label, full opacity, visible glyph box - not a dark
    gap (Textual's base *:disabled:can-focus dims to 0.7 and the stock
    toggle box is panel-on-panel).

    task-2233: the legibility fix covers the SHOWN case - the checkbox is
    hidden outright until a profile is assigned, so this test assigns one
    first (still disabled here: nothing is selected/exportable).
    """
    from textual.color import Color

    class StyledInspectorApp(ConsolidatedCSSApp):
        CSS_PATH = str(
            Path(__file__).resolve().parents[2]
            / "tldw_chatbook"
            / "css"
            / "tldw_cli_modular.tcss"
        )

        def compose(self):
            yield PersonasInspectorPane(id="personas-inspector-pane")

    app = StyledInspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        checkbox = pilot.app.query_one("#personas-export-include-tts", Checkbox)
        assert checkbox.display is False  # task-2233: no profile assigned
        pane.set_tts_export_available(True)
        await pilot.pause()
        assert checkbox.display is True
        assert checkbox.disabled is True
        variables = app.get_css_variables()
        # Full opacity (Textual's base disabled rule dims to 0.7)...
        assert checkbox.styles.opacity == 1.0
        # ...with the label dimmed per the disabled idiom (the theme's
        # text-disabled is "auto 38%": foreground at 38% alpha).
        assert checkbox.styles.color.a == 0.38
        # ...and the glyph box paints a real surface, not a dark gap.
        glyph_styles = checkbox.get_component_styles("toggle--button")
        assert glyph_styles.background == Color.parse(variables["surface"])


async def test_tts_checkbox_hidden_until_a_profile_is_assigned():
    """task-2233: the checkbox renders only when the selected character has
    a voice profile to include - no assignment, no disabled dark smear.
    Clearing the assignment hides it again; the kind gate still applies."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        checkbox = pilot.app.query_one("#personas-export-include-tts", Checkbox)

        pane.show_selection(name="Tutor", kind="character")
        await pilot.pause()
        assert checkbox.display is False

        pane.set_tts_export_available(True)
        await pilot.pause()
        assert checkbox.display is True

        pane.set_tts_export_available(False)
        await pilot.pause()
        assert checkbox.display is False

        # Even with an assignment on record, non-character kinds never show
        # it (the kind gate is independent of availability).
        pane.show_selection(name="Archivist", kind="persona")
        pane.set_tts_export_available(True)
        await pilot.pause()
        assert checkbox.display is False


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
        assert "Chat now and Send to Console draft blocked: select an item" in str(
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
            assert pilot.app.query_one(button_id, Button).display is True, button_id


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
            assert pilot.app.query_one(button_id, Button).display is True, button_id
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
            assert pilot.app.query_one(button_id, Button).display is False, button_id
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
            assert pilot.app.query_one(button_id, Button).display is False, button_id
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
            assert pilot.app.query_one(button_id, Button).display is True, button_id
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
        assert "Chat now and Send to Console draft blocked: select an item" in str(
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
        assert (
            "Chat now and Send to Console draft blocked: prompts are not attachable"
            in str(
                pilot.app.query_one("#personas-readiness-console", Static).renderable
            )
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


async def test_initial_paginated_conversations_end_with_actionable_load_tail():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations(
            (("conv-1", "First case"), ("conv-2", "Cold trail")),
            has_more=True,
        )
        await pilot.pause()

        list_view = app.query_one("#personas-conversations-list", ListView)
        items = list(list_view.children)
        assert [_row_text(item) for item in items] == [
            "First case",
            "Cold trail",
            f"Load {constants.PERSONAS_CONVERSATIONS_PAGE_SIZE} older conversations",
        ]

        tail = items[-1]
        list_view.focus()
        list_view.index = len(items) - 1
        await pilot.pause()
        assert list_view.highlighted_child is tail

        await pilot.press("enter")
        await pilot.pause()

        assert len(app.older_conversation_messages) == 1
        assert app.conversation_messages == []


async def test_older_loading_replaces_only_tail_and_is_highlightable_but_inert():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations(
            (("conv-1", "First case"), ("conv-2", "Cold trail")),
            has_more=True,
        )
        list_view = app.query_one("#personas-conversations-list", ListView)
        first_row = list_view.children[0]
        old_tail = list_view.children[-1]
        list_view.focus()
        list_view.index = len(list_view.children) - 1
        await pilot.pause()

        await pane.show_older_conversations_loading()
        await pilot.pause()

        loading_tail = list_view.children[-1]
        assert list_view.children[0] is first_row
        assert loading_tail is old_tail
        assert _row_text(loading_tail) == "Loading older conversations..."
        assert loading_tail.disabled is False
        assert list_view.highlighted_child is loading_tail
        assert loading_tail.highlighted is True
        assert loading_tail.has_class("-highlight") is True

        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.pause()

        assert app.older_conversation_messages == []
        assert app.conversation_messages == []


async def test_older_loading_retains_tail_highlight_only_while_list_has_focus():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations((("conv-1", "First case"),), has_more=True)
        list_view = app.query_one("#personas-conversations-list", ListView)
        list_view.focus()
        list_view.index = len(list_view.children) - 1
        await pilot.pause()
        tail = list_view.highlighted_child
        app.query_one("#personas-inspector-rail-collapse", Button).focus()
        await pilot.pause()

        await pane.show_older_conversations_loading()
        await pilot.pause()

        assert list_view.highlighted_child is not tail


async def test_initial_conversation_loading_is_disabled_and_clears_old_rows():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations((("conv-1", "First case"),), has_more=True)

        await pane.show_conversations_loading()
        await pilot.pause()

        list_view = app.query_one("#personas-conversations-list", ListView)
        assert len(list_view.children) == 1
        loading = list_view.children[0]
        assert _row_text(loading) == "Loading conversations..."
        assert loading.disabled is True
        assert len(app.query(".personas-conversation-row")) == 0


async def test_initial_conversation_failure_is_two_line_actionable_retry():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")

        await pane.show_conversations_failure(initial=True)
        await pilot.pause()

        list_view = app.query_one("#personas-conversations-list", ListView)
        retry_tail = list_view.children[0]
        assert _row_text(retry_tail).splitlines() == [
            "Load failed.",
            "Retry conversations",
        ]

        list_view.focus()
        list_view.index = 0
        await pilot.press("enter")
        await pilot.pause()

        assert len(app.older_conversation_messages) == 1
        assert app.conversation_messages == []


async def test_append_failure_preserves_rows_and_retries_the_tail():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations(
            (("conv-1", "First case"), ("conv-2", "Cold trail")),
            has_more=True,
        )
        list_view = app.query_one("#personas-conversations-list", ListView)
        first_row = list_view.children[0]
        await pane.show_older_conversations_loading()

        await pane.show_conversations_failure(initial=False)
        await pilot.pause()

        retry_tail = list_view.children[-1]
        assert list_view.children[0] is first_row
        assert _row_text(retry_tail).splitlines() == [
            "Load failed.",
            "Retry older conversations",
        ]

        list_view.focus()
        list_view.index = len(list_view.children) - 1
        await pilot.press("enter")
        await pilot.pause()

        assert len(app.older_conversation_messages) == 1
        assert app.conversation_messages == []


@pytest.mark.parametrize(
    ("rows", "expected_copy"),
    (
        ((), "No saved conversations."),
        ((("conv-1", "First case"),), "All conversations shown."),
    ),
)
async def test_empty_and_exhausted_conversation_states_are_distinct_and_inert(
    rows: tuple[tuple[str, str], ...], expected_copy: str
):
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")

        await pane.show_conversations(rows, has_more=False)
        await pilot.pause()

        list_view = app.query_one("#personas-conversations-list", ListView)
        assert _row_text(list_view.children[-1]) == expected_copy
        list_view.focus()
        list_view.index = len(list_view.children) - 1
        await pilot.press("enter")
        await pilot.pause()

        assert app.older_conversation_messages == []
        assert app.conversation_messages == []


async def test_append_keeps_old_widgets_and_new_rows_use_conversation_selection():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations((("conv-1", "First case"),), has_more=True)
        list_view = app.query_one("#personas-conversations-list", ListView)
        original_row = list_view.children[0]
        await pane.show_older_conversations_loading()

        await pane.append_conversations(
            (("conv-2", "Cold trail"), ("conv-3", "Closed file")),
            has_more=False,
        )
        await pilot.pause()

        assert list_view.children[0] is original_row
        assert [_row_text(item) for item in list_view.children] == [
            "First case",
            "Cold trail",
            "Closed file",
            "All conversations shown.",
        ]

        await pilot.click("#personas-conversation-row-conv-3")
        await pilot.pause()

        assert [
            message.conversation_id for message in app.conversation_messages
        ] == ["conv-3"]
        assert app.older_conversation_messages == []


async def test_append_highlights_first_new_row_only_from_focused_loading_tail():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations(
            (("conv-1", "First case"), ("conv-2", "Cold trail")),
            has_more=True,
        )
        list_view = app.query_one("#personas-conversations-list", ListView)
        list_view.focus()
        list_view.index = len(list_view.children) - 1
        await pilot.pause()
        await pane.show_older_conversations_loading()
        loading_tail = list_view.children[-1]
        assert list_view.has_focus is True
        assert list_view.highlighted_child is loading_tail

        await pane.append_conversations((("conv-3", "Closed file"),), has_more=True)
        await pilot.pause()

        new_row = app.query_one("#personas-conversation-row-conv-3", ListItem)
        assert list_view.has_focus is True
        assert list_view.highlighted_child is new_row
        assert new_row.highlighted is True
        assert new_row.has_class("-highlight") is True
        assert loading_tail.highlighted is False
        assert loading_tail.has_class("-highlight") is False
        assert [item for item in list_view.children if item.highlighted] == [new_row]


@pytest.mark.parametrize("interaction", ("cursor", "focus"))
async def test_append_completion_preserves_interaction_during_real_mount(
    interaction: str, monkeypatch
):
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations(
            (("conv-1", "First case"), ("conv-2", "Cold trail")),
            has_more=True,
        )
        list_view = app.query_one("#personas-conversations-list", ListView)
        list_view.focus()
        list_view.index = len(list_view.children) - 1
        await pilot.pause()
        await pane.show_older_conversations_loading()
        loading_tail = list_view.children[-1]
        mount_settled = asyncio.Event()
        release_mount = asyncio.Event()
        real_mount = list_view.mount

        async def gated_real_mount(*widgets, **kwargs):
            await real_mount(*widgets, **kwargs)
            mount_settled.set()
            await release_mount.wait()

        monkeypatch.setattr(list_view, "mount", gated_real_mount)
        append_task = asyncio.create_task(
            pane.append_conversations((("conv-3", "Closed file"),), has_more=False)
        )
        await mount_settled.wait()
        assert append_task.done() is False
        assert loading_tail.highlighted is True
        assert loading_tail.has_class("-highlight") is True

        if interaction == "cursor":
            list_view.index = 0
            expected_highlight = list_view.children[0]
            expected_focus = list_view
        else:
            expected_focus = app.query_one(
                "#personas-inspector-rail-collapse", Button
            )
            expected_focus.focus()
            expected_highlight = loading_tail
        await pilot.pause()

        release_mount.set()
        await append_task
        await pilot.pause()

        new_row = app.query_one("#personas-conversation-row-conv-3", ListItem)
        assert app.focused is expected_focus
        assert list_view.highlighted_child is expected_highlight
        assert [item for item in list_view.children if item.highlighted] == [
            expected_highlight
        ]
        assert new_row.highlighted is (expected_highlight is new_row)


@pytest.mark.parametrize("tail_state", ("load", "retry", "exhausted"))
async def test_append_does_not_advance_from_a_non_loading_tail(tail_state: str):
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations(
            (("conv-1", "First case"),),
            has_more=tail_state != "exhausted",
        )
        if tail_state == "retry":
            await pane.show_conversations_failure(initial=False)
        list_view = app.query_one("#personas-conversations-list", ListView)
        list_view.focus()
        list_view.index = len(list_view.children) - 1
        await pilot.pause()
        highlighted_tail = list_view.highlighted_child

        await pane.append_conversations((("conv-2", "Cold trail"),), has_more=False)
        await pilot.pause()

        new_row = app.query_one("#personas-conversation-row-conv-2", ListItem)
        assert list_view.highlighted_child is highlighted_tail
        assert list_view.highlighted_child is not new_row


async def test_append_does_not_steal_focus_or_highlight_a_new_row():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations((("conv-1", "First case"),), has_more=True)
        list_view = app.query_one("#personas-conversations-list", ListView)
        list_view.focus()
        list_view.index = len(list_view.children) - 1
        await pane.show_older_conversations_loading()
        collapse_button = app.query_one(
            "#personas-inspector-rail-collapse", Button
        )
        collapse_button.focus()
        await pilot.pause()
        assert collapse_button.has_focus is True

        await pane.append_conversations((("conv-2", "Cold trail"),), has_more=False)
        await pilot.pause()

        new_row = app.query_one("#personas-conversation-row-conv-2", ListItem)
        assert collapse_button.has_focus is True
        assert list_view.highlighted_child is not new_row


async def test_append_preserves_another_highlighted_conversation():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations(
            (("conv-1", "First case"), ("conv-2", "Cold trail")),
            has_more=True,
        )
        list_view = app.query_one("#personas-conversations-list", ListView)
        list_view.focus()
        list_view.index = len(list_view.children) - 1
        await pane.show_older_conversations_loading()
        list_view.index = 0
        selected_row = list_view.children[0]

        await pane.append_conversations((("conv-3", "Closed file"),), has_more=False)
        await pilot.pause()

        assert list_view.highlighted_child is selected_row


async def test_append_with_no_new_row_does_not_advance_past_exhausted_tail():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        await pane.show_conversations((("conv-1", "First case"),), has_more=True)
        list_view = app.query_one("#personas-conversations-list", ListView)
        list_view.focus()
        list_view.index = len(list_view.children) - 1
        await pane.show_older_conversations_loading()

        await pane.append_conversations((), has_more=False)
        await pilot.pause()

        assert [_row_text(item) for item in list_view.children] == [
            "First case",
            "All conversations shown.",
        ]
        assert list_view.highlighted_child is not list_view.children[0]


@pytest.mark.parametrize("size", ((24, 20), (80, 24)))
@pytest.mark.parametrize("initial", (True, False))
async def test_retry_tail_wraps_without_changing_conversation_row_ellipsis(
    size: tuple[int, int], initial: bool
):
    app = StyledInspectorApp()
    async with app.run_test(size=size) as pilot:
        pane = app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character")
        if initial:
            await pane.show_conversations_failure(initial=True)
            expected_retry = "Retry conversations"
        else:
            await pane.show_conversations(
                (("conv-1", "A conversation title much wider than the inspector"),),
                has_more=True,
            )
            await pane.show_older_conversations_loading()
            await pane.show_conversations_failure(initial=False)
            expected_retry = "Retry older conversations"
        await pilot.pause()

        list_view = app.query_one("#personas-conversations-list", ListView)
        tail = list_view.children[-1]
        tail_copy = tail.query_one(Static)
        assert app.screen.region.contains_region(pane.region)
        assert tail.region.height >= 2
        assert tail_copy.region.height >= 2
        painted_copy = " ".join("\n".join(_painted_region_text(app, tail)).split())
        assert "Load failed." in painted_copy
        assert expected_retry in painted_copy

        if not initial:
            row = list_view.children[0]
            row_copy = row.query_one(Static)
            assert row.region.height == 1
            assert row_copy.region.height == 1
            assert str(row_copy.styles.text_wrap) == "nowrap"
            assert str(row_copy.styles.text_overflow) == "ellipsis"


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


async def test_state_pushed_before_children_mount_defers_then_replays():
    """task-2727: PersonasScreen._load_after_mount races the pane's child
    mounting; a push landing early crashed with NoMatches and aborted the
    screen's initial load (pending restore + auto-select silently skipped).
    Early pushes must defer quietly and be applied once children exist."""
    pane = PersonasInspectorPane(id="personas-inspector-pane")
    # Selection state normally arrives via show_selection(), which needs
    # mounted children itself — set the attributes directly to model the
    # racing screen-side sync.
    pane._has_selection = True
    pane._selected_kind = "character"

    # The exact call observed crashing live, issued pre-mount:
    pane.set_console_actions_enabled(False, reason="task-2727-probe")

    class LateStateApp(ConsolidatedCSSApp):
        def compose(self):
            yield pane

    app = LateStateApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        readiness = str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )
        assert "task-2727-probe" in readiness, (
            f"pre-mount push was dropped; readiness reads: {readiness!r}"
        )


async def test_avatar_thumbnail_static_matches_mosaic_grid_no_fold():
    """task-3793: the inspector portrait must not fold into black stripes.

    The thumb box used to add padding 0 1 on top of max-width 24, so the
    24-column mosaic baked by PersonasScreen._build_avatar_pixels folded at
    22 content columns -- every folded line painted a black continuation row
    (the striped portrait the owner reported) and the folded stack clipped
    at max-height 10. The box is now padding-free and set_avatar_thumbnail
    sizes the Static explicitly from the renderable grid.
    """
    from PIL import Image
    from textual.containers import Container

    from tldw_chatbook.UI.Screens.personas_screen import (
        AVATAR_THUMB_COLS,
        AVATAR_THUMB_LINES,
    )
    from tldw_chatbook.Utils.mosaic_render import mosaic_from_image

    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        mosaic = mosaic_from_image(
            Image.new("RGB", (820, 1230), (64, 200, 180)),
            AVATAR_THUMB_COLS,
            AVATAR_THUMB_LINES,
            fit="cover",
        )
        pane.set_avatar_thumbnail(mosaic)
        await pilot.pause()
        thumb_box = pane.query_one("#personas-inspector-avatar-thumb", Container)
        static = thumb_box.query_one(Static)
        # Before task-3793 the painted height was ~2x the mosaic rows (fold).
        assert static.region.height == len(mosaic.plain.split("\n"))
        assert static.region.width <= thumb_box.content_size.width
        padding = thumb_box.styles.padding
        assert (padding.top, padding.right, padding.bottom, padding.left) == (
            0,
            0,
            0,
            0,
        )


async def test_avatar_thumbnail_falls_back_to_box_dims_for_non_text_renderable():
    """Non-Text renderables get the thumb box dims, not a default-width Static.

    Pins the documented ``explicit_cell_size`` contract (PR #1434 review):
    when the renderable's grid cannot be read (e.g. a ``rich_pixels``
    renderable), ``set_avatar_thumbnail`` falls back to the container box
    (AVATAR_THUMB_COLS x AVATAR_THUMB_LINES), matching the console builder.
    """
    from rich.panel import Panel
    from textual.containers import Container

    from tldw_chatbook.UI.Screens.personas_screen import (
        AVATAR_THUMB_COLS,
        AVATAR_THUMB_LINES,
    )

    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.set_avatar_thumbnail(Panel("portrait"))
        await pilot.pause()
        thumb_box = pane.query_one("#personas-inspector-avatar-thumb", Container)
        static = thumb_box.query_one(Static)
        assert static.styles.width.value == AVATAR_THUMB_COLS
        assert static.styles.height.value == AVATAR_THUMB_LINES
