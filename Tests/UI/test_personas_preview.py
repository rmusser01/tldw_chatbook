"""Mounted tests for the Personas preview-conversation pane."""

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    PreviewOpenInConsoleRequested,
    PreviewReplyRequested,
    PreviewResetRequested,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
    PersonasPreviewPane,
)

pytestmark = pytest.mark.asyncio


class PreviewApp(App):
    def __init__(self):
        super().__init__()
        self.replies: list[str] = []
        self.resets = 0
        self.opens = 0

    def compose(self):
        yield PersonasPreviewPane(id="personas-preview-pane")

    def on_preview_reply_requested(self, message: PreviewReplyRequested) -> None:
        self.replies.append(message.user_message)

    def on_preview_reset_requested(self, message: PreviewResetRequested) -> None:
        self.resets += 1

    def on_preview_open_in_console_requested(
        self, message: PreviewOpenInConsoleRequested
    ) -> None:
        self.opens += 1


def _line_texts(app) -> list[str]:
    return [str(line.renderable) for line in app.query(".personas-preview-line")]


async def test_collapsed_by_default_and_toggle_expands():
    app = PreviewApp()
    async with app.run_test() as pilot:
        body = pilot.app.query_one("#personas-preview-body")
        assert body.display is False
        pilot.app.query_one("#personas-preview-toggle", Button).press()
        await pilot.pause()
        assert body.display is True
        pilot.app.query_one("#personas-preview-toggle", Button).press()
        await pilot.pause()
        assert body.display is False


async def test_buttons_carry_shared_flat_button_classes():
    app = PreviewApp()
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#personas-preview-test-reply", Button).has_class(
            "console-action-secondary"
        )
        assert pilot.app.query_one("#personas-preview-reset", Button).has_class(
            "console-action-subdued"
        )
        assert pilot.app.query_one("#personas-preview-open-console", Button).has_class(
            "console-action-subdued"
        )
        assert pilot.app.query_one("#personas-preview-toggle", Button).has_class(
            "console-action-subdued"
        )


async def test_expand_api_shows_body():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        assert pilot.app.query_one("#personas-preview-body").display is True


async def test_seed_append_reset_roundtrip():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Greetings, detective.")
        await pilot.pause()
        assert _line_texts(pilot.app) == ["character: Greetings, detective."]
        pane.append_user("Who are you?")
        pane.append_reply("Your humble narrator.")
        await pilot.pause()
        assert _line_texts(pilot.app) == [
            "character: Greetings, detective.",
            "you: Who are you?",
            "character: Your humble narrator.",
        ]
        assert pane.transcript_text() == (
            "character: Greetings, detective.\n"
            "you: Who are you?\n"
            "character: Your humble narrator."
        )
        pane.set_status("Ready")
        await pane.reset()
        await pilot.pause()
        assert _line_texts(pilot.app) == ["character: Greetings, detective."]
        assert str(
            pilot.app.query_one("#personas-preview-status", Static).renderable
        ) == ""


async def test_seed_empty_greeting_clears_transcript():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello.")
        pane.append_user("Hi")
        await pilot.pause()
        await pane.seed_greeting("")
        await pilot.pause()
        assert _line_texts(pilot.app) == []
        assert pane.transcript_text() == ""


async def test_double_seed_same_tick_does_not_crash():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("First greeting.")
        await pane.seed_greeting("Second greeting.")
        await pilot.pause()
        assert _line_texts(pilot.app) == ["character: Second greeting."]


async def test_transcript_lines_carry_role_classes():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello.")
        pane.append_user("Hi")
        await pilot.pause()
        you_lines = pilot.app.query(".personas-preview-line-you")
        character_lines = pilot.app.query(".personas-preview-line-character")
        assert [str(l.renderable) for l in you_lines] == ["you: Hi"]
        assert [str(l.renderable) for l in character_lines] == ["character: Hello."]


async def test_test_reply_posts_message_and_clears_input():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        pilot.app.query_one("#personas-preview-input", Input).value = "Hi there"
        pilot.app.query_one("#personas-preview-test-reply", Button).press()
        await pilot.pause()
        assert pilot.app.replies == ["Hi there"]
        assert pilot.app.query_one("#personas-preview-input", Input).value == ""
        assert _line_texts(pilot.app) == ["you: Hi there"]


async def test_enter_in_input_submits_like_test_reply():
    """Enter in the preview input takes the same path as the Test Reply button."""
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        field = pilot.app.query_one("#personas-preview-input", Input)
        field.focus()
        await pilot.pause()
        field.value = "Hi there"
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.replies == ["Hi there"]
        assert field.value == ""
        assert _line_texts(pilot.app) == ["you: Hi there"]


async def test_enter_with_empty_input_is_a_noop():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        field = pilot.app.query_one("#personas-preview-input", Input)
        field.focus()
        await pilot.pause()
        field.value = "   "
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.replies == []
        assert _line_texts(pilot.app) == []


async def test_streaming_reply_updates_one_line_progressively():
    """begin_reply/append_reply_chunk grow a single character line in place."""
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello.")
        await pilot.pause()
        pane.begin_reply()
        pane.append_reply_chunk("Your humble ")
        await pilot.pause()
        assert _line_texts(pilot.app) == [
            "character: Hello.",
            "character: Your humble ",
        ]
        pane.append_reply_chunk("narrator.")
        await pilot.pause()
        assert _line_texts(pilot.app) == [
            "character: Hello.",
            "character: Your humble narrator.",
        ]
        assert pane.transcript_text() == (
            "character: Hello.\ncharacter: Your humble narrator."
        )
        pane.finalize_reply()
        # A finalized line is committed: a later discard must not remove it.
        await pane.discard_partial_reply()
        await pilot.pause()
        assert "character: Your humble narrator." in pane.transcript_text()


async def test_discard_partial_reply_removes_in_progress_line():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello.")
        await pilot.pause()
        pane.begin_reply()
        pane.append_reply_chunk("Half a tho")
        await pilot.pause()
        await pane.discard_partial_reply()
        await pilot.pause()
        assert _line_texts(pilot.app) == ["character: Hello."]
        assert pane.transcript_text() == "character: Hello."
        # Discard with no partial in progress is a no-op.
        await pane.discard_partial_reply()
        pane.append_user("Still works")
        await pilot.pause()
        assert pane.transcript_text() == "character: Hello.\nyou: Still works"


async def test_seed_greeting_clears_partial_reply_state():
    """A reseed mid-stream wipes the partial line; discard after is a no-op."""
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello.")
        pane.begin_reply()
        pane.append_reply_chunk("Half a tho")
        await pilot.pause()
        await pane.seed_greeting("Fresh greeting.")
        await pilot.pause()
        await pane.discard_partial_reply()
        await pilot.pause()
        assert _line_texts(pilot.app) == ["character: Fresh greeting."]
        assert pane.transcript_text() == "character: Fresh greeting."


async def test_test_reply_with_empty_input_is_a_noop():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        pilot.app.query_one("#personas-preview-input", Input).value = "   "
        pilot.app.query_one("#personas-preview-test-reply", Button).press()
        await pilot.pause()
        assert pilot.app.replies == []
        assert _line_texts(pilot.app) == []


async def test_reset_button_restores_greeting_and_posts_reset():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pane.seed_greeting("Hello.")
        pane.append_user("Hi")
        pane.append_reply("Hey.")
        pane.set_status("Ready")
        await pilot.pause()
        pilot.app.query_one("#personas-preview-reset", Button).press()
        await pilot.pause()
        assert pilot.app.resets == 1
        assert _line_texts(pilot.app) == ["character: Hello."]
        assert str(
            pilot.app.query_one("#personas-preview-status", Static).renderable
        ) == ""


async def test_open_in_console_posts_message():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        pilot.app.query_one("#personas-preview-open-console", Button).press()
        await pilot.pause()
        assert pilot.app.opens == 1


async def test_markup_like_transcript_content_renders_without_raising():
    """Greeting/user/reply text with Rich-markup-looking content renders literally."""
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pane.seed_greeting("[/oops]")
        pane.append_user("[/bad user]")
        pane.append_reply("[bold]unclosed")
        await pilot.pause()  # would raise MarkupError at render with markup on
        assert _line_texts(pilot.app) == [
            "character: [/oops]",
            "you: [/bad user]",
            "character: [bold]unclosed",
        ]


async def test_status_is_readable():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.set_status("Running")
        await pilot.pause()
        status = str(pilot.app.query_one("#personas-preview-status", Static).renderable)
        assert status == "Running"
        assert "Traceback" not in status
