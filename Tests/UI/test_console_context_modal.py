import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Label, Static, TextArea

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_context_modal import ConsoleContextModal


SNAPSHOT = ConsoleContextSnapshot(
    current_messages=[
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="Hello"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="Hi"),
    ],
    next_send_payload={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    },
)

EMPTY_SNAPSHOT = ConsoleContextSnapshot(current_messages=[], next_send_payload={})


async def _snapshot_factory() -> ConsoleContextSnapshot:
    return SNAPSHOT


async def _empty_factory() -> ConsoleContextSnapshot:
    return EMPTY_SNAPSHOT


class ModalHarness(App):
    def compose(self) -> ComposeResult:
        yield Static("background")

    def on_mount(self) -> None:
        self.push_screen(ConsoleContextModal(_snapshot_factory, token_estimate=42))


@pytest.mark.asyncio
async def test_context_modal_renders_tabs():
    app = ModalHarness()

    async with app.run_test(size=(100, 40)) as _pilot:
        modal = app.screen
        header = modal.query_one("#console-context-header", Static)
        header_text = str(header.renderable)
        assert "Chat Context" in header_text
        assert "42 tokens" in header_text

        current_container = modal.query_one(
            "#console-context-current-body", Vertical
        )
        text_areas = current_container.query(TextArea)
        assert any("Hello" in ta.text for ta in text_areas)

        next_container = modal.query_one(
            "#console-context-next-send-body", Vertical
        )
        labels = list(next_container.query(Label))
        assert any("gpt-4" in str(label.renderable) for label in labels)


@pytest.mark.asyncio
async def test_context_modal_empty_state():
    app = ModalHarness()
    app._push_empty = lambda: app.push_screen(
        ConsoleContextModal(_empty_factory)
    )

    async with app.run_test(size=(100, 40)) as pilot:
        app._push_empty()
        await pilot.pause()
        modal = app.screen
        current_container = modal.query_one(
            "#console-context-current-body", Vertical
        )
        labels = list(current_container.query(Label))
        assert any(
            "No conversation context" in str(label.renderable)
            for label in labels
        )


@pytest.mark.asyncio
async def test_context_modal_in_progress_warning():
    app = ModalHarness()
    app._push_in_progress = lambda: app.push_screen(
        ConsoleContextModal(_snapshot_factory, in_progress=True)
    )

    async with app.run_test(size=(100, 40)) as pilot:
        app._push_in_progress()
        await pilot.pause()
        modal = app.screen
        warning = modal.query_one("#console-context-warning", Static)
        assert "in progress" in str(warning.renderable)
        refresh_button = modal.query_one("#console-context-refresh", Button)
        assert refresh_button.disabled
