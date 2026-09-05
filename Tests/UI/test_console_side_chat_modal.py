"""Ephemeral Console side-chat modal (selection menu phase 2, task 4).

Covers: More Details auto-send on mount + streaming into the reply area, Ask
mode waiting for Send, Stop cancellation (Cancelling… then the cancelled
outcome with Retry — and, in Ask mode, Send returning so an edited prompt is
submittable while Retry resends the last prompt unchanged), provider errors
surfacing inline with Retry (mid-stream errors keep the streamed partial
text), Escape mid-stream cancelling the worker and dismissing, reply-buffer
tail capping, the read-only quote block, the provider·model summary line,
and worker isolation (non-exclusive ``console-side-chat`` group). All
against the real ``ConsoleSideChatService`` backed by a fake gateway — no
live LLM.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from textual.app import App
from textual.screen import Screen
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_side_chat import (
    SIDE_CHAT_BUFFER_CAP,
    ConsoleSideChatService,
)
from tldw_chatbook.Widgets.Console.console_side_chat_modal import (
    ConsoleSideChatModal,
)

# ---------------------------------------------------------------------------
# Fakes (mirrors Tests/Chat/test_console_side_chat_service.py)
# ---------------------------------------------------------------------------


@dataclass
class FakeResolution:
    provider: str = "openai"
    model: str = "gpt-test"
    ready: bool = True
    visible_copy: str = ""


class FakeGateway:
    """Records selections/messages; replays chunks, errors, or blocks."""

    def __init__(
        self,
        chunks: list[Any] | None = None,
        error: Exception | None = None,
        resolution: FakeResolution | None = None,
        block_after_first_chunk: bool = False,
        error_after_chunks: bool = False,
    ) -> None:
        self.chunks = chunks if chunks is not None else ["Hello", " world"]
        self.error = error
        self.resolution = resolution or FakeResolution()
        self.block_after_first_chunk = block_after_first_chunk
        # When False (default) the error fires before any chunk; when True
        # all chunks stream first and the error lands mid-stream afterwards.
        self.error_after_chunks = error_after_chunks
        self.first_chunk_sent = asyncio.Event()
        self.messages: list[list[dict[str, str]]] = []
        self.stream_calls = 0

    async def resolve_for_send(
        self, selection: ConsoleProviderSelection
    ) -> FakeResolution:
        return self.resolution

    async def stream_chat(
        self,
        resolution: FakeResolution,
        messages: list[dict[str, str]],
        *,
        route=None,
    ):
        self.stream_calls += 1
        self.messages.append(messages)
        if self.error is not None and not self.error_after_chunks:
            raise self.error
        for index, chunk in enumerate(self.chunks):
            yield chunk
            if self.block_after_first_chunk and index == 0:
                self.first_chunk_sent.set()
                await asyncio.Event().wait()  # never set: holds the stream open
        if self.error is not None:
            raise self.error


QUOTE = "the quoted transcript text"
AUTO_PROMPT = "Explain this"
_SESSION_SELECTION = ConsoleProviderSelection(provider="openai")


def _service(gateway: FakeGateway) -> ConsoleSideChatService:
    return ConsoleSideChatService(gateway)


def _modal(
    gateway: FakeGateway,
    *,
    auto: bool = True,
    quote: str = QUOTE,
    provider_selection: ConsoleProviderSelection | None = _SESSION_SELECTION,
    sidechat_model: str = "",
    callback: Any = None,
) -> ConsoleSideChatModal:
    return ConsoleSideChatModal(
        service=_service(gateway),
        provider_selection=provider_selection,
        sidechat_model=sidechat_model,
        quote=quote,
        auto_send_prompt=AUTO_PROMPT if auto else None,
        callback=callback,
    )


class _SideChatApp(App[None]):
    CSS = "Screen { align: center middle; }"

    def __init__(self) -> None:
        super().__init__()
        self.results: list[object] = []


def _text(modal: ConsoleSideChatModal, selector: str) -> str:
    return str(modal.query_one(selector, Static).render())


def _displayed(modal: ConsoleSideChatModal, selector: str) -> bool:
    return bool(modal.query_one(selector).display)


async def _await_status(
    modal: ConsoleSideChatModal, expected: str, timeout: float = 5.0
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while _text(modal, "#console-side-chat-status") != expected:
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(
                f"status never became {expected!r}; "
                f"last={_text(modal, '#console-side-chat-status')!r}"
            )
        await asyncio.sleep(0.02)


async def _await_button_settled(
    modal: ConsoleSideChatModal, selector: str, timeout: float = 5.0
) -> None:
    """Wait out Textual's ``-active`` click debounce: a second click on a
    still-``-active`` Button is intentionally swallowed by ``_on_click``."""
    button = modal.query_one(selector, Button)
    deadline = asyncio.get_running_loop().time() + timeout
    while button.has_class("-active"):
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(f"{selector} never settled after its click animation")
        await asyncio.sleep(0.02)


async def _await_stream_calls(
    gateway: FakeGateway, expected: int, timeout: float = 5.0
) -> None:
    """Wait until the gateway has seen ``expected`` stream_chat calls."""
    deadline = asyncio.get_running_loop().time() + timeout
    while gateway.stream_calls < expected:
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(
                f"stream_calls never reached {expected}; "
                f"last={gateway.stream_calls}"
            )
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# More Details (auto-send) mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_more_details_auto_sends_on_mount_and_streams_into_reply():
    gateway = FakeGateway(chunks=["Hel", "lo ", "world"])
    modal = _modal(gateway)

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as _pilot:
        await app.push_screen(modal, callback=app.results.append)
        await _await_status(modal, "Complete.")

        assert _text(modal, "#console-side-chat-reply") == "Hello world"
        # One send, carrying the rendered prompt and the quoted selection.
        assert gateway.stream_calls == 1
        user_content = gateway.messages[0][1]["content"]
        assert AUTO_PROMPT in user_content
        assert QUOTE in user_content
        # Auto mode hides the prompt input and all ask/stream affordances.
        assert not _displayed(modal, "#console-side-chat-prompt")
        assert not _displayed(modal, "#console-side-chat-send")
        assert not _displayed(modal, "#console-side-chat-stop")
        assert not _displayed(modal, "#console-side-chat-retry")
        assert _displayed(modal, "#console-side-chat-close")
        # Worker isolation: non-exclusive, side-chat-only group (spec §2).
        worker = modal._sidechat_worker
        assert worker is not None
        assert worker.group == "console-side-chat"


# ---------------------------------------------------------------------------
# Ask (freeform) mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_mode_waits_for_send_then_streams():
    gateway = FakeGateway(chunks=["Answer"])
    modal = _modal(gateway, auto=False)

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        prompt_area = modal.query_one("#console-side-chat-prompt", TextArea)
        assert prompt_area.display  # visible and empty in Ask mode
        assert prompt_area.text == ""
        assert gateway.stream_calls == 0  # nothing sent until Send

        # Blank Send is refused inline.
        await pilot.click("#console-side-chat-send")
        await pilot.pause()
        assert gateway.stream_calls == 0
        assert _text(modal, "#console-side-chat-status") == "Type a question first."

        prompt_area.text = "What does this mean?"
        await _await_button_settled(modal, "#console-side-chat-send")
        await pilot.click("#console-side-chat-send")
        await _await_status(modal, "Complete.")

        assert gateway.stream_calls == 1
        assert "What does this mean?" in gateway.messages[0][1]["content"]
        assert _text(modal, "#console-side-chat-reply") == "Answer"


# ---------------------------------------------------------------------------
# Stop / cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_cancels_then_shows_cancelled_outcome_and_retry_reruns():
    gateway = FakeGateway(chunks=["partial "], block_after_first_chunk=True)
    modal = _modal(gateway)

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await asyncio.wait_for(gateway.first_chunk_sent.wait(), timeout=5)

        # The intermediate "Cancelling…" state is set synchronously by Stop.
        modal._stop_streaming()
        assert _text(modal, "#console-side-chat-status") == "Cancelling…"

        await _await_status(modal, "Cancelled.")
        assert "partial" in _text(modal, "#console-side-chat-reply")
        assert _displayed(modal, "#console-side-chat-retry")
        assert not _displayed(modal, "#console-side-chat-stop")

        # Retry re-runs the same prompt.
        await pilot.click("#console-side-chat-retry")
        await pilot.pause()
        assert gateway.stream_calls == 2
        assert "Explain this" in gateway.messages[1][1]["content"]

        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_stop_button_cancels_a_streaming_side_chat():
    gateway = FakeGateway(chunks=["partial "], block_after_first_chunk=True)
    modal = _modal(gateway)

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await asyncio.wait_for(gateway.first_chunk_sent.wait(), timeout=5)

        await pilot.click("#console-side-chat-stop")
        await _await_status(modal, "Cancelled.")

        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_ask_mode_after_stop_send_uses_edited_text_and_retry_uses_original():
    """Ask mode must not dead-end after a stop: the editable prompt area is
    still up, so Send (reads the TextArea) returns alongside Retry (resends
    the last prompt unchanged)."""
    gateway = FakeGateway(chunks=["partial "], block_after_first_chunk=True)
    modal = _modal(gateway, auto=False)

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        prompt_area = modal.query_one("#console-side-chat-prompt", TextArea)
        prompt_area.text = "original question"
        await _await_button_settled(modal, "#console-side-chat-send")
        await pilot.click("#console-side-chat-send")
        await asyncio.wait_for(gateway.first_chunk_sent.wait(), timeout=5)

        modal._stop_streaming()
        await _await_status(modal, "Cancelled.")
        assert _displayed(modal, "#console-side-chat-send")
        assert _displayed(modal, "#console-side-chat-retry")
        assert not _displayed(modal, "#console-side-chat-stop")

        # Retry resends the last prompt unchanged, ignoring the edited area.
        prompt_area.text = "edited question"
        await _await_button_settled(modal, "#console-side-chat-retry")
        await pilot.click("#console-side-chat-retry")
        await _await_stream_calls(gateway, 2)
        assert "original question" in gateway.messages[1][1]["content"]
        assert "edited question" not in gateway.messages[1][1]["content"]

        # Stop again, then Send submits the edited TextArea content.
        modal._stop_streaming()
        await _await_status(modal, "Cancelled.")
        await _await_button_settled(modal, "#console-side-chat-send")
        await pilot.click("#console-side-chat-send")
        await _await_stream_calls(gateway, 3)
        assert "edited question" in gateway.messages[2][1]["content"]

        await pilot.press("escape")
        await pilot.pause()


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_error_shows_inline_and_retry_reruns():
    gateway = FakeGateway(error=ChatProviderError("safe copy"))
    modal = _modal(gateway)

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await _await_status(modal, "Provider error: safe copy")

        assert _displayed(modal, "#console-side-chat-retry")
        assert not _displayed(modal, "#console-side-chat-stop")
        assert _text(modal, "#console-side-chat-reply") == ""

        await pilot.click("#console-side-chat-retry")
        await pilot.pause()
        assert gateway.stream_calls == 2

        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_ask_mode_provider_error_keeps_send_for_edited_prompt():
    gateway = FakeGateway(error=ChatProviderError("safe copy"))
    modal = _modal(gateway, auto=False)

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        prompt_area = modal.query_one("#console-side-chat-prompt", TextArea)
        prompt_area.text = "first question"
        await _await_button_settled(modal, "#console-side-chat-send")
        await pilot.click("#console-side-chat-send")
        await _await_status(modal, "Provider error: safe copy")

        # Send must stay available alongside Retry so an edited prompt can
        # be submitted without reopening the modal.
        assert _displayed(modal, "#console-side-chat-send")
        assert _displayed(modal, "#console-side-chat-retry")

        prompt_area.text = "second question"
        await _await_button_settled(modal, "#console-side-chat-send")
        await pilot.click("#console-side-chat-send")
        await _await_stream_calls(gateway, 2)
        assert "second question" in gateway.messages[1][1]["content"]

        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_provider_error_mid_stream_keeps_partial_reply():
    """A mid-stream provider error must not blank the already-streamed text."""
    gateway = FakeGateway(
        chunks=["partial answer"],
        error=ChatProviderError("mid-stream boom"),
        error_after_chunks=True,
    )
    modal = _modal(gateway)

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await _await_status(modal, "Provider error: mid-stream boom")

        assert _text(modal, "#console-side-chat-reply") == "partial answer"
        assert _displayed(modal, "#console-side-chat-retry")

        await pilot.press("escape")
        await pilot.pause()


# ---------------------------------------------------------------------------
# Escape mid-stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escape_mid_stream_cancels_worker_and_dismisses():
    gateway = FakeGateway(chunks=["partial "], block_after_first_chunk=True)
    modal = _modal(gateway)

    app = _SideChatApp()
    host: Screen[None] | None = None
    async with app.run_test(size=(100, 40)) as pilot:
        host = app.screen
        await app.push_screen(modal, callback=app.results.append)
        await asyncio.wait_for(gateway.first_chunk_sent.wait(), timeout=5)

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert app.screen is host
        assert app.results == [None]
        worker = modal._sidechat_worker
        assert worker is not None
        assert worker.is_cancelled or worker.is_finished


# ---------------------------------------------------------------------------
# Reply buffer cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_buffer_is_tail_capped():
    raw = "a" * (SIDE_CHAT_BUFFER_CAP + 10_000)
    gateway = FakeGateway(chunks=[raw])
    modal = _modal(gateway)

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as _pilot:
        await app.push_screen(modal, callback=app.results.append)
        await _await_status(modal, "Complete.")

        displayed = _text(modal, "#console-side-chat-reply")
        assert displayed.startswith("…\n")
        assert len(displayed) == SIDE_CHAT_BUFFER_CAP + 2
        assert displayed.endswith(raw[-SIDE_CHAT_BUFFER_CAP:])


# ---------------------------------------------------------------------------
# Quote + identity lines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quote_is_displayed_read_only():
    gateway = FakeGateway(chunks=["ok"])
    modal = _modal(gateway, auto=False, quote="a very specific quote")

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        assert _text(modal, "#console-side-chat-quote") == "a very specific quote"
        assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_summary_line_shows_requested_then_resolved_identity():
    gateway = FakeGateway(
        chunks=["ok"],
        resolution=FakeResolution(provider="zai", model="glm-test"),
        block_after_first_chunk=True,
    )
    modal = _modal(
        gateway,
        provider_selection=ConsoleProviderSelection(
            provider="openai", explicit_model="gpt-4o"
        ),
        sidechat_model="",
    )

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as _pilot:
        await app.push_screen(modal, callback=app.results.append)
        await asyncio.wait_for(gateway.first_chunk_sent.wait(), timeout=5)

        # Mid-stream the header still shows the REQUESTED identity.
        identity = _text(modal, "#console-side-chat-identity")
        assert "openai" in identity and "gpt-4o" in identity

        # The resolution arrives with the outcome and replaces it.
        modal._sidechat_worker.cancel()
        await _await_status(modal, "Cancelled.")
        resolved = _text(modal, "#console-side-chat-identity")
        assert "zai" in resolved and "glm-test" in resolved


@pytest.mark.asyncio
async def test_summary_line_prefers_configured_sidechat_model():
    gateway = FakeGateway(chunks=["ok"], block_after_first_chunk=True)
    modal = _modal(
        gateway,
        provider_selection=ConsoleProviderSelection(
            provider="openai", explicit_model="gpt-4o"
        ),
        sidechat_model="mistral/mistral-large-latest",
    )

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await asyncio.wait_for(gateway.first_chunk_sent.wait(), timeout=5)
        identity = _text(modal, "#console-side-chat-identity")
        assert "mistral/mistral-large-latest" in identity

        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_close_button_dismisses_without_result():
    gateway = FakeGateway(chunks=["ok"])
    modal = _modal(gateway)

    app = _SideChatApp()
    async with app.run_test(size=(100, 40)) as pilot:
        host = app.screen
        await app.push_screen(modal, callback=app.results.append)
        await _await_status(modal, "Complete.")

        await pilot.click("#console-side-chat-close")
        await pilot.pause()
        assert app.screen is host
        assert app.results == [None]
