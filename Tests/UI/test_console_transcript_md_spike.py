"""SPIKE smoke tests: env-gated Markdown-widget rendering for assistant rows.

Not intended to merge — validates the TLDW_CONSOLE_MD_SPIKE prototype:
assistant rows become ConsoleMarkdownMessage, streaming growth goes through
Markdown.append() (never a full re-parse), and non-prefix edits fall back to
Markdown.update().
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Markdown

import tldw_chatbook.Widgets.Console.console_transcript as ct
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleMarkdownMessage,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)


class SpikeHarness(App):
    CSS = "ConsoleTranscript { height: 24; }"

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _assistant(content: str, status: str = "streaming") -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
        status=status,
        id="a1",
    )


@pytest.mark.asyncio
async def test_spike_renders_assistant_rows_as_markdown(monkeypatch):
    monkeypatch.setattr(ct, "_MD_SPIKE_ENABLED", True)
    app = SpikeHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="**hi**", id="u1"
                ),
                _assistant(""),
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()

        user_row = transcript.query_one("#console-message-u1")
        assert isinstance(user_row, ConsoleTranscriptMessage)  # USER stays plain
        row = transcript.query_one("#console-message-a1")
        assert isinstance(row, ConsoleMarkdownMessage)
        header = row.query_one(".console-md-spike-header")
        assert "Generating…" in header.renderable.plain


@pytest.mark.asyncio
async def test_spike_streams_via_append_not_reparse(monkeypatch):
    monkeypatch.setattr(ct, "_MD_SPIKE_ENABLED", True)
    app = SpikeHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_assistant("## Title\n\nfirst ")])
        await transcript.refresh_messages()
        await pilot.pause()

        row = transcript.query_one("#console-message-a1")
        md = row.query_one(Markdown)
        update_calls = []
        append_calls = []
        original_update = md.update
        original_append = md.append
        monkeypatch.setattr(
            md, "update", lambda text: update_calls.append(text) or original_update(text)
        )
        monkeypatch.setattr(
            md, "append", lambda text: append_calls.append(text) or original_append(text)
        )

        # Streaming growth: strict prefix -> append only.
        transcript.set_messages(
            [_assistant("## Title\n\nfirst **bold** and\n\n- item 1\n- item 2\n")]
        )
        await transcript.refresh_messages()
        await pilot.pause()
        assert len(append_calls) == 1
        assert not update_calls
        assert append_calls[0].startswith("**bold**") or append_calls[0].startswith(
            "**"
        ) is False  # delta is the tail, not the whole body
        assert "## Title" not in append_calls[0]

        # Completion with identical content: no body work at all.
        transcript.set_messages(
            [
                _assistant(
                    "## Title\n\nfirst **bold** and\n\n- item 1\n- item 2\n",
                    status="complete",
                )
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()
        assert len(append_calls) == 1
        assert not update_calls
        header = row.query_one(".console-md-spike-header")
        assert "[streaming]" not in header.renderable.plain

        # Non-prefix change (an edit): full update fallback.
        transcript.set_messages([_assistant("rewritten", status="complete")])
        await transcript.refresh_messages()
        await pilot.pause()
        assert len(update_calls) == 1


@pytest.mark.asyncio
async def test_gate_off_keeps_plain_rows(monkeypatch):
    monkeypatch.setattr(ct, "_MD_SPIKE_ENABLED", False)
    app = SpikeHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_assistant("## Title", status="complete")])
        await transcript.refresh_messages()
        await pilot.pause()
        row = transcript.query_one("#console-message-a1")
        assert isinstance(row, ConsoleTranscriptMessage)
        assert not isinstance(row, ConsoleMarkdownMessage)
