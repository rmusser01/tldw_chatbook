"""TASK-1990: assistant transcript rows render through Textual's Markdown widget.

Covers the config gate (``[chat_defaults] assistant_markdown``, default on),
append-only streaming (never a full re-parse on prefix growth), the
chips/citation footer, role scoping (USER/SYSTEM/TOOL stay plain), and the
explicit link policy (http(s) to the browser, other schemes notify-only).
"""

import webbrowser
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Markdown

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleCitationPhase,
    ConsoleCitationPresentation,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleMarkdownMessage,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
    get_console_assistant_markdown,
)


class MarkdownHarness(App):
    CSS = "ConsoleTranscript { height: 24; }"

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _assistant(content: str, status: str = "streaming", **kwargs) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
        status=status,
        id=kwargs.pop("id", "a1"),
        **kwargs,
    )


def test_gate_resolution_defaults_and_types():
    """Missing/None/non-bool config resolves True; explicit False resolves False."""
    assert get_console_assistant_markdown(None) is True
    assert get_console_assistant_markdown({}) is True
    assert get_console_assistant_markdown({"chat_defaults": "bogus"}) is True
    assert (
        get_console_assistant_markdown({"chat_defaults": {"assistant_markdown": "no"}})
        is True
    )
    assert (
        get_console_assistant_markdown({"chat_defaults": {"assistant_markdown": False}})
        is False
    )


@pytest.mark.asyncio
async def test_assistant_rows_render_markdown_and_other_roles_stay_plain():
    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="**hi**", id="u1"
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.SYSTEM, content="## sys", id="s1"
                ),
                _assistant("", id="a1"),
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()

        assert isinstance(
            transcript.query_one("#console-message-u1"), ConsoleTranscriptMessage
        )
        assert isinstance(
            transcript.query_one("#console-message-s1"), ConsoleTranscriptMessage
        )
        row = transcript.query_one("#console-message-a1")
        assert isinstance(row, ConsoleMarkdownMessage)
        header = row.query_one(".console-markdown-header")
        assert "Generating…" in header.renderable.plain
        # Empty message: footer hidden.
        assert row.query_one(".console-markdown-footer").display is False


@pytest.mark.asyncio
async def test_config_off_keeps_plain_assistant_rows():
    app = MarkdownHarness()
    app.app_config = {"chat_defaults": {"assistant_markdown": False}}
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_assistant("## Title", status="complete")])
        await transcript.refresh_messages()
        await pilot.pause()
        row = transcript.query_one("#console-message-a1")
        assert isinstance(row, ConsoleTranscriptMessage)
        assert not isinstance(row, ConsoleMarkdownMessage)


@pytest.mark.asyncio
async def test_streaming_appends_without_reparse(monkeypatch):
    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_assistant("## Title\n\nfirst ")])
        await transcript.refresh_messages()
        await pilot.pause()

        row = transcript.query_one("#console-message-a1")
        md = row.query_one(Markdown)
        update_calls, append_calls = [], []
        original_update, original_append = md.update, md.append
        monkeypatch.setattr(
            md, "update", lambda text: update_calls.append(text) or original_update(text)
        )
        monkeypatch.setattr(
            md, "append", lambda text: append_calls.append(text) or original_append(text)
        )

        # Streaming growth: strict prefix -> append only, delta only.
        transcript.set_messages(
            [_assistant("## Title\n\nfirst **bold** and\n\n- item 1\n- item 2\n")]
        )
        await transcript.refresh_messages()
        await pilot.pause()
        assert len(append_calls) == 1
        assert not update_calls
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
        header = row.query_one(".console-markdown-header")
        assert "[streaming]" not in header.renderable.plain

        # Non-prefix change (an edit / variant switch): full update fallback.
        transcript.set_messages([_assistant("rewritten", status="complete")])
        await transcript.refresh_messages()
        await pilot.pause()
        assert len(update_calls) == 1


@pytest.mark.asyncio
async def test_footer_renders_chips_and_citation_notice():
    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        message = _assistant(
            "body text",
            status="complete",
            image_mime_type="image/png",
            attachment_label="photo.png · 11 B",
            citation_presentation=ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.CHECKING
            ),
        )
        transcript.set_messages([message])
        await transcript.refresh_messages()
        await pilot.pause()

        row = transcript.query_one("#console-message-a1")
        assert isinstance(row, ConsoleMarkdownMessage)
        footer = row.query_one(".console-markdown-footer")
        assert footer.display is True
        footer_text = footer.renderable.plain
        assert "photo.png · 11 B" in footer_text
        assert "Checking citations…" in footer_text
        # The chip never enters the markdown body (would break prefix appends).
        assert "photo.png" not in row.query_one(Markdown).source


@pytest.mark.asyncio
async def test_link_policy_http_opens_browser_other_schemes_do_not(monkeypatch):
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(
            [_assistant("[x](https://example.com)", status="complete")]
        )
        await transcript.refresh_messages()
        await pilot.pause()
        row = transcript.query_one("#console-message-a1")

        row._open_link(SimpleNamespace(href="https://example.com", stop=lambda: None))
        assert opened == ["https://example.com"]

        row._open_link(SimpleNamespace(href="file:///etc/passwd", stop=lambda: None))
        row._open_link(SimpleNamespace(href="javascript:alert(1)", stop=lambda: None))
        assert opened == ["https://example.com"]  # nothing else opened


@pytest.mark.asyncio
async def test_plain_text_export_unchanged_by_renderer():
    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_assistant("## Title\n\n**bold**", status="complete")])
        await transcript.refresh_messages()
        await pilot.pause()
        exported = transcript.to_plain_text(width=40)
        # Export keeps raw markdown text — renderer choice never rewrites it.
        assert "## Title" in exported
        assert "**bold**" in exported
