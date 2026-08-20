"""TASK-1990: assistant transcript rows render through Textual's Markdown widget.

Covers the config gate (``[chat_defaults] assistant_markdown``, default on),
append-only streaming (never a full re-parse on prefix growth), the
chips/citation footer, role scoping (USER/SYSTEM/TOOL stay plain), and the
explicit link policy (http(s) to the browser, other schemes notify-only).
"""

import webbrowser
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleCitationPhase,
    ConsoleCitationPresentation,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_roleplay_identity import (
    ConsolePresentationContext,
    ConsoleTranscriptStyle,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleMarkdownMessage,
    ConsoleMessageHeader,
    ConsoleRoleplayMarkdown,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
    _resolve_textual_roleplay_blocks,
    get_console_assistant_markdown,
)


class MarkdownHarness(App):
    CSS = "ConsoleTranscript { height: 24; }"

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class StyledMarkdownHarness(MarkdownHarness):
    CSS_PATH = str(_BUNDLE)


def _painted_style_of_text(app: App, region, needle: str):
    """Return the compositor style painting ``needle`` inside a region."""
    strips = list(app.screen._compositor.render_strips())
    for y in range(region.y, region.bottom):
        if y >= len(strips):
            break
        segments = list(strips[y]._segments)
        row_text = "".join(segment.text for segment in segments)
        index = row_text.find(needle)
        if index == -1:
            continue
        offset = 0
        for segment in segments:
            if offset + len(segment.text) > index:
                return segment.style
            offset += len(segment.text)
    return None


def _relative_luminance(color) -> float:
    triplet = color.get_truecolor()

    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


def _contrast(first, second) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


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


def test_roleplay_markdown_annotates_flavor_without_rewriting_source():
    source = (
        'Narration. "Speech with **weight**, [link](https://example.com), and `code`." '
        "Then *paces with `step()` slowly* and **Listen.**"
    )
    markdown = ConsoleRoleplayMarkdown(source, open_links=False)
    blocks = markdown._build_from_source(source)
    paragraph = blocks[0]
    content = paragraph._content

    assert markdown._initial_markdown == source
    assert content.plain == (
        'Narration. "Speech with weight, link, and code." '
        "Then paces with step() slowly and Listen."
    )
    flavor_spans = [
        (content.plain[span.start : span.end], span.style)
        for span in content.spans
        if isinstance(span.style, str) and span.style.startswith(".console-rp-")
    ]
    assert ('"Speech with weight, ', ".console-rp-speech") in flavor_spans
    assert ("link", ".console-rp-speech") not in flavor_spans
    assert ("code", ".console-rp-speech") not in flavor_spans
    assert ("paces with ", ".console-rp-action") in flavor_spans
    assert ("step()", ".console-rp-action") not in flavor_spans
    assert (" slowly", ".console-rp-action") in flavor_spans
    assert ("Listen.", ".console-rp-strong") in flavor_spans


def test_roleplay_markdown_leaves_unclosed_flavor_literal():
    source = 'Narration. "unfinished speech and *unfinished action'
    markdown = ConsoleRoleplayMarkdown(source, open_links=False)
    content = markdown._build_from_source(source)[0]._content

    assert content.plain == source
    assert not any(
        isinstance(span.style, str) and span.style.startswith(".console-rp-")
        for span in content.spans
    )


def test_textual_block_compatibility_guard_falls_back(monkeypatch):
    monkeypatch.setattr(Markdown, "BLOCKS", {})

    assert _resolve_textual_roleplay_blocks() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_immersive_markdown_flavor_is_distinct_and_accessibly_painted(theme):
    app = StyledMarkdownHarness()
    async with app.run_test(size=(100, 20)) as pilot:
        app.theme = theme
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_presentation_context(
            ConsolePresentationContext(
                assistant_kind="character",
                character_name="Alraune",
                transcript_style=ConsoleTranscriptStyle.IMMERSIVE_RP,
            )
        )
        transcript.set_messages(
            [
                _assistant(
                    'Narration. "Spoken words." Then *paces slowly* and **Listen.**',
                    status="complete",
                )
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()

        row = transcript.query_one("#console-message-a1", ConsoleMarkdownMessage)
        markdown = row.query_one(ConsoleRoleplayMarkdown)
        header = row.query_one(".console-markdown-header", ConsoleMessageHeader)
        label = header.query_one(".console-transcript-speaker-label", Static)
        assert label.renderable.plain == "Alraune"

        styles = {
            role: _painted_style_of_text(app, markdown.region, text)
            for role, text in {
                "narration": "Narration.",
                "speech": '"Spoken words."',
                "action": "paces slowly",
                "strong": "Listen.",
            }.items()
        }
        assert all(style is not None for style in styles.values())
        blocks = list(markdown.query(".console-roleplay-markdown-block"))
        assert len({style.color for style in styles.values()}) == len(styles), (
            row.classes,
            [block.classes for block in blocks],
            {
                component: blocks[0].get_component_rich_style(component)
                for component in (
                    "console-rp-speech",
                    "console-rp-action",
                    "console-rp-strong",
                )
            }
            if blocks
            else {},
            styles,
        )
        for role, style in styles.items():
            assert style.color is not None and style.bgcolor is not None
            ratio = _contrast(style.color, style.bgcolor)
            assert (
                ratio >= 4.5
            ), f"{role} contrast is {ratio:.2f}:1 under {theme}; expected 4.5:1"

        transcript.select_message("a1")
        await pilot.pause()
        selected_styles = {
            _painted_style_of_text(app, markdown.region, text).color
            for text in ('"Spoken words."', "paces slowly", "Listen.")
        }
        assert len(selected_styles) == 1

        transcript.action_clear_selection()
        transcript.set_messages(
            [
                _assistant(
                    'Narration. "Spoken words." Then *paces slowly* and **Listen.**',
                    status="failed",
                )
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()
        failed_markdown = transcript.query_one(ConsoleRoleplayMarkdown)
        failed_styles = {
            _painted_style_of_text(app, failed_markdown.region, text).color
            for text in ('"Spoken words."', "paces slowly", "Listen.")
        }
        assert len(failed_styles) == 1


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
        header = row.query_one(".console-markdown-header", ConsoleMessageHeader)
        label = header.query_one(".console-transcript-speaker-label", Static)
        assert "Generating…" in label.renderable.plain
        # Empty message: footer hidden.
        assert row.query_one(".console-markdown-footer").display is False


@pytest.mark.asyncio
async def test_roleplay_renderer_preserves_full_markdown_block_structure():
    source = """# Heading

- list item

> quoted block

| Column |
| --- |
| Cell |

```python
print("literal")
```
"""
    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_assistant(source, status="complete")])
        await transcript.refresh_messages()
        await pilot.pause()

        markdown = transcript.query_one(ConsoleRoleplayMarkdown)
        widget_types = {type(widget).__name__ for widget in markdown.walk_children()}

        assert markdown.source == source
        assert {
            "ConsoleRoleplayMarkdownH1",
            "MarkdownBulletList",
            "MarkdownBlockQuote",
            "MarkdownTable",
            "MarkdownFence",
        } <= widget_types


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
        header = row.query_one(".console-markdown-header", ConsoleMessageHeader)
        label = header.query_one(".console-transcript-speaker-label", Static)
        assert "[streaming]" not in label.renderable.plain

        # Non-prefix change (an edit / variant switch): full update fallback.
        transcript.set_messages([_assistant("rewritten", status="complete")])
        await transcript.refresh_messages()
        await pilot.pause()
        assert len(update_calls) == 1


@pytest.mark.asyncio
async def test_streaming_append_activates_flavor_when_marker_closes(monkeypatch):
    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_assistant("Narration. *paces slow")])
        await transcript.refresh_messages()
        await pilot.pause()

        row = transcript.query_one("#console-message-a1", ConsoleMarkdownMessage)
        markdown = row.query_one(ConsoleRoleplayMarkdown)
        append_calls, update_calls = [], []
        original_append, original_update = markdown.append, markdown.update
        monkeypatch.setattr(
            markdown,
            "append",
            lambda text: append_calls.append(text) or original_append(text),
        )
        monkeypatch.setattr(
            markdown,
            "update",
            lambda text: update_calls.append(text) or original_update(text),
        )

        transcript.set_messages([_assistant("Narration. *paces slowly*")])
        await transcript.refresh_messages()
        await pilot.pause()

        assert append_calls == ["ly*"]
        assert update_calls == []
        assert any(
            block._content.plain[span.start : span.end] == "paces slowly"
            and span.style == ".console-rp-action"
            for block in markdown.query(".console-roleplay-markdown-block")
            for span in block._content.spans
        )


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


@pytest.mark.asyncio
async def test_roleplay_markdown_row_uses_literal_named_label_and_updates_in_place():
    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_presentation_context(
            ConsolePresentationContext(
                user_name="Captain [Rowan]",
                assistant_kind="character",
                character_name="Alraune [bold red]",
                transcript_style=ConsoleTranscriptStyle.ROLE_ACCENTS,
                revision=1,
            )
        )
        transcript.set_messages(
            [
                _assistant(
                    "Body [red]stays literal[/red]",
                    status="complete",
                    id="a-roleplay",
                )
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()

        row = transcript.query_one("#console-message-a-roleplay")
        assert isinstance(row, ConsoleMarkdownMessage)
        markdown = row.query_one(ConsoleRoleplayMarkdown)
        label = row.query_one(".console-transcript-speaker-label")
        assert label.renderable.plain == "Alraune [bold red]"
        assert "console-transcript-roleplay-character-label" in label.classes
        assert "console-transcript-message-roleplay-character" in row.classes
        assert row.query_one(Markdown).source == "Body [red]stays literal[/red]"

        transcript.set_presentation_context(
            ConsolePresentationContext(
                user_name="Captain [Rowan]",
                assistant_kind="character",
                character_name="Cecelia",
                transcript_style=ConsoleTranscriptStyle.ROLE_ACCENTS,
                revision=2,
            )
        )
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.query_one("#console-message-a-roleplay") is row
        assert label.renderable.plain == "Cecelia"

        transcript.set_presentation_context(
            ConsolePresentationContext(
                user_name="Captain [Rowan]",
                assistant_kind="character",
                character_name="Cecelia",
                transcript_style=ConsoleTranscriptStyle.IMMERSIVE_RP,
                revision=3,
            )
        )
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.query_one("#console-message-a-roleplay") is row
        assert row.query_one(ConsoleRoleplayMarkdown) is markdown
        assert "console-transcript-message-immersive-character" in row.classes
