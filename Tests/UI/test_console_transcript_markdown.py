"""TASK-372: render assistant markdown emphasis in the Console transcript.

Headings / **bold** / `code` were shown with literal marker characters. They now
render with terminal emphasis, and -- critically -- via literal-text + styled
spans, so a message can never inject Rich markup (the transcript's safety
guarantee is preserved).
"""

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Widgets.Console.console_transcript import (
    _markdown_body_spans,
    _message_render_text,
)


def test_markdown_spans_render_heading_bold_and_code():
    """Headings, **bold**, and `code` map to the expected styled segments."""
    assert _markdown_body_spans("### Understanding WAL") == [("Understanding WAL", "bold underline")]
    assert _markdown_body_spans("use **local RAG** now") == [
        "use ",
        ("local RAG", "bold"),
        " now",
    ]
    assert _markdown_body_spans("run `pytest` here") == [
        "run ",
        ("pytest", "italic"),
        " here",
    ]


def test_markdown_spans_leave_unclosed_markers_literal():
    """A half-streamed (unclosed) bold marker stays literal until it closes."""
    assert _markdown_body_spans("**bold not closed") == ["**bold not closed"]


def test_assistant_render_strips_heading_markers():
    """An assistant heading renders without its literal ``#`` markers."""
    msg = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="#### The Checkpointing Process",
        status="complete",
    )
    plain = _message_render_text(msg, selected=False).plain
    assert "The Checkpointing Process" in plain
    assert "####" not in plain


def test_assistant_render_does_not_interpret_injected_markup():
    """Bracket tokens in an assistant reply render literally, never as markup."""
    msg = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="danger [bold red]x[/bold red] end",
        status="complete",
    )
    plain = _message_render_text(msg, selected=False).plain
    # The bracket tokens survive as literal text (not parsed/stripped as markup).
    assert "[bold red]x[/bold red]" in plain


def test_user_message_text_is_left_verbatim():
    """User input is shown verbatim -- its markdown markers are not restyled."""
    msg = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="literal **stars** and ## hash",
        status="complete",
    )
    plain = _message_render_text(msg, selected=False).plain
    assert "**stars**" in plain
    assert "## hash" in plain


def test_system_and_tool_messages_are_left_verbatim():
    """Qodo #823: SYSTEM diagnostics and TOOL output keep their literal markers
    (their #/**/backtick characters may be meaningful) -- only ASSISTANT renders."""
    for role in (ConsoleMessageRole.SYSTEM, ConsoleMessageRole.TOOL):
        msg = ConsoleChatMessage(
            role=role,
            content="### not a heading  **not bold**  `not code`",
            status="complete",
        )
        plain = _message_render_text(msg, selected=False).plain
        assert "### not a heading" in plain
        assert "**not bold**" in plain
        assert "`not code`" in plain
