"""TASK-372: render assistant markdown emphasis in the Console transcript.

Headings / **bold** / `code` were shown with literal marker characters. They now
render with terminal emphasis, and -- critically -- via literal-text + styled
spans, so a message can never inject Rich markup (the transcript's safety
guarantee is preserved).
"""

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_roleplay_identity import (
    ConsolePresentationContext,
    resolve_console_message_presentation,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    _markdown_body_spans,
    _message_render_text,
)


def test_markdown_spans_render_heading_bold_and_code():
    """Headings, **bold**, and `code` map to the expected styled segments."""
    from tldw_chatbook.Widgets.Console.console_transcript import _BOLD_STYLE

    assert _markdown_body_spans("### Understanding WAL") == [("Understanding WAL", "bold underline")]
    assert _markdown_body_spans("use **local RAG** now") == [
        "use ",
        ("local RAG", _BOLD_STYLE),
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


def test_plain_renderer_uses_roleplay_presentation_without_parsing_the_name():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="hello",
        status="complete",
    )
    presentation = resolve_console_message_presentation(
        message,
        ConsolePresentationContext(
            user_name="Captain [Rowan]",
            assistant_kind="character",
            character_name="Alraune [bold red]",
            revision=1,
        ),
    )

    plain = _message_render_text(
        message, selected=False, presentation=presentation
    ).plain

    assert "Alraune [bold red]" in plain
    assert "Assistant" not in plain


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


# ---- Roleplay flavor styling (task-1536) ----


def test_flavor_quoted_speech_gets_speech_style():
    """Straight double-quoted speech styles as one speech-colored span."""
    from tldw_chatbook.Widgets.Console.console_transcript import _SPEECH_STYLE

    assert _markdown_body_spans('She said "hello there" softly') == [
        "She said ",
        ('"hello there"', _SPEECH_STYLE),
        " softly",
    ]


def test_flavor_curly_quoted_speech_gets_speech_style():
    """Curly-quoted speech styles as one speech-colored span."""
    from tldw_chatbook.Widgets.Console.console_transcript import _SPEECH_STYLE

    assert _markdown_body_spans("“hello” she offered") == [
        ("“hello”", _SPEECH_STYLE),
        " she offered",
    ]


def test_flavor_single_quoted_thought_preserves_quotes_and_contraction():
    """Removing legacy thought styling collapses the phrase into narration."""
    from tldw_chatbook.Widgets.Console.console_transcript import (
        _ACTION_STYLE,
        _BOLD_STYLE,
        _SPEECH_STYLE,
    )

    spans = _markdown_body_spans("She wondered, 'I don't know.'")

    assert len(spans) == 2
    assert spans[0] == "She wondered, "
    thought, style = spans[1]
    assert thought == "'I don't know.'"
    assert "italic" in style
    assert style not in {_ACTION_STYLE, _BOLD_STYLE, _SPEECH_STYLE}


def test_flavor_curly_single_quoted_thought_preserves_curly_contraction():
    """Curly thought delimiters and a word-internal apostrophe stay visible."""
    from tldw_chatbook.Widgets.Console.console_transcript import _THOUGHT_STYLE

    assert _markdown_body_spans("She wondered, ‘I don’t know.’") == [
        "She wondered, ",
        ("‘I don’t know.’", _THOUGHT_STYLE),
    ]


def test_flavor_ordinary_apostrophe_and_unclosed_thought_stay_literal():
    """Contractions and incomplete thoughts never become thought spans."""
    assert _markdown_body_spans("Don't panic.") == ["Don't panic."]
    assert _markdown_body_spans("She wondered, 'not yet") == [
        "She wondered, 'not yet"
    ]


def test_flavor_single_asterisk_action_gets_action_style():
    """*action* text styles with the action style, markers stripped."""
    from tldw_chatbook.Widgets.Console.console_transcript import _ACTION_STYLE

    assert _markdown_body_spans("*leans forward* Tell me more") == [
        ("leans forward", _ACTION_STYLE),
        " Tell me more",
    ]


def test_flavor_bold_action_speech_are_mutually_distinct():
    """Bold, action, and speech spans use three distinct styles."""
    from tldw_chatbook.Widgets.Console.console_transcript import (
        _ACTION_STYLE,
        _BOLD_STYLE,
        _SPEECH_STYLE,
    )

    spans = _markdown_body_spans('**Loud** *quiet* "spoken"')
    styles = [s for item in spans if isinstance(item, tuple) for s in [item[1]]]
    assert styles == [_BOLD_STYLE, _ACTION_STYLE, _SPEECH_STYLE]
    assert len({_BOLD_STYLE, _ACTION_STYLE, _SPEECH_STYLE}) == 3


def test_flavor_unclosed_quote_and_asterisk_stay_literal():
    """Unclosed quote/asterisk markers stay literal (mid-stream safety)."""
    assert _markdown_body_spans('He said "wait') == ['He said "wait']
    assert _markdown_body_spans("*unfinished action") == ["*unfinished action"]


def test_flavor_quote_containing_action_marker_styles_as_speech():
    """A quote swallows markers inside it and styles wholly as speech."""
    from tldw_chatbook.Widgets.Console.console_transcript import _SPEECH_STYLE

    assert _markdown_body_spans('"I *mean* it"') == [
        ('"I *mean* it"', _SPEECH_STYLE),
    ]


def test_flavor_outer_speech_swallows_nested_single_quotes():
    """Outer speech precedence keeps nested single quotes in one span."""
    from tldw_chatbook.Widgets.Console.console_transcript import _SPEECH_STYLE

    straight = '"I said \'no\'."'
    curly = "“I said ‘no’.”"

    assert _markdown_body_spans(straight) == [(straight, _SPEECH_STYLE)]
    assert _markdown_body_spans(curly) == [(curly, _SPEECH_STYLE)]
