"""TOOL transcript rows carry a dim role class, pinned in source + bundle."""
from pathlib import Path

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscriptMessage


def test_tool_message_row_has_tool_class():
    msg = ConsoleChatMessage(role=ConsoleMessageRole.TOOL, content="called calculator -> 42")
    row = ConsoleTranscriptMessage(msg)
    assert "console-transcript-message-tool" in row.classes
    assert "console-transcript-message" in row.classes


def test_tool_row_class_is_styled_in_source_and_bundle():
    for path in (
        Path("tldw_chatbook/css/components/_agentic_terminal.tcss"),
        Path("tldw_chatbook/css/tldw_cli_modular.tcss"),
    ):
        assert ".console-transcript-message-tool" in path.read_text(encoding="utf-8")
