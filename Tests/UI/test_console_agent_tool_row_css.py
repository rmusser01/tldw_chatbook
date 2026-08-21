"""Console agent-transcript styling contracts in source and production CSS."""

import re
from pathlib import Path

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscriptMessage


_STYLESHEETS = (
    Path("tldw_chatbook/css/components/_agentic_terminal.tcss"),
    Path("tldw_chatbook/css/tldw_cli_modular.tcss"),
)


def _css_block(text: str, selector: str) -> str:
    """Return the declaration block for one exact selector in a selector list."""
    uncommented = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    for match in re.finditer(r"\{(?P<body>[^{}]*)\}", uncommented, re.DOTALL):
        prefix = uncommented[: match.start()]
        selector_start = max(prefix.rfind("}"), prefix.rfind(";")) + 1
        selectors = {
            item.strip() for item in prefix[selector_start : match.start()].split(",")
        }
        if selector in selectors:
            return match.group("body")
    raise AssertionError(f"missing stylesheet selector: {selector}")


def test_tool_message_row_has_tool_class():
    msg = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL, content="called calculator -> 42"
    )
    row = ConsoleTranscriptMessage(msg)
    assert "console-transcript-message-tool" in row.classes
    assert "console-transcript-message" in row.classes


def test_tool_row_class_is_styled_in_source_and_bundle():
    for path in _STYLESHEETS:
        assert ".console-transcript-message-tool" in path.read_text(encoding="utf-8")


def test_agent_rail_section_css_is_styled_in_source_and_bundle():
    for path in _STYLESHEETS:
        assert ".console-agent-section-steps" in path.read_text(encoding="utf-8")


def test_tool_diff_row_class_is_styled_in_source_and_bundle():
    """TASK-1366: the inline diff row under a file-write TOOL marker."""
    for path in _STYLESHEETS:
        assert ".console-transcript-tool-diff" in path.read_text(encoding="utf-8")


def test_assistant_turn_stylesheet_contract_in_source_and_bundle() -> None:
    """The generated app bundle must carry every authored ownership rule."""
    contract = {
        ".console-assistant-turn": (
            "height: auto;",
            "background: $ds-surface-panel;",
        ),
        ".console-assistant-activity-stack": ("height: auto;",),
        ".console-activity-disclosure": (
            "height: auto;",
            "background: $ds-surface-raised;",
        ),
        ".console-activity-header": ("height: 1;", "width: 100%;"),
        ".console-activity-label": (
            "width: 1fr;",
            "min-width: 0;",
            "text-wrap: nowrap;",
            "text-overflow: ellipsis;",
        ),
        ".console-activity-status": (
            "width: 9;",
            "min-width: 9;",
            "max-width: 9;",
            "text-align: right;",
        ),
        ".console-activity-header:focus": (
            "background: $ds-focus-bg;",
            "color: $ds-focus-fg;",
            "text-style: bold underline;",
        ),
        ".console-activity-header-selected": (
            "background: $ds-focus-bg;",
            "color: $ds-focus-fg;",
            "text-style: bold;",
        ),
        ".console-activity-disclosure-selected > .console-activity-action-stack": (
            "background: $ds-focus-bg;",
            "color: $ds-focus-fg;",
        ),
        ".console-activity-detail-stack": (
            "height: auto;",
            "border: solid $ds-grid-line;",
        ),
        ".console-activity-disclosure-expanded > .console-activity-detail-stack": (
            "background: $ds-surface-panel;",
        ),
        ".console-activity-detail-stack .console-transcript-tool-diff": (
            "padding: 0 0 0 2;",
        ),
        ".console-assistant-turn > .console-transcript-message-selected": (
            "background: $ds-focus-bg;",
            "color: $ds-focus-fg;",
        ),
    }
    status_contract = {
        ".console-activity-status-success": "$ds-status-ready",
        ".console-activity-status-blocked": "$ds-status-blocked",
        ".console-activity-status-failed": "$ds-status-error-readable",
        ".console-activity-status-done": "$ds-text-muted",
    }

    for path in _STYLESHEETS:
        text = path.read_text(encoding="utf-8")
        for selector, declarations in contract.items():
            block = _css_block(text, selector)
            for declaration in declarations:
                assert declaration in block, f"{path}: {selector} lacks {declaration}"
        for selector, token in status_contract.items():
            block = _css_block(text, selector)
            assert f"color: {token};" in block

        dark = _css_block(text, ".-dark-mode .console-assistant-turn")
        light = _css_block(text, ".-light-mode .console-assistant-turn")
        assert "border: tall $ds-chat-assistant-accent-dark;" in dark
        assert "border: tall $ds-chat-assistant-accent-light;" in light
