# Tests/Chat/test_approval_constants.py
from pathlib import Path

import tldw_chatbook.Chat.approval_constants as module


def test_module_has_no_textual_dependency() -> None:
    source = Path(module.__file__).read_text()
    assert "textual" not in source


def test_value_matches_widget_re_export() -> None:
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        TOOL_DESCRIPTION_CAPTURE_CAP as widget_value,
    )
    assert module.TOOL_DESCRIPTION_CAPTURE_CAP == widget_value
