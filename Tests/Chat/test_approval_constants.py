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


def test_approval_formatting_importable_without_textual() -> None:
    import subprocess
    import sys

    code = (
        "import sys, tldw_chatbook.Chat.approval_formatting as m;"
        "assert m.summarize_arguments({'a': 1}) == '{\"a\":1}';"
        "assert m.format_context_line('x') == 'x';"
        "bad = sorted(k for k in sys.modules if k == 'textual' or k.startswith('textual.'));"
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_widget_reexports_formatting_helpers() -> None:
    from tldw_chatbook.Chat import approval_formatting as fmt
    from tldw_chatbook.Widgets.Chat_Widgets import chat_approval_card as card

    for name in ("summarize_arguments", "summarize_row_arguments", "format_context_line",
                 "_summarize_arguments", "RATIONALE_DISPLAY_CAP", "CONTEXT_LABEL"):
        assert getattr(card, name) is getattr(fmt, name)
