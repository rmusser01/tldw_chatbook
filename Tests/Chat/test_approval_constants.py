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


def test_interrupt_defaults_keep_original_controller_aliases_and_values() -> None:
    from tldw_chatbook.Chat import console_chat_controller as controller

    expected = {
        "_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS": 0.0,
        "_MCP_APPROVAL_POLL_SECONDS": 1.0,
        "_DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS": 0.0,
        "_DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS": 0.0,
        "_DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS": 0.0,
        "_CHAT_CREATE_SESSION_GONE": "session_gone",
        "_DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS": 0.0,
        "_DEFAULT_ASK_USER_TIMEOUT_SECONDS": 0.0,
        "ASK_USER_TIMEOUT_ENV_VAR": "TLDW_CONSOLE_ASK_USER_TIMEOUT_SECONDS",
        "_MAX_TRACKED_QUESTION_BOUNCE_RUNS": 64,
        "_LEGACY_PENDING_APPROVAL_ROUND_ID": "__legacy_pending_approval__",
        "_APPROVAL_SCOPE_RANK": {
            "approve_once": 1,
            "approve_session": 2,
            "always_allow": 3,
        },
        "_APPROVING_DECISIONS": ("approve_once", "approve_session", "always_allow"),
    }
    for name, value in expected.items():
        canonical = getattr(module, name)
        assert canonical == value
        assert getattr(controller, name) is canonical


def test_interrupt_host_reads_original_controller_patch_seams_live(monkeypatch) -> None:
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    owner = controller.ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=None
    )
    host = owner._interrupt_host
    monkeypatch.setattr(controller, "_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS", 13.5)
    monkeypatch.setattr(controller, "_MCP_APPROVAL_POLL_SECONDS", 0.01)
    assert host.read_global__DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS() == 13.5
    assert host.read_global__MCP_APPROVAL_POLL_SECONDS() == 0.01
    assert module._DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS == 0.0
    assert module._MCP_APPROVAL_POLL_SECONDS == 1.0


def test_scope_inventory_reexport_keeps_shared_mutable_identity(monkeypatch) -> None:
    from tldw_chatbook.Chat import console_chat_controller as controller

    assert controller._APPROVAL_SCOPE_RANK is module._APPROVAL_SCOPE_RANK
    monkeypatch.setitem(controller._APPROVAL_SCOPE_RANK, "approve_once", 42)
    assert module._APPROVAL_SCOPE_RANK["approve_once"] == 42


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
