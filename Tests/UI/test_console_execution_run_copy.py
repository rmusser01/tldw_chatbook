"""Run-chip display priority over published facts; no execution-owner substitute."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_execution_composition_values import _encode
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_execution_composition_values import (
    decode_session_composition,
)
from tldw_chatbook.Chat.console_execution_projection_values import (
    decode_session_projection,
    encode_session_projection,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


@pytest.mark.parametrize(
    "status,changes,ack,expected",
    [
        (ConsoleRunStatus.IDLE, {}, "", ""),
        (
            ConsoleRunStatus.STREAMING,
            {"pending_interrupt_copy": "Waiting for review"},
            "",
            "Waiting for review.",
        ),
        (
            ConsoleRunStatus.IDLE,
            {"response_rule_phase": "checking", "response_rule_copy": "Checking rules"},
            "",
            "Checking rules",
        ),
        (
            ConsoleRunStatus.IDLE,
            {"blocked_turn_reason": "Capture needs recovery"},
            "",
            "Blocked — Capture needs recovery",
        ),
        (ConsoleRunStatus.IDLE, {}, "Preparing…", "Preparing…"),
        (
            ConsoleRunStatus.STREAMING,
            {"run_feedback_copy": "Streaming reply…"},
            "",
            "Streaming reply…",
        ),
    ],
)
def test_selected_run_copy_uses_exact_published_facts(
    status, changes, ack, expected, monkeypatch
):
    from tldw_chatbook.UI.Console_Modules import execution_state

    session = ConsoleChatSession()
    row = decode_session_projection(encode_session_projection(session))
    facts = replace(
        decode_session_composition(_encode(session), row),
        **({"pending_interrupt_copy": "", "blocked_turn_reason": ""} | changes),
    )
    client = SimpleNamespace(session_composition=facts)
    screen = SimpleNamespace(
        _console_chat_store=SimpleNamespace(active_session_id=session.id),
        _console_chat_controller=SimpleNamespace(run_state=ConsoleRunState(status)),
        _image=SimpleNamespace(
            _h3_image_edit_registry=lambda: SimpleNamespace(active=lambda sid: None)
        ),
        _context_spend=SimpleNamespace(
            compaction_pause_expected=lambda: False,
            compaction_forecast_unavailable=lambda: False,
        ),
        _console_send_ack=SimpleNamespace(run_copy=lambda sid: ack),
    )
    monkeypatch.setattr(
        execution_state,
        "selected_client",
        lambda owner: client if owner is screen else None,
    )
    assert ChatScreen._console_active_run_copy(screen) == expected
