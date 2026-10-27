"""TASK-34100.2 review round 1 (F2): the composer's send gate.

The send-outcome blockers (a 401 'key rejected', a 404 'model not found') hard-
blocked the composer with the wrong words -- 'Add an API key' above a key that
exists, 'Select a model' above a selected model -- and only a reply could clear
them, which no blocked composer can produce. They now explain the last failure
on every surface but never refuse the next send (that send is the retry). The
capacity blocker does refuse, and names its own fix instead of 'finish provider
setup'.

Rebase onto dev (TASK-33621.11): dev's ``console_display_state`` already
owns the composer's two gate functions (``console_send_gate_setup_copy``,
``console_setup_blocker``), so this branch's duplicate ``console_send_gate``
module is gone and the rule lives in dev's functions, gated on
``ConsoleSettingsReadiness.refuses_turn``.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_display_state import (
    build_console_disabled_reason,
    console_send_gate_setup_copy as send_blocked_copy,
    console_setup_blocked_reason,
    console_setup_blocker,
)
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_SEND_OUTCOMES,
    ConsoleSessionSettings,
    build_console_settings_readiness,
)

KEY = "sk-test-console-send-gate"
OPENAI = {"api_settings": {"openai": {"api_key": KEY}}}


@pytest.fixture(autouse=True)
def _fresh_ledger():
    CONSOLE_SEND_OUTCOMES.clear()
    yield
    CONSOLE_SEND_OUTCOMES.clear()


def _readiness(model: str = "gpt-4o", max_tokens: int | None = None):
    return build_console_settings_readiness(
        ConsoleSessionSettings(provider="openai", model=model, max_tokens=max_tokens),
        app_config=OPENAI,
        environ={},
    )


def readiness_refuses_turn(readiness) -> bool:
    return readiness.refuses_turn


def _composer_line(readiness) -> str:
    setup_reason = console_setup_blocker(readiness)
    return build_console_disabled_reason(
        action_id="send",
        has_draft=True,
        send_blocked=bool(setup_reason),
        setup_blocked_reason=setup_reason,
    )


@pytest.mark.parametrize("status", [401, 404])
def test_a_failed_last_send_never_refuses_the_next_one(status: int) -> None:
    CONSOLE_SEND_OUTCOMES.record_status("openai", "gpt-4o", KEY, status)
    readiness = _readiness()

    assert readiness.operability == "not_ready"  # every surface still says why
    assert readiness_refuses_turn(readiness) is False
    assert send_blocked_copy(readiness) == ""
    assert console_setup_blocked_reason(readiness) == ""


@pytest.mark.parametrize("status", [401, 404])
def test_a_retry_in_flight_reads_as_an_active_run(status: int) -> None:
    """The retry a past 401/404 now allows is a run like any other: while it
    is in flight the verdict is the active run (which keeps Settings from
    mutating mid-run), not the soft outcome of the send before it."""
    CONSOLE_SEND_OUTCOMES.record_status("openai", "gpt-4o", KEY, status)
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="openai", model="gpt-4o"),
        app_config=OPENAI,
        environ={},
        active_run=True,
    )

    assert readiness.blocker == "active_run"
    assert readiness.recovery_action == "wait_for_active_run"


def test_a_full_context_refuses_and_names_its_fix() -> None:
    readiness = _readiness(max_tokens=128_000)  # gpt-4o's whole 128k window

    assert readiness_refuses_turn(readiness) is True
    send = send_blocked_copy(readiness)
    setup = console_setup_blocked_reason(readiness)
    assert "context window" in send and "response maximum" in send
    assert "Finish provider setup" not in send
    assert "context window" in setup and "Fix model limits" in setup
    # Rebase onto dev (TASK-33005.7): the strip's copy fits it whole.
    assert _composer_line(readiness) == "Send blocked — fix model limits to continue"


def test_provider_setup_blockers_keep_their_copy() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="openai", model="gpt-4o"),
        app_config={"api_settings": {"openai": {}}},
        environ={},
    )

    assert readiness_refuses_turn(readiness) is True
    assert send_blocked_copy(readiness) == (
        "Console send blocked: Add an API key for OpenAI before sending."
    )
    assert console_setup_blocked_reason(readiness) == (
        "Add API key in Settings > Providers & Models before sending."
    )


def test_a_ready_verdict_is_never_refused() -> None:
    readiness = _readiness()

    assert readiness_refuses_turn(readiness) is False
    assert send_blocked_copy(readiness) == ""
    assert console_setup_blocked_reason(readiness) == ""
