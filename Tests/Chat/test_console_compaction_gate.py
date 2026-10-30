"""TASK-33621.4 AC#2: turns that cannot show the hold card stop before commit.

Continue, Regenerate and Retry over the Ask trigger used to commit an
assistant row, mark it Failed and explain afterwards ("Use Compact now in
Chat settings ..."). The gate asks the same side-effect-free preflight probe
the composer hold uses, and refuses before anything is committed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_compaction_gate import refuse_at_ask_trigger
from tldw_chatbook.Chat.console_context_budget_copy import ask_trigger_refusal_copy
from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode
from tldw_chatbook.Chat.console_turn_preparation import ContextCompactionHold

_HOLD = ContextCompactionHold(
    session_id="session-1",
    used_tokens=1_656,
    trigger_tokens=1_200,
    budget_tokens=1_500,
    estimated=False,
)


class _Controller:
    def __init__(self, *, mode, hold) -> None:
        self.mode = mode
        self.hold = hold
        self.blocked: list[tuple[str, str]] = []
        self.assessed: list[dict] = []
        self.store = SimpleNamespace(sessions=lambda: [SimpleNamespace(id="session-1")])

    def _compaction_mode_for(self, _session):
        return self.mode

    async def _assess_context_compaction(self, **kwargs):
        self.assessed.append(kwargs)
        return self.hold, None

    def _block(self, session_id: str, copy: str):
        self.blocked.append((session_id, copy))
        return SimpleNamespace(accepted=False, visible_copy=copy)


def _run(controller):
    return asyncio.run(
        refuse_at_ask_trigger(
            controller, "session-1", object(), [{"role": "user", "content": "x"}], "continued"
        )
    )


def test_over_the_ask_trigger_the_turn_is_refused_before_commit() -> None:
    controller = _Controller(mode=ContextCompactionMode.ASK, hold=_HOLD)

    result = _run(controller)

    assert result is not None and not result.accepted
    assert controller.assessed[0]["uncommitted_user_message_id"] is None
    (session_id, copy), = controller.blocked
    assert session_id == "session-1"
    assert copy.startswith("Not continued:")
    assert "1,656 of 1,500 tokens" in copy and "1,200" in copy
    assert "/compact" in copy


@pytest.mark.parametrize(
    ("mode", "hold"),
    [
        (ContextCompactionMode.ASK, None),  # below the trigger
        (ContextCompactionMode.AUTOMATIC, _HOLD),  # compacts on its own
        (ContextCompactionMode.OFF, _HOLD),
    ],
)
def test_otherwise_the_turn_goes_on_unprobed_or_unblocked(mode, hold) -> None:
    controller = _Controller(mode=mode, hold=hold)

    assert _run(controller) is None
    assert controller.blocked == []
    if mode is not ContextCompactionMode.ASK:
        assert controller.assessed == []  # no probe cost outside Ask


def test_the_refusal_copy_flags_an_assumed_window() -> None:
    copy = ask_trigger_refusal_copy(
        action="regenerated",
        used_tokens=30_000,
        budget_tokens=32_000,
        trigger_tokens=25_600,
        estimated=True,
    )

    assert copy.startswith("Not regenerated:")
    assert "assumed" in copy
