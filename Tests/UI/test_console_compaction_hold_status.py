"""TASK-33621.4 AC#3: a send held for compaction does not read "Ready".

On dev a send held at the Ask compaction threshold left the header badge,
the run chip and Inspect's "What happens if I send now?" Run line all saying
"Ready" (live 2026-10-29, 160x45: the card said "Context limit reached; your
message is held" while Inspect said "Run: Ready"). ``blocked_turn_reason`` is
the one input those three surfaces share (TASK-33621.2); it projected only
pauses with a card state, and a compaction hold's state needs the
controller's numbers, which it never asked for.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
    ContextCompactionHold,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    blocked_turn_reason,
)

pytestmark = pytest.mark.bootstrap_profile

_HOLD = ContextCompactionHold(
    session_id="session-1",
    used_tokens=1_656,
    trigger_tokens=1_200,
    budget_tokens=1_500,
    estimated=False,
)


def _controller(*, hold: ContextCompactionHold | None) -> SimpleNamespace:
    preparation = SimpleNamespace(
        preparation_id="preparation-1",
        state=ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.CONTEXT_COMPACTION,
    )
    return SimpleNamespace(
        run_state=SimpleNamespace(status=ConsoleRunStatus.BLOCKED),
        trace_call_recovery_preparation=lambda: preparation,
        context_compaction_hold=lambda preparation_id: (
            hold if preparation_id == "preparation-1" else None
        ),
    )


def test_a_send_held_for_compaction_is_a_stuck_turn_with_a_short_reason() -> None:
    reason = blocked_turn_reason(_controller(hold=_HOLD))

    assert reason == "compaction ask"
    # The run chip shows "Run: Blocked — <reason>" inside a 30-cell cap.
    assert len(f"Blocked — {reason}") <= 30


def test_a_compaction_pause_without_a_hold_has_no_card_and_no_reason() -> None:
    assert blocked_turn_reason(_controller(hold=None)) == ""

