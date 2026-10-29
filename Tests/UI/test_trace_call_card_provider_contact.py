"""TASK-33621.5 AC#2: the card's Impact line comes from a recorded fact.

"Impact: The provider was not contacted" was hard-coded on every trace card,
including after the provider had answered HTTP 429. The line now follows the
controller's provider-contacted record (yes, no or unknown).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_trace_call_recovery import (
    TraceCallProviderContact,
    TraceProviderRejection,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    _trace_call_copy,
    trace_call_recovery_state,
)

pytestmark = pytest.mark.bootstrap_profile

NOT_CONTACTED = "Impact: The provider was not contacted. Choose how to continue."


def _paused(kind: ConsolePreparationPauseKind) -> SimpleNamespace:
    return SimpleNamespace(
        preparation_id="p1",
        state=ConsoleTurnPreparationState.PAUSED,
        pause_kind=kind,
    )


def _impact(kind, contact) -> str:
    state = trace_call_recovery_state(_paused(kind), provider_contact=contact)
    copy = _trace_call_copy(state)
    return NOT_CONTACTED if copy is None else copy[2]


@pytest.mark.parametrize(
    ("contact", "impact"),
    [
        (TraceCallProviderContact("no"), NOT_CONTACTED),
        (None, "Impact: It is not known whether the provider was contacted."),
        (
            TraceCallProviderContact("unknown"),
            "Impact: It is not known whether the provider was contacted.",
        ),
        (
            TraceCallProviderContact("yes"),
            "Impact: The provider was contacted earlier in this send; this "
            "request was not sent.",
        ),
        (
            TraceCallProviderContact(
                "yes", TraceProviderRejection("Anthropic", 504, "gateway timed out")
            ),
            "Impact: Anthropic was contacted; no reply arrived. Note: the "
            "automatic retry was not sent",
        ),
    ],
)
def test_a_trace_call_card_says_what_was_recorded(contact, impact) -> None:
    assert _impact(ConsolePreparationPauseKind.TRACE_CALL, contact).startswith(impact)


def test_a_recorded_refusal_leads_and_names_status_and_meaning() -> None:
    state = trace_call_recovery_state(
        _paused(ConsolePreparationPauseKind.TRACE_CALL),
        provider_contact=TraceCallProviderContact(
            "yes", TraceProviderRejection("OpenAI", 429, "rate limit exceeded")
        ),
    )
    title, problem, _impact_line = _trace_call_copy(state)
    assert title == "OpenAI returned HTTP 429 (rate limit exceeded)"
    assert problem == "Problem: OpenAI returned HTTP 429 (rate limit exceeded)."


@pytest.mark.parametrize(
    "kind",
    [
        ConsolePreparationPauseKind.TRACE_PROVENANCE,
        ConsolePreparationPauseKind.TEMPORARY_CAPTURE,
    ],
)
def test_pauses_before_any_call_still_say_not_contacted(kind) -> None:
    """Both pause before dispatch; a stray record never reaches their card."""
    contact = TraceCallProviderContact("yes")
    assert _impact(kind, contact) == NOT_CONTACTED
