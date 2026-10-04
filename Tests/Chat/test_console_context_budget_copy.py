"""TASK-34350: copy for a Console send the context budget cannot take.

The owner ruling (2026-10-03): a send over budget prompts to compact,
auto-compacts when that is enabled, or alerts when compacting cannot make
room. The alert must name what fills the window and the setting that changes
it, and say when the window is only an estimate and where to set the real one.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_context_budget_copy import (
    MAX_TOKENS_SETTING,
    MODEL_WINDOW_SETTING,
    ContextOverflowCause,
    compaction_hold_detail,
    compaction_prompt_copy,
    context_overflow_alert_copy,
)


def _alert(cause, *, estimated=False, **overrides):
    values = dict(
        model="gpt-test",
        window_tokens=4_096,
        window_estimated=estimated,
        response_tokens=4_096,
        input_ceiling_tokens=0,
    )
    values.update(overrides)
    return context_overflow_alert_copy(cause, **values)


@pytest.mark.parametrize("cause", list(ContextOverflowCause))
def test_every_alert_says_the_message_was_not_sent(cause) -> None:
    assert _alert(cause).startswith("Your message was not sent:")


def test_reservation_alert_names_max_tokens_and_its_value() -> None:
    copy = _alert(ContextOverflowCause.NO_INPUT_CAPACITY)

    assert "4,096-token context window" in copy
    assert "Max tokens (4,096)" in copy
    assert "compacting older turns cannot make room" in copy
    assert MAX_TOKENS_SETTING in copy


def test_mandatory_alert_names_what_fills_the_window() -> None:
    copy = _alert(
        ContextOverflowCause.MANDATORY_EXCEEDS,
        window_tokens=128_000,
        response_tokens=4_096,
        input_ceiling_tokens=121_344,
    )

    assert "system prompt, tools and attached context" in copy
    assert "121,344 tokens" in copy
    assert MAX_TOKENS_SETTING in copy


def test_nothing_to_compact_alert_offers_shortening_or_a_new_chat() -> None:
    copy = _alert(ContextOverflowCause.NOTHING_TO_COMPACT, window_tokens=32_000)

    assert "no older complete turns to compact" in copy
    assert "start a new chat" in copy


@pytest.mark.parametrize("cause", list(ContextOverflowCause))
def test_an_estimated_window_is_called_an_estimate_with_where_to_fix_it(cause) -> None:
    copy = _alert(cause, estimated=True)

    assert "an estimate" in copy
    assert MODEL_WINDOW_SETTING in copy


@pytest.mark.parametrize("cause", list(ContextOverflowCause))
def test_a_verified_window_is_not_called_an_estimate(cause) -> None:
    copy = _alert(cause, estimated=False)

    assert "estimate" not in copy


def test_compaction_prompt_states_the_threshold_and_the_choice() -> None:
    copy = compaction_prompt_copy(
        used_tokens=1_650, budget_tokens=1_800, estimated=False
    )

    assert "1,650 of 1,800 tokens" in copy
    for choice in ("Compact and send", "Send without compacting", "Cancel"):
        assert choice in copy
    assert "estimate" not in copy


def test_compaction_prompt_flags_an_estimated_budget() -> None:
    copy = compaction_prompt_copy(
        used_tokens=30_000, budget_tokens=32_000, estimated=True
    )

    assert "estimate" in copy
    assert MODEL_WINDOW_SETTING in copy


def test_hold_detail_gives_the_numbers_and_what_each_choice_does() -> None:
    usage, choices = compaction_hold_detail(
        used_tokens=1_650, budget_tokens=1_800, estimated=False
    )

    assert "1,650 of 1,800 tokens" in usage
    assert "estimate" not in usage
    assert "Nothing was sent" in choices
    assert "one extra model call" in choices
    assert "back in the composer" in choices


def test_hold_detail_flags_an_estimated_window() -> None:
    usage, _choices = compaction_hold_detail(
        used_tokens=30_000, budget_tokens=32_000, estimated=True
    )

    assert "estimate" in usage
    assert MODEL_WINDOW_SETTING in usage
