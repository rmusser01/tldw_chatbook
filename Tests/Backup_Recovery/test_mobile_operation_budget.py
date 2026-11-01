"""The fixed public envelope cannot enlarge the original native proof allowance."""

import inspect

import pytest

from tldw_chatbook.Chat.console_trace_read_budget import (
    _TraceReadBudget,
    _TraceReadLimit,
)


def test_ordinary_native_budget_stays_sixteen_mib_and_refuses_atomically():
    budget = _TraceReadBudget()
    assert budget._remaining == 16 * 1024**2
    budget.reserve(budget._remaining)
    with pytest.raises(_TraceReadLimit):
        budget.reserve(1)
    assert budget._remaining == 0
    assert not inspect.signature(_TraceReadBudget).parameters


def test_fixed_pair_charges_native_and_aggregate_once():
    from tldw_chatbook.Chat.console_trace_read_budget import _mobile_operation_budgets

    aggregate, native = _mobile_operation_budgets()
    assert type(aggregate) is type(native) is _TraceReadBudget
    assert native._remaining == 16 * 1024**2
    assert 4 * 1024**3 - 1024 <= aggregate._remaining <= 4 * 1024**3
    before = aggregate._remaining
    native.reserve(37)
    assert native._remaining == 16 * 1024**2 - 37
    assert aggregate._remaining == before - 37
    assert not inspect.signature(_mobile_operation_budgets).parameters


@pytest.mark.parametrize("limiting", ["native", "aggregate"])
def test_either_bound_refuses_without_charging_either_counter(limiting):
    from tldw_chatbook.Chat.console_trace_read_budget import _mobile_operation_budgets

    aggregate, native = _mobile_operation_budgets()
    selected = native if limiting == "native" else aggregate
    selected.reserve(selected._remaining - 4)
    before = aggregate._remaining, native._remaining
    with pytest.raises(_TraceReadLimit):
        native.reserve(5)
    assert (aggregate._remaining, native._remaining) == before
    native.reserve(4)
    assert (aggregate._remaining, native._remaining) == (before[0] - 4, before[1] - 4)


def test_before_effect_capacity_check_honors_aggregate_without_consuming():
    from tldw_chatbook.Chat.console_trace_read_budget import _mobile_operation_budgets

    aggregate, native = _mobile_operation_budgets()
    aggregate.reserve(aggregate._remaining - 3)
    before = aggregate._remaining, native._remaining
    native.require_capacity(3)
    with pytest.raises(_TraceReadLimit):
        native.require_capacity(4)
    assert (aggregate._remaining, native._remaining) == before
