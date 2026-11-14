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


def test_budget_compatibility_exports_share_one_canonical_owner():
    from tldw_chatbook.Backup_Recovery import limits, models
    from tldw_chatbook.Chat import console_trace_read_budget as legacy

    assert legacy._TraceReadBudget is models._TraceReadBudget
    assert legacy._TraceReadLimit is models._TraceReadLimit is limits._TraceReadLimit
    assert legacy._mobile_operation_budgets is models._mobile_operation_budgets
    assert legacy._mobile_budget_pair is models._mobile_budget_pair
    assert legacy._mobile_pairs is models._mobile_pairs


def test_model_issued_pair_retains_legacy_validation_and_atomic_bounds():
    from tldw_chatbook.Backup_Recovery import models
    from tldw_chatbook.Chat import console_trace_read_budget as legacy

    aggregate, native = models._mobile_operation_budgets()
    assert legacy._require_mobile_budget_pair((aggregate, native)) is aggregate
    native.reserve(37)
    assert native._remaining == 16 * 1024**2 - 37
    assert aggregate._remaining == 4 * 1024**3 - 512 - 37
    forged = models._TraceReadBudget()
    forged._parent = aggregate
    assert not legacy._mobile_budget_pair(aggregate, forged)
    before = aggregate._remaining, forged._remaining
    with pytest.raises(ValueError, match="^mobile_budget_pair_changed$"):
        forged.reserve(1)
    assert (aggregate._remaining, forged._remaining) == before


def test_method_patch_at_legacy_class_reaches_model_issued_budget(monkeypatch):
    from tldw_chatbook.Backup_Recovery import models
    from tldw_chatbook.Chat import console_trace_read_budget as legacy

    calls = []
    original = legacy._TraceReadBudget.reserve

    def observed(self, amount):
        calls.append((self, amount))
        return original(self, amount)

    monkeypatch.setattr(legacy._TraceReadBudget, "reserve", observed)
    aggregate, native = models._mobile_operation_budgets()
    native.reserve(11)
    assert calls == [(aggregate, 512), (native, 11)]
    assert aggregate._remaining == 4 * 1024**3 - 512 - 11
    assert native._remaining == 16 * 1024**2 - 11


def test_shared_pair_registry_releases_only_the_collected_pair():
    import gc
    import weakref

    from tldw_chatbook.Backup_Recovery import models
    from tldw_chatbook.Chat import console_trace_read_budget as legacy

    aggregate, native = legacy._mobile_operation_budgets()
    aggregate_ref, native_ref = weakref.ref(aggregate), weakref.ref(native)
    assert models._mobile_pairs[native] is aggregate
    del aggregate, native
    gc.collect()
    assert aggregate_ref() is None and native_ref() is None
