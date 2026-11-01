"""Fixed cutover accounting controls; no positive recovery authority is mocked."""

import inspect

import pytest

from tldw_chatbook.Chat import console_trace_read_budget as accounting


def test_cutover_issuer_is_fixed_and_cannot_change_existing_defaults():
    ordinary = accounting._TraceReadBudget()
    aggregate, native = accounting._mobile_operation_budgets()
    before = ordinary._remaining, aggregate._remaining, native._remaining
    budget = accounting._mobile_cutover_budget()
    assert type(budget) is accounting._TraceReadBudget
    assert budget._parent is None
    assert budget._remaining == 64 * 1024**2 - 512
    assert before == (16 * 1024**2, 4 * 1024**3 - 512, 16 * 1024**2)
    assert (ordinary._remaining, aggregate._remaining, native._remaining) == before
    assert accounting._mobile_budget_pair(aggregate, native)
    assert not inspect.signature(accounting._mobile_cutover_budget).parameters
    assert accounting._mobile_cutover_budget() is not budget


def test_cutover_exhaustion_refuses_without_partial_debit():
    budget = accounting._mobile_cutover_budget()
    budget.reserve(budget._remaining - 3)
    with pytest.raises(accounting._TraceReadLimit, match="mobile_trace_input_limit"):
        budget.reserve(4)
    assert budget._remaining == 3
    budget.reserve(3)
    with pytest.raises(accounting._TraceReadLimit):
        budget.reserve(1)
    assert budget._remaining == 0


@pytest.mark.parametrize("argument", [16, 128 * 1024**2])
def test_cutover_issuer_accepts_no_caller_capacity(argument):
    with pytest.raises(TypeError):
        accounting._mobile_cutover_budget(argument)


def test_depleted_cutover_counter_refuses_before_source_discovery(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import mobile_consolidation as mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_recovery import (
        _require_mobile_cutover,
    )

    budget = accounting._mobile_cutover_budget()
    budget.reserve(budget._remaining)
    issued = []

    def exhausted():
        issued.append(budget)
        return budget

    monkeypatch.setattr(accounting, "_mobile_cutover_budget", exhausted)
    monkeypatch.setattr(
        mobile,
        "_source_claim_locations",
        lambda *args: pytest.fail("source discovery preceded reservation"),
    )
    stores = mobile.CanonicalConsoleStores(
        *(tmp_path / name for name in ("chat.db", "workspaces.db", "runs.db"))
    )
    before_names = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(accounting._TraceReadLimit, match="mobile_trace_input_limit"):
        _require_mobile_cutover(tmp_path / "source", stores)
    assert issued == [budget]
    assert budget._remaining == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == before_names
