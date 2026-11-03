"""Compatibility exports and fixed cutover accounting for retained trace reads."""

from tldw_chatbook.Backup_Recovery.limits import (
    _TraceReadLimit as _TraceReadLimit,  # noqa: PLC0414 - canonical compatibility export
)
from tldw_chatbook.Backup_Recovery.models import (
    _mobile_budget_pair,
    _TraceReadBudget,
)
from tldw_chatbook.Backup_Recovery.models import (
    _mobile_operation_budgets as _mobile_operation_budgets,  # noqa: PLC0414 - canonical compatibility export
)
from tldw_chatbook.Backup_Recovery.models import (
    _mobile_pairs as _mobile_pairs,  # noqa: PLC0414 - canonical compatibility export
)


def _mobile_cutover_budget():
    """Issue one fixed allowance for a complete ordinary cutover observation."""
    budget = _TraceReadBudget()
    budget._remaining = 64 * 1024**2
    budget.reserve(512)
    return budget


def _require_mobile_budget_pair(pair):
    if type(pair) is not tuple or len(pair) != 2 or not _mobile_budget_pair(*pair):
        raise ValueError("mobile_budget_pair_changed")
    return pair[0]
