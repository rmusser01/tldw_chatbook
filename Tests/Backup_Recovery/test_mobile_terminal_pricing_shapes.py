"""Local terminal sizing mechanics, not a positive recovery authority proof."""

import copy
import json

import pytest

from tldw_chatbook.Backup_Recovery.mobile_runs_goal_budget import (
    _json_width,
    _serialization,
)
from tldw_chatbook.Backup_Recovery.mobile_terminal_pricing import (
    _COUNT,
    _UINT64,
    _check_terminal_dimensions,
)
from tldw_chatbook.Chat.console_trace_read_budget import (
    _TraceReadBudget,
    _TraceReadLimit,
)


def test_non_bmp_native_json_width_and_atomic_preallocation():
    value = {"path": '/tmp/😀/\\"', "count": _COUNT}
    budget = _TraceReadBudget()
    width = _json_width(value, budget)
    assert width >= len(json.dumps(value, sort_keys=True, separators=(",", ":")))
    budget.reserve(budget._remaining)
    with pytest.raises(_TraceReadLimit):
        _serialization(value, budget, encoded=True)


def test_terminal_actual_max_dimensions_and_overflow():
    from tldw_chatbook.Backup_Recovery import bootstrap

    value = {
        "targets": [
            {
                "witnesses": [{"device": _UINT64, "inode": _UINT64, "mode": _UINT64}],
                "sidecars": [],
            }
        ],
        "owners": [{"entry_count": _COUNT, "command_history_count": _COUNT}],
        "pending": {"device": _UINT64, "inode": _UINT64, "size": bootstrap.MAX_RECORD},
        "bootstrap": {"device": _UINT64, "inode": _UINT64},
    }
    _check_terminal_dimensions(value, _TraceReadBudget())
    for group, field in (("pending", "size"), ("bootstrap", "inode")):
        bad = copy.deepcopy(value)
        bad[group][field] += 1
        with pytest.raises(ValueError, match="mobile_terminal_dimension_invalid"):
            _check_terminal_dimensions(bad, _TraceReadBudget())
    bad = copy.deepcopy(value)
    bad["owners"][0]["entry_count"] += 1
    with pytest.raises(ValueError, match="mobile_terminal_dimension_invalid"):
        _check_terminal_dimensions(bad, _TraceReadBudget())


@pytest.mark.parametrize("encoded", [False, True])
def test_shared_model_json_accounting_covers_native_escaped_encoding(encoded):
    from tldw_chatbook.Backup_Recovery import models

    text = 'native\\"\x00\x7f😀'
    copies = 2 if encoded else 1
    budget = _TraceReadBudget()
    before = budget._remaining
    raw = json.dumps(text, ensure_ascii=True)
    models._serialization(text, budget, encoded=encoded)
    assert before - budget._remaining == copies * (64 + 4 * len(raw))

    value = {"text": text, "values": [None, True, False, 2**4096, -(2**4096), 1.25]}
    raw = json.dumps(value, ensure_ascii=True, allow_nan=False)
    budget = _TraceReadBudget()
    before = budget._remaining
    models._serialization(value, budget, encoded=encoded)
    assert before - budget._remaining >= copies * (64 + 4 * len(raw))

    budget = _TraceReadBudget()
    before = budget._remaining
    with pytest.raises(ValueError, match="^native_goal_json_shape_unproved$"):
        models._serialization(object(), budget, encoded=encoded)
    assert budget._remaining == before
