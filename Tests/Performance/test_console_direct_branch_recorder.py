"""Repeated observed branches retain multiplicity without relaxing the bound."""

import pytest

from Tests.UI.test_console_pending_facts_direct_control_cold_readiness import (
    _OriginalColdRead,
)

pytestmark = pytest.mark.bootstrap_profile


def test_direct_branch_recorder_counts_repeats_and_retains_distinct_limit():
    recorder = object.__new__(_OriginalColdRead)
    recorder.rows, recorder.invalid, recorder._branch_rows = [], [], {}
    row = {"site": "checked_status", "result": False}
    for _ in range(100):
        recorder._record_branch(row)
    row["result"] = True
    assert recorder.rows == [
        {"site": "checked_status", "result": False, "observations": 100}
    ]
    for value in range(63):
        recorder._record_branch({"site": "other", "value": value})
    assert len(recorder.rows) == 64 and not recorder.invalid
    recorder._record_branch({"site": "checked_status", "result": False})
    assert recorder.rows[0]["observations"] == 101 and not recorder.invalid
    recorder._record_branch({"site": "overflow", "value": 64})
    assert recorder.invalid == ["bounded_branch_receipt_overflow"]
    assert len(recorder.rows) == 64
    recorder._record_branch({"site": "bad", "value": object()})
    assert recorder.invalid[-1] == "nonplain_branch_receipt"
    assert len(recorder.rows) == 64
