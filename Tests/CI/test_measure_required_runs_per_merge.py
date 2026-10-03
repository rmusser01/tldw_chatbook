"""Classifier and summary of the merge-queue success measure (spec section 13)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("measure", ROOT / "scripts" / "measure_required_runs_per_merge.py")
m = importlib.util.module_from_spec(_spec)
sys.modules["measure"] = m
_spec.loader.exec_module(m)


def test_classify_orders_causes():
    sync, rebase = {"s1"}, {"r1"}
    assert m.classify({"event": "pull_request", "conclusion": "action_required", "head_sha": "r1"}, sync, rebase) is None
    assert m.classify({"event": "workflow_dispatch", "conclusion": "success", "head_sha": "r1"}, sync, rebase) == "queue"
    assert m.classify({"event": "pull_request", "conclusion": "success", "head_sha": "s1"}, sync, rebase) == "sync"
    assert m.classify({"event": "pull_request", "conclusion": "failure", "head_sha": "r1"}, sync, rebase) == "rebase"
    assert m.classify({"event": "pull_request", "conclusion": "success", "head_sha": "c1"}, sync, rebase) == "content"


def test_summarize_counts_resync_beyond_one():
    per_pr = [{"content": 2, "sync": 3}, {"content": 1, "queue": 1}, {"content": 1}]
    s = m.summarize(per_pr)
    assert s == {"prs": 3, "runs": 8, "resync_median": 1, "resync_beyond_one": 2}


def test_summarize_empty():
    assert m.summarize([]) == {"prs": 0, "runs": 0, "resync_median": 0, "resync_beyond_one": 0}
