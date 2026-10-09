"""Table tests for the merge queue's pure rules (spec section 6)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("merge_queue", ROOT / "scripts" / "merge_queue.py")
mq = importlib.util.module_from_spec(_spec)
sys.modules['merge_queue'] = mq
_spec.loader.exec_module(mq)

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _pr(**kw) -> "mq.PrState":
    base = dict(
        number=1, node_id="PR_1", head_sha="a" * 40, head_ref="feat/x", same_repo=True,
        armed_at=NOW - timedelta(hours=1), is_draft=False, merge_state="CLEAN",
        head_committed_at=NOW - timedelta(hours=2), checks=(),
    )
    base.update(kw)
    return mq.PrState(**base)


def _check(conclusion="success", minutes_ago=1, status="completed", url="u") -> "mq.CheckRun":
    done = NOW - timedelta(minutes=minutes_ago) if status == "completed" else None
    return mq.CheckRun(status=status, conclusion=conclusion if status == "completed" else None, completed_at=done, url=url)


def test_line_is_armed_non_draft_same_repo_oldest_first():
    a = _pr(number=1, armed_at=NOW - timedelta(minutes=5))
    b = _pr(number=2, armed_at=NOW - timedelta(minutes=50))
    unarmed = _pr(number=3, armed_at=None)
    draft = _pr(number=4, is_draft=True)
    fork = _pr(number=5, same_repo=False)
    assert [p.number for p in mq.line_of([a, b, unarmed, draft, fork])] == [2, 1]


def test_rearmed_pr_rejoins_at_the_back():
    first = _pr(number=1, armed_at=NOW - timedelta(minutes=30))
    rearmed = _pr(number=2, armed_at=NOW - timedelta(minutes=1))  # was first before eviction
    assert [p.number for p in mq.line_of([rearmed, first])] == [1, 2]


@pytest.mark.parametrize(
    ("pr", "kind"),
    [
        (_pr(merge_state="UNKNOWN"), "wait"),
        (_pr(merge_state="BEHIND"), "rebase"),
        (_pr(merge_state="DIRTY"), "evict"),
        (_pr(merge_state="BLOCKED", checks=(_check(status="in_progress"),)), "wait"),
        (_pr(merge_state="BLOCKED", checks=()), "start"),
        (_pr(merge_state="BLOCKED", checks=(_check("cancelled"),)), "start"),
        (_pr(merge_state="BLOCKED", checks=(), head_committed_at=NOW - timedelta(minutes=2)), "wait"),
        (_pr(merge_state="BLOCKED", checks=(), head_committed_at=NOW + timedelta(minutes=5)), "start"),
        (_pr(merge_state="BLOCKED", checks=(_check("failure"),)), "retry"),
        (_pr(merge_state="BLOCKED", checks=(_check("failure", minutes_ago=30), _check("failure", minutes_ago=2))), "evict"),
        (_pr(merge_state="CLEAN", checks=(_check(minutes_ago=5),)), "wait"),
        (_pr(merge_state="UNSTABLE", checks=(_check(minutes_ago=5),)), "wait"),
        (_pr(merge_state="CLEAN", checks=(_check(minutes_ago=16),)), "evict"),
        (_pr(merge_state="BLOCKED", checks=(_check(minutes_ago=1),), unresolved_threads=1), "evict"),
        (_pr(merge_state="BLOCKED", checks=(_check(minutes_ago=1),), unresolved_threads=0), "wait"),
        (_pr(merge_state="CLEAN", checks=(_check("failure", minutes_ago=20), _check(minutes_ago=3))), "wait"),
    ],
    ids=[
        "unknown-waits", "behind-rebases", "dirty-evicts", "running-waits", "no-run-starts",
        "only-cancelled-starts", "young-head-waits", "future-dated-head-starts", "first-failure-retries", "second-failure-evicts",
        "green-clean-waits", "green-unstable-waits", "stuck-green-evicts", "green-blocked-unresolved-evicts",
        "green-blocked-nothing-unresolved-waits", "retry-that-passed-waits",
    ],
)
def test_decide_front_table(pr, kind):
    assert mq.decide_front(pr, NOW).kind == kind


def test_second_failure_links_both_runs():
    pr = _pr(merge_state="BLOCKED", checks=(_check("failure", 30, url="r1"), _check("failure", 2, url="r2")))
    assert mq.decide_front(pr, NOW).links == ("r1", "r2")


def test_neutral_counts_as_passing():
    pr = _pr(merge_state="CLEAN", checks=(_check("neutral", minutes_ago=2),))
    assert mq.decide_front(pr, NOW).kind == "wait"
