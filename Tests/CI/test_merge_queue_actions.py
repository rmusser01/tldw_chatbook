"""The merge queue's action layer and run loop, against a fake gh (spec sections 6-8)."""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "merge_queue.py"
_spec = importlib.util.spec_from_file_location("merge_queue", SCRIPT)
mq = importlib.util.module_from_spec(_spec)
sys.modules["merge_queue"] = mq
_spec.loader.exec_module(mq)

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
OLD = "a" * 40
NEW = "b" * 40


@pytest.fixture(autouse=True)
def _no_real_run_id(monkeypatch):
    """Isolate GITHUB_RUN_ID: these tests must not depend on (or be confused by) this
    process's own CI run id. Tests that exercise self-exclusion set it explicitly."""
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)


def _node(number, *, head=OLD, armed="2026-10-03T10:00:00Z", state="BEHIND", repo=None, draft=False,
          committed="2026-10-03T09:00:00Z", ref=None, threads=()):
    return {
        "number": number, "id": f"PR_{number}", "isDraft": draft, "headRefOid": head,
        "headRefName": ref or f"feat/{number}", "mergeStateStatus": state,
        "headRepository": {"nameWithOwner": repo or mq.REPO},
        "autoMergeRequest": {"enabledAt": armed} if armed else None,
        "commits": {"nodes": [{"commit": {"committedDate": committed}}]},
        "reviewThreads": {"nodes": [{"isResolved": resolved} for resolved in threads]},
    }


def _check(conclusion="success", completed="2026-10-03T11:58:00Z", url="https://run/1", suite=None):
    check = {"status": "completed", "conclusion": conclusion, "completed_at": completed, "html_url": url}
    if suite is not None:
        check["check_suite"] = {"id": suite}
    return check


class FakeGh:
    """Records every mutating call; serves scripted reads.

    Like the real API (spike evidence), the rebase mutation returns the PRE-rebase head and the
    branch moves a moment later: the first `rebase_lag` single-PR rereads after a successful
    rebase still show the old head, later ones show NEW. `rebase_lag=None`: it never moves.
    `events` holds the recorded calls interleaved with ("read_pr", head) for every single-PR read.
    """

    def __init__(self, nodes, *, checks=None, runs=None, comments=None, rebase_error=False,
                 reread=None, dispatch_refused=(), rebase_lag=1, disarm_error=False):
        self.nodes = {n["number"]: n for n in nodes}
        self.checks = checks or {}
        self.runs = runs or {}
        self.comments = comments or {}
        self.rebase_error = rebase_error
        self.reread = reread or {}
        self.dispatch_refused = set(dispatch_refused)
        self.rebase_lag = rebase_lag
        self.disarm_error = disarm_error
        self.rereads_since_rebase = None
        self.calls = []
        self.events = []
        self.reads = 0

    def _record(self, call):
        self.calls.append(call)
        self.events.append(call)

    def graphql(self, query, **v):
        if "updatePullRequestBranch" in query:
            self._record(("rebase", v["id"], v["oid"]))
            if self.rebase_error:
                raise mq.GhError("rebase refused")
            self.rereads_since_rebase = 0
            return {"data": {"updatePullRequestBranch": {"pullRequest": {"headRefOid": v["oid"]}}}}
        if "disablePullRequestAutoMerge" in query:
            self._record(("disarm", v["id"]))
            if self.disarm_error:
                raise mq.GhError("Pull request is not in the correct state to disable auto-merge")
            return {"data": {}}
        self.reads += 1
        if "comments(last" in query:
            bodies = self.comments.get(v["number"], [])
            return {"data": {"repository": {"pullRequest": {"comments": {"nodes": [{"body": b} for b in bodies]}}}}}
        if "pullRequest(number" in query:
            node = self.reread.get(v["number"], self.nodes[v["number"]])
            if self.rereads_since_rebase is not None:
                self.rereads_since_rebase += 1
                if self.rebase_lag is not None and self.rereads_since_rebase > self.rebase_lag:
                    node = dict(node, headRefOid=NEW, mergeStateStatus="BLOCKED")
            self.events.append(("read_pr", node["headRefOid"]))
            return {"data": {"repository": {"pullRequest": node}}}
        if "pullRequests(" in query:
            return {"data": {"repository": {"pullRequests": {"nodes": list(self.nodes.values())}}}}
        raise AssertionError(f"unexpected query {query[:60]!r}")

    def rest(self, method, path, fields=None):
        if method == "GET" and "/check-runs" in path:
            self.reads += 1
            return {"check_runs": self.checks.get(path.split("/commits/")[1].split("/")[0], [])}
        if method == "GET" and "/actions/runs?" in path:
            self.reads += 1
            sha = path.split("head_sha=")[1].split("&")[0]
            runs = self.runs.get(sha, [])
            if "status=action_required" in path:
                runs = [r for r in runs if r.get("conclusion") == "action_required"]
            return {"workflow_runs": runs}
        if method == "POST" and path.endswith("/dispatches"):
            workflow = path.split("/workflows/")[1].split("/")[0]
            self._record(("dispatch", workflow, dict(fields or {})))
            if workflow in self.dispatch_refused:
                raise mq.GhError("HTTP 422: Workflow does not have 'workflow_dispatch' trigger")
            return None
        if method == "POST" and path.endswith("/cancel"):
            self._record(("cancel", path.split("/runs/")[1].split("/")[0]))
            return None
        if method == "DELETE" and "/actions/runs/" in path:
            self._record(("delete", path.rsplit("/", 1)[1]))
            return None
        if method == "POST" and path.endswith("/comments"):
            self._record(("comment", int(path.split("/issues/")[1].split("/")[0]), fields["body"]))
            return None
        raise AssertionError(f"unexpected rest call {method} {path}")


def _run(gh, mode="on"):
    return mq.run(gh, mode, now=lambda: NOW, sleep=lambda s: None, log=lambda m: None)


def test_mode_values():
    for mode in ("", "off", "yes", "true", None):
        gh = FakeGh([_node(1)])
        assert _run(gh, mode) == [] and gh.calls == [] and gh.reads == 0
    for mode in ("On", " on ", "DRY"):
        gh = FakeGh([_node(1)])
        assert _run(gh, mode)[0][1].kind == "rebase"


def test_dry_mode_makes_no_mutating_calls():
    gh = FakeGh([_node(1, state="DIRTY"), _node(2, armed="2026-10-03T11:00:00Z")])
    decisions = _run(gh, "dry")
    assert [(n, a.kind) for n, a in decisions] == [(1, "evict"), (2, "rebase")]
    assert gh.calls == []


def test_on_mode_rebases_front_only():
    runs = {OLD: [
        {"id": 11, "path": ".github/workflows/derived-artifacts.yml", "event": "pull_request", "status": "in_progress", "conclusion": None},
        {"id": 12, "path": ".github/workflows/perf-guard.yml", "event": "pull_request", "status": "completed", "conclusion": "success"},
        {"id": 13, "path": ".github/workflows/task-598-platform-evidence.yml", "event": "pull_request", "status": "completed", "conclusion": "skipped"},
        {"id": 14, "path": ".github/workflows/merge-queue.yml", "event": "pull_request", "status": "completed", "conclusion": "success"},
    ]}
    gh = FakeGh([_node(1), _node(2, armed="2026-10-03T11:00:00Z")], runs=runs)
    _run(gh)
    kinds = [c[0] for c in gh.calls]
    assert ("rebase", "PR_1", OLD) in gh.calls
    assert ("cancel", "11") in gh.calls
    assert ("dispatch", "derived-artifacts.yml", {"ref": "feat/1", "inputs[pr]": "1"}) in gh.calls
    assert ("dispatch", "perf-guard.yml", {"ref": "feat/1"}) in gh.calls
    dispatched = [c[1] for c in gh.calls if c[0] == "dispatch"]
    assert "task-598-platform-evidence.yml" not in dispatched and "merge-queue.yml" not in dispatched
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert comment[1] == 1 and f"<!-- merge-queue:rebased:{NEW} -->" in comment[2]
    assert all(c[1] != "PR_2" for c in gh.calls if c[0] in ("rebase", "disarm")) and kinds.count("rebase") == 1


def test_rebase_failure_with_moved_head_never_evicts():
    gh = FakeGh([_node(1)], rebase_error=True, reread={1: _node(1, head=NEW, state="BEHIND")})
    _run(gh)
    assert [c[0] for c in gh.calls] == ["rebase"]


def test_rebase_refused_on_conflict_evicts():
    gh = FakeGh([_node(1)], rebase_error=True, reread={1: _node(1, state="DIRTY")})
    _run(gh)
    assert ("disarm", "PR_1") in gh.calls
    assert any(c[0] == "comment" and "conflicts with dev" in c[2] for c in gh.calls)


def test_evicted_front_hands_over_in_the_same_run():
    gh = FakeGh([_node(1, state="DIRTY"), _node(2, armed="2026-10-03T11:00:00Z")])
    _run(gh)
    assert gh.calls[0] == ("disarm", "PR_1")
    assert ("rebase", "PR_2", OLD) in gh.calls


def test_dispatch_refused_is_named_in_the_comment():
    runs = {OLD: [{"id": 12, "path": ".github/workflows/task-19642-smoke-clock-matrix.yml",
                   "event": "pull_request", "status": "completed", "conclusion": "success"}]}
    gh = FakeGh([_node(1)], runs=runs, dispatch_refused={"task-19642-smoke-clock-matrix.yml"})
    _run(gh)
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert "task-19642-smoke-clock-matrix.yml" in comment[2]


def test_comments_are_deduplicated_by_marker():
    gh = FakeGh([_node(1, state="DIRTY")], comments={1: [f"<!-- merge-queue:evict-conflict:{OLD} -->\nold"]})
    _run(gh)
    assert ("disarm", "PR_1") in gh.calls
    assert not any(c[0] == "comment" for c in gh.calls)


def test_eviction_for_a_new_reason_on_the_same_head_still_comments():
    """A re-armed PR evicted again on the same head, for a different reason, must be told why."""
    gh = FakeGh([_node(1, state="DIRTY")], comments={1: [f"<!-- merge-queue:evict-stuck:{OLD} -->\nold"]})
    _run(gh)
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"<!-- merge-queue:evict-conflict:{OLD} -->" in comment[2]


def test_disarm_failure_still_comments():
    """A merge fires push:dev and pull_request:closed; the losing run's disarm hits an
    already-disarmed PR. That must not crash the run or skip the comment."""
    gh = FakeGh([_node(1, state="DIRTY")], disarm_error=True)
    _run(gh)
    assert ("disarm", "PR_1") in gh.calls
    assert any(c[0] == "comment" and "conflicts with dev" in c[2] for c in gh.calls)


def test_first_failure_dispatches_a_retry():
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: [_check("failure")]})
    _run(gh)
    assert ("dispatch", "derived-artifacts.yml", {"ref": "feat/1", "inputs[pr]": "1"}) in gh.calls
    assert any(c[0] == "comment" and f"merge-queue:retry:{OLD}" in c[2] for c in gh.calls)


def test_queue_tick_failure_is_not_a_ci_failure():
    """The run failed (its queue-tick did), but its check suite reported the required check green."""
    runs = {OLD: [{"id": 9, "path": ".github/workflows/derived-artifacts.yml", "event": "pull_request",
                   "status": "completed", "conclusion": "failure", "check_suite_id": 900,
                   "updated_at": "2026-10-03T11:59:00Z"}]}
    gh = FakeGh([_node(1, state="CLEAN")], checks={OLD: [_check("success", suite=900)]}, runs=runs)
    assert _run(gh)[0][1].kind == "wait"
    assert gh.calls == []


def _broken_run(rid, conclusion, updated):
    """A required-workflow run that completed red without ever reporting the required check."""
    return {"id": rid, "path": ".github/workflows/derived-artifacts.yml", "event": "workflow_dispatch",
            "status": "completed", "conclusion": conclusion, "check_suite_id": 500 + rid,
            "updated_at": updated, "html_url": f"https://run/{rid}"}


def test_run_that_failed_without_reporting_the_check_counts_as_a_failure():
    """A startup failure (e.g. a broken derived-artifacts.yml on the branch) reports no check run.
    It must count as a failure -- one retries, two evict -- not as 'no run', which would
    re-dispatch forever."""
    once = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_broken_run(31, "startup_failure", "2026-10-03T11:00:00Z")]})
    assert _run(once)[0][1].kind == "retry"
    twice = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [
        _broken_run(31, "startup_failure", "2026-10-03T11:00:00Z"),
        _broken_run(32, "timed_out", "2026-10-03T11:30:00Z"),
    ]})
    decision = _run(twice)[0][1]
    assert decision.kind == "evict" and decision.links == ("https://run/31", "https://run/32")
    assert ("disarm", "PR_1") in twice.calls


def test_cleanup_deletes_only_bot_approval_runs():
    runs = {OLD: [
        {"id": 21, "conclusion": "action_required", "triggering_actor": {"login": "github-actions[bot]"}},
        {"id": 22, "conclusion": "action_required", "triggering_actor": {"login": "someone"}},
    ]}
    gh = FakeGh([_node(1, state="CLEAN")], checks={OLD: [_check()]}, runs=runs)
    _run(gh)
    assert ("delete", "21") in gh.calls and ("delete", "22") not in gh.calls


def test_armed_fork_gets_one_comment_and_is_never_queued():
    gh = FakeGh([_node(1, repo="someone/fork", state="BEHIND")])
    decisions = _run(gh)
    assert decisions == []
    assert [c[0] for c in gh.calls] == ["comment"]
    assert "fork" in gh.calls[0][2]


def test_unknown_state_is_reread_before_deciding():
    gh = FakeGh([_node(1, state="UNKNOWN")], reread={1: _node(1, state="DIRTY")})
    assert _run(gh)[0][1].kind == "evict"


def test_the_queue_can_never_merge_arm_or_push():
    """Spec section 7: the queue may only DISABLE auto-merge. The eviction comment's
    re-arm hint (`gh pr merge <n> --auto --merge`) is text for a human, not a call."""
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("enablePullRequestAutoMerge", "mergePullRequest", '/merge"', "/merge'", "git push", '"push"'):
        assert forbidden not in source, forbidden
    assert not re.search(r"\[\s*[\"']pr[\"']", source), "the queue never runs `gh pr ...` subcommands"


def test_gh_only_ever_calls_the_api_subcommand():
    seen = []
    gh = mq.Gh(runner=lambda args: (seen.append(args), "{}")[1])
    gh.graphql("query { viewer { login } }", n=1)
    gh.rest("POST", "repos/x/y/issues/1/comments", {"body": "b"})
    assert seen and all(args[0] == "api" for args in seen)


def test_live_required_run_without_check_waits():
    """The required check is a needs-gated aggregate: it has NO check run while its lanes are
    still running. A live derived-artifacts.yml run on the head must still read as 'in flight',
    not as 'no run at all' (which would dispatch a duplicate on every tick)."""
    runs = {OLD: [{"id": 77, "path": ".github/workflows/derived-artifacts.yml",
                   "status": "in_progress", "html_url": "https://run/77"}]}
    gh = FakeGh([_node(1, state="BLOCKED")], runs=runs)
    decisions = _run(gh)
    assert decisions[0][1].kind == "wait"
    assert gh.calls == []


def test_live_retry_run_after_one_failure_waits():
    runs = {OLD: [{"id": 78, "path": ".github/workflows/derived-artifacts.yml",
                   "status": "in_progress", "html_url": "https://run/78"}]}
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: [_check("failure")]}, runs=runs)
    decisions = _run(gh)
    assert decisions[0][1].kind == "wait"
    assert not any(c[0] == "dispatch" for c in gh.calls)


def test_own_run_is_not_counted_as_live(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "55")
    runs = {OLD: [{"id": 55, "path": ".github/workflows/derived-artifacts.yml",
                   "status": "in_progress", "html_url": "https://run/55"}]}
    gh = FakeGh([_node(1, state="BLOCKED")], runs=runs)
    decisions = _run(gh)
    assert decisions[0][1].kind == "dispatch"


def test_rebase_dispatches_before_cancelling_and_spares_the_queue(monkeypatch):
    """queue-tick runs inside its own derived-artifacts run (id 11), and a merge-queue.yml run
    (id 99) shares the old head too -- cancelling either would kill the run doing the cancelling.
    Only the unrelated perf-guard run (id 12) is fair game."""
    monkeypatch.setenv("GITHUB_RUN_ID", "11")
    runs = {OLD: [
        {"id": 99, "path": ".github/workflows/merge-queue.yml", "event": "pull_request", "status": "in_progress"},
        {"id": 11, "path": ".github/workflows/derived-artifacts.yml", "event": "pull_request", "status": "in_progress"},
        {"id": 12, "path": ".github/workflows/perf-guard.yml", "event": "pull_request", "status": "in_progress"},
    ]}
    gh = FakeGh([_node(1)], runs=runs)
    _run(gh)
    rebase_idx = next(i for i, c in enumerate(gh.calls) if c[0] == "rebase")
    assert gh.calls[rebase_idx + 1] == ("dispatch", "derived-artifacts.yml", {"ref": "feat/1", "inputs[pr]": "1"})
    assert ("cancel", "99") not in gh.calls
    assert ("cancel", "11") not in gh.calls
    assert ("cancel", "12") in gh.calls


def test_max_fronts_per_run_is_bounded():
    nodes = [_node(i, state="DIRTY") for i in range(1, 13)]
    gh = FakeGh(nodes)
    decisions = _run(gh, "dry")
    assert len(decisions) == mq.MAX_FRONTS_PER_RUN


def test_still_unknown_after_rereads_waits():
    """After each merge the next front is routinely UNKNOWN for a while: 12 rereads, 10 s apart."""
    sleeps = []
    gh = FakeGh([_node(1, state="UNKNOWN")], reread={1: _node(1, state="UNKNOWN")})
    decisions = mq.run(gh, "on", now=lambda: NOW, sleep=lambda s: sleeps.append(s), log=lambda m: None)
    assert decisions[0][1].kind == "wait"
    assert sleeps == [10] * 12


def test_first_rebase_failure_warns_without_evicting():
    gh = FakeGh([_node(1)], rebase_error=True, reread={1: _node(1, state="BEHIND")})
    _run(gh)
    assert [c[0] for c in gh.calls] == ["rebase", "comment"]
    body = gh.calls[1][2]
    assert f"<!-- merge-queue:rebase-failed:{OLD} -->" in body
    assert "rebasing onto dev failed (rebase refused); will retry once" in body


def test_repeated_rebase_failure_evicts():
    """A PR that stays BEHIND (not DIRTY) while every rebase fails would otherwise stall the line."""
    gh = FakeGh([_node(1)], rebase_error=True, reread={1: _node(1, state="BEHIND")},
                comments={1: [f"<!-- merge-queue:rebase-failed:{OLD} -->\nfirst failure"]})
    _run(gh)
    assert ("disarm", "PR_1") in gh.calls
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"<!-- merge-queue:evict-rebase:{OLD} -->" in comment[2]
    assert "rebase onto dev keeps failing: rebase refused" in comment[2]


def test_rebase_dispatches_only_after_the_new_head_appears():
    """updatePullRequestBranch returns the PRE-rebase head and the branch moves about 1 s later
    (spike). The queue polls until a reread shows the new head, and only then dispatches CI and
    comments with the NEW sha."""
    sleeps = []
    gh = FakeGh([_node(1)], rebase_lag=2)
    mq.run(gh, "on", now=lambda: NOW, sleep=sleeps.append, log=lambda m: None)
    required = ("dispatch", "derived-artifacts.yml", {"ref": "feat/1", "inputs[pr]": "1"})
    assert gh.events.index(("read_pr", NEW)) < gh.events.index(required)
    assert sleeps == [mq.REBASE_POLL_S] * 3
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"<!-- merge-queue:rebased:{NEW} -->" in comment[2] and f"`{NEW[:10]}`" in comment[2]
    assert OLD not in comment[2]


def test_rebase_whose_head_never_moves_dispatches_nothing():
    sleeps = []
    gh = FakeGh([_node(1)], rebase_lag=None)
    mq.run(gh, "on", now=lambda: NOW, sleep=sleeps.append, log=lambda m: None)
    assert [c[0] for c in gh.calls] == ["rebase"]
    assert sleeps == [mq.REBASE_POLL_S] * mq.REBASE_POLLS


def test_blocked_green_evicts_only_with_unresolved_threads():
    """A green check with BLOCKED is often mergeStateStatus lagging; only real unresolved
    conversations evict at once."""
    gh = FakeGh([_node(1, state="BLOCKED", threads=(True, False))], checks={OLD: [_check()]})
    decision = _run(gh)[0][1]
    assert decision.kind == "evict" and "unresolved conversations" in decision.reason
    gh = FakeGh([_node(1, state="BLOCKED", threads=(True,))], checks={OLD: [_check()]})
    assert _run(gh)[0][1].kind == "wait"
    assert gh.calls == []


def test_gh_argument_typing():
    seen = []
    gh = mq.Gh(runner=lambda args: (seen.append(args), "{}")[1])
    gh.graphql("query { x }", number=5, id="X")
    mq.dispatch(gh, "w.yml", "feat/1", 7)
    graphql_args, dispatch_args = seen
    assert graphql_args[graphql_args.index("-F") + 1] == "number=5"
    assert "id=X" in graphql_args
    assert "ref=feat/1" in dispatch_args and "inputs[pr]=7" in dispatch_args
