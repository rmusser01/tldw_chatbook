"""The merge queue's action layer and run loop, against a fake gh (spec sections 6-8)."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
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
          committed="2026-10-03T09:00:00Z", ref=None, threads=(), author="User"):
    return {
        "number": number, "id": f"PR_{number}", "isDraft": draft, "headRefOid": head,
        "headRefName": ref or f"feat/{number}", "mergeStateStatus": state,
        "author": {"__typename": author} if author else None,
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


# gh's real error format for API failures: `gh: <message> (HTTP NNN)` (verified live).
DISPATCH_ERRORS = {
    422: "gh: Workflow does not have 'workflow_dispatch' trigger (HTTP 422)",
    404: "gh: No ref found for: feat/1 (HTTP 404)",
    403: "gh: Resource not accessible by integration (HTTP 403)",
    502: "gh: Server Error (HTTP 502)",
}


class FakeGh:
    """Records every mutating call; serves scripted reads.

    Like the real API (spike evidence), the rebase mutation returns the PRE-rebase head and the
    branch moves a moment later: the first `rebase_lag` single-PR rereads after a successful
    rebase still show the old head, later ones show NEW. `rebase_lag=None`: it never moves.
    `events` holds the recorded calls interleaved with ("read_pr", head) for every single-PR read.
    A `reread` value may be a list: successive single-PR reads walk it, the last entry sticks.
    Lists page like the real API: the PR line `line_page_size` at a time through a cursor, REST
    lists `mq.PER_PAGE` at a time. `late_runs[sha]` joins the head's runs from its second
    unfiltered runs read on, i.e. a run another queue run started after this one decided.
    A dispatch of a `dispatch_refused` workflow fails with `DISPATCH_ERRORS[dispatch_status]`.
    """

    def __init__(self, nodes, *, checks=None, runs=None, comments=None, rebase_error=False,
                 reread=None, dispatch_refused=(), rebase_lag=1, disarm_error=False,
                 line_page_size=None, late_runs=None, dispatch_status=422, approve_error=None, rerun_error=None,
                 run_status=None):
        self.nodes = {n["number"]: n for n in nodes}
        self.checks = checks or {}
        self.runs = runs or {}
        self.comments = comments or {}
        self.rebase_error = rebase_error
        self.reread = reread or {}
        self.dispatch_refused = set(dispatch_refused)
        self.rebase_lag = rebase_lag
        self.disarm_error = disarm_error
        self.line_page_size = line_page_size
        self.late_runs = late_runs or {}
        self.dispatch_status = dispatch_status
        self.approve_error = approve_error
        self.rerun_error = rerun_error
        # run_status[run_id]: successive statuses a single-run read walks (the last one sticks).
        self.run_status = run_status or {}
        self.run_reads = {}
        self.runs_reads = {}
        self.reread_counts = {}
        self.rereads_since_rebase = None
        self.calls = []
        self.events = []
        self.reads = 0
        self.line_cursors = []

    def _set_run(self, rid, **fields):
        for runs in self.runs.values():
            for run in runs:
                if run.get("id") == rid:
                    run.update(fields)

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
            if isinstance(node, list):
                seen = self.reread_counts.get(v["number"], 0)
                self.reread_counts[v["number"]] = seen + 1
                node = node[min(seen, len(node) - 1)]
            if self.rereads_since_rebase is not None:
                self.rereads_since_rebase += 1
                if self.rebase_lag is not None and self.rereads_since_rebase > self.rebase_lag:
                    node = dict(node, headRefOid=NEW, mergeStateStatus="BLOCKED")
            self.events.append(("read_pr", node["headRefOid"]))
            return {"data": {"repository": {"pullRequest": node}}}
        if "pullRequests(" in query:
            nodes = list(self.nodes.values())
            self.line_cursors.append(v.get("after"))
            size = self.line_page_size or len(nodes) or 1
            start = int(v.get("after") or 0)
            more = start + size < len(nodes)
            return {"data": {"repository": {"pullRequests": {
                "pageInfo": {"hasNextPage": more, "endCursor": str(start + size) if more else None},
                "nodes": nodes[start:start + size],
            }}}}
        raise AssertionError(f"unexpected query {query[:60]!r}")

    @staticmethod
    def _page(items, path):
        page = int(re.search(r"[?&]page=(\d+)", path).group(1))
        assert f"per_page={mq.PER_PAGE}" in path
        return items[(page - 1) * mq.PER_PAGE:page * mq.PER_PAGE]

    def rest(self, method, path, fields=None):
        if method == "GET" and "/check-runs" in path:
            self.reads += 1
            checks = self.checks.get(path.split("/commits/")[1].split("/")[0], [])
            return {"total_count": len(checks), "check_runs": self._page(checks, path)}
        if method == "GET" and "/actions/runs?" in path:
            self.reads += 1
            sha = path.split("head_sha=")[1].split("&")[0]
            runs = self.runs.get(sha, [])
            if "status=action_required" in path:
                runs = [r for r in runs if r.get("conclusion") == "action_required"]
            elif "page=1" in re.findall(r"[?&](page=\d+)", path):
                self.runs_reads[sha] = self.runs_reads.get(sha, 0) + 1
            if self.runs_reads.get(sha, 0) >= 2 and "status=" not in path:
                runs = runs + self.late_runs.get(sha, [])
            return {"total_count": len(runs), "workflow_runs": self._page(runs, path)}
        if method == "POST" and path.endswith("/dispatches"):
            workflow = path.split("/workflows/")[1].split("/")[0]
            self._record(("dispatch", workflow, dict(fields or {})))
            if workflow in self.dispatch_refused:
                raise mq.GhError(DISPATCH_ERRORS[self.dispatch_status])
            return None
        if method == "POST" and path.endswith("/approve"):
            rid = path.split("/runs/")[1].split("/")[0]
            self._record(("approve", rid))
            if self.approve_error:
                raise mq.GhError(self.approve_error)
            self._set_run(int(rid), status="queued", conclusion=None)  # GitHub starts it
            return None
        if method == "POST" and path.endswith(("/rerun", "/rerun-failed-jobs")):
            rid = path.split("/runs/")[1].split("/")[0]
            self._record(("rerun", rid, "failed" if path.endswith("-failed-jobs") else "all"))
            if self.rerun_error:
                raise mq.GhError(self.rerun_error)
            self._set_run(int(rid), status="queued", conclusion=None)
            return None
        if method == "GET" and re.search(r"/actions/runs/\d+$", path):
            rid = int(path.rsplit("/", 1)[1])
            seen = self.run_reads.get(rid, 0)
            self.run_reads[rid] = seen + 1
            if rid in self.run_status:
                statuses = self.run_status[rid]
                return {"id": rid, "status": statuses[min(seen, len(statuses) - 1)]}
            run = next((r for runs in self.runs.values() for r in runs if r.get("id") == rid), {"status": "completed"})
            return dict(run)
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


def _held(rid, workflow="derived-artifacts.yml", actor="github-actions[bot]", event="pull_request"):
    """A run GitHub holds for approval because the queue's token caused its event (spec F4)."""
    return {"id": rid, "path": f".github/workflows/{workflow}", "event": event, "status": "completed",
            "conclusion": "action_required", "triggering_actor": {"login": actor}, "html_url": f"https://run/{rid}"}


def _pr_run(rid, suite, conclusion="failure", status="completed"):
    """A required-workflow pull_request run whose check suite is `suite`."""
    return {"id": rid, "path": ".github/workflows/derived-artifacts.yml", "event": "pull_request",
            "status": status, "conclusion": conclusion, "check_suite_id": suite,
            "created_at": "2026-10-03T11:00:00Z", "updated_at": "2026-10-03T11:58:00Z", "html_url": f"https://run/{rid}"}


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
        {"id": 14, "path": ".github/workflows/merge-queue.yml", "event": "pull_request", "status": "completed", "conclusion": "success"},
    ], NEW: [_held(31), _held(32, "perf-guard.yml")]}
    gh = FakeGh([_node(1), _node(2, armed="2026-10-03T11:00:00Z")], runs=runs)
    _run(gh)
    kinds = [c[0] for c in gh.calls]
    assert ("rebase", "PR_1", OLD) in gh.calls
    assert ("cancel", "11") in gh.calls
    assert ("approve", "31") in gh.calls and ("approve", "32") in gh.calls
    assert "dispatch" not in kinds, "a dispatched check never counts toward mergeability (spec V4)"
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert comment[1] == 1 and f"<!-- merge-queue:rebased:{NEW} -->" in comment[2] and "approved its CI" in comment[2]
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


def test_rebase_whose_held_runs_never_appear_says_so_and_dispatches_nothing():
    sleeps = []
    gh = FakeGh([_node(1)])
    mq.run(gh, "on", now=lambda: NOW, sleep=sleeps.append, log=lambda m: None)
    assert sleeps[-mq.HELD_RUN_POLLS:] == [mq.HELD_RUN_POLL_S] * mq.HELD_RUN_POLLS
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert "not approved yet; a later tick approves it" in comment[2]
    assert not any(c[0] in ("approve", "dispatch") for c in gh.calls)


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


def test_first_failure_reruns_the_failed_run_in_its_own_check_suite():
    """A re-run lands in the same check suite, replacing the failure for branch protection
    (spec V3); a dispatched run would not count at all (V4)."""
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: [_check("failure", suite=700)]},
                runs={OLD: [_pr_run(70, 700)]})
    _run(gh)
    assert ("rerun", "70", "failed") in gh.calls
    assert not any(c[0] == "dispatch" for c in gh.calls)
    assert any(c[0] == "comment" and f"merge-queue:retry:{OLD}" in c[2] and "re-running it" in c[2] for c in gh.calls)


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


def test_only_the_queues_own_held_pull_request_runs_are_approved():
    """A held run another actor caused, or a held non-pull_request run, is not the queue's to approve."""
    runs = {OLD: [_held(21), _held(22, actor="someone"), _held(23, event="workflow_dispatch")]}
    gh = FakeGh([_node(1, state="CLEAN")], checks={OLD: [_check()]}, runs=runs)
    _run(gh)
    assert [c[1] for c in gh.calls if c[0] == "approve"] == ["21"]
    assert not any(c[0] == "delete" for c in gh.calls), "held runs are approved, never deleted (spec V4)"


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
    decisions = _run(gh, "dry")
    assert decisions[0][1].kind == "start"


def test_rebase_approves_before_cancelling_and_spares_the_queue(monkeypatch):
    """queue-tick runs inside its own derived-artifacts run (id 11), and a merge-queue.yml run
    (id 99) shares the old head too -- cancelling either would kill the run doing the cancelling.
    Only the unrelated perf-guard run (id 12) is fair game."""
    monkeypatch.setenv("GITHUB_RUN_ID", "11")
    runs = {OLD: [
        {"id": 99, "path": ".github/workflows/merge-queue.yml", "event": "pull_request", "status": "in_progress"},
        {"id": 11, "path": ".github/workflows/derived-artifacts.yml", "event": "pull_request", "status": "in_progress"},
        {"id": 12, "path": ".github/workflows/perf-guard.yml", "event": "pull_request", "status": "in_progress"},
    ], NEW: [_held(31)]}
    gh = FakeGh([_node(1)], runs=runs)
    _run(gh)
    rebase_idx = next(i for i, c in enumerate(gh.calls) if c[0] == "rebase")
    assert gh.calls[rebase_idx + 1] == ("approve", "31")
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


def test_rebase_approves_only_after_the_new_head_appears():
    """updatePullRequestBranch returns the PRE-rebase head and the branch moves about 1 s later
    (spike). The queue polls until a reread shows the new head, and only then approves its held
    CI and comments with the NEW sha."""
    sleeps = []
    gh = FakeGh([_node(1)], rebase_lag=2, runs={NEW: [_held(31)]})
    mq.run(gh, "on", now=lambda: NOW, sleep=sleeps.append, log=lambda m: None)
    assert gh.events.index(("read_pr", NEW)) < gh.events.index(("approve", "31"))
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
    mq.wake_after(gh, 7)
    graphql_args, wake_args = seen
    assert graphql_args[graphql_args.index("-F") + 1] == "number=5"
    assert "id=X" in graphql_args
    assert f"repos/{mq.REPO}/actions/workflows/merge-queue.yml/dispatches" in wake_args
    assert "ref=dev" in wake_args and "inputs[wait_run]=7" in wake_args


def _required_run(rid, status="in_progress", conclusion=None):
    return {"id": rid, "path": ".github/workflows/derived-artifacts.yml", "event": "workflow_dispatch",
            "status": status, "conclusion": conclusion, "html_url": f"https://run/{rid}"}


def test_line_is_read_across_pages_and_the_oldest_armed_pr_leads():
    """Qodo #1: with more open PRs than one page, the oldest armed PR may sit on a later page."""
    gh = FakeGh([
        _node(1, armed="2026-10-03T11:00:00Z", state="DIRTY"),
        _node(2, armed="2026-10-03T11:30:00Z", state="DIRTY"),
        _node(3, armed="2026-10-03T09:00:00Z", state="DIRTY"),
    ], line_page_size=2)
    decisions = _run(gh, "dry")
    assert [n for n, _ in decisions] == [3, 1, 2]
    assert gh.line_cursors == [None, "2"]


def test_a_line_longer_than_the_page_cap_fails_instead_of_deciding_on_part_of_it():
    gh = FakeGh([_node(i, state="DIRTY") for i in range(1, mq.MAX_PAGES + 2)], line_page_size=1)
    with pytest.raises(mq.GhError, match="partial line"):
        _run(gh, "dry")


def test_a_failure_on_the_second_page_of_check_runs_is_seen():
    """Qodo #6: 100 cancelled runs fill page 1; the failure on page 2 must still count."""
    cancelled = [_check("cancelled", url=f"https://run/c{i}") for i in range(mq.PER_PAGE)]
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: cancelled + [_check("failure", url="https://run/f")]})
    decision = _run(gh, "dry")[0][1]
    assert decision.kind == "retry" and decision.links == ("https://run/f",)


def test_a_live_required_run_on_the_second_page_of_workflow_runs_is_seen():
    """Qodo #6: a live required run behind 100 other runs on the head still means 'in flight'."""
    others = [{"id": 1000 + i, "path": ".github/workflows/perf-guard.yml", "event": "pull_request",
               "status": "completed", "conclusion": "success"} for i in range(mq.PER_PAGE)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: others + [_required_run(77)]})
    assert _run(gh)[0][1].kind == "wait"
    assert not any(c[0] == "dispatch" for c in gh.calls)


@pytest.mark.parametrize("status", [403, 409, 422])
def test_a_refused_rerun_evicts_and_the_line_moves_on(status):
    """GitHub will not re-run the failed run (too old, not re-runnable): evict with the reason,
    then decide for the next PR in the same run. A dispatch is no fallback: it never counts (V4).

    Args:
        status: The refusal's HTTP status.
    """
    gh = FakeGh([_node(1, state="BLOCKED"), _node(2, armed="2026-10-03T11:00:00Z", state="DIRTY")],
                checks={OLD: [_check("failure", suite=700)]}, runs={OLD: [_pr_run(70, 700)]},
                rerun_error=f"gh: This workflow run cannot be rerun (HTTP {status})")
    decisions = _run(gh)
    assert [(n, a.kind) for n, a in decisions] == [(1, "retry"), (2, "evict")]
    assert ("disarm", "PR_1") in gh.calls and ("disarm", "PR_2") in gh.calls
    comment = next(c for c in gh.calls if c[0] == "comment" and c[1] == 1)
    assert f"<!-- merge-queue:evict-rerun:{OLD} -->" in comment[2] and "refused to re-run" in comment[2]
    assert not any(c[0] == "dispatch" for c in gh.calls)


def test_start_with_nothing_to_approve_or_rerun_evicts_and_the_line_moves_on():
    """No held run and no cancelled run on an up-to-date head: nothing the queue can start counts,
    so evict with how to start CI, and move on."""
    gh = FakeGh([_node(1, state="BLOCKED"), _node(2, armed="2026-10-03T11:00:00Z", state="DIRTY")])
    decisions = _run(gh)
    assert [(n, a.kind) for n, a in decisions] == [(1, "start"), (2, "evict")]
    comment = next(c for c in gh.calls if c[0] == "comment" and c[1] == 1)
    assert f"<!-- merge-queue:evict-no-run:{OLD} -->" in comment[2] and "close and reopen" in comment[2]
    assert not any(c[0] == "dispatch" for c in gh.calls)


def test_start_reruns_a_cancelled_run_in_full():
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: [_check("cancelled", suite=800)]},
                runs={OLD: [_pr_run(80, 800, conclusion="cancelled")]})
    decisions = _run(gh)
    assert decisions[0][1].kind == "start"
    assert ("rerun", "80", "all") in gh.calls and not any(c[0] == "disarm" for c in gh.calls)


def test_start_approves_a_held_run_strictly():
    """Reached when the tick's best-effort approval pass did not get there first."""
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_held(31)]})
    pr = mq.read_prs(gh)[0]
    assert mq._start(gh, pr, lambda m: None, lambda s: None) is False
    assert ("approve", "31") in gh.calls and not any(c[0] == "disarm" for c in gh.calls)


@pytest.mark.parametrize("status", [502, 500])
@pytest.mark.parametrize("path", ["start", "retry"])
def test_a_github_side_error_fails_the_run_and_disarms_nobody(path, status):
    """Spec section 8: a 5xx is GitHub's error, not the branch's. Evicting on it, with the
    same-run hand-over, would disarm every front one outage touches. The run must fail instead,
    before any disarm, so the next event retries.

    Args:
        path: Which action meets the error: approving a held run, or re-running a failed one.
        status: The HTTP status GitHub returns.
    """
    nodes = [_node(1, state="BLOCKED")] + [_node(i, armed=f"2026-10-03T1{i}:00:00Z", state="BLOCKED") for i in range(2, 6)]
    error = f"gh: Server Error (HTTP {status})"
    if path == "start":
        gh = FakeGh(nodes, runs={OLD: [_held(31)]}, approve_error=error)
    else:
        gh = FakeGh(nodes, checks={OLD: [_check("failure", suite=700)]}, runs={OLD: [_pr_run(70, 700)]},
                    rerun_error=error)
    with pytest.raises(mq.GhError, match=rf"\(HTTP {status}\)"):
        _run(gh)
    assert not any(c[0] == "disarm" for c in gh.calls)
    assert not any(c[0] == "comment" for c in gh.calls)


def test_rest_pages_stop_at_total_count_without_a_false_cap_error():
    """Exactly MAX_PAGES full pages is a complete list when total_count says so; one more item is not."""
    full = [_check("cancelled", url=f"https://run/c{i}") for i in range(mq.MAX_PAGES * mq.PER_PAGE)]
    assert len(mq.read_checks(FakeGh([], checks={OLD: full}), OLD)) == mq.MAX_PAGES * mq.PER_PAGE
    with pytest.raises(mq.GhError, match="partial view"):
        mq.read_checks(FakeGh([], checks={OLD: full + [_check()]}), OLD)


@pytest.mark.parametrize("checks", [[], [_check("failure")]], ids=["start", "retry"])
def test_a_required_run_that_appeared_since_the_decision_stops_the_action(checks):
    """Qodo #4: merge-queue.yml and a queue-tick can decide the same action for one head. The one
    that acts second re-reads the live runs first and stands down.

    Args:
        checks: The required check's runs on the head.
    """
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: checks}, late_runs={OLD: [_required_run(88)]})
    decisions = _run(gh)
    assert decisions[0][1].kind in ("start", "retry")
    assert gh.calls == []


def test_refused_rebase_rereads_once_more_before_counting_it_as_a_failure():
    """The parked race: a racing run's rebase was accepted, but the ref moves about 1 s later. Our
    pinned-head mutation is refused inside that window; the first reread still shows the old head.
    One more look after REBASE_POLL_S sees the moved head: no comment, no eviction."""
    sleeps = []
    gh = FakeGh([_node(1)], rebase_error=True,
                reread={1: [_node(1, state="BEHIND"), _node(1, head=NEW, state="BLOCKED")]})
    mq.run(gh, "on", now=lambda: NOW, sleep=sleeps.append, log=lambda m: None)
    assert [c[0] for c in gh.calls] == ["rebase"]
    assert sleeps == [mq.REBASE_POLL_S]


def test_bot_authored_prs_are_never_queued():
    """Spec section 9: a queue dispatch runs as github-actions[bot], so it would skip the token
    cap GitHub puts on Dependabot runs and the approval gate on agent pushes. Bot PRs are told
    once and left for a maintainer; the next user PR is the front."""
    gh = FakeGh([_node(1, author="Bot"), _node(2, armed="2026-10-03T11:00:00Z", state="DIRTY")])
    decisions = _run(gh)
    assert [n for n, _ in decisions] == [2]
    bot_comment = next(c for c in gh.calls if c[0] == "comment" and c[1] == 1)
    assert f"<!-- merge-queue:bot:{OLD} -->" in bot_comment[2] and "bot or app" in bot_comment[2]
    assert not any(c[0] in ("rebase", "dispatch", "disarm") and "PR_1" in c for c in gh.calls)
    assert _run(FakeGh([_node(1, author=None)]), "dry") == [], "a deleted (ghost) author is not a user"


def test_main_drives_the_real_gh_argv_end_to_end(monkeypatch, tmp_path):
    """Qodo #7: main() through the real Gh and _run_gh down to the subprocess argv, with canned
    JSON for a BEHIND front PR: line query, approval pass, pinned-head rebase, poll reread, the
    bounded wait for the new head's held runs (none appear here), comment. No dispatch."""
    owner, name = mq.REPO.split("/")
    node = _node(7, ref="feat/queue-me")
    seen = []

    def fake_subprocess_run(argv, **kwargs):
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        seen.append(argv)
        args = argv[1:]
        if args[:2] == ["api", "graphql"]:
            query = args[3].removeprefix("query=")
            out = {
                mq.LINE_QUERY: {"data": {"repository": {"pullRequests": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": [node]}}}},
                mq.REBASE_MUTATION: {"data": {"updatePullRequestBranch": {"pullRequest": {"headRefOid": OLD}}}},
                mq.PR_QUERY: {"data": {"repository": {"pullRequest": dict(node, headRefOid=NEW,
                                                                          mergeStateStatus="BLOCKED")}}},
                mq.COMMENTS_QUERY: {"data": {"repository": {"pullRequest": {"comments": {"nodes": []}}}}},
            }[query]
        elif args[2] == "GET":
            out = {"total_count": 0, "check_runs": [], "workflow_runs": []}
        elif args[3].endswith("/dispatches"):
            out = None  # 204 No Content
        else:
            out = {"id": 1}
        return subprocess.CompletedProcess(argv, 0, stdout="" if out is None else json.dumps(out), stderr="")

    monkeypatch.setattr(mq.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(mq, "REBASE_POLL_S", 0)
    monkeypatch.setattr(mq, "HELD_RUN_POLL_S", 0)
    monkeypatch.setenv("MERGE_QUEUE", "on")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert mq.main() == 0

    repo = f"repos/{owner}/{name}"
    runs_old = f"{repo}/actions/runs?head_sha={OLD}&per_page=100&page=1"
    held_old = f"{repo}/actions/runs?head_sha={OLD}&status=action_required&per_page=100&page=1"
    held_new = f"{repo}/actions/runs?head_sha={NEW}&status=action_required&per_page=100&page=1"
    who = ["-f", f"owner={owner}", "-f", f"name={name}"]
    assert seen[:-1] == [
        ["gh", "api", "graphql", "-f", f"query={mq.LINE_QUERY}", *who],
        ["gh", "api", "-X", "GET", held_old],
        ["gh", "api", "-X", "GET", f"{repo}/commits/{OLD}/check-runs"
         "?check_name=Derived%20artifacts%20reproduce%20from%20their%20sources&filter=all&per_page=100&page=1"],
        ["gh", "api", "-X", "GET", runs_old],
        ["gh", "api", "graphql", "-f", f"query={mq.REBASE_MUTATION}", "-f", "id=PR_7", "-f", f"oid={OLD}"],
        ["gh", "api", "graphql", "-f", f"query={mq.PR_QUERY}", *who, "-F", "number=7"],
        ["gh", "api", "-X", "GET", runs_old],
        *[["gh", "api", "-X", "GET", held_new]] * mq.HELD_RUN_POLLS,
        ["gh", "api", "graphql", "-f", f"query={mq.COMMENTS_QUERY}", *who, "-F", "number=7"],
    ]
    assert not any("/dispatches" in " ".join(argv) for argv in seen)
    assert seen[-1][:6] == ["gh", "api", "-X", "POST", f"{repo}/issues/7/comments", "-f"]
    assert seen[-1][6].startswith(f"body=<!-- merge-queue:rebased:{NEW} -->\nMerge queue: this PR is next.")
    assert "| #7 | rebase | behind dev |" in summary.read_text(encoding="utf-8")


def test_the_failed_runs_own_tick_wakes_a_queue_run(monkeypatch):
    """The retry is usually decided by the failed run's own queue-tick while that run is still in
    progress, and GitHub only re-runs a completed run. The tick wakes a queue run that waits for it."""
    monkeypatch.setenv("GITHUB_RUN_ID", "70")
    own = _pr_run(70, 700, conclusion=None, status="in_progress")
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: [_check("failure", suite=700)]}, runs={OLD: [own]})
    assert _run(gh)[0][1].kind == "retry"
    assert ("dispatch", "merge-queue.yml", {"ref": "dev", "inputs[wait_run]": "70"}) in gh.calls
    assert not any(c[0] in ("rerun", "comment") for c in gh.calls)


def test_a_run_that_is_live_again_is_left_to_finish():
    """A racing queue run re-ran it first; a second re-run would be refused. Stand down."""
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_pr_run(70, 700, conclusion=None, status="queued")]})
    pr = mq.read_prs(gh)[0]
    action = mq.Action("retry", "required check failed once; retrying", ("https://run/70",), suite_id=700)
    assert mq._retry(gh, pr, action, lambda m: None, lambda s: None) is False
    assert not any(c[0] in ("rerun", "dispatch", "comment", "disarm") for c in gh.calls)


def test_a_broken_runs_stand_in_is_rerun_in_full():
    """A run that failed without reporting the check (e.g. a startup failure) has no suite on the
    retry; the queue re-runs that run in full."""
    broken = dict(_pr_run(31, 531, conclusion="startup_failure"))
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [broken]})
    assert _run(gh)[0][1].kind == "retry"
    assert ("rerun", "31", "all") in gh.calls


def test_a_rerun_held_for_approval_is_approved():
    """The re-run's actor is the queue's token, so GitHub may hold it again; the queue approves it."""
    run = _pr_run(70, 700)
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: [_check("failure", suite=700)]}, runs={OLD: [run]})
    original = gh.rest

    def rest(method, path, fields=None):
        out = original(method, path, fields)
        if path.endswith("/rerun-failed-jobs"):
            gh._set_run(70, status="completed", conclusion="action_required",
                        event="pull_request", triggering_actor={"login": "github-actions[bot]"})
        return out

    gh.rest = rest
    _run(gh)
    assert gh.calls.index(("rerun", "70", "failed")) < gh.calls.index(("approve", "70"))


def test_a_woken_run_waits_for_the_failed_run_then_reruns_it():
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: [_check("failure", suite=700)]},
                runs={OLD: [_pr_run(70, 700)]}, run_status={70: ["in_progress", "in_progress", "completed", "queued"]})
    slept = []
    mq.run(gh, "on", now=lambda: NOW, sleep=slept.append, log=lambda m: None, wait_run=70)
    assert gh.run_reads[70] >= 3 and slept[:2] == [mq.WAIT_RUN_DELAY_S] * 2
    assert ("rerun", "70", "failed") in gh.calls


def test_the_wait_is_bounded():
    gh = FakeGh([_node(1, state="CLEAN")], checks={OLD: [_check()]}, run_status={70: ["in_progress"]})
    lines = []
    mq.run(gh, "on", now=lambda: NOW, sleep=lambda s: None, log=lines.append, wait_run=70)
    assert gh.run_reads[70] == mq.WAIT_RUN_TRIES
    assert any("still live after" in line for line in lines)


def test_dry_mode_never_approves_reruns_or_wakes(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "70")
    runs = {OLD: [_pr_run(70, 700, conclusion=None, status="in_progress"), _held(31)]}
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: [_check("failure", suite=700)]}, runs=runs)
    assert _run(gh, "dry")[0][1].kind == "retry"
    assert gh.calls == []


def test_a_failed_rerun_still_counts_as_the_second_failure():
    """After a re-run the old attempt's failed check stays listed (filter=all, verified live on
    #3019's head 6a11342309), so a re-run that fails again evicts instead of re-running forever."""
    checks = [_check("failure", completed="2026-10-03T11:00:00Z", url="https://run/a", suite=700),
              _check("failure", completed="2026-10-03T11:40:00Z", url="https://run/b", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: checks}, runs={OLD: [_pr_run(70, 700)]})
    assert _run(gh)[0][1].kind == "evict"
    assert not any(c[0] == "rerun" for c in gh.calls)
