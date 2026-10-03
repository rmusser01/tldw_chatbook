"""The merge queue's action layer and run loop, against a fake gh (spec sections 6-8)."""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "merge_queue.py"
_spec = importlib.util.spec_from_file_location("merge_queue", SCRIPT)
mq = importlib.util.module_from_spec(_spec)
sys.modules["merge_queue"] = mq
_spec.loader.exec_module(mq)

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
OLD = "a" * 40
NEW = "b" * 40


def _node(number, *, head=OLD, armed="2026-10-03T10:00:00Z", state="BEHIND", repo=None, draft=False,
          committed="2026-10-03T09:00:00Z", ref=None):
    return {
        "number": number, "id": f"PR_{number}", "isDraft": draft, "headRefOid": head,
        "headRefName": ref or f"feat/{number}", "mergeStateStatus": state,
        "headRepository": {"nameWithOwner": repo or mq.REPO},
        "autoMergeRequest": {"enabledAt": armed} if armed else None,
        "commits": {"nodes": [{"commit": {"committedDate": committed}}]},
    }


def _check(conclusion="success", completed="2026-10-03T11:58:00Z", url="https://run/1"):
    return {"status": "completed", "conclusion": conclusion, "completed_at": completed, "html_url": url}


class FakeGh:
    """Records every mutating call; serves scripted reads."""

    def __init__(self, nodes, *, checks=None, runs=None, comments=None, rebase_error=False,
                 reread=None, dispatch_refused=()):
        self.nodes = {n["number"]: n for n in nodes}
        self.checks = checks or {}
        self.runs = runs or {}
        self.comments = comments or {}
        self.rebase_error = rebase_error
        self.reread = reread or {}
        self.dispatch_refused = set(dispatch_refused)
        self.calls = []
        self.reads = 0

    def graphql(self, query, **v):
        if "updatePullRequestBranch" in query:
            self.calls.append(("rebase", v["id"], v["oid"]))
            if self.rebase_error:
                raise mq.GhError("rebase refused")
            return {"data": {"updatePullRequestBranch": {"pullRequest": {"headRefOid": NEW}}}}
        if "disablePullRequestAutoMerge" in query:
            self.calls.append(("disarm", v["id"]))
            return {"data": {}}
        self.reads += 1
        if "comments(last" in query:
            bodies = self.comments.get(v["number"], [])
            return {"data": {"repository": {"pullRequest": {"comments": {"nodes": [{"body": b} for b in bodies]}}}}}
        if "pullRequest(number" in query:
            node = self.reread.get(v["number"], self.nodes[v["number"]])
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
            self.calls.append(("dispatch", workflow, dict(fields or {})))
            if workflow in self.dispatch_refused:
                raise mq.GhError("HTTP 422: Workflow does not have 'workflow_dispatch' trigger")
            return None
        if method == "POST" and path.endswith("/cancel"):
            self.calls.append(("cancel", path.split("/runs/")[1].split("/")[0]))
            return None
        if method == "DELETE" and "/actions/runs/" in path:
            self.calls.append(("delete", path.rsplit("/", 1)[1]))
            return None
        if method == "POST" and path.endswith("/comments"):
            self.calls.append(("comment", int(path.split("/issues/")[1].split("/")[0]), fields["body"]))
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
    gh = FakeGh([_node(1, state="DIRTY")], comments={1: [f"<!-- merge-queue:evict:{OLD} -->\nold"]})
    _run(gh)
    assert ("disarm", "PR_1") in gh.calls
    assert not any(c[0] == "comment" for c in gh.calls)


def test_first_failure_dispatches_a_retry():
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: [_check("failure")]})
    _run(gh)
    assert ("dispatch", "derived-artifacts.yml", {"ref": "feat/1", "inputs[pr]": "1"}) in gh.calls
    assert any(c[0] == "comment" and f"merge-queue:retry:{OLD}" in c[2] for c in gh.calls)


def test_queue_tick_failure_is_not_a_ci_failure():
    runs = {OLD: [{"id": 9, "path": ".github/workflows/derived-artifacts.yml", "event": "pull_request",
                   "status": "completed", "conclusion": "failure"}]}
    gh = FakeGh([_node(1, state="CLEAN")], checks={OLD: [_check("success")]}, runs=runs)
    assert _run(gh)[0][1].kind == "wait"
    assert gh.calls == []


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
