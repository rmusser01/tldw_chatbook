#!/usr/bin/env python3
"""One-at-a-time merge queue for `dev`.

Spec: Docs/superpowers/specs/2026-10-03-merge-queue-design.md

Runs inside GitHub Actions with the built-in GITHUB_TOKEN (merge-queue.yml and the
queue-tick job in derived-artifacts.yml). Each run reads the line of armed PRs, decides
one action for the PR at the front and, in `on` mode, performs it. PRs behind the front
are never touched.

Never enables auto-merge, never merges, never pushes: a merge made with GITHUB_TOKEN
pushes to dev without triggering any workflow, which would silently stop this queue and
dev's post-merge checks (spec section 7).

Mode comes from the MERGE_QUEUE repository variable: unset/off = do nothing,
dry = decide and log only, on = act.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol
from urllib.parse import quote

REPO = os.environ.get("GITHUB_REPOSITORY", "rmusser01/tldw_chatbook")
BASE = "dev"
REQUIRED_CHECK = "Derived artifacts reproduce from their sources"
REQUIRED_WORKFLOW = "derived-artifacts.yml"
QUEUE_WORKFLOW = "merge-queue.yml"
YOUNG_HEAD = timedelta(minutes=3)
STUCK_GREEN = timedelta(minutes=15)
MAX_FRONTS_PER_RUN = 10
UNKNOWN_REREADS = 3
UNKNOWN_SLEEP_S = 5
PASSING = frozenset({"success", "neutral", "skipped"})
LIVE_RUN_STATUSES = frozenset({"queued", "in_progress", "waiting", "requested", "pending"})


@dataclass(frozen=True)
class CheckRun:
    """One run of the required check on a commit."""

    status: str
    conclusion: str | None
    completed_at: datetime | None
    url: str


@dataclass(frozen=True)
class PrState:
    """What the queue needs to know about one open PR into dev."""

    number: int
    node_id: str
    head_sha: str
    head_ref: str
    same_repo: bool
    armed_at: datetime | None
    is_draft: bool
    merge_state: str
    head_committed_at: datetime
    checks: tuple[CheckRun, ...] = ()


@dataclass(frozen=True)
class Action:
    """The single decision for the front PR: wait, rebase, dispatch, retry or evict."""

    kind: str
    reason: str
    links: tuple[str, ...] = ()


def line_of(prs: list[PrState]) -> list[PrState]:
    """Return the queue: armed, non-draft, same-repo PRs, oldest arming first.

    Args:
        prs: Every open PR into dev.

    Returns:
        The PRs in queue order.
    """
    eligible = [p for p in prs if p.armed_at is not None and not p.is_draft and p.same_repo]
    return sorted(eligible, key=lambda p: (p.armed_at, p.number))


def decide_front(pr: PrState, now: datetime) -> Action:
    """Decide the one action for the PR at the front of the line (spec section 6).

    Args:
        pr: The front PR, with its required-check runs on the current head.
        now: The current time (UTC).

    Returns:
        The action to take.
    """
    state = pr.merge_state
    if state == "UNKNOWN":
        return Action("wait", "merge state unknown")
    if state == "BEHIND":
        return Action("rebase", "behind dev")
    if state == "DIRTY":
        return Action("evict", "conflicts with dev")
    if any(c.status != "completed" for c in pr.checks):
        return Action("wait", "required check running")
    finished = sorted(
        (c for c in pr.checks if c.conclusion != "cancelled"),
        key=lambda c: c.completed_at or now,
    )
    if not finished:
        if now - pr.head_committed_at <= YOUNG_HEAD:
            return Action("wait", "head is under 3 minutes old; its own run may not be visible yet")
        return Action("dispatch", "no required-check run on the up-to-date head")
    latest = finished[-1]
    failed = [c for c in finished if c.conclusion not in PASSING]
    if latest.conclusion not in PASSING:
        if len(failed) >= 2:
            return Action("evict", "required check failed twice", tuple(c.url for c in failed[-2:]))
        return Action("retry", "required check failed once; retrying", (latest.url,))
    if state in ("CLEAN", "UNSTABLE", "HAS_HOOKS"):
        if latest.completed_at is not None and now - latest.completed_at > STUCK_GREEN:
            return Action("evict", "green for over 15 minutes but auto-merge did not fire; re-arm to retry")
        return Action("wait", "green; auto-merge should fire")
    if state == "BLOCKED":
        return Action("evict", "green but blocked by unresolved conversations or reviews")
    return Action("wait", f"merge state {state}")


class GhError(RuntimeError):
    """A gh CLI call failed."""


class GhApi(Protocol):
    """What the queue needs from GitHub; `Gh` in production, a fake in tests."""

    def graphql(self, query: str, **variables: object) -> dict: ...

    def rest(self, method: str, path: str, fields: dict[str, str] | None = None) -> object: ...


def _run_gh(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise GhError(f"gh {' '.join(args[:3])} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


class Gh:
    """Thin wrapper over the gh CLI (preinstalled on hosted runners; auth via GH_TOKEN)."""

    def __init__(self, runner: Callable[[list[str]], str] | None = None) -> None:
        self._run = runner or _run_gh

    def graphql(self, query: str, **variables: object) -> dict:
        args = ["api", "graphql", "-f", f"query={query}"]
        for key, value in variables.items():
            args += ["-F" if isinstance(value, int) else "-f", f"{key}={value}"]
        return json.loads(self._run(args))

    def rest(self, method: str, path: str, fields: dict[str, str] | None = None) -> object:
        args = ["api", "-X", method, path]
        for key, value in (fields or {}).items():
            args += ["-f", f"{key}={value}"]
        out = self._run(args)
        return json.loads(out) if out.strip() else None


PR_FIELDS = """
  number id isDraft headRefOid headRefName mergeStateStatus
  headRepository { nameWithOwner }
  autoMergeRequest { enabledAt }
  commits(last: 1) { nodes { commit { committedDate } } }
"""
LINE_QUERY = (
    "query($owner: String!, $name: String!) { repository(owner: $owner, name: $name) {"
    f' pullRequests(states: OPEN, baseRefName: "{BASE}", first: 100) {{ nodes {{ {PR_FIELDS} }} }} }} }}'
)
PR_QUERY = (
    "query($owner: String!, $name: String!, $number: Int!) { repository(owner: $owner, name: $name) {"
    f" pullRequest(number: $number) {{ {PR_FIELDS} }} }} }}"
)
COMMENTS_QUERY = (
    "query($owner: String!, $name: String!, $number: Int!) { repository(owner: $owner, name: $name) {"
    " pullRequest(number: $number) { comments(last: 100) { nodes { body } } } } }"
)
REBASE_MUTATION = (
    "mutation($id: ID!, $oid: GitObjectID!) { updatePullRequestBranch(input: "
    "{pullRequestId: $id, expectedHeadOid: $oid, updateMethod: REBASE}) { pullRequest { headRefOid } } }"
)
DISARM_MUTATION = (
    "mutation($id: ID!) { disablePullRequestAutoMerge(input: {pullRequestId: $id}) { clientMutationId } }"
)


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _owner_name() -> tuple[str, str]:
    owner, name = REPO.split("/")
    return owner, name


def _parse_pr(node: dict) -> PrState:
    commits = node["commits"]["nodes"]
    auto = node.get("autoMergeRequest")
    head_repo = (node.get("headRepository") or {}).get("nameWithOwner")
    return PrState(
        number=node["number"],
        node_id=node["id"],
        head_sha=node["headRefOid"],
        head_ref=node["headRefName"],
        same_repo=head_repo == REPO,
        armed_at=_ts(auto["enabledAt"]) if auto else None,
        is_draft=node["isDraft"],
        merge_state=node["mergeStateStatus"],
        head_committed_at=_ts(commits[0]["commit"]["committedDate"]) if commits else datetime.now(timezone.utc),
    )


def read_prs(gh: GhApi) -> list[PrState]:
    owner, name = _owner_name()
    data = gh.graphql(LINE_QUERY, owner=owner, name=name)
    return [_parse_pr(n) for n in data["data"]["repository"]["pullRequests"]["nodes"]]


def read_pr(gh: GhApi, number: int) -> PrState:
    owner, name = _owner_name()
    data = gh.graphql(PR_QUERY, owner=owner, name=name, number=number)
    return _parse_pr(data["data"]["repository"]["pullRequest"])


def read_checks(gh: GhApi, sha: str) -> tuple[CheckRun, ...]:
    path = f"repos/{REPO}/commits/{sha}/check-runs?check_name={quote(REQUIRED_CHECK)}&filter=all&per_page=100"
    data = gh.rest("GET", path) or {}
    return tuple(
        CheckRun(c["status"], c.get("conclusion"), _ts(c.get("completed_at")), c["html_url"])
        for c in data.get("check_runs", [])
    )


def runs_on(gh: GhApi, sha: str, status: str | None = None) -> list[dict]:
    path = f"repos/{REPO}/actions/runs?head_sha={sha}&per_page=100"
    if status:
        path += f"&status={status}"
    return (gh.rest("GET", path) or {}).get("workflow_runs", [])


def _best_effort(log: Callable[[str], None], what: str, fn: Callable[[], object]) -> None:
    try:
        fn()
    except GhError as exc:
        log(f"  best-effort {what} failed: {exc}")


def comment_once(gh: GhApi, number: int, kind: str, sha: str, body: str) -> None:
    """Post a comment unless one of this kind already exists for this head."""
    marker = f"<!-- merge-queue:{kind}:{sha} -->"
    owner, name = _owner_name()
    data = gh.graphql(COMMENTS_QUERY, owner=owner, name=name, number=number)
    bodies = [n.get("body") or "" for n in data["data"]["repository"]["pullRequest"]["comments"]["nodes"]]
    if any(marker in b for b in bodies):
        return
    gh.rest("POST", f"repos/{REPO}/issues/{number}/comments", {"body": f"{marker}\n{body}"})


def dispatch(gh: GhApi, workflow: str, ref: str, pr_number: int | None = None) -> None:
    fields = {"ref": ref}
    if pr_number is not None:
        fields["inputs[pr]"] = str(pr_number)
    gh.rest("POST", f"repos/{REPO}/actions/workflows/{workflow}/dispatches", fields)


def _workflows_to_redispatch(runs: list[dict]) -> list[str]:
    """Workflows (other than the required one and the queue) that ran on the old head."""
    names = set()
    for run in runs:
        name = str(run.get("path", "")).split("/")[-1].split("@")[0]
        if not name or name in (REQUIRED_WORKFLOW, QUEUE_WORKFLOW):
            continue
        if run.get("event") not in ("pull_request", "workflow_dispatch"):
            continue
        if run.get("conclusion") in ("skipped", "action_required"):
            continue
        names.add(name)
    return sorted(names)


def _evict(gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None]) -> None:
    gh.graphql(DISARM_MUTATION, id=pr.node_id)
    links = "".join(f"\n- {u}" for u in action.links)
    comment_once(
        gh, pr.number, "evict", pr.head_sha,
        f"Merge queue: removed from the line ({action.reason}). Auto-merge is now off. Fix the cause, then "
        f"re-arm with `gh pr merge {pr.number} --auto --merge` to rejoin at the back.{links}",
    )
    log(f"  evicted #{pr.number}: {action.reason}")


def _rebase(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> None:
    try:
        result = gh.graphql(REBASE_MUTATION, id=pr.node_id, oid=pr.head_sha)
    except GhError as exc:
        fresh = read_pr(gh, pr.number)
        if fresh.head_sha != pr.head_sha:
            log(f"  rebase skipped: head moved to {fresh.head_sha[:10]}")
        elif fresh.merge_state == "DIRTY":
            _evict(gh, fresh, Action("evict", "conflicts with dev (rebase refused)"), log)
        else:
            log(f"  rebase failed, left alone: {exc}")
        return
    new_head = result["data"]["updatePullRequestBranch"]["pullRequest"]["headRefOid"]
    old_runs = runs_on(gh, pr.head_sha)
    for run in old_runs:
        if run.get("status") in LIVE_RUN_STATUSES:
            _best_effort(log, f"cancel run {run['id']}",
                         lambda rid=run["id"]: gh.rest("POST", f"repos/{REPO}/actions/runs/{rid}/cancel"))
    dispatch(gh, REQUIRED_WORKFLOW, pr.head_ref, pr.number)
    refused = []
    for workflow in _workflows_to_redispatch(old_runs):
        try:
            dispatch(gh, workflow, pr.head_ref)
        except GhError:
            refused.append(workflow)
    note = f"\n\nNot re-run (dispatch refused): {', '.join(refused)}" if refused else ""
    comment_once(
        gh, pr.number, "rebased", new_head,
        f"Merge queue: this PR is next. Rebased onto `{BASE}` (head `{new_head[:10]}`) and started CI.{note}",
    )
    log(f"  rebased #{pr.number} {pr.head_sha[:10]} -> {new_head[:10]}")


def apply(gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None]) -> None:
    """Perform one decided action (spec section 7)."""
    if action.kind == "rebase":
        _rebase(gh, pr, log)
    elif action.kind in ("dispatch", "retry"):
        dispatch(gh, REQUIRED_WORKFLOW, pr.head_ref, pr.number)
        if action.kind == "retry":
            comment_once(
                gh, pr.number, "retry", pr.head_sha,
                f"Merge queue: the required check failed once on `{pr.head_sha[:10]}`; retrying with a fresh run. "
                f"Failed run: {action.links[0]}",
            )
    elif action.kind == "evict":
        _evict(gh, pr, action, log)


def cleanup_approval_runs(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> None:
    """Delete the empty approval-pending runs our own token rebase created (spec F4, V1)."""
    for run in runs_on(gh, pr.head_sha, status="action_required"):
        if (run.get("triggering_actor") or {}).get("login") != "github-actions[bot]":
            continue
        _best_effort(log, f"delete approval-pending run {run['id']}",
                     lambda rid=run["id"]: gh.rest("DELETE", f"repos/{REPO}/actions/runs/{rid}"))


def comment_forks(gh: GhApi, prs: list[PrState], mode: str, log: Callable[[str], None]) -> None:
    for pr in prs:
        if pr.armed_at is None or pr.same_repo:
            continue
        log(f"#{pr.number}: fork PR armed; not queued")
        if mode == "on":
            comment_once(
                gh, pr.number, "fork", pr.head_sha,
                "Merge queue: fork PRs are not queued, because GitHub cannot dispatch workflows on a fork's "
                "branch. A maintainer merges this one by hand.",
            )


def settle_unknown(gh: GhApi, pr: PrState, sleep: Callable[[float], None]) -> PrState:
    for _ in range(UNKNOWN_REREADS):
        if pr.merge_state != "UNKNOWN":
            return pr
        sleep(UNKNOWN_SLEEP_S)
        pr = read_pr(gh, pr.number)
    return pr


def run(
    gh: GhApi,
    mode: str | None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> list[tuple[int, Action]]:
    """One queue pass. Returns the (PR, action) decisions taken or, in dry mode, proposed."""
    mode = (mode or "").strip().lower()
    if mode not in ("dry", "on"):
        log("merge queue is off (MERGE_QUEUE is not 'dry' or 'on')")
        return []
    prs = read_prs(gh)
    comment_forks(gh, prs, mode, log)
    decisions: list[tuple[int, Action]] = []
    for pr in line_of(prs)[:MAX_FRONTS_PER_RUN]:
        pr = settle_unknown(gh, pr, sleep)
        if pr.armed_at is None:
            continue
        pr = replace(pr, checks=read_checks(gh, pr.head_sha))
        action = decide_front(pr, now())
        decisions.append((pr.number, action))
        log(f"#{pr.number}: {action.kind} - {action.reason}")
        if mode == "on":
            cleanup_approval_runs(gh, pr, log)
            apply(gh, pr, action, log)
        if action.kind != "evict":
            break
    return decisions


def main() -> int:
    decisions = run(Gh(), os.environ.get("MERGE_QUEUE"))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Merge queue\n\n| PR | Action | Reason |\n|---|---|---|\n")
            for number, action in decisions:
                fh.write(f"| #{number} | {action.kind} | {action.reason} |\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
