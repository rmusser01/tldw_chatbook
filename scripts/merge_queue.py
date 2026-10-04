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
UNKNOWN_REREADS = 12
UNKNOWN_SLEEP_S = 10
# updatePullRequestBranch returns the PRE-rebase head; the branch moves about 1 s later (spike).
REBASE_POLLS = 10
REBASE_POLL_S = 3
# Every list read follows pages to the end. A list longer than this fails the run rather than
# deciding on a partial view (an armed PR or a live run on a later page would be invisible).
PER_PAGE = 100
MAX_PAGES = 10
PASSING = frozenset({"success", "neutral", "skipped"})
LIVE_RUN_STATUSES = frozenset({"queued", "in_progress", "waiting", "requested", "pending"})
BROKEN_RUN_CONCLUSIONS = frozenset({"failure", "startup_failure", "timed_out"})
# gh reports API errors as `gh: <message> (HTTP NNN)` (verified live). These two mean the branch
# itself refuses the required-check dispatch; every other error is GitHub's and is retried.
BRANCH_REFUSALS = ("(HTTP 422)", "(HTTP 404)")


@dataclass(frozen=True)
class CheckRun:
    """One run of the required check on a commit.

    Attributes:
        status: The run's status (`queued`, `in_progress`, `completed`, ...).
        conclusion: The conclusion once completed (`success`, `failure`, ...), else None.
        completed_at: When it completed, else None.
        url: The run's web URL, quoted in comments.
        suite_id: The check suite it belongs to, which links it to its workflow run.
    """

    status: str
    conclusion: str | None
    completed_at: datetime | None
    url: str
    suite_id: int | None = None


@dataclass(frozen=True)
class PrState:
    """What the queue needs to know about one open PR into dev.

    Attributes:
        number: The PR number.
        node_id: The PR's GraphQL node id, used by mutations.
        head_sha: The current head commit.
        head_ref: The head branch name, the ref workflows are dispatched on.
        same_repo: Whether the head branch lives in this repository (not a fork).
        armed_at: When auto-merge was enabled, or None if it is not armed.
        is_draft: Whether the PR is a draft.
        merge_state: GitHub's `mergeStateStatus` (`BEHIND`, `CLEAN`, `DIRTY`, ...).
        head_committed_at: The head commit's date.
        checks: Required-check runs on the head, plus live or failed workflow-run stand-ins.
        unresolved_threads: How many review threads are unresolved.
        human_author: Whether a user, not a bot or app, opened the PR (spec section 9).
    """

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
    unresolved_threads: int = 0
    human_author: bool = True


@dataclass(frozen=True)
class Action:
    """The single decision for the front PR: wait, rebase, dispatch, retry or evict.

    `slug` names an eviction's cause in its comment marker (`evict-<slug>`), so a re-armed PR
    evicted again on the same head for a different reason is still told why.

    Attributes:
        kind: `wait`, `rebase`, `dispatch`, `retry` or `evict`.
        reason: A human-readable cause, logged and quoted in comments.
        links: Run URLs quoted in the comment (the failed runs).
        slug: An eviction's cause, used in its comment marker.
    """

    kind: str
    reason: str
    links: tuple[str, ...] = ()
    slug: str = ""


def line_of(prs: list[PrState]) -> list[PrState]:
    """Return the queue: armed, non-draft, same-repo, user-authored PRs, oldest arming first.

    Args:
        prs: Every open PR into dev.

    Returns:
        The PRs in queue order.
    """
    eligible = [p for p in prs if p.armed_at is not None and not p.is_draft and p.same_repo and p.human_author]
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
        return Action("evict", "conflicts with dev", slug="conflict")
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
            return Action("evict", "required check failed twice", tuple(c.url for c in failed[-2:]), "failed-twice")
        return Action("retry", "required check failed once; retrying", (latest.url,))
    if state == "BLOCKED" and pr.unresolved_threads > 0:
        return Action("evict", "green but blocked by unresolved conversations", slug="blocked")
    # BLOCKED with nothing unresolved is usually mergeStateStatus lagging the green check.
    if state in ("CLEAN", "UNSTABLE", "HAS_HOOKS", "BLOCKED"):
        if latest.completed_at is not None and now - latest.completed_at > STUCK_GREEN:
            return Action(
                "evict", "green for over 15 minutes but auto-merge did not fire; re-arm to retry", slug="stuck",
            )
        return Action("wait", "green; auto-merge should fire")
    return Action("wait", f"merge state {state}")


class GhError(RuntimeError):
    """A gh CLI call failed, or a list read was longer than the queue will page through."""


class GhApi(Protocol):
    """What the queue needs from GitHub; `Gh` in production, a fake in tests."""

    def graphql(self, query: str, **variables: object) -> dict:
        """Run a GraphQL query or mutation.

        Args:
            query: The GraphQL document.
            **variables: Its variables; ints are sent typed, everything else as strings.

        Returns:
            The decoded response (`{"data": ...}`).

        Raises:
            GhError: The call failed.
        """
        ...

    def rest(self, method: str, path: str, fields: dict[str, str] | None = None) -> object:
        """Call a REST endpoint.

        Args:
            method: The HTTP method.
            path: The path relative to the API root, query string included.
            fields: String fields sent as the request body.

        Returns:
            The decoded JSON response, or None for an empty body.

        Raises:
            GhError: The call failed.
        """
        ...


def _run_gh(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise GhError(f"gh {' '.join(args[:3])} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


class Gh:
    """Thin wrapper over the gh CLI (preinstalled on hosted runners; auth via GH_TOKEN).

    Only ever runs `gh api`; see `GhApi` for the method contracts.
    """

    def __init__(self, runner: Callable[[list[str]], str] | None = None) -> None:
        """Create the wrapper.

        Args:
            runner: Runs `gh` with the given arguments and returns its stdout; defaults to a
                subprocess call that raises GhError on a non-zero exit.
        """
        self._run = runner or _run_gh

    def graphql(self, query: str, **variables: object) -> dict:
        """See `GhApi.graphql`."""
        args = ["api", "graphql", "-f", f"query={query}"]
        for key, value in variables.items():
            args += ["-F" if isinstance(value, int) else "-f", f"{key}={value}"]
        return json.loads(self._run(args))

    def rest(self, method: str, path: str, fields: dict[str, str] | None = None) -> object:
        """See `GhApi.rest`."""
        args = ["api", "-X", method, path]
        for key, value in (fields or {}).items():
            args += ["-f", f"{key}={value}"]
        out = self._run(args)
        return json.loads(out) if out.strip() else None


PR_FIELDS = """
  number id isDraft headRefOid headRefName mergeStateStatus
  author { __typename }
  headRepository { nameWithOwner }
  autoMergeRequest { enabledAt }
  commits(last: 1) { nodes { commit { committedDate } } }
  reviewThreads(first: 100) { nodes { isResolved } }
"""
LINE_QUERY = (
    "query($owner: String!, $name: String!, $after: String) { repository(owner: $owner, name: $name) {"
    f' pullRequests(states: OPEN, baseRefName: "{BASE}", first: {PER_PAGE}, after: $after) {{'
    f" pageInfo {{ hasNextPage endCursor }} nodes {{ {PR_FIELDS} }} }} }} }}"
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
        unresolved_threads=sum(1 for t in node["reviewThreads"]["nodes"] if not t["isResolved"]),
        human_author=(node.get("author") or {}).get("__typename") == "User",
    )


def read_prs(gh: GhApi) -> list[PrState]:
    """Read every open PR into dev, following the GraphQL cursor to the last page.

    Args:
        gh: The GitHub client.

    Returns:
        Every open PR into dev, in API order.

    Raises:
        GhError: A call failed, or there are more than MAX_PAGES pages of open PRs.
    """
    owner, name = _owner_name()
    prs: list[PrState] = []
    cursor = None
    for _ in range(MAX_PAGES):
        variables = {"owner": owner, "name": name}
        if cursor:
            variables["after"] = cursor
        page = gh.graphql(LINE_QUERY, **variables)["data"]["repository"]["pullRequests"]
        prs += [_parse_pr(n) for n in page["nodes"]]
        if not page["pageInfo"]["hasNextPage"]:
            return prs
        cursor = page["pageInfo"]["endCursor"]
    raise GhError(f"more than {MAX_PAGES * PER_PAGE} open PRs into {BASE}; refusing to decide on a partial line")


def read_pr(gh: GhApi, number: int) -> PrState:
    """Read one PR's current state.

    Args:
        gh: The GitHub client.
        number: The PR number.

    Returns:
        The PR's state, without checks.

    Raises:
        GhError: The call failed.
    """
    owner, name = _owner_name()
    data = gh.graphql(PR_QUERY, owner=owner, name=name, number=number)
    return _parse_pr(data["data"]["repository"]["pullRequest"])


def _rest_pages(gh: GhApi, path: str, key: str) -> list[dict]:
    """Every item of a paged REST list: pages are followed until `total_count` (when the response
    has one) is reached or a short page comes back."""
    items: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        data = gh.rest("GET", f"{path}&per_page={PER_PAGE}&page={page}") or {}
        batch = data.get(key, [])
        items += batch
        total = data.get("total_count")
        if len(batch) < PER_PAGE or (total is not None and len(items) >= total):
            return items
    raise GhError(f"more than {MAX_PAGES * PER_PAGE} {key} for {path}; refusing to decide on a partial view")


def read_checks(gh: GhApi, sha: str) -> tuple[CheckRun, ...]:
    """Read every run of the required check on a commit.

    Args:
        gh: The GitHub client.
        sha: The commit.

    Returns:
        The required check's runs, all pages.

    Raises:
        GhError: A call failed, or there are more than MAX_PAGES pages.
    """
    path = f"repos/{REPO}/commits/{sha}/check-runs?check_name={quote(REQUIRED_CHECK)}&filter=all"
    return tuple(
        CheckRun(c["status"], c.get("conclusion"), _ts(c.get("completed_at")), c["html_url"],
                 (c.get("check_suite") or {}).get("id"))
        for c in _rest_pages(gh, path, "check_runs")
    )


def runs_on(gh: GhApi, sha: str, status: str | None = None) -> list[dict]:
    """Read every workflow run on a commit.

    Args:
        gh: The GitHub client.
        sha: The head commit.
        status: Only runs with this status (e.g. `action_required`), if given.

    Returns:
        The workflow runs, all pages, as the API returns them.

    Raises:
        GhError: A call failed, or there are more than MAX_PAGES pages.
    """
    path = f"repos/{REPO}/actions/runs?head_sha={sha}"
    if status:
        path += f"&status={status}"
    return _rest_pages(gh, path, "workflow_runs")


def _workflow_name(run: dict) -> str:
    """The workflow file's basename, stripped of any `@ref` suffix the API may add."""
    return str(run.get("path", "")).split("/")[-1].split("@")[0]


def required_run_stand_ins(gh: GhApi, sha: str, checks: tuple[CheckRun, ...]) -> tuple[CheckRun, ...]:
    """Required-workflow runs on this head that have not reported the required check, as stand-ins.

    - A LIVE run stands in as an in-flight check: the required check is a needs-gated aggregate
      with no check run until the lanes finish (verified live: run 37156228421 showed
      `total_count: 0` while its lanes ran), so an in-flight front PR would otherwise look like
      it has no run and get re-dispatched on every tick.
    - A COMPLETED run that failed (`BROKEN_RUN_CONCLUSIONS`) with no required check run in its
      check suite stands in as one failure: a startup failure, e.g. a broken
      derived-artifacts.yml on the branch, would otherwise read as "no run" and be re-dispatched
      forever. A failed run whose suite DID report the required check is not a CI failure (its
      queue-tick may have failed); that check run already speaks for it.

    A workflow-run conclusion is only ever read as a failure here, never as a merge signal
    (spec section 6). The queue's own run (GITHUB_RUN_ID) is never counted.

    Args:
        gh: The GitHub client.
        sha: The head commit.
        checks: The required check's runs on that head, from `read_checks`.

    Returns:
        One stand-in per live run and per failed run that reported no required check.

    Raises:
        GhError: A call failed.
    """
    own_run_id = int(os.environ.get("GITHUB_RUN_ID", "0") or 0)
    reported_suites = {c.suite_id for c in checks if c.suite_id is not None}
    stand_ins = []
    for run in runs_on(gh, sha):
        if _workflow_name(run) != REQUIRED_WORKFLOW or run.get("id") == own_run_id:
            continue
        url = run.get("html_url", "")
        if run.get("status") in LIVE_RUN_STATUSES:
            stand_ins.append(CheckRun(run["status"], None, None, url))
        elif (run.get("status") == "completed" and run.get("conclusion") in BROKEN_RUN_CONCLUSIONS
              and run.get("check_suite_id") not in reported_suites):
            stand_ins.append(CheckRun("completed", "failure", _ts(run.get("updated_at")), url))
    return tuple(stand_ins)


def _best_effort(log: Callable[[str], None], what: str, fn: Callable[[], object]) -> None:
    try:
        fn()
    except GhError as exc:
        log(f"  best-effort {what} failed: {exc}")


def comment_once(gh: GhApi, number: int, kind: str, sha: str, body: str) -> bool:
    """Post a comment unless one of this kind already exists for this head.

    Args:
        gh: The GitHub client.
        number: The PR number.
        kind: The comment's kind, part of its hidden marker.
        sha: The head the comment is about, part of its hidden marker.
        body: The visible text.

    Returns:
        True if it posted, False if the marker was already there.

    Raises:
        GhError: A call failed.
    """
    marker = f"<!-- merge-queue:{kind}:{sha} -->"
    owner, name = _owner_name()
    data = gh.graphql(COMMENTS_QUERY, owner=owner, name=name, number=number)
    bodies = [n.get("body") or "" for n in data["data"]["repository"]["pullRequest"]["comments"]["nodes"]]
    if any(marker in b for b in bodies):
        return False
    gh.rest("POST", f"repos/{REPO}/issues/{number}/comments", {"body": f"{marker}\n{body}"})
    return True


def dispatch(gh: GhApi, workflow: str, ref: str, pr_number: int | None = None) -> None:
    """Start a workflow_dispatch run.

    Args:
        gh: The GitHub client.
        workflow: The workflow file's basename.
        ref: The branch to run it on (its copy of the workflow is the one that runs).
        pr_number: Sent as the `pr` input, if given (only the required workflow takes it).

    Raises:
        GhError: GitHub refused the dispatch (e.g. HTTP 422, no workflow_dispatch trigger).
    """
    fields = {"ref": ref}
    if pr_number is not None:
        fields["inputs[pr]"] = str(pr_number)
    gh.rest("POST", f"repos/{REPO}/actions/workflows/{workflow}/dispatches", fields)


def _workflows_to_redispatch(runs: list[dict]) -> list[str]:
    """Workflows (other than the required one and the queue) that ran on the old head."""
    names = set()
    for run in runs:
        name = _workflow_name(run)
        if not name or name in (REQUIRED_WORKFLOW, QUEUE_WORKFLOW):
            continue
        if run.get("event") not in ("pull_request", "workflow_dispatch"):
            continue
        if run.get("conclusion") in ("skipped", "action_required"):
            continue
        names.add(name)
    return sorted(names)


def _evict(gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None]) -> bool:
    # Best-effort: a merge fires both push:dev and pull_request:closed, so two racing runs can
    # evict the same PR and the second disarm hits an already-disarmed PR.
    _best_effort(log, f"disarm #{pr.number}", lambda: gh.graphql(DISARM_MUTATION, id=pr.node_id))
    links = "".join(f"\n- {u}" for u in action.links)
    comment_once(
        gh, pr.number, f"evict-{action.slug}", pr.head_sha,
        f"Merge queue: removed from the line ({action.reason}). Auto-merge is now off. Fix the cause, then "
        f"re-arm with `gh pr merge {pr.number} --auto --merge` to rejoin at the back.{links}",
    )
    log(f"  evicted #{pr.number}: {action.reason}")
    return True


def _dispatch_required(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> bool:
    """Dispatch the required check on the PR's head; a branch-side refusal evicts the PR.

    Only HTTP 422 (the branch's workflow is broken or lacks the trigger) and HTTP 404 (the ref
    is gone) are the branch's fault. Anything else (5xx, rate limit, network, 403) re-raises, so
    the run fails and the next event retries (spec section 8): evicting on those would disarm
    every front an outage touches.

    Returns True if the PR was evicted (the line moves on), False if the run was started.
    """
    try:
        dispatch(gh, REQUIRED_WORKFLOW, pr.head_ref, pr.number)
    except GhError as exc:
        if not any(code in str(exc) for code in BRANCH_REFUSALS):
            raise
        return _evict(gh, pr, Action("evict", f"CI dispatch failed: {str(exc)[:200]}", slug="dispatch"), log)
    return False


def _rebase(gh: GhApi, pr: PrState, log: Callable[[str], None], sleep: Callable[[float], None]) -> bool:
    try:
        gh.graphql(REBASE_MUTATION, id=pr.node_id, oid=pr.head_sha)
    except GhError as exc:
        fresh = read_pr(gh, pr.number)
        if fresh.head_sha == pr.head_sha and fresh.merge_state != "DIRTY":
            # A racing run's rebase may have been accepted with its ref update still landing
            # (about 1 s): look once more before counting this as our own failure.
            sleep(REBASE_POLL_S)
            fresh = read_pr(gh, pr.number)
        if fresh.head_sha != pr.head_sha:
            log(f"  rebase skipped: head moved to {fresh.head_sha[:10]}")
            return False
        if fresh.merge_state == "DIRTY":
            return _evict(gh, fresh, Action("evict", "conflicts with dev (rebase refused)", slug="conflict"), log)
        error = str(exc)[:200]
        posted = comment_once(
            gh, pr.number, "rebase-failed", pr.head_sha,
            f"Merge queue: rebasing onto dev failed ({error}); will retry once, then remove from the line.",
        )
        if posted:
            log(f"  rebase failed, will retry once: {exc}")
            return False
        return _evict(gh, fresh, Action("evict", f"rebase onto dev keeps failing: {error}", slug="rebase"), log)
    # The mutation returns the PRE-rebase headRefOid and the branch moves about 1 s later, so
    # wait for the new head to appear before dispatching CI on it.
    new_head = None
    for _ in range(REBASE_POLLS):
        sleep(REBASE_POLL_S)
        fresh = read_pr(gh, pr.number)
        if fresh.head_sha != pr.head_sha:
            new_head = fresh.head_sha
            break
    if new_head is None:
        log(f"  rebase of #{pr.number} accepted but the head never moved; a later tick recovers it")
        return False
    old_runs = runs_on(gh, pr.head_sha)
    # Dispatch the required check *before* cancelling anything on the old head: queue-tick runs
    # inside its own derived-artifacts run, and an auto_merge_enabled-triggered merge-queue.yml
    # run shares this same head too. Cancelling first would kill the run executing this script
    # (spec section 7).
    if _dispatch_required(gh, replace(pr, head_sha=new_head), log):
        return True
    own_run_id = int(os.environ.get("GITHUB_RUN_ID", "0") or 0)
    for run in old_runs:
        if run.get("status") not in LIVE_RUN_STATUSES:
            continue
        if run.get("id") == own_run_id or _workflow_name(run) == QUEUE_WORKFLOW:
            continue
        _best_effort(log, f"cancel run {run['id']}",
                     lambda rid=run["id"]: gh.rest("POST", f"repos/{REPO}/actions/runs/{rid}/cancel"))
    refused = []
    for workflow in _workflows_to_redispatch(old_runs):
        try:
            dispatch(gh, workflow, pr.head_ref)
        except GhError as exc:
            refused.append(f"{workflow} ({str(exc)[:120]})")
            log(f"  re-dispatch of {workflow} failed: {exc}")
    note = f"\n\nNot re-run (dispatch failed): {', '.join(refused)}" if refused else ""
    comment_once(
        gh, pr.number, "rebased", new_head,
        f"Merge queue: this PR is next. Rebased onto `{BASE}` (head `{new_head[:10]}`) and started CI.{note}",
    )
    log(f"  rebased #{pr.number} {pr.head_sha[:10]} -> {new_head[:10]}")
    return False


def apply(
    gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None], sleep: Callable[[float], None],
) -> bool:
    """Perform one decided action (spec section 7).

    Args:
        gh: The GitHub client.
        pr: The front PR, as decided on.
        action: The decided action.
        log: Receives one line per notable step.
        sleep: Waits between rereads (injected by tests).

    Returns:
        True if the PR left the line (evicted, by decision or because an action failed), so
        the caller moves on to the next front; False otherwise.

    Raises:
        GhError: A read or a non-best-effort call failed.
    """
    if action.kind == "rebase":
        return _rebase(gh, pr, log, sleep)
    if action.kind in ("dispatch", "retry"):
        # Two queue runs (merge-queue.yml and a queue-tick) can decide the same dispatch for the
        # same head. Once the first one's run is listed, the other sees it here and stands down.
        # This narrows the race but cannot close it: the window left is the time between this
        # read and the POST, plus GitHub's delay between a dispatch returning and its run showing
        # up in the runs list, which no read can see. A duplicate is one extra queued run (spec
        # section 7).
        if any(c.status != "completed" for c in required_run_stand_ins(gh, pr.head_sha, ())):
            log(f"  #{pr.number}: a required run appeared on {pr.head_sha[:10]} since the decision; not dispatching")
            return False
        if _dispatch_required(gh, pr, log):
            return True
        if action.kind == "retry":
            comment_once(
                gh, pr.number, "retry", pr.head_sha,
                f"Merge queue: the required check failed once on `{pr.head_sha[:10]}`; retrying with a fresh run. "
                f"Failed run: {action.links[0]}",
            )
        return False
    if action.kind == "evict":
        return _evict(gh, pr, action, log)
    return False


def cleanup_approval_runs(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> None:
    """Delete the empty approval-pending runs our own token rebase created (spec F4, V1).

    Best-effort end to end: a failed listing must not abort the tick.

    Args:
        gh: The GitHub client.
        pr: The front PR; runs on its head are cleaned.
        log: Receives a line per failed best-effort call.
    """
    try:
        runs = runs_on(gh, pr.head_sha, status="action_required")
    except GhError as exc:
        log(f"  best-effort list approval-pending runs failed: {exc}")
        return
    for run in runs:
        if (run.get("triggering_actor") or {}).get("login") != "github-actions[bot]":
            continue
        _best_effort(log, f"delete approval-pending run {run['id']}",
                     lambda rid=run["id"]: gh.rest("DELETE", f"repos/{REPO}/actions/runs/{rid}"))


UNQUEUED_NOTES = {
    "fork": "Merge queue: fork PRs are not queued, because GitHub cannot dispatch workflows on a fork's branch. "
            "A maintainer merges this one by hand.",
    "bot": "Merge queue: PRs opened by a bot or app are not queued, because a queue dispatch would run this "
           "branch's workflows without the token limits and approval gates GitHub applies to bot-authored "
           "runs. A maintainer merges this one by hand.",
}


def comment_unqueued(gh: GhApi, prs: list[PrState], mode: str, log: Callable[[str], None]) -> None:
    """Tell armed fork and bot-authored PRs, once per head, that the queue skips them.

    Best-effort: a failed comment never aborts the run.

    Args:
        gh: The GitHub client.
        prs: Every open PR into dev.
        mode: `dry` only logs; `on` also comments.
        log: Receives one line per skipped PR.
    """
    for pr in prs:
        if pr.armed_at is None or (pr.same_repo and pr.human_author):
            continue
        kind = "fork" if not pr.same_repo else "bot"
        log(f"#{pr.number}: {kind} PR armed; not queued")
        if mode == "on":
            _best_effort(
                log, f"comment on {kind} PR #{pr.number}",
                lambda p=pr, k=kind: comment_once(gh, p.number, k, p.head_sha, UNQUEUED_NOTES[k]),
            )


def settle_unknown(gh: GhApi, pr: PrState, sleep: Callable[[float], None]) -> PrState:
    """Re-read a PR whose merge state is UNKNOWN until it settles, up to UNKNOWN_REREADS times.

    Args:
        gh: The GitHub client.
        pr: The front PR.
        sleep: Waits between rereads (injected by tests).

    Returns:
        The latest state read; still UNKNOWN if it never settled.

    Raises:
        GhError: A reread failed.
    """
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
    """One queue pass: decide for the front PR and, in `on` mode, act; repeat after evictions.

    Args:
        gh: The GitHub client.
        mode: The MERGE_QUEUE value; only `dry` and `on` (any case or padding) do anything.
        now: The clock (injected by tests).
        sleep: Waits between rereads (injected by tests).
        log: Receives the run's log lines.

    Returns:
        The (PR number, action) decisions taken or, in dry mode, proposed.

    Raises:
        GhError: A read or a non-best-effort call failed; the next event retries.
    """
    mode = (mode or "").strip().lower()
    if mode not in ("dry", "on"):
        log("merge queue is off (MERGE_QUEUE is not 'dry' or 'on')")
        return []
    prs = read_prs(gh)
    comment_unqueued(gh, prs, mode, log)
    decisions: list[tuple[int, Action]] = []
    for pr in line_of(prs)[:MAX_FRONTS_PER_RUN]:
        pr = settle_unknown(gh, pr, sleep)
        if pr.armed_at is None:
            continue
        checks = read_checks(gh, pr.head_sha)
        pr = replace(pr, checks=checks + required_run_stand_ins(gh, pr.head_sha, checks))
        action = decide_front(pr, now())
        decisions.append((pr.number, action))
        log(f"#{pr.number}: {action.kind} - {action.reason}")
        left_line = action.kind == "evict"
        if mode == "on":
            cleanup_approval_runs(gh, pr, log)
            left_line = apply(gh, pr, action, log, sleep)
        if not left_line:
            break
    return decisions


def main() -> int:
    """Run one queue pass with the real gh CLI and append the decisions to the job summary.

    Returns:
        The process exit code (0; a failed call raises instead).
    """
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
