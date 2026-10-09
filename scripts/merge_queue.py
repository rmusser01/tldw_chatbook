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
QUEUE_WORKFLOW = "merge-queue.yml"  # never cancelled by a rebase; never dispatched (not on main)
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
# gh reports API errors as `gh: <message> (HTTP NNN)` (verified live).
# A queue rebase (GITHUB_TOKEN) makes GitHub create the PR's pull_request runs held for approval
# (spec F4). Those are the runs whose checks count toward mergeability; a workflow_dispatch run's
# check does not appear in the PR's status rollup at all (spec V4, 2026-10-06). So the queue
# approves the held runs and never dispatches the required check.
QUEUE_ACTOR = "github-actions[bot]"
HELD_RUN_POLLS = 10
HELD_RUN_POLL_S = 3
# How a re-run that GitHub answered with an error is read (`_rerun_error`):
# - refused, for good: 409/422 (not re-runnable), or a 403 saying the run is over a month old;
# - transient: a 5xx, a 429, a rate limit (primary and secondary limits both answer 403), or no HTTP
#   status at all (a network error). These re-raise, so the run fails and the next event retries;
#   they never disarm a front (spec section 8);
# - unknown, anything else (GitHub answers a re-run of a broken workflow file with a 403, as does a
#   permission error): the first one re-raises with a `rerun-error` comment, and the second on the
#   same head counts as refused, so a lasting error cannot fail every queue run forever.
RERUN_REFUSALS = ("(HTTP 409)", "(HTTP 422)")
RERUN_REFUSAL_403_TEXT = "month ago"
# How long a woken queue run waits for a run to complete before deciding: well past the seconds the
# waited run's queue-tick needs to finish, and inside queue-tick's own 10-minute timeout with room
# left to act.
# ponytail: a run still live after this is a GitHub fault; the tick decides anyway and the next event recovers.
WAIT_RUN_TRIES = 50
WAIT_RUN_DELAY_S = 6.0


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
    """The single decision for the front PR: wait, rebase, start, retry or evict.

    `slug` names an eviction's cause in its comment marker (`evict-<slug>`), so a re-armed PR
    evicted again on the same head for a different reason is still told why.

    Attributes:
        kind: `wait`, `rebase`, `start` (get a counted CI run going), `retry` or `evict`.
        reason: A human-readable cause, logged and quoted in comments.
        links: Run URLs quoted in the comment (the failed runs).
        slug: An eviction's cause, used in its comment marker; `young` on the young-head wait.
        suite_id: For a retry, the failed check's check suite, which names the run to re-run;
            None when the failure has no check run (a broken run's stand-in).
    """

    kind: str
    reason: str
    links: tuple[str, ...] = ()
    slug: str = ""
    suite_id: int | None = None


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
            return Action("wait", "head is under 3 minutes old; its own run may not be visible yet", slug="young")
        return Action("start", "no required-check run on the up-to-date head")
    latest = finished[-1]
    failed = [c for c in finished if c.conclusion not in PASSING]
    if latest.conclusion not in PASSING:
        if len(failed) >= 2:
            return Action("evict", "required check failed twice", tuple(c.url for c in failed[-2:]), "failed-twice")
        return Action("retry", "required check failed once; retrying", (latest.url,), suite_id=latest.suite_id)
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


def counted(checks: tuple[CheckRun, ...], runs: list[dict]) -> tuple[CheckRun, ...]:
    """Drop required-check runs that came from a non-pull_request run (e.g. workflow_dispatch).

    Those never satisfy the required check, being absent from the PR's status rollup (spec V4).
    A check whose run is not listed is kept.

    Args:
        checks: The required check's runs on the head, from `read_checks`.
        runs: Every workflow run on the head, from `runs_on`.

    Returns:
        The checks that count.
    """
    not_counted = {r.get("check_suite_id") for r in runs
                   if _workflow_name(r) == REQUIRED_WORKFLOW and r.get("event") != "pull_request"}
    return tuple(c for c in checks if c.suite_id is None or c.suite_id not in not_counted)


def required_run_stand_ins(gh: GhApi, sha: str, checks: tuple[CheckRun, ...],
                           runs: list[dict] | None = None) -> tuple[CheckRun, ...]:
    """Required-workflow runs on this head that have not reported the required check, as stand-ins.

    - A LIVE run stands in as an in-flight check: the required check is a needs-gated aggregate
      with no check run until the lanes finish (verified live: run 37156228421 showed
      `total_count: 0` while its lanes ran), so an in-flight front PR would otherwise look like
      it has no run and be started again on every tick.
    - A COMPLETED run that failed (`BROKEN_RUN_CONCLUSIONS`) with no required check run in its
      check suite stands in as one failure: a startup failure, e.g. a broken
      derived-artifacts.yml on the branch, would otherwise read as "no run" instead of being
      retried once and then evicted. A failed run whose suite DID report the required check is not
      a CI failure (its queue-tick may have failed); that check run already speaks for it.

    A workflow-run conclusion is only ever read as a failure here, never as a merge signal
    (spec section 6). The queue's own run (GITHUB_RUN_ID) is never counted.

    Only pull_request runs stand in: a workflow_dispatch run can never satisfy the required
    check (spec V4), so waiting on one would only stall the line.

    Args:
        gh: The GitHub client.
        sha: The head commit.
        checks: The required check's runs on that head, from `read_checks`.
        runs: The head's workflow runs, if already read; read here otherwise.

    Returns:
        One stand-in per live run and per failed run that reported no required check.

    Raises:
        GhError: A call failed.
    """
    own_run_id = int(os.environ.get("GITHUB_RUN_ID", "0") or 0)
    reported_suites = {c.suite_id for c in checks if c.suite_id is not None}
    stand_ins = []
    for run in runs_on(gh, sha) if runs is None else runs:
        if (_workflow_name(run) != REQUIRED_WORKFLOW or run.get("id") == own_run_id
                or run.get("event") != "pull_request"):
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


def has_comment(gh: GhApi, number: int, kind: str, sha: str) -> bool:
    """Whether the queue already posted a comment of this kind for this head.

    Args:
        gh: The GitHub client.
        number: The PR number.
        kind: The comment's kind, part of its hidden marker.
        sha: The head, part of its hidden marker.

    Returns:
        True if the marker is there.

    Raises:
        GhError: The read failed.
    """
    marker = f"<!-- merge-queue:{kind}:{sha} -->"
    owner, name = _owner_name()
    data = gh.graphql(COMMENTS_QUERY, owner=owner, name=name, number=number)
    nodes = data["data"]["repository"]["pullRequest"]["comments"]["nodes"]
    return any(marker in (n.get("body") or "") for n in nodes)


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


def _is_held(run: dict) -> bool:
    """Whether GitHub holds this run for approval (it lists as completed, `action_required`)."""
    return "action_required" in (run.get("status"), run.get("conclusion"))


def _queue_held(run: dict) -> bool:
    """Whether this is a held run the queue may approve: a pull_request run its own token caused.

    Never a run from another repository (GitHub's approval gate for outside contributors) or one
    another actor caused, such as an agent's push waiting for a human's approval (spec section 9).

    Args:
        run: A workflow run, as the API returns it.

    Returns:
        True only for the queue's own held pull_request runs on this repository's branches.
    """
    return (_is_held(run) and run.get("event") == "pull_request"
            and (run.get("triggering_actor") or {}).get("login") == QUEUE_ACTOR
            and (run.get("head_repository") or {}).get("full_name") == REPO)


def _approve_strictly(gh: GhApi, run: dict, log: Callable[[str], None]) -> None:
    # Not best-effort: on a GitHub outage the queue run must fail (the next event retries), never
    # fall through to an eviction and disarm the PR (spec section 8).
    gh.rest("POST", f"repos/{REPO}/actions/runs/{run['id']}/approve")
    log(f"  approved held run {run['id']} ({_workflow_name(run)})")


def approve_held_runs(gh: GhApi, sha: str, log: Callable[[str], None]) -> set[str]:
    """Approve the runs GitHub holds on this head because the queue's own token caused them.

    After a GITHUB_TOKEN rebase, GitHub creates the PR's pull_request runs "in an
    approval-required state" (spec F4, GitHub docs). Approving them makes them run as ordinary
    pull_request runs, whose required check counts toward mergeability (spec V4, verified live on
    #3033, 2026-10-06). Only runs triggered by the queue's own actor are approved, and only for the
    front PR: a same-repo, user-authored PR that a human armed.

    Best-effort per run: one refused approval must not stop the others.

    Args:
        gh: The GitHub client.
        sha: The front PR's head.
        log: Receives a line per approval and per failure.

    Returns:
        The workflow names whose held runs were approved.
    """
    try:
        held = runs_on(gh, sha, status="action_required")
    except GhError as exc:
        log(f"  best-effort list held runs failed: {exc}")
        return set()
    approved: set[str] = set()
    for run in held:
        if not _queue_held(run):
            continue
        try:
            gh.rest("POST", f"repos/{REPO}/actions/runs/{run['id']}/approve")
        except GhError as exc:
            log(f"  best-effort approve run {run['id']} failed: {exc}")
            continue
        approved.add(_workflow_name(run))
        log(f"  approved held run {run['id']} ({_workflow_name(run)})")
    return approved


def _approve_until_required(gh: GhApi, sha: str, log: Callable[[str], None], sleep: Callable[[float], None]) -> bool:
    """Approve held runs on a fresh head, waiting (bounded) for the required workflow's to appear.

    Args:
        gh: The GitHub client.
        sha: The new head.
        log: Receives one line per approval.
        sleep: Waits between polls (injected by tests).

    Returns:
        True once the required workflow's held run was approved; False if it never appeared.
    """
    approved: set[str] = set()
    for _ in range(HELD_RUN_POLLS):
        approved |= approve_held_runs(gh, sha, log)
        if REQUIRED_WORKFLOW in approved:
            return True
        sleep(HELD_RUN_POLL_S)
    return False


def wake(gh: GhApi, wait_run: int | None = None) -> None:
    """Start a queue tick on dev: a `derived-artifacts.yml` dispatch without `pr` (a queue kick).

    `derived-artifacts.yml` is dispatched, not `merge-queue.yml`, because GitHub only starts a
    workflow_dispatch run when the workflow file exists on the default branch (`main`), and of the
    two only `derived-artifacts.yml` does. On dev with no `pr` its lanes skip and only queue-tick
    runs. Ceiling: GitHub keeps one pending run per concurrency group, so a newer kick replaces a
    pending older one; the queue only ever acts on its single front PR, so one wait is in play.

    Args:
        gh: The GitHub client.
        wait_run: A run the woken tick waits for (bounded) before deciding, if any.

    Raises:
        GhError: The dispatch failed.
    """
    fields = {"ref": BASE}
    if wait_run is not None:
        fields["inputs[wait_run]"] = str(wait_run)
    gh.rest("POST", f"repos/{REPO}/actions/workflows/{REQUIRED_WORKFLOW}/dispatches", fields)


def wait_for_run(gh: GhApi, run_id: int, sleep: Callable[[float], None], log: Callable[[str], None]) -> None:
    """Wait, bounded, for a workflow run to complete; a run still live after the bound is left.

    Args:
        gh: The GitHub client.
        run_id: The run to wait for.
        sleep: Waits between reads (injected by tests).
        log: Receives a line if the bound is reached.
    """
    error: GhError | None = None
    for _ in range(WAIT_RUN_TRIES):
        try:
            if gh.rest("GET", f"repos/{REPO}/actions/runs/{run_id}").get("status") not in LIVE_RUN_STATUSES:
                return
        except GhError as exc:  # a failed read is retried within the bound, never fatal
            error = exc
        sleep(WAIT_RUN_DELAY_S)
    seen = f"last read failed: {error}" if error else "still live"
    log(f"run {run_id} not seen complete after {WAIT_RUN_TRIES * WAIT_RUN_DELAY_S:.0f}s ({seen}); deciding anyway")


def _required_runs(gh: GhApi, sha: str) -> list[dict]:
    """The required workflow's pull_request runs on this head, newest first."""
    runs = [r for r in runs_on(gh, sha) if _workflow_name(r) == REQUIRED_WORKFLOW and r.get("event") == "pull_request"]
    return sorted(runs, key=lambda r: r.get("created_at") or "", reverse=True)


def _going_again(gh: GhApi, run_id: int) -> bool:
    """Whether a run is live or held again, i.e. a racing queue run has re-run it since we looked."""
    fresh = gh.rest("GET", f"repos/{REPO}/actions/runs/{run_id}")
    return fresh.get("status") in LIVE_RUN_STATUSES or _is_held(fresh)


def _rerun_error(exc: GhError) -> str:
    """Classify a re-run error as `refused`, `transient` or `unknown` (see RERUN_REFUSALS).

    Args:
        exc: The error the re-run call raised.

    Returns:
        The error's class.
    """
    text = str(exc)
    lowered = text.lower()
    if any(code in text for code in RERUN_REFUSALS) or ("(HTTP 403)" in text and RERUN_REFUSAL_403_TEXT in lowered):
        return "refused"
    if "(HTTP 5" in text or "(HTTP 429)" in text or "rate limit" in lowered or "(HTTP " not in text:
        return "transient"
    return "unknown"


def _rerun(gh: GhApi, pr: PrState, run: dict, failed_only: bool, log: Callable[[str], None],
           sleep: Callable[[float], None]) -> str:
    """Re-run a required-workflow run in place, then approve the re-run if GitHub holds it.

    A re-run adds a new attempt to the same check suite, so it replaces the old result for branch
    protection while `read_checks` (filter=all) still lists the old one (spec V3).

    Args:
        gh: The GitHub client.
        pr: The front PR.
        run: The workflow run to re-run.
        failed_only: Re-run only the failed jobs (`rerun-failed-jobs`) instead of the whole run.
        log: Receives one line per notable step.
        sleep: Waits between polls (injected by tests).

    Returns:
        `started` (by this run, or by a racing queue run whose re-run made GitHub refuse this
        one), or `refused` (GitHub will not re-run it).

    Raises:
        GhError: A transient error, or the first unknown one on this head (see RERUN_REFUSALS).
    """
    path = f"repos/{REPO}/actions/runs/{run['id']}/{'rerun-failed-jobs' if failed_only else 'rerun'}"
    try:
        gh.rest("POST", path)
    except GhError as exc:
        kind = _rerun_error(exc)
        if kind == "transient":
            raise
        # Whatever code GitHub refused with, a racing queue run may have re-run it first.
        if _going_again(gh, run["id"]):
            log(f"  #{pr.number}: run {run['id']} was re-run by a racing queue run; standing down")
            return "started"
        if kind == "unknown" and comment_once(
            gh, pr.number, "rerun-error", pr.head_sha,
            f"Merge queue: re-running the required check failed ({str(exc)[:200]}); will try once more, "
            "then remove from the line.",
        ):
            raise
        log(f"  #{pr.number}: re-run of run {run['id']} refused: {exc}")
        return "refused"
    # The re-run's actor is the queue's token, so GitHub may hold it for approval again.
    for _ in range(HELD_RUN_POLLS):
        sleep(HELD_RUN_POLL_S)
        approve_held_runs(gh, pr.head_sha, log)
        if gh.rest("GET", f"repos/{REPO}/actions/runs/{run['id']}").get("status") in LIVE_RUN_STATUSES:
            break
    else:
        # A held run never executes, so its own queue-tick never comes: wake one to approve it.
        _best_effort(log, "wake a queue tick", lambda: wake(gh))
    log(f"  #{pr.number}: re-ran run {run['id']}")
    return "started"


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
    # wait for the new head to appear before approving its CI.
    new_head = None
    for _ in range(REBASE_POLLS):
        sleep(REBASE_POLL_S)
        fresh = read_pr(gh, pr.number)
        if fresh.head_sha != pr.head_sha:
            new_head = fresh.head_sha
            break
    if new_head is None:
        log(f"  rebase of #{pr.number} accepted but the head never moved")
        # If the branch moves later, its new head's runs are held and nothing would wake the
        # queue to approve them. One kick per head; after that the next event recovers it.
        if comment_once(gh, pr.number, "rebase-unmoved", pr.head_sha,
                        "Merge queue: the rebase onto dev was accepted, but the branch has not moved yet; "
                        "a queue run will look again shortly."):
            _best_effort(log, "wake a queue tick", lambda: wake(gh))
        return False
    old_runs = runs_on(gh, pr.head_sha)
    # The new head's pull_request runs arrive held for approval (spec F4); approving them is what
    # makes its required check count (V4). Approve before cancelling anything on the old head:
    # queue-tick runs inside its own derived-artifacts run, and an auto_merge_enabled-triggered
    # merge-queue.yml run shares this same head too. Cancelling first would kill the run
    # executing this script (spec section 7).
    approved = _approve_until_required(gh, new_head, log, sleep)
    own_run_id = int(os.environ.get("GITHUB_RUN_ID", "0") or 0)
    for run in old_runs:
        if run.get("status") not in LIVE_RUN_STATUSES:
            continue
        if run.get("id") == own_run_id or _workflow_name(run) == QUEUE_WORKFLOW:
            continue
        _best_effort(log, f"cancel run {run['id']}",
                     lambda rid=run["id"]: gh.rest("POST", f"repos/{REPO}/actions/runs/{rid}/cancel"))
    if not approved:
        # The PR's own runs stay held, so no queue-tick will run for it: wake one.
        _best_effort(log, "wake a queue tick", lambda: wake(gh))
    started = "approved its CI" if approved else "its CI is not approved yet; a woken tick approves it"
    comment_once(
        gh, pr.number, "rebased", new_head,
        f"Merge queue: this PR is next. Rebased onto `{BASE}` (head `{new_head[:10]}`); {started}.",
    )
    log(f"  rebased #{pr.number} {pr.head_sha[:10]} -> {new_head[:10]}")
    return False


def _start(gh: GhApi, pr: PrState, log: Callable[[str], None], sleep: Callable[[float], None]) -> bool:
    """Get a counted required-check run going on an up-to-date head that has none.

    Approve a held run, else re-run a cancelled one. A dispatched run would not count (spec V4),
    so with neither the PR is evicted: a push, or closing and reopening it, starts its CI. Before
    evicting it looks again for up to HELD_RUN_POLLS x HELD_RUN_POLL_S: the head's age comes from
    its commit date, not its push, so a commit made well before it was pushed can reach this
    while GitHub has yet to list its run.

    Args:
        gh: The GitHub client.
        pr: The front PR.
        log: Receives one line per notable step.
        sleep: Waits between polls (injected by tests).

    Returns:
        True if the PR was evicted.

    Raises:
        GhError: An approval or re-run failed for a reason other than a refusal.
    """
    for attempt in range(HELD_RUN_POLLS):
        if attempt:
            sleep(HELD_RUN_POLL_S)
        held = [r for r in runs_on(gh, pr.head_sha, status="action_required")
                if _queue_held(r) and _workflow_name(r) == REQUIRED_WORKFLOW]
        if held:
            for run in held:
                _approve_strictly(gh, run, log)
            return False
        runs = _required_runs(gh, pr.head_sha)
        if any(r.get("status") in LIVE_RUN_STATUSES for r in runs):
            log(f"  #{pr.number}: its required run appeared on {pr.head_sha[:10]}; standing down")
            return False
        cancelled = next((r for r in runs if r.get("conclusion") == "cancelled"), None)
        if cancelled is not None:
            if _rerun(gh, pr, cancelled, False, log, sleep) == "started":
                return False
            break
    return _evict(gh, pr, Action(
        "evict", "no CI run on this head that the queue can start; push a commit, or close and reopen the PR",
        slug="no-run"), log)


def _retry(gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None], sleep: Callable[[float], None]) -> bool:
    """Re-run the run whose required check failed once, in its own check suite (spec V3).

    The retry is usually decided by the failed run's own queue-tick while that run is still in
    progress, and GitHub only re-runs a completed run, so that case wakes a queue run that waits
    for it (`wake`). A run that is live again (re-run by a racing queue run) is left alone, and
    one GitHub holds again (its re-run's approval failed) is approved. One retry per head: a run
    that failed without reporting the check keeps its run id when re-run, so its failures never
    add up to two; the retry comment marker caps it instead. A retry attempt that was cancelled,
    not failed, is re-run without counting against that cap.

    Args:
        gh: The GitHub client.
        pr: The front PR.
        action: The retry decision, carrying the failed check's suite and run URL.
        log: Receives one line per notable step.
        sleep: Waits between polls (injected by tests).

    Returns:
        True if the PR was evicted.

    Raises:
        GhError: A re-run or the wake failed for a reason other than a refusal.
    """
    runs = _required_runs(gh, pr.head_sha)
    run = next((r for r in runs if action.suite_id is not None and r.get("check_suite_id") == action.suite_id), None)
    if run is None:  # a broken run's stand-in: its failed run reported no check
        run = next((r for r in runs if r.get("conclusion") in BROKEN_RUN_CONCLUSIONS), None)
    if run is None:
        return _evict(gh, pr, Action("evict", "required check failed and its run cannot be found to re-run",
                                     action.links, "rerun"), log)
    if run.get("id") == int(os.environ.get("GITHUB_RUN_ID", "0") or 0):
        wake(gh, run["id"])
        log(f"  #{pr.number}: the failed run is this run ({run['id']}); woke a queue run to re-run it once it completes")
        return False
    if run.get("status") in LIVE_RUN_STATUSES:
        log(f"  #{pr.number}: run {run['id']} is live again; standing down")
        return False
    if _queue_held(run):
        _approve_strictly(gh, run, log)
        return False
    cancelled = run.get("conclusion") == "cancelled"
    if not cancelled and has_comment(gh, pr.number, "retry", pr.head_sha):
        if _going_again(gh, run["id"]):
            log(f"  #{pr.number}: run {run['id']} was re-run by a racing queue run; standing down")
            return False
        return _evict(gh, pr, Action("evict", "required check failed again after its retry", action.links,
                                     "failed-twice"), log)
    if _rerun(gh, pr, run, action.suite_id is not None and not cancelled, log, sleep) == "refused":
        return _evict(gh, pr, Action("evict", "required check failed once and GitHub refused to re-run it",
                                     action.links, "rerun"), log)
    comment_once(
        gh, pr.number, "retry", pr.head_sha,
        f"Merge queue: the required check failed once on `{pr.head_sha[:10]}`; re-running it. "
        f"Failed run: {action.links[0]}",
    )
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
    if action.kind in ("start", "retry"):
        # Two queue runs (merge-queue.yml and a queue-tick) can decide the same action for the same
        # head. Once the first one's run is live, the other sees it here and stands down.
        if any(c.status != "completed" for c in required_run_stand_ins(gh, pr.head_sha, ())):
            log(f"  #{pr.number}: a required run is live on {pr.head_sha[:10]}; standing down")
            return False
        return _start(gh, pr, log, sleep) if action.kind == "start" else _retry(gh, pr, action, log, sleep)
    if action.kind == "evict":
        return _evict(gh, pr, action, log)
    return False


UNQUEUED_NOTES = {
    "fork": "Merge queue: fork PRs are not queued, because approving a fork's held runs would bypass GitHub's "
            "approval gate for outside contributors' workflows. A maintainer merges this one by hand.",
    "bot": "Merge queue: PRs opened by a bot or app are not queued, because a queue-approved run would run this "
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


def _decide(gh: GhApi, pr: PrState, mode: str, now: Callable[[], datetime],
            log: Callable[[str], None]) -> tuple[PrState, Action]:
    """Read the front PR's counted checks and runs, then decide (spec section 6).

    Args:
        gh: The GitHub client.
        pr: The front PR, its merge state settled.
        mode: `dry` or `on`.
        now: The clock.
        log: Receives approval lines.

    Returns:
        The PR with its checks filled in, and the decision.

    Raises:
        GhError: A read failed.
    """
    if mode == "on" and pr.merge_state not in ("BEHIND", "DIRTY"):
        # Held runs left by an earlier rebase or re-run: approving them first lets the
        # decision below see them as the live runs they become. A BEHIND front is about to
        # be rebased (its old head's runs cancelled) and a DIRTY one evicted.
        approve_held_runs(gh, pr.head_sha, log)
    checks = read_checks(gh, pr.head_sha)
    runs = runs_on(gh, pr.head_sha)
    checks = counted(checks, runs)
    pr = replace(pr, checks=checks + required_run_stand_ins(gh, pr.head_sha, checks, runs))
    return pr, decide_front(pr, now())


def run(
    gh: GhApi,
    mode: str | None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    wait_run: int | None = None,
) -> list[tuple[int, Action]]:
    """One queue pass: decide for the front PR and, in `on` mode, act; repeat after evictions.

    Args:
        gh: The GitHub client.
        mode: The MERGE_QUEUE value; only `dry` and `on` (any case or padding) do anything.
        now: The clock (injected by tests).
        sleep: Waits between rereads (injected by tests).
        log: Receives the run's log lines.
        wait_run: A run to wait for (bounded) before deciding, set when a queue-tick woke this
            run because its own run failed the required check (`wake`).

    Returns:
        The (PR number, action) decisions taken or, in dry mode, proposed.

    Raises:
        GhError: A read or a non-best-effort call failed; the next event retries.
    """
    mode = (mode or "").strip().lower()
    if mode not in ("dry", "on"):
        log("merge queue is off (MERGE_QUEUE is not 'dry' or 'on')")
        return []
    if wait_run:
        wait_for_run(gh, wait_run, sleep, log)
    prs = read_prs(gh)
    comment_unqueued(gh, prs, mode, log)
    decisions: list[tuple[int, Action]] = []
    for pr in line_of(prs)[:MAX_FRONTS_PER_RUN]:
        pr = settle_unknown(gh, pr, sleep)
        if pr.armed_at is None:
            continue
        pr, action = _decide(gh, pr, mode, now, log)
        if mode == "on" and action.slug == "young":
            # A tick woken right after a rebase whose held runs had not appeared, or an arm right
            # after a push, can land before GitHub lists the head's run, and nothing may wake the
            # queue again for this head. Wait out the window once and decide again.
            # Capped: the commit date comes from the author's clock and can lie in the future.
            left = (pr.head_committed_at + YOUNG_HEAD - now()).total_seconds()
            sleep(min(max(0.0, left), YOUNG_HEAD.total_seconds()) + 1)
            fresh = settle_unknown(gh, read_pr(gh, pr.number), sleep)
            if fresh.armed_at != pr.armed_at:
                # Disarmed, or re-armed at the back of the line, meanwhile; that event wakes a
                # fresh queue run, which reads the line again.
                break
            pr, action = _decide(gh, fresh, mode, now, log)
        decisions.append((pr.number, action))
        log(f"#{pr.number}: {action.kind} - {action.reason}")
        left_line = action.kind == "evict"
        if mode == "on":
            left_line = apply(gh, pr, action, log, sleep)
        if not left_line:
            break
    return decisions


def main() -> int:
    """Run one queue pass with the real gh CLI and append the decisions to the job summary.

    Returns:
        The process exit code (0; a failed call raises instead).
    """
    wait_run = (os.environ.get("WAIT_RUN") or "").strip()
    decisions = run(Gh(), os.environ.get("MERGE_QUEUE"), wait_run=int(wait_run) if wait_run.isdigit() else None)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Merge queue\n\n| PR | Action | Reason |\n|---|---|---|\n")
            for number, action in decisions:
                fh.write(f"| #{number} | {action.kind} | {action.reason} |\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
