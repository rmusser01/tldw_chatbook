# In-repo merge queue for `dev` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PRs armed for auto-merge into `dev` merge one at a time, in arming order. Only the front PR is rebased and tested; every other armed PR is left untouched.

**Architecture:**
- One stdlib Python script (`scripts/merge_queue.py`) holds a pure decision function and a thin action layer over the `gh` CLI.
- It runs inside GitHub Actions on the built-in `GITHUB_TOKEN`, from two entry points:
  - a new `merge-queue.yml`, woken by PR arm, disarm and close events and by pushes to `dev`;
  - a new non-required `queue-tick` job in `derived-artifacts.yml`, which runs after CI.
- `derived-artifacts.yml` gains a `workflow_dispatch` path, so the queue can start the required check on a rebased head.

**Tech Stack:**
- Python 3.12, standard library only;
- the `gh` CLI (preinstalled on GitHub-hosted runners);
- GitHub Actions YAML;
- pytest and PyYAML for tests (already in the fast-lane install).

**Spec:** `Docs/superpowers/specs/2026-10-03-merge-queue-design.md`. Read it before any task. Section numbers below refer to it.

## Global Constraints

- **Credentials:** only the built-in `GITHUB_TOKEN`. No PAT, GitHub App key or deploy key anywhere (spec section 3).
- **Triggers:** nothing on `main`, so no `schedule`, `workflow_run`, `check_suite`, `check_run` or `pull_request_target`
  trigger (spec section 3, D3).
- **Forbidden queue actions:** enabling auto-merge, merging a PR, or any git push (spec section 7).
- **Required check name** (string constant, never renamed): `Derived artifacts reproduce from their sources`.
- **Mode variable:** `MERGE_QUEUE`. Unset or any value other than `dry`/`on` means off.
- **Timing constants:** young-head window 3 minutes; stuck-green window 15 minutes; at most 10 front PRs evaluated per run;
  `UNKNOWN` merge state re-read 12 times, 10 seconds apart (raised from 3 x 5 s in the final review); after a rebase,
  the PR re-read up to 10 times, 3 seconds apart, until the head moves.
- **Rebase rule:** rebase onto `dev`, never merge `dev` in. Push with `--force-with-lease` only.
- **Python for local runs:** `PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. Never run the app.
  `git stash` is forbidden (it is shared across worktrees).
- **Shell discipline:**
  - Start every multi-step shell command with `cd <worktree> && [ "$(pwd)" = <worktree> ]`, under `bash -euo pipefail -c`.
  - Never chain with `;` after a `cd`.
  - Brace variables before `:` in zsh (`"${b}:path"`).
- **Implementation worktree:** `/private/tmp/mq-impl`, branch `feat/merge-queue`, created from `docs/merge-queue-spec`, so
  the one PR carries the spec, this plan and the code.
- **Preflight:** run `PYTHON=$PY ./scripts/preflight.sh` before every push. Capture the exit code with `out=$(...); rc=$?`,
  never through `| tail`.

## Review Focus

These are the failure modes the spec implies that are most likely to bite. Each has a pinned test in the task that owns it.

1. **A failing `queue-tick` job must never count as a CI failure.** It turns the derived-artifacts *workflow run* red, but
   the queue reads only the required check runs. Test: `test_queue_tick_failure_is_not_a_ci_failure` (Task 3).
2. **Approval-pending runs appear after the queue's rebase, not during it.** Cleanup must happen lazily on every tick, and
   delete only runs triggered by `github-actions[bot]`. Test: `test_cleanup_deletes_only_bot_approval_runs` (Task 3).
3. **A PR re-armed after eviction rejoins at the back**, because `enabledAt` resets. Test:
   `test_rearmed_pr_rejoins_at_the_back` (Task 2).
4. **A conflicting front PR followed by a behind PR:** one run evicts the first and rebases the second. Test:
   `test_evicted_front_hands_over_in_the_same_run` (Task 3).
5. **Odd values of the mode variable** (`On`, ` on `, `yes`) must resolve deterministically: normalised `on`/`dry` act,
   anything else is off. Test: `test_mode_values` (Task 3).

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/merge_queue.py` (new) | Queue rules (`line_of`, `decide_front`), GitHub reads, the action layer, `run()` and `main()` |
| `.github/workflows/merge-queue.yml` (new) | Entry point for PR arm, disarm and close events and pushes to `dev` |
| `.github/workflows/derived-artifacts.yml` (modify) | `workflow_dispatch` with input `pr`, the shared lane condition, and the `queue-tick` job |
| `.github/workflows/perf-guard.yml`, `task-19642-smoke-clock-matrix.yml`, `task-32011-linux-storage-evidence.yml` (modify) | Made dispatch-safe |
| `scripts/measure_required_runs_per_merge.py` (new) | Success measure: required runs per merged PR, by cause |
| `Tests/CI/test_merge_queue_rules.py` (new) | `line_of` and `decide_front` table tests |
| `Tests/CI/test_merge_queue_actions.py` (new) | `run()` and action layer against a fake `gh`, plus the guard test |
| `Tests/CI/test_merge_queue_workflow.py` (new) | Shape of `merge-queue.yml` |
| `Tests/CI/test_pr_workflows_dispatch_safe.py` (new) | Every PR workflow can be dispatched safely |
| `Tests/CI/test_measure_required_runs_per_merge.py` (new) | Classifier and summary of the measurement script |
| `Tests/CI/test_derived_artifacts_workflow.py`, `Tests/CI/test_ci_queue_pressure_contract.py` (modify) | Update the 6 pins; add the `queue-tick` shape |
| `CLAUDE.md`, `AGENTS.md` (modify) | Mode-dependent merge rules (spec section 10) |
| `backlog/decisions/<NNN>-in-repo-merge-queue.md` (new), `103-*.md` (modify) | ADR for the queue, and the cross-reference |
| `backlog/docs/branch-protection-baseline.md` (modify) | Records the queue alongside the protection settings |
| `backlog/tasks/task-<ID> - Build-the-in-repo-merge-queue-for-dev.md` (new) | Tracking task |

All new tests live in `Tests/CI`, which is already a PR fast-lane target, so they are gated without any workflow change.

---

### Task 1: Tracking task and the V1/V2 token-permission check

**Files:**
- Create: `backlog/tasks/task-<ID> - Build-the-in-repo-merge-queue-for-dev.md`
- Modify: `Docs/superpowers/specs/2026-10-03-merge-queue-design.md` (section 4, the V-table)
- Throwaway, never committed to `feat/merge-queue`: branch `spike/merge-queue-v1v2`, with `.github/workflows/spike-v1v2.yml`

**Interfaces:**
- Produces: the V1/V2 results that Task 3 relies on. If V1 is refuted, Task 3 drops `cleanup_approval_runs`. If V2 is
  refuted, Task 3 drops the cancel loop in `_rebase`.

- [ ] **Step 1: Create the implementation worktree**

```bash
bash -euo pipefail -c '
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git fetch -q origin
git worktree add -q -b feat/merge-queue /private/tmp/mq-impl docs/merge-queue-spec
cd /private/tmp/mq-impl; [ "$(pwd)" = /private/tmp/mq-impl ]
git rebase origin/dev
git log --oneline -3'
```

Expected: the two spec commits and the plan commit sit on top of the current `origin/dev`.

- [ ] **Step 2: Pick the task ID.** Use the highest ID across every origin ref and every open PR, plus 10:

```bash
bash -euo pipefail -c '
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
m1=$(git for-each-ref --format="%(refname)" refs/remotes/origin | while read r; do git ls-tree --name-only "${r}" backlog/tasks/ 2>/dev/null; done | grep -oE "task-[0-9]+" | sed "s/task-//" | sort -n | tail -1)
m2=$(gh pr list -R rmusser01/tldw_chatbook --state open --limit 100 --json files --jq ".[].files[].path" | grep -oE "backlog/tasks/task-[0-9]+" | sed "s/.*task-//" | sort -n | tail -1)
echo "use ID: $(( (m1 > m2 ? m1 : m2) + 10 ))"'
```

- [ ] **Step 3: Write the task file.** Replace `<ID>` with the number from Step 2, and use today's date and time:

```markdown
---
id: TASK-<ID>
title: Build the in-repo merge queue for dev
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-03 21:00'
labels:
  - ci-throughput
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under strict protection only one PR can merge into dev per CI cycle, yet every armed PR is re-synced and re-tested after each merge: 47% of required runs over 80 merged PRs (2026-09-21..28) were re-sync churn, about 740 runner-minutes a day. The owner asked (2026-10-03) for CI that pushes one PR at a time. Spec: Docs/superpowers/specs/2026-10-03-merge-queue-design.md. Mechanics verified by spike PR #2985.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Armed PRs merge into dev one at a time in arming order; PRs behind the front are never rebased, dispatched or commented on
- [ ] #2 The front PR is rebased with the built-in token, and its required check and other PR workflows run on the rebased head
- [ ] #3 Conflicting, twice-failed, blocked or stuck PRs are evicted with a comment; a single CI failure is retried once
- [ ] #4 The queue never enables auto-merge, merges or pushes (guard test)
- [ ] #5 MERGE_QUEUE unset/off has no effect; dry logs decisions with no side effects; on acts
- [ ] #6 CLAUDE.md and AGENTS.md carry the same mode-dependent merge rules
- [ ] #7 Re-sync runs per merged PR are measurable before and after with a committed script
<!-- AC:END -->
```

- [ ] **Step 4: Run the V1/V2 spike.** In a separate throwaway worktree `/private/tmp/mq-v1v2`, on branch
  `spike/merge-queue-v1v2` from `origin/dev`, add this file:

`.github/workflows/spike-v1v2.yml`:

```yaml
name: SPIKE merge-queue V1 V2

# THROWAWAY (merge-queue plan Task 1). Never merged. Checks whether GITHUB_TOKEN can
# cancel (V2) and delete (V1) a workflow run.
on:
  pull_request:
    types: [opened]
    branches: [dev]

permissions:
  actions: write
  contents: read

jobs:
  probe:
    if: startsWith(github.head_ref, 'spike/merge-queue-v1v2')
    runs-on: ubuntu-latest
    timeout-minutes: 25
    env:
      GH_TOKEN: ${{ github.token }}
      REPO: ${{ github.repository }}
      HEAD: ${{ github.event.pull_request.head.sha }}
    steps:
      - name: V2 cancel this PR's derived-artifacts run
        run: |
          RID=""
          for i in $(seq 1 40); do
            RID=$(gh api "repos/$REPO/actions/workflows/derived-artifacts.yml/runs?head_sha=$HEAD" --jq '.workflow_runs[0].id // empty')
            [ -n "$RID" ] && break
            sleep 15
          done
          echo "derived run: $RID" | tee -a "$GITHUB_STEP_SUMMARY"
          if gh api -X POST "repos/$REPO/actions/runs/$RID/cancel"; then echo "V2 cancel: OK" | tee -a "$GITHUB_STEP_SUMMARY"; else echo "V2 cancel: REFUSED" | tee -a "$GITHUB_STEP_SUMMARY"; fi
          echo "RID=$RID" >> "$GITHUB_ENV"
      - name: V1 delete that run once completed
        run: |
          for i in $(seq 1 60); do
            S=$(gh api "repos/$REPO/actions/runs/$RID" --jq .status)
            [ "$S" = completed ] && break
            sleep 15
          done
          if gh api -X DELETE "repos/$REPO/actions/runs/$RID"; then echo "V1 delete: OK" | tee -a "$GITHUB_STEP_SUMMARY"; else echo "V1 delete: REFUSED" | tee -a "$GITHUB_STEP_SUMMARY"; fi
```

Commit it, push the branch, and open a **draft** PR titled `[SPIKE - DO NOT MERGE] merge-queue V1/V2`. Wait for the probe
run, then read its two result lines:

```bash
gh run list -R rmusser01/tldw_chatbook --branch spike/merge-queue-v1v2 --workflow spike-v1v2.yml --limit 1 --json databaseId --jq '.[0].databaseId' | xargs -I{} gh run view {} -R rmusser01/tldw_chatbook --log | grep -E 'V[12] (cancel|delete):'
```

Expected: two lines, each `OK` or `REFUSED`.

- [ ] **Step 5: Close the spike.** Close the PR without merging (`--delete-branch`) and remove `/private/tmp/mq-v1v2`.

- [ ] **Step 6: Record the results in the spec.** In section 4's V-table, change each V row's answer to `Verified (run <id>)`
  or `Refuted (run <id>) - fallback applies`.

- [ ] **Step 7: Commit**

```bash
bash -euo pipefail -c '
cd /private/tmp/mq-impl; [ "$(pwd)" = /private/tmp/mq-impl ]
git add backlog/tasks/task-*Build-the-in-repo-merge-queue-for-dev.md Docs/superpowers/specs/2026-10-03-merge-queue-design.md
git commit -q -m "chore(backlog): track the in-repo merge queue; record V1/V2 token results" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"'
```

---

### Task 2: Queue rules — `line_of` and `decide_front`

**Files:**
- Create: `scripts/merge_queue.py` (rules section only; Task 3 appends the rest)
- Test: `Tests/CI/test_merge_queue_rules.py`

**Interfaces:**
- Produces, used by Task 3:
  - `CheckRun(status: str, conclusion: str | None, completed_at: datetime | None, url: str)`
  - `PrState(number, node_id, head_sha, head_ref, same_repo, armed_at, is_draft, merge_state, head_committed_at, checks)`
  - `Action(kind: str, reason: str, links: tuple[str, ...] = ())`
  - `line_of(prs: list[PrState]) -> list[PrState]`
  - `decide_front(pr: PrState, now: datetime) -> Action`
  - constants `REPO`, `BASE`, `REQUIRED_CHECK`, `REQUIRED_WORKFLOW`, `QUEUE_WORKFLOW`

- [ ] **Step 1: Write the failing tests** in `Tests/CI/test_merge_queue_rules.py`:

```python
"""Table tests for the merge queue's pure rules (spec section 6)."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("merge_queue", ROOT / "scripts" / "merge_queue.py")
mq = importlib.util.module_from_spec(_spec)
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


def _check(conclusion="success", status="completed", minutes_ago=1, url="u") -> "mq.CheckRun":
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
        (_pr(merge_state="BLOCKED", checks=()), "dispatch"),
        (_pr(merge_state="BLOCKED", checks=(_check("cancelled"),)), "dispatch"),
        (_pr(merge_state="BLOCKED", checks=(), head_committed_at=NOW - timedelta(minutes=2)), "wait"),
        (_pr(merge_state="BLOCKED", checks=(_check("failure"),)), "retry"),
        (_pr(merge_state="BLOCKED", checks=(_check("failure", minutes_ago=30), _check("failure", minutes_ago=2))), "evict"),
        (_pr(merge_state="CLEAN", checks=(_check(minutes_ago=5),)), "wait"),
        (_pr(merge_state="UNSTABLE", checks=(_check(minutes_ago=5),)), "wait"),
        (_pr(merge_state="CLEAN", checks=(_check(minutes_ago=16),)), "evict"),
        (_pr(merge_state="BLOCKED", checks=(_check(minutes_ago=1),)), "evict"),
        (_pr(merge_state="CLEAN", checks=(_check("failure", minutes_ago=20), _check(minutes_ago=3))), "wait"),
    ],
    ids=[
        "unknown-waits", "behind-rebases", "dirty-evicts", "running-waits", "no-run-dispatches",
        "only-cancelled-dispatches", "young-head-waits", "first-failure-retries", "second-failure-evicts",
        "green-clean-waits", "green-unstable-waits", "stuck-green-evicts", "green-blocked-evicts",
        "retry-that-passed-waits",
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /private/tmp/mq-impl && $PY -m pytest -q -p no:cacheprovider Tests/CI/test_merge_queue_rules.py`

Expected: errors, because `scripts/merge_queue.py` does not exist (`FileNotFoundError`).

- [ ] **Step 3: Write the rules** in `scripts/merge_queue.py`:

```python
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

import os
from dataclasses import dataclass
from datetime import datetime, timedelta

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /private/tmp/mq-impl && $PY -m pytest -q -p no:cacheprovider Tests/CI/test_merge_queue_rules.py`

Expected: `18 passed`.

- [ ] **Step 5: Commit**

```bash
bash -euo pipefail -c '
cd /private/tmp/mq-impl; [ "$(pwd)" = /private/tmp/mq-impl ]
git add scripts/merge_queue.py Tests/CI/test_merge_queue_rules.py
git commit -q -m "feat(ci): merge-queue rules (line order and front-PR decision)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"'
```

---

### Task 3: Reads, actions, `run()` and `main()`

**Files:**
- Modify: `scripts/merge_queue.py` (append below the rules)
- Test: `Tests/CI/test_merge_queue_actions.py`

**Interfaces:**
- Consumes: everything Task 2 produces.
- Produces, used by Tasks 4 and 5 through `python3 scripts/merge_queue.py`:
  - `run(gh, mode, now=..., sleep=..., log=...) -> list[tuple[int, Action]]`
  - `main() -> int`
  - `Gh(runner=None)` with `.graphql(query, **vars) -> dict` and `.rest(method, path, fields=None) -> object`
  - `GhError`
- Task 1 results: if V1 was refuted, omit `cleanup_approval_runs`, its call in `run()`, and
  `test_cleanup_deletes_only_bot_approval_runs`. If V2 was refuted, omit the cancel loop in `_rebase` and the `cancel`
  assertion in `test_on_mode_rebases_front_only`.

- [ ] **Step 1: Write the failing tests** in `Tests/CI/test_merge_queue_actions.py`:

```python
"""The merge queue's action layer and run loop, against a fake gh (spec sections 6-8)."""

from __future__ import annotations

import importlib.util
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "merge_queue.py"
_spec = importlib.util.spec_from_file_location("merge_queue", SCRIPT)
mq = importlib.util.module_from_spec(_spec)
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /private/tmp/mq-impl && $PY -m pytest -q -p no:cacheprovider Tests/CI/test_merge_queue_actions.py`

Expected: failures with `AttributeError: module 'merge_queue' has no attribute 'run'`. The guard test passes already.

- [ ] **Step 3: Replace the import block and append the rest.** At the top of `scripts/merge_queue.py`, replace the four
  import lines from Task 2 (everything after `from __future__ import annotations`) with:

```python
import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol
from urllib.parse import quote
```

Then append the read layer, the action layer and the entry points to the end of the file:

```python
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
```

- [ ] **Step 4: Run both test files to verify they pass**

Run: `cd /private/tmp/mq-impl && $PY -m pytest -q -p no:cacheprovider Tests/CI/test_merge_queue_rules.py Tests/CI/test_merge_queue_actions.py`

Expected: `33 passed` (18 rules + 15 actions).

- [ ] **Step 5: Negative control.** With the Edit tool, temporarily change `if action.kind != "evict":` (in `run()`) to
  `if True:`, then run `test_evicted_front_hands_over_in_the_same_run`. Expected: FAIL. Change it back with the Edit tool.
  Never use `git checkout`: the Task 3 code is not committed yet, and checkout would wipe it. Re-run: PASS.

- [ ] **Step 6: Commit**

```bash
bash -euo pipefail -c '
cd /private/tmp/mq-impl; [ "$(pwd)" = /private/tmp/mq-impl ]
git add scripts/merge_queue.py Tests/CI/test_merge_queue_actions.py
git commit -q -m "feat(ci): merge-queue action layer and run loop (rebase, dispatch, retry, evict)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"'
```

---

### Task 4: `derived-artifacts.yml` dispatch path and the `queue-tick` job

**Files:**
- Modify: `.github/workflows/derived-artifacts.yml` (triggers; the `if:` of both lanes; both verdict steps; new job at the
  end)
- Modify: `Tests/CI/test_derived_artifacts_workflow.py` (4 pins, plus new tests)
- Modify: `Tests/CI/test_ci_queue_pressure_contract.py` (2 pins)

**Interfaces:**
- Consumes: `python3 scripts/merge_queue.py` (Task 3).
- Produces:
  - the dispatch input `pr`, which Task 3's `dispatch()` sends as `inputs[pr]`;
  - the lane-condition string `LANES` below, verbatim.

The shared lane condition, used verbatim everywhere:

```
github.event_name == 'pull_request' || (github.event_name == 'workflow_dispatch' && inputs.pr != '')
```

- [ ] **Step 1: Update the pins first** so they fail against the current workflow. In
  `Tests/CI/test_ci_queue_pressure_contract.py`:
  - At module level, after `PUSH_ONLY_CANCELLATION`, add:
    ```python
    LANES = "github.event_name == 'pull_request' || (github.event_name == 'workflow_dispatch' && inputs.pr != '')"
    ```
  - In `_assert_required_aggregation`, replace both verdict `if` expectations:
    ```python
        assert verdict["if"] == f"${{{{ ({LANES}) && needs.pr-fast-lane.result != 'success' }}}}"
    ```
    ```python
        assert ui_verdict["if"] == f"${{{{ ({LANES}) && needs.ui-fast-lane.result != 'success' }}}}"
    ```
  - In `test_fast_lane_is_one_serial_minimal_python_312_job`, replace the `fast["if"]` expectation:
    ```python
        assert fast["if"] == LANES
    ```

  In `Tests/CI/test_derived_artifacts_workflow.py`:
  - After `WORKFLOW_PATH`, add the same `LANES` constant.
  - In `test_workflow_has_one_fast_prerequisite_and_one_required_aggregator`, the expected job list becomes
    `["pr-fast-lane", "ui-fast-lane", "derived-artifacts", "queue-tick"]`. Add this docstring sentence: "`queue-tick` (the
    merge queue, spec 2026-10-03) runs after the aggregate and is never a required context."
  - In `test_required_aggregator_fails_when_either_lane_fails`:
    ```python
            assert verdict["if"] == f"${{{{ ({LANES}) && needs.{lane}.result != 'success' }}}}"
    ```
  - In `test_ui_fast_lane_runs_the_census_serially_on_the_minimal_dep_set`:
    ```python
        assert job["if"] == LANES
    ```
  - In `test_triggers_are_not_path_filtered`:
    ```python
        assert set(triggers) == {"pull_request", "push", "workflow_dispatch"}
    ```
  - Add the new tests at the end of `Tests/CI/test_derived_artifacts_workflow.py`:

```python
def test_dispatch_input_lets_the_queue_name_the_pr():
    """A queue dispatch names its PR; a manual kick names none, so the lanes skip."""
    dispatch = _workflow()[True]["workflow_dispatch"]
    assert dispatch["inputs"]["pr"] == {"description": "PR number (set by the merge queue)", "required": False,
                                        "type": "string", "default": ""}


def test_queue_tick_runs_after_ci_and_is_never_required():
    jobs = _workflow()["jobs"]
    tick = jobs["queue-tick"]
    assert tick["needs"] == ["derived-artifacts"]
    assert "queue-tick" not in jobs["derived-artifacts"].get("needs", [])
    assert tick["if"].startswith("always() &&")
    assert "vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on'" in tick["if"]
    assert "github.event.pull_request.auto_merge != null" in tick["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in tick["if"]
    assert "push" not in tick["if"], "pushes to dev are merge-queue.yml's job"
    assert tick["permissions"] == {"contents": "write", "pull-requests": "write", "actions": "write"}
    assert _workflow()["permissions"] == {"contents": "read"}
    checkout = tick["steps"][0]
    assert checkout["uses"] == "actions/checkout@v4" and checkout["with"] == {"ref": "dev"}
    assert tick["steps"][1]["run"] == "python3 scripts/merge_queue.py"
    assert tick["steps"][1]["env"] == {"GH_TOKEN": "${{ github.token }}", "MERGE_QUEUE": "${{ vars.MERGE_QUEUE }}"}
```

- [ ] **Step 2: Run the CI pin tests to verify they fail**

Run: `cd /private/tmp/mq-impl && $PY -m pytest -q -p no:cacheprovider Tests/CI/test_derived_artifacts_workflow.py Tests/CI/test_ci_queue_pressure_contract.py`

Expected: the 6 updated pins and the 2 new tests FAIL, because the workflow is unchanged.

- [ ] **Step 3: Edit `.github/workflows/derived-artifacts.yml`.**
  1. Triggers: after the `push:` block, add:
     ```yaml
       # Merge queue (Docs/superpowers/specs/2026-10-03-merge-queue-design.md): the queue
       # dispatches this workflow on a rebased PR head with `pr` set, so the required check
       # runs on exactly the head that merges. A dispatch without `pr` is a manual queue
       # kick: the lanes skip and only queue-tick runs.
       workflow_dispatch:
         inputs:
           pr:
             description: PR number (set by the merge queue)
             required: false
             type: string
             default: ''
     ```
  2. In both `pr-fast-lane` and `ui-fast-lane`, replace `if: github.event_name == 'pull_request'` with:
     ```yaml
         if: github.event_name == 'pull_request' || (github.event_name == 'workflow_dispatch' && inputs.pr != '')
     ```
  3. In both verdict steps, replace the `if:` with:
     ```yaml
             if: ${{ (github.event_name == 'pull_request' || (github.event_name == 'workflow_dispatch' && inputs.pr != '')) && needs.pr-fast-lane.result != 'success' }}
     ```
     and, for the UI step:
     ```yaml
             if: ${{ (github.event_name == 'pull_request' || (github.event_name == 'workflow_dispatch' && inputs.pr != '')) && needs.ui-fast-lane.result != 'success' }}
     ```
  4. Append the job at the end of `jobs:`:
     ```yaml
       queue-tick:
         # Wakes the merge queue after CI finishes, for the queue's own dispatched runs and for
         # armed same-repo PRs. NOT a required check and NOT in the aggregate's needs: a queue
         # failure can never turn `Derived artifacts reproduce from their sources` red. Pushes to
         # dev are merge-queue.yml's job. Spec: Docs/superpowers/specs/2026-10-03-merge-queue-design.md
         name: Merge queue tick
         needs: [derived-artifacts]
         if: >-
           always() &&
           (vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on') &&
           (github.event_name == 'workflow_dispatch' ||
            (github.event_name == 'pull_request' &&
             github.event.pull_request.auto_merge != null &&
             github.event.pull_request.head.repo.full_name == github.repository))
         runs-on: ubuntu-latest
         timeout-minutes: 10
         permissions:
           contents: write
           pull-requests: write
           actions: write
         steps:
           - uses: actions/checkout@v4
             with:
               ref: dev
           - name: Advance the queue
             env:
               GH_TOKEN: ${{ github.token }}
               MERGE_QUEUE: ${{ vars.MERGE_QUEUE }}
             run: python3 scripts/merge_queue.py
     ```

  The new job references `github.event.pull_request`, but its condition also admits `workflow_dispatch`, so Task 6's
  dispatch-safety test accepts it.

- [ ] **Step 4: Run the whole `Tests/CI` directory to verify everything passes**

Run: `cd /private/tmp/mq-impl && out=$($PY -m pytest -q -p no:cacheprovider Tests/CI 2>&1); rc=$?; echo "$out" | tail -3; echo rc=$rc`

Expected: `rc=0`. If another `Tests/CI` file fails on the new job or trigger, search it untruncated
(`grep -n "derived-artifacts" Tests/CI/*.py`), update the pin to the new shape, and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
bash -euo pipefail -c '
cd /private/tmp/mq-impl; [ "$(pwd)" = /private/tmp/mq-impl ]
git add .github/workflows/derived-artifacts.yml Tests/CI/test_derived_artifacts_workflow.py Tests/CI/test_ci_queue_pressure_contract.py
git commit -q -m "ci: dispatch path and non-required queue-tick job in derived-artifacts" -m "Updates the six lane/trigger/job-list pins to the dispatch-aware shape." -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"'
```

---

### Task 5: `merge-queue.yml`

**Files:**
- Create: `.github/workflows/merge-queue.yml`
- Test: `Tests/CI/test_merge_queue_workflow.py`

**Interfaces:**
- Consumes: `python3 scripts/merge_queue.py` (Task 3); `QUEUE_WORKFLOW = "merge-queue.yml"` (Task 2) excludes this file
  from re-dispatch.

- [ ] **Step 1: Write the failing test** in `Tests/CI/test_merge_queue_workflow.py`:

```python
"""Shape of the merge-queue entry point (spec sections 3, 5 and 9)."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "merge-queue.yml"


def _wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_triggers_are_dev_events_only():
    on = _wf()[True]
    assert set(on) == {"pull_request", "push"}
    assert on["pull_request"] == {"types": ["auto_merge_enabled", "auto_merge_disabled", "closed"], "branches": ["dev"]}
    assert on["push"] == {"branches": ["dev"]}


def test_never_reads_main():
    text = WORKFLOW.read_text(encoding="utf-8")
    for trigger in ("pull_request_target", "schedule", "workflow_run", "check_suite", "check_run"):
        assert f"{trigger}:" not in text, trigger


def test_write_permissions_are_job_level_only():
    wf = _wf()
    assert wf["permissions"] == {"contents": "read"}
    job = wf["jobs"]["queue"]
    assert job["permissions"] == {"contents": "write", "pull-requests": "write", "actions": "write"}


def test_runs_devs_script_only_when_enabled_and_never_for_forks():
    job = _wf()["jobs"]["queue"]
    assert "vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on'" in job["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in job["if"]
    assert job["steps"][0] == {"uses": "actions/checkout@v4", "with": {"ref": "dev"}}
    assert job["steps"][1]["run"] == "python3 scripts/merge_queue.py"
    assert job["steps"][1]["env"] == {"GH_TOKEN": "${{ github.token }}", "MERGE_QUEUE": "${{ vars.MERGE_QUEUE }}"}
    assert "concurrency" not in _wf() and "concurrency" not in job, "races are made safe in the script (spec 7)"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /private/tmp/mq-impl && $PY -m pytest -q -p no:cacheprovider Tests/CI/test_merge_queue_workflow.py`

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Create `.github/workflows/merge-queue.yml`:**

```yaml
name: Merge Queue

# One-at-a-time merge queue for dev.
# Spec: Docs/superpowers/specs/2026-10-03-merge-queue-design.md
#
# Wakes on a PR being armed, disarmed or closed, and on pushes to dev. derived-artifacts.yml's
# queue-tick job wakes it after CI. It uses no schedule, workflow_run or pull_request_target
# trigger, because those read main's copy and nothing here may depend on main. No concurrency
# group: racing runs are made safe in scripts/merge_queue.py (pinned-head rebase, idempotent
# evictions).
#
# Kill switch: repository variable MERGE_QUEUE. Unset/off = nothing; dry = decisions logged
# to the job summary only; on = act.
on:
  pull_request:
    types: [auto_merge_enabled, auto_merge_disabled, closed]
    branches: [dev]
  push:
    branches: [dev]

permissions:
  contents: read

jobs:
  queue:
    name: Merge queue
    if: >-
      (vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on') &&
      (github.event_name == 'push' ||
       github.event.pull_request.head.repo.full_name == github.repository)
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      contents: write
      pull-requests: write
      actions: write
    steps:
      - uses: actions/checkout@v4
        with:
          ref: dev
      - name: Advance the queue
        env:
          GH_TOKEN: ${{ github.token }}
          MERGE_QUEUE: ${{ vars.MERGE_QUEUE }}
        run: python3 scripts/merge_queue.py
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd /private/tmp/mq-impl && $PY -m pytest -q -p no:cacheprovider Tests/CI/test_merge_queue_workflow.py`

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
bash -euo pipefail -c '
cd /private/tmp/mq-impl; [ "$(pwd)" = /private/tmp/mq-impl ]
git add .github/workflows/merge-queue.yml Tests/CI/test_merge_queue_workflow.py
git commit -q -m "ci: merge-queue.yml entry point (arm/disarm/close, pushes to dev)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"'
```

---

### Task 6: Make every PR workflow dispatch-safe

**Files:**
- Modify: `.github/workflows/perf-guard.yml`, `.github/workflows/task-19642-smoke-clock-matrix.yml`,
  `.github/workflows/task-32011-linux-storage-evidence.yml`
- Test: `Tests/CI/test_pr_workflows_dispatch_safe.py`

**Interfaces:**
- Consumes: Task 3's `_workflows_to_redispatch`, which dispatches any workflow that ran on the old head.
- Audit result (2026-10-03, untruncated): these 8 are already safe — task-19637, task-2062-1, task-2062-2, task-598,
  task-601, task-602, task-603 and voice-aec-wheels. Each has `workflow_dispatch`, and its PR checkout falls back to
  `github.sha`. Nothing needs an owner decision.

- [ ] **Step 1: Write the failing test** in `Tests/CI/test_pr_workflows_dispatch_safe.py`:

```python
"""Every pull_request workflow must also run correctly from workflow_dispatch.

The merge queue re-runs, on the rebased head, every workflow that ran on the old head.
A rebase made with GITHUB_TOKEN gets only approval-pending pull_request runs (spec F4),
so dispatch is the only way they run again (spec section 5.4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
# The queue's own entry point reacts to PR events but is never re-dispatched (it is not on
# main, so it cannot be dispatched, and scripts/merge_queue.py excludes it by design).
QUEUE = "merge-queue.yml"


def _pr_workflows():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        if path.name == QUEUE:
            continue
        wf = yaml.safe_load(path.read_text(encoding="utf-8"))
        on = wf.get(True, wf.get("on"))
        if isinstance(on, dict) and "pull_request" in on:
            yield path, wf, on


def test_there_are_pull_request_workflows():
    assert len(list(_pr_workflows())) >= 10


def test_every_pull_request_workflow_is_dispatchable():
    missing = [p.name for p, wf, on in _pr_workflows() if "workflow_dispatch" not in on]
    assert missing == []


def test_pr_context_job_guards_also_admit_dispatch():
    bad = []
    for path, wf, on in _pr_workflows():
        for job_id, job in wf["jobs"].items():
            cond = str(job.get("if", ""))
            if ("github.event.pull_request" in cond or "github.event.label" in cond) and "workflow_dispatch" not in cond:
                bad.append(f"{path.name}:{job_id}")
    assert bad == []


def test_pr_head_checkouts_fall_back_to_github_sha():
    bad = []
    for path, wf, on in _pr_workflows():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "github.event.pull_request.head.sha" in line and "github.sha" not in line:
                bad.append(f"{path.name}: {line.strip()}")
    assert bad == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /private/tmp/mq-impl && $PY -m pytest -q -p no:cacheprovider Tests/CI/test_pr_workflows_dispatch_safe.py`

Expected:
- `test_every_pull_request_workflow_is_dispatchable` fails, listing `perf-guard.yml`, `task-19642-smoke-clock-matrix.yml`
  and `task-32011-linux-storage-evidence.yml`;
- the guard test fails on `task-32011-linux-storage-evidence.yml:linux-storage`;
- the checkout test fails on task-32011's `ref:` line.

- [ ] **Step 3: Edit the three workflows.**
  - `perf-guard.yml` and `task-19642-smoke-clock-matrix.yml`: add `workflow_dispatch:` as the last key under `on:`, with
    this comment above it:
    ```yaml
      # Merge queue re-runs this on the rebased head (spec 2026-10-03, section 5.4).
      workflow_dispatch:
    ```
  - `task-32011-linux-storage-evidence.yml`: add the same `workflow_dispatch:` trigger, then replace the job `if:` with:
    ```yaml
        if: >-
          github.event_name == 'workflow_dispatch' ||
          (contains(github.event.pull_request.labels.*.name, 'task-32011-linux-evidence') &&
          (github.event.action == 'synchronize' || github.event.label.name == 'task-32011-linux-evidence'))
    ```
    and replace the checkout `ref:` with:
    ```yaml
              ref: ${{ github.event.pull_request.head.sha || github.sha }}
    ```

- [ ] **Step 4: Run the whole `Tests/CI` directory to verify everything passes**

Run: `cd /private/tmp/mq-impl && out=$($PY -m pytest -q -p no:cacheprovider Tests/CI 2>&1); rc=$?; echo "$out" | tail -3; echo rc=$rc`

Expected: `rc=0`. `test_focused_guards_keep_dev_pr_and_dev_main_push_coverage` still passes, because perf-guard's PR
types and branches are unchanged.

- [ ] **Step 5: Commit**

```bash
bash -euo pipefail -c '
cd /private/tmp/mq-impl; [ "$(pwd)" = /private/tmp/mq-impl ]
git add .github/workflows/perf-guard.yml .github/workflows/task-19642-smoke-clock-matrix.yml .github/workflows/task-32011-linux-storage-evidence.yml Tests/CI/test_pr_workflows_dispatch_safe.py
git commit -q -m "ci: make every pull_request workflow dispatch-safe for the merge queue" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"'
```

---

### Task 7: Success-measure script

**Files:**
- Create: `scripts/measure_required_runs_per_merge.py`
- Test: `Tests/CI/test_measure_required_runs_per_merge.py`

**Interfaces:**
- Produces: `classify(run: dict, sync_oids: set[str], rebase_oids: set[str]) -> str | None` and
  `summarize(per_pr: list[dict[str, int]]) -> dict[str, float]`, plus the CLI `--prs N --since DATE`.

- [ ] **Step 1: Write the failing test** in `Tests/CI/test_measure_required_runs_per_merge.py`:

```python
"""Classifier and summary of the merge-queue success measure (spec section 13)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("measure", ROOT / "scripts" / "measure_required_runs_per_merge.py")
m = importlib.util.module_from_spec(_spec)
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /private/tmp/mq-impl && $PY -m pytest -q -p no:cacheprovider Tests/CI/test_measure_required_runs_per_merge.py`

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Create `scripts/measure_required_runs_per_merge.py`:**

```python
#!/usr/bin/env python3
"""Required-workflow runs per merged PR into dev, split by cause.

The merge-queue success measure (Docs/superpowers/specs/2026-10-03-merge-queue-design.md,
section 13): with the queue on, sync + rebase + queue runs per merged PR should have a
median of at most 1. Baseline (2026-09-21..28, 80 PRs): 186 re-sync runs beyond one per PR.

Cause of each derived-artifacts.yml run on the PR branch, between PR creation and merge:
  queue   - event workflow_dispatch (the merge queue's dispatches)
  sync    - its head is a merge commit that brought dev in
  rebase  - its head came from a force-push (a manual or queue rebase)
  content - anything else
Runs that concluded action_required never ran and are excluded.

Read-only. Needs `gh` authenticated for the repository.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from urllib.parse import quote

REPO = "rmusser01/tldw_chatbook"
TIMELINE = """query($n: Int!) { repository(owner: "rmusser01", name: "tldw_chatbook") { pullRequest(number: $n) {
  timelineItems(first: 250, itemTypes: [HEAD_REF_FORCE_PUSHED_EVENT, PULL_REQUEST_COMMIT]) { nodes { __typename
    ... on HeadRefForcePushedEvent { afterCommit { oid } }
    ... on PullRequestCommit { commit { oid messageHeadline parents { totalCount } } } } } } } }"""


def classify(run: dict, sync_oids: set[str], rebase_oids: set[str]) -> str | None:
    """Return the cause of one required-workflow run, or None if it never ran."""
    if run.get("conclusion") == "action_required":
        return None
    if run.get("event") == "workflow_dispatch":
        return "queue"
    if run["head_sha"] in sync_oids:
        return "sync"
    if run["head_sha"] in rebase_oids:
        return "rebase"
    return "content"


def summarize(per_pr: list[dict[str, int]]) -> dict[str, float]:
    """Totals plus the re-sync median and the re-sync runs beyond one per PR."""
    resync = [p.get("sync", 0) + p.get("rebase", 0) + p.get("queue", 0) for p in per_pr]
    return {
        "prs": len(per_pr),
        "runs": sum(sum(p.values()) for p in per_pr),
        "resync_median": statistics.median(resync) if resync else 0,
        "resync_beyond_one": sum(max(0, r - 1) for r in resync),
    }


def _gh(*args: str) -> object:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out) if out.strip() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Required-workflow runs per merged PR into dev, by cause.")
    parser.add_argument("--prs", type=int, default=80, help="how many recent merged PRs (default 80)")
    parser.add_argument("--since", help="only PRs merged at or after this ISO date, e.g. 2026-10-05")
    args = parser.parse_args(argv)
    if args.prs < 1:
        parser.error("--prs must be at least 1")
    prs = _gh("pr", "list", "-R", REPO, "--state", "merged", "--base", "dev", "--limit", str(args.prs),
              "--json", "number,headRefName,createdAt,mergedAt")
    if args.since:
        prs = [p for p in prs if p["mergedAt"] >= args.since]
    per_pr = []
    for pr in prs:
        items = _gh("api", "graphql", "-f", f"query={TIMELINE}", "-F", f"n={pr['number']}")
        nodes = items["data"]["repository"]["pullRequest"]["timelineItems"]["nodes"]
        sync = {
            n["commit"]["oid"] for n in nodes
            if n["__typename"] == "PullRequestCommit"
            and n["commit"]["parents"]["totalCount"] > 1 and "dev" in n["commit"]["messageHeadline"]
        }
        rebase = {n["afterCommit"]["oid"] for n in nodes if n["__typename"] == "HeadRefForcePushedEvent" and n.get("afterCommit")}
        runs = _gh("api", f"repos/{REPO}/actions/workflows/derived-artifacts.yml/runs"
                          f"?branch={quote(pr['headRefName'], safe='')}&per_page=100")["workflow_runs"]
        counts: dict[str, int] = {}
        for run in runs:
            if not pr["createdAt"] <= run["created_at"] <= pr["mergedAt"]:
                continue
            cause = classify(run, sync, rebase)
            if cause:
                counts[cause] = counts.get(cause, 0) + 1
        per_pr.append(counts)
    s = summarize(per_pr)
    print(f"merged PRs: {s['prs']}  required runs: {s['runs']}")
    print(f"re-sync runs per PR (sync+rebase+queue): median {s['resync_median']}, beyond one: {s['resync_beyond_one']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd /private/tmp/mq-impl && $PY -m pytest -q -p no:cacheprovider Tests/CI/test_measure_required_runs_per_merge.py`

Expected: `3 passed`.

- [ ] **Step 5: Capture the pre-queue baseline.** This makes live API calls, read-only:

```bash
cd /private/tmp/mq-impl && $PY scripts/measure_required_runs_per_merge.py --prs 80 | tee /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/70ad6d88-1faa-4139-95bd-02684062ad14/scratchpad/mq-baseline.txt
```

Copy the two output lines into the task's Implementation Notes in Task 9, because the scratchpad does not survive a
session restart.

- [ ] **Step 6: Commit**

```bash
bash -euo pipefail -c '
cd /private/tmp/mq-impl; [ "$(pwd)" = /private/tmp/mq-impl ]
git add scripts/measure_required_runs_per_merge.py Tests/CI/test_measure_required_runs_per_merge.py
git commit -q -m "feat(ci): measure required runs per merged PR by cause (merge-queue success measure)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"'
```

---

### Task 8: Agent rules, ADR and baseline doc

**Files:**
- Modify: `CLAUDE.md` (the "Merging into `dev`" block, currently starting with `- Re-sync a PR that is behind by
  **rebasing** it`)
- Modify: `AGENTS.md` (new `### Merging into \`dev\`` section after `### Testing`, before `## Special Systems`)
- Create: `backlog/decisions/<NNN>-in-repo-merge-queue.md`
- Modify: `backlog/decisions/103-*.md` (one cross-reference line)
- Modify: `backlog/docs/branch-protection-baseline.md` (new dated section at the top)

**Interfaces:**
- Consumes: the mode values and the forbidden-action rule (Global Constraints).

- [ ] **Step 1: Replace the CLAUDE.md bullets.** Keep the three protection bullets and "These apply to admins too."
  unchanged. Replace everything from `- Re-sync a PR that is behind by **rebasing** it` through `pushing any further work
  to that PR.` with:

```markdown
The merge queue (`.github/workflows/merge-queue.yml`, spec
`Docs/superpowers/specs/2026-10-03-merge-queue-design.md`) runs in GitHub Actions. It works the same from any machine,
session or tool. Check its mode with `gh variable get MERGE_QUEUE`.

**Queue `on`:**
- Arm auto-merge (`gh pr merge <n> --auto --merge`) only when both of these hold:
  - Qodo has posted its review on the *current* head;
  - every thread on that head is resolved.

  Then walk away. The queue rebases the PR when it reaches the front, starts its CI, and lets auto-merge land it.
- Never `gh pr update-branch` an armed PR, and never merge an armed PR by hand. Either one makes the front PR's CI run go
  to waste.
- Before pushing more work to an armed PR, run `gh pr merge <n> --disable-auto`. Then `git pull --rebase`, because the
  queue may have rebased your branch, and push with `--force-with-lease`. Never use a plain force push.
- A conflict-free queue rebase needs no fresh Qodo review, because CI tests the combined result. New Qodo threads on the
  rebased head block the merge, and the queue evicts the PR with the reason.
- Never click "Approve and run" on a queue-rebased PR. Those runs are the token rebase's empty duplicates.
- An evicted PR has auto-merge off and a comment saying why. Fix the cause and re-arm; it rejoins at the back.

**Queue `off` or `dry`:**
- Re-sync a PR that is behind by **rebasing** it: `gh pr update-branch --rebase <n>`, or a local `git rebase origin/dev`
  pushed with `--force-with-lease`. Never merge `dev` into the branch (plain `gh pr update-branch` does exactly that).
- Re-sync only the PR you are about to merge. Under strict protection, one PR merges per CI cycle.
- Arm auto-merge only under the same Qodo and threads rule as above, and run `gh pr merge <n> --disable-auto` before
  pushing more work.
```

- [ ] **Step 2: Add the same rules to AGENTS.md.** Insert a `### Merging into \`dev\`` heading after the `### Testing`
  section, then the three protection bullets from CLAUDE.md, then the block from Step 1, word for word.

- [ ] **Step 3: Create the ADR.** First recompute the number, since ADR numbers churn across peer PRs:

```bash
bash -euo pipefail -c '
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git fetch -q origin
n=$(git for-each-ref --format="%(refname)" refs/remotes/origin | while read r; do git ls-tree --name-only "${r}" backlog/decisions/ 2>/dev/null; done | sed -E "s#backlog/decisions/([0-9]+).*#\1#" | grep -E "^[0-9]+$" | sort -n | tail -1)
echo "use ADR $((n + 1))"'
```

  Write `backlog/decisions/<NNN>-in-repo-merge-queue.md`:

```markdown
# ADR-<NNN>: In-repo merge queue for dev

Status: Accepted — owner decisions 2026-10-03 (build in-repo, not an org transfer; spec and plan approved)
Date: 2026-10-03

## Context

`dev` is strict-protected with one required check. Only one PR can merge per CI cycle, yet every armed PR was re-synced and
re-tested after each merge. Over 80 merged PRs, 47% of required runs were re-sync churn, about 740 runner-minutes a day.
GitHub's native merge queue needs an organization-owned repository, and this one is user-owned.

## Decision

- A queue workflow in this repository, `merge-queue.yml` plus the `queue-tick` job in `derived-artifacts.yml`, runs
  `scripts/merge_queue.py` on the built-in token.
- It rebases only the front armed PR, dispatches its CI, and lets auto-merge land it.
- It evicts PRs that conflict, fail twice, stay blocked, or get stuck green.
- It never arms, merges or pushes.
- The mode is set by the `MERGE_QUEUE` variable (off, dry or on).

## Consequences

- One CI cycle per merge, without the churn. The ceiling of one merge per cycle is unchanged.
- Nothing depends on `main`. A stall with no activity waits for the next event, or a manual
  `gh workflow run derived-artifacts.yml --ref dev`.
- Every pull_request workflow must stay dispatch-safe (`Tests/CI/test_pr_workflows_dispatch_safe.py`).
- Fork PRs stay manual.

Spec: `Docs/superpowers/specs/2026-10-03-merge-queue-design.md`. Plan: `Docs/superpowers/plans/2026-10-03-merge-queue.md`.
Extends: ADR-103.
```

- [ ] **Step 4: Cross-reference from ADR-103.** Append to the end of `backlog/decisions/103-*.md`:

```markdown

2026-10-03: the required check is now also started by the merge queue's `workflow_dispatch` (input `pr`); see ADR-<NNN>.
```

- [ ] **Step 5: Add a section to `backlog/docs/branch-protection-baseline.md`**, directly under the title:

```markdown
## 2026-10-03: in-repo merge queue (ADR-<NNN>)

Protection is unchanged: strict, required conversation resolution, enforce_admins, and one required check. The queue
works within these settings. It rebases the front armed PR with `GITHUB_TOKEN`, dispatches `derived-artifacts.yml`
(`pr=<n>`), and leaves the merge to auto-merge. Its mode is the repository variable `MERGE_QUEUE` (off, dry or on), and
setting it needs an admin. Agent rules are in `CLAUDE.md` and `AGENTS.md`, under "Merging into `dev`".
```

- [ ] **Step 6: Check the docs render and the IDs are unique**

Run:

```bash
cd /private/tmp/mq-impl && $PY scripts/check_backlog_task_ids.py && grep -c "MERGE_QUEUE" CLAUDE.md AGENTS.md
```

Expected: the ID check passes, and both files report a count of at least 2.

- [ ] **Step 7: Commit**

```bash
bash -euo pipefail -c '
cd /private/tmp/mq-impl; [ "$(pwd)" = /private/tmp/mq-impl ]
git add CLAUDE.md AGENTS.md backlog/decisions backlog/docs/branch-protection-baseline.md
git commit -q -m "docs: merge-queue agent rules (CLAUDE.md + AGENTS.md), ADR, baseline doc" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"'
```

---

### Task 9: Verify, open the PR dark, and roll out

**Files:**
- Modify: the task file from Task 1 (Implementation Notes, ACs, status)

- [ ] **Step 1: Full `Tests/CI` run and preflight**

```bash
cd /private/tmp/mq-impl && out=$($PY -m pytest -q -p no:cacheprovider Tests/CI 2>&1); rc=$?; echo "$out" | tail -3; echo pytest_rc=$rc
cd /private/tmp/mq-impl && out=$(PYTHON=$PY ./scripts/preflight.sh 2>&1); rc=$?; echo "$out" | tail -4; echo preflight_rc=$rc
```

Expected: both exit codes are 0. If preflight's diagnostic inventory reports drift, read the named rows before running
`--write`. Only rows for `scripts/merge_queue.py` or `scripts/measure_required_runs_per_merge.py` are expected.

- [ ] **Step 2: Fill in the task's Implementation Notes.** Include:
  - the approach;
  - the V1/V2 results;
  - the baseline lines from Task 7 Step 5;
  - the files changed;
  - the dispatch-safety audit result.

  Tick ACs #2-#7. AC #1 is ticked only after the live trial (Step 6). Keep the status at `In Progress`.

- [ ] **Step 3: Push and open the PR.** Confirm first that `MERGE_QUEUE` is unset or `off`
  (`gh variable get MERGE_QUEUE -R rmusser01/tldw_chatbook` returns an error or `off`). The PR body must state that the
  queue ships dark and that the owner flips the variable.

```bash
bash -euo pipefail -c '
cd /private/tmp/mq-impl; [ "$(pwd)" = /private/tmp/mq-impl ]
git push -q -u origin feat/merge-queue
gh pr create -R rmusser01/tldw_chatbook --base dev --head feat/merge-queue --title "ci: in-repo merge queue for dev (ships dark)" --body-file /dev/stdin <<EOF
Implements Docs/superpowers/specs/2026-10-03-merge-queue-design.md (plan: Docs/superpowers/plans/2026-10-03-merge-queue.md), TASK-<ID>.

Ships **dark**: MERGE_QUEUE is unset, so the queue does nothing until the owner runs \`gh variable set MERGE_QUEUE --body dry\` (decisions logged only), then \`on\`.

What it does, how it was verified (spike #2985 + Task 1), and the dispatch-safety audit are in the task notes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF'
```

- [ ] **Step 4: Merge the PR the old way.** This PR changes the workflows the queue relies on, so it cannot go through the
  queue. Rebase-sync with `gh pr update-branch --rebase`, address every Qodo comment, and merge once the required check is
  green and all threads are resolved.

- [ ] **Step 5: Dry run (owner action).** Ask the owner to run `gh variable set MERGE_QUEUE --body dry -R
  rmusser01/tldw_chatbook`. For about a day, compare each `Merge queue` job summary with what actually happened to the
  armed PRs. Every proposed action must match the spec's table. Record any mismatch as a fix before Step 6.

- [ ] **Step 6: Go live (owner action).** Ask the owner to run `gh variable set MERGE_QUEUE --body on`. Arm 2-3 low-risk
  PRs first, and confirm:
  - one is rebased at a time;
  - CI runs on the rebased head;
  - auto-merge lands each PR;
  - the next PR starts after the merge.

  Then tick AC #1.

- [ ] **Step 7: Measure after 7 days live**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && $PY scripts/measure_required_runs_per_merge.py --prs 80 --since <go-live date>
```

Expected: re-sync median of at most 1, plus the other spec section 13 measures. Record the results in the task notes, then
mark the task Done with `backlog task edit <ID> -s Done`, following the repo's Definition of Done.

- [ ] **Step 8: Resume sub-project 3** (the nightly), as the spec's non-goals note.
