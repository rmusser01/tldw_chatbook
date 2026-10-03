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
