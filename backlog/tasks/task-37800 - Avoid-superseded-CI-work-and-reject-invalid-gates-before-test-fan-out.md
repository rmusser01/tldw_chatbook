---
id: TASK-37800
title: Avoid superseded CI work and reject invalid gates before test fan-out
status: In Progress
created_date: 2026-10-30 05:40
assignee:
- '@codex'
labels:
- ci
- testing
priority: high
references:
- Docs/superpowers/specs/2026-10-29-ci-preflight-and-superseded-runs.md
- backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
- backlog/decisions/218-in-repo-merge-queue.md
updated_date: 2026-11-01 12:32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce avoidable runner contention observed with 17 armed PRs and hundreds of queued jobs, while preserving every required test and merge protection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Only active Derived Artifacts pull_request runs for the same open same-repository dev PR whose head SHA is obsolete may be cancelled; current-head, unrelated, fork, completed and manual runs are preserved, including when the head changes during cleanup.
- [x] #2 Existing artifact guards and PR test-target validation run before the expensive test lanes, and failure, cancellation or unexpected skipping cannot produce a green required check.
- [x] #3 All existing test targets, shard limits, dependency sets, required check name, merge-queue retry policy and branch-protection behavior remain intact.
- [x] #4 Targeted behavioral and workflow-contract tests, formatting and diff checks pass; ADR-103 and ADR-218 record the new policy and local versus hosted evidence is explicit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend existing decisions). ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md and backlog/decisions/218-in-repo-merge-queue.md. Reason: CI ordering and narrowly scoped cancellation authority. Follow Docs/superpowers/plans/2026-10-29-ci-preflight-and-superseded-runs.md: test the actual cleanup script before implementing its isolated synchronize job; add failing preflight graph and aggregate contracts; move all artifact guards and static selector validation ahead of unchanged lanes; run targeted checks and independent review; record local and hosted evidence separately.
2026-11-01 backlog integration follows Docs/superpowers/plans/2026-11-01-pr3316-backlog-integration.md. Retain current-dev 32 UI shards, every existing lane, selector, install, timeout, UAT step and guard. Rebase after PR3315 merges, run targeted CI contracts and artifact guards, obtain requesting-code-review on the final head, then merge under the explicit hosted-CI waiver. Live PR is non-draft; historical draft delivery note is superseded by current user scope. Hosted cancellation effectiveness and throughput remain unqualified. ADR required: no new decision beyond existing ADR103/ADR218 source amendments.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added checkout-free superseded-run cleanup to merge-queue.yml and moved all 13 artifact guards plus both existing selector checks into a preflight dependency in derived-artifacts.yml. The stable required aggregate now reports verdicts only. Application selectors, dependency sets, 28 UI shards, max-parallel 17, timeouts and queue retry policy remain unchanged.

The cleanup uses live PR identity/head checks, exact workflow/run metadata and a final live read before cancellation. It preserves same-head runs, forks, manual/completed/unrelated runs and runs whose required verdict or queue tick has appeared. Independent review identified the controller interruption risk; eight reproducing cases failed before the fix and all 46 cleanup tests now pass.

Validation: 631 targeted CI/queue tests passed, 28 unchanged exhaustive shard-distribution cases deselected. All 13 real artifact guards and both selector checks passed. The existing Mermaid checker passed with Python 3.12.11 against an isolated export with core.autocrlf=false and verified Git blob bytes; normal Windows CRLF conversion caused the earlier local mismatch. Other guards used local Python 3.12.10; hosted preflight retains 3.12.11. Scoped Ruff checks on changed test files, formatting of the new test, and git diff --check passed. Parsed comparison confirmed all seven application jobs differ only by needs and all 13 moved guards are identical.

ADR required: yes; amended backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md and backlog/decisions/218-in-repo-merge-queue.md. Spec, plan, five CI test files and lessons-testing-evidence.md record contracts and evidence.

Hosted qualification remains outstanding; task stays In Progress. No full suite, actual cancellation, throughput measurement, merge, auto-merge or branch-protection change was performed. The read/cancel API is not atomic; the immediate live read narrows races without claiming an absolute guarantee. See Docs/superpowers/plans/2026-10-29-ci-preflight-and-superseded-runs.md for the exact targeted command and review ledger.
2026-11-01 provisional integration onto PR3315 head5d7cad1c: retained32 shards and all7 application lanes. CI contracts:641 passed/2 failed, then642 passed/1 failed; failures reproduce on base and are corrected by expanding stale fixture-derived selectors to full functions and retaining the terminal-input target. Final affected contracts64 passed; all306 selectors resolve. All13 artifact guards now pass with verified complete LF Mermaid export and a two-line literal Kanban notification repair. Mounted Kanban regressions failed before the fix and both pass after it; census unchanged. See Docs/superpowers/qa/2026-11-01-pr3316-backlog-integration.md. Final confirmed-dev rebase, independent review and publication remain pending. Hosted throughput/cancellation remain unqualified under the user CI waiver. No draft, release or main actions.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
