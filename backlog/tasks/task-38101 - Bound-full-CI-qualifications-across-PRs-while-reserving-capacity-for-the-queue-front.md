---
id: TASK-38101
title: Bound full CI qualifications across PRs while reserving capacity for the queue
  front
status: In Progress
created_date: 2026-10-30 15:20
assignee:
- '@codex'
labels:
- ci
- testing
priority: high
updated_date: 2026-10-30 15:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce competition among full PR qualifications while preserving automatic complete regression feedback and giving the live merge-queue front a separate qualification slot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Heavy qualifications use one queue-front slot and one background slot across PRs, with useful in-progress evidence preserved and queue overflow never treated as successful qualification.
- [x] #2 Preflight continues on every existing trigger; all existing application test targets, dependencies, timeouts and 28 UI shards remain, with UI concurrency 17 for the front and 8 for background.
- [x] #3 Only a verified live eligible queue-front head receives the front class; uncertainty uses background, and scheduling classification cannot change merge or approval authority.
- [ ] #4 The stable required aggregate rejects failed, skipped, cancelled or missing qualification/lane results; existing failed-job-only retries and queue control remain intact.
- [x] #5 Focused behavior, graph, selector and job-name compatibility tests pass, ADR103/218 and documentation explain bounds and rollout, and hosted scheduling evidence is distinguished from local contracts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md and backlog/decisions/218-in-repo-merge-queue.md. Reason: cross-PR qualification admission and read-only front classification. Follow Docs/superpowers/plans/2026-10-30-ci-bounded-qualifications.md: test classifier and raw admission graph before implementation; move all unchanged lanes into one queued reusable workflow; preserve explicit lane and wrapper verdicts; update actual-source guards and capacity statements; compare conserved selectors/dependencies; run focused tests and independent review; keep hosted scheduling/retry evidence separate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented one native queued reusable-workflow call per front/background class, preserving all seven lane definitions and 28 UI shards with UI concurrency 17/8. Preflight classifies the live queue with bounded read-only pagination and candidate reread; malformed/uncertain metadata falls back to background. The stable parent gate rejects unsuccessful wrapper and individual lane outputs; source guards read the actual caller/callee, manual dispatch inputs and replay names are preserved. ADR103/218 apply. Independent review P2 timestamp/empty-metadata finding fixed with eight RED/GREEN probes; no other actionable source finding. Verification: 806 focused tests passed,28 unchanged exhaustive shard cases deselected;150 admission/aggregation tests passed after corrections; Ruff and diff checks pass; all seven parsed jobs conserved; live read-only GraphQL probe passed. Actionlint1.7.12 has only documented queue/result schema gaps (QA details). Hosted slot/retry/output/overflow qualification remains pending: keep In Progress with AC1/4 open. Evidence: Docs/superpowers/qa/2026-10-30-ci-bounded-qualifications.md. Updated workflow/contract/capacity docs and recorded the output-key/job-header parser incident in lessons-testing-evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
