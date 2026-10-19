---
id: TASK-35615
title: Reduce Console attachment completion latency
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-19 21:06'
updated_date: '2026-10-19 21:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate the measured return attachment regression, remove verified redundant work, and retain responsive typing and all storage authority checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Completed attachment improves in serial native Windows profiles without losing responsive typing or drafts.
- [x] #2 Storage, source, visit, maintenance and physical retirement checks remain intact with targeted regression coverage.
- [ ] #3 Document remaining hotspots and verification results, review the changes and update PR 3166.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Existing contracts: [ADR-171](../decisions/171-console-conversation-review-and-attention.md), [ADR-126](../decisions/126-complete-local-backup-and-recovery.md), [ADR-230](../decisions/230-console-send-preparation-and-io-ownership.md).
ADR required: no unless investigation changes an authority or lifecycle contract. ADR path: N/A; existing ADR-126/230 apply. Reason: expected scheduling or redundant presentation correction within existing boundaries. 1. Trace attachment phases against current head 3742efae94 and identify a reproducible root cause. 2. Add a failing targeted regression, apply the smallest correction and verify lifecycle/safety neighbors. 3. Run serial native before/after profiles measuring input and completed attachment independently. 4. Self-review, document evidence and limitations, update PR and clean exact owned temporary profiles.

Investigation identified `console-manual-read-return` requesting FULL even when its captured visit is absent or has no unread token. Add absence/refusal/no-token and real-token controls in `Tests/UI/test_console_manual_unread.py`, then return before acknowledgement-only FULL in `ConsoleSessionController.acknowledge_explicit_console_return`. Preserve ordinary attachment FULL and the original marked-chat path, including fresh token reads, painted-session and generation/profile checks. ADR required: no. ADR path: N/A; existing ADR-171 plus ADR-126/230 apply. Reason: removing work for an already-ineligible acknowledgement changes no storage, reminder or attachment contract. Timer and interpreter-handoff diagnostic experiments worsened total latency and will not ship. Profile the small correction against exact head 3742efae9428dc10f879606b519215913263ac4e with serial ABBA runs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The implementation preserves [ADR-171 reminder semantics](../decisions/171-console-conversation-review-and-attention.md) and [ADR-230 finite native custody](../decisions/230-console-send-preparation-and-io-ownership.md), with [ADR-126 storage admission](../decisions/126-complete-local-backup-and-recovery.md) unchanged.
Four-line guard removes acknowledgement-only FULL when the original captured reminder visit/token is absent, preserving ordinary attachment and marked-chat paint/token checks (existing ADR-171/126/230; no new ADR). Original regression 3 fail/2 pass; 57 connected tests plus 10 mounted real-SQLite reminder tests pass. Nine safety sources and seven reminder methods unchanged; no added lint findings and independent review found no issues. Serial native ABBA: 24 successful returns, median attachment 1149.08 to 979.41 ms (14.8% lower), input 191.33 to 169.09 ms (11.6% lower). Worst updated attachment remains 2106.49 ms; tail improvement is not established. Timer and 100us scheduling experiments worsened discovery results and were rejected. Evidence in ATTACHMENT_RESULTS.md; no full suite. PR publication and final task closeout pending.
<!-- SECTION:NOTES:END -->
