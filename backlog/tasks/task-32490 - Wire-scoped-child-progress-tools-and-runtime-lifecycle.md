---
id: TASK-32490
title: Wire scoped child progress tools and runtime lifecycle
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 23:54'
updated_date: '2026-09-11 02:29'
labels:
  - agents
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the approved report and collect tools through exact live agent capabilities while preserving budgets, continuation authority, and private metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Live threaded children can report; authorized primaries can collect within conversation and automatic-chain scope.
- [x] #2 Productive reads, native restoration, failure barriers, and body-free metadata follow ADR-136.
- [x] #3 Native runtime, session close, and child termination revoke stale messaging capabilities and release owned queue state.
- [x] #4 Saving a temporary conversation preserves its progress owner for live reporting, later primary collection, and close cleanup without resetting limits or chain scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md
Reason: Direct implementation of the accepted scoped messaging and existing ADR-063 continuation contracts.

1. After Task 1 review, follow Task 2 in Docs/superpowers/plans/2026-09-10-scoped-agent-messaging.md and its recorded runtime-readiness checks.
2. Add failing real-service relay, scope, privacy, continuation, and lifecycle tests before production changes.
3. Wire exact child senders and primary readers, dedicated reserved tools, productive-read cycle handling, and restored-call refusal while preserving execution barriers and budgets.
4. Add native bridge store ownership and close-before-cancel hooks; retain screen-navigation lifetime and reject stale owners after reopen.
5. Align existing steering fixtures with current internal continuation metadata; run focused new and affected tests plus scoped static checks.
6. Review the task's before-image diff, record actual evidence, and leave changes uncommitted.

Final review I1 correction (AC4): Follow the stable progress ownership correction in Docs/superpowers/plans/2026-09-10-scoped-agent-messaging.md and the final-fix-1-proposal.md checkpoint. Use an eager nonserialized native owner shared by live restored siblings, with atomic open/bind/release and last-binding close-before-cancel. Preserve persisted fleet/run identities, source IDs, lifetime/chain bounds, and exact native-session/inbox UI fencing. Verify actual submit/report/SQLite Save and rollback/cancel/close races, later primary reads, live producers, sibling cleanup and rendered navigation. Review the scoped final-fix before-image diff before completing AC4. Existing ADR-136 is amended; no new ADR, schema or authority change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact child report capabilities and primary inbox readers with reserved runtime tools, strict validation, automatic-chain checks, request-preview parity and body-free step metadata. Native private continuation retains protocol results under ADR-063; restored pending messaging calls refuse before queue effects. Reader access stays available when threaded spawning is disabled, and only trusted positive collection resets the no-progress cycle detector. The native denied/restored-reader cycle defect found in task review was fixed with regression coverage.

Final combined review found Save changing the progress lookup identity. An eager nonserialized native token now survives Save, rollback and cancelled awaits. Live restored siblings share it; last binding closes the queue before cancellation. One typed MessageStore registration covers direct store close/state replacement and fences stale runtime teardown. The controller captures expected ownership before preparation/worker scheduling, and the bridge validates it at entry and binding so old queued work cannot acquire replacement authority. Persisted fleet/run/budget keys, source IDs, chain scope and lifetime counters are unchanged.

Validation: final affected run 400 passed across nine files, including 14 identity cases and seven rendered UI cases; separate queue/coordinator/tool run 84 passed. Earlier complete messaging backend run passed 586. These overlap and are not additive. Nine corrected Python files introduce zero Ruff findings (204 inherited before/after); corrected ranges are formatted and new progress test files pass whole-file checks. Source/test hashes match the verified tree. All task reviews and the single final scoped re-review approved; no actionable finding remains in this feature scope.

ADR: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md, clarified for stable ownership and exact dispatch; no new ADR or schema. Final evidence, decisions, warnings and limitations: Docs/superpowers/reviews/2026-09-10-scoped-agent-messaging-implementation.md. Existing dependency/pytest cleanup warnings remain, and one unawaited UI coroutine warning has an unestablished cause/pre-fix baseline. No full suite, provider network call, staging or commit. Source changes are in the runtime/tool, bridge/controller/store and targeted test files listed in the implementation report.
<!-- SECTION:NOTES:END -->
