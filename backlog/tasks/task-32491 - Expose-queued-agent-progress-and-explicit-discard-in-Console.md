---
id: TASK-32491
title: Expose queued agent progress and explicit discard in Console
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
Let users inspect pending progress and recover queue capacity without starting agents or changing completion attention.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Queued progress counts and a literal-text view remain reachable after child pruning and with agent mode disabled.
- [x] #2 Explicit discard only removes selected snapshot IDs and preserves concurrent arrivals, child budgets, and completion state.
- [x] #3 Rendered UI tests cover refresh, navigation, privacy copy, and existing fleet behavior.
- [x] #4 Progress inspection and navigation remain attached to the correct owner when a temporary conversation is saved, including an already-open or stale view.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md
Reason: Implements the accepted explicit progress inspection, scoped discard, and session-only visibility contract.

1. After runtime integration review, follow Task 3 in Docs/superpowers/plans/2026-09-10-scoped-agent-messaging.md and its UI-readiness notes.
2. Add a failing rendered test using the real store for inspect/select/concurrent arrival/discard.
3. Add the literal-text progress modal and reachable count/action in the Agents rail; bind callbacks to the captured inbox and resolved runtime conversation identity.
4. Add separate body-free progress counts to existing conversation navigation, including unsaved native sessions; preserve historical run counts and completion attention.
5. Verify pruning, mode-off access, live collection refresh, close/reopen stale callbacks, and navigation with rendered tests. Correct affected legacy harness setup using the confirmed real-DB fixture diagnosis.
6. Update user guidance and implementation status after verification; run targeted checks, review the combined messaging diff, and record actual evidence. No full suite, commit or staging.

Final review I1 correction (AC4): Follow the stable progress ownership correction in Docs/superpowers/plans/2026-09-10-scoped-agent-messaging.md and the final-fix-1-proposal.md checkpoint. Use an eager nonserialized native owner shared by live restored siblings, with atomic open/bind/release and last-binding close-before-cancel. Preserve persisted fleet/run identities, source IDs, lifetime/chain bounds, and exact native-session/inbox UI fencing. Verify actual submit/report/SQLite Save and rollback/cancel/close races, later primary reads, live producers, sibling cleanup and rendered navigation. Review the scoped final-fix before-image diff before completing AC4. Existing ADR-136 is amended; no new ADR, schema or authority change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the explicit queued-progress action in the Console Agent rail, a literal-text inspection modal, and body-free native-session counts in conversation navigation. Reports remain reachable with agent mode off and after child handle pruning. Inspection does not consume or clear completion attention. Discard removes only checked IDs from the displayed snapshot, preserves concurrent arrivals, releases pending capacity exactly once and does not refund lifetime allowance.

The modal captures the exact native session and inbox. Saving a temporary conversation preserves an already-open selected view and navigation counts; switching, closing or replacing its owner invalidates stale callbacks. Live saved siblings share progress until last close. Count polling refreshes navigation only on changes and stops with the view. Report selection and multiline literal detail use separate widgets; row-specific compositor assertions verify the actual narrow navigation count rather than matching the duplicate Agent label. Existing historical run counts and recovery behavior are preserved.

Validation: the final affected run passed 400 cases, including all seven styled UI cases and 14 ownership cases. The rendered actual-child/Save test covers pre-Save selection, a post-Save report, selected discard, counts and stale same-ID replacement. Earlier affected UI/navigation evidence and wide/narrow production-style captures are retained in the implementation report. All five CSS bundles reproduce. New files and edited ranges pass static checks; the initial missed SIM102 finding and incorrect lint claim were corrected and independently re-reviewed. Task behavior/visual review and final scoped identity re-review approved with no new findings.

ADR: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md; direct implementation with ownership clarification, no new ADR. User guide, implementation plan and orchestration ledger updated. Full evidence: Docs/superpowers/reviews/2026-09-10-scoped-agent-messaging-implementation.md. The pre-existing ChatScreen size ratchet stays unchanged; an observed unawaited UI coroutine warning has no established cause/pre-fix baseline. No full suite, provider network call, staging or commit. Changes are in the progress modal, Console modules/navigation, CSS, bridge ownership integration and targeted tests.
<!-- SECTION:NOTES:END -->
