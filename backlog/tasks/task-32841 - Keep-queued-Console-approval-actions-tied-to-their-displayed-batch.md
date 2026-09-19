---
id: TASK-32841
title: Keep queued Console approval actions tied to their displayed batch
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 15:02'
updated_date: '2026-09-19 15:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent delayed bulk or submission gestures from changing or deciding a replacement approval batch, while keeping current keyboard actions usable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A queued bulk or submit action cannot affect a changed, cleared, finishing or replacement approval batch.
- [x] #2 Each current batch emits at most one decision; unchanged resyncs preserve choices and fresh batches remain actionable.
- [x] #3 Targeted mounted and controller checks plus native dark/light verification preserve decision scope, raw-shell exclusions and existing UI conventions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce shared-button queued delivery across actual set_batch transitions and rapid repeated submissions.
2. Bind action messages to their originating batch generation and reject resolved/inactive batches using existing decision and round-id paths. Preserve current control layout and decision options.
3. Verify targeted card/controller cases, static/derived guards and independent review; inspect native dark/light keyboard bulk and Submit flows in a private profile. Save an independent draft against dev.
ADR required: no
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: routine repair to existing approval-event ownership, with no new policy, persistence, service boundary or interaction model. The proposed MCP matrix bulk-actions ADR is not implemented by this review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bounded repair complete: approval buttons stamp their batch generation at publication, stale/cleared/finishing gestures are ignored, and submission locks the current controls before publishing one decision. Unchanged identified resyncs remain stable; fresh rounds re-enable normally. Existing scopes, raw-shell exclusions, layout and controller policy are preserved.

Final isolated targeted run: 123 passed (including all 22 new ownership cases), 19 failed. All 19 failures reproduce on unchanged dev with identical names/messages/final traceback lines; they remain baseline profile/CSS/controller test debt. No full suite. No new Ruff diagnostics; changed/new formatting checked; all seven preflight guards passed including a separate integrity-pinned Canvas download retry. Independent review found no blocker and its additional fresh-round regression passes.

Eight inspected native run-003 captures use the real card/controller at 120x40 and 170x48 in both themes, with synthetic calls and no tool dispatch/network attempts. Normal exit, released lock, ten healthy private databases, no persisted conversations/messages, unchanged defaults and exact source hashes verified. Earlier overlay-obscured and concurrent-CSS runs are explicitly superseded. A live-verification lesson records the visibility assertion trap.

Evidence: Docs/superpowers/qa/2026-09-19-approval-action-ownership/README.md and GALLERY.md. Only owned files were copied into codex/approval-batch-action-review after concurrent branch switching/edits appeared in the original checkout; original and unrelated files are preserved. Census-selected TASK-32841 corrected the CLI's stale newly created 32829 before implementation; two uncommitted copies represent one logical task.

ADR required: no new ADR. Existing backlog/decisions/032-local-agent-tool-permission-boundary.md and backlog/decisions/150-design-token-system-and-design-language.md govern this routine event-ownership fix. Independent draft against dev; current-head CI/review and final visual approval still required before merge. Wider component work remains open.
<!-- SECTION:NOTES:END -->
