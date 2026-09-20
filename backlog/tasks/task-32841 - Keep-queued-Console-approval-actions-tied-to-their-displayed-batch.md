---
id: TASK-32841
title: Keep queued Console approval actions tied to their displayed batch
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 15:02'
updated_date: '2026-09-20 16:29'
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

## Current-dev integration (2026-09-20)

1. Preserve saved head 10747313ea6ec2c8d40381518ca31708cc94fa68 and replay only TASK-32841 onto merged dev de10a62e67124a2b21b78edf1a4887cea03ff139. Retain both review histories and existing lessons; inspect every conflict and product overlap.
2. Requalify the existing card/controller and ownership cases on this combined tree. Attribute remaining baseline failures against unchanged current dev before making regression claims. Port the saved native runner to the accepted shared CLI/image-warmup boundary only if needed, with its existing boundary tests.
3. Run relevant artifact/static checks and independent review. Repeat the real private-profile native keyboard Deny all, Approve all and Submit matrix in both themes and sizes, inspect all captures, and verify normal shutdown and unchanged user defaults.
4. Update the task, evidence and ledgers, push the existing bounded PR, address accumulated Qodo findings and exact-head CI, then present its own final visual approval. Do not merge without that approval or expand into remaining MCP journeys.

ADR required: no
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: integration and qualification of the existing approved-scope event-ownership repair, preserving policy, layout, storage and service boundaries.

Qodo follow-up: apply the local ApprovalActionButton casing cleanup and one shared WORKER_TIMEOUT_SECONDS value in the native runner. These are mechanical naming changes with no behavior, layout or policy change. Re-run affected targeted checks and current-source native evidence before republishing; existing ADR-032/150 still apply, no new ADR.

Approved closeout, 2026-09-20: preserve approved fd382754, rebase onto dev d1a0649 (Library/Artifacts navigation integration), verify unchanged approval source/controller/runner and no conflicts, rerun targeted cases and baseline attribution plus native matrix and guards, then publish and complete current-head CI/Qodo before the authorized merge. Record actual merged-tree provenance. No new ADR; existing ADR-032/150 govern the approval repair, upstream ADR-172 governs the navigation change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bounded repair complete: approval buttons stamp their batch generation at publication, stale/cleared/finishing gestures are ignored, and submission locks the current controls before publishing one decision. Unchanged identified resyncs remain stable; fresh rounds re-enable normally. Existing scopes, raw-shell exclusions, layout and controller policy are preserved.

Final isolated targeted run: 123 passed (including all 22 new ownership cases), 19 failed. All 19 failures reproduce on unchanged dev with identical names/messages/final traceback lines; they remain baseline profile/CSS/controller test debt. No full suite. No new Ruff diagnostics; changed/new formatting checked; all seven preflight guards passed including a separate integrity-pinned Canvas download retry. Independent review found no blocker and its additional fresh-round regression passes.

Eight inspected native run-003 captures use the real card/controller at 120x40 and 170x48 in both themes, with synthetic calls and no tool dispatch/network attempts. Normal exit, released lock, ten healthy private databases, no persisted conversations/messages, unchanged defaults and exact source hashes verified. Earlier overlay-obscured and concurrent-CSS runs are explicitly superseded. A live-verification lesson records the visibility assertion trap.

Evidence: Docs/superpowers/qa/2026-09-19-approval-action-ownership/README.md and GALLERY.md. Only owned files were copied into codex/approval-batch-action-review after concurrent branch switching/edits appeared in the original checkout; original and unrelated files are preserved. Census-selected TASK-32841 corrected the CLI's stale newly created 32829 before implementation; two uncommitted copies represent one logical task.

ADR required: no new ADR. Existing backlog/decisions/032-local-agent-tool-permission-boundary.md and backlog/decisions/150-design-token-system-and-design-language.md govern this routine event-ownership fix. Independent draft against dev; current-head CI/review and final visual approval still required before merge. Wider component work remains open.

Current-dev integration (2026-09-20): replayed only the saved approval fix onto merged dev de10a62e67 after PR2728 closeout. Both report conflicts retain all histories; no product conflict. Production code is unchanged from saved 10747313. Added three isolated publication/submission checks and moved the native runner to the shared CLI validator and application image warmup. Independent review caught editable-package import ordering; final runner pins the worktree before any project import and asserts six loaded module origins. Integrated native run 001 is unqualified, 002 superseded, and final 003 supplies all eight inspected dark/light captures and healthy lifecycle receipts. No tool dispatch or network attempt. Targeted union: 210 distinct passes (126 product, 84 runner); all 19 failures reproduce on unchanged current dev after normalizing process-specific object addresses in two assertions. No full sweep. Seven guards pass; no new static diagnostics and changed formatting passes. Current evidence and conflict choices: Docs/superpowers/qa/2026-09-19-approval-action-ownership/CURRENT-DEV-REVIEW.md. Existing ADR-032/150 apply; no new ADR. Local implementation/review complete; PR2730 current-head CI/Qodo and its own visual approval remain merge gates.

Qodo reported zero bugs and two maintainability findings on f1af5ee17c. Both addressed: helper class renamed ApprovalActionButton and the two native worker waits share WORKER_TIMEOUT_SECONDS=8. Production AST is identical after normalizing only the helper name. All 109 affected tests and seven fresh guards pass, with no new Ruff diagnostics. Final native integration run 006 passes all four controller journeys and lifecycle checks; all eight captures are pixel-identical to inspected run 003. Launches 004/005 were refused before app imports because required private directories were missing; neither is qualified. Current source and runner hashes match the new receipts. Own visual approval and current-head CI/Qodo remain merge gates.

Owner approved the fd382754 Console gallery and merge. Rebased onto current dev d1a0649 (PR2754 Library/Artifacts) without conflicts. Approval source/controller/Console/runner/tests remain byte-identical to the approved head. Fresh targeted run: 211 passed, 18 failed; all 18 failures reproduce unchanged on current dev after object-address normalization. Previously failing human-input-wait case passes on both current trees; no fix is attributed to this PR. All seven guards and native lifecycle pass. Across all eight images, only upstream navigation pixels differ; approved controls remain pixel-identical. Four current submit views inspected across themes/sizes. Evidence: Docs/superpowers/qa/2026-09-19-approval-action-ownership/CLOSEOUT.md. Existing ADR-032/150; upstream ADR-172. Rebased-head CI/Qodo and actual merge verification remain before authorized closeout.
<!-- SECTION:NOTES:END -->
