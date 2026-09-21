---
id: TASK-32825
title: Keep MCP server actions bound to their displayed target
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 00:33'
updated_date: '2026-09-21 01:49'
labels:
  - mcp
  - ui
  - ownership
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Queued server actions must never connect, disconnect, edit or delete a different profile when selection or readiness changes. Accepted deletion must retain the exact confirmed target while the toolbar updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Retired or hidden server toolbar controls cannot dispatch or arm actions for a replacement detail view.
- [x] #2 An accepted delete confirmation retains its original server identity across asynchronous toolbar teardown and subsequent selection changes.
- [x] #3 Current toolbar actions, safe Keep and Escape behavior, and keyboard-visible confirmation remain usable at compact and wide sizes.
- [x] #4 Deterministic regression, targeted neighboring tests and bounded native evidence qualify the repair and document remaining server lifecycle scope.
- [x] #5 Queued toolbar controls cannot act after a Servers to another mode to Servers round-trip, while disabled directly or through an ancestor, or while covered by another screen.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/161-component-pattern-library.md; ADR-150 and current MCP lifecycle contracts apply.
Reason: restore displayed-action identity, confirmation lifetime and compact token-backed toolbar behavior within existing UI handlers; no new storage, permissions, service or runtime boundary.
1. Resume saved PR2712 on merged PR2716 dev 9cf5ba67b2; preserve current lifecycle/cancellation fixes. Reproduce the saved ownership/layout regressions against this baseline.
2. Integrate the minimal saved toolbar target ownership and compact grid changes, rebuilding generated CSS from current modules. Add deterministic queued-event round-trip, disabled-ancestor and covered-screen regressions before repairing any remaining gaps. Preserve confirmations accepted before presentation awaits and normal current controls.
3. Run focused server/Workbench lifecycle and design-governance tests, baseline-relative static checks and artifact guards. Independently review the bounded patch.
4. Retire the historical native executable behind its immutable source reference. Qualify a current runner with shared CLI/private-profile validation, correct import provenance, supported terminal warmup, network guard, occupied-ID refusal and owned cleanup. Inspect real keyboard dark/light 80x24 and170x48 Keep/Escape, retired action and exact-target deletion journeys with full paint/hit checks and clean lifecycle/default fingerprints.
5. Update existing draft PR2712 and the review ledgers, then present fresh native evidence for owner visual approval. Current-head CI, accumulated review and current-dev/conflict checks remain separate merge gates. Compact Test Tool, Audit and other screen reviews remain out of this slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resumed draft PR2712 on merged PR2716 dev 9cf5ba67b2, retaining current lifecycle/cancellation behavior. Toolbar widget identities retain displayed targets; unavailable controls and mode departures revoke queued presses synchronously. Captured revisions preserve newer controls and prevent overlapping toolbar publication, while accepted deletion retains its target. Compact toolbar uses existing token-backed two-column action classes; wide layout unchanged. 232 distinct targeted cases pass, including 31 ownership/departure regressions, and all eight artifact guards pass. Scoped Ruff finds no introduced diagnostics; new files and changed ranges are formatted. Independent review caught two timing gaps, both reproduced and fixed; final production and native-runner reviews are clear. Four fresh native theme/size journeys and 16 inspected captures pass with real private profile persistence, Keep/Escape, retired presses and Alpha-only deletion. Clean exit, lock release, ten healthy databases, unchanged defaults, matching source hashes and no network attempts verified. Existing ADR-150/161 apply; no new ADR. Historical native executable retired behind immutable saved-head reference; current runner uses shared admission and owned cleanup. Evidence: Docs/superpowers/qa/2026-09-18-mcp-server-actions/current-dev/README.md. Remains In Progress pending fresh owner visual approval, current-head CI, accumulated review and final dev/conflict checks. Compact Test Tool, Audit and remaining screens stay separate.

Qodo mount-completion finding investigated against Textual 8.2.8: registration is synchronous before AwaitMount, whose completion cannot republish controls. Two added real-mount overlap cases pass, and independent source/test review confirms no additional lock or production change is warranted. Native hashes and all 16 captures remain applicable. The first reproduction harness incorrectly awaited an intentionally cancelled exclusive worker; corrected to observe replacement mount completion. Evidence and rationale are recorded in current-dev/independent-review.md.
<!-- SECTION:NOTES:END -->
