---
id: TASK-32825
title: Keep MCP server actions bound to their displayed target
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 00:33'
updated_date: '2026-09-21 06:12'
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
- [x] #6 The merged server-action repair satisfies the newly landed Workbench size ratchet without increasing its budget or changing mode cleanup behavior.
- [x] #7 The current-dev integration stays within the existing UI-ready module budget while preserving Advanced execution permission and audit behavior.
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
6. Post-merge integration follow-up: PR2744 landed the Workbench size ratchet at merge time. Move the existing mode-dependent inspector clearing sequence into MCPInspector, retain the deferred async worker and order, and lower the ratchet to the measured count. Verify architecture, mode transitions and queued server-action regressions; submit a bounded follow-up PR. This is a routine ownership refactor within the existing inspector API boundary, so no new ADR is required.
7. Current-dev CI follow-up (existing ADR097): the upstream UTC timestamp helper raised the UI-ready census to1027 vs1026. Defer the MCP readiness import used only by Advanced execution until that operation, preserving the existing constant and all admission/permission/audit calls. Add an isolated import-closure regression and verify boot census plus Advanced execution tests; do not raise the guard.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resumed draft PR2712 on merged PR2716 dev 9cf5ba67b2, retaining current lifecycle/cancellation behavior. Toolbar widget identities retain displayed targets; unavailable controls and mode departures revoke queued presses synchronously. Captured revisions preserve newer controls and prevent overlapping toolbar publication, while accepted deletion retains its target. Compact toolbar uses existing token-backed two-column action classes; wide layout unchanged. 232 distinct targeted cases pass, including 31 ownership/departure regressions, and all eight artifact guards pass. Scoped Ruff finds no introduced diagnostics; new files and changed ranges are formatted. Independent review caught two timing gaps, both reproduced and fixed; final production and native-runner reviews are clear. Four fresh native theme/size journeys and 16 inspected captures pass with real private profile persistence, Keep/Escape, retired presses and Alpha-only deletion. Clean exit, lock release, ten healthy databases, unchanged defaults, matching source hashes and no network attempts verified. Existing ADR-150/161 apply; no new ADR. Historical native executable retired behind immutable saved-head reference; current runner uses shared admission and owned cleanup. Evidence: Docs/superpowers/qa/2026-09-18-mcp-server-actions/current-dev/README.md. Remains In Progress pending fresh owner visual approval, current-head CI, accumulated review and final dev/conflict checks. Compact Test Tool, Audit and remaining screens stay separate.

Qodo mount-completion finding investigated against Textual 8.2.8: registration is synchronous before AwaitMount, whose completion cannot republish controls. Two added real-mount overlap cases pass, and independent source/test review confirms no additional lock or production change is warranted. Native hashes and all 16 captures remain applicable. The first reproduction harness incorrectly awaited an intentionally cancelled exclusive worker; corrected to observe replacement mount completion. Evidence and rationale are recorded in current-dev/independent-review.md.

PR2712 merged as 4b61a5ca8f1dd33a4cc839b3f329a068decedea5 after visual approval, current-head CI and resolved Qodo review. The actual parent advanced to PR2744 architecture tests at merge time; post-merge checks passed 66 cases but exposed Workbench at 6771 lines against the new 6761-line limit. Task remains In Progress for a bounded integration follow-up; compact Test Tool review follows it.

Post-merge integration repair moves the unchanged tool/audit/finding clearing sequence into MCPInspector.clear_mode_view. Workbench keeps the same deferred async worker and exclusive group; size falls to6760 and the ratchet is tightened. All106 focused architecture/MCP cases and nine artifact guards pass; no new Ruff findings and all17 changed ranges formatted. Independent review finds no blockers and confirms prior native visuals apply to this mechanical move. No new ADR required. Follow-up PR/CI and merge remain pending.

PR2769 current-dev CI exposed the upstream timestamp helper raising UI-ready modules to1027. Deferred the existing Advanced execution readiness import (ADR-097), restoring1026 without changing budget, permission or audit logic. The isolated import regression is red before and green after;12 boot/import checks and8 Advanced execution cases pass. Three execution cases retained their collection-selected private profile using the existing bootstrap_profile marker after identical failures were reproduced on unchanged HEAD. Independent review clear; no introduced Ruff diagnostics and all5 changed ranges formatted. Final-head CI/current-dev merge checks remain pending.
<!-- SECTION:NOTES:END -->
