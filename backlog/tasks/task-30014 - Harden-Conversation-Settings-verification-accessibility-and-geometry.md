---
id: TASK-30014
title: Harden Conversation Settings verification accessibility and geometry
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 18:04'
updated_date: '2026-09-04 01:42'
labels:
  - console
  - ux
  - accessibility
  - testing
dependencies:
  - TASK-30013
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish Conversation settings with honest supported connection probes, an optional explicitly authorized paid generation check, deterministic keyboard/accessibility behavior, and verified rendering at compact, normal, and wide terminal sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Test connection appears only when an existing provider service supports a meaningful bounded non-generating probe; other providers state that no non-billable live check is available
- [x] #2 Any paid generation check is optional, provider-matrix-backed, cancellable, sanitized, and requires explicit usage confirmation before every request
- [x] #3 Network workers reject stale results, use bounded timeouts, preserve the draft on failure, and never display or log secrets, raw response bodies, or credential-bearing URLs
- [x] #4 Visible labels, Textual-supported accessible names/descriptions, selected-state semantics, non-color-only statuses, single announcements, and deterministic focus order are covered by focused tests
- [x] #5 The approved save accelerator is discoverable and does not shadow terminal-convention or global bindings
- [x] #6 At 80x24, 100x30, and 160x40 the Connection flow and completion actions remain visible/reachable without horizontal scrolling; compact actions stack or wrap with full labels
- [x] #7 Isolated live UAT captures the cloud deep-link/return, local failure/recovery, provenance, disclosure, keyboard, disabled-action, and Ready to send evidence without touching the real profile
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR paths: backlog/decisions/006-provider-aware-generation-settings.md; backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/012-provider-credential-settings-boundary.md; backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/097-boot-budget-ratchets.md
Reason: This hardens existing UI, provider probe, credential-safety, and first-paint performance contracts; paid generation remains explicit and optional and does not become a new runtime boundary.

Execute the red-green checklist in Docs/superpowers/plans/2026-09-02-task-30014-verification-accessibility-geometry.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented and verified the final Conversation settings hardening. Added provider-matrix-backed bounded non-generating connection/model-list probes; explicit fresh-confirmation one-token generation tests with actual timeout/retry controls, cancellation truth, sanitized result categories, exact identity fencing, and independent endpoint/model/generation evidence; deterministic keyboard focus, Ctrl+Enter primary activation, picker Escape behavior, accessible labels/tooltips, single settled-result announcements, and responsive one-scroll-owner geometry at 80x24, 100x30, and 160x40. Final UAT used an isolated temporary profile, fake cloud key, and disposable localhost provider for refused, zero/one/many-model, provenance, confirmation/cancel, generation success, stale-invalidation, and deep-link/real-router-return flows; real and decoy profile hashes were unchanged. Evidence: /tmp/task-30014-final-uat.fXycHU/evidence, including cloud-return-restored-100x30.svg, and /tmp/task-30014-geometry-evidence. Verification: the first two planned groups passed 1,063 tests; native Console passed 359; rail passed 53; Conversation settings passed 433 with only the separately proven pre-existing unrelated roleplay-GC poison node deselected; geometry passed 11. Task5 also fixed a real active-run queue/readiness conflict, delayed-focus-after-unmount guard, current retrieval/skill test harnesses, truthful in-modal endpoint recovery, and a re-entrant modal focus callback; focused regressions pass. Post-review CI hardening defers the modal, provider picker, and endpoint probe until first use and re-keys three modal button selectors to scoped classes, restoring the ADR-097 UI-ready and selector ratchets to 968/972 modules and 274/274 selectors locally; a subprocess import-closure regression guard and 76 focused open/probe/deep-link/return tests pass. Ruff, compileall, CSS bundle reproduction, and git diff --check passed. ADR required: no new ADR; ADR-006, ADR-011, ADR-012, ADR-033, and ADR-097 remain authoritative. No real provider request or billing occurred.
<!-- SECTION:NOTES:END -->
