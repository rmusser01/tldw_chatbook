---
id: TASK-21500
title: 'Console: expose Improve current draft and recommend Review'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 04:46'
updated_date: '2026-08-24 06:01'
labels:
  - console
  - prompts
  - ux
  - uat
dependencies: []
references:
  - >-
    .impeccable/critique/2026-08-24T04-39-32Z__chatbook-widgets-console-console-prompts-modal-py.md
  - Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md
  - >-
    backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user with an unsent Console draft begin improvement directly, without first navigating through the Prompt Library. Preserve Library browsing as a separate destination and make Review the visibly recommended, safe first-use improvement mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the composer contains a nonblank unsent message, its menu exposes a distinct `Improve current draft…` item that opens the improvement mode chooser directly.
- [x] #2 `Browse Prompt Library…` remains a separate, plainly named destination; an empty composer does not offer an actionable improvement path and still provides direct Library access.
- [x] #3 The improvement chooser visibly marks Review as `Recommended`, gives it initial keyboard focus on first entry, and does not start any provider request until the user chooses a mode.
- [x] #4 Direct entry captures the current unsent message, optional current System prompt, and current Console provider/model exactly as the existing workbench does; it does not silently select another provider or model.
- [x] #5 The user can include or exclude the System prompt before starting any of the three improvement modes, with the choice preserved when entering Recipe mode.
- [x] #6 Menu order, focus restoration, Escape behavior, and the 140x40, 100x30, and 80x24 layouts remain keyboard-reachable and free of clipped labels or actions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the current composer-menu, Prompt controller, and modal-entry contracts on current dev and retain the existing provider/model, captured-context, focus, and safe-dismissal boundaries.
2. Add failing mounted Textual tests for a direct `Improve current draft…` path, separate `Browse Prompt Library…` path, empty-draft behavior, and Review's recommended initial focus.
3. Implement the smallest native Textual changes that route direct improvement into the existing modal state without duplicating the workbench or provider logic.
4. Run the targeted composer-menu, Prompt-modal, workbench-contract, and native-flow checks, then inspect rendered 140x40, 100x30, and 80x24 evidence for clipping and focus order.
5. Update user-facing Prompt guidance if the navigation labels or first-use behavior are documented, then complete an accessibility, privacy, and self-review pass.

ADR required: no
ADR path: N/A; existing ADR-040 remains applicable.
Reason: this changes entry routing, labels, and initial focus inside the existing Prompt workbench without altering storage, provider, privacy, security, or cross-module ownership boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added separate composer-menu destinations for direct draft improvement and Prompt Library browsing, with direct entry conditional on a nonblank draft.
- Reused the existing Prompt workbench in an initial Improve mode, made Review visibly recommended and initially focused, and ensured focused actions are scrolled fully into view at 140x40, 100x30, and 80x24.
- Kept chooser entry provider-free. Provider resolution now starts only after Auto, Review, or Recipe Fill is chosen; Structured Recipe remains available without resolution. System-context opt-out survives mode changes and activation.
- Preserved the exact opening provider/model/endpoint disclosure with canonical endpoint comparison and a reopen-required drift guard. Provider, protected-draft, and stale-context blockers now expose only relevant recovery actions.
- Updated Console user guidance and the real-app QA capture flow. The checked-in full QA runner still aborts later on its pre-existing removed `ChatScreen._ensure_active_console_session_settings()` call; responsive captures completed before that point and mounted contracts provide the authoritative behavioral evidence.
- Verification: 218 composer-menu/modal/workbench tests passed; 25 native prompt-improvement tests passed; focused stale-context and direct-entry regressions passed; targeted Ruff and `git diff --check` passed. Independent review reported no Critical or Important findings, and the Impeccable detector returned `[]`.
- ADR required: no. ADR-040 remains the applicable provider/privacy/transaction boundary; no storage or cross-module ownership decision changed.
<!-- SECTION:NOTES:END -->
