---
id: TASK-16266
title: Harden Speech view switching and first-run test seam
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 19:07'
updated_date: '2026-08-14 20:14'
labels:
  - testing
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent rapid Speech rail navigation from pruning partially mounted controls and keep the first-run fake service aligned with the current audio.cpp lifecycle contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rapid Speech rail navigation does not prune a Playground while its nested controls are mounting.
- [x] #2 The first-run audio.cpp handoff fake exercises the current lifecycle-backed Test Connection operation.
- [x] #3 Narrow clone-setup reachability waits for Textual's asynchronous scroll convergence without weakening the viewport assertion.
- [x] #4 Focused Speech modules and the original 25-file checkpoint pass with static checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the two deterministic failures and identify their independent causes.
2. Defer reactive Speech view replacement until the current refresh/mount boundary completes.
3. Give the first-run fake service the existing lifecycle operation used by the production button.
4. Wait on the clone-setup test's observable scroll result instead of one scheduler turn.
5. Run focused tests, affected modules, the original checkpoint, and static checks.

ADR required: no
ADR path: N/A
Reason: narrow lifecycle bug fix and test-double reconciliation within existing Speech ownership boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Serialized and awaited Speech view replacement through the strict axis descendants so rapid rail changes cannot remove partially mounted nested controls. Deferred the language-axis option mutation from the pane Mount event to its first refresh because Textual had not always composed the nested Select label yet.
- Restored Refresh after successful audio.cpp lifecycle operations and updated the first-run fake for passive runtime observation, current effective-selection provenance, and sanitized projection copies.
- Replaced guessed scheduler turns with observable lifecycle, mounted-Playground, and viewport convergence checks. The rail regression passed 20 consecutive fresh-app runs after the final fix.
- Verification: rail module 7 passed; affected lifecycle samples 4 passed; original 25-file checkpoint 459 passed in 431.41s with two dependency warnings and one existing session-level FD-growth warning. Ruff lint, source compilation, privacy scan, and diff hygiene passed. Ruff format remains baseline-red on the same three files at HEAD.
<!-- SECTION:NOTES:END -->
