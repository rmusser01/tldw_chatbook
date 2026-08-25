---
id: TASK-22130
title: 'Console: surface improvement Undo and before-after review'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 04:46'
updated_date: '2026-08-24 06:23'
labels:
  - console
  - prompts
  - ux
  - recovery
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
Make an automatic prompt replacement immediately understandable and reversible from the composer. A user should not need to reopen the composer menu to discover that the previous draft can be restored or to inspect what changed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The automatic mode is labeled `Replace draft automatically` before execution so its effect is clear.
- [x] #2 After a successful automatic replacement, the composer immediately shows a persistent `Draft improved` status with keyboard-reachable `Undo` and `Review changes` actions.
- [x] #3 Undo restores the exact pre-improvement composer transaction, including draft text and attachment-related state, and remains safe when invoked repeatedly or after unrelated late provider results.
- [x] #4 `Review changes` opens a before-and-after comparison without first reverting the improved draft; the improved version remains editable and the user can keep it or restore the original.
- [x] #5 The visible improvement status is cleared only by a subsequent draft edit, send, explicit restoration, or session/context replacement, and stale actions cannot mutate newer composer state.
- [x] #6 Success, failure, and stale-result status changes are textually exposed for keyboard/accessibility users without logging prompt bodies or changing the existing sensitive-provider boundary.
- [x] #7 The composer-menu Undo remains available as a secondary recovery path, and both recovery surfaces behave consistently at supported terminal sizes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the existing one-shot improvement snapshot, composer mutation events, and status-row patterns.
2. Add failing mounted tests for the renamed automatic action, persistent Undo/Review actions, exact restoration, comparison behavior, and expiry on edit/send/session replacement.
3. Implement a composer-owned improvement recovery presentation that reuses the existing immutable snapshot and stale guards.
4. Add the before/after review modal with keep/restore outcomes and keyboard-safe focus.
5. Run targeted composer, Prompt-modal, workbench, and native-flow verification plus responsive visual inspection.

ADR required: no
ADR path: N/A; ADR-040 remains applicable.
Reason: this surfaces existing transaction state and recovery actions without changing storage, provider, privacy, or ownership boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Renamed automatic improvement to `Replace draft automatically` and added a conditional composer-owned `Draft improved` row with direct Undo and Review actions. Both direct Undo and the existing composer-menu Undo share the same exact-snapshot restoration and session-draft persistence path.
- Added a read-only before/after comparison with Keep and Restore decisions. Inline-file bodies are replaced by protected labels in both panes; the exact private snapshot remains composer-owned and is used only for restoration.
- Bound recovery visibility to the existing one-shot snapshot lifecycle, including edit, send, load/session-context replacement, explicit restore, and collapse/expand behavior. No new persistence, provider, or telemetry boundary was introduced.
- Added mounted lifecycle, privacy, modal-flow, responsive geometry, and exact-restoration coverage. The deterministic real-app responsive stage passed and produced recovery/comparison captures at 140x40, 100x30, and 80x24; all were visually inspected without clipping or unreachable actions.
- Targeted prompt/improvement verification passed (189 tests, followed by 2 updated contract expectations); focused composer/native tests passed; Ruff, compileall, and `git diff --check` passed. The broader composer-collapse file reported 76 passed and four transcript-overflow setup failures; the same representative failure reproduces on untouched base `f4995603`, so it is recorded as baseline rather than attributed to this change.

ADR required: no. ADR-040 remains the governing transaction/privacy decision.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task was originally filed as `TASK-21501`. During the final rebase for PR #2053,
add-commit provenance showed that the unrelated private-MCP-tool-surface task reached
`dev` first in `3f439ce7a`; this prompt-workbench task arrived later in `5a3b53802`.
Per the TASK-19601 older-arrival rule, the older task keeps `TASK-21501` and this task
moves to `TASK-22130`, selected after sweeping all current local and remote refs.

No inbound reference identified this task by its old ID. Remaining `TASK-21501`
references belong to the older private-MCP task or the separately renumbered
cursor-blink task lineage and are intentionally unchanged here.
