---
id: TASK-32821
title: Honor shared dialog action alignment
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 21:06'
updated_date: '2026-09-18 21:23'
labels:
  - design-system
  - components
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The documented dialog-buttons and button-group composition centers actions even when a caller selects left or right alignment. Restore the shared component contract so the Pattern Gallery and real consumers agree without local exceptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Explicit left and right dialog action modifiers place the group at their requested edge; plain and explicitly centered dialog rows retain centered layout.
- [x] #2 The real Pattern Gallery paints its dialog actions at the trailing edge with readable keyboard focus at compact and wide sizes in both themes.
- [x] #3 Roleplay recovery keeps its verified layout and Retry/Stay/Escape results using the shared rule, with its local alignment override removed.
- [x] #4 Reviewed gallery snapshots, targeted component/consumer checks and original CSS budgets pass with source-bound native visual and lifecycle evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/161-component-pattern-library.md. Reason: restore existing component catalog alignment contract; no new UX or runtime boundary. 1. Read all dialog action consumers and current CSS cascade; retain baseline gallery snapshot status. 2. Add painted geometry regressions for explicit left/right, plain/explicit-center rows and the real gallery consumer. 3. Add compound component rules so explicit modifiers win; remove the Roleplay-specific workaround and rebuild CSS. 4. Run targeted tests serially in fresh profiles, deliberately review and update only intended gallery snapshot changes, and check token/build/byte/selector budgets. 5. Capture native palette-to-gallery focus at compact/wide dark/light sizes plus Roleplay continuity; verify process/log/lock/private DB/default-state lifecycle. 6. Record evidence, independent review, task/ledger and draft PR update. All-ref and33-worktree allocation scan: max32820, reserved32821.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored explicit left/right dialog action alignment with shared compound selectors, preserving default/centered rows and vertical centering. Removed the Roleplay local override, rebuilt generated CSS, updated component documentation and deliberately reviewed both gallery snapshots. New geometry/paint/focus regressions plus existing consumers and guards qualify 53 distinct cases. Eight final dark/light compact/wide native captures were inspected; palette entry, gallery focus/Escape and direct Roleplay Retry/Stay/Escape pass. Final lifecycle confirms normal exit, released lock, ten healthy private databases, unchanged defaults, no errors and 11 matching source hashes. CSS 584,093/608,090 bytes and 274/274 selectors; all seven preflight guards and Ruff/format pass. Independent final code/evidence review found no actionable issue. QA: Docs/superpowers/qa/2026-09-18-dialog-action-alignment/README.md retains red cases, interrupted first SVG comparison, concise failures, final snapshots and first toast capture. Added the observed SVG diagnostic trap to lessons-testing-evidence.md. No new ADR: repairs existing backlog/decisions/161-component-pattern-library.md contract. Broader review remains open; PR2707 stays draft/unmerged.
<!-- SECTION:NOTES:END -->
