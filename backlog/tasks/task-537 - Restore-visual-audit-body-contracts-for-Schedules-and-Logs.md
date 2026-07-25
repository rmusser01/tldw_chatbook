---
id: TASK-537
title: Restore visual audit body contracts for Schedules and Logs
status: Done
assignee: []
created_date: '2026-07-24 21:15'
updated_date: '2026-07-25 19:12'
labels:
  - ui
  - navigation
  - schedules
  - logs
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the stable destination-body contracts omitted when `SchedulesWorkbench` replaced `SchedulesScreen` and when Logs returned to the top-level shell, so cross-destination visual and focus audits match the canonical route inventory at every supported terminal size.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `SchedulesWorkbench` mounts the stable `#schedules-shell` destination body around its current sync status, queue, detail, inspector, and conflicts content.
- [x] #2 Phase-1 and phase-6 visual-audit body-selector maps cover every canonical shell destination, including the top-level Logs destination.
- [x] #3 Compact, laptop/default, and large/wide phase-1 and phase-6 visual-audit matrices pass without missing-body or route-inventory failures.
- [x] #4 Focused Schedules and shell-navigation tests plus Ruff, formatting, compile, and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the clean-HEAD Schedules failure as the regression baseline and add a focused shell assertion for `SchedulesWorkbench`.
2. Restore the stable outer `#schedules-shell` wrapper around the current workbench content without changing scheduling services or route ownership.
3. Add the existing Logs destination header to both visual-audit body-selector maps and amend ADR-015's route inventory to reflect the already-established top-level Logs decision.
4. Run the focused Schedules/shell-navigation suite and both phase-1 and phase-6 terminal-size matrices.
5. Run static checks, inspect the bounded diff, and request independent review before closeout.

ADR required: yes
ADR path: backlog/decisions/015-shell-destination-ia.md
Reason: The canonical shell ADR still describes Logs as folded under Settings even though the established route inventory and navigation tests make Logs a top-level destination; this task amends the existing ADR rather than creating a duplicate decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reconciliation note (2026-07-25): the original task record and implementation
  landed on a different branch, while this branch later acquired the Schedules
  visual-parity changes without the stable shell, Logs selector maps, or ADR
  amendment. The task was reopened here to merge both change sets and rerun the
  sentinel and visual matrices before declaring it complete. The merge also
  retargets the newer direct-child height rule through `#schedules-shell`; the
  unadjusted rule let the workbench overflow three rows below the viewport.
- Restored the stable `#schedules-shell` outer body around the current `SchedulesWorkbench` content without changing scheduling services, workers, events, or responsive class ownership.
- Added the existing `#logs-destination-header` to both cross-destination visual selector maps and added direct phase-6 route-inventory coverage so future destination additions cannot fail only deep inside the size sweep.
- Amended [ADR-015](../decisions/015-shell-destination-ia.md) to record the already-established 13-destination taxonomy, top-level Logs ownership, fold map, palette semantics, rationale, and consequences.
- Reconciliation verification: all 9 phase-1/phase-6 visual-audit tests
  passed across compact, laptop, and large sizes; all 6 Schedules geometry
  cases passed; and all 43 Schedules, CSS-integrity, and duplicate-task
  sentinel tests passed. The direct shell/selector checks and original compact
  failure also passed. Ruff, formatting, `compileall`, and diff checks passed.
- The original implementation received independent review. A final
  reconciliation self-review found and corrected the stale direct-child CSS
  selector before closeout.
<!-- SECTION:NOTES:END -->
