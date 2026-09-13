---
id: TASK-31911
title: Restore Skills shadow-name coverage for current runtime and Console commands
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 20:03'
updated_date: '2026-09-05 20:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The resumed dev sweep shows that fifteen registered runtime tools and Console commands bypass the existing skill-name collision warning. Restore the fixed reserved-name contract without weakening the four-source drift guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every currently registered runtime, catalog, gated tool, and Console command name is covered by the existing shadow-name guard
- [x] #2 The fifteen newly uncovered names are recognized through both exact and normalized user input while ordinary skill names remain allowed
- [x] #3 Complete affected tests and static checks pass with no guard exemptions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the full Skills state baseline and read the prior drift-guard task.
2. Add literal behavior regressions and verify they fail before restoring only the missing reserved names.
3. Run the complete Skills state and related guard selections, lint and formatting; record evidence and review the scoped diff.
ADR required: no
ADR path: N/A
Reason: Routine correction to the existing reserved-name contract established by TASK-580; no new security or runtime boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored exactly15 missing names in the existing fixed reserved-name set:3runtime tools and12Consolecommands. Added literal exact/normalized/nearby-noncollision behavior cases; the four-source runtime/catalog/gate-table/command drift guard is unchanged. Baseline27passed1failed; added regression produced27passed2failed. State file now29passed; initial complete state+real-service flow51passed1failed exposed unrelated bootstrap browse refresh, reproduced at pre-assembly commitf8e63aea4f and separately repaired under TASK-31751. Final complete state+flow53passed including new retained-draft journey. Whole changed-file Ruff/format and whitespace checks pass. No dependencies/runtime registry or trust boundary changed; ADR not required (existing TASK-580 fixed-name contract). Files:library_skills_state.py and its complete state test file.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31748 was renumbered to TASK-31911 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
