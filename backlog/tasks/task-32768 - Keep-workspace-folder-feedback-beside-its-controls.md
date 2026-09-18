---
id: TASK-32768
title: Keep workspace folder feedback beside its controls
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 04:10'
updated_date: '2026-09-18 04:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At compact size, adding an invalid workspace folder leaves the error in an offscreen status above the selected card. Users need visible feedback and a retained correction path beside the folder controls, while immediate folder-access semantics remain unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Folder validation and operation outcomes remain visible beside the controls in compact and wide layouts and both themes.
- [x] #2 Invalid or failed adds retain the editable path; successful add defaults to read-only, access changes and removal reach the real registry, and refusal can be retried.
- [x] #3 Feedback stays scoped to the selected workspace and control focus remains usable after refresh and removal.
- [x] #4 Targeted regressions and representative native private-profile evidence record the repair, including existing workspace lifecycle behavior and remaining review limits.
- [x] #5 Bound-folder paths and read-only/read-write labels render literally and remain visible after selecting the workspace again.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/028-settings-workspaces-category-and-folder-roots.md, backlog/decisions/033-settings-commit-models-three-honestly-labeled.md, backlog/decisions/150-design-token-system-and-design-language.md, backlog/decisions/161-component-pattern-library.md (existing). Reason: repair the existing inline-feedback and focus contract without changing folder authority, commit semantics or storage. 1. Preserve the compact red capture and failing compositor assertion; inspect the approved Workspace management spec and prior lifecycle tasks. 2. Place folder-operation feedback beside its controls using the existing token-backed status pattern, keep it workspace-scoped and retain invalid input. 3. Exercise real keyboard invalid/retry/add/access/remove flows in both sizes/themes, fresh registry reads, selection changes and focus after recomposition; preserve original assertions with private-profile ownership where needed. 4. Run targeted existing lifecycle/regression checks and scoped static/governance checks, inspect native private-profile captures with clean lifecycle, and record independent review and remaining Workspaces subflow boundaries. Allocation scan: 273 refs and30 worktrees, max32767; CLI probe offered32768.

The keyboard matrix also reproduced a collapsed workspace list (height 1 for 3 rows), clipping selection targets before the parent could scroll them. Set only settings-workspaces-list to natural height and rebuild the generated bundle; the unchanged full journey is the regression gate.

Native inspection also found the folder row interprets [ro]/[rw] and bracketed paths as Rich markup. Render the row as literal text and assert painted access labels and bracketed fixture names before refreshing native evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Folder outcomes now use token-backed local status rows, retain rejected drafts, and remain scoped to the workspace/binding. The workspace list uses natural height, folder paths/access labels render literally, and removal restores focus on the replacement Add after pane rebuild. A native focus race is documented in lessons-testing-evidence. Verified 37 distinct targeted cases: four folder journeys, 20 existing workspace cases and 13 token/CSS guards; repeated removal cases counted once. Scoped static checks and baseline comparison introduce no new diagnostics. Native run005 covers dark/light at 80x24 and 170x48 with twelve inspected captures, real registry access/removal, literal bracketed paths, live Tab routing, normal Ctrl+Q exit, 11 healthy private databases and unchanged default-profile fingerprints. Independent review found no remaining blocker. QA: Docs/superpowers/qa/2026-09-17-settings-workspace-folders/README.md. Existing ADR-028/033/150/161 apply; no new ADR. Assistant defaults, Change Review journeys and workspace lifecycle modals remain separate reviews; the broader workstream and draft PR merge gate remain open.
<!-- SECTION:NOTES:END -->
