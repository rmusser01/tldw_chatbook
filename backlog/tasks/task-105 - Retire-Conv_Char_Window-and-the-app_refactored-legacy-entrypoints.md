---
id: TASK-105
title: Retire Conv_Char_Window and the app_refactored legacy entrypoints
status: Done
assignee: []
created_date: '2026-06-11 23:46'
updated_date: '2026-06-26 16:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CCPWindow (Conv_Char_Window.py) is reachable only from app_refactored_v2.py's lazy fallback, which has no in-package consumers; app_refactored.py and tldw_chatbook/navigation/screen_registry.py are similarly dead. Retire them together after confirming nothing external depends on the alternate entrypoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No dead alternate entrypoints remain,CCP_Modules TYPE_CHECKING hints updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/004-personas-destination-native-workbench.md; backlog/decisions/007-personas-workbench-route-consolidation.md
Reason: Existing ADRs already accept retiring legacy CCP route/window surfaces; this task removes dead alternate app entrypoints and type-only references without changing the active app contract.

1. Confirm no packaged entrypoint or in-package runtime consumer depends on tldw_chatbook.app_refactored, tldw_chatbook.navigation, or tldw_chatbook.UI.Conv_Char_Window.
2. Add regression guards that retired modules are absent, CCP handler type-only references point at PersonasScreen, and the active lazy registry still resolves the ccp alias to PersonasScreen.
3. Delete the retired alternate entrypoint modules/backups and update CCP_Modules TYPE_CHECKING annotations.
4. Run focused tests, import checks, and git diff hygiene before marking the task done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retired the dead alternate app entrypoints and legacy CCP window file set: removed tldw_chatbook.app_refactored, the old tldw_chatbook.navigation package, Conv_Char_Window.py, and its backup. Updated reused CCP handler TYPE_CHECKING annotations and stale handler docs to target PersonasScreen, preserving the active UI/Navigation lazy registry and ccp -> Personas route. Added Docs/superpowers/plans/2026-06-26-retire-legacy-entrypoints.md and Tests/UI/test_legacy_entrypoints_retired.py to guard the retirement and active alias. PR review fixes added test docstrings, contextual error logging, idempotent PersonasScreen setup for reused CCP enhancements, and silent background refresh loading notifications so startup refreshes do not look like user action success. Verification: 218 focused UI tests passed; import guard confirmed app_refactored/navigation specs are absent and ccp resolves to personas; git diff --check passed.
<!-- SECTION:NOTES:END -->
