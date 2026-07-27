---
id: TASK-944
title: Repair Console test fixture after runtime backend became read-only
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 16:39'
updated_date: '2026-07-27 22:01'
labels:
  - tests
  - console
  - baseline
dependencies: []
references:
  - Tests/UI/test_screen_navigation.py
  - Tests/UI/test_console_native_transcript.py
  - Tests/UI/test_console_tick_gating.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the shared Console navigation test harness after current_runtime_backend became a read-only derived property on dev. The fixture must configure runtime state through the supported owner instead of assigning the property directly, without changing production runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console native-transcript and tick-gating tests construct TldwCli without assigning current_runtime_backend
- [x] #2 The affected tests pass against current dev without changing production runtime ownership
- [x] #3 The repair remains isolated from TASK-553.16 citation implementation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the current fixture failure in the three affected Console UI test files.
2. Update only the shared test harness to configure the runtime backend through its supported owner.
3. Rerun only those three test files, then run Ruff on the touched test file and git diff --check.
4. Review the diff, record verification, complete the acceptance criteria, and mark the task Done.

ADR required: no
ADR path: N/A
Reason: Test-fixture-only repair using the existing runtime ownership contract; no production behavior or architecture changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
No implementation change was needed. Commit 384623d5b (`test(ui): update mounted app runtime harness`) had already repaired the shared `_build_test_app()` fixture on current `dev` by replacing direct `current_runtime_backend` assignment with `_publish_runtime_policy_projection(context.state)`, the existing runtime-policy owner path.

Fresh scoped verification passed all 83 native-transcript and tick-gating tests; the only output was the repository baseline RequestsDependencyWarning. Static inspection found no `current_runtime_backend =` assignment in any of the three task-referenced files. A broader initial run of all three files produced 138 passes plus one unrelated stale navigation test (`test_deferred_initial_tab_uses_first_run_home_route`) that calls the removed `_set_initial_tab` method; that separate baseline is tracked by TASK-1078 and was not changed here.

The plan correctly stopped before its conditional fixture edit because the required change was already present. No Python file changed, so a touched-code Ruff run was not applicable; task-document diff validation replaced it. TASK-1078 was allocated after the final rebase because current `dev` already owned TASK-1077.

ADR required: no. ADR path: N/A. Test-fixture closeout only; runtime ownership and production behavior are unchanged.
<!-- SECTION:NOTES:END -->
