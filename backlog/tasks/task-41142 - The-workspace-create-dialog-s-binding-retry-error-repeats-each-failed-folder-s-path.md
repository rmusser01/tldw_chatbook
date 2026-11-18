---
id: TASK-41142
title: "The workspace create dialog's binding-retry error repeats each failed folder's path"
status: Done
created_date: 2026-11-15 02:20
updated_date: 2026-11-15 19:27
assignee:
- '@claude'
labels:
- follow-up
- settings
- workspaces
- copy
priority: low
references:
- tldw_chatbook/Widgets/workspace_create_modal.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A long path printed twice per failed folder pushes the retry action further down a dialog that already needed a scroll fix at 80x24 (TASK-40318).

**Evidence.** `workspace_create_modal.py:541-545` builds the error as `f"{folder}: {message}"` per failed folder, and `message` is `str(exc)` from the binding attempt (`:519`), which the round-14 Settings stream reports already contains the path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each failed folder's path appears once in the retry error
- [x] #2 A test with a failing binding checks the error text, red before the change
<!-- AC:END -->

## Implementation Plan

1. Confirm which binding errors already name the folder (`_validate_folder_path_rules`, `_validate_folder_overlap`, `save_runtime_binding` in `Workspaces/registry_service.py`).
2. Pin the retry error in the laned journey `test_partial_create_keeps_the_committed_workspace_clear` (80x24): the failed folder's path appears exactly once. Show it red.
3. In `_finish_or_retry_bindings`, print the message alone when it already contains the folder; keep the `folder: ` prefix otherwise.

## Implementation Notes

- **Approach.** `_finish_or_retry_bindings` (`tldw_chatbook/Widgets/workspace_create_modal.py`) now writes each failure as `message` when the message already contains the folder, and as `folder: message` otherwise. The folders in the dialog are already resolved locators (`_add_folder` stores `str(resolved)`), and the errors that name a path name the resolved one, so a substring check is exact. Not every binding error names the path: the filesystem-root and home-directory refusals, the Default-workspace refusal and `WorkspaceNotFound` do not, so those keep the prefix and still show the path once.
- **Test.** `Tests/UI/test_settings_workspace_lifecycle_journeys.py::test_partial_create_keeps_the_committed_workspace_clear` (all three params; whole-file UI census entries) now asserts the failed folder's path appears once in `#workspace-create-error`. Red before the change: `AssertionError: assert 2 == 1` (the error read `<path>: Folder does not exist or is not a directory: <path>`). Green after.
- **Runs** (run.sh, load 50-110 on 18 CPUs): the journey file, `Tests/Workspaces/test_workspace_create_modal.py` and `Tests/UI/test_workspace_persona_creation.py` at `-n 4`: 41 passed, 1 failed. The failure, `test_workspace_persona_creation.py::test_console_bootstrap_workspace_activation_and_details_use_target_default` (`NoMatches '#workspace-default-persona'` in the private child), is intermittent and not this change: it passed serially both with this branch's sources and with the base sources (82.7 s and 93.2 s at load ~110).
- No User Guide page describes the retry error's format, so no doc change.
- Files: `tldw_chatbook/Widgets/workspace_create_modal.py`, `Tests/UI/test_settings_workspace_lifecycle_journeys.py`.
