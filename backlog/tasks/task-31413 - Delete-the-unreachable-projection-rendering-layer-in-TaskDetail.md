---
id: TASK-31413
title: Delete the unreachable projection-rendering layer in TaskDetail
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-04 22:39'
updated_date: '2026-09-05 22:00'
labels:
  - scheduling
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`TaskDetail` still composes and display-toggles a `#scheduling-task-detail-legacy-fields` container (the Type/Schedule rows) for a `ScheduledTask` projection, plus the `is_reminder` branch in `set_task` that exists only to hide or show it. No projection can reach that code.

Redesign PR-4's final review (ruling 5) established the unreachability argument when it deleted the sibling task-23106 ownership line, and the argument is recorded verbatim at `task_detail.py:1425`: `TaskDetail(` is constructed in exactly TWO places in the repo, both in `schedules_workbench.py` (the docked detail pane and task 6's per-push overlay instance); both are fed through the SAME seam, `_update_detail_for_index` (`_detail_panes` is one list, not a second data path), whose data comes from `load_tasks` -> `list_tasks(owner_id=None, include_projections=False)` filtered to `ReminderTask`, and which asserts `isinstance(task, ReminderTask)` before every call.

The same argument retires the projection-rendering layer: the legacy container is composed on every mount and painted never. It is not merely unused — it is provably unreachable, which is the standard ruling 5 set for deleting rather than deprecating.

Scope guard: the live `None`/assert guards in the `task_detail.py:897-1098` region (10 by grep at `b7f8efde73`) are NOT part of this layer. They guard row references taken before mount, they fire on the real reminder path, and they must survive the deletion untouched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The #scheduling-task-detail-legacy-fields container and its Type/Schedule rows are gone from TaskDetail's widget tree
- [x] #2 The is_reminder display-toggling branch that existed only to hide or show that container is gone with it
- [x] #3 A test asserts the unreachability claim (TaskDetail is only ever fed a ReminderTask) rather than restating it in a comment
- [x] #4 The pre-mount None/assert guards in the task_detail.py:897-1098 region still guard the real reminder path
- [x] #5 The scheduling UI suite and the boot-css and bare-type ratchets stay green, with the freed bytes banked rather than left as headroom
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify unreachability at HEAD (TaskDetail construction sites, _update_detail_for_index assert)
2. Delete the #scheduling-task-detail-legacy-fields container from compose()
3. Delete the is_reminder toggle line for that container in set_task(), keeping the groups toggle and the ~6 live None-guards untouched
4. Delete the now-dead _update_static(type/schedule) calls and their _task_type_label/_task_schedule_label helpers
5. Add a runtime assert + tightened type hint at set_task's boundary so a ScheduledTask can never silently regress in
6. Add a test pinning the assert (test_set_task_rejects_a_scheduled_task_projection) and fix the one test that asserted on the deleted container
7. Run the scheduling UI suite, boot-css ratchet, and bare-type ratchet
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-verified unreachability at HEAD (87759522f): TaskDetail( is constructed at exactly schedules_workbench.py:716,1810, both fed through _update_detail_for_index, which asserts isinstance(task, ReminderTask) (schedules_workbench.py:1946) before ever calling set_task -- a ScheduledTask projection provably never reaches TaskDetail.

Deleted from task_detail.py: the #scheduling-task-detail-legacy-fields Vertical (Type/Schedule Horizontal rows) from compose(); the is_reminder toggle line that hid/showed it in set_task (the groups-display toggle and the ~6 live isinstance/None-guards at :897-1098 were left untouched, per the scope guard); the now-orphaned _update_static calls for the Type/Schedule Statics; and the _task_type_label/_task_schedule_label helper functions that became dead once those calls were gone (their underlying _humanize_schedule_kind/_humanize_schedule stay in use elsewhere).

Hardened the boundary instead of just deleting: tightened set_task's task param from ReminderTask | ScheduledTask | None to ReminderTask | None and added a runtime assert isinstance(task, ReminderTask) right after the None-check, so a future regression fails loudly instead of silently painting a near-empty pane. Added test_set_task_rejects_a_scheduled_task_projection (Tests/UI/test_schedules_workbench.py) pinning that assert directly, per AC3 -- fixed the one existing test that queried the now-deleted container (test_task_detail_renders_selected_task), and refreshed two stale comments that referenced the deleted container/general-purpose claim.

Verified: full scheduling UI suite (test_schedules_workbench.py + 7 sibling files) 296 passed; boot-css-bytes ratchet passed unchanged (786624/804000, headroom 17376 -- no CSS rule keyed off the deleted ids, so there were no bytes to bank); bare-type-rule-count ratchet passed. Tests/Architecture/test_persistent_diagnostic_inventory.py has 2 pre-existing failures (test_inventory_excludes_nested_virtualenv_but_keeps_application_sources, a task_31551_calls schema-key drift) confirmed present at HEAD via a throwaway detached worktree before this change -- unrelated to task-31413, not touched.

Modified files: tldw_chatbook/UI/Screens/scheduling/task_detail.py, Tests/UI/test_schedules_workbench.py.
<!-- SECTION:NOTES:END -->
