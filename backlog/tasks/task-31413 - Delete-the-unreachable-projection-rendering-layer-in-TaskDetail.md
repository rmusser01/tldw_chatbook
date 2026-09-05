---
id: TASK-31413
title: Delete the unreachable projection-rendering layer in TaskDetail
status: To Do
assignee: []
created_date: '2026-09-04 22:39'
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
- [ ] #1 The #scheduling-task-detail-legacy-fields container and its Type/Schedule rows are gone from TaskDetail's widget tree
- [ ] #2 The is_reminder display-toggling branch that existed only to hide or show that container is gone with it
- [ ] #3 A test asserts the unreachability claim (TaskDetail is only ever fed a ReminderTask) rather than restating it in a comment
- [ ] #4 The pre-mount None/assert guards in the task_detail.py:897-1098 region still guard the real reminder path
- [ ] #5 The scheduling UI suite and the boot-css and bare-type ratchets stay green, with the freed bytes banked rather than left as headroom
<!-- AC:END -->
