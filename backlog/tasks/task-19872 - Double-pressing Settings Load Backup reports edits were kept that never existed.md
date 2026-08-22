---
id: TASK-19872
title: >-
  Double-pressing Settings Load Backup reports edits were kept that never existed
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - ux
  - settings
  - concurrency
priority: low
dependencies:
  - TASK-19559
---

## Description

Source: surfaced by **TASK-19559**'s reviewer while checking the arrival-time
guards that task introduces.

**Precondition, stated up front: this does not reproduce on `dev` today.** The
copy it describes ("unsaved edits were kept") and the comparison that produces
it are introduced by TASK-19559, which is still in flight. At `3605bd52d`
`_advanced_load_backup_worker` (`UI/Screens/settings_screen.py:8601`) is a plain
`@work(exclusive=True, thread=True)` with no such guard. This is filed now so
the observation survives that task's fix round.

The shape: TASK-19559 replaces the worker's exclusivity with an arrival-time
guard — before applying a loaded backup, the callback compares the editor's
current text against what it expects, and declines to overwrite if the user has
typed something in the meantime. That is the right instinct. But the guard
compares against the editor's *live* content, and the first callback's own
write into the config `TextArea` is indistinguishable from a user edit.

So on a rapid double-press of "Load Backup":

1. two workers are dispatched
2. the first completes and writes the backup text into the editor
3. the second completes, compares, sees text it did not expect, and declines —
   reporting that the user's unsaved edits were preserved

The user had no unsaved edits. The application invented a reason for declining,
and told the user about work it protected that never existed.

**Reasoned from the code, not driven.** The sequence above follows from reading
the callback and the worker dispatch; it has not been reproduced in a running
app. Whoever picks this up should confirm the interleaving before fixing it.

No data is at risk — declining is the safe direction, and the backup text is
already in the editor from the first callback. The defect is that the message
is false, which is the same family as TASK-19550 / TASK-19861 / TASK-19869:
*the app describes an outcome it did not produce.*

## Acceptance Criteria

- [ ] Pressing "Load Backup" twice in quick succession does not report that
      unsaved edits were kept when the user made none
- [ ] The guard distinguishes a write the application itself made from a genuine
      user edit
- [ ] A real unsaved user edit is still protected — a backup load that would
      overwrite typed-but-unsaved config text still declines and still says so
- [ ] A test drives both cases (double-press with no user edit; single press
      with a pending user edit) and asserts the message, and is
      mutation-checked
- [ ] The interleaving is reproduced before the fix is written, and what was
      observed is recorded — this task was reasoned from code, not driven

## Notes

Low severity and deliberately so: the outcome is correct, only the explanation
is wrong. It is worth fixing because a spurious "we protected your edits"
teaches users to distrust the message on the occasion when it is true.
