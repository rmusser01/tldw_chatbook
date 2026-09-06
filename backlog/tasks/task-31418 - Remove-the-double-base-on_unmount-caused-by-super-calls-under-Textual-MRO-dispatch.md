---
id: TASK-31418
title: >-
  Remove the double base on_unmount caused by super calls under Textual MRO
  dispatch
status: To Do
assignee: []
created_date: '2026-09-04 22:41'
labels:
  - textual
  - cleanup
  - reliability
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Textual dispatches a lifecycle handler by walking the MRO and calling EVERY distinct implementation it finds (`MessagePump._get_dispatch_methods`: `for cls in self.__class__.__mro__`). A subclass that both overrides `on_unmount` AND calls `super().on_unmount()` therefore runs the base implementation TWICE — once from its own explicit call, once from Textual's own walk.

Probed against the installed Textual (8.2.8) with a two-level `Screen` subclass: the base handler fires twice per unmount (`['child', 'base', 'base']`).

The pattern is repo-wide, not scheduling-specific — `super().on_unmount()` appears in at least eight screens and widgets, including `schedules_workbench.py:712`, `chat_screen.py`, `library_screen.py`, `personas_screen.py`, `artifacts_screen.py`, `logs_screen.py`, `watchlists_collections_screen.py` and two Console modals. Found during the schedules-handoff PR-6 Task 4 review, where the workbench's new observer-stopping `on_unmount` added another instance.

It is harmless TODAY because `BaseAppScreen.on_unmount` (`UI/Navigation/base_app_screen.py:377`) happens to be idempotent: it increments a generation counter, nulls a view reference, releases an interaction capture on a reference it just nulled, and logs. The second call is a no-op plus a duplicate log line. That is a property of the current base body, not a guarantee — the next non-idempotent teardown added there (a close, a release, a decrement, a dispatch) becomes a double-teardown bug in every one of these screens at once, and the symptom will appear far from the line that caused it.

This is a convention fix, not a bug fix: pick one discipline and make it uniform.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A base screen's on_unmount body runs exactly once per unmount for every screen that overrides it
- [x] #2 A regression test pins the count directly, so a future re-introduction fails rather than staying invisible
- [x] #3 One convention is chosen and applied to every site (either no explicit super call, or a base whose body is not itself a handler), and the choice is recorded where the next author will read it
- [x] #4 The audit covers every super().on_unmount() and the same MRO question for on_mount and on_screen_resume, and records which are affected
- [x] #5 The lessons-textual entry documents the mechanism with the probe result, not just the rule
<!-- AC:END -->

## Implementation Notes

**Approach:** Applied the repo's existing lifecycle convention (already in force
for `on_mount` from prior work) to `on_unmount`: a subclass handler for an
MRO-dispatched lifecycle event does NOT call `super().on_*()`, because Textual's
`_get_dispatch_methods` walks the MRO and invokes the base handler itself — an
explicit `super()` call runs the base body a second time.

**Probe (Textual 8.2.8):** a two-level `Screen` subclass appending its name in
each `on_unmount` yields `['child', 'base', 'base']` with an explicit
`super().on_unmount()`, and `['child', 'base']` without it. The same double-fire
reproduced for `on_mount` and `on_screen_resume` — all three MRO-dispatched
handlers are affected (AC#4).

**Sites:** removed the redundant `super().on_unmount()` from every subclass that
carried it — `BaseAppScreen` subclasses (`chat_screen`, `library_screen`,
`personas_screen`, `artifacts_screen`, `logs_screen`, `meetings_screen`,
`watchlists_collections_screen`, `schedules_workbench`) and the two
`SafeModalDismissMixin` Console modals (`console_workspace_files_modal`,
`console_session_switcher_modal`), each replaced with a `# No super().on_unmount()`
comment naming the base whose handler Textual dispatches separately. Zero
`super().on_unmount()` calls remain in the package.

**Classification:** every removed `super().on_unmount()` was verified redundant —
its base (`BaseAppScreen.on_unmount` or `SafeModalDismissMixin.on_unmount`, both
defined in their own class `__dict__`) is separately MRO-dispatched, so the base
teardown still runs exactly once after removal. No `on_unmount` site was
load-bearing. (Correction after review: there is no "safe `super().on_*()` to a
dispatched handler" — an explicit call always double-fires; the genuine
run-once-with-explicit-invocation pattern is `BaseWizard`'s plain
`_post_mount_hook()`, not `super()`. The repo's remaining `super().on_mount()`
calls — `change_review_screen.py`, the two Console modals, ~19 total — are latent
instances of the same bug on the mount side, out of this `on_unmount`-scoped task
and filed as a follow-up.)

**Guard (AC#2):** `Tests/UI/test_on_unmount_mro_convention.py` — a runtime count
test pins the base `on_unmount` firing exactly once under the no-super
convention (revert-checked: it fires twice with `super()`), plus an AST scan
that fails if any `on_unmount` re-introduces a `super().on_unmount()` call.

**Docs (AC#3/#5):** the mechanism, the probe result, and the convention
are recorded in `BaseAppScreen`'s docstring and in a new
`backlog/docs/lessons-textual.md` section.

**Verification:** guard tests pass (2 passed); all 11 touched modules import
cleanly; the console-modal suites pass except 6 tests in
`test_console_modal_dismissal.py` that fail identically at the pre-Task-4 base
`564e34cbc` (pre-existing branch debt, unrelated to this change — verified in a
throwaway worktree).

**Modified/added files:** `UI/Navigation/base_app_screen.py`,
`UI/Screens/{chat,library,personas,artifacts,logs,meetings,watchlists_collections}_screen.py`,
`UI/Screens/scheduling/schedules_workbench.py`,
`Widgets/Console/{console_workspace_files_modal,console_session_switcher_modal}.py`,
`backlog/docs/lessons-textual.md`, `Tests/UI/test_on_unmount_mro_convention.py` (new).
