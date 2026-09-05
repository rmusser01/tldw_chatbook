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
- [ ] #1 A base screen's on_unmount body runs exactly once per unmount for every screen that overrides it
- [ ] #2 A regression test pins the count directly, so a future re-introduction fails rather than staying invisible
- [ ] #3 One convention is chosen and applied to every site (either no explicit super call, or a base whose body is not itself a handler), and the choice is recorded where the next author will read it
- [ ] #4 The audit covers every super().on_unmount() and the same MRO question for on_mount and on_screen_resume, and records which are affected
- [ ] #5 The lessons-textual entry documents the mechanism with the probe result, not just the rule
<!-- AC:END -->
