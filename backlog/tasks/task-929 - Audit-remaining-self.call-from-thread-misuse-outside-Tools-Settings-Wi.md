---
id: TASK-929
title: >-
  Audit remaining self.call_from_thread misuse outside Tools_Settings_Window
status: To Do
assignee: []
created_date: '2026-07-27 09:00'
labels:
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While fixing TASK-899 it emerged that all four database-maintenance workers called `self.call_from_thread(...)` on `ToolsSettingsWindow`, which is a `Container`. That method exists only on `App` — verified: `Widget` and `Container` both lack it. Every notification raised from those worker threads would therefore have raised `AttributeError` rather than notifying, which is why the failures were never seen.

All 39 call sites in that file now use `self.app.call_from_thread`, and a guard test asserts the bare form does not come back. That guard is file-scoped.

The same mistake is plausible anywhere a `Widget`/`Container` subclass runs a threaded worker, and it is invisible until the error path executes. Sweep the codebase for `self.call_from_thread` on non-`App` classes and fix or clear each.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Every `self.call_from_thread` call site outside `App` subclasses is identified
- [ ] Each is fixed to use `self.app.call_from_thread` or confirmed to be on an `App`
- [ ] A repo-wide guard replaces or supplements the file-scoped one
<!-- AC:END -->
