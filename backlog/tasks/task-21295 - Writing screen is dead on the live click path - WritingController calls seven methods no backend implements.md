---
id: TASK-21295
title: >-
  Writing screen is dead on the live click path - WritingController calls seven
  methods no backend implements
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - bug
  - writing
  - ui
dependencies: []
priority: high
---

## Description

Source: TASK-21125 review (perf work on the same screen). `WritingController`
was written against an interface that neither writing backend provides. Seven of
the methods it calls exist on **none** of `WritingScopeService`,
`LocalWritingService`, or `ServerWritingService`, and none of the three defines
`__getattr__`, so each call raises `AttributeError` at runtime rather than a
handled "unsupported" result.

The missing names, with the `UI/Writing_Modules/writing_controller.py` line that
calls each:

| controller line | method called |
| --- | --- |
| 64 | `get_project_structure` |
| 107 | `autosave_scene` |
| 215 | `assign_chapter` |
| 232 | `reorder_items` |
| 247 | `move_scene` |
| 266 | `search_project` |
| 313 | `restore_version_to_working_state` |

The first one is on the **live click path and is unguarded**: selecting a
project in the sidebar list runs
`Writing_Window._handle_project_selected` (`UI/Writing_Window.py:295-299`) →
`WritingWindow.load_project_structure` (`UI/Writing_Window.py:85-93`) →
`WritingController.load_project_structure`. Neither the handler nor the window
method has a `try/except`, so the `AttributeError` escapes a Textual message
handler. Every screen path that depends on the outline — which is the screen's
whole purpose — is therefore unreachable in the shipped app; only
`load_projects` (which calls `list_projects`, a method that does exist) works.

The reason this is not visible in CI: `Tests/UI/test_writing_screen.py` drives
the controller through `FakeWritingScopeService`, a test double that implements
all seven. The mounted-app tests are green over an API the app does not wire.

Scope note: TASK-21125 (held connection + thread offload) deliberately left this
unchanged. Its `_ThreadOffloadedBackend` proxy forwards attribute lookups with
`getattr(self._backend, name)`, so a missing method still raises `AttributeError`
exactly as before — the proxy neither hides nor worsens this.

## Acceptance Criteria

- [ ] Selecting a project in the Writing sidebar loads its outline in the shipped app (no `AttributeError` escapes a Textual handler)
- [ ] Each of the seven controller-called methods either exists on the wired backend(s) or is removed from the controller, with the affected UI affordance disabled through the existing `get_capability` / unsupported-reason seam rather than failing at call time
- [ ] A test drives at least the project-select → outline path against the REAL `WritingScopeService` + `LocalWritingService` (not `FakeWritingScopeService`), so the wiring gap cannot reopen silently
- [ ] `WritingWindow.load_project_structure` and its handler degrade to a status message on backend failure, as `_handle_outline_node_selected` already does
