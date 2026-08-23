---
id: TASK-21119
title: >-
  Every Chat-screen press walks the whole screen DOM twice for selection-menu
  dismissal
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 17:28'
labels:
  - performance
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21119).

`chat_screen.py:18939-18990`: `_dismiss_console_selection_menus_outside_transcript` runs
`self.query(ConsoleTranscript)` and `self.query(ConsoleSelectionMenu)` - two full-screen DOM
traversals - and is invoked on BOTH on_mouse_down and on_click of the same physical press
(~4 traversals per click) on the largest-DOM screen in the app. A direct contributor to the
click-lag symptom on every click.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dismissal early-returns via a mounted-menu flag/registry (at most one menu is ever mounted) and a cached transcript reference - no full-screen queries when nothing is mounted
- [ ] #2 Selection-menu dismissal behavior is unchanged (covered by existing selection tests)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the real per-press cost with a counter probe over the production Console pilot (instrument screen.query, count ConsoleTranscript/ConsoleSelectionMenu walks per physical press) -- red-first.
2. Add constructor-registered candidate registries (WeakSet) for ConsoleSelectionMenu and ConsoleTranscript; re-derive attachment from the live DOM at read time so the registry can over-report but never miss a mounted node.
3. Add SelectionManager.is_idle + ConsoleTranscript.has_pending_selection_ui so the screen handler can prove its per-transcript cleanup is a no-op (keyboard-selection state has no menu but must still clear).
4. Rewrite _dismiss_console_selection_menus_outside_transcript to gate on the registries and early-return before any DOM work; keep the ancestor guard and the removal semantics identical.
5. Route ConsoleTranscript._attached_selection_menus through the same registry (it made a third full-screen walk on every in-transcript press).
6. Control arms: menu mounted on the screen still dismissed; selection-without-menu still cleared; in-transcript press still left alone. A/B every red against the base.
<!-- SECTION:PLAN:END -->
