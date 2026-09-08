---
id: TASK-32047
title: >-
  Critique #7 Qodo follow-ups: empty-list select reason, / route test, doc +
  docstrings
status: To Do
assignee: []
created_date: '2026-09-08 18:01'
labels:
  - library
  - media
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidated Qodo follow-ups from the critique #7 fix PRs. (a, BUG) The zero-selection bulk-action reason 'Select items to enable.' (task-32045) shows whenever selected_count==0 without checking whether any rows exist, so a SUCCESSFUL empty media list in select mode asks the user to select from nothing. (b) The `/`-to-canvas-filter fix (task-32046) routes Media AND Prompts but only Media is pinned; the Prompts route is untested. (c) The two new size-parametrized keyboard tests lack Google-style docstring Args for the size tuple. (d) The shortcut guide (library.md) over-claims the `/` fallback for canvases without their own filter. Also fold in the media test-app fixture gap that leaves several media pins red for lacking local prompt/study/quiz scope-service backends, IF that fix is localized to the media test app builder; otherwise leave it as its own rider.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The zero-selection reason is shown only when nothing is selected AND at least one selectable row exists; a successful empty list in select mode shows no 'select items' message (kept mounted for layout), with a test for an empty refresh while select mode is active
- [ ] #2 An integration test opens the Prompts canvas, presses /, and asserts focus lands on the Prompts filter (not the rail search)
- [ ] #3 The two size-parametrized keyboard tests carry a one-line summary + Args: describing the size tuple
- [ ] #4 The shortcut guide's `/` description matches the actual per-canvas routing (no over-claimed fallback)
<!-- AC:END -->
