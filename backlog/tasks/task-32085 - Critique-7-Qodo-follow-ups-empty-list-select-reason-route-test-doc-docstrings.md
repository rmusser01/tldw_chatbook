---
id: TASK-32085
title: >-
  Critique #7 Qodo follow-ups: empty-list select reason, / route test, doc +
  docstrings
status: Done
assignee: []
created_date: '2026-09-08 18:01'
updated_date: '2026-09-08 18:35'
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
- [x] #1 The zero-selection reason is shown only when nothing is selected AND at least one selectable row exists; a successful empty list in select mode shows no 'select items' message (kept mounted for layout), with a test for an empty refresh while select mode is active
- [x] #2 An integration test opens the Prompts canvas, presses /, and asserts focus lands on the Prompts filter (not the rail search)
- [x] #3 The two size-parametrized keyboard tests carry a one-line summary + Args: describing the size tuple
- [x] #4 The shortcut guide's `/` description matches the actual per-canvas routing (no over-claimed fallback)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Consolidated critique #7 Qodo follow-ups. (a, BUG) The zero-selection bulk-action reason (task-32045) is now VISIBLE only when `selected_count==0 AND rendered_count>0 AND not bulk_list_failed` -- a successful EMPTY media list in select mode paints nothing (widget stays mounted via visibility for layout); red-first pin + the empty-refresh case. (b) Added a Prompts `/`-route integration test (seeds a real prompt, opens the canvas, / focuses #library-prompts-filter, not the rail search) -- the route from task-32046 was Media-only pinned. (c) Args: docstrings on the two size-parametrized keyboard tests. (d) library.md `/` fallback reworded to describe the ROUTING (`/` isn't wired to Skills/Collections/Search-RAG/Study, so it focuses the rail search) rather than falsely claiming those canvases lack a filter -- they DO have filter/query inputs, / just doesn't route there (fixed at review after the first wording over-claimed). (e) CORRECTED a controller misdiagnosis: the media scroll-restore pins' RED was NOT a test-fixture backend gap (`Local prompt/study/quiz backend is unavailable` is caught-and-logged noise at library_screen.py:13083, tolerated) -- it was a STALE-ATTRIBUTE migration miss in test_library_media_reader_scroller_resolution.py (`screen._library_media_reader_session/_content_mode/_read_scroll_by_id` -> `screen._media_state.*`, the real LibraryMediaState fields; LibraryScreen defines no such bare attrs). 6-line test-only fix; the file is 6/6 green and the crit6 scroll-restore pins (task-31968) pass again. No production change for (b)-(e); the only production file touched is library_media_canvas.py (item a). Files: library_media_canvas.py, Tests/UI/test_library_shell.py, Tests/UI/test_library_media_render_fixes.py, Tests/UI/test_library_media_reader_scroller_resolution.py, Docs/User_Guide/library.md.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Renumbered from TASK-32047 to TASK-32085 during the Console logging/trace PR update to dev on 2026-09-08. The logging task was created at 16:58 UTC; this Library task was created at 18:01 UTC and added by f7a85ded21a5f0f80ee8925360d53d62ec7c4c46. The repository older-keeps-ID rule applies regardless of Done status. Updated the Library guide and its code/test references; no Library behavior changed.
