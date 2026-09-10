---
id: TASK-32106
title: >-
  Library Notes editor: Tab burst appends the body to the title; keywords field
  lacks the focused-field guard; editor-owned refresh skip is unpinned
status: Done
assignee:
  - '@claude'
created_date: '2026-09-08 22:43'
updated_date: '2026-09-10 16:56'
labels:
  - library
  - notes
  - bug
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-32062 fix rounds (PR #2531): with title + Tab + body typed in one uninterrupted burst, Tab's focus move lands after the burst and the body text is appended to the title (nothing is lost; 1 s gaps behave); `apply_session_state` writes `wide_keywords.value` with no `has_focus` guard (`library_notes_canvas.py:1687-1689`) — the same stale-snapshot clobber class as the title bug one field over, and `_NOTE_EDITOR_INPUT_IDS` excludes keywords; nothing pins that the editor-owned refresh skip is instance-scoped to the work pane (moving the guard to the screen would freeze the list silently); the list scroll-offset re-apply is skipped while the editor owns focus. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A title, Tab, body burst lands the body in the body field
- [x] #2 The keywords field is treated as its own authority while focused, like title and body
- [x] #3 A test pins that a sync while the editor has focus still repaints the list pane
- [x] #4 The list scroll offset survives a sync that lands mid-edit at compact widths
<!-- AC:END -->

## Critique #9 evidence (2026-09-10)

Assessor B: the Notes list does not live-update a note's title while its editor is open (D13). Related to the editor-owned refresh skip this task tracks; include it in the pin.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify each premise against current dev (peer Notes wave 1 landed).
   - AC#2 ALREADY SHIPPED: commit 97626354ee ('the keyword boxes are editor fields', PR #2531 review) put both keyword ids in _NOTE_EDITOR_INPUT_IDS and gave each assignment the same 'not has_focus' guard the title has. No test pins it -- add one.
   - AC#1 NOT REPRODUCIBLE in the harness on current dev: title + Tab + body in one uninterrupted pilot.press burst lands correctly, with or without a sync at the Tab boundary or a graduation mid-burst. Textual 8's parser has no burst-to-Paste heuristic, so keys stay ordered; the reported corruption is the task-32062 stale-snapshot clobber, now guarded. Pin the reported gesture as a regression test.
   - AC#3 + AC#4 REPRODUCED at 100x30 and 170x48: a sync while the editor has focus DOES recompose the list pane (rows are new objects, the editor's Input is retained and still focused) -- but the list's scroll offset went to 0 (measured: 6 -> 0).
2. Fix AC#4 only: when the editor-owned skip fires, queue a follow-up that re-applies the Items pane's own scroll offset -- and only that, never focus, which is what the skip exists to prevent.
3. Tests in Tests/UI/test_library_crit8_polish_shell.py beside the task-32062 group: the burst pin, the keywords-authority pin, and one test covering both the list repaint and the surviving offset.
4. Docs: Notes guide stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Premise re-verified per AC against current dev (e6cb464239, after the peer's Notes wave 1). Two of the four had already shipped or did not reproduce; only AC#4 needed code.

AC#1 (Tab burst) -- NOT REPRODUCIBLE on dev. Title + Tab + body sent as ONE uninterrupted key burst lands correctly, with a sync at the Tab boundary and with the graduation landing mid-burst. Textual 8's XTermParser only emits Paste inside a bracketed paste, so fast typing stays ordered Key events and Tab's focus move cannot 'land after the burst'. The reported corruption is the task-32062 stale-snapshot clobber ('Mhello from jordan, testing the libraryy first note'), which the has_focus guards and the recompose skip already fix -- per lessons-live-verification, a live cause attribution that the code contradicts. Pinned as a regression test rather than re-fixed.

AC#2 (keywords authority) -- ALREADY SHIPPED in commit 97626354ee ('the keyword boxes are editor fields', PR #2531 review): both ids joined _NOTE_EDITOR_INPUT_IDS and each assignment took the 'not has_focus' guard. It had NO test; added one.

AC#3 + AC#4 -- reproduced. A sync while the editor has focus does repaint the Items pane (editor_has_focus is instance-scoped: the work pane is a sibling canvas, not an ancestor of the list), and the editor's Input is retained and still focused -- but the list's scroll offset went to the top with the recompose (measured at 100x30: 6 -> 0), because the follow-up that re-applies it is skipped whenever the editor owns focus. Fix: when that skip fires, canvas_sync now queues a follow-up that re-applies the Items pane's own offset and nothing else -- never focus, which is what the skip exists to prevent. Deferred through call_after_refresh for the same reason the identity restore defers its own offset (unlaid-out rows clamp it to 0).

On critique #9 D13 (the list not live-updating a title while the editor is open): the pin shows the list DOES repaint under an editor-owned sync. A title that never appears is a list whose DATA has not changed -- the tree projection reads loaded slices, not the unsaved editor buffer -- not a list frozen by this guard.

Every new test was mutation-checked and each goes red under the guard it pins: disabling the new scroll re-apply reds the mid-edit test (2 failed), dropping the wide_keywords has_focus guard reds the keywords pin, making editor_has_focus screen-wide reds the repaint pin (2 failed), and removing the recompose skip reds the burst pin.

Files: tldw_chatbook/UI/Library_Modules/canvas_sync.py, Tests/UI/test_library_crit8_polish_shell.py, Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
