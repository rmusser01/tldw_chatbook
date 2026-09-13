---
id: TASK-31968
title: >-
  Library media: the Reader's reading-position restore does not land on a mode
  change
status: Done
assignee: []
created_date: '2026-09-07 20:27'
updated_date: '2026-09-08 05:40'
labels:
  - library
  - media
  - reader
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by PR L (task-31954). On Read → Analysis → Read the reader progress restore is scheduled once now, but the position is not visibly restored — reproduced on the merge-base 801216bb0 too, so this predates the rider work. `_restore_library_media_loaded_progress` calls `scroll_to` while the rendered Markdown body is still parsing, so the scroll lands on a shorter document and is lost when the body grows. Two ineffective schedules became one ineffective schedule; the user still loses their place on every mode change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Switching Read → Analysis → Read returns the Reader to the same scroll offset (painted pin, not a state probe)
- [x] #2 The restore waits for the rendered body's layout (or re-applies once after the parse settles) rather than racing it
- [x] #3 No second scheduler is introduced; the one owner from task-31954 stays the only one
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace where the media reader scroll is captured/re-applied on a mode change and why the re-apply lands on an unlaid-out Markdown body. 2. Mirror the NOTES scroll-restore precedent: scroll_to(immediate=True) + a bounded re-apply after the body lays out, staying task-31954's single owner. 3. Painted pin on the real scroll container, red first.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: on Read->Analysis->Read the media reading-position restore fired one deferred scroll_to against a Markdown body still parsing (Textual parses in an executor and mounts blocks across later refreshes), so max_scroll_y was 0 and the offset clamped to the top. Fix in the one shared `_restore_library_media_loaded_progress` (covers mode-change AND initial-load): scroll_to(immediate=True) clamps against current layout, and while `scroller.scroll_y < offset` an inner `settle()` closure re-arms one `call_after_refresh` at a time (<=8 hops, re-checking loaded_id each hop so a stale offset never lands on a newer item), so the offset re-lands as the body grows. Stays task-31954's single owner: settle() is a continuation of the one call, never re-enters the public method, never re-claims `_library_media_progress_restored_id` (AC#3). Determinism ruling: the production race is sub-frame and the pilot harness collapses parse+mount+layout+deferred-scroll into one pause(), so the end-to-end round-trip cannot go RED in-harness (kept as a regression guard); the RED->GREEN pin asserts the deterministic half -- the restore lands synchronously on the REAL laid-out scroll container (reverting immediate=True fails assert 0 == 14). Bounded Minors (not fixed, YAGNI): rapid same-item switching stacks convergent settle loops; a scroll-up within the <=8-hop window is pulled back. Files: tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_media_reader_scroller_resolution.py, Tests/UI/test_library_media_reader_flow.py (progress unit contract).
<!-- SECTION:NOTES:END -->
