---
id: task-481
title: P3b follow-up — sweep deferred editor-polish minors
status: To Do
assignee: []
labels:
  - tech-debt
  - personas
  - p3b
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cleanup of Minor findings deferred from the Roleplay P3b whole-branch review (PR #772). None are user-visible data bugs; all are cosmetic, self-healing, or coverage/robustness nits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Avatar thumbnail: after opening the character editor and immediately Saving (save-in-place), the thumbnail no longer risks staying blank — re-dispatch `_render_character_editor_avatar()` on the stay-in-editor branch of `_after_character_save` (today the `_character_editor_generation` bump can drop an in-flight open-time render; text status stays correct).
- [ ] #2 Character editor gains a re-save dedup guard equivalent to the persona `_profile_save_inflight` (save-in-place keeps the Save button live; sequential no-edit Save clicks each run a redundant `update_character` that bumps `version`; no data loss, but wasteful).
- [ ] #3 `personas_screen.py` `_handle_profile_save_requested`: move the `mode = current_mode()` / `persona_id = ...` lines inside the try/except that resets `_profile_save_inflight`, so a raise there can't latch the flag.
- [ ] #4 Add a test that editing `personality_traits` to `""` on a persona Update actually clears the stored value (today only non-empty→non-empty is covered; the path is traced-correct but untested).
- [ ] #5 Add a screen-level test that presses Save twice in one character session (edit greeting → save → edit again → save) to lock in greetings + version preservation across consecutive save-in-place (today only the widget-level `version==2` assertion covers it).
- [ ] #6 (pre-existing, separate concern) `Widgets/Console/console_transcript.py:_image_row_widget` shares the max-width/height-only `textual_image.Image` sizing that caused a zero-size `ValueError` under a live render in P3b Task 2 — apply the same explicit-size guard there (the avatar path already sidesteps it via `_fit_avatar_cell_size`).
<!-- AC:END -->
