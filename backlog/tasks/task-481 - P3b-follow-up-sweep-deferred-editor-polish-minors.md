---
id: task-481
title: P3b follow-up — sweep deferred editor-polish minors
status: Done
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
- [x] #1 Avatar thumbnail re-render on save-in-place — DONE in PR #772 fix wave (cc1c04ad0): `_after_character_save` stay-in-editor branch now re-dispatches `_render_character_editor_avatar()` after `mark_saved`.
- [x] #2 Character re-save dedup — DONE (cc1c04ad0): `_character_save_inflight` blocks a re-entrant (concurrent double-Ctrl+S) Save; a sequential no-edit re-save (minutes later, fully settled) is still allowed by design (trivial, no data loss).
- [x] #3 `_profile_save_inflight` try-placement — DONE (cc1c04ad0): `mode`/`persona_id` reads moved inside the try.
- [x] #4 `personality_traits` clear-to-empty Update test — DONE (cc1c04ad0).
- [x] #5 twice-save-in-place greeting/version-preservation test — DONE (cc1c04ad0).
- [x] #6 `console_transcript.py:_image_row_widget` explicit-size guard — DONE in PR #775 (cf8d69c63): extracted `fit_image_cell_size` pure helper into `console_image_view.py`, set explicit fitted cell dims on the graphics image (pixels Static keeps its safe max-* cap).
<!-- AC:END -->
