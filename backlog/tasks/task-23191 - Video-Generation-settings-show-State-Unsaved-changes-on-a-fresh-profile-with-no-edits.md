---
id: TASK-23191
title: >-
  Video Generation settings show 'State: Unsaved changes' on a fresh profile
  with no edits
status: Done
assignee: []
created_date: '2026-08-29 02:25'
labels:
  - ux
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a newly created profile, opening Settings -> Video Generation reports unsaved changes before the user has edited anything, so the dirty indicator cannot be trusted and a Revert appears to be needed on a page nobody touched. Observed during the TASK-23109 verification pass on an isolated profile; the draft for this category appears to initialize dirty rather than adopting the persisted values. The State banner is the sole carrier of the save contract (task-1717, TASK-23104), so a false dirty state undermines the mechanism the whole screen relies on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening Video Generation on a fresh profile with no user edits reports no unsaved changes
- [x] #2 The dirty state appears only after an actual user edit and clears after save or revert
- [x] #3 A test mounts the category on a fresh profile and asserts the clean state, so the regression cannot return silently
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the false dirty state in a mounted test before touching production code.
2. Find WHY the draft initializes dirty, and check whether the cause reaches sibling categories.
3. Fix at the cause, mutation-check the test in both directions, and verify live on a fresh profile.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Root cause.** A freshly composed Textual `Select` re-posts `Changed` from
its own `_on_mount` whenever it carries a non-blank value. Video Gen
composes TWO of them, and this category deliberately diffs the draft
against the RAW `[video_generation]` TOML table (never the resolved
config) so an untouched key is never written out -- while the panel itself
composes from the RESOLVED config. On a fresh profile the shipped config
template writes no `[video_generation]` table at all (verified live: the
app's own boot rewrite does not add one), so the retention `Select` mounted
with the effective default `"session"` and its mount echo was staged as
`retention: None -> "session"`. That is a difference, but not a user edit.
`default_backend` was already protected by a suppression queue; retention
had no guard.

**Fix.** `expected_select_mount_values()` in
`settings_video_gen_defaults.py` mirrors the panel's compose and reports
what EVERY Video Gen `Select` will mount with. The screen's suppression
queue became a per-Select mapping (`_video_gen_select_suppress_queues`,
`_consume_video_gen_select_mount_echo`) so one Select's echo can never be
mistaken for the other's, and it is cleared on category leave -- matching
the image-gen and RAG precedents, since a stale expectation would
otherwise swallow the first genuine edit of a later visit.

**Sibling check (no other category has this shape).** Image Gen uses the
same raw-table diff but composes a single, already-suppressed `Select`,
and its `[image_generation]` section IS in the shipped template. Every
other staging helper (`_stage_appearance_value`,
`_stage_console_default_value`, `_stage_library_rag_value`,
`_stage_storage_value`, ...) takes its "original" from the resolved
`_*_loaded_values()` -- exactly what the widget mounts with -- so a mount
echo compares equal there by construction.

**Trade-off.** The suppression queue is a per-Select FIFO rather than a
general "is this a user gesture?" test, because Textual gives the handler
no way to tell a mount echo from a real `Changed`. The mirror between
`expected_select_mount_values()` and the panel's compose must be kept in
step by hand; it is one function with a docstring saying so, instead of
the two divergent inline expressions it replaced.

**Verification.** `test_video_gen_opens_clean_on_a_fresh_profile` mounts
the category with no persisted table and asserts the clean draft AND the
clean banner, then drives a REAL retention edit through the widget
(dirty), then Revert (clean again). Mutation-checked both ways: removing
the guard fails on `fresh profile staged: {'retention'}`; suppressing every
retention `Changed` fails on `a genuine retention edit must stage`.

Live, isolated fresh profile:

    before:  State: Unsaved changes | Save (s) or Revert (r) - switching
             categories keeps this draft.
    after:   State: Draft - save/revert below | Defaults affect future
             Console video generations.

**Files.** `tldw_chatbook/UI/Screens/settings_video_gen_defaults.py`,
`tldw_chatbook/UI/Screens/settings_screen.py`,
`tldw_chatbook/Widgets/settings_video_gen_panel.py`,
`Tests/UI/test_settings_video_gen_defaults.py`.
<!-- SECTION:NOTES:END -->
