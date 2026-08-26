---
id: TASK-3020
title: Media select-mode polish batch from the P1 arc final review
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 12:20'
updated_date: '2026-08-08 21:23'
labels:
  - library
  - media
  - polish
  - uat-2026-08-06
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The `fix/library-uat-p1s` final whole-branch review (2026-08-07) triaged these as ride-as-follow-up
minors, all in the Media select-mode neighborhood shipped by task-2853/2856. File positions cite
that branch's head `6672ed276`.

1. `library_screen.py:8751-8769` — the bulk-delete confirm handler starts a bare `run_worker`; a
   fast double-press launches two workers over the same ids. `mark_as_trash` is idempotent but the
   second worker decrements the rail count again (floored at 0) → transient under-report. Fix:
   `exclusive=True` worker group or an in-flight guard.
2. Escape inconsistency between the two delete confirms: the single-item viewer confirm cancels on
   Escape ("back a step"); an armed bulk-delete confirm does not — Escape moves focus to the rail
   and leaves the armed "Delete N…?" row behind. Add a `confirming_bulk_delete` branch to the
   Escape chain (cancel first, like its own Cancel button).
3. Partial-failure bulk delete leaves nothing focused (the confirm row's Delete button is removed
   by the recompose; entry focus deliberately not re-armed). Focus the first still-checked row.
4. The skill editor's footer does not advertise its working `esc`/`ctrl+s`
   (`_register_footer_shortcuts` has no skill-editor branch) — under-advertising, now a visible
   asymmetry beside the note/prompt editors.
5. The single-item viewer delete never decrements the rail's "Media N" count (pre-existing) —
   now a visible inconsistency beside the bulk path, which does.
6. Copy: `library_screen.py:8846` uses "item(s)" while sibling strings compute singular/plural.
7. `Docs/User_Guide/library/media-and-conversations.md` stamp names only TASK-2857 though
   task-2853's content shipped on the same page (content correct; stamp lags).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A double-press on the bulk-delete confirm cannot run two delete workers or double-decrement the rail count
- [x] #2 Escape cancels an armed bulk-delete confirm (parity with the single-item confirm), covered by a test
- [x] #3 After a partial-failure bulk delete, keyboard focus lands on a still-checked row
- [x] #4 The skill editor's footer advertises its working keys
- [x] #5 Single-item viewer delete updates the rail count in place, like the bulk path
- [x] #6 Pluralization and the docs stamp are corrected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Guard the bulk-delete confirm button with an in-flight flag + exclusive worker group so a fast double-press cannot launch two delete workers.
2. Add a confirming_bulk_delete Escape gate before the broader focus-rail gate (declaration-order precedence), reusing the Cancel button's own dismiss path.
3. Extend _focus_library_list_entry to prefer the first still-checked Media row over the literal first row whenever Select mode holds a non-empty selection (partial/total bulk-delete failure retry).
4. Add a skill-editor branch to _library_footer_shortcuts_for_current_state (and a bulk-delete-confirm branch) so the footer/F1 advertise the working ctrl+s/esc keys honestly.
5. Mirror the bulk path's rail-count decrement into the single-item viewer delete handler.
6. Pluralize the partial-failure warning off the confirmed batch size instead of a bare item(s) string; refresh the media-and-conversations.md verified-against stamp.
7. Cover each AC with unit tests in test_library_multiselect_media.py / test_screen_navigation.py; live-verify in a scratch tmux profile.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Takeover note: this task's implementation was built by a prior agent that died mid-live-verification (API limit) before writing a report or committing; its work sat uncommitted (backup: .superpowers/sdd/library-polish-batch/task-4-takeover-backup.diff). A second agent (this session) verified the inherited diff against all 6 ACs, ran the covering suites with real counts, reconstructed RED evidence for two of the changes via Edit-based hunk reverts (restored after), completed the live TUI verification the original agent was mid-way through, and finished backlog hygiene -- no production code changed from the inherited diff.

AC1 (double-press guard): _library_media_bulk_delete_in_flight, set synchronously before scheduling the worker and cleared in a finally, plus exclusive=True/group="library_media_bulk_delete" as a second line of defense. RED-verified by reverting the guard: 2 of 3 new guard tests failed for the right reason (second press launched a second worker; exclusive kwarg missing), then restored. Live-verified: rapid double SGR press+release on Delete against a real scratch DB (2 selected media rows) trashed exactly 2 rows (Media rail 4->2), confirmed against the DB directly (is_trash=1 on exactly the 2 selected ids, not more).

AC2 (Escape cancels an armed bulk-delete confirm): new escape/library_media_bulk_delete_cancel BINDINGS entry declared before library_list_focus_rail (declaration-order precedence, both gates are check_action-True on an armed confirm), sharing _cancel_library_media_bulk_delete with the Cancel button. RED-verified by removing the check_action branch: test_check_action_gates_media_bulk_delete_cancel_to_armed_confirm failed (fell through to the bare default True), then restored. Live-verified: armed a 2-item confirm, first Escape cancelled it in place (row selection preserved, footer reverted to "esc focus rail"), second Escape then moved focus to the rail search box -- matches the matrix.

AC3 (partial-failure focus): _focus_library_list_entry now prefers the first still-checked Media row over the literal first row when Select mode holds a non-empty selection; _delete_library_media_selection now arms entry focus on every completion path, not just full success. Covered by unit tests (fixture-level fake rows); not separately live-reproduced (simulating a real delete_media_item failure live was out of scope for this session's required live-verification list).

AC4 (skill-editor footer): LIBRARY_SKILL_EDITOR_SHORTCUTS + LIBRARY_MEDIA_BULK_DELETE_CONFIRM_SHORTCUTS added to _library_footer_shortcuts_for_current_state, both feeding F1 via the same shared seam. Live-verified: opened Create > New skill, footer read "ctrl+s save skill | esc back to skills list" (previously would have fallen through to the bare general set); armed bulk-delete confirm showed "esc cancel delete".

AC5 (single-item viewer delete decrements the rail count): _local_source_counts["media"] decremented in _delete_library_media_item, mirroring the bulk path. Live-verified: deleted one item from the viewer, Media rail dropped 2->1 in place; confirmed the DB row flipped is_trash=1.

AC6 (pluralization + docs stamp): partial-failure warning now pluralizes off len(media_ids) instead of a bare "item(s)" string (grepped clean, no literal "item(s)" remains); Docs/User_Guide/library.md and .../media-and-conversations.md both got a fresh "Verified against dev @ 023a04a48" stamp plus a short prose note describing the new Escape/rail-count behavior.

Test counts (foreground, venv python -m pytest, real counts read): Tests/UI/test_library_multiselect_media.py + Tests/UI/test_screen_navigation.py: 152 passed. Tests/UI/test_library_skills_canvas.py: 107 passed, 1 failed (test_action_library_skill_back_honors_dirty_guard -- confirmed PRE-EXISTING via a throwaway git-worktree checkout at clean HEAD, same AttributeError on _library_list_entry_focus_timer, a fixture-bypass bug outside task-3022's exit-bar suites; not caused by this task, not touched). Tests/UI/test_library_shell.py -k media: 66 passed, 301 deselected.

Files changed: tldw_chatbook/UI/Screens/library_screen.py (BINDINGS, check_action, footer shortcut sets, bulk-delete guard/cancel/focus, single-item rail decrement, pluralization); Tests/UI/test_library_multiselect_media.py, Tests/UI/test_screen_navigation.py (new coverage); Docs/User_Guide/library.md, Docs/User_Guide/library/media-and-conversations.md (stamps); backlog/tasks/task-3020 (this file).
<!-- SECTION:NOTES:END -->
