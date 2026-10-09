---
id: TASK-32451
title: Library Notes Manage sync folders shows a placeholder name for every root
status: Done
assignee:
  - '@claude'
created_date: '2026-09-11 16:10'
updated_date: '2026-10-09 17:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - sync
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while walking the lasting-sync chapter for task-32269, once task-32243 made a root reachable at all.

Every row in Manage sync folders reads "Sync folder (name unavailable before cutover)", including a root the user just created and typed a display name for. `library_notes_sync_controller.py:709` hard-codes that literal for every row because the path-free `NotesSyncRootRuntimeSnapshot` carries only root_id/status/next_action -- the display name the user typed at setup has no route to the row, and neither does the folder. With one root the row is merely unhelpful; with two it is unusable, because nothing on screen distinguishes them.

The name is not private the way the path is: the user typed it, it is already shown in the notes folder tree ("Vault sync"), and it is validated as a bounded non-path label by `NotesSyncRootSetup`. The fix is a route for it, not a new path field.

Evidence: captures cap-06 and cap-07 under the wave-3 sync scratchpad (live walk at 235x52 against a 179-file vault, 2026-09-11).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each row in Manage sync folders shows the display name the user gave that root
- [x] #2 Two roots are distinguishable from the list alone
- [x] #3 No absolute path reaches the row or any log record through this change
- [x] #4 A migrated legacy candidate, which has no user-typed name, still shows something honest rather than a placeholder that claims a name is unavailable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED first (new `Tests/Notes/test_notes_sync_root_display_name.py`, real `build_notes_sync_runtime_owner` over a tmp
   `CharactersRAGDB` + state DB): the published snapshot carries the folder name the user typed (AC#1), two roots are
   told apart from the snapshot alone (AC#2), a restart republishes the names from the store, a migrated candidate
   without a folder reads "Migrated notes — review to finish setup" and a folder-less root "Sync folder (setting up)"
   (AC#4), no vault path reaches the snapshot, the row or a loguru record (AC#3), and a soft-deleted managed folder
   still names its root. Rewrite the three pilot pins that asserted the placeholder.
2. Route the name through the projection: `NotesSyncRootRuntimeSnapshot.display_name` (defaulted, validated as a
   bounded non-path label); `_RuntimeAdapter.root_display_name(root)` implemented on the production adapter from the
   root's logical folder (`include_deleted=True`); an owner-side `_root_names` map filled at startup, at setup review
   and at activation, cleared with the setup authority; `_publish` (still the only construction site) stamps it; the
   honest fallbacks are computed in the owner where the status code is known.
3. Controller: `_project_root` uses `root.display_name or "Sync folder"`; delete the placeholder literal.
4. Canvas pins for name safety: a markup-shaped name paints literally; a 160-character name keeps the row's controls.
5. Guide (`Docs/User_Guide/library/notes.md`), FAST_LANE + workflow entries, size ratchet if the controller grew.
6. Live verification on the isolated harness at 160x45 and 100x30 with the seeded "VSync" root; captures under
   `qa/notes-library-ux-review-2026-10-02/fixes/task-32451/`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** The row title now comes from the runtime's projection: `NotesSyncRootRuntimeSnapshot.display_name`
(defaulted last, validated as a bounded non-path label when set) is stamped by `_publish` -- still the only
construction site -- from an owner-side `_root_names` map. The production adapter reads a root's name from its
managed folder (`root_display_name`: `get_note_folder_by_id_for_sync(..., include_deleted=True)` → `folder.name`,
shaped by `_bounded_root_label`); the owner fills the map at startup (`_load_root_names`, right after
`_load_roots`, through `getattr(adapter, "root_display_name", None)` so a test fake lands on the fallback), at
setup review (the typed name, so a status published during the review is titled) and at activation (once the
folder is assigned), and releases it with the setup authority. Fallbacks are computed in the owner where the
status code is known: a PAUSED `migration_review_required` root reads "Migrated notes — review to finish setup",
any other folder-less root "Sync folder (setting up)". `LibraryNotesSyncController._project_root` uses
`root.display_name or "Sync folder"`; the placeholder literal is gone. The canvas already paints the title with
`markup=False`, so a markup-shaped folder name is literal and inert; a 160-character name wraps inside its row.

**Verified (branch `fix/task-34000-clarify` on 8ccf012971, 2026-10-09).** RED on base: the new
`Tests/Notes/test_notes_sync_root_display_name.py` failed 7/7 with `AttributeError: 'NotesSyncRootRuntimeSnapshot'
object has no attribute 'display_name'`; GREEN after: 9 passed with the two architecture pins. Re-runs: Notes
pure + architecture + CI contract 292 passed (2 pre-existing failures in `test_notes_sync_cutover.py` reproduce
identically on a base probe worktree: an unrelated `get_cli_setting` allowlist miss and a local `RecoveryRequired`);
w4 + journey + controller + lasting-flow with the scratch plugin 224 passed, 0 failed. Live on the isolated
harness with the seeded "VSync" root: 160x45 and 100x30 both title the row `VSync` with "✓ Up to date as of HH:MM
· Next: Check changes", and the Notes tree shows "▸ VSync ⇄ Sync managed"; 0 Tracebacks and no vault path in
either run's log. Captures: `qa/notes-library-ux-review-2026-10-02/fixes/task-32451/`.

**Trade-offs.** No new DB read on the render path (names resolve on the maintenance path at startup/activation).
The size ratchet row for the controller rose 2375 -> 2378 (owner ruling).

**Fix round 1 (review 1, Important 1).** A root WITH a folder id is never titled "setting up". Fallback table:
no folder id + PAUSED `migration_review_required` -> "Migrated notes — review to finish setup"; no folder id,
otherwise -> "Sync folder (setting up)"; folder id set and the adapter returned no name (row gone from
`note_folders`, or a blank name) -> "Sync folder (missing from Notes)"; the name read raised, or the adapter has no
`root_display_name` route -> "Sync folder (name unavailable)"; a soft-deleted folder still gives its real name.
Pinned by four new pure tests (row hard-deleted, blank name, raising read, fake adapter with a folder id), RED on
edb4a879c9, and documented in the guide paragraph. **Known limitation (review 1, Minor 1).** Names are read at
startup, setup review and activation only; a managed folder renamed in the Notes tree keeps its old row title
until the next start. The repository refuses a rename on a subtree holding managed notes
(`_require_manual_folder_subtree`), so this reaches only a managed folder with no managed notes or an out-of-band
write; no in-process rename signal reaches the runtime today, and adding polling or UI-thread I/O for it was
ruled out, so it is recorded here rather than fixed.

**Files.** `tldw_chatbook/Notes/notes_sync_runtime.py`, `tldw_chatbook/UI/Library_Modules/
library_notes_sync_controller.py`, `Tests/Notes/test_notes_sync_root_display_name.py` (new; FAST_LANE + workflow),
`Tests/Widgets/Library/test_library_notes_sync_roots_canvas.py`, `Tests/UI/test_library_notes_w4_sync_roots.py`,
`Tests/UI/test_library_notes_files_sync_journey.py`, `Tests/Architecture/test_library_modules_size_ratchet.py`,
`Tests/CI/test_ci_queue_pressure_contract.py`, `.github/workflows/derived-artifacts.yml`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
