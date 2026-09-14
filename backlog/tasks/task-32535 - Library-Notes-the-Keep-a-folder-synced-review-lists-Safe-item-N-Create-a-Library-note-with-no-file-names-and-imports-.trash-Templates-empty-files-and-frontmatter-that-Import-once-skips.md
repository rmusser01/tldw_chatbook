---
id: TASK-32535
title: >-
  Library Notes: the Keep-a-folder-synced review lists "Safe item N / Create a
  Library note" with no file names, and imports .trash, Templates, empty files
  and frontmatter that Import once skips
status: Done
assignee:
  - '@claude'
created_date: '2026-09-13 06:45'
updated_date: '2026-09-13 15:22'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, persona Alex (Obsidian sync) and Jordan. P1 (A) / D4 (B).

**What happened.** Check changes on the 71-file vault → "60 safe · 0 need attention · 0 skipped · 0 managed placements" then sixty rows reading only "Safe item N / Create a Library note" — no path, no destination, no effect detail (A 62; B 43). 60 = every `.md` including `.trash/Old idea.md`, `Templates/Daily.md`, the 0-byte `Inbox/Untitled.md` and the whitespace-only `Untitled 1.md`; after Activate the synced folder's first row is "Daily" (the template) and every note sits flat under `PowerVault ⇄ Sync managed` (B 45/46); sqlite shows `has_frontmatter=1` for the daily notes, i.e. YAML kept in the body. Import once, one button to the left on the same vault, skips those three folders with reasons, treats the two empty files as "Empty file — nothing to import", lifts the frontmatter into title/keywords and keeps the folder tree (A 31; B 29). Captures: A 31, 62; B 29, 43, 45, 46.

**Cause.** INFERRED: the sync review renders one generic line per planned effect and the sync planner has no Obsidian pass. Docs contradicted: notes.md says the sync review shows "safe actions, attention items, skips, filesystem effects, and deletion-like effects", and its Obsidian section is written under Import once with no statement that the two paths differ. Related improvement idea (critique-3 ideas task): one review renderer for both paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each sync review row names the file path, the effect and the destination folder, using the Import once row grammar (path · what will happen · where)
- [x] #2 Groups carry per-group counts and a uniform run collapses to one summary row with a disclosure, as the Import once review does
- [x] #3 The Obsidian toggle (skip .obsidian/.trash/Templates, frontmatter → title and keywords; the frontmatter block stays byte-exact in the synced note body — controller ruling, wave 4: lasting sync is bidirectional and UPDATE_FILE writes the note body back to disk, so stripping it would delete the user's Obsidian properties on the next Chatbook edit; Folder files keeps the block the same way, task-32264) is offered default-on for a vault in the sync setup, and empty or whitespace-only files are skipped with a reason
- [x] #4 Synced notes are placed in the sync-managed root folder, and the review names that folder as each row's destination rather than promising a nested one — amended from "keep the vault's folder structure" (controller ruling, wave 4). The nested form is blocked two ways in the folder layer, proven against a live profile's database: `note_folders.create_folder` refuses a manual child of any subtree that already holds an active managed placement (`_require_manual_folder_subtree`, `FolderCapabilityError` reason `sync_managed_folder`), so only the first operation of an activation can create its folder; and a subfolder that was created then reads `has_managed_folder_ownership`, which `NotesScopeSyncAuthority._verified_folder` rejects as `folder_authority_changed`, so on the next run even the existing folder stops resolving and every placement recomputes back to the root. Lasting sync needs a folder-creation door of its own before a nested tree can ship; filed separately by the controller
- [x] #5 notes.md states what differs between Import once and Keep a folder synced on the same folder, or that nothing does; stamp updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce headlessly (plan carries only action ids) and live on a fresh scratch profile + 71-file vault
2. RED tests per AC: review rows carry path/effect/destination; 60 creates collapse to one run; reconciler item skips (obsidian_config/trash/template/empty_file) behind obsidian_mode; sync setup offers the Obsidian checkbox for a vault; executor places People/Sam.md under <root>/People and lifts frontmatter title/tags (block kept in the body)
3. Label seam: RuntimeBindingLabel + NotesSyncRuntime.binding_labels(root_id, binding_ids); controller fetches labels for every binding row; canvas renders path · effect · where, grouped with the Import once helpers made public
4. Obsidian pass: NotesSyncRootSetup.obsidian_mode persisted as an obsidian_mode:<root_id> store setting; reconciler item_skips; setup checkbox; executor CREATE_NOTE keeps the vault folder chain under the root folder and lifts title/keywords
5. GREEN + live captures at 235x52 and 100x30; guide notes.md updated and stamped
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All five ACs are met and live-verified at 235x52 and 100x30 on a fresh scratch
profile against a 59-file Obsidian vault. AC#4 was amended by controller ruling
to what ships — placement in the root folder, named truthfully by the review —
because the nested form is blocked in the folder layer; the AC carries both
refusals so the follow-up task inherits the evidence.

**What the review does now (AC#1/#2).** A plan carries only opaque ids, so the
review had nothing to name a row with. `RuntimeBindingLabel` +
`NotesSyncRuntime.binding_labels(root_id, binding_ids)` are the seam that
carries the path; the controller fetches one label per bound row,
`build_reconciliation_review(labels=...)` fills them in, and the canvas renders
Import once's grammar under a heading per effect with its count, collapsing a
uniform run behind a disclosure. Live: "56 safe · 0 need attention · 4 skipped",
"Create a Library note (56)", "▶ Archive · 45 files · Create a Library note ·
PowerVault", "Daily/2026-09-06.md · Create a Library note · PowerVault".

Two defects only the live run found, both now pinned on the real route:
- the canvas patches the configure form in place instead of recomposing, so the
  Obsidian checkbox (a widget that exists only for a vault) never appeared
  however correct the state was;
- a plan orders actions by binding id — a digest — so the vault's 45 Archive
  files arrived shuffled and `uniform_runs`, which only collapses CONSECUTIVE
  members, produced no run at all. Each group is now sorted by path.

**The Obsidian pass (AC#3).** `ReconciliationInput.obsidian_mode` drops the
vault's own root folders from the plan and records `item_skips` with Import
once's own reason table (now public — one source of truth). An empty or
whitespace-only file is skipped whatever the toggle says, because Import once
refuses an empty source in any folder. The frontmatter lift reuses Import
once's parser: `title` and `tags`/`aliases` become the note's title and
keywords while the block stays byte-exact in the body (controller ruling:
`UPDATE_FILE` writes the body back to disk).

The flag is NOT persisted. `notes_sync_store_settings` CHECK-constrains
`setting_key` to two literal keys, so the store-setting approach a previous
implementer took raised `IntegrityError` on every activation (and hung one
path for 300 s); persisting there needs a device-schema version and a table
rebuild, because the schema module byte-compares each table's DDL. It is
resolved per observation from the vault marker the discovery walk already
reports — no extra filesystem read, and a folder that stops being a vault stops
being treated as one. Ceiling: a user who declines the toggle at setup gets the
pass back after a restart; the pass only ever leaves NEVER-BOUND files alone,
so nothing already synced changes.

**AC#4 — amended and reverted whole.** Live activation ended "⚠ Partial ·
Activation needs attention" with `folder_mutation_failed` after three notes.
Reproduced against the live profile's own database:
`note_folders.create_folder` refuses a manual child of a subtree that already
holds a managed placement (`_require_manual_folder_subtree`, reason
`sync_managed_folder`), so only the first operation can create its folder; and
the folder that did get made then reads `has_managed_folder_ownership`, which
the authority's verify path rejects as `folder_authority_changed` — on the next
run even the existing folder stops resolving and the placements collapse back
to the root. A half-nested tree that flattens itself on the next sync is worse
than the flat tree the critique filed, so the chain was reverted whole
(`_derived_folder_id`, `_sync_folder_id`, `ensure_sync_subfolder`,
`active_binding_placements`). The runtime pin now holds the flat placement so
no half version can ship, the review's destination says the root folder for
every row rather than promising a folder that is never created, and the guide
states the difference. Sync needs a folder-creation door of its own (a
sync-owned create the manual guard does not apply to) before a nested tree can
ship; AC#4 is amended to what ships and the controller is filing that separately, along with persisting the
Obsidian choice (which needs a device-schema version). The setup copy and the
guide now say the choice lasts only until Chatbook quits.

**Files.** `Notes/notes_sync_runtime.py`, `notes_sync_reconciler.py`,
`notes_sync_executor.py`, `notes_sync_authority.py`, `notes_scope_service.py`,
`note_import_discovery.py`; `Library/library_notes_lasting_sync_state.py`;
`UI/Library_Modules/library_notes_sync_controller.py`;
`Widgets/Library/library_notes_add_from_files_canvas.py`,
`library_note_import_canvas.py` (three helpers made public, old names aliased);
`Docs/User_Guide/library/notes.md`; pins in `Tests/UI/test_library_notes_w4_sync_review.py`
(new), `Tests/Notes/test_notes_sync_{runtime,reconciler,executor}.py`,
`Tests/Library/test_library_notes_lasting_sync_state.py`.

Tests vs a detached `origin/dev` worktree: Notes sync suites 1073 passed /
1 failed vs 1069 / 1; Library-UI suites 255 / 1 vs 245 / 1. Both failures are
the same pre-existing names on dev
(`test_legacy_sync_config_is_read_only_and_only_the_migrator_reads_it`,
`test_database_notes_import_once_journey_is_painted_focused_and_retained[size1]`).
Captures under `wave4-caps/sync-review/`. `./scripts/preflight.sh` green.
<!-- SECTION:NOTES:END -->
