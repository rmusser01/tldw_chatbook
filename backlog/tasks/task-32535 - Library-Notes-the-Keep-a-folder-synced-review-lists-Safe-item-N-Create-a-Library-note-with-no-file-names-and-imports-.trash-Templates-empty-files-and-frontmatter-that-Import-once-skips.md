---
id: TASK-32535
title: >-
  Library Notes: the Keep-a-folder-synced review lists "Safe item N / Create a
  Library note" with no file names, and imports .trash, Templates, empty files
  and frontmatter that Import once skips
status: In Progress
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
- [ ] #1 Each sync review row names the file path, the effect and the destination folder, using the Import once row grammar (path · what will happen · where)
- [ ] #2 Groups carry per-group counts and a uniform run collapses to one summary row with a disclosure, as the Import once review does
- [ ] #3 The Obsidian toggle (skip .obsidian/.trash/Templates, frontmatter → title and keywords; the frontmatter block stays byte-exact in the synced note body — controller ruling, wave 4: lasting sync is bidirectional and UPDATE_FILE writes the note body back to disk, so stripping it would delete the user's Obsidian properties on the next Chatbook edit; Folder files keeps the block the same way, task-32264) is offered default-on for a vault in the sync setup, and empty or whitespace-only files are skipped with a reason
- [ ] #4 Synced notes keep the vault's folder structure under the sync-managed folder — NOT DELIVERED, blocked in the folder layer and reverted (wave 4): `note_folders` refuses a manual child of any subtree that already holds a managed placement (`_require_manual_folder_subtree`, reason `sync_managed_folder`), so only the first operation's folder can be created; and a subfolder that was created then reads as managed-owned, which the sync authority's verify path rejects as `folder_authority_changed` on the next run — the tree would collapse back to flat on the following sync. Both refusals reproduced against the live profile's database. Sync needs its own folder-creation door (a sync-owned create that the manual guard does not apply to) before this AC can be met; controller decision requested
- [ ] #5 notes.md states what differs between Import once and Keep a folder synced on the same folder, or that nothing does; stamp updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce headlessly (plan carries only action ids) and live on a fresh scratch profile + 71-file vault
2. RED tests per AC: review rows carry path/effect/destination; 60 creates collapse to one run; reconciler item skips (obsidian_config/trash/template/empty_file) behind obsidian_mode; sync setup offers the Obsidian checkbox for a vault; executor places People/Sam.md under <root>/People and lifts frontmatter title/tags (block kept in the body)
3. Label seam: RuntimeBindingLabel + NotesSyncRuntime.binding_labels(root_id, binding_ids); controller fetches labels for every binding row; canvas renders path · effect · where, grouped with the Import once helpers made public
4. Obsidian pass: NotesSyncRootSetup.obsidian_mode persisted as an obsidian_mode:<root_id> store setting; reconciler item_skips; setup checkbox; executor CREATE_NOTE keeps the vault folder chain under the root folder and lifts title/keywords
5. GREEN + live captures at 235x52 and 100x30; guide notes.md updated and stamped
<!-- SECTION:PLAN:END -->
