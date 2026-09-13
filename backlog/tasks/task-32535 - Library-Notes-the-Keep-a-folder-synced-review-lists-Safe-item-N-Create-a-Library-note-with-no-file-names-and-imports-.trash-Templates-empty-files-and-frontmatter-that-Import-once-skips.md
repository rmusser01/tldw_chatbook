---
id: TASK-32535
title: >-
  Library Notes: the Keep-a-folder-synced review lists "Safe item N / Create a
  Library note" with no file names, and imports .trash, Templates, empty files
  and frontmatter that Import once skips
status: To Do
assignee: []
created_date: '2026-09-13 06:45'
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
- [ ] #3 The Obsidian toggle (skip .obsidian/.trash/Templates, frontmatter → title and keywords, block stripped) is offered default-on for a vault in the sync setup, and empty or whitespace-only files are skipped with a reason
- [ ] #4 Synced notes keep the vault's folder structure under the sync-managed folder
- [ ] #5 notes.md states what differs between Import once and Keep a folder synced on the same folder, or that nothing does; stamp updated
<!-- AC:END -->
