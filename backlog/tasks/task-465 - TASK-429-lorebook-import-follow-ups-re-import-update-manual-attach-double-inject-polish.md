---
id: TASK-465
title: TASK-429 lorebook-import follow-ups (re-import update, manual-attach double-inject, polish)
status: To Do
assignee: []
created_date: '2026-07-21 22:00'
labels:
  - roleplay
  - import
  - lore
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred items from TASK-429 (see Docs/superpowers/specs/2026-07-21-lorebook-import-preservation-design.md and its whole-branch review). None block the shipped fresh-import behavior.
1. Re-import / update-existing character (C1): a name-conflict re-import updates NO fields today (pre-existing whole-card behavior); TASK-429 only made the toast honest. Real re-import/update — including re-attaching an updated lorebook — is a separate feature spanning all card fields and needs a merge-vs-overwrite product decision.
2. Manual-attach + legacy double-inject (I2): a legacy character still carrying extensions['character_book'] that gets a standalone world book manually attached via the Personas UI ends up with BOTH keys (double-injects at send-time, since _collect_active_world_books unions them with no cross-dedup). A blind pop in WorldBookManager._write_character_world_books would itself lose unconverted lore, so the fix must convert-then-strip the legacy character_book there.
3. Polish: parse_v2_card drops a TOP-LEVEL character_book whose entries are ALL inert (0 salvaged) rather than preserving its metadata (asymmetric with the nested case; benign, untested); the import toast names character_world_books[0] rather than specifically the newly-imported block (hand-crafted multi-book edge); add a top-level-all-inert test + a one-line spec note.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Re-import/update-existing is either implemented (with a merge-vs-overwrite decision) or explicitly documented as unsupported
- [ ] #2 A legacy character with a manually-attached world book cannot double-inject its embedded character_book
- [ ] #3 The top-level all-inert-book behavior is covered by a test and the import toast names the newly-imported block
<!-- AC:END -->
