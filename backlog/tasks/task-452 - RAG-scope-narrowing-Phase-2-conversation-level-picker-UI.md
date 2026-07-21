---
id: TASK-452
title: 'RAG scope narrowing Phase 2: conversation-level picker UI'
status: Done
assignee: []
created_date: '2026-07-21 21:20'
updated_date: '2026-07-21 21:28'
labels:
  - rag
  - scope
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 of the RAG scope narrowing program (spec Docs/superpowers/specs/2026-07-21-rag-scope-narrowing-design.md §4; plan Docs/superpowers/plans/2026-07-21-rag-scope-narrowing.md Tasks 8-11). Adds the conversation-level UI on top of Phase 1's enforcement (task-405): the ConsoleScopePickerModal (sortable, tag-filterable, All/Selected views, select-all-matching), the Inspector 'Retrieval scope' row below the Sources tray, and the effective-count header chip. Workspace-level scope is Phase 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scope picker modal lists media/notes with type tabs, sort, tag filter, All/Selected views, and select-all-matching
- [x] #2 Inspector row shows scope state and opens the picker; header chip shows effective count
- [x] #3 Scope persists on save (conversation metadata or session holder → flush at first persistence) and survives resume
- [x] #4 Zero DB reads on compose/recompose; single state source drives row and chip
- [x] #5 QA captures reviewed and approved by owner before merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 2 (Tasks 8-11) via SDD: ConsoleScopePickerModal, Inspector retrieval-scope row, header chip. Every task independent spec+quality review (fix waves all re-verified). Live QA captures in Docs/superpowers/qa/rag-scope-2026-07/. Follow-ups filed at Phase-1: 406-409; Phase-2 close-out follow-ups noted for Phase 3: search_media_db COLLATE title-sort bug (bounded workaround in place), media multi-tag-OR id-window cap (500), row-lacks-alert-emphasis + row-lacks-scoped-tooltip cosmetics (reachable at Phase 3).
<!-- SECTION:NOTES:END -->
