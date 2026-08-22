---
id: TASK-19905
title: chunk_preview_modal.py is an orphan — delete it or give it a live importer
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - tech-debt
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Widgets/chunk_preview_modal.py` (`ChunkPreviewModal`) has no live importer: its only import is `media_details_widget.py:753`, and `MediaDetailsWidget` is itself unreachable (TASK-19641 owns its disposition). The mentions in `RAG_Search/enhanced_chunking_service.py:18` and `parent_child_adapter.py:99` are docstring references, not imports.

Filed from the chunking template parity design spec §11 item 2 (`Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`; ADR-078). Re-verified live 2026-08-21: importer grep in a worktree at/after the spec's pin, file untouched on `origin/dev` since.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reachability is re-verified at implementation time (importer graph recorded in the Implementation Notes) before anything is deleted
- [ ] #2 Either the module is deleted with no dangling references (imports, CSS selectors — the `chunk-preview` selector family, tests, docstring mentions updated), or it is mounted from a reachable production surface
- [ ] #3 The disposition is coordinated with TASK-19641 so the two tasks cannot strand each other (deleting only the importer leaves this module; deleting only this module breaks nothing but leaves the dead widget)
<!-- AC:END -->
