---
id: TASK-79
title: Functionalize Settings Library and RAG defaults
status: To Do
labels:
- settings
- library
- rag
- configuration
- ux
priority: high
documentation:
- Docs/superpowers/plans/2026-06-07-settings-library-rag-defaults.md
- backlog/decisions/003-settings-library-rag-defaults.md
- Docs/superpowers/specs/2026-05-24-settings-configuration-hub-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn Settings > Library & RAG from a read-only ownership contract into a real guided configuration category for existing Library search, RAG retrieval, citation, and snippet defaults while preserving Library and Console as the workflow owners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings > Library & RAG loads existing persisted defaults for search mode, result limits, citation display, snippet display, and context limits from the current config source.
- [ ] #2 Users can edit, validate, save, and revert Library/RAG defaults without changing indexing, embedding model lifecycle, source staging, or active query workflows.
- [ ] #3 Invalid numeric or enum values block save, keep the category dirty, and show visible recovery copy in the detail pane and inspector.
- [ ] #4 Library/RAG Settings copy clearly distinguishes global defaults from active Library queries, Console staged context, workspace eligibility, and indexing operations owned elsewhere.
- [ ] #5 Focused automated tests cover load, edit, validation failure, save, revert, ownership copy, and no-regression behavior for existing domain category contract tests.
- [ ] #6 Actual Textual-web/CDP screenshots verify the changed Library & RAG category and are approved before PR creation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/003-settings-library-rag-defaults.md
Reason: This task defines the persisted Settings/Library/RAG configuration boundary and adds a new snippet display default under the existing RAG config mapping.

Use `Docs/superpowers/plans/2026-06-07-settings-library-rag-defaults.md` as the implementation plan. Keep the implementation PR scoped to Library/RAG defaults only. Add explicit RAG config-model support for new display defaults (`citation_style` and `snippet_max_chars`) before rendering them as editable Settings fields.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
