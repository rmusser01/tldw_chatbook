---
id: TASK-79
title: Functionalize Settings Library and RAG defaults
status: Done
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
- [x] #1 Settings > Library & RAG loads existing persisted defaults for search mode, result limits, citation display, snippet display, and context limits from the current config source.
- [x] #2 Users can edit, validate, save, and revert Library/RAG defaults without changing indexing, embedding model lifecycle, source staging, or active query workflows.
- [x] #3 Invalid numeric or enum values block save, keep the category dirty, and show visible recovery copy in the detail pane and inspector.
- [x] #4 Library/RAG Settings copy clearly distinguishes global defaults from active Library queries, Console staged context, workspace eligibility, and indexing operations owned elsewhere.
- [x] #5 Focused automated tests cover load, edit, validation failure, save, revert, ownership copy, and no-regression behavior for existing domain category contract tests.
- [x] #6 Actual Textual-web/CDP screenshots verify the changed Library & RAG category and are approved before PR creation.
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
- Added a Settings-owned Library/RAG defaults model for loading, validating, and building deep-merged save payloads under `AppRAGSearchConfig`.
- Replaced the read-only Library & RAG detail with guided controls for search mode, result limits, retriever balance, citations, snippet length, and context budget while preserving Library as the runtime workflow owner.
- Added visible validation and recovery copy in the detail pane and inspector, including focused invalid-field styling that keeps typed values readable.
- Extended RAG config defaults for citation style and snippet display length, and added focused regression coverage for load, edit, validation, save, revert, ownership copy, and CSS readability.
- Captured approved Textual-web/CDP QA screenshots under `Docs/superpowers/qa/product-maturity/screen-qa/settings/`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Verification: `python -m pytest -q Tests/UI/test_settings_library_rag_defaults.py Tests/UI/test_settings_configuration_hub.py --tb=short` passed with 192 tests and 1 existing dependency warning.
- Verification: `git diff --check` passed.
- QA evidence: approved actual screenshots for baseline, search dropdown, focused valid snippet input, and focused invalid snippet input are included in the Settings screen QA folder.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
