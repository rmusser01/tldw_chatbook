---
id: TASK-24404
title: 'Settings form for creating and editing chunking templates'
status: To Do
assignee: []
created_date: '2026-08-29'
updated_date: '2026-08-29'
labels:
  - chunking
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The chunking program (PRs #1852, #1938, #1952, #1976, #1984, #1990; ADR-078)
gave chunking templates a canonical store (`ChunkingTemplates`, media DB schema
v7) and three consumers: the Library ingest canvas picker ("None (manual
settings)" / "Auto" / saved template — `Widgets/Library/library_ingest_canvas.py`),
the per-item chunking config in `Widgets/media_details_widget.py` (template
select + Preview chunks modal), and the bulk legacy re-chunk
(`RAG_Admin/local_rag_admin_service.rechunk_legacy_media`).

But template **creation** has no human-facing surface: it exists only in the
service layer (`Chunking/chunking_interop_library.py` `create_template` /
`update_template`, and the RAG-admin local/scope/server variants) plus the
runtime-policy `library.templates/save` verb (the Console agent path). A user
without the agent cannot mint or edit a named template from the TUI. This task
closes that loop in the canonical settings surface (F9,
`UI/Screens/settings_screen.py` and its `settings_*` modules). Per AGENTS.md
the legacy `Tools_Settings_Window` / `enhanced_settings_sidebar` are deprecated
parallels — nothing may be added there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Settings (F9) Library/RAG area offers create and edit of chunking templates (name, method, chunk options, tags), writing through the existing service layer (`chunking_interop_library` / RAG-admin `create_template` / `update_template`) — no new direct DB writes
- [ ] Validate-on-write verdicts surface to the user: the reserved name `auto` (case-insensitive, whole word) and invalid bodies are refused with the validator's field/message summary (`InvalidTemplateError`); builtin templates are not editable (`BuiltinTemplateError`)
- [ ] The form lists existing templates with the same validity/reserved decoration rules as the ingest picker
- [ ] A template created or edited in Settings is immediately selectable in the Library ingest picker and the media-details template select (template-name cache invalidation covered)
- [ ] Nothing is added to the deprecated settings surfaces (`Tools_Settings_Window` / `enhanced_settings_sidebar`)
- [ ] Targeted tests: a form contract test (create happy path + reserved-name and builtin refusals) and a picker-cache invalidation test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate the canonical Settings surface for Library/RAG chunking defaults (`UI/Screens/settings_library_rag_defaults.py`, how `settings_screen.py` composes it) and the service seam (`chunking_interop_library` create/update/list, RAG-admin variants)
2. Add the template-management section (list + create/edit form) calling the service layer only; mirror the ingest picker's reserved/valid decoration
3. Wire cache invalidation after a successful save (ingest canvas template-name cache + media-details selects)
4. Tests: UI contract test for the form (happy path, reserved name, builtin refusal), cache-invalidation test; targeted runs only
<!-- SECTION:PLAN:END -->
