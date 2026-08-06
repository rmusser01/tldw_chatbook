---
id: TASK-1341
title: Settings unify save commit models
status: Done
assignee: []
created_date: '2026-08-04 23:47'
updated_date: '2026-08-05 02:14'
labels:
  - settings
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three commit models coexist: staged s/r (Providers endpoint/key/model), instant-persist checkboxes (settings_screen.py:6264-6266), and editor-own buttons (Theme). UAT confirmed the confusion path: toggle auto-save checkbox, press s, get 'no changes to save'. Users cannot answer 'did that save?'
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Staged save is the default model across categories,Any intentional instant-apply control is labeled inline ('applies immediately - no Save needed') and visually separated from staged fields,Per-category save behavior is documented in the field inspector
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit commit models: staged (GUIDED_SETTINGS_MUTATION_CATEGORIES: providers/appearance/console/library-rag/storage) stays the default; model-catalog toggles + stale-hours (ADR-020 operational flags) and splash defaults stay instant; theme editor stays editor-owned. No persistence behavior changes.
2. AC2: add inline 'applies immediately - no Save needed' hint + bordered group class around the Providers 'Automatic refresh (ADR-020)' block; add the same hint to the Splash viewer 'Startup defaults' section; CSS in tldw_cli_modular.tcss.
3. AC3: add a 'Save:' row to every branch of _provider_field_guidance_rows (incl. a new model-catalog branch: instant), _appearance_field_guidance_rows, _storage_field_guidance_rows; add Save rows to Console Behavior / Library&RAG / Theme / Splash inspector sections.
4. TDD: new Tests/UI/test_settings_save_commit_models.py pinning inline labels, visual-separation class, and per-field Save rows.
5. Run settings suites (hub, footer hints, sweep) + new file; confirm only known pre-existing failures.
ADR required: no
ADR path: N/A
Reason: no change to what is persisted or when - ADR-020 already governs model-catalog instant persistence and ADR-031 governs hint honesty; this task only labels/documents the existing models.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Follow-up (code quality review, same tree): splash guided message now reads 'Splash defaults: applies immediately - no Save needed.'; the stale-hours Input got its own inspector branch ('Refresh after (hours)', model_catalog.stale_after_hours, numeric validation) instead of sharing checkbox copy; the Console 'Save scope' row was renamed 'Scope' to sit cleanly next to 'Save: staged...'; INSTANT_APPLY_BEHAVIOR_COPY cross-comments its mirror in settings_splash_screen_viewer.py; splash hint now reads 'applies immediately - no Save needed; text fields apply on Enter' (duration/animation-speed persist on Input.Submitted); added a uniform-row-count invariant test sweeping every guidance branch per category plus a stale-hours branch test (7 tests in the file now). Full run: 260 passed across save-commit-models + hub + footer-hints + sweep.
<!-- SECTION:NOTES:END -->
