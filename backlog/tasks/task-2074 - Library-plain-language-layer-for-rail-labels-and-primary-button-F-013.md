---
id: TASK-2074
title: 'Library: plain-language layer for rail labels and primary button (F-013)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 02:51'
labels:
  - ux-review
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ingest, RAG, Skills, Collections, Runtime are load-bearing labels with zero gloss; the guidance sentence presumes 'ingest'. Evidence: library_shell_state.py:7. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Jargon rows carry dim plain-language subtitles,Primary button reads as a plain verb phrase (e.g. Add content…),Landing guidance no longer requires knowing 'ingest',Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (labels/copy only). Steps: 1. RED tests: shell-state unit test that jargon rows (Media, Prompts, Skills, Collections, Search / RAG) carry plain-language subtitles; rail render test that a subtitle renders as a dim markup suffix on the same one-line row; updated pins for the new landing copy and the 'Add content…' top button (Tests/Library/test_library_shell_state.py, Tests/UI/test_library_shell.py, Tests/UI/test_library_screen.py, Tests/UI/test_command_palette_providers.py, Tests/UI/test_destination_shells.py, Tests/UI/test_destination_visual_parity_correction.py, Tests/UI/test_product_maturity_phase6_recovery_docs.py). 2. library_shell_state.py: LibraryRailRow.subtitle field; gloss the five jargon rows; rewrite LIBRARY_CANVAS_LANDING_COPY without 'ingest'. 3. library_rail.py: render subtitle as an escaped [dim]— suffix on the row line; Details 'Runtime' label -> 'Source'. 4. library_screen.py: top button 'Ingest content…' -> 'Add content…' with plain tooltip; app.py palette entry 'Library: Ingest content…' -> 'Library: Add content…'. 5. Docs/User_Guide/library.md: update rail/button/landing/footer section (covers F-011/F-012 doc drift too). 6. Run shell-state/rail/shell/parity/palette/library-screen tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plain-language layer shipped: (1) LibraryRailRow gains a subtitle field; the five jargon rows (Media 'imported files & transcripts', Prompts 'saved instructions for the AI', Skills 'installable AI abilities', Collections 'saved groups of content', Search / RAG 'search everything') render it as a dim em-dash suffix on the SAME one-line row -- the F-011 height contract is untouched, and the gloss sits after the count so narrow rails clip the expendable gloss first. (2) Rail-top primary button 'Ingest content…' -> 'Add content…' with plain tooltip; command palette entry follows ('Library: Add content…'); 'ingest' stays inside the ingest canvas. (3) Landing copy rewritten without 'ingest': 'Search everything, pick a section on the left, or add something new.' (4) Details 'Runtime' label -> 'Source'. (5) Docs/User_Guide/library.md + import-and-export.md updated (also covers F-011/F-012 doc drift). Tests: new subtitle pins in Tests/Library/test_library_shell_state.py; rendered dim-span test in Tests/UI/test_library_shell.py; updated pins in test_library_screen.py, test_command_palette_providers.py, test_destination_shells.py (incl. the wrapper param 'content type' -> 'pick a section'), test_destination_visual_parity_correction.py, test_product_maturity_phase6_recovery_docs.py. Verified: full test_library_shell.py 310 passed; fast batch 89 passed; destination/parity/library-screen batch 251 passed + 1 skip with only the 3 pre-existing dev-broken failures (confirmed failing at F-010 parent); usability smoke 3 passed. Ruff clean on all changed files. Follow-up noted: RAG empty-index recovery copy (library_local_rag_search_service.py:937,945, semantic_availability.py:70) still says 'Ingest content...' -- a separate surface, left for a later copy pass. ADR: not required (labels/copy only). Commit b37ea8c9e.
<!-- SECTION:NOTES:END -->
