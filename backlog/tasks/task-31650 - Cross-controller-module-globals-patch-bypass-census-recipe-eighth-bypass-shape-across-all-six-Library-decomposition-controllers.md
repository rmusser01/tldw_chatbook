---
id: TASK-31650
title: >-
  Cross-controller module-globals patch-bypass census (recipe eighth bypass
  shape) across all six Library decomposition controllers
status: To Do
assignee: []
created_date: '2026-09-05 14:04'
labels:
  - library
  - decomposition
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave-5 ingest task 2's post-landing review found the recipe's eighth bypass shape: a moved method body's bare module-global read (an ordinary 'from module import name' in the pre-move file) can be patched at the OLD screen-scoped module path by a test whose own assertions still pass after the move anyway -- a green-but-vacuous test, never a red one, because an independently-true condition makes the assertion pass whether or not the patch is actually observed. The ingest series' own worked example, _resolve_ingest_source reading validate_path_simple/validate_url, was an ACTIVE collision (fixed by exclusion); its own module docstring also confirms _apply_library_ingest_backend_save's bare _sync_library_canvas read is LATENT (10 files/38 sites, none reaching the mover's own call path). _sync_library_canvas is the shared cross-subsystem canvas-sync dispatcher every one of the six Library decomposition controllers (conversations, export, collections, search+RAG, skills, ingest) imports the identical way, and every one of the five controllers landed BEFORE this mechanical census existed shipped without it ever being run against their own moved-method sets. This is a census, not a presumed bug: the point is to know, not to guess, whether any of the five prior controllers carries an ACTIVE (not just latent) version of the same shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The recipe's own eighth-bypass-shape 4-step mechanical module-globals census (backlog/docs/library-decomposition-recipe.md section 3) has been run against each of the six Library decomposition controllers' own moved-method sets (library_conversations_controller.py, library_export_controller.py, library_collections_controller.py, library_rag_search_controller.py, library_skills_controller.py, library_ingest_controller.py), with every bare-module-global finding classified ACTIVE or LATENT and the verdict plus supporting evidence recorded in that controller's own module docstring
- [ ] #2 Any ACTIVE collision the census finds is fixed in the same task by excluding the affected method (reverted to the screen byte-for-byte) and rebinding its mover caller through a named late-binding dependency, verified with an existing-file probe (a temporary, uncommitted stub change) showing the covering test fails pre-fix and passes post-fix, mirroring the ingest series' own _resolve_ingest_source remedy
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed at wave-5 close per the ledger's cross-wave follow-up flag (wave-5 task 2, e3d85ad21) and recipe section 3's eighth-bypass-shape entry. Motivating example: the _sync_library_canvas latency already confirmed present (latent) in prior-wave controllers per that entry's own worked example.
<!-- SECTION:NOTES:END -->
