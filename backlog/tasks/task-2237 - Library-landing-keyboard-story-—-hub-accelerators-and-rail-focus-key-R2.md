---
id: TASK-2237
title: 'Library: landing keyboard story — hub accelerators and rail focus key (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 20:47'
labels:
  - ux-review
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The landing offers exactly one working key (/). Hub CTAs need single-letter accelerators (advertised in footer) and the rail needs a focus key. Post-fix re-review P1. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hub next-action rows have working advertised accelerators,A rail-focus key exists and is advertised,Tests updated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landing keyboard story: 'i'/'n' open Add content / New note from the landing via the same guarded row-switch dispatch as the hub action rows (screen-level on_key handler, never a Binding, never while an Input/TextArea owns focus, landing-scoped). F6 works for the first time on Library: _WORKBENCH_FOCUS_TARGETS (rail -> #library-search-input, landing canvas -> #library-hub-action-import) via the shared focus_relative_workbench_pane idiom -- previously F6 dead-ended in a 'no target' notification. Footer is now three honest contexts: LIBRARY_SHORTCUTS (search row, unchanged), LIBRARY_LANDING_SHORTCUTS (/, i, n, F6 -- landing only), LIBRARY_GENERAL_SHORTCUTS (/, F6 -- other canvases), so no key is advertised where it doesn't work. Files: library_screen.py (constants, _register_footer_shortcuts, on_key, action_focus_next_workbench_pane, workbench_focus import), Tests/UI/test_library_shell.py (4 new/updated tests), test_screen_footer_hints.py (tuple pin + production route test string), test_library_skills_canvas.py (registration assertion now LIBRARY_GENERAL_SHORTCUTS), Docs/User_Guide/library.md. Verified: 3 RED->GREEN (guard test passed pre-implementation as the correct pin); targeted shell keys tests 8 passed; footer-hints + skills files 113 passed; ruff clean (1 pre-existing F401 untouched). Note: committed on targeted-suite evidence per session instruction; the broader background sweep (shell+footer+skills+destination+visual audit) is confirmatory -- will verify on completion. ADR: not required (key handling + footer contexts; no schema/boundary changes). Commit 96ce4924e.
<!-- SECTION:NOTES:END -->
