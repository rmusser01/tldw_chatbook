---
id: TASK-2073
title: 'Library: keyboard affordances on the landing state (F-012)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 02:34'
labels:
  - ux-review
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Footer shows only quit/palette on landing; the u (use in Console) hint registers only when Search row is selected; no focus-search key; teaching copy is hover-only. Evidence: library_screen.py:1413-1417. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Landing footer advertises the keys that work in that state,A focus-search shortcut works and is advertised,Key tooltip-only teaching copy also appears inline or in footer,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (footer hint contexts + one focus key; no schema/boundary changes). Steps: 1. RED tests: landing footer shows '/ focus search' (replacing the bare default); search-row footer gains the '/' hint; '/' focuses #library-search-input from landing; '/' while another Input has focus types literally; '/' on the focused rail box re-arms (select-all, settings task-1584 pattern); nav-context deep link to the Search canvas registers the 'u' hint (today only the rail-row switch re-registers). 2. library_screen.py: add LIBRARY_LANDING_SHORTCUTS, extend LIBRARY_SHORTCUTS with ('/', 'focus search'), make _register_footer_shortcuts a two-context map (search row vs everywhere else), add settings-style on_key '/' handler (skip when Input/TextArea focused), re-register at the end of _apply_navigation_context_state. 3. library_rail.py: LibraryRailSearchInput subclass whose '/' re-arms (select-all) instead of inserting, mirroring SettingsCategorySearchInput. 4. Update pinned footer strings in test_screen_footer_hints.py. 5. Run footer-hint/shell/rail/parity tests + ruff. Skill-editor ctrl+s/esc hints deliberately NOT in scope (F-019/task-2078).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two-context footer map: LIBRARY_SHORTCUTS (Search/RAG row: u/enter/o plus '/') vs new LIBRARY_LANDING_SHORTCUTS ('/ focus search') everywhere else, so each state advertises exactly the keys that work there -- the landing state no longer shows the bare global default. '/' focuses the rail search box via a screen-level on_key handler (settings task-1715 pattern; never a Binding, so no key-palette noise and no focus-steal out of Inputs/TextAreas). A second '/' on the focused box re-arms (select-all) via new LibraryRailSearchInput, mirroring SettingsCategorySearchInput (task-1584). The u-hint gap is closed: _apply_navigation_context_state now re-registers the footer, so deep links into the Search canvas (mode=search) advertise u where it works. Files: library_screen.py (shortcut constants, _register_footer_shortcuts, on_key, nav-context re-register), library_rail.py (LibraryRailSearchInput), Tests/UI/test_library_shell.py (5 new tests: landing footer, '/' focus, literal typing in other inputs, re-arm, deep-link registration), Tests/UI/test_screen_footer_hints.py (pinned strings updated). Verified: 5 new tests RED->GREEN; Tests/UI/test_screen_footer_hints.py + Tests/Widgets/Library/test_library_rail.py green (7 passed); full Tests/UI/test_library_shell.py green (309 passed). Ruff clean on changed files. Skill-editor ctrl+s/escape hints deferred to task-2078 (F-019). Docs: Docs/User_Guide/library.md footer/row updates land with the F-013 commit. ADR: not required (footer hint contexts + one focus key; no schema/boundary changes). Commit 79e328363.
<!-- SECTION:NOTES:END -->
