---
id: TASK-2079
title: 'Library: scroll affordance for clipped rail sections (F-020)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 06:30'
labels:
  - ux-review
  - library
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rail scrolls but shows no scrollbar/affordance, so Create and Details are undiscoverable when clipped. Evidence: library-100x30.png. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clipped rail content has a visible affordance (scrollbar, fade, or more-indicator),Tests/snapshot updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (one CSS token + tests). The affordance convention already exists: task-1712 fixed this exact bug class on #settings-category-list ('$ds-grid-line blended into the panel... the overflow cue must actually read as one') by moving the thumb to $ds-text-muted. Steps: 1. RED tests: (a) at 100x30 the rail's content overflows and rail.show_vertical_scrollbar is True; (b) drift guard: the #library-rail block in _agentic_terminal.tcss carries scrollbar-color: $ds-text-muted (the task-1712 token), not the panel-blending $ds-grid-line. 2. Change the one token in _agentic_terminal.tcss, regenerate tldw_cli_modular.tcss via build_css.py, check_bundle_sync.py green. 3. Run shell suite + rail tests + css sync guard + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The rail already styled its scrollbar, but the thumb used $ds-grid-line on a $ds-surface-panel track -- invisible in practice (the F-020 capture shows a track with no readable thumb, so clipped sections were undiscoverable). Fix matches the app's existing convention for this exact bug class: task-1712 moved #settings-category-list's thumb to $ds-text-muted ('the overflow cue must actually read as one'); #library-rail now uses the same token (hover/active unchanged). One token in _agentic_terminal.tcss; bundle regenerated via build_css.py; check_bundle_sync.py green. Files: tldw_chatbook/css/components/_agentic_terminal.tcss, tldw_chatbook/css/tldw_cli_modular.tcss, Tests/UI/test_library_shell.py (new test_rail_shows_a_visible_scrollbar_when_content_overflows: 100x30 overflow -> show_vertical_scrollbar + token drift guard; existing test_library_rail_css_scrolls_vertically_with_scrollbar_styling pin updated to the new token on both CSS files). Verified: new/updated tests RED->GREEN; full test_library_shell.py 314 passed + the one stale pin (fixed and re-verified green in the targeted run; the full-suite failure was the test-only token string, no production code involved); destination/parity/contract sweep 225 passed + 1 skip; visual: live 100x30 capture shows a clearly readable thumb marking more content below. Ruff clean. ADR: not required (one CSS token + tests). Commit c047b5757.
<!-- SECTION:NOTES:END -->
