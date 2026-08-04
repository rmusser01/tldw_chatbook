---
id: TASK-2231
title: >-
  Roleplay: redesign center pane — character card fills the view, attachments
  collapsible below (R2)
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 17:36'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the post-fix re-review: Dictionaries + World Books panels own ~50% of the center canvas even when empty, the card is clipped at 100x30, and a ~10-line dead void sits between the panels at 170x50 (bottom-dock workaround). User direction: give the center space to the current focus, let the user scroll the center area, make chat dictionaries + world books collapsible; by default the center view is filled with the character info, then the user scrolls down to attachments. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Center area is one scrollable column with character card/content first and attachments below
- [x] #2 Default view shows the character info filling the center (attachments collapsed or below the fold)
- [x] #3 Dictionaries and World Books are collapsible sections the user expands in place
- [x] #4 Empty attachment panels render as one collapsed line (not 16 lines of 'nothing attached')
- [x] #5 At 100x30 the character card remains fully visible (real min-height)
- [x] #6 No dead void between panels at 170x50
- [x] #7 Existing attach/detach flows and tests keep working
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no - UI layout change within the Roleplay center pane; no storage/schema, sync, data-ownership, provider/runtime-boundary, service-contract, or security implications. ADR path: N/A.
1. Make #personas-detail-stack a VerticalScroll; remove the bottom-dock wrapper CSS (dock + max-height caps) so the character card and the attachment sections flow in document order (card fills the viewport, attachments below the fold).
2. Turn the dictionaries/world-books panels into collapsible sections using the screen's established preview-pane disclosure idiom (full-width toggle button, arrow-first label with live count, e.g. '▸ Dictionaries (2)'); collapsed by default; collapsed state kept in a widget attribute that load_* never resets (persists for the session); an empty section renders as exactly one line.
3. Give the character card a real min-height on top of its viewport-filling height so empty panels can never displace it at 100x30.
4. Update tests asserting the old dock layout (geometry class, attach-flow clicks now expand the section first); add layout tests for the new contract at 170x50 and 100x30.
5. Run targeted persona tests + ruff; capture headless screenshots at both sizes and eyeball them.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Center pane redesigned per the user's direction. ADR: none (UI layout only, stated in the plan).

Approach: #personas-detail-stack is now a VerticalScroll and the character card keeps height: 100% (plus a new min-height: 10), so the card fills the center viewport by default and the attachment sections flow below the fold in document order - the bottom-dock wrapper and its max-height caps are gone (AC#1/#2/#5/#6). PersonasCharacterDictionariesWidget / PersonasCharacterWorldBooksWidget are now collapsible sections in the screen's established preview-pane disclosure idiom: a full-width toggle button as the section header carrying the live count ('▸ Dictionaries (0)' / '▾ World Books (2)'), body hidden by default, empty section = exactly one line (AC#3/#4). Collapse state is a widget attribute that load_* never resets, so it survives attach/detach refreshes for the session.

Trade-off / follow-on fix: making the stack scrollable changed height resolution for siblings shown together - the conversation transcript (height: 100%) + 3-line actions row overflowed by 3 and focus auto-scroll hid the Back/Continue buttons. Fixed by sizing the transcript 1fr (viewport minus the actions row), restoring the pre-scroll geometry in conversation view (18 conversation tests green).

Tests: new Tests/UI/test_personas_center_canvas_layout.py (6 tests pinning the contract at 170x50 and 100x30: card fills viewport, sections one line each and adjacent, scroll reveals, expand/collapse + count + state persistence, card content visible at 100x30); collapse-contract tests added to both panel widget test files; attach-flow tests now expand the section before clicking (AC#7); the dock-era geometry class in test_personas_character_world_books_screen.py rewritten to the flow contract. Full personas UI suite: 788 passed, 4 failed - the 4 (Tests/UI/test_personas_generation_wiring.py, editor generation wiring) fail identically on the pristine branch (verified via stash); pre-existing and unrelated. ruff clean on all touched files; check_bundle_sync green (no app-tier CSS touched).

Visual: headless before/after captures at both sizes in output/ux-r2/ (roleplay-{before,after}-{170x50,100x30}.svg.png plus after-170x50-scrolled) - before: card fully displaced by empty panels at 100x30 and dead void at 170x50; after: card fills the viewport at both sizes and the sections reveal as one-line headers on scroll.

Files: tldw_chatbook/UI/Screens/personas_screen.py; tldw_chatbook/Widgets/Persona_Widgets/{personas_character_dictionaries,personas_character_world_books,personas_character_card_widget,personas_conversation_transcript_widget}.py; Tests/UI/test_personas_center_canvas_layout.py (new); Tests/UI/{test_personas_character_dictionaries,test_personas_character_world_books,test_personas_character_attach,test_personas_character_world_books_screen}.py.
<!-- SECTION:NOTES:END -->
