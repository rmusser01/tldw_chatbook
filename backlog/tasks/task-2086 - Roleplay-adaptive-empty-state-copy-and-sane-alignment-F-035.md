---
id: TASK-2086
title: 'Roleplay: adaptive empty-state copy and sane alignment (F-035)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 10:04'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Empty copy renders 'use New or Import' even with characters present, centered in a huge void (reads broken, right-aligned at some widths). Evidence: personas_screen.py:263-269,870-874. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty copy adapts to whether the library has items,Copy alignment matches app conventions (left/centered deliberately),Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests first: empty library shows New/Import copy; non-empty library with cleared selection (mode round-trip) shows picker copy; styled test pins left/top alignment per .chat-empty-state convention. 2. personas_screen.py: _characters_empty_guidance_text adapts on _character_total; empty-library copy stops claiming 'pick from the list'; #personas-characters-empty CSS switches from center-middle to left-top alignment. 3. Run guidance + suites + ruff. ADR required: no - empty-state copy/CSS only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Empty-state copy now adapts on _character_total: empty library -> 'No characters yet — use New or Import to add one.' (no longer claims a list exists); non-empty library with cleared selection -> 'Pick a character from the list to see it here.' Reconciliation with task-2082 auto-select: first paint auto-selects, so the picker variant serves post-delete and mode-round-trip no-selection states, and the New/Import variant serves first-run/empty libraries. Alignment switched from center-middle-in-a-void to left/top per the app's empty-state convention (.chat-empty-state). Files: tldw_chatbook/UI/Screens/personas_screen.py (constants, _characters_empty_guidance_text, #personas-characters-empty CSS); tests in TestCharactersEmptyStateGuidance (adaptive copy both ways + styled alignment pin). Verified: guidance class 8 passed; gate 299 passed (full workbench + phase6 replay); ruff clean. ADR: not required (copy/CSS only).
<!-- SECTION:NOTES:END -->
