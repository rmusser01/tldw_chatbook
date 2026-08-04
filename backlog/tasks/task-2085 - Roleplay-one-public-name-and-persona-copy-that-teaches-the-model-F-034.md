---
id: TASK-2085
title: 'Roleplay: one public name and persona copy that teaches the model (F-034)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 09:50'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nav 'Roleplay' vs header 'Roleplay & Chat Dictionaries' vs mode 'Personas' ('assistant profiles') muddles who-plays-who. Evidence: personas_screen.py:771,3133. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Screen uses one public name consistently,Personas descriptor teaches 'who you play',Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update tests asserting 'Roleplay & Chat Dictionaries' / 'assistant profiles' first. 2. personas_screen.py: header title -> 'Roleplay' at both sites (compose + _update_title); personas mode descriptor -> 'Personas — who you play in the chat.' 3. shell_destinations.py full_label and app.py action descriptions -> 'Roleplay' for one consistent public name. 4. Update palette/shell-destination test assertions; run destination/palette/personas suites + ruff. ADR required: no - user-facing label/copy unification; no behavior, schema, or boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Unified the public name as 'Roleplay': screen header title (both personas_screen.py sites), shell_destinations full_label, and the six app.py action descriptions; the retired 'Roleplay & Chat Dictionaries' long form also left the palette alias terms. Personas descriptor is now 'Personas — who you play in the chat.' (teaches the genre convention against Characters' 'who the AI plays') without touching the human-identity prohibition (forbidden-copy test retained, renamed). Files: tldw_chatbook/UI/Screens/personas_screen.py, tldw_chatbook/UI/Navigation/shell_destinations.py, tldw_chatbook/app.py; tests in Tests/UI/test_{shell_destinations,command_palette_providers,command_palette_shell_routes,personas_workbench}.py. Verified: 556 passed/1 skipped gate (workbench, shell destinations, palette x2, screen navigation, destination shells); ruff clean. ADR: not required (label/copy unification only).
<!-- SECTION:NOTES:END -->
