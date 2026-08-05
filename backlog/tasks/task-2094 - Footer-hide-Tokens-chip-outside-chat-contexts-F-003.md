---
id: TASK-2094
title: 'Footer: hide Tokens chip outside chat contexts (F-003)'
status: Done
assignee: []
created_date: '2026-08-03 17:25'
updated_date: '2026-08-04 12:57'
labels:
  - ux-review
  - chrome
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Tokens: --' renders as dead chrome on authoring/config destinations (Roleplay, MCP). Evidence: roleplay/mcp-170x50.png. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tokens chip shows only where token counts are meaningful,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (footer chip visibility; no behavior change where tokens exist). Flow found: every screen mounts its own AppFooterStatus whose token chip starts as 'Tokens: --'; the 10s/0.5s updaters (db_status_manager.update_token_count_display) write real counts on TAB_CHAT and '' elsewhere, so non-chat destinations render placeholder-then-empty dead chrome. Deliberately NOT touching the DB-sizes path (F-014 lives on the library branch). Steps: 1. RED tests: (a) widget-level in Tests/UI/test_app_footer_shortcut_context.py -- a fresh AppFooterStatus renders the token chip empty and hidden; update_token_count(text) reveals it, update_token_count('') hides it again; (b) screen-level on MCP (non-chat): the chip never shows 'Tokens: --' and stays hidden. 2. AppFooterStatus: chip starts empty + display False; update_token_count toggles display on content (the reflow already measures its renderable, so priority math is unaffected). 3. Run footer tests (test_app_footer_shortcut_context, test_screen_footer_hints) + MCP/roleplay screen tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The token chip (per-screen AppFooterStatus) started as 'Tokens: --' everywhere; the periodic updater (db_status_manager.update_token_count_display, 0.5s one-shot + 10s interval) writes real counts only on TAB_CHAT and '' elsewhere, so Roleplay/MCP rendered placeholder-then-empty dead chrome. Fix in AppFooterStatus only: the chip starts empty + display False, and update_token_count toggles display on content (text reveals, '' hides). The TASK-451 reflow already measures the chip's renderable, so priority math needed no change; one existing reflow test (test_footer_reflows_when_counts_change_without_a_resize) was calibrated on the placeholder's 10 cells and is recalibrated 100->90 cols with a comment. The DB-sizes path is deliberately untouched (F-014 lives on the library branch, unmerged). Files: tldw_chatbook/Widgets/AppFooterStatus.py, Tests/UI/test_app_footer_shortcut_context.py (new chip-visibility test + recalibration), Tests/UI/test_screen_footer_hints.py (production test pins the chip hidden on MCP, initially and past the 0.5s updater tick). Verified: both RED->GREEN; footer files 12 passed; ui_responsiveness 14 passed; final sweep (destination_shells + visual audit + footer files + responsiveness) 134 passed + 1 skip; ruff clean. Live MCP 170x50 capture: footer shows only '1-4 mode | a add server | r refresh' -- no Tokens chip. Deferral: the word-count chip has the same always-rendered shape but is meaningful on notes/editor surfaces -- out of F-003's scope. ADR: not required (chip visibility only). Commit 85ce5f4de.
<!-- SECTION:NOTES:END -->
