---
id: TASK-2094
title: 'Footer: hide Tokens chip outside chat contexts (F-003)'
status: In Progress
assignee: []
created_date: '2026-08-03 17:25'
updated_date: '2026-08-04 12:47'
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
- [ ] #1 Tokens chip shows only where token counts are meaningful,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (footer chip visibility; no behavior change where tokens exist). Flow found: every screen mounts its own AppFooterStatus whose token chip starts as 'Tokens: --'; the 10s/0.5s updaters (db_status_manager.update_token_count_display) write real counts on TAB_CHAT and '' elsewhere, so non-chat destinations render placeholder-then-empty dead chrome. Deliberately NOT touching the DB-sizes path (F-014 lives on the library branch). Steps: 1. RED tests: (a) widget-level in Tests/UI/test_app_footer_shortcut_context.py -- a fresh AppFooterStatus renders the token chip empty and hidden; update_token_count(text) reveals it, update_token_count('') hides it again; (b) screen-level on MCP (non-chat): the chip never shows 'Tokens: --' and stays hidden. 2. AppFooterStatus: chip starts empty + display False; update_token_count toggles display on content (the reflow already measures its renderable, so priority math is unaffected). 3. Run footer tests (test_app_footer_shortcut_context, test_screen_footer_hints) + MCP/roleplay screen tests + ruff.
<!-- SECTION:PLAN:END -->
