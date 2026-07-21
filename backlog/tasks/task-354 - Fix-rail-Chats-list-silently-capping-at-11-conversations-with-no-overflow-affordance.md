---
id: TASK-354
title: Fix rail Chats list silently capping at 11 conversations with no overflow affordance
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With 12 seeded conversations, the rail's Chats section listed 11 (Websocket...Debug flaky) in every state, including after wheel-scrolling the rail to its end (blank rows below the last item). 'Refactor auth middleware plan' (the oldest) was absent across all sessions. It IS reachable via Ctrl+K fuzzy search ('refac' finds it), but nothing in the rail discloses that items are hidden, so a user scanning the rail would reasonably conclude the conversation was deleted. There is also no search/filter affordance inside the rail itself.

**Repro:** Seed 12 conversations (mk_home variant seeded). Open Console, expand/scroll the rail Chats section to its bottom: only 11 saved rows exist, oldest missing, no indicator. Ctrl+K + 'refac' finds the hidden one.

**Verifier note:** Code-verified silent cap: CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT=12 (conversation_browser_state.py:14), _visible_rows drops overflow and computes hidden_count, but ConsoleWorkspaceContextTray never renders it, and _build_status_copy returns '' unless a search query is active (line 628) — so the default view has zero disclosure ('Showing X of Y' exists only mid-search). task-138 shipped 'cap states with explicit copy' for the old rail's searchable subsection; the newer grouped browser's no-query view is an uncovered gap, not a re-report. 12 seeded + native sessions exceed the cap, hiding the oldest with no hint.

**Source:** Console UX expert review 2026-07-20 (finding j2-rail-hides-oldest-conversation; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J2 returning power user journey. Evidence: `j2-04-rail-scrolled.png`, `j2-20-boot2-rail.png`, `j2-36-switcher-refac.png`, `j2-55-final-boot-rail.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Either list all conversations (scrollable), or show an explicit 'N more — search with Ctrl+K / Show all' disclosure row at the section end
<!-- AC:END -->
