---
id: TASK-354
title: >-
  Fix rail Chats list silently capping at 11 conversations with no overflow
  affordance
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-22 05:18'
labels:
  - console
  - ux
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
- [x] #1 Either list all conversations (scrollable), or show an explicit 'N more — search with Ctrl+K / Show all' disclosure row at the section end
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (code-verified): the grouped browser caps each group at
`CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT` (12); `_visible_rows` drops the
overflow and computes `hidden_count`, but `_build_status_copy` returned `""`
unless a search query was active, so the default no-query view had ZERO
disclosure — the oldest conversation was simply gone with no hint (only Ctrl+K
fuzzy-search could still reach it). task-138 covered the OLD rail's searchable
subsection; the newer grouped browser's no-query view was uncovered.

Fix (chose the AC's "explicit disclosure" option, not "list all"): compute the
silently-capped count and surface it as the existing `status_copy` line (which
the tray already renders unconditionally, right under the always-present search
box). `_build_status_copy` now, when no query is active and rows were capped,
returns `"{n} more conversation(s) — search with Ctrl+K"`. The count comes from
a new `_capped_hidden_count(sections)` = `sum(section.hidden_count for
non-collapsed sections)`: `_visible_rows` reports `hidden_count=0` for collapsed
groups, so a non-collapsed section's `hidden_count` is PURE cap overflow —
user-collapsed sections (which show their own header count) are correctly
excluded and never inflate the disclosure. The search-mode "Showing X of Y" copy
is unchanged.

Verified: 3 state unit tests (discloses when capped; no disclosure when nothing
capped; excludes user-collapsed sections) + 1 tray render test mounting the real
`ConsoleWorkspaceContextTray` with 15 conversations and asserting the
`#console-workspace-conversation-search-status` static shows "3 more … Ctrl+K"
in the no-query view. Files:
`tldw_chatbook/Workspaces/conversation_browser_state.py`,
`Tests/Workspaces/test_console_conversation_browser_state.py`,
`Tests/UI/test_console_rail_sections.py`.
<!-- SECTION:NOTES:END -->
