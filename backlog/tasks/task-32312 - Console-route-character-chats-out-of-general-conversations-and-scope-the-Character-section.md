---
id: TASK-32312
title: >-
  Console: route character chats out of general conversations and scope the
  Character section
status: Done
assignee:
  - '@robert'
created_date: '2026-09-11 14:58'
updated_date: '2026-09-11 15:08'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Character conversations currently appear in the Conversations section's flat list mixed with regular chats, and the Character section counts every local character conversation regardless of workspace scope. Apply the one-owner routing rule end to end: character conversations leave the flat lane (they belong to the Character section, or their workspace Tree node when workspace-scoped), and the Character section's browse/page/search/unavailable queries exclude workspace-scoped character conversations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Global/Default character conversations no longer render in the Conversations flat list (persisted and live native rows)
- [x] Workspace-scoped character conversations appear only in their workspace Tree node (never under their character)
- [x] Character section recent groups, per-character paging, keyword search, and unavailable queries exclude workspace-scoped character conversations
- [x] Character row identity (character_id/character_label) survives normalization and the markers overlay
- [x] Targeted tests cover the routing rules; docs updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no\nADR path: N/A\nReason: routing policy over existing data and queries; no schema, storage, or interface-boundary change.\n\n1. TDD: state-builder tests — character rows (global or Default scope) leave the flat Chats lane; identity survives normalization and overlay.\n2. TDD: controller tests — persisted global rows carry character_id/character_label from the cards DB; live native character sessions carry identity through local_character_id().\n3. TDD: repository tests — recent groups, per-character paging exclude workspace-scoped character conversations (workspace wins).\n4. Implement the shared _CHARACTER_SCOPE_SQL predicate across the Character section's read queries (browse, page, keyword, unavailable); leave repair/indexing paths unfiltered.\n5. Update Docs/User_Guide/console/sessions-tabs-workspaces.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** Character conversations now follow the rail's one-owner rule
end to end: they no longer appear in the Conversations section's flat list,
and the Character section's groups, paging, keyword search, and unavailable
pages count and list only global/Default-scope character conversations
(workspace-scoped character chats stay in their workspace's Tree node).

**Approach.**

- `Workspaces/conversation_browser_state.py`: `ConsoleConversationBrowser
  InputRow`/`Row` gained `character_id`/`character_label`; `_belongs_to_chats`
  (the flat lane's predicate) excludes rows with a character id. Identity is
  threaded through `_normalize_input_row`, `_to_browser_row`, and the markers
  overlay, so excluded rows stay self-describing for their owning lanes.
- `UI/Console_Modules/workspace.py`: the persisted global/Default fetch reads
  `character_id` from the normalized conversation payload (already present —
  no service or DB API change) and labels it from a new
  `_console_browser_character_labels()` helper over
  `chachanotes_db.list_character_cards` (deleted cards degrade to an empty
  label). `_native_console_browser_rows` carries the live session's
  `local_character_id()`/`character_name`.
- `DB/character_conversation_search.py`: one shared `_CHARACTER_SCOPE_SQL`
  predicate (global/NULL scope, or workspace scope bound to Default/NULL)
  applied to the Character section's eight read queries — recent-group
  summaries and rows, per-character count and keyset page, keyword-search
  count and candidates, and the unavailable total/sources. Repair and
  keyword-index maintenance paths stay unfiltered: they maintain data health,
  not lane ownership, and a workspace-scoped chat with a missing card still
  needs repair visibility.

**Tests (TDD; each watched failing first).** State-builder: flat lane
excludes global and Default-scoped character rows; identity survives
normalization and the overlay. Controller: persisted rows route character
identity with cards-DB labels and degraded fallback; the built flat state
shows no character rows end-to-end; native character sessions carry
identity. Repository: recent groups and per-character paging exclude
workspace-scoped conversations. Runs: 23 state tests, 84 navigation/state
tests, 57 repository tests (incl. 2 new), 61 Character-context UI tests
pass. `test_console_workspace_controller.py` shows 2 pre-existing failures
on clean dev (constructor-docstring gap for `notify_character_navigation`;
one cancellation rollback case) — verified identical with this branch's
changes stashed.

**Docs.** `Docs/User_Guide/console/sessions-tabs-workspaces.md`: the
Conversations section now states character conversations never appear there,
and the Character section states the workspace-wins exclusion.

**Modified files.** `tldw_chatbook/Workspaces/conversation_browser_state.py`,
`tldw_chatbook/UI/Console_Modules/workspace.py`,
`tldw_chatbook/DB/character_conversation_search.py`,
`Tests/Workspaces/test_console_conversation_browser_state.py`,
`Tests/UI/test_console_workspace_controller.py`,
`Tests/DB/test_character_conversation_search_projection.py`, and the doc
above.

**ADR check.** Not required — routing policy over existing data and queries;
no schema, storage, or interface-boundary change.

**Lessons.** None generalizable beyond this task surfaced.
<!-- SECTION:NOTES:END -->
