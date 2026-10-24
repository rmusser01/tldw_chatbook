---
id: TASK-36519
title: Approval cards, provider-error copy and the character tools still send users to the retired MCP screen
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- console
- copy
- mcp
priority: low
references:
- tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/MCP/builtin_tool_policy.py
- tldw_chatbook/assets/skills/character-creator/SKILL.md
- Docs/User_Guide/console/agent-runs-and-tools.md
dependencies:
- TASK-36387
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36387's implementation (PR #3226, 2026-10-23). That task moved the Settings copy to Settings ▸ Tools & MCP. It left the Console copy as out of scope, and in agent-runs-and-tools.md it added a note explaining that "MCP" on the approval card means the Settings category.

Since task-35150 (#3140) and ADR-240 there is no MCP screen. Ctrl+9 deep-links to Settings ▸ Tools & MCP (`MCP_COMPATIBILITY_SHORTCUT`, shell_destinations.py). These strings still name a place that does not exist, both on dev 5b5920b8e2 and at PR #3226's head:
- **The approval card's scope lines** (`DECISION_SCOPE_COPY`, chat_approval_card.py:93-102): "Remove it under MCP ▸ Tools ▸ this tool." and "Change it under MCP ▸ Permissions." The row has a width budget: `.approval-row-decision`'s width minus 8 cells (chat_approval_card.py:75).
- **Provider tool-rejection copy** (console_provider_gateway.py:1803-1822): "turn MCP servers off on the MCP screen", "on the MCP screen, then send again", "has a switch on the MCP screen".
- **External-client refusals** (MCP/builtin_tool_policy.py:53 and :57): "allow this tool in MCP Permissions" and "disabled in MCP Permissions".
- **The character-creator skill** (assets/skills/character-creator/SKILL.md:17-19), which the model reads and repeats to the user: "set those two read tools to Allow in the MCP hub (Permissions)".

These pins quote the old strings: `Tests/UI/test_chat_approval_card.py`, `Tests/Chat/test_console_tool_definition_rejection.py`, `Tests/Agents/test_provider_tool_schema_conformance.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 None of the listed strings names an MCP screen, MCP hub, 'MCP Permissions' or 'MCP ▸ …' as a place; each names Settings ▸ Tools & MCP and, where it matters, the mode (Tools or Permissions)
- [ ] #2 The approval card's scope line stays on one row at the widths the approval card tests pin
- [ ] #3 The User Guide quotes the new lines, and the note explaining the old wording is gone
- [ ] #4 A test fails if one of these surfaces names MCP as a place of its own again
<!-- AC:END -->
