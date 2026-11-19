---
id: TASK-36519
title: Approval cards, provider-error copy and the character tools still send users to the retired MCP screen
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: '2026-11-05 21:09'
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
- **The approval card's scope lines** (`DECISION_SCOPE_COPY`, chat_approval_card.py:93-102): "Remove it under MCP ▸ Tools ▸ this tool." and "Change it under MCP ▸ Permissions." The scope line is its own `.approval-row-scope` Static, and `Tests/UI/test_approval_row_information_budget.py::test_the_scope_line_states_the_selected_decision_and_follows_it` pins it to one row painted at 80 columns. A longer "Settings ▸ Tools & MCP ▸ Permissions" has to fit there.
- **Provider tool-rejection copy** (console_provider_gateway.py:1803-1822): "turn MCP servers off on the MCP screen", "on the MCP screen, then send again", "has a switch on the MCP screen".
- **External-client refusals** (MCP/builtin_tool_policy.py:53 and :57): "allow this tool in MCP Permissions" and "disabled in MCP Permissions".
- **The character-creator skill** (assets/skills/character-creator/SKILL.md:17-19), which the model reads and repeats to the user: "set those two read tools to Allow in the MCP hub (Permissions)".

These pins quote the old strings: `Tests/UI/test_chat_approval_card.py`, `Tests/Chat/test_console_tool_definition_rejection.py`, `Tests/Agents/test_provider_tool_schema_conformance.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 None of the listed strings names an MCP screen, MCP hub, 'MCP Permissions' or 'MCP ▸ …' as a place; each names Settings ▸ Tools & MCP and, where it matters, the mode (Tools or Permissions)
- [x] #2 The approval card's scope line stays on one row at 80 columns, and test_the_scope_line_states_the_selected_decision_and_follows_it passes
- [x] #3 The User Guide quotes the new lines, and the note explaining the old wording is gone
- [x] #4 A test fails if one of these surfaces names MCP as a place of its own again
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the approval card's scope-line width at 80 columns (74 cells) and rewrite both persistent scope lines to name Settings ▸ Tools & MCP plus the mode, each inside that width.
2. Rewrite the three provider tool-rejection sentences (now in `Chat/refused_stream.py`) to name Settings ▸ Tools & MCP ▸ Servers, and the built-in group switch's real pane path (`TOOL_GATES_PANE_PATH`).
3. Rewrite the two external-client refusals in `MCP/builtin_tool_policy.py` and the character-creator SKILL.md line; re-pin the skill's SHA-256 in `Skills_Interop/builtin_skills.py`.
4. Add one guard test in a lane-run file that renders every listed surface and fails if MCP is named as a place of its own; show it red on dev's copy first.
5. Update the pins that quote the old strings, and the User Guide table; delete the note that explained the old wording.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Every listed surface now names Settings ▸ Tools & MCP and the mode:
- Approval scope lines (`chat_approval_card.py` `DECISION_SCOPE_COPY`): "Remembered for these args. Remove in Settings ▸ Tools & MCP ▸ Tools." and "Remembered for this tool. Edit in Settings ▸ Tools & MCP ▸ Permissions." The row's scope Static is 74 cells wide at 80 columns (measured); both lines are 68 and 71 cells. The old allow_matching line was 80 cells and already wrapped to two rows.
- Provider tool-rejection copy (it moved from `console_provider_gateway.py` to `Chat/refused_stream.py`): MCP servers are turned off in Settings ▸ Tools & MCP ▸ Servers; a Chatbook tool's group switch is named by the existing `TOOL_GATES_PANE_PATH` (Settings ▸ Tools & MCP ▸ Servers ▸ built-in row ▸ Tool gates).
- External-client refusals (`MCP/builtin_tool_policy.py`) and the character-creator SKILL.md name Settings ▸ Tools & MCP ▸ Permissions. The skill's SHA-256 pin in `Skills_Interop/builtin_skills.py` is re-pinned.
- User Guide (`console/agent-runs-and-tools.md`): the scope table quotes the new lines, the "On these lines MCP is ..." note is gone (one line says Ctrl+9 opens Settings ▸ Tools & MCP), and the tool-rejection troubleshooting entry names the Servers mode and the Tool gates pane.

Guard (AC#4): `Tests/Agents/test_provider_tool_schema_conformance.py::test_no_surface_names_the_retired_mcp_screen_as_a_place` renders all eight surfaces (two scope lines, three rejection sentences, two external refusals, the skill file) and fails if one lacks "Settings ▸ Tools & MCP" or uses "MCP" other than as part of that name or before "server"/"tool". It renders rather than greps because two sentences were split across source lines. The file runs whole in derived-artifacts' admission-sensitive Console P0 step. Red on dev's copy: all 8 surfaces listed ("Left contains 8 more items"); green after.

AC#2: `test_the_scope_line_states_the_selected_decision_and_follows_it` now checks both persistent lines are one row and painted at 80 columns. Red with dev's card module ("allow_matching wraps at 80 columns"), green after.

Re-pinned old strings in `Tests/UI/test_chat_approval_card.py`, `Tests/Chat/test_console_tool_definition_rejection.py`, `Tests/Agents/test_provider_tool_schema_conformance.py` and `Tests/Skills/test_builtin_skills.py` (it counted "MCP hub").

Covering runs (load 35-50): the changed test files plus every test importing the changed modules, `-n 4`. The remaining reds are also red on dev source with the same file-level command: `test_console_mcp_approval.py` (19, CSS-bundle pins and harness failures), `test_console_provider_gateway.py` (2, base-URL pins), `test_builtin_skills.py::test_app_wires_loader_to_in_memory_config`. One load flake (`test_a_revoked_childs_card_can_no_longer_execute_its_tool`, precondition timing) passed 3/3 when rerun alone.
<!-- SECTION:NOTES:END -->
