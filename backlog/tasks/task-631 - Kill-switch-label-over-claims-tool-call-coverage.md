---
id: TASK-631
title: Kill switch label over-claims tool-call coverage
status: Done
assignee: []
created_date: '2026-07-25'
labels: [ux, security, mcp, tools]
dependencies: [TASK-545]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-545/P1 relabelled the MCP Permissions mode kill switch to stop reading as MCP-only, since `BuiltinToolProvider.invoke` now honors the same `service.get_kill_switch()`/`set_kill_switch()` pair. The button label (`UI/MCP_Modules/mcp_permissions_mode.py`, `_kill_switch_label`) now reads `"block tool calls in chat: yes/no ▸"` and its tooltip says "Master kill switch for chat tool calls — takes effect with the chat bridge." Both now claim broad coverage ("tool calls in chat" / "chat tool calls").

That claim is broader than reality. `build_tool_review_hook`'s own docstring is explicit that "a name neither provider claims (a skill, `spawn_subagent`, `find_tools`, ...) passes through unreviewed" — and those names are never routed through `MCPToolProvider.invoke` or `BuiltinToolProvider.invoke` at all (`find_tools`/`load_tools`/`spawn_subagent` are native closures defined directly in `Agents/agent_service.py`, not catalog-registered tools), so the kill switch's `get_kill_switch()` check is never even consulted for them. A user who flips the switch to stop all tool calls — the behavior the label promises — will still have skill tools, `spawn_subagent`, `find_tools`, and `load_tools` running normally. This is a false sense of security in a security-relevant control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Either (a) the kill switch's label/tooltip wording is narrowed to accurately describe what it covers today (MCP tools + built-in agent-runtime tools, not skills/spawn/find/load), or (b) the switch's enforcement is extended so skill tools, `spawn_subagent`, `find_tools`, and `load_tools` all also honor it
- [x] Whichever approach is chosen, a test or docstring makes the actual coverage explicit so a future reader cannot reintroduce the same gap silently
- [x] If narrowing wording only: the change is confined to the label/tooltip strings, with no behavior change
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Option (b): enforcement extended, label untouched -- the label's promise is now TRUE.**

`build_tool_review_hook` takes `kill_switch: Callable[[], bool] | None` and, when it reports on, refuses EVERY call in the batch without prompting -- `KILL_SWITCH_REFUSAL` per call id (name for id-less fence calls, fail-closed per TASK-1861's reasoning). The hook is the one place every parsed call passes, including the four families neither provider claims (skills, `spawn_subagent`, `find_tools`, `load_tools`) that previously ran normally with the switch on; the runtime already converts non-"proceed" verdicts into results without dispatch (pinned by the TASK-1861 tests), so no new plumbing.

A callable, read fresh per turn, so a mid-run flip takes effect on the next batch; an unreadable switch fails CLOSED -- the only safe answer for a security control that cannot be read. Wired via `_console_tool_kill_switch_reader()` (None without a service, matching `_compose_mcp_provider`).

Mutation-verified: deleting the hook's check fails the refusal test. Also corrected the hook docstring's stale "verdict map is purely documentary" claim, false since TASK-1861.
<!-- SECTION:NOTES:END -->
