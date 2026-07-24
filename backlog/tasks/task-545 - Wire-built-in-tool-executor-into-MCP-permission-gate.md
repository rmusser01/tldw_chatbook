---
id: TASK-545
title: Wire built-in tool executor into MCP permission gate
status: To Do
assignee: []
created_date: '2026-07-24 12:00'
labels: [tools, security]
dependencies: [TASK-331]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Built-in fs/mutating tools (`write_file`, `create_note`, `update_note`, all default-off) auto-execute on model tool_calls with no allow/ask/deny gate. Wire `ToolExecutor` (Site A: `Event_Handlers/…execute_tool_calls`, main-loop) and/or the agent-runtime `BuiltinToolProvider` (Site B: worker-thread) into the existing `MCP/permission_store.py` model (`resolve_effective_state`, `EffectiveToolState`, kill switch, `HIGH_RISK_TAGS`). Add a risk-tag field to the `Tool` ABC and tag the mutating tools with it. Reuse `Widgets/Chat_Widgets/chat_approval_card.py` for the "ask" confirmation. TASK-331's sandbox fix made these tools functional-within-a-sandbox, so this gate is the intended protection layer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] A mutating built-in tool requested by the model is gated allow/ask/deny before execution
- [ ] Tool ABC has a risk_tags field and mutating tools are tagged with HIGH_RISK or similar
- [ ] Integration with permission_store.py resolve_effective_state is complete
- [ ] ask confirmation reuses chat_approval_card.py
- [ ] Unit test covers tool execution gates (allow, ask, deny scenarios)
<!-- AC:END -->
