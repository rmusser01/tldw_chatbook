---
id: TASK-1842
title: 'Tool markers are destroyed by the next message (data loss)'
status: In Progress
assignee: []
created_date: '2026-08-01 19:30'
labels:
  - console
  - agents
  - bug
  - data-loss
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A user reported tool output appearing and then vanishing, replaced by `[failed]`. Reproduced deterministically, and the two halves are independent -- the output is lost whether or not the run fails.

TOOL-role markers are display-only and never persisted (`console_chat_store.py:1052-1060`): they are deliberately NOT tree nodes, and `append_message` returns before persisting. `_recompute_active_path` is documented as "the SINGLE writer" of `_messages_by_session` and rebuilds that view **from tree nodes only** -- so every marker is erased the moment anything recomputes the path, which `append_message` does for every non-TOOL message.

Reproduction (store-level, no UI needed): append USER, ASSISTANT, then two TOOL markers -> 4 rows, 2 markers. Append one more USER message -> **3 rows, 0 markers**.

Consequences: the record of what an agent did to the user's machine is gone after their next message; it does not survive session switch or restart; and because `format_agent_step_marker` only ever stored a preview capped at `tool_result_display_chars` (default 160), the full output was never in the transcript to begin with. The complete result survives only in `AgentRunsDB`, reachable via "View full log" in the Agent rail section -- collapsed by default and below the fold at 48 rows.

This matters more given that tools are the agent's route to the outside world: the transcript is the user's only in-context record of what left the machine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tool markers survive subsequent messages in the same session
- [x] #2 Tool markers survive session switch and app restart, or are re-derived from AgentRunsDB on resume
- [ ] #3 The full tool result is reachable from the transcript without navigating to a collapsed rail section
- [x] #4 A failed or denied tool call retains its output rather than being replaced
- [x] #5 A regression test appends a message after tool markers and asserts they are still present -- the exact reproduction above
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Do not simply make TOOL rows tree nodes: the invariant comment at `console_chat_store.py:1052` explains that a marker becoming a parent would corrupt the message chain for the next real message. Likely shape is a separate per-session marker store that `_recompute_active_path` merges into the view, or re-deriving from `AgentRunsDB` (which `resume_marker_messages` already does for resume).

Related: `_console_tool_result_display_cap()` in `console_agent_bridge.py` (default 160, range 20-2000) governs the live summary, the transcript marker, and resumed markers from one number.
**AC #3 is NOT delivered and is carved out to TASK-1860.** This task fixed the DESTRUCTION of markers; the full result is still only a preview capped by `_console_tool_result_display_cap()`, with no expand affordance and no truncation indicator. Left In Progress rather than closed so the gap is not lost.

Review follow-up (same PR): `_tool_markers_by_session` had no lifecycle. `close_session` popped every other per-session structure but left the registry keyed by a dead session, retaining every marker object for the life of the process; `delete_message` purges the whole subtree from the node structures but markers are display-only and are not tree nodes, so deleting a branch left markers registered with their ids still in `_message_session_index` -- claiming a session owned messages it could never render, since `_with_tool_markers` drops off-path anchors. `_purge_tool_markers` now handles both paths.
<!-- SECTION:NOTES:END -->
